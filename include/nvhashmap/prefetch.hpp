/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "common.hpp"

namespace nvhm {

template <int_t Idx, typename T, int_t Capacity>
constexpr void shift(T (&__restrict arr)[Capacity]) noexcept {
  static_assert(Idx > 0);

  if constexpr (Idx < Capacity) {
    // Forcefully unroll, to avoid weird compiler behaviors when it comes to loop unrolling.
    arr[Idx - 1] = std::move(arr[Idx]);
    shift<Idx + 1>(arr);
  }
}

enum class queue_t : int_t {
  shift,
  ring
};

constexpr std::string to_string(queue_t type) {
  switch (type) {
    case queue_t::shift: return "shift";
    case queue_t::ring: return "ring";
  }
  return "error";
}

inline std::ostream& operator<<(std::ostream& os, queue_t type) {
  return os << to_string(type);
}

template <typename Self, queue_t Type, typename Key, typename Value, int_t Capacity>
class alignas(cache_line_size) prefetch_queue : public self_aware<Self> {
 public:
  using base_type = self_aware<Self>;
  using self_type = typename base_type::self_type;

  constexpr static queue_t type{Type};
  using key_type = Key;
  using value_type = Value;
  using const_entry_type = std::pair<const_view_t<key_type>, const_view_t<value_type>>;
  using entry_type = std::pair<key_type, value_type>;

  constexpr static int_t capacity{Capacity};
  static_assert(capacity > 0);

  constexpr prefetch_queue() = default;

  using base_type::self;

  template <typename Map, typename K>
  [[nodiscard]] constexpr entry_type push_read(const Map& map, K&& key) noexcept {
    return self()->push(std::forward<K>(key), map.read_prefetch(key));
  }

  template <typename Map, typename K>
  [[nodiscard]] constexpr entry_type push_write(Map& map, K&& key) noexcept {
    return self()->push(std::forward<K>(key), map.write_prefetch(key));
  }

 protected:
  alignas(cache_line_size / 2) key_type keys_[capacity];
  alignas(cache_line_size / 2) value_type values_[capacity];
};

template <typename Key, typename Value, int_t Capacity>
class alignas(cache_line_size) shift_prefetch_queue
  : public prefetch_queue<shift_prefetch_queue<Key, Value, Capacity>, queue_t::shift, Key, Value, Capacity> {
 public:
  using base_type = prefetch_queue<shift_prefetch_queue, queue_t::shift, Key, Value, Capacity>;
  using self_type = typename base_type::self_type;

  using key_type = typename base_type::key_type;
  using value_type = typename base_type::value_type;
  using const_entry_type = typename base_type::const_entry_type;
  using entry_type = typename base_type::entry_type;

  using base_type::capacity;

  shift_prefetch_queue() = default;

  using base_type::self;

  constexpr const_entry_type front() const noexcept { return {keys_[0], values_[0]}; }
  constexpr const_entry_type back() const noexcept { return {keys_[capacity - 1], values_[capacity - 1]}; }

  template <typename K, typename V>
  [[nodiscard]] constexpr entry_type push(K&& key, V&& value) noexcept {
    key_type key0{std::move(keys_[0])};
    value_type value0{std::move(values_[0])};
    skip();
    keys_[capacity - 1] = std::forward<K>(key);
    values_[capacity - 1] = std::forward<V>(value);
    return {std::move(key0), std::move(value0)};
  }

  constexpr entry_type pop() noexcept {
    key_type key0{std::move(keys_[0])};
    value_type value0{std::move(values_[0])};
    skip();
    return {std::move(key0), std::move(value0)};
  }

  constexpr self_type& skip() noexcept {
    shift<1>(keys_);
    shift<1>(values_);
    return *self();
  }

  template<typename K, typename V>
  constexpr self_type& prefill(int_t idx, K&& key, V&& value) noexcept {
    NVHM_ASSERT_(idx >= 0 && idx < capacity);
    keys_[idx] = std::forward<K>(key);
    values_[idx] = std::forward<V>(value);
    return *self();
  }

  template <typename Map, typename K>
  constexpr self_type& prefill_read(int_t idx, const Map& map, K&& key) noexcept {
    return prefill(idx, std::forward<K>(key), map.read_prefetch(key));
  }

  template<typename Map, typename K>
  constexpr self_type& prefill_write(int_t idx, Map& map, K&& key) noexcept {
    return prefill(idx, std::forward<K>(key), map.write_prefetch(key));
  }

 protected:
  using base_type::keys_;
  using base_type::values_;
};

template <typename Key, typename Value, int_t Capacity>
class alignas(cache_line_size) ring_prefetch_queue
  : public prefetch_queue<ring_prefetch_queue<Key, Value, Capacity>, queue_t::ring, Key, Value, Capacity> {
 public:
  using base_type = prefetch_queue<ring_prefetch_queue, queue_t::ring, Key, Value, Capacity>;
  using self_type = typename base_type::self_type;

  using key_type = typename base_type::key_type;
  using value_type = typename base_type::value_type;
  using const_entry_type = typename base_type::const_entry_type;
  using entry_type = typename base_type::entry_type;

  using base_type::capacity;
  static_assert(has_single_bit(capacity));
  constexpr static bitmask_t capacity_mask{size_mask_v<capacity>};

  ring_prefetch_queue() = default;

  using base_type::self;

  /**
   * Returns the front entry of the queue (the oldest entry).
   *
   * @return The front entry of the queue.
   */
  constexpr const_entry_type front() const noexcept {
    NVHM_ASSERT_(!empty());
    const int_t idx{align_pos<capacity_mask>(begin_)};
    return {keys_[idx], values_[idx]};
  }
  /**
   * Returns the back entry of the queue (the newest entry).
   *
   * @return The back entry of the queue.
   */
  constexpr const_entry_type back() const noexcept {
    NVHM_ASSERT_(!empty());
    const int_t idx{align_pos<capacity_mask>(end_ - 1)};
    return {keys_[idx], values_[idx]};
  }

  constexpr int_t size() const noexcept { return end_ - begin_; }
  constexpr bool empty() const noexcept { return begin_ == end_; }
  constexpr bool full() const noexcept { return size() == capacity; }

  template <typename K, typename V>
  [[nodiscard]] constexpr entry_type push(K&& key, V&& value) noexcept {
    NVHM_ASSERT_(full(), "Queue must be full to push.");
    
    int_t idx{align_pos<capacity_mask>(begin_++)};
    key_type key0{std::move(keys_[idx])};
    value_type value0{std::move(values_[idx])};

    idx = align_pos<capacity_mask>(end_++);
    keys_[idx] = std::forward<K>(key);
    values_[idx] = std::forward<V>(value);

    return {std::move(key0), std::move(value0)};
  }

  constexpr entry_type pop() noexcept {
    NVHM_ASSERT_(!empty());
    const int_t idx{align_pos<capacity_mask>(begin_++)};
    return {std::move(keys_[idx]), std::move(values_[idx])};
  }

  constexpr self_type& skip() noexcept {
    NVHM_ASSERT_(!empty());
    ++begin_;
    return *self();
  }
  
  template<typename K, typename V>
  constexpr self_type& prefill(K&& key, V&& value) noexcept {
    const int_t idx{align_pos<capacity_mask>(end_++)};
    keys_[idx] = std::forward<K>(key);
    values_[idx] = std::forward<V>(value);
    return *self();
  }

  template <typename Map, typename K>
  constexpr self_type& prefill_read(const Map& map, K&& key) noexcept {
    return prefill(std::forward<K>(key), map.read_prefetch(key));
  }

  template<typename Map, typename K>
  constexpr self_type& prefill_write(Map& map, K&& key) noexcept {
    return prefill(std::forward<K>(key), map.write_prefetch(key));
  }

 protected:
  using base_type::keys_;
  using base_type::values_;
  int_t begin_{};
  int_t end_{};
};

}