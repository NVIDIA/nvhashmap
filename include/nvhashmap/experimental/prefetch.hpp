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

#include "../common.hpp"
//#include "debug.hpp"

namespace nvhm { namespace experimental {

template <typename T, typename U>
constexpr void shift(T& __restrict a, U& __restrict b) noexcept {
  a = b;
}

template <typename T, typename U, typename... Args>
constexpr void shift(T& __restrict a, U& __restrict b, Args&&... args) noexcept {
  a = b;
  shift(b, args...);
}

template <size_t I, typename T, size_t N>
constexpr void shift(T (&__restrict arr)[N]) noexcept {
  static_assert(I > 0);

  if constexpr (I < N) {
    // Forcefully unroll, to avoid weird compiler behaviors when it comes to loop unrolling.
    arr[I - 1] = arr[I];
    shift<I + 1>(arr);
  }
}

template <typename Key, typename Value, size_t N>
class alignas(cache_line_size) shift_prefetch_queue {
 public:
  using key_type = Key;
  using value_type = Value;

  static constexpr size_t capacity{N};
  static_assert(capacity > 0 && capacity < 64);

  shift_prefetch_queue() = default;

  constexpr const key_type& front_key() const noexcept { return keys_[0]; }
  constexpr const value_type& front_value() const noexcept { return values_[0]; }

  constexpr key_type& front_key() noexcept { return keys_[0]; }
  constexpr value_type& front_value() noexcept { return values_[0]; }

  constexpr const key_type& back_key() const noexcept { return keys_[capacity - 1]; }
  constexpr const value_type& back_value() const noexcept { return values_[capacity - 1]; }

  constexpr key_type& back_key() noexcept { return keys_[capacity - 1]; }
  constexpr value_type& back_value() noexcept { return values_[capacity - 1]; }

  constexpr std::pair<key_type, value_type> front() const noexcept {
    return {front_key(), front_value()};
  }

  constexpr std::pair<key_type, value_type> push(
    const key_type& __restrict key, const value_type& __restrict value
  ) noexcept {
    const key_type key0{front_key()};
    const value_type value0{front_value()};
    shift<1>(keys_);
    shift<1>(values_);
    back_key() = key;
    back_value() = value;
    return {key0, value0};
  }

  constexpr std::pair<key_type, value_type> pop() noexcept {
    const key_type key0{front_key()};
    const value_type value0{front_value()};
    shift<1>(keys_);
    shift<1>(values_);
    return {key0, value0};
  }

  template <typename Map>
  constexpr std::pair<key_type, value_type> prepare_lookup(
    const Map& __restrict map, const key_type& __restrict key, const bool optimistic
  ) noexcept {
    return push(key, map.read_prefetch(key, optimistic));
  }

  template <typename Map>
  constexpr std::pair<key_type, value_type> prepare_upsert(
    Map& __restrict map, const key_type& __restrict key, const bool optimistic
  ) noexcept {
    return push(key, map.write_prefetch(key, optimistic));
  }

  template <typename Map>
  constexpr size_t pop_and_lookup(const Map& __restrict map) noexcept {
    const auto [key, value]{pop()};
    return map.lookup(key, value);
  }

 private:
  alignas(cache_line_size / 2) key_type keys_[capacity];
  alignas(cache_line_size / 2) value_type values_[capacity];
};

template <typename Key, typename Value, size_t N>
class alignas(cache_line_size) ring_prefetch_queue {
 public:
  using key_type = Key;
  using value_type = Value;

  static constexpr size_t capacity{N};
  static_assert(has_single_bit(capacity));
  static constexpr size_t capacity_mask{capacity - 1};

  ring_prefetch_queue() = default;

  constexpr size_t front_idx() const noexcept { return r_ & capacity_mask; }
  constexpr size_t back_idx() const noexcept { return w_ & capacity_mask; }

  constexpr const key_type& front_key() const noexcept { return keys_[front_idx()]; }
  constexpr const value_type& front_value() const noexcept { return values_[front_idx()]; }

  constexpr key_type& front_key() noexcept { return keys_[front_idx()]; }
  constexpr value_type& front_value() noexcept { return values_[front_idx()]; }

  constexpr const key_type& back_key() const noexcept { return keys_[back_idx()]; }
  constexpr const value_type& back_value() const noexcept { return values_[back_idx()]; }

  constexpr key_type& back_key() noexcept { return keys_[back_idx()]; }
  constexpr value_type& back_value() noexcept { return values_[back_idx()]; }

  constexpr std::pair<key_type, value_type> front() const noexcept {
    const size_t idx{r_ & capacity_mask};
    return {keys_[idx], values_[idx]};
  }

  constexpr bool empty() const noexcept { return r_ == w_; }

  constexpr bool full() const noexcept { return size() == capacity; }

  constexpr size_t size() const noexcept { return w_ - r_; }

  constexpr ring_prefetch_queue& push(
    const key_type& __restrict key, const value_type& __restrict value
  ) noexcept {
    NVHM_ASSERT_(!full());
    const size_t idx{w_++ & capacity_mask};
    keys_[idx] = key;
    values_[idx] = value;
    return *this;
  }

  constexpr std::pair<key_type, value_type> pop() noexcept {
    NVHM_ASSERT_(!empty());
    const size_t idx{r_++ & capacity_mask};
    const key_type key{keys_[idx]};
    const value_type value{values_[idx]};
    return {key, value};
  }

  constexpr void skip() noexcept {
    NVHM_ASSERT_(!empty());
    ++r_;
  }

  template <typename Map>
  constexpr ring_prefetch_queue& prepare_lookup(
    const Map& __restrict map, const key_type& __restrict key, const bool optimistic
  ) noexcept {
    return push(key, map.read_prefetch(key, optimistic));
  }

  template <typename Map>
  constexpr ring_prefetch_queue& prepare_upsert(
    Map& __restrict map, const key_type& __restrict key, const bool optimistic
  ) noexcept {
    return push(key, map.write_prefetch(key, optimistic));
  }

  template <typename Map>
  constexpr size_t pop_and_lookup(const Map& __restrict map) noexcept {
    const auto [key, value]{pop()};
    return map.lookup(key, value);
  }

  constexpr size_t read_offset() const noexcept { return r_; }

  constexpr size_t write_offset() const noexcept { return w_; }

 private:
  size_t r_{};
  size_t w_{};
  alignas(cache_line_size / 2) key_type keys_[capacity];
  alignas(cache_line_size / 2) value_type values_[capacity];
};

}}