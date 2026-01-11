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

#include "allocator.hpp"
#include "hash.hpp"
#include "kernel.hpp"
#include <variant>

namespace nvhm { namespace experimental {

/**
 * Set-associative cache with LRU replacement policy.
 * The kernel size determines the set-associativity.
 */
template <
  typename Key, typename Value = void, typename RawValue = char, typename Kernel = default_kernel_t,
  typename Allocator = default_allocator<>>
class cache {
 public:
  using key_type = Key;
  using value_type = std::conditional_t<std::is_void_v<Value>, std::monostate, Value>;
  static constexpr bool has_values{!std::is_same_v<value_type, std::monostate>};
  using raw_value_type = RawValue;

  using kernel_type = Kernel;
  using mask_type = typename kernel_type::mask_type;
  using probe_seq_type = void;
  constexpr static bool minimize_psl{true};
  constexpr static bool auto_shrink{false};
  constexpr static bool allow_reclaim{true};
  using allocator_type = Allocator;

#if defined(NDEBUG)
  using prefetch_type = raw_pos_t;
#else
  using prefetch_type = std::tuple<raw_pos_t>;
#endif
  using read_pos_type = raw_pos_t;
  using write_pos_type = raw_pos_t;

  // Not much point in having a table that is smaller than the size of a cache line.
  static constexpr size_t min_capacity{std::max(cache_line_size, kernel_type::size)};
  static_assert(has_single_bit(min_capacity));
  static_assert(min_capacity % kernel_type::size == 0);
  constexpr static size_t default_capacity{std::max<size_t>(min_capacity, 4096)};

  inline cache()
  : cache(default_capacity) {}

  inline cache(
    size_t capacity, const size_t raw_value_size = 0, size_t raw_value_alignment = sizeof(float)
  ) {
    // Ensure alignment is power-2-based.
    raw_value_size_ = raw_value_size;
    raw_value_alignment_ = raw_value_alignment = bit_ceil(raw_value_alignment);
    raw_value_pitch_ = (raw_value_size + raw_value_alignment - 1) & ~(raw_value_alignment - 1);

    capacity = bit_ceil(std::max(capacity, min_capacity));

    // Wrap around at capacity and align at kernel size.
    bucket_mask_ = (capacity - 1) & ~kernel_type::size_mask;
    states_ = allocator_type::template alloc<state_t>(capacity * 2);
    clear();

    keys_ = allocator_type::template alloc<key_type>(capacity);
    if constexpr (has_values) {
      values_ = allocator_type::template alloc<value_type>(capacity);
    }
    raw_values_ = allocator_type::template alloc<raw_value_type>(capacity * raw_value_pitch_);
  }

  inline cache(const cache&) = default;
  inline cache& operator=(const cache&) = default;

  inline cache(cache&&) = default;
  inline cache& operator=(cache&&) = default;

  inline ~cache() = default;

  inline void swap(cache& __restrict that) noexcept {
    std::swap(raw_value_size_, that.raw_value_size_);
    std::swap(raw_value_alignment_, that.raw_value_alignment_);
    std::swap(raw_value_pitch_, that.raw_value_pitch_);

    std::swap(bucket_mask_, that.bucket_mask_);
    std::swap(states_, that.states_);

    std::swap(keys_, that.keys_);
    std::swap(values_, that.values_);
    std::swap(raw_values_, that.raw_values_);
  }

  inline size_t raw_value_size() const noexcept { return raw_value_size_; }

  inline size_t capacity() const noexcept { return bucket_mask_ + kernel_type::size; }

  inline size_t size() const noexcept {
    size_t n{};
    const raw_pos_t end{capacity()};
    for (raw_pos_t off{}; off != end; off += kernel_type::size) {
      const auto kern{kernel_type::load(&states_[off * 2])};
      const auto mask{kernel_type::mask_hash(kern)};
      n += mask_type::count(mask);
    }
    return n;
  }

  inline bool empty() const noexcept { return size() == 0; }

  inline bool full() const noexcept { return size() == capacity(); }

 public:
  /**
   * Mark all slots as free.
   */
  inline void clear() noexcept {
    state_t* const __restrict states{states_.get()};

    const raw_pos_t end2{capacity() * 2};
    for (raw_pos_t off2{}; off2 < end2; off2 += kernel_type::size * 2) {
      std::fill_n(&states[off2], kernel_type::size, kernel_type::empty);
      std::fill_n(
        &states[off2 + kernel_type::size], kernel_type::size, static_cast<state_t>(max_lru)
      );
    }
  }

  /**
   * Check whether the provided `key` is currently in the cache.
   *
   * @param key The key to find.
   * @return Whether or not the key is in in the cache.
   */
  inline bool contains(const key_type& key) const noexcept { return lookup(key) != npos; }

  /**
   * Check whether the provided `key` is currently in the cache.
   *
   * @param key The key to find.
   * @param pre Prefetch token retunred by previous call `prefetch_*`.
   * @return Whether or not the key is in the map.
   */
  inline bool contains(const key_type& key, const prefetch_type& pre) const noexcept {
    return lookup(key, pre) != npos;
  }

  /**
   * Find the key in the cache, and mark the slot as free.
   *
   * @param key The key to find.
   * @return True if the key was in the cache (i.e., the erasure was successful).
   */
  inline bool erase(const key_type& key) noexcept { return erase_(key, key_to_hash(key)); }

  /**
   * Find the key in the cache, and mark the slot as free.
   *
   * @param key The key to find.
   * @param pre Prefetch token retunred by previous call `prefetch_*`.
   * @return True if the key was in the map (i.e., the erasure was successful).
   */
  inline bool erase(const key_type& key, const prefetch_type& pre) {
#ifndef NDEBUG
    return erase_(key, std::get<0>(pre));
#else
    return erase_(key, pre);
#endif
  }

  /**
   * Erase whatever key was at the provided cache-position.
   *
   * @param pos The position to investigate.
   * @return True if the position was occupied (i.e., the erasure was successful).
   */
  inline bool erase_at(const write_pos_type& __restrict pos) {
    return pos < capacity() ? erase_at_(pos) : false;
  }

  /**
   * Locate a key in the cache and return its position.
   *
   * @param key The key to find.
   * @return The position of the key, if it exists in the table, otherwise `npos`.
   */
  inline read_pos_type lookup(const key_type& key) const noexcept {
    return find_(key, key_to_hash(key));
  }

  /**
   * Locate a key in the map and return its position.
   *
   * @param key The key to find.
   * @param pre Prefetch token retunred by previous call `prefetch_*`.
   * @return The position of the key, if it exists in the table, otherwise `npos`.
   */
  inline read_pos_type lookup(const key_type& key, const prefetch_type& pre) const noexcept {
#ifndef NDEBUG
    return find_(key, std::get<0>(pre));
#else
    return find_(key, pre);
#endif
  }

  /**
   * Locate a key in the map and return its position.
   *
   * @param key The key to find.
   * @return The position of the key, if it exists in the table, otherwise `npos`.
   */
  inline write_pos_type update(const key_type& key) noexcept {
    return find_(key, key_to_hash(key));
  }

  /**
   * Locate a key in the map and return its position.
   *
   * @param key The key to find.
   * @param pre Prefetch token retunred by previous call `prefetch_*`.
   * @return The position of the key, if it exists in the table, otherwise `npos`.
   */
  inline write_pos_type update(const key_type& key, const prefetch_type& pre) noexcept {
#ifndef NDEBUG
    return find_(key, std::get<0>(pre));
#else
    return find_(key, pre);
#endif
  }

  /**
   * Inserts a key into the cache if it doesn't exist yet. May replace another key if necessary.
   *
   * @param key The key to find/insert.
   * @return The position of the key.
   */
  inline write_pos_type upsert(const key_type& key) noexcept {
    std::monostate it;
    return upsert(key, it);
  }

  /**
   * Inserts a key into the map if it doesn't exist yet. Grows the map if necessary.
   *
   * @param key The key to find/insert.
   * @param pre Prefetch token retunred by previous call `prefetch_*`.
   * @return The position of the key.
   */
  inline write_pos_type upsert(const key_type& key, const prefetch_type& pre) noexcept {
    std::monostate it;
    return upsert(key, it, pre);
  }

  /**
   * Inserts a key into the map if it doesn't exist yet. Grows the map if necessary. Increments
   * `counter` if the key was actually inserted.
   *
   * @param key The key to find/insert.
   * @param it Some iterator to increment.
   * @return The position of the key.
   */
  template <typename It>
  inline write_pos_type upsert(const key_type& key, It& it) noexcept {
    return insert_(key, it, key_to_hash(key));
  }

  /**
   * Inserts a key into the map if it doesn't exist yet. Grows the map if necessary. Increments
   * supplied iterattor if the key was actually inserted.
   *
   * @param key The key to find/insert.
   * @param it Some iterator to increment.
   * @param pre Prefetch token retunred by previous call `prefetch_*`.
   * @return The position of the key.
   */
  template <typename It>
  inline write_pos_type upsert(const key_type& key, It& it, const prefetch_type& pre) noexcept {
#ifndef NDEBUG
    return insert_(key, it, std::get<0>(pre));
#else
    return insert_(key, it, pre);
#endif
  }

  /**
   * Decode the key and prefetch memory to facilitate a future `lookup`-operation.
   *
   * @param key The key to prefetch.
   * @param optimistic If `true`, will do a more thorough prefetch.
   * @return Prefetch token. Intended to be used in conjunction with `find_ex`.
   */
  inline prefetch_type read_prefetch(const key_type& key, const bool optimistic) const noexcept {
    const size_t bucket_mask{bucket_mask_};
    const state_t* const __restrict states{states_.get()};

    const hash_t h{key_to_hash(key)};
    const raw_pos_t off{hash_to_pos<kernel_type>(h, bucket_mask)};

    nvhm::read_prefetch<kernel_type::size * 2>(&states[off * 2]);
    if (optimistic) {
      const key_type* const __restrict keys{keys_.get()};
      nvhm::read_prefetch<kernel_type::size * sizeof(key_type)>(&keys[off]);
    }

#ifndef NDEBUG
    return {h};
#else
    return h;
#endif
  }

  /**
   * Decode the key and prefetch memory to facilitate a future `update` or `upsert`-operation.
   *
   * @param key The key to prefetch.
   * @param optimistic If `true`, will do a more thorough prefetch.
   * @return Prefetch token. Intended to be used in conjunction with `insert_ex`.
   */
  inline prefetch_type write_prefetch(const key_type& key, const bool optimistic) noexcept {
    const size_t bucket_mask{bucket_mask_};
    state_t* const __restrict states{states_.get()};

    const hash_t h{key_to_hash(key)};
    const raw_pos_t off{hash_to_pos<kernel_type>(h, bucket_mask)};

    nvhm::write_prefetch<kernel_type::size * 2>(&states[off * 2]);
    if (optimistic) {
      key_type* const __restrict keys{keys_.get()};
      nvhm::write_prefetch<kernel_type::size * sizeof(key_type)>(&keys[off]);
    }

#ifndef NDEBUG
    return {h};
#else
    return h;
#endif
  }

 public:
  /**
   * Scan through the map and return the position of the matching entry.
   *
   * @return The position of the valid key, or `npos`.
   */
  inline read_pos_type find_if(
    const std::function<bool(const read_pos_type&)>& __restrict f
  ) const {
    const state_t* const __restrict states{states_.get()};

    const raw_pos_t end{capacity()};
    for (raw_pos_t off{}; off != end; off += kernel_type::size) {
      const auto kern{kernel_type::load(&states[off * 2])};

      auto mask{kernel_type::mask_hash(kern)};
      for (; mask_type::has_next(mask); mask = mask_type::step(mask)) {
        const read_pos_type pos{mask_type::next(mask, off)};
        if (f(pos)) return pos;
      }
    }

    return npos;
  }

  /**
   * Scan through the map and return the position of the matching entry.
   *
   * @return The position of the valid key, or `npos`.
   */
  inline write_pos_type find_if(const std::function<bool(const write_pos_type&)>& __restrict f) {
    const state_t* const __restrict states{states_.get()};

    const raw_pos_t end{capacity()};
    for (raw_pos_t off{}; off != end; off += kernel_type::size) {
      const auto kern{kernel_type::load(&states[off * 2])};

      auto mask{kernel_type::mask_hash(kern)};
      for (; mask_type::has_next(mask); mask = mask_type::step(mask)) {
        const write_pos_type pos{mask_type::next(mask, off)};
        if (f(pos)) return pos;
      }
    }

    return npos;
  }

  /**
   * Scan through the map and return the position of the first valid entry.
   *
   * @return The position of the valid key, or `npos`.
   */
  inline read_pos_type first() const noexcept {
    return find_if([](const read_pos_type&) { return true; });
  }

  /**
   * Scan through the map and return the position of the first valid entry.
   *
   * @return The position of the valid key, or `npos`.
   */
  inline write_pos_type first() noexcept {
    return find_if([](const write_pos_type&) { return true; });
  }

  /**
   * Iterate over the cache and call a function for each filled slot.
   *
   * @param f The function to call.
   */
  inline void for_each(const std::function<void(const read_pos_type&)>& __restrict f) const {
    const state_t* const __restrict states{states_.get()};

    const raw_pos_t end{capacity()};
    for (raw_pos_t off{}; off != end; off += kernel_type::size) {
      const auto kern{kernel_type::load(&states[off * 2])};

      auto mask{kernel_type::mask_hash(kern)};
      for (; mask_type::has_next(mask); mask = mask_type::step(mask)) {
        f(mask_type::next(mask, off));
      }
    }
  }

  /**
   * Retrieve all keys in the map.
   *
   * @param it An output iterator to write the keys to.
   */
  template <typename OutIt>
  inline OutIt keys(OutIt it) const noexcept {
    for_each([&](const read_pos_type& __restrict pos) { *it++ = key_at(pos); });
    return it;
  }

  /**
   * Retrieve all keys and values in the map.
   *
   * @param it An output iterator to write the keys and values to.
   */
  template <typename OutIt>
  inline OutIt keys_and_values(OutIt it) const noexcept {
    for_each([&](const read_pos_type& __restrict pos) { *it++ = {key_at(pos), value_at(pos)}; });
    return it;
  }

  /**
   * Retrieve all valid map positions.
   *
   * @param it An output iterator to write the positions to.
   */
  template <typename OutIt>
  inline OutIt read_positions(OutIt it) const noexcept {
    for_each([&](const read_pos_type& __restrict pos) { *it++ = pos; });
    return it;
  }

  /**
   * Quickly transform all normal values.
   *
   * @param f The transformation to apply.
   */
  inline void transform_values(const std::function<void(value_type&)>& __restrict f) {
    value_type* const __restrict values{values_.get()};
    std::for_each(values, &values[capacity()], f);
  }

  /**
   * Quickly transform all raw values.
   *
   * @param f The transformation to apply.
   */
  inline void transform_raw_values(
    const std::function<raw_value_type(raw_value_type)>& __restrict f
  ) {
    raw_value_type* const __restrict raw_values{raw_values_.get()};
    std::transform(raw_values, &raw_values[capacity() * raw_value_pitch_], raw_values, f);
  }

 public:
  /**
   * Check if the slot at `pos` is occupied.
   *
   * @param pos The position to query.
   * @return True if the slot contains a key.
   */
  inline bool is_occupied(const read_pos_type& __restrict pos) const noexcept {
    return slot_is_occupied(state_at(pos));
  }

  /**
   * Fetch the state/fingerprint of `pos`.
   *
   * @param pos The position to query.
   * @return The associated slot data.
   */
  inline state_t state_at(const read_pos_type& __restrict pos) const noexcept {
    NVHM_ASSERT_(pos < capacity(), "pos = ", pos, ", capacity = ", capacity());
    const raw_pos_t off{pos & ~kernel_type::size_mask};
    const raw_pos_t idx{pos & kernel_type::size_mask};
    return states_[off * 2 + idx];
  }

  /**
   * Fetch the LRU value stored for `pos`.
   *
   * @param pos The position to query.
   * @return The associated LRU value.
   */
  inline lru_t lru_at(const read_pos_type& __restrict pos) const noexcept {
    NVHM_ASSERT_(is_occupied(pos));
    const raw_pos_t off{pos & ~kernel_type::size_mask};
    const raw_pos_t idx{pos & kernel_type::size_mask};
    return static_cast<lru_t>(states_[off * 2 + idx + kernel_type::size_mask]);
  }

  /**
   * Fetch the LRU value stored for `pos`.
   *
   * @param pos The position to query.
   * @return The associated LRU value.
   */
  inline lru_t& lru_at(const write_pos_type& __restrict pos) noexcept {
    NVHM_ASSERT_(is_occupied(pos));
    const raw_pos_t off{pos & ~kernel_type::size_mask};
    const raw_pos_t idx{pos & kernel_type::size_mask};
    return static_cast<lru_t&>(states_[off * 2 + idx + kernel_type::size_mask]);
  }

  /**
   * Fetch the key at `pos`.
   *
   * @param pos The position to query.
   * @return The associated key.
   */
  inline const key_type& key_at(const read_pos_type& __restrict pos) const noexcept {
    NVHM_ASSERT_(is_occupied(pos));
    return keys_[pos];
  }

  /**
   * Fetch the value for `pos`.
   *
   * @param pos The position to query.
   * @return The associated value.
   */
  inline const value_type& value_at(const read_pos_type& __restrict pos) const noexcept {
    static_assert(has_values);
    NVHM_ASSERT_(is_occupied(pos));
    return values_[pos];
  }

  /**
   * Fetch the value for `pos`.
   *
   * @param pos The position to query.
   * @return The associated value.
   */
  inline value_type& value_at(const write_pos_type& __restrict pos) noexcept {
    static_assert(has_values);
    NVHM_ASSERT_(is_occupied(pos));
    return values_[pos];
  }

  /**
   * Fetch pointer to the raw values for `pos`.
   *
   * @param pos The position to query.
   * @return Pointer the associated raw values.
   */
  inline const raw_value_type* raw_values_at(const read_pos_type& __restrict pos) const noexcept {
    NVHM_ASSERT_(is_occupied(pos));
    NVHM_ASSERT_(raw_value_pitch_, "raw_value_pitch = ", raw_value_pitch_);
    return &raw_values_[pos * raw_value_pitch_];
  }

  /**
   * Fetch pointer to the raw values for `pos`.
   *
   * @param pos The position to query.
   * @return Pointer the associated raw values.
   */
  inline raw_value_type* raw_values_at(const write_pos_type& __restrict pos) noexcept {
    NVHM_ASSERT_(is_occupied(pos));
    NVHM_ASSERT_(raw_value_pitch_, "raw_value_pitch = ", raw_value_pitch_);
    return &raw_values_[pos * raw_value_pitch_];
  }

  inline void get_raw_values_at(
    const read_pos_type& pos, raw_value_type* const dst, const size_t n
  ) const noexcept {
    NVHM_ASSERT_(n <= raw_value_size_, "n = ", n, ", raw_value_size = ", raw_value_size_);
    fast_copy(dst, raw_values_at(pos), n);
  }

  inline void set_raw_values_at(
    const write_pos_type& pos, const raw_value_type* const src, const size_t n
  ) noexcept {
    NVHM_ASSERT_(n <= raw_value_size_, "n = ", n, ", raw_value_size = ", raw_value_size_);
    fast_copy(raw_values_at(pos), src, n);
  }

 public:
  inline psl_t psl(const key_type&) const noexcept { return kernel_type::size; }

  inline std::array<size_t, kernel_type::size> state_collisions() const {
    std::array<size_t, kernel_type::size> res{};

    const raw_pos_t end{capacity()};
    for (raw_pos_t off{}; off != end; off += kernel_type::size) {
      const auto kern{kernel_type::load(&states_[off * 2])};

      std::array<state_t, kernel_type::size> tmp;
      size_t n{};

      auto mask{kernel_type::mask_hash(kern)};
      for (; mask_type::has_next(mask); mask = mask_type::step(mask)) {
        const size_t idx{mask_type::next(mask)};
        NVHM_ASSERT_(n < kernel_type::size);
        tmp[n++] = kernel_type::at(kern, idx);
      }
      std::sort(tmp.begin(), tmp.begin() + n);

      const auto tmp_end{std::unique(tmp.begin(), tmp.begin() + n)};
      for (auto it{tmp.begin()}; it != tmp_end; ++it) {
        mask = kernel_type::mask(kern, *it);
        ++res[mask_type::count(mask) - 1];
      }
    }

    return res;
  }

 protected:
  inline bool erase_(const key_type& __restrict key, const hash_t h) noexcept {
    NVHM_ASSERT_(key_to_hash(key) == h, "Supplied key and hash do not match!");
    const size_t bucket_mask{bucket_mask_};
    state_t* const __restrict states{states_.get()};
    const key_type* const __restrict keys{keys_.get()};

    const state_t s{hash_to_state(h)};

    const raw_pos_t off{hash_to_pos<kernel_type>(h, bucket_mask)};
    const raw_pos_t off2{off * 2};
    const auto kern{kernel_type::load(&states[off2])};

    // Find slot.
    auto mask{kernel_type::mask(kern, s)};
    for (; mask_type::has_next(mask); mask = mask_type::step(mask)) {
      const raw_pos_t pos{mask_type::next(mask, off)};
      if (NVHM_UNLIKELY_(keys[pos] != key)) continue;

      const raw_pos_t idx{pos & kernel_type::size_mask};
      states[off2 + idx] = kernel_type::empty;
      states[off2 + idx + kernel_type::size] = static_cast<state_t>(max_lru);
      return true;
    }

    return false;
  }

  inline bool erase_at_(const raw_pos_t pos) noexcept {
    NVHM_ASSERT_(pos < capacity(), "`pos` is out of bounds");
    state_t* const __restrict states{states_.get()};

    const raw_pos_t off{pos & ~kernel_type::size_mask};
    const raw_pos_t off2{off * 2};
    const raw_pos_t idx{pos & kernel_type::size_mask};

    if (slot_not_occupied(states[off2 + idx])) return false;

    states[off2 + idx] = kernel_type::empty;
    states[off2 + idx + kernel_type::size] = static_cast<state_t>(max_lru);
    return true;
  }

  inline raw_pos_t find_(const key_type& __restrict k, const hash_t h) const noexcept {
    NVHM_ASSERT_(key_to_hash(k) == h, "Supplied key and hash do not match!");
    const size_t bucket_mask{bucket_mask_};
    state_t* const __restrict states{states_.get()};
    const key_type* const __restrict keys{keys_.get()};

    const state_t s{hash_to_state(h)};

    const raw_pos_t off{hash_to_pos<kernel_type>(h, bucket_mask)};
    const raw_pos_t off2{off * 2};
    const auto kern{kernel_type::load(&states[off2])};

    auto mask{kernel_type::mask(kern, s)};
    for (; mask_type::has_next(mask); mask = mask_type::step(mask)) {
      const raw_pos_t pos{mask_type::next(mask, off)};
      if (NVHM_UNLIKELY_(keys[pos] != k)) continue;

      // Increment LRU, and rescale bucket if necessary.
      auto lru_kern{kernel_type::load(&states[off2 + kernel_type::size])};
      lru_kern = kernel_type::lru_update(kern, lru_kern, mask);
      kernel_type::store(lru_kern, &states[off2 + kernel_type::size]);
      return pos;
    }

    return npos;
  }

  template <typename It>
  inline raw_pos_t insert_(
    const key_type& __restrict k, It& __restrict it, const hash_t h
  ) noexcept {
    NVHM_ASSERT_(key_to_hash(k) == h, "Supplied key and hash do not match!");
    const size_t bucket_mask{bucket_mask_};
    state_t* const __restrict states{states_.get()};
    key_type* const __restrict keys{keys_.get()};

    const state_t s{hash_to_state(h)};

    // Try to locate the slot.
    const raw_pos_t off{hash_to_pos<kernel_type>(h, bucket_mask)};
    const raw_pos_t off2{off * 2};
    auto kern{kernel_type::load(&states[off2])};

    auto mask{kernel_type::mask(kern, s)};
    for (; mask_type::has_next(mask); mask = mask_type::step(mask)) {
      const raw_pos_t pos{mask_type::next(mask, off)};
      if (NVHM_UNLIKELY_(keys[pos] != k)) continue;

      // Increment LRU, and rescale bucket if necessary.
      auto lru_kern{kernel_type::load(&states[off2 + kernel_type::size])};
      lru_kern = kernel_type::lru_update(kern, lru_kern, mask);
      kernel_type::store(lru_kern, &states[off2 + kernel_type::size]);
      return pos;
    }

    // Have slot available?
    size_t idx;
    mask = kernel_type::mask_non_hash(kern);
    if (mask_type::has_next(mask)) {
      idx = mask_type::next(mask);
      if constexpr (!std::is_same_v<It, std::monostate>) {
        ++it;
      }
    } else {
      // Displace least recently used key.
      kern = kernel_type::load(&states[off2 + kernel_type::size]);
      idx = kernel_type::min_lru_index(kern);
    }
    const raw_pos_t pos{off + idx};

    states[off2 + idx] = s;
    states[off2 + idx + kernel_type::size] = static_cast<state_t>(default_lru);
    keys[pos] = k;
    return pos;
  }

 protected:
  size_t raw_value_size_;
  size_t raw_value_alignment_;
  size_t raw_value_pitch_;

  size_t bucket_mask_;
  aligned_unique_ptr<state_t[]> states_;

  aligned_unique_ptr<key_type[]> keys_;
  aligned_unique_ptr<value_type[]> values_;
  aligned_unique_ptr<raw_value_type[]> raw_values_;
};

}}