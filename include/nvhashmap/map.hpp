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
#include "memory.hpp"
#include "probe_seq.hpp"

#include <algorithm>
#include <variant>

namespace nvhm {

/**
 * Mutable growable hashmap.
 *
 * @tparam Key Type to use for keys.
 * @tparam Value Type to use for values.
 * @tparam Kernel The kernel to use for parsing state arrays.
 * @tparam ProbeSeq The order in which to traverse the map to find a key.
 * @tparam MinimizeProbeLength Shorten the probe length if possible. This may slighly increase the
 *         time required for insertions. Highly recommended if keys are `erased` frequently.
 * @tparam MaxProbeLength Caps probe length.
 */
template <
  typename Key, typename Value = void, typename RawValue = char, typename Kernel = default_kernel_t,
  typename ProbeSeq = default_seq_t, bool MinimizePSL = false, bool AutoShrink = false,
  bool AllowReclaim = true, typename Allocator = default_allocator<>>
class map {
 public:
  using key_type = Key;
  using value_type = std::conditional_t<std::is_void_v<Value>, std::monostate, Value>;
  constexpr static bool has_values{!std::is_same_v<value_type, std::monostate>};
  using raw_value_type = RawValue;

  using kernel_type = Kernel;
  using mask_type = typename kernel_type::mask_type;
  using probe_seq_type = ProbeSeq;
  constexpr static bool minimize_psl{MinimizePSL};
  constexpr static bool auto_shrink{AutoShrink};
  constexpr static bool allow_reclaim{AllowReclaim};
  using allocator_type = Allocator;

#ifndef NDEBUG
  using prefetch_type = std::tuple<raw_pos_t>;
#else
  using prefetch_type = raw_pos_t;
#endif
  using read_pos_type = raw_pos_t;
  using write_pos_type = raw_pos_t;

  // Not much point in having a table that is smaller than the size of a cache line.
  constexpr static size_t min_capacity{std::max(cache_line_size, kernel_type::size)};
  static_assert(has_single_bit(min_capacity));
  static_assert(min_capacity % kernel_type::size == 0);
  constexpr static size_t default_capacity{std::max<size_t>(min_capacity, 4096)};

  // Reserve capacity bias and related calculations are always based on the min_capacity.
  static_assert(min_capacity % 8 == 0);
  constexpr static size_t max_reserve_capacity_bias{min_capacity / 2};      // 50%
  constexpr static size_t default_reserve_capacity_bias{min_capacity / 8};  // 12.5%
  static_assert(default_reserve_capacity_bias <= max_reserve_capacity_bias);

  inline map()
  : map(default_capacity) {}

  inline map(
    size_t initial_capacity, const size_t raw_value_size = {},
    size_t raw_value_alignment = sizeof(float),
    const size_t reserve_capacity_bias = default_reserve_capacity_bias
  ) {
    // Ensure alignment is power-2-based.
    raw_value_size_ = raw_value_size;
    raw_value_alignment_ = raw_value_alignment = bit_ceil(raw_value_alignment);
    raw_value_pitch_ = (raw_value_size + raw_value_alignment - 1) & ~(raw_value_alignment - 1);

    // Bias for controlling preemptive growth.
    if (reserve_capacity_bias >= max_reserve_capacity_bias) {
      throw std::out_of_range(
        "Reserve capacity bias is out of bounds [0, min(64, min_capacity) - 1]!"
      );
    }
    reserve_capacity_bias_ = reserve_capacity_bias;

    initial_capacity = bit_ceil(std::max(initial_capacity, min_capacity));
    reset_(initial_capacity);
  }

  inline map(const map&) = default;
  inline map& operator=(const map&) = default;

  inline map(map&&) = default;
  inline map& operator=(map&&) = default;

  inline ~map() = default;

  inline void swap(map& __restrict that) noexcept {
    std::swap(raw_value_size_, that.raw_value_size_);
    std::swap(raw_value_alignment_, that.raw_value_alignment_);
    std::swap(raw_value_pitch_, that.raw_value_pitch_);
    std::swap(reserve_capacity_bias_, that.reserve_capacity_bias_);

    std::swap(bucket_mask_, that.bucket_mask_);
    std::swap(growth_threshold_, that.growth_threshold_);
    std::swap(states_, that.states_);
    std::swap(num_used_, that.num_used_);
    std::swap(num_tombstones_, that.num_tombstones_);

    std::swap(keys_, that.keys_);
    std::swap(values_, that.values_);
    std::swap(raw_values_, that.raw_values_);
  }

  inline size_t raw_value_size() const noexcept { return raw_value_size_; }

  inline size_t capacity() const noexcept { return bucket_mask_ + kernel_type::size; }

  inline size_t size() const noexcept { return num_used_ - num_tombstones_; }

  inline bool empty() const noexcept { return size() == 0; }

  inline bool full() const noexcept { return size() == capacity(); }

 public:
  /**
   * Mark all slots as free.
   */
  inline void clear() {
    bool should_shrink;
    if constexpr (auto_shrink) {
      should_shrink = capacity() > min_capacity;
    } else {
      should_shrink = false;
    }

    if (should_shrink) {
      reset_(min_capacity);
    } else {
      std::fill_n(states_.get(), capacity(), kernel_type::empty);
      num_used_ = {};
      num_tombstones_ = {};
    }
  }

  /**
   * Check whether the provided `key` is currently in the map.
   *
   * @param key The key to find.
   * @return Whether or not the key is in the map.
   */
  inline bool contains(const key_type& key) const noexcept { return lookup(key) != npos; }

  /**
   * Check whether the provided `key` is currently in the map.
   *
   * @param key The key to find.
   * @param pre Prefetch token retunred by previous call `prefetch_*`.
   * @return Whether or not the key is in the map.
   */
  inline bool contains(const key_type& key, const prefetch_type& pre) const noexcept {
    return lookup(key, pre) != npos;
  }

  /**
   * Find the key in the map, and mark the slot as free.
   *
   * @param key The key to find.
   * @return True if the key was in the map (i.e., the erasure was successful).
   */
  inline bool erase(const key_type& key) {
    psl_t psl;
    return erase_(key, psl, key_to_hash(key));
  }

  /**
   * Find the key in the map, and mark the slot as free.
   *
   * @param key The key to find.
   * @param pre Prefetch token retunred by previous call `prefetch_*`.
   * @return True if the key was in the map (i.e., the erasure was successful).
   */
  inline bool erase(const key_type& key, const prefetch_type& pre) {
    psl_t psl;
#ifndef NDEBUG
    return erase_(key, psl, std::get<0>(pre));
#else
    return erase_(key, psl, pre);
#endif
  }

  /**
   * Erase whatever key was at the provided map-position.
   *
   * @param pos The position to investigate.
   * @return True if the position was occupied (i.e., the erasure was successful).
   */
  inline bool erase_at(const write_pos_type& __restrict pos) {
    return pos < capacity() ? erase_at_(pos) : false;
  }

  /**
   * Locate a key in the map and return its position.
   *
   * @param key The key to find.
   * @return The position of the key, if it exists in the table, otherwise `npos`.
   */
  inline read_pos_type lookup(const key_type& key) const noexcept {
    psl_t psl;
    return find_(key, psl, key_to_hash(key));
  }

  /**
   * Locate a key in the map and return its position.
   *
   * @param key The key to find.
   * @param pre Prefetch token retunred by previous call `prefetch_*`.
   * @return The position of the key, if it exists in the table, otherwise `npos`.
   */
  inline read_pos_type lookup(const key_type& key, const prefetch_type& pre) const noexcept {
    psl_t psl;
#ifndef NDEBUG
    return find_(key, psl, std::get<0>(pre));
#else
    return find_(key, psl, pre);
#endif
  }

  /**
   * Locate a key in the map and return its position.
   *
   * @param key The key to find.
   * @return The position of the key, if it exists in the table, otherwise `npos`.
   */
  inline write_pos_type update(const key_type& key) noexcept {
    psl_t psl;
    return find_(key, psl, key_to_hash(key));
  }

  /**
   * Locate a key in the map and return its position.
   *
   * @param key The key to find.
   * @param pre Prefetch token retunred by previous call `prefetch_*`.
   * @return The position of the key, if it exists in the table, otherwise `npos`.
   */
  inline write_pos_type update(const key_type& key, const prefetch_type& pre) noexcept {
    psl_t psl;
#ifndef NDEBUG
    return find_(key, psl, std::get<0>(pre));
#else
    return find_(key, psl, pre);
#endif
  }

  /**
   * Inserts a key into the map if it doesn't exist yet. Grows the map if necessary.
   *
   * @param key The key to find/insert.
   * @return The position of the key.
   */
  inline write_pos_type upsert(const key_type& key) {
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
  inline write_pos_type upsert(const key_type& key, const prefetch_type& pre) {
    std::monostate it;
    return upsert(key, it, pre);
  }

  /**
   * Inserts a key into the map if it doesn't exist yet. Grows the map if necessary. Increments
   * supplied iterattor if the key was actually inserted.
   *
   * @param key The key to find/insert.
   * @param it Some iterator to increment.
   * @return The position of the key.
   */
  template <typename It>
  inline write_pos_type upsert(const key_type& key, It& it) {
    psl_t psl;
    return insert_(key, psl, it, key_to_hash(key));
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
  inline write_pos_type upsert(const key_type& key, It& it, const prefetch_type& pre) {
    psl_t psl;
#ifndef NDEBUG
    return insert_(key, psl, it, std::get<0>(pre));
#else
    return insert_(key, psl, it, pre);
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

    nvhm::read_prefetch<kernel_type::size>(&states[off]);
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
   * Decode the key and prefetch memory to facilitate a future update` or `upsert`-operation.
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

    nvhm::write_prefetch<kernel_type::size>(&states[off]);
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
      const auto kern{kernel_type::load(&states[off])};

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
      const auto kern{kernel_type::load(&states[off])};

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
   * Iterate over the map and call a function for each filled slot.
   *
   * @param f The function to call.
   */
  inline void for_each(const std::function<void(const read_pos_type&)>& __restrict f) const {
    const state_t* const __restrict states{states_.get()};

    const raw_pos_t end{capacity()};
    for (raw_pos_t off{}; off != end; off += kernel_type::size) {
      const auto kern{kernel_type::load(&states[off])};

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
   * Number of slots currently occupied with a valid hash.
   */
  inline size_t num_used() const noexcept { return num_used_; }

  /**
   * Number of slots currently occupied buy tombstone markers.
   */
  inline size_t num_tombstones() const noexcept { return num_tombstones_; }

  /**
   * Returns the current growth limit of the map.
   *
   * @return The limit beyond which the next growth can happen.
   */
  inline size_t growth_threshold() const noexcept { return growth_threshold_; }

  /**
   * Check if a map grow.
   *
   * @return Whether or not the table should grow.
   */
  inline bool should_grow() const noexcept { return num_used_ >= growth_threshold_; }

  /**
   * Check if the table is supposed to shrink.
   *
   * @return Whether or not the table should shrink.
   */
  inline bool should_shrink() const noexcept {
    const size_t cap{capacity()};
    return cap > min_capacity && size() < cap / 4;
  }

  /**
   * Check if the table is supposed to be compacted.
   *
   * @return Whether or not the table should be compacted.
   */
  inline bool should_compact() const noexcept { return num_tombstones_ >= capacity() / 4; }

  /**
   * Grow the table now.
   */
  inline void grow() { realloc_(capacity() << 1); }

  /**
   * Apply table compactification now.
   */
  inline void compact() {
    if (num_tombstones_) {
      realloc_(capacity());
    }
  }

  /**
   * Optimize the table, eventually freeing up some space.
   */
  inline void optimize() {
    if (should_shrink()) {
      NVHM_DBG_LOG_(
        "should shrink, size/capacity:", size(), '/', capacity(),
        ", before shrink: ", should_shrink(), '\n'
      );
      realloc_(std::max(capacity() >> 1, min_capacity));
      NVHM_DBG_LOG_(
        "should shrink, size/capacity:", size(), '/', capacity(),
        ", after shrink: ", should_shrink(), '\n'
      );
    } else {
      compact();
    }
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
    return states_[pos];
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
  inline psl_t psl(const key_type& key) const noexcept {
    psl_t psl;
    find_(key, psl, key_to_hash(key));
    return psl;
  }

  inline std::array<size_t, kernel_type::size> state_collisions() const {
    state_t* const __restrict states{states_.get()};
    std::array<size_t, kernel_type::size> res{};

    const raw_pos_t end{capacity()};
    for (raw_pos_t off{}; off != end; off += kernel_type::size) {
      const auto kern{kernel_type::load(&states[off])};

      std::array<state_t, kernel_type::size> tmp;
      size_t n{};

      auto mask{kernel_type::mask_hash(kern)};
      for (; mask_type::has_next(mask); mask = mask_type::step(mask)) {
        const size_t i{mask_type::next(mask)};
        NVHM_ASSERT_(n < kernel_type::size);
        tmp[n++] = kernel_type::at(kern, i);
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
  inline bool erase_(const key_type& __restrict k, psl_t& __restrict psl, const hash_t h) {
    NVHM_ASSERT_(key_to_hash(k) == h, "Supplied key and hash do not match!");
    const size_t bucket_mask{bucket_mask_};
    state_t* const __restrict states{states_.get()};
    const key_type* const __restrict keys{keys_.get()};

    const state_t s{hash_to_state(h)};

    raw_pos_t seq{hash_to_pos<kernel_type>(h, bucket_mask)};
    for (psl = {}; probe_seq_type::has_next(psl);) {
      NVHM_ASSERT_(
        psl != static_cast<psl_t>(capacity()), "The table is full. This should not happen!"
      );

      const raw_pos_t off{probe_seq_type::next(seq, psl, bucket_mask)};
      psl += kernel_type::size;
      const auto kern{kernel_type::load(&states[off])};

      // Is the key present in this bucket?
      auto mask{kernel_type::mask(kern, s)};
      for (; mask_type::has_next(mask); mask = mask_type::step(mask)) {
        const raw_pos_t pos{mask_type::next(mask, off)};
        if (NVHM_UNLIKELY_(keys[pos] != k)) continue;

        bool can_reclaim;
        if constexpr (allow_reclaim) {
          can_reclaim = kernel_type::has_empty(kern);
        } else {
          can_reclaim = false;
        }
        release_slot_(states[pos], can_reclaim);

        if constexpr (auto_shrink) {
          if (NVHM_UNLIKELY_(should_shrink())) {
            NVHM_DBG_LOG_(
              "should shrink, size/capacity:", size(), '/', capacity(),
              ", before shrink: ", should_shrink(), '\n'
            );
            realloc_(std::max(capacity() >> 1, min_capacity));
            NVHM_DBG_LOG_(
              "should shrink, size/capacity:", size(), '/', capacity(),
              ", after shrink: ", should_shrink(), '\n'
            );
          }
        }

        return true;
      }

      // Empty slots mark the end of a probe sequence.
      if (NVHM_LIKELY_(kernel_type::has_empty(kern))) break;
      seq = probe_seq_type::step(seq, psl, bucket_mask);
    }

    return false;
  }

  inline bool erase_at_(const raw_pos_t pos) {
    NVHM_ASSERT_(pos < capacity(), "`pos` is out of bounds");
    state_t* const __restrict states{states_.get()};

    bool can_reclaim;
    if constexpr (allow_reclaim) {
      const raw_pos_t off{pos & ~kernel_type::size_mask};
      const auto kern{kernel_type::load(&states[off])};

      // Can reclaim if have an empty slot is nearby.
      if (slot_not_occupied(kernel_type::at(kern, pos - off))) return false;
      can_reclaim = kernel_type::has_empty(kern);
    } else {
      if (slot_not_occupied(states[pos])) return false;
      can_reclaim = false;
    }
    release_slot_(states[pos], can_reclaim);

    if constexpr (auto_shrink) {
      if (should_shrink()) {
        NVHM_DBG_LOG_(
          "should shrink, size/capacity:", size(), '/', capacity(),
          ", before shrink: ", should_shrink(), '\n'
        );
        realloc_(capacity() > 1);
        NVHM_DBG_LOG_(
          "should shrink, size/capacity:", size(), '/', capacity(),
          ", after shrink: ", should_shrink(), '\n'
        );
      }
    }
    return true;
  }

  inline raw_pos_t find_(
    const key_type& __restrict k, psl_t& __restrict psl, const hash_t h
  ) const noexcept {
    NVHM_ASSERT_(key_to_hash(k) == h, "Supplied key and hash do not match!");
    const size_t bucket_mask{bucket_mask_};
    const state_t* const __restrict states{states_.get()};
    const key_type* const __restrict keys{keys_.get()};

    const state_t s{hash_to_state(h)};

    raw_pos_t seq{hash_to_pos<kernel_type>(h, bucket_mask)};
    for (psl = {}; probe_seq_type::has_next(psl);) {
      NVHM_ASSERT_(
        psl != static_cast<psl_t>(capacity()), "The table is full. This should not happen!"
      );

      const raw_pos_t off{probe_seq_type::next(seq, psl, bucket_mask)};
      psl += kernel_type::size;
      const auto kern{kernel_type::load(&states[off])};

      // if constexpr (kernel_type::size * sizeof(key_type) <= cache_line_size) {
      //   // TODO: Is this still necessary if we use explicit prefetch?
      //   // TODO: The question is, how optimistic are we are bout the PSL and the success chance.
      //   nvhm::read_prefetch<kernel_type::size * sizeof(key_type)>(&keys[off]);
      // }

      // Is the key present in this bucket?
      auto mask{kernel_type::mask(kern, s)};
      for (; mask_type::has_next(mask); mask = mask_type::step(mask)) {
        const raw_pos_t pos{mask_type::next(mask, off)};
        if (NVHM_LIKELY_(keys[pos] == k)) {
          return pos;
        }
      }

      // Empty slots mark the end of a probe sequence.
      if (NVHM_LIKELY_(kernel_type::has_empty(kern))) break;
      seq = probe_seq_type::step(seq, psl, bucket_mask);
    }

    return npos;
  }

  template <typename It>
  inline raw_pos_t insert_(
    const key_type& __restrict k, psl_t& __restrict psl, It& __restrict it, const hash_t h
  ) {
    NVHM_ASSERT_(key_to_hash(k) == h, "Supplied key and hash do not match!");

    // Grow the table if there are on average less than 2 empty spaces per segment.
    if (NVHM_UNLIKELY_(should_grow())) {
      size_t new_capacity{capacity()};
      NVHM_DBG_LOG_(
        "should grow, size/capacity:", size(), '/', new_capacity, ", should grow: ", should_grow(),
        ", should compact: ", should_compact(), '\n'
      );
      if (NVHM_LIKELY_(!should_compact())) {
        new_capacity <<= 1;
        new_capacity <<= new_capacity < 1024 * 1024;
      }
      realloc_(new_capacity);
      NVHM_DBG_LOG_(
        "after grow, size/capacity:", size(), '/', capacity(), "(desired: ", new_capacity,
        "), should grow: ", should_grow(), ", should compact: ", should_compact(), '\n'
      );
      NVHM_ASSERT_(!should_grow());
    }

    while (true) {
      const size_t bucket_mask{bucket_mask_};
      state_t* const __restrict states{states_.get()};
      key_type* const __restrict keys{keys_.get()};

      const state_t s{hash_to_state(h)};
      raw_pos_t tomb_pos{npos};

      raw_pos_t seq{hash_to_pos<kernel_type>(h, bucket_mask)};
      for (psl = {}; probe_seq_type::has_next(psl);) {
        NVHM_ASSERT_(
          psl != static_cast<psl_t>(capacity()), "The table is full. This should not happen!"
        );

        const raw_pos_t off{probe_seq_type::next(seq, psl, bucket_mask)};
        psl += kernel_type::size;
        const auto kern{kernel_type::load(&states[off])};

        // Is the key present in this bucket?
        auto mask{kernel_type::mask(kern, s)};
        for (; mask_type::has_next(mask); mask = mask_type::step(mask)) {
          const raw_pos_t pos{mask_type::next(mask, off)};
          if (NVHM_LIKELY_(keys[pos] == k)) {
            return pos;
          }
        }

        // Empty slots mark the end of a probe sequence.
        mask = kernel_type::mask_empty(kern);
        if (NVHM_LIKELY_(mask_type::has_next(mask))) {
          raw_pos_t pos;
          if constexpr (minimize_psl) {
            if (tomb_pos == npos) {
              pos = mask_type::next(mask, off);
            } else {
              pos = tomb_pos;
              --num_tombstones_;
            }
          } else {
            pos = mask_type::next(mask, off);
          }

          states[pos] = s;
          keys[pos] = k;
          ++num_used_;
          if constexpr (!std::is_same_v<It, std::monostate>) {
            ++it;
          }
          return pos;
        }

        // Remember location of the first tombstone.
        if constexpr (minimize_psl) {
          if (tomb_pos == npos) {
            mask = kernel_type::mask_non_hash(kern);
            if (mask_type::has_next(mask)) {
              tomb_pos = mask_type::next(mask, off);
            }
          }
        }

        seq = probe_seq_type::step(seq, psl, bucket_mask);
      }

      // Probe sequence limit reached. Increase table size and try again.
      realloc_((bucket_mask + kernel_type::size) << 1);
    }
  }

  void realloc_(size_t new_capacity) {
    const size_t raw_value_size{raw_value_size_};
    const size_t raw_value_pitch{raw_value_pitch_};

    const size_t old_bucket_mask{bucket_mask_};
    const size_t old_capacity{old_bucket_mask + kernel_type::size};
    aligned_unique_ptr<state_t[]> old_states{std::move(states_)};
    const size_t old_size{size()};
    aligned_unique_ptr<key_type[]> old_keys{std::move(keys_)};
    aligned_unique_ptr<value_type[]> old_values{std::move(values_)};
    aligned_unique_ptr<raw_value_type[]> old_raw_values{std::move(raw_values_)};

    for (;; new_capacity <<= 1) {
      __label__ on_success, on_error;
      NVHM_ASSERT_(
        new_capacity > size(), "old_capacity = ", old_capacity, ", new_capacity = ", new_capacity,
        ", size() = ", size()
      );

      reset_(new_capacity);
      const size_t bucket_mask{bucket_mask_};
      state_t* const __restrict states{states_.get()};
      key_type* const __restrict keys{keys_.get()};
      value_type* const __restrict values{values_.get()};
      raw_value_type* const __restrict raw_values{raw_values_.get()};

      // Insert all KV-pairs into the new container.
      for (raw_pos_t off{}; off != old_capacity; off += kernel_type::size) {
        const auto kern{kernel_type::load(&old_states[off])};

        auto mask{kernel_type::mask_hash(kern)};
        for (; mask_type::has_next(mask); mask = mask_type::step(mask)) {
          const raw_pos_t old_pos{mask_type::next(mask, off)};
          const key_type& __restrict k{old_keys[old_pos]};

          const hash_t h{key_to_hash(k)};

          raw_pos_t seq{hash_to_pos<kernel_type>(h, bucket_mask)};
          for (psl_t psl{}; probe_seq_type::has_next(psl);) {
            NVHM_ASSERT_(
              psl != static_cast<psl_t>(capacity()), "The table is full. This should not happen!"
            );

            const raw_pos_t off{probe_seq_type::next(seq, psl, bucket_mask)};
            psl += kernel_type::size;
            const auto kern{kernel_type::load(&states[off])};

            auto mask{kernel_type::mask_non_hash(kern)};
            for (; NVHM_LIKELY_(mask_type::has_next(mask)); mask = mask_type::step(mask)) {
              const raw_pos_t pos{mask_type::next(mask, off)};
              states[pos] = hash_to_state(h);
              keys[pos] = k;
#ifndef NDEBUG
              ++num_used_;
#endif

              if constexpr (has_values) {
                values[pos] = old_values[old_pos];
              }
              fast_copy(
                &raw_values[pos * raw_value_pitch], &old_raw_values[old_pos * raw_value_pitch],
                raw_value_size
              );
              goto on_success;
            }

            seq = probe_seq_type::step(seq, psl, bucket_mask);
          }

          goto on_error;
        on_success:;
        }
      }

      break;
    on_error:;
    }

    NVHM_ASSERT_(num_used_ == old_size, "num_used_ = ", num_used_, ", old_size = ", old_size);
    num_used_ = old_size;
  }

  inline void release_slot_(state_t& __restrict state, const bool reclaim) noexcept {
    NVHM_ASSERT_(slot_is_occupied(state));
    if (reclaim) {
      state = kernel_type::empty;
      --num_used_;
    } else {
      state = kernel_type::tombstone;
      ++num_tombstones_;
    }
  }

  inline void reset_(const size_t capacity) {
    NVHM_ASSERT_(capacity >= min_capacity && has_single_bit(capacity));

    // Wrap around at capacity and align at kernel size.
    bucket_mask_ = capacity - kernel_type::size;
    growth_threshold_ = capacity - capacity * reserve_capacity_bias_ / min_capacity;

    states_ = allocator_type::template alloc<state_t>(capacity);
    std::fill_n(states_.get(), capacity, kernel_type::empty);
    num_used_ = {};
    num_tombstones_ = {};

    keys_ = allocator_type::template alloc<key_type>(capacity);
    if constexpr (has_values) {
      values_ = allocator_type::template alloc<value_type>(capacity);
    }
    raw_values_ = allocator_type::template alloc<raw_value_type>(capacity * raw_value_pitch_);
  }

 protected:
  size_t raw_value_size_;
  size_t raw_value_alignment_;
  size_t raw_value_pitch_;
  size_t reserve_capacity_bias_;

  size_t bucket_mask_;
  size_t growth_threshold_;
  aligned_unique_ptr<state_t[]> states_;
  size_t num_used_;
  size_t num_tombstones_;

  aligned_unique_ptr<key_type[]> keys_;
  aligned_unique_ptr<value_type[]> values_;
  aligned_unique_ptr<raw_value_type[]> raw_values_;
};

}  // namespace nvhm