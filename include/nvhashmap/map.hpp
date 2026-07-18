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

#include "swiss_map_base.hpp"
#include "probe_seq.hpp"
#include "conf.hpp"

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
  typename Key, typename Value = void,
  flags_t Flags = flags_t::none, typename Kernel = default_kernel_t<>,
  typename ProbeSeq = default_seq_t, typename Allocator = default_allocator_t>
class map : public swiss_map_base<
  map<Key, Value, Flags, Kernel, ProbeSeq, Allocator>,
  conf<Key, Value, Flags, Kernel::size>, ProbeSeq, Allocator
> {
 public:
  using base_type = swiss_map_base<map, conf<Key, Value, Flags, Kernel::size>, ProbeSeq, Allocator>;

  using self_type = typename base_type::self_type;
  using conf_type = typename base_type::conf_type;

  using key_type = typename base_type::key_type;
  using value_type = typename base_type::value_type;
  using base_type::has_values;
  using base_type::flags;
  using base_type::has_blobs;
  using base_type::kernel_size;
  
  using const_entry_type = typename base_type::const_entry_type;
  using entry_type = typename base_type::entry_type;
  using const_mapped_type = typename base_type::const_mapped_type;
  using mapped_type = typename base_type::mapped_type;
  
  using probe_seq_type = typename base_type::probe_seq_type;
  using allocator_type = typename base_type::allocator_type;

  using kernel_type = Kernel;
  static_assert(kernel_type::size == kernel_size, "Provided `kernel::size` configured `kernel_size` do not match!");
  using kernel_repr_type = typename kernel_type::repr_type;
  using mask_type = typename kernel_type::mask_type;

  constexpr map(const map& other) : base_type{other} {
    grow_threshold_ = other.grow_threshold_;
    states_ = make_copy(other.states_, capacity());
    num_empty_ = other.num_empty_;
    num_tombstone_ = other.num_tombstone_;
  }
  constexpr map& operator=(const map& __restrict other) {
    if (self() == &other) return *self();
    base_type::operator=(other);

    const raw_pos_t end{capacity()};
    NVHM_ASSERT_(end == other.capacity(), "end = ", end, ", other.capacity = ", other.capacity());

    grow_threshold_ = other.grow_threshold_;
    std::copy_n(other.states_.get(), end, states_.get());
    num_empty_ = other.num_empty_;
    num_tombstone_ = other.num_tombstone_;
    return *self();
  }
  constexpr map(map&& other) noexcept = default;
  constexpr map& operator=(map&& other) noexcept = default;

  constexpr map() : map{conf_type()} {}
  constexpr map(int_t init_capacity) : map{conf_type().set_capacity(init_capacity)} {}
  constexpr map(const conf_type& conf) : base_type{conf} {
    const raw_pos_t end{capacity()};
    grow_threshold_ = conf.grow_threshold(end);
    states_ = make_unique<state_t, allocator_type>(end);
    reset_();
  }

  using base_type::capacity;
  using base_type::self;
  using base_type::size;
  
  /**
   * Returns the current growth limit of the map.
   *
   * @return The limit beyond which the next growth can happen.
   */
  constexpr int_t grow_threshold() const noexcept { return grow_threshold_; }

  constexpr friend void swap(map& lhs, map& rhs) noexcept {
    if (&lhs == &rhs) return;
    swap(static_cast<base_type&>(lhs), static_cast<base_type&>(rhs));

    std::swap(lhs.grow_threshold_, rhs.grow_threshold_);
    std::swap(lhs.states_, rhs.states_);
    std::swap(lhs.num_empty_, rhs.num_empty_);
    std::swap(lhs.num_tombstone_, rhs.num_tombstone_);
  }

 protected:
  template <typename, typename, typename, typename>
  friend class raw_map_base;
  template <typename, typename, typename, typename> 
  friend class swiss_map_base;

  using base_type::conf_;
  using base_type::bucket_mask_;
  using base_type::values_;
  using base_type::blobs_;
  using base_type::keys_;
  int_t grow_threshold_;
  unique_ptr_t<state_t[], allocator_type> states_;
  int_t num_empty_;
  int_t num_tombstone_;

  using base_type::contains_at_;

  constexpr raw_pos_t alloc_(raw_pos_t end) {
    end = base_type::alloc_(end);

    grow_threshold_ = conf_.grow_threshold(end);
    states_ = make_unique<state_t, allocator_type>(end);
    return end;
  }

  constexpr int_t bucket_size_at_(const raw_pos_t pos) const noexcept {
    NVHM_ASSERT_(pos >= 0 && pos < capacity(), "pos = ", pos, ", capacity = ", capacity());
    const state_t* const __restrict states{states_.get()};

    const raw_pos_t off{align_pos<~kernel_type::size_mask>(pos)};
    const auto k{kernel_type::load(&states[off])};
    // TODO: Add optimized kernel function for this?
    return mask_type::count(kernel_type::mask_hash(k));
  }

  constexpr int_t bucket_num_empty_slots_at_(const raw_pos_t pos) const noexcept {
    NVHM_ASSERT_(pos >= 0 && pos < capacity(), "pos = ", pos, ", capacity = ", capacity());
    const state_t* const __restrict states{states_.get()};

    const raw_pos_t off{align_pos<~kernel_type::size_mask>(pos)};
    const auto k{kernel_type::load(&states[off])};
    // TODO: Add optimized kernel function for this?
    return mask_type::count(kernel_type::mask_empty(k));
  }

  constexpr int_t bucket_num_tombstone_slots_at_(const raw_pos_t pos) const noexcept {
    NVHM_ASSERT_(pos >= 0 && pos < capacity(), "pos = ", pos, ", capacity = ", capacity());
    const state_t* const __restrict states{states_.get()};

    const raw_pos_t off{align_pos<~kernel_type::size_mask>(pos)};
    const auto k{kernel_type::load(&states[off])};
    // TODO: Add optimized kernel function for this?
    return mask_type::count(kernel_type::mask_tombstone(k));
  }

  constexpr bool check_integrity_() const noexcept {
    const int_t end{capacity()};
    const state_t* const __restrict states{states_.get()};
    
    int_t num_empty{};
    int_t num_tombstone{};

    for (raw_pos_t off{}; off < end; off += kernel_size) {
      const auto k{kernel_type::load(&states[off])};
      
      // TODO: Add optimized kernel function for this?
      num_empty += mask_type::count(kernel_type::mask_empty(k));
      num_tombstone += mask_type::count(kernel_type::mask_tombstone(k));
    }

    NVHM_ASSERT_(num_empty == num_empty_, "num_empty = ", num_empty, ", num_empty_ = ", num_empty_);
    NVHM_ASSERT_(num_tombstone == num_tombstone_, "num_tombstone = ", num_tombstone, ", num_tombstone_ = ", num_tombstone_);
    return num_empty == num_empty_ && num_tombstone == num_tombstone_;
  }

  constexpr int_t clear_() {
    if constexpr (test_flags(flags, flags_t::auto_shrink)) {
      int_t end{conf_.init_capacity()};
      if (end != capacity()) {
        end = alloc_(end);
      }
      NVHM_ASSERT_(end == capacity(), "end = ", end, ", capacity = ", capacity());
    }
    return reset_();
  }

  constexpr void count_kernel_populations_(std::array<int_t, kernel_size + 1>& counts) const noexcept {
    const raw_pos_t end{capacity()};
    const state_t* const __restrict states{states_.get()};

    for (raw_pos_t off{}; off < end; off += kernel_size) {
      auto k{kernel_type::load(&states[off])};
      auto m{kernel_type::mask_hash(k)};

      ++counts[to_uint(mask_type::count(m))];
    }
  }

  constexpr void count_state_collisions_(std::array<int_t, kernel_size>& counts) const noexcept {
    const raw_pos_t end{capacity()};
    const state_t* const __restrict states{states_.get()};

    for (raw_pos_t off{}; off < end; off += kernel_size) {
      count_collisions<kernel_type>(&states[off], counts);
    }
  }

  constexpr std::tuple<bool, raw_pos_t, probe_seq_type> erase_first_(const key_type& __restrict key, const hash_t hash) {
    NVHM_ASSERT_(key_to_hash(key) == hash, "Supplied key and hash do not match!");

    const bitmask_t bucket_mask{bucket_mask_};
    const key_type* const __restrict keys{keys_.get()};
    state_t* const __restrict states{states_.get()};

    const state_t s{hash_to_state(hash)};
    const raw_pos_t end{aligned_mask_to_capacity<kernel_size>(bucket_mask)};

    probe_seq_type seq{hash_to_pos<kernel_size>(hash)};
    for (; seq.has_next(); seq += kernel_size) {
      NVHM_ASSERT_(seq.psl() < end, "The table is full. This should not happen! (psl = ", seq.psl(), ", capacity = ", end, ')');

      const raw_pos_t off{align_pos(seq.next(), bucket_mask)};
      const auto k{kernel_type::load(&states[off])};

      // Is the key present in this bucket?
      for (auto m{kernel_type::mask(k, s)}; mask_type::has_next(m); m = mask_type::step(m)) {
        raw_pos_t pos{mask_type::next(m, off)};
        if (NVHM_UNLIKELY_(keys[pos] != key)) continue;
        
        pos = reset_slot_(end, states[pos], kernel_type::has_empty(k)) ? pos : npos;
        return {true, pos, std::move(seq)};
      }

      // Empty slots mark the end of a probe sequence.
      if (NVHM_LIKELY_(kernel_type::has_empty(k))) break;
    }

    return {false, npos, std::move(seq)};
  }

  template <typename PS>
  NVHM_ALWAYS_INLINE constexpr std::tuple<bool, raw_pos_t, probe_seq_type> erase_next_(raw_pos_t pos, PS&& __restrict seq, const key_type& __restrict key, const hash_t hash) {
    static_assert(std::is_same_v<remove_cvref_t<PS>, probe_seq_type>, "`PS` must be `probe_seq_type`!");
    NVHM_ASSERT_(key_to_hash(key) == hash, "Supplied key and hash do not match!");

    if constexpr (!test_flags(flags, flags_t::duplicates)) {
      return {false, npos, std::move(seq)};
    }

    const bitmask_t bucket_mask{bucket_mask_};
    NVHM_ASSERT_(align_pos(seq.psl(), bucket_mask) == seq.psl(), "psl = ", seq.psl(), ", bucket_mask = ", std::hex, bucket_mask, std::dec);
    const key_type* const __restrict keys{keys_.get()};
    state_t* const __restrict states{states_.get()};

    const state_t s{hash_to_state(hash)};
    const raw_pos_t end{aligned_mask_to_capacity<kernel_size>(bucket_mask)};
    NVHM_ASSERT_(pos >= 0 && pos < end, "pos = ", pos, ", capacity = ", end);

    auto [off, idx]{splice_pos<kernel_size>(pos)};
    NVHM_ASSERT_(seq.has_next() && align_pos(seq.next(), bucket_mask) == off);
    NVHM_ASSERT_(seq.psl() < end, "The table is full. This should not happen! (psl = ", seq.psl(), ", capacity = ", end, ')');

    if (idx >= kernel_size - 1) {
      seq += kernel_size;
      if (NVHM_UNLIKELY_(!seq.has_next())) {
        return {false, npos, std::move(seq)};
      }
      NVHM_ASSERT_(seq.psl() < end, "The table is full. This should not happen! (psl = ", seq.psl(), ", capacity = ", end, ')');
      off = align_pos(seq.next(), bucket_mask);
      idx = -1;
    }

    auto k{kernel_type::load(&states[off])};
    auto m{kernel_type::mask(k, s)};
    m = mask_type::above(m, idx);

    while (true) {
      // Is the key present in this bucket?
      for (; mask_type::has_next(m); m = mask_type::step(m)) {
        pos = mask_type::next(m, off);
        if (NVHM_UNLIKELY_(keys[pos] != key)) continue;

        pos = reset_slot_(end, states[pos], kernel_type::has_empty(k)) ? pos : npos;
        return {true, pos, std::move(seq)};
      }

      // Empty slots mark the end of a probe sequence.
      if (NVHM_LIKELY_(kernel_type::has_empty(k))) break;

      seq += kernel_size;
      if (!seq.has_next()) break;
      NVHM_ASSERT_(seq.psl() < end, "The table is full. This should not happen! (psl = ", seq.psl(), ", capacity = ", end, ')');

      off = align_pos(seq.next(), bucket_mask);
      k = kernel_type::load(&states[off]);
      m = kernel_type::mask(k, s);
    }

    return {false, npos, std::move(seq)};
  }

  NVHM_ALWAYS_INLINE constexpr std::pair<bool, raw_pos_t> erase_at_(raw_pos_t pos) {
    NVHM_ASSERT_(pos >= 0 && pos < capacity(), "pos = ", pos, ", capacity = ", capacity());
    const raw_pos_t end{capacity()};
    state_t* const __restrict states{states_.get()};

    const auto [off, idx]{splice_pos<kernel_size>(pos)};
    
    const auto k{kernel_type::load(&states[off])};
    if (NVHM_LIKELY_(is_hash(kernel_type::at(k, idx)))) {
      pos = reset_slot_(end, states[pos], kernel_type::has_empty(k)) ? pos : npos;
      return {true, pos};
    }
  
    return {false, pos};
  }

  template <typename Pred>
  NVHM_ALWAYS_INLINE constexpr int_t erase_if_(const Pred& __restrict pred) {
    static_assert(std::is_invocable_r_v<bool, Pred, raw_pos_t>, "`pred` must be pred(raw_pos_t) -> bool");

    const raw_pos_t end{capacity()};
    state_t* const __restrict states{states_.get()};

    int_t num_empty{num_empty_};
    int_t num_tombstone{num_tombstone_};
    for (raw_pos_t off{}; off < end; off += kernel_size) {
      const auto k{kernel_type::load(&states[off])};

      for (auto m{kernel_type::mask_hash(k)}; mask_type::has_next(m); m = mask_type::step(m)) {
        const raw_pos_t pos{mask_type::next(m, off)};
        if (!pred(pos)) continue;

        if (kernel_type::has_empty(k)) {
          states[pos] = kernel_type::empty;
          ++num_empty;
        } else {
          states[pos] = kernel_type::tombstone;
          ++num_tombstone;
        }
      }
    }

    const int_t num_emptied{num_empty - num_empty_};
    const int_t num_burried{num_tombstone - num_tombstone_};

    const int_t num_erased{num_emptied + num_burried};
    if (NVHM_LIKELY_(num_erased)) {
      num_empty_ = num_empty;
      num_tombstone_ = num_tombstone;
      NVHM_ASSERT_(check_integrity_());

      auto_shrink_and_scub_(end, num_emptied, num_burried);
    }
    return num_erased;
  }

  NVHM_ALWAYS_INLINE constexpr int_t erase_all_(const key_type& __restrict key, const hash_t hash) {
    NVHM_ASSERT_(key_to_hash(key) == hash, "Supplied key and hash do not match!");

    if constexpr (!test_flags(flags, flags_t::duplicates)) {
      return std::get<0>(erase_first_(key, hash));
    }

    const bitmask_t bucket_mask{bucket_mask_};
    const key_type* const __restrict keys{keys_.get()};
    state_t* const __restrict states{states_.get()};

    const state_t s{hash_to_state(hash)};

    int_t num_empty{num_empty_};
    int_t num_tombstone{num_tombstone_};
    probe_seq_type seq{hash_to_pos<kernel_size>(hash)};
    for (; seq.has_next(); seq += kernel_size) {
      NVHM_ASSERT_(seq.psl() < capacity(), "The table is full. This should not happen! (psl = ", seq.psl(), ", capacity = ", capacity(), ')');

      const raw_pos_t off{align_pos(seq.next(), bucket_mask)};
      const auto k{kernel_type::load(&states[off])};

      // Is the key present in this bucket?
      for (auto m{kernel_type::mask(k, s)}; mask_type::has_next(m); m = mask_type::step(m)) {
        const raw_pos_t pos{mask_type::next(m, off)};
        if (NVHM_UNLIKELY_(keys[pos] != key)) continue;

        if (kernel_type::has_empty(k)) {
          states[pos] = kernel_type::empty;
          ++num_empty;
        } else {
          states[pos] = kernel_type::tombstone;
          ++num_tombstone;
        }
      }

      // Empty slots mark the end of a probe sequence.
      if (NVHM_LIKELY_(kernel_type::has_empty(k))) break;
    }

    const int_t num_emptied{num_empty - num_empty_};
    const int_t num_burried{num_tombstone - num_tombstone_};

    const int_t num_erased{num_emptied + num_burried};
    if (NVHM_LIKELY_(num_erased)) {
      num_empty_ = num_empty;
      num_tombstone_ = num_tombstone;
      NVHM_ASSERT_(check_integrity_());

      const raw_pos_t end{aligned_mask_to_capacity<kernel_size>(bucket_mask)};
      auto_shrink_and_scub_(end, num_emptied, num_burried);
    }
    return num_erased;
  }

  NVHM_ALWAYS_INLINE constexpr std::pair<raw_pos_t, probe_seq_type> find_first_(const key_type& __restrict key, const hash_t hash) const {
    NVHM_ASSERT_(key_to_hash(key) == hash, "Supplied key and hash do not match!");

    const bitmask_t bucket_mask{bucket_mask_};
    const key_type* const __restrict keys{keys_.get()};
    const state_t* const __restrict states{states_.get()};

    const state_t s{hash_to_state(hash)};

    probe_seq_type seq{hash_to_pos<kernel_size>(hash)};
    for (; seq.has_next(); seq += kernel_size) {
      NVHM_ASSERT_(seq.psl() < capacity(), "The table is full. This should not happen! (psl = ", seq.psl(), ", capacity = ", capacity(), ')');

      const raw_pos_t off{align_pos(seq.next(), bucket_mask)};
      const auto k{kernel_type::load(&states[off])};

      // if constexpr (kernel_size * sizeof(key_type) <= cache_line_size) {
      //   // TODO: Is this still necessary if we use explicit prefetch?
      //   // TODO: The question is, how optimistic are we are about the PSL and the success chance.
      //   // TODO: Also, should we prefetch more than one cache line here?
      //   nvhm::read_prefetch<kernel_size>(&keys[off]);
      // }

      // Is the key present in this bucket?
      for (auto m{kernel_type::mask(k, s)}; mask_type::has_next(m); m = mask_type::step(m)) {
        const raw_pos_t pos{mask_type::next(m, off)};
        if (NVHM_UNLIKELY_(keys[pos] != key)) continue;

        return {pos, std::move(seq)};
      }

      // Empty slots mark the end of a probe sequence.
      if (NVHM_LIKELY_(kernel_type::has_empty(k))) break;
    }

    return {npos, std::move(seq)};
  }

  template <typename PS>
  NVHM_ALWAYS_INLINE constexpr std::pair<raw_pos_t, probe_seq_type> find_next_(raw_pos_t pos, PS&& __restrict seq, const key_type& __restrict key, const hash_t hash) const {
    static_assert(std::is_same_v<remove_cvref_t<PS>, probe_seq_type>, "`PS` must be `probe_seq_type`!");
    NVHM_ASSERT_(key_to_hash(key) == hash, "Supplied key and hash do not match!");

    if constexpr (!test_flags(flags, flags_t::duplicates)) {
      return {npos, std::move(seq)};
    }

    const bitmask_t bucket_mask{bucket_mask_};
    const key_type* const __restrict keys{keys_.get()};
    const state_t* const __restrict states{states_.get()};

    const state_t s{hash_to_state(hash)};
    const raw_pos_t end{aligned_mask_to_capacity<kernel_size>(bucket_mask)};
    NVHM_ASSERT_(pos >= 0 && pos < end, "pos = ", pos, ", capacity = ", end);

    auto [off, idx]{splice_pos<kernel_size>(pos)};
    NVHM_ASSERT_(seq.has_next() && align_pos(seq.next(), bucket_mask) == off);
    NVHM_ASSERT_(seq.psl() < end, "The table is full. This should not happen! (psl = ", seq.psl(), ", capacity = ", end, ')');

    if (idx >= kernel_size - 1) {
      seq += kernel_size;
      if (NVHM_UNLIKELY_(!seq.has_next())) {
        return {npos, std::move(seq)};
      }
      NVHM_ASSERT_(seq.psl() < end, "The table is full. This should not happen! (psl = ", seq.psl(), ", capacity = ", end, ')');
      off = align_pos(seq.next(), bucket_mask);
      idx = -1;
    }

    auto k{kernel_type::load(&states[off])};
    auto m{kernel_type::mask(k, s)};
    m = mask_type::above(m, idx);

    while (true) {
      // Is the key present in this bucket?
      for (; mask_type::has_next(m); m = mask_type::step(m)) {
        pos = mask_type::next(m, off);
        if (NVHM_UNLIKELY_(keys[pos] != key)) continue;

        return {pos, std::move(seq)};
      }

      // Empty slots mark the end of a probe sequence.
      if (NVHM_LIKELY_(kernel_type::has_empty(k))) break;
      
      seq += kernel_size;
      if (!seq.has_next()) break;
      NVHM_ASSERT_(seq.psl() < capacity(), "The table is full. This should not happen! (psl = ", seq.psl(), ", capacity = ", end, ')');

      off = align_pos(seq.next(), bucket_mask);
      k = kernel_type::load(&states[off]);
      m = kernel_type::mask(k, s);
    }

    return {npos, std::move(seq)};
  }

  template <typename Pred>
  NVHM_ALWAYS_INLINE constexpr raw_pos_t find_if_(const Pred& __restrict pred) const {
    static_assert(std::is_invocable_r_v<bool, Pred, raw_pos_t>, "`pred` must be pred(raw_pos_t) -> bool");

    const raw_pos_t end{capacity()};
    const state_t* const __restrict states{states_.get()};

    for (raw_pos_t off{}; off < end; off += kernel_size) {
      const auto k{kernel_type::load(&states[off])};

      for (auto m{kernel_type::mask_hash(k)}; mask_type::has_next(m); m = mask_type::step(m)) {
        const raw_pos_t pos{mask_type::next(m, off)};
        if (pred(pos)) return pos;
      }
    }
    return npos;
  }

  template <typename Func>
  NVHM_ALWAYS_INLINE constexpr void for_each_(const Func& __restrict func) const {
    const raw_pos_t end{capacity()};
    const state_t* const __restrict states{states_.get()};

    for (raw_pos_t off{}; off < end; off += kernel_size) {
      const auto k{kernel_type::load(&states[off])};

      for (auto m{kernel_type::mask_hash(k)}; mask_type::has_next(m); m = mask_type::step(m)) {
        func(mask_type::next(m, off));
      }
    }
  }

  template <typename Func>
  NVHM_ALWAYS_INLINE constexpr void for_each_(const key_type& __restrict key, const hash_t hash, const Func& __restrict func) const {
    static_assert(std::is_invocable_v<Func, raw_pos_t, probe_seq_type>, "`func` must be `func(raw_pos_t, probe_seq_type)`!");
    NVHM_ASSERT_(key_to_hash(key) == hash, "Supplied key and hash do not match!");

    const bitmask_t bucket_mask{bucket_mask_};
    const key_type* const __restrict keys{keys_.get()};
    const state_t* const __restrict states{states_.get()};

    const state_t s{hash_to_state(hash)};

    probe_seq_type seq{hash_to_pos<kernel_size>(hash)};
    for (; seq.has_next(); seq += kernel_size) {
      NVHM_ASSERT_(seq.psl() < capacity(), "The table is full. This should not happen! (psl = ", seq.psl(), ", capacity = ", capacity(), ')');

      const raw_pos_t off{align_pos(seq.next(), bucket_mask)};
      const auto k{kernel_type::load(&states[off])};

      // Is the key present in this bucket?
      for (auto m{kernel_type::mask(k, s)}; mask_type::has_next(m); m = mask_type::step(m)) {
        const raw_pos_t pos{mask_type::next(m, off)};
        if (NVHM_UNLIKELY_(keys[pos] != key)) continue;

        func(pos, seq);
        if constexpr (!test_flags(flags, flags_t::duplicates)) {
          return;
        }
      }

      // Empty slots mark the end of a probe sequence.
      if (NVHM_LIKELY_(kernel_type::has_empty(k))) break;
    }
  }

  template <typename Func>
  NVHM_ALWAYS_INLINE constexpr void for_each_lru_(const Func& __restrict func) const {
    const raw_pos_t end{capacity()};
    const state_t* const __restrict states{states_.get()};

    for (raw_pos_t off{}; off < end; off += kernel_size) {
      const auto k{kernel_type::load(&states[off])};

      for (auto m{kernel_type::mask_hash(k)}; mask_type::has_next(m); m = mask_type::step(m)) {
        func(max_lru);
      }
    }
  }

  template <typename Func>
  NVHM_ALWAYS_INLINE constexpr void for_each_state_(const Func& __restrict func) const {
    const raw_pos_t end{capacity()};
    const state_t* const __restrict states{states_.get()};

    for (raw_pos_t off{}; off < end; off += kernel_size) {
      const auto k{kernel_type::load(&states[off])};

      for (auto m{kernel_type::mask_hash(k)}; mask_type::has_next(m); m = mask_type::step(m)) {
        const int_t idx{mask_type::next(m)};
        func(kernel_type::at(k, idx));
      }
    }
  }

  template <typename K>
  NVHM_ALWAYS_INLINE constexpr std::tuple<raw_pos_t, probe_seq_type, insert_op_t> insert_(K&& __restrict key, const hash_t hash) {
    static_assert(std::is_same_v<remove_cvref_t<K>, key_type>, "K must be `key_type`!");
    if constexpr (std::is_floating_point_v<key_type>) {
      if (key != key) {
        throw std::runtime_error("Provided key is not a discernable floating point value!");
      }
    }
    NVHM_ASSERT_(key_to_hash(key) == hash, "Supplied key and hash do not match!");

    // Preemptively grow table if there is not enough reserve space left.
    raw_pos_t end{capacity()};
    if (NVHM_UNLIKELY_(is_full_())) {
      end = resize_("Preemptive Grow", end << 1);
      NVHM_ASSERT_(!is_full_());
    }
    
    const state_t s{hash_to_state(hash)};
    while (true) {
      const bitmask_t bucket_mask{make_aligned_mask<kernel_size>(end)};
      key_type* const __restrict keys{keys_.get()};
      state_t* const __restrict states{states_.get()};

      probe_seq_type seq{hash_to_pos<kernel_size>(hash)};
      if (num_tombstone_) {
        // We have tombstones. And we want to consume them if possible.
        raw_pos_t ins_pos{npos};

        for (; seq.has_next(); seq += kernel_size) {
          NVHM_ASSERT_(seq.psl() < end, "The table is full. This should not happen! (psl = ", seq.psl(), ", end = ", end, ')');

          const raw_pos_t off{align_pos(seq.next(), bucket_mask)};
          const auto k{kernel_type::load(&states[off])};
          
          if constexpr (!test_flags(flags, flags_t::duplicates)) {
            // Is the key present in this bucket?
            for (auto m{kernel_type::mask(k, s)}; mask_type::has_next(m); m = mask_type::step(m)) {
              const raw_pos_t pos{mask_type::next(m, off)};
              if (NVHM_LIKELY_(keys[pos] != key)) continue;

              return {pos, std::move(seq), insert_op_t::found};
            }
          }

          int_t state_is_empty{};
          if (ins_pos == npos) {
            // Remember first tombstone we encounter.
            auto m{kernel_type::mask_tombstone(k)};
            if (mask_type::has_next(m)) {
              ins_pos = mask_type::next(m, off);
              NVHM_ASSERT_(!kernel_type::has_empty(k), "Table corrupted!");
              continue;
            }

            // Empty slot marks the end of a probe sequence. This `ins_pos` is the best ideal position.
            m = kernel_type::mask_not_hash(k);
            if (NVHM_UNLIKELY_(!mask_type::has_next(m))) continue;

            ins_pos = mask_type::next(m, off);
            state_is_empty = 1;
          } else {
            // Empty slot marks the end of the probe sequence.
            auto m{kernel_type::mask_empty(k)};
            if (NVHM_UNLIKELY_(!mask_type::has_next(m))) continue;
          }
          
          keys[ins_pos] = std::forward<K>(key);
          states[ins_pos] = s;

          // We do this after key to avoid having an inconsistent state if the key's constructor throws.
          num_empty_ -= state_is_empty;
          num_tombstone_ -= 1 - state_is_empty;
          NVHM_ASSERT_(check_integrity_());
          return {ins_pos, std::move(seq), insert_op_t::insert};
        }
      } else {
        // We have ensured that there are no tombstones. So, we can insert into the first available slot.
        for (; seq.has_next(); seq += kernel_size) {
          NVHM_ASSERT_(seq.psl() < end, "The table is full. This should not happen! (psl = ", seq.psl(), ", end = ", end, ')');

          const raw_pos_t off{align_pos(seq.next(), bucket_mask)};
          const auto k{kernel_type::load(&states[off])};
          
          if constexpr (!test_flags(flags, flags_t::duplicates)) {
            // Is the key present in this bucket?
            for (auto m{kernel_type::mask(k, s)}; mask_type::has_next(m); m = mask_type::step(m)) {
              const raw_pos_t pos{mask_type::next(m, off)};
              if (NVHM_LIKELY_(keys[pos] != key)) continue;

              return {pos, std::move(seq), insert_op_t::found};
            }
          }

          // Any non-hash must be an empty slot.
          auto m{kernel_type::mask_not_hash(k)};
          if (NVHM_UNLIKELY_(!mask_type::has_next(m))) continue;

          raw_pos_t pos{mask_type::next(m, off)};

          keys[pos] = std::forward<K>(key);
          states[pos] = s;
          
          --num_empty_;
          NVHM_ASSERT_(check_integrity_());
          return {pos, std::move(seq), insert_op_t::insert};
        }
      }

      // Probe sequence limit reached. Increase table size and try again.
      if constexpr (probe_seq_type::is_bounded) {
        end = resize_("Insert Error", end << 1);
      } else {
        NVHM_ASSERT_(false, "Insert error! This should not happen!");
      }
    }
  }

  NVHM_ALWAYS_INLINE constexpr bool is_full_() const noexcept {
    return conf_.should_grow_precalc(num_empty_, grow_threshold_);
  }

  NVHM_ALWAYS_INLINE constexpr lru_t lru_at_([[maybe_unused]] raw_pos_t pos) const noexcept {
    NVHM_ASSERT_(contains_at_(pos), "pos = ", pos);
    return max_lru;
  }

  NVHM_ALWAYS_INLINE constexpr double max_load_factor_() const noexcept { return conf_.max_load_factor(); }

  NVHM_ALWAYS_INLINE constexpr int_t num_empty_slots_() const noexcept { return num_empty_; }
  NVHM_ALWAYS_INLINE constexpr int_t num_tombstone_slots_() const noexcept { return num_tombstone_; }
  NVHM_ALWAYS_INLINE constexpr int_t num_not_hash_slots_() const noexcept { return num_empty_ + num_tombstone_; }

  NVHM_ALWAYS_INLINE constexpr void read_prefetch_states_(raw_pos_t off) const noexcept {
    NVHM_ASSERT_(off >= 0 && off <= to_int(bucket_mask_), "off = ", off, ", bucket_mask = ", std::hex, bucket_mask_, std::dec);
    nvhm::read_prefetch<kernel_size>(&states_[to_uint(off)]);
  }

  NVHM_ALWAYS_INLINE constexpr raw_pos_t reset_() noexcept {
    const raw_pos_t end{capacity()};

    std::fill_n(states_.get(), end, kernel_type::empty);
    num_empty_ = end;
    num_tombstone_ = 0;
    NVHM_ASSERT_(check_integrity_());

    return end;
  }

  NVHM_ALWAYS_INLINE constexpr bool reset_slot_(raw_pos_t end, state_t& __restrict slot, const bool have_empty_neighbor) {
    NVHM_ASSERT_(is_hash(slot));

    if (NVHM_LIKELY_(have_empty_neighbor)) {
      slot = kernel_type::empty;
      ++num_empty_;
      NVHM_ASSERT_(check_integrity_());

      return auto_shrink_and_scub_(end, 1, 0);
    } else {
      slot = kernel_type::tombstone;
      ++num_tombstone_;
      NVHM_ASSERT_(check_integrity_());

      return auto_shrink_and_scub_(end, 0, 1);
    }
  }

  NVHM_ALWAYS_INLINE constexpr bool auto_shrink_and_scub_(raw_pos_t end, int_t /*num_emptied*/, int_t num_burried) {
    if constexpr (test_flags(flags, flags_t::auto_shrink)) {
      if (NVHM_UNLIKELY_(conf_.should_shrink(end, num_empty_, num_tombstone_))) {
        end = resize_("Auto Shrink", end >> 1);
        return false;
      }
    }

    if constexpr (test_flags(flags, flags_t::auto_scrub)) {
      if (num_burried) {
        if (NVHM_UNLIKELY_(conf_.should_scrub(end, num_tombstone_))) {
          end = scrub_("Auto Scrub", max_lru);
          return false;
        }
      }
    }

    return true;
  }

  NVHM_NO_INLINE raw_pos_t resize_(const char reason[], raw_pos_t end) {
    if (end < conf_.min_capacity()) end = conf_.min_capacity();
    NVHM_ASSERT_(end >= conf_.min_capacity(), "reason = ", reason, ", new_capacity = ", end, ", min_capacity = ", conf_.min_capacity());
    NVHM_ASSERT_(has_single_bit(end), "reason = ", reason, ", new_capacity = ", end, " is not a power of two!");

    const raw_pos_t old_end{capacity()};
    const raw_pos_t old_size{old_end - num_not_hash_slots_()};
    end = std::max(end, bit_ceil(old_size + conf_.grow_threshold(bit_ceil(old_size))));
    if (NVHM_UNLIKELY_(end == old_end)) {
      NVHM_LOG_(log_level_t::debug, reason, " resizing (abort): ", old_size, '/', old_end, " -> ", end, '\n');
      return end;
    }
    NVHM_LOG_(log_level_t::debug, reason, " resizing (before): ", old_size, '/', old_end, " -> ", end, '\n');

    const int_t blob_size{conf_.blob_size()};
    const int_t blob_stride{conf_.blob_stride()};
    
    unique_ptr_t<value_type[], allocator_type> old_values_ptr{std::move(values_)};
    unique_ptr_t<std::byte[], allocator_type> old_blobs_ptr{std::move(blobs_)};
    unique_ptr_t<key_type[], allocator_type> old_keys_ptr{std::move(keys_)};
    unique_ptr_t<state_t[], allocator_type> old_states_ptr{std::move(states_)};
    const value_type* const __restrict old_values{old_values_ptr.get()};
    const std::byte* const __restrict old_blobs{old_blobs_ptr.get()};
    const key_type* const __restrict old_keys{old_keys_ptr.get()};
    const state_t* const __restrict old_states{old_states_ptr.get()};
    
    for (;; end <<= 1) {
      __label__ next_old_pos, resize_error;

      end = alloc_(end);
      reset_();

      const bitmask_t bucket_mask{bucket_mask_};
      value_type* const __restrict new_values{values_.get()};
      std::byte* const __restrict new_blobs{blobs_.get()};
      key_type* const __restrict new_keys{keys_.get()};
      state_t* const __restrict new_states{states_.get()};

      // Re-insert all KV-pairs, taking some shortcuts because no collisions are possible.
      for (raw_pos_t old_off{}; old_off < old_end; old_off += kernel_size) {
        const auto k_old{kernel_type::load(&old_states[old_off])};
        
        for (auto m_old{kernel_type::mask_hash(k_old)}; mask_type::has_next(m_old); m_old = mask_type::step(m_old)) {
          const raw_pos_t old_pos{mask_type::next(m_old, old_off)};
          const hash_t hash{key_to_hash(old_keys[old_pos])};

          probe_seq_type seq{hash_to_pos<kernel_size>(hash)};
          for (; seq.has_next(); seq += kernel_size) {
            NVHM_ASSERT_(seq.psl() < end, "The table is full. This should not happen! (psl = ", seq.psl(), ", end = ", end, ')');

            // TODO: Shouldn't there only be two offsets possible. Can't we exploit that?
            const raw_pos_t new_off{align_pos(seq.next(), bucket_mask)};
            const auto k_new{kernel_type::load(&new_states[new_off])};
            
            for (auto m_new{kernel_type::mask_not_hash(k_new)}; NVHM_LIKELY_(mask_type::has_next(m_new)); m_new = mask_type::step(m_new)) {
              const raw_pos_t new_pos{mask_type::next(m_new, new_off)};

              new_states[new_pos] = hash_to_state(hash);
              if constexpr (probe_seq_type::is_bounded) {
                new_keys[new_pos] = old_keys[old_pos];
              } else {
                new_keys[new_pos] = std::move(old_keys[old_pos]);
              }
              if constexpr (has_values) {
                if constexpr (probe_seq_type::is_bounded) {
                  new_values[new_pos] = old_values[old_pos];
                } else {
                  new_values[new_pos] = std::move(old_values[old_pos]);
                }
              }
              if constexpr (has_blobs) {
                copy_blob(&new_blobs[new_pos * blob_stride], &old_blobs[old_pos * blob_stride], blob_size);
              }

              goto next_old_pos;
            }
          }

          if constexpr (probe_seq_type::is_unbounded) {
            throw std::runtime_error("Resizing error! This should not happen!");
          }
          goto resize_error;
        next_old_pos:;
        }
      }

      break;
    resize_error:;
      NVHM_LOG_(log_level_t::debug, reason, " resizing (error): ", size(), '/', capacity(), '\n');
    }

    num_empty_ -= old_size;
    NVHM_LOG_(log_level_t::debug, reason, " resizing (after): ", size(), '/', capacity(), '\n');
    NVHM_ASSERT_(check_integrity_());
    return end;
  }

  NVHM_NO_INLINE int_t scrub_(const char reason[], lru_t) {
    if (num_tombstone_ == 0) return false;
    NVHM_LOG_(log_level_t::debug, reason, " scrubbing (before): size = ", size(), ", num_empty = ", num_empty_, ", num_tombstone = ", num_tombstone_, '\n');

    const int_t blob_size{conf_.blob_size()};
    const int_t blob_stride{conf_.blob_stride()};

    const raw_pos_t end{capacity()};
    const bitmask_t bucket_mask{make_aligned_mask<kernel_size>(end)};
    value_type* const __restrict values{values_.get()};
    std::byte* const __restrict blobs{blobs_.get()};
    key_type* const __restrict keys{keys_.get()};
    state_t* const __restrict states{states_.get()};

    // Scrubbing consists of two phases.
    // 1. Re-insert KV-pairs upstream. As we try to reinsert KV-pairs upstream, we encounter a tombstone and replace it.
    //    -- The original location has an adjacent empty slot. Then we can reclaim that slot.
    //    -- The original location has no adjacent empty slot. Then we just propagate the tombstone.
    // 2. At the end, assuming that movements happend, we may be able to reclaim tombstones. This is possible because, we
    //    are ensured that all valid KV-pairs are now in the most upstream location "permissible".
    //    -- If all pairs are in the opitmal bucket (i.e. the PSL never exceeded 0), we can prune all tombstones.
    //    -- Otherwise, we can still prune tombstones in buckets that consist only of tombstones.
    int_t num_empty{num_empty_};
    int_t num_tombstone{num_tombstone_};
    int_t num_moves{};
    psl_t psl_bits;
    for (int_t phase{};; ++phase) {
      const int_t num_moves_prev{num_moves};
      psl_bits = 0;

      for (raw_pos_t old_off{}; old_off < end; old_off += kernel_size) {
        __label__ next_old_pos;

        // Re-insert KV-pairs upstream, if they can replace a tombstone.
        const auto old_k{kernel_type::load(&states[old_off])};

        for (auto old_m{kernel_type::mask_hash(old_k)}; mask_type::has_next(old_m); old_m = mask_type::step(old_m)) {
          const raw_pos_t old_pos{mask_type::next(old_m, old_off)};

          const hash_t hash{key_to_hash(keys[old_pos])};

          probe_seq_type seq{hash_to_pos<kernel_size>(hash)};
          for (; seq.has_next(); seq += kernel_size) {
            NVHM_ASSERT_(seq.psl() < end, "The table is full. This should not happen! (psl = ", seq.psl(), ", end = ", end, ')');

            const raw_pos_t new_off{align_pos(seq.next(), bucket_mask)};
            if (NVHM_LIKELY_(new_off == old_off)) goto next_old_pos;  // Already in the best possible bucket.

            const auto new_k{kernel_type::load(&states[new_off])};

            for (auto new_m{kernel_type::mask_not_hash(new_k)}; mask_type::has_next(new_m); new_m = mask_type::step(new_m)) {
              const raw_pos_t new_pos{mask_type::next(new_m, new_off)};
              NVHM_ASSERT_(states[new_pos] == kernel_type::tombstone);

              states[new_pos] = hash_to_state(hash);
              keys[new_pos] = std::move(keys[old_pos]);
              if constexpr (has_values) {
                values[new_pos] = std::move(values[old_pos]);
              }
              if constexpr (has_blobs) {
                copy_blob(&blobs[new_pos * blob_stride], &blobs[old_pos * blob_stride], blob_size);
              }

              if (kernel_type::has_empty(old_k)) {
                states[old_pos] = kernel_type::empty;
                ++num_empty;
                --num_tombstone;
              } else {
                states[old_pos] = kernel_type::tombstone;
              }
              ++num_moves;
              goto next_old_pos;
            }
          }
          NVHM_ASSERT_(false, "Probe sequence has exhausted. This should not happen! (psl = ", seq.psl(), ", max_length = ", probe_seq_type::max_length, ')');

        next_old_pos:;
          psl_bits |= seq.psl();
        }
      }
      NVHM_LOG_(log_level_t::debug, reason, " scrubbing (phase ", phase, "): size = ", size(), ", num_empty = ", num_empty_, " -> ", num_empty, ", num_tombstone = ", num_tombstone_, " -> ", num_tombstone, ", num_moves = ", num_moves, ", psl_bits = ", psl_bits, '\n');

      // TODO: Can we avoid the final full pass?
      if (num_moves - num_moves_prev <= 1) break;
    }
    num_empty_ = num_empty;
    num_tombstone_ = num_tombstone;
    NVHM_ASSERT_(check_integrity_());

    if (num_moves != 0) {
      // If all KV-pairs were placed in the best bucket, we can convert tombstones to empty slots.
      if (psl_bits == 0) {
        for (raw_pos_t off{}; off < end; off += kernel_size) {
          auto k{kernel_type::load(&states[off])};
          k = kernel_type::not_hash_to_empty(k);
          kernel_type::store(k, &states[off]);
        }
        num_empty_ += num_tombstone;
        num_tombstone_ = 0;
      } else {
        // Alternatively, we can try locating and earsing dead buckets (this is only permissible if we completed all passes).
        int_t n{};
        for (raw_pos_t off{}; off < end; off += kernel_size) {
          auto k{kernel_type::load(&states[off])};
          if (kernel_type::has_not_tombstone(k)) continue;
          
          std::fill_n(&states[off], kernel_size, kernel_type::empty);
          n += kernel_size;
        }
        num_empty_ += n;
        num_tombstone_ -= n;
      }

      NVHM_ASSERT_(check_integrity_());
    }

    NVHM_LOG_(log_level_t::debug, reason, " scrubbing (after): size = ", size(), ", num_empty = ", num_empty_, ", num_tombstone = ", num_tombstone_, ", num_moves = ", num_moves, '\n');
    return num_moves;
  }

  NVHM_ALWAYS_INLINE constexpr int_t shrink_target_() const noexcept {
    const int_t num_empty{num_empty_};
    const int_t num_tombstone{num_tombstone_};
    const int_t size{capacity() - (num_empty + num_tombstone)};
    const int_t min_end{std::max(bit_ceil(size), conf_.min_capacity())};

    raw_pos_t end{capacity()};
    for (; end > min_end; end >>= 1) {
      if (!conf_.should_shrink(end, num_empty, num_tombstone)) break;
    }
    return end;
  }

  NVHM_ALWAYS_INLINE constexpr state_t state_at_(const raw_pos_t pos) const noexcept {
    NVHM_ASSERT_(pos >= 0 && pos < capacity(), "pos = ", pos, ", capacity = ", capacity());
    return states_[to_uint(pos)];
  }

  NVHM_ALWAYS_INLINE constexpr void write_prefetch_states_(raw_pos_t off) noexcept {
    NVHM_ASSERT_(off >= 0 && off <= to_int(bucket_mask_), "off = ", off, ", bucket_mask = ", std::hex, bucket_mask_, std::dec);
    nvhm::write_prefetch<kernel_size>(&states_[to_uint(off)]);
  }
};

template <
  typename Key,
  flags_t Flags = flags_t::none, typename Kernel = default_kernel_t<>,
  typename ProbeSeq = default_seq_t, typename Allocator = default_allocator_t>
using set = map<Key, void, Flags & ~flags_t::blobs, Kernel, ProbeSeq, Allocator>;

}  // namespace nvhm