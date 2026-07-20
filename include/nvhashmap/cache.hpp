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
#include "conf.hpp"
#include "probe_seq.hpp"

namespace nvhm {

template<int_t Size>
constexpr raw_pos_t pos_to_pos2(raw_pos_t pos) noexcept {
  constexpr bitmask_t mask{size_mask_v<Size>};
  return pos + align_pos<~mask>(pos);
}

/**
 * Set-associative cache with LRU replacement policy.
 * The kernel size determines the set-associativity.
 */
template <
  typename Key, typename Value = void, flags_t Flags = flags_t::none, typename Kernel = default_kernel_t<>,
  typename Allocator = default_allocator_t>
class cache : public swiss_map_base<
  cache<Key, Value, Flags, Kernel, Allocator>,
  conf<Key, Value, Flags, Kernel::size>, linear_seq<0, Kernel::size>, Allocator
> {
 public:
  using base_type = swiss_map_base<cache, conf<Key, Value, Flags, Kernel::size>, linear_seq<0, Kernel::size>, Allocator>;

  using self_type = typename base_type::self_type;
  using conf_type = typename base_type::conf_type;

  using key_type = typename base_type::key_type;
  using value_type = typename base_type::value_type;
  using base_type::has_values;
  using base_type::flags;
  using base_type::has_blobs;
  using base_type::kernel_size;

  using probe_seq_type = typename base_type::probe_seq_type;
  static_assert(probe_seq_type::max_length == kernel_size);
  using allocator_type = typename base_type::allocator_type;

  using kernel_type = Kernel;
  static_assert(kernel_type::size == conf_type::kernel_size);
  using mask_type = typename kernel_type::mask_type;
  
  constexpr cache(const cache& other) : base_type{other} {
    states_and_lrus_ = make_copy(other.states_and_lrus_, capacity() * 2);
    num_empty_ = other.num_empty_;
  }
  constexpr cache& operator=(const cache& other) {
    if (self() == &other) return *self();
    base_type::operator=(other);

    const raw_pos_t end{capacity()};
    NVHM_ASSERT_(end == other.capacity(), "end = ", end, ", other.capacity = ", other.capacity());

    std::copy_n(other.states_and_lrus_.get(), end * 2, states_and_lrus_.get());
    num_empty_ = other.num_empty_;
    return *self();
  }
  constexpr cache(cache&& other) noexcept = default;
  constexpr cache& operator=(cache&& other) noexcept = default;

  constexpr cache() : cache{conf_type()} {}
  constexpr cache(int_t init_capacity) : cache{conf_type().set_capacity(init_capacity)} {}
  constexpr cache(const conf_type& conf) : base_type{conf} {
    states_and_lrus_ = make_unique<std::byte, allocator_type>(capacity() * 2);
    reset_();
  }

  using base_type::capacity;
  using base_type::self;
  using base_type::size;

  constexpr friend void swap(cache& lhs, cache& rhs) noexcept {
    if (&lhs == &rhs) return;
    swap(static_cast<base_type&>(lhs), static_cast<base_type&>(rhs));

    std::swap(lhs.states_and_lrus_, rhs.states_and_lrus_);
    std::swap(lhs.num_empty_, rhs.num_empty_);
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
  mutable unique_ptr_t<std::byte[], allocator_type> states_and_lrus_;
  int_t num_empty_;

  using base_type::contains_at_;

  constexpr raw_pos_t alloc_(raw_pos_t end) {
    end = base_type::alloc_(end);

    states_and_lrus_ = make_unique<std::byte, allocator_type>(end * 2);
    return end;
  }

  constexpr int_t bucket_size_at_(const raw_pos_t pos) const noexcept {
    NVHM_ASSERT_(pos >= 0 && pos < capacity(), "pos = ", pos, ", capacity = ", capacity());
    const state_t* const __restrict states{states_()};

    const raw_pos_t off{align_pos<~kernel_type::size_mask>(pos)};
    const auto k{kernel_type::load(&states[off * 2])};
    // TODO: Add optimized kernel function for this?
    return mask_type::count(kernel_type::mask_hash(k));
  }

  constexpr int_t bucket_num_empty_slots_at_(const raw_pos_t pos) const noexcept {
    NVHM_ASSERT_(pos >= 0 && pos < capacity(), "pos = ", pos, ", capacity = ", capacity());
    const state_t* const __restrict states{states_()};

    const raw_pos_t off{align_pos<~kernel_type::size_mask>(pos)};
    const auto k{kernel_type::load(&states[off * 2])};
    // TODO: Add optimized kernel function for this?
    return mask_type::count(kernel_type::mask_empty(k));
  }

  constexpr int_t bucket_num_tombstone_slots_at_(const raw_pos_t pos) const noexcept {
    NVHM_ASSERT_(pos >= 0 && pos < capacity(), "pos = ", pos, ", capacity = ", capacity());
    const state_t* const __restrict states{states_()};

    const raw_pos_t off{align_pos<~kernel_type::size_mask>(pos)};
    const auto k{kernel_type::load(&states[off * 2])};
    // TODO: Add optimized kernel function for this?
    return mask_type::count(kernel_type::mask_tombstone(k));
  }
  
  constexpr bool check_integrity_() const noexcept {
    const int_t end{capacity()};
    const state_t* const __restrict states{states_()};
    
    int_t num_empty{};
    for (raw_pos_t off{}; off < end; off += kernel_size) {
      const auto k{kernel_type::load(&states[off * 2])};
      
      // TODO: Add optimized kernel function for this?
      num_empty += mask_type::count(kernel_type::mask_empty(k));
    }

    NVHM_ASSERT_(num_empty == num_empty_, "num_empty = ", num_empty, ", num_empty_ = ", num_empty_);
    return num_empty == num_empty_;
  }

  constexpr int_t clear_() noexcept { return reset_(); }

  constexpr void count_kernel_populations_(std::array<int_t, kernel_size + 1>& counts) const noexcept {
    const raw_pos_t end{capacity()};
    const state_t* const __restrict states{states_()};

    for (raw_pos_t off{}; off < end; off += kernel_size) {
      auto k{kernel_type::load(&states[off * 2])};
      auto m{kernel_type::mask_hash(k)};

      ++counts[to_uint(mask_type::count(m))];
    }
  }

  constexpr void count_state_collisions_(std::array<int_t, kernel_size>& counts) const noexcept {
    const raw_pos_t end{capacity()};
    const state_t* const __restrict states{states_()};

    for (raw_pos_t off{}; off < end; off += kernel_size) {
      count_collisions<kernel_type>(&states[off * 2], counts);
    }
  }

  NVHM_ALWAYS_INLINE constexpr std::tuple<bool, raw_pos_t, probe_seq_type> erase_first_(const key_type& __restrict key, const hash_t hash) {
    NVHM_ASSERT_(key_to_hash(key) == hash, "Supplied key and hash do not match!");

    const bitmask_t bucket_mask{bucket_mask_};
    const key_type* const __restrict keys{keys_.get()};
    state_t* const __restrict states{states_()};

    const state_t s{hash_to_state(hash)};

    probe_seq_type seq{hash_to_pos<kernel_size>(hash)};
    NVHM_ASSERT_(seq.psl() < capacity(), "This should not happen! (psl = ", seq.psl(), ", capacity = ", capacity(), ')');
    
    const raw_pos_t off{align_pos(seq.next(), bucket_mask)};
    const auto k{kernel_type::load(&states[off * 2])};

    // Is the key present in this bucket?
    for (auto m{kernel_type::mask(k, s)}; mask_type::has_next(m); m = mask_type::step(m)) {
      const int_t idx{mask_type::next(m)};
      const raw_pos_t pos{off + idx};
      if (NVHM_UNLIKELY_(keys[pos] != key)) continue;

      reset_slot_(states[off * 2 + idx]);
      return {true, pos, std::move(seq)};
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
    const key_type* const __restrict keys{keys_.get()};
    state_t* const __restrict states{states_()};

    const state_t s{hash_to_state(hash)};
    const raw_pos_t end{aligned_mask_to_capacity<kernel_size>(bucket_mask)};
    NVHM_ASSERT_(pos >= 0 && pos < end, "pos = ", pos, ", capacity = ", end);

    auto [off, idx]{splice_pos<kernel_size>(pos)};
    NVHM_ASSERT_(seq.has_next() && align_pos(seq.next(), bucket_mask) == off);
    NVHM_ASSERT_(seq.psl() < end, "This should not happen! (psl = ", seq.psl(), ", capacity = ", end, ')');

    auto k{kernel_type::load(&states[off * 2])};

    // Is the key present in this bucket?
    for (auto m{mask_type::above(kernel_type::mask(k, s), idx)}; mask_type::has_next(m); m = mask_type::step(m)) {
      idx = mask_type::next(m);
      pos = off + idx;
      if (NVHM_UNLIKELY_(keys[pos] != key)) continue;

      reset_slot_(states[off * 2 + idx]);
      return {true, pos, std::move(seq)};
    }

    return {false, npos, std::move(seq)};
  }

  NVHM_ALWAYS_INLINE constexpr std::pair<bool, raw_pos_t> erase_at_(raw_pos_t pos) {
    NVHM_ASSERT_(pos >= 0 && pos < capacity(), "pos = ", pos, ", capacity = ", capacity());
    state_t* const __restrict states{states_()};

    raw_pos_t pos2{pos_to_pos2<kernel_size>(pos)};

    if (NVHM_LIKELY_(is_hash(states[pos2]))) {
      reset_slot_(states[pos2]);
      return {true, pos};
    }
  
    return {false, pos};
  }

  template <typename Pred>
  NVHM_ALWAYS_INLINE constexpr int_t erase_if_(const Pred& __restrict pred) {
    const raw_pos_t end{capacity()};
    state_t* const __restrict states{states_()};

    int_t num_empty{num_empty_};
    for (raw_pos_t off{}; off < end; off += kernel_size) {
      const auto k{kernel_type::load(&states[off * 2])};

      for (auto m{kernel_type::mask_hash(k)}; mask_type::has_next(m); m = mask_type::step(m)) {
        const int_t idx{mask_type::next(m)};
        const raw_pos_t pos{off + idx};
        if (!pred(pos)) continue;

        states[off * 2 + idx] = kernel_type::empty;
        ++num_empty;
      }
    }

    const int_t num_erased{num_empty - num_empty_};
    num_empty_ = num_empty;
    NVHM_ASSERT_(check_integrity_());
    return num_erased;
  }

  NVHM_ALWAYS_INLINE constexpr int_t erase_all_(const key_type& __restrict key, const hash_t hash) {
    NVHM_ASSERT_(key_to_hash(key) == hash, "Supplied key and hash do not match!");

    const bitmask_t bucket_mask{bucket_mask_};
    const key_type* const __restrict keys{keys_.get()};
    state_t* const __restrict states{states_()};

    const state_t s{hash_to_state(hash)};

    int_t num_empty{num_empty_};
    probe_seq_type seq{hash_to_pos<kernel_size>(hash)};
    NVHM_ASSERT_(seq.psl() < capacity(), "This should not happen! (psl = ", seq.psl(), ", capacity = ", capacity(), ')');
    
    const raw_pos_t off{align_pos(seq.next(), bucket_mask)};
    const auto k{kernel_type::load(&states[off * 2])};

    // Is the key present in this bucket?
    for (auto m{kernel_type::mask(k, s)}; mask_type::has_next(m); m = mask_type::step(m)) {
      const int_t idx{mask_type::next(m)};
      if (NVHM_UNLIKELY_(keys[off + idx] != key)) continue;

      states[off * 2 + idx] = kernel_type::empty;
      ++num_empty;
    }

    const int_t num_erased{num_empty - num_empty_};
    num_empty_ = num_empty;
    NVHM_ASSERT_(check_integrity_());

    return num_erased;
  }

  NVHM_ALWAYS_INLINE constexpr std::pair<raw_pos_t, probe_seq_type> find_first_(const key_type& __restrict key, const hash_t hash) const {
    NVHM_ASSERT_(key_to_hash(key) == hash, "Supplied key and hash do not match!");

    const bitmask_t bucket_mask{bucket_mask_};
    const key_type* const __restrict keys{keys_.get()};
    const state_t* const __restrict states{states_()};
    lru_t* const __restrict lrus{lrus_()};

    const state_t s{hash_to_state(hash)};

    probe_seq_type seq{hash_to_pos<kernel_size>(hash)};
    NVHM_ASSERT_(seq.psl() < capacity(), "This should not happen! (psl = ", seq.psl(), ", capacity = ", capacity(), ')');

    const raw_pos_t off{align_pos(seq.next(), bucket_mask)};
    const auto k{kernel_type::load(&states[off * 2])};

    // Is the key present in this bucket?
    for (auto m{kernel_type::mask(k, s)}; mask_type::has_next(m); m = mask_type::step(m)) {
      const raw_pos_t pos{mask_type::next(m, off)};
      if (NVHM_UNLIKELY_(keys[pos] != key)) continue;

      auto l{kernel_type::load_lru(&lrus[off * 2])};
      l = kernel_type::update_lru(l, m);
      kernel_type::store_lru(l, &lrus[off * 2]);

      return {pos, std::move(seq)};
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
    const state_t* const __restrict states{states_()};
    lru_t* const __restrict lrus{lrus_()};

    const state_t s{hash_to_state(hash)};
    NVHM_ASSERT_(pos >= 0 && pos < capacity(), "pos = ", pos, ", capacity = ", capacity());

    auto [off, idx]{splice_pos<kernel_size>(pos)};
    NVHM_ASSERT_(seq.has_next() && align_pos(seq.next(), bucket_mask) == off);
    NVHM_ASSERT_(seq.psl() < capacity(), "This should not happen! (psl = ", seq.psl(), ", capacity = ", capacity(), ')');

    const auto k{kernel_type::load(&states[off * 2])};

    // Is the key present in this bucket?
    for (auto m{mask_type::above(kernel_type::mask(k, s), idx)}; mask_type::has_next(m); m = mask_type::step(m)) {
      pos = mask_type::next(m, off);
      if (NVHM_UNLIKELY_(keys[pos] != key)) continue;

      auto l{kernel_type::load_lru(&lrus[off * 2])};
      l = kernel_type::update_lru(l, m);
      kernel_type::store_lru(l, &lrus[off * 2]);

      return {pos, std::move(seq)};
    }

    return {npos, std::move(seq)};
  }

  template <typename Pred>
  NVHM_ALWAYS_INLINE constexpr raw_pos_t find_if_(const Pred& __restrict pred) const {
    const raw_pos_t end{capacity()};
    const state_t* const __restrict states{states_()};
    lru_t* const __restrict lrus{lrus_()};
    
    for (raw_pos_t off{}; off < end; off += kernel_size) {
      const auto k{kernel_type::load(&states[off * 2])};

      for (auto m{kernel_type::mask_hash(k)}; mask_type::has_next(m); m = mask_type::step(m)) {
        const raw_pos_t pos{mask_type::next(m, off)};
        if (!pred(pos)) continue;

        auto l{kernel_type::load_lru(&lrus[off * 2])};
        l = kernel_type::update_lru(l, m);
        kernel_type::store_lru(l, &lrus[off * 2]);

        return pos;
      }
    }
    return npos;
  }

  template <typename Func>
  NVHM_ALWAYS_INLINE constexpr void for_each_(const Func& __restrict func) const {
    const raw_pos_t end{capacity()};
    const state_t* const __restrict states{states_()};
    
    for (raw_pos_t off{}; off < end; off += kernel_size) {
      const auto k{kernel_type::load(&states[off * 2])};

      for (auto m{kernel_type::mask_hash(k)}; mask_type::has_next(m); m = mask_type::step(m)) {
        func(mask_type::next(m, off));
      }
    }
  }

  template <typename Func>
  NVHM_ALWAYS_INLINE constexpr void for_each_lru_(const Func& __restrict func) const {
    const raw_pos_t end{capacity()};
    const state_t* const __restrict states{states_()};
    const lru_t* const __restrict lrus{lrus_()};

    for (raw_pos_t off{}; off < end; off += kernel_size) {
      const auto k{kernel_type::load(&states[off * 2])};

      for (auto m{kernel_type::mask_hash(k)}; mask_type::has_next(m); m = mask_type::step(m)) {
        const int_t idx{mask_type::next(m)};
        func(lrus[off * 2 + idx]);
      }
    }
  }

  template <typename Func>
  NVHM_ALWAYS_INLINE constexpr void for_each_state_(const Func& __restrict func) const {
    const raw_pos_t end{capacity()};
    const state_t* const __restrict states{states_()};

    for (raw_pos_t off{}; off < end; off += kernel_size) {
      const auto k{kernel_type::load(&states[off * 2])};

      for (auto m{kernel_type::mask_hash(k)}; mask_type::has_next(m); m = mask_type::step(m)) {
        const int_t idx{mask_type::next(m)};
        func(kernel_type::at(k, idx));
      }
    }
  }

  template <typename Func>
  NVHM_ALWAYS_INLINE constexpr void for_each_(const key_type& __restrict key, const hash_t hash, const Func& __restrict func) const {
    static_assert(std::is_invocable_v<Func, raw_pos_t, probe_seq_type>, "`func` must be `func(raw_pos_t, probe_seq_type)`!");
    NVHM_ASSERT_(key_to_hash(key) == hash, "Supplied key and hash do not match!");

    const bitmask_t bucket_mask{bucket_mask_};
    const key_type* const __restrict keys{keys_.get()};
    const state_t* const __restrict states{states_()};
    lru_t* const __restrict lrus{lrus_()};

    const state_t s{hash_to_state(hash)};

    probe_seq_type seq{hash_to_pos<kernel_size>(hash)};
    NVHM_ASSERT_(seq.psl() < capacity(), "This should not happen! (psl = ", seq.psl(), ", capacity = ", capacity(), ')');

    const raw_pos_t off{align_pos(seq.next(), bucket_mask)};
    const auto k{kernel_type::load(&states[off * 2])};

    // Is the key present in this bucket?
    for (auto m{kernel_type::mask(k, s)}; mask_type::has_next(m); m = mask_type::step(m)) {
      const raw_pos_t pos{mask_type::next(m, off)};
      if (NVHM_UNLIKELY_(keys[pos] != key)) continue;

      auto l{kernel_type::load_lru(&lrus[off * 2])};
      l = kernel_type::update_lru(l, m);
      kernel_type::store_lru(l, &lrus[off * 2]);

      func(pos, seq);
      if constexpr (!test_flags(flags, flags_t::duplicates)) {
        return;
      }
    }
  }

  template <typename K>
  inline std::tuple<raw_pos_t, probe_seq_type, insert_op_t> insert_(K&& __restrict key, const hash_t hash) {
    static_assert(std::is_same_v<remove_cvref_t<K>, key_type>, "K must be `key_type`!");
    if constexpr (std::is_floating_point_v<key_type>) {
      if (key != key) {
        throw std::runtime_error("Provided key is not a discernable floating point value!");
      }
    }
    NVHM_ASSERT_(key_to_hash(key) == hash, "Supplied key and hash do not match!");

    const bitmask_t bucket_mask{bucket_mask_};
    key_type* const __restrict keys{keys_.get()};
    state_t* const __restrict states{states_()};
    lru_t* const __restrict lrus{lrus_()};

    const state_t s{hash_to_state(hash)};

    probe_seq_type seq{hash_to_pos<kernel_size>(hash)};
    NVHM_ASSERT_(seq.psl() < capacity(), "This should not happen! (psl = ", seq.psl(), ", end = ", capacity(), ')');

    const raw_pos_t off{align_pos(seq.next(), bucket_mask)};
    const auto k{kernel_type::load(&states[off * 2])};

    if constexpr (!test_flags(flags, flags_t::duplicates)) {
      // Is the key present in this bucket?
      for (auto m{kernel_type::mask(k, s)}; mask_type::has_next(m); m = mask_type::step(m)) {
        const raw_pos_t pos{mask_type::next(m, off)};
        if (NVHM_LIKELY_(keys[pos] != key)) continue;

        // Increment LRU, and rescale bucket if necessary.
        auto l{kernel_type::load_lru(&lrus[off * 2])};
        l = kernel_type::update_lru(l, m);
        kernel_type::store_lru(l, &lrus[off * 2]);

        return {pos, std::move(seq), insert_op_t::found};
      }
    }

    // Have slot available?
    auto m{kernel_type::mask_not_hash(k)};

    insert_op_t op;
    if (NVHM_LIKELY_(mask_type::has_next(m))) {
      op = insert_op_t::insert;
    } else {
      m = kernel_type::mask_min_lru(kernel_type::load_lru(&lrus[off * 2]));
      op = insert_op_t::replace;
    }
    int_t idx{mask_type::next(m)};

    const raw_pos_t pos{off + idx};
    keys[pos] = std::forward<K>(key);
    states[off * 2 + idx] = s;
    lrus[off * 2 + idx] = default_lru;

    num_empty_ -= (op == insert_op_t::insert);
    NVHM_ASSERT_(check_integrity_());

    return {pos, std::move(seq), op};
  }

  constexpr bool is_full_() const noexcept { return size() >= capacity(); }

  constexpr lru_t lru_at_(raw_pos_t pos) const noexcept {
    NVHM_ASSERT_(contains_at_(pos), "pos = ", pos);
    return lrus_()[pos_to_pos2<kernel_size>(pos)];
  }

  constexpr double max_load_factor_() const noexcept { return 1.0; }

  constexpr int_t num_empty_slots_() const noexcept { return num_empty_; }
  constexpr int_t num_tombstone_slots_() const noexcept { return 0; }

  constexpr void read_prefetch_states_(raw_pos_t off) const noexcept {
    NVHM_ASSERT_(off >= 0 && off <= to_int(bucket_mask_), "off = ", off, ", bucket_mask = ", std::hex, bucket_mask_, std::dec);
    nvhm::read_prefetch<kernel_size * 2>(&states_()[off * 2]);
  }

  constexpr raw_pos_t reset_() noexcept {
    const raw_pos_t end{capacity()};
    state_t* const __restrict states{states_()};
    lru_t* const __restrict lrus{lrus_()};

    for (raw_pos_t off{}; off < end; off += kernel_size) {
      std::fill_n(&states[off * 2], kernel_size, kernel_type::empty);
      std::fill_n(&lrus[off * 2], kernel_size, max_lru);
    } 
    num_empty_ = end;
    NVHM_ASSERT_(check_integrity_());

    return end;
  }

  constexpr void reset_slot_(state_t& __restrict slot) {
    NVHM_ASSERT_(is_hash(slot));

    slot = kernel_type::empty;
    ++num_empty_;
    NVHM_ASSERT_(check_integrity_());
  }

  raw_pos_t resize_(const char reason[], raw_pos_t end) {
    NVHM_ASSERT_(end >= conf_.min_capacity(), "new_capacity = ", end, ", min_capacity = ", conf_.min_capacity());
    NVHM_ASSERT_(has_single_bit(end), "new_capacity = ", end, " is not a power of two!");

    const raw_pos_t old_end{capacity()};
    end = std::max(end, conf_.min_capacity());
    if (NVHM_UNLIKELY_(end == old_end)) {
      NVHM_LOG_(log_level_t::debug, reason, " resizing (abort): ", size(), '/', old_end, " -> ", end, '\n');
      return end;
    }
    NVHM_LOG_(log_level_t::debug, reason, " resizing (before): ", size(), '/', old_end, " -> ", end, '\n');

    const int_t blob_size{conf_.blob_size()};
    const int_t blob_stride{conf_.blob_stride()};

    value_type* const __restrict old_values{values_.get()};
    std::byte* const __restrict old_blobs{blobs_.get()};
    key_type* const __restrict old_keys{keys_.get()};
    state_t* const __restrict old_states{states_()};
    lru_t* const __restrict old_lrus{lrus_()};
    unique_ptr_t<value_type[], allocator_type> old_values_ptr{std::move(values_)};
    unique_ptr_t<std::byte[], allocator_type> old_blobs_ptr{std::move(blobs_)};
    unique_ptr_t<key_type[], allocator_type> old_keys_ptr{std::move(keys_)};
    unique_ptr_t<std::byte[], allocator_type> old_states_and_lrus_ptr{std::move(states_and_lrus_)};

    end = alloc_(end);
    reset_();

    const bitmask_t bucket_mask{bucket_mask_};
    value_type* const __restrict values{values_.get()};
    std::byte* const __restrict blobs{blobs_.get()};
    key_type* const __restrict keys{keys_.get()};
    state_t* const __restrict states{states_()};
    lru_t* const __restrict lrus{lrus_()};
    
    int_t num_insert{};
    for (raw_pos_t old_off{}; old_off < old_end; old_off += kernel_size) {
      const auto k_old{kernel_type::load(&old_states[old_off * 2])};

      for (auto m_old{kernel_type::mask_hash(k_old)}; mask_type::has_next(m_old); m_old = mask_type::step(m_old)) {
        const int_t old_idx{mask_type::next(m_old)};
        const raw_pos_t old_pos{old_off + old_idx};
        
        const lru_t old_lru{old_lrus[old_off * 2 + old_idx]};
        const hash_t hash{key_to_hash(old_keys[old_pos])};

        const probe_seq_type seq{hash_to_pos<kernel_size>(hash)};
        NVHM_ASSERT_(seq.psl() < end, "This should not happen! (psl = ", seq.psl(), ", end = ", end, ')');
        const raw_pos_t new_off{align_pos(seq.next(), bucket_mask)};
        
        // Have slot available?
        const auto k_new{kernel_type::load(&states[new_off * 2])};
        auto m_new{kernel_type::mask_not_hash(k_new)};

        lru_t new_lru;
        int_t new_idx;
        if (NVHM_LIKELY_(mask_type::has_next(m_new))) {
          new_idx = mask_type::next(m_new);
          new_lru = lrus[new_off * 2 + new_idx];
          ++num_insert;
        } else {
          const auto l_new{kernel_type::load_lru(&lrus[new_off * 2])};

          std::tie(new_lru, new_idx) = kernel_type::min_lru(l_new);
          if (NVHM_LIKELY_(old_lru < new_lru)) {
            continue;
          }
        }

        const raw_pos_t new_pos{new_off + new_idx};
        if constexpr (has_values) {
          values[new_pos] = std::move(old_values[old_pos]);
        }
        if constexpr (has_blobs) {
          copy_blob(&blobs[new_pos * blob_stride], &old_blobs[old_pos * blob_stride], blob_size);
        }
        keys[new_pos] = std::move(old_keys[old_pos]);
        states[new_off * 2 + new_idx] = hash_to_state(hash);
        lrus[new_off * 2 + new_idx] = old_lru;
      }
    }

    num_empty_ -= num_insert;
    NVHM_LOG_(log_level_t::debug, reason, " resizing (after): ", size(), '/', end, '\n');
    NVHM_ASSERT_(check_integrity_());
    return end;
  }

  constexpr int_t scrub_(const char reason[], lru_t threshold) noexcept {
    if (threshold >= max_lru) return 0;
    NVHM_LOG_(log_level_t::debug, reason, " scrubbing (before): num_empty = ", num_empty_, '\n');

    const raw_pos_t end{capacity()};
    state_t* const __restrict states{states_()};
    const lru_t* const __restrict lrus{lrus_()};

    int_t num_erased{};
    for (raw_pos_t off{}; off < end; off += kernel_size) {
      const auto k0{kernel_type::load(&states[off * 2])};
      const auto l{kernel_type::load_lru(&lrus[off * 2])};

      const auto k1{kernel_type::to_empty_if_lru_below(k0, l, threshold)};
      num_erased += kernel_type::count_not_equal(k0, k1);

      kernel_type::store(k1, &states[off * 2]);
    }
    num_empty_ += num_erased;

    NVHM_LOG_(log_level_t::debug, reason, " scrubbing (after): num_empty = ", num_empty_, ", num_erased = ", num_erased, '\n');
    NVHM_ASSERT_(check_integrity_());
    return num_erased;
  }

  constexpr int_t shrink_target_() const noexcept {
    return std::max(capacity() >> 1, conf_.min_capacity());
  }

  constexpr state_t state_at_(const raw_pos_t pos) const noexcept {
    NVHM_ASSERT_(pos >= 0 && pos < capacity(), "pos = ", pos, ", capacity = ", capacity());
    return states_()[pos_to_pos2<kernel_size>(pos)];
  }

  constexpr void write_prefetch_states_(raw_pos_t off) noexcept {
    NVHM_ASSERT_(off >= 0 && off <= to_int(bucket_mask_), "off = ", off, ", bucket_mask = ", std::hex, bucket_mask_, std::dec);
    nvhm::write_prefetch<kernel_size * 2>(&states_()[off * 2]);
  }

  constexpr const state_t* states_() const noexcept {
    return reinterpret_cast<const state_t*>(states_and_lrus_.get());
  }
  constexpr state_t* states_() noexcept {
    return reinterpret_cast<state_t*>(states_and_lrus_.get());
  }

  constexpr lru_t* lrus_() const noexcept {
    return reinterpret_cast<lru_t*>(states_and_lrus_.get() + kernel_size);
  }
  constexpr lru_t* lrus_() noexcept {
    return reinterpret_cast<lru_t*>(states_and_lrus_.get()) + kernel_size;
  }
};

template <
  typename Key,
  flags_t Flags = flags_t::none, typename Kernel = default_kernel_t<>,
  typename Allocator = default_allocator_t>
using cache_set = cache<Key, void, Flags & ~flags_t::blobs, Kernel, Allocator>;

}