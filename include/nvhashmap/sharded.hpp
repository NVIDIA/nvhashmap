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

#include "container.hpp"
#include "memory.hpp"

namespace nvhm {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wc++20-extensions"

// template <typename Key>
// constexpr int_t key_to_shard(const Key& k, const bitmask_t shard_mask) noexcept {
//   hash_t h{hasher<Key>(k)};
// 
//   // TODO: Not yet configurable because we may want to change that later.
//   constexpr uint64_t c{UINT64_C(0x63a0'8307'3c4c'e4a1)};
//   const auto h128{static_cast<__uint128_t>(h) * c};
//   h = static_cast<hash_t>(h128 >> 64) + static_cast<hash_t>(h128);
// 
//   return h & shard_mask;
// }

template <typename Inner>
class sharded_pos : public wrapped_pos<Inner> {
 public:
  using base_type = wrapped_pos<Inner>;
  using inner_type = typename base_type::inner_type;

  sharded_pos() = delete;

#if defined(NVHM_DEBUG_API_)
  constexpr shard_idx_t shard_idx() const noexcept { return shard_idx_; }
#endif

  template <typename RhsInner>
  constexpr friend bool operator==(const sharded_pos& lhs, const sharded_pos<RhsInner>& rhs) {
    if (lhs.inner_ == npos) return rhs.inner_ == npos;
    if (rhs.inner_ == npos) return false;
    return lhs.shard_idx_ == rhs.shard_idx_ && lhs.inner_ == rhs.inner_;
  }
  template <typename RhsInner>
  constexpr friend bool operator!=(const sharded_pos& lhs, const sharded_pos<RhsInner>& rhs) { return !(lhs == rhs); }
  template <typename RhsInner>
  constexpr friend bool operator<(const sharded_pos& lhs, const sharded_pos<RhsInner>& rhs) {
    if (lhs.inner_ == npos) return rhs.inner_ != npos;
    if (rhs.inner_ == npos) return false;
    return lhs.shard_idx_ < rhs.shard_idx_ || (lhs.shard_idx_ == rhs.shard_idx_ && lhs.inner_ < rhs.inner_);
  }
  template <typename RhsInner>
  constexpr friend bool operator>(const sharded_pos& lhs, const sharded_pos<RhsInner>& rhs) { return rhs < lhs; }
  template <typename RhsInner>
  constexpr friend bool operator<=(const sharded_pos& lhs, const sharded_pos<RhsInner>& rhs) { return !(lhs > rhs); }
  template <typename RhsInner>
  constexpr friend bool operator>=(const sharded_pos& lhs, const sharded_pos<RhsInner>& rhs) { return !(lhs < rhs); }

  constexpr friend void swap(sharded_pos& lhs, sharded_pos& rhs) noexcept {
    if (&lhs == &rhs) return;
    swap(static_cast<base_type&>(lhs), static_cast<base_type&>(rhs));

    std::swap(lhs.shard_idx_, rhs.shard_idx_);
  }

 protected:
  using base_type::inner_;
  shard_idx_t shard_idx_;

  constexpr explicit sharded_pos(const sharded_pos&) noexcept = default;
  constexpr sharded_pos& operator=(const sharded_pos&) noexcept = default;
  constexpr explicit sharded_pos(sharded_pos&&) noexcept = default;
  constexpr sharded_pos& operator=(sharded_pos&&) noexcept = default;

  constexpr explicit sharded_pos(inner_type&& inner, shard_idx_t shard_idx) noexcept
    : base_type{std::move(inner)}, shard_idx_{shard_idx} {}

  template <typename>
  friend class sharded;
};

template <typename Inner>
class sharded_read_pos : public sharded_pos<Inner> {
 public:
  using base_type = sharded_pos<Inner>;
  using inner_type = typename base_type::inner_type;

  sharded_read_pos() = delete;
  constexpr sharded_read_pos(const sharded_read_pos&) noexcept = default;
  constexpr sharded_read_pos& operator=(const sharded_read_pos&) noexcept = default;
  constexpr sharded_read_pos(sharded_read_pos&&) noexcept = default;
  constexpr sharded_read_pos& operator=(sharded_read_pos&&) noexcept = default;

  constexpr explicit sharded_read_pos(inner_type&& inner, shard_idx_t shard_idx) noexcept
    : base_type{std::move(inner), shard_idx} {}
};

template <typename Inner>
class sharded_write_pos : public sharded_pos<Inner> {
 public:
  using base_type = sharded_pos<Inner>;
  using inner_type = typename base_type::inner_type;
  
  sharded_write_pos() = delete;
  constexpr sharded_write_pos(const sharded_write_pos&) noexcept = default;
  constexpr sharded_write_pos& operator=(const sharded_write_pos&) noexcept = default;
  constexpr sharded_write_pos(sharded_write_pos&&) noexcept = default;
  constexpr sharded_write_pos& operator=(sharded_write_pos&&) noexcept = default;

  constexpr explicit sharded_write_pos(inner_type&& inner, shard_idx_t shard_idx) noexcept
    : base_type{std::move(inner), shard_idx} {}
};

template <typename Shard>
class sharded : public container<sharded<Shard>> {
 public:
  static_assert(!std::is_reference_v<Shard>, "Cannot encapuslate reference types with sharded!");
  using base_type = container<sharded<Shard>>;
  using self_type = typename base_type::self_type;
  using shard_type = Shard;
  using conf_type = typename shard_type::conf_type;

  using key_type = typename shard_type::key_type;
  using value_type = typename shard_type::value_type;
  constexpr static bool has_values{shard_type::has_values};
  constexpr static flags_t flags{shard_type::flags};
  constexpr static bool has_blobs{shard_type::has_blobs};
  constexpr static int_t kernel_size{shard_type::kernel_size};

  using const_entry_type = typename shard_type::const_entry_type;
  using entry_type = typename shard_type::entry_type;
  using const_mapped_type = typename shard_type::const_mapped_type;
  using mapped_type = typename shard_type::mapped_type;

  using probe_seq_type = typename shard_type::probe_seq_type;
  using allocator_type = typename shard_type::allocator_type;

  using shard_read_pos_type = typename shard_type::read_pos;
  using shard_write_pos_type = typename shard_type::write_pos;

  template <typename InnerPos>
  using pos = sharded_pos<InnerPos>;
  using read_pos = sharded_read_pos<shard_read_pos_type>;
  using write_pos = sharded_write_pos<shard_write_pos_type>;

  using shard_const_iterator_type = typename shard_type::const_iterator;
  using shard_iterator_type = typename shard_type::iterator;

  template <typename Self, typename InnerIter, typename ShardsVec>
  class iterator_base : public wrapped_map_iterator<Self, InnerIter> {
   public:
    using base_type = wrapped_map_iterator<Self, InnerIter>;
    using self_type = typename base_type::self_type;
    using inner_type = typename base_type::inner_type;
    using difference_type = typename base_type::difference_type;

    using shards_vector_type = ShardsVec;

    using base_type::self;

    constexpr shard_idx_t min_shard_idx() const noexcept { return 0; }
    constexpr shard_idx_t max_shard_idx() const noexcept { return to_int(shards_->size() - 1); }

    constexpr bool is_left() const noexcept { return shard_idx_ <= min_shard_idx() && inner_.is_left(); }
    constexpr bool is_right() const noexcept { return shard_idx_ >= max_shard_idx() && inner_.is_right(); }

    constexpr self_type& operator++() {
      ++inner_;
      if (inner_.is_right()) {
        while (shard_idx_ < max_shard_idx()) {
          inner_ = (*shards_)[to_uint(++shard_idx_)].begin();
          if (!inner_.is_right()) return *self();
        }
      }
      return *self();
    }
    constexpr self_type& operator--() {
      --inner_;
      if (inner_.is_left()) {
        while (shard_idx_ > min_shard_idx()) {
          inner_ = (*shards_)[to_uint(--shard_idx_)].end();
          if (!(--inner_).is_left()) return *self();
        }
      }
      return *self();
    }

    constexpr self_type& operator+=(difference_type n) {
      if (n > 0) {
        while (n--) ++(*self());
      } else {
        while (n++) --(*self());
      }
      return *self();
    }
  
    template <typename RhsSelf, typename RhsInner, typename RhsOuter>
    constexpr friend difference_type operator-(const iterator_base& lhs, const iterator_base<RhsSelf, RhsInner, RhsOuter>& rhs) {
      iterator_base l{lhs};
      iterator_base<RhsSelf, RhsInner, RhsOuter> r{rhs};

      difference_type n{};
      for (; l < r; ++l) { --n; }
      for (; l > r; --l) { ++n; }
      return n;
    }

    template <typename RhsSelf, typename RhsInner, typename RhsOuter>
    constexpr friend bool operator==(const iterator_base& lhs, const iterator_base<RhsSelf, RhsInner, RhsOuter>& rhs) noexcept { return lhs.shard_idx_ == rhs.shard_idx_ && lhs.inner_ == rhs.inner_; }
    template <typename RhsSelf, typename RhsInner, typename RhsOuter>
    constexpr friend bool operator!=(const iterator_base& lhs, const iterator_base<RhsSelf, RhsInner, RhsOuter>& rhs) noexcept { return !(lhs == rhs); }
    template <typename RhsSelf, typename RhsInner, typename RhsOuter>
    constexpr friend bool operator<(const iterator_base& lhs, const iterator_base<RhsSelf, RhsInner, RhsOuter>& rhs) noexcept { return lhs.shard_idx_ < rhs.shard_idx_ || (lhs.shard_idx_ == rhs.shard_idx_ && lhs.inner_ < rhs.inner_); }
    template <typename RhsSelf, typename RhsInner, typename RhsOuter>
    constexpr friend bool operator>(const iterator_base& lhs, const iterator_base<RhsSelf, RhsInner, RhsOuter>& rhs) noexcept { return lhs.shard_idx_ > rhs.shard_idx_ || (lhs.shard_idx_ == rhs.shard_idx_ && lhs.inner_ > rhs.inner_); }
    template <typename RhsSelf, typename RhsInner, typename RhsOuter>
    constexpr friend bool operator<=(const iterator_base& lhs, const iterator_base<RhsSelf, RhsInner, RhsOuter>& rhs) noexcept { return !(lhs > rhs); }
    template <typename RhsSelf, typename RhsInner, typename RhsOuter>
    constexpr friend bool operator>=(const iterator_base& lhs, const iterator_base<RhsSelf, RhsInner, RhsOuter>& rhs) noexcept { return !(lhs < rhs); }

    constexpr friend void swap(iterator_base& lhs, iterator_base& rhs) noexcept {
      if (&lhs == &rhs) return;
      swap(static_cast<base_type&>(lhs), static_cast<base_type&>(rhs));

      std::swap(lhs.shards_, rhs.shards_);
      std::swap(lhs.shard_idx_, rhs.shard_idx_);
    }

   protected:
    using base_type::inner_;
    shards_vector_type* shards_;
    shard_idx_t shard_idx_;

    constexpr iterator_base(inner_type&& it, shards_vector_type& shards, shard_idx_t shard_idx) noexcept
      : base_type{std::move(it)}, shards_{&shards}, shard_idx_{shard_idx} {}

    iterator_base() = delete;
    constexpr iterator_base(const iterator_base&) noexcept = default;
    constexpr iterator_base& operator=(const iterator_base&) noexcept = default;
    constexpr iterator_base(iterator_base&&) noexcept = default;
    constexpr iterator_base& operator=(iterator_base&&) noexcept = default;

    template <typename, typename, typename>
    friend class iterator_base;
  };
  
  class const_iterator : public iterator_base<const_iterator, shard_const_iterator_type, const std::vector<shard_type>> {
   public:
    using base_type = iterator_base<const_iterator, shard_const_iterator_type, const std::vector<shard_type>>;
    using self_type = typename base_type::self_type;
    using inner_type = typename base_type::inner_type;
    using shards_vector_type = typename base_type::shards_vector_type;

    const_iterator() = delete;
    constexpr const_iterator(const const_iterator&) noexcept = default;
    constexpr const_iterator& operator=(const const_iterator&) noexcept = default;
    constexpr const_iterator(const_iterator&&) noexcept = default;
    constexpr const_iterator& operator=(const_iterator&&) noexcept = default;

   protected:
    constexpr const_iterator(inner_type&& it, shards_vector_type& shards, shard_idx_t shard_idx) noexcept
      : base_type{std::move(it), shards, shard_idx} {}

    friend class sharded;
  };

  class iterator : public iterator_base<iterator, shard_iterator_type, std::vector<shard_type>> {
   public:
    using base_type = iterator_base<iterator, shard_iterator_type, std::vector<shard_type>>;
    using self_type = typename base_type::self_type;
    using inner_type = typename base_type::inner_type;
    using shards_vector_type = typename base_type::shards_vector_type;

    iterator() = delete;
    constexpr iterator(const iterator&) noexcept = default;
    constexpr iterator& operator=(const iterator&) noexcept = default;
    constexpr iterator(iterator&&) noexcept = default;
    constexpr iterator& operator=(iterator&&) noexcept = default;

    constexpr bool erase() { return inner_.erase(); }

   protected:
    using base_type::inner_;

    constexpr iterator(inner_type&& it, shards_vector_type& shards, shard_idx_t shard_idx) noexcept
      : base_type{std::move(it), shards, shard_idx} {}

    friend class sharded;
  };

  using shard_prefetch_hint_type = typename shard_type::prefetch_hint;

  class prefetch_hint {
   public:
    constexpr prefetch_hint() noexcept : shard_idx_{} {}
    constexpr prefetch_hint(const key_type& key, bitmask_t shard_mask) noexcept : inner_{key} {
      shard_idx_ = align_pos(hash_to_shard_idx(inner_.hash()), shard_mask);
    }

    constexpr const shard_prefetch_hint_type& inner() const noexcept { return inner_; }
    constexpr hash_t hash() const noexcept { return inner_.hash(); }
    constexpr shard_idx_t shard_idx() const noexcept { return shard_idx_; }

    constexpr friend bool operator==(const prefetch_hint& lhs, const prefetch_hint& rhs) noexcept {
      return lhs.shard_idx_ == rhs.shard_idx_ && lhs.inner_ == rhs.inner_;
    }
    constexpr friend bool operator!=(const prefetch_hint& lhs, const prefetch_hint& rhs) noexcept { return !(lhs == rhs); }

   protected:
    shard_prefetch_hint_type inner_;
    shard_idx_t shard_idx_;
  };

  constexpr sharded() : sharded{1} {};
  template <typename... ShardArgs>
  constexpr sharded(const int_t num_shards, ShardArgs&&... shard_args) {
    if (num_shards < 1 || num_shards > max_num_shards || !has_single_bit(num_shards)) {
      throw std::out_of_range(
        "Number of shards must be within [1, max_num_shards] and a power of 2."
      );
    }
    
    shards_.reserve(to_uint(num_shards));
    for (int_t i{1}; i < num_shards; ++i) {
      shards_.emplace_back(shard_args...);
    }
    shards_.emplace_back(std::forward<ShardArgs>(shard_args)...);
  }

  using base_type::self;
  using base_type::cself;
  NVHM_CONTAINER_INTERFACE_()

  inline std::vector<blob_t> all_blobs_for(const key_type& key) const {
    return to_shard(key).first.all_blobs_for(key);
  }
  constexpr std::vector<value_type> all_values_for(const key_type& key) const {
    return to_shard(key).first.all_values_for(key);
  }

  template<typename InnerPos>
  constexpr const_entry_type at(const pos<InnerPos>& pos) const {
    return to_shard(pos).first.at(pos.inner_);
  }
  constexpr entry_type at(const write_pos& pos) {
    return to_shard(pos).first.at(pos.inner_);
  }

  constexpr const_iterator begin() const {
    const int_t n{num_shards()};
    for (int_t i{}; i < n; ++i) {
      shard_const_iterator_type it{shards_[to_uint(i)].begin()};
      if (!it.is_right()) {
        return {std::move(it), shards_, i};
      }
    }
    return {shards_.back().end(), shards_, n - 1};
  }
  constexpr iterator begin() {
    const int_t n{num_shards()};
    for (int_t i{}; i < n; ++i) {
      shard_iterator_type it{shards_[to_uint(i)].begin()};
      if (!it.is_right()) {
        return {std::move(it), shards_, i};
      }
    }
    return {shards_.back().end(), shards_, n - 1};
  }

  template<typename InnerPos>
  constexpr const blob_t* blob_at(const pos<InnerPos>& pos) const {
    return to_shard(pos).first.blob_at(pos.inner_);
  }
  constexpr blob_t* blob_at(const write_pos& pos) {
    return to_shard(pos).first.blob_at(pos.inner_);
  }

  template<typename InnerPos>
  constexpr int_t bucket_size_at(const pos<InnerPos>& pos) const {
    return to_shard(pos).first.bucket_size_at(pos.inner_);
  }
  template<typename InnerPos>
  constexpr int_t bucket_num_empty_slots_at(const pos<InnerPos>& pos) const {
    return to_shard(pos).first.bucket_num_empty_slots_at(pos.inner_);
  }
  template<typename InnerPos>
  constexpr int_t bucket_num_tombstone_slots_at(const pos<InnerPos>& pos) const {
    return to_shard(pos).first.bucket_num_tombstone_slots_at(pos.inner_);
  }

  constexpr int_t capacity() const {
    int_t n{};
    for (const shard_type& shard : shards_) n += shard.capacity();
    return n;
  }

  constexpr const_iterator cbegin() const { return begin(); }
  constexpr const_iterator cend() const { return end(); }

  constexpr bool check_integrity() const {
    for (const shard_type& shard : shards_) {
      if (!shard.check_integrity()) {
        return false;
      }
    }
    return true;
  }

  constexpr int_t clear() {
    int_t n{};
    for (shard_type& shard : shards_) n += shard.clear();
    return n;
  }
  constexpr int_t clear(int_t new_capacity) {
    new_capacity = ceil_div(new_capacity, num_shards());

    int_t n{};
    for (shard_type& shard : shards_) n += shard.clear(new_capacity);
    return n;
  }

  constexpr const conf_type& conf() const noexcept { return shards_.front().conf(); }

  constexpr bool contains(const key_type& key) const { return to_shard(key).first.contains(key); }
  constexpr bool contains(const key_type& key, const prefetch_hint& hint) const {
    return to_shard(key, hint).first.contains(key, hint.inner());
  }
  template<typename InnerPos>
  constexpr bool contains_at(const pos<InnerPos>& pos) const {
    return to_shard(pos).first.contains_at(pos.inner_);
  }

  constexpr int_t count(const key_type& key) const { return to_shard(key).first.count(key); }
  template <typename Pred>
  constexpr int_t count_if(const Pred& pred) const {
    static_assert(std::is_invocable_r_v<bool, Pred, read_pos>, "`pred` must be pred(read_pos) -> bool");

    int_t n{};
    for_each_shard_(
      [&](const shard_type& shard, shard_idx_t i) {
        n += shard.count_if(
          [&](shard_read_pos_type&& pos) { return pred(read_pos{std::move(pos), i}); }
        );
      }
    );
    return n;
  }
  constexpr void count_state_collisions(std::array<int_t, kernel_size>& counts) const {
    for (const shard_type& shard : shards_) shard.count_state_collisions(counts);
  }

  constexpr const_iterator end() const { return {shards_.back().end(), shards_, num_shards() - 1}; }
  constexpr iterator end() { return {shards_.back().end(), shards_, num_shards() - 1}; }

  constexpr bool erase(const key_type& key) {
    return to_shard(key).first.erase(key);
  }
  constexpr bool erase(const key_type& key, const prefetch_hint& hint) {
    return to_shard(key, hint).first.erase(key, hint.inner());
  }
  constexpr std::tuple<bool, write_pos, probe_seq_type> erase_first(const key_type& key) {
    auto [shard, i]{to_shard(key)};
    auto [b, p, s]{shard.erase_first(key)};
    return {b, write_pos{std::move(p), i}, std::move(s)};
  }
  constexpr std::tuple<bool, write_pos, probe_seq_type> erase_first(const key_type& key, const prefetch_hint& hint) {
    auto [shard, i]{to_shard(key, hint)};
    auto [b, p, s]{shard.erase_first(key, hint.inner())};
    return {b, write_pos{std::move(p), i}, std::move(s)};
  }
  template <typename PS>
  constexpr std::tuple<bool, write_pos, probe_seq_type> erase_next(write_pos&& pos, PS&& seq, const key_type& key) {
    auto [shard, i]{to_shard(pos, key)};
    auto [b, p, s]{shard.erase_next(std::move(pos.inner_), std::forward<PS>(seq), key)};
    return {b, write_pos{std::move(p), i}, std::move(s)};
  }
  template <typename PS>
  constexpr std::tuple<bool, write_pos, probe_seq_type> erase_next(write_pos&& pos, PS&& seq, const key_type& key, const prefetch_hint& hint) {
    auto [shard, i]{to_shard(pos, key, hint)};
    auto [b, p, s]{shard.erase_next(std::move(pos.inner_), std::forward<PS>(seq), key, hint.inner())};
    return {b, write_pos{std::move(p), i}, std::move(s)};
  }
  constexpr std::pair<bool, write_pos> erase_at(write_pos&& pos) {
    auto [b, p]{to_shard(pos).first.erase_at(std::move(pos.inner_))};
    return {b, write_pos{std::move(p), pos.shard_idx_}};
  }
  template <typename Pred>
  constexpr int_t erase_if(const Pred& pred) {
    static_assert(std::is_invocable_r_v<bool, Pred, write_pos>, "`pred` must be pred(const write_pos&) -> bool");
    static_assert(arg_type_v<arg_n_t<Pred, 0>> == arg_type_t::const_lvalue_ref, "`pred` must be pred(const write_pos&) -> bool");

    int_t n{};
    for_each_shard_(
      [&](shard_type& shard, shard_idx_t i) {
        n += shard.erase_if(
          [&](shard_write_pos_type&& pos) { return pred(write_pos{std::move(pos), i}); }
        );
      }
    );
    return n;
  }
  constexpr int_t erase_all(const key_type& key) { return to_shard(key).first.erase_all(key); }

  constexpr read_pos find(const key_type& key) const {
    auto [shard, i]{to_shard(key)};

    return read_pos{shard.find(key), i};
  }
  constexpr read_pos find(const key_type& key, const prefetch_hint& hint) const {
    auto [shard, i]{to_shard(key, hint)};

    return read_pos{shard.find(key, hint.inner()), i};
  }
  constexpr std::pair<read_pos, probe_seq_type> find_first(const key_type& key) const {
    auto [shard, i]{to_shard(key)};

    auto [p, s]{shard.find_first(key)};
    return {read_pos{std::move(p), i}, std::move(s)};
  }
  constexpr std::pair<read_pos, probe_seq_type> find_first(const key_type& key, const prefetch_hint& hint) const {
    auto [shard, i]{to_shard(key, hint)};

    auto [p, s]{shard.find_first(key, hint.inner())};
    return {read_pos{std::move(p), i}, std::move(s)};
  }
  template <typename InnerPos, typename PS>
  constexpr std::pair<read_pos, probe_seq_type> find_next(const pos<InnerPos>& pos, PS&& seq, const key_type& key) const {
    auto [shard, i]{to_shard(pos, key)};

    auto [p, s]{shard.find_next(pos.inner_, std::forward<PS>(seq), key)};
    return {read_pos{std::move(p), i}, std::move(s)};
  }
  template <typename InnerPos, typename PS>
  constexpr std::pair<read_pos, probe_seq_type> find_next(const pos<InnerPos>& pos, PS&& seq, const key_type& key, const prefetch_hint& hint) const {
    auto [shard, i]{to_shard(pos, key, hint)};

    auto [p, s]{shard.find_next(pos.inner_, std::forward<PS>(seq), key, hint.inner())};
    return {read_pos{std::move(p), i}, std::move(s)};
  }
  template <typename Pred>
  constexpr read_pos find_if(const Pred& pred) const {
    static_assert(std::is_invocable_r_v<bool, Pred, read_pos>, "`pred` must be pred(read_pos) -> bool");
    static_assert(arg_type_v<arg_n_t<Pred, 0>> != arg_type_t::lvalue_ref, "`pred` cannot be pred(read_pos&) -> bool");

    const shard_idx_t end{num_shards()};

    for (shard_idx_t i{}; i < end; ++i) {
      shard_read_pos_type pos{shards_[to_uint(i)].find_if(
        [&](shard_read_pos_type&& pos) { return pred(read_pos{std::move(pos), i}); }
      )};
      if (pos != npos) return read_pos{std::move(pos), i};
    }
    return read_npos();
  }
  constexpr std::vector<read_pos> find_all(const key_type& key) const {
    const auto [shard, i]{to_shard(key)};

    std::vector<read_pos> res;
    shard.for_each(key,
      [&](shard_read_pos_type&& pos, const probe_seq_type&) {
        res.emplace_back(std::move(pos), i);
      }
    );
    return res;
  }

  template <typename Func>
  constexpr void for_each(const Func& func) const {
    static_assert(std::is_invocable_v<Func, read_pos>, "`func` must be a `func(read_pos)`!");

    for_each_shard_(
      [&](const shard_type& shard, shard_idx_t i) {
        shard.for_each([&](shard_read_pos_type&& pos) { func(read_pos{std::move(pos), i}); });
      }
    );
  }
  template <typename Func>
  constexpr void for_each(const Func& func) {
    if constexpr (std::is_invocable_v<Func, read_pos>) {
      cself()->for_each(func);
    } else if constexpr (std::is_invocable_v<Func, write_pos>) {
      for_each_shard_(
        [&](shard_type& shard, shard_idx_t i) {
          shard.for_each([&](shard_write_pos_type&& pos) { func(write_pos{std::move(pos), i}); });
        }
      );
    } else {
      static_assert(dependent_false_v<Func>, "`func` must be a `func(read_pos)` or `func(write_pos)`!");
    }
  }
  template <typename Func>
  constexpr void for_each_blob(const Func& func) const {
    for (const shard_type& shard : shards_) shard.for_each_blob(func);
  }
  template <typename Func>
  constexpr void for_each_blob(const Func& func) {
    for (shard_type& shard : shards_) shard.for_each_blob(func);
  }
  template <typename Func>
  constexpr void for_each_entry(const Func& func) const {
    for (const shard_type& shard : shards_) shard.for_each_entry(func);
  }
  template <typename Func>
  constexpr void for_each_entry(const Func& func) {
    for (shard_type& shard : shards_) shard.for_each_entry(func);
  }
  template <typename Func>
  constexpr void for_each_key(const Func& func) const {
    for (const shard_type& shard : shards_) shard.for_each_key(func);
  }
  template <typename Func>
  constexpr void for_each_lru(const Func& func) const {
    for (const shard_type& shard : shards_) shard.for_each_lru(func);
  }
  template <typename Func>
  constexpr void for_each_state(const Func& func) const {
    for (const shard_type& shard : shards_) shard.for_each_state(func);
  }
  template <typename Func>
  constexpr void for_each_value(const Func& func) const {
    for (const shard_type& shard : shards_) shard.for_each_value(func);
  }
  template <typename Func>
  constexpr void for_each_value(const Func& func) {
    for (shard_type& shard : shards_) shard.for_each_value(func);
  }

  template <typename Func>
  constexpr void for_each(const key_type& key, const Func& func) const {
    static_assert(std::is_invocable_v<Func, read_pos, probe_seq_type>, "`func` must be `func(read_pos, probe_seq_type)`!");

    const auto [shard, i]{to_shard(key)};
    shard.for_each(key,
      [&](shard_read_pos_type&& pos, const probe_seq_type& seq) { func(read_pos{std::move(pos), i}, seq); }
    );
  }
  template <typename Func>
  constexpr void for_each(const key_type& key, const Func& func) {
    const auto [shard, i]{to_shard(key)};

    if constexpr (std::is_invocable_v<Func, read_pos, probe_seq_type>) {
      cself()->for_each(key, func);
    } else if constexpr (std::is_invocable_v<Func, write_pos, probe_seq_type>) {
      shard.for_each(key,
        [&](shard_write_pos_type&& pos, const probe_seq_type& seq) { func(write_pos{std::move(pos), i}, seq); }
      );
    } else {
      static_assert(dependent_false_v<Func>, "`func` must be `func(read_pos, probe_seq_type)` or `func(write_pos, probe_seq_type)`!");
    }
  }

  template <typename InnerPos>
  constexpr void get_blob_at(const pos<InnerPos>& pos, void* dst) const {
    to_shard(pos).first.get_blob_at(pos.inner_, dst);
  }
  template <typename InnerPos>
  constexpr void get_blob_at(const pos<InnerPos>& pos, void* dst, int_t n) const {
    to_shard(pos).first.get_blob_at(pos.inner_, dst, n);
  }

  constexpr int_t grow() {
    int_t n{};
    for (shard_type& shard : shards_) n += shard.grow();
    return n;
  }

  template <typename K>
  constexpr write_pos insert(K&& key) {
    auto [shard, i]{to_shard(key)};
    return write_pos{shard.insert(std::forward<K>(key)), i};
  }
  template <typename K>
  constexpr write_pos insert(K&& key, const prefetch_hint& hint) {
    auto [shard, i]{to_shard(key, hint)};
    return write_pos{shard.insert(std::forward<K>(key), hint.inner()), i};
  }
  template <typename K>
  constexpr std::tuple<write_pos, probe_seq_type, insert_op_t> insert_ex(K&& key) {
    auto [shard, i]{to_shard(key)};
    auto [p, s, op]{shard.insert_ex(std::forward<K>(key))};
    return {write_pos{std::move(p), i}, std::move(s), op};
  }
  template <typename K>
  constexpr std::tuple<write_pos, probe_seq_type, insert_op_t> insert_ex(K&& key, const prefetch_hint& hint) {
    auto [shard, i]{to_shard(key, hint)};
    auto [p, s, op]{shard.insert_ex(std::forward<K>(key), hint.inner())};
    return {write_pos{std::move(p), i}, std::move(s), op};
  }

  constexpr bool is_empty() const {
    for (const shard_type& shard : shards_) {
      if (!shard.is_empty()) return false;
    }
    return true;
  }
  constexpr bool is_full() const {
    for (const shard_type& shard : shards_) {
      if (!shard.is_full()) return false;
    }
    return true;
  }

  template <typename InnerPos>
  constexpr const key_type& key_at(const pos<InnerPos>& pos) const {
    return to_shard(pos).first.key_at(pos.inner_);
  }
  
  constexpr double load_factor() const noexcept {
    int_t s{}, c{};
    for (const shard_type& shard : shards_) {
      s += shard.size();
      c += shard.capacity();
    }
    return static_cast<double>(s) / static_cast<double>(c);
  }

  template<typename InnerPos>
  constexpr lru_t lru_at(const pos<InnerPos>& pos) const {
    return to_shard(pos).first.lru_at(pos.inner_);
  }

  template <typename InnerPos>
  constexpr const_mapped_type mapped_at(const pos<InnerPos>& pos) const {
    return to_shard(pos).first.mapped_at(pos.inner_);
  }
  constexpr mapped_type mapped_at(const write_pos& pos) {
    return to_shard(pos).first.mapped_at(pos.inner_);
  }

  constexpr double max_load_factor() const {
    return shards_.front().max_load_factor();
  }

  constexpr int_t num_empty_slots() const {
    int_t n{};
    for (const shard_type& shard : shards_) n += shard.num_empty_slots();
    return n;
  }
  constexpr int_t num_not_hash_slots() const {
    int_t n{};
    for (const shard_type& shard : shards_) n += shard.num_not_hash_slots();
    return n;
  }
  constexpr int_t num_tombstone_slots() const {
    int_t n{};
    for (const shard_type& shard : shards_) n += shard.num_tombstone_slots();
    return n;
  }
  constexpr shard_idx_t num_shards() const noexcept { return to_int(shards_.size()); }

  constexpr read_pos read_npos() const noexcept { return read_pos{shards_.back().read_npos(), num_shards() - 1}; }

  constexpr prefetch_hint read_prefetch(const key_type& key) const {
    prefetch_hint hint{key, shard_mask_()};
    shards_[to_uint(hint.shard_idx())].read_prefetch(key, hint.inner());
    return hint;
  }
  constexpr void read_prefetch(const key_type& key, const prefetch_hint& hint) const {
    to_shard(key, hint).first.read_prefetch(key, hint.inner());
  }
  template<typename InnerPos>
  constexpr void read_prefetch_value_at(const pos<InnerPos>& pos) const {
    to_shard(pos).first.read_prefetch_value_at(pos.inner_);
  }
  template<typename InnerPos>
  constexpr void read_prefetch_blob_at(const pos<InnerPos>& pos) const {
    to_shard(pos).first.read_prefetch_blob_at(pos.inner_);
  }

  constexpr void render(std::ostream& os, bool with_values = has_values, blob_render_t with_blobs = has_blobs ? blob_render_t::size : blob_render_t::hide) const {
    for (const shard_type& shard : shards_) {
      shard.render(os, with_values, with_blobs);
    }
  }
  
  constexpr int_t reserve(int_t new_capacity) {
    new_capacity = ceil_div(new_capacity, num_shards());

    int_t n{};
    for (shard_type& shard : shards_) n += shard.reserve(new_capacity);
    return n;
  }

  constexpr int_t resize(int_t new_capacity) {
    new_capacity = ceil_div(new_capacity, num_shards());

    int_t n{};
    for (shard_type& shard : shards_) n += shard.resize(new_capacity);
    return n;
  }

  constexpr int_t scrub(lru_t threshold = max_lru) {
    int_t n{};
    for (shard_type& shard : shards_) n += shard.scrub(threshold);
    return n;
  }

  constexpr void set_blob_at(const write_pos& pos, const void* src) {
    to_shard(pos).first.set_blob_at(pos.inner_, src);
  }
  constexpr void set_blob_at(const write_pos& pos, const void* src, int_t n) {
    to_shard(pos).first.set_blob_at(pos.inner_, src, n);
  }

  template<typename V>
  constexpr void set_value_at(const write_pos& pos, V&& value) {
    to_shard(pos).first.set_value_at(pos.inner_, std::forward<V>(value));
  }

  constexpr int_t shrink() {
    int_t n{};
    for (shard_type& shard : shards_) n += shard.shrink();
    return n;
  }

  constexpr int_t size() const {
    int_t n{};
    for (const shard_type& shard : shards_) n += shard.size();
    return n;
  }

  template <typename InnerPos>
  constexpr state_t state_at(const pos<InnerPos>& pos) const {
    return to_shard(pos).first.state_at(pos.inner_);
  }

  template <typename InnerPos>
  constexpr const_iterator to_iterator(pos<InnerPos>&& pos) const {
    auto [shard, i]{to_shard(pos)};
    return {shard.to_iterator(std::move(pos.inner_)), shards_, i};
  }
  constexpr iterator to_iterator(write_pos&& pos) {
    auto [shard, i]{to_shard(pos)};
    return {shard.to_iterator(std::move(pos.inner_)), shards_, i};
  }
  template <typename InnerPos>
  constexpr const_iterator to_citerator(pos<InnerPos>&& pos) const { return to_iterator(std::move(pos)); }

  constexpr std::pair<const shard_type&, shard_idx_t> to_shard(const key_type& key) const noexcept {
    shard_idx_t i{to_shard_idx(key)};
    return {shards_[to_uint(i)], i};
  }
  constexpr std::pair<shard_type&, shard_idx_t> to_shard(const key_type& key) noexcept {
    shard_idx_t i{to_shard_idx(key)};
    return {shards_[to_uint(i)], i};
  }
  constexpr std::pair<const shard_type&, shard_idx_t> to_shard(const key_type& key, const prefetch_hint& hint) const {
    shard_idx_t i{to_shard_idx(key, hint)};
    return {shards_[to_uint(i)], i};
  }
  constexpr std::pair<shard_type&, shard_idx_t> to_shard(const key_type& key, const prefetch_hint& hint) {
    shard_idx_t i{to_shard_idx(key, hint)};
    return {shards_[to_uint(i)], i};
  }

  template<typename InnerPos>
  constexpr std::pair<const shard_type&, shard_idx_t> to_shard(const pos<InnerPos>& pos) const {
    shard_idx_t i{to_shard_idx(pos)};
    return {shards_[to_uint(i)], i};
  }
  constexpr std::pair<shard_type&, shard_idx_t> to_shard(const write_pos& pos) {
    shard_idx_t i{to_shard_idx(pos)};
    return {shards_[to_uint(i)], i};
  }
  template<typename InnerPos>
  constexpr std::pair<const shard_type&, shard_idx_t> to_shard(const pos<InnerPos>& pos, const key_type& key) const {
    shard_idx_t i{to_shard_idx(pos, key)};
    return {shards_[to_uint(i)], i};
  }
  constexpr std::pair<shard_type&, shard_idx_t> to_shard(const write_pos& pos, const key_type& key) {
    shard_idx_t i{to_shard_idx(pos, key)};
    return {shards_[to_uint(i)], i};
  }
  template<typename InnerPos>
  constexpr std::pair<const shard_type&, shard_idx_t> to_shard(const pos<InnerPos>& pos, const key_type& key, const prefetch_hint& hint) const {
    shard_idx_t i{to_shard_idx(pos, key, hint)};
    return {shards_[to_uint(i)], i};
  }
  constexpr std::pair<shard_type&, shard_idx_t> to_shard(const write_pos& pos, const key_type& key, const prefetch_hint& hint) {
    shard_idx_t i{to_shard_idx(pos, key, hint)};
    return {shards_[to_uint(i)], i};
  }

  constexpr shard_idx_t to_shard_idx(const key_type& key) const noexcept {
    return align_pos(hash_to_shard_idx(key_to_hash(key)), shard_mask_());
  }
  constexpr shard_idx_t to_shard_idx(const key_type& key, const prefetch_hint& hint) const {
    NVHM_ASSUME_((prefetch_hint{key, shard_mask_()} == hint));
    return hint.shard_idx();
  }
  template<typename InnerPos>
  constexpr shard_idx_t to_shard_idx(const pos<InnerPos>& pos) const {
    shard_idx_t i{pos.shard_idx_};
    NVHM_ASSUME_(i >= 0 && i < num_shards(), "i = ", i, ", num_shards = ", num_shards());
    return i;
  }
  template<typename InnerPos>
  constexpr shard_idx_t to_shard_idx(const pos<InnerPos>& pos, const key_type& key) const {
    shard_idx_t i{to_shard_idx(pos)};
    NVHM_ASSUME_(i == to_shard_idx(key), "i = ", i, ", to_shard_idx(key) = ", to_shard_idx(key));
    return i;
  }
  template<typename InnerPos>
  constexpr shard_idx_t to_shard_idx(const pos<InnerPos>& pos, const key_type& key, const prefetch_hint& hint) const {
    shard_idx_t i{to_shard_idx(pos, key)};
    NVHM_ASSUME_(i == hint.shard_idx(), "i = ", i, ", hint.shard_idx = ", hint.shard_idx());
    return i;
  }

  constexpr write_pos update(const key_type& key) {
    auto [shard, i]{to_shard(key)};
    return write_pos{shard.update(key), i};
  }
  constexpr write_pos update(const key_type& key, const prefetch_hint& hint) {
    auto [shard, i]{to_shard(key, hint)};
    return write_pos{shard.update(key, hint.inner()), i};
  }
  constexpr std::pair<write_pos, probe_seq_type> update_first(const key_type& key) {
    auto [shard, i]{to_shard(key)};
    auto [p, s]{shard.update_first(key)};
    return {write_pos{std::move(p), i}, std::move(s)};
  }
  constexpr std::pair<write_pos, probe_seq_type> update_first(const key_type& key, const prefetch_hint& hint) {
    auto [shard, i]{to_shard(key, hint)};
    auto [p, s]{shard.update_first(key, hint.inner())};
    return {write_pos{std::move(p), i}, std::move(s)};
  }
  template <typename PS>
  constexpr std::pair<write_pos, probe_seq_type> update_next(write_pos&& pos, PS&& seq, const key_type& key) {
    auto [shard, i]{to_shard(pos, key)};
    auto [p, s]{shard.update_next(std::move(pos.inner_), std::forward<PS>(seq), key)};
    return {write_pos{std::move(p), i}, std::move(s)};
  }
  template <typename PS>
  constexpr std::pair<write_pos, probe_seq_type> update_next(write_pos&& pos, PS&& seq, const key_type& key, const prefetch_hint& hint) {
    auto [shard, i]{to_shard(pos, key, hint)};
    auto [p, s]{shard.update_next(std::move(pos.inner_), std::forward<PS>(seq), key, hint.inner())};
    return {write_pos{std::move(p), i}, std::move(s)};
  }
  template <typename Pred>
  constexpr write_pos update_if(const Pred& pred) {
    static_assert(std::is_invocable_r_v<bool, Pred, write_pos>, "`pred` must be pred(const write_pos&) -> bool");
    static_assert(arg_type_v<arg_n_t<Pred, 0>> == arg_type_t::const_lvalue_ref, "`pred` must be pred(const write_pos&) -> bool");

    const shard_idx_t end{num_shards()};

    for (shard_idx_t i{}; i < end; ++i) {
      shard_write_pos_type pos{shards_[to_uint(i)].update_if(
        [&](shard_write_pos_type&& pos) { return pred(write_pos{std::move(pos), i}); }
      )};
      if (pos != npos) return write_pos{std::move(pos), i};
    }
    return write_npos();
  }
  constexpr std::vector<write_pos> update_all(const key_type& key) {
    auto [shard, i]{to_shard(key)};

    std::vector<write_pos> res;
    shard.for_each(key,
      [&](shard_write_pos_type&& pos, const probe_seq_type&) {
        res.emplace_back(std::move(pos), i);
      }
    );
    return res;
  }

  template<typename InnerPos>
  constexpr const value_type& value_at(const pos<InnerPos>& pos) const {
    return to_shard(pos).first.value_at(pos.inner_);
  }
  constexpr value_type& value_at(const write_pos& pos) {
    return to_shard(pos).first.value_at(pos.inner_);
  }

  constexpr write_pos write_npos() noexcept {
    return write_pos{shards_.back().write_npos(), num_shards() - 1};
  }

  constexpr prefetch_hint write_prefetch(const key_type& key) {
    prefetch_hint hint{key, shard_mask_()};
    shards_[to_uint(hint.shard_idx())].write_prefetch(key, hint.inner());
    return hint;
  }
  constexpr void write_prefetch(const key_type& key, const prefetch_hint& hint) {
    to_shard(key, hint).first.write_prefetch(key, hint.inner());
  }
  constexpr void write_prefetch_value_at(const write_pos& pos) {
    to_shard(pos).first.write_prefetch_value_at(pos.inner_);
  }
  constexpr void write_prefetch_blob_at(const write_pos& pos) {
    to_shard(pos).first.write_prefetch_blob_at(pos.inner_);
  }

  template <typename K>
  constexpr mapped_type operator[](K&& key) { return to_shard(key).first[std::forward<K>(key)]; }

  constexpr friend bool operator==(const sharded& lhs, const sharded& rhs) {
    return (&lhs == &rhs) || (lhs.shards_ == rhs.shards_);
  }
  constexpr friend bool operator!=(const sharded& lhs, const sharded& rhs) { return !(lhs == rhs); }

  constexpr friend void swap(sharded& lhs, sharded& rhs) {
    if (&lhs == &rhs) return;

    std::swap(lhs.shards_, rhs.shards_);
  }

 protected:
  std::vector<shard_type> shards_;

  template <typename Func>
  constexpr void for_each_shard_(Func func) const {
    static_assert(std::is_invocable_v<Func, const shard_type&, shard_idx_t>, "`func` must be a valid function!");
    const shard_idx_t end{num_shards()};

    for (shard_idx_t i{}; i < end; ++i) {
      func(shards_[to_uint(i)], i);
    }
  }
  template <typename Func>
  constexpr void for_each_shard_(Func func) {
    static_assert(std::is_invocable_v<Func, shard_type&, shard_idx_t>, "`func` must be a valid function!");
    const shard_idx_t end{num_shards()};

    for (shard_idx_t i{}; i < end; ++i) {
      func(shards_[to_uint(i)], i);
    }
  }

  constexpr bitmask_t shard_mask_() const noexcept { return make_size_mask(num_shards()); }
};

#pragma GCC diagnostic pop
}