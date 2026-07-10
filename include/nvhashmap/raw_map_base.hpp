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

#include <optional>
#include "container.hpp"
#include "allocator.hpp"
#include "kernel.hpp"
#include "memory.hpp"

namespace nvhm {

class raw_map_pos : public wrapped_pos<raw_pos_t> {
 public:
  using base_type = wrapped_pos<raw_pos_t>;

  raw_map_pos() = delete;

  constexpr raw_pos_t inner() const noexcept { return inner_; }

  constexpr friend bool operator==(const raw_map_pos& lhs, const raw_map_pos& rhs) noexcept { return lhs.inner_ == rhs.inner_; }
  constexpr friend bool operator!=(const raw_map_pos& lhs, const raw_map_pos& rhs) noexcept { return !(lhs == rhs); }
  constexpr friend bool operator<(const raw_map_pos& lhs, const raw_map_pos& rhs) noexcept { return lhs.inner_ < rhs.inner_; }
  constexpr friend bool operator>(const raw_map_pos& lhs, const raw_map_pos& rhs) noexcept { return lhs.inner_ > rhs.inner_; }
  constexpr friend bool operator<=(const raw_map_pos& lhs, const raw_map_pos& rhs) noexcept { return !(lhs > rhs); }
  constexpr friend bool operator>=(const raw_map_pos& lhs, const raw_map_pos& rhs) noexcept { return !(lhs < rhs); }

 protected:
  constexpr explicit raw_map_pos(const raw_map_pos&) noexcept = default;
  constexpr raw_map_pos& operator=(const raw_map_pos&) noexcept = default;
  constexpr explicit raw_map_pos(raw_map_pos&&) noexcept = default;
  constexpr raw_map_pos& operator=(raw_map_pos&&) noexcept = default;

  constexpr explicit raw_map_pos(raw_pos_t inner) noexcept : base_type{inner} {}
};

class raw_map_read_pos : public raw_map_pos {
 public:
  using base_type = raw_map_pos;

  raw_map_read_pos() = delete;
  constexpr raw_map_read_pos(const raw_map_read_pos&) noexcept = default;
  constexpr raw_map_read_pos& operator=(const raw_map_read_pos&) noexcept = default;
  constexpr raw_map_read_pos(raw_map_read_pos&&) noexcept = default;
  constexpr raw_map_read_pos& operator=(raw_map_read_pos&&) noexcept = default;

  constexpr explicit raw_map_read_pos(raw_pos_t inner) noexcept : base_type{inner} {}
};

class raw_map_write_pos : public raw_map_pos {
 public:
  using base_type = raw_map_pos;

  raw_map_write_pos() = delete;
  constexpr raw_map_write_pos(const raw_map_write_pos&) noexcept = default;
  constexpr raw_map_write_pos& operator=(const raw_map_write_pos&) noexcept = default;
  constexpr raw_map_write_pos(raw_map_write_pos&&) noexcept = default;
  constexpr raw_map_write_pos& operator=(raw_map_write_pos&&) noexcept = default;

  constexpr explicit raw_map_write_pos(raw_pos_t inner) noexcept : base_type{inner} {}
};

template <typename Self, typename Conf, typename ProbeSeq, typename Allocator>
class raw_map_base : public container<Self> {
 public:
  using base_type = container<Self>;
  using self_type = typename base_type::self_type;
  using conf_type = Conf;

  using key_type = typename conf_type::key_type;
  using value_type = typename conf_type::value_type;
  constexpr static bool has_values{conf_type::has_values};
  constexpr static flags_t flags{conf_type::flags};
  constexpr static bool has_blobs{conf_type::has_blobs};
  constexpr static int_t kernel_size{conf_type::kernel_size};
  using const_value_type = const_view_t<value_type>;

  using const_entry_type = const_entry_t<key_type, value_type, has_blobs>;
  using entry_type = entry_t<key_type, value_type, has_blobs>;
  using const_mapped_type = const_mapped_t<value_type, has_blobs>;
  using mapped_type = mapped_t<value_type, has_blobs>;

  using probe_seq_type = ProbeSeq;
  using allocator_type = Allocator;

  using pos = raw_map_pos;
  using read_pos = raw_map_read_pos;
  using write_pos = raw_map_write_pos;

  template <typename IterSelf, typename Outer>
  class iterator_base : public wrapped_iterator<IterSelf, raw_pos_t, raw_pos_t> {
   public:
    using base_type = wrapped_iterator<IterSelf, raw_pos_t, raw_pos_t>;
    using self_type = typename base_type::self_type;
    using difference_type = typename base_type::difference_type;

    using iterator_category = std::bidirectional_iterator_tag;

    using outer_type = Outer;
    constexpr static bool is_readonly{std::is_const_v<outer_type>};
    using key_type = typename outer_type::key_type;
    using value_type = std::conditional_t<is_readonly, typename outer_type::const_value_type, typename outer_type::value_type&>;
    using blob_type = std::conditional_t<is_readonly, const blob_t*, blob_t*>;
    using entry_type = std::conditional_t<is_readonly, typename outer_type::const_entry_type, typename outer_type::entry_type>;
    using mapped_type = std::conditional_t<is_readonly, typename outer_type::const_mapped_type, typename outer_type::mapped_type>;
    using entry_ptr_type = std::remove_reference_t<entry_type>;

    iterator_base() = delete;

    using base_type::self;

    constexpr blob_type blob() const noexcept { return outer_->blob_at_(inner_); }
    constexpr entry_type entry() const noexcept { return outer_->at_(inner_); }
    constexpr const key_type& key() const noexcept { return outer_->key_at_(inner_); }
    constexpr lru_t lru() const noexcept { return outer_->lru_at_(inner_); }
    constexpr bool query() const noexcept { return outer_->contains_at_(inner_); }
    constexpr value_type value() const noexcept { return outer_->value_at_(inner_); }
    constexpr mapped_type mapped() const noexcept { return outer_->mapped_at_(inner_); }

    constexpr entry_type operator*() const noexcept { return entry(); }
    constexpr const entry_ptr_type* operator->() const noexcept {
      cache_.emplace(entry());
      return &cache_.value();
    }
    
    constexpr bool is_left() const noexcept { return inner_ < 0; }
    constexpr bool is_right() const noexcept { return inner_ >= outer_->capacity(); }

    constexpr self_type& operator++() {
      raw_pos_t& __restrict pos{inner_};
      if (pos < -1) {
        ++pos;
        return *self();
      }

      const raw_pos_t end{outer_->capacity()};
      while (NVHM_LIKELY_(++pos < end)) {
        if (outer_->contains_at_(pos)) return *self();
      }
      return *self();
    }
    constexpr self_type& operator--() {
      raw_pos_t& __restrict pos{inner_};
      if (pos > outer_->capacity()) {
        --pos;
        return *self();
      }

      while (NVHM_LIKELY_(--pos >= 0)) {
        if (outer_->contains_at_(pos)) return *self();
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

    template <typename RhsSelf, typename RhsOuter>
    constexpr friend difference_type operator-(const iterator_base& lhs, const iterator_base<RhsSelf, RhsOuter>& rhs) {
      iterator_base l{lhs};
      iterator_base<RhsSelf, RhsOuter> r{rhs};

      difference_type n{};
      for (; l < r; ++l) { --n; }
      for (; l > r; --l) { ++n; }
      return n;
    }

    template <typename RhsSelf, typename RhsOuter>
    constexpr friend bool operator==(const iterator_base& lhs, const iterator_base<RhsSelf, RhsOuter>& rhs) noexcept { return lhs.inner_ == rhs.inner_; }
    template <typename RhsSelf, typename RhsOuter>
    constexpr friend bool operator!=(const iterator_base& lhs, const iterator_base<RhsSelf, RhsOuter>& rhs) noexcept { return !(lhs == rhs); }
    template <typename RhsSelf, typename RhsOuter>
    constexpr friend bool operator<(const iterator_base& lhs, const iterator_base<RhsSelf, RhsOuter>& rhs) noexcept { return lhs.inner_ < rhs.inner_; }
    template <typename RhsSelf, typename RhsOuter>
    constexpr friend bool operator>(const iterator_base& lhs, const iterator_base<RhsSelf, RhsOuter>& rhs) noexcept { return lhs.inner_ > rhs.inner_; }
    template <typename RhsSelf, typename RhsOuter>
    constexpr friend bool operator<=(const iterator_base& lhs, const iterator_base<RhsSelf, RhsOuter>& rhs) noexcept { return !(lhs > rhs); }
    template <typename RhsSelf, typename RhsOuter>
    constexpr friend bool operator>=(const iterator_base& lhs, const iterator_base<RhsSelf, RhsOuter>& rhs) noexcept { return !(lhs < rhs); }

    constexpr friend void swap(iterator_base& lhs, iterator_base& rhs) noexcept {
      if (&lhs == &rhs) return;
      swap(static_cast<base_type&>(lhs), static_cast<base_type&>(rhs));

      std::swap(lhs.outer_, rhs.outer_);
      lhs.cache_.reset();
      rhs.cache_.reset();
    }

   protected:
    using base_type::inner_;
    outer_type* outer_;
    mutable std::optional<entry_ptr_type> cache_;

    constexpr iterator_base(const iterator_base& other) noexcept
      : base_type{other}, outer_{other.outer_} {}
    constexpr iterator_base& operator=(const iterator_base& other) noexcept {
      base_type::operator=(other);
      outer_ = other.outer_;
      cache_.reset();
      return *this;
    }
    constexpr iterator_base(iterator_base&& other) noexcept
      : base_type{std::move(other)}, outer_{std::move(other.outer_)} {}
    constexpr iterator_base& operator=(iterator_base&& other) noexcept {
      base_type::operator=(std::move(other));
      outer_ = std::move(other.outer_);
      cache_.reset();
      other.cache_.reset();
      return *this;
    }

    constexpr explicit iterator_base(raw_pos_t inner, outer_type* outer) noexcept
      : base_type{inner}, outer_{outer} {}

    template <typename, typename>
    friend class iterator_base;
  };

  class const_iterator : public iterator_base<const_iterator, const Self> {
   public:
    using base_type = iterator_base<const_iterator, const Self>;
    using self_type = typename base_type::self_type;
    using outer_type = typename base_type::outer_type;

    const_iterator() = delete;
    constexpr const_iterator(const const_iterator&) noexcept = default;
    constexpr const_iterator& operator=(const const_iterator&) noexcept = default;
    constexpr const_iterator(const_iterator&&) noexcept = default;
    constexpr const_iterator& operator=(const_iterator&&) noexcept = default;

   protected:
    constexpr const_iterator(outer_type* outer, raw_pos_t inner) noexcept : base_type{inner, outer} {}

    friend class raw_map_base;
  };

  class iterator : public iterator_base<iterator, Self> {
   public:
    using base_type = iterator_base<iterator, Self>;
    using self_type = typename base_type::self_type;
    using outer_type = typename base_type::outer_type;
    
    iterator() = delete;
    constexpr iterator(const iterator&) noexcept = default;
    constexpr iterator& operator=(const iterator&) noexcept = default;
    constexpr iterator(iterator&&) noexcept = default;
    constexpr iterator& operator=(iterator&&) noexcept = default;

    constexpr bool erase() const { return outer_->erase_at_(inner_).first; }

   protected:
    using base_type::inner_;
    using base_type::outer_;

    constexpr iterator(outer_type* outer, raw_pos_t inner) noexcept : base_type{inner, outer} {}

    friend class raw_map_base;
  };

  template <typename P>
  constexpr static bool is_pos_v{std::is_same_v<P, read_pos> || std::is_same_v<P, write_pos>};

  class prefetch_hint {
   public:
    constexpr explicit prefetch_hint() noexcept : hash_{} {}
    constexpr explicit prefetch_hint(const key_type& key) noexcept : hash_{key_to_hash(key)} {}
    
    constexpr hash_t hash() const noexcept { return hash_; }

    constexpr friend bool operator==(const prefetch_hint& lhs, const prefetch_hint& rhs) noexcept { return lhs.hash_ == rhs.hash_; }
    constexpr friend bool operator!=(const prefetch_hint& lhs, const prefetch_hint& rhs) noexcept { return !(lhs == rhs); }

   protected:
    hash_t hash_;
  };

  raw_map_base() = delete;
  constexpr raw_map_base(const raw_map_base& other)
    : base_type{other}, conf_{other.conf_}, bucket_mask_{other.bucket_mask_} {
    const raw_pos_t end{capacity()};
    if constexpr (has_values) {
      values_ = make_copy(other.values_, end);
    }
    if constexpr (has_blobs) {
      blobs_ = make_copy(other.blobs_, end * conf_.blob_stride());
    }
  }
  constexpr raw_map_base& operator=(const raw_map_base& other) {
    if (self() == &other) return *self();
    base_type::operator=(other);

    raw_pos_t end{other.capacity()};
    bool need_alloc{end != capacity() || other.conf_.blob_stride() != conf_.blob_stride()};

    conf_ = other.conf_;
    if (need_alloc) {
      end = self()->alloc_(end);
    }
    NVHM_ASSERT_(end == capacity(), "end = ", end, ", capacity = ", capacity());

    if constexpr (has_values) {
      std::copy_n(other.values_.get(), end, values_.get());
    }
    if constexpr (has_blobs) {
      std::copy_n(other.blobs_.get(), end * conf_.blob_stride(), blobs_.get());
    }
    return *self();
  }
  constexpr raw_map_base(raw_map_base&& other) noexcept = default;
  constexpr raw_map_base& operator=(raw_map_base&& other) noexcept = default;

  constexpr raw_map_base(const conf_type& conf) : conf_{conf} {
    if (!conf.is_valid()) {
      throw std::invalid_argument("`conf` is not valid!");
    }
    const raw_pos_t end{conf.init_capacity()};

    bucket_mask_ = make_aligned_mask<kernel_size>(conf.init_capacity());
    if constexpr (has_values) {
      values_ = make_unique<value_type, allocator_type>(end);
    }
    if constexpr (has_blobs) {
      blobs_ = make_unique<std::byte, allocator_type>(end * conf_.blob_stride());
    }
  }

  using base_type::self;
  NVHM_CONTAINER_INTERFACE_()

  /**
   * Fetch all blobs for a matching key in the container.
   *
   * @param key The key to query.
   * @return Pointers to the blobs of all matchin entries.
   */
  inline std::vector<blob_t> all_blobs_for(const key_type& key) const {
    static_assert(has_blobs, "`all_blobs_for` is not supported for this configuration!");
    const int_t blob_size{conf_.blob_size()};
    const int_t blob_stride{conf_.blob_stride()};
    const blob_t* const __restrict blobs{blobs_.get()};

    std::vector<blob_t> res(to_uint(count(key) * blob_size));
    size_t n{};
    self()->for_each_(key, key_to_hash(key),
      [&](raw_pos_t pos, const probe_seq_type&) {
        std::copy_n(&blobs[pos * blob_stride], blob_size, &res[n]);
        n += to_uint(blob_size);
      }
    );
    NVHM_ASSERT_(n == res.size(), "n = ", n, ", res.size() = ", res.size());
    return res;
  }
  /**
   * Fetch all values for a matching key in the container.
   *
   * @param key The key to query.
   * @return The values of all matching entries.
   */
  constexpr std::vector<value_type> all_values_for(const key_type& key) const {
    static_assert(has_values, "`all_values_for` is not supported for this configuration!");
    const value_type* const __restrict values{values_.get()};

    std::vector<value_type> res(to_uint(count(key)));
    size_t n{};
    self()->for_each_(key, key_to_hash(key),
      [&](raw_pos_t pos, const probe_seq_type&) { res[n++] = values[pos]; }
    );
    NVHM_ASSERT_(n == res.size(), "n = ", n, ", res.size() = ", res.size());
    return res;
  }

  /**
   * Fetch the entry at the provided position.
   *
   * @param pos The position to query.
   * @return The associated entry.
   */
  constexpr const_entry_type at(const pos& pos) const { return self()->at_(validate(pos)); }
  /**
   * Fetch the entry at the provided position.
   *
   * @param pos The position to query.
   * @return The associated entry.
   */
  constexpr entry_type at(const write_pos& pos) { return self()->at_(validate(pos)); }

  /**
   * @return Iterator pointing to beginning of the data structure.
   */
  constexpr const_iterator begin() const noexcept {
    raw_pos_t pos{front_()};
    return {self(), pos == npos ? capacity() : pos};
  }
  /**
   * @return Iterator pointing to beginning of the data structure.
   */
  constexpr iterator begin() noexcept {
    raw_pos_t pos{front_()};
    return {self(), pos == npos ? capacity() : pos};
  }

  /**
   * Fetch pointer to the blob at `pos`.
   *
   * @param pos The position to query.
   * @return Pointer the associated blob.
   */
  constexpr const blob_t* blob_at(const pos& pos) const { return blob_at_(validate(pos)); }
  /**
   * Fetch pointer to the blob at `pos`.
   *
   * @param pos The position to query.
   * @return Pointer the associated blob.
   */
  constexpr blob_t* blob_at(const write_pos& pos) { return blob_at_(validate(pos)); }

  /**
   * @return The current storage capacity of the data structure.
   */
  constexpr int_t capacity() const noexcept { return aligned_mask_to_capacity<kernel_size>(bucket_mask_); }

  /**
   * @return Iterator pointing to beginning of the data structure.
   */
  constexpr const_iterator cbegin() const noexcept { return begin(); }
  /**
   * @return Iterator pointing to end of the data structure.
   */
  constexpr const_iterator cend() const noexcept { return end(); }

  /**
   * Check if the data structure is in a consistent state.
   *
   * @return `true` if no errors were found.
   */
  constexpr bool check_integrity() const noexcept {
    return self()->check_integrity_();
  }

  /**
   * Mark all slots as free.
   */
  constexpr int_t clear() { return self()->clear_(); }
  /**
   * Evict all entries from the data structure.
   *
   * @param new_capacity Desired new capacity.
   * @return Actual new capacity.
   */
  constexpr int_t clear(int_t new_capacity) {
    new_capacity = bit_ceil(std::max(new_capacity, conf_.min_capacity()));
    if (new_capacity != capacity()) {
      self()->alloc_(new_capacity);
    }
    return self()->reset_();
  }

  /**
   * @return The arguments of the data structure.
   */
  constexpr const conf_type& conf() const noexcept { return conf_; }

  /**
   * Determine whether the provided `key` is stored in the data structure.
   *
   * @param key The key to check.
   * @return Whether or not the key could be located.
   */
  constexpr bool contains(const key_type& key) const noexcept { return find(key) != npos; }
  /**
   * Determine whether the provided `key` is stored in the data structure.
   *
   * @param key The key to check.
   * @param hint Hint from preceeding call to `read_prefetch`.
   * @return Whether or not the key could be located.
   */
  constexpr bool contains(const key_type& key, const prefetch_hint& hint) const {
    return find(key, hint) != npos;
  }
  /**
   * Determine if the `pos` contains actual data.
   *
   * @param pos The position to query.
   * @return True if the slot contains actual data.
   */
  constexpr bool contains_at(const pos& pos) const { return self()->contains_at_(validate_range(pos)); }

  /**
   * Count the number of entries in the data structure that match the key.
   *
   * @param key The key to check.
   * @return The number of entries that match.
   */
  constexpr int_t count(const key_type& key) const noexcept {
    int_t n{};
    self()->for_each_(key, key_to_hash(key),
      [&](raw_pos_t, const probe_seq_type&) { ++n; }
    );
    return n;
  }
  /**
   * Count the number of entries in the data structure that match the predicate.
   *
   * @param f Predicate function to determine whether an entry matches.
   * @return The number of entries that match.
   */
  template <typename Pred>
  constexpr int_t count_if(const Pred& pred) const {
    static_assert(std::is_invocable_r_v<bool, Pred, read_pos>, "`pred` must be pred(read_pos) -> bool");
    int_t n{};
    self()->for_each_(
      [&](raw_pos_t pos) { n += pred(read_pos{pos}); }
    );
    return n;
  }

  /**
   * @return Iterator pointing to end of the data structure.
   */
  constexpr const_iterator end() const noexcept { return {self(), capacity()}; }
  /**
   * @return Iterator pointing to end of the data structure.
   */
  constexpr iterator end() noexcept { return {self(), capacity()}; }

  /**
   * Erase a entry from the data structure.
   *
   * @param key The key to check.
   * @return True if the key was in the container (i.e., the erasure was successful).
   */
  constexpr bool erase(const key_type& key) {
    return std::get<0>(self()->erase_first_(key, key_to_hash(key)));
  }
  /**
   * Erase a entry from the data structure.
   *
   * @param key The key to check.
   * @param hint Hint from preceeding call to `write_prefetch`.
   * @return True if the key was in the container (i.e., the erasure was successful).
   */
  constexpr bool erase(const key_type& key, const prefetch_hint& hint) {
    return std::get<0>(self()->erase_first_(key, hint_to_hash_(key, hint)));
  }
  /**
   * Erase the first entry with a matching key in the container.
   *
   * @param key The key to erase.
   * @return The position of the first key and the associated probe sequence.
   */
  constexpr std::tuple<bool, write_pos, probe_seq_type> erase_first(const key_type& key) {
    auto [b, p, s]{self()->erase_first_(key, key_to_hash(key))};
    return {b, write_pos{p}, s};
  }
  /**
   * Erase the first entry with a matching key in the container.
   *
   * @param key The key to erase.
   * @param hint Hint from preceeding call to `write_prefetch`.
   * @return The position of the first key and the associated probe sequence.
   */
  constexpr std::tuple<bool, write_pos, probe_seq_type> erase_first(const key_type& key, const prefetch_hint& hint) {
    auto [b, p, s]{self()->erase_first_(key, hint_to_hash_(key, hint))};
    return {b, write_pos{p}, s};
  }
  /**
   * Erase the next entry (after pos) with a matching key in the container.
   *
   * @param pos The position to start the search from.
   * @param seq The probe sequence to use.
   * @param key The key to erase.
   * @return The position of the next key and the associated probe sequence.
   */
  template <typename PS>
  constexpr std::tuple<bool, write_pos, probe_seq_type> erase_next(write_pos&& pos, PS&& seq, const key_type& key) {
    auto [b, p, s]{self()->erase_next_(validate_range(std::move(pos)), std::forward<PS>(seq), key, key_to_hash(key))};
    return {b, write_pos{p}, s};
  }
  /**
   * Erase the next entry (after pos) with a matching key in the container.
   *
   * @param pos The position to start the search from.
   * @param seq The probe sequence to use.
   * @param key The key to erase.
   * @param hint Hint from preceeding call to `write_prefetch`.
   * @return The position of the next key and the associated probe sequence.
   */
  template <typename PS>
  constexpr std::tuple<bool, write_pos, probe_seq_type> erase_next(write_pos&& pos, PS&& seq, const key_type& key, const prefetch_hint& hint) {
    auto [b, p, s]{self()->erase_next_(validate_range(std::move(pos)), std::forward<PS>(seq), key, hint_to_hash_(key, hint))};
    return {b, write_pos{p}, s};
  }
  /**
   * Prune the entry at the provided position from the data structure.
   *
   * @param pos The position to investigate.
   * @return The position if it is still valid, otherwise `npos`.
   */
  constexpr std::pair<bool, write_pos> erase_at(write_pos&& pos) {
    auto [b, p]{self()->erase_at_(validate_range(std::move(pos)))};
    return {b, write_pos{p}};
  }
  /**
   * Scan through the data structure and erase all matching entries.
   *
   * @param pred Predicate function to determine whether an entry matches.
   */
  template <typename Pred>
  constexpr int_t erase_if(const Pred& pred) {
    static_assert(std::is_invocable_r_v<bool, Pred, write_pos>, "`pred` must be pred(write_pos) -> bool");
    return self()->erase_if_([&](raw_pos_t pos) { return pred(write_pos{pos}); });
  }
  /**
   * Erase all entries with a matching key in the container.
   *
   * @param key The key to erase.
   * @return The number of keys that were actually erased.
   */
  constexpr int_t erase_all(const key_type& key) {
    return self()->erase_all_(key, key_to_hash(key));
  }

  /**
   * Locate an entry in the container.
   *
   * @param key The key to check.
   * @return The position of the entry if it exists, otherwise `npos`.
   */
  constexpr read_pos find(const key_type& key) const noexcept {
    return read_pos{self()->find_first_(key, key_to_hash(key)).first};
  }
  /**
   * Locate an entry in the container.
   *
   * @param key The key to check.
   * @param hint Hint from preceeding call to `read_prefetch`.
   * @return The position of the entry if it exists, otherwise `npos`.
   */
  constexpr read_pos find(const key_type& key, const prefetch_hint& hint) const {
    return read_pos{self()->find_first_(key, hint_to_hash_(key, hint)).first};
  }
  /**
   * Locate the first entry with a matching key in the container.
   *
   * @param key The key to find.
   * @return The position of the first key and the associated probe sequence.
   */
  constexpr std::pair<read_pos, probe_seq_type> find_first(const key_type& key) const noexcept {
    auto [p, s]{self()->find_first_(key, key_to_hash(key))};
    return {read_pos{p}, s};
  }
  /**
   * Locate the first entry with a matching key in the container.
   *
   * @param key The key to find.
   * @param hint Hint from preceeding call to `read_prefetch`.
   * @return The position of the first key and the associated probe sequence.
   */
  constexpr std::pair<read_pos, probe_seq_type> find_first(const key_type& key, const prefetch_hint& hint) const {
    auto [p, s]{self()->find_first_(key, hint_to_hash_(key, hint))};
    return {read_pos{p}, s};
  }
  /**
   * Locate the next key (after pos) in the container.
   *
   * @param pos The position to start the search from.
   * @param seq The probe sequence to use.
   * @param key The key to find.
   * @return Result of `on_find`.
   */
  template <typename Pos, typename PS>
  constexpr std::pair<read_pos, probe_seq_type> find_next(Pos&& pos, PS&& seq, const key_type& key) const {
    auto [p, s]{self()->find_next_(validate_range(std::forward<Pos>(pos)), std::forward<PS>(seq), key, key_to_hash(key))};
    return {read_pos{p}, s};
  }
  /**
   * Locate the next key (after pos) in the container.
   *
   * @param pos The position to start the search from.
   * @param seq The probe sequence to use.
   * @param key The key to find.
   * @param hint Hint from preceeding call to `read_prefetch`.
   * @return Result of `on_find`.
   */
  template <typename Pos, typename PS>
  constexpr std::pair<read_pos, probe_seq_type> find_next(Pos&& pos, PS&& seq, const key_type& key, const prefetch_hint& hint) const {
    auto [p, s]{self()->find_next_(validate_range(std::forward<Pos>(pos)), std::forward<PS>(seq), key, hint_to_hash_(key, hint))};
    return {read_pos{p}, s};
  }
  /**
   * Scan through the data structure and return the position of the matching entry.
   *
   * @param func Predicate function to determine whether an entry matches.
   * @return Either the position of the entry, or `npos`.
   */
  template <typename Pred>
  constexpr read_pos find_if(const Pred& pred) const {
    static_assert(std::is_invocable_r_v<bool, Pred, read_pos>, "`pred` must be pred(read_pos) -> bool");
    raw_pos_t p{self()->find_if_([&](raw_pos_t pos) { return pred(read_pos{pos}); })};
    return read_pos{p};
  }
  /**
   * Get `read_pos`s of all entries with a matching key in the container.
   *
   * @param key The key to erase.
   * @return All found positions.
   */
  inline std::vector<read_pos> find_all(const key_type& key) const {
    std::vector<read_pos> res;
    self()->for_each_(key, key_to_hash(key),
      [&](raw_pos_t pos, const probe_seq_type&) { res.emplace_back(pos); }
    );
    return res;
  }

  /**
   * Iterate over the data structure and call a function for each filled slot.
   *
   * @param func The function to call.
   */
  template <typename Func>
  constexpr void for_each(const Func& func) const {
    static_assert(std::is_invocable_v<Func, read_pos>, "`func` must be `func(read_pos)`!");
    self()->for_each_([&](raw_pos_t pos) { func(read_pos{pos}); });
  }
  /**
   * Iterate over the data structure and call a function for each filled slot.
   *
   * @param func The function to call.
   */
  template <typename Func>
  constexpr void for_each(const Func& func) {
    if constexpr (std::is_invocable_v<Func, read_pos>) {
      self()->for_each_([&](raw_pos_t pos) { func(read_pos{pos}); });
    } else if constexpr (std::is_invocable_v<Func, write_pos>) {
      self()->for_each_([&](raw_pos_t pos) { func(write_pos{pos}); });
    } else {
      static_assert(dependent_false_v<Func>, "`func` must be `func(read_pos)` or `func(write_pos)`!");
    }
  }
  /**
   * Iterate over the data structure and call a function for each blob.
   *
   * @param func The function to call.
   */
  template <typename Func>
  constexpr void for_each_blob(const Func& func) const {
    static_assert(std::is_invocable_v<Func, const blob_t*>, "`func` must be a valid `func(const blob_t*)`!");
    static_assert(has_blobs, "`for_each_blob` is not supported for this configuration!");

    const int_t blob_stride{conf_.blob_stride()};
    const blob_t* const __restrict blobs{blobs_.get()};

    self()->for_each_(
      [&](raw_pos_t pos) { func(&blobs[pos * blob_stride]); }
    );
  }
  /**
   * Iterate over the data structure and call a function for each blob.
   *
   * @param func The function to call.
   */
  template <typename Func>
  constexpr void for_each_blob(const Func& func) {
    static_assert(std::is_invocable_v<Func, const blob_t*> || std::is_invocable_v<Func, blob_t*>, "`func` must be `func(const blob_t*)` or `func(blob_t*)`!");
    static_assert(has_blobs, "`for_each_blob` is not supported for this configuration!");

    const int_t blob_stride{conf_.blob_stride()};
    blob_t* const __restrict blobs{blobs_.get()};

    self()->for_each_(
      [&](raw_pos_t pos) { func(&blobs[pos * blob_stride]); }
    );
  }
  /**
   * Iterate over the data structure and call a function for each LRU.
   *
   * @param func The function to call.
   */
  template <typename Func>
  constexpr void for_each_lru(const Func& func) const {
    static_assert(std::is_invocable_v<Func, lru_t>, "`func` must be `func(lru_t)`!");
    self()->for_each_lru_(func);
  }
  /**
   * Iterate over the data structure and call a function for each state.
   *
   * @param func The function to call.
   */
  template <typename Func>
  constexpr void for_each_state(const Func& func) const {
    static_assert(std::is_invocable_v<Func, state_t>, "`func` must be `func(state_t)`!");
    self()->for_each_state_(func);
  }
  /**
   * Iterate over the data structure and call a function for each value.
   *
   * @param func The function to call.
   */
  template <typename Func>
  constexpr void for_each_value(const Func& func) const {
    static_assert(has_values, "`for_each_value` is not supported for this configuration!");
    static_assert(std::is_invocable_v<Func, value_type>, "`func` must be `func(value_type)`!");
    const value_type* const __restrict values{values_.get()};

    self()->for_each_(
      [&](raw_pos_t pos) { func(values[pos]); }
    );
  }
  /**
   * Iterate over the data structure and call a function for each value.
   *
   * @param func The function to call.
   */
  template <typename Func>
  constexpr void for_each_value(const Func& func) {
    static_assert(has_values, "`for_each_value` is not supported for this configuration!");
    static_assert(std::is_invocable_v<Func, value_type&>, "`func` must be `func(value_type&)`!");
    value_type* const __restrict values{values_.get()};

    self()->for_each_([&](raw_pos_t pos) { func(values[pos]); });
  }

  /**
   * Iteratore over all entries that match the supplied key.
   *
   * @param key The key to query.
   * @param func Handler that is called for each matching entry.
   */
  template <typename Func>
  constexpr void for_each(const key_type& key, const Func& func) const {
    static_assert(std::is_invocable_v<Func, read_pos, probe_seq_type>, "`func` must be `func(read_pos, probe_seq_type)`!");

    self()->for_each_(key, key_to_hash(key),
      [&](raw_pos_t pos, const probe_seq_type& seq) { func(read_pos{pos}, seq); }
    );
  }
  /**
   * Iteratore over all entries that match the supplied key.
   *
   * @param key The key to query.
   * @param func Handler that is called for each matching entry.
   */
  template <typename Func>
  constexpr void for_each(const key_type& key, const Func& func) {
    if constexpr (std::is_invocable_v<Func, read_pos, probe_seq_type>) {
      self()->for_each_(key, key_to_hash(key),
        [&](raw_pos_t pos, const probe_seq_type& seq) { func(read_pos{pos}, seq); }
      );
    } else if constexpr (std::is_invocable_v<Func, write_pos, probe_seq_type>) {
      self()->for_each_(key, key_to_hash(key),
        [&](raw_pos_t pos, const probe_seq_type& seq) { func(write_pos{pos}, seq); }
      );
    } else {
      static_assert(dependent_false_v<Func>, "`func` must be `func(read_pos, probe_seq_type)` or `func(write_pos, probe_seq_type)`!");
    }
  }

  /**
   * Retrieve the blob at `pos` and copy it to `dst`.
   *
   * @param pos The position to query.
   * @param dst The destination buffer.
   */
  constexpr void get_blob_at(const pos& pos, void* dst) const {
    copy_blob(dst, blob_at(pos), conf_.blob_size());
  }
  /**
   * Retrieve the blob at `pos` and copy it to `dst`.
   *
   * @param pos The position to query.
   * @param dst The destination buffer.
   * @param n The number of bytes to copy.
   */
  constexpr void get_blob_at(const pos& pos, void* dst, int_t n) const {
    const int_t blob_size{conf_.blob_size()};
    NVHM_ASSUME_(n <= blob_size, "n = ", n, ", blob_size = ", blob_size);
    copy_blob(dst, self()->blob_at(pos), n);
  }

  /**
   * Doubles the storage capacity of the data structure.
   * @return New capacity.
   */
  constexpr int_t grow() { return self()->resize_("Grow API call", capacity() << 1); }

  /**
   * Inserts a key into the container if it doesn't exist yet.
   *
   * @param key The key to find/insert.
   * @param on_insert Handler that is called after insertion has concluded.
   * @return Insertion position, or `npos` if the insertion failed.
   */
  template <typename K>
  constexpr write_pos insert(K&& key) {
    return write_pos{std::get<0>(self()->insert_(std::forward<K>(key), key_to_hash(key)))};
  }
  /**
   * Inserts a key into the container if it doesn't exist yet.
   *
   * @param key The key to find/insert.
   * @param hint Prefetch hint returned by previous call `prefetch_*`.
   * @return Insertion position, or `npos` if the insertion failed.
   */
  template <typename K>
  constexpr write_pos insert(K&& key, const prefetch_hint& hint) {
    return write_pos{std::get<0>(self()->insert_(std::forward<K>(key), hint_to_hash_(key, hint)))};
  }
  /**
   * Inserts a key into the container if it doesn't exist yet.
   *
   * @param key The key to find/insert.
   * @return The insertion position, the associated probe sequence, and the insert operation conducted.
   */
  template <typename K>
  constexpr std::tuple<write_pos, probe_seq_type, insert_op_t> insert_ex(K&& key) {
    auto [p, s, op]{self()->insert_(std::forward<K>(key), key_to_hash(key))};
    return {write_pos{p}, std::move(s), op};
  }
  /**
   * Inserts a key into the container if it doesn't exist yet.
   *
   * @param key The key to find/insert.
   * @param hint Prefetch hint returned by previous call `prefetch_*`.
   * @return The insertion position, the associated probe sequence, and the insert operation conducted.
   */
  template <typename K>
  constexpr std::tuple<write_pos, probe_seq_type, insert_op_t> insert_ex(K&& key, const prefetch_hint& hint) {
    auto [p, s, op]{self()->insert_(std::forward<K>(key), hint_to_hash_(key, hint))};
    return {write_pos{p}, std::move(s), op};
  }

  /**
   * @return True, if the data structure contains no data.
   */
  constexpr bool is_empty() const noexcept { return capacity() == self()->num_not_hash_slots(); }
  /**
   * By definition, a dynamic map cannot never be full. It will just grow. Hence,
   * this for such maps, this returns `true`, if the very next `insert` operation
   * would cause the map to grow.
   *
   * @return True, if the data structure has no free space left.
   */
  constexpr bool is_full() const noexcept { return self()->is_full_(); }

  /**
   * Fetch the key at `pos`.
   *
   * @param pos The position to query.
   * @return The associated key.
   */
  constexpr const key_type& key_at(const pos& pos) const noexcept {
    return self()->key_at_(pos.inner());
  }

  /**
   * @return The load factor of the data structure.
   */
  constexpr double load_factor() const noexcept {
    return static_cast<double>(self()->size()) / static_cast<double>(capacity());
  }

  /**
   * Fetch the LRU value at `pos`.
   *
   * @param pos The position to query.
   * @return The associated LRU value.
   */
  constexpr lru_t lru_at(const pos& pos) const noexcept {
    return self()->lru_at_(pos.inner());
  }

  /**
   * Fetch the mapped value at the provided position.
   *
   * @param pos The position to query.
   * @return The associated entry.
   */
  constexpr const_mapped_type mapped_at(const pos& pos) const noexcept {
    return mapped_at_(pos.inner());
  }
  /**
   * Fetch the mapped value at the provided position.
   *
   * @param pos The position to query.
   * @return The associated entry.
   */
  constexpr mapped_type mapped_at(const write_pos& pos) noexcept {
    return mapped_at_(pos.inner());
  }

  /**
   * @return The maximum permissible load factor of the data structure.
   */
  constexpr double max_load_factor() const noexcept { return self()->max_load_factor_(); }

  /**
   * Return a dummy `read_pos` that is not associated with any position in the data structure.
   *
   * @return A dummy `read_pos`.
   */
  constexpr static read_pos read_npos() noexcept { return read_pos{npos}; }

  /**
   * Decode the key and prefetch memory to facilitate a future `read`-operation.
   *
   * @param key The key for which to prefetch prefetch.
   * @return Prefetch token.
   */
  constexpr prefetch_hint read_prefetch(const key_type& key) const noexcept {
    prefetch_hint hint{key};
    self()->read_prefetch_(hint.hash());
    return hint;
  }
  /**
   * Uses existing prefetch hint to prefetch memory to facilitate a future `read`-operation.
   *
   * @param hint The prefetch hint.
   */
  constexpr void read_prefetch(const key_type& key, const prefetch_hint& hint) const {
    self()->read_prefetch_(hint_to_hash_(key, hint));
  }
  /**
   * Prefetch the value at `pos` for read operations.
   *
   * @param pos The position to query.
   */
  constexpr void read_prefetch_value_at(const pos& pos) const noexcept {
    nvhm::read_prefetch<1>(&value_at_(pos.inner()));
  }
  /**
   * Prefetch the value at `pos` for read operations.
   *
   * @param pos The position to query.
   */
  constexpr void read_prefetch_blob_at(const pos& pos) const noexcept {
    nvhm::read_prefetch(blob_at_(pos.inner()), conf_.blob_size());
  }

  /**
   * Render the data structure to an output stream.
   *
   * @param os The output stream.
   * @param with_values Whether to render values.
   * @param with_blobs Whether and how to render blobs.
   * @return The output stream.
   */
  constexpr void render(std::ostream& os, bool with_values = has_values, blob_render_t with_blobs = has_blobs ? blob_render_t::size : blob_render_t::hide) const {
    if (!with_values && with_blobs == blob_render_t::hide) {
      const char* sep{""};
      self()->for_each_key([&](const key_type& key) {
        os << sep << key;
        sep = ", ";
      });
    } else {
      const int_t blob_size{conf_.blob_size()};

      const char* value_beg{""};
      const char* value_sep{""};
      const char* value_end{""};
      if (with_blobs != blob_render_t::hide) {
        value_sep = " | ";
        if (with_values) {
          value_beg = "{";
          value_end = "}";
        }
      }
      
      const char* sep{""};
      self()->for_each_([&](raw_pos_t pos) {
        os << sep << self()->key_at_(pos) << " -> ";

        os << value_beg;
        if constexpr (has_values) {
          if constexpr (is_ostreamable_v<value_type>) {
            os << value_at_(pos) << value_sep;
          } else {
            throw std::runtime_error("`value_type` is not compatible with ostream!");
          }
        } else if (with_values) {
          throw std::runtime_error("Requested printing values. But no values present!");
        }

        if constexpr (has_blobs) {
          switch (with_blobs) {
            case blob_render_t::hide: {
            } break;
            case blob_render_t::size: {
              os << blob_size << " bytes";
            } break;
            case blob_render_t::full: {
              render_n(blob_at_(pos), blob_size, os);
            } break;
          }
        } else if (with_blobs == blob_render_t::full) {
          throw std::runtime_error("Requested printing blobs. But no blobs present!");
        }
        os << value_end;
        sep = ", ";
      });
    }
  }

  /**
   * Ensures the data structure has at least the provided capacity.
   * @param new_capacity Desired new capacity.
   * @return Actual new capacity.
   */
  constexpr int_t reserve(int_t new_capacity) {
    const int_t end{capacity()};
    if (new_capacity <= end) return end;

    new_capacity = std::max(bit_ceil(new_capacity), conf_.min_capacity());
    new_capacity = self()->resize_("Reserve API call", new_capacity);
    return new_capacity;
  }

  /**
   * Attempts to resize the data structure.
   *
   * @param new_capacity Desired new capacity.
   * @return Actual new capacity.
   */
  constexpr int_t resize(int_t new_capacity) {
    new_capacity = std::max(bit_ceil(new_capacity), conf_.min_capacity());
    return self()->resize_("Resize API call", new_capacity);
  }

  /**
   * Scrub the data structure of entries with an LRU less than the provided threshold.
   *
   * @param threshold The threshold LRU value.
   */
  constexpr int_t scrub(lru_t threshold = max_lru) {
    return self()->scrub_("Forced Scrub", threshold);
  }

  /**
   * Overwrite the blob at `pos` with the contents of `src`.
   *
   * @param pos The position to query.
   * @param src The source buffer.
   */
  constexpr void set_blob_at(const write_pos& pos, const void* src) {
    copy_blob(blob_at(pos), src, conf_.blob_size());
  }
  /**
   * Overwrite the blob at `pos` with the contents of `src`.
   *
   * @param pos The position to query.
   * @param src The source buffer.
   * @param n The number of bytes to copy.
   */
  constexpr void set_blob_at(const write_pos& pos, const void* src, int_t n) {
    NVHM_ASSUME_(n <= conf_.blob_size(), "n = ", n, ", blob_size = ", conf_.blob_size());
    copy_blob(blob_at(pos), src, n);
  }

  /**
   * Assigns the value at `pos` to the provided value.
   */
  template<typename V>
  constexpr void set_value_at(const write_pos& pos, V&& value) {
    value_at(pos) = std::forward<V>(value);
  }

  /**
   * Attempt to shrink the data structure.
   * @return New capacity.
   */
  constexpr int_t shrink() {
    const int_t old_end{capacity()};
    int_t new_end{self()->shrink_target_()};
    if (new_end != old_end) {
      new_end = self()->resize_("Try Shrink", new_end);
    }
    return new_end;
  }

  /**
   * Transforms a position into a readonly iterator.
   *
   * @param pos The position to create an iterator from.
   * @return The iterator.
   */
  constexpr const_iterator to_iterator(pos&& pos) const {
    if (pos == npos) {
      return end();
    } else {
      return {self(), validate_range(std::move(pos))};
    }
  }
  /**
   * Transforms a position into a readonly iterator.
   *
   * @param pos The position to create an iterator from.
   * @return The iterator.
   */
  constexpr iterator to_iterator(write_pos&& pos) {
    if (pos == npos) {
      return end();
    } else {
      return {self(), validate_range(std::move(pos))};
    }
  }
  /**
   * Transforms a position into a readonly iterator.
   *
   * @param pos The position to create an iterator from.
   * @return The iterator.
   */
  constexpr const_iterator to_citerator(pos&& pos) const { return to_iterator(std::move(pos)); }

  /**
   * Locate an entry in the container.
   *
   * @param key The key to check.
   * @return Result of `on_update`.
   */
  constexpr write_pos update(const key_type& key) {
    return write_pos{self()->find_first_(key, key_to_hash(key)).first};
  }
  /**
   * Locate an entry in the container.
   *
   * @param key The key to check.
   * @param hint Hint from preceeding call to `write_prefetch`.
   * @return The position of the entry if it exists, otherwise `npos`.
   */
  constexpr write_pos update(const key_type& key, const prefetch_hint& hint) {
    return write_pos{self()->find_first_(key, hint_to_hash_(key, hint)).first};
  }
  /**
   * Locate the first entry with a matching key in the container.
   *
   * @param key The key to find.
   * @return The position of the first key and the associated probe sequence.
   */
  constexpr std::pair<write_pos, probe_seq_type> update_first(const key_type& key) noexcept {
    auto [p, s]{self()->find_first_(key, key_to_hash(key))};
    return {write_pos{p}, std::move(s)};
  }
  /**
   * Locate the first entry with a matching key in the container.
   *
   * @param key The key to find.
   * @param hint Hint from preceeding call to `read_prefetch`.
   * @return The position of the first key and the associated probe sequence.
   */
  constexpr std::pair<write_pos, probe_seq_type> update_first(const key_type& key, const prefetch_hint& hint) {
    auto [p, s]{self()->find_first_(key, hint_to_hash_(key, hint))};
    return {write_pos{p}, std::move(s)};
  }
  /**
   * Locate the next key (after pos) in the container.
   *
   * @param pos The position to start the search from.
   * @param seq The probe sequence to use.
   * @param key The key to find.
   * @return The position of the next key and the associated probe sequence.
   */
  template <typename PS>
  constexpr std::pair<write_pos, probe_seq_type> update_next(write_pos&& pos, PS&& seq, const key_type& key) {
    auto [p, s]{self()->find_next_(validate_range(std::move(pos)), std::forward<PS>(seq), key, key_to_hash(key))};
    return {write_pos{p}, std::move(s)};
  }
  /**
   * Locate the next key (after pos) in the container.
   *
   * @param pos The position to start the search from.
   * @param seq The probe sequence to use.
   * @param key The key to find.
   * @param hint Hint from preceeding call to `read_prefetch`.
   * @return The position of the next key and the associated probe sequence.
   */
  template <typename PS>
  constexpr std::pair<write_pos, probe_seq_type> update_next(write_pos&& pos, PS&& seq, const key_type& key, const prefetch_hint& hint) {
    auto [p, s]{self()->find_next_(validate_range(std::move(pos)), std::forward<PS>(seq), key, hint_to_hash_(key, hint))};
    return {write_pos{p}, std::move(s)};
  }
  /**
   * Scan through the data structure and return the position of the matching entry.
   *
   * @return Either the position of the entry, or `npos`.
   */
  template <typename Pred>
  constexpr write_pos update_if(const Pred& pred) {
    static_assert(std::is_invocable_r_v<bool, Pred, write_pos>, "`pred` must be pred(write_pos) -> bool");
    raw_pos_t p{self()->find_if_([&](raw_pos_t pos) { return pred(write_pos{pos}); })};
    return write_pos{p};
  }
  /**
   * Get `write_pos`s of all entries with a matching key in the container.
   *
   * @param key The key to erase.
   * @return All found positions.
   */
  inline std::vector<write_pos> update_all(const key_type& key) {
    std::vector<write_pos> res;
    self()->for_each_(key, key_to_hash(key),
      [&](raw_pos_t pos, const probe_seq_type&) { res.emplace_back(pos); }
    );
    return res;
  }

  constexpr raw_pos_t validate(const pos& pos) const {
    NVHM_ASSUME_(contains_at(pos), "pos = ", pos);
    return pos.inner();
  }
  constexpr raw_pos_t validate(pos&& pos) const {
    NVHM_ASSUME_(contains_at(pos), "pos = ", pos);
    return pos.inner();
  }

  constexpr raw_pos_t validate_range(const pos& pos) const {
    raw_pos_t inner{pos.inner()};
    NVHM_ASSUME_(inner >= 0 && inner < capacity(), "pos = ", inner, ", capacity = ", capacity());
    return inner;
  }
  constexpr raw_pos_t validate_range(pos&& pos) const {
    raw_pos_t inner{pos.inner()};
    NVHM_ASSUME_(inner >= 0 && inner < capacity(), "pos = ", inner, ", capacity = ", capacity());
    return inner;
  }

  /**
   * Fetch the value at `pos`.
   *
   * @param pos The position to query.
   * @return Associated value.
   */
  constexpr const value_type& value_at(const pos& pos) const {
    return value_at_(pos.inner());
  }
  /**
   * Fetch the value at `pos`.
   *
   * @param pos The position to query.
   * @return Associated value.
   */
  constexpr value_type& value_at(const write_pos& pos) {
    return value_at_(pos.inner());
  }

  /**
   * Return a dummy `write_pos` that is not associated with any position in the data structure.
   *
   * @return A dummy `write_pos`.
   */
  constexpr static write_pos write_npos() noexcept { return write_pos{npos}; }

  /**
   * Decode the key and prefetch memory to facilitate a future update` or `upsert`-operation.
   *
   * @param key The key for which to prefetch prefetch.
   * @return Prefetch token. Intended to be used in conjunction with `insert_ex`.
   */
  constexpr prefetch_hint write_prefetch(const key_type& key) noexcept {
    prefetch_hint hint{key};
    self()->write_prefetch_(hint.hash());
    return hint;
  }
  /**
   * Uses existing prefetch hint to prefetch memory to facilitate a future `write`-operation.
   *
   * @param hint The prefetch hint.
   * @param num_keys The number of keys to prefetch.
   */
  constexpr void write_prefetch(const key_type& key, const prefetch_hint& hint) {
    self()->write_prefetch_(hint_to_hash_(key, hint));
  }
  /**
   * Prefetch the value at `pos` for write operations.
   *
   * @param pos The position to query.
   */
  constexpr void write_prefetch_value_at(const write_pos& pos) noexcept {
    nvhm::write_prefetch<1>(&value_at_(pos.inner()));
  }
  /**
   * Prefetch the value at `pos` for write operations.
   *
   * @param pos The position to query.
   */
  constexpr void write_prefetch_blob_at(const write_pos& pos) noexcept {
    nvhm::write_prefetch(blob_at_(pos.inner()), conf_.blob_size());
  }

  /**
   * Retrieve or insert key into the data structure, and returns the associated value.
   *
   * @param key The key to retrieve or insert.
   * @return The mapped value.
   */
  template <typename K>
  constexpr mapped_type operator[](K&& key) {
    raw_pos_t pos{std::get<0>(self()->insert_(std::forward<K>(key), key_to_hash(key)))};
    if (pos == npos) {
      throw std::runtime_error("Failed to insert key into the data structure!");
    }
    return mapped_at_(pos);
  }

  constexpr friend void swap(raw_map_base& lhs, raw_map_base& rhs) noexcept {
    if (&lhs == &rhs) return;

    std::swap(lhs.conf_, rhs.conf_);
    std::swap(lhs.bucket_mask_, rhs.bucket_mask_);
    std::swap(lhs.values_, rhs.values_);
    std::swap(lhs.blobs_, rhs.blobs_);
  }

 protected:
  conf_type conf_;
  bitmask_t bucket_mask_;
  unique_ptr_t<value_type[], allocator_type> values_;
  unique_ptr_t<std::byte[], allocator_type> blobs_;

  constexpr raw_pos_t alloc_(raw_pos_t end) {
    NVHM_ASSERT_(has_single_bit(end) && end >= conf_.min_capacity(), "new_capacity = ", end, ", min_capacity = ", conf_.min_capacity());

    bucket_mask_ = make_aligned_mask<kernel_size>(end);
    if constexpr (has_values) {
      values_ = make_unique<value_type, allocator_type>(end);
    }
    if constexpr (has_blobs) {
      blobs_ = make_unique<std::byte, allocator_type>(end * conf_.blob_stride());
    }
    return end;
  }

  constexpr const blob_t* blob_at_(const raw_pos_t pos) const noexcept {
    static_assert(has_blobs, "`blob_at_` is not supported for this configuration!");
    NVHM_ASSERT_(self()->contains_at_(pos), "pos = ", pos);
    
    return &blobs_[to_uint(pos * conf_.blob_stride())];
  }
  constexpr blob_t* blob_at_(const raw_pos_t pos) noexcept {
    static_assert(has_blobs, "`blob_at_` is not supported for this configuration!");
    NVHM_ASSERT_(self()->contains_at_(pos), "pos = ", pos);

    return &blobs_[to_uint(pos * conf_.blob_stride())];
  }

  constexpr raw_pos_t front_() const noexcept { return self()->find_if_([](raw_pos_t) { return true; }); }

  constexpr hash_t hint_to_hash_(const key_type& key, const prefetch_hint& hint) const {
    NVHM_ASSUME_(key_to_hash(key) == hint.hash(), "key = ", key, ", hash = ", key_to_hash(key), ", hint = ", hint.hash());
    return hint.hash();
  }

  constexpr const_mapped_type mapped_at_(const raw_pos_t pos) const noexcept {
    NVHM_ASSERT_(self()->contains_at_(pos), "pos = ", pos);

    const int_t blob_stride{conf_.blob_stride()};
    value_type* const __restrict values{values_.get()};
    blob_t* const __restrict blobs{blobs_.get()};

    if constexpr (has_values && has_blobs) {
      return {values[pos], &blobs[pos * blob_stride]};
    } else if constexpr (has_values) {
      return values[pos];
    } else if constexpr (has_blobs) {
      return &blobs[pos * blob_stride];
    }
  }
  constexpr mapped_type mapped_at_(const raw_pos_t pos) noexcept {
    NVHM_ASSERT_(self()->contains_at_(pos), "pos = ", pos);

    const int_t blob_stride{conf_.blob_stride()};
    value_type* const __restrict values{values_.get()};
    blob_t* const __restrict blobs{blobs_.get()};

    if constexpr (has_values && has_blobs) {
      return {values[pos], &blobs[pos * blob_stride]};
    } else if constexpr (has_values) {
      return values[pos];
    } else if constexpr (has_blobs) {
      return &blobs[pos * blob_stride];
    }
  }

  constexpr const value_type& value_at_(const raw_pos_t pos) const noexcept {
    static_assert(has_values, "`value_offset_at_` is not supported for this configuration!");
    NVHM_ASSERT_(self()->contains_at_(pos), "pos = ", pos);

    return values_[to_uint(pos)];
  }
  constexpr value_type& value_at_(const raw_pos_t pos) noexcept {
    static_assert(has_values, "`value_offset_at_` is not supported for this configuration!");
    NVHM_ASSERT_(self()->contains_at_(pos), "pos = ", pos);

    return values_[to_uint(pos)];
  }
};

template<typename T>
T* view_blob_as(std::byte* ptr) noexcept {
  return reinterpret_cast<T*>(ptr);
}
template<typename T>
const T* view_blob_as(const std::byte* ptr) noexcept {
  return reinterpret_cast<const T*>(ptr);
}

} // namespace nvhm