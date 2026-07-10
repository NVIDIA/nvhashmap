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

#include <mutex>
#include <shared_mutex>
#include "hash.hpp"
#include "container.hpp"

namespace nvhm {

using guarded_mutex_type = std::shared_mutex;

template <typename Inner, typename Lock>
class guarded_pos : public wrapped_pos<Inner> {
 public:
  using base_type = wrapped_pos<Inner>;
  using inner_type = typename base_type::inner_type;

  using mutex_type = guarded_mutex_type;
  using lock_type = Lock;

  template <typename RhsInner, typename RhsLock>
  constexpr friend bool operator==(const guarded_pos& lhs, const guarded_pos<RhsInner, RhsLock>& rhs) noexcept { return lhs.inner_ == rhs.inner_; }
  template <typename RhsInner, typename RhsLock>
  constexpr friend bool operator!=(const guarded_pos& lhs, const guarded_pos<RhsInner, RhsLock>& rhs) noexcept { return !(lhs == rhs); }
  template <typename RhsInner, typename RhsLock>
  constexpr friend bool operator<(const guarded_pos& lhs, const guarded_pos<RhsInner, RhsLock>& rhs) noexcept { return lhs.inner_ < rhs.inner_; }
  template <typename RhsInner, typename RhsLock>
  constexpr friend bool operator>(const guarded_pos& lhs, const guarded_pos<RhsInner, RhsLock>& rhs) noexcept { return lhs.inner_ > rhs.inner_; }
  template <typename RhsInner, typename RhsLock>
  constexpr friend bool operator<=(const guarded_pos& lhs, const guarded_pos<RhsInner, RhsLock>& rhs) noexcept { return !(lhs > rhs); }
  template <typename RhsInner, typename RhsLock>
  constexpr friend bool operator>=(const guarded_pos& lhs, const guarded_pos<RhsInner, RhsLock>& rhs) noexcept { return !(lhs < rhs); }

  constexpr friend void swap(guarded_pos& lhs, guarded_pos& rhs) noexcept {
    if (&lhs == &rhs) return;
    swap(static_cast<base_type&>(lhs), static_cast<base_type&>(rhs));

    std::swap(lhs.lock_, rhs.lock_);
  }

 protected:
  using base_type::inner_;
  lock_type lock_;
  
  guarded_pos() = delete;
  constexpr explicit guarded_pos(const guarded_pos&) noexcept = default;
  constexpr guarded_pos& operator=(const guarded_pos&) noexcept = default;
  constexpr explicit guarded_pos(guarded_pos&&) noexcept = default;
  constexpr guarded_pos& operator=(guarded_pos&&) noexcept = default;

  constexpr explicit guarded_pos(inner_type&& inner) noexcept
    : base_type{std::move(inner)} {}
  constexpr explicit guarded_pos(inner_type&& inner, lock_type&& lock) noexcept
    : base_type{std::move(inner)}, lock_{std::move(lock)} {}

  constexpr const mutex_type* mutex_() const noexcept { return lock_.mutex(); }

  template <typename>
  friend class guarded;
};

template <typename Inner>
class guarded_read_pos : public guarded_pos<Inner, std::shared_lock<guarded_mutex_type>> {
 public:
  using base_type = guarded_pos<Inner, std::shared_lock<guarded_mutex_type>>;
  using inner_type = typename base_type::inner_type;
  using lock_type = typename base_type::lock_type;

  guarded_read_pos() = delete;
  constexpr guarded_read_pos(const guarded_read_pos&) noexcept = delete;
  constexpr guarded_read_pos& operator=(const guarded_read_pos&) noexcept = delete;
  constexpr guarded_read_pos(guarded_read_pos&&) noexcept = default;
  constexpr guarded_read_pos& operator=(guarded_read_pos&&) noexcept = default;

  constexpr explicit guarded_read_pos(inner_type&& inner) noexcept
    : base_type{std::move(inner)} {}
  constexpr explicit guarded_read_pos(inner_type&& inner, lock_type&& lock) noexcept
    : base_type{std::move(inner), std::move(lock)} {}
};

template <typename Inner>
class guarded_write_pos : public guarded_pos<Inner, std::unique_lock<guarded_mutex_type>> {
 public:
  using base_type = guarded_pos<Inner, std::unique_lock<guarded_mutex_type>>;
  using inner_type = typename base_type::inner_type;
  using lock_type = typename base_type::lock_type;

  guarded_write_pos() = delete;
  constexpr guarded_write_pos(const guarded_write_pos&) = delete;
  constexpr guarded_write_pos& operator=(const guarded_write_pos&) = delete;
  constexpr guarded_write_pos(guarded_write_pos&&) noexcept = default;
  constexpr guarded_write_pos& operator=(guarded_write_pos&&) noexcept = default;

  constexpr explicit guarded_write_pos(inner_type&& inner) noexcept
    : base_type{std::move(inner)} {}
  constexpr explicit guarded_write_pos(inner_type&& inner, lock_type&& lock) noexcept
    : base_type{std::move(inner), std::move(lock)} {}
};

template <typename Inner>
class guarded : public container<guarded<Inner>> {
 public:
  static_assert(!std::is_reference_v<Inner>, "Cannot encapuslate reference types with guarded!");

  using base_type = container<guarded<Inner>>;
  using self_type = typename base_type::self_type;
  using inner_type = Inner;
  using conf_type = typename inner_type::conf_type;

  using key_type = typename inner_type::key_type;
  using value_type = typename inner_type::value_type;
  constexpr static bool has_values{inner_type::has_values};
  constexpr static flags_t flags{inner_type::flags};
  constexpr static bool has_blobs{inner_type::has_blobs};
  constexpr static int_t kernel_size{inner_type::kernel_size};

  using const_entry_type = typename inner_type::const_entry_type;
  using entry_type = typename inner_type::entry_type;
  using const_mapped_type = typename inner_type::const_mapped_type;
  using mapped_type = typename inner_type::mapped_type;

  using probe_seq_type = typename inner_type::probe_seq_type;
  using allocator_type = typename inner_type::allocator_type;

  using inner_read_pos_type = typename inner_type::read_pos;
  using inner_write_pos_type = typename inner_type::write_pos;

  template <typename InnerPos, typename Lock>
  using pos = guarded_pos<InnerPos, Lock>;
  using read_pos = guarded_read_pos<inner_read_pos_type>;
  using write_pos = guarded_write_pos<inner_write_pos_type>;

  using mutex_type = guarded_mutex_type;
  using read_lock_type = typename read_pos::lock_type;
  using write_lock_type = typename write_pos::lock_type;
  
  using inner_const_iterator_type = typename inner_type::const_iterator;
  using inner_iterator_type = typename inner_type::iterator;

  template <typename Self, typename InnerIterator, typename Lock>
  class iterator_base : public wrapped_map_iterator<Self, InnerIterator> {
   public:
    using base_type = wrapped_map_iterator<Self, InnerIterator>;
    using self_type = typename base_type::self_type;
    using inner_type = typename base_type::inner_type;
    using difference_type = typename base_type::difference_type;
    using lock_type = Lock;

    using base_type::self;

    constexpr bool is_left() const noexcept { return inner_.is_left(); }
    constexpr bool is_right() const noexcept { return inner_.is_right(); }

    constexpr self_type& operator++() { 
      NVHM_ASSUME_(lock_.owns_lock(), "Unsupported operation with this iterator!");
      ++inner_;
      return *self();
    }
    constexpr self_type& operator--() {
      NVHM_ASSUME_(lock_.owns_lock(), "Unsupported operation with this iterator!");
      --inner_;
      return *self();
    }

    constexpr self_type& operator+=(difference_type n) {
      NVHM_ASSUME_(lock_.owns_lock(), "Unsupported operation with this iterator!");
      inner_ += n; return *self();
    }
    
    template <typename RhsSelf, typename RhsInner, typename RhsLock>
    constexpr friend difference_type operator-(const iterator_base& lhs, const iterator_base<RhsSelf, RhsInner, RhsLock>& rhs) { return lhs.inner_ - rhs.inner_; }

    template <typename RhsSelf, typename RhsInner, typename RhsLock>
    constexpr friend bool operator==(const iterator_base& lhs, const iterator_base<RhsSelf, RhsInner, RhsLock>& rhs) noexcept { return lhs.inner_ == rhs.inner_; }
    template <typename RhsSelf, typename RhsInner, typename RhsLock>
    constexpr friend bool operator!=(const iterator_base& lhs, const iterator_base<RhsSelf, RhsInner, RhsLock>& rhs) noexcept { return !(lhs == rhs); }
    template <typename RhsSelf, typename RhsInner, typename RhsLock>
    constexpr friend bool operator<(const iterator_base& lhs, const iterator_base<RhsSelf, RhsInner, RhsLock>& rhs) noexcept { return lhs.inner_ < rhs.inner_; }
    template <typename RhsSelf, typename RhsInner, typename RhsLock>
    constexpr friend bool operator>(const iterator_base& lhs, const iterator_base<RhsSelf, RhsInner, RhsLock>& rhs) noexcept { return lhs.inner_ > rhs.inner_; }
    template <typename RhsSelf, typename RhsInner, typename RhsLock>
    constexpr friend bool operator<=(const iterator_base& lhs, const iterator_base<RhsSelf, RhsInner, RhsLock>& rhs) noexcept { return !(lhs > rhs); }
    template <typename RhsSelf, typename RhsInner, typename RhsLock>
    constexpr friend bool operator>=(const iterator_base& lhs, const iterator_base<RhsSelf, RhsInner, RhsLock>& rhs) noexcept { return !(lhs < rhs); }

    constexpr friend void swap(iterator_base& lhs, iterator_base& rhs) noexcept {
      if (&lhs == &rhs) return;
      swap(static_cast<base_type&>(lhs), static_cast<base_type&>(rhs));

      std::swap(lhs.lock_, rhs.lock_);
    }

   protected:
    using base_type::inner_;
    lock_type lock_;

    constexpr explicit iterator_base(inner_type&& pos) noexcept
      : base_type{std::move(pos)} {}
    constexpr explicit iterator_base(inner_type&& pos, lock_type&& lock) noexcept
      : base_type{std::move(pos)}, lock_{std::move(lock)} {}

    iterator_base() = delete;
    iterator_base(const iterator_base&) = delete;
    iterator_base& operator=(const iterator_base&) = delete;
    constexpr explicit iterator_base(iterator_base&&) noexcept = default;
    constexpr iterator_base& operator=(iterator_base&&) noexcept = default;

    template <typename, typename, typename>
    friend class iterator_base;
  };

  class const_iterator : public iterator_base<const_iterator, inner_const_iterator_type, read_lock_type> {
   public:
    using base_type = iterator_base<const_iterator, inner_const_iterator_type, read_lock_type>;
    using self_type = typename base_type::self_type;
    using inner_type = typename base_type::inner_type;
    using difference_type = typename base_type::difference_type;
    using lock_type = typename base_type::lock_type;

    const_iterator() = delete;
    const_iterator(const inner_type& pos) = delete;
    const_iterator& operator=(const const_iterator&) = delete;
    constexpr const_iterator(const_iterator&&) noexcept = default;
    constexpr const_iterator& operator=(const_iterator&&) noexcept = default;

    using base_type::self;

   protected:
    using base_type::inner_;
    using base_type::lock_;

    constexpr explicit const_iterator(inner_type&& pos) noexcept
      : base_type{std::move(pos)} {}
    constexpr explicit const_iterator(inner_type&& pos, lock_type&& lock) noexcept
      : base_type{std::move(pos), std::move(lock)} {}

    friend class guarded;
  };

  class iterator : public iterator_base<iterator, inner_iterator_type, write_lock_type> {
   public:
    using base_type = iterator_base<iterator, inner_iterator_type, write_lock_type>;
    using self_type = typename base_type::self_type;
    using inner_type = typename base_type::inner_type;
    using lock_type = typename base_type::lock_type;

    iterator() = delete;
    iterator(const inner_type& pos) = delete;
    iterator& operator=(const iterator&) = delete;
    constexpr iterator(iterator&&) noexcept = default;
    constexpr iterator& operator=(iterator&&) noexcept = default;

    constexpr bool erase() { return inner_.erase(); }

   protected:
    using base_type::inner_;

    constexpr explicit iterator(inner_type&& pos) noexcept
      : base_type{std::move(pos)} {}
    constexpr explicit iterator(inner_type&& pos, lock_type&& lock) noexcept
      : base_type{std::move(pos), std::move(lock)} {}

    friend class guarded;
  };

  using inner_prefetch_hint_type = typename inner_type::prefetch_hint;

  class prefetch_hint {
   public:
    constexpr prefetch_hint() noexcept {}
    constexpr prefetch_hint(inner_prefetch_hint_type&& inner) noexcept : inner_{std::move(inner)} {}

    constexpr const inner_prefetch_hint_type& inner() const noexcept { return inner_; }
    constexpr hash_t hash() const noexcept { return inner_.hash(); }

    constexpr friend bool operator==(const prefetch_hint& lhs, const prefetch_hint& rhs) noexcept { return lhs.inner_ == rhs.inner_; }
    constexpr friend bool operator!=(const prefetch_hint& lhs, const prefetch_hint& rhs) noexcept { return !(lhs == rhs); }

   protected:
    inner_prefetch_hint_type inner_;
  };

  constexpr guarded(const guarded& other)
    : inner_{[&]() {
      read_lock_type other_lock{other.lock_()};
      return other.inner_;
    }()} {}
  inline guarded& operator=(const guarded& other) {
    if (&other == this) return *self();

    write_lock_type lock{mutex_, std::defer_lock};
    read_lock_type other_lock{other.mutex_, std::defer_lock};
    std::lock(lock, other_lock);

    inner_ = other.inner_;
    return *self();
  }
  constexpr guarded(guarded&& other)
    : inner_{[&]() {
      write_lock_type other_lock{other.lock_()};
      return std::move(other.inner_);
    }()} {}
  inline guarded& operator=(guarded&& other) {
    if (&other == this) return *self();
    
    write_lock_type lock{mutex_, std::defer_lock};
    write_lock_type other_lock{other.mutex_, std::defer_lock};
    std::lock(lock, other_lock);

    inner_ = std::move(other.inner_);
    return *self();
  }

  constexpr guarded() {}
  template <typename Arg0, typename... Args, typename = std::enable_if_t<!std::is_same_v<remove_cvref_t<Arg0>, guarded>>>
  constexpr guarded(Arg0&& arg0, Args&&... args)
    : inner_{std::forward<Arg0>(arg0), std::forward<Args>(args)...} {}

  using base_type::self;
  using base_type::cself;
  NVHM_CONTAINER_INTERFACE_()

  inline std::vector<blob_t> all_blobs_for(const key_type& key) const {
    read_lock_type lock{lock_()};
    return inner_.all_blobs_for(key);
  }
  constexpr std::vector<value_type> all_values_for(const key_type& key) const {
    read_lock_type lock{lock_()};
    return inner_.all_values_for(key);
  }

  template<typename InnerPos, typename Lock>
  constexpr const_entry_type at(const pos<InnerPos, Lock>& pos) const {
    NVHM_ASSUME_(pos.mutex_() == &mutex_);
    return inner_.at(pos.inner_);
  }
  constexpr entry_type at(const write_pos& pos) {
    NVHM_ASSUME_(pos.mutex_() == &mutex_);
    return inner_.at(pos.inner_);
  }

  constexpr const_iterator begin() const {
    read_lock_type lock{lock_()};
    return const_iterator{inner_.begin(), std::move(lock)};
  }
  constexpr iterator begin() {
    write_lock_type lock{lock_()};
    return iterator{inner_.begin(), std::move(lock)};
  }

  template<typename InnerPos, typename Lock>
  constexpr const blob_t* blob_at(const pos<InnerPos, Lock>& pos) const {
    NVHM_ASSUME_(pos.mutex_() == &mutex_);
    return inner_.blob_at(pos.inner_);
  }
  constexpr blob_t* blob_at(const write_pos& pos) {
    NVHM_ASSUME_(pos.mutex_() == &mutex_);
    return inner_.blob_at(pos.inner_);
  }
  
  template<typename InnerPos, typename Lock>
  constexpr int_t bucket_size_at(const pos<InnerPos, Lock>& pos) const {
    NVHM_ASSUME_(pos.mutex_() == &mutex_);
    return inner_.bucket_size_at(pos.inner_);
  }
  template<typename InnerPos, typename Lock>
  constexpr int_t bucket_num_empty_slots_at(const pos<InnerPos, Lock>& pos) const {
    NVHM_ASSUME_(pos.mutex_() == &mutex_);
    return inner_.bucket_num_empty_slots_at(pos.inner_);
  }
  template<typename InnerPos, typename Lock>
  constexpr int_t bucket_num_tombstone_slots_at(const pos<InnerPos, Lock>& pos) const {
    NVHM_ASSUME_(pos.mutex_() == &mutex_);
    return inner_.bucket_num_tombstone_slots_at(pos.inner_);
  }

  constexpr int_t capacity() const {
    read_lock_type lock{lock_()};
    return inner_.capacity();
  }
  
  constexpr const_iterator cbegin() const { return begin(); }
  constexpr const_iterator cend() const { return end(); }

  constexpr bool check_integrity() const {
    read_lock_type lock{lock_()};
    return inner_.check_integrity();
  }

  constexpr int_t clear() {
    write_lock_type lock{lock_()};
    return inner_.clear();
  }
  constexpr int_t clear(int_t new_capacity) {
    write_lock_type lock{lock_()};
    return inner_.clear(new_capacity);
  }

  constexpr const conf_type& conf() const noexcept { return inner_.conf(); }

  constexpr bool contains(const key_type& key) const {
    read_lock_type lock{lock_()};
    return inner_.contains(key);
  }
  constexpr bool contains(const key_type& key, const prefetch_hint& hint) const {
    read_lock_type lock{lock_()};
    return inner_.contains(key, hint.inner());
  }
  template<typename InnerPos, typename Lock>
  constexpr bool contains_at(const pos<InnerPos, Lock>& pos) const {
    NVHM_ASSUME_(pos.mutex_() == &mutex_);
    return inner_.contains_at(pos.inner_);
  }

  constexpr int_t count(const key_type& key) const {
    read_lock_type lock{lock_()};
    return inner_.count(key);
  }
  template <typename Pred>
  constexpr int_t count_if(const Pred& pred) const {
    static_assert(std::is_invocable_r_v<bool, Pred, read_pos>, "`pred` must be pred(read_pos) -> bool");
    static_assert(arg_type_v<arg_n_t<Pred, 0>> != arg_type_t::lvalue_ref, "`pred` cannot be pred(read_pos&) -> bool");

    read_lock_type lock{lock_()};
    return inner_.count_if(
      [&](inner_read_pos_type&& pos) {
        read_pos r_lock{std::move(pos), std::move(lock)};
        bool res{pred(r_lock)};
        pos = std::move(r_lock.inner_);
        lock = std::move(r_lock.lock_);
        return res;
      }
    );
  }
  constexpr void count_state_collisions(std::array<int_t, kernel_size>& counts) const {
    read_lock_type lock{lock_()};
    inner_.count_state_collisions(counts);
  }

  constexpr const_iterator end() const {
    // We need to die either way. Either `begin()` or `end()` can hold the lock.
    // The more likely case is `begin()`, so we do the locking there and prevent
    // access using the `end()` iterator.
    return const_iterator{inner_.end()};
  }
  constexpr iterator end() {
    // We need to die either way. Either `begin()` or `end()` can hold the lock.
    // The more likely case is `begin()`, so we do the locking there and prevent
    // access using the `end()` iterator.
    return iterator{inner_.end()};
  }

  constexpr bool erase(const key_type& key) {
    write_lock_type lock{lock_()};
    return inner_.erase(key);
  }
  constexpr bool erase(const key_type& key, const prefetch_hint& hint) {
    write_lock_type lock{lock_()};
    return inner_.erase(key, hint.inner());
  }
  constexpr std::tuple<bool, write_pos, probe_seq_type> erase_first(const key_type& key) {
    write_lock_type lock{lock_()};
    auto [b, p, s]{inner_.erase_first(key)};
    return {b, write_pos{std::move(p), std::move(lock)}, std::move(s)};
  }
  constexpr std::tuple<bool, write_pos, probe_seq_type> erase_first(const key_type& key, const prefetch_hint& hint) {
    write_lock_type lock{lock_()};
    auto [b, p, s]{inner_.erase_first(key, hint.inner())};
    return {b, write_pos{std::move(p), std::move(lock)}, std::move(s)};
  }
  template <typename PS>
  constexpr std::tuple<bool, write_pos, probe_seq_type> erase_next(write_pos&& prev_pos, PS&& seq, const key_type& key) {
    NVHM_ASSUME_(prev_pos.mutex_() == &mutex_);
    auto [b, p, s]{inner_.erase_next(std::move(prev_pos.inner_), std::forward<PS>(seq), key)};
    return {b, write_pos{std::move(p), std::move(prev_pos.lock_)}, std::move(s)};
  }
  template <typename PS>
  constexpr std::tuple<bool, write_pos, probe_seq_type> erase_next(write_pos&& prev_pos, PS&& seq, const key_type& key, const prefetch_hint& hint) {
    NVHM_ASSUME_(prev_pos.mutex_() == &mutex_);
    auto [b, p, s]{inner_.erase_next(std::move(prev_pos.inner_), std::forward<PS>(seq), key, hint.inner())};
    return {b, write_pos{std::move(p), std::move(prev_pos.lock_)}, std::move(s)};
  }
  constexpr std::pair<bool, write_pos> erase_at(write_pos&& pos) {
    NVHM_ASSUME_(pos.mutex_() == &mutex_);
    auto [b, p]{inner_.erase_at(std::move(pos.inner_))};
    return {b, write_pos{std::move(p), std::move(pos.lock_)}};
  }
  template <typename Pred>
  constexpr int_t erase_if(const Pred& pred) {
    static_assert(std::is_invocable_r_v<bool, Pred, write_pos>, "`pred` must be pred(const write_pos&) -> bool");
    static_assert(arg_type_v<arg_n_t<Pred, 0>> == arg_type_t::const_lvalue_ref, "`pred` must be pred(const write_pos&) -> bool");

    write_lock_type lock{lock_()};
    return inner_.erase_if(
      [&](inner_write_pos_type&& pos) {
        write_pos w_lock{std::move(pos), std::move(lock)};
        bool res{pred(w_lock)};
        pos = std::move(w_lock.inner_);
        lock = std::move(w_lock.lock_);
        return res;
      }
    );
  }
  constexpr int_t erase_all(const key_type& key) {
    write_lock_type lock{lock_()};
    return inner_.erase_all(key);
  }

  constexpr read_pos find(const key_type& key) const {
    read_lock_type lock{lock_()};
    return read_pos{inner_.find(key), std::move(lock)};
  }
  constexpr read_pos find(const key_type& key, const prefetch_hint& hint) const {
    read_lock_type lock{lock_()};
    return read_pos{inner_.find(key, hint.inner()), std::move(lock)};
  }
  constexpr std::pair<read_pos, probe_seq_type> find_first(const key_type& key) const {
    read_lock_type lock{lock_()};

    auto [p, s]{inner_.find_first(key)};
    return {read_pos{std::move(p), std::move(lock)}, std::move(s)};
  }
  constexpr std::pair<read_pos, probe_seq_type> find_first(const key_type& key, const prefetch_hint& hint) const {
    read_lock_type lock{lock_()};

    auto [p, s]{inner_.find_first(key, hint.inner())};
    return {read_pos{std::move(p), std::move(lock)}, std::move(s)};
  }
  template <typename PS>
  constexpr std::pair<read_pos, probe_seq_type> find_next(read_pos&& pos, PS&& seq, const key_type& key) const {
    NVHM_ASSUME_(pos.mutex_() == &mutex_);

    auto [p, s]{inner_.find_next(pos.inner_, std::forward<PS>(seq), key)};
    return {read_pos{std::move(p), std::move(pos.lock_)}, std::move(s)};
  }
  template <typename PS>
  constexpr std::pair<read_pos, probe_seq_type> find_next(write_pos&& pos, PS&& seq, const key_type& key) const {
    NVHM_ASSUME_(pos.mutex_() == &mutex_);
    pos.lock_.unlock();  // Not ideal but the best we can do without downgradable locks.
    read_lock_type lock{lock_()};

    auto [p, s]{inner_.find_next(pos.inner_, std::forward<PS>(seq), key)};
    return {read_pos{std::move(p), std::move(lock)}, std::move(s)};
  }
  template <typename PS>
  constexpr std::pair<read_pos, probe_seq_type> find_next(read_pos&& pos, PS&& seq, const key_type& key, const prefetch_hint& hint) const {
    NVHM_ASSUME_(pos.mutex_() == &mutex_);

    auto [p, s]{inner_.find_next(pos.inner_, std::forward<PS>(seq), key, hint.inner())};
    return {read_pos{std::move(p), std::move(pos.lock_)}, std::move(s)};
  }
  template <typename PS>
  constexpr std::pair<read_pos, probe_seq_type> find_next(write_pos&& pos, PS&& seq, const key_type& key, const prefetch_hint& hint) const {
    NVHM_ASSUME_(pos.mutex_() == &mutex_);
    pos.lock_.unlock();  // Not ideal but the best we can do without downgradable locks.
    read_lock_type lock{lock_()};

    auto [p, s]{inner_.find_next(pos.inner_, std::forward<PS>(seq), key, hint.inner())};
    return {read_pos{std::move(p), std::move(lock)}, std::move(s)};
  }
  template <typename Pred>
  constexpr read_pos find_if(const Pred& pred) const {
    static_assert(std::is_invocable_r_v<bool, Pred, read_pos>, "`pred` must be pred(read_pos) -> bool");
    static_assert(arg_type_v<arg_n_t<Pred, 0>> != arg_type_t::lvalue_ref, "`pred` cannot be pred(read_pos&) -> bool");

    read_lock_type lock{lock_()};
    inner_read_pos_type pos{inner_.find_if(
      [&](inner_read_pos_type&& pos) {
        read_pos r_lock{std::move(pos), std::move(lock)};
        bool res{pred(r_lock)};
        pos = std::move(r_lock.inner_);
        lock = std::move(r_lock.lock_);
        return res;
      }
    )};
    return read_pos{std::move(pos), std::move(lock)};
  }
  constexpr std::vector<read_pos> find_all(const key_type& key) const {
    read_lock_type lock{lock_()};

    // This is kind of hack. Since we cannot RAII share a write lock without using a `shared_ptr`.
    // Sure, we could use a `shared_ptr`, but that would add a lot of overhead that only makes
    // sense if you really really need these write_positions to be usable.
    //
    // So for now, unless I get a request to change this, we'll keep it like this.
    std::vector<read_pos> res;
    inner_.for_each(key, [&](inner_read_pos_type&& pos, const probe_seq_type&) {
      res.emplace_back(std::move(pos), std::move(lock));
      lock = std::move(res.back().lock_);
    });
    if (!res.empty()) {
      res.back().lock_ = std::move(lock);
    }
    return res;
  }

  template <typename Func>
  constexpr void for_each(const Func& func) const {
    static_assert(std::is_invocable_v<Func, read_pos>, "`func` must be `func(const read_pos&)`!");
    static_assert(arg_type_v<arg_n_t<Func, 0>> == arg_type_t::const_lvalue_ref, "`func` must be `func(const read_pos&)`!");

    read_lock_type lock{lock_()};
    inner_.for_each(
      [&](inner_read_pos_type&& pos) {
        read_pos r_lock{std::move(pos), std::move(lock)};
        func(r_lock);
        pos = std::move(r_lock.inner_);
        lock = std::move(r_lock.lock_);
      }
    );
  }
  template <typename Func>
  constexpr void for_each(const Func& func) {
    if constexpr (std::is_invocable_v<Func, read_pos>) {
      cself()->for_each(func);
    } else if constexpr (std::is_invocable_v<Func, write_pos>) {
      static_assert(arg_type_v<arg_n_t<Func, 0>> == arg_type_t::const_lvalue_ref, "`func` must be `func(const write_pos&)`!");

      write_lock_type lock{lock_()};
      inner_.for_each(
        [&](inner_write_pos_type&& pos) {
          write_pos w_lock{std::move(pos), std::move(lock)};
          func(w_lock);
          pos = std::move(w_lock.inner_);
          lock = std::move(w_lock.lock_);
        }
      );
    } else {
      static_assert(dependent_false_v<Func>, "`func` must be `func(const read_pos&)` or `func(const write_pos&)`!");
    }
  }
  template <typename Func>
  inline void for_each_blob(const Func& func) const {
    read_lock_type lock{lock_()};
    inner_.for_each_blob(func);
  }
  template <typename Func>
  inline void for_each_blob(const Func& func) {
    write_lock_type lock{lock_()};
    inner_.for_each_blob(func);
  }
  template <typename Func>
  inline void for_each_entry(const Func& func) const {
    read_lock_type lock{lock_()};
    inner_.for_each_entry(func);
  }
  template <typename Func>
  inline void for_each_entry(const Func& func) {
    write_lock_type lock{lock_()};
    inner_.for_each_entry(func);
  }
  template <typename Func>
  inline void for_each_key(const Func& func) const {
    read_lock_type lock{lock_()};
    inner_.for_each_key(func);
  }
  template <typename Func>
  inline void for_each_lru(const Func& func) const {
    read_lock_type lock{lock_()};
    inner_.for_each_lru(func);
  }
  template <typename Func>
  constexpr void for_each_state(const Func& func) const {
    read_lock_type lock{lock_()};
    inner_.for_each_state(func);
  }
  template <typename Func>
  inline void for_each_value(const Func& func) const {
    read_lock_type lock{lock_()};
    inner_.for_each_value(func);
  }
  template <typename Func>
  inline void for_each_value(const Func& func) {
    write_lock_type lock{lock_()};
    inner_.for_each_value(func);
  }

  template <typename Func>
  constexpr void for_each(const key_type& key, const Func& func) const {
    static_assert(std::is_invocable_v<Func, read_pos, probe_seq_type>, "`func` must be `func(const read_pos&, probe_seq_type)`!");
    static_assert(arg_type_v<arg_n_t<Func, 0>> == arg_type_t::const_lvalue_ref, "`func` must be `func(const read_pos&, probe_seq_type)`!");

    read_lock_type lock{lock_()};
    inner_.for_each(key,
      [&](inner_read_pos_type&& pos, const probe_seq_type& seq) { 
        read_pos r_lock{std::move(pos), std::move(lock)};
        func(r_lock, seq);
        pos = std::move(r_lock.inner_);
        lock = std::move(r_lock.lock_);
      }
    );
  }
  template <typename Func>
  constexpr void for_each(const key_type& key, const Func& func) {
    if constexpr (std::is_invocable_v<Func, read_pos, probe_seq_type>) {
      cself()->for_each(key, func);
    } else if constexpr (std::is_invocable_v<Func, write_pos, probe_seq_type>) {
      static_assert(arg_type_v<arg_n_t<Func, 0>> == arg_type_t::const_lvalue_ref, "`func` must be `func(const write_pos&, probe_seq_type)`!");

      write_lock_type lock{lock_()};
      inner_.for_each(key,
        [&](inner_write_pos_type&& pos, const probe_seq_type& seq) {
          write_pos w_lock{std::move(pos), std::move(lock)};  
          func(w_lock, seq);
          pos = std::move(w_lock.inner_);
          lock = std::move(w_lock.lock_);
        }
      );
    } else {
      static_assert(dependent_false_v<Func>, "`func` must be `func(read_pos, probe_seq_type)` or `func(const write_pos&, probe_seq_type)`!");
    }
  }

  constexpr void get_blob_at(const read_pos& pos, void* dst) const {
    NVHM_ASSUME_(pos.mutex_() == &mutex_);
    inner_.get_blob_at(pos.inner_, dst);
  }
  constexpr void get_blob_at(const write_pos& pos, void* dst) const {
    NVHM_ASSUME_(pos.mutex_() == &mutex_);
    inner_.get_blob_at(pos.inner_, dst);
  }
  constexpr void get_blob_at(const write_pos& pos, void* dst) {
    NVHM_ASSUME_(pos.mutex_() == &mutex_);
    inner_.get_blob_at(pos.inner_, dst);
  }
  constexpr void get_blob_at(const read_pos& pos, void* dst, int_t n) const {
    NVHM_ASSUME_(pos.mutex_() == &mutex_);
    inner_.get_blob_at(pos.inner_, dst, n);
  }
  constexpr void get_blob_at(const write_pos& pos, void* dst, int_t n) const {
    NVHM_ASSUME_(pos.mutex_() == &mutex_);
    inner_.get_blob_at(pos.inner_, dst, n);
  }
  constexpr void get_blob_at(const write_pos& pos, void* dst, int_t n) {
    NVHM_ASSUME_(pos.mutex_() == &mutex_);
    inner_.get_blob_at(pos.inner_, dst, n);
  }

  constexpr int_t grow() {
    write_lock_type lock{lock_()};
    return inner_.grow();
  }

  template <typename K>
  constexpr write_pos insert(K&& key) {
    write_lock_type lock{lock_()};
    return write_pos{inner_.insert(std::forward<K>(key)), std::move(lock)};
  }
  template <typename K>
  constexpr write_pos insert(K&& key, const prefetch_hint& hint) {
    write_lock_type lock{lock_()};
    return write_pos{inner_.insert(std::forward<K>(key), hint.inner()), std::move(lock)};
  }
  template <typename K>
  constexpr std::tuple<write_pos, probe_seq_type, insert_op_t> insert_ex(K&& key) {
    write_lock_type lock{lock_()};
    auto [p, s, op]{inner_.insert_ex(std::forward<K>(key))};
    return {write_pos{std::move(p), std::move(lock)}, std::move(s), op};
  }
  template <typename K>
  constexpr std::tuple<write_pos, probe_seq_type, insert_op_t> insert_ex(K&& key, const prefetch_hint& hint) {
    write_lock_type lock{lock_()};
    auto [p, s, op]{inner_.insert_ex(std::forward<K>(key), hint.inner())};
    return {write_pos{std::move(p), std::move(lock)}, std::move(s), op};
  }

  constexpr bool is_empty() const {
    read_lock_type lock{lock_()};
    return inner_.is_empty();
  }
  constexpr bool is_full() const {
    read_lock_type lock{lock_()};
    return inner_.is_full();
  }

  constexpr const key_type& key_at(const read_pos& pos) const {
    NVHM_ASSUME_(pos.mutex_() == &mutex_);
    return inner_.key_at(pos.inner_);
  }
  constexpr const key_type& key_at(const write_pos& pos) const {
    NVHM_ASSUME_(pos.mutex_() == &mutex_);
    return inner_.key_at(pos.inner_);
  }

  constexpr double load_factor() const {
    read_lock_type lock{lock_()};
    return inner_.load_factor();
  }

  template <typename InnerPos, typename Lock>
  constexpr lru_t lru_at(const pos<InnerPos, Lock>& pos) const {
    NVHM_ASSUME_(pos.mutex_() == &mutex_);
    return inner_.lru_at(pos.inner_);
  }

  template <typename InnerPos, typename Lock>
  constexpr const_mapped_type mapped_at(const pos<InnerPos, Lock>& pos) const {
    NVHM_ASSUME_(pos.mutex_() == &mutex_);
    return inner_.mapped_at(pos.inner_);
  }
  constexpr mapped_type mapped_at(const write_pos& pos) {
    NVHM_ASSUME_(pos.mutex_() == &mutex_);
    return inner_.mapped_at(pos.inner_);
  }

  constexpr double max_load_factor() const {
    read_lock_type lock{lock_()};
    return inner_.max_load_factor();
  }

  constexpr int_t num_empty_slots() const {
    read_lock_type lock{lock_()};
    return inner_.num_empty_slots();
  }
  constexpr int_t num_not_hash_slots() const {
    read_lock_type lock{lock_()};
    return inner_.num_not_hash_slots();
  }
  constexpr int_t num_tombstone_slots() const {
    read_lock_type lock{lock_()};
    return inner_.num_tombstone_slots();
  }

  constexpr read_pos read_npos() const noexcept { return read_pos{inner_.read_npos()}; }

  constexpr prefetch_hint read_prefetch(const key_type& key) const {
    read_lock_type lock{lock_()};
    return inner_.read_prefetch(key);
  }
  constexpr void read_prefetch(const key_type& key, const prefetch_hint& hint) const {
    read_lock_type lock{lock_()};
    inner_.read_prefetch(key, hint.inner());
  }
  template <typename InnerPos, typename Lock>
  constexpr void read_prefetch_value_at(const pos<InnerPos, Lock>& pos) const {
    NVHM_ASSUME_(pos.mutex_() == &mutex_);
    inner_.read_prefetch_value_at(pos.inner_);
  }
  template <typename InnerPos, typename Lock>
  constexpr void read_prefetch_blob_at(const pos<InnerPos, Lock>& pos) const {
    NVHM_ASSUME_(pos.mutex_() == &mutex_);
    inner_.read_prefetch_blob_at(pos.inner_);
  }

  constexpr void render(std::ostream& os, bool with_values = has_values, blob_render_t with_blobs = has_blobs ? blob_render_t::size : blob_render_t::hide) const {
    read_lock_type lock{lock_()};
    return inner_.render(os, with_values, with_blobs);
  }

  constexpr int_t reserve(int_t n) {
    write_lock_type lock{lock_()};
    return inner_.reserve(n);
  }

  constexpr int_t resize(int_t n) {
    write_lock_type lock{lock_()};
    return inner_.resize(n);
  }

  constexpr int_t scrub(lru_t threshold = max_lru) {
    write_lock_type lock{lock_()};
    return inner_.scrub(threshold);
  }

  constexpr void set_blob_at(const write_pos& pos, const void* src) {
    NVHM_ASSUME_(pos.mutex_() == &mutex_);
    inner_.set_blob_at(pos.inner_, src);
  }
  constexpr void set_blob_at(const write_pos& pos, const void* src, int_t n) {
    NVHM_ASSUME_(pos.mutex_() == &mutex_);
    inner_.set_blob_at(pos.inner_, src, n);
  }

  template <typename V>
  constexpr void set_value_at(const write_pos& pos, V&& value) {
    NVHM_ASSUME_(pos.mutex_() == &mutex_);
    inner_.set_value_at(pos.inner_, std::forward<V>(value));
  }

  constexpr int_t shrink() {
    write_lock_type lock{lock_()};
    return inner_.shrink();
  }

  constexpr int_t size() const {
    read_lock_type lock{lock_()};
    return inner_.size();
  }

  template <typename InnerPos, typename Lock>
  constexpr state_t state_at(const pos<InnerPos, Lock>& pos) const {
    NVHM_ASSUME_(pos.mutex_() == &mutex_);
    return inner_.state_at(pos.inner_);
  }

  constexpr const_iterator to_iterator(read_pos&& pos) const {
    NVHM_ASSUME_(pos.mutex_() == &mutex_);
    return const_iterator{inner_.to_iterator(std::move(pos.inner_)), std::move(pos.lock_)};
  }
  constexpr const_iterator to_iterator(write_pos&& pos) const {
    NVHM_ASSUME_(pos.mutex_() == &mutex_);

    // TODO: This is not a clean solution, but the best we can do without a downgradable lock.
    pos.lock_.unlock();
    read_lock_type lock{lock_()};
    return const_iterator{inner_.to_iterator(std::move(pos.inner_)), std::move(lock)};
  }
  constexpr iterator to_iterator(write_pos&& pos) {
    NVHM_ASSUME_(pos.mutex_() == &mutex_);
    return iterator{inner_.to_iterator(std::move(pos.inner_)), std::move(pos.lock_)};
  }
  constexpr const_iterator to_citerator(read_pos&& pos) const {
    return to_iterator(std::move(pos));
  }
  constexpr const_iterator to_citerator(write_pos&& pos) const {
    return to_iterator(std::move(pos));
  }

  constexpr write_pos update(const key_type& key) {
    write_lock_type lock{lock_()};
    return write_pos{inner_.update(key), std::move(lock)};
  }
  constexpr write_pos update(const key_type& key, const prefetch_hint& hint) {
    write_lock_type lock{lock_()};
    return write_pos{inner_.update(key, hint.inner()), std::move(lock)};
  }
  constexpr std::pair<write_pos, probe_seq_type> update_first(const key_type& key) {
    write_lock_type lock{lock_()};
    auto [p, s]{inner_.update_first(key)};
    return {write_pos{std::move(p), std::move(lock)}, std::move(s)};
  }
  constexpr std::pair<write_pos, probe_seq_type> update_first(const key_type& key, const prefetch_hint& hint) {
    write_lock_type lock{lock_()};
    auto [p, s]{inner_.update_first(key, hint.inner())};
    return {write_pos{std::move(p), std::move(lock)}, std::move(s)};
  }
  template <typename PS>
  constexpr std::pair<write_pos, probe_seq_type> update_next(write_pos&& prev_pos, PS&& seq, const key_type& key) {
    NVHM_ASSUME_(prev_pos.mutex_() == &mutex_);
    auto [p, s]{inner_.update_next(std::move(prev_pos.inner_), std::forward<PS>(seq), key)};
    return {write_pos{std::move(p), std::move(prev_pos.lock_)}, std::move(s)};
  }
  template <typename PS>
  constexpr std::pair<write_pos, probe_seq_type> update_next(write_pos&& prev_pos, PS&& seq, const key_type& key, const prefetch_hint& hint) {
    NVHM_ASSUME_(prev_pos.mutex_() == &mutex_);
    auto [p, s]{inner_.update_next(std::move(prev_pos.inner_), std::forward<PS>(seq), key, hint.inner())};
    return {write_pos{std::move(p), std::move(prev_pos.lock_)}, std::move(s)};
  }
  template <typename Pred>
  constexpr write_pos update_if(const Pred& pred) {
    static_assert(std::is_invocable_r_v<bool, Pred, write_pos>, "`pred` must be pred(const write_pos&) -> bool");
    static_assert(arg_type_v<arg_n_t<Pred, 0>> == arg_type_t::const_lvalue_ref, "`pred` must be pred(const write_pos&) -> bool");

    write_lock_type lock{lock_()};
    inner_write_pos_type pos{inner_.update_if(
      [&](inner_write_pos_type&& pos) {
        write_pos w_lock{std::move(pos), std::move(lock)};
        bool res{pred(w_lock)};
        pos = std::move(w_lock.inner_);
        lock = std::move(w_lock.lock_);
        return res;
      }
    )};
    return write_pos{std::move(pos), std::move(lock)};
  }
  constexpr std::vector<write_pos> update_all(const key_type& key) {
    write_lock_type lock{lock_()};

    // This is kind of hack. Since we cannot RAII share a write lock without using a `shared_ptr`.
    // Sure, we could use a `shared_ptr`, but that would add a lot of overhead that only makes
    // sense if you really really need these write_positions to be usable.
    //
    // So for now, unless I get a request to change this, we'll keep it like this.
    std::vector<write_pos> res;
    inner_.for_each(key, [&](inner_write_pos_type&& pos, const probe_seq_type&) {
      res.emplace_back(std::move(pos), std::move(lock));
      lock = std::move(res.back().lock_);
    });
    if (!res.empty()) {
      res.back().lock_ = std::move(lock);
    }
    return res;
  }

  constexpr const value_type& value_at(const read_pos& pos) const {
    NVHM_ASSUME_(pos.mutex_() == &mutex_);
    return inner_.value_at(pos.inner_);
  }
  constexpr const value_type& value_at(const write_pos& pos) const {
    NVHM_ASSUME_(pos.mutex_() == &mutex_);
    return inner_.value_at(pos.inner_);
  }
  constexpr value_type& value_at(const write_pos& pos) {
    NVHM_ASSUME_(pos.mutex_() == &mutex_);
    return inner_.value_at(pos.inner_);
  }

  constexpr write_pos write_npos() noexcept { return write_pos{inner_.write_npos()}; }

  constexpr prefetch_hint write_prefetch(const key_type& key) {
    write_lock_type lock{lock_()};
    return inner_.write_prefetch(key);
  }
  constexpr void write_prefetch(const key_type& key, const prefetch_hint& hint) {
    write_lock_type lock{lock_()};
    inner_.write_prefetch(key, hint.inner());
  }
  constexpr void write_prefetch_value_at(const write_pos& pos) {
    NVHM_ASSUME_(pos.mutex_() == &mutex_);
    inner_.write_prefetch_value_at(pos.inner_);
  }
  constexpr void write_prefetch_blob_at(const write_pos& pos) {
    NVHM_ASSUME_(pos.mutex_() == &mutex_);
    inner_.write_prefetch_blob_at(pos.inner_);
  }

  template <typename K>
  constexpr mapped_type operator[](K&&) {
    throw std::runtime_error("`guarded` does not support this operator[] access!");
  }
  
  constexpr friend bool operator==(const guarded& lhs, const guarded& rhs) {
    if (&lhs == &rhs) return true;

    read_lock_type lhs_lock{lhs.mutex_, std::defer_lock};
    read_lock_type rhs_lock{rhs.mutex_, std::defer_lock};
    std::lock(lhs_lock, rhs_lock);

    return lhs.inner_ == rhs.inner_;
  }
  constexpr friend bool operator!=(const guarded& lhs, const guarded& rhs) { return !(lhs == rhs); }

  constexpr friend void swap(guarded& lhs, guarded& rhs) {
    if (&lhs == &rhs) return;

    write_lock_type lhs_lock{lhs.mutex_, std::defer_lock};
    write_lock_type rhs_lock{rhs.mutex_, std::defer_lock};
    std::lock(lhs_lock, rhs_lock);  

    swap(lhs.inner_, rhs.inner_);
  }

 protected:
  inner_type inner_;
  mutable mutex_type mutex_;

  inline read_lock_type lock_() const { return read_lock_type{mutex_}; }
  inline write_lock_type lock_() { return write_lock_type{mutex_}; }
  inline read_lock_type clock_() const { return read_lock_type{mutex_}; }
};

}  // namespace nvhm