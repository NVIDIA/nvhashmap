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
#include <mutex>
#include <shared_mutex>

namespace nvhm {

template <typename Sub>
class guarded {
 public:
  using sub_type = Sub;
  using key_type = typename sub_type::key_type;
  using value_type = typename sub_type::value_type;
  constexpr static bool has_values{sub_type::has_values};
  using raw_value_type = typename sub_type::raw_value_type;

  using kernel_type = typename sub_type::kernel_type;
  using mask_type = typename sub_type::mask_type;
  using probe_seq_type = typename sub_type::probe_seq_type;
  constexpr static bool minimize_psl{sub_type::minimize_psl};
  constexpr static bool auto_shrink{sub_type::auto_shrink};
  constexpr static bool allow_reclaim{sub_type::allow_reclaim};
  using allocator_type = typename sub_type::allocator_type;

  using mutex_type = std::shared_mutex;
  using read_lock_type = std::shared_lock<mutex_type>;
  using write_lock_type = std::unique_lock<mutex_type>;

  class read_pos final {
   public:
    using pos_type = typename sub_type::read_pos_type;

    read_pos() = delete;
    read_pos(const read_pos&) = delete;
    read_pos(read_pos&&) = default;
    read_pos& operator=(const read_pos&) = delete;
    read_pos& operator=(read_pos&&) = default;

    read_pos(read_lock_type&& lock, pos_type&& pos)
    : lock_{std::move(lock)}
    , pos_{std::move(pos)} {}

    inline operator const pos_type&() const noexcept { return pos_; }

    inline bool operator==(const raw_pos_t& that) const noexcept { return pos_ == that; }
    inline bool operator!=(const raw_pos_t& that) const noexcept { return pos_ != that; }

   private:
    read_lock_type lock_;
    pos_type pos_;
  };

  class write_pos final {
   public:
    using pos_type = typename sub_type::write_pos_type;

    write_pos() = delete;
    write_pos(const write_pos&) = delete;
    write_pos(write_pos&&) = default;
    write_pos& operator=(const write_pos&) = delete;
    write_pos& operator=(write_pos&&) = default;

    write_pos(write_lock_type&& lock, pos_type&& pos)
    : lock_{std::move(lock)}
    , pos_{std::move(pos)} {}

    inline operator const pos_type&() const noexcept { return pos_; }

    inline bool operator==(const raw_pos_t& that) const noexcept { return pos_ == that; }
    inline bool operator!=(const raw_pos_t& that) const noexcept { return pos_ != that; }

   private:
    write_lock_type lock_;
    pos_type pos_;
  };

  using prefetch_type = std::tuple<typename sub_type::prefetch_type>;
  using read_pos_type = read_pos;
  using write_pos_type = write_pos;

  constexpr static size_t min_capacity{sub_type::min_capacity};
  constexpr static size_t default_capacity{sub_type::default_capacity};

  inline guarded()
  : sub_() {}

  template <typename... SubArgs>
  inline guarded(SubArgs&&... args)
  : sub_(args...) {}

  inline guarded(const guarded& that) {
    read_lock_type lock(that.mutex_);
    sub_ = that.sub_;
  }

  inline guarded& operator=(const guarded& that) {
    read_lock_type lock(that.mutex_);
    sub_ = that.sub_;
    return *this;
  }

  inline guarded(guarded&& that) {
    write_lock_type lock(that.mutex_);
    sub_ = std::move(that.sub_);
  }

  inline guarded& operator=(guarded&& that) {
    write_lock_type lock(that.mutex_);
    sub_ = std::move(that.sub_);
    return *this;
  }

  inline ~guarded() { write_lock_type lock(mutex_); }

  inline void swap(guarded& __restrict that) noexcept {
    write_lock_type this_lock(mutex_), that_lock(that.mutex_);
    std::swap(sub_, that.sub_);
  }

  inline size_t raw_value_size() const noexcept {
    read_lock_type lock(mutex_);
    return sub_.raw_value_size();
  }

  inline size_t capacity() const noexcept {
    read_lock_type lock(mutex_);
    return sub_.capacity();
  }

  inline size_t size() const noexcept {
    read_lock_type lock(mutex_);
    return sub_.size();
  }

  inline bool empty() const noexcept {
    read_lock_type lock(mutex_);
    return sub_.empty();
  }

  inline bool full() const noexcept {
    read_lock_type lock(mutex_);
    return sub_.full();
  }

 public:
  inline void clear() noexcept {
    write_lock_type lock(mutex_);
    sub_.clear();
  }

  inline bool contains(const key_type& key) const noexcept {
    read_lock_type lock(mutex_);
    return sub_.contains(key);
  }

  inline bool contains(const key_type& key, const prefetch_type& pre) const noexcept {
    read_lock_type lock(mutex_);
    return sub_.contains(key, pre);
  }

  inline bool erase(const key_type& key) {
    write_lock_type lock(mutex_);
    return sub_.erase(key);
  }

  inline bool erase(const key_type& key, const prefetch_type& pre) {
    write_lock_type lock(mutex_);
    return sub_.erase(key, pre);
  }

  inline bool erase_at(const write_pos_type& pos) noexcept { return sub_.erase_at(pos); }

  inline read_pos_type lookup(const key_type& key) const noexcept {
    read_lock_type lock(mutex_);
    return {std::move(lock), sub_.lookup(key)};
  }

  inline read_pos_type lookup(const key_type& key, const prefetch_type& pre) const noexcept {
    read_lock_type lock(mutex_);
    return {std::move(lock), sub_.lookup(key, std::get<0>(pre))};
  }

  inline write_pos_type update(const key_type& key) noexcept {
    write_lock_type lock(mutex_);
    return {std::move(lock), sub_.update(key)};
  }

  inline write_pos_type update(const key_type& key, const prefetch_type& pre) noexcept {
    write_lock_type lock(mutex_);
    return {std::move(lock), sub_.update(key, std::get<0>(pre))};
  }

  inline write_pos_type upsert(const key_type& key) {
    write_lock_type lock(mutex_);
    return {std::move(lock), sub_.upsert(key)};
  }

  inline write_pos_type upsert(const key_type& key, const prefetch_type& pre) {
    write_lock_type lock(mutex_);
    return {std::move(lock), sub_.upsert(key, std::get<0>(pre))};
  }

  template <typename It>
  inline write_pos_type upsert(const key_type& key, It& it) {
    write_lock_type lock(mutex_);
    return {std::move(lock), sub_.upsert(key, it)};
  }

  template <typename It>
  inline write_pos_type upsert(const key_type& key, It& it, const prefetch_type& pre) {
    write_lock_type lock(mutex_);
    return {&sub_, sub_.upsert(key, it, std::get<0>(pre)), std::move(lock)};
  }

  inline prefetch_type read_prefetch(const key_type& key, const bool optimistic) const noexcept {
    read_lock_type lock(mutex_);
    return {sub_.read_prefetch(key, optimistic)};
  }

  inline prefetch_type write_prefetch(const key_type& key, const bool optimistic) noexcept {
    write_lock_type lock(mutex_);
    return {sub_.write_prefetch(key, optimistic)};
  }

 public:
  template <typename OutIt>
  inline OutIt keys(OutIt it) const noexcept {
    read_lock_type lock(mutex_);
    return sub_.keys(it);
  }

  template <typename OutIt>
  inline OutIt keys_and_values(OutIt it) const noexcept {
    read_lock_type lock(mutex_);
    return sub_.keys_and_values(it);
  }

  inline void transform_values(const std::function<void(value_type& __restrict)>& __restrict f) {
    write_lock_type lock(mutex_);
    sub_.transform_values(f);
  }

  inline void transform_raw_values(
    const std::function<raw_value_type(raw_value_type)>& __restrict f
  ) {
    write_lock_type lock(mutex_);
    sub_.transform_raw_values(f);
  }

 public:
  inline bool is_occupied(const read_pos_type& pos) const noexcept { return sub_.is_occupied(pos); }

  inline state_t state_at(const read_pos_type& pos) const noexcept { return sub_.state_at(pos); }

  inline const key_type& key_at(const read_pos_type& pos) const noexcept {
    return sub_.key_at(pos);
  }

  inline const value_type& value_at(const read_pos_type& pos) const noexcept {
    return sub_.value_at(pos);
  }

  inline value_type& value_at(const write_pos_type& pos) noexcept { return sub_.value_at(pos); }

  inline const raw_value_type* raw_values_at(const read_pos_type& pos) const noexcept {
    return sub_.raw_values_at(pos);
  }

  inline raw_value_type* raw_values_at(const write_pos_type& pos) noexcept {
    return sub_.raw_values_at(pos);
  }

  inline const raw_value_type* get_raw_values(
    const read_pos_type& pos, raw_value_type* const dst, const size_t n
  ) const noexcept {
    return sub_.get_raw_values(pos, dst, n);
  }

  inline void set_raw_values(
    const write_pos_type& pos, const raw_value_type* const src, const size_t n
  ) noexcept {
    sub_.set_raw_values(pos, src, n);
  }

 public:
  inline size_t psl(const key_type& key) const noexcept {
    read_lock_type lock(mutex_);
    return sub_.psl(key);
  }

  inline std::array<size_t, kernel_type::size> state_collisions() const noexcept {
    read_lock_type lock(mutex_);
    return sub_.state_collisions();
  }

 protected:
  mutable mutex_type mutex_;
  sub_type sub_;
};

}  // namespace nvhm