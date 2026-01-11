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
#include <stdexcept>

namespace nvhm { namespace experimental {

template <typename Sub>
class std_map_shim {
 public:
  using sub_type = Sub;
  using key_type = typename sub_type::key_type;
  using mapped_type = typename sub_type::value_type;
  using value_type = std::pair<key_type, mapped_type>;
  constexpr static bool has_values{sub_type::has_values};

  using read_pos_type = typename sub_type::read_pos_type;
  using write_pos_type = typename sub_type::write_pos_type;

  inline std_map_shim()
  : sub_() {}

  template <typename... SubArgs>
  inline std_map_shim(SubArgs&&... args)
  : sub_(args...) {}

  inline size_t capacity() const noexcept { return sub_.capacity(); }

  inline size_t size() const noexcept { return sub_.size(); }

  inline bool empty() const noexcept { return sub_.empty(); }

  template <typename Pos>
  class iterator_base {
   public:
    using pos_type = Pos;

    inline iterator_base() = delete;

    inline iterator_base(const pos_type& pos)
    : pos_{pos} {}

    inline iterator_base(const iterator_base&) = default;
    inline iterator_base(iterator_base&&) = default;
    inline iterator_base& operator=(const iterator_base&) = default;
    inline iterator_base& operator=(iterator_base&&) = default;

    inline bool operator==(const iterator_base& that) const noexcept { return pos_ == that.pos_; }

    inline bool operator!=(const iterator_base& that) const noexcept { return pos_ != that.pos_; }

    inline iterator_base& operator++() noexcept {
      ++pos_;
      return *this;
    }

    inline explicit operator pos_type() const noexcept { return pos_; }

   protected:
    pos_type pos_;
  };

  class const_iterator : public iterator_base<read_pos_type> {
   public:
    using base_type = iterator_base<read_pos_type>;
    using pos_type = typename base_type::pos_type;

    const_iterator() = delete;

    inline const_iterator(const pos_type& pos, const sub_type* const sub)
    : base_type(pos)
    , sub_{sub} {}

    inline const_iterator(const const_iterator& that)
    : base_type(that)
    , sub_{that.sub_} {}

    inline const_iterator(const_iterator&& that)
    : base_type(that)
    , sub_{std::move(that.sub_)} {}

    inline const_iterator& operator=(const const_iterator& that) {
      base_type::operator=(that);
      sub_ = that.sub_;
      return *this;
    }

    inline const_iterator& operator=(const_iterator&& that) {
      base_type::operator=(that);
      sub_ = std::move(that.sub_);
      return *this;
    }

    inline const value_type* operator->() const noexcept {
      const raw_pos_t pos{this->pos_};
      // TODO: Can we somehow make this so that it will return a persistant reference?
      pair_ = {sub_->key_at(pos), sub_->value_at(pos)};
      return &pair_;
    }

    inline const value_type& operator*() const noexcept { return *operator->(); }

   private:
    const sub_type* sub_;
    mutable value_type pair_;
  };

  class iterator : public iterator_base<write_pos_type> {
   public:
    using base_type = iterator_base<write_pos_type>;
    using pos_type = typename base_type::pos_type;

    iterator() = delete;

    inline iterator(const pos_type& pos, sub_type* const sub)
    : base_type(pos)
    , sub_{sub} {}

    inline iterator(const iterator& that)
    : base_type(that)
    , sub_{that.sub_} {}

    inline iterator(iterator&& that)
    : base_type(that)
    , sub_{std::move(that.sub_)} {}

    inline iterator& operator=(const iterator& that) {
      base_type::operator=(that);
      sub_ = that.sub_;
      return *this;
    }

    inline iterator& operator=(iterator&& that) {
      base_type::operator=(that);
      sub_ = std::move(that.sub_);
      return *this;
    }

    const value_type* operator->() const noexcept {
      const raw_pos_t pos{this->pos_};
      // TODO: Can we somehow make this so that it will return a persistant reference?
      pair_ = {sub_->key_at(pos), sub_->value_at(pos)};
      return &pair_;
    }

    value_type* operator->() noexcept {
      const raw_pos_t pos{this->pos_};
      // TODO: Can we somehow make this so that it will return a persistant reference?
      pair_ = {sub_->key_at(pos), sub_->value_at(pos)};
      return &pair_;
    }

    inline const value_type& operator*() const noexcept { return *operator->(); }

    inline value_type& operator*() noexcept { return *operator->(); }

    inline operator const_iterator() const noexcept { return {this->pos_, sub_}; }

   private:
    sub_type* sub_;
    mutable value_type pair_;
  };
  
  inline const_iterator begin() const { return {{}, sub_}; }
  inline iterator begin() { return {{}, sub_}; }

  inline const_iterator end() const { return {sub_.capacity(), &sub_}; }
  inline iterator end() { return {sub_.capacity(), &sub_}; }

  inline void clear() { sub_.clear(); }

  inline bool contains(const key_type& key) const noexcept { return sub_.contains(key); }

  inline iterator erase(const key_type& key) {
    sub_.erase(key);
    return end();
  }

  inline iterator erase(const iterator& it) {
    sub_.erase_at(static_cast<size_t>(it));
    return end();
  }

  inline const_iterator find(const key_type& key) const noexcept {
    return {sub_.lookup(key), &sub_};
  }

  inline iterator find(const key_type& key) noexcept { return {sub_.lookup(key), &sub_}; }

  inline std::pair<iterator, bool> emplace(const key_type& key, const mapped_type value) {
    const write_pos_type pos{sub_.upsert(key)};
    sub_.value_at(pos) = value;
    return {{pos, &sub_}, pos != npos};
  }

  inline std::pair<iterator, bool> insert(const value_type& kv) {
    return emplace(kv.first, kv.second);
  }

  inline const mapped_type& at(const key_type& key) const {
    const read_pos_type pos{sub_.find(key)};
    if (pos == npos) {
      throw std::out_of_range("Key not in map!");
    }
    return sub_.value_at(pos);
  }

  inline mapped_type& at(const key_type& key) {
    const write_pos_type pos{sub_.find(key)};
    if (pos == npos) {
      throw std::out_of_range("Key not in map!");
    }
    return sub_.value_at(pos);
  }

  const mapped_type& operator[](const key_type& key) const noexcept { return find(key)->second; }

  mapped_type& operator[](const key_type& key) noexcept { return insert(key)->second; }

 private:
  sub_type sub_;
};

}}