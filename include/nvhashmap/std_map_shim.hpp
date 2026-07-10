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
#include <optional>

namespace nvhm {

template <typename Inner>
class std_map_shim {
 public:
  using inner_type = Inner;

  using key_type = typename inner_type::key_type;
  using const_mapped_type = typename inner_type::const_mapped_type;
  using mapped_type = typename inner_type::mapped_type;
  using value_type = std::pair<const key_type, remove_cvref_t<mapped_type>>;
  
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  using inner_read_pos_type = typename inner_type::read_pos;
  using inner_write_pos_type = typename inner_type::write_pos;
  using inner_const_iterator_type = typename inner_type::const_iterator;
  using inner_iterator_type = typename inner_type::iterator;

  class const_iterator;
  class iterator;

  template <typename Self, typename InnerIter, typename Outer>
  class iterator_base : public wrapped_iterator<Self, InnerIter, typename InnerIter::difference_type> {
   public:
    using base_type = wrapped_iterator<Self, InnerIter, typename InnerIter::difference_type>;
    using self_type = typename base_type::self_type;
    using inner_type = typename base_type::inner_type;
    using difference_type = typename base_type::difference_type;
    
    using outer_type = Outer;
    constexpr static bool is_readonly{std::is_const_v<outer_type>};
    using mapped_type = typename inner_type::mapped_type;
    using value_type = std::pair<const_view_t<key_type>, mapped_type>;

    using base_type::self;

    constexpr value_type operator*() const { return {inner_.key(), inner_.mapped()}; }
    constexpr const value_type* operator->() const {
      cache_.emplace(self()->operator*());
      return &cache_.value();
    }

    constexpr self_type& operator++() { ++inner_; return *self(); }
    constexpr self_type& operator--() { --inner_; return *self(); }

    constexpr self_type& operator+=(difference_type n) { inner_ += n; return *self(); }

    template <typename RhsSelf, typename RhsInner, typename RhsOuter>
    constexpr friend difference_type operator-(const iterator_base& lhs, const iterator_base<RhsSelf, RhsInner, RhsOuter>& rhs) { return lhs.inner_ - rhs.inner_; }

    template <typename RhsSelf, typename RhsInner, typename RhsOuter>
    constexpr friend bool operator==(const iterator_base& lhs, const iterator_base<RhsSelf, RhsInner, RhsOuter>& rhs) { return lhs.inner_ == rhs.inner_; }
    template <typename RhsSelf, typename RhsInner, typename RhsOuter>
    constexpr friend bool operator!=(const iterator_base& lhs, const iterator_base<RhsSelf, RhsInner, RhsOuter>& rhs) { return !(lhs == rhs); }
    template <typename RhsSelf, typename RhsInner, typename RhsOuter>
    constexpr friend bool operator<(const iterator_base& lhs, const iterator_base<RhsSelf, RhsInner, RhsOuter>& rhs) { return lhs.inner_ < rhs.inner_; }
    template <typename RhsSelf, typename RhsInner, typename RhsOuter>
    constexpr friend bool operator>(const iterator_base& lhs, const iterator_base<RhsSelf, RhsInner, RhsOuter>& rhs) { return lhs.inner_ > rhs.inner_; }
    template <typename RhsSelf, typename RhsInner, typename RhsOuter>
    constexpr friend bool operator<=(const iterator_base& lhs, const iterator_base<RhsSelf, RhsInner, RhsOuter>& rhs) { return !(lhs > rhs); }
    template <typename RhsSelf, typename RhsInner, typename RhsOuter>
    constexpr friend bool operator>=(const iterator_base& lhs, const iterator_base<RhsSelf, RhsInner, RhsOuter>& rhs) { return !(lhs < rhs); }

    constexpr friend void swap(iterator_base& lhs, iterator_base& rhs) {
      if (&lhs == &rhs) return;
      swap(static_cast<base_type&>(lhs), static_cast<base_type&>(rhs));

      lhs.cache_.reset();
      rhs.cache_.reset();
    }

   protected:
    using base_type::inner_;
    mutable std::optional<value_type> cache_;

    iterator_base() = delete;
    constexpr iterator_base(const iterator_base& other) noexcept : base_type(other) {}
    constexpr iterator_base& operator=(const iterator_base& other) noexcept {
      base_type::operator=(other);
      cache_.reset();
      return *this;
    }
    constexpr iterator_base(iterator_base&& other) noexcept : base_type(std::move(other)) {}
    constexpr iterator_base& operator=(iterator_base&& other) noexcept {
      base_type::operator=(std::move(other));
      cache_.reset();
      other.cache_.reset();
      return *this;
    }

    constexpr explicit iterator_base(inner_type&& inner) : base_type(std::move(inner)) {}

    template <typename, typename, typename>
    friend class iterator_base;
  };

  class const_iterator : public iterator_base<const_iterator, inner_const_iterator_type, std_map_shim> {
   public:
    using base_type = iterator_base<const_iterator, inner_const_iterator_type, std_map_shim>;
    using self_type = typename base_type::self_type;
    using inner_type = typename base_type::inner_type;

    using outer_type = typename base_type::outer_type;

    const_iterator() = delete;
    constexpr const_iterator(const const_iterator&) noexcept = default;
    constexpr const_iterator& operator=(const const_iterator&) noexcept = default;
    constexpr const_iterator(const_iterator&&) noexcept = default;
    constexpr const_iterator& operator=(const_iterator&&) noexcept = default;

   protected:
    constexpr const_iterator(inner_type&& inner) : base_type(std::move(inner)) {}

    friend class std_map_shim;
  };

  class iterator : public iterator_base<iterator, inner_iterator_type, std_map_shim> {
   public:
    using base_type = iterator_base<iterator, inner_iterator_type, std_map_shim>;
    using self_type = typename base_type::self_type;
    using inner_type = typename base_type::inner_type;

    iterator() = delete;
    constexpr iterator(const iterator&) noexcept = default;
    constexpr iterator& operator=(const iterator&) noexcept = default;
    constexpr iterator(iterator&&) noexcept = default;
    constexpr iterator& operator=(iterator&&) noexcept = default;

   protected:
    constexpr iterator(inner_type&& inner) : base_type(std::move(inner)) {}

    friend class std_map_shim;
  };
  
  constexpr std_map_shim() {}
  template <typename Arg0, typename... Args, typename = std::enable_if_t<!std::is_same_v<remove_cvref_t<Arg0>, std_map_shim>>>
  constexpr std_map_shim(Arg0&& arg0, Args&&... args)
    : inner_{std::forward<Arg0>(arg0), std::forward<Args>(args)...} {}

  constexpr const_mapped_type at(const key_type& key) const {
    const auto pos{inner_.find(key)};
    if (pos == npos) {
      throw std::out_of_range("Key is not in the map!");
    }
    return inner_.mapped_at(pos);
  }
  constexpr mapped_type at(const key_type& key) {
    const auto pos{inner_.update(key)};
    if (pos == npos) {
      throw std::out_of_range("Key is not in the map!");
    }
    return inner_.mapped_at(pos);
  }

  constexpr const_iterator begin() const { return inner_.begin(); }
  constexpr iterator begin() { return inner_.begin(); }

  constexpr size_type capacity() const { return to_uint(inner_.capacity()); }

  constexpr const_iterator cbegin() const { return begin(); }
  constexpr const_iterator cend() const { return end(); }

  constexpr void clear() { inner_.clear(); }

  constexpr bool contains(const key_type& key) const { return inner_.contains(key); }
  
  constexpr size_type count(const key_type& key) const { return to_uint(inner_.count(key)); }

  template <typename K, typename V>
  constexpr std::pair<iterator, bool> emplace(K&& key, V&& value) {
    return insert(std::forward<K>(key), std::forward<V>(value));
  }
  
  constexpr bool empty() const { return inner_.is_empty(); }

  constexpr const_iterator end() const { return inner_.end(); }
  constexpr iterator end() { return inner_.end(); }
  
  std::pair<const_iterator, const_iterator> equal_range(const key_type& key) const {
    auto pos{inner_.find(key)};
    bool is_npos{pos == npos};
    auto it1{inner_.to_iterator(std::move(pos))};
    if (!is_npos) {
      ++it1;
    }
    auto it0{inner_.to_iterator(inner_.find(key))};
    return {std::move(it0), std::move(it1)};
  }
  std::pair<iterator, iterator> equal_range(const key_type& key) {
    auto pos{inner_.update(key)};
    bool is_npos{pos == npos};
    auto it1{inner_.to_iterator(std::move(pos))};
    if (!is_npos) {
      ++it1;
    }
    auto it0{inner_.to_iterator(inner_.update(key))};
    return {std::move(it0), std::move(it1)};
  }

  constexpr iterator erase(const key_type& key) {
    auto pos{std::get<1>(inner_.erase_first(key))};
    bool is_npos{pos == npos};
    auto it{inner_.to_iterator(std::move(pos))};
    if (!is_npos) {
      ++it;
    }
    return it;
  }
  constexpr void erase(iterator& pos) {
    if (!pos.inner_.erase()) {
      throw std::out_of_range("The iterator did not point to a valid entry!");
    }
  }
  template<typename LastIt>
  constexpr iterator erase(iterator&& first, const LastIt& last) {
    if constexpr (test_flags(inner_type::flags, flags_t::auto_shrink)) {
      throw std::runtime_error("Erasing a range is not supported when `auto_shrink` is enabled!");
    }

    for (; first < last; ++first) {
      erase(first);
    }
    return std::move(first);
  }

  constexpr const_iterator find(const key_type& key) const {
    return inner_.to_iterator(inner_.find(key));
  }
  constexpr iterator find(const key_type& key) {
    return inner_.to_iterator(inner_.update(key));
  }

  template <typename K, typename V>
  constexpr std::pair<iterator, bool> insert(K&& key, V&& value) {
    auto [pos, seq, op]{inner_.insert_ex(std::forward<K>(key))};
    if (op == insert_op_t::reject) {
      throw std::runtime_error("Key insertion was rejected by the underlying container!");
    }

    bool inserted{op != insert_op_t::found};
    if (inserted) {
      inner_.mapped_at(pos) = std::forward<V>(value);
    }

    return {inner_.to_iterator(std::move(pos)), inserted};
  }
  template <typename K, typename V>
  constexpr std::pair<iterator, bool> insert_or_assign(K&& key, V&& value) {
    auto [pos, _, op]{inner_.insert_ex(std::forward<K>(key))};
    if (op == insert_op_t::reject) {
      throw std::runtime_error("Key insertion was rejected by the underlying container!");
    }

    inner_.mapped_at(pos) = std::forward<V>(value);

    return {inner_.to_iterator(std::move(pos)), op != insert_op_t::found};
  }
  
  constexpr float load_factor() const { return static_cast<float>(inner_.load_factor()); }

  constexpr float max_load_factor() const { return static_cast<float>(inner_.max_load_factor()); }
  constexpr void max_load_factor(float) {
    throw std::out_of_range("Setting t he `max_load_factor` is currently not supported!");
  }

  constexpr size_type max_size() const { return to_uint(max_capacity); }

  constexpr void rehash(size_type count) { inner_.resize(to_int(count)); }

  constexpr void reserve(size_type count) { inner_.reserve(to_int(count)); }

  constexpr size_type size() const { return to_uint(inner_.size()); }

  template<typename K, typename... Args>
  constexpr std::pair<iterator, bool> try_emplace(K&& key, Args&&... args) {
    auto [pos, _, op]{inner_.insert_ex(std::forward<K>(key))};
    if (op == insert_op_t::reject) {
      throw std::runtime_error("Key insertion was rejected by the underlying container!");
    }

    bool inserted{op != insert_op_t::found};
    if (inserted) {
      // Should use construct_at here. But will need to do this later.
      //construct_at(&inner_.value_at(pos), std::forward<Args>(args)...);
      inner_.value_at(pos) = typename inner_type::value_type{std::forward<Args>(args)...};
    }

    return {inner_.to_iterator(std::move(pos)), inserted};
  }

  template <typename K>
  constexpr mapped_type operator[](K&& key) { return inner_[key]; }

  constexpr friend bool operator==(const std_map_shim& lhs, const std_map_shim& rhs) {
    return (&lhs == &rhs) || (lhs.inner_ == rhs.inner_);
  }
  constexpr friend bool operator!=(const std_map_shim& lhs, const std_map_shim& rhs) { return !(lhs == rhs); }

  constexpr friend void swap(std_map_shim& lhs, std_map_shim& rhs) { 
    if (&lhs == &rhs) return;

    std::swap(lhs.inner_, rhs.inner_);
  }

 private:
  inner_type inner_;
};

}  // namespace nvhm