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

#include "hash.hpp"

namespace nvhm {

#if defined(NVHM_CONTAINER_INTERFACE_)
#error NVHM_CONTAINER_INTERFACE_ was defined elsewhere. Something is wrong.
#endif
#define NVHM_CONTAINER_INTERFACE_()\
  /**
   * Count the number of collisions for each bucket.
   *
   * @return An array of size `kernel_size` containing the number of collision distribution.
   */\
  constexpr std::array<int_t, kernel_size> state_collisions() const {\
    std::array<int_t, kernel_size> counts{};\
    self()->count_state_collisions(counts);\
    return counts;\
  }\
  \
  /**
   * Retrieve all keys in the data structure.
   *
   * @return The keys.
   */\
  constexpr std::vector<key_type> keys() const {\
    std::vector<key_type> res;\
    self()->for_each_key([&](const key_type& key) { res.emplace_back(key); });\
    return res;\
  }\
  \
  /**
   * Retrieve all value views in the data structure.
   *
   * @return The value views.
   */\
   constexpr std::vector<value_type> values() const {\
    std::vector<value_type> res;\
    self()->for_each_value([&](const value_type& value) { res.emplace_back(value); });\
    return res;\
  }

template <typename Self>
class container : public self_aware<Self> {
 public:
  using base_type = self_aware<Self>;
  using self_type = typename base_type::self_type;

  constexpr container() = default;

  using base_type::self;

  /**
   * Retrieve all blobs in the data structure.
   *
   * @return The blobs.
   */
  inline std::vector<blob_t> blobs() const {
    const int_t blob_size{self()->conf().blob_size()};

    std::vector<blob_t> res;
    self()->for_each_blob([&](const blob_t* blob) {
      for (int_t i{}; i < blob_size; ++i) {
        res.emplace_back(blob[i]);
      }
    });
    return res;
  }

  /**
   * Erases a range of keys from the container.
   *
   * @param first The first key to insert.
   * @param last The last key to insert.
   * @return The number of keys that were actually erased.
   */
  template <typename It>
  constexpr int_t erase_range(It first, It last) {
    int_t n{};
    for (; first != last; ++first) {
      n += self()->erase(*first);
    }
    return n;
  }

  /**
   * Checks whether a range of keys exist in the container.
   *
   * @param first The first key to insert.
   * @param last The last key to insert.
   * @return The number of keys that were successfully located.
   */
  template <typename It>
  constexpr int_t find_range(It first, It last) const {
    int_t n{};
    for (; first != last; ++first) {
      n += self()->contains(*first);
    }
    return n;
  }

  /**
   * Inserts a range of keys into the container.
   *
   * @param first The first key to insert.
   * @param last The last key to insert.
   * @return The number of keys inserted.
   */
  template <typename It>
  constexpr int_t insert_range(It first, It last) {
    int_t n{};
    for (; first != last; ++first) {
      n += self()->insert(*first) != npos;
    }
    return n;
  }

  /**
   * Retrieve all LRUs in the data structucre.
   *
   * @return The LRUs.
   */
  inline std::vector<lru_t> lrus() const {
    std::vector<lru_t> res;
    self()->for_each_lru([&](lru_t lru) { res.emplace_back(lru); });
    return res;
  }
  
  /**
   * Retrieve all "valid" states in the data structucre.
   *
   * @return The states.
   */
   inline std::vector<state_t> states() const {
    std::vector<state_t> res;
    self()->for_each_state([&](state_t state) { res.emplace_back(state); });
    return res;
  }

  /**
   * Render the data structure to a string.
   *
   * @return The string.
   */
  inline std::string to_string() const {
    std::ostringstream os;
    self()->render(os);
    return os.str();
  }
  

  /**
   * Render the data structure to an output stream.
   *
   * @param os The output stream.
   * @param container The container to render.
   * @return The output stream.
   */
  constexpr friend std::ostream& operator<<(std::ostream& lhs, const self_type& rhs) {
    rhs.render(lhs);
    return lhs;
  }
};

template <typename Inner>
class wrapped {
 public:
  using inner_type = Inner;

  wrapped() = delete;

  constexpr friend void swap(wrapped& lhs, wrapped& rhs) noexcept {
    if (&lhs == &rhs) return;

    std::swap(lhs.inner_, rhs.inner_);
  }

 protected:
  inner_type inner_;

  constexpr explicit wrapped(const inner_type& v) noexcept : inner_{v} {}
  constexpr explicit wrapped(inner_type&& inner) noexcept : inner_{std::move(inner)} {}
};

template <typename Inner>
class wrapped_pos : public wrapped<Inner> {
 public:
  using base_type = wrapped<Inner>;
  using inner_type = typename base_type::inner_type;

  wrapped_pos() = delete;

  constexpr raw_pos_t raw() const noexcept {
    if constexpr (std::is_same_v<inner_type, raw_pos_t>) {
      return inner_;
    } else {
      return inner_.raw();
    }
  }

  constexpr friend bool operator==(const wrapped_pos& lhs, raw_pos_t rhs) noexcept { return lhs.inner_ == rhs; }
  constexpr friend bool operator!=(const wrapped_pos& lhs, raw_pos_t rhs) noexcept { return !(lhs == rhs); }
  constexpr friend bool operator<(const wrapped_pos& lhs, raw_pos_t rhs) noexcept { return lhs.inner_ < rhs; }
  constexpr friend bool operator>(const wrapped_pos& lhs, raw_pos_t rhs) noexcept { return lhs.inner_ > rhs; }
  constexpr friend bool operator<=(const wrapped_pos& lhs, raw_pos_t rhs) noexcept { return !(lhs > rhs); }
  constexpr friend bool operator>=(const wrapped_pos& lhs, raw_pos_t rhs) noexcept { return !(lhs < rhs); }

  constexpr friend std::ostream& operator<<(std::ostream& os, const wrapped_pos& p) { return os << p.inner_; }

 protected:
  using base_type::inner_;
 
  constexpr explicit wrapped_pos(const inner_type& v) noexcept : base_type{v} {}
  constexpr explicit wrapped_pos(inner_type&& v) noexcept : base_type{std::move(v)} {}
};

template<typename Self, typename Inner, typename DifferenceType>
class wrapped_iterator : public self_aware<Self>, public wrapped<Inner> {
 public:
  using self_type = typename self_aware<Self>::self_type;
  using base_type = wrapped<Inner>;
  using inner_type = Inner;

  using difference_type = DifferenceType;

  wrapped_iterator() = delete;

  using self_aware<Self>::self;

  constexpr self_type& operator-=(difference_type n) { return *self() += -n; }

  constexpr friend self_type operator++(self_type& it, int) { self_type tmp{it}; ++it; return tmp; }
  constexpr friend self_type operator--(self_type& it, int) { self_type tmp{it}; --it; return tmp; }

  constexpr friend self_type operator+(self_type it, difference_type n) { it += n; return it; }
  constexpr friend self_type operator-(self_type it, difference_type n) { it -= n; return it; }

  constexpr auto operator[](difference_type n) const { return *(*self() + n); }

 protected:
  constexpr explicit wrapped_iterator(const inner_type& inner) noexcept : base_type{inner} {}
  constexpr explicit wrapped_iterator(inner_type&& inner) noexcept : base_type{std::move(inner)} {}
};

template<typename Self, typename Inner>
class wrapped_map_iterator : public wrapped_iterator<Self, Inner, typename Inner::difference_type> {
 public:
  using base_type = wrapped_iterator<Self, Inner, typename Inner::difference_type>;
  using self_type = typename base_type::self_type;
  using inner_type = typename base_type::inner_type;

  using iterator_category = typename inner_type::iterator_category;

  using key_type = typename inner_type::key_type;
  using value_type = typename inner_type::value_type;
  using blob_type = typename inner_type::blob_type;
  using entry_type = typename inner_type::entry_type;
  using mapped_type = typename inner_type::mapped_type;
  using entry_ptr_type = typename inner_type::entry_ptr_type;

  wrapped_map_iterator() = delete;

  constexpr blob_type blob() const noexcept { return inner_.blob(); }
  constexpr entry_type entry() const noexcept { return inner_.entry(); }
  constexpr key_type key() const noexcept { return inner_.key(); }
  constexpr lru_t lru() const noexcept { return inner_.lru(); }
  constexpr bool query() const noexcept { return inner_.query(); }
  constexpr value_type value() const noexcept { return inner_.value(); }
  constexpr mapped_type mapped() const noexcept { return inner_.mapped(); }

  constexpr entry_ptr_type operator*() const noexcept { return inner_.operator*(); }
  constexpr const entry_ptr_type* operator->() const noexcept { return inner_.operator->(); }

 protected:
  using base_type::inner_;

  constexpr explicit wrapped_map_iterator(const inner_type& inner) noexcept : base_type{inner} {}
  constexpr explicit wrapped_map_iterator(inner_type&& inner) noexcept : base_type{std::move(inner)} {}
};

enum class insert_op_t : int_t {
  reject = 0,
  found = 1,
  insert = 2,
  replace = 3,
};

constexpr const char* to_string(insert_op_t op) {
  switch (op) {
    case insert_op_t::reject:
      return "reject";
    case insert_op_t::found:
      return "found";
    case insert_op_t::insert:
      return "insert";
    case insert_op_t::replace:
      return "replace";
  }
  return "error";
}

inline std::ostream& operator<<(std::ostream& os, insert_op_t op) { return os << to_string(op); }

enum class blob_render_t : int_t {
  hide = 0,
  size = 1,
  full = 2,
};

constexpr const char* to_string(blob_render_t br) {
  switch (br) {
    case blob_render_t::hide:
      return "hide";
    case blob_render_t::size:
      return "size";
    case blob_render_t::full:
      return "full";
  }
  return "error";
}

inline std::ostream& operator<<(std::ostream& os, blob_render_t br) { return os << to_string(br); }

template <typename Self, typename Key, typename Value>
class nested_container : public self_aware<Self> {
 public:
  using base_type = self_aware<Self>;
  using self_type = typename base_type::self_type;

  using key_type = Key;
  using value_type = Value;
};

} // namespace nvhm