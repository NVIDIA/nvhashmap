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

#include "raw_map_base.hpp"

namespace nvhm {

template <typename Self, typename Conf, typename ProbeSeq, typename Allocator>
class swiss_map_base : public raw_map_base<Self, Conf, ProbeSeq, Allocator> {
 public:
  using base_type = raw_map_base<Self, Conf, ProbeSeq, Allocator>;

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

  using pos = typename base_type::pos;
  using read_pos = typename base_type::read_pos;
  using write_pos = typename base_type::write_pos;

  using prefetch_hint = typename base_type::prefetch_hint;

  swiss_map_base() = delete;
  constexpr swiss_map_base(const conf_type& conf) : base_type{conf} {
    keys_ = make_unique<key_type, allocator_type>(capacity());
  }
  constexpr swiss_map_base(const swiss_map_base& other) : base_type{other} {
    keys_ = make_copy(other.keys_, capacity());
  }
  constexpr swiss_map_base& operator=(const swiss_map_base& other) {
    if (this == &other) return *this;
    base_type::operator=(other);

    std::copy_n(other.keys_.get(), capacity(), keys_.get());
    return *this;
  }
  constexpr swiss_map_base(swiss_map_base&& other) noexcept = default;
  constexpr swiss_map_base& operator=(swiss_map_base&& other) noexcept = default;

  using base_type::capacity;
  using base_type::self;
  using base_type::validate_range;

  /**
   * Fetch the number of entries in the bucket at `pos`.
   *
   * @param pos The position to query.
   * @return The number of entries in the bucket.
   */
  constexpr int_t bucket_size_at(const pos& pos) const {
    return self()->bucket_size_at_(validate_range(pos));
  }
  /**
   * Fetch the number of empty slots in the bucket at `pos`.
   *
   * @param pos The position to query.
   * @return The number of empty slots in the bucket.
   */
  constexpr int_t bucket_num_empty_slots_at(const pos& pos) const {
    return self()->bucket_num_empty_slots_at_(validate_range(pos));
  }
  /**
   * Fetch the number of tombstone slots in the bucket at `pos`.
   *
   * @param pos The position to query.
   * @return The number of tombstone slots in the bucket.
   */
  constexpr int_t bucket_num_tombstone_slots_at(const pos& pos) const {
    return self()->bucket_num_tombstone_slots_at_(validate_range(pos));
  }

  /**
   * Count the number of collisions for each bucket.
   *
   * @param counts An array of size `kernel_type::size` containing the number of collision distribution. Not zeroed before use.
   */
  constexpr void count_state_collisions(std::array<int_t, kernel_size>& counts) const noexcept {
    self()->count_state_collisions_(counts);
  }

  /**
   * Iterate over the data structure and call a function for each entry.
   *
   * @param func The function to call.
   */
  template <typename Func>
  constexpr void for_each_entry(const Func& func) const {
    if constexpr (has_values && has_blobs) {
      static_assert(std::is_invocable_v<Func, key_type, value_type, const blob_t*>, "`func` must be `func(key_type, value_type, const blob_t*)`!");
    } else if constexpr (has_values) {
      static_assert(std::is_invocable_v<Func, key_type, value_type>, "`func` must be `func(key_type, value_type)`!");
    } else if constexpr (has_blobs) {
      static_assert(std::is_invocable_v<Func, key_type, const blob_t*>, "`func` must be `func(key_type, const blob_t*)`!");
    } else {
      static_assert(std::is_invocable_v<Func, key_type>, "`func` must be `func(key_type)`!");
    }
    const int_t blob_stride{conf_.blob_stride()};

    value_type* const __restrict values{values_.get()};
    std::byte* const __restrict blobs{blobs_.get()};
    const key_type* const __restrict keys{keys_.get()};

    self()->for_each_([&](raw_pos_t pos) {
      if constexpr (has_values && has_blobs) {
        func(keys[pos], values[pos], &blobs[pos * blob_stride]);
      } else if constexpr (has_values) {
        func(keys[pos], values[pos]);
      } else if constexpr (has_blobs) {
        func(keys[pos], &blobs[pos * blob_stride]);
      } else {
        func(keys[pos]);
      }
    });
  }
  /**
   * Iterate over the data structure and call a function for each entry.
   *
   * @param func The function to call.
   */
  template <typename Func>
  constexpr void for_each_entry(const Func& func) {
    if constexpr (has_values && has_blobs) {
      static_assert(std::is_invocable_v<Func, key_type, value_type&, blob_t*>, "`func` must be `func(key_type, value_type&, blob_t*)`!");
    } else if constexpr (has_values) {
      static_assert(std::is_invocable_v<Func, key_type, value_type&>, "`func` must be `func(key_type, value_type&)`!");
    } else if constexpr (has_blobs) {
      static_assert(std::is_invocable_v<Func, key_type, blob_t*>, "`func` must be `func(key_type, blob_t*)`!");
    } else {
      static_assert(std::is_invocable_v<Func, key_type>, "`func` must be `func(key_type)`!");
    }
    const int_t blob_stride{conf_.blob_stride()};

    value_type* const __restrict values{values_.get()};
    std::byte* const __restrict blobs{blobs_.get()};
    const key_type* const __restrict keys{keys_.get()};

    self()->for_each_([&](raw_pos_t pos) {
      if constexpr (has_values && has_blobs) {
        func(keys[pos], values[pos], &blobs[pos * blob_stride]);
      } else if constexpr (has_values) {
        func(keys[pos], values[pos]);
      } else if constexpr (has_blobs) {
        func(keys[pos], &blobs[pos * blob_stride]);
      } else {
        func(keys[pos]);
      }
    });
  }
  /**
   * Iterate over the data structure and call a function for each key.
   *
   * @param func The function to call.
   */
  template <typename Func>
  constexpr void for_each_key(const Func& func) const {
    static_assert(std::is_invocable_v<Func, key_type>, "`func` must be `func(key_type)`!");
    const key_type* const __restrict keys{keys_.get()};

    self()->for_each_([&](raw_pos_t pos) { func(keys[pos]); });
  }

  /**
   * @return Current number of empty slots.
   */
  constexpr int_t num_empty_slots() const noexcept { return self()->num_empty_slots_(); }
  /**
   * @return Current number of slots occupied by either a hash or tombstone.
   */
  constexpr int_t num_not_hash_slots() const noexcept { return num_empty_slots() + num_tombstone_slots(); }
  /**
   * @return Current number of tombstone slots.
   */
  constexpr int_t num_tombstone_slots() const noexcept { return self()->num_tombstone_slots_(); }

  /**
   * @return The current number of entries stored in the data structure.
   */
  constexpr int_t size() const noexcept { return capacity() - num_not_hash_slots(); }

  /**
   * Fetch the state state byte at `pos`.
   *
   * @param pos The position to query.
   * @return The associated state byte.
   */
  constexpr state_t state_at(const pos& pos) const { return self()->state_at_(validate_range(pos)); }

  constexpr friend bool operator==(const swiss_map_base& lhs, const swiss_map_base& rhs) {
    static_assert(!test_flags(flags, flags_t::duplicates), "Not implemented yet!");
    if (&lhs == &rhs) return true;

    if (lhs.size() != rhs.size()) return false;

    const int_t blob_size{lhs.conf_.blob_size()};
    if (blob_size != rhs.conf_.blob_size()) return false;

    const int_t l_blob_stride{lhs.conf_.blob_stride()};
    const value_type* const __restrict l_values{lhs.values_.get()};
    const blob_t* const __restrict l_blobs{lhs.blobs_.get()};
    const key_type* const __restrict l_keys{lhs.keys_.get()};
    
    const int_t r_blob_stride{rhs.conf_.blob_stride()};
    const value_type* const __restrict r_values{rhs.values_.get()};
    const blob_t* const __restrict r_blobs{rhs.blobs_.get()};
    const key_type* const __restrict r_keys{rhs.keys_.get()};

    auto pos{lhs.self()->find_if_([&](raw_pos_t l_pos) {
      const key_type& __restrict l_key{l_keys[l_pos]};

      const raw_pos_t r_pos{rhs.self()->find_first_(l_key, key_to_hash(l_key)).first};
      if (r_pos == npos) return true;

      if (l_key != r_keys[r_pos]) return true;
      if constexpr (has_values) {
        if (l_values[l_pos] != r_values[r_pos]) return true;
      }
      if constexpr (has_blobs) {
        if (!equal_n(
          &l_blobs[l_pos * l_blob_stride],
          &r_blobs[r_pos * r_blob_stride],
          blob_size
        )) return true;
      }

      return false;
    })};
    return pos == npos;
  }
  constexpr friend bool operator!=(const swiss_map_base& lhs, const swiss_map_base& rhs) { return !(lhs == rhs); }

  constexpr friend void swap(swiss_map_base& lhs, swiss_map_base& rhs) noexcept {
    if (&lhs == &rhs) return;
    swap(static_cast<base_type&>(lhs), static_cast<base_type&>(rhs));

    std::swap(lhs.keys_, rhs.keys_);
  }
  
 protected:
  using base_type::conf_;
  using base_type::bucket_mask_;
  using base_type::values_;
  using base_type::blobs_;
  unique_ptr_t<key_type[], allocator_type> keys_;

  using base_type::hint_to_hash_;

  constexpr raw_pos_t alloc_(raw_pos_t end) {
    end = base_type::alloc_(end);
    keys_ = make_unique<key_type, allocator_type>(end);
    return end;
  }

  constexpr const_entry_type at_(const raw_pos_t pos) const noexcept {
    NVHM_ASSERT_(contains_at_(pos), "pos = ", pos);

    const int_t blob_stride{conf_.blob_stride()};
    const value_type* const __restrict values{values_.get()};
    const blob_t* const __restrict blobs{blobs_.get()};
    const key_type* const __restrict keys{keys_.get()};

    if constexpr (has_values && has_blobs) {
      return {keys[pos], values[pos], &blobs[pos * blob_stride]};
    } else if constexpr (has_values) {
      return {keys[pos], values[pos]};
    } else if constexpr (has_blobs) {
      return {keys[pos], &blobs[pos * blob_stride]};
    } else {
      return keys[pos];
    }
  }
  constexpr entry_type at_(const raw_pos_t pos) noexcept {
    NVHM_ASSERT_(contains_at_(pos), "pos = ", pos);

    const int_t blob_stride{conf_.blob_stride()};
    value_type* const __restrict values{values_.get()};
    blob_t* const __restrict blobs{blobs_.get()};
    const key_type* const __restrict keys{keys_.get()};

    if constexpr (has_values && has_blobs) {
      return {keys[pos], values[pos], &blobs[pos * blob_stride]};
    } else if constexpr (has_values) {
      return {keys[pos], values[pos]};
    } else if constexpr (has_blobs) {
      return {keys[pos], &blobs[pos * blob_stride]};
    } else {
      return keys[pos];
    }
  }
  
  constexpr bool contains_at_(raw_pos_t pos) const noexcept {
    return is_hash(self()->state_at_(pos));
  }

  constexpr const key_type& key_at_(raw_pos_t pos) const noexcept {
    NVHM_ASSERT_(contains_at_(pos), "pos = ", pos);
    return keys_[to_uint(pos)];
  }

  constexpr void read_prefetch_(const hash_t hash) const noexcept {
    const bitmask_t bucket_mask{bucket_mask_};
    
    probe_seq_type seq{hash_to_pos<kernel_size>(hash)};
    NVHM_ASSERT_(seq.has_next(), "Probe sequence is incompatible!");

    const raw_pos_t off{align_pos(seq.next(), bucket_mask)};
    self()->read_prefetch_states_(off);

    if constexpr (test_flags(flags, flags_t::aggressive_prefetch)) {
      NVHM_ASSERT_(off >= 0 && off <= to_int(bucket_mask_), "off = ", off, ", bucket_mask = ", std::hex, bucket_mask_, std::dec);
      nvhm::read_prefetch<kernel_size>(&keys_[to_uint(off)]);
    }
  }
  
  constexpr void write_prefetch_(const hash_t hash) noexcept {
    const bitmask_t bucket_mask{bucket_mask_};

    probe_seq_type seq{hash_to_pos<kernel_size>(hash)};
    NVHM_ASSERT_(seq.has_next(), "Probe sequence is incompatible!");
    
    const raw_pos_t off{align_pos(seq.next(), bucket_mask)};
    self()->write_prefetch_states_(off);

    if constexpr (test_flags(flags, flags_t::aggressive_prefetch)) {
      NVHM_ASSERT_(off >= 0 && off <= to_int(bucket_mask_), "off = ", off, ", bucket_mask = ", std::hex, bucket_mask_, std::dec);
      nvhm::write_prefetch<kernel_size>(&keys_[to_uint(off)]);
    }
  }
};

} // namespace nvhm