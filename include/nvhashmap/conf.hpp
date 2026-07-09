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
#include "kernel.hpp"

namespace nvhm {


template<typename Key, typename Value = void, flags_t Flags = flags_t::none, int_t KernelSize = default_kernel_size>
class conf : public self_aware<conf<Key, Value, Flags, KernelSize>> {
 public:
  using base_type = self_aware<conf<Key, Value, Flags, KernelSize>>;
  using self_type = typename base_type::self_type;

  using key_type = Key;

  using value_type = value_t<Value>;
  constexpr static bool has_values{is_value_v<value_type>};
  constexpr static flags_t flags{Flags};
  constexpr static bool has_blobs{test_flags(flags, flags_t::blobs)};
  constexpr static int_t kernel_size{KernelSize};

  // Reserve bias for growth (relative number of empty slots).
  constexpr static int_t min_grow_bias{2}; // 50%
  constexpr static int_t max_grow_bias{cache_line_size};  // 1.5625% (if `cache_line_size` is 64)
  static_assert(min_grow_bias <= max_grow_bias);
  
  constexpr static int_t default_grow_bias{8}; // 12.5%
  static_assert(default_grow_bias >= min_grow_bias && default_grow_bias <= max_grow_bias);

  // Reserve bias for scrubbing (relative number of tombstone slots).
  constexpr static int_t min_scrub_bias{2};      // 50%
  constexpr static int_t max_scrub_bias{cache_line_size};  // 98.4375% (if `cache_line_size` is 64)
  static_assert(min_scrub_bias <= max_scrub_bias);
  
  constexpr static int_t default_scrub_bias{4};  // 25%
  static_assert(default_scrub_bias >= min_scrub_bias && default_scrub_bias <= max_scrub_bias);

  // Reserve bias for shrink (relative number of non-hash slots).
  constexpr static int_t min_shrink_bias{3};      // 33% before shrink, 66% after shrink.
  constexpr static int_t max_shrink_bias{cache_line_size};  // 1.5625% before shrink, 3.125% after shrink (if `cache_line_size` is 64).
  static_assert(min_shrink_bias <= max_shrink_bias);
  
  constexpr static int_t default_shrink_bias{4};  // 25% before shrink, 50 after shrink.
  static_assert(default_shrink_bias >= min_shrink_bias && default_shrink_bias <= max_shrink_bias);
  
  constexpr static int_t min_capacity_limit(int_t grow_bias) noexcept {
    // Not much point in having a table that is smaller than the size of a cache line.
    const int_t n{std::max(bit_ceil(grow_bias), kernel_size)};
    NVHM_ASSERT_(n > 0 && has_single_bit(n));
    return n;
  }

  constexpr static int_t default_capacity(int_t grow_bias) noexcept {
    int_t n{std::max(num_bytes_v<key_type>, num_bytes_v<value_type>)};
    n = std::min(n, page_size);
    n = bit_ceil(page_size / n);
    n = std::max(n, min_capacity_limit(grow_bias));
    NVHM_ASSERT_(n > 0 && has_single_bit(n));
    return n;
  }

  constexpr int_t min_capacity() const noexcept { return min_capacity_; }
  constexpr int_t init_capacity() const noexcept { return init_capacity_; }

  /**
   * @return Number of objects of `blob_t` in each blob.
   */
  constexpr int_t blob_size() const noexcept { return blob_size_; }

  /**
   * @return Stride between blobs in memory.
   */
  constexpr int_t blob_stride() const noexcept { return blob_stride_; }

  constexpr int_t scrub_bias() const noexcept { return scrub_bias_; }
  constexpr int_t grow_bias() const noexcept { return grow_bias_; }
  constexpr int_t shrink_bias() const noexcept { return shrink_bias_; }

  constexpr conf& set_blob(int_t size = 0, int_t stride = 0) {
    static_assert(has_blobs, "`set_blob` is not supported for this configuration!");

    // Prefer raw values to be cache-line-aligned.
    if (size <= 0) {
      if (stride != 0) {
        throw std::out_of_range("If blob `size` is zero, blob `stride` must also be zero!");
      }
      size = 0;
    } else if (stride <= 0) {
      if (size < cache_line_size) {
        stride = bit_ceil(size);
      } else {
        stride = round_up<cache_line_size / 2>(size);
      }
    } else if (stride < size) {
      throw std::out_of_range("Blob `stride` must be greater than or equal to `size`!");
    }
    blob_size_ = size;
    blob_stride_ = stride;
    return *this;
  }

  constexpr conf& set_capacity(int_t init_capacity) {
    return set_capacity(init_capacity, min_capacity_limit(grow_bias_));
  }

  constexpr conf& set_capacity(int_t init_capacity, int_t min_capacity) {
    min_capacity = bit_ceil(min_capacity);
    min_capacity_ = min_capacity;
    init_capacity_ = std::max(bit_ceil(init_capacity), min_capacity);
    return *this;
  }

  constexpr conf& set_grow_bias(int_t grow_bias) {
    if (grow_bias < min_grow_bias || grow_bias > max_grow_bias) {
      throw std::out_of_range("`grow_bias` bias is out of bounds!");
    }
    grow_bias_ = grow_bias;
    return *this;
  }

  constexpr conf& set_scrub_bias(int_t scrub_bias) {
    if (scrub_bias < min_scrub_bias || scrub_bias > max_scrub_bias) {
      throw std::out_of_range("`scrub_bias` bias is out of bounds!");
    }
    scrub_bias_ = scrub_bias;
    return *this;
  }

  constexpr conf& set_shrink_bias(int_t shrink_bias) {
    if (shrink_bias < min_shrink_bias || shrink_bias > max_shrink_bias) {
      throw std::out_of_range("`shrink_bias` bias is out of bounds!");
    }
    shrink_bias_ = shrink_bias;
    return *this;
  }

  constexpr bool is_valid() const noexcept {
    return min_capacity_ >= min_capacity_limit(grow_bias_);
  }

  constexpr conf& auto_adjust() {
    const int_t min_cap_limit{min_capacity_limit(grow_bias_)};
    int_t min_capacity{min_capacity_};
    int_t init_capacity{init_capacity_};

    if (min_capacity < min_cap_limit) {
      min_capacity = min_cap_limit;
    }
    init_capacity = std::max(init_capacity, min_capacity);

    min_capacity_ = min_capacity;
    init_capacity_ = init_capacity;
    return *this;
  }

  constexpr bool should_grow(int_t capacity, int_t num_empty) const noexcept {
    return num_empty <= grow_threshold(capacity);
  }

  constexpr bool should_grow_precalc(int_t num_empty, int_t grow_threshold) const noexcept {
    NVHM_ASSERT_(grow_threshold > 0);
    return num_empty <= grow_threshold;
  }

  constexpr bool should_scrub(int_t capacity, int_t num_tombstone) const noexcept {
    return capacity <= num_tombstone * scrub_bias_;
  }

  constexpr bool should_shrink(int_t capacity, int_t num_empty, int_t num_tombstone) const noexcept {
    const int_t num_not_hash{num_empty + num_tombstone};
    const int_t size{capacity - num_not_hash};
    return capacity > min_capacity_ && size * shrink_bias_ <= capacity;
  }
  
  constexpr int_t grow_threshold(int_t capacity) const noexcept {
    const int_t grow_bias{grow_bias_};
    capacity = std::max(capacity, min_capacity_);
    NVHM_ASSERT_(capacity >= grow_bias);
    return capacity / grow_bias;
  }

  constexpr double max_load_factor() const noexcept {
    int_t min_cap{min_capacity_};
    int_t grow_thres{grow_threshold(min_cap)};
    return 1.0 - (static_cast<double>(grow_thres) / static_cast<double>(min_cap));
  }

  constexpr friend bool operator==(const conf& lhs, const conf& rhs) noexcept {
    if (&lhs == &rhs) return true;

    return lhs.min_capacity_ == rhs.min_capacity_ &&
           lhs.init_capacity_ == rhs.init_capacity_ &&
           lhs.blob_size_ == rhs.blob_size_ &&
           lhs.blob_stride_ == rhs.blob_stride_ &&
           lhs.scrub_bias_ == rhs.scrub_bias_ &&
           lhs.grow_bias_ == rhs.grow_bias_ &&
           lhs.shrink_bias_ == rhs.shrink_bias_;
  }
  constexpr friend bool operator!=(const conf& lhs, const conf& rhs) noexcept { return !(lhs == rhs); }

  constexpr friend void swap(conf& lhs, conf& rhs) noexcept {
    if (&lhs == &rhs) return;
    swap(static_cast<base_type&>(lhs), static_cast<base_type&>(rhs));

    std::swap(lhs.min_capacity_, rhs.min_capacity_);
    std::swap(lhs.init_capacity_, rhs.init_capacity_);
    std::swap(lhs.blob_size_, rhs.blob_size_);
    std::swap(lhs.blob_stride_, rhs.blob_stride_);
    std::swap(lhs.scrub_bias_, rhs.scrub_bias_);
    std::swap(lhs.grow_bias_, rhs.grow_bias_);
    std::swap(lhs.shrink_bias_, rhs.shrink_bias_);
  }

 protected:
  int_t min_capacity_{min_capacity_limit(default_grow_bias)};
  int_t init_capacity_{default_capacity(default_grow_bias)};

  int_t blob_size_{};
  int_t blob_stride_{};

  int_t scrub_bias_{default_scrub_bias};
  int_t grow_bias_{default_grow_bias};
  int_t shrink_bias_{default_shrink_bias};
};

} // namespace nvhm