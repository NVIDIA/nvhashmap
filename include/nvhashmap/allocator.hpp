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

#include <ratio>
#include <memory>
#include <stdexcept>
#include <sys/mman.h>

namespace nvhm {

template<typename T, typename Allocator>
using deleter_t = typename Allocator::template deleter_type<T>;

template<typename T, typename Allocator>
using unique_ptr_t = std::unique_ptr<T, deleter_t<T, Allocator>>;

template <typename T, typename Deleter>
constexpr std::unique_ptr<T, Deleter> make_copy(const std::unique_ptr<T, Deleter>& src) {
  using allocator_t = typename Deleter::allocator_type;
  auto [p, d]{allocator_t::template alloc_<T>(1)};
  std::uninitialized_copy_n(src.get(), 1, p);
  return {p, std::move(d)};
}

template <typename T, typename Deleter>
constexpr std::unique_ptr<T[], Deleter> make_copy(const std::unique_ptr<T[], Deleter>& src, int_t n) {
  using allocator_t = typename Deleter::allocator_type;
  auto [p, d]{allocator_t::template alloc_<T>(n)};
  std::uninitialized_copy_n(src.get(), n, p);
  return {p, std::move(d)};
}

template <typename T, typename Allocator>
constexpr std::shared_ptr<T> make_shared() {
  using allocator_t = Allocator;
  auto [p, d]{allocator_t::template alloc_<T>(1)};
  if constexpr (!std::is_trivially_default_constructible_v<T>) {
    std::uninitialized_default_construct_n(p, 1);
  }
  return {p, std::move(d)};
}

template <typename T, typename Allocator>
constexpr std::shared_ptr<T[]> make_shared(int_t n) {
  using allocator_t = Allocator;
  auto [p, d]{allocator_t::template alloc_<T>(n)};
  if constexpr (!std::is_trivially_default_constructible_v<T>) {
    std::uninitialized_default_construct_n(p, n);
  }
  return {p, std::move(d)};
}

template <typename T, typename Allocator>
constexpr std::unique_ptr<T, deleter_t<T, Allocator>> make_unique() {
  using allocator_t = Allocator;
  auto [p, d]{allocator_t::template alloc_<T>(1)};
  if constexpr (!std::is_trivially_default_constructible_v<T>) {
    std::uninitialized_default_construct_n(p, 1);
  }
  return {p, std::move(d)};
}

template <typename T, typename Allocator>
constexpr std::unique_ptr<T[], deleter_t<T, Allocator>> make_unique(int_t n) {
  using allocator_t = Allocator;
  auto [p, d]{allocator_t::template alloc_<T>(n)};
  if constexpr (!std::is_trivially_default_constructible_v<T>) {
    std::uninitialized_default_construct_n(p, n);
  }
  return {p, std::move(d)};
}

template <typename T, bool PageAlign, bool AllowHugePages, typename SwitchThreshold>
constexpr std::pair<int_t, int_t> alloc_size(int_t n) noexcept {
  const bool page_align = PageAlign;
  const bool allow_hugepages = AllowHugePages;
  using switch_threshold_t = SwitchThreshold;

  n *= num_bytes_v<T>;

  if constexpr (page_align) {
    if constexpr (allow_hugepages) {
      if (n >= switch_threshold_t::num * hugepage_size / switch_threshold_t::den) {
        return {hugepage_size, round_up<hugepage_size>(n)};
      }
    }

    if (n >= switch_threshold_t::num * page_size / switch_threshold_t::den) {
      return {page_size, round_up<page_size>(n)};
    }
  }

  return {cache_line_size, round_up<cache_line_size>(n)};
}

/**
 * Base class for all deleters.
 */
template <typename T>
class deleter_base {
 public:
  constexpr deleter_base() noexcept : n_{} {}
  constexpr deleter_base(int_t n) noexcept : n_{n} {
    NVHM_ASSERT_(n >= 0);
  }

  constexpr void operator()(T* p) const {
    if constexpr (!std::is_trivially_destructible_v<T>) {
      std::destroy_n(p, n_);
    }
  }

 protected:
  int_t n_;
};

template <typename Self>
class allocator_base {
 public:
  using self_type = Self;

  NVHM_MAKE_NOT_INSTANTIABLE_(allocator_base);
};

using default_switch_threshold = std::ratio<2, 3>;

template <bool PageAlign = page_align_by_default, bool UseHugePages = use_hugepages_by_default, typename SwitchThreshold = default_switch_threshold>
class std_allocator : public allocator_base<std_allocator<PageAlign, UseHugePages, SwitchThreshold>> {
 public:
  using base_type = allocator_base<std_allocator<PageAlign, UseHugePages, SwitchThreshold>>;
  constexpr static bool page_align{PageAlign};
  constexpr static bool use_hugepages{UseHugePages};
  using switch_threshold_type = SwitchThreshold;

  template <typename T>
  class deleter : public deleter_base<T> {
   public:
    using base_type = deleter_base<T>;
    using allocator_type = std_allocator;

    constexpr deleter() = default;
    constexpr deleter(int_t n) noexcept : base_type{n} {}
  
    constexpr void operator()(T* p) const {
      base_type::operator()(p);
      std::free(p);
    }
  };

  template <typename T>
  using deleter_type = deleter<std::conditional_t<std::is_array_v<T>, std::remove_extent_t<T>, T>>;

  template <typename T>
  using unique_ptr_type = std::unique_ptr<T, deleter_type<T>>;

  NVHM_MAKE_NOT_INSTANTIABLE_(std_allocator);

 protected:
  template <typename T>
  constexpr static std::pair<T*, deleter_type<T>> alloc_(const int_t n) {
    if (n <= 0) return {nullptr, {}};
    const auto [align_size, n_bytes]{alloc_size<T, page_align, use_hugepages, switch_threshold_type>(n)};

    T* p{static_cast<T*>(std::aligned_alloc(to_uint(align_size), to_uint(n_bytes)))};
    if (NVHM_UNLIKELY_(!p)) throw std::bad_alloc();
    NVHM_ASSERT_((reinterpret_cast<uintptr_t>(p) & make_size_mask(align_size)) == 0);

    if constexpr (page_align && use_hugepages) {
      if (align_size >= hugepage_size) {
        if (NVHM_UNLIKELY_(madvise(p, to_uint(n_bytes), MADV_HUGEPAGE) != 0)) {
          throw std::runtime_error("madvise hugepage failure! Check errno for details.");
        }
      }
    }

    return {p, n};
  }

  template <typename T, typename Deleter>
  constexpr friend std::unique_ptr<T, Deleter> make_copy(const std::unique_ptr<T, Deleter>&);
  template <typename T, typename Deleter>
  constexpr friend std::unique_ptr<T[], Deleter> make_copy(const std::unique_ptr<T[], Deleter>&, int_t);
  template <typename T, typename>
  constexpr friend std::shared_ptr<T> make_shared();
  template <typename T, typename>
  constexpr friend std::shared_ptr<T[]> make_shared(int_t);
  template <typename T, typename Allocator>
  constexpr friend std::unique_ptr<T, deleter_t<T, Allocator>> make_unique();
  template <typename T, typename Allocator>
  constexpr friend std::unique_ptr<T[], deleter_t<T, Allocator>> make_unique(int_t);
};

template <bool UseHugePages = use_hugepages_by_default, typename SwitchThreshold = default_switch_threshold>
class mmap_allocator : public allocator_base<mmap_allocator<UseHugePages, SwitchThreshold>> {
 public:
  using base_type = allocator_base<mmap_allocator<UseHugePages, SwitchThreshold>>;
  constexpr static bool use_hugepages{UseHugePages};
  using switch_threshold_type = SwitchThreshold;

  template <typename T>
  class deleter : public deleter_base<T> {
   public:
    using base_type = deleter_base<T>;
    using allocator_type = mmap_allocator;
   
    constexpr deleter() noexcept : base_type{}, n_bytes_{} {}
    constexpr deleter(int_t n, int_t n_bytes) noexcept : base_type{n}, n_bytes_{n_bytes} {
      NVHM_ASSERT_(n_bytes >= n * num_bytes_v<T>);
    }
  
    constexpr void operator()(T* p) const {
      base_type::operator()(p);
      if (NVHM_UNLIKELY_(munmap(p, n_bytes_) != 0)) {
        NVHM_LOG_(log_level_t::error, "munmap failed, errno = ", errno);
      }
    }
  
   protected:
    int_t n_bytes_;
  };
  
  template <typename T>
  using deleter_type = deleter<std::conditional_t<std::is_array_v<T>, std::remove_extent_t<T>, T>>;

  template <typename T>
  using unique_ptr_type = std::unique_ptr<T, deleter_type<T>>;

  NVHM_MAKE_NOT_INSTANTIABLE_(mmap_allocator);

 protected:
  template <typename T>
  constexpr static std::pair<T*, deleter_type<T>> alloc_(const int_t n) {
    if (n <= 0) return {nullptr, {}};
    const auto [align_size, n_bytes]{alloc_size<T, true, use_hugepages, switch_threshold_type>(n)};

    int flags{MAP_PRIVATE | MAP_ANONYMOUS};
    if constexpr (use_hugepages) {
      if (align_size >= hugepage_size) {  
        flags |= MAP_HUGETLB;
        flags |= countr_zero(hugepage_size) << MAP_HUGE_SHIFT;
      }
    }

    T* p{static_cast<T*>(mmap(nullptr, n_bytes, PROT_READ | PROT_WRITE, flags, -1, 0))};
    if (NVHM_UNLIKELY_(p == MAP_FAILED)) throw std::bad_alloc();
    NVHM_ASSERT_((reinterpret_cast<uintptr_t>(p) & make_size_mask(align_size)) == 0);

    return {p, {n, n_bytes}};
  }

  template <typename T, typename Deleter>
  constexpr friend std::unique_ptr<T, Deleter> make_copy(const std::unique_ptr<T, Deleter>&);
  template <typename T, typename Deleter>
  constexpr friend std::unique_ptr<T[], Deleter> make_copy(const std::unique_ptr<T[], Deleter>&, int_t);
  template <typename T, typename>
  constexpr friend std::shared_ptr<T> make_shared();
  template <typename T, typename>
  constexpr friend std::shared_ptr<T[]> make_shared(int_t);
  template <typename T, typename Allocator>
  constexpr friend std::unique_ptr<T, deleter_t<T, Allocator>> make_unique();
  template <typename T, typename Allocator>
  constexpr friend std::unique_ptr<T[], deleter_t<T, Allocator>> make_unique(int_t);
};

using default_allocator_t = std_allocator<>;

}  // namespace nvhm