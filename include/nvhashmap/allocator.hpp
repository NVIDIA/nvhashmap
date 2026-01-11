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
#include "debug.hpp"

#include <memory>
#include <stdexcept>
#include <sys/mman.h>

namespace nvhm {

static constexpr bool use_explicit_hugepages{true};

struct default_deleter {
  void operator()(void* const p) const noexcept { std::free(p); }
};

template <typename T>
using aligned_unique_ptr = std::unique_ptr<T, std::function<void(void*)>>;

/**
 * Allocator we use buffers.
 */
template <
  size_t Alignment = cache_line_size, bool AllowHugePages = use_hugepages,
  bool PreferTransparentHugePages = use_transparent_hugepages>
struct default_allocator {
  default_allocator() = delete;

  constexpr static size_t alignment{Alignment};
  static_assert(has_single_bit(alignment) && alignment >= cache_line_size);
  constexpr static size_t alignment_mask{alignment - 1};

  constexpr static bool allow_hugepages{AllowHugePages};
  constexpr static bool prefer_transparent_hugepages{PreferTransparentHugePages};

  template <typename T>
  inline static aligned_unique_ptr<T[]> alloc(size_t n) {
    n *= sizeof(T);
    n = (n + alignment_mask) & ~alignment_mask;

    if constexpr (allow_hugepages) {
      if (n >= hugepage_size / 4 * 3) {
        n = (n + hugepage_size) & ~hugepage_size;

        if constexpr (prefer_transparent_hugepages) {
          void* const p{std::aligned_alloc(hugepage_size, n)};
          if (!p) throw std::bad_alloc();
          NVHM_ASSERT_((reinterpret_cast<uintptr_t>(p) & alignment_mask) == 0);

          if (madvise(p, n, MADV_HUGEPAGE) != 0) {
            throw std::runtime_error("memadvise failure. Check errno for details.");
          }

          return aligned_unique_ptr<T[]>{static_cast<T*>(p), std::free};
        } else {
          int flags{MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB};
          flags |= countr_zero(hugepage_size) << MAP_HUGE_SHIFT;

          void* const p{mmap(nullptr, n, PROT_READ | PROT_WRITE, flags, -1, 0)};
          if (p == MAP_FAILED) throw std::bad_alloc();

          return aligned_unique_ptr<T[]>{static_cast<T*>(p), [n](void* p) {
            if (munmap(p, n) != 0) {
              throw std::runtime_error("munmap failed. Check errno for details.");
            }
          }};
        }
      }
    }

    n = (n + alignment_mask) & ~alignment_mask;
    void* const p{std::aligned_alloc(hugepage_size, n)};
    if (!p) throw std::bad_alloc();
    NVHM_ASSERT_((reinterpret_cast<uintptr_t>(p) & alignment_mask) == 0);

    return aligned_unique_ptr<T[]>{static_cast<T*>(p), std::free};
  }
};

}  // namespace nvhm