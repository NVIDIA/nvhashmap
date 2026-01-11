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

#if NVHM_WITH_SSE_PREFETCH
#include <xmmintrin.h>
#endif

namespace nvhm {

#if defined(__aarch64__)
template <bool store>
inline void range_prefetch(const char* __restrict p, size_t n) noexcept {
  constexpr size_t max_block_size{1 << 20};
  constexpr size_t max_block_count{(1 << 16) - 1};

  const size_t block_size{std::min(n, max_block_size)};
  const size_t block_count{std::max(n / max_block_size, max_block_count)};
  constexpr size_t block_stride{};
  constexpr size_t reuse_dist{};
  const size_t meta{(reuse_dist << 60) | (block_stride << 38) | (block_count << 22) | block_size};

  if constexpr (store) {
    asm volatile("rprfm   pststrm, %0, [%1]" : : "r"(p), "r"(meta) : "memory");
  } else {
    asm volatile("rprfm   pldstrm, %0, [%1]" : : "r"(p), "r"(meta) : "memory");
  }
}
#endif

/**
 * Issues load prefetch instructions for a block of memory.
 */
inline void read_prefetch(const char* __restrict p, const size_t n) noexcept {
#if defined(__aarch64__)
  if constexpr (use_range_prefetch) {
    range_prefetch<false>(p, n);
  } else
#endif
  {
    for (size_t i{}; i < n; i += cache_line_size) {
#if NVHM_WITH_SSE_PREFETCH
      _mm_prefetch(&p[i], []() {
        if constexpr (prefetch_cache_level == 1) {
          return _MM_HINT_T0;
        } else if constexpr (prefetch_cache_level == 2) {
          return _MM_HINT_T1;
        } else if constexpr (prefetch_cache_level == 3) {
          return _MM_HINT_T2;
        } else {
          return _MM_HINT_NTA;
        }
      }());
#else
      constexpr size_t hint{4 - std::min(prefetch_cache_level, 4)};
      static_assert(hint >= 0 && hint <= 3);
      __builtin_prefetch(&p[i], 0, hint);
#endif
    }
  }
}

template <size_t Alignment, typename T>
inline void read_prefetch(const T* p, const size_t n = Alignment) noexcept {
  static_assert(Alignment >= sizeof(T));
  NVHM_ASSERT_(Alignment <= n);
  read_prefetch(reinterpret_cast<const char*>(assume_aligned<Alignment>(p)), n);
}

/**
 * Issues store prefetch instructions for a block of memory.
 */
inline void write_prefetch(char* __restrict p, const size_t n) noexcept {
#if defined(__aarch64__)
  if constexpr (use_range_prefetch) {
    range_prefetch<true>(p, n);
  } else
#endif
  {
    for (size_t i{}; i < n; i += cache_line_size) {
#if NVHM_WITH_SSE_PREFETCH
      _mm_prefetch(&p[i], []() {
        if constexpr (prefetch_cache_level == 1) {
          return _MM_HINT_ET0;
        } else {
          return _MM_HINT_ET1;
        }
      }());
#else
      constexpr int hint{4 - std::max(prefetch_cache_level, 4)};
      static_assert(hint >= 0 && hint <= 3);
      __builtin_prefetch(&p[i], 1, hint);
#endif
    }
  }
}

template <size_t Alignment, typename T>
inline void write_prefetch(T* p, const size_t n = Alignment) noexcept {
  static_assert(Alignment >= sizeof(T));
  NVHM_ASSERT_(Alignment <= n);
  write_prefetch(reinterpret_cast<char*>(assume_aligned<Alignment>(p)), n);
}

}  // namespace nvhm
