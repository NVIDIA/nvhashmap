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

namespace nvhm { namespace detail {

#if defined(__aarch64__)
template <bool store>
inline void arm64_rprfm(const std::byte* __restrict p, int_t n) noexcept {
  constexpr int_t min_block_size{1};
  constexpr int_t max_block_size{1 << 20};
  constexpr int_t max_block_count{(1 << 16) - 1};

  const int_t block_size{std::min(std::max(n, min_block_size), max_block_size)};
  const int_t block_count{std::min((n + block_size - 1) / block_size, max_block_count)};
  constexpr int_t block_stride{};
  constexpr int_t reuse_dist{};
  const int_t meta{(reuse_dist << 60) | (block_stride << 38) | (block_count << 22) | block_size};

  if constexpr (store) {
    asm volatile("rprfm   pststrm, %0, [%1]" : : "r"(meta), "r"(p) : "memory");
  } else {
    asm volatile("rprfm   pldstrm, %0, [%1]" : : "r"(meta), "r"(p) : "memory");
  }
}
#endif

}}  // namespace nvhm::detail

#include <cstring>

#if NVHM_WITH_SSE
#include <xmmintrin.h>
#endif

namespace nvhm {

namespace detail {

#if NVHM_WITH_SSE
#if (defined(__GNUC__) && (__GNUC__ < 15)) || defined(__clang__)
using mm_hint_t = int;
#else
using mm_hint_t = _mm_hint;
#endif
#endif

}  // namespace detail

template <typename T>
inline void read_prefetch(const T* p, int_t n) noexcept {
  read_prefetch(reinterpret_cast<const std::byte*>(p), n * num_bytes_v<T>);
}

/**
 * Issues load prefetch instructions for a block of memory.
 */
template <>
inline void read_prefetch<std::byte>(const std::byte* __restrict p, int_t n) noexcept {
#if defined(__aarch64__)
  if constexpr (use_range_prefetch) {
    detail::arm64_rprfm<false>(p, n);
    return;
  }
#endif

  for (int_t i{}; i < n; i += cache_line_size) {
#if NVHM_WITH_SSE
    if constexpr (use_sse_prefetch) {
      constexpr detail::mm_hint_t hint{[]() {
        if constexpr (prefetch_cache_level == 1) {
          return _MM_HINT_T0;
        } else if constexpr (prefetch_cache_level == 2) {
          return _MM_HINT_T1;
        } else if constexpr (prefetch_cache_level == 3) {
          return _MM_HINT_T2;
        } else {
          return _MM_HINT_NTA;
        }
      }()};
      _mm_prefetch(&p[i], hint);
      continue;
    }
#endif

    constexpr int hint{4 - std::min(prefetch_cache_level, 4)};
    static_assert(hint >= 0 && hint <= 3);
    __builtin_prefetch(&p[i], 0, hint);
  }
}

template <int_t N, typename T>
constexpr void read_prefetch(const T* p) noexcept { read_prefetch<T>(p, N); }

template <typename T>
inline void write_prefetch(T* p, int_t n) noexcept {
  write_prefetch(reinterpret_cast<std::byte*>(p), n * num_bytes_v<T>);
}

/**
 * Issues store prefetch instructions for a block of memory.
 */
template <>
inline void write_prefetch<std::byte>(std::byte* __restrict p, int_t n) noexcept {
#if defined(__aarch64__)
  if constexpr (use_range_prefetch) {
    detail::arm64_rprfm<true>(p, n);
    return;
  }
#endif

  for (int_t i{}; i < n; i += cache_line_size) {
    #if NVHM_WITH_SSE
    if constexpr (use_sse_prefetch) {
      #if defined(_MM_HINT_ET1)
      constexpr detail::mm_hint_t hint{prefetch_cache_level == 1 ? _MM_HINT_ET0 : _MM_HINT_ET1};
      #else
      constexpr detail::mm_hint_t hint{_MM_HINT_ET0};
      #endif
      _mm_prefetch(&p[i], hint);
      continue;
    }
    #endif

    constexpr int hint{4 - std::min(prefetch_cache_level, 4)};
    static_assert(hint >= 0 && hint <= 3);
    __builtin_prefetch(&p[i], 1, hint);
  }
}

template <int_t N, typename T>
constexpr void write_prefetch(T* p) noexcept { write_prefetch<T>(p, N); }

}  // namespace nvhm
