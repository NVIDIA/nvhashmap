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


namespace nvhm {

namespace detail {
  
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

}  // namespace detail

}  // namespace nvhm

#include <cstring>

#if NVHM_WITH_SSE
#include <xmmintrin.h>
#endif

namespace nvhm {

namespace detail {

#if (defined(__GNUC__) && (__GNUC__ < 15)) || defined(__clang__)
using mm_hint_t = int;
#else
using mm_hint_t = _mm_hint;
#endif

}


template <typename T, int_t Alignment = num_bytes_v<T>>
inline void read_prefetch(const T* p, int_t n) noexcept {
  static_assert(Alignment % num_bytes_v<T> == 0);
  read_prefetch(reinterpret_cast<const std::byte*>(assume_aligned<Alignment>(p)), n * num_bytes_v<T>);
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

template <int_t N, typename T, int_t Alignment = num_bytes_v<T>>
constexpr void read_prefetch(const T* p) noexcept { read_prefetch<T, Alignment>(p, N); }

template <typename T, int_t Alignment = num_bytes_v<T>>
inline void write_prefetch(T* p, int_t n) noexcept {
  static_assert(Alignment % num_bytes_v<T> == 0);
  write_prefetch(reinterpret_cast<std::byte*>(assume_aligned<Alignment>(p)), n * num_bytes_v<T>);
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
      constexpr detail::mm_hint_t hint{prefetch_cache_level == 1 ? _MM_HINT_ET0 : _MM_HINT_ET1};
      _mm_prefetch(&p[i], hint);
      continue;
    }
#endif

    constexpr int hint{4 - std::min(prefetch_cache_level, 4)};
    static_assert(hint >= 0 && hint <= 3);
    __builtin_prefetch(&p[i], 1, hint);
  }
}

template <int_t N, typename T, int_t Alignment = num_bytes_v<T>>
constexpr void write_prefetch(T* p) noexcept { write_prefetch<T, Alignment>(p, N); }

}  // namespace nvhm

#if NVHM_WITH_SVE
#include <arm_sve.h>
#endif

namespace nvhm {

namespace detail {
/**
 * Lightweight memcpy for blobs.
 */
inline void copy_blob(int8_t* __restrict dst, const int8_t* __restrict src, int_t n) noexcept {
#if NVHM_WITH_SVE
  // SVE fast-path considered but disabled: only outperforms memcpy for very small copies,
  // and typical blob strides are aligned, so memcpy wins in the common case.
  if constexpr (false && use_sve_copy) {
    int_t i{};
    do {
      svbool_t p{svwhilelt_b8(i, n)};
      svst1_s8(p, &dst[i], svld1_s8(p, &src[i]));
      i += svcntb();
      // `while (svptest_last(svptrue_b8(), p))` would save one instruction per iteration,
      // but runs through the loop again if the `n` is aligned with the SVE register
      // size. Unfortunately, most times, the blob array is likely aligned.
    } while (i < n);
    return;
  }
#endif

  std::memcpy(dst, src, to_uint(n));
}
}  // namespace detail

inline void copy_blob(std::byte* dst, const void* src, int_t n) noexcept {
  return detail::copy_blob(reinterpret_cast<int8_t*>(dst), reinterpret_cast<const int8_t*>(src), n);
}

inline void copy_blob(void* dst, const std::byte* src, int_t n) noexcept {
  return detail::copy_blob(static_cast<int8_t*>(dst), reinterpret_cast<const int8_t*>(src), n);
}

inline void copy_blob(std::byte* dst, const std::byte* src, int_t n) noexcept {
  return detail::copy_blob(reinterpret_cast<int8_t*>(dst), reinterpret_cast<const int8_t*>(src), n);
}

}  // namespace nvhm
