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

#include <cassert>
#include <cstddef>
#include <cstdint>

namespace nvhm { namespace std_ext {

template <size_t N, typename T>
constexpr T* assume_aligned(T* p) noexcept {
  return p;
}

constexpr int countl_zero(uint32_t x) noexcept { return __builtin_clz(x); }
constexpr int countl_zero(uint64_t x) noexcept { return __builtin_clzl(x); }
constexpr int countl_zero(__uint128_t x) noexcept {
  const uint64_t x_lo{static_cast<uint64_t>(x)};
  const uint64_t x_hi{static_cast<uint64_t>(x >> 64)};
  return x_hi ? countl_zero(x_hi) : 64 + countl_zero(x_lo);
}

constexpr int countr_zero(uint32_t x) noexcept { return __builtin_ctz(x); }
constexpr int countr_zero(uint64_t x) noexcept { return __builtin_ctzl(x); }
constexpr int countr_zero(__uint128_t x) noexcept {
  const uint64_t x_lo{static_cast<uint64_t>(x)};
  const uint64_t x_hi{static_cast<uint64_t>(x >> 64)};
  return x_lo ? countr_zero(x_lo) : 64 + countr_zero(x_hi);
}

constexpr int popcount(uint32_t x) noexcept { return __builtin_popcount(x); }
constexpr int popcount(uint64_t x) noexcept { return __builtin_popcountl(x); }
constexpr int popcount(__uint128_t x) noexcept {
  const uint64_t x_lo{static_cast<uint64_t>(x)};
  const uint64_t x_hi{static_cast<uint64_t>(x >> 64)};
  return popcount(x_lo) + popcount(x_hi);
}

constexpr uint32_t rotl(uint32_t x, int n) noexcept {
  assert(n >= 0 && n < 32);
#if defined(__clang__)
  return __builtin_rotateleft32(x, static_cast<uint32_t>(n));
#else
  return (x << n) | (x >> (32 - n));
#endif
}
constexpr uint64_t rotl(uint64_t x, int n) noexcept {
  assert(n >= 0 && n < 64);
#if defined(__clang__)
  return __builtin_rotateleft64(x, static_cast<uint64_t>(n));
#else
  return (x << n) | (x >> (64 - n));
#endif
}

constexpr uint32_t rotr(uint32_t x, int n) noexcept {
  assert(n >= 0 && n < 32);
#if defined(__clang__)
  return __builtin_rotateright32(x, static_cast<uint32_t>(n));
#else
  return (x >> n) | (x << (32 - n));
#endif
}
constexpr uint64_t rotr(uint64_t x, int n) noexcept {
  assert(n >= 0 && n < 64);
#if defined(__clang__)
  return __builtin_rotateright64(x, static_cast<uint64_t>(n));
#else
  return (x >> n) | (x << (64 - n));
#endif
}

constexpr uint32_t bit_ceil(uint32_t x) noexcept {
  if (!x) x = 1;
  uint32_t y{UINT32_C(0x8000'0000)};
  y >>= countl_zero(x);
  y = rotl(y, y < x);
  return y;
}
constexpr uint64_t bit_ceil(uint64_t x) noexcept {
  if (!x) x = 1;
  uint64_t y{UINT64_C(0x8000'0000'0000'0000)};
  y >>= countl_zero(x);
  y = rotl(y, y < x);
  return y;
}

constexpr bool has_single_bit(uint32_t x) noexcept { return x && !(x & (x - 1)); }
constexpr bool has_single_bit(uint64_t x) noexcept { return x && !(x & (x - 1)); }

}}  // namespace nvhm::std_ext

#if defined(__cpp_lib_assume_aligned)
#include <memory>
#endif

#if defined(__cpp_lib_bitops) || defined(__cpp_lib_int_pow2)
#include <bit>
#endif

namespace nvhm {

#if defined(__cpp_lib_assume_aligned)
using std::assume_aligned;
#else
using std_ext::assume_aligned;
#endif

#if defined(__cpp_lib_bitops)
using std::countl_zero;
using std::countr_zero;
using std::popcount;
using std::rotl;
using std::rotr;
#else
using std_ext::countl_zero;
using std_ext::countr_zero;
using std_ext::popcount;
using std_ext::rotl;
using std_ext::rotr;
#endif

#if defined(__cpp_lib_int_pow2)
using std::bit_ceil;
using std::has_single_bit;
#else
using std_ext::bit_ceil;
using std_ext::has_single_bit;
#endif

}  // namespace nvhm