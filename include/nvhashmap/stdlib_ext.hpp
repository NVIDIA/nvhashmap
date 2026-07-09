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

#include <sys/types.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#if defined(__cpp_lib_assume_aligned)
#include <memory>
#endif

#if defined(__cpp_lib_bitops) || defined(__cpp_lib_int_pow2)
#include <bit>
#endif

#if defined(__BMI__) || defined(__POPCNT__) || defined(__LZCNT__)
#include <immintrin.h>
#endif


namespace nvhm { 

template<typename T>
constexpr bool is_unsigned_v{std::is_unsigned_v<T> || std::is_same_v<T, __uint128_t>};

#if defined(__cpp_lib_remove_cvref)
using std::remove_cvref;
using std::remove_cvref_t;
#else
template <typename T>
struct remove_cvref {
  using type = std::remove_cv_t<std::remove_reference_t<T>>;
};
template <typename T>
using remove_cvref_t = typename remove_cvref<T>::type;
#endif


template <typename T, typename U>
constexpr T broadcast(const U x) noexcept {
  static_assert(is_unsigned_v<T> && is_unsigned_v<U>);
  static_assert(sizeof(T) >= sizeof(U) && sizeof(T) % sizeof(U) == 0);

  T y{x};
  for (size_t i{sizeof(U)}; i < sizeof(T); i *= 2) {
    y |= y << (i * 8);
  }
  return y;
}

constexpr uint64_t low_bits(__uint128_t x) noexcept { return static_cast<uint64_t>(x); }
constexpr uint64_t high_bits(__uint128_t x) noexcept { return static_cast<uint64_t>(x >> 64); }

namespace std_ext {

template <size_t N, typename T>
constexpr T* assume_aligned(T* p) noexcept {
  static_assert(N >= alignof(T) && N % alignof(T) == 0);

#if defined(__GNUC__) || defined(__clang__)
#if __has_builtin(__builtin_assume_aligned)
  return static_cast<T*>(__builtin_assume_aligned(p, N));
#endif
#endif

  return p;
}

template <class T, class... Args>
constexpr T* construct_at(T* ptr, Args&&... args) {
  if constexpr (std::is_array_v<T>) {
    return ::new (static_cast<void*>(ptr)) T[1]();
  } else {
    return ::new (static_cast<void*>(ptr)) T(std::forward<Args>(args)...);
  }
}

template<typename T>
constexpr int popcount_fallback(T x) noexcept {
  static_assert(is_unsigned_v<T>);
  static_assert(sizeof(T) <= 16, "Top byte summing trick only works for types up to 16 bytes!");

  constexpr T b01{broadcast<T, uint8_t>(0x55)};
  constexpr T b0011{broadcast<T, uint8_t>(0x33)};
  constexpr T b00001111{broadcast<T, uint8_t>(0x0f)};

  x -= (x >> 1) & b01;  // 2-bit sums
  x = (x & b0011) + ((x >> 2) & b0011);  // 4-bit sums
  x = (x + (x >> 4)) & b00001111;  // 8-bit sums

  // Sum up in top byte.
  constexpr int n{sizeof(T) * 8};
  x *= broadcast<T, uint8_t>(0x01);
  x >>= (n - 8);
  return static_cast<int>(x);
}

template <typename T>
constexpr int popcount(T x) noexcept {
  static_assert(is_unsigned_v<T>);

#if defined(__POPCNT__)
  if constexpr (sizeof(T) <= sizeof(uint32_t)) {
    return _mm_popcnt_u32(x);
  }
  if constexpr (sizeof(T) <= sizeof(uint64_t)) {
    return _mm_popcnt_u64(x);
  }
  if constexpr (sizeof(T) <= sizeof(__uint128_t)) {
    uint64_t lo{low_bits(x)};
    uint64_t hi{high_bits(x)};
    return _mm_popcnt_u64(lo) + _mm_popcnt_u64(hi);
  }

#elif defined(__GNUC__) || defined(__clang__)
#if __has_builtin(__builtin_popcountg)
  return __builtin_popcountg(x);
#endif

#if __has_builtin(__builtin_popcount)
  if constexpr (sizeof(T) <= sizeof(uint32_t)) {
    return __builtin_popcount(x);
  }
#endif

#if __has_builtin(__builtin_popcountll)
  if constexpr (sizeof(T) <= sizeof(uint64_t)) {
    return __builtin_popcountll(x);
  }
  if constexpr (sizeof(T) <= sizeof(__uint128_t)) {
    uint64_t lo{low_bits(x)};
    uint64_t hi{high_bits(x)};
    return __builtin_popcountll(lo) + __builtin_popcountll(hi);
  }
#endif

#endif

  return popcount_fallback(x);
}

template <typename T>
constexpr int countl_zero_fallback(T x) noexcept {
  static_assert(is_unsigned_v<T>);

  // Smear highest bit down.
  constexpr int n{sizeof(T) * 8};
  for (int i{1}; i < n; i *= 2) {
    x |= x >> i;
  }
  return n - popcount_fallback(x);
}

template <typename T>
constexpr int countl_zero(T x) noexcept {
  static_assert(is_unsigned_v<T>);
  constexpr int n{sizeof(T) * 8};

#if defined(__LZCNT__)
  if constexpr (sizeof(T) <= sizeof(uint32_t)) {
    return _lzcnt_u32(x) - (sizeof(uint32_t) * 8 - n);
  }
  if constexpr (sizeof(T) <= sizeof(uint64_t)) {
    return _lzcnt_u64(x);
  }
  if constexpr (sizeof(T) <= sizeof(__uint128_t)) {
    uint64_t lo{low_bits(x)};
    uint64_t hi{high_bits(x)};
    return hi ? _lzcnt_u64(hi) : 64 + _lzcnt_u64(lo);
  }

#elif defined(__GNUC__) || defined(__clang__)

#if __has_builtin(__builtin_clzg)
  return __builtin_clzg(x, n);
#endif

#if __has_builtin(__builtin_clz)
  if constexpr (sizeof(T) <= sizeof(uint32_t)) {
    constexpr int n32{sizeof(uint32_t) * 8};
    return x ? __builtin_clz(x) - (n32 - n) : n;
  }
#endif

#if __has_builtin(__builtin_clzll)  
  if constexpr (sizeof(T) <= sizeof(uint64_t)) {
    return x ? __builtin_clzll(x) : n;
  }
  if constexpr (sizeof(T) <= sizeof(__uint128_t)) {
    uint64_t lo{low_bits(x)};
    uint64_t hi{high_bits(x)};
    return hi ? __builtin_clzll(hi) : 64 + countl_zero(lo);
  }
#endif
#endif

  return countl_zero_fallback(x);
}

template <typename T>
constexpr int countr_zero_fallback(T x) noexcept {
  static_assert(is_unsigned_v<T>);

  int z{x != 0};
  x = z ? x ^ (x - 1) : broadcast<T, uint8_t>(0xff);
  return popcount_fallback(x) - z;
}

template <typename T>
constexpr int countr_zero(T x) noexcept {
  static_assert(is_unsigned_v<T>);
  constexpr int n{sizeof(T) * 8};

#if defined(__BMI__)
  if constexpr (sizeof(T) <= sizeof(uint16_t)) {
    return _tzcnt_u32(x | (UINT32_C(1) << n));
  }
  if constexpr (sizeof(T) <= sizeof(uint32_t)) {
    return _tzcnt_u32(x);
  }
  if constexpr (sizeof(T) <= sizeof(uint64_t)) {
    return _tzcnt_u64(x);
  }
  if constexpr (sizeof(T) <= sizeof(__uint128_t)) {
    uint64_t lo{low_bits(x)};
    uint64_t hi{high_bits(x)};
    return lo ? _tzcnt_u64(lo) : 64 + _tzcnt_u64(hi);
  }

#elif defined(__GNUC__) || defined(__clang__)
#if __has_builtin(__builtin_ctzg)
  return __builtin_ctzg(x, n);
#endif

#if __has_builtin(__builtin_ctz)
  if constexpr (sizeof(T) <= sizeof(uint16_t)) {
    return __builtin_ctz(x | (UINT32_C(1) << n));
  }
#endif

#if __has_builtin(__builtin_ctzll)
  if constexpr (sizeof(T) <= sizeof(uint32_t)) {
    return __builtin_ctzll(x | (UINT64_C(1) << n));
  }
  if constexpr (sizeof(T) <= sizeof(uint64_t)) {
    return x ? __builtin_ctzll(x) : n;
  }
  if constexpr (sizeof(T) <= sizeof(__uint128_t)) {
    uint64_t lo{low_bits(x)};
    uint64_t hi{high_bits(x)};
    return lo ? __builtin_ctzll(lo) : 64 + countr_zero(hi);
  }
#endif
#endif

  return countr_zero_fallback(x);
}

template <typename T>
constexpr T rotl_fallback(T x, int s) noexcept {
  static_assert(is_unsigned_v<T>);

  constexpr int n{sizeof(T) * 8};
  const auto y{(x << (s & (n - 1))) | (x >> (-s & (n - 1)))};

  if constexpr (sizeof(T) < sizeof(uint32_t)) {
    return static_cast<T>(y);
  } else {
    return y;
  }
}

template <typename T>
constexpr T rotl(T x, int s) noexcept {
  static_assert(is_unsigned_v<T>);

#if defined(__GNUC__) || defined(__clang__)
#if __has_builtin(__builtin_rotateleft8)
  if constexpr (sizeof(T) == sizeof(uint8_t)) {
    return __builtin_rotateleft8(x, static_cast<uint8_t>(s));
  }
#endif

#if __has_builtin(__builtin_rotateleft16)
  if constexpr (sizeof(T) == sizeof(uint16_t)) {
    return __builtin_rotateleft16(x, static_cast<uint16_t>(s));
  }
#endif

#if __has_builtin(__builtin_rotateleft32)
  if constexpr (sizeof(T) == sizeof(uint32_t)) {
    return __builtin_rotateleft32(x, static_cast<uint32_t>(s));
  }
#endif

#if __has_builtin(__builtin_rotateleft64)
  if constexpr (sizeof(T) == sizeof(uint64_t)) {
    return __builtin_rotateleft64(x, static_cast<uint64_t>(s));
  }
#endif
#endif

  return rotl_fallback(x, s);
}

template <typename T>
constexpr T rotr_fallback(T x, int s) noexcept {
  static_assert(is_unsigned_v<T>);

  constexpr int n{sizeof(T) * 8};
  const auto y{(x >> (s & (n - 1))) | (x << (-s & (n - 1)))};

  if constexpr (sizeof(T) < sizeof(uint32_t)) {
    return static_cast<T>(y);
  } else {
    return y;
  }
}

template <typename T>
constexpr T rotr(T x, int s) noexcept {
  static_assert(is_unsigned_v<T>);

#if defined(__GNUC__) || defined(__clang__)
#if __has_builtin(__builtin_rotateright8)
  if constexpr (sizeof(T) == sizeof(uint8_t)) {
    return __builtin_rotateright8(x, static_cast<uint8_t>(s));
  }
#endif

#if __has_builtin(__builtin_rotateright16)
  if constexpr (sizeof(T) == sizeof(uint16_t)) {
    return __builtin_rotateright16(x, static_cast<uint16_t>(s));
  }
#endif

#if __has_builtin(__builtin_rotateright32)
  if constexpr (sizeof(T) == sizeof(uint32_t)) {
    return __builtin_rotateright32(x, static_cast<uint32_t>(s));
  }
#endif

#if __has_builtin(__builtin_rotateright64)
  if constexpr (sizeof(T) == sizeof(uint64_t)) {
    return __builtin_rotateright64(x, static_cast<uint64_t>(s));
  }
#endif
#endif

  return rotr_fallback(x, s);
}

template <typename T>
constexpr T bit_ceil_fallback(T x) noexcept {
  static_assert(is_unsigned_v<T>);

  // Annoying, but makes it compatible with std::bit_ceil.
  if (x == 0) return 1;

  // Shuffle down highest bit after substract 1. Then add 1 back to flip the `next` bit.
  --x;
  constexpr int n{sizeof(T) * 8};
  for (int i{1}; i < n; i *= 2) {
    x |= x >> i;
  }
  ++x;

  return x;
}

template <typename T>
constexpr T bit_ceil(T x) noexcept {
  static_assert(is_unsigned_v<T>);

  // Annoying, but makes it compatible with std::bit_ceil.
  if (x == 0) return 1;

  int n{sizeof(T) * 8};
  int s{countl_zero(--x)};

  x = s != 0;
  x <<= n - s;

  return x;
}

template <typename T>
constexpr bool has_single_bit_fallback(T x) noexcept {
  static_assert(is_unsigned_v<T>);

  return (x != 0) & !(x & (x - 1));
}

template <typename T>
constexpr bool has_single_bit(T x) noexcept {
  static_assert(is_unsigned_v<T>);

#if defined(__clang__)
  return popcount(x) == 1;
#endif

  return has_single_bit_fallback(x);
}

}}  // namespace nvhm::std_ext


namespace nvhm {

template <size_t N, typename T>
constexpr T* assume_aligned(T* p) noexcept {
  static_assert(N >= alignof(T) && N % alignof(T) == 0);
#if defined(__cpp_lib_assume_aligned)
  return std::assume_aligned<N>(p);
#else
  return std_ext::assume_aligned<N>(p);
#endif
}

#if defined(__cpp_lib_construct_at)
using std::construct_at;
#else
using std_ext::construct_at;
#endif

template <typename T>
constexpr int countl_zero(T x) noexcept {
  if constexpr (std::is_signed_v<T>) {
    return countl_zero(static_cast<std::make_unsigned_t<T>>(x));
#if defined(__cpp_lib_bitops)
  } else if constexpr (std::is_unsigned_v<T>) {
    return std::countl_zero(x);
#endif
  } else {
    return std_ext::countl_zero(x);
  }
}

template <typename T>
constexpr int countr_zero(T x) noexcept {
  if constexpr (std::is_signed_v<T>) {
    return countr_zero(static_cast<std::make_unsigned_t<T>>(x));
#if defined(__cpp_lib_bitops)
  } else if constexpr (std::is_unsigned_v<T>) {
    return std::countr_zero(x);
#endif
  } else {
    return std_ext::countr_zero(x);
  }
}

template <typename T>
constexpr int popcount(T x) noexcept {
  if constexpr (std::is_signed_v<T>) {
    return popcount(static_cast<std::make_unsigned_t<T>>(x));
#if defined(__cpp_lib_bitops)
  } else if constexpr (std::is_unsigned_v<T>) {
    return std::popcount(x);
#endif
  } else {
    return std_ext::popcount(x);
  }
}

template <typename T>
constexpr T rotl(T x, int s) noexcept {
  if constexpr (std::is_signed_v<T>) {
    return rotl(static_cast<std::make_unsigned_t<T>>(x), s);
#if defined(__cpp_lib_bitops)
  } else if constexpr (std::is_unsigned_v<T>) {
    return std::rotl(x, s);
#endif
  } else {
    return std_ext::rotl(x, s);
  }
}

template <typename T>
constexpr T rotr(T x, int s) noexcept {
  if constexpr (std::is_signed_v<T>) {
    return rotr(static_cast<std::make_unsigned_t<T>>(x), s);
#if defined(__cpp_lib_bitops)
  } else if constexpr (std::is_unsigned_v<T>) {
    return std::rotr(x, s);
#endif
  } else {
    return std_ext::rotr(x, s);
  }
}

template <typename T>
constexpr T bit_ceil(T x) noexcept {
  if constexpr (std::is_signed_v<T>) {
    return static_cast<T>(bit_ceil(static_cast<std::make_unsigned_t<T>>(x)));
#if defined(__cpp_lib_bitops)
  } else if constexpr (std::is_unsigned_v<T>) {
    return std::bit_ceil(x);
#endif
  } else {
    return std_ext::bit_ceil(x);
  }
}

template <typename T>
constexpr bool has_single_bit(T x) noexcept {
  if constexpr (std::is_signed_v<T>) {
    return has_single_bit(static_cast<std::make_unsigned_t<T>>(x));
#if defined(__cpp_lib_bitops)
  } else if constexpr (std::is_unsigned_v<T>) {
    return std::has_single_bit(x);
#endif
  } else {
    return std_ext::has_single_bit(x);
  }
}

}  // namespace nvhm