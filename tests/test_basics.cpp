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

#include "test_common.hpp"
#include <gtest/gtest.h>
#include <iomanip>
#include <nvhashmap/common.hpp>
#include <nvhashmap/stdlib_ext.hpp>
#include <random>

using namespace nvhm;

template <typename T>
const static auto gen{[]() {
  static std::uniform_int_distribution<T> dist;
  return dist(rng);
}};

template <typename T> constexpr static T zero{};
template <typename T> constexpr static T full{static_cast<T>(~zero<T>)};
template <typename T> constexpr static T one{1};

constexpr static int_t num_random_tests{1024 * 1024};

template <typename T>
void test_broadcast() {
  T n;
  
  if constexpr (sizeof(T) >= sizeof(uint8_t)) {
    for (size_t i{}; i < 256; ++i) {
      std::array<uint8_t, sizeof(T)> m;
      uint8_t x{static_cast<uint8_t>(i)};

      std::fill(m.begin(), m.end(), x);
      n = broadcast<T>(x);

      EXPECT_EQ(*reinterpret_cast<const T*>(m.data()), n);
    }
  }

  if constexpr (sizeof(T) >= sizeof(uint16_t)) {
    for (size_t i{}; i < 65536; ++i) {
      std::array<uint16_t, sizeof(T) / 2> m;
      uint16_t x{static_cast<uint16_t>(i)};

      std::fill(m.begin(), m.end(), x);
      n = broadcast<T>(x);

      EXPECT_EQ(*reinterpret_cast<const T*>(m.data()), n);
    }
  }

  if constexpr (sizeof(T) >= sizeof(uint32_t)) {
    for (size_t i{}; i < num_random_tests; ++i) {
      std::array<uint32_t, sizeof(T) / 4> m;
      uint32_t x{gen<uint32_t>()};

      std::fill(m.begin(), m.end(), x);
      n = broadcast<T>(x);

      EXPECT_EQ(*reinterpret_cast<const T*>(m.data()), n);
    }
  }

  if constexpr (sizeof(T) >= sizeof(uint64_t)) {
    for (size_t i{}; i < num_random_tests; ++i) {
      std::array<uint64_t, sizeof(T) / 8> m;
      uint64_t x{gen<uint64_t>()};

      std::fill(m.begin(), m.end(), x);
      n = broadcast<T>(x);

      EXPECT_EQ(*reinterpret_cast<const T*>(m.data()), n);
    }
  }

  if constexpr (sizeof(T) >= sizeof(__uint128_t)) {
    for (size_t i{}; i < num_random_tests; ++i) {
      std::array<__uint128_t, sizeof(T) / 8> m;
      __uint128_t x{gen<__uint128_t>()};

      std::fill(m.begin(), m.end(), x);
      n = broadcast<T>(x);

      EXPECT_EQ(*reinterpret_cast<const T*>(m.data()), n);
    }
  }
}

#define TEST_BROADCAST_(_X_) \
  TEST(test_broadcast, _X_) { test_broadcast<_X_>(); }

NVHM_FOR_EACH_(
  TEST_BROADCAST_,
  uint8_t,
  uint16_t,
  uint32_t,
  uint64_t,
  __uint128_t
);

template <typename T>
void test_popcount() {
  T n;
  
  n = zero<T>;
  for (int_t i{}; i <= num_bits_v<T>; ++i) {
    EXPECT_EQ(popcount(n), i) << "n: " << n;
    EXPECT_EQ(popcount(n), std_ext::popcount(n)) << "n: " << n;
    EXPECT_EQ(popcount(n), std_ext::popcount_fallback(n)) << "n: " << n;
#if defined(__cpp_lib_bitops)
    EXPECT_EQ(popcount(n), std::popcount(n)) << "n: " << n;
#endif
    n = static_cast<T>((n << 1) | 1);
  }

  for (int_t i{}; i <= num_bits_v<T>; ++i) {
    EXPECT_EQ(popcount(n), num_bits_v<T> - i) << "n: " << n;
    EXPECT_EQ(popcount(n), std_ext::popcount(n)) << "n: " << n;
    EXPECT_EQ(popcount(n), std_ext::popcount_fallback(n)) << "n: " << n;
#if defined(__cpp_lib_bitops)
    EXPECT_EQ(popcount(n), std::popcount(n)) << "n: " << n;
#endif
    n <<= 1;
  }

  for (int_t i{}; i < num_random_tests; ++i) {
    n = gen<T>();
    EXPECT_EQ(popcount(n), std_ext::popcount(n)) << "n: " << n;
    EXPECT_EQ(popcount(n), std_ext::popcount_fallback(n)) << "n: " << n;
#if defined(__cpp_lib_bitops)
    EXPECT_EQ(popcount(n), std::popcount(n)) << "n: " << n;
#endif
  }
}

#define TEST_POPCOUNT_(_X_) \
  TEST(test_popcount, _X_) { test_popcount<_X_>(); }

NVHM_FOR_EACH_(
  TEST_POPCOUNT_,
  uint8_t,
  uint16_t,
  uint32_t,
  uint64_t,
  __uint128_t
);

template <typename T>
void test_countl_zero() {
  T n;
  
  n = zero<T>;
  EXPECT_EQ(countl_zero(n), num_bits_v<T>) << "n: " << n;
  EXPECT_EQ(countl_zero(n), std_ext::countl_zero(n)) << "n: " << n;
  EXPECT_EQ(countl_zero(n), std_ext::countl_zero_fallback(n)) << "n: " << n;
#if defined(__cpp_lib_bitops)
  EXPECT_EQ(countl_zero(n), std::countl_zero(n)) << "n: " << n;
#endif

  n = full<T>;
  EXPECT_EQ(countl_zero(n), 0) << "n: " << n;
  EXPECT_EQ(countl_zero(n), std_ext::countl_zero(n)) << "n: " << n;
  EXPECT_EQ(countl_zero(n), std_ext::countl_zero_fallback(n)) << "n: " << n;
#if defined(__cpp_lib_bitops)
  EXPECT_EQ(countl_zero(n), std::countl_zero(n)) << "n: " << n;
#endif

  for (int i{}; i < num_bits_v<T>; ++i) {
    n = static_cast<T>(one<T> << i);
    EXPECT_EQ(countl_zero(n), num_bits_v<T> - 1 - i) << "n: " << n;
    EXPECT_EQ(countl_zero(n), std_ext::countl_zero(n)) << "n: " << n;
    EXPECT_EQ(countl_zero(n), std_ext::countl_zero_fallback(n)) << "n: " << n;
#if defined(__cpp_lib_bitops)
    EXPECT_EQ(std_ext::countl_zero(n), std::countl_zero(n)) << "n: " << n;
#endif
  }

  for (int_t i{}; i < num_random_tests; ++i) {
    n = gen<T>();
    EXPECT_EQ(countl_zero(n), std_ext::countl_zero(n)) << "n: " << n;
    EXPECT_EQ(countl_zero(n), std_ext::countl_zero_fallback(n)) << "n: " << n;
#if defined(__cpp_lib_bitops)
    EXPECT_EQ(countl_zero(n), std::countl_zero(n)) << "n: " << n;
#endif
  }
}

#define TEST_COUNTL_ZERO_(_X_) \
  TEST(test_countl_zero, _X_) { test_countl_zero<_X_>(); }

NVHM_FOR_EACH_(
  TEST_COUNTL_ZERO_,
  uint8_t,
  uint16_t,
  uint32_t,
  uint64_t,
  __uint128_t
);

template <typename T>
void test_countr_zero() {
  T n;
  
  n = zero<T>;
  EXPECT_EQ(countr_zero(n), num_bits_v<T>) << "n: " << n;
  EXPECT_EQ(countr_zero(n), std_ext::countr_zero(n)) << "n: " << n;
  EXPECT_EQ(countr_zero(n), std_ext::countr_zero_fallback(n)) << "n: " << n;
#if defined(__cpp_lib_bitops)
  EXPECT_EQ(countr_zero(n), std::countr_zero(n)) << "n: " << n;
#endif

  n = full<T>;
  EXPECT_EQ(countr_zero(n), 0) << "n: " << n;
  EXPECT_EQ(countr_zero(n), std_ext::countr_zero(n)) << "n: " << n;
  EXPECT_EQ(countr_zero(n), std_ext::countr_zero_fallback(n)) << "n: " << n;
  #if defined(__cpp_lib_bitops)
  EXPECT_EQ(countr_zero(n), std::countr_zero(n)) << "n: " << n;
#endif

  for (int i{}; i < num_bits_v<T>; ++i) {
    n = static_cast<T>(one<T> << i);
    EXPECT_EQ(countr_zero(n), i) << "n: " << n;
    EXPECT_EQ(countr_zero(n), std_ext::countr_zero(n)) << "n: " << n;
    EXPECT_EQ(countr_zero(n), std_ext::countr_zero_fallback(n)) << "n: " << n;
#if defined(__cpp_lib_bitops)
    EXPECT_EQ(countr_zero(n), std::countr_zero(n)) << "n: " << n;
#endif
  }

  for (int_t i{}; i < num_random_tests; ++i) {
    n = gen<T>();
    EXPECT_EQ(countr_zero(n), std_ext::countr_zero(n)) << "n: " << n;
    EXPECT_EQ(countr_zero(n), std_ext::countr_zero_fallback(n)) << "n: " << n;
#if defined(__cpp_lib_bitops)
    EXPECT_EQ(countr_zero(n), std::countr_zero(n)) << "n: " << n;
#endif
  }
}

#define TEST_COUNTR_ZERO_(_X_) \
  TEST(test_countr_zero, _X_) { test_countr_zero<_X_>(); }

NVHM_FOR_EACH_(
  TEST_COUNTR_ZERO_,
  uint8_t,
  uint16_t,
  uint32_t,
  uint64_t,
  __uint128_t
);

template <typename T>
void test_rotl() {
  T n;
  int s;

  for (int i{}; i < num_random_tests; ++i) {
    n = gen<T>();
    EXPECT_EQ(rotl(n, i % num_bits_v<T>), std_ext::rotl(n, i % num_bits_v<T>)) << "n: " << n;
    EXPECT_EQ(rotl(n, i % num_bits_v<T>), std_ext::rotl_fallback(n, i % num_bits_v<T>)) << "n: " << n;
#if defined(__cpp_lib_bitops)
    EXPECT_EQ(rotl(n, i % num_bits_v<T>), std::rotl(n, i % num_bits_v<T>)) << "n: " << n;
#endif
  }

  for (int s{-1024}; s <= 1024; ++s) {
    n = gen<T>();
    EXPECT_EQ(rotl(n, s), std_ext::rotl(n, s)) << "n: " << n;
    EXPECT_EQ(rotl(n, s), std_ext::rotl_fallback(n, s)) << "n: " << n;
#if defined(__cpp_lib_bitops)
    EXPECT_EQ(rotl(n, s), std::rotl(n, s)) << "n: " << n;
#endif
  }

  for (int_t i{}; i < num_random_tests; ++i) {
    n = gen<T>();
    s = gen<int>();
    EXPECT_EQ(rotl(n, s), std_ext::rotl(n, s)) << "n: " << n;
    EXPECT_EQ(rotl(n, s), std_ext::rotl_fallback(n, s)) << "n: " << n;
#if defined(__cpp_lib_bitops)
    EXPECT_EQ(rotl(n, s), std::rotl(n, s)) << "n: " << n;
#endif
  }
}

#define TEST_ROTL_(_X_) \
  TEST(test_rotl, _X_) { test_rotl<_X_>(); }

NVHM_FOR_EACH_(
  TEST_ROTL_,
  uint8_t,
  uint16_t,
  uint32_t,
  uint64_t,
  __uint128_t
);

template <typename T>
void test_rotr() {
  T n;
  int s;

  for (int i{}; i < num_random_tests; ++i) {
    n = gen<T>();
    EXPECT_EQ(rotr(n, i % num_bits_v<T>), std_ext::rotr(n, i % num_bits_v<T>)) << "n: " << n;
    EXPECT_EQ(rotr(n, i % num_bits_v<T>), std_ext::rotr_fallback(n, i % num_bits_v<T>)) << "n: " << n;
#if defined(__cpp_lib_bitops)
    EXPECT_EQ(rotr(n, i % num_bits_v<T>), std::rotr(n, i % num_bits_v<T>)) << "n: " << n;
#endif
  }

  for (int s{-1024}; s <= 1024; ++s) {
    n = gen<T>();
    EXPECT_EQ(rotr(n, s), std_ext::rotr(n, s)) << "n: " << n;
    EXPECT_EQ(rotr(n, s), std_ext::rotr_fallback(n, s)) << "n: " << n;
#if defined(__cpp_lib_bitops)
    EXPECT_EQ(rotr(n, s), std::rotr(n, s)) << "n: " << n;
#endif
  }

  for (int_t i{}; i < num_random_tests; ++i) {
    n = gen<T>();
    s = gen<int>();
    EXPECT_EQ(rotr(n, s), std_ext::rotr(n, s)) << "n: " << n;
    EXPECT_EQ(rotr(n, s), std_ext::rotr_fallback(n, s)) << "n: " << n;
#if defined(__cpp_lib_bitops)
    EXPECT_EQ(rotr(n, s), std::rotr(n, s)) << "n: " << n;
#endif
  }
}

#define TEST_ROTR_(_X_) \
  TEST(test_rotr, _X_) { test_rotr<_X_>(); }

NVHM_FOR_EACH_(
  TEST_ROTR_,
  uint8_t,
  uint16_t,
  uint32_t,
  uint64_t,
  __uint128_t
);

template <typename T>
void test_bit_ceil() {
  T n, r;

  // Should round up zero.
  n = zero<T>;
#if defined(__cpp_lib_int_pow2)
  r = std::bit_ceil(n);
#else
  r = one<T>;
#endif
  EXPECT_EQ(bit_ceil(n), one<T>) << "n: " << n << " r: " << r;
  EXPECT_EQ(bit_ceil(n), std_ext::bit_ceil(n)) << "n: " << n << " r: " << r;
  EXPECT_EQ(bit_ceil(n), std_ext::bit_ceil_fallback(n)) << "n: " << n << " r: " << r;
#if defined(__cpp_lib_int_pow2)
  EXPECT_EQ(bit_ceil(n), std::bit_ceil(n)) << "n: " << n << " r: " << r;
#endif

  // Should reproduce 2^n values.
  for (int_t i{}; i < num_bits_v<T>; ++i) {
    n = static_cast<T>(one<T> << i);
#if defined(__cpp_lib_int_pow2)
    r = std::bit_ceil(n);
#else
    r = n;
#endif
    EXPECT_EQ(bit_ceil(n), n) << "n: " << n << " r: " << r;
    EXPECT_EQ(bit_ceil(n), std_ext::bit_ceil(n)) << "n: " << n << " r: " << r;
    EXPECT_EQ(bit_ceil(n), std_ext::bit_ceil_fallback(n)) << "n: " << n << " r: " << r;
#if defined(__cpp_lib_int_pow2)
    EXPECT_EQ(std_ext::bit_ceil(n), std::bit_ceil(n)) << "n: " << n << " r: " << r;
#endif
  }

  // Should round to next 2^n value.
  for (int_t i{}; i < num_bits_v<T> - 1; ++i) {
    n = static_cast<T>((one<T> << i) + 1);
#if defined(__cpp_lib_int_pow2)
    r = std::bit_ceil(n);
#else
    r = static_cast<T>(one<T> << (i + 1));
#endif
    EXPECT_EQ(bit_ceil(n), one<T> << (i + 1)) << "n: " << n << " r: " << static_cast<__uint128_t>(r);
    EXPECT_EQ(bit_ceil(n), std_ext::bit_ceil(n)) << "n: " << n << " r: " << static_cast<__uint128_t>(r);
    EXPECT_EQ(bit_ceil(n), std_ext::bit_ceil_fallback(n)) << "n: " << n << " r: " << static_cast<__uint128_t>(r);
#if defined(__cpp_lib_int_pow2)
    EXPECT_EQ(bit_ceil(n), std::bit_ceil(n)) << "n: " << n << " r: " << static_cast<__uint128_t>(r);
#endif
  }

  
  n = (one<T> << (num_bits_v<T> - 1)) | (one<T> << (num_bits_v<T> / 2));
#if defined(__cpp_lib_int_pow2)
  r = std::bit_ceil(n);
#else
  r = zero<T>;
#endif
  EXPECT_EQ(std_ext::bit_ceil(n), zero<T>) << "n: " << static_cast<__uint128_t>(n) << " r: " << static_cast<__uint128_t>(r);
  EXPECT_EQ(std_ext::bit_ceil_fallback(n), zero<T>) << "n: " << static_cast<__uint128_t>(n) << " r: " << static_cast<__uint128_t>(r);
  
  n = full<T>;
#if defined(__cpp_lib_int_pow2)
  r = std::bit_ceil(n);
#else
  r = zero<T>;
#endif
  EXPECT_EQ(std_ext::bit_ceil(n), zero<T>) << "n: " << n << " r: " << static_cast<__uint128_t>(r);
  EXPECT_EQ(std_ext::bit_ceil_fallback(n), zero<T>) << "n: " << n << " r: " << static_cast<__uint128_t>(r);

  for (int_t i{}; i < num_random_tests; ++i) {
    n = gen<T>();
    // Note: Overflow behavior is actually undefined for std::bit_ceil.
    if (countl_zero(n) == 0 && popcount(n) > 1) {
      n >>= 1;
    }

#if defined(__cpp_lib_int_pow2)
    r = std::bit_ceil(n);
#else
    r = std_ext::bit_ceil_fallback(n);
#endif
    EXPECT_EQ(bit_ceil(n), std_ext::bit_ceil(n)) << "n: " << n << " r: " << static_cast<__uint128_t>(r);
    EXPECT_EQ(bit_ceil(n), std_ext::bit_ceil_fallback(n)) << "n: " << n << " r: " << static_cast<__uint128_t>(r);
#if defined(__cpp_lib_int_pow2)
    EXPECT_EQ(bit_ceil(n), std::bit_ceil(n)) << "n: " << n << " r: " << static_cast<__uint128_t>(r);
#endif
  }
}

#define TEST_BIT_CEIL_(_X_) \
  TEST(test_bit_ceil, _X_) { test_bit_ceil<_X_>(); }

NVHM_FOR_EACH_(
  TEST_BIT_CEIL_,
  uint8_t,
  uint16_t,
  uint32_t,
  uint64_t,
  __uint128_t
);
