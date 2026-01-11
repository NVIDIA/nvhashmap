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

#include <gtest/gtest.h>
#include <iomanip>
#include <nvhashmap/stdlib_ext.hpp>
#include <random>

using namespace nvhm;

std::random_device rd;

static constexpr size_t num_random_tests{100'000'000};

template <typename T>
void test_countl_zero() {
  T n{};
  EXPECT_EQ(std_ext::countl_zero(n), sizeof(T) * 8);
#if defined(__cpp_lib_bitops)
  EXPECT_EQ(std_ext::countl_zero(n), std::countl_zero(n));
#endif

  n = ~n;
  EXPECT_EQ(std_ext::countl_zero(n), {});
#if defined(__cpp_lib_bitops)
  EXPECT_EQ(std_ext::countl_zero(n), std::countl_zero(n));
#endif

  for (size_t i{}; i != sizeof(T) * 8; ++i) {
    n = T{1} << i;
    EXPECT_EQ(std_ext::countl_zero(n), sizeof(T) * 8 - 1 - i);
#if defined(__cpp_lib_bitops)
    EXPECT_EQ(std_ext::countl_zero(n), std::countl_zero(n));
#endif
  }

#if defined(__cpp_lib_bitops)
  std::mt19937_64 rng{rd()};
  for (T i{}; i < num_random_tests; ++i) {
    n = static_cast<T>(rng());
    EXPECT_EQ(std_ext::countl_zero(n), std::countl_zero(n));
  }
#else
  (void)num_random_tests;
#endif
}

TEST(TestBitFunc, CountlZero) {
  test_countl_zero<uint32_t>();
  test_countl_zero<uint64_t>();
  test_countl_zero<__uint128_t>();
}

template <typename T>
void test_countr_zero() {
  T n{};
  EXPECT_EQ(std_ext::countr_zero(n), sizeof(T) * 8);
#if defined(__cpp_lib_bitops)
  EXPECT_EQ(std_ext::countr_zero(n), std::countr_zero(n));
#endif

  n = ~n;
  EXPECT_EQ(std_ext::countr_zero(n), {});
#if defined(__cpp_lib_bitops)
  EXPECT_EQ(std_ext::countr_zero(n), std::countr_zero(n));
#endif

  for (size_t i{}; i != sizeof(T) * 8; ++i) {
    n = T{1} << i;
    EXPECT_EQ(std_ext::countr_zero(n), i);
#if defined(__cpp_lib_bitops)
    EXPECT_EQ(std_ext::countr_zero(n), std::countr_zero(n));
#endif
  }

#if defined(__cpp_lib_bitops)
  std::mt19937_64 rng{rd()};
  for (T i{}; i < num_random_tests; ++i) {
    n = static_cast<T>(rng());
    EXPECT_EQ(std_ext::countr_zero(n), std::countr_zero(n));
  }
#else
  (void)num_random_tests;
#endif
}

TEST(TestBitFunc, CountrZero) {
  test_countr_zero<uint32_t>();
  test_countr_zero<uint64_t>();
  test_countr_zero<__uint128_t>();
}

template <typename T>
void test_popcount() {
  T n{};
  for (size_t i{}; i <= sizeof(T) * 8; ++i) {
    EXPECT_EQ(std_ext::popcount(n), i);
#if defined(__cpp_lib_bitops)
    EXPECT_EQ(std_ext::popcount(n), std::popcount(n));
#endif
    n = (n << 1) | 1;
  }

#if defined(__cpp_lib_bitops)
  std::mt19937_64 rng{rd()};
  for (T i{}; i < num_random_tests; ++i) {
    n = static_cast<T>(rng());
    EXPECT_EQ(std_ext::popcount(n), std::popcount(n));
  }
#else
  (void)num_random_tests;
#endif
}

TEST(TestBitFunc, PopCount) {
  test_popcount<uint32_t>();
  test_popcount<uint64_t>();
}

template <typename T>
void test_rotl() {
#if defined(__cpp_lib_bitops)
  std::mt19937_64 rng{rd()};
  for (T i{}; i != num_random_tests; ++i) {
    const T n{static_cast<T>(rng())};
    EXPECT_EQ(std_ext::rotl(n, i % (sizeof(T) * 8)), std::rotl(n, i % (sizeof(T) * 8)));
  }
#else
  (void)num_random_tests;
#endif
}

TEST(TestBitFunc, Rotl) {
  test_rotl<uint32_t>();
  test_rotl<uint64_t>();
}

template <typename T>
void test_rotr() {
#if defined(__cpp_lib_bitops)
  std::mt19937_64 rng{rd()};
  for (T i{}; i != num_random_tests; ++i) {
    const T n{static_cast<T>(rng())};
    EXPECT_EQ(std_ext::rotr(n, i % (sizeof(T) * 8)), std::rotr(n, i % (sizeof(T) * 8)));
  }
#else
  (void)num_random_tests;
#endif
}

TEST(TestBitFunc, Rotr) {
  test_rotr<uint32_t>();
  test_rotr<uint64_t>();
}

template <typename T>
void test_bit_ceil() {
  constexpr T zero{};
  constexpr T one{1};
  T n{zero};

  // Should round up zero.
  EXPECT_EQ(std_ext::bit_ceil(n), one);
#if defined(__cpp_lib_int_pow2)
  EXPECT_EQ(std_ext::bit_ceil(n), std::bit_ceil(n));
#endif

  // Should reproduce 2^n values.
  for (T i{}; i != sizeof(T) * 8; ++i) {
    n = one << i;
    EXPECT_EQ(std_ext::bit_ceil(n), n);
#if defined(__cpp_lib_int_pow2)
    EXPECT_EQ(std_ext::bit_ceil(n), std::bit_ceil(n));
#endif
  }

  // Should round to next 2^n value.
  for (T i{}; i != sizeof(T) * 8 - 1; ++i) {
    n = (one << i) + 1;
    EXPECT_EQ(std_ext::bit_ceil(n), one << (i + 1));
#if defined(__cpp_lib_int_pow2)
    EXPECT_EQ(std_ext::bit_ceil(n), std::bit_ceil(n));
#endif
  }

  // Wrap around behavior should be identical.
  n = (one << (sizeof(T) * 8 - 1)) | (one << (sizeof(T) * 8) / 2);
  EXPECT_EQ(std_ext::bit_ceil(n), one);
#if defined(__cpp_lib_int_pow2)
  // With GCC, on ARM64 std::bit_ceil does return different values for hard-coded values if the
  // upper most bit is set.
#if defined(__clang__)
  EXPECT_EQ(std_ext::bit_ceil(n), std::bit_ceil(n));
#endif
#endif

  n = ~zero;
  EXPECT_EQ(std_ext::bit_ceil(n), one);
#if defined(__cpp_lib_int_pow2)
  // With GCC, on ARM64 std::bit_ceil does return different values for hard-coded values if the
  // upper most bit is set.
#if defined(__clang__)
  EXPECT_EQ(std_ext::bit_ceil(n), std::bit_ceil(n));
#endif
#endif

#if defined(__cpp_lib_int_pow2)
  std::mt19937_64 rng{static_cast<T>(rd())};
  for (T i{}; i != num_random_tests; ++i) {
    n = static_cast<T>(rng());
    EXPECT_EQ(std_ext::bit_ceil(n), std::bit_ceil(n));
  }
#else
  (void)num_random_tests;
#endif
}

TEST(TestBitFunc, BitCeil) {
  test_bit_ceil<uint32_t>();
  test_bit_ceil<uint64_t>();
}
