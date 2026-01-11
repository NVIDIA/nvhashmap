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
#include <nvhashmap/guarded.hpp>
#include <nvhashmap/map.hpp>

using namespace nvhm;

template <typename Map, size_t initial_capacity = 0, size_t value_size = 10>
void test_insert_lookup() {
  using map_t = Map;
  using key_t = typename map_t::key_type;
  using read_pos_t = typename map_t::read_pos_type;
  using write_pos_t = typename map_t::write_pos_type;

  map_t map(initial_capacity, value_size);
  key_t k;

  // Should be empty.
  for (k = -999; k <= 999; ++k) {
    EXPECT_FALSE(map.contains(k));
  }

  // Insert some values.
  for (k = -9999; k <= 9999; ++k) {
    if (k % 12 == 0) {
      map.upsert(k);
      EXPECT_TRUE(map.contains(k));

      write_pos_t pos{map.update(k)};
      EXPECT_NE(pos, npos);

      std::sprintf(map.raw_values_at(pos), "%ld", k);
      map.value_at(pos) = static_cast<double>(-k);
    }
  }

  map.clear();
  for (k = -9999; k <= 9999; ++k) {
    if (k % 12 == 0) {
      const write_pos_t pos{map.upsert(k)};
      std::sprintf(map.raw_values_at(pos), "%ld", k);
      map.value_at(pos) = static_cast<double>(-k);
    }
  }

  // Should exist and have retained the value and meta, or not exist at all.
  for (k = -9999; k <= 9999; ++k) {
    if (k % 12 == 0) {
      const read_pos_t pos{map.lookup(k)};
      EXPECT_NE(pos, npos);
      EXPECT_EQ(map.raw_values_at(pos), std::to_string(k));
      EXPECT_EQ(map.value_at(pos), static_cast<double>(-k));
    } else {
      EXPECT_FALSE(map.contains(k));
    }
  }

  // Delete 1/2 of keys and reallocate on the way.
  for (k = -9999; k <= 9999; ++k) {
    if (k % 24 == 0) {
      map.erase(k);
    }
  }

  // Increase a whole bunch of keys.
  for (k = -9999; k <= 9999; ++k) {
    if (k % 3 == 0) {
      const write_pos_t pos{map.upsert(k)};
      // if (map.raw_values_at(p) == std::to_string(k)) {
      //   std::cout << "recycling " << k << "   ==>  " << p << '\n';
      // }
      if (k % 12 != 0) {
        std::strcpy(map.raw_values_at(pos), "");
        map.value_at(pos) = 0.0;
      }
    }
  }

  // Check again if the values we expect to be retained have been retained.
  for (k = -9999; k <= 9999; ++k) {
    if (k % 12 == 0 && k % 24 != 0) {
      const read_pos_t pos{map.lookup(k)};
      EXPECT_NE(pos, npos);
      EXPECT_EQ(map.raw_values_at(pos), std::to_string(k));
      EXPECT_EQ(map.value_at(pos), static_cast<double>(-k));
    }
  }
}

template <typename Kernel>
void test_insert_find() {
  {
    using map_t = map<int64_t, double, char, Kernel, linear_seq<>, false, true>;
    test_insert_lookup<map_t>();
    test_insert_lookup<guarded<map_t>>();
    test_insert_lookup<guarded<guarded<map_t>>>();
  }
  // TODO: Support testing the cache.
  //{
  //  using cache_t = cache<int64_t, double, char, Kernel>;
  //  test_insert_lookup<cache_t>();
  //  test_insert_lookup<guarded<cache_t>>();
  //}
}

#define TEST_INSERT_FIND_(_X_) \
  TEST(test_insert_find, _X_) { test_insert_find<_X_>(); }

NVHM_FOR_EACH_(
  TEST_INSERT_FIND_,
  // array kernels
  array_kernel8_t, array_kernel16_t, array_kernel32_t, array_kernel64_t, array_kernel128_t,
  array_kernel256_t, array_kernel512_t, array_kernel1024_t,
  // uint kernels
  uint_kernel8_t, uint_kernel16_t, uint_kernel32_t, uint_kernel64_t, uint_kernel128_t
#if NVHM_WITH_SSE2
  ,
  sse_kernel_t
#endif
#if NVHM_WITH_AVX2
  ,
  avx_kernel_t
#endif
#if NVHM_WITH_AVX512
  ,
  avx512_kernel_t
#endif
#if NVHM_WITH_NEON
  ,
  neon_kernel64_t, neon_kernel128_t, neon_kernel256_t, neon_kernel512_t
#endif
#if NVHM_WITH_SVE
  ,
  sve_kernel8_t, sve_kernel16_t, sve_kernel32_t, sve_kernel64_t, sve_kernel128_t
#if __ARM_FEATURE_SVE_BITS >= 256
  ,
  sve_kernel256_t
#endif
#if __ARM_FEATURE_SVE_BITS >= 512
  ,
  sve_kernel512_t
#endif
#if __ARM_FEATURE_SVE_BITS >= 1024
  ,
  sve_kernel1024_t
#endif
#if __ARM_FEATURE_SVE_BITS >= 2048
  ,
  sve_kernel2048_t
#endif
#endif
);

template <typename Map>
void test_grow_shrink_2() {
  using map_t = Map;
  using key_t = typename map_t::key_type;

  map_t map(0, 1);
  key_t k;

  // Should not grow until we reach this point.
  size_t capacity0{map.capacity()};
  size_t limit0{map.growth_threshold()};
  for (k = 0; static_cast<size_t>(k) < limit0; ++k) {
    EXPECT_FALSE(map.contains(k));
    map.upsert(k);
    EXPECT_TRUE(map.contains(k));
    EXPECT_EQ(map.size(), k + 1);
    EXPECT_EQ(map.capacity(), capacity0);
  }
  // std::cout << map.size() << " / " << map.capacity() << '\n';

  constexpr size_t growth_factor{4};

  // Should grow now.
  map.upsert(k);
  EXPECT_EQ(map.size(), k + 1);
  EXPECT_EQ(map.capacity(), capacity0 * growth_factor);
  ++k;
  // std::cout << map.size() << " / " << map.capacity() << '\n';

  // Should not grow
  capacity0 *= growth_factor;
  limit0 *= growth_factor;
  for (; static_cast<size_t>(k) < limit0; ++k) {
    map.upsert(k);
    EXPECT_EQ(map.size(), k + 1);
    EXPECT_EQ(map.capacity(), capacity0);
  }

  // Should grow now.
  map.upsert(k);
  EXPECT_EQ(map.size(), k + 1);
  EXPECT_EQ(map.capacity(), capacity0 * growth_factor);
  ++k;

  // std::cout << map.size() << " / " << map.capacity() << ", " << map.num_used() << ", " <<
  // map.num_tombstones() << '\n';

  // Should not shrink.
  capacity0 *= growth_factor;
  for (k = 0; map.size() > capacity0 / 4; ++k) {
    map.erase(k);
    // EXPECT_EQ(map.num_tombstones(), k + 1);
    EXPECT_EQ(map.capacity(), capacity0);
  }

  // Should shrink now.
  map.erase(k);
  EXPECT_EQ(map.num_tombstones(), 0);
  EXPECT_EQ(map.capacity(), capacity0 / 2);
  ++k;
  // std::cout << map.size() << " / " << map.capacity() << ", " << map.num_used() << ", " <<
  // map.num_tombstones() << '\n';
}

template <typename Kernel>
void test_grow_shrink() {
  using gs_map = map<int64_t, void, char, Kernel, default_seq_t, false, true>;

  test_grow_shrink_2<gs_map>();
}

#define TEST_GROW_SHRINK_(_X_) \
  TEST(test_grow_shrink, _X_) { test_grow_shrink<_X_>(); }

NVHM_FOR_EACH_(
  TEST_GROW_SHRINK_,
  // array kernels
  array_kernel8_t, array_kernel16_t, array_kernel32_t, array_kernel64_t, array_kernel128_t,
  array_kernel256_t, array_kernel512_t, array_kernel1024_t,
  // uint kernels
  uint_kernel8_t, uint_kernel16_t, uint_kernel32_t, uint_kernel64_t, uint_kernel128_t
#if NVHM_WITH_SSE2
  ,
  sse_kernel_t
#endif
#if NVHM_WITH_AVX2
  ,
  avx_kernel_t
#endif
#if NVHM_WITH_AVX512
  ,
  avx512_kernel_t
#endif
#if NVHM_WITH_NEON
  ,
  neon_kernel64_t, neon_kernel128_t, neon_kernel256_t, neon_kernel512_t
#endif
#if NVHM_WITH_SVE
  ,
  sve_kernel8_t, sve_kernel16_t, sve_kernel32_t, sve_kernel64_t, sve_kernel128_t
#if __ARM_FEATURE_SVE_BITS >= 256
  ,
  sve_kernel256_t
#endif
#if __ARM_FEATURE_SVE_BITS >= 512
  ,
  sve_kernel512_t
#endif
#if __ARM_FEATURE_SVE_BITS >= 1024
  ,
  sve_kernel1024_t
#endif
#if __ARM_FEATURE_SVE_BITS >= 2048
  ,
  sve_kernel2048_t
#endif
#endif
);

template <typename Map>
void test_clear_2() {
  using map_t = Map;
  using key_t = typename map_t::key_type;

  map_t map(size_t{}, size_t{1});

  // Fit up map.
  for (key_t k{-999}; k <= 999; ++k) {
    map.upsert(k);
  }
  // std::cout << map.size() << " / " << map.capacity() << '\n';
  EXPECT_EQ(map.size(), 1999);
  EXPECT_EQ(map.capacity(), 4096);

  // Clear with/without shrink.
  const size_t expected_capacity{map_t::auto_shrink ? map.min_capacity : map.capacity()};
  map.clear();

  // std::cout << map.size() << " / " << map.capacity() << '\n';
  EXPECT_EQ(map.size(), 0);
  EXPECT_EQ(map.capacity(), expected_capacity);
}

template <typename Key, typename Kernel>
void test_clear() {
  using map_n = map<Key, void, char, Kernel, default_seq_t, false, false>;
  using map_a = map<Key, void, char, Kernel, default_seq_t, false, true>;

  test_clear_2<map_n>();
  test_clear_2<guarded<map_n>>();
  test_clear_2<guarded<guarded<map_n>>>();

  test_clear_2<map_a>();
  test_clear_2<guarded<map_a>>();
  test_clear_2<guarded<guarded<map_a>>>();

  // TODO: Support testing the cache.
  // using cache = cache<int64_t, void, char, Kernel>;
  // test_clear_2<cache>();
}

#define TEST_CLEAR_(_X_)                                          \
  TEST(test_clear, int32_t_##_X_) { test_clear<int32_t, _X_>(); } \
  TEST(test_clear, int64_t_##_X_) { test_clear<int64_t, _X_>(); }

NVHM_FOR_EACH_(
  TEST_CLEAR_,
  // array kernels
  array_kernel8_t, array_kernel16_t, array_kernel32_t, array_kernel64_t, array_kernel128_t,
  array_kernel256_t,
  array_kernel512_t
  // TODO: Fix test.
  //, array_kernel1024_t
  // uint kernels
  ,
  uint_kernel8_t, uint_kernel16_t, uint_kernel32_t, uint_kernel64_t, uint_kernel128_t
#if NVHM_WITH_SSE2
  ,
  sse_kernel_t
#endif
#if NVHM_WITH_AVX2
  ,
  avx_kernel_t
#endif
#if NVHM_WITH_AVX512
  ,
  avx512_kernel_t
#endif
#if NVHM_WITH_NEON
  ,
  neon_kernel64_t, neon_kernel128_t, neon_kernel256_t, neon_kernel512_t
#endif
#if NVHM_WITH_SVE
  ,
  sve_kernel8_t, sve_kernel16_t, sve_kernel32_t, sve_kernel64_t, sve_kernel128_t
#if __ARM_FEATURE_SVE_BITS >= 256
  ,
  sve_kernel256_t
#endif
#if __ARM_FEATURE_SVE_BITS >= 512
  ,
  sve_kernel512_t
#endif
// TODO: Fix test.
// #if __ARM_FEATURE_SVE_BITS >= 1024
//  , sve_kernel1024_t
// #endif
// #if __ARM_FEATURE_SVE_BITS >= 2048
//  , sve_kernel2048_t
// #endif
#endif
);
