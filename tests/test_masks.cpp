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
#include <nvhashmap/kernel.hpp>
#include <random>

using namespace nvhm;

template <typename Mask>
void test_empty() {
  using mask_t = Mask;
  using repr_t = typename mask_t::repr_type;
  repr_t m;

  m = mask_t::empty();
  EXPECT_FALSE(mask_t::has_next(m));
}

template <typename Mask>
void test_full() {
  using mask_t = Mask;
  using repr_t = typename mask_t::repr_type;
  repr_t m;

  m = mask_t::full();
  for (raw_pos_t i{}; i < mask_t::max_count; ++i) {
    EXPECT_TRUE(mask_t::has_next(m));
    EXPECT_EQ(mask_t::next(m), i);
    m = mask_t::step(m);
  }
  EXPECT_FALSE(mask_t::has_next(m));
}

template <typename Mask>
void test_single() {
  using mask_t = Mask;
  using repr_t = typename mask_t::repr_type;
  repr_t m;

  for (int_t i{}; i < mask_t::max_count; ++i) {
    m = mask_t::single(i);
    EXPECT_TRUE(mask_t::has_next(m));
    EXPECT_EQ(mask_t::next(m), i);
    m = mask_t::step(m);
    EXPECT_FALSE(mask_t::has_next(m));
  }
}

template <typename Mask>
void test_contains() {
  using mask_t = Mask;
  using repr_t = typename mask_t::repr_type;
  repr_t m, n;

  m = mask_t::full();
  EXPECT_TRUE(mask_t::contains(m, mask_t::empty()));
  EXPECT_FALSE(mask_t::contains(mask_t::empty(), m));

  for (raw_pos_t i{}; i < mask_t::max_count; ++i) {
    m = mask_t::single(i);
    EXPECT_TRUE(mask_t::contains(m, mask_t::empty()));
    EXPECT_FALSE(mask_t::contains(mask_t::empty(), m));

    EXPECT_TRUE(mask_t::contains(m, m));
    for (raw_pos_t j{}; j < mask_t::max_count; ++j) {
      n = mask_t::single(j);
      EXPECT_EQ(mask_t::contains(m, n), j == i);
    }
  }

  m = mask_t::full();
  for (raw_pos_t i{}; i < mask_t::max_count; ++i) {
    n = mask_t::full();
    EXPECT_TRUE(mask_t::contains(n, m));;

    for (raw_pos_t j{}; j < mask_t::max_count; ++j) {
      EXPECT_TRUE(mask_t::contains(m, n));;
      n = mask_t::step(n);
      EXPECT_FALSE(mask_t::contains(n, m));;
    }
    EXPECT_TRUE(mask_t::contains(m, n));;
  }
}

template <typename Mask>
void test_count() {
  using mask_t = Mask;
  using repr_t = typename mask_t::repr_type;
  repr_t m;

  m = mask_t::empty();
  EXPECT_EQ(mask_t::count(m), 0);

  for (raw_pos_t i{}; i < mask_t::max_count; ++i) {
    m = mask_t::single(i);
    EXPECT_EQ(mask_t::count(m), 1);
    m = mask_t::step(m);
    EXPECT_EQ(mask_t::count(m), 0);
  }

  m = mask_t::full();
  for (raw_pos_t i{}; i < mask_t::max_count; ++i) {
    EXPECT_EQ(mask_t::count(m), mask_t::max_count - i);
    m = mask_t::step(m);
  }
  EXPECT_EQ(mask_t::count(m), 0);
}

template <typename Mask>
void test_next() {
  using mask_t = Mask;
  using repr_t = typename mask_t::repr_type;
  repr_t m;

  for (raw_pos_t i{}; i < mask_t::max_count; ++i) {
    m = mask_t::single(i);
    EXPECT_TRUE(mask_t::has_next(m));
    for (raw_pos_t j{}; j < mask_t::max_count; ++j) {
      const raw_pos_t off{j * 4711};
      EXPECT_EQ(mask_t::next(m, off), off + i);
    }
  }
}

template <typename Mask>
void test_truncate() {
  using mask_t = Mask;
  using repr_t = typename mask_t::repr_type;
  repr_t m;

  m = mask_t::empty();
  EXPECT_FALSE(mask_t::has_next(m));
  m = mask_t::truncate(m);
  EXPECT_FALSE(mask_t::has_next(m));
  m = mask_t::truncate(m);
  EXPECT_FALSE(mask_t::has_next(m));

  for (raw_pos_t i{}; i < mask_t::max_count; ++i) {
    m = mask_t::single(i);
    EXPECT_TRUE(mask_t::has_next(m));
    EXPECT_EQ(mask_t::next(m), i);
    m = mask_t::truncate(m);
    EXPECT_TRUE(mask_t::has_next(m));
    EXPECT_EQ(mask_t::next(m), i);
    m = mask_t::truncate(m);
    EXPECT_TRUE(mask_t::has_next(m));
    EXPECT_EQ(mask_t::next(m), i);
    m = mask_t::step(m);
    EXPECT_FALSE(mask_t::has_next(m));
  }

  for (int_t i{}; i < mask_t::max_count; ++i) {
    m = mask_t::full();
    EXPECT_TRUE(mask_t::has_next(m));
    EXPECT_EQ(mask_t::next(m), 0);

    for (int_t j{}; j < i;) {   
      m = mask_t::step(m);
      EXPECT_TRUE(mask_t::has_next(m));
      EXPECT_EQ(mask_t::next(m), ++j);
    }

    m = mask_t::truncate(m);
    EXPECT_TRUE(mask_t::has_next(m));
    EXPECT_EQ(mask_t::next(m), i);
    m = mask_t::truncate(m);
    EXPECT_TRUE(mask_t::has_next(m));
    EXPECT_EQ(mask_t::next(m), i);
    m = mask_t::step(m);
    EXPECT_FALSE(mask_t::has_next(m));
  }
}

template <typename Mask>
const static std::function<int_t()> size_gen{[]() {
  using mask_t = Mask;
  static std::uniform_int_distribution<int_t> dist(0, mask_t::max_count);
  return dist(rng);
}};

template <typename Mask>
const static std::function<int_t()> idx_gen{[]() {
  using mask_t = Mask;
  static std::uniform_int_distribution<int_t> dist(0, mask_t::max_count - 1);
  return dist(rng);
}};

const static std::function<bool()> bool_gen{[]() {
  static std::uniform_int_distribution<size_t> dist;
  return dist(rng) % 2 != 0;
}};

template <typename Mask>
void test_at(const int_t num_iter = 1024) {
  using mask_t = Mask;
  using repr_t = typename mask_t::repr_type;
  repr_t m;

  for (int_t k{}; k < num_iter; ++k) {
    bool x[mask_t::max_count];
    std::generate(std::begin(x), std::end(x), bool_gen);

    m = mask_t::empty();
    for (int_t i{}; i < mask_t::max_count; ++i) {
      if (x[i]) {
        m = mask_t::join(m, mask_t::single(i));
      }
    }

    for (int_t i{}; i < mask_t::max_count; ++i) {
      EXPECT_EQ(mask_t::at(m, i), x[i]);
    }
  }
}

template <typename Mask>
void test_above(const int_t num_iter = 1024) {
  using mask_t = Mask;
  using repr_t = typename mask_t::repr_type;
  repr_t m, n;

  for (int_t i{-1}; i < mask_t::max_count; ++i) {
    m = mask_t::above(mask_t::full(), i);

    for (int_t j{}; j < mask_t::max_count; ++j) {
      EXPECT_EQ(mask_t::at(m, j), j > i) << "i: " << i << ", j: " << j;
    }
  }

  for (int_t k{}; k < num_iter; ++k) {
    bool x[mask_t::max_count];
    std::generate(std::begin(x), std::end(x), bool_gen);

    m = mask_t::empty();
    for (int_t j{}; j < mask_t::max_count; ++j) {
      if (x[j]) {
        m = mask_t::join(m, mask_t::single(j));
      }
    }

    for (int_t i{-1}; i < mask_t::max_count; ++i) {
      n = mask_t::above(m, i);
  
      for (int_t j{}; j < mask_t::max_count; ++j) {
        EXPECT_EQ(mask_t::at(n, j), j > i && x[j]) << "i: " << i << ", j: " << j;
      }
    }
  }
}

template <typename Mask>
void test_join(const int_t num_iter = 1024) {
  using mask_t = Mask;
  using repr_t = typename mask_t::repr_type;
  repr_t m, n;

  m = mask_t::empty();
  n = mask_t::empty();
  EXPECT_FALSE(mask_t::has_next(m));
  EXPECT_FALSE(mask_t::has_next(n));
  m = mask_t::join(m, n);
  EXPECT_FALSE(mask_t::has_next(m));

  m = mask_t::full();
  n = mask_t::full();
  EXPECT_EQ(mask_t::count(m), mask_t::max_count);
  EXPECT_EQ(mask_t::count(n), mask_t::max_count);
  m = mask_t::join(m, n);
  EXPECT_EQ(mask_t::count(m), mask_t::max_count);

  bool x[mask_t::max_count];
  bool y[mask_t::max_count];
  bool z[mask_t::max_count];

  for (int_t i{}; i < num_iter; ++i) {
    const int_t m_max_size{size_gen<mask_t>()};
    const int_t n_max_size{size_gen<mask_t>()};

    m = mask_t::empty();
    for (int_t j{}; j < m_max_size; ++j) {
      m = mask_t::join(m, mask_t::single(idx_gen<mask_t>()));
    }
    mask_to_array<mask_t>(m, x);

    n = mask_t::empty();
    for (int_t j{}; j < n_max_size; ++j) {
      n = mask_t::join(n, mask_t::single(idx_gen<mask_t>()));
    }
    mask_to_array<mask_t>(n, y);

    m = mask_t::join(m, n);
    mask_to_array<mask_t>(m, z);
    for (int_t j{}; j < mask_t::max_count; ++j) {
      EXPECT_EQ(z[j], x[j] || y[j]);
    }
  }
}

template <typename Mask>
void test_intersect(const int_t num_iter = 1024) {
  using mask_t = Mask;
  using repr_t = typename mask_t::repr_type;
  repr_t m, n;

  m = mask_t::empty();
  n = mask_t::empty();
  EXPECT_FALSE(mask_t::has_next(m));
  EXPECT_FALSE(mask_t::has_next(n));
  m = mask_t::intersect(m, n);
  EXPECT_FALSE(mask_t::has_next(m));

  m = mask_t::full();
  n = mask_t::full();
  EXPECT_EQ(mask_t::count(m), mask_t::max_count);
  EXPECT_EQ(mask_t::count(n), mask_t::max_count);
  m = mask_t::intersect(m, n);
  EXPECT_EQ(mask_t::count(m), mask_t::max_count);

  bool x[mask_t::max_count];
  bool y[mask_t::max_count];
  bool z[mask_t::max_count];

  for (int_t i{}; i < num_iter; ++i) {
    const int_t m_max_size{size_gen<mask_t>()};
    const int_t n_max_size{size_gen<mask_t>()};

    m = mask_t::empty();
    for (int_t j{}; j < m_max_size; ++j) {
      m = mask_t::join(m, mask_t::single(idx_gen<mask_t>()));
    }
    mask_to_array<mask_t>(m, x);

    n = mask_t::empty();
    for (int_t j{}; j < n_max_size; ++j) {
      n = mask_t::join(n, mask_t::single(idx_gen<mask_t>()));
    }
    mask_to_array<mask_t>(n, y);

    m = mask_t::intersect(m, n);
    mask_to_array<mask_t>(m, z);
    for (int_t j{}; j < mask_t::max_count; ++j) {
      EXPECT_EQ(z[j], x[j] && y[j]);
    }
  }
}

#define TEST_MASKS_(_X_) \
  TEST(test_empty, _X_) { test_empty<_X_>(); } \
  TEST(test_full, _X_) { test_full<_X_>(); } \
  TEST(test_single, _X_) { test_single<_X_>(); } \
  TEST(test_contains, _X_) { test_contains<_X_>(); } \
  TEST(test_count, _X_) { test_count<_X_>(); } \
  TEST(test_next, _X_) { test_next<_X_>(); } \
  TEST(test_truncate, _X_) { test_truncate<_X_>(); } \
  TEST(test_at, _X_) { test_at<_X_>(); } \
  TEST(test_above, _X_) { test_above<_X_>(); } \
  TEST(test_join, _X_) { test_join<_X_>(); } \
  TEST(test_intersect, _X_) { test_intersect<_X_>(); }

// clang-format off
NVHM_FOR_EACH_(
  TEST_MASKS_,
  bitset_mask1_t,
  bitset_mask2_t,
  bitset_mask4_t,
  bitset_mask8_t,
  bitset_mask16_t,
  bitset_mask32_t,
  bitset_mask64_t,
  bitset_mask128_t,
  bitset_mask256_t,
  bitset_mask512_t
)

NVHM_FOR_EACH_(
  TEST_MASKS_,
  uint32_mask1_1r_t,
  uint32_mask1_1l_t,
  uint32_mask2_1r_t, uint32_mask2_2r_t,
  uint32_mask2_1l_t, uint32_mask2_2l_t,
  uint32_mask4_1r_t, uint32_mask4_2r_t, uint32_mask4_4r_t,
  uint32_mask4_1l_t, uint32_mask4_2l_t, uint32_mask4_4l_t,
  uint32_mask8_1r_t, uint32_mask8_2r_t, uint32_mask8_4r_t, uint32_mask8_8r_t,
  uint32_mask8_1l_t, uint32_mask8_2l_t, uint32_mask8_4l_t, uint32_mask8_8l_t,
  uint32_mask16_1r_t, uint32_mask16_2r_t, uint32_mask16_4r_t, uint32_mask16_8r_t,
  uint32_mask16_1l_t, uint32_mask16_2l_t, uint32_mask16_4l_t, uint32_mask16_8l_t,
  uint32_mask32_1r_t, uint32_mask32_2r_t, uint32_mask32_4r_t, uint32_mask32_8r_t,
  uint32_mask32_1l_t, uint32_mask32_2l_t, uint32_mask32_4l_t, uint32_mask32_8l_t
)

NVHM_FOR_EACH_(
  TEST_MASKS_,
  uint64_mask1_1r_t,
  uint64_mask1_1l_t,
  uint64_mask2_1r_t, uint64_mask2_2r_t,
  uint64_mask2_1l_t, uint64_mask2_2l_t,
  uint64_mask4_1r_t, uint64_mask4_2r_t, uint64_mask4_4r_t,
  uint64_mask4_1l_t, uint64_mask4_2l_t, uint64_mask4_4l_t,
  uint64_mask8_1r_t, uint64_mask8_2r_t, uint64_mask8_4r_t, uint64_mask8_8r_t,
  uint64_mask8_1l_t, uint64_mask8_2l_t, uint64_mask8_4l_t, uint64_mask8_8l_t,
  uint64_mask16_1r_t, uint64_mask16_2r_t, uint64_mask16_4r_t, uint64_mask16_8r_t,
  uint64_mask16_1l_t, uint64_mask16_2l_t, uint64_mask16_4l_t, uint64_mask16_8l_t,
  uint64_mask32_1r_t, uint64_mask32_2r_t, uint64_mask32_4r_t, uint64_mask32_8r_t,
  uint64_mask32_1l_t, uint64_mask32_2l_t, uint64_mask32_4l_t, uint64_mask32_8l_t,
  uint64_mask64_1r_t, uint64_mask64_2r_t, uint64_mask64_4r_t, uint64_mask64_8r_t,
  uint64_mask64_1l_t, uint64_mask64_2l_t, uint64_mask64_4l_t, uint64_mask64_8l_t
)

NVHM_FOR_EACH_(
  TEST_MASKS_,
  uint128_mask1_1r_t,
  uint128_mask1_1l_t,
  uint128_mask2_1r_t, uint128_mask2_2r_t,
  uint128_mask2_1l_t, uint128_mask2_2l_t,
  uint128_mask4_1r_t, uint128_mask4_2r_t, uint128_mask4_4r_t,
  uint128_mask4_1l_t, uint128_mask4_2l_t, uint128_mask4_4l_t,
  uint128_mask8_1r_t, uint128_mask8_2r_t, uint128_mask8_4r_t, uint128_mask8_8r_t,
  uint128_mask8_1l_t, uint128_mask8_2l_t, uint128_mask8_4l_t, uint128_mask8_8l_t,
  uint128_mask16_1r_t, uint128_mask16_2r_t, uint128_mask16_4r_t, uint128_mask16_8r_t,
  uint128_mask16_1l_t, uint128_mask16_2l_t, uint128_mask16_4l_t, uint128_mask16_8l_t,
  uint128_mask32_1r_t, uint128_mask32_2r_t, uint128_mask32_4r_t, uint128_mask32_8r_t,
  uint128_mask32_1l_t, uint128_mask32_2l_t, uint128_mask32_4l_t, uint128_mask32_8l_t,
  uint128_mask64_1r_t, uint128_mask64_2r_t, uint128_mask64_4r_t, uint128_mask64_8r_t,
  uint128_mask64_1l_t, uint128_mask64_2l_t, uint128_mask64_4l_t, uint128_mask64_8l_t,
  uint128_mask128_1r_t, uint128_mask128_2r_t, uint128_mask128_4r_t, uint128_mask128_8r_t,
  uint128_mask128_1l_t, uint128_mask128_2l_t, uint128_mask128_4l_t, uint128_mask128_8l_t
)

#if NVHM_WITH_SVE
NVHM_FOR_EACH_(
  TEST_MASKS_
#if NVHM_WITH_SVE_SIZE >= 1
  , sve_mask1_t
#endif
#if NVHM_WITH_SVE_SIZE >= 2
  , sve_mask2_t
#endif
#if NVHM_WITH_SVE_SIZE >= 4
  , sve_mask4_t
#endif
#if NVHM_WITH_SVE_SIZE >= 8
  , sve_mask8_t
#endif
#if NVHM_WITH_SVE_SIZE >= 16
  , sve_mask16_t
#endif
#if NVHM_WITH_SVE_SIZE >= 32
  , sve_mask32_t
#endif
#if NVHM_WITH_SVE_SIZE >= 64
  , sve_mask64_t
#endif
#if NVHM_WITH_SVE_SIZE >= 128
  , sve_mask128_t
#endif
#if NVHM_WITH_SVE_SIZE >= 256
  , sve_mask256_t
#endif
)
#endif
// clang-format on