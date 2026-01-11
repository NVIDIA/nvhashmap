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

using namespace nvhm;

template <typename Mask>
void test_masking() {
  using mask_t = Mask;

  auto mask{mask_t::empty()};
  EXPECT_FALSE(mask_t::has_next(mask));

  mask = mask_t::full();
  EXPECT_EQ(mask_t::count(mask), mask_t::max_count);

  for (size_t i{}; i != mask_t::max_count; ++i) {
    EXPECT_TRUE(mask_t::has_next(mask));
    EXPECT_EQ(mask_t::next(mask), i);
    EXPECT_EQ(mask_t::next(mask, 4711), i + 4711);

    auto trunc_mask{mask_t::truncate(mask)};
    EXPECT_EQ(mask_t::next(trunc_mask), mask_t::next(mask));
    trunc_mask = mask_t::step(trunc_mask);
    EXPECT_FALSE(mask_t::has_next(trunc_mask));

    mask = mask_t::step(mask);
    if (!mask_t::has_next(mask)) {
      break;
    }
  }
}

#define TEST_MASKING_(_X_) \
  TEST(test_masking, _X_) { test_masking<_X_>(); }

NVHM_FOR_EACH_(
  TEST_MASKING_,
  // uint mask8
  uint_mask8_1_t, uint_mask8_2r_t, uint_mask8_2l_t, uint_mask8_4r_t, uint_mask8_4l_t,
  uint_mask8_8r_t, uint_mask8_8l_t,
  // uint mask16
  uint_mask16_1_t, uint_mask16_2r_t, uint_mask16_2l_t, uint_mask16_4r_t, uint_mask16_4l_t,
  uint_mask16_8r_t, uint_mask16_8l_t,
  // uint mask32
  uint_mask32_1_t, uint_mask32_2r_t, uint_mask32_2l_t, uint_mask32_4r_t, uint_mask32_4l_t,
  uint_mask32_8r_t, uint_mask32_8l_t,
  // uint mask64
  uint_mask64_1_t, uint_mask64_2r_t, uint_mask64_2l_t, uint_mask64_4r_t, uint_mask64_4l_t,
  uint_mask64_8r_t, uint_mask64_8l_t,
  // uint mask128
  uint_mask128_1_t, uint_mask128_2r_t, uint_mask128_2l_t, uint_mask128_4r_t, uint_mask128_4l_t,
  uint_mask128_8r_t, uint_mask128_8l_t
#if NVHM_WITH_SVE
  ,
  sve_mask8_t, sve_mask16_t, sve_mask32_t, sve_mask64_t, sve_mask128_t
#if __ARM_FEATURE_SVE_BITS >= 256
  ,
  sve_mask256_t
#endif
#if __ARM_FEATURE_SVE_BITS >= 512
  ,
  sve_mask512_t
#endif
#if __ARM_FEATURE_SVE_BITS >= 1024
  ,
  sve_mask1024_t
#endif
#if __ARM_FEATURE_SVE_BITS >= 2048
  ,
  sve_mask2048_t
#endif
  ,
  sve_mask_t
#endif
);
