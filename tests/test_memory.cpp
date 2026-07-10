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

#include <nvhashmap/memory.hpp>

#include "test_common.hpp"
#include <random>
#include <gtest/gtest.h>

using namespace nvhm;

const static std::function<blob_t()> blob_gen{[]() {
  static std::uniform_int_distribution<uint8_t> dist;
  return static_cast<blob_t>(dist(rng));
}};

template <int_t N>
void test_copy_blob(const int_t num_iter = 4096) {
  std::array<blob_t, N> src, dst;

  for (int_t i{}; i < num_iter; ++i) {
    std::generate(src.begin(), src.end(), blob_gen);

    std::transform(src.begin(), src.end(), dst.begin(), [](std::byte b) { return b ^ std::byte{0x55}; });
    EXPECT_NE(src, dst);

    copy_blob(dst.data(), src.data(), N);
    EXPECT_EQ(src, dst);
  }
}

TEST(test_copy_bytes_blob,    1) { test_copy_blob<   1>(); }
TEST(test_copy_bytes_blob,    2) { test_copy_blob<   2>(); }
TEST(test_copy_bytes_blob,    3) { test_copy_blob<   3>(); }
TEST(test_copy_bytes_blob,    4) { test_copy_blob<   4>(); }
TEST(test_copy_bytes_blob,    5) { test_copy_blob<   5>(); }
TEST(test_copy_bytes_blob,    6) { test_copy_blob<   6>(); }
TEST(test_copy_bytes_blob,    7) { test_copy_blob<   7>(); }
TEST(test_copy_bytes_blob,    8) { test_copy_blob<   8>(); }
TEST(test_copy_bytes_blob,    9) { test_copy_blob<   9>(); }
TEST(test_copy_bytes_blob,   10) { test_copy_blob<  10>(); }
TEST(test_copy_bytes_blob,   11) { test_copy_blob<  11>(); }
TEST(test_copy_bytes_blob,   12) { test_copy_blob<  12>(); }
TEST(test_copy_bytes_blob,   13) { test_copy_blob<  13>(); }
TEST(test_copy_bytes_blob,   14) { test_copy_blob<  14>(); }
TEST(test_copy_bytes_blob,   15) { test_copy_blob<  15>(); }
TEST(test_copy_bytes_blob,   16) { test_copy_blob<  16>(); }
TEST(test_copy_bytes_blob,   32) { test_copy_blob<  32>(); }
TEST(test_copy_bytes_blob,   64) { test_copy_blob<  64>(); }
TEST(test_copy_bytes_blob,  128) { test_copy_blob< 128>(); }
TEST(test_copy_bytes_blob,  256) { test_copy_blob< 256>(); }
TEST(test_copy_bytes_blob,  512) { test_copy_blob< 512>(); }
TEST(test_copy_bytes_blob, 1024) { test_copy_blob<1024>(); }
TEST(test_copy_bytes_blob, 1337) { test_copy_blob<1337>(); }
TEST(test_copy_bytes_blob, 4096) { test_copy_blob<4096>(); }
TEST(test_copy_bytes_blob, 4711) { test_copy_blob<4711>(); }
