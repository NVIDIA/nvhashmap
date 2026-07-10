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

#include <nvhashmap/probe_seq.hpp>
#include <nvhashmap/kernel.hpp>

#include "test_common.hpp"
#include <gtest/gtest.h>
#include <limits>
#include <map>

#include <set>
#include <type_traits>
#include <utility>
#include <vector>

using namespace nvhm;

static const std::function<raw_pos_t()> pos_gen{[]() {
  static std::uniform_int_distribution<raw_pos_t> dist(0, std::numeric_limits<raw_pos_t>::max());
  return dist(rng);
}};

template <typename Seq>
void test_probe_seq_step(int_t step_size, int_t num_iter = 1024) {
  using seq_t = Seq;

  bitmask_t m{0};
  m += bitmask_t{1} << countr_zero(m);
  EXPECT_TRUE(!m || has_single_bit(m));

  for (int_t i{}; i < num_iter; ++i) {
    seq_t seq(pos_gen());

    raw_pos_t n0{seq.next()};
    seq += step_size;
    raw_pos_t n1{seq.next()};

    if (step_size == 0) {
      EXPECT_EQ(n0, n1) << "step_size: " << step_size;
    } else {
      EXPECT_NE(n0, n1) << "step_size: " << step_size;
    }
  }
}

template <typename Seq>
void test_probe_seq_has_next(int_t step_size, int_t num_iter = 4096) {
  using seq_t = Seq;

  for (int_t i{}; i < num_iter; ++i) {
    seq_t seq{pos_gen()};

    for (int_t n{}; n < 4096; n += step_size) {
      if constexpr (seq_t::is_unbounded) {
        EXPECT_TRUE(seq.has_next());
      } else {
        EXPECT_EQ(seq.has_next(), n < seq_t::max_length) << "step_size: " << step_size << ", n: " << n << ", max_length: " << seq_t::max_length;
      }
      
      seq += step_size;
    }
  }
}

template <typename Seq>
void test_probe_seq_should_visit_all_slots(int_t step_size, int_t num_iter = 64) {
  using seq_t = Seq;

  int_t capacity;
  if constexpr (seq_t::is_bounded) {
    capacity = seq_t::max_length;
  } else {
    capacity = 4096;
  }
  NVHM_ASSERT_(has_single_bit(capacity));
  bitmask_t bucket_mask{make_aligned_mask(capacity, step_size)};

  for (int_t i{}; i < num_iter; ++i) {
    std::vector<bool> visited(to_uint(capacity), false);
    
    int_t j{1};
    for (seq_t seq{pos_gen()}; seq.has_next(); seq += step_size) {
      raw_pos_t off{align_pos(seq.next(), bucket_mask)};
      for (int_t i{}; i < step_size; ++i) {
        EXPECT_FALSE(visited.at(to_uint(off + i)));
        visited.at(to_uint(off + i)) = true;
      }
      if (all_of(visited.begin(), visited.end(), [](bool v) { return v; })) break;

      if (++j >= 1024 * 1024) break;
    }
    EXPECT_LT(j, 1024 * 1024) << "step_size: " << step_size;
    EXPECT_TRUE(all_of(visited.begin(), visited.end(), [](bool v) { return v; })) << "step_size: " << step_size;
    EXPECT_EQ(j * step_size, capacity) << "j: " << j << ", step_size: " << step_size;
  }
}

#define EVAL_PROBE_SEQUENCES_(_X_) \
  TEST(test_probe_seq_step_0, _X_) { test_probe_seq_step<_X_>(0); }\
  TEST(test_probe_seq_step_1, _X_) { test_probe_seq_step<_X_>(1); }\
  TEST(test_probe_seq_step_2, _X_) { test_probe_seq_step<_X_>(2); }\
  TEST(test_probe_seq_step_4, _X_) { test_probe_seq_step<_X_>(4); }\
  TEST(test_probe_seq_step_8, _X_) { test_probe_seq_step<_X_>(8); }\
  TEST(test_probe_seq_step_16, _X_) { test_probe_seq_step<_X_>(16); }\
  TEST(test_probe_seq_step_32, _X_) { test_probe_seq_step<_X_>(32); }\
  TEST(test_probe_seq_step_64, _X_) { test_probe_seq_step<_X_>(64); }\
  TEST(test_probe_seq_step_128, _X_) { test_probe_seq_step<_X_>(128); }\
  TEST(test_probe_seq_has_next_1, _X_) { test_probe_seq_has_next<_X_>(1); }\
  TEST(test_probe_seq_has_next_2, _X_) { test_probe_seq_has_next<_X_>(2); }\
  TEST(test_probe_seq_has_next_4, _X_) { test_probe_seq_has_next<_X_>(4); }\
  TEST(test_probe_seq_has_next_8, _X_) { test_probe_seq_has_next<_X_>(8); }\
  TEST(test_probe_seq_has_next_16, _X_) { test_probe_seq_has_next<_X_>(16); }\
  TEST(test_probe_seq_has_next_32, _X_) { test_probe_seq_has_next<_X_>(32); }\
  TEST(test_probe_seq_has_next_64, _X_) { test_probe_seq_has_next<_X_>(64); }\
  TEST(test_probe_seq_has_next_128, _X_) { test_probe_seq_has_next<_X_>(128); }\
  TEST(test_probe_seq_should_visit_all_slots_1, _X_) { test_probe_seq_should_visit_all_slots<_X_>(1); }\
  TEST(test_probe_seq_should_visit_all_slots_2, _X_) { test_probe_seq_should_visit_all_slots<_X_>(2); }\
  TEST(test_probe_seq_should_visit_all_slots_4, _X_) { test_probe_seq_should_visit_all_slots<_X_>(4); }\
  TEST(test_probe_seq_should_visit_all_slots_8, _X_) { test_probe_seq_should_visit_all_slots<_X_>(8); }\
  TEST(test_probe_seq_should_visit_all_slots_16, _X_) { test_probe_seq_should_visit_all_slots<_X_>(16); }\
  TEST(test_probe_seq_should_visit_all_slots_32, _X_) { test_probe_seq_should_visit_all_slots<_X_>(32); }\
  TEST(test_probe_seq_should_visit_all_slots_64, _X_) { test_probe_seq_should_visit_all_slots<_X_>(64); }\
  TEST(test_probe_seq_should_visit_all_slots_128, _X_) { test_probe_seq_should_visit_all_slots<_X_>(128); }

using linear_seq___0_inf_t = linear_seq<0>;
using linear_seq___1_inf_t = linear_seq<1>;
using linear_seq___2_inf_t = linear_seq<2>;
using linear_seq___4_inf_t = linear_seq<4>;
using linear_seq___8_inf_t = linear_seq<8>;
using linear_seq__16_inf_t = linear_seq<16>;
using linear_seq__32_inf_t = linear_seq<32>;
using linear_seq__64_inf_t = linear_seq<64>;
using linear_seq_128_inf_t = linear_seq<128>;

using linear_seq___0_128_t = linear_seq<0, 128>;
using linear_seq___1_128_t = linear_seq<1, 128>;
using linear_seq___2_128_t = linear_seq<2, 128>;
using linear_seq___4_128_t = linear_seq<4, 128>;
using linear_seq___8_128_t = linear_seq<8, 128>;
using linear_seq__16_128_t = linear_seq<16, 128>;
using linear_seq__32_128_t = linear_seq<32, 128>;
using linear_seq__64_128_t = linear_seq<64, 128>;
using linear_seq_128_128_t = linear_seq<128, 128>;

using linear_seq___0__1k_t = linear_seq<0, 1024>;
using linear_seq___1__1k_t = linear_seq<1, 1024>;
using linear_seq___2__1k_t = linear_seq<2, 1024>;
using linear_seq___4__1k_t = linear_seq<4, 1024>;
using linear_seq___8__1k_t = linear_seq<8, 1024>;
using linear_seq__16__1k_t = linear_seq<16, 1024>;
using linear_seq__32__1k_t = linear_seq<32, 1024>;
using linear_seq__64__1k_t = linear_seq<64, 1024>;
using linear_seq_128__1k_t = linear_seq<128, 1024>;

using quadratic_seq___0_inf_t = quadratic_seq<0>;
using quadratic_seq___1_inf_t = quadratic_seq<1>;
using quadratic_seq___2_inf_t = quadratic_seq<2>;
using quadratic_seq___4_inf_t = quadratic_seq<4>;
using quadratic_seq___8_inf_t = quadratic_seq<8>;
using quadratic_seq__16_inf_t = quadratic_seq<16>;
using quadratic_seq__32_inf_t = quadratic_seq<32>;
using quadratic_seq__64_inf_t = quadratic_seq<64>;
using quadratic_seq_128_inf_t = quadratic_seq<128>;

using quadratic_seq___0_128_t = quadratic_seq<0, 128>;
using quadratic_seq___1_128_t = quadratic_seq<1, 128>;
using quadratic_seq___2_128_t = quadratic_seq<2, 128>;
using quadratic_seq___4_128_t = quadratic_seq<4, 128>;
using quadratic_seq___8_128_t = quadratic_seq<8, 128>;
using quadratic_seq__16_128_t = quadratic_seq<16, 128>;
using quadratic_seq__32_128_t = quadratic_seq<32, 128>;
using quadratic_seq__64_128_t = quadratic_seq<64, 128>;
using quadratic_seq_128_128_t = quadratic_seq<128, 128>;

using quadratic_seq___0__1k_t = quadratic_seq<0, 1024>;
using quadratic_seq___1__1k_t = quadratic_seq<1, 1024>;
using quadratic_seq___2__1k_t = quadratic_seq<2, 1024>;
using quadratic_seq___4__1k_t = quadratic_seq<4, 1024>;
using quadratic_seq___8__1k_t = quadratic_seq<8, 1024>;
using quadratic_seq__16__1k_t = quadratic_seq<16, 1024>;
using quadratic_seq__32__1k_t = quadratic_seq<32, 1024>;
using quadratic_seq__64__1k_t = quadratic_seq<64, 1024>;
using quadratic_seq_128__1k_t = quadratic_seq<128, 1024>;

NVHM_FOR_EACH_(
  EVAL_PROBE_SEQUENCES_,
  linear_seq___0_inf_t, linear_seq___0_128_t, linear_seq___0__1k_t,
  linear_seq___1_inf_t, linear_seq___1_128_t, linear_seq___1__1k_t,
  linear_seq___2_inf_t, linear_seq___2_128_t, linear_seq___2__1k_t,
  linear_seq___4_inf_t, linear_seq___4_128_t, linear_seq___4__1k_t,
  linear_seq___8_inf_t, linear_seq___8_128_t, linear_seq___8__1k_t,
  linear_seq__16_inf_t, linear_seq__16_128_t, linear_seq__16__1k_t,
  linear_seq__32_inf_t, linear_seq__32_128_t, linear_seq__32__1k_t,
  linear_seq__64_inf_t, linear_seq__64_128_t, linear_seq__64__1k_t,
  linear_seq_128_inf_t, linear_seq_128_128_t, linear_seq_128__1k_t,
  quadratic_seq___0_inf_t, quadratic_seq___0_128_t, quadratic_seq___0__1k_t,
  quadratic_seq___1_inf_t, quadratic_seq___1_128_t, quadratic_seq___1__1k_t,
  quadratic_seq___2_inf_t, quadratic_seq___2_128_t, quadratic_seq___2__1k_t,
  quadratic_seq___4_inf_t, quadratic_seq___4_128_t, quadratic_seq___4__1k_t,
  quadratic_seq___8_inf_t, quadratic_seq___8_128_t, quadratic_seq___8__1k_t,
  quadratic_seq__16_inf_t, quadratic_seq__16_128_t, quadratic_seq__16__1k_t,
  quadratic_seq__32_inf_t, quadratic_seq__32_128_t, quadratic_seq__32__1k_t,
  quadratic_seq__64_inf_t, quadratic_seq__64_128_t, quadratic_seq__64__1k_t,
  quadratic_seq_128_inf_t, quadratic_seq_128_128_t, quadratic_seq_128__1k_t
);

// Walk a probe sequence for `n` steps and return the visited offsets.
template <typename Seq>
std::vector<raw_pos_t> walk(int_t capacity, int_t step_size, raw_pos_t pos, psl_t max_psl) {
  using seq_t = Seq;

  std::vector<raw_pos_t> res;
  res.reserve(to_uint(max_psl));

  const bitmask_t bucket_mask{make_aligned_mask(capacity, step_size)};
  for (seq_t seq{pos}; seq.has_next(); seq += step_size) {
    if (seq.psl() >= max_psl) break;

    raw_pos_t off{align_pos(seq.next(), bucket_mask)};
    res.emplace_back(off);
  }

  return res;
}

template <typename Seq>
void test_probe_seq_correctness(int_t kernel_size) {
  using seq_t = Seq;

  constexpr static int_t capacity{256};
  constexpr static raw_pos_t pos{511};
  
  auto offs{walk<seq_t>(capacity, kernel_size, pos, capacity)};

  std::vector<raw_pos_t> ref_offs;
  if (kernel_size == 32) {
    if constexpr (std::is_same_v<seq_t, linear_seq___0_inf_t>) {
      ref_offs = {224, 0, 32, 64, 96, 128, 160, 192};
    } else if constexpr (std::is_same_v<seq_t, quadratic_seq___0_inf_t>) {
      ref_offs = {224, 0, 64, 160, 32, 192, 128, 96};
    } else if constexpr (std::is_same_v<seq_t, linear_seq__64_inf_t>) {
      ref_offs = {224, 192, 32, 0, 96, 64, 160, 128};
    } else if constexpr (std::is_same_v<seq_t, quadratic_seq__64_inf_t>) {
      ref_offs = {224, 192, 32, 0, 160, 128, 96, 64};
    } else {
      static_assert(dependent_false_v<seq_t>);
    }
  } else if (kernel_size == 16) {
    if constexpr (std::is_same_v<seq_t, linear_seq___0_inf_t>) {
      ref_offs = {240, 0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224};
    } else if constexpr (std::is_same_v<seq_t, quadratic_seq___0_inf_t>) {
      ref_offs = {240, 0, 32, 80, 144, 224, 64, 176, 48, 192, 96, 16, 208, 160, 128, 112};
    } else if constexpr (std::is_same_v<seq_t, linear_seq__64_inf_t>) {
      ref_offs = {240, 192, 208, 224, 48, 0, 16, 32, 112, 64, 80, 96, 176, 128, 144, 160};
    } else if constexpr (std::is_same_v<seq_t, quadratic_seq__64_inf_t>) {
      ref_offs = {240, 192, 208, 224, 48, 0, 16, 32, 176, 128, 144, 160, 112, 64, 80, 96};
    } else {
      static_assert(dependent_false_v<seq_t>);
    }
  }

  EXPECT_EQ(offs, ref_offs);
}

TEST(test_probe_seq_correctness_32, linear_seq___0_inf_t) { test_probe_seq_correctness<linear_seq___0_inf_t>(32); }
TEST(test_probe_seq_correctness_32, linear_seq__64_inf_t) { test_probe_seq_correctness<linear_seq__64_inf_t>(32); }
TEST(test_probe_seq_correctness_32, quadratic_seq___0_inf_t) { test_probe_seq_correctness<quadratic_seq___0_inf_t>(32); }
TEST(test_probe_seq_correctness_32, quadratic_seq__64_inf_t) { test_probe_seq_correctness<quadratic_seq__64_inf_t>(32); }

TEST(test_probe_seq_correctness_16, linear_seq___0_inf_t) { test_probe_seq_correctness<linear_seq___0_inf_t>(16); }
TEST(test_probe_seq_correctness_16, linear_seq__64_inf_t) { test_probe_seq_correctness<linear_seq__64_inf_t>(16); }
TEST(test_probe_seq_correctness_16, quadratic_seq___0_inf_t) { test_probe_seq_correctness<quadratic_seq___0_inf_t>(16); }
TEST(test_probe_seq_correctness_16, quadratic_seq__64_inf_t) { test_probe_seq_correctness<quadratic_seq__64_inf_t>(16); }


template <typename UnalignedSeq, typename AlignedSeq>
void test_probe_seq_symmetry() {
  constexpr static int_t capacity{1024};
  constexpr static raw_pos_t pos{511};

  for (int_t kernel_size : {1, 2, 4, 8, 16, 32, 64, 128}) {
    auto offs_unaligned{walk<UnalignedSeq>(capacity, kernel_size, pos, capacity)};
    auto offs_aligned{walk<AlignedSeq>(capacity, kernel_size, pos, capacity)};

    EXPECT_EQ(offs_unaligned, offs_aligned);
  }
}

TEST(test_probe_seq_symmetry, linear_seq___0_inf_t) { test_probe_seq_symmetry<linear_seq___0_inf_t, linear_seq___1_inf_t>(); }
TEST(test_probe_seq_symmetry, linear_seq___0_128_t) { test_probe_seq_symmetry<linear_seq___0_128_t, linear_seq___1_128_t>(); }
TEST(test_probe_seq_symmetry, linear_seq___0__1k_t) { test_probe_seq_symmetry<linear_seq___0__1k_t, linear_seq___1__1k_t>(); }
TEST(test_probe_seq_symmetry, quadratic_seq___0_inf_t) { test_probe_seq_symmetry<quadratic_seq___0_inf_t, quadratic_seq___1_inf_t>(); }
TEST(test_probe_seq_symmetry, quadratic_seq___0_128_t) { test_probe_seq_symmetry<quadratic_seq___0_128_t, quadratic_seq___1_128_t>(); }
TEST(test_probe_seq_symmetry, quadratic_seq___0__1k_t) { test_probe_seq_symmetry<quadratic_seq___0__1k_t, quadratic_seq___1__1k_t>(); }
