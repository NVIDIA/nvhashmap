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

#include <nvhashmap/allocator.hpp>
#include <nvhashmap/kernel.hpp>

#include "test_common.hpp"
#include <algorithm>
#include <gtest/gtest.h>
#include <random>
#include <iomanip>

using namespace nvhm;

template <typename Kernel>
void test_kernel_masking() {
  using kernel_t = Kernel;
  using mask_t = typename kernel_t::mask_type;
  using mask_repr_t = typename mask_t::repr_type;

  constexpr state_t empty_slot{kernel_t::empty};
  constexpr state_t tombstone_slot{kernel_t::tombstone};

  auto slots{default_allocator<>::template alloc<state_t>(kernel_t::size)};

  for (int f{-128}; f < 128; ++f) {
    std::fill_n(slots.get(), kernel_t::size, f);

    auto kern{kernel_t::load(slots.get())};
    mask_repr_t mask;

    // Explicitly test for match.
    for (int m{-128}; m < 128; ++m) {
      mask = kernel_t::mask(kern, static_cast<state_t>(m));
      if (m == f) {
        EXPECT_EQ(mask_t::count(mask), kernel_t::size);
        for (size_t i{}; mask_t::has_next(mask); mask = mask_t::step(mask)) {
          EXPECT_EQ(mask_t::next(mask), i++);
        }
        EXPECT_EQ(mask_t::has_next(mask), false);
      } else {
        EXPECT_EQ(mask_t::count(mask), 0);
        EXPECT_EQ(mask_t::has_next(mask), false);
      }
    }

    // Test hash / non-hash.
    mask = kernel_t::mask_hash(kern);
    if (f >= 0) {
      EXPECT_EQ(mask_t::count(mask), kernel_t::size);
      for (size_t i{}; mask_t::has_next(mask); mask = mask_t::step(mask)) {
        EXPECT_EQ(mask_t::next(mask), i++);
      }
      EXPECT_EQ(mask_t::has_next(mask), false);
    } else {
      EXPECT_EQ(mask_t::count(mask), 0);
      EXPECT_EQ(mask_t::has_next(mask), false);
    }

    mask = kernel_t::mask_non_hash(kern);
    if (f < 0) {
      EXPECT_EQ(mask_t::count(mask), kernel_t::size);
      for (size_t i{}; mask_t::has_next(mask); mask = mask_t::step(mask)) {
        EXPECT_EQ(mask_t::next(mask), i++);
      }
      EXPECT_EQ(mask_t::has_next(mask), false);
    } else {
      EXPECT_EQ(mask_t::count(mask), 0);
      EXPECT_EQ(mask_t::has_next(mask), false);
    }

    // Test special values.
    mask = kernel_t::mask_empty(kern);
    if (f == empty_slot) {
      EXPECT_EQ(mask_t::count(mask), kernel_t::size);
      for (size_t i{}; mask_t::has_next(mask); mask = mask_t::step(mask)) {
        EXPECT_EQ(mask_t::next(mask), i++);
      }
      EXPECT_EQ(mask_t::has_next(mask), false);
    } else if (f == tombstone_slot || f >= 0) {
      EXPECT_EQ(mask_t::count(mask), 0);
      EXPECT_EQ(mask_t::has_next(mask), false);
    }

    mask = kernel_t::mask_tombstone(kern);
    if (f == tombstone_slot) {
      EXPECT_EQ(mask_t::count(mask), kernel_t::size);
      for (size_t i{}; mask_t::has_next(mask); mask = mask_t::step(mask)) {
        EXPECT_EQ(mask_t::next(mask), i++);
      }
      EXPECT_EQ(mask_t::has_next(mask), false);
    } else if (f == empty_slot || f >= 0) {
      EXPECT_EQ(mask_t::count(mask), 0);
      EXPECT_EQ(mask_t::has_next(mask), false);
    }

    // Fast conversion.
    kern = kernel_t::hash_to_tombstone(kern);
    mask = kernel_t::mask_tombstone(kern);
    if (f >= 0 || f == tombstone_slot) {
      EXPECT_EQ(mask_t::count(mask), kernel_t::size);
      for (size_t i{}; mask_t::has_next(mask); mask = mask_t::step(mask)) {
        EXPECT_EQ(mask_t::next(mask), i++);
      }
      EXPECT_EQ(mask_t::has_next(mask), false);
    } else if (f == empty_slot) {
      EXPECT_EQ(mask_t::count(mask), 0);
      EXPECT_EQ(mask_t::has_next(mask), false);
    }
  }

  for (size_t i{}; i < kernel_t::size; ++i) {
    slots[i] = static_cast<state_t>(100 + i);
  }
  auto kern{kernel_t::load(slots.get())};
  for (size_t i{}; i < kernel_t::size; ++i) {
    EXPECT_EQ(kernel_t::at(kern, i), slots[i]);
  }
}

#define TEST_KERNEL_MASKING_(_X_) \
  TEST(test_kernel_masking, _X_) { test_kernel_masking<_X_>(); }

NVHM_FOR_EACH_(
  TEST_KERNEL_MASKING_,
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

static std::random_device rd;

template <typename Kernel, size_t MaxIter, bool Verbose>
void eval_random_values() {
  using kern_t = Kernel;
  using kern_repr_t = typename kern_t::repr_type;
  using mask_t = typename kern_t::mask_type;
  using mask_repr_t = typename mask_t::repr_type;

  using ref_kern_t = array_kernel<kern_t::size>;
  using ref_kern_repr_t = typename ref_kern_t::repr_type;
  using ref_mask_t = typename ref_kern_t::mask_type;
  using ref_mask_repr_t = typename ref_mask_t::repr_type;

  auto compare_masks{[](ref_mask_repr_t m0, mask_repr_t m1, bool allow_false_positives = false) {
    if (allow_false_positives && ref_mask_t::count(m0) != mask_t::count(m1)) {
      EXPECT_LE(ref_mask_t::count(m0), mask_t::count(m1));
      while (ref_mask_t::has_next(m0) && mask_t::has_next(m1)) {
        if (ref_mask_t::next(m0) == mask_t::next(m1)) {
          m0 = ref_mask_t::step(m0);
          m1 = mask_t::step(m1);
        } else {
          m1 = mask_t::step(m1);
        }
      }
      EXPECT_FALSE(ref_mask_t::has_next(m0));
    } else {
      // std::cout << "cnt: " << ref_mask_t::count(m0) << " ~ " << mask_t::count(m1) << '\n';
      EXPECT_EQ(ref_mask_t::count(m0), mask_t::count(m1));
      while (ref_mask_t::has_next(m0) && mask_t::has_next(m1)) {
        // std::cout << "nxt: " << ref_mask_t::next(m0) << " ~ " << mask_t::next(m1) << '\n';
        EXPECT_EQ(ref_mask_t::next(m0), mask_t::next(m1));

        m0 = ref_mask_t::step(m0);
        m1 = mask_t::step(m1);
      }
      EXPECT_FALSE(ref_mask_t::has_next(m0));
      EXPECT_FALSE(mask_t::has_next(m1));
    }
  }};

  auto compare_kern{[](ref_kern_repr_t k0, kern_repr_t k1) {
    for (size_t j{}; j < kern_t::size; ++j) {
      switch (kern_t::at(k1, j)) {
        case kern_t::empty: {
          EXPECT_EQ(ref_kern_t::at(k0, j), ref_kern_t::empty);
        } break;
        case kern_t::tombstone: {
          EXPECT_EQ(ref_kern_t::at(k0, j), ref_kern_t::tombstone);
        } break;
        default: {
          EXPECT_EQ(ref_kern_t::at(k0, j), kern_t::at(k1, j));
        } break;
      }
    }
  }};

  std::mt19937 gen(rd());
  std::uniform_int_distribution<state_t> state_dist(-20, 127);
  std::uniform_int_distribution<size_t> idx_dist(0, kern_t::size - 1);

  for (size_t i{}; i < MaxIter; ++i) {
    alignas(kern_t::size) ref_kern_repr_t ref_buf, kern_buf;
    for (size_t j{}; j < ref_kern_t::size; ++j) {
      const state_t s{state_dist(gen)};
      if (s < -10) {
        ref_buf[j] = ref_kern_t::empty;
        kern_buf[j] = kern_t::empty;
      } else if (s < 0) {
        ref_buf[j] = ref_kern_t::tombstone;
        kern_buf[j] = kern_t::tombstone;
      } else {
        ref_buf[j] = s;
        kern_buf[j] = s;
      }
    }

    // Load.
    ref_kern_repr_t r{ref_kern_t::load(ref_buf.begin())};
    kern_repr_t k{kern_t::load(kern_buf.begin())};

    // At.
    if constexpr (Verbose) {
      std::cout << std::hex;
    }
    for (size_t j{}; j < kern_t::size; ++j) {
      EXPECT_EQ(ref_kern_t::at(r, j), ref_kern_t::at(ref_buf, j));
      if constexpr (Verbose) {
        std::cout << (kern_t::at(k, j) & 0xff) << ' ';
      }
    }
    if constexpr (Verbose) {
      std::cout << '\n';
    }
    compare_kern(r, k);

    // Mask.
    ref_mask_repr_t r_mask;
    mask_repr_t k_mask;

    r_mask = ref_kern_t::mask_hash(r);
    k_mask = kern_t::mask_hash(k);
    //if constexpr (Verbose) {
    //  std::cout << "mask_hash " << std::hex << render_mask<ref_mask_t>(r_mask) << " ~ "
    //            << render_mask<mask_t>(k_mask) << '\n';
    //}
    compare_masks(r_mask, k_mask);

    r_mask = ref_kern_t::mask_non_hash(r);
    k_mask = kern_t::mask_non_hash(k);
    //if constexpr (Verbose) {
    //  std::cout << "mask_non_hash " << std::hex << render_mask<ref_mask_t>(r_mask) << " ~ "
    //            << render_mask<mask_t>(k_mask) << '\n';
    //}
    compare_masks(r_mask, k_mask);

    r_mask = ref_kern_t::mask_empty(r);
    k_mask = kern_t::mask_empty(k);
    //if constexpr (Verbose) {
    //  std::cout << "mask_empty " << std::hex << render_mask<ref_mask_t>(r_mask) << " ~ "
    //            << render_mask<mask_t>(k_mask) << '\n';
    //}
    compare_masks(r_mask, k_mask);

    r_mask = ref_kern_t::mask_tombstone(r);
    k_mask = kern_t::mask_tombstone(k);
    //if constexpr (Verbose) {
    //  std::cout << "mask_tombstone " << std::hex << render_mask<ref_mask_t>(r_mask) << " ~ "
    //            << render_mask<mask_t>(k_mask) << '\n';
    //}
    compare_masks(r_mask, k_mask);

    size_t buf_idx{idx_dist(gen)};
    r_mask = ref_kern_t::mask(r, ref_buf[buf_idx]);
    k_mask = kern_t::mask(k, kern_buf[buf_idx]);
    //if constexpr (Verbose) {
    //  std::cout << "mask[" << buf_idx << ", " << static_cast<int>(ref_buf[buf_idx]) << " / "
    //            << static_cast<int>(kern_buf[buf_idx]) << "] ";
    //  std::cout << render_mask<ref_mask_t>(r_mask) << " ~ " << render_mask<mask_t>(k_mask) << '\n';
    //}
    compare_masks(r_mask, k_mask, allow_false_positive_matches);

    r_mask = ref_kern_t::mask(r, 0x19);
    k_mask = kern_t::mask(k, 0x19);
    //if constexpr (Verbose) {
    //  std::cout << "mask[19] " << render_mask<ref_mask_t>(r_mask) << " ~ " << render_mask<mask_t>(k_mask)
    //            << '\n';
    //}
    compare_masks(r_mask, k_mask, allow_false_positive_matches);

    // Has empty
    const bool r_empty{ref_kern_t::has_empty(r)};
    const bool k_empty{kern_t::has_empty(k)};
    //if constexpr (Verbose) {
    //  std::cout << "has_empty " << render_mask<ref_mask_t>(r_mask) << " ~ " << render_mask<mask_t>(k_mask)
    //            << " | ";
    //  std::cout << r_empty << " ~ " << k_empty << '\n';
    //}
    EXPECT_EQ(r_empty, k_empty);

    // Hashes_to_tombstones
    const ref_kern_repr_t r2{ref_kern_t::hash_to_tombstone(r)};
    const kern_repr_t k2{kern_t::hash_to_tombstone(k)};
    if constexpr (Verbose) {
      std::cout << "hashes_to_tombstones " << '\n';
    }
    compare_kern(r2, k2);

    // Min LRU.
    const size_t r_min{ref_kern_t::min_lru_index(r)};
    const size_t k_min{kern_t::min_lru_index(k)};
    EXPECT_EQ(r_min, k_min);
    if constexpr (Verbose) {
      std::cout << "min_lru " << r_min << " ~ " << k_min << '\n';
    }
  }

  alignas(kern_t::size) ref_kern_repr_t k_buf, l_buf;
  ref_mask_repr_t r_m;
  do {
    std::generate(k_buf.begin(), k_buf.end(), [&]() -> state_t { return state_dist(gen); });
    r_m = ref_kern_t::mask_hash(ref_kern_t::load(k_buf.begin()));
  } while (!ref_mask_t::has_next(r_m));

  std::generate(l_buf.begin(), l_buf.end(), [&]() -> state_t { return state_dist(gen); });

  const ref_kern_repr_t r_k{ref_kern_t::load(k_buf.begin())};
  const kern_repr_t k_k{kern_t::load(k_buf.begin())};

  mask_repr_t k_m{kern_t::mask_hash(k_k)};
  EXPECT_EQ(ref_mask_t::count(r_m), mask_t::count(k_m));

  ref_kern_repr_t r_l{ref_kern_t::load(l_buf.begin())};
  kern_repr_t k_l{kern_t::load(l_buf.begin())};

#if defined(NDEBUG)
  // Technically speaking the implementation should work for any mask index (even if that doesn't
  // make sense in practice). The debug mode sanity checks in the kernels prevent it however.
  r_m = ref_mask_t::full();
  k_m = mask_t::full();
#endif
  //if constexpr (Verbose) {
  //  std::cout << "r_m (count = " << ref_mask_t::count(r_m) << "): " << render_mask<ref_mask_t>(r_m)
  //            << '\n';
  //  std::cout << "k_m (count = " << mask_t::count(k_m) << "): " << render_mask<mask_t>(k_m) << '\n';
  //}

  // LRU update.
  for (; ref_mask_t::has_next(r_m) && mask_t::has_next(k_m);
       r_m = ref_mask_t::step(r_m), k_m = mask_t::step(k_m)) {
    const size_t inc_idx{mask_t::next(k_m)};
    EXPECT_LT(inc_idx, kern_t::size);
    EXPECT_EQ(inc_idx, ref_mask_t::next(r_m));

    for (size_t n{}; n < 256; ++n) {
      if constexpr (Verbose) {
        std::cout << "k_k0: " << std::hex;
        for (size_t j{}; j < kern_t::size; ++j) {
          std::cout << std::setw(2) << (kern_t::at(k_k, j) & 0xff) << ' ';
        }
        std::cout << "\nk_l0: " << std::hex;
        for (size_t j{}; j < kern_t::size; ++j) {
          std::cout << std::setw(2) << (kern_t::at(k_l, j) & 0xff) << ' ';
        }
        std::cout << "\ninc_idx = " << std::dec << inc_idx << '\n';
      }

      const ref_kern_repr_t r_l2{ref_kern_t::lru_update(r_k, r_l, r_m)};
      const kern_repr_t k_l2{kern_t::lru_update(k_k, k_l, k_m)};

      if constexpr (Verbose) {
        std::cout << "r_l2: " << std::hex;
        for (size_t j{}; j < kern_t::size; ++j) {
          std::cout << std::setw(2) << (ref_kern_t::at(r_l2, j) & 0xff) << ' ';
        }
        std::cout << "\nk_l2: " << std::hex;
        for (size_t j{}; j < kern_t::size; ++j) {
          std::cout << std::setw(2) << (kern_t::at(k_l2, j) & 0xff) << ' ';
        }
        std::cout << "\ninc_idx = " << std::dec << inc_idx << std::hex << '\n';
        //std::cout << "\nk_mask (size = " << kern_t::size << "): " << render_mask<mask_t>(k_m) << '\n';
      }

      for (size_t j{}; j < kern_t::size; ++j) {
        const uint8_t rj{static_cast<uint8_t>(ref_kern_t::at(r_l, j))};
        const uint8_t kj{static_cast<uint8_t>(kern_t::at(k_l, j))};
        if (rj != kj) {
          std::cout << "j = " << std::dec << j;
          std::cout << "\nrj = " << std::hex << (rj & 0xff);
          std::cout << "\nkj = " << std::hex << (kj & 0xff) << '\n';
          throw std::runtime_error("rj != kj line: " + std::to_string(__LINE__));
        }
        EXPECT_EQ(rj, kj);

        const uint8_t r2j{static_cast<uint8_t>(ref_kern_t::at(r_l2, j))};
        const uint8_t k2j{static_cast<uint8_t>(kern_t::at(k_l2, j))};
        if (r2j != k2j) {
          std::cout << "j = " << std::dec << j;
          std::cout << "\nr2j = " << std::hex << (r2j & 0xff);
          std::cout << "\nk2j = " << std::hex << (k2j & 0xff) << '\n';
          throw std::runtime_error("r2j != k2j line: " + std::to_string(__LINE__));
        }
        EXPECT_EQ(r2j, k2j);

        const int div{
          slot_is_occupied(ref_kern_t::at(r_k, j)) && ref_kern_t::at(r_l, inc_idx) == -1 ? 2 : 1
        };
        if (r2j != static_cast<uint8_t>(rj / div + (j == inc_idx))) {
          std::cout << "j = " << std::dec << j;
          std::cout << "\nrj = " << std::hex << (rj & 0xff);
          std::cout << "\nkj = " << std::hex << (kj & 0xff);
          std::cout << "\ndiv = " << std::hex << div;
          std::cout << "\nj == inc_idx = " << std::hex << (j == inc_idx);
          std::cout << "\nr2j = " << std::hex << (r2j & 0xff);
          std::cout << "\nk2j = " << std::hex << (k2j & 0xff);
          std::cout << "\nr2j/div = " << std::hex
                    << (static_cast<uint8_t>(rj / div + (j == inc_idx)) & 0xff);
          std::cout << "\nk2j/div = " << std::hex
                    << (static_cast<uint8_t>(kj / div + (j == inc_idx)) & 0xff) << '\n';
          throw std::runtime_error("r2j != static_cast<uint8_t>(rj / div + (j == inc_idx)");
        }
        EXPECT_EQ(r2j, static_cast<uint8_t>(rj / div + (j == inc_idx)));
        EXPECT_EQ(k2j, static_cast<uint8_t>(kj / div + (j == inc_idx)));
      }
      if constexpr (Verbose) {
        std::cout << "lru_update " << '\n';
      }

      r_l = r_l2;
      k_l = k_l2;
    }
  }
}

#define EVAL_RANDOM_VALUES_(_X_) \
  TEST(eval_random_values, _X_) { eval_random_values<_X_, 1024 * 1024, false>(); }

NVHM_FOR_EACH_(
  EVAL_RANDOM_VALUES_,
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
