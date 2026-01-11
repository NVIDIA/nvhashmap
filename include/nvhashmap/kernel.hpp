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

#include "common.hpp"
#include "debug.hpp"

#if NVHM_WITH_SSE2
#include <emmintrin.h>
#endif

#if NVHM_WITH_SSE3
#include <tmmintrin.h>
#endif

#if NVHM_WITH_SSE4
#include <smmintrin.h>
#endif

#if NVHM_WITH_AVX2 || NVHM_WITH_AVX512 || NVHM_WITH_AVX_FVL || NVHM_WITH_AVX_BWVL || \
  NVHM_WITH_AVX_VBMI
#include <immintrin.h>
#endif

#if NVHM_WITH_NEON
#include <arm_neon.h>
#endif

#if NVHM_WITH_SVE
#include <arm_sve.h>
#endif

#include <array>
#include <cstring>

namespace nvhm {

template <typename T, typename U>
constexpr T broadcast_bits(const U v) noexcept {
  static_assert(std::is_unsigned_v<U>);
  static_assert(sizeof(T) % sizeof(U) == 0);

  T x{static_cast<T>(v)};
  for (size_t i{num_bits<T> / 2}; i >= num_bits<U>; i /= 2) {
    x |= x << i;
  }
  return x;
}

inline void fast_copy(
  void* const __restrict dst, const void* const __restrict src, const size_t n
) noexcept {
#if NVHM_WITH_SVE
  size_t i{};
  do {
    const svbool_t pg{svwhilelt_b8(i, n)};
    const svint8_t d{svld1_s8(pg, &static_cast<const int8_t*>(src)[i])};
    svst1_s8(pg, &static_cast<int8_t*>(dst)[i], d);
    i += svcntb();
  } while (i < n);
#else
  std::memcpy(dst, src, n);
#endif
}

template <typename This, typename Repr>
struct mask {
  using this_type = This;
  using repr_type = Repr;

  mask() = delete;
};

template <
  typename Repr, size_t BitsPerSlot, bool LeftAligned,
  size_t MaxCount = num_bits<Repr> / BitsPerSlot>
struct uint_mask final : public mask<uint_mask<Repr, BitsPerSlot, LeftAligned, MaxCount>, Repr> {
  using base_type = mask<uint_mask<Repr, BitsPerSlot, LeftAligned, MaxCount>, Repr>;
  using repr_type = typename base_type::repr_type;
  static_assert(std::is_unsigned_v<repr_type> && sizeof(repr_type) >= sizeof(uint32_t));

  constexpr static size_t bits_per_slot{BitsPerSlot};
  static_assert(bits_per_slot > 0 && bits_per_slot <= 8);
  static_assert(has_single_bit(bits_per_slot));

  constexpr static size_t max_count{MaxCount};
  static_assert(max_count <= num_bits<repr_type> / bits_per_slot);
  constexpr static int shift{countr_zero(bits_per_slot)};

  uint_mask() = delete;

  constexpr static repr_type empty() noexcept { return {}; }

  constexpr static repr_type full() noexcept {
    repr_type m{broadcast_bits<repr_type, uint8_t>(0x01)};
    if constexpr (bits_per_slot < 8) {
      m |= m << 4;
    }
    if constexpr (bits_per_slot < 4) {
      m |= m << 2;
    }
    if constexpr (bits_per_slot < 2) {
      m |= m << 1;
    }
    if constexpr (LeftAligned) {
      m <<= (bits_per_slot - 1);
    }
    m >>= (num_bits<repr_type> - max_count * bits_per_slot);
    return m;
  }

  constexpr static bool has_next(const repr_type m) noexcept { return m != 0; }

  constexpr static size_t count(const repr_type m) noexcept {
    return static_cast<size_t>(popcount(m));
  }

  constexpr static size_t next(const repr_type m) noexcept {
    return static_cast<size_t>(countr_zero(m)) >> shift;
  }

  constexpr static raw_pos_t next(const repr_type m, const raw_pos_t off) noexcept {
    return off + next(m);
  }

  constexpr static repr_type step(const repr_type m) noexcept { return m & (m - 1); }

  constexpr static repr_type truncate(const repr_type m) noexcept { return m & (m ^ (m - 1)); }

  static_assert(count(full()) == max_count);
};

using uint_mask8_1_t = uint_mask<uint32_t, 1, false, 8>;
using uint_mask8_2r_t = uint_mask<uint32_t, 2, false, 4>;
using uint_mask8_2l_t = uint_mask<uint32_t, 2, true, 4>;
using uint_mask8_4r_t = uint_mask<uint32_t, 4, false, 2>;
using uint_mask8_4l_t = uint_mask<uint32_t, 4, true, 2>;
using uint_mask8_8r_t = uint_mask<uint32_t, 8, false, 1>;
using uint_mask8_8l_t = uint_mask<uint32_t, 8, true, 1>;

using uint_mask16_1_t = uint_mask<uint32_t, 1, false, 16>;
using uint_mask16_2r_t = uint_mask<uint32_t, 2, false, 8>;
using uint_mask16_2l_t = uint_mask<uint32_t, 2, true, 8>;
using uint_mask16_4r_t = uint_mask<uint32_t, 4, false, 4>;
using uint_mask16_4l_t = uint_mask<uint32_t, 4, true, 4>;
using uint_mask16_8r_t = uint_mask<uint32_t, 8, false, 2>;
using uint_mask16_8l_t = uint_mask<uint32_t, 8, true, 2>;

using uint_mask32_1_t = uint_mask<uint32_t, 1, false>;
using uint_mask32_2r_t = uint_mask<uint32_t, 2, false>;
using uint_mask32_2l_t = uint_mask<uint32_t, 2, true>;
using uint_mask32_4r_t = uint_mask<uint32_t, 4, false>;
using uint_mask32_4l_t = uint_mask<uint32_t, 4, true>;
using uint_mask32_8r_t = uint_mask<uint32_t, 8, false>;
using uint_mask32_8l_t = uint_mask<uint32_t, 8, true>;

using uint_mask64_1_t = uint_mask<uint64_t, 1, false>;
using uint_mask64_2r_t = uint_mask<uint64_t, 2, false>;
using uint_mask64_2l_t = uint_mask<uint64_t, 2, true>;
using uint_mask64_4r_t = uint_mask<uint64_t, 4, false>;
using uint_mask64_4l_t = uint_mask<uint64_t, 4, true>;
using uint_mask64_8r_t = uint_mask<uint64_t, 8, false>;
using uint_mask64_8l_t = uint_mask<uint64_t, 8, true>;

using uint_mask128_1_t = uint_mask<__uint128_t, 1, false>;
using uint_mask128_2r_t = uint_mask<__uint128_t, 2, false>;
using uint_mask128_2l_t = uint_mask<__uint128_t, 2, true>;
using uint_mask128_4r_t = uint_mask<__uint128_t, 4, false>;
using uint_mask128_4l_t = uint_mask<__uint128_t, 4, true>;
using uint_mask128_8r_t = uint_mask<__uint128_t, 8, false>;
using uint_mask128_8l_t = uint_mask<__uint128_t, 8, true>;

#if NVHM_WITH_SVE
template <size_t MaxCount>
struct sve_mask final : public mask<sve_mask<MaxCount>, svbool_t> {
  using base_type = mask<sve_mask<MaxCount>, svbool_t>;
  using repr_type = typename base_type::repr_type;

  constexpr static size_t max_count{MaxCount};
  static_assert(has_single_bit(max_count));

  sve_mask() = delete;

  inline static repr_type empty() noexcept { return svpfalse_b(); }

  inline static repr_type full() noexcept {
    if constexpr (max_count == 1) {
      return svptrue_pat_b8(SV_VL1);
    } else if constexpr (max_count == 2) {
      return svptrue_pat_b8(SV_VL2);
    } else if constexpr (max_count == 4) {
      return svptrue_pat_b8(SV_VL4);
    } else if constexpr (max_count == 8) {
      return svptrue_pat_b8(SV_VL8);
    } else if constexpr (max_count == 16) {
      return svptrue_pat_b8(SV_VL16);
    } else if constexpr (max_count == 32) {
      return svptrue_pat_b8(SV_VL32);
    } else if constexpr (max_count == 64) {
      return svptrue_pat_b8(SV_VL64);
    } else if constexpr (max_count == 128) {
      return svptrue_pat_b8(SV_VL128);
    } else if constexpr (max_count == 256) {
      return svptrue_pat_b8(SV_VL256);
    } else {
      static_assert(!max_count, "Shouldn't happen!");
    }
  }

  inline static bool has_next(const repr_type m) noexcept { return svptest_any(m, m); }

  inline static size_t count(const repr_type m) noexcept { return svcntp_b8(m, m); }

  inline static size_t next(const repr_type m) noexcept { return count(svbrkb_b_z(full(), m)); }

  inline static raw_pos_t next(const repr_type m, const raw_pos_t off) noexcept {
    // Why not non-quantizing incp?
    return svqincp_n_u64_b8(off, svbrkb_b_z(full(), m));
  }

  inline static repr_type step(const repr_type m) noexcept {
    return svnot_b_z(m, svbrka_b_z(m, m));
  }

  inline static repr_type truncate(const repr_type m) noexcept { return svbrka_b_z(m, m); }
};

using sve_mask8_t = sve_mask<1>;
using sve_mask16_t = sve_mask<2>;
using sve_mask32_t = sve_mask<4>;
using sve_mask64_t = sve_mask<8>;
using sve_mask128_t = sve_mask<16>;
#if __ARM_FEATURE_SVE_BITS >= 256
using sve_mask256_t = sve_mask<32>;
#endif
#if __ARM_FEATURE_SVE_BITS >= 512
using sve_mask512_t = sve_mask<64>;
#endif
#if __ARM_FEATURE_SVE_BITS >= 1024
using sve_mask1024_t = sve_mask<128>;
#endif
#if __ARM_FEATURE_SVE_BITS >= 2048
using sve_mask2048_t = sve_mask<256>;
#endif
using sve_mask_t = sve_mask<__ARM_FEATURE_SVE_BITS / 8>;
#endif

struct kernel {
  kernel() = delete;
};

template <size_t Size>
struct array_kernel final : public kernel {
  constexpr static size_t size{Size};
  static_assert(has_single_bit(size) && size <= num_bits<__uint128_t>);
  constexpr static size_t size_mask{size - 1};

  using repr_type = std::array<state_t, size>;

  using mask_repr_type = std::conditional_t<
    size <= num_bits<uint32_t>, uint32_t,
    std::conditional_t<
      size <= num_bits<uint64_t>, uint64_t,
      std::conditional_t<size <= num_bits<__uint128_t>, __uint128_t, void>>>;
  static_assert(sizeof(mask_repr_type) * 8 >= sizeof(repr_type));
  using mask_type = uint_mask<mask_repr_type, 1, false, size>;
  static_assert(mask_type::max_count == size);

  constexpr static state_t empty{-128};      // 0x80 == msb
  constexpr static state_t tombstone{-127};  // 0x81 == msb | lsb
  static_assert(empty < tombstone && tombstone < 0);

  array_kernel() = delete;

  /**
   * Loads slot states from memory. The memory address is guaranteed to be aligned.
   */
  constexpr static repr_type load(const state_t* __restrict ptr) noexcept {
    repr_type k;
    std::memcpy(k.begin(), assume_aligned<size>(ptr), size);
    return k;
  }

  /**
   * Stores slot states to memory. The memory address is guaranteed to be aligned.
   */
  constexpr static void store(const repr_type& k, state_t* __restrict ptr) noexcept {
    std::memcpy(assume_aligned<size>(ptr), k.begin(), size);
  }

  /**
   * Masks slots that match the provided hash.
   */
  constexpr static mask_repr_type mask(const repr_type& k, state_t s) noexcept {
    mask_repr_type m{};
    for (size_t i{}; i < size; ++i) {
      m |= static_cast<mask_repr_type>(k[i] == s) << i;
    }
    return m;
  }

  /**
   * Returns mask that identifies all slots that contains actual hashes.
   */
  constexpr static mask_repr_type mask_hash(const repr_type& k) noexcept {
    mask_repr_type m{};
    for (size_t i{}; i < size; ++i) {
      m |= static_cast<mask_repr_type>(k[i] >= 0) << i;
    }
    return m;
  }

  /**
   * Returns mask that identifies all slots that are either `empty` or `tombstones`.
   */
  constexpr static mask_repr_type mask_non_hash(const repr_type& k) noexcept {
    mask_repr_type m{};
    for (size_t i{}; i < size; ++i) {
      m |= static_cast<mask_repr_type>(k[i] < 0) << i;
    }
    return m;
  }

  /**
   * Returns mask that identifies all empty slots. May assume only valid values are present.
   */
  constexpr static mask_repr_type mask_empty(const repr_type& k) noexcept { return mask(k, empty); }

  /**
   * Returns mask that identifies all tombstone slots. May assume only valid values are present.
   */
  constexpr static mask_repr_type mask_tombstone(const repr_type& k) noexcept {
    return mask(k, tombstone);
  }

  constexpr static repr_type all(state_t s) noexcept {
    repr_type k;
    std::fill(k.begin(), k.end(), s);
    return k;
  }

  constexpr static repr_type all_empty() noexcept { return all(empty); }

  constexpr static repr_type all_tombstone() noexcept { return all(tombstone); }

  constexpr static bool has_empty(const repr_type& k) noexcept {
    bool b{};
    for (size_t i{}; i < size; ++i) {
      b |= k[i] == empty;
    }
    return b;
  }

  constexpr static repr_type hash_to_tombstone(const repr_type& k) noexcept {
    repr_type r;
    for (size_t i{}; i < size; ++i) {
      r[i] = std::min(k[i], tombstone);
    }
    return r;
  }

  constexpr static state_t at(const repr_type& k, const size_t i) noexcept {
    NVHM_ASSERT_(i < size, "i = ", i, ", size = ", size);
    return k[i];
  }

  constexpr static lru_t lru_at(const repr_type& k, const size_t i) noexcept {
    return static_cast<lru_t>(at(k, i));
  }

  constexpr static size_t min_lru_index(const repr_type& k) noexcept {
    lru_t lru{lru_at(k, 0)};
    size_t lru_idx{};

    for (size_t i{1}; i < size; ++i) {
      const lru_t ki{lru_at(k, i)};
      if (ki < lru) {
        lru = ki;
        lru_idx = i;
      }
    }

    return lru_idx;
  }

  constexpr static repr_type lru_update(
    const repr_type& k, const repr_type& l, const mask_repr_type& m
  ) {
    NVHM_ASSERT_(mask_type::has_next(m), "m = ", render_mask<mask_type>(m));
    const size_t i{mask_type::next(m)};
    NVHM_ASSERT_(i < size, "i = ", i, ", size = ", size);
    NVHM_ASSERT_(slot_is_occupied(at(k, i)));

    const int shift{lru_at(l, i) >= max_lru};

    repr_type r;
    for (size_t j{}; j < size; ++j) {
      uint32_t lj{lru_at(l, j)};
      if (slot_is_occupied(at(k, j))) {
        lj >>= shift;
      }
      lj += (j == i);
      r[j] = static_cast<state_t>(lj);
    }
    return r;
  }
};

using array_kernel8_t = array_kernel<1>;
using array_kernel16_t = array_kernel<2>;
using array_kernel32_t = array_kernel<4>;
using array_kernel64_t = array_kernel<8>;
using array_kernel128_t = array_kernel<16>;
using array_kernel256_t = array_kernel<32>;
using array_kernel512_t = array_kernel<64>;
using array_kernel1024_t = array_kernel<128>;

template <typename Repr>
struct uint_kernel final : public kernel {
  using repr_type = Repr;
  static_assert(std::is_unsigned_v<repr_type>);

  constexpr static size_t size{sizeof(repr_type)};
  static_assert(has_single_bit(size));
  constexpr static size_t size_mask{size - 1};

  using mask_repr_type =
    std::conditional_t<sizeof(repr_type) < sizeof(uint32_t), uint32_t, repr_type>;
  static_assert(sizeof(mask_repr_type) >= sizeof(repr_type));
  using mask_type = uint_mask<mask_repr_type, 8, true, size>;
  static_assert(mask_type::max_count == size);

  constexpr static state_t empty{-128};      // 0x80 == msb
  constexpr static state_t tombstone{-127};  // 0x81 == msb | lsb
  static_assert(empty < tombstone && tombstone < 0);

  constexpr static repr_type lsbs{broadcast_bits<repr_type, uint8_t>(0x01)};
  constexpr static repr_type msbs{broadcast_bits<repr_type, uint8_t>(0x80)};

  uint_kernel() = delete;

  /**
   * Loads slot states from memory. The memory address is guaranteed to be aligned.
   */
  constexpr static repr_type load(const state_t* __restrict ptr) noexcept {
    return *reinterpret_cast<const repr_type* __restrict>(assume_aligned<size>(ptr));
  }

  /**
   * Stores slot states to memory. The memory address is guaranteed to be aligned.
   */
  constexpr static void store(repr_type k, state_t* __restrict ptr) noexcept {
    *reinterpret_cast<repr_type* __restrict>(assume_aligned<size>(ptr)) = k;
  }

  /**
   * Masks slots that match the provided hash (may return false positives).
   */
  constexpr static mask_repr_type mask(repr_type k, state_t s) noexcept {
    k ^= lsbs * static_cast<lru_t>(s);
    if constexpr (allow_false_positive_matches) {
      // Compiles to one assembly instruction less.
      // https://graphics.stanford.edu/~seander/bithacks.html#ZeroInWord
      k = (k - lsbs) & ~k & msbs;
    } else {
      k |= ~msbs + (k & ~msbs);  // Set 7th bits to 1 if any bit below is not zero.
      k = ~k & msbs;             // Isolate 7th bit.
    }
    return k;
  }

  /**
   * Returns mask that identifies all slots that contains actual hashes.
   */
  constexpr static mask_repr_type mask_hash(repr_type k) noexcept { return ~k & msbs; }

  /**
   * Returns mask that identifies all slots that are either `empty` or `tombstones`.
   */
  constexpr static mask_repr_type mask_non_hash(repr_type k) noexcept { return k & msbs; }

  /**
   * Returns mask that identifies all empty slots. May assume only valid values are present.
   *   0 ~ 0  =>  0
   *   0 ~ 1  =>  0
   *   1 ~ 0  =>  1
   *   1 ~ 1  =>  0
   */
  constexpr static mask_repr_type mask_empty(repr_type k) noexcept { return k & (~k << 7) & msbs; }

  /**
   * Returns mask that identifies all tombstone slots. May assume only valid values are present.
   */
  constexpr static mask_repr_type mask_tombstone(repr_type k) noexcept {
    return k & (k << 7) & msbs;
  }

  constexpr static repr_type all_empty() noexcept { return msbs; }

  constexpr static repr_type all_tombstone() noexcept { return msbs | lsbs; }

  constexpr static bool has_empty(repr_type k) noexcept { return mask_empty(k) != 0; }

  constexpr static repr_type hash_to_tombstone(repr_type k) noexcept {
    // 0 ~ 0  =>  1 ~ 1
    // 0 ~ 1  =>  1 ~ 1
    // 1 ~ 0  =>  1 ~ 0
    // 1 ~ 1  =>  1 ~ 1
    return msbs | (((~k >> 7) | k) & lsbs);
  }

  constexpr static state_t at(repr_type k, size_t i) noexcept {
    NVHM_ASSERT_(i < size, "i = ", i, ", size = ", size);
    return static_cast<state_t>(k >> (i * num_bits<state_t>));
  }

  constexpr static lru_t lru_at(repr_type k, size_t i) noexcept {
    return static_cast<lru_t>(at(k, i));
  }

  constexpr static size_t min_lru_index(repr_type k) noexcept {
    lru_t lru{static_cast<lru_t>(k)};
    size_t lru_idx{};

    for (size_t i{1}; i < size; ++i) {
      k >>= 8;
      const lru_t ki{static_cast<lru_t>(k)};
      if (ki < lru) {
        lru = ki;
        lru_idx = i;
      }
    }

    return lru_idx;
  }

  constexpr static repr_type lru_update(const repr_type k, const repr_type l, mask_repr_type m) {
    NVHM_ASSERT_(mask_type::has_next(m), "m = ", render_mask<mask_type>(m));
    const size_t i{mask_type::next(m)};
    NVHM_ASSERT_(i < size, "i = ", i, ", size = ", size);
    NVHM_ASSERT_(slot_is_occupied(at(k, i)));

    // Rescale if l[i] hit maximum value.
    mask_repr_type r;
    if (NVHM_LIKELY_(lru_at(l, i) < max_lru)) {
      r = l;
    } else {
      r = (l >> 1) & ~msbs;
      m = (mask_non_hash(k) >> 7) * 0xff;
      // https://graphics.stanford.edu/~seander/bithacks.html#MaskedMerge
      r ^= (r ^ l) & m;
    }

    // Add 1 to l[i].
    m = mask_repr_type{0xff} << (i * num_bits<lru_t>);
    // https://graphics.stanford.edu/~seander/bithacks.html#MaskedMerge
    r ^= (r ^ (r + (mask_repr_type{0x01} << (i * num_bits<lru_t>)))) & m;
    return static_cast<repr_type>(r);
  }
};

template <>
struct uint_kernel<uint8_t> final : public kernel {
  using repr_type = uint8_t;

  constexpr static size_t size{sizeof(repr_type)};
  static_assert(has_single_bit(size));
  constexpr static size_t size_mask{size - 1};

  using mask_type = uint_mask32_1_t;
  using mask_repr_type = typename mask_type::repr_type;
  static_assert(sizeof(mask_repr_type) >= sizeof(repr_type));

  constexpr static state_t empty{-128};      // 0x80 == msb
  constexpr static state_t tombstone{-127};  // 0x81 == msb | lsb
  static_assert(empty < tombstone && tombstone < 0);

  uint_kernel() = delete;

  /**
   * Loads slot states from memory. The memory address is guaranteed to be aligned.
   */
  inline static repr_type load(const state_t* __restrict ptr) noexcept {
    return *reinterpret_cast<const repr_type*>(assume_aligned<size>(ptr));
  }

  /**
   * Stores slot states to memory. The memory address is guaranteed to be aligned.
   */
  inline static void store(repr_type k, state_t* __restrict ptr) noexcept {
    *reinterpret_cast<repr_type*>(assume_aligned<size>(ptr)) = k;
  }

  /**
   * Masks slots that match the provided hash (may return false positives).
   */
  constexpr static mask_repr_type mask(repr_type k, state_t s) noexcept {
    return static_cast<mask_repr_type>(k == static_cast<repr_type>(s));
  }

  /**
   * Returns mask that identifies all slots that contains actual hashes.
   */
  constexpr static mask_repr_type mask_hash(repr_type k) noexcept { return (k >> 7) ^ 0x1; }

  /**
   * Returns mask that identifies all slots that are either `empty` or `tombstones`.
   */
  constexpr static mask_repr_type mask_non_hash(repr_type k) noexcept { return k >> 7; }

  /**
   * Returns mask that identifies all empty slots. May assume only valid values are present.
   */
  constexpr static mask_repr_type mask_empty(repr_type k) noexcept { return mask(k, empty); }

  /**
   * Returns mask that identifies all tombstone slots. May assume only valid values are present.
   */
  constexpr static mask_repr_type mask_tombstone(repr_type k) noexcept {
    return mask(k, tombstone);
  }

  constexpr static repr_type all_empty() noexcept { return static_cast<repr_type>(empty); }

  constexpr static repr_type all_tombstone() noexcept { return static_cast<repr_type>(tombstone); }

  constexpr static bool has_empty(repr_type k) noexcept { return k == all_empty(); }

  constexpr static repr_type hash_to_tombstone(repr_type k) noexcept {
    return static_cast<repr_type>(std::min(at(k, 0), tombstone));
  }

  constexpr static state_t at(repr_type k, [[maybe_unused]] size_t i) noexcept {
    NVHM_ASSERT_(i < size, "i = ", i, ", size = ", size);
    return static_cast<state_t>(k);
  }

  constexpr static lru_t lru_at(repr_type k, size_t i) noexcept {
    return static_cast<lru_t>(at(k, i));
  }

  constexpr static size_t min_lru_index(repr_type) noexcept { return {}; }

  constexpr static repr_type lru_update(
    const repr_type k, repr_type l, [[maybe_unused]] const mask_repr_type m
  ) {
    NVHM_ASSERT_(mask_type::has_next(m) && mask_type::next(m) < size, "m = ", render_mask<mask_type>(m));
    NVHM_ASSERT_(slot_is_occupied(at(k, 0)));

    l >>= mask_hash(k) & (l >= max_lru);
    l += 1;
    return l;
  }
};

using uint_kernel8_t = uint_kernel<uint8_t>;
using uint_kernel16_t = uint_kernel<uint16_t>;
using uint_kernel32_t = uint_kernel<uint32_t>;
using uint_kernel64_t = uint_kernel<uint64_t>;
using uint_kernel128_t = uint_kernel<__uint128_t>;

#if NVHM_WITH_SSE2
struct sse_kernel final : public kernel {
  using repr_type = __m128i;

  constexpr static size_t size{sizeof(repr_type)};
  static_assert(has_single_bit(size));
  constexpr static size_t size_mask{size - 1};

  using mask_type = uint_mask16_1_t;
  using mask_repr_type = typename mask_type::repr_type;

  constexpr static state_t empty{-128};      // 0x80 == msb
  constexpr static state_t tombstone{-127};  // 0x81 == msb | lsb
  static_assert(empty < tombstone && tombstone < 0);

  sse_kernel() = delete;

  inline static repr_type load(const state_t* __restrict ptr) noexcept {
    return _mm_load_si128(reinterpret_cast<const repr_type*>(assume_aligned<size>(ptr)));
  }

  inline static void store(repr_type k, state_t* __restrict ptr) noexcept {
    _mm_store_si128(reinterpret_cast<repr_type*>(assume_aligned<size>(ptr)), k);
  }

  inline static mask_repr_type mask(repr_type k, state_t s) noexcept {
    return mask_(k, _mm_set1_epi8(s));
  }

  inline static mask_repr_type mask_hash(repr_type k) noexcept { return mask_non_hash(k) ^ 0xffff; }

  inline static mask_repr_type mask_non_hash(repr_type k) noexcept {
#if NVHM_WITH_AVX_BWVL
    return _mm_movepi8_mask(k);
#else
    return static_cast<mask_repr_type>(_mm_movemask_epi8(k));
#endif
  }

  inline static mask_repr_type mask_empty(repr_type k) noexcept {
#if NVHM_WITH_SSE3
    return mask_non_hash(_mm_sign_epi8(k, k));
#else
    return mask_(k, all_empty());
#endif
  }

  inline static mask_repr_type mask_tombstone(repr_type k) noexcept {
    return mask_(k, all_tombstone());
  }

  inline static repr_type all_empty() noexcept { return _mm_set1_epi8(empty); }

  inline static repr_type all_tombstone() noexcept { return _mm_set1_epi8(tombstone); }

  inline static bool has_empty(repr_type k) noexcept { return mask_empty(k) != 0; }

  inline static repr_type hash_to_tombstone(repr_type k) noexcept {
#if NVHM_WITH_SSE4
    k = _mm_min_epi8(k, all_tombstone());
#else
    k = _mm_cmplt_epi8(k, all_tombstone());
    k = _mm_add_epi8(k, all_tombstone());
#endif
    return k;
  }

  inline static state_t at(repr_type k, size_t i) noexcept {
    NVHM_ASSERT_(i < size, "i = ", i, ", size = ", size);
    int64_t s;
#if NVHM_WITH_AVX_VBMI
    k = _mm_permutexvar_epi8(_mm_set1_epi8(static_cast<char>(i)), k);
    s = _mm_cvtsi128_si64(k);
#elif NVHM_WITH_SSE3
    k = _mm_shuffle_epi8(k, _mm_set1_epi8(static_cast<char>(i)));
    s = _mm_cvtsi128_si64(k);
#else
    const int64_t k0{_mm_cvtsi128_si64(k)};
    k = _mm_srli_si128(k, sizeof(int64_t));
    const int64_t k1{_mm_cvtsi128_si64(k)};
    s = (i < sizeof(int64_t) ? k0 : k1) >> (i % sizeof(int64_t) * 8);
#endif
    return static_cast<state_t>(s);
  }

  inline static repr_type min_lru(repr_type k) noexcept {
    k = _mm_min_epu8(k, _mm_srli_si128(k, 1));
#if NVHM_WITH_SSE4
    k = _mm_and_si128(k, _mm_set1_epi16(0xff));
    k = _mm_minpos_epu16(k);
#else
    k = _mm_min_epu8(k, _mm_srli_si128(k, 2));
    k = _mm_min_epu8(k, _mm_srli_si128(k, 4));
    k = _mm_min_epu8(k, _mm_srli_si128(k, 8));
#endif
    return k;
  }

  inline static mask_repr_type mask_min_lru(repr_type k) noexcept {
    repr_type s{min_lru(k)};
#if NVHM_WITH_AVX2
    s = _mm_broadcastb_epi8(s);
#elif NVHM_WITH_SSE3
    s = _mm_shuffle_epi8(s, _mm_setzero_si128());
#else
    s = _mm_set1_epi8(static_cast<char>(_mm_cvtsi128_si32(s)));
#endif
    return mask_(k, s);
  }

  inline static size_t min_lru_index(repr_type k) noexcept {
    return mask_type::next(mask_min_lru(k));
  }

  inline static repr_type lru_update(const repr_type k, repr_type l, mask_repr_type m) {
    NVHM_ASSERT_(mask_type::has_next(m) && mask_type::next(m) < size, "m = ", render_mask<mask_type>(m));
    m = mask_type::truncate(m);
    NVHM_ASSERT_(slot_is_occupied(at(k, mask_type::next(m))));

    // Rescale if l[i] hit maximum value.
    if (NVHM_UNLIKELY_(mask_type::has_next(mask(l, static_cast<state_t>(max_lru)) & m))) {
      __m128i tmp;
      tmp = _mm_srli_epi64(l, 1);
      tmp = _mm_and_si128(tmp, _mm_set1_epi8(0x7f));
#if NVHM_WITH_AVX_BWVL
      l = _mm_mask_blend_epi8(static_cast<__mmask16>(mask_non_hash(k)), tmp, l);
#else
      const __m128i msk{_mm_cmplt_epi8(k, _mm_setzero_si128())};
#if NVHM_WITH_SSE4
      l = _mm_blendv_epi8(tmp, l, msk);
#else
      l = _mm_and_si128(msk, l);
      tmp = _mm_andnot_si128(msk, tmp);
      l = _mm_or_si128(tmp, l);
#endif
#endif
    }

    // Add 1 to l[i].
#if NVHM_WITH_AVX_BWVL
    l = _mm_mask_add_epi8(l, static_cast<__mmask16>(m), l, _mm_set1_epi8(1));
#else
    size_t i{mask_type::next(m)};
    const int64_t l0{static_cast<int64_t>(i < sizeof(int64_t))};
    const int64_t l1{static_cast<int64_t>(i >= sizeof(int64_t))};
    i = i % sizeof(int64_t) * num_bits<lru_t>;

    __m128i inc;
    inc = _mm_cvtsi64_si128(l0 << i);
#if NVHM_WITH_SSE4
    inc = _mm_insert_epi64(inc, l1 << i, 1);
#else
    inc = _mm_unpacklo_epi64(inc, _mm_cvtsi64_si128(l1 << i));
#endif

    l = _mm_add_epi8(l, inc);
#endif
    return l;
  }

 private:
  inline static mask_repr_type mask_(repr_type k, repr_type s) noexcept {
#if NVHM_WITH_AVX_BWVL
    return _mm_cmpeq_epi8_mask(k, s);
#else
    return mask_non_hash(_mm_cmpeq_epi8(k, s));
#endif
  }
};

using sse_kernel_t = sse_kernel;
#endif

#if NVHM_WITH_AVX2
struct avx_kernel final : public kernel {
  using repr_type = __m256i;

  constexpr static size_t size{sizeof(repr_type)};
  static_assert(has_single_bit(size));
  constexpr static size_t size_mask{size - 1};

  using mask_type = uint_mask32_1_t;
  using mask_repr_type = typename mask_type::repr_type;

  constexpr static state_t empty{-128};      // 0x80 == msb
  constexpr static state_t tombstone{-127};  // 0x81 == msb | lsb
  static_assert(empty < tombstone && tombstone < 0);

  avx_kernel() = delete;

  inline static repr_type load(const state_t* __restrict ptr) noexcept {
    return _mm256_load_si256(reinterpret_cast<const __m256i*>(assume_aligned<size>(ptr)));
  }

  inline static void store(const repr_type k, state_t* __restrict ptr) noexcept {
    _mm256_store_si256(reinterpret_cast<__m256i*>(assume_aligned<size>(ptr)), k);
  }

  inline static mask_repr_type mask(repr_type k, state_t s) noexcept {
    return mask_(k, _mm256_set1_epi8(s));
  }

  inline static mask_repr_type mask_hash(repr_type k) noexcept { return ~mask_non_hash(k); }

  inline static mask_repr_type mask_non_hash(repr_type k) noexcept {
#if NVHM_WITH_AVX_BWVL
    return _mm256_movepi8_mask(k);
#else
    return static_cast<mask_repr_type>(_mm256_movemask_epi8(k));
#endif
  }

  inline static mask_repr_type mask_empty(repr_type k) noexcept {
    return mask_non_hash(_mm256_sign_epi8(k, k));
  }

  inline static mask_repr_type mask_tombstone(repr_type k) noexcept {
    return mask_(k, all_tombstone());
  }

  inline static repr_type all_empty() noexcept { return _mm256_set1_epi8(empty); }

  inline static repr_type all_tombstone() noexcept { return _mm256_set1_epi8(tombstone); }

  inline static bool has_empty(repr_type k) noexcept { return mask_empty(k) != 0; }

  inline static repr_type hash_to_tombstone(repr_type k) noexcept {
    return _mm256_min_epi8(k, all_tombstone());
  }

  inline static state_t at(repr_type k, size_t i) noexcept {
    NVHM_ASSERT_(i < size, "i = ", i, ", size = ", size);
    int s;
#if NVHM_WITH_AVX_VBMI
    k = _mm256_permutexvar_epi8(_mm256_set1_epi8(static_cast<char>(i)), k);
    s = _mm256_cvtsi256_si32(k);
#else
    k = _mm256_shuffle_epi8(k, _mm256_set1_epi8(static_cast<char>(i)));
#if NVHM_WITH_AVX_FVL
    s = _mm_cvtsi128_si32(_mm256_cvtepi64_epi8(k));
    s >>= i / sizeof(int64_t) * 8;
#else
    const int k0{_mm256_cvtsi256_si32(k)};
    const int k4{_mm256_extract_epi32(k, sizeof(__m128i) / sizeof(int32_t))};
    s = (i < sizeof(__m128i)) ? k0 : k4;
#endif
#endif
    return static_cast<state_t>(s);
  }

  inline static __m128i min_lru(repr_type k) noexcept {
    const __m128i k0{_mm256_castsi256_si128(k)};
    const __m128i k1{_mm256_extracti128_si256(k, 1)};
    return sse_kernel::min_lru(_mm_min_epu8(k0, k1));
  }

  inline static mask_repr_type mask_min_lru(repr_type k) noexcept {
    return mask_(k, _mm256_broadcastb_epi8(min_lru(k)));
  }

  inline static size_t min_lru_index(repr_type k) noexcept {
    return mask_type::next(mask_min_lru(k));
  }

  inline static repr_type lru_update(const repr_type k, repr_type l, mask_repr_type m) {
    NVHM_ASSERT_(mask_type::has_next(m) && mask_type::next(m) < size, "m = ", render_mask<mask_type>(m));
    m = mask_type::truncate(m);
    NVHM_ASSERT_(slot_is_occupied(at(k, mask_type::next(m))));

    // Rescale if l[i] hit maximum value.
    if (NVHM_UNLIKELY_(mask_type::has_next(mask(l, static_cast<state_t>(max_lru)) & m))) {
      __m256i tmp;
      tmp = _mm256_srli_epi64(l, 1);
      tmp = _mm256_and_si256(tmp, _mm256_set1_epi8(0x7f));
#if NVHM_WITH_AVX_BWVL
      l = _mm256_mask_blend_epi8(mask_non_hash(k), tmp, l);
#else
      l = _mm256_blendv_epi8(l, tmp, _mm256_cmpgt_epi8(k, _mm256_set1_epi8(-1)));
#endif
    }

    // Add 1 to l[i].
#if NVHM_WITH_AVX_BWVL
    l = _mm256_mask_add_epi8(l, m, l, _mm256_set1_epi8(1));
#else
    const size_t i{mask_type::next(m)};
    const __m128i maj{_mm_cvtsi64_si128(INT64_C(0xff) << (i / sizeof(int64_t) * num_bits<lru_t>))};
    const __m256i min{_mm256_set1_epi64x(INT64_C(0x1) << (i % sizeof(int64_t) * num_bits<lru_t>))};
    const __m256i inc{_mm256_and_si256(_mm256_cvtepi8_epi64(maj), min)};
    l = _mm256_add_epi8(l, inc);
#endif
    return l;
  }

 private:
  inline static mask_repr_type mask_(repr_type k, repr_type s) noexcept {
#if NVHM_WITH_AVX_BWVL
    return _mm256_cmpeq_epi8_mask(k, s);
#else
    return mask_non_hash(_mm256_cmpeq_epi8(k, s));
#endif
  }
};

using avx_kernel_t = avx_kernel;
#endif

#if NVHM_WITH_AVX512
struct avx512_kernel final : public kernel {
  using repr_type = __m512i;

  constexpr static size_t size{sizeof(repr_type)};
  static_assert(has_single_bit(size));
  constexpr static size_t size_mask{size - 1};

  using mask_type = uint_mask64_1_t;
  using mask_repr_type = typename mask_type::repr_type;

  constexpr static state_t empty{-128};      // 0x80 == msb
  constexpr static state_t tombstone{-127};  // 0x81 == msb | lsb
  static_assert(empty < tombstone && tombstone < 0);

  avx512_kernel() = delete;

  inline static repr_type load(const state_t* __restrict ptr) noexcept {
    return _mm512_load_si512(assume_aligned<size>(ptr));
  }

  inline static void store(repr_type k, state_t* __restrict ptr) noexcept {
    _mm512_store_si512(assume_aligned<size>(ptr), k);
  }

  inline static mask_repr_type mask(repr_type k, const state_t s) noexcept {
    return mask_(k, _mm512_set1_epi8(s));
  }

  inline static mask_repr_type mask_hash(repr_type k) noexcept { return ~mask_non_hash(k); }

  inline static mask_repr_type mask_non_hash(repr_type k) noexcept {
    return _mm512_movepi8_mask(k);
  }

  inline static mask_repr_type mask_empty(repr_type k) noexcept { return mask_(k, all_empty()); }

  inline static mask_repr_type mask_tombstone(repr_type k) noexcept {
    return mask_(k, all_tombstone());
  }

  inline static repr_type all_empty() noexcept { return _mm512_set1_epi8(empty); }

  inline static repr_type all_tombstone() noexcept { return _mm512_set1_epi8(tombstone); }

  inline static bool has_empty(repr_type k) noexcept { return mask_empty(k) != 0; }

  inline static repr_type hash_to_tombstone(repr_type k) noexcept {
    return _mm512_min_epi8(k, all_tombstone());
  }

  inline static state_t at(repr_type k, size_t i) noexcept {
    NVHM_ASSERT_(i < size, "i = ", i, ", size = ", size);
#if NVHM_WITH_AVX_VBMI
    k = _mm512_permutexvar_epi8(_mm512_set1_epi8(static_cast<char>(i)), k);
    const int s{_mm512_cvtsi512_si32(k)};
    return static_cast<state_t>(s);
#else
    k = _mm512_shuffle_epi8(k, _mm512_set1_epi8(static_cast<char>(i)));
    const int64_t s{_mm_cvtsi128_si64(_mm512_cvtepi64_epi8(k))};
    return static_cast<state_t>(s >> (i / sizeof(int64_t) * 8));
#endif
  }

  inline static __m128i min_lru(const repr_type k) noexcept {
    const __m256i k0{_mm512_castsi512_si256(k)};
    const __m256i k1{_mm512_extracti64x4_epi64(k, 1)};
    return avx_kernel::min_lru(_mm256_min_epu8(k0, k1));
  }

  inline static mask_repr_type mask_min_lru(repr_type k) noexcept {
    return mask_(k, _mm512_broadcastb_epi8(min_lru(k)));
  }

  inline static size_t min_lru_index(repr_type k) noexcept {
    return mask_type::next(mask_min_lru(k));
  }

  inline static repr_type lru_update(const repr_type k, repr_type l, mask_repr_type m) {
    NVHM_ASSERT_(mask_type::has_next(m) && mask_type::next(m) < size, "m = ", render_mask<mask_type>(m));
    m = mask_type::truncate(m);
    NVHM_ASSERT_(slot_is_occupied(at(k, mask_type::next(m))));

    // Rescale if l[i] hit maximum value.
    if (NVHM_UNLIKELY_(mask_type::has_next(mask(l, static_cast<state_t>(max_lru)) & m))) {
      __m512i tmp;
      tmp = _mm512_srli_epi64(l, 1);
      tmp = _mm512_and_si512(tmp, _mm512_set1_epi8(0x7f));
      l = _mm512_mask_blend_epi8(mask_non_hash(k), tmp, l);
    }

    // Add 1 to l[i].
    l = _mm512_mask_add_epi8(l, m, l, _mm512_set1_epi8(1));
    return l;
  }

 private:
  inline static mask_repr_type mask_(repr_type k, const repr_type s) noexcept {
    return _mm512_cmpeq_epi8_mask(k, s);
  }
};

using avx512_kernel_t = avx512_kernel;
#endif

#if NVHM_WITH_NEON
struct neon_kernel64 final : public kernel {
  using repr_type = int8x8_t;

  constexpr static size_t size{sizeof(repr_type)};
  static_assert(has_single_bit(size));
  constexpr static size_t size_mask{size - 1};

  using mask_type = uint_mask64_8r_t;
  using mask_repr_type = typename mask_type::repr_type;
  constexpr static mask_repr_type full_mask{mask_type::full()};

  constexpr static state_t empty{-128};      // 0x80 == msb
  constexpr static state_t tombstone{-127};  // 0x81 == msb | lsb
  static_assert(empty < tombstone && tombstone < 0);

  neon_kernel64() = delete;

  inline static repr_type load(const state_t* __restrict ptr) noexcept {
    return vld1_s8(assume_aligned<size>(ptr));
  }

  inline static void store(repr_type k, state_t* __restrict ptr) noexcept {
    vst1_s8(assume_aligned<size>(ptr), k);
  }

  inline static mask_repr_type mask(repr_type k, state_t s) noexcept {
    return full_mask & collapse_mask_(vceq_s8(k, vdup_n_s8(s)));
  }

  inline static mask_repr_type mask_hash(repr_type k) noexcept {
    return full_mask & collapse_mask_(vcgez_s8(k));
  }

  inline static mask_repr_type mask_non_hash(repr_type k) noexcept {
    return full_mask & collapse_mask_(vcltz_s8(k));
  }

  inline static mask_repr_type mask_empty(repr_type k) noexcept {
    return full_mask & collapse_mask_(vceq_s8(k, all_empty()));
  }

  inline static mask_repr_type mask_tombstone(repr_type k) noexcept {
    return full_mask & collapse_mask_(vceq_s8(k, all_tombstone()));
  }

  inline static repr_type all_empty() noexcept { return vdup_n_s8(empty); }

  inline static repr_type all_tombstone() noexcept { return vdup_n_s8(tombstone); }

  inline static bool has_empty(repr_type k) noexcept { return vminv_s8(k) == empty; }

  inline static repr_type hash_to_tombstone(repr_type k) noexcept {
    return vmin_s8(k, all_tombstone());
  }

  inline static state_t at(repr_type k, size_t i) noexcept {
    NVHM_ASSERT_(i < size, "i = ", i, ", size = ", size);
    k = vtbl1_s8(k, vdup_n_s8(static_cast<int8_t>(i)));
    return vget_lane_s8(k, 0);
  }

  inline static uint8_t min_lru(repr_type k) noexcept { return vminv_u8(vreinterpret_u8_s8(k)); }

  inline static mask_repr_type mask_min_lru(repr_type k) noexcept {
    return mask(k, static_cast<state_t>(min_lru(k)));
  }

  inline static size_t min_lru_index(repr_type k) noexcept {
    const repr_type s{vreinterpret_s8_u8(vdup_n_u8(min_lru(k)))};
    const mask_repr_type m{collapse_mask_(vceq_s8(k, s))};
    return mask_type::next(m);
  }

  inline static repr_type lru_update(const repr_type k, const repr_type l, const mask_repr_type m) {
    NVHM_ASSERT_(mask_type::has_next(m) && mask_type::next(m) < size, "m = ", render_mask<mask_type>(m));
    NVHM_ASSERT_(slot_is_occupied(at(k, mask_type::next(m))));

    return lru_update_(k, vreinterpret_u8_s8(l), expand_mask_(mask_type::truncate(m)));
  }

 private:
  inline static mask_repr_type collapse_mask_(uint8x8_t m) noexcept {
    return vget_lane_u64(vreinterpret_u64_u8(m), 0);
  }

  inline static uint8x8_t expand_mask_(mask_repr_type m) noexcept {
    return vcgtz_s8(vcreate_s8(m));
  }

  inline static repr_type lru_update_(const repr_type k, uint8x8_t l, const uint8x8_t m) {
    // Mask slots that need rescaling.
    const uint64x1_t r64{vreinterpret_u64_u8(vand_u8(vceq_u8(l, m), m))};
    uint8x8_t r{vreinterpret_u8_u64(vtst_u64(r64, r64))};
    r = vand_u8(r, vcgez_s8(k));

    // Rescale and increment.
    l = vshl_u8(l, vreinterpret_s8_u8(r));
    l = vsub_u8(l, m);

    return vreinterpret_s8_u8(l);
  }
};

struct neon_kernel128 final : public kernel {
  using repr_type = int8x16_t;

  constexpr static size_t size{sizeof(repr_type)};
  static_assert(has_single_bit(size));
  constexpr static size_t size_mask{size - 1};

  using mask_type = uint_mask64_4r_t;
  using mask_repr_type = typename mask_type::repr_type;
  constexpr static mask_repr_type full_mask{mask_type::full()};

  constexpr static state_t empty{-128};      // 0x80 == msb
  constexpr static state_t tombstone{-127};  // 0x81 == msb | lsb
  static_assert(empty < tombstone && tombstone < 0);

  neon_kernel128() = delete;

  inline static repr_type load(const state_t* __restrict ptr) noexcept {
    return vld1q_s8(assume_aligned<size>(ptr));
  }

  inline static void store(repr_type k, state_t* __restrict ptr) noexcept {
    vst1q_s8(assume_aligned<size>(ptr), k);
  }

  inline static mask_repr_type mask(repr_type k, state_t s) noexcept {
    return full_mask & mask_(k, vdupq_n_s8(s));
  }

  inline static mask_repr_type mask_hash(repr_type k) noexcept {
    return full_mask & collapse_mask_(vcgezq_s8(k));
  }

  inline static mask_repr_type mask_non_hash(repr_type k) noexcept {
    return full_mask & collapse_mask_(vcltzq_s8(k));
  }

  inline static mask_repr_type mask_empty(repr_type k) noexcept {
    return full_mask & mask_(k, all_empty());
  }

  inline static mask_repr_type mask_tombstone(repr_type k) noexcept {
    return full_mask & mask_(k, all_tombstone());
  }

  inline static repr_type all_empty() noexcept { return vdupq_n_s8(empty); }

  inline static repr_type all_tombstone() noexcept { return vdupq_n_s8(tombstone); }

  inline static bool has_empty(repr_type k) noexcept { return min(k) == empty; }

  inline static repr_type hash_to_tombstone(repr_type k) noexcept {
    return vminq_s8(k, all_tombstone());
  }

  inline static state_t at(repr_type k, size_t i) noexcept {
    NVHM_ASSERT_(i < size, "i = ", i, ", size = ", size);
    k = vqtbl1q_s8(k, vdupq_n_u8(static_cast<uint8_t>(i)));
    return vgetq_lane_s8(k, 0);
  }

  inline static state_t min(repr_type k) noexcept { return vminvq_s8(k); }

  inline static uint8_t min_lru(repr_type k) noexcept { return vminvq_u8(vreinterpretq_u8_s8(k)); }

  inline static mask_repr_type mask_min_lru(repr_type k) noexcept {
    return mask(k, static_cast<state_t>(min_lru(k)));
  }

  inline static size_t min_lru_index(repr_type k) noexcept {
    const repr_type s{vreinterpretq_s8_u8(vdupq_n_u8(min_lru(k)))};
    const mask_repr_type m{collapse_mask_(vceqq_s8(k, s))};
    return mask_type::next(m);
  }

  inline static repr_type lru_update(const repr_type k, const repr_type l, const mask_repr_type m) {
    NVHM_ASSERT_(mask_type::has_next(m) && mask_type::next(m) < size, "m = ", render_mask<mask_type>(m));
    NVHM_ASSERT_(slot_is_occupied(at(k, mask_type::next(m))));

    return lru_update_(k, vreinterpretq_u8_s8(l), expand_mask_(mask_type::truncate(m)));
  }

 private:
  inline static mask_repr_type mask_(repr_type k, repr_type s) noexcept {
    return collapse_mask_(vceqq_s8(k, s));
  }

  /**
   * Truncate NEON 8-bit mask (0x00 or 0xff) into a 4-bit mask (0x0 or 0xf).
   * See also:
   * https://community.arm.com/arm-community-blogs/b/infrastructure-solutions-blog/posts/porting-x86-vector-bitmask-optimizations-to-arm-neon
   */
  inline static mask_repr_type collapse_mask_(uint8x16_t m8) noexcept {
    const uint8x8_t m4{vshrn_n_u16(vreinterpretq_u16_u8(m8), 4)};
    return vget_lane_u64(vreinterpret_u64_u8(m4), 0);
  }

  inline static uint8x16_t expand_mask_(mask_repr_type m) noexcept {
    const uint8x16_t m4{vreinterpretq_u8_u16(vshll_n_u8(vcreate_u8(m), 4))};
    return vtstq_u8(m4, m4);
  }

  inline static repr_type lru_update_(const repr_type k, uint8x16_t l, const uint8x16_t m) {
    // Mask slots that need rescaling.
    uint8x16_t r{vandq_u8(vceqq_u8(l, m), m)};
    r = vdupq_n_u8(vmaxvq_u8(r));
    r = vandq_u8(r, vcgezq_s8(k));

    // Rescale and increment.
    l = vshlq_u8(l, vreinterpretq_s8_u8(r));
    l = vsubq_u8(l, m);

    return vreinterpretq_s8_u8(l);
  }
};

struct neon_kernel256 final : public kernel {
  using repr_type = int8x16x2_t;

  constexpr static size_t size{sizeof(repr_type)};
  static_assert(has_single_bit(size));
  constexpr static size_t size_mask{size - 1};

  using mask_type = uint_mask64_2r_t;
  using mask_repr_type = typename mask_type::repr_type;
  constexpr static mask_repr_type full_mask{mask_type::full()};

  constexpr static state_t empty{-128};      // 0x80 == msb
  constexpr static state_t tombstone{-127};  // 0x81 == msb | lsb
  static_assert(empty < tombstone && tombstone < 0);

  neon_kernel256() = delete;

  inline static repr_type load(const state_t* __restrict ptr) noexcept {
    return vld1q_s8_x2(assume_aligned<size>(ptr));
  }

  inline static void store(repr_type k, state_t* __restrict ptr) noexcept {
    vst1q_s8_x2(assume_aligned<size>(ptr), k);
  }

  inline static mask_repr_type mask(repr_type k, state_t s) noexcept {
    return mask_(k, vdupq_n_s8(s));
  }

  inline static mask_repr_type mask_hash(repr_type k) noexcept {
    return collapse_mask_(vcgezq_s8(k.val[0]), vcgezq_s8(k.val[1]));
  }

  inline static mask_repr_type mask_non_hash(repr_type k) noexcept {
    return collapse_mask_(vcltzq_s8(k.val[0]), vcltzq_s8(k.val[1]));
  }

  inline static mask_repr_type mask_empty(repr_type k) noexcept {
    return mask_(k, neon_kernel128::all_empty());
  }

  inline static mask_repr_type mask_tombstone(repr_type k) noexcept {
    return mask_(k, neon_kernel128::all_tombstone());
  }

  inline static bool has_empty(repr_type k) noexcept { return min(k) == empty; }

  inline static repr_type hash_to_tombstone(repr_type k) noexcept {
    return {
      neon_kernel128::hash_to_tombstone(k.val[0]), neon_kernel128::hash_to_tombstone(k.val[1])
    };
  }

  inline static state_t at(repr_type k, size_t i) noexcept {
    NVHM_ASSERT_(i < size, "i = ", i, ", size = ", size);
    const int8x16_t k0{vqtbl2q_s8(k, vdupq_n_u8(static_cast<uint8_t>(i)))};
    return vgetq_lane_s8(k0, 0);
  }

  inline static state_t min(repr_type k) noexcept {
    return neon_kernel128::min(vminq_s8(k.val[0], k.val[1]));
  }

  inline static uint8_t min_lru(repr_type k) noexcept {
    return vminvq_u8(vminq_u8(vreinterpretq_u8_s8(k.val[0]), vreinterpretq_u8_s8(k.val[1])));
  }

  inline static mask_repr_type mask_min_lru(repr_type k) noexcept {
    return mask(k, static_cast<state_t>(min_lru(k)));
  }

  inline static size_t min_lru_index(repr_type k) noexcept {
    return mask_type::next(mask_min_lru(k));
  }

  inline static repr_type lru_update(const repr_type k, const repr_type l, const mask_repr_type m) {
    NVHM_ASSERT_(mask_type::has_next(m) && mask_type::next(m) < size, "m = ", render_mask<mask_type>(m));
    NVHM_ASSERT_(slot_is_occupied(at(k, mask_type::next(m))));

    return lru_update_(
      k, vreinterpretq_u8_s8(l.val[0]), vreinterpretq_u8_s8(l.val[1]),
      expand_mask_(mask_type::truncate(m))
    );
  }

 private:
  inline static mask_repr_type mask_(repr_type k, int8x16_t s) noexcept {
    return collapse_mask_(vceqq_s8(k.val[0], s), vceqq_s8(k.val[1], s));
  }

  /**
   * Truncate NEON 8-bit mask (0x00 or 0xff) into a 2-bit mask (0x1 or 0x2).
   */
  inline static mask_repr_type collapse_mask_(uint8x16_t m8_0, uint8x16_t m8_1) noexcept {
    const uint8x16_t bits{vreinterpretq_u8_u32(vdupq_n_u32(UINT32_C(0x4010'0401)))};
    m8_0 = vandq_u8(m8_0, bits);  // Isolate bits.
    m8_1 = vandq_u8(m8_1, bits);

    m8_0 = vpaddq_u8(m8_0, m8_1);  // 01|04, 10|40, ...
    m8_0 = vpaddq_u8(m8_0, m8_0);  // 01|04|10|40, ...

    return vgetq_lane_u64(vreinterpretq_u64_u8(m8_0), 0);
  }

  inline static uint8x16x2_t expand_mask_(mask_repr_type m) noexcept {
    const uint8x16_t m2{vreinterpretq_u8_u64(vdupq_n_u64(m))};
    const uint16x8_t m4{vreinterpretq_u16_u8(vzip1q_u8(m2, m2))};

    uint8x16_t m8_0{vreinterpretq_u8_u16(vzip1q_u16(m4, m4))};
    uint8x16_t m8_1{vreinterpretq_u8_u16(vzip2q_u16(m4, m4))};

    const uint8x16_t bits{vreinterpretq_u8_u32(vdupq_n_u32(UINT32_C(0x4010'0401)))};
    m8_0 = vtstq_u8(m8_0, bits);
    m8_1 = vtstq_u8(m8_1, bits);
    return {m8_0, m8_1};
  }

  inline static repr_type lru_update_(
    const repr_type k, uint8x16_t l0, uint8x16_t l1, const uint8x16x2_t m
  ) {
    // Mask slots that need rescaling.
    uint8x16_t r{vorrq_u8(
      vandq_u8(vceqq_u8(l0, m.val[0]), m.val[0]), vandq_u8(vceqq_u8(l1, m.val[1]), m.val[1])
    )};
    r = vdupq_n_u8(vmaxvq_u8(r));
    const uint8x16_t r0{vandq_u8(r, vcgezq_s8(k.val[0]))};
    const uint8x16_t r1{vandq_u8(r, vcgezq_s8(k.val[1]))};

    // Rescale and increment.
    l0 = vshlq_u8(l0, vreinterpretq_s8_u8(r0));
    l1 = vshlq_u8(l1, vreinterpretq_s8_u8(r1));

    l0 = vsubq_u8(l0, m.val[0]);
    l1 = vsubq_u8(l1, m.val[1]);

    return {vreinterpretq_s8_u8(l0), vreinterpretq_s8_u8(l1)};
  }
};

struct neon_kernel512 final : public kernel {
  using repr_type = int8x16x4_t;

  constexpr static size_t size{sizeof(repr_type)};
  static_assert(has_single_bit(size));
  constexpr static size_t size_mask{size - 1};

  using mask_type = uint_mask64_1_t;
  using mask_repr_type = typename mask_type::repr_type;
  constexpr static mask_repr_type full_mask{mask_type::full()};

  constexpr static state_t empty{-128};      // 0x80 == msb
  constexpr static state_t tombstone{-127};  // 0x81 == msb | lsb
  static_assert(empty < tombstone && tombstone < 0);

  neon_kernel512() = delete;

  inline static repr_type load(const state_t* __restrict ptr) noexcept {
    return vld1q_s8_x4(assume_aligned<size>(ptr));
  }

  inline static void store(repr_type k, state_t* __restrict ptr) noexcept {
    vst1q_s8_x4(assume_aligned<size>(ptr), k);
  }

  inline static mask_repr_type mask(repr_type k, state_t s) noexcept {
    return mask_(k, vdupq_n_s8(s));
  }

  inline static mask_repr_type mask_hash(repr_type k) noexcept {
    return collapse_mask_(
      vcgezq_s8(k.val[0]), vcgezq_s8(k.val[1]), vcgezq_s8(k.val[2]), vcgezq_s8(k.val[3])
    );
  }

  inline static mask_repr_type mask_non_hash(repr_type k) noexcept {
    return collapse_mask_(
      vcltzq_s8(k.val[0]), vcltzq_s8(k.val[1]), vcltzq_s8(k.val[2]), vcltzq_s8(k.val[3])
    );
  }

  inline static mask_repr_type mask_empty(repr_type k) noexcept {
    return mask_(k, neon_kernel128::all_empty());
  }

  inline static mask_repr_type mask_tombstone(repr_type k) noexcept {
    return mask_(k, neon_kernel128::all_tombstone());
  }

  inline static bool has_empty(repr_type k) noexcept { return min(k) == empty; }

  inline static repr_type hash_to_tombstone(repr_type k) noexcept {
    return {
      neon_kernel128::hash_to_tombstone(k.val[0]),
      neon_kernel128::hash_to_tombstone(k.val[1]),
      neon_kernel128::hash_to_tombstone(k.val[2]),
      neon_kernel128::hash_to_tombstone(k.val[3]),
    };
  }

  inline static state_t at(repr_type k, size_t i) noexcept {
    NVHM_ASSERT_(i < size, "i = ", i, ", size = ", size);
    const int8x16_t k0{vqtbl4q_s8(k, vdupq_n_u8(static_cast<uint8_t>(i)))};
    return vgetq_lane_s8(k0, 0);
  }

  inline static state_t min(repr_type k) noexcept {
    return neon_kernel128::min(vminq_s8(vminq_s8(k.val[0], k.val[1]), vminq_s8(k.val[2], k.val[3]))
    );
  }

  inline static uint8_t min_lru(repr_type k) noexcept {
    return vminvq_u8(vminq_u8(
      vminq_u8(vreinterpretq_u8_s8(k.val[0]), vreinterpretq_u8_s8(k.val[1])),
      vminq_u8(vreinterpretq_u8_s8(k.val[2]), vreinterpretq_u8_s8(k.val[3]))
    ));
  }

  inline static mask_repr_type mask_min_lru(repr_type k) noexcept {
    return mask(k, static_cast<state_t>(min_lru(k)));
  }

  inline static size_t min_lru_index(repr_type k) noexcept {
    return mask_type::next(mask_min_lru(k));
  }

  inline static repr_type lru_update(const repr_type k, const repr_type l, const mask_repr_type m) {
    NVHM_ASSERT_(mask_type::has_next(m) && mask_type::next(m) < size, "m = ", render_mask<mask_type>(m));
    NVHM_ASSERT_(slot_is_occupied(at(k, mask_type::next(m))));

    return lru_update_(
      k, vreinterpretq_u8_s8(l.val[0]), vreinterpretq_u8_s8(l.val[1]),
      vreinterpretq_u8_s8(l.val[2]), vreinterpretq_u8_s8(l.val[3]),
      expand_mask_(mask_type::truncate(m))
    );
  }

 private:
  inline static mask_repr_type mask_(repr_type k, int8x16_t s) noexcept {
    return collapse_mask_(
      vceqq_s8(k.val[0], s), vceqq_s8(k.val[1], s), vceqq_s8(k.val[2], s), vceqq_s8(k.val[3], s)
    );
  }

  /**
   * Truncate NEON 8-bit mask (0x00 or 0xff) into a 1-bit mask (0 or 1).
   */
  inline static mask_repr_type collapse_mask_(
    uint8x16_t m8_0, uint8x16_t m8_1, uint8x16_t m8_2, uint8x16_t m8_3
  ) noexcept {
    const uint8x16_t bits{vreinterpretq_u8_u64(vdupq_n_u64(UINT64_C(0x8040'2010'0804'0201)))};
    m8_0 = vandq_u8(m8_0, bits);  // Isolate bits.
    m8_1 = vandq_u8(m8_1, bits);
    m8_2 = vandq_u8(m8_2, bits);
    m8_3 = vandq_u8(m8_3, bits);

    m8_0 = vpaddq_u8(m8_0, m8_1);  // 01|02, 04|08, 10|20, 40|80, ...
    m8_2 = vpaddq_u8(m8_2, m8_3);
    m8_0 = vpaddq_u8(m8_0, m8_2);  // 01|02|04|08, 10|20|40|80, ...
    m8_0 = vpaddq_u8(m8_0, m8_0);  // 01|02|04|08|10|20|40|80, ...

    return vgetq_lane_u64(vreinterpretq_u64_u8(m8_0), 0);
  }

  inline static uint8x16x4_t expand_mask_(mask_repr_type m) noexcept {
    const uint32x2_t m1{vcreate_u32(m)};
    const uint8x16_t m1_0{vreinterpretq_u8_u32(vdupq_lane_u32(m1, 0))};
    const uint8x16_t m1_1{vreinterpretq_u8_u32(vdupq_lane_u32(m1, 1))};

    const uint16x8_t m2_0{vreinterpretq_u16_u8(vzip1q_u8(m1_0, m1_0))};
    const uint16x8_t m2_1{vreinterpretq_u16_u8(vzip1q_u8(m1_1, m1_1))};

    const uint32x4_t m4_0{vreinterpretq_u32_u16(vzip1q_u16(m2_0, m2_0))};
    const uint32x4_t m4_1{vreinterpretq_u32_u16(vzip1q_u16(m2_1, m2_1))};

    uint8x16_t m8_0{vreinterpretq_u8_u32(vzip1q_u32(m4_0, m4_0))};
    uint8x16_t m8_1{vreinterpretq_u8_u32(vzip2q_u32(m4_0, m4_0))};
    uint8x16_t m8_2{vreinterpretq_u8_u32(vzip1q_u32(m4_1, m4_1))};
    uint8x16_t m8_3{vreinterpretq_u8_u32(vzip2q_u32(m4_1, m4_1))};

    const uint8x16_t bits{vreinterpretq_u8_u64(vdupq_n_u64(UINT64_C(0x8040'2010'0804'0201)))};
    m8_0 = vtstq_u8(m8_0, bits);
    m8_1 = vtstq_u8(m8_1, bits);
    m8_2 = vtstq_u8(m8_2, bits);
    m8_3 = vtstq_u8(m8_3, bits);
    return {m8_0, m8_1, m8_2, m8_3};
  }

  inline static repr_type lru_update_(
    const repr_type k, uint8x16_t l0, uint8x16_t l1, uint8x16_t l2, uint8x16_t l3,
    const uint8x16x4_t m
  ) {
    // Mask slots that need rescaling.
    uint8x16_t r{vorrq_u8(
      vorrq_u8(
        vandq_u8(vceqq_u8(l0, m.val[0]), m.val[0]), vandq_u8(vceqq_u8(l1, m.val[1]), m.val[1])
      ),
      vorrq_u8(
        vandq_u8(vceqq_u8(l2, m.val[2]), m.val[2]), vandq_u8(vceqq_u8(l3, m.val[3]), m.val[3])
      )
    )};
    r = vdupq_n_u8(vmaxvq_u8(r));
    const uint8x16_t r0{vandq_u8(r, vcgezq_s8(k.val[0]))};
    const uint8x16_t r1{vandq_u8(r, vcgezq_s8(k.val[1]))};
    const uint8x16_t r2{vandq_u8(r, vcgezq_s8(k.val[2]))};
    const uint8x16_t r3{vandq_u8(r, vcgezq_s8(k.val[3]))};

    // Rescale and increment.
    l0 = vshlq_u8(l0, vreinterpretq_s8_u8(r0));
    l1 = vshlq_u8(l1, vreinterpretq_s8_u8(r1));
    l2 = vshlq_u8(l2, vreinterpretq_s8_u8(r2));
    l3 = vshlq_u8(l3, vreinterpretq_s8_u8(r3));

    l0 = vsubq_u8(l0, m.val[0]);
    l1 = vsubq_u8(l1, m.val[1]);
    l2 = vsubq_u8(l2, m.val[2]);
    l3 = vsubq_u8(l3, m.val[3]);

    return {
      vreinterpretq_s8_u8(l0), vreinterpretq_s8_u8(l1), vreinterpretq_s8_u8(l2),
      vreinterpretq_s8_u8(l3)
    };
  }
};

using neon_kernel64_t = neon_kernel64;
using neon_kernel128_t = neon_kernel128;
using neon_kernel256_t = neon_kernel256;
using neon_kernel512_t = neon_kernel512;
#endif

#if NVHM_WITH_SVE
template <size_t Size>
struct sve_kernel final : public kernel {
  using repr_type = svint8_t;

  constexpr static size_t size{Size};
  static_assert(has_single_bit(size));
  constexpr static size_t size_mask{size - 1};

  using mask_type = sve_mask<size>;
  using mask_repr_type = typename mask_type::repr_type;

  constexpr static state_t empty{-16};      // Can be used as an immediate for SVE comparison ops.
  constexpr static state_t tombstone{-15};  // Can be used as an immediate for SVE comparison ops.
  static_assert(empty < tombstone && tombstone < 0);

  sve_kernel() = delete;

  inline static repr_type load(const state_t* __restrict ptr) noexcept {
    return svld1_s8(mask_type::full(), assume_aligned<size>(ptr));
  }

  inline static void store(repr_type k, state_t* __restrict ptr) noexcept {
    svst1_s8(mask_type::full(), assume_aligned<size>(ptr), k);
  }

  inline static mask_repr_type mask(repr_type k, state_t s) noexcept {
    return svcmpeq_n_s8(mask_type::full(), k, s);
  }

  inline static mask_repr_type mask_hash(repr_type k) noexcept {
    return svcmpge_n_s8(mask_type::full(), k, 0);
  }

  inline static mask_repr_type mask_non_hash(const repr_type k) noexcept {
    return svcmplt_n_s8(mask_type::full(), k, 0);
  }

  inline static mask_repr_type mask_empty(const repr_type k) noexcept { return mask(k, empty); }

  inline static mask_repr_type mask_tombstone(const repr_type k) noexcept {
    return mask(k, tombstone);
  }

  inline static bool has_empty(const repr_type k) noexcept {
    return mask_type::has_next(mask_empty(k));
  }

  inline static repr_type hash_to_tombstone(repr_type k) noexcept {
    return svmin_n_s8_x(mask_type::full(), k, tombstone);
  }

  inline static state_t at(repr_type k, size_t i) noexcept {
    NVHM_ASSERT_(i < size, "i = ", i, ", size = ", size);
    k = svtbl_s8(k, svdup_n_u8(static_cast<uint8_t>(i)));
    return svlasta_s8(svpfalse_b(), k);
  }

  inline static uint8_t min_lru(repr_type k) noexcept {
    return svminv_u8(mask_type::full(), svreinterpret_u8(k));
  }

  inline static mask_repr_type mask_min_lru(repr_type k) noexcept {
    return mask(k, static_cast<state_t>(min_lru(k)));
  }

  inline static size_t min_lru_index(repr_type k) noexcept {
    return mask_type::next(mask_min_lru(k));
  }

  inline static repr_type lru_update(const repr_type k, const repr_type l, const mask_repr_type m) {
    NVHM_ASSERT_(mask_type::has_next(m) && mask_type::next(m) < size, "m = ", render_mask<mask_type>(m));
    NVHM_ASSERT_(slot_is_occupied(at(k, mask_type::next(m))));

    return lru_update_(k, svreinterpret_u8(l), mask_type::truncate(m));
  }

 private:
  inline static repr_type lru_update_(const repr_type k, svuint8_t l, const mask_repr_type m) {
    // Mask slots that need rescaling.
    svbool_t r{svcmpeq_n_u8(m, l, 0xff)};
    r = svbrkn_b_z(m, r, mask_hash(k));

    // Rescale and increment.
    l = svlsr_n_u8_m(r, l, 1);
    l = svadd_n_u8_m(m, l, 1);

    return svreinterpret_s8(l);
  }
};

using sve_kernel8_t = sve_kernel<1>;
using sve_kernel16_t = sve_kernel<2>;
using sve_kernel32_t = sve_kernel<4>;
using sve_kernel64_t = sve_kernel<8>;
using sve_kernel128_t = sve_kernel<16>;
using sve_kernel256_t = sve_kernel<32>;
using sve_kernel512_t = sve_kernel<64>;
using sve_kernel1024_t = sve_kernel<128>;
using sve_kernel2048_t = sve_kernel<256>;
#endif

using default_kernel8_t = uint_kernel8_t;
using default_kernel16_t = uint_kernel16_t;
using default_kernel32_t = uint_kernel32_t;

#if NVHM_WITH_SVE && NVHM_WITH_SVE_OVER_NEON
using default_kernel64_t =
  std::conditional_t<prefer_sve_over_neon, sve_kernel64_t, neon_kernel64_t>;
#elif NVHM_WITH_NEON
using default_kernel64_t = neon_kernel64_t;
#else
using default_kernel64_t = uint_kernel64_t;
#endif

#if NVHM_WITH_SSE2
using default_kernel128_t = sse_kernel_t;
#elif NVHM_WITH_SVE
using default_kernel128_t =
  std::conditional_t<prefer_sve_over_neon, sve_kernel128_t, neon_kernel128_t>;
#elif NVHM_WITH_NEON
using default_kernel128_t = neon_kernel128_t;
#else
using default_kernel128_t = uint_kernel128_t;
#endif

#if NVHM_WITH_AVX2
using default_kernel256_t = avx_kernel_t;
#elif NVHM_WITH_SVE && __ARM_FEATURE_SVE_BITS >= 256
using default_kernel256_t =
  std::conditional_t<prefer_sve_over_neon, sve_kernel256_t, neon_kernel256_t>;
#elif NVHM_WITH_NEON
using default_kernel256_t = neon_kernel256_t;
#else
using default_kernel256_t = array_kernel256_t;
#endif

#if NVHM_WITH_AVX512
using default_kernel512_t = avx512_kernel_t;
#elif NVHM_WITH_SVE && __ARM_FEATURE_SVE_BITS >= 512
using default_kernel512_t =
  std::conditional_t<prefer_sve_over_neon, sve_kernel512_t, neon_kernel512_t>;
#elif NVHM_WITH_NEON
using default_kernel512_t = neon_kernel512_t;
#else
using default_kernel512_t = array_kernel512_t;
#endif

#if NVHM_WITH_SVE && __ARM_FEATURE_SVE_BITS >= 1024
using default_kernel1024_t = sve_kernel1024_t;
#else
using default_kernel1024_t = array_kernel1024_t;
#endif

#if NVHM_WITH_SVE && __ARM_FEATURE_SVE_BITS >= 2048
using default_kernel2048_t = arm_sve_kernel2048_t;
#endif

using default_kernel_t = std::conditional_t<
  std::is_same_v<default_kernel128_t, uint_kernel128_t>, default_kernel64_t, default_kernel128_t>;

}  // namespace nvhm
