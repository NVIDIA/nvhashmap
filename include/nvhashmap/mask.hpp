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
#include <bitset>

namespace nvhm {

template<typename Mask>
constexpr void mask_to_array(typename Mask::repr_type m, bool (&a)[Mask::max_count]) noexcept {
  using mask_t = Mask;

  std::fill(std::begin(a), std::end(a), false);
  for (; mask_t::has_next(m); m = mask_t::step(m)) {
    int_t i{mask_t::next(m)};
    NVHM_ASSERT_(i >= 0 && i < mask_t::max_count);
    a[to_uint(i)] = true;
  }
}

template <typename Mask, typename Repr = typename Mask::repr_type>
inline std::string render_mask(Repr m) {
  using mask_t = Mask;

  std::ostringstream os;
  for (int_t i{}; mask_t::has_next(m); m = mask_t::step(m)) {
    os << (i++ ? ", " : "") << mask_t::next(m);
  }
  return os.str();
}

struct mask {
  NVHM_MAKE_NOT_INSTANTIABLE_(mask);
};

enum class mask_align_t {
  left,
  right
};

constexpr const char* to_string(mask_align_t a) noexcept {
  switch (a) {
    case mask_align_t::left:
      return "left";
    case mask_align_t::right:
      return "right";
  }
  return "error";
}

inline std::ostream& operator<<(std::ostream& os, mask_align_t a) {
  return os << to_string(a);
}

template <int_t MaxCount>
struct bitset_mask final : public mask {
  constexpr static int_t max_count{MaxCount};
  static_assert(max_count > 0);

  using repr_type = std::bitset<to_uint(max_count)>;
  
  NVHM_MAKE_NOT_INSTANTIABLE_(bitset_mask);

  constexpr static repr_type empty() noexcept { return {}; }
  constexpr static repr_type full() noexcept {
    repr_type m;
    m.set();
    return m;
  }
  constexpr static repr_type single(int_t i) noexcept {
    NVHM_ASSERT_(i >= 0 && i < max_count);
    repr_type m{};
    m[to_uint(i)] = true;
    return m;
  }

  constexpr static repr_type from(repr_type m, int_t i) noexcept {
    NVHM_ASSERT_(i >= 0 && i < max_count);
    for (int_t j{}; j < i; ++j) {
      m[to_uint(j)] = false;
    }
    return m;
  }
  constexpr static repr_type above(repr_type m, int_t i) noexcept {
    NVHM_ASSERT_(i >= -1 && i < max_count);
    for (int_t j{}; j <= i; ++j) {
      m[to_uint(j)] = false;
    }
    return m;
  }

  constexpr static repr_type intersect(const repr_type& m, const repr_type& n) noexcept { return m & n; }
  constexpr static repr_type join(const repr_type& m, const repr_type& n) noexcept { return m | n; }
  constexpr static repr_type drop(const repr_type& m, const repr_type& n) noexcept {return m & ~n; }
  constexpr static bool contains(const repr_type& m, const repr_type& n) noexcept { return (m & n) == n; }

  constexpr static bool has_next(const repr_type& m) noexcept { return m.any(); }
  constexpr static int_t count(const repr_type& m) noexcept { return to_int(m.count()); }

  constexpr static int_t next(const repr_type& m) noexcept {
    NVHM_ASSERT_(has_next(m));
    int_t i{};
    for (; !m[to_uint(i)]; ++i) {}
    return i;
  }
  constexpr static raw_pos_t next(const repr_type& m, raw_pos_t off) noexcept {
    return off + next(m);
  }

  constexpr static repr_type step(repr_type m) noexcept {
    int_t i{next(m)};
    m[to_uint(i)] = false;
    return m;
  }
  constexpr static repr_type truncate(const repr_type& m) noexcept {
    repr_type n{};
    for (int_t i{}; i < max_count; ++i) {
      if (m[to_uint(i)]) {
        n[to_uint(i)] = true;
        break;
      }
    }
    return n;
  }

  constexpr static bool at(const repr_type& m, int_t i) noexcept {
    NVHM_ASSERT_(i >= 0 && i < max_count);
    return m[to_uint(i)];
  }
};

using bitset_mask1_t = bitset_mask<1>;
using bitset_mask2_t = bitset_mask<2>;
using bitset_mask4_t = bitset_mask<4>;
using bitset_mask8_t = bitset_mask<8>;
using bitset_mask16_t = bitset_mask<16>;
using bitset_mask32_t = bitset_mask<32>;
using bitset_mask64_t = bitset_mask<64>;
using bitset_mask128_t = bitset_mask<128>;
using bitset_mask256_t = bitset_mask<256>;
using bitset_mask512_t = bitset_mask<512>;

template <typename Repr, int_t NumBits, int_t BitsPerSlot, mask_align_t Alignment>
struct uint_mask final : public mask {
  using repr_type = Repr;
  static_assert(is_unsigned_v<repr_type> && sizeof(repr_type) >= sizeof(uint32_t));

  constexpr static int_t max_num_bits{num_bits_v<repr_type>};
  static_assert(has_single_bit(max_num_bits));

  constexpr static int_t num_bits{NumBits};
  static_assert(num_bits > 0 && num_bits <= max_num_bits);
  static_assert(has_single_bit(num_bits));

  constexpr static int_t bits_per_slot{BitsPerSlot};
  static_assert(bits_per_slot > 0 && bits_per_slot <= 8);
  static_assert(has_single_bit(bits_per_slot));
  static_assert(num_bits % bits_per_slot == 0);
  constexpr static int_t shift{countr_zero(bits_per_slot)};
  constexpr static int_t max_count{num_bits / bits_per_slot};
  constexpr static bool fully_utilized{max_count * bits_per_slot == max_num_bits};

  constexpr static mask_align_t alignment{Alignment};
  static_assert(alignment == mask_align_t::left || alignment == mask_align_t::right);

  NVHM_MAKE_NOT_INSTANTIABLE_(uint_mask);

  constexpr static repr_type empty() noexcept { return {}; }
  constexpr static repr_type full() noexcept {
    repr_type m{broadcast<repr_type, uint8_t>(0x01)};
    if constexpr (bits_per_slot < 8) {
      m |= m << 4;
    }
    if constexpr (bits_per_slot < 4) {
      m |= m << 2;
    }
    if constexpr (bits_per_slot < 2) {
      m |= m << 1;
    }
    m >>= num_bits_v<repr_type> - max_count * bits_per_slot;
    if constexpr (alignment == mask_align_t::left) {
      m <<= bits_per_slot - 1;
    }
    return m;
  }
  constexpr static repr_type single(int_t i, bool v = true) noexcept {
    NVHM_ASSERT_(i >= 0 && i < max_count);
    repr_type m{v};
    m <<= i * bits_per_slot;
    if constexpr (alignment == mask_align_t::left) {
      m <<= bits_per_slot - 1;
    }
    return m;
  }

  constexpr static repr_type above(repr_type m, int_t i) noexcept {
    //repr_type n{i < 0 ? empty() : (single(i) << 1) - 1};
    int_t i0{std::max(i, {})};
    repr_type n{single(i0, i0 == i) << 1};
    n -= i0 == i;
    //repr_type n{(single(std::max(i, {}), i >= 0) << 1) - 1};
    return (m ^ n) & m;
  }
  constexpr static repr_type from(repr_type m, int_t i) noexcept {
    repr_type n{single(i) - 1};
    return (m ^ n) & m;
  }

  constexpr static repr_type intersect(repr_type m, repr_type n) noexcept { return m & n; }
  constexpr static repr_type join(repr_type m, repr_type n) noexcept { return m | n; }
  constexpr static repr_type drop(repr_type m, repr_type n) noexcept { return m & ~n; }
  constexpr static bool contains(repr_type m, repr_type n) noexcept { return (m & n) == n; }

  constexpr static bool has_next(repr_type m) noexcept { return m != 0; }
  constexpr static int_t count(repr_type m) noexcept { return popcount(m); }
  static_assert(count(empty()) == 0);
  static_assert(count(full()) == max_count);

  constexpr static int_t next(repr_type m) noexcept {
    NVHM_ASSERT_(has_next(m));
    return countr_zero(m) >> shift;
  }
  constexpr static raw_pos_t next(repr_type m, raw_pos_t off) noexcept {
    return off + next(m);
  }

  constexpr static repr_type step(repr_type m) noexcept { return m & (m - 1); }
  constexpr static repr_type truncate(repr_type m) noexcept { return m ^ step(m); }  // return m & (m ^ (m - 1));

  constexpr static bool at(repr_type m, int_t i) noexcept { return (m & single(i)) != 0; }
};

template <int_t NumBits, int_t BitsPerSlot, mask_align_t Alignment>
using uint32_mask_t = uint_mask<uint32_t, NumBits, BitsPerSlot, Alignment>;

template <int_t NumBits, int_t BitsPerSlot, mask_align_t Alignment>
using uint64_mask_t = uint_mask<uint64_t, NumBits, BitsPerSlot, Alignment>;

template <int_t NumBits, int_t BitsPerSlot, mask_align_t Alignment>
using uint128_mask_t = uint_mask<__uint128_t, NumBits, BitsPerSlot, Alignment>;

using uint32_mask1_1r_t = uint32_mask_t<1, 1, mask_align_t::right>;
using uint32_mask1_1l_t = uint32_mask_t<1, 1, mask_align_t::left>;

using uint32_mask2_1r_t = uint32_mask_t<2, 1, mask_align_t::right>;
using uint32_mask2_2r_t = uint32_mask_t<2, 2, mask_align_t::right>;
using uint32_mask2_1l_t = uint32_mask_t<2, 1, mask_align_t::left>;
using uint32_mask2_2l_t = uint32_mask_t<2, 2, mask_align_t::left>;

using uint32_mask4_1r_t = uint32_mask_t<4, 1, mask_align_t::right>;
using uint32_mask4_2r_t = uint32_mask_t<4, 2, mask_align_t::right>;
using uint32_mask4_4r_t = uint32_mask_t<4, 4, mask_align_t::right>;
using uint32_mask4_1l_t = uint32_mask_t<4, 1, mask_align_t::left>;
using uint32_mask4_2l_t = uint32_mask_t<4, 2, mask_align_t::left>;
using uint32_mask4_4l_t = uint32_mask_t<4, 4, mask_align_t::left>;

using uint32_mask8_1r_t = uint32_mask_t<8, 1, mask_align_t::right>;
using uint32_mask8_2r_t = uint32_mask_t<8, 2, mask_align_t::right>;
using uint32_mask8_4r_t = uint32_mask_t<8, 4, mask_align_t::right>;
using uint32_mask8_8r_t = uint32_mask_t<8, 8, mask_align_t::right>;
using uint32_mask8_1l_t = uint32_mask_t<8, 1, mask_align_t::left>;
using uint32_mask8_2l_t = uint32_mask_t<8, 2, mask_align_t::left>;
using uint32_mask8_4l_t = uint32_mask_t<8, 4, mask_align_t::left>;
using uint32_mask8_8l_t = uint32_mask_t<8, 8, mask_align_t::left>;

using uint32_mask16_1r_t = uint32_mask_t<16, 1, mask_align_t::right>;
using uint32_mask16_2r_t = uint32_mask_t<16, 2, mask_align_t::right>;
using uint32_mask16_4r_t = uint32_mask_t<16, 4, mask_align_t::right>;
using uint32_mask16_8r_t = uint32_mask_t<16, 8, mask_align_t::right>;
using uint32_mask16_1l_t = uint32_mask_t<16, 1, mask_align_t::left>;
using uint32_mask16_2l_t = uint32_mask_t<16, 2, mask_align_t::left>;
using uint32_mask16_4l_t = uint32_mask_t<16, 4, mask_align_t::left>;
using uint32_mask16_8l_t = uint32_mask_t<16, 8, mask_align_t::left>;

using uint32_mask32_1r_t = uint32_mask_t<32, 1, mask_align_t::right>;
using uint32_mask32_2r_t = uint32_mask_t<32, 2, mask_align_t::right>;
using uint32_mask32_4r_t = uint32_mask_t<32, 4, mask_align_t::right>;
using uint32_mask32_8r_t = uint32_mask_t<32, 8, mask_align_t::right>;
using uint32_mask32_1l_t = uint32_mask_t<32, 1, mask_align_t::left>;
using uint32_mask32_2l_t = uint32_mask_t<32, 2, mask_align_t::left>;
using uint32_mask32_4l_t = uint32_mask_t<32, 4, mask_align_t::left>;
using uint32_mask32_8l_t = uint32_mask_t<32, 8, mask_align_t::left>;

using uint64_mask1_1r_t = uint64_mask_t<1, 1, mask_align_t::right>;
using uint64_mask1_1l_t = uint64_mask_t<1, 1, mask_align_t::left>;

using uint64_mask2_1r_t = uint64_mask_t<2, 1, mask_align_t::right>;
using uint64_mask2_2r_t = uint64_mask_t<2, 2, mask_align_t::right>;
using uint64_mask2_1l_t = uint64_mask_t<2, 1, mask_align_t::left>;
using uint64_mask2_2l_t = uint64_mask_t<2, 2, mask_align_t::left>;

using uint64_mask4_1r_t = uint64_mask_t<4, 1, mask_align_t::right>;
using uint64_mask4_2r_t = uint64_mask_t<4, 2, mask_align_t::right>;
using uint64_mask4_4r_t = uint64_mask_t<4, 4, mask_align_t::right>;
using uint64_mask4_1l_t = uint64_mask_t<4, 1, mask_align_t::left>;
using uint64_mask4_2l_t = uint64_mask_t<4, 2, mask_align_t::left>;
using uint64_mask4_4l_t = uint64_mask_t<4, 4, mask_align_t::left>;

using uint64_mask8_1r_t = uint64_mask_t<8, 1, mask_align_t::right>;
using uint64_mask8_2r_t = uint64_mask_t<8, 2, mask_align_t::right>;
using uint64_mask8_4r_t = uint64_mask_t<8, 4, mask_align_t::right>;
using uint64_mask8_8r_t = uint64_mask_t<8, 8, mask_align_t::right>;
using uint64_mask8_1l_t = uint64_mask_t<8, 1, mask_align_t::left>;
using uint64_mask8_2l_t = uint64_mask_t<8, 2, mask_align_t::left>;
using uint64_mask8_4l_t = uint64_mask_t<8, 4, mask_align_t::left>;
using uint64_mask8_8l_t = uint64_mask_t<8, 8, mask_align_t::left>;

using uint64_mask16_1r_t = uint64_mask_t<16, 1, mask_align_t::right>;
using uint64_mask16_2r_t = uint64_mask_t<16, 2, mask_align_t::right>;
using uint64_mask16_4r_t = uint64_mask_t<16, 4, mask_align_t::right>;
using uint64_mask16_8r_t = uint64_mask_t<16, 8, mask_align_t::right>;
using uint64_mask16_1l_t = uint64_mask_t<16, 1, mask_align_t::left>;
using uint64_mask16_2l_t = uint64_mask_t<16, 2, mask_align_t::left>;
using uint64_mask16_4l_t = uint64_mask_t<16, 4, mask_align_t::left>;
using uint64_mask16_8l_t = uint64_mask_t<16, 8, mask_align_t::left>;

using uint64_mask32_1r_t = uint64_mask_t<32, 1, mask_align_t::right>;
using uint64_mask32_2r_t = uint64_mask_t<32, 2, mask_align_t::right>;
using uint64_mask32_4r_t = uint64_mask_t<32, 4, mask_align_t::right>;
using uint64_mask32_8r_t = uint64_mask_t<32, 8, mask_align_t::right>;
using uint64_mask32_1l_t = uint64_mask_t<32, 1, mask_align_t::left>;
using uint64_mask32_2l_t = uint64_mask_t<32, 2, mask_align_t::left>;
using uint64_mask32_4l_t = uint64_mask_t<32, 4, mask_align_t::left>;
using uint64_mask32_8l_t = uint64_mask_t<32, 8, mask_align_t::left>;

using uint64_mask64_1r_t = uint64_mask_t<64, 1, mask_align_t::right>;
using uint64_mask64_2r_t = uint64_mask_t<64, 2, mask_align_t::right>;
using uint64_mask64_4r_t = uint64_mask_t<64, 4, mask_align_t::right>;
using uint64_mask64_8r_t = uint64_mask_t<64, 8, mask_align_t::right>;
using uint64_mask64_1l_t = uint64_mask_t<64, 1, mask_align_t::left>;
using uint64_mask64_2l_t = uint64_mask_t<64, 2, mask_align_t::left>;
using uint64_mask64_4l_t = uint64_mask_t<64, 4, mask_align_t::left>;
using uint64_mask64_8l_t = uint64_mask_t<64, 8, mask_align_t::left>;

using uint128_mask1_1r_t = uint128_mask_t<1, 1, mask_align_t::right>;
using uint128_mask1_1l_t = uint128_mask_t<1, 1, mask_align_t::left>;

using uint128_mask2_1r_t = uint128_mask_t<2, 1, mask_align_t::right>;
using uint128_mask2_2r_t = uint128_mask_t<2, 2, mask_align_t::right>;
using uint128_mask2_1l_t = uint128_mask_t<2, 1, mask_align_t::left>;
using uint128_mask2_2l_t = uint128_mask_t<2, 2, mask_align_t::left>;

using uint128_mask4_1r_t = uint128_mask_t<4, 1, mask_align_t::right>;
using uint128_mask4_2r_t = uint128_mask_t<4, 2, mask_align_t::right>;
using uint128_mask4_4r_t = uint128_mask_t<4, 4, mask_align_t::right>;
using uint128_mask4_1l_t = uint128_mask_t<4, 1, mask_align_t::left>;
using uint128_mask4_2l_t = uint128_mask_t<4, 2, mask_align_t::left>;
using uint128_mask4_4l_t = uint128_mask_t<4, 4, mask_align_t::left>;

using uint128_mask8_1r_t = uint128_mask_t<8, 1, mask_align_t::right>;
using uint128_mask8_2r_t = uint128_mask_t<8, 2, mask_align_t::right>;
using uint128_mask8_4r_t = uint128_mask_t<8, 4, mask_align_t::right>;
using uint128_mask8_8r_t = uint128_mask_t<8, 8, mask_align_t::right>;
using uint128_mask8_1l_t = uint128_mask_t<8, 1, mask_align_t::left>;
using uint128_mask8_2l_t = uint128_mask_t<8, 2, mask_align_t::left>;
using uint128_mask8_4l_t = uint128_mask_t<8, 4, mask_align_t::left>;
using uint128_mask8_8l_t = uint128_mask_t<8, 8, mask_align_t::left>;

using uint128_mask16_1r_t = uint128_mask_t<16, 1, mask_align_t::right>;
using uint128_mask16_2r_t = uint128_mask_t<16, 2, mask_align_t::right>;
using uint128_mask16_4r_t = uint128_mask_t<16, 4, mask_align_t::right>;
using uint128_mask16_8r_t = uint128_mask_t<16, 8, mask_align_t::right>;
using uint128_mask16_1l_t = uint128_mask_t<16, 1, mask_align_t::left>;
using uint128_mask16_2l_t = uint128_mask_t<16, 2, mask_align_t::left>;
using uint128_mask16_4l_t = uint128_mask_t<16, 4, mask_align_t::left>;
using uint128_mask16_8l_t = uint128_mask_t<16, 8, mask_align_t::left>;

using uint128_mask32_1r_t = uint128_mask_t<32, 1, mask_align_t::right>;
using uint128_mask32_2r_t = uint128_mask_t<32, 2, mask_align_t::right>;
using uint128_mask32_4r_t = uint128_mask_t<32, 4, mask_align_t::right>;
using uint128_mask32_8r_t = uint128_mask_t<32, 8, mask_align_t::right>;
using uint128_mask32_1l_t = uint128_mask_t<32, 1, mask_align_t::left>;
using uint128_mask32_2l_t = uint128_mask_t<32, 2, mask_align_t::left>;
using uint128_mask32_4l_t = uint128_mask_t<32, 4, mask_align_t::left>;
using uint128_mask32_8l_t = uint128_mask_t<32, 8, mask_align_t::left>;

using uint128_mask64_1r_t = uint128_mask_t<64, 1, mask_align_t::right>;
using uint128_mask64_2r_t = uint128_mask_t<64, 2, mask_align_t::right>;
using uint128_mask64_4r_t = uint128_mask_t<64, 4, mask_align_t::right>;
using uint128_mask64_8r_t = uint128_mask_t<64, 8, mask_align_t::right>;
using uint128_mask64_1l_t = uint128_mask_t<64, 1, mask_align_t::left>;
using uint128_mask64_2l_t = uint128_mask_t<64, 2, mask_align_t::left>;
using uint128_mask64_4l_t = uint128_mask_t<64, 4, mask_align_t::left>;
using uint128_mask64_8l_t = uint128_mask_t<64, 8, mask_align_t::left>;

using uint128_mask128_1r_t = uint128_mask_t<128, 1, mask_align_t::right>;
using uint128_mask128_2r_t = uint128_mask_t<128, 2, mask_align_t::right>;
using uint128_mask128_4r_t = uint128_mask_t<128, 4, mask_align_t::right>;
using uint128_mask128_8r_t = uint128_mask_t<128, 8, mask_align_t::right>;
using uint128_mask128_1l_t = uint128_mask_t<128, 1, mask_align_t::left>;
using uint128_mask128_2l_t = uint128_mask_t<128, 2, mask_align_t::left>;
using uint128_mask128_4l_t = uint128_mask_t<128, 4, mask_align_t::left>;
using uint128_mask128_8l_t = uint128_mask_t<128, 8, mask_align_t::left>;

template<int_t NumBits, int_t BitsPerSlot, mask_align_t Alignment>
using uint_mask_t = std::conditional_t<NumBits * BitsPerSlot <= num_bits_v<uint32_t>,
  uint32_mask_t<NumBits, BitsPerSlot, Alignment>,
  std::conditional_t<NumBits * BitsPerSlot <= num_bits_v<uint64_t>,
    uint64_mask_t<NumBits, BitsPerSlot, Alignment>,
    uint128_mask_t<NumBits, BitsPerSlot, Alignment>
  >>;

}

#if NVHM_WITH_SVE
#include <arm_sve.h>

namespace nvhm {

// TODO: Check if we should use `full` more often to avoid shuffling around p-registers that cannot be used as gate.
template <int_t MaxCount>
struct sve_mask final : public mask {
#if defined(__ARM_FEATURE_SVE_BITS) && (NVHM_WITH_SVE_SIZE * 8 <= __ARM_FEATURE_SVE_BITS)
  using repr_type = svbool_t __attribute__((arm_sve_vector_bits(__ARM_FEATURE_SVE_BITS)));
#else
  using repr_type = svbool_t;
#endif

  constexpr static int_t max_count{MaxCount};
  static_assert(has_single_bit(max_count));

  NVHM_MAKE_NOT_INSTANTIABLE_(sve_mask);

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
  inline static repr_type single(int_t i) noexcept { return svpnext_b8(full(), until(i)); }
  inline static repr_type until(int_t i) noexcept {
    NVHM_ASSERT_(i >= 0 && i < max_count);
    return svwhilelt_b8_s64(0, i);
  }
  inline static repr_type to(int_t i) noexcept {
    NVHM_ASSERT_(i >= -1 && i < max_count);
    return svwhilele_b8_s64(0, i);
  }
  
  inline static repr_type from(repr_type m, int_t i) noexcept { return drop(m, until(i)); }
  inline static repr_type above(repr_type m, int_t i) noexcept { return drop(m, to(i)); }

  inline static repr_type intersect(repr_type m, repr_type n) noexcept {
    // Tends to produce better code on Clang, then predicating on svptrue_b8().
    return svand_b_z(m, m, n);
  }
  inline static repr_type join(repr_type m, repr_type n) noexcept {
    // OR with svptrue_b8() or full() materializes the predicate with Clang and GCC.
    // return svorr_b_z(svptrue_b8(), m, n);
    return svsel_b(m, m, n);
  }
  inline static repr_type drop(repr_type m, repr_type n) noexcept { return svnot_b_z(m, n); }
  inline static bool contains(repr_type m, repr_type n) noexcept {
    n = sveor_b_z(n, n, m);
    return !svptest_any(n, n);
  }

  inline static bool has_next(repr_type m) noexcept { return svptest_any(m, m); }
  inline static int_t count(repr_type m) noexcept {
    // Both, svptrue_b8() will be eliminated on both Clang and GCC in actual loops
    // Of note: GCC won't optimize svcntp_b8(m, m) for some reason.
    return static_cast<int_t>(svcntp_b8(svptrue_b8(), m));
  }
  
  inline static int_t next(repr_type m) noexcept {
    NVHM_ASSERT_(has_next(m));
    return count(svbrkb_b_z(full(), m));
  }
  inline static raw_pos_t next(repr_type m, raw_pos_t off) noexcept {
    // Confirmed to produce `incp` on both Clang and GCC.
    return off + next(m);
    // m = svbrkb_b_z(svptrue_b8(), m);
    // if constexpr (sizeof(raw_pos_t) == sizeof(int64_t)) {
    //   return svqincp_n_s64_b8(off, m);
    // } else if constexpr (sizeof(raw_pos_t) == sizeof(int32_t)) {
    //   return svqincp_n_s32_b8(off, m);
    // } else {
    //   static_assert(dependent_type_false_v<raw_pos_t>, "Shouldn't happen!");
    // }
  }

  inline static repr_type step(repr_type m) noexcept { return svnot_b_z(m, truncate(m)); }
  inline static repr_type truncate(repr_type m) noexcept { return svbrka_b_z(m, m); }

  inline static bool at(repr_type m, int_t i) noexcept {
    NVHM_ASSERT_(i >= 0);
    return svptest_last(to(i), m);
  }
};

template <>
constexpr int_t num_bits_v<svbool_t>{NVHM_WITH_SVE_SIZE};

#if NVHM_WITH_SVE_SIZE >= 1
using sve_mask1_t = sve_mask<1>;
#endif
#if NVHM_WITH_SVE_SIZE >= 2
using sve_mask2_t = sve_mask<2>;
#endif
#if NVHM_WITH_SVE_SIZE >= 4
using sve_mask4_t = sve_mask<4>;
#endif
#if NVHM_WITH_SVE_SIZE >= 8
using sve_mask8_t = sve_mask<8>;
#endif
#if NVHM_WITH_SVE_SIZE >= 16
using sve_mask16_t = sve_mask<16>;
#endif
#if NVHM_WITH_SVE_SIZE >= 32
using sve_mask32_t = sve_mask<32>;
#endif
#if NVHM_WITH_SVE_SIZE >= 64
using sve_mask64_t = sve_mask<64>;
#endif
#if NVHM_WITH_SVE_SIZE >= 128
using sve_mask128_t = sve_mask<128>;
#endif
#if NVHM_WITH_SVE_SIZE >= 256
using sve_mask256_t = sve_mask<256>;
#endif

}  // namespace nvhm
#endif

namespace nvhm {

template <int_t Size>
struct default_mask;

template <>
struct default_mask<1> {
  // TODO: Prefer SVE if available?
  using type = uint32_mask1_1l_t;
};

template <>
struct default_mask<2> {
  // TODO: Prefer SVE if available?
  using type = uint32_mask2_1l_t;
};

template <>
struct default_mask<4> {
  // TODO: Prefer SVE if available?
  using type = uint32_mask4_1l_t;
};

template <>
struct default_mask<8> {
  // TODO: Prefer SVE if available?
  using type = uint32_mask8_1l_t;
};

template <>
struct default_mask<16> {
#if NVHM_WITH_SVE && NVHM_WITH_SVE_SIZE >= 16
  using type = sve_mask16_t;
#else
  using type = uint32_mask16_1l_t;
#endif
};

template <>
struct default_mask<32> {
#if NVHM_WITH_SVE && NVHM_WITH_SVE_SIZE >= 32
  using type = sve_mask32_t;
#else
  using type = uint32_mask32_1l_t;
#endif
};

template <>
struct default_mask<64> {
#if NVHM_WITH_SVE && NVHM_WITH_SVE_SIZE >= 64
  using type = sve_mask64_t;
#else
  using type = uint64_mask64_1l_t;
#endif
};

template <>
struct default_mask<128> {
#if NVHM_WITH_SVE && NVHM_WITH_SVE_SIZE >= 128
  using type = sve_mask128_t;
#else
  using type = uint128_mask128_1l_t;
#endif
};

template <>
struct default_mask<256> {
#if NVHM_WITH_SVE && NVHM_WITH_SVE_SIZE >= 256
  using type = sve_mask256_t;
#else
  using type = bitset_mask256_t;
#endif
};

template <>
struct default_mask<512> {
  using type = bitset_mask512_t;
};

using default_mask1_t = typename default_mask<1>::type;
using default_mask2_t = typename default_mask<2>::type;
using default_mask4_t = typename default_mask<4>::type;
using default_mask8_t = typename default_mask<8>::type;
using default_mask16_t = typename default_mask<16>::type;
using default_mask32_t = typename default_mask<32>::type;
using default_mask64_t = typename default_mask<64>::type;
using default_mask128_t = typename default_mask<128>::type;
using default_mask256_t = typename default_mask<256>::type;
using default_mask512_t = typename default_mask<512>::type;

constexpr int_t default_mask_size{
#if NVHM_WITH_SVE && NVHM_WITH_SVE_SIZE >= 16
  num_bits_v<svbool_t>
#else
  sizeof(int_t)
#endif
};

template <int_t Size = default_mask_size>
using default_mask_t = typename default_mask<Size>::type;

}  // namespace nvhm