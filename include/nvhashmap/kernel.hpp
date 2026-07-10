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
#include "mask.hpp"

#include <array>

namespace nvhm {

template <typename Kernel, typename Count>
void count_collisions(const state_t* states, std::array<Count, Kernel::size>& acc) noexcept {
  using kern_t = Kernel;
  using mask_t = typename kern_t::mask_type;

  auto k{kern_t::load(states)};

  std::array<state_t, kern_t::size> tmp;
  auto it{tmp.begin()};

  for (auto m{kern_t::mask_hash(k)}; mask_t::has_next(m); m = mask_t::step(m)) {
    const int_t idx{mask_t::next(m)};
    NVHM_ASSERT_(idx < kern_t::size);
    *it++ = kern_t::at(k, idx);
  }
  // Not exactly sure why GCC complains about this. I cannot see how this can even happen?!
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-warning-option"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds="
  std::sort(tmp.begin(), it);
#pragma GCC diagnostic pop
#pragma clang diagnostic pop
#pragma GCC diagnostic pop

  it = std::unique(tmp.begin(), it);
  while (it-- > tmp.begin()) {
    const auto m{kern_t::mask(k, *it)};
    const int_t i{mask_t::count(m) - 1};
    NVHM_ASSERT_(i >= 0 && i < kern_t::size);
    ++acc[to_uint(i)];
  }
}

template <typename Kernel>
inline std::string render_kernel(const typename Kernel::repr_type& k) {
  using kern_t = Kernel;

  std::ostringstream os;
  os << std::hex;
  for (int_t i{}; i < kern_t::size; ++i) {
    os << (i == 0 ? "" : ", ");
    os << std::setw(2) << (kern_t::at(k, i) & 0xff);
  }
  return os.str();
}

template <typename Kernel>
inline std::string render_lru_kernel(const typename Kernel::lru_repr_type& l) {
  using kern_t = Kernel;

  std::ostringstream os;
  os << std::hex;
  for (int_t i{}; i < kern_t::size; ++i) {
    os << (i == 0 ? "" : ", ");
    os << std::setw(2) << (kern_t::lru_at(l, i) & 0xff);
  }
  return os.str();
}

template <state_t Empty, state_t Tombstone, bool ProducesFalsePositives>
struct kernel {
  constexpr static state_t empty{Empty};
  constexpr static state_t tombstone{Tombstone};
  static_assert(empty < tombstone && tombstone < 0);

  /**
   * @return Whether the kernel may produce false positives in its masks. This only affects `mask(k, s)`.
   */
  constexpr static bool produces_false_positives{ProducesFalsePositives};

  NVHM_MAKE_NOT_INSTANTIABLE_(kernel);
};

// empty: 0x80 == msb
// tombstone: 0x81 == msb | lsb
template <int_t Size>
struct array_kernel final : public kernel<-128, -127, false> {
  constexpr static int_t size{Size};
  constexpr static bitmask_t size_mask{size_mask_v<size>};

  using repr_type = std::array<state_t, size>;
  using lru_repr_type = std::array<lru_t, size>;

  using mask_type = std::conditional_t<size <= num_bits_v<__uint128_t>,
    uint_mask_t<size, 1, mask_align_t::right>,
    bitset_mask<size>
  >;
  static_assert(mask_type::max_count == size);

  using mask_repr_type = typename mask_type::repr_type;
  static_assert(num_bits_v<mask_repr_type> >= size);

  NVHM_MAKE_NOT_INSTANTIABLE_(array_kernel);

  /**
   * Loads slot states from memory. The memory address is guaranteed to be aligned.
   */
  inline static repr_type load(const state_t* ptr) noexcept {
    repr_type k;
    std::copy_n(assume_aligned<std::min(size, cache_line_size)>(ptr), k.size(), k.begin());
    return k;
  }

  /**
   * Stores slot states to memory. The memory address is guaranteed to be aligned.
   */
  inline static void store(const repr_type& k, state_t* ptr) noexcept {
    std::copy_n(k.begin(), k.size(), assume_aligned<std::min(size, cache_line_size)>(ptr));
  }

  /**
   * Masks slots that match the provided state.
   */
  constexpr static mask_repr_type mask(const repr_type& k, state_t s) noexcept {
    mask_repr_type m{};
    for (int_t i{}; i < size; ++i) {
      m |= static_cast<mask_repr_type>(at(k, i) == s) << to_uint(i);
    }
    return m;
  }
  /**
   * Masks slots that do not match the provided state.
   */
  constexpr static mask_repr_type mask_not(const repr_type& k, state_t s) noexcept {
    mask_repr_type m{};
    for (int_t i{}; i < size; ++i) {
      m |= static_cast<mask_repr_type>(at(k, i) != s) << to_uint(i);
    }
    return m;
  }

  /**
   * Returns mask that identifies all slots that contains actual hashes.
   */
  constexpr static mask_repr_type mask_hash(const repr_type& k) noexcept {
    mask_repr_type m{};
    for (int_t i{}; i < size; ++i) {
      m |= static_cast<mask_repr_type>(is_hash(at(k, i))) << to_uint(i);
    }
    return m;
  }
  /**
   * Returns mask that identifies all slots that are either `empty` or `tombstones`.
   */
  constexpr static mask_repr_type mask_not_hash(const repr_type& k) noexcept {
    mask_repr_type m{};
    for (int_t i{}; i < size; ++i) {
      m |= static_cast<mask_repr_type>(is_not_hash(at(k, i))) << to_uint(i);
    }
    return m;
  }

  /**
   * Returns mask that identifies all empty slots. May assume only valid values are present.
   */
  constexpr static mask_repr_type mask_empty(const repr_type& k) noexcept { return mask(k, empty); }
  /**
   * Returns mask that identifies all slots that are not `empty`.
   */
  constexpr static mask_repr_type mask_not_empty(const repr_type& k) noexcept { return mask_not(k, empty); }
  /**
   * Returns mask that identifies all tombstone slots. May assume only valid values are present.
   */
  constexpr static mask_repr_type mask_tombstone(const repr_type& k) noexcept { return mask(k, tombstone); }
  /**
   * Returns mask that identifies all slots that are not `tombstone`.
   */
  constexpr static mask_repr_type mask_not_tombstone(const repr_type& k) noexcept { return mask_not(k, tombstone); }

  /**
   * Returns mask that identifies all slots that are equal.
   */
  constexpr static mask_repr_type mask_equal(const repr_type& k0, const repr_type& k1) noexcept {
    mask_repr_type m{};
    for (int_t i{}; i < size; ++i) {
      m |= static_cast<mask_repr_type>(at(k0, i) == at(k1, i)) << to_uint(i);
    }
    return m;
  }
  /**
   * Returns mask that identifies all slots that are equal.
   */
  constexpr static mask_repr_type mask_not_equal(const repr_type& k0, const repr_type& k1) noexcept {
    mask_repr_type m{};
    for (int_t i{}; i < size; ++i) {
      m |= static_cast<mask_repr_type>(at(k0, i) != at(k1, i)) << to_uint(i);
    }
    return m;
  }

  /**
   * Returns true if the kernel contains the given value.
   */
  constexpr static bool has(const repr_type& k, const state_t s) noexcept {
    return std::any_of(k.begin(), k.end(), [s](state_t x) { return x == s; });
  }
  /**
   * Returns true if the kernel does not contain the given value.
   */
  constexpr static bool has_not(const repr_type& k, const state_t s) noexcept {
    return std::any_of(k.begin(), k.end(), [s](state_t x) { return x != s; });
  }

  /**
   * Returns true if the kernel contains any hash values.
   */
  constexpr static bool has_hash(const repr_type& k) noexcept {
    return std::any_of(k.begin(), k.end(), [](state_t s) { return is_hash(s); });
  }
  /**
   * Returns true if the kernel contains any non-hash values.
   */
  constexpr static bool has_not_hash(const repr_type& k) noexcept {
    return std::any_of(k.begin(), k.end(), [](state_t s) { return is_not_hash(s); });
  }

  /**
   * Returns true if the kernel contains any `empty` values.
   */
  constexpr static bool has_empty(const repr_type& k) noexcept { return has(k, empty); }
  /**
   * Returns true if the kernel contains any not `empty` values.
   */
  constexpr static bool has_not_empty(const repr_type& k) noexcept { return has_not(k, empty); }
  /**
   * Returns true if the kernel contains any `tombstone` values.
   */
  constexpr static bool has_tombstone(const repr_type& k) noexcept { return has(k, tombstone); }
  /**
   * Returns true if the kernel contains any not `tombstone` values.
   */
  constexpr static bool has_not_tombstone(const repr_type& k) noexcept { return has_not(k, tombstone); }

  /**
   * Returns the number of slots that are equal.
   */
  constexpr static int_t count_equal(const repr_type& k0, const repr_type& k1) noexcept {
    int_t n{};
    for (int_t i{}; i < size; ++i) {
      n += at(k0, i) == at(k1, i);
    }
    return n;
  }
  /**
   * Returns the number of slots that are notequal.
   */
  constexpr static int_t count_not_equal(const repr_type& k0, const repr_type& k1) noexcept {
    int_t n{};
    for (int_t i{}; i < size; ++i) {
      n += at(k0, i) != at(k1, i);
    }
    return n;
  }

  /**
   * Returns the number of slots that are `empty`.
   */
  constexpr static int_t count_empty(const repr_type& k) noexcept {
    return std::count_if(k.begin(), k.end(), [](state_t s) { return s == empty; });
  }
  /**
   * Returns the number of slots that are `tombstone`.
   */
  constexpr static int_t count_tombstone(const repr_type& k) noexcept {
    return std::count_if(k.begin(), k.end(), [](state_t s) { return s == tombstone; });
  }

  constexpr static repr_type hash_to_tombstone(const repr_type& k) noexcept {
    // 0x00 ~ 0x7f => 0x81
    // 0x80 => 0x80
    // 0x81 => 0x81
    repr_type r;
    for (int_t i{}; i < size; ++i) {
      set_at(r, i, std::min(at(k, i), tombstone));
    }
    return r;
  }

  constexpr static repr_type not_hash_to_empty(const repr_type& k) noexcept {
    // 0x00 ~ 0x7f => keep
    // 0x80 ~ 0x81 => 0x80
    constexpr uint_t e{to_uint(empty)};
    repr_type r;
    for (int_t i{}; i < size; ++i) {
      uint_t ki{to_uint(at(k, i))};
      set_at(r, i, static_cast<state_t>(std::min(ki, e)));
    }
    return r;
  }

  constexpr static repr_type to_empty(const repr_type& k, mask_repr_type m) noexcept {
    repr_type r;
    for (int_t i{}; i < size; ++i) {
      set_at(r, i, mask_type::at(m, i) ? empty : at(k, i));
    }
    return r;
  }

  constexpr static state_t at(const repr_type& k, int_t i) noexcept {
    NVHM_ASSERT_(i < size, "i = ", i, ", size = ", size);
    return k[to_uint(i)];
  }

  constexpr static void set_at(repr_type& k, int_t i, state_t s) noexcept {
    NVHM_ASSERT_(i < size, "i = ", i, ", size = ", size);
    k[to_uint(i)] = s;
  }

  inline static lru_repr_type load_lru(const lru_t* ptr) noexcept {
    lru_repr_type l;
    std::copy_n(assume_aligned<std::min(size, cache_line_size)>(ptr), size, l.begin());
    return l;
  }

  inline static void store_lru(const lru_repr_type& l, lru_t* ptr) noexcept {
    std::copy_n(l.begin(), size, assume_aligned<std::min(size, cache_line_size)>(ptr));
  }

  constexpr static std::pair<lru_t, int_t> min_lru(const lru_repr_type& l) noexcept {
    lru_t lru{lru_at(l, 0)};
    int_t idx{};
    for (int_t i{1}; i < size; ++i) {
      lru_t li{lru_at(l, i)};
      if (li < lru) {
        lru = li;
        idx = i;
      }
    }
    return {lru, idx};
  }

  constexpr static mask_repr_type mask_min_lru(const lru_repr_type& l) noexcept {
    return mask_type::single(min_lru(l).second);
  }

  constexpr static mask_repr_type mask_lru_below(const lru_repr_type& l, lru_t t) noexcept {
    mask_repr_type m{};
    for (int_t i{}; i < size; ++i) {
      if (l[to_uint(i)] < t) {
        m |= mask_type::single(i);
      }
    }
    return m;
  }

  constexpr static lru_repr_type update_lru(const lru_repr_type& l, mask_repr_type n) noexcept {
    NVHM_ASSERT_(mask_type::has_next(n) && mask_type::next(n) < size, "n = ", render_mask<mask_type>(n));
    const int_t i{mask_type::next(n)};

    int_t rescale_shift{lru_at(l, i) >= max_lru};
    lru_repr_type r;
    for (int_t j{}; j < size; ++j) {
      uint_t lj{lru_at(l, j)};
      lj = (lj >> rescale_shift) + (j == i);
      set_lru_at(r, j, static_cast<lru_t>(lj));
    }
    return r;
  }

  constexpr static repr_type to_empty_if_lru_below(const repr_type& k, const lru_repr_type& l, lru_t t) noexcept {
    repr_type r;
    for (int_t i{}; i < size; ++i) {
      set_at(r, i, lru_at(l, i) < t ? empty : at(k, i));
    }
    return r;
  }

  constexpr static lru_t lru_at(const lru_repr_type& l, int_t i) noexcept {
    NVHM_ASSERT_(i < size, "i = ", i, ", size = ", size);
    return l[to_uint(i)];
  }

  constexpr static void set_lru_at(lru_repr_type& l, int_t i, lru_t lru) noexcept {
    NVHM_ASSERT_(i < size, "i = ", i, ", size = ", size);
    l[to_uint(i)] = lru;
  }
};

using array_kernel1_t = array_kernel<1>;
using array_kernel2_t = array_kernel<2>;
using array_kernel4_t = array_kernel<4>;
using array_kernel8_t = array_kernel<8>;
using array_kernel16_t = array_kernel<16>;
using array_kernel32_t = array_kernel<32>;
using array_kernel64_t = array_kernel<64>;
using array_kernel128_t = array_kernel<128>;
using array_kernel256_t = array_kernel<256>;
using array_kernel512_t = array_kernel<512>;

// empty: 0x80 == msb
// tombstone: 0x81 == msb | lsb
template <typename Repr, bool ProducesFalsePositives>
struct uint_kernel final : public kernel<-128, -127, ProducesFalsePositives> {
  using base_type = kernel<-128, -127, ProducesFalsePositives>;

  using repr_type = Repr;
  static_assert(std::is_unsigned_v<repr_type>);
  using lru_repr_type = Repr;
  static_assert(std::is_unsigned_v<lru_repr_type>);

  using base_type::produces_false_positives;

  constexpr static int_t size{num_bytes_v<repr_type>};
  constexpr static bitmask_t size_mask{size_mask_v<size>};

  using mask_type = uint_mask_t<num_bits_v<repr_type>, 8, mask_align_t::left>;
  static_assert(mask_type::max_count == size);

  using mask_repr_type = typename mask_type::repr_type;
  static_assert(num_bytes_v<mask_repr_type> >= size);

  constexpr static mask_repr_type lsbs{broadcast<repr_type, uint8_t>(0x01)};
  constexpr static mask_repr_type msbs{broadcast<repr_type, uint8_t>(0x80)};
  constexpr static mask_repr_type all_bits{broadcast<repr_type, uint8_t>(0xff)};

  NVHM_MAKE_NOT_INSTANTIABLE_(uint_kernel);

  /**
   * Loads slot states from memory. The memory address is guaranteed to be aligned.
   */
  inline static repr_type load(const state_t* ptr) noexcept {
    return *reinterpret_cast<const repr_type*>(assume_aligned<size>(ptr));
  }

  /**
   * Stores slot states to memory. The memory address is guaranteed to be aligned.
   */
  inline static void store(repr_type k, state_t* ptr) noexcept {
    *reinterpret_cast<repr_type*>(assume_aligned<size>(ptr)) = k;
  }

  /**
   * Masks slots that match the provided hash (may return false positives).
   */
  constexpr static mask_repr_type mask(repr_type k, state_t s) noexcept {
    mask_repr_type m{k};
    m ^= lsbs * static_cast<uint8_t>(s);
    if constexpr (produces_false_positives) {
      // Compiles to one assembly instruction less.
      // https://graphics.stanford.edu/~seander/bithacks.html#ZeroInWord
      m = (m - lsbs) & (~m & msbs);
    } else {
      m = ((~m & ~msbs) + lsbs) & (~m & msbs);
    }
    return m;
  }
  constexpr static mask_repr_type mask_not(repr_type k, state_t s) noexcept {
    return mask(k, s) ^ msbs;
  }

  constexpr static mask_repr_type mask_hash(repr_type k) noexcept {
    mask_repr_type m{k};
    return ~m & msbs;
  }
  constexpr static mask_repr_type mask_not_hash(repr_type k) noexcept { return k & msbs; }

  /**
   * Returns mask that identifies all empty slots. May assume only valid values are present.
   *   0 ~ 0  =>  0
   *   0 ~ 1  =>  0
   *   1 ~ 0  =>  1
   *   1 ~ 1  =>  0
   */
  constexpr static mask_repr_type mask_empty(repr_type k) noexcept { return k & (~k << 7) & msbs; }
  constexpr static mask_repr_type mask_not_empty(repr_type k) noexcept { return mask_empty(k) ^ msbs; }
  constexpr static mask_repr_type mask_tombstone(repr_type k) noexcept { return k & (k << 7) & msbs; }
  constexpr static mask_repr_type mask_not_tombstone(repr_type k) noexcept { return mask_tombstone(k) ^ msbs; }

  constexpr static mask_repr_type mask_equal(repr_type k0, repr_type k1) noexcept {
    mask_repr_type m{k0};
    m ^= k1;
    m = ((~m & ~msbs) + lsbs) & (~m & msbs);
    return m;
  }
  constexpr static mask_repr_type mask_not_equal(repr_type k0, repr_type k1) noexcept {
    return mask_equal(k0, k1) ^ msbs;
  }

  constexpr static bool has_hash(repr_type k) noexcept { return mask_hash(k) != 0; }
  constexpr static bool has_not_hash(repr_type k) noexcept { return mask_not_hash(k) != 0; }

  constexpr static bool has_empty(repr_type k) noexcept { return mask_empty(k) != 0; }
  constexpr static bool has_not_empty(repr_type k) noexcept { return mask_not_empty(k) != 0; }
  constexpr static bool has_tombstone(repr_type k) noexcept { return mask_tombstone(k) != 0; }
  constexpr static bool has_not_tombstone(repr_type k) noexcept { return mask_not_tombstone(k) != 0; }

  constexpr static int_t count_equal(const repr_type& k0, const repr_type& k1) noexcept {
    return mask_type::count(mask_equal(k0, k1));
  }
  constexpr static int_t count_not_equal(const repr_type& k0, const repr_type& k1) noexcept {
    return mask_type::count(mask_not_equal(k0, k1));
  }

  constexpr static repr_type hash_to_tombstone(repr_type k) noexcept {
    // 0x00 ~ 0x7f => 0x81
    // 0x80 => 0x80
    // 0x81 => 0x81
    mask_repr_type m{k};
    return msbs | (((~m >> 7) | m) & lsbs);
  }

  constexpr static repr_type not_hash_to_empty(repr_type k) noexcept {
    // 0x00 ~ 0x7f => keep
    // 0x80 ~ 0x81 => 0x80
    mask_repr_type m{k};
    return static_cast<repr_type>(m & (~lsbs | (~m >> 7)));
  }

  constexpr static repr_type to_empty(repr_type k, mask_repr_type m) noexcept {
    m = (m >> 7) * 0xff;
    // https://graphics.stanford.edu/~seander/bithacks.html#MaskedMerge
    m = k ^ ((k ^ msbs) & m);
    return static_cast<repr_type>(m);
  }

  constexpr static state_t at(repr_type k, int_t i) noexcept {
    NVHM_ASSERT_(i < size, "i = ", i, ", size = ", size);
    return static_cast<state_t>(k >> (i * num_bits_v<state_t>));
  }

  inline static lru_repr_type load_lru(const lru_t* ptr) noexcept {
    return *reinterpret_cast<const lru_repr_type*>(assume_aligned<size>(ptr));
  }

  inline static void store_lru(lru_repr_type l, lru_t* ptr) noexcept {
    *reinterpret_cast<lru_repr_type*>(assume_aligned<size>(ptr)) = l;
  }

  constexpr static std::pair<lru_t, int_t> min_lru(lru_repr_type l) noexcept {
    lru_t lru{lru_at(l, 0)};
    int_t idx{};
    for (int_t i{1}; i < size; ++i) {
      lru_t li{lru_at(l, i)};
      if (li < lru) {
        lru = li;
        idx = i;
      }
    }
    return {lru, idx};
  }

  constexpr static mask_repr_type mask_min_lru(lru_repr_type l) noexcept {
    return mask_type::single(min_lru(l).second);
  }

  constexpr static mask_repr_type mask_lru_below(lru_repr_type l, lru_t t) noexcept {
    mask_repr_type m{};
    for (int_t i{}; i < size; ++i) {
      mask_repr_type c{lru_at(l, i) < t};
      m |= c << (i * num_bits_v<lru_t>);
    }
    return m << 7;
  }

  constexpr static lru_repr_type update_lru(lru_repr_type l, mask_repr_type n) noexcept {
    NVHM_ASSERT_(mask_type::has_next(n) && mask_type::next(n) < size, "n = ", render_mask<mask_type>(n));
    int_t i{mask_type::next(n)};

    int_t rescale_shift{lru_at(l, i) >= max_lru};
    n = l >> rescale_shift;
    n &= rescale_shift ? ~msbs : all_bits;

    l = 1;
    l <<= i * num_bits_v<lru_t>;
    n += l;
    return static_cast<lru_repr_type>(n);
  }

  constexpr static repr_type to_empty_if_lru_below(repr_type k, lru_repr_type l, lru_t t) noexcept {
    return to_empty(k, mask_lru_below(l, t));
  }

  constexpr static lru_t lru_at(lru_repr_type l, int_t i) noexcept {
    NVHM_ASSERT_(i < size, "i = ", i, ", size = ", size);
    return static_cast<lru_t>(l >> (i * num_bits_v<lru_t>));
  }
};

using uint_kernel1_t = uint_kernel<uint8_t, false>;
using uint_kernel2_t = uint_kernel<uint16_t, false>;
using uint_kernel4_t = uint_kernel<uint32_t, false>;
using uint_kernel8_t = uint_kernel<uint64_t, false>;
using uint_kernel16_t = uint_kernel<__uint128_t, false>;

using fast_uint_kernel1_t = uint_kernel<uint8_t, true>;
using fast_uint_kernel2_t = uint_kernel<uint16_t, true>;
using fast_uint_kernel4_t = uint_kernel<uint32_t, true>;
using fast_uint_kernel8_t = uint_kernel<uint64_t, true>;
using fast_uint_kernel16_t = uint_kernel<__uint128_t, true>;

}  // namespace nvhm


#if NVHM_WITH_SSE || NVHM_WITH_AVX || NVHM_WITH_AVX512

#if NVHM_WITH_AVX || NVHM_WITH_AVX512 || NVHM_WITH_AVX_FVL || NVHM_WITH_AVX_BWVL || NVHM_WITH_AVX_VBMI
#include <immintrin.h>
#elif NVHM_WITH_SSE >= 4
#include <smmintrin.h>
#elif NVHM_WITH_SSE >= 3
#include <tmmintrin.h>
#elif NVHM_WITH_SSE >= 2
#include <emmintrin.h>
#endif

namespace nvhm {

#if NVHM_WITH_SSE >= 2
// TODO: Splice implementations to make them separately testable.

// empty: 0x80 == msb
// tombstone: 0x81 == msb | lsb
struct sse_kernel final : public kernel<-128, -127, false> {
  using repr_type = __m128i;
  using lru_repr_type = __m128i;

  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wignored-attributes"
  constexpr static int_t size{num_bytes_v<repr_type>};
  static_assert(size == 16, "We rely on this! Even with ignored attributes, size must be 16!");
  #pragma GCC diagnostic pop
  constexpr static bitmask_t size_mask{size_mask_v<size>};

  using mask_type = uint32_mask16_1r_t;
  static_assert(mask_type::max_count == size && mask_type::max_num_bits == size * 2);
  using mask_repr_type = typename mask_type::repr_type;

  constexpr static mask_repr_type full_mask{mask_type::full()};

  NVHM_MAKE_NOT_INSTANTIABLE_(sse_kernel);

  inline static repr_type load(const state_t* ptr) noexcept {
    return _mm_load_si128(reinterpret_cast<const repr_type*>(assume_aligned<size>(ptr)));
  }

  inline static void store(repr_type k, state_t* ptr) noexcept {
    _mm_store_si128(reinterpret_cast<repr_type*>(assume_aligned<size>(ptr)), k);
  }

  inline static mask_repr_type mask(repr_type k, state_t s) noexcept { return mask_equal(k, _mm_set1_epi8(s)); }
  inline static mask_repr_type mask_not(repr_type k, state_t s) noexcept { return mask_not_equal(k, _mm_set1_epi8(s)); }

  inline static mask_repr_type mask_hash(repr_type k) noexcept { return mask_not_hash(k) ^ full_mask; }
  inline static mask_repr_type mask_not_hash(repr_type k) noexcept {
#if NVHM_WITH_AVX_BWVL
    return _mm_movepi8_mask(k);
#else
    return static_cast<mask_repr_type>(_mm_movemask_epi8(k));
#endif
  }

  inline static mask_repr_type mask_empty(repr_type k) noexcept {
#if NVHM_WITH_AVX_BWVL
    // TODO: Less register dependencies. True. But is it faster?
    return mask(k, empty);
#else
#if NVHM_WITH_SSE >= 3
    k = _mm_sign_epi8(k, k);
#else
    k = _mm_cmpeq_epi8(k, _mm_set1_epi8(empty));
#endif
    return mask_not_hash(k);
#endif
  }
  inline static mask_repr_type mask_not_empty(repr_type k) noexcept {
#if NVHM_WITH_AVX_BWVL
    // TODO: Less register dependencies. True. But is it faster?
    return mask_not(k, empty);
#else
#if NVHM_WITH_SSE >= 3
    k = _mm_sign_epi8(k, k);
#else
    k = _mm_cmpeq_epi8(k, _mm_set1_epi8(empty));
#endif
    return mask_hash(k);
#endif
  }
  inline static mask_repr_type mask_tombstone(repr_type k) noexcept { return mask(k, tombstone); }
  inline static mask_repr_type mask_not_tombstone(repr_type k) noexcept { return mask_not(k, tombstone); }

  inline static mask_repr_type mask_equal(repr_type k0, repr_type k1) noexcept {
#if NVHM_WITH_AVX_BWVL
    return _mm_cmpeq_epi8_mask(k0, k1);
#else
    return mask_not_hash(_mm_cmpeq_epi8(k0, k1));
#endif
  }
  inline static mask_repr_type mask_not_equal(repr_type k0, repr_type k1) noexcept {
#if NVHM_WITH_AVX_BWVL
    return _mm_cmpneq_epi8_mask(k0, k1);
#else
    return mask_hash(_mm_cmpeq_epi8(k0, k1));
#endif
  }

  inline static bool has(repr_type k, state_t s) noexcept { return mask(k, s) != 0; }
  inline static bool has_not(repr_type k, state_t s) noexcept { return mask_not(k, s) != 0; }

  inline static bool has_hash(repr_type k) noexcept { return mask_hash(k) != 0; }
  inline static bool has_not_hash(repr_type k) noexcept { return mask_not_hash(k) != 0; }

  inline static bool has_empty(repr_type k) noexcept { return mask_empty(k) != 0; }
  inline static bool has_not_empty(repr_type k) noexcept { return mask_not_empty(k) != 0; }
  inline static bool has_tombstone(repr_type k) noexcept { return mask_tombstone(k) != 0; }
  inline static bool has_not_tombstone(repr_type k) noexcept { return mask_not_tombstone(k) != 0; }

  inline static int_t count_equal(repr_type k0, repr_type k1) noexcept {
    #if defined(__GNUC__) && !defined(__clang__)
    // TODO: GCC compiles this wrong. Comment out and run test_kernels to see what I mean.
    return mask_type::count(mask_not_hash(_mm_cmpeq_epi8(k0, k1)));
    #else
    return mask_type::count(mask_equal(k0, k1));
    #endif
  }
  inline static int_t count_not_equal(repr_type k0, repr_type k1) noexcept {
    return mask_type::count(mask_not_equal(k0, k1));
  }

  inline static repr_type hash_to_tombstone(repr_type k) noexcept {
    repr_type t{_mm_set1_epi8(tombstone)};
#if NVHM_WITH_SSE >= 4
    k = _mm_min_epi8(k, t);
#else
    k = _mm_add_epi8(_mm_cmplt_epi8(k, t), t);
#endif
    return k;
  }

  inline static repr_type not_hash_to_empty(repr_type k) noexcept {
    return _mm_min_epu8(k, _mm_set1_epi8(empty));
  }

  inline static repr_type to_empty(repr_type k, mask_repr_type m) noexcept {
#if NVHM_WITH_AVX_BWVL
    return _mm_mask_set1_epi8(k, static_cast<__mmask16>(m), empty);
#else
    repr_type mx{_mm_set1_epi32(static_cast<int32_t>(m))}; // [01xy01xy 01xy01xy]
    mx = _mm_unpacklo_epi8(mx, mx);  // [0011xxyy 0011xxyy]
    mx = _mm_unpacklo_epi8(mx, mx);  // [00001111 xxxxyyyy]
    mx = _mm_unpacklo_epi8(mx, mx);  // [00000000 11111111]
    mx = _mm_and_si128(mx, _mm_set1_epi64x(static_cast<int64_t>(0x8040'2010'0804'0201)));
    mx = _mm_cmpeq_epi8(mx, _mm_setzero_si128());
    return blendv_epi8(_mm_set1_epi8(empty), k, mx);
  #endif
  }

  inline static state_t at(repr_type k, int_t i) noexcept {
    NVHM_ASSERT_(i < size, "i = ", i, ", size = ", size);
    int64_t s;
#if NVHM_WITH_AVX_VBMI
    k = _mm_permutexvar_epi8(_mm_set1_epi8(static_cast<char>(i)), k);
    s = _mm_cvtsi128_si64(k);
#elif NVHM_WITH_SSE >= 3
    k = _mm_shuffle_epi8(k, _mm_set1_epi8(static_cast<char>(i)));
    s = _mm_cvtsi128_si64(k);
#else
    int64_t k0{_mm_cvtsi128_si64(k)};
    int64_t k8{_mm_cvtsi128_si64(_mm_srli_si128(k, num_bytes_v<int64_t>))};
    s = i < num_bytes_v<int64_t> ? k0 : k8;
    s >>= i % num_bytes_v<int64_t> * num_bits_v<state_t>;
#endif
    return static_cast<state_t>(s);
  }

  inline static lru_repr_type load_lru(const lru_t* ptr) noexcept {
    return _mm_load_si128(reinterpret_cast<const lru_repr_type*>(assume_aligned<size>(ptr)));
  }

  inline static void store_lru(lru_repr_type l, lru_t* ptr) noexcept {
    _mm_store_si128(reinterpret_cast<lru_repr_type*>(assume_aligned<size>(ptr)), l);
  }

  inline static std::pair<lru_t, int_t> min_lru(lru_repr_type l) noexcept {
    lru_repr_type s{min_reduce_epu8(l)};
    lru_t lru{static_cast<lru_t>(_mm_cvtsi128_si32(s))};
#if NVHM_WITH_SSE >= 3
    s = _mm_shuffle_epi8(s, _mm_setzero_si128());
#else
    s = _mm_set1_epi8(static_cast<char>(lru));
#endif
    return {lru, mask_type::next(mask_equal(l, s))};

  }

  inline static mask_repr_type mask_min_lru(lru_repr_type l) noexcept {
    lru_repr_type s{min_reduce_epu8(l)};
#if NVHM_WITH_AVX >= 2
    s = _mm_broadcastb_epi8(s);
#elif NVHM_WITH_SSE >= 3
    s = _mm_shuffle_epi8(s, _mm_setzero_si128());
#else
    s = _mm_set1_epi8(static_cast<char>(_mm_cvtsi128_si32(s)));
#endif
    return mask_equal(l, s);
  }

  inline static mask_repr_type mask_lru_below(lru_repr_type l, lru_t t) noexcept {
    repr_type tx{_mm_set1_epi8(static_cast<char>(t))};
#if NVHM_WITH_AVX_BWVL
    return _mm_cmplt_epu8_mask(l, tx);
#else
    return mask_not_hash(cmplt_epu8(l, tx));
#endif
  }

  inline static lru_repr_type update_lru(lru_repr_type l, mask_repr_type n) noexcept {
    NVHM_ASSERT_(mask_type::has_next(n) && mask_type::next(n) < size, "n = ", render_mask<mask_type>(n));
    n = mask_type::truncate(n);

    // Rescale if l[i] hit maximum value.
    if (NVHM_UNLIKELY_(mask_type::has_next(mask(l, static_cast<state_t>(max_lru)) & n))) {
      l = _mm_and_si128(_mm_srli_epi64(l, 1), _mm_set1_epi8(0x7f));
    }

    // Add 1 to l[i].
#if NVHM_WITH_AVX_BWVL
    l = _mm_mask_add_epi8(l, static_cast<__mmask16>(n), l, _mm_set1_epi8(1));
#else
    int_t i{mask_type::next(n)};
    int64_t inc_lo{i < num_bytes_v<int64_t>};
    int64_t inc_hi{i >= num_bytes_v<int64_t>};
    i = i % num_bytes_v<int64_t> * num_bits_v<lru_t>;
    inc_lo <<= i;
    inc_hi <<= i;

    lru_repr_type inc{_mm_cvtsi64_si128(inc_lo)};
#if NVHM_WITH_SSE >= 4
    inc = _mm_insert_epi64(inc, inc_hi, 1);
#else
    inc = _mm_unpacklo_epi64(inc, _mm_cvtsi64_si128(inc_hi));
#endif
    l = _mm_add_epi8(l, inc);
#endif
    return l;
  }

  inline static repr_type to_empty_if_lru_below(repr_type k, lru_repr_type l, lru_t t) noexcept {
    l = cmplt_epu8(l, _mm_set1_epi8(static_cast<char>(t)));
    k = blendv_epi8(k, _mm_set1_epi8(static_cast<char>(0x80)), l);
    return k;
  }

  inline static lru_t lru_at(lru_repr_type l, int_t i) noexcept {
    return static_cast<lru_t>(at(l, i));
  }

 private:
  inline static __m128i blendv_epi8(__m128i a, __m128i b, __m128i m) noexcept {
#if NVHM_WITH_SSE >= 4
    return _mm_blendv_epi8(a, b, m);
#else
    a = _mm_andnot_si128(m, a);
    b = _mm_and_si128(m, b);
    return _mm_or_si128(a, b);
#endif
  }
  inline static __m128i cmplt_epu8(__m128i a, __m128i b) noexcept {
    const __m128i bias{_mm_set1_epi8(static_cast<char>(0x80))};
    a = _mm_xor_si128(a, bias);
    b = _mm_xor_si128(b, bias);
    // a < b is equivalent to b > a
    return _mm_cmpgt_epi8(b, a);
  }

  inline static __m128i min_reduce_epu8(__m128i l) noexcept {
    __m128i s;
    s = _mm_min_epu8(l, _mm_srli_si128(l, 1));
#if NVHM_WITH_SSE >= 4
    s = _mm_and_si128(s, _mm_set1_epi16(0xff));
    s = _mm_minpos_epu16(s);
#else
    s = _mm_min_epu8(s, _mm_srli_si128(s, 2));
    s = _mm_min_epu8(s, _mm_srli_si128(s, 4));
    s = _mm_min_epu8(s, _mm_srli_si128(s, 8));
#endif
    return s;
  }
};

using sse_kernel_t = sse_kernel;
#endif

#if NVHM_WITH_AVX >= 2
// empty: 0x80 == msb
// tombstone: 0x81 == msb | lsb
struct avx_kernel final : public kernel<-128, -127, false> {
  using repr_type = __m256i;
  using lru_repr_type = __m256i;

  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wignored-attributes"
  constexpr static int_t size{num_bytes_v<repr_type>};
  static_assert(size == 32, "We rely on this! Even with ignored attributes, size must be 32!");
  #pragma GCC diagnostic pop
  constexpr static bitmask_t size_mask{size_mask_v<size>};

  using mask_type = uint32_mask32_1r_t;
  static_assert(mask_type::max_count == size && mask_type::fully_utilized);
  using mask_repr_type = typename mask_type::repr_type;

  NVHM_MAKE_NOT_INSTANTIABLE_(avx_kernel);

  inline static repr_type load(const state_t* ptr) noexcept {
    return _mm256_load_si256(reinterpret_cast<const repr_type*>(assume_aligned<size>(ptr)));
  }

  inline static void store(const repr_type k, state_t* ptr) noexcept {
    _mm256_store_si256(reinterpret_cast<repr_type*>(assume_aligned<size>(ptr)), k);
  }

  inline static mask_repr_type mask(repr_type k, state_t s) noexcept {
    return mask_equal(k, _mm256_set1_epi8(s));
  }
  inline static mask_repr_type mask_not(repr_type k, state_t s) noexcept {
    return mask_not_equal(k, _mm256_set1_epi8(s));
  }

  inline static mask_repr_type mask_hash(repr_type k) noexcept { return ~mask_not_hash(k); }
  inline static mask_repr_type mask_not_hash(repr_type k) noexcept {
#if NVHM_WITH_AVX_BWVL
    return _mm256_movepi8_mask(k);
#else
    return static_cast<mask_repr_type>(_mm256_movemask_epi8(k));
#endif
  }

  inline static mask_repr_type mask_empty(repr_type k) noexcept { return mask_not_hash(_mm256_sign_epi8(k, k)); }
  inline static mask_repr_type mask_not_empty(repr_type k) noexcept { return mask_hash(_mm256_sign_epi8(k, k)); }
  inline static mask_repr_type mask_tombstone(repr_type k) noexcept { return mask(k, tombstone); }
  inline static mask_repr_type mask_not_tombstone(repr_type k) noexcept { return mask_not(k, tombstone); }

  inline static mask_repr_type mask_equal(repr_type k0, repr_type k1) noexcept {
#if NVHM_WITH_AVX_BWVL
    return _mm256_cmpeq_epi8_mask(k0, k1);
#else
    return mask_not_hash(_mm256_cmpeq_epi8(k0, k1));
#endif
  }
  inline static mask_repr_type mask_not_equal(repr_type k0, repr_type k1) noexcept {
#if NVHM_WITH_AVX_BWVL
    return _mm256_cmpneq_epi8_mask(k0, k1);
#else
    return mask_hash(_mm256_cmpeq_epi8(k0, k1));
#endif
  }

  inline static bool has(repr_type k, state_t s) noexcept { return mask(k, s) != 0; }
  inline static bool has_not(repr_type k, state_t s) noexcept { return mask_not(k, s) != 0; }

  inline static bool has_hash(repr_type k) noexcept { return mask_hash(k) != 0; }
  inline static bool has_not_hash(repr_type k) noexcept { return mask_not_hash(k) != 0; }

  inline static bool has_empty(repr_type k) noexcept { return mask_empty(k) != 0; }
  inline static bool has_not_empty(repr_type k) noexcept { return mask_not_empty(k) != 0; }
  inline static bool has_tombstone(repr_type k) noexcept { return mask_tombstone(k) != 0; }
  inline static bool has_not_tombstone(repr_type k) noexcept { return mask_not_tombstone(k) != 0; }

  inline static int_t count_equal(repr_type k0, repr_type k1) noexcept {
    return mask_type::count(mask_equal(k0, k1));
  }
  inline static int_t count_not_equal(repr_type k0, repr_type k1) noexcept {
    return mask_type::count(mask_not_equal(k0, k1));
  }

  inline static repr_type hash_to_tombstone(repr_type k) noexcept {
    return _mm256_min_epi8(k, _mm256_set1_epi8(tombstone));
  }

  inline static repr_type not_hash_to_empty(repr_type k) noexcept {
    return _mm256_min_epu8(k, _mm256_set1_epi8(empty));
  }

  inline static repr_type to_empty(repr_type k, mask_repr_type m) noexcept {
#if NVHM_WITH_AVX_BWVL
    return _mm256_mask_set1_epi8(k, m, empty);
#else
    repr_type mx{_mm256_set1_epi32(static_cast<int32_t>(m))}; // [01230123 01230123 01230123 01230123]
    mx = _mm256_unpacklo_epi8(mx, mx);  // [00112233 00112233 00112233 00112233]
    mx = _mm256_unpacklo_epi8(mx, mx);  // [00001111 22223333 00001111 22223333]
    mx = _mm256_permute4x64_epi64(mx, _MM_SHUFFLE(1, 1, 0, 0));  // [00001111 00001111 22223333 22223333]
    mx = _mm256_unpacklo_epi8(mx, mx);  // [00000000 11111111 22222222 33333333]
    mx = _mm256_and_si256(mx, _mm256_set1_epi64x(static_cast<int64_t>(0x8040'2010'0804'0201)));
    mx = _mm256_cmpeq_epi8(mx, _mm256_setzero_si256());
    return _mm256_blendv_epi8(_mm256_set1_epi8(empty), k, mx);
#endif
  }

  inline static state_t at(repr_type k, int_t i) noexcept {
    NVHM_ASSERT_(i < size, "i = ", i, ", size = ", size);
    int32_t s;
#if NVHM_WITH_AVX_VBMI
    k = _mm256_permutexvar_epi8(_mm256_set1_epi8(static_cast<char>(i)), k);
    s = _mm256_cvtsi256_si32(k);
#else
    k = _mm256_permutevar8x32_epi32(k, _mm256_set1_epi32(static_cast<int32_t>(i / num_bytes_v<int32_t>)));
    s = _mm256_cvtsi256_si32(k);
    s >>= i % num_bytes_v<int32_t> * num_bits_v<state_t>;
#endif
    return static_cast<state_t>(s);
  }

  inline static lru_repr_type load_lru(const lru_t* ptr) noexcept {
    return _mm256_load_si256(reinterpret_cast<const lru_repr_type*>(assume_aligned<size>(ptr)));
  }

  inline static void store_lru(lru_repr_type l, lru_t* ptr) noexcept {
    _mm256_store_si256(reinterpret_cast<lru_repr_type*>(assume_aligned<size>(ptr)), l);
  }

  inline static std::pair<lru_t, int_t> min_lru(lru_repr_type l) noexcept {
    __m128i s{_mm_min_epu8(_mm256_castsi256_si128(l), _mm256_extracti128_si256(l, 1))};
    s = _mm_min_epu8(s, _mm_srli_si128(s, 1));
    s = _mm_and_si128(s, _mm_set1_epi16(0xff));
    s = _mm_minpos_epu16(s);
    lru_t lru{static_cast<lru_t>(_mm_cvtsi128_si32(s))};
    return {lru, mask_type::next(mask_equal(l, _mm256_broadcastb_epi8(s)))};
  }

  inline static mask_repr_type mask_min_lru(lru_repr_type l) noexcept {
    __m128i s{min_reduce_epu8(l)};
    return mask_equal(l, _mm256_broadcastb_epi8(s));
  }

  inline static mask_repr_type mask_lru_below(lru_repr_type l, lru_t t) noexcept {
    repr_type tx{_mm256_set1_epi8(static_cast<char>(t))};
    mask_repr_type m;
#if NVHM_WITH_AVX_BWVL
    m = _mm256_cmplt_epu8_mask(l, tx);
#else
    l = cmplt_epu8(l, tx);
    m = static_cast<mask_repr_type>(_mm256_movemask_epi8(l));
#endif
    return m;
  }

  inline static lru_repr_type update_lru(lru_repr_type l, mask_repr_type n) noexcept {
    NVHM_ASSERT_(mask_type::has_next(n) && mask_type::next(n) < size, "n = ", render_mask<mask_type>(n));
    n = mask_type::truncate(n);

    // Rescale if l[i] hit maximum value.
    if (NVHM_UNLIKELY_(mask_type::has_next(mask(l, static_cast<state_t>(max_lru)) & n))) {
      l = _mm256_and_si256(_mm256_srli_epi32(l, 1), _mm256_set1_epi8(0x7f));
    }

    // Add 1 to l[i].
#if NVHM_WITH_AVX_BWVL
    l = _mm256_mask_add_epi8(l, n, l, _mm256_set1_epi8(1));
#else
    int_t i{mask_type::next(n)};
    __m128i maj{_mm_cvtsi64_si128(INT64_C(0xff) << (i / num_bytes_v<int64_t> * num_bits_v<lru_t>))};
    __m256i min{_mm256_set1_epi64x(INT64_C(0x01) << (i % num_bytes_v<int64_t> * num_bits_v<lru_t>))};
    l = _mm256_add_epi8(l, _mm256_and_si256(_mm256_cvtepi8_epi64(maj), min));
#endif
    return l;
  }

  inline static repr_type to_empty_if_lru_below(repr_type k, lru_repr_type l, lru_t t) noexcept {
    l = cmplt_epu8(l, _mm256_set1_epi8(static_cast<char>(t)));
    k = _mm256_blendv_epi8(k, _mm256_set1_epi8(static_cast<char>(0x80)), l);
    return k;
  }

  inline static lru_t lru_at(lru_repr_type l, int_t i) noexcept {
    return static_cast<lru_t>(at(l, i));
  }

 private:
  inline static __m256i cmplt_epu8(__m256i a, __m256i b) noexcept {
    __m256i bias{_mm256_set1_epi8(static_cast<char>(0x80))};
    a = _mm256_xor_si256(a, bias);
    b = _mm256_xor_si256(b, bias);
    // a < b is equivalent to b > a
    return _mm256_cmpgt_epi8(b, a);
  }

  inline static __m128i min_reduce_epu8(lru_repr_type l) noexcept {
    __m128i s{_mm_min_epu8(_mm256_castsi256_si128(l), _mm256_extracti128_si256(l, 1))};
    s = _mm_min_epu8(s, _mm_srli_si128(s, 1));
    s = _mm_and_si128(s, _mm_set1_epi16(0xff));
    s = _mm_minpos_epu16(s);
    return s;
  }
};

using avx_kernel_t = avx_kernel;
#endif

#if NVHM_WITH_AVX512
// empty: 0x80 == msb
// tombstone: 0x81 == msb | lsb
struct avx512_kernel final : public kernel<-128, -127, false> {
  using repr_type = __m512i;
  using lru_repr_type = __m512i;

  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wignored-attributes"
  constexpr static int_t size{num_bytes_v<repr_type>};
  static_assert(size == 64, "We rely on this! Even with ignored attributes, size must be 64!");
  #pragma GCC diagnostic pop
  constexpr static bitmask_t size_mask{size_mask_v<size>};

  using mask_type = uint64_mask64_1r_t;
  static_assert(mask_type::max_count == size && mask_type::fully_utilized);
  using mask_repr_type = typename mask_type::repr_type;

  NVHM_MAKE_NOT_INSTANTIABLE_(avx512_kernel);

  inline static repr_type load(const state_t* ptr) noexcept {
    return _mm512_load_si512(assume_aligned<size>(ptr));
  }

  inline static void store(repr_type k, state_t* ptr) noexcept {
    _mm512_store_si512(assume_aligned<size>(ptr), k);
  }

  inline static mask_repr_type mask(repr_type k, const state_t s) noexcept {
    return mask_equal(k, _mm512_set1_epi8(s));
  }
  inline static mask_repr_type mask_not(repr_type k, state_t s) noexcept {
    return mask_not_equal(k, _mm512_set1_epi8(s));
  }

  inline static mask_repr_type mask_hash(repr_type k) noexcept { return ~mask_not_hash(k); }
  inline static mask_repr_type mask_not_hash(repr_type k) noexcept { return _mm512_movepi8_mask(k); }

  inline static mask_repr_type mask_empty(repr_type k) noexcept { return mask(k, empty); }
  inline static mask_repr_type mask_not_empty(repr_type k) noexcept { return mask_not(k, empty); }
  inline static mask_repr_type mask_tombstone(repr_type k) noexcept { return mask(k, tombstone); }
  inline static mask_repr_type mask_not_tombstone(repr_type k) noexcept { return mask_not(k, tombstone); }

  inline static mask_repr_type mask_equal(repr_type k0, repr_type k1) noexcept {
    return _mm512_cmpeq_epi8_mask(k0, k1);
  }
  inline static mask_repr_type mask_not_equal(repr_type k0, repr_type k1) noexcept {
    return _mm512_cmpneq_epi8_mask(k0, k1);
  }

  inline static bool has(repr_type k, state_t s) noexcept { return mask(k, s) != 0; }
  inline static bool has_not(repr_type k, state_t s) noexcept { return mask_not(k, s) != 0; }

  inline static bool has_hash(repr_type k) noexcept { return mask_hash(k) != 0; }
  inline static bool has_not_hash(repr_type k) noexcept { return mask_not_hash(k) != 0; }

  inline static bool has_empty(repr_type k) noexcept { return mask_empty(k) != 0; }
  inline static bool has_not_empty(repr_type k) noexcept { return mask_not_empty(k) != 0; }
  inline static bool has_tombstone(repr_type k) noexcept { return mask_tombstone(k) != 0; }
  inline static bool has_not_tombstone(repr_type k) noexcept { return mask_not_tombstone(k) != 0; }

  inline static int_t count_equal(repr_type k0, repr_type k1) noexcept {
    return mask_type::count(mask_equal(k0, k1));
  }
  inline static int_t count_not_equal(repr_type k0, repr_type k1) noexcept {
    return mask_type::count(mask_not_equal(k0, k1));
  }

  inline static repr_type hash_to_tombstone(repr_type k) noexcept {
    return _mm512_min_epi8(k, _mm512_set1_epi8(tombstone));
  }

  inline static repr_type not_hash_to_empty(repr_type k) noexcept {
    return _mm512_min_epu8(k, _mm512_set1_epi8(empty));
  }

  inline static repr_type to_empty(repr_type k, mask_repr_type m) noexcept {
    return _mm512_mask_set1_epi8(k, m, empty);
  }

  inline static state_t at(repr_type k, int_t i) noexcept {
    NVHM_ASSERT_(i < size, "i = ", i, ", size = ", size);
    int32_t s;
#if NVHM_WITH_AVX_VBMI
    k = _mm512_permutexvar_epi8(_mm512_set1_epi8(static_cast<char>(i)), k);
    s = _mm512_cvtsi512_si32(k);
#else
    k = _mm512_permutexvar_epi32(_mm512_set1_epi32(static_cast<int32_t>(i / num_bytes_v<int32_t>)), k);
    s = _mm512_cvtsi512_si32(k);
    s >>= i % num_bytes_v<int32_t> * num_bits_v<state_t>;
#endif
    return static_cast<state_t>(s);
  }

  inline static lru_repr_type load_lru(const lru_t* ptr) noexcept {
    return _mm512_load_si512(assume_aligned<size>(ptr));
  }

  inline static void store_lru(lru_repr_type l, lru_t* ptr) noexcept {
    _mm512_store_si512(assume_aligned<size>(ptr), l);
  }

  inline static std::pair<lru_t, int_t> min_lru(lru_repr_type l) noexcept {
    __m128i s{min_reduce_epu8(l)};
    lru_t lru{static_cast<lru_t>(_mm_cvtsi128_si32(s))};
    return {lru, mask_type::next(mask_equal(l, _mm512_broadcastb_epi8(s)))};
  }

  inline static mask_repr_type mask_min_lru(lru_repr_type l) noexcept {
    __m128i s{min_reduce_epu8(l)};
    return mask_equal(l, _mm512_broadcastb_epi8(s));
  }

  inline static mask_repr_type mask_lru_below(lru_repr_type l, lru_t t) noexcept {
    return _mm512_cmplt_epu8_mask(l, _mm512_set1_epi8(static_cast<char>(t)));
  }

  inline static lru_repr_type update_lru(lru_repr_type l, mask_repr_type n) noexcept {
    NVHM_ASSERT_(mask_type::has_next(n) && mask_type::next(n) < size, "n = ", render_mask<mask_type>(n));
    n = mask_type::truncate(n);

    // Rescale if l[i] hits maximum value.
    if (NVHM_UNLIKELY_(mask_type::has_next(mask(l, static_cast<state_t>(max_lru)) & n))) {
      l = _mm512_and_si512(_mm512_srli_epi32(l, 1), _mm512_set1_epi8(0x7f));
    }

    // Add 1 to l[i].
    l = _mm512_mask_add_epi8(l, n, l, _mm512_set1_epi8(1));
    return l;
  }

  inline static repr_type to_empty_if_lru_below(repr_type k, lru_repr_type l, lru_t t) noexcept {
    return to_empty(k, mask_lru_below(l, t));
  }

  inline static lru_t lru_at(lru_repr_type l, int_t i) noexcept {
    return static_cast<lru_t>(at(l, i));
  }

 protected:
  inline static __m128i min_reduce_epu8(lru_repr_type l) noexcept {
    __m256i ss{_mm256_min_epu8(_mm512_castsi512_si256(l), _mm512_extracti64x4_epi64(l, 1))};
    __m128i s{_mm_min_epu8(_mm256_castsi256_si128(ss), _mm256_extracti128_si256(ss, 1))};
    s = _mm_min_epu8(s, _mm_srli_si128(s, 1));
    s = _mm_and_si128(s, _mm_set1_epi16(0xff));
    s = _mm_minpos_epu16(s);
    return s;
  }
};

using avx512_kernel_t = avx512_kernel;
#endif
}
#endif


#if NVHM_WITH_NEON
#include <arm_neon.h>

namespace nvhm {

// empty: 0x80 == msb
// tombstone: 0x81 == msb | lsb
template <int_t Size>
struct neon_kernel : public kernel<-128, -127, false> {
  NVHM_MAKE_NOT_INSTANTIABLE_(neon_kernel);
};

template <>
struct neon_kernel<8> final : public kernel<-128, -127, false> {
  using repr_type = int8x8_t;
  using lru_repr_type = uint8x8_t;

  constexpr static int_t size{num_bytes_v<repr_type>};
  constexpr static bitmask_t size_mask{size_mask_v<size>};

  using mask_type = uint64_mask64_8l_t;
  static_assert(mask_type::max_count == size && mask_type::fully_utilized);

  using mask_repr_type = typename mask_type::repr_type;
  static_assert(num_bytes_v<mask_repr_type> == size);

  constexpr static mask_repr_type full_mask{mask_type::full()};

  NVHM_MAKE_NOT_INSTANTIABLE_(neon_kernel);

  inline static repr_type load(const state_t* ptr) noexcept {
    return vld1_s8(assume_aligned<size>(ptr));
  }

  inline static void store(repr_type k, state_t* ptr) noexcept {
    vst1_s8(assume_aligned<size>(ptr), k);
  }

  inline static mask_repr_type mask(repr_type k, state_t s) noexcept {
    return mask_equal(k, vdup_n_s8(s));
  }
  inline static mask_repr_type mask_not(repr_type k, state_t s) noexcept {
    return mask_not_equal(k, vdup_n_s8(s));
  }

  inline static mask_repr_type mask_hash(repr_type k) noexcept {
    return mask_not_hash(k) ^ full_mask;
    //return full_mask & collapse_(vcgez_s8(k));
  }
  inline static mask_repr_type mask_not_hash(repr_type k) noexcept {
    return full_mask & collapse_(vreinterpret_u8_s8(k));
    //return full_mask & collapse_(vcltz_s8(k));
  }

  inline static mask_repr_type mask_empty(repr_type k) noexcept { return mask(k, empty); }
  inline static mask_repr_type mask_not_empty(repr_type k) noexcept { return mask_not(k, empty); }
  inline static mask_repr_type mask_tombstone(repr_type k) noexcept { return mask(k, tombstone); }
  inline static mask_repr_type mask_not_tombstone(repr_type k) noexcept { return mask_not(k, tombstone); }

  inline static mask_repr_type mask_equal(repr_type k0, repr_type k1) noexcept {
    return full_mask & collapse_(vceq_s8(k0, k1));
  }
  inline static mask_repr_type mask_not_equal(repr_type k0, repr_type k1) noexcept {
    return mask_equal(k0, k1) ^ full_mask;
  }

  inline static state_t min(repr_type k) noexcept { return vminv_s8(k); }
  inline static state_t max(repr_type k) noexcept { return vmaxv_s8(k); }

  inline static bool has_hash(repr_type k) noexcept { return is_hash(max(k)); }
  inline static bool has_not_hash(repr_type k) noexcept { return is_not_hash(min(k)); }

  inline static bool has(repr_type k, state_t s) noexcept {
    return vmaxv_u8(vceq_s8(k, vdup_n_s8(s))) != 0;
  }
  inline static bool has_not(repr_type k, state_t s) noexcept {
    return vminv_u8(vceq_s8(k, vdup_n_s8(s))) == 0;
  }

  inline static bool has_empty(repr_type k) noexcept { return min(k) == empty; }
  inline static bool has_not_empty(repr_type k) noexcept { return max(k) != empty; }
  inline static bool has_tombstone(repr_type k) noexcept { return has(k, tombstone); }
  inline static bool has_not_tombstone(repr_type k) noexcept { return has_not(k, tombstone); }

  inline static int_t count_equal(repr_type k0, repr_type k1) noexcept {
    return -vaddv_s8(vreinterpret_s8_u8(vceq_s8(k0, k1)));
  }
  inline static int_t count_not_equal(repr_type k0, repr_type k1) noexcept {
    return size + vaddv_s8(vreinterpret_s8_u8(vceq_s8(k0, k1)));
  }

  inline static repr_type hash_to_tombstone(repr_type k) noexcept {
    return vmin_s8(k, vdup_n_s8(tombstone));
  }

  inline static repr_type not_hash_to_empty(repr_type k) noexcept {
    uint8x8_t e{vreinterpret_u8_s8(vdup_n_s8(empty))};
    return vreinterpret_s8_u8(vmin_u8(vreinterpret_u8_s8(k), e));
  }

  inline static repr_type to_empty(repr_type k, mask_repr_type m) noexcept {
    uint8x8_t mx{expand_(m)};
    return vbsl_s8(mx, vdup_n_s8(empty), k);
  }

  inline static state_t at(repr_type k, int_t i) noexcept {
    NVHM_ASSERT_(i < size, "i = ", i, ", size = ", size);
    k = vtbl1_s8(k, vdup_n_s8(static_cast<int8_t>(i)));
    return vget_lane_s8(k, 0);
  }

  inline static lru_repr_type load_lru(const lru_t* ptr) noexcept {
    return vld1_u8(assume_aligned<size>(ptr));
  }

  inline static void store_lru(lru_repr_type l, lru_t* ptr) noexcept {
    vst1_u8(assume_aligned<size>(ptr), l);
  }

  inline static std::pair<lru_t, int_t> min_lru(lru_repr_type l) noexcept {
    lru_t lru{vminv_u8(l)};
    int_t idx{mask_type::next(mask_lru(l, lru))};
    return {lru, idx};
  }

  inline static mask_repr_type mask_lru(lru_repr_type l, lru_t lru) noexcept {
    return full_mask & collapse_(vceq_u8(l, vdup_n_u8(lru)));
  }

  inline static mask_repr_type mask_min_lru(lru_repr_type l) noexcept {
    return full_mask & collapse_(vceq_u8(l, vdup_n_u8(vminv_u8(l))));
  }

  inline static mask_repr_type mask_lru_below(lru_repr_type l, lru_t t) noexcept {
    return full_mask & collapse_(vclt_u8(l, vdup_n_u8(t)));
  }

  inline static lru_repr_type update_lru(lru_repr_type l, mask_repr_type n) noexcept {
    NVHM_ASSERT_(mask_type::has_next(n) && mask_type::next(n) < size, "n = ", render_mask<mask_type>(n));
    n = mask_type::truncate(n);
    uint8x8_t nx{expand_(n)};

    // Mask slots if they need rescaling.
    uint64x1_t r{vreinterpret_u64_u8(vceq_u8(l, vdup_n_u8(max_lru)))};
    r = vtst_u64(r, vreinterpret_u64_u8(nx));
    
    // Rescale and increment.
    l = vshl_u8(l, vreinterpret_s8_u64(r));
    l = vsub_u8(l, nx);
    return l;
  }

  inline static repr_type to_empty_if_lru_below(repr_type k, lru_repr_type l, lru_t t) noexcept {
    l = vclt_u8(l, vdup_n_u8(t));
    return vbsl_s8(l, vdup_n_s8(empty), k);
  }

  inline static lru_t lru_at(lru_repr_type l, int_t i) noexcept {
    NVHM_ASSERT_(i < size, "i = ", i, ", size = ", size);
    l = vtbl1_u8(l, vdup_n_u8(static_cast<uint8_t>(i)));
    return vget_lane_u8(l, 0);
  }

 private:
  inline static mask_repr_type collapse_(uint8x8_t m) noexcept {
    return vget_lane_u64(vreinterpret_u64_u8(m), 0);
  }

  inline static uint8x8_t expand_(mask_repr_type m) noexcept {
    uint8x8_t mx{vcreate_u8(m)};
    return vtst_u8(mx, mx);
  }
};

template <>
struct neon_kernel<16> final : public kernel<-128, -127, false> {
  using repr_type = int8x16_t;
  using lru_repr_type = uint8x16_t;

  constexpr static int_t size{num_bytes_v<repr_type>};
  constexpr static bitmask_t size_mask{size_mask_v<size>};

  using mask_type = uint64_mask64_4r_t;
  static_assert(mask_type::max_count == size && mask_type::fully_utilized);
  
  using mask_repr_type = typename mask_type::repr_type;
  static_assert(num_bytes_v<mask_repr_type> * 2 == size);

  constexpr static mask_repr_type full_mask{mask_type::full()};

  NVHM_MAKE_NOT_INSTANTIABLE_(neon_kernel);

  inline static repr_type load(const state_t* ptr) noexcept {
    return vld1q_s8(assume_aligned<size>(ptr));
  }

  inline static void store(repr_type k, state_t* ptr) noexcept {
    vst1q_s8(assume_aligned<size>(ptr), k);
  }

  inline static mask_repr_type mask(repr_type k, state_t s) noexcept {
    return mask_equal(k, vdupq_n_s8(s));
  }
  inline static mask_repr_type mask_not(repr_type k, state_t s) noexcept {
    return mask_not_equal(k, vdupq_n_s8(s));
  }

  inline static mask_repr_type mask_hash(repr_type k) noexcept {
    return full_mask & collapse_(vcgezq_s8(k));
  }
  inline static mask_repr_type mask_not_hash(repr_type k) noexcept {
    return full_mask & collapse_(vcltzq_s8(k));
  }

  inline static mask_repr_type mask_empty(repr_type k) noexcept { return mask(k, empty); }
  inline static mask_repr_type mask_not_empty(repr_type k) noexcept { return mask_not(k, empty); }
  inline static mask_repr_type mask_tombstone(repr_type k) noexcept { return mask(k, tombstone); }
  inline static mask_repr_type mask_not_tombstone(repr_type k) noexcept { return mask_not(k, tombstone); }

  inline static mask_repr_type mask_equal(repr_type k0, repr_type k1) noexcept {
    return full_mask & collapse_(vceqq_s8(k0, k1));
  }
  inline static mask_repr_type mask_not_equal(repr_type k0, repr_type k1) noexcept {
    return mask_equal(k0, k1) ^ full_mask;
  }

  inline static state_t min(repr_type k) noexcept { return vminvq_s8(k); }
  inline static state_t max(repr_type k) noexcept { return vmaxvq_s8(k); }

  inline static bool has_hash(repr_type k) noexcept { return is_hash(max(k)); }
  inline static bool has_not_hash(repr_type k) noexcept { return is_not_hash(min(k)); }

  inline static bool has(repr_type k, state_t s) noexcept {
    return vmaxvq_u8(vceqq_s8(k, vdupq_n_s8(s))) != 0;
  }
  inline static bool has_not(repr_type k, state_t s) noexcept {
    return vminvq_u8(vceqq_s8(k, vdupq_n_s8(s))) == 0;
  }

  inline static bool has_empty(repr_type k) noexcept { return min(k) == empty; }
  inline static bool has_not_empty(repr_type k) noexcept { return max(k) != empty; }
  inline static bool has_tombstone(repr_type k) noexcept { return has(k, tombstone); }
  inline static bool has_not_tombstone(repr_type k) noexcept { return has_not(k, tombstone); }

  inline static int_t count_equal(repr_type k0, repr_type k1) noexcept {
    return -vaddvq_s8(vreinterpretq_s8_u8(vceqq_s8(k0, k1)));
  }
  inline static int_t count_not_equal(repr_type k0, repr_type k1) noexcept {
    return size + vaddvq_s8(vreinterpretq_s8_u8(vceqq_s8(k0, k1)));
  }

  inline static repr_type hash_to_tombstone(repr_type k) noexcept {
    return vminq_s8(k, vdupq_n_s8(tombstone));
  }

  inline static repr_type not_hash_to_empty(repr_type k) noexcept {
    uint8x16_t e{vreinterpretq_u8_s8(vdupq_n_s8(empty))};
    return vreinterpretq_s8_u8(vminq_u8(vreinterpretq_u8_s8(k), e));
  }

  inline static repr_type to_empty(repr_type k, mask_repr_type m) noexcept {
    uint8x16_t mx{expand_(m)};
    return vbslq_s8(mx, vdupq_n_s8(empty), k);
  }

  inline static state_t at(repr_type k, int_t i) noexcept {
    NVHM_ASSERT_(i < size, "i = ", i, ", size = ", size);
    k = vqtbl1q_s8(k, vdupq_n_u8(static_cast<uint8_t>(i)));
    return vgetq_lane_s8(k, 0);
  }

  inline static lru_repr_type load_lru(const lru_t* ptr) noexcept {
    return vld1q_u8(assume_aligned<size>(ptr));
  }

  inline static void store_lru(lru_repr_type l, lru_t* ptr) noexcept {
    vst1q_u8(assume_aligned<size>(ptr), l);
  }

  inline static std::pair<lru_t, int_t> min_lru(lru_repr_type l) noexcept {
    lru_t lru{vminvq_u8(l)};
    int_t idx{mask_type::next(mask_lru(l, lru))};
    return {lru, idx};
  }

  inline static mask_repr_type mask_lru(lru_repr_type l, lru_t lru) noexcept {
    return full_mask & collapse_(vceqq_u8(l, vdupq_n_u8(lru)));
  }

  inline static mask_repr_type mask_min_lru(lru_repr_type l) noexcept {
    return full_mask & collapse_(vceqq_u8(l, vdupq_n_u8(vminvq_u8(l))));
  }

  inline static mask_repr_type mask_lru_below(lru_repr_type l, lru_t t) noexcept {
    return full_mask & collapse_(vcltq_u8(l, vdupq_n_u8(t)));
  }

  inline static lru_repr_type update_lru(lru_repr_type l, mask_repr_type n) noexcept {
    NVHM_ASSERT_(mask_type::has_next(n) && mask_type::next(n) < size, "n = ", render_mask<mask_type>(n));
    n = mask_type::truncate(n);
    uint8x16_t nx{expand_(n)};

    // Mask slots if they need rescaling.
    uint32x4_t r{vreinterpretq_u32_u8(vceqq_u8(l, vdupq_n_u8(max_lru)))};
    r = vtstq_u32(r, vreinterpretq_u32_u8(nx));
    r = vdupq_n_u32(vmaxvq_u32(r));

    // Rescale and increment.
    l = vshlq_u8(l, vreinterpretq_s8_u32(r));
    l = vsubq_u8(l, nx);
    return l;
  }

  inline static repr_type to_empty_if_lru_below(repr_type k, lru_repr_type l, lru_t t) noexcept {
    l = vcltq_u8(l, vdupq_n_u8(t));
    return vbslq_s8(l, vdupq_n_s8(empty), k);
  }

  inline static lru_t lru_at(lru_repr_type l, int_t i) noexcept {
    NVHM_ASSERT_(i < size, "i = ", i, ", size = ", size);
    l = vqtbl1q_u8(l, vdupq_n_u8(static_cast<uint8_t>(i)));
    return vgetq_lane_u8(l, 0);
  }

 private:
  /**
   * Truncate NEON 8-bit mask (0x00 or 0xff) into a 4-bit mask (0x0 or 0xf).
   * See also:
   * https://community.arm.com/arm-community-blogs/b/infrastructure-solutions-blog/posts/porting-x86-vector-bitmask-optimizations-to-arm-neon
   */
  inline static mask_repr_type collapse_(uint8x16_t m8) noexcept {
    uint8x8_t m4{vshrn_n_u16(vreinterpretq_u16_u8(m8), 4)};
    return vget_lane_u64(vreinterpret_u64_u8(m4), 0);
  }

  inline static uint8x16_t expand_(mask_repr_type m) noexcept {
    uint8x16_t mx{vreinterpretq_u8_u16(vshll_n_u8(vcreate_u8(m), 4))};
    return vtstq_u8(mx, mx);
  }
};

template <>
struct neon_kernel<32> final : public kernel<-128, -127, false> {
  using repr_type = int8x16x2_t;
  using lru_repr_type = uint8x16x2_t;

  constexpr static int_t size{num_bytes_v<repr_type>};
  constexpr static bitmask_t size_mask{size_mask_v<size>};

  using mask_type = uint64_mask64_2r_t;
  static_assert(mask_type::max_count == size && mask_type::fully_utilized);
  
  using mask_repr_type = typename mask_type::repr_type;
  static_assert(num_bytes_v<mask_repr_type> * 4 == size);

  constexpr static mask_repr_type full_mask{mask_type::full()};

  NVHM_MAKE_NOT_INSTANTIABLE_(neon_kernel);

  inline static repr_type load(const state_t* ptr) noexcept {
    return vld1q_s8_x2(assume_aligned<size>(ptr));
  }

  inline static void store(repr_type k, state_t* ptr) noexcept {
    vst1q_s8_x2(assume_aligned<size>(ptr), k);
  }

  inline static mask_repr_type mask(repr_type k, state_t s) noexcept {
    int8x16_t sx{vdupq_n_s8(s)};
    uint8x16_t m0{vceqq_s8(k.val[0], sx)};
    uint8x16_t m1{vceqq_s8(k.val[1], sx)};
    return collapse_(m0, m1);
  }
  inline static mask_repr_type mask_not(repr_type k, state_t s) noexcept {
    return mask(k, s) ^ full_mask;
  }

  inline static mask_repr_type mask_hash(repr_type k) noexcept {
    uint8x16_t m0{vcgezq_s8(k.val[0])};
    uint8x16_t m1{vcgezq_s8(k.val[1])};
    return collapse_(m0, m1);
  }
  inline static mask_repr_type mask_not_hash(repr_type k) noexcept {
    uint8x16_t m0{vcltzq_s8(k.val[0])};
    uint8x16_t m1{vcltzq_s8(k.val[1])};
    return collapse_(m0, m1);
  }

  inline static mask_repr_type mask_empty(repr_type k) noexcept { return mask(k, empty); }
  inline static mask_repr_type mask_not_empty(repr_type k) noexcept { return mask_not(k, empty); }
  inline static mask_repr_type mask_tombstone(repr_type k) noexcept { return mask(k, tombstone); }
  inline static mask_repr_type mask_not_tombstone(repr_type k) noexcept { return mask_not(k, tombstone); }

  inline static mask_repr_type mask_equal(repr_type k0, repr_type k1) noexcept {
    uint8x16_t m0{vceqq_s8(k0.val[0], k1.val[0])};
    uint8x16_t m1{vceqq_s8(k0.val[1], k1.val[1])};
    return collapse_(m0, m1);
  }
  inline static mask_repr_type mask_not_equal(repr_type k0, repr_type k1) noexcept {
    return mask_equal(k0, k1) ^ full_mask;
  }

  inline static state_t min(repr_type k) noexcept { return vminvq_s8(vminq_s8(k.val[0], k.val[1])); }
  inline static state_t max(repr_type k) noexcept { return vmaxvq_s8(vmaxq_s8(k.val[0], k.val[1])); }

  inline static bool has_hash(repr_type k) noexcept { return is_hash(max(k)); }
  inline static bool has_not_hash(repr_type k) noexcept { return is_not_hash(min(k)); }

  inline static bool has(repr_type k, state_t s) noexcept {
    int8x16_t t{vdupq_n_s8(s)};
    uint8x16_t m0{vceqq_s8(k.val[0], t)};
    uint8x16_t m1{vceqq_s8(k.val[1], t)};
    return vmaxvq_u8(vmaxq_u8(m0, m1)) != 0;
  }
  inline static bool has_not(repr_type k, state_t s) noexcept {
    int8x16_t t{vdupq_n_s8(s)};
    uint8x16_t m0{vceqq_s8(k.val[0], t)};
    uint8x16_t m1{vceqq_s8(k.val[1], t)};
    return vminvq_u8(vminq_u8(m0, m1)) == 0;
  }

  inline static bool has_empty(repr_type k) noexcept { return min(k) == empty; }
  inline static bool has_not_empty(repr_type k) noexcept { return max(k) != empty; }
  inline static bool has_tombstone(repr_type k) noexcept { return has(k, tombstone); }
  inline static bool has_not_tombstone(repr_type k) noexcept { return has_not(k, tombstone); }

  inline static int_t count_equal(repr_type k0, repr_type k1) noexcept {
    int8x16_t m0{vreinterpretq_s8_u8(vceqq_s8(k0.val[0], k1.val[0]))};
    int8x16_t m1{vreinterpretq_s8_u8(vceqq_s8(k0.val[1], k1.val[1]))};
    return -vaddvq_s8(vaddq_s8(m0, m1));
  }
  inline static int_t count_not_equal(repr_type k0, repr_type k1) noexcept {
    int8x16_t m0{vreinterpretq_s8_u8(vceqq_s8(k0.val[0], k1.val[0]))};
    int8x16_t m1{vreinterpretq_s8_u8(vceqq_s8(k0.val[1], k1.val[1]))};
    return size + vaddvq_s8(vaddq_s8(m0, m1));
  }

  inline static repr_type hash_to_tombstone(repr_type k) noexcept {
    const int8x16_t t{vdupq_n_s8(tombstone)};
    const int8x16_t k0{vminq_s8(k.val[0], t)};
    const int8x16_t k1{vminq_s8(k.val[1], t)};
    return {k0, k1};
  }

  inline static repr_type not_hash_to_empty(repr_type k) noexcept {
    const uint8x16_t e{vreinterpretq_u8_s8(vdupq_n_s8(empty))};
    return {
      vreinterpretq_s8_u8(vminq_u8(vreinterpretq_u8_s8(k.val[0]), e)),
      vreinterpretq_s8_u8(vminq_u8(vreinterpretq_u8_s8(k.val[1]), e))
    };
  }

  inline static repr_type to_empty(repr_type k, mask_repr_type m) noexcept {
    const int8x16_t e{vdupq_n_s8(empty)};
    uint8x16x2_t mx{expand_(m)};
    return {
      vbslq_s8(mx.val[0], e, k.val[0]),
      vbslq_s8(mx.val[1], e, k.val[1])
    };
  }

  inline static state_t at(repr_type k, int_t i) noexcept {
    NVHM_ASSERT_(i < size, "i = ", i, ", size = ", size);
    return vgetq_lane_s8(vqtbl2q_s8(k, vdupq_n_u8(static_cast<uint8_t>(i))), 0);
  }

  inline static lru_repr_type load_lru(const lru_t* ptr) noexcept {
    return vld1q_u8_x2(assume_aligned<size>(ptr));
  }

  inline static void store_lru(lru_repr_type l, lru_t* ptr) noexcept {
    vst1q_u8_x2(assume_aligned<size>(ptr), l);
  }

  inline static std::pair<lru_t, int_t> min_lru(lru_repr_type l) noexcept {
    lru_t lru{min_lru_(l)};
    int_t idx{mask_type::next(mask_lru(l, lru))};
    return {lru, idx};
  }

  inline static mask_repr_type mask_lru(lru_repr_type l, lru_t lru) noexcept {
    uint8x16_t nx{vdupq_n_u8(lru)};
    return collapse_(vceqq_u8(l.val[0], nx), vceqq_u8(l.val[1], nx));
  }

  inline static mask_repr_type mask_min_lru(lru_repr_type l) noexcept {
    return mask_lru(l, min_lru_(l));
  }

  inline static mask_repr_type mask_lru_below(lru_repr_type l, lru_t t) noexcept {
    uint8x16_t tx{vdupq_n_u8(t)};
    return collapse_(vcltq_u8(l.val[0], tx), vcltq_u8(l.val[1], tx));
  }

  inline static lru_repr_type update_lru(lru_repr_type l, mask_repr_type n) noexcept {
    NVHM_ASSERT_(mask_type::has_next(n) && mask_type::next(n) < size, "n = ", render_mask<mask_type>(n));
    n = mask_type::truncate(n);
    uint8x16x2_t nx{expand_(n)};

    // Mask slots if they need rescaling.
    uint8x16_t max_lru_u8{vdupq_n_u8(max_lru)};
    uint32x4_t r0{vreinterpretq_u32_u8(vceqq_u8(l.val[0], max_lru_u8))};
    uint32x4_t r1{vreinterpretq_u32_u8(vceqq_u8(l.val[1], max_lru_u8))};
    r0 = vtstq_u32(r0, vreinterpretq_u32_u8(nx.val[0]));
    r1 = vtstq_u32(r1, vreinterpretq_u32_u8(nx.val[1]));
    r0 = vorrq_u32(r0, r1);
    r0 = vdupq_n_u32(vmaxvq_u32(r0));

    // Rescale and increment.
    l.val[0] = vshlq_u8(l.val[0], vreinterpretq_s8_u32(r0));
    l.val[1] = vshlq_u8(l.val[1], vreinterpretq_s8_u32(r0));
    l.val[0] = vsubq_u8(l.val[0], nx.val[0]);
    l.val[1] = vsubq_u8(l.val[1], nx.val[1]);
    return l;
  }

  inline static repr_type to_empty_if_lru_below(repr_type k, lru_repr_type l, lru_t t) noexcept {
    uint8x16_t tx{vdupq_n_u8(t)};
    l.val[0] = vcltq_u8(l.val[0], tx);
    l.val[1] = vcltq_u8(l.val[1], tx);

    const int8x16_t e{vdupq_n_s8(empty)};
    return {
      vbslq_s8(l.val[0], e, k.val[0]),
      vbslq_s8(l.val[1], e, k.val[1])
    };
  }

  inline static lru_t lru_at(lru_repr_type l, int_t i) noexcept {
    NVHM_ASSERT_(i < size, "i = ", i, ", size = ", size);
    return vgetq_lane_u8(vqtbl2q_u8(l, vdupq_n_u8(static_cast<uint8_t>(i))), 0);
  }

 private:
  /**
   * Truncate NEON 8-bit mask (0x00 or 0xff) into a 2-bit masks (0x1, 0x4, 0x10, 0x40).
   */
  inline static mask_repr_type collapse_(uint8x16_t m8_0, uint8x16_t m8_1) noexcept {
    // Isolate bits.
    const uint32x4_t bits{vdupq_n_u32(UINT32_C(0x4010'0401))};
    m8_0 = vandq_u8(m8_0, vreinterpretq_u8_u32(bits));
    m8_1 = vandq_u8(m8_1, vreinterpretq_u8_u32(bits));

    m8_0 = vpaddq_u8(m8_0, m8_1);  // 01|04, 10|40, ...
    m8_0 = vpaddq_u8(m8_0, m8_0);  // 01|04|10|40, ...

    return vgetq_lane_u64(vreinterpretq_u64_u8(m8_0), 0);
  }

  inline static uint8x16x2_t expand_(mask_repr_type m) noexcept {
    uint8x16_t m2{vreinterpretq_u8_u64(vdupq_n_u64(m))};

    uint16x8_t m4{vreinterpretq_u16_u8(vzip1q_u8(m2, m2))};

    uint8x16_t m8_0{vreinterpretq_u8_u16(vzip1q_u16(m4, m4))};
    uint8x16_t m8_1{vreinterpretq_u8_u16(vzip2q_u16(m4, m4))};

    const uint32x4_t bits{vdupq_n_u32(UINT32_C(0x4010'0401))};
    m8_0 = vtstq_u8(m8_0, vreinterpretq_u8_u32(bits));
    m8_1 = vtstq_u8(m8_1, vreinterpretq_u8_u32(bits));

    return {m8_0, m8_1};
  }

  inline static lru_t min_lru_(lru_repr_type l) noexcept {
    return vminvq_u8(vminq_u8(l.val[0], l.val[1]));
  }
};

template <>
struct neon_kernel<64> final : public kernel<-128, -127, false> {
  using repr_type = int8x16x4_t;
  using lru_repr_type = uint8x16x4_t;

  constexpr static int_t size{num_bytes_v<repr_type>};
  constexpr static bitmask_t size_mask{size_mask_v<size>};

  using mask_type = uint64_mask64_1r_t;
  static_assert(mask_type::max_count == size && mask_type::fully_utilized);
  
  using mask_repr_type = typename mask_type::repr_type;
  static_assert(num_bytes_v<mask_repr_type> * 8 == size);

  NVHM_MAKE_NOT_INSTANTIABLE_(neon_kernel);

  inline static repr_type load(const state_t* ptr) noexcept {
    return vld1q_s8_x4(assume_aligned<size>(ptr));
  }

  inline static void store(repr_type k, state_t* ptr) noexcept {
    vst1q_s8_x4(assume_aligned<size>(ptr), k);
  }

  inline static mask_repr_type mask(repr_type k, state_t s) noexcept {
    int8x16_t sx{vdupq_n_s8(s)};
    uint8x16_t m0{vceqq_s8(k.val[0], sx)};
    uint8x16_t m1{vceqq_s8(k.val[1], sx)};
    uint8x16_t m2{vceqq_s8(k.val[2], sx)};
    uint8x16_t m3{vceqq_s8(k.val[3], sx)};
    return collapse_(m0, m1, m2, m3);
  }
  inline static mask_repr_type mask_not(repr_type k, state_t s) noexcept { return ~mask(k, s); }

  inline static mask_repr_type mask_hash(repr_type k) noexcept {
    uint8x16_t m0{vcgezq_s8(k.val[0])};
    uint8x16_t m1{vcgezq_s8(k.val[1])};
    uint8x16_t m2{vcgezq_s8(k.val[2])};
    uint8x16_t m3{vcgezq_s8(k.val[3])};
    return collapse_(m0, m1, m2, m3);
  }
  inline static mask_repr_type mask_not_hash(repr_type k) noexcept {
    uint8x16_t m0{vcltzq_s8(k.val[0])};
    uint8x16_t m1{vcltzq_s8(k.val[1])};
    uint8x16_t m2{vcltzq_s8(k.val[2])};
    uint8x16_t m3{vcltzq_s8(k.val[3])};
    return collapse_(m0, m1, m2, m3);
  }

  inline static mask_repr_type mask_empty(repr_type k) noexcept { return mask(k, empty); }
  inline static mask_repr_type mask_not_empty(repr_type k) noexcept { return mask_not(k, empty); }
  inline static mask_repr_type mask_tombstone(repr_type k) noexcept { return mask(k, tombstone); }
  inline static mask_repr_type mask_not_tombstone(repr_type k) noexcept { return mask_not(k, tombstone); }

  inline static mask_repr_type mask_equal(repr_type k0, repr_type k1) noexcept {
    uint8x16_t m0{vceqq_s8(k0.val[0], k1.val[0])};
    uint8x16_t m1{vceqq_s8(k0.val[1], k1.val[1])};
    uint8x16_t m2{vceqq_s8(k0.val[2], k1.val[2])};
    uint8x16_t m3{vceqq_s8(k0.val[3], k1.val[3])};
    return collapse_(m0, m1, m2, m3);
  }
  inline static mask_repr_type mask_not_equal(repr_type k0, repr_type k1) noexcept { return ~mask_equal(k0, k1); }

  inline static state_t min(repr_type k) noexcept {
    return vminvq_s8(vminq_s8(
      vminq_s8(k.val[0], k.val[1]),
      vminq_s8(k.val[2], k.val[3])
    ));
  }
  inline static state_t max(repr_type k) noexcept {
    return vmaxvq_s8(vmaxq_s8(
      vmaxq_s8(k.val[0], k.val[1]),
      vmaxq_s8(k.val[2], k.val[3])
    ));
  }

  inline static bool has_hash(repr_type k) noexcept { return is_hash(max(k)); }
  inline static bool has_not_hash(repr_type k) noexcept { return is_not_hash(min(k)); }

  inline static bool has(repr_type k, state_t s) noexcept {
    int8x16_t sx{vdupq_n_s8(s)};
    uint8x16_t m0{vceqq_s8(k.val[0], sx)};
    uint8x16_t m1{vceqq_s8(k.val[1], sx)};
    uint8x16_t m2{vceqq_s8(k.val[2], sx)};
    uint8x16_t m3{vceqq_s8(k.val[3], sx)};
    m0 = vmaxq_u8(m0, m1);
    m2 = vmaxq_u8(m2, m3);
    m0 = vmaxq_u8(m0, m2);
    return vmaxvq_u8(m0) != 0;
  }
  inline static bool has_not(repr_type k, state_t s) noexcept {
    int8x16_t sx{vdupq_n_s8(s)};
    uint8x16_t m0{vceqq_s8(k.val[0], sx)};
    uint8x16_t m1{vceqq_s8(k.val[1], sx)};
    uint8x16_t m2{vceqq_s8(k.val[2], sx)};
    uint8x16_t m3{vceqq_s8(k.val[3], sx)};
    m0 = vminq_u8(m0, m1);
    m2 = vminq_u8(m2, m3);
    m0 = vminq_u8(m0, m2);
    return vminvq_u8(m0) == 0;
  }

  inline static bool has_empty(repr_type k) noexcept { return min(k) == empty; }
  inline static bool has_not_empty(repr_type k) noexcept { return max(k) != empty; }
  inline static bool has_tombstone(repr_type k) noexcept { return has(k, tombstone); }
  inline static bool has_not_tombstone(repr_type k) noexcept { return has_not(k, tombstone); }

  inline static int_t count_equal(repr_type k0, repr_type k1) noexcept {
    int8x16_t m0{vreinterpretq_s8_u8(vceqq_s8(k0.val[0], k1.val[0]))};
    int8x16_t m1{vreinterpretq_s8_u8(vceqq_s8(k0.val[1], k1.val[1]))};
    int8x16_t m2{vreinterpretq_s8_u8(vceqq_s8(k0.val[2], k1.val[2]))};
    int8x16_t m3{vreinterpretq_s8_u8(vceqq_s8(k0.val[3], k1.val[3]))};
    m0 = vaddq_s8(m0, m1);
    m2 = vaddq_s8(m2, m3);
    return -vaddvq_s8(vaddq_s8(m0, m2));
  }
  inline static int_t count_not_equal(repr_type k0, repr_type k1) noexcept {
    int8x16_t m0{vreinterpretq_s8_u8(vceqq_s8(k0.val[0], k1.val[0]))};
    int8x16_t m1{vreinterpretq_s8_u8(vceqq_s8(k0.val[1], k1.val[1]))};
    int8x16_t m2{vreinterpretq_s8_u8(vceqq_s8(k0.val[2], k1.val[2]))};
    int8x16_t m3{vreinterpretq_s8_u8(vceqq_s8(k0.val[3], k1.val[3]))};
    m0 = vaddq_s8(m0, m1);
    m2 = vaddq_s8(m2, m3);
    return size + vaddvq_s8(vaddq_s8(m0, m2));
  }

  inline static repr_type hash_to_tombstone(repr_type k) noexcept {
    int8x16_t t{vdupq_n_s8(tombstone)};
    return {
      vminq_s8(k.val[0], t),
      vminq_s8(k.val[1], t),
      vminq_s8(k.val[2], t),
      vminq_s8(k.val[3], t)
    };
  }

  inline static repr_type not_hash_to_empty(repr_type k) noexcept {
    uint8x16_t e{vreinterpretq_u8_s8(vdupq_n_s8(empty))};
    return {
      vreinterpretq_s8_u8(vminq_u8(vreinterpretq_u8_s8(k.val[0]), e)),
      vreinterpretq_s8_u8(vminq_u8(vreinterpretq_u8_s8(k.val[1]), e)),
      vreinterpretq_s8_u8(vminq_u8(vreinterpretq_u8_s8(k.val[2]), e)),
      vreinterpretq_s8_u8(vminq_u8(vreinterpretq_u8_s8(k.val[3]), e))
    };
  }

  inline static repr_type to_empty(repr_type k, mask_repr_type m) noexcept {
    const int8x16_t e{vdupq_n_s8(empty)};
    uint8x16x4_t mx{expand_(m)};
    return {
      vbslq_s8(mx.val[0], e, k.val[0]),
      vbslq_s8(mx.val[1], e, k.val[1]),
      vbslq_s8(mx.val[2], e, k.val[2]),
      vbslq_s8(mx.val[3], e, k.val[3])
    };
  }

  inline static state_t at(repr_type k, int_t i) noexcept {
    NVHM_ASSERT_(i < size, "i = ", i, ", size = ", size);
    return vgetq_lane_s8(vqtbl4q_s8(k, vdupq_n_u8(static_cast<uint8_t>(i))), 0);
  }

  inline static lru_repr_type load_lru(const lru_t* ptr) noexcept {
    return vld1q_u8_x4(assume_aligned<size>(ptr));
  }

  inline static void store_lru(lru_repr_type l, lru_t* ptr) noexcept {
    vst1q_u8_x4(assume_aligned<size>(ptr), l);
  }

  inline static std::pair<lru_t, int_t> min_lru(lru_repr_type l) noexcept {
    lru_t lru{min_lru_(l)};
    int_t idx{mask_type::next(mask_lru(l, lru))};
    return {lru, idx};
  }

  inline static mask_repr_type mask_lru(lru_repr_type l, lru_t lru) noexcept {
    uint8x16_t nx{vdupq_n_u8(lru)};
    return collapse_(
      vceqq_u8(l.val[0], nx),
      vceqq_u8(l.val[1], nx),
      vceqq_u8(l.val[2], nx),
      vceqq_u8(l.val[3], nx)
    );
  }

  inline static mask_repr_type mask_min_lru(lru_repr_type l) noexcept { return mask_lru(l, min_lru_(l)); }

  inline static mask_repr_type mask_lru_below(lru_repr_type l, lru_t t) noexcept {
    uint8x16_t tx{vdupq_n_u8(t)};
    return collapse_(
      vcltq_u8(l.val[0], tx),
      vcltq_u8(l.val[1], tx),
      vcltq_u8(l.val[2], tx),
      vcltq_u8(l.val[3], tx)
    );
  }

  inline static lru_repr_type update_lru(lru_repr_type l, mask_repr_type n) noexcept {
    NVHM_ASSERT_(mask_type::has_next(n) && mask_type::next(n) < size, "n = ", render_mask<mask_type>(n));
    n = mask_type::truncate(n);
    uint8x16x4_t nx{expand_(n)};

    // Mask slots if they need rescaling.
    uint8x16_t max_lru_u8{vdupq_n_u8(max_lru)};
    uint32x4_t r0{vreinterpretq_u32_u8(vceqq_u8(l.val[0], max_lru_u8))};
    uint32x4_t r1{vreinterpretq_u32_u8(vceqq_u8(l.val[1], max_lru_u8))};
    uint32x4_t r2{vreinterpretq_u32_u8(vceqq_u8(l.val[2], max_lru_u8))};
    uint32x4_t r3{vreinterpretq_u32_u8(vceqq_u8(l.val[3], max_lru_u8))};
    r0 = vtstq_u32(r0, vreinterpretq_u32_u8(nx.val[0]));
    r1 = vtstq_u32(r1, vreinterpretq_u32_u8(nx.val[1]));
    r2 = vtstq_u32(r2, vreinterpretq_u32_u8(nx.val[2]));
    r3 = vtstq_u32(r3, vreinterpretq_u32_u8(nx.val[3]));
    r0 = vorrq_u32(vorrq_u32(r0, r1), vorrq_u32(r2, r3));
    r0 = vdupq_n_u32(vmaxvq_u32(r0));

    // Rescale and increment.
    l.val[0] = vshlq_u8(l.val[0], vreinterpretq_s8_u32(r0));
    l.val[1] = vshlq_u8(l.val[1], vreinterpretq_s8_u32(r0));
    l.val[2] = vshlq_u8(l.val[2], vreinterpretq_s8_u32(r0));
    l.val[3] = vshlq_u8(l.val[3], vreinterpretq_s8_u32(r0));
    l.val[0] = vsubq_u8(l.val[0], nx.val[0]);
    l.val[1] = vsubq_u8(l.val[1], nx.val[1]);
    l.val[2] = vsubq_u8(l.val[2], nx.val[2]);
    l.val[3] = vsubq_u8(l.val[3], nx.val[3]);
    return l;
  }

  inline static repr_type to_empty_if_lru_below(repr_type k, lru_repr_type l, lru_t t) noexcept {
    uint8x16_t tx{vdupq_n_u8(t)};
    l.val[0] = vcltq_u8(l.val[0], tx);
    l.val[1] = vcltq_u8(l.val[1], tx);
    l.val[2] = vcltq_u8(l.val[2], tx);
    l.val[3] = vcltq_u8(l.val[3], tx);

    const int8x16_t e{vdupq_n_s8(empty)};
    return {
      vbslq_s8(l.val[0], e, k.val[0]),
      vbslq_s8(l.val[1], e, k.val[1]),
      vbslq_s8(l.val[2], e, k.val[2]),
      vbslq_s8(l.val[3], e, k.val[3])
    };
  }

  inline static lru_t lru_at(lru_repr_type l, int_t i) noexcept {
    NVHM_ASSERT_(i < size, "i = ", i, ", size = ", size);
    return vgetq_lane_u8(vqtbl4q_u8(l, vdupq_n_u8(static_cast<uint8_t>(i))), 0);
  }

 private:
  /**
   * Truncate NEON 8-bit mask (0x00 or 0xff) into a single-bit masks (i.e., binary 0 or 1).
   */
  inline static mask_repr_type collapse_(
    uint8x16_t m8_0, uint8x16_t m8_1, uint8x16_t m8_2, uint8x16_t m8_3
  ) noexcept {
    // Isolate bits.
    const uint64x2_t bits{vdupq_n_u64(UINT64_C(0x8040'2010'0804'0201))};
    m8_0 = vandq_u8(m8_0, vreinterpretq_u8_u64(bits));
    m8_1 = vandq_u8(m8_1, vreinterpretq_u8_u64(bits));
    m8_2 = vandq_u8(m8_2, vreinterpretq_u8_u64(bits));
    m8_3 = vandq_u8(m8_3, vreinterpretq_u8_u64(bits));

    m8_0 = vpaddq_u8(m8_0, m8_1);  // 01|02, 04|08, 10|20, 40|80, ...
    m8_2 = vpaddq_u8(m8_2, m8_3);
    m8_0 = vpaddq_u8(m8_0, m8_2);  // 01|02|04|08, 10|20|40|80, ...
    m8_0 = vpaddq_u8(m8_0, m8_0);  // 01|02|04|08|10|20|40|80, ...

    return vgetq_lane_u64(vreinterpretq_u64_u8(m8_0), 0);
  }

  inline static uint8x16x4_t expand_(mask_repr_type m) noexcept {
    uint32x2_t m1{vcreate_u32(m)};
    uint8x16_t m1_0{vreinterpretq_u8_u32(vdupq_lane_u32(m1, 0))};
    uint8x16_t m1_1{vreinterpretq_u8_u32(vdupq_lane_u32(m1, 1))};

    uint16x8_t m2_0{vreinterpretq_u16_u8(vzip1q_u8(m1_0, m1_0))};
    uint16x8_t m2_1{vreinterpretq_u16_u8(vzip1q_u8(m1_1, m1_1))};

    uint32x4_t m4_0{vreinterpretq_u32_u16(vzip1q_u16(m2_0, m2_0))};
    uint32x4_t m4_1{vreinterpretq_u32_u16(vzip1q_u16(m2_1, m2_1))};

    uint8x16_t m8_0{vreinterpretq_u8_u32(vzip1q_u32(m4_0, m4_0))};
    uint8x16_t m8_1{vreinterpretq_u8_u32(vzip2q_u32(m4_0, m4_0))};
    uint8x16_t m8_2{vreinterpretq_u8_u32(vzip1q_u32(m4_1, m4_1))};
    uint8x16_t m8_3{vreinterpretq_u8_u32(vzip2q_u32(m4_1, m4_1))};

    const uint64x2_t bits{vdupq_n_u64(UINT64_C(0x8040'2010'0804'0201))};
    m8_0 = vtstq_u8(m8_0, vreinterpretq_u8_u64(bits));
    m8_1 = vtstq_u8(m8_1, vreinterpretq_u8_u64(bits));
    m8_2 = vtstq_u8(m8_2, vreinterpretq_u8_u64(bits));
    m8_3 = vtstq_u8(m8_3, vreinterpretq_u8_u64(bits));
    return {m8_0, m8_1, m8_2, m8_3};
  }

  inline static lru_t min_lru_(lru_repr_type l) noexcept {
    return vminvq_u8(vminq_u8(
      vminq_u8(l.val[0], l.val[1]),
      vminq_u8(l.val[2], l.val[3])
    ));
  }
};

using neon_kernel8_t = neon_kernel<8>;
using neon_kernel16_t = neon_kernel<16>;
using neon_kernel32_t = neon_kernel<32>;
using neon_kernel64_t = neon_kernel<64>;

}  // namespace nvhm
#endif


#if NVHM_WITH_SVE
#include <arm_sve.h>

namespace nvhm {

// We use valuse that can be used as immediates in SVE comparison ops.
// empty: -16
// tombstone: -15
template <int_t Size>
struct sve_kernel final : public kernel<-16, -15, false> {
#if defined(__ARM_FEATURE_SVE_BITS) && (NVHM_WITH_SVE_SIZE * 8 <= __ARM_FEATURE_SVE_BITS)
  using repr_type = svint8_t __attribute__((arm_sve_vector_bits(__ARM_FEATURE_SVE_BITS)));
  using lru_repr_type = svuint8_t __attribute__((arm_sve_vector_bits(__ARM_FEATURE_SVE_BITS)));
#else
  using repr_type = svint8_t;
  using lru_repr_type = svuint8_t;
#endif

  constexpr static int_t size{Size};
  constexpr static bitmask_t size_mask{size_mask_v<size>};

  using mask_type = sve_mask<size>;
  using mask_repr_type = typename mask_type::repr_type;

  NVHM_MAKE_NOT_INSTANTIABLE_(sve_kernel);

  inline static repr_type load(const state_t* ptr) noexcept {
    return svld1_s8(mask_type::full(), assume_aligned<std::min(size, cache_line_size)>(ptr));
  }

  inline static void store(repr_type k, state_t* ptr) noexcept {
    svst1_s8(mask_type::full(), assume_aligned<std::min(size, cache_line_size)>(ptr), k);
  }

  inline static mask_repr_type mask(repr_type k, state_t s) noexcept {
    return svcmpeq_n_s8(mask_type::full(), k, s);
  }
  inline static mask_repr_type mask_not(repr_type k, state_t s) noexcept {
    return svcmpne_n_s8(mask_type::full(), k, s);
  }

  inline static mask_repr_type mask_hash(repr_type k) noexcept {
    return svcmpge_n_s8(mask_type::full(), k, 0);
  }
  inline static mask_repr_type mask_not_hash(repr_type k) noexcept {
    return svcmplt_n_s8(mask_type::full(), k, 0);
  }

  inline static mask_repr_type mask_empty(repr_type k) noexcept { return mask(k, empty); }
  inline static mask_repr_type mask_not_empty(repr_type k) noexcept { return mask_not(k, empty); }
  inline static mask_repr_type mask_tombstone(repr_type k) noexcept { return mask(k, tombstone); }
  inline static mask_repr_type mask_not_tombstone(repr_type k) noexcept { return mask_not(k, tombstone); }

  inline static mask_repr_type mask_equal(repr_type k0, repr_type k1) noexcept {
    return svcmpeq_s8(mask_type::full(), k0, k1);
  }
  inline static mask_repr_type mask_not_equal(repr_type k0, repr_type k1) noexcept {
    return svcmpne_s8(mask_type::full(), k0, k1);
  }

  inline static bool has(repr_type k, state_t s) noexcept { return mask_type::has_next(mask(k, s)); }
  inline static bool has_not(repr_type k, state_t s) noexcept { return mask_type::has_next(mask_not(k, s)); }

  inline static bool has_hash(repr_type k) noexcept { return mask_type::has_next(mask_hash(k)); }
  inline static bool has_not_hash(repr_type k) noexcept { return mask_type::has_next(mask_not_hash(k)); }

  inline static bool has_empty(repr_type k) noexcept { return has(k, empty); }
  inline static bool has_not_empty(repr_type k) noexcept { return has_not(k, empty); }
  inline static bool has_tombstone(repr_type k) noexcept { return has(k, tombstone); }
  inline static bool has_not_tombstone(repr_type k) noexcept { return has_not(k, tombstone); }

  inline static int_t count_equal(repr_type k0, repr_type k1) noexcept {
    return mask_type::count(mask_equal(k0, k1));
  }
  inline static int_t count_not_equal(repr_type k0, repr_type k1) noexcept {
    return size - count_equal(k0, k1);
  }

  inline static repr_type hash_to_tombstone(repr_type k) noexcept {
    return svmin_n_s8_x(svptrue_b8(), k, tombstone);
  }

  inline static repr_type not_hash_to_empty(repr_type k) noexcept {
    return svreinterpret_s8_u8(svmin_n_u8_x(svptrue_b8(), svreinterpret_u8_s8(k), static_cast<uint8_t>(empty)));
  }

  inline static repr_type to_empty(repr_type k, mask_repr_type m) noexcept {
    return svdup_n_s8_m(k, m, empty);
  }

  inline static state_t at(repr_type k, int_t i) noexcept {
    NVHM_ASSERT_(i < size, "i = ", i, ", size = ", size);
    return svlasta_s8(mask_type::until(i), k);
  }

  inline static lru_repr_type load_lru(const lru_t* ptr) noexcept {
    return svld1_u8(mask_type::full(), assume_aligned<size>(ptr));
  }

  inline static void store_lru(lru_repr_type l, lru_t* ptr) noexcept {
    svst1_u8(mask_type::full(), assume_aligned<size>(ptr), l);
  }

  inline static std::pair<lru_t, int_t> min_lru(lru_repr_type l) noexcept {
    lru_t lru{min_lru_(l)};
    int_t idx{mask_type::next(mask_lru(l, lru))};
    return {lru, idx};
  }

  inline static mask_repr_type mask_lru(lru_repr_type l, lru_t lru) noexcept {
    return svcmpeq_n_u8(mask_type::full(), l, lru);
  }

  inline static mask_repr_type mask_min_lru(lru_repr_type l) noexcept { return mask_lru(l, min_lru_(l)); }

  inline static mask_repr_type mask_lru_below(lru_repr_type l, lru_t t) noexcept {
    return svcmplt_n_u8(mask_type::full(), l, t);
  }

  inline static lru_repr_type update_lru(lru_repr_type l, mask_repr_type n) noexcept {
    NVHM_ASSERT_(mask_type::has_next(n) && mask_type::next(n) < size, "n = ", render_mask<mask_type>(n));

    // Mask slots if they need rescaling.
    svbool_t r{svcmpeq_n_u8(n, l, max_lru)};
    n = mask_type::truncate(n);
    r = svbrkn_b_z(n, r, mask_type::full());

    // Rescale and increment.
    l = svlsr_n_u8_m(r, l, 1);
    l = svadd_n_u8_m(n, l, 1);

    return l;
  }

  inline static repr_type to_empty_if_lru_below(repr_type k, lru_repr_type l, lru_t t) noexcept {
    return to_empty(k, mask_lru_below(l, t));
  }

  inline static lru_t lru_at(lru_repr_type l, int_t i) noexcept {
    NVHM_ASSERT_(i < size, "i = ", i, ", size = ", size);
    return svlasta_u8(mask_type::until(i), l);
  }

 private:
  inline static lru_t min_lru_(lru_repr_type l) noexcept {
    return svminv_u8(mask_type::full(), l);
  }
};

template <>
constexpr int_t num_bytes_v<svint8_t>{NVHM_WITH_SVE_SIZE};
template <>
constexpr int_t num_bytes_v<svuint8_t>{NVHM_WITH_SVE_SIZE};

#if NVHM_WITH_SVE_SIZE >= 1
using sve_kernel1_t = sve_kernel<1>;
#endif
#if NVHM_WITH_SVE_SIZE >= 2
using sve_kernel2_t = sve_kernel<2>;
#endif
#if NVHM_WITH_SVE_SIZE >= 4
using sve_kernel4_t = sve_kernel<4>;
#endif
#if NVHM_WITH_SVE_SIZE >= 8
using sve_kernel8_t = sve_kernel<8>;
#endif
#if NVHM_WITH_SVE_SIZE >= 16
using sve_kernel16_t = sve_kernel<16>;
#endif
#if NVHM_WITH_SVE_SIZE >= 32
using sve_kernel32_t = sve_kernel<32>;
#endif
#if NVHM_WITH_SVE_SIZE >= 64
using sve_kernel64_t = sve_kernel<64>;
#endif
#if NVHM_WITH_SVE_SIZE >= 128
using sve_kernel128_t = sve_kernel<128>;
#endif
#if NVHM_WITH_SVE_SIZE >= 256
using sve_kernel256_t = sve_kernel<256>;
#endif

}  // namespace nvhm
#endif

namespace nvhm {

template <int_t Size>
struct default_kernel;
  
template <>
struct default_kernel<1> {
  // TODO: Prefer SVE if available?
  using type = fast_uint_kernel1_t;
};

template <>
struct default_kernel<2> {
  // TODO: Prefer SVE if available?
  using type = fast_uint_kernel2_t;
};

template <>
struct default_kernel<4> {
  // TODO: Prefer SVE if available?
  // TODO: MMX?
  using type = fast_uint_kernel4_t;
};

template <>
struct default_kernel<8> {
#if NVHM_WITH_SVE && NVHM_WITH_SVE_SIZE >= 8
  // TODO: Is this really better?
  using type = sve_kernel8_t;
#elif NVHM_WITH_NEON
  using type = neon_kernel8_t;
#else
  using type = fast_uint_kernel8_t;
#endif
};

template <>
struct default_kernel<16> {
#if NVHM_WITH_SSE >= 2
  using type = sse_kernel_t;
#elif NVHM_WITH_SVE && NVHM_WITH_SVE_SIZE >= 16
  using type = sve_kernel16_t;
#elif NVHM_WITH_NEON
  using type = neon_kernel16_t;
#else
  using type = fast_uint_kernel16_t;
#endif
};

template <>
struct default_kernel<32> {
#if NVHM_WITH_AVX >= 2
  using type = avx_kernel_t;
#elif NVHM_WITH_SVE && NVHM_WITH_SVE_SIZE >= 32
  using type = sve_kernel32_t;
#elif NVHM_WITH_NEON
  using type = neon_kernel32_t;
#else
  using type = array_kernel32_t;
#endif
};

template <>
struct default_kernel<64> {
#if NVHM_WITH_AVX512
  using type = avx512_kernel_t;
#elif NVHM_WITH_SVE && NVHM_WITH_SVE_SIZE >= 64
  using type = sve_kernel64_t;
#elif NVHM_WITH_NEON
  using type = neon_kernel64_t;
#else
  using type = array_kernel64_t;
#endif
};

template <>
struct default_kernel<128> {
#if NVHM_WITH_SVE && NVHM_WITH_SVE_SIZE >= 128
  using type = sve_kernel128_t;
#else
  using type = array_kernel128_t;
#endif
};

template <>
struct default_kernel<256> {
#if NVHM_WITH_SVE && NVHM_WITH_SVE_SIZE >= 256
  using type = sve_kernel256_t;
#else
  using type = array_kernel256_t;
#endif
};

template <>
struct default_kernel<512> {
  using type = array_kernel512_t;
};

using default_kernel1_t = typename default_kernel<1>::type;
using default_kernel2_t = typename default_kernel<2>::type;
using default_kernel4_t = typename default_kernel<4>::type;
using default_kernel8_t = typename default_kernel<8>::type;
using default_kernel16_t = typename default_kernel<16>::type;
using default_kernel32_t = typename default_kernel<32>::type;
using default_kernel64_t = typename default_kernel<64>::type;
using default_kernel128_t = typename default_kernel<128>::type;
using default_kernel256_t = typename default_kernel<256>::type;
using default_kernel512_t = typename default_kernel<512>::type;

constexpr int_t default_kernel_size{
#if NVHM_WITH_SVE && NVHM_WITH_SVE_SIZE >= 16
  num_bytes_v<svint8_t>
#elif NVHM_WITH_SSE >= 2
  sse_kernel_t::size
#else
  num_bytes_v<int_t>
#endif
};

template <int_t Size = default_kernel_size>
using default_kernel_t = typename default_kernel<Size>::type;

}  // namespace nvhm
