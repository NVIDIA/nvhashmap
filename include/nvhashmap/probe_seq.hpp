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

namespace nvhm {

template<int_t Size>
constexpr std::pair<raw_pos_t, int_t> splice_pos(raw_pos_t pos) noexcept {
  constexpr bitmask_t mask{size_mask_v<Size>};
  return {
    align_pos<~mask>(pos),
    align_pos<mask>(pos)
  };
}

template <typename Self, int_t Alignment, psl_t MaxLength>
class probe_seq : public self_aware<Self> {
 public:
  using base_type = self_aware<Self>;
  using self_type = typename base_type::self_type;

  constexpr static int_t alignment{Alignment};
  constexpr static bitmask_t alignment_mask{size_mask_v<alignment>};

  constexpr static psl_t max_length{MaxLength};
  static_assert(max_length >= alignment && max_length <= inf_psl);
  constexpr static bool is_bounded{max_length < inf_psl};
  constexpr static bool is_unbounded{max_length == inf_psl};

  constexpr probe_seq() : psl_{} {}

  using base_type::self;

  constexpr bool has_next() const noexcept {
    if constexpr (is_bounded) {
      return psl_ < max_length;
    } else {
      return true;
    }
  }

  constexpr psl_t psl() const noexcept { return psl_; }

  constexpr self_type& operator+=(int_t n) noexcept {
    NVHM_ASSERT_(n >= 0, "n = ", n);
    //NVHM_ASSERT_(psl_ >= 0 && psl_ < max_length, "Probe sequence is out of bounds! (psl = ", psl_, ", max_length = ", max_length, ')');
    psl_ += n;
    NVHM_ASSERT_(psl_ >= 0);
    return *self();
  }

  constexpr friend bool operator==(const probe_seq& lhs, const probe_seq& rhs) noexcept { return lhs.psl_ == rhs.psl_; }
  constexpr friend bool operator!=(const probe_seq& lhs, const probe_seq& rhs) noexcept { return !(lhs == rhs); }

  constexpr friend self_type operator+(self_type seq, int_t n) noexcept { seq += n; return seq; }

  constexpr friend void swap(probe_seq& lhs, probe_seq& rhs) noexcept {
    std::swap(lhs.psl_, rhs.psl_);
  }

 protected:
  psl_t psl_;
};

template <typename Self, psl_t MaxLength>
class unaligned_probe_seq : public probe_seq<Self, 0, MaxLength> {
 public:
  using base_type = probe_seq<Self, 0, MaxLength>;

  unaligned_probe_seq() = delete;
  constexpr unaligned_probe_seq(raw_pos_t pos) : base_{pos} {
    NVHM_ASSERT_(pos >= 0, "pos = ", pos);
  }

  constexpr raw_pos_t base() const noexcept { return base_; }

  constexpr friend bool operator==(const unaligned_probe_seq& lhs, const unaligned_probe_seq& rhs) noexcept {
    return static_cast<const base_type&>(lhs) == static_cast<const base_type&>(rhs) &&
      lhs.base_ == rhs.base_;
  }
  constexpr friend bool operator!=(const unaligned_probe_seq& lhs, const unaligned_probe_seq& rhs) noexcept { return !(lhs == rhs); }

  constexpr friend void swap(unaligned_probe_seq& lhs, unaligned_probe_seq& rhs) noexcept {
    swap(static_cast<base_type&>(lhs), static_cast<base_type&>(rhs));

    std::swap(lhs.base_, rhs.base_);
  }

 protected:
  raw_pos_t base_;
};

template <typename Self, int_t Alignment, psl_t MaxLength>
class aligned_probe_seq : public probe_seq<Self, Alignment, MaxLength> {
 public:
  using base_type = probe_seq<Self, Alignment, MaxLength>;

  using base_type::alignment;

  aligned_probe_seq() = delete;
  constexpr aligned_probe_seq(raw_pos_t pos) {
    std::tie(base_, shift_) = splice_pos<alignment>(pos);
  }

  constexpr raw_pos_t base() const noexcept { return base_; }
  constexpr int_t shift() const noexcept { return shift_; }

  constexpr friend bool operator==(const aligned_probe_seq& lhs, const aligned_probe_seq& rhs) noexcept {
    return static_cast<const base_type&>(lhs) == static_cast<const base_type&>(rhs) &&
      lhs.base_ == rhs.base_ &&
      lhs.shift_ == rhs.shift_;
  }
  constexpr friend bool operator!=(const aligned_probe_seq& lhs, const aligned_probe_seq& rhs) noexcept { return !(lhs == rhs); }

  constexpr friend void swap(aligned_probe_seq& lhs, aligned_probe_seq& rhs) noexcept {
    swap(static_cast<base_type&>(lhs), static_cast<base_type&>(rhs));

    std::swap(lhs.base_, rhs.base_);
    std::swap(lhs.shift_, rhs.shift_);
  }

 protected:
  raw_pos_t base_;
  int_t shift_;
};

template <int_t Alignment, psl_t MaxLength = inf_psl>
class linear_seq : public aligned_probe_seq<linear_seq<Alignment, MaxLength>, Alignment, MaxLength> {
 public:
  using base_type = aligned_probe_seq<linear_seq, Alignment, MaxLength>;

  using base_type::alignment_mask;

  linear_seq() = delete;
  constexpr linear_seq(raw_pos_t pos) : base_type(pos) {}

  constexpr raw_pos_t next() const noexcept {
    psl_t psl{psl_};
    raw_pos_t maj{align_pos<~alignment_mask>(psl)};
    raw_pos_t min{align_pos<alignment_mask>(psl + shift_)};
    return base_ + maj + min;
  }

 protected:
  using base_type::psl_;
  using base_type::base_;
  using base_type::shift_;
};

template <psl_t MaxLength>
class linear_seq<0, MaxLength> : public unaligned_probe_seq<linear_seq<0, MaxLength>, MaxLength> {
 public:
  using base_type = unaligned_probe_seq<linear_seq, MaxLength>;

  linear_seq() = delete;
  constexpr linear_seq(raw_pos_t pos) : base_type(pos) {}

  constexpr raw_pos_t next() const noexcept { return base_ + psl_; }

 protected:
  using base_type::psl_;
  using base_type::base_;
};

template <int_t Alignment, psl_t MaxLength = inf_psl>
class quadratic_seq : public aligned_probe_seq<quadratic_seq<Alignment, MaxLength>, Alignment, MaxLength> {
 public:
  using base_type = aligned_probe_seq<quadratic_seq, Alignment, MaxLength>;
  using self_type = typename base_type::self_type;

  using base_type::alignment_mask;
  
  quadratic_seq() = delete;
  constexpr quadratic_seq(raw_pos_t pos) : base_type(pos) {}

  using base_type::self;

  constexpr raw_pos_t next() const noexcept {
    return base_ + align_pos<alignment_mask>(psl_ + shift_);
  }

  constexpr self_type& operator+=(int_t n) noexcept {
    base_type::operator+=(n);
    psl_t psl{psl_};
    base_ += align_pos<alignment_mask>(psl) ? 0 : psl;

    return *self();
  }

 protected:
  using base_type::psl_;
  using base_type::base_;
  using base_type::shift_;
};

template <psl_t MaxLength>
class quadratic_seq<0, MaxLength> : public unaligned_probe_seq<quadratic_seq<0, MaxLength>, MaxLength> {
 public:
  using base_type = unaligned_probe_seq<quadratic_seq, MaxLength>;
  using self_type = typename base_type::self_type;

  quadratic_seq() = delete;
  constexpr quadratic_seq(raw_pos_t pos) : base_type(pos) {}

  using base_type::self;

  constexpr raw_pos_t next() const noexcept { return base_; }

  constexpr self_type& operator+=(int_t n) noexcept {
    base_type::operator+=(n);
    base_ += psl_;

    return *self();
  }

 protected:
  using base_type::psl_;
  using base_type::base_;
};

using default_seq_t = quadratic_seq<cache_line_size, inf_psl>;

}  // namespace nvhm