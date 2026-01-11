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

namespace nvhm {

template <psl_t MaxLength>
struct probe_seq {
  constexpr static psl_t max_length{MaxLength};
  static_assert(MaxLength >= 0);

  probe_seq() = delete;
  probe_seq(const probe_seq&) = delete;
  probe_seq(probe_seq&&) = delete;
  probe_seq& operator=(const probe_seq&) = delete;
  probe_seq& operator=(probe_seq&&) = delete;

  constexpr static bool has_next(const psl_t psl) noexcept {
    if constexpr (max_length > 0) {
      return psl < max_length;
    } else {
      return true;
    }
  }
};

template <psl_t MaxLength = 0>
struct linear_seq final : public probe_seq<MaxLength> {
  using base_type = probe_seq<MaxLength>;
  constexpr static psl_t max_length{base_type::max_length};

  linear_seq() = delete;
  linear_seq(const linear_seq&) = delete;
  linear_seq(linear_seq&&) = delete;
  linear_seq& operator=(const linear_seq&) = delete;
  linear_seq& operator=(linear_seq&&) = delete;

  constexpr static raw_pos_t next(
    const raw_pos_t seq, const psl_t psl, const size_t bucket_mask
  ) noexcept {
    NVHM_ASSERT_(
      psl >= 0 && (!max_length || psl < max_length), "The probe sequence is out of bounds"
    );
    return (seq + static_cast<raw_pos_t>(psl)) & bucket_mask;
  }

  constexpr static raw_pos_t step(const raw_pos_t seq, const psl_t, const size_t) noexcept {
    return seq;
  }
};

template <psl_t MaxLength = 0>
struct quadratic_seq final : public probe_seq<MaxLength> {
  using base_type = probe_seq<MaxLength>;
  constexpr static psl_t max_length{base_type::max_length};

  quadratic_seq() = delete;
  quadratic_seq(const quadratic_seq&) = delete;
  quadratic_seq(quadratic_seq&&) = delete;
  quadratic_seq& operator=(const quadratic_seq&) = delete;
  quadratic_seq& operator=(quadratic_seq&&) = delete;

  constexpr static raw_pos_t next(const raw_pos_t seq, const psl_t, const size_t) noexcept {
    return seq;
  }

  constexpr static raw_pos_t step(
    const raw_pos_t seq, const psl_t psl, const size_t bucket_mask
  ) noexcept {
    NVHM_ASSERT_(
      psl >= 0 && (!max_length || psl < max_length), "The probe sequence is out of bounds"
    );
    return (seq + static_cast<raw_pos_t>(psl)) & bucket_mask;
  }
};

using default_seq_t = quadratic_seq<>;

}  // namespace nvhm