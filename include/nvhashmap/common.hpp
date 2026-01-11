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

#include "stdlib_ext.hpp"

#include <functional>

/**
 * `nvhashmap_conf.hpp` is generated during CMake in your build directory under
 * `${CMAKE_BINARY_DIR}/include`. If your compiler fails locating it, something in your
 * configuration is odd. Try adding the it to the search path manually, by ensuring that
 * `${CMAKE_BINARY_DIR}/include` is part in your include file search path.
 */
#include <nvhashmap_conf.hpp>

#define NVHM_LIKELY_(_expr_) __builtin_expect((_expr_), 1)
#define NVHM_UNLIKELY_(_expr_) __builtin_expect((_expr_), 0)

#include <limits>
#include <type_traits>

namespace nvhm {

template <typename T>
constexpr static size_t num_bits{sizeof(T) * 8};

/**
 * Data type used for slots. Slots either contain a hash or `empty` or `tombstone`.
 * `empty` means that the slot was never used since the last rehash, or it was reclaimed.
 * `tombstone` means that the slot was used, then freed, but couldn't be reclaimed.
 */
using state_t = int8_t;
static_assert(std::is_signed_v<state_t> && sizeof(state_t) == 1);

constexpr static size_t num_state_bits{num_bits<state_t> - 1};
static_assert(
  num_state_bits == 7
);  // If you change this, you likely need to check everything else.

constexpr static size_t state_mask{(1 << num_state_bits) - 1};

constexpr bool slot_is_occupied(const state_t s) noexcept { return s >= 0; }
constexpr bool slot_not_occupied(const state_t s) noexcept { return s < 0; }

using lru_t = uint8_t;
static_assert(std::is_unsigned_v<lru_t> && sizeof(lru_t) == sizeof(state_t));

constexpr static lru_t max_lru{0xff};
constexpr static lru_t default_lru{max_lru / 2};
static_assert(default_lru < max_lru);

using raw_pos_t = size_t;
constexpr static raw_pos_t npos{std::numeric_limits<raw_pos_t>::max()};

using psl_t = intptr_t;

}  // namespace nvhm
