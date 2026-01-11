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

/**
 * hash layout (traditional):
 * |    6          5          4          3          2          1            |
 * | 3210 9876543210 9876543210 9876543210 9876543210 9876543210 9876543210 |
 * | <-------------------------------------------------------------><-----> |
 * | HHHH HHHHHHHHHH HHHHHHHHHH HHHHHHHHHH HHHHHHHHHH HHHHHHHHHH HHHSSSSSSS |
 *
 * hash layout (monkey_wrench):
 * |    6          5          4          3          2          1            |
 * | 3210 9876543210 9876543210 9876543210 9876543210 9876543210 9876543210 |
 * | <------><------------------------------------------------------------> |
 * | SSSS SSSHHHHHHH HHHHHHHHHH HHHHHHHHHH HHHHHHHHHH HHHHHHHHHH HHHHHHHHHH |
 *
 * 7 bits = state / fingerprint
 * 57 bits = hashspace
 *
 * Although it looks counter-intuitive at first, this works better as those
 * bits are less correlated, which leads to shorter average probe sequences.
 *
 * By taking advantage of the CPU zero-extending right shifts we can save one
 * logic operation by taking advantage of the CPU zero-extending right shifts.
 */
using hash_t = uint64_t;
static_assert(std::is_unsigned_v<hash_t> && sizeof(hash_t) > sizeof(state_t));

template <typename Key>
constexpr hash_t key_to_hash(const Key& k) noexcept {
  hash_t h{hasher<Key>(k)};

  if constexpr (hash_mixer == hash_mixer_t::marsaglia) {
    // Mix bits by running them through a single-step Marsaglia-style multiply-with-carry generator.
    //
    // The prime here originally comes from Robin Hood hash:
    // https://github.com/martinus/better-faster-stronger-mixer/blob/master/include/mixer/robin_hood_hash_int.h

    // TODO: Do some quantiative experiments to see if we can find a better prime.
    constexpr uint64_t c{UINT64_C(0xde5f'b9d2'6304'58e9)};
    const auto h128{static_cast<__uint128_t>(h) * c};
    h = static_cast<hash_t>(h128 >> 64) + static_cast<hash_t>(h128);
  }

  return h;
}

/**
 * Initializes a sequence.
 */
template <typename Kernel>
constexpr static raw_pos_t hash_to_pos(hash_t h, const size_t bucket_mask) noexcept {
  using kernel_t = Kernel;

  if constexpr (hash_layout == hash_layout_t::traditional) {
    if constexpr (kernel_t::size <= 128) {
      h = h / (128 / kernel_t::size);
    } else {
      h = h / 128 * kernel_t::size;
    }
  } else if constexpr (hash_layout == hash_layout_t::monkey_wrench) {
    h = h * kernel_t::size;
  }

  return h & bucket_mask;
}

constexpr state_t hash_to_state(hash_t h) noexcept {
  if constexpr (hash_layout == hash_layout_t::traditional) {
    h &= state_mask;
  } else if constexpr (hash_layout == hash_layout_t::monkey_wrench) {
    h >>= num_bits<hash_t> - num_state_bits;
  }

  return static_cast<state_t>(h);
}

}  // namespace nvhm
