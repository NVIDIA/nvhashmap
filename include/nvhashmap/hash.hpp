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
 * | RRRR RRRRRRRRRR HHHHHHHHHH HHHHHHHHHH HHHHHHHHHH HHHHHHHHHH HHHSSSSSSS |
 *
 * hash layout (monkey_wrench):
 * |    6          5          4          3          2          1            |
 * | 3210 9876543210 9876543210 9876543210 9876543210 9876543210 9876543210 |
 * | <------><------------------------------------------------------------> |
 * | SSSS SSSRRRRRRR RRRRRRRHHH HHHHHHHHHH HHHHHHHHHH HHHHHHHHHH HHHHHHHHHH |
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
static_assert(is_unsigned_v<hash_t> && sizeof(hash_t) >= sizeof(int_t));

template <int_t KernelSize>
constexpr raw_pos_t hash_to_pos(hash_t h) noexcept {
  static_assert(has_single_bit(KernelSize));
  constexpr int_t num_kernel_bits{countr_zero(KernelSize)};

  // TODO: Needs more theoretical analysis.
  if constexpr (hash_layout == hash_layout_t::traditional) {
    if constexpr (num_kernel_bits >= num_state_bits) {
      h <<= num_kernel_bits - num_state_bits;
      
    } else {
      h >>= num_state_bits - num_kernel_bits;
    }
  } else if constexpr (hash_layout == hash_layout_t::monkey_wrench) {
    // TODO: Is this really necessary?
    h <<= num_kernel_bits;
  }
  return static_cast<raw_pos_t>(h);
}

constexpr state_t hash_to_state(hash_t h) noexcept {
  if constexpr (hash_layout == hash_layout_t::traditional) {
    h &= state_mask;
  } else if constexpr (hash_layout == hash_layout_t::monkey_wrench) {
    h >>= num_bits_v<hash_t> - num_state_bits;
  }
  return static_cast<state_t>(h);
}

constexpr static int_t max_num_shards{16'384};
static_assert(has_single_bit(max_num_shards));
constexpr static int_t num_shard_bits{countr_zero(max_num_shards)};
static_assert(num_shard_bits >= 4 && num_shard_bits <= 16);

using shard_idx_t = int_t;

constexpr shard_idx_t hash_to_shard_idx(hash_t h) noexcept {
  if constexpr (hash_layout == hash_layout_t::traditional) {
    h >>= num_bits_v<hash_t> - num_shard_bits;
  } else if constexpr (hash_layout == hash_layout_t::monkey_wrench) {
    h >>= num_bits_v<hash_t> - num_state_bits - num_shard_bits;
  }
  return static_cast<int_t>(h);
}

template <typename Key>
constexpr hash_t key_to_hash(const Key& key) noexcept {
  hash_t h{hasher<Key>(key)};
  // TODO: We need a better hasher for strings.
  //if constexpr (std::is_same_v<Key, std::string>) {
  //  if (key.size() <= 8) {
  //    h = 0;
  //    memcpy(&h, key.data(), key.size());
  //  } else {
  //    h = hasher<Key>(key);
  //  }
  //}

  if constexpr (hash_mixer == hash_mixer_t::marsaglia) {
    // Mix bits by running them through a single-step Marsaglia-style multiply-with-carry generator.
    //
    // The prime here originally comes from Robin Hood hash:
    // https://github.com/martinus/better-faster-stronger-mixer/blob/master/include/mixer/robin_hood_hash_int.h

    // TODO: Do some quantiative experiments to see if we can find a better prime.
    constexpr uint64_t c{UINT64_C(0xde5f'b9d2'6304'58e9)};
    const __uint128_t h128{static_cast<__uint128_t>(h) * c};
    h = low_bits(h128) + high_bits(h128);
  }

  return h;
}

template <typename Key>
constexpr state_t key_to_state(const Key& key) noexcept {
  return hash_to_state(key_to_hash(key));
}

constexpr static int_t num_pos_bits{num_bits_v<int_t> - num_state_bits - num_shard_bits};
static_assert(num_pos_bits >= 32 && num_pos_bits <= 64);
constexpr static int_t max_capacity{int_t{1} << num_pos_bits};

}  // namespace nvhm
