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

#include <cxxabi.h>
#include <iomanip>
#include <iostream>
#include <nvhashmap/map.hpp>
#include <random>

using namespace nvhm;

static std::random_device rd;

template <typename T>
std::string type_to_string() {
  int status;
  char* real_name{abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, &status)};
  std::string s;
  if (real_name) {
    s = real_name;
  }
  std::free(real_name);
  return s;
}

template <typename Map>
int probe_seq_len(
  const size_t seed, const size_t num_keys, const bool shuffle_keys, const size_t interval
) {
  using map_t = Map;
  const std::string map_type_name{type_to_string<map_t>()};
  using key_t = typename map_t::key_type;

  std::vector<key_t> keys(num_keys);
  for (size_t i{}; i != keys.size(); ++i) {
    keys[i] = static_cast<key_t>(i * 10);
  }
  if (shuffle_keys) {
    std::mt19937 g(seed);
    std::shuffle(keys.begin(), keys.end(), g);
  }

  map_t map(num_keys);
  for (const key_t k : keys) {
    map.upsert(k);
    if (map.size() % interval != 0) {
      continue;
    }

    std::vector<size_t> overall_psl;
    for (const key_t k : keys) {
      const psl_t psl{map.psl(k)};

      overall_psl.resize(std::max(overall_psl.size(), static_cast<size_t>(psl + 1)));
      ++overall_psl[static_cast<size_t>(psl)];
    }

    std::cout << "Map Type: " << map_type_name << ", size = " << std::setw(7) << map.size() << "/"
              << map.capacity() << ", num_keys = " << num_keys << " ~ ";
    for (size_t i{}; i != overall_psl.size(); ++i) {
      if (overall_psl[i]) {
        std::cout << "   [" << i << "] " << std::setw(4) << overall_psl[i];
      }
    }
    std::cout << '\n';
  }

  return 0;
}

int main() {
  const size_t num_keys{940'000};
  const bool shuffle_keys{true};
  const size_t interval{50'000};

  using key_t = int64_t;
  using meta_t = time_t;

  const size_t seed{rd()};

  using seq_t = default_seq_t;
  // using seq_t = linear_seq<>;
  // using seq_t = quadratic_seq<32>;

  probe_seq_len<map<key_t, meta_t, char, default_kernel32_t, seq_t>>(
    seed, num_keys, shuffle_keys, interval
  );

  probe_seq_len<map<key_t, meta_t, char, default_kernel64_t, seq_t>>(
    seed, num_keys, shuffle_keys, interval
  );

  probe_seq_len<map<key_t, meta_t, char, default_kernel128_t, seq_t>>(
    seed, num_keys, shuffle_keys, interval
  );

  probe_seq_len<map<key_t, meta_t, char, default_kernel256_t, seq_t>>(
    seed, num_keys, shuffle_keys, interval
  );

  probe_seq_len<map<key_t, meta_t, char, default_kernel512_t, seq_t>>(
    seed, num_keys, shuffle_keys, interval
  );
  return 0;
}