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

enum class goal_t {
  interval,
  load_factor,
};

template <typename Map, goal_t Goal>
int count_collisions(
  const size_t seed, const size_t num_keys, const bool shuffle_keys, const size_t param
) {
  using map_t = Map;
  using key_t = typename map_t::key_type;

  std::vector<key_t> keys(num_keys);
  for (size_t i{}; i != keys.size(); ++i) {
    keys[i] = static_cast<key_t>(i);
  }
  if (shuffle_keys) {
    std::mt19937_64 g(seed);
    std::shuffle(keys.begin(), keys.end(), g);
  }

  const int size_alignment{static_cast<int>(std::to_string(num_keys).size())};

  std::cout << type_to_string<map_t>() << ", ";
  if constexpr (Goal == goal_t::interval) {
    std::cout << type_to_string<map_t>() << ", collisions @ " << param << " interval\n";
    map_t map(num_keys);

    for (const key_t k : keys) {
      map.upsert(k);
      if (map.size() % param == 0) {
        std::cout << "size = " << std::setw(size_alignment) << map.size() << " /"
                  << std::setw(size_alignment + 1) << map.capacity() << ": ";
        const auto colls{map.state_collisions()};
        for (size_t i{}; i != colls.size(); ++i) {
          if (colls[i] > 0) {
            std::cout << "     [" << i << "] " << std::setw(size_alignment) << colls[i];
          }
        }
        std::cout << '\n';
      }
    }
  } else if constexpr (Goal == goal_t::load_factor) {
    std::cout << "collisions @ " << param << "%\n";
    map_t map(map_t::min_capacity);

    size_t prev_capacity{};
    for (const key_t k : keys) {
      map.upsert(k);
      if (map.capacity() != prev_capacity && map.size() * 100 / map.capacity() == param) {
        std::cout << "size = " << std::setw(size_alignment) << map.size() << " /"
                  << std::setw(size_alignment + 1) << map.capacity() << ": ";
        const auto colls{map.state_collisions()};
        for (size_t i{}; i != colls.size(); ++i) {
          if (colls[i] > 0) {
            std::cout << "     [" << i << "] " << std::setw(size_alignment) << colls[i];
          }
        }
        std::cout << '\n';
        prev_capacity = map.capacity();
      }
    }
  }

  std::cout << '\n';
  return 0;
}

int main() {
  const bool shuffle_keys{true};
  using key_t = int64_t;
  using meta_t = time_t;

  const size_t seed{rd()};

  const goal_t goal{goal_t::load_factor};

  if constexpr (goal == goal_t::interval) {
    constexpr size_t num_keys{1'000'000};
    constexpr size_t interval{50'000};
    count_collisions<map<key_t, meta_t, char, default_kernel32_t, default_seq_t>, goal_t::interval>(
      seed, num_keys, shuffle_keys, interval
    );
    count_collisions<map<key_t, meta_t, char, default_kernel64_t, default_seq_t>, goal_t::interval>(
      seed, num_keys, shuffle_keys, interval
    );
    count_collisions<
      map<key_t, meta_t, char, default_kernel128_t, default_seq_t>, goal_t::interval>(
      seed, num_keys, shuffle_keys, interval
    );
    count_collisions<
      map<key_t, meta_t, char, default_kernel256_t, default_seq_t>, goal_t::interval>(
      seed, num_keys, shuffle_keys, interval
    );
    count_collisions<
      map<key_t, meta_t, char, default_kernel512_t, default_seq_t>, goal_t::interval>(
      seed, num_keys, shuffle_keys, interval
    );
    count_collisions<
      map<key_t, meta_t, char, default_kernel1024_t, default_seq_t>, goal_t::interval>(
      seed, num_keys, shuffle_keys, interval
    );
  } else if constexpr (goal == goal_t::load_factor) {
    constexpr size_t num_keys{60'000'000};
    constexpr size_t load_factor{85};
    count_collisions<
      map<key_t, meta_t, char, default_kernel32_t, default_seq_t>, goal_t::load_factor>(
      seed, num_keys, shuffle_keys, load_factor
    );
    count_collisions<
      map<key_t, meta_t, char, default_kernel64_t, default_seq_t>, goal_t::load_factor>(
      seed, num_keys, shuffle_keys, load_factor
    );
    count_collisions<
      map<key_t, meta_t, char, default_kernel128_t, default_seq_t>, goal_t::load_factor>(
      seed, num_keys, shuffle_keys, load_factor
    );
    count_collisions<
      map<key_t, meta_t, char, default_kernel256_t, default_seq_t>, goal_t::load_factor>(
      seed, num_keys, shuffle_keys, load_factor
    );
    count_collisions<
      map<key_t, meta_t, char, default_kernel512_t, default_seq_t>, goal_t::load_factor>(
      seed, num_keys, shuffle_keys, load_factor
    );
    count_collisions<
      map<key_t, meta_t, char, default_kernel1024_t, default_seq_t>, goal_t::load_factor>(
      seed, num_keys, shuffle_keys, load_factor
    );
  }

  return 0;
}