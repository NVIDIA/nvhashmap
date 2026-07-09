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

#include <iomanip>
#include <iostream>
#include <nvhashmap/map.hpp>

using namespace nvhm;
#include "utils.hpp"

template <typename Map>
void collisions_by_interval(const std::vector<uint_t>& keys, const int_t interval) {
  const int_t num_keys{to_int(keys.size())};
  const int num_align{rendered_length(num_keys)};

  using map_t = Map;
  std::cout << type_to_string<map_t>() << ", collisions @ " << interval << " interval\n";
  map_t map(num_keys);

  for (uint_t k : keys) {
    map.insert(k);
    if (map.size() % interval != 0) continue;

    auto stats{map.state_collisions()};
    std::reverse(stats.begin(), stats.end());
    auto last{std::find_if(stats.begin(), stats.end(), [](int_t s) { return s > 0; })};

    std::cout << "size = " << std::setw(num_align) << map.size() << " /"
              << std::setw(num_align + 1) << map.capacity() << ": ";
    for (auto it{stats.end()}; --it >= last;) {
      int_t i{stats.end() - it + 1};
      std::cout << "    [" << std::setw(2) << i << "] " << std::setw(num_align) << *it;
    }
    std::cout << '\n';
  }
  
  std::cout << '\n';
}

template <typename Map>
void collisions_by_load_factor(const std::vector<uint_t>& keys, const int_t load_perc) {
  const int_t num_keys{to_int(keys.size())};
  const int num_align{rendered_length(num_keys)};

  using map_t = Map;
  std::cout << type_to_string<map_t>() << ", collisions @ " << load_perc << "%\n";
  map_t map(0);

  int_t prev_capacity{};
  for (uint_t k : keys) {
    map.insert(k);
    if (map.capacity() == prev_capacity) continue;
    if (map.size() * 100 / map.capacity() < load_perc) continue;

    auto stats{map.state_collisions()};
    std::reverse(stats.begin(), stats.end());
    auto last{std::find_if(stats.begin(), stats.end(), [](int_t s) { return s > 0; })};

    std::cout << "size = " << std::setw(num_align) << map.size() << " /"
              << std::setw(num_align + 1) << map.capacity() << ": ";

    for (auto it{stats.end()}; --it >= last;) {
      int_t i{stats.end() - it + 1};
      std::cout << "    [" <<  std::setw(2) << i << "] " << std::setw(num_align) << *it;
    }

    std::cout << '\n';
    prev_capacity = map.capacity();
  }

  std::cout << '\n';
}

enum class goal_t : uint_t {
  interval,
  load_factor,
};

int main() {
  using probe_seq_t = default_seq_t;

  constexpr goal_t goal{goal_t::load_factor};
  constexpr flags_t flags{flags_t::none};
  
  constexpr int_t num_keys{[]() {
    if constexpr (goal == goal_t::interval) {
      return 1'000'000;
    } else if constexpr (goal == goal_t::load_factor) {
      return 60'000'000;
    } else {
      static_assert(goal == goal_t::interval || goal == goal_t::load_factor, "Invalid goal");
    }
  }()};
  constexpr bool shuffle_keys{true};
  const std::vector<uint_t> keys{make_keys<uint_t>(num_keys, shuffle_keys, rd())};

  if constexpr (goal == goal_t::interval) {
    constexpr int_t interval{50'000};
    collisions_by_interval<set<uint_t, flags, default_kernel1_t, probe_seq_t>>(keys, interval);
    collisions_by_interval<set<uint_t, flags, default_kernel2_t, probe_seq_t>>(keys, interval);
    collisions_by_interval<set<uint_t, flags, default_kernel4_t, probe_seq_t>>(keys, interval);
    collisions_by_interval<set<uint_t, flags, default_kernel8_t, probe_seq_t>>(keys, interval);
    collisions_by_interval<set<uint_t, flags, default_kernel16_t, probe_seq_t>>(keys, interval);
    collisions_by_interval<set<uint_t, flags, default_kernel32_t, probe_seq_t>>(keys, interval);
    collisions_by_interval<set<uint_t, flags, default_kernel64_t, probe_seq_t>>(keys, interval);
    collisions_by_interval<set<uint_t, flags, default_kernel128_t, probe_seq_t>>(keys, interval);

  } else if constexpr (goal == goal_t::load_factor) {
    constexpr int_t load_perc{85};
    collisions_by_load_factor<set<uint_t, flags, default_kernel1_t, probe_seq_t>>(keys, load_perc);
    collisions_by_load_factor<set<uint_t, flags, default_kernel2_t, probe_seq_t>>(keys, load_perc);
    collisions_by_load_factor<set<uint_t, flags, default_kernel4_t, probe_seq_t>>(keys, load_perc);
    collisions_by_load_factor<set<uint_t, flags, default_kernel8_t, probe_seq_t>>(keys, load_perc);
    collisions_by_load_factor<set<uint_t, flags, default_kernel16_t, probe_seq_t>>(keys, load_perc);
    collisions_by_load_factor<set<uint_t, flags, default_kernel32_t, probe_seq_t>>(keys, load_perc);
    collisions_by_load_factor<set<uint_t, flags, default_kernel64_t, probe_seq_t>>(keys, load_perc);
    collisions_by_load_factor<set<uint_t, flags, default_kernel128_t, probe_seq_t>>(keys, load_perc);
  }

  return 0;
}