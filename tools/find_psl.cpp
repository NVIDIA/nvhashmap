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
#include <map>
#include <nvhashmap/map.hpp>

using namespace nvhm;
#include "utils.hpp"

template <typename Map>
void probe_seq_len(const std::vector<uint_t>& keys, const int_t interval) {
  using map_t = Map;
  
  const int_t num_keys{to_int(keys.size())};
  const int num_align{rendered_length(num_keys)};

  std::cout << type_to_string<map_t>() << ", probe sequence lengths @ " << interval << " interval (num_keys = " << num_keys << ")\n";
  map_t map(num_keys);

  for (const uint_t k : keys) {
    map.insert(k);
    if (map.size() % interval != 0) continue;

    // TODO: Is it better to test all keys, or just the already inserted ones?
    std::map<psl_t, int_t> psl_counts;
    for (const uint_t k : keys) {
      auto [pos, seq]{map.find_first(k)};
      if (pos == npos) continue;
      ++psl_counts[seq.psl()];
    }

    std::cout << "size = " << std::setw(num_align) << map.size() << " /"
              << std::setw(num_align + 1) << map.capacity() << " ~ ";
    for (const auto& [psl, count] : psl_counts) {
      std::cout << "   [" << std::setw(3) << psl << "] " << std::setw(num_align) << count;
    }
    std::cout << '\n';
  }

  std::cout << '\n';
}

int main() {
  using probe_seq_t = default_seq_t;
  // using probe_seq_t = linear_seq<>;
  // using probe_seq_t = quadratic_seq<32>;

  constexpr flags_t flags{flags_t::none};

  constexpr int_t num_keys{940'000};
  constexpr bool shuffle_keys{true};
  const std::vector<uint_t> keys{make_keys<uint_t>(num_keys, shuffle_keys, rd())};

  constexpr int_t interval{50'000};
  probe_seq_len<map<uint_t, void, flags, default_kernel1_t, probe_seq_t>>(keys, interval);
  probe_seq_len<map<uint_t, void, flags, default_kernel2_t, probe_seq_t>>(keys, interval);
  probe_seq_len<map<uint_t, void, flags, default_kernel4_t, probe_seq_t>>(keys, interval);
  probe_seq_len<map<uint_t, void, flags, default_kernel8_t, probe_seq_t>>(keys, interval);
  probe_seq_len<map<uint_t, void, flags, default_kernel16_t, probe_seq_t>>(keys, interval);
  probe_seq_len<map<uint_t, void, flags, default_kernel32_t, probe_seq_t>>(keys, interval);
  probe_seq_len<map<uint_t, void, flags, default_kernel64_t, probe_seq_t>>(keys, interval);
  probe_seq_len<map<uint_t, void, flags, default_kernel128_t, probe_seq_t>>(keys, interval);

  return 0;
}