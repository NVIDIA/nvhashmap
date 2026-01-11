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

#include <chrono>
#include <cxxabi.h>
#include <iomanip>
#include <iostream>
#include <map>
#include <nvhashmap/map.hpp>
#include <nvhashmap/experimental/shim.hpp>
#include <nvhashmap/experimental/prefetch.hpp>
#include <random>
#include <thread>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#include <phmap.h>
#pragma GCC diagnostic pop

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#include <absl/container/flat_hash_map.h>
#pragma GCC diagnostic pop

using namespace nvhm;
using namespace nvhm::experimental;

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

constexpr int32_t key_scale{1};

#define INSERT_VALUE_AND_META_NEXT_(_k_, _p_)                            \
  do {                                                                   \
    const key_t k{_k_};                                                  \
    const prefetch_t p{_p_};                                             \
    const write_pos_t pos{map.upsert(k, p)};                             \
    if (value_size) {                                                    \
      if constexpr (FillInsteadOfCopy) {                                 \
        char* const __restrict v{map.raw_values_at(pos)};                \
        std::fill_n(v, value_size, static_cast<char>(~(k / key_scale))); \
      } else {                                                           \
        map.set_raw_values_at(pos, vbuffer.data(), value_size);          \
      }                                                                  \
    }                                                                    \
    if constexpr (map_t::has_values) {                                   \
      map.value_at(pos) = -(k / key_scale);                              \
    }                                                                    \
  } while (false)

#define CHECK_POS_VALUE_AND_META_(_i_, _k_)                                                       \
  do {                                                                                            \
    if (is_valid_key(_i_)) {                                                                      \
      if (pos == npos) {                                                                          \
        std::cerr << "Find error! i = " << (_i_) << ", k = " << (_k_) << " (" << __LINE__         \
                  << ")\n";                                                                       \
        worker_result[w] = false;                                                                 \
        return;                                                                                   \
      }                                                                                           \
      if (value_size) {                                                                           \
        if constexpr (FillInsteadOfCopy) {                                                        \
          const char* const __restrict v{map.raw_values_at(pos)};                                 \
          if (static_cast<size_t>(std::count(v, v + value_size, static_cast<char>(~(_k_)))) !=    \
              value_size) {                                                                       \
            std::cerr << "Value error! i = " << (_i_) << ", k = " << (_k_) << " (" << __LINE__    \
                      << ")\n";                                                                   \
            worker_result[w] = false;                                                             \
            return;                                                                               \
          }                                                                                       \
        } else {                                                                                  \
          map.get_raw_values_at(pos, vbuffer.data(), value_size);                                 \
        }                                                                                         \
      }                                                                                           \
      if constexpr (TestMeta && map_t::has_values) {                                              \
        if (map.value_at(pos) != -(_k_)) {                                                        \
          std::cerr << "Meta error! i = " << (_i_) << ", k = " << (_k_) << " (" << __LINE__       \
                    << ")\n";                                                                     \
          worker_result[w] = false;                                                               \
          return;                                                                                 \
        }                                                                                         \
      }                                                                                           \
    } else if (pos != npos) {                                                                     \
      std::cerr << "Miss error! i = " << (_i_) << ", k = " << (_k_) << " (" << __LINE__ << ")\n"; \
      worker_result[w] = false;                                                                   \
      return;                                                                                     \
    }                                                                                             \
  } while (false)

#define CHECK_POS_VALUE_AND_META_STD_(_i_, _k_)                                                   \
  do {                                                                                            \
    if (is_valid_key(_i_)) {                                                                      \
      if (it == map_end) {                                                                        \
        std::cerr << "Find error! i = " << (_i_) << ", k = " << (_k_) << " (" << __LINE__         \
                  << ")\n";                                                                       \
        worker_result[w] = false;                                                                 \
        return;                                                                                   \
      }                                                                                           \
      const char* const __restrict vm{it->second};                                                \
      if (value_size) {                                                                           \
        if constexpr (FillInsteadOfCopy) {                                                        \
          if (static_cast<size_t>(std::count(vm, vm + value_size, static_cast<char>(~(_k_)))) !=  \
              value_size) {                                                                       \
            std::cerr << "Value error! i = " << (_i_) << ", k = " << (_k_) << " (" << __LINE__    \
                      << ")\n";                                                                   \
            worker_result[w] = false;                                                             \
            return;                                                                               \
          }                                                                                       \
        } else {                                                                                  \
          fast_copy(vbuffer.data(), vm, value_size);                                              \
        }                                                                                         \
      }                                                                                           \
      if constexpr (TestMeta && meta_size) {                                                      \
        if (*reinterpret_cast<const meta_t*>(&vm[value_size]) != -(_k_)) {                        \
          std::cerr << "Meta error! i = " << (_i_) << ", k = " << (_k_) << " (" << __LINE__       \
                    << ")\n";                                                                     \
          worker_result[w] = false;                                                               \
          return;                                                                                 \
        }                                                                                         \
      }                                                                                           \
    } else if (it != map_end) {                                                                   \
      std::cerr << "Miss error! i = " << (_i_) << ", k = " << (_k_) << " (" << __LINE__ << ")\n"; \
      worker_result[w] = false;                                                                   \
      return;                                                                                     \
    }                                                                                             \
  } while (false)

template <typename Map, bool FillInsteadOfCopy, bool TestMeta>
bool __attribute__((noinline)) benchmark_10M(
  const size_t num_trials, const size_t num_keys, const bool shuffle_keys, const size_t value_size,
  const size_t num_workers, const size_t max_insert_prefetch, const size_t max_find_prefetch
) {
  using map_t = Map;
  const std::string map_type{type_to_string<map_t>()};
  using key_t = typename map_t::key_type;
  using prefetch_t = typename map_t::prefetch_type;
  using read_pos_t = typename map_t::read_pos_type;
  using write_pos_t = typename map_t::write_pos_type;

  std::vector<key_t> keys(num_keys);
  for (size_t i{}; i != keys.size(); ++i) {
    keys[i] = static_cast<key_t>(i);
  }
  std::random_device rd;
  std::mt19937 g(rd());

  std::vector<char> vbuffer(value_size);
  for (size_t i{}; i != vbuffer.size(); ++i) {
    vbuffer[i] = static_cast<char>(i % 100);
  }
  map_t map(map_t::default_capacity, value_size);

  constexpr size_t max_prefetch_cache_size{16};
  if (max_insert_prefetch >= max_prefetch_cache_size) {
    std::cout << "`max_insert_prefetch` is out of bounds!\n";
    return false;
  }
  if (max_find_prefetch >= max_prefetch_cache_size) {
    std::cout << "`max_find_prefetch` is out of bounds!\n";
    return false;
  }

  // Insert
  std::cout << map_type << ", " << std::setw(15) << "Insert (no prefetch):  ";
  std::cout.flush();
  for (size_t trial{}; trial != num_trials; ++trial) {
    if (shuffle_keys) {
      std::shuffle(keys.begin(), keys.end(), g);
    }
    map.clear();
    const auto begin{std::chrono::high_resolution_clock::now()};
    for (const key_t k : keys) {
      const write_pos_t pos{map.upsert(k * key_scale)};
      if (value_size) {
        if constexpr (FillInsteadOfCopy) {
          char* const __restrict v{map.raw_values_at(pos)};
          std::fill_n(v, value_size, static_cast<char>(~k));
        } else {
          map.set_raw_values_at(pos, vbuffer.data(), value_size);
        }
      }
      if constexpr (map_t::has_values) {
        map.value_at(pos) = -k;
      }
    }
    const auto end{std::chrono::high_resolution_clock::now()};

    std::cout << (trial ? ", " : "") << std::setw(5)
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms";
    std::cout.flush();
  }
  std::cout << ", size: " << map.size() << " / " << map.capacity() << '\n';

  if (keys.size() >= max_insert_prefetch) {
    for (size_t insert_prefetch{1}; insert_prefetch < max_insert_prefetch; ++insert_prefetch) {
      switch (insert_prefetch) {
        case 1: {
          std::cout << map_type << ", " << std::setw(15) << "Insert (prefetch = dq  1): ";
          std::cout.flush();
          for (size_t trial{}; trial != num_trials; ++trial) {
            if (shuffle_keys) {
              std::shuffle(keys.begin(), keys.end(), g);
            }
            map.clear();
            const auto begin{std::chrono::high_resolution_clock::now()};

            auto it{keys.begin()};
            key_t k0{*it++ * key_scale};

            prefetch_t h0{map.write_prefetch(k0, true)};

            while (NVHM_LIKELY_(it != keys.end())) {
              const key_t k1{*it++ * key_scale};
              const prefetch_t h1{map.write_prefetch(k1, true)};
              INSERT_VALUE_AND_META_NEXT_(k0, h0);
              shift(k0, k1);
              shift(h0, h1);
            }

            INSERT_VALUE_AND_META_NEXT_(k0, h0);
            const auto end{std::chrono::high_resolution_clock::now()};

            std::cout << (trial ? ", " : "") << std::setw(5)
                      << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
                      << "ms";
            std::cout.flush();
          }
          std::cout << ", size: " << map.size() << " / " << map.capacity() << '\n';

          std::cout << map_type << ", " << std::setw(15) << "Insert (prefetch = sq  1): ";
          std::cout.flush();
          for (size_t trial{}; trial != num_trials; ++trial) {
            if (shuffle_keys) {
              std::shuffle(keys.begin(), keys.end(), g);
            }
            map.clear();
            const auto begin{std::chrono::high_resolution_clock::now()};
            shift_prefetch_queue<key_t, prefetch_t, 1> queue;

            auto it{keys.begin()};
            for (size_t i{}; i != queue.capacity; ++i) {
              queue.prepare_upsert(map, *it++ * key_scale, true);
            }

            while (NVHM_LIKELY_(it != keys.end())) {
              const auto [k0, h0]{queue.prepare_upsert(map, *it++ * key_scale, true)};
              INSERT_VALUE_AND_META_NEXT_(k0, h0);
            }

            for (size_t i{}; i != queue.capacity; ++i) {
              const auto [k0, h0]{queue.pop()};
              INSERT_VALUE_AND_META_NEXT_(k0, h0);
            }
            const auto end{std::chrono::high_resolution_clock::now()};

            std::cout << (trial ? ", " : "") << std::setw(5)
                      << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
                      << "ms";
            std::cout.flush();
          }
          std::cout << ", size: " << map.size() << " / " << map.capacity() << '\n';
        } break;
        case 2: {
          std::cout << map_type << ", " << std::setw(15) << "Insert (prefetch = dq  2): ";
          std::cout.flush();
          for (size_t trial{}; trial != num_trials; ++trial) {
            if (shuffle_keys) {
              std::shuffle(keys.begin(), keys.end(), g);
            }
            map.clear();
            const auto begin{std::chrono::high_resolution_clock::now()};

            auto it{keys.begin()};
            key_t k0{*it++ * key_scale};
            key_t k1{*it++ * key_scale};

            prefetch_t h0{map.write_prefetch(k0, true)};
            prefetch_t h1{map.write_prefetch(k1, true)};

            while (NVHM_LIKELY_(it != keys.end())) {
              const key_t k2{*it++ * key_scale};
              const prefetch_t h2{map.write_prefetch(k2, true)};
              INSERT_VALUE_AND_META_NEXT_(k0, h0);
              shift(k0, k1, k2);
              shift(h0, h1, h2);
            }

            INSERT_VALUE_AND_META_NEXT_(k0, h0);
            INSERT_VALUE_AND_META_NEXT_(k1, h1);
            const auto end{std::chrono::high_resolution_clock::now()};

            std::cout << (trial ? ", " : "") << std::setw(5)
                      << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
                      << "ms";
            std::cout.flush();
          }
          std::cout << ", size: " << map.size() << " / " << map.capacity() << '\n';

          std::cout << map_type << ", " << std::setw(15) << "Insert (prefetch = sq  2): ";
          std::cout.flush();
          for (size_t trial{}; trial != num_trials; ++trial) {
            if (shuffle_keys) {
              std::shuffle(keys.begin(), keys.end(), g);
            }
            map.clear();
            const auto begin{std::chrono::high_resolution_clock::now()};
            shift_prefetch_queue<key_t, prefetch_t, 2> queue;

            auto it{keys.begin()};
            for (size_t i{}; i != queue.capacity; ++i) {
              queue.prepare_upsert(map, *it++ * key_scale, true);
            }

            while (NVHM_LIKELY_(it != keys.end())) {
              const auto [k0, h0]{queue.prepare_upsert(map, *it++ * key_scale, true)};
              INSERT_VALUE_AND_META_NEXT_(k0, h0);
            }

            for (size_t i{}; i != queue.capacity; ++i) {
              const auto [k0, h0]{queue.pop()};
              INSERT_VALUE_AND_META_NEXT_(k0, h0);
            }
            const auto end{std::chrono::high_resolution_clock::now()};

            std::cout << (trial ? ", " : "") << std::setw(5)
                      << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
                      << "ms";
            std::cout.flush();
          }
          std::cout << ", size: " << map.size() << " / " << map.capacity() << '\n';
        } break;
        case 3: {
          std::cout << map_type << ", " << std::setw(15) << "Insert (prefetch = dq  3): ";
          std::cout.flush();
          for (size_t trial{}; trial != num_trials; ++trial) {
            if (shuffle_keys) {
              std::shuffle(keys.begin(), keys.end(), g);
            }
            map.clear();
            const auto begin{std::chrono::high_resolution_clock::now()};

            auto it{keys.begin()};
            key_t k0{*it++ * key_scale};
            key_t k1{*it++ * key_scale};
            key_t k2{*it++ * key_scale};

            prefetch_t h0{map.write_prefetch(k0, true)};
            prefetch_t h1{map.write_prefetch(k1, true)};
            prefetch_t h2{map.write_prefetch(k2, true)};

            while (NVHM_LIKELY_(it != keys.end())) {
              const key_t k3{*it++ * key_scale};
              const prefetch_t h3{map.write_prefetch(k3, true)};
              INSERT_VALUE_AND_META_NEXT_(k0, h0);
              shift(k0, k1, k2, k3);
              shift(h0, h1, h2, h3);
            }

            INSERT_VALUE_AND_META_NEXT_(k0, h0);
            INSERT_VALUE_AND_META_NEXT_(k1, h1);
            INSERT_VALUE_AND_META_NEXT_(k2, h2);
            const auto end{std::chrono::high_resolution_clock::now()};

            std::cout << (trial ? ", " : "") << std::setw(5)
                      << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
                      << "ms";
            std::cout.flush();
          }
          std::cout << ", size: " << map.size() << " / " << map.capacity() << '\n';

          std::cout << map_type << ", " << std::setw(15) << "Insert (prefetch = sq  3): ";
          std::cout.flush();
          for (size_t trial{}; trial != num_trials; ++trial) {
            if (shuffle_keys) {
              std::shuffle(keys.begin(), keys.end(), g);
            }
            map.clear();
            const auto begin{std::chrono::high_resolution_clock::now()};
            shift_prefetch_queue<key_t, prefetch_t, 3> queue;

            auto it{keys.begin()};
            for (size_t i{}; i != queue.capacity; ++i) {
              queue.prepare_upsert(map, *it++ * key_scale, true);
            }

            while (NVHM_LIKELY_(it != keys.end())) {
              const auto [k0, h0]{queue.prepare_upsert(map, *it++ * key_scale, true)};
              INSERT_VALUE_AND_META_NEXT_(k0, h0);
            }

            for (size_t i{}; i != queue.capacity; ++i) {
              const auto [k0, h0]{queue.pop()};
              INSERT_VALUE_AND_META_NEXT_(k0, h0);
            }
            const auto end{std::chrono::high_resolution_clock::now()};

            std::cout << (trial ? ", " : "") << std::setw(5)
                      << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
                      << "ms";
            std::cout.flush();
          }
          std::cout << ", size: " << map.size() << " / " << map.capacity() << '\n';
        } break;
        case 4: {
          std::cout << map_type << ", " << std::setw(15) << "Insert (prefetch = dq  4): ";
          std::cout.flush();
          for (size_t trial{}; trial != num_trials; ++trial) {
            if (shuffle_keys) {
              std::shuffle(keys.begin(), keys.end(), g);
            }
            map.clear();
            const auto begin{std::chrono::high_resolution_clock::now()};

            auto it{keys.begin()};
            key_t k0{*it++ * key_scale};
            key_t k1{*it++ * key_scale};
            key_t k2{*it++ * key_scale};
            key_t k3{*it++ * key_scale};

            prefetch_t h0{map.write_prefetch(k0, true)};
            prefetch_t h1{map.write_prefetch(k1, true)};
            prefetch_t h2{map.write_prefetch(k2, true)};
            prefetch_t h3{map.write_prefetch(k3, true)};

            while (NVHM_LIKELY_(it != keys.end())) {
              const key_t k4{*it++ * key_scale};
              const prefetch_t h4{map.write_prefetch(k4, true)};
              INSERT_VALUE_AND_META_NEXT_(k0, h0);
              shift(k0, k1, k2, k3, k4);
              shift(h0, h1, h2, h3, h4);
            }

            INSERT_VALUE_AND_META_NEXT_(k0, h0);
            INSERT_VALUE_AND_META_NEXT_(k1, h1);
            INSERT_VALUE_AND_META_NEXT_(k2, h2);
            INSERT_VALUE_AND_META_NEXT_(k3, h3);
            const auto end{std::chrono::high_resolution_clock::now()};

            std::cout << (trial ? ", " : "") << std::setw(5)
                      << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
                      << "ms";
            std::cout.flush();
          }
          std::cout << ", size: " << map.size() << " / " << map.capacity() << '\n';

          std::cout << map_type << ", " << std::setw(15) << "Insert (prefetch = sq  4): ";
          std::cout.flush();
          for (size_t trial{}; trial != num_trials; ++trial) {
            if (shuffle_keys) {
              std::shuffle(keys.begin(), keys.end(), g);
            }
            map.clear();
            const auto begin{std::chrono::high_resolution_clock::now()};
            shift_prefetch_queue<key_t, prefetch_t, 4> queue;

            auto it{keys.begin()};
            for (size_t i{}; i != queue.capacity; ++i) {
              queue.prepare_upsert(map, *it++ * key_scale, true);
            }

            while (NVHM_LIKELY_(it != keys.end())) {
              const auto [k0, h0]{queue.prepare_upsert(map, *it++ * key_scale, true)};
              INSERT_VALUE_AND_META_NEXT_(k0, h0);
            }

            for (size_t i{}; i != queue.capacity; ++i) {
              const auto [k0, h0]{queue.pop()};
              INSERT_VALUE_AND_META_NEXT_(k0, h0);
            }
            const auto end{std::chrono::high_resolution_clock::now()};

            std::cout << (trial ? ", " : "") << std::setw(5)
                      << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
                      << "ms";
            std::cout.flush();
          }
          std::cout << ", size: " << map.size() << " / " << map.capacity() << '\n';
        } break;
        case 5: {
          std::cout << map_type << ", " << std::setw(15) << "Insert (prefetch = dq  5): ";
          std::cout.flush();
          for (size_t trial{}; trial != num_trials; ++trial) {
            if (shuffle_keys) {
              std::shuffle(keys.begin(), keys.end(), g);
            }
            map.clear();
            const auto begin{std::chrono::high_resolution_clock::now()};

            auto it{keys.begin()};
            key_t k0{*it++ * key_scale};
            key_t k1{*it++ * key_scale};
            key_t k2{*it++ * key_scale};
            key_t k3{*it++ * key_scale};
            key_t k4{*it++ * key_scale};

            prefetch_t h0{map.write_prefetch(k0, true)};
            prefetch_t h1{map.write_prefetch(k1, true)};
            prefetch_t h2{map.write_prefetch(k2, true)};
            prefetch_t h3{map.write_prefetch(k3, true)};
            prefetch_t h4{map.write_prefetch(k4, true)};

            while (NVHM_LIKELY_(it != keys.end())) {
              const key_t k5{*it++ * key_scale};
              const prefetch_t h5{map.write_prefetch(k5, true)};
              INSERT_VALUE_AND_META_NEXT_(k0, h0);
              shift(k0, k1, k2, k3, k4, k5);
              shift(h0, h1, h2, h3, h4, h5);
            }

            INSERT_VALUE_AND_META_NEXT_(k0, h0);
            INSERT_VALUE_AND_META_NEXT_(k1, h1);
            INSERT_VALUE_AND_META_NEXT_(k2, h2);
            INSERT_VALUE_AND_META_NEXT_(k3, h3);
            INSERT_VALUE_AND_META_NEXT_(k4, h4);
            const auto end{std::chrono::high_resolution_clock::now()};

            std::cout << (trial ? ", " : "") << std::setw(5)
                      << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
                      << "ms";
            std::cout.flush();
          }
          std::cout << ", size: " << map.size() << " / " << map.capacity() << '\n';

          std::cout << map_type << ", " << std::setw(15) << "Insert (prefetch = sq  5): ";
          std::cout.flush();
          for (size_t trial{}; trial != num_trials; ++trial) {
            if (shuffle_keys) {
              std::shuffle(keys.begin(), keys.end(), g);
            }
            map.clear();
            const auto begin{std::chrono::high_resolution_clock::now()};
            shift_prefetch_queue<key_t, prefetch_t, 5> queue;

            auto it{keys.begin()};
            for (size_t i{}; i != queue.capacity; ++i) {
              queue.prepare_upsert(map, *it++ * key_scale, true);
            }

            while (NVHM_LIKELY_(it != keys.end())) {
              const auto [k0, h0]{queue.prepare_upsert(map, *it++ * key_scale, true)};
              INSERT_VALUE_AND_META_NEXT_(k0, h0);
            }

            for (size_t i{}; i != queue.capacity; ++i) {
              const auto [k0, h0]{queue.pop()};
              INSERT_VALUE_AND_META_NEXT_(k0, h0);
            }
            const auto end{std::chrono::high_resolution_clock::now()};

            std::cout << (trial ? ", " : "") << std::setw(5)
                      << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
                      << "ms";
            std::cout.flush();
          }
          std::cout << ", size: " << map.size() << " / " << map.capacity() << '\n';
        } break;
        case 6: {
          std::cout << map_type << ", " << std::setw(15) << "Insert (prefetch = dq  6): ";
          std::cout.flush();
          for (size_t trial{}; trial != num_trials; ++trial) {
            if (shuffle_keys) {
              std::shuffle(keys.begin(), keys.end(), g);
            }
            map.clear();
            const auto begin{std::chrono::high_resolution_clock::now()};

            auto it{keys.begin()};
            key_t k0{*it++ * key_scale};
            key_t k1{*it++ * key_scale};
            key_t k2{*it++ * key_scale};
            key_t k3{*it++ * key_scale};
            key_t k4{*it++ * key_scale};
            key_t k5{*it++ * key_scale};

            prefetch_t h0{map.write_prefetch(k0, true)};
            prefetch_t h1{map.write_prefetch(k1, true)};
            prefetch_t h2{map.write_prefetch(k2, true)};
            prefetch_t h3{map.write_prefetch(k3, true)};
            prefetch_t h4{map.write_prefetch(k4, true)};
            prefetch_t h5{map.write_prefetch(k5, true)};

            while (NVHM_LIKELY_(it != keys.end())) {
              const key_t k6{*it++ * key_scale};
              const prefetch_t h6{map.write_prefetch(k6, true)};
              INSERT_VALUE_AND_META_NEXT_(k0, h0);
              shift(k0, k1, k2, k3, k4, k5, k6);
              shift(h0, h1, h2, h3, h4, h5, h6);
            }

            INSERT_VALUE_AND_META_NEXT_(k0, h0);
            INSERT_VALUE_AND_META_NEXT_(k1, h1);
            INSERT_VALUE_AND_META_NEXT_(k2, h2);
            INSERT_VALUE_AND_META_NEXT_(k3, h3);
            INSERT_VALUE_AND_META_NEXT_(k4, h4);
            INSERT_VALUE_AND_META_NEXT_(k5, h5);
            const auto end{std::chrono::high_resolution_clock::now()};

            std::cout << (trial ? ", " : "") << std::setw(5)
                      << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
                      << "ms";
            std::cout.flush();
          }
          std::cout << ", size: " << map.size() << " / " << map.capacity() << '\n';

          std::cout << map_type << ", " << std::setw(15) << "Insert (prefetch = sq  6): ";
          std::cout.flush();
          for (size_t trial{}; trial != num_trials; ++trial) {
            if (shuffle_keys) {
              std::shuffle(keys.begin(), keys.end(), g);
            }
            map.clear();
            const auto begin{std::chrono::high_resolution_clock::now()};
            shift_prefetch_queue<key_t, prefetch_t, 6> queue;

            auto it{keys.begin()};
            for (size_t i{}; i != queue.capacity; ++i) {
              queue.prepare_upsert(map, *it++ * key_scale, true);
            }

            while (NVHM_LIKELY_(it != keys.end())) {
              const auto [k0, h0]{queue.prepare_upsert(map, *it++ * key_scale, true)};
              INSERT_VALUE_AND_META_NEXT_(k0, h0);
            }

            for (size_t i{}; i != queue.capacity; ++i) {
              const auto [k0, h0]{queue.pop()};
              INSERT_VALUE_AND_META_NEXT_(k0, h0);
            }
            const auto end{std::chrono::high_resolution_clock::now()};

            std::cout << (trial ? ", " : "") << std::setw(5)
                      << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
                      << "ms";
            std::cout.flush();
          }
          std::cout << ", size: " << map.size() << " / " << map.capacity() << '\n';
        } break;
        case 7: {
          std::cout << map_type << ", " << std::setw(15) << "Insert (prefetch = dq  7): ";
          std::cout.flush();
          for (size_t trial{}; trial != num_trials; ++trial) {
            if (shuffle_keys) {
              std::shuffle(keys.begin(), keys.end(), g);
            }
            map.clear();
            const auto begin{std::chrono::high_resolution_clock::now()};

            auto it{keys.begin()};
            key_t k0{*it++ * key_scale};
            key_t k1{*it++ * key_scale};
            key_t k2{*it++ * key_scale};
            key_t k3{*it++ * key_scale};
            key_t k4{*it++ * key_scale};
            key_t k5{*it++ * key_scale};
            key_t k6{*it++ * key_scale};

            prefetch_t h0{map.write_prefetch(k0, true)};
            prefetch_t h1{map.write_prefetch(k1, true)};
            prefetch_t h2{map.write_prefetch(k2, true)};
            prefetch_t h3{map.write_prefetch(k3, true)};
            prefetch_t h4{map.write_prefetch(k4, true)};
            prefetch_t h5{map.write_prefetch(k5, true)};
            prefetch_t h6{map.write_prefetch(k6, true)};

            while (NVHM_LIKELY_(it != keys.end())) {
              const key_t k7{*it++ * key_scale};
              const prefetch_t h7{map.write_prefetch(k7, true)};
              INSERT_VALUE_AND_META_NEXT_(k0, h0);
              shift(k0, k1, k2, k3, k4, k5, k6, k7);
              shift(h0, h1, h2, h3, h4, h5, h6, h7);
            }

            INSERT_VALUE_AND_META_NEXT_(k0, h0);
            INSERT_VALUE_AND_META_NEXT_(k1, h1);
            INSERT_VALUE_AND_META_NEXT_(k2, h2);
            INSERT_VALUE_AND_META_NEXT_(k3, h3);
            INSERT_VALUE_AND_META_NEXT_(k4, h4);
            INSERT_VALUE_AND_META_NEXT_(k5, h5);
            INSERT_VALUE_AND_META_NEXT_(k6, h6);
            const auto end{std::chrono::high_resolution_clock::now()};

            std::cout << (trial ? ", " : "") << std::setw(5)
                      << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
                      << "ms";
            std::cout.flush();
          }
          std::cout << ", size: " << map.size() << " / " << map.capacity() << '\n';

          std::cout << map_type << ", " << std::setw(15) << "Insert (prefetch = sq  7): ";
          std::cout.flush();
          for (size_t trial{}; trial != num_trials; ++trial) {
            if (shuffle_keys) {
              std::shuffle(keys.begin(), keys.end(), g);
            }
            map.clear();
            const auto begin{std::chrono::high_resolution_clock::now()};
            shift_prefetch_queue<key_t, prefetch_t, 6> queue;

            auto it{keys.begin()};
            for (size_t i{}; i != queue.capacity; ++i) {
              queue.prepare_upsert(map, *it++ * key_scale, true);
            }

            while (NVHM_LIKELY_(it != keys.end())) {
              const auto [k0, h0]{queue.prepare_upsert(map, *it++ * key_scale, true)};
              INSERT_VALUE_AND_META_NEXT_(k0, h0);
            }

            for (size_t i{}; i != queue.capacity; ++i) {
              const auto [k0, h0]{queue.pop()};
              INSERT_VALUE_AND_META_NEXT_(k0, h0);
            }
            const auto end{std::chrono::high_resolution_clock::now()};

            std::cout << (trial ? ", " : "") << std::setw(5)
                      << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
                      << "ms";
            std::cout.flush();
          }
          std::cout << ", size: " << map.size() << " / " << map.capacity() << '\n';
        } break;
      }

      std::cout << map_type << ", " << std::setw(15) << "Insert (prefetch = rq " << std::setw(2)
                << insert_prefetch << "): ";
      std::cout.flush();

      for (size_t trial{}; trial != num_trials; ++trial) {
        if (shuffle_keys) {
          std::shuffle(keys.begin(), keys.end(), g);
        }
        map.clear();
        const auto begin{std::chrono::high_resolution_clock::now()};
        ring_prefetch_queue<key_t, prefetch_t, max_prefetch_cache_size> queue;

        auto it{keys.begin()};
        while (queue.size() != insert_prefetch) {
          queue.prepare_upsert(map, *it++ * key_scale, true);
        }

        while (NVHM_LIKELY_(it != keys.end())) {
          const auto [k0, h0]{queue.prepare_upsert(map, *it++ * key_scale, true).pop()};
          INSERT_VALUE_AND_META_NEXT_(k0, h0);
        }

        while (!queue.empty()) {
          const auto [k0, h0]{queue.pop()};
          INSERT_VALUE_AND_META_NEXT_(k0, h0);
        }
        const auto end{std::chrono::high_resolution_clock::now()};

        std::cout << (trial ? ", " : "") << std::setw(5)
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
                  << "ms";
        std::cout.flush();
      }
      std::cout << ", size: " << map.size() << " / " << map.capacity() << '\n';
    }
  }

  // Find.
  std::vector<std::thread> workers(num_workers);
  std::atomic_size_t worker_ready;
  std::vector<bool> worker_result(num_workers, true);
  std::vector<std::chrono::milliseconds> worker_times(num_workers);
  const size_t num_keys_per_worker{(keys.size() + num_workers - 1) / num_workers};

  const auto launch_and_time_workers{
    [&](const size_t trial, const std::function<void(size_t, size_t, size_t)>& worker_fn) -> bool {
    worker_ready.store(0, std::memory_order_relaxed);
    for (size_t w{}; w != num_workers; ++w) {
      workers[w] = std::thread([&, w]() {
        const size_t i0{std::min(w * num_keys_per_worker, keys.size())};
        const size_t i1{std::min(i0 + num_keys_per_worker, keys.size())};

        worker_ready.fetch_add(1, std::memory_order_relaxed);
        while (worker_ready.load(std::memory_order_relaxed) != num_workers) {
          // Spin until all threads are ready.
        }

        const auto begin{std::chrono::high_resolution_clock::now()};
        worker_fn(w, i0, i1);
        const auto end{std::chrono::high_resolution_clock::now()};
        worker_times[w] = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
      });
    }

    std::chrono::milliseconds sum_time{};
    for (size_t w{}; w != num_workers; ++w) {
      workers[w].join();
      if (!worker_result[w]) {
        std::cerr << "Worker #" << w << " failed!";
        return false;
      }
      sum_time += worker_times[w];
    }
    sum_time /= static_cast<int64_t>(num_workers);
    std::cout << (trial ? ", " : "") << std::setw(5) << sum_time.count() << "ms";
    std::cout.flush();
    return true;
  }
  };

  {
    constexpr bool optimisitic_prefetch{true};
    const auto is_valid_key{[](const size_t) { return true; }};

    std::cout << map_type << ", " << std::setw(15) << "Find 100% hit:         ";
    std::cout.flush();
    for (size_t trial{}; trial != num_trials; ++trial) {
      if (shuffle_keys) {
        std::shuffle(keys.begin(), keys.end(), g);
      }

      const auto worker_fn{[&](const size_t w, size_t i, const size_t i1) -> void {
        for (; NVHM_LIKELY_(i != i1); ++i) {
          const key_t k{keys[i] * key_scale};
          const read_pos_t pos{map.lookup(k)};
          CHECK_POS_VALUE_AND_META_(i, k);
        }
      }};
      if (!launch_and_time_workers(trial, worker_fn)) return false;
    }
    std::cout << " (no prefetch)\n";

    if (num_keys_per_worker >= max_find_prefetch) {
      for (size_t find_prefetch{1}; find_prefetch < max_find_prefetch; ++find_prefetch) {
        switch (find_prefetch) {
          case 1: {
            std::cout << map_type << ", " << std::setw(15) << "Find 100% hit (sq  1): ";
            std::cout.flush();
            for (size_t trial{}; trial != num_trials; ++trial) {
              if (shuffle_keys) {
                std::shuffle(keys.begin(), keys.end(), g);
              }

              const auto worker_fn{[&](const size_t w, size_t i, const size_t i1) -> void {
#if 1
                shift_prefetch_queue<key_t, prefetch_t, 1> queue;

                for (size_t j{}; j != queue.capacity; ++j) {
                  queue.prepare_lookup(map, keys[i++] * key_scale, optimisitic_prefetch);
                }

                for (; NVHM_LIKELY_(i != i1); ++i) {
                  const auto [k0, h0]{
                    queue.prepare_lookup(map, keys[i] * key_scale, optimisitic_prefetch)
                  };
                  const read_pos_t pos{map.lookup(k0, h0)};
                  CHECK_POS_VALUE_AND_META_(i - queue.capacity, k0);
                }

                i -= queue.capacity;
                for (size_t j{}; j != queue.capacity; ++j) {
                  const auto [k0, h0]{queue.pop()};
                  const read_pos_t pos{map.lookup(k0, h0)};
                  CHECK_POS_VALUE_AND_META_(i, k0);
                  ++i;
                }
#else
                key_t k0{keys[i++] * key_scale};

                prefetch_t h0{map.read_prefetch(k0, optimisitic_prefetch)};

                for (; NVHM_LIKELY_(i != i1); ++i) {
                  const key_t k1{keys[i] * key_scale};
                  const prefetch_t h1{map.read_prefetch(k1)};
                  const read_pos_t pos{map.lookup(k0, h0)};
                  CHECK_POS_VALUE_AND_META_(i - 1, k0);
                  shift(k0, k1);
                  shift(h0, h1);
                }

                const read_pos_t pos{map.lookup(k0, h0)};
                CHECK_POS_VALUE_AND_META_(i, k0);
                ++i;
#endif
              }};
              if (!launch_and_time_workers(trial, worker_fn)) return false;
            }
            std::cout << " (prefetch = 1)\n";
          } break;
          case 2: {
            std::cout << map_type << ", " << std::setw(15) << "Find 100% hit (sq  2): ";
            std::cout.flush();
            for (size_t trial{}; trial != num_trials; ++trial) {
              if (shuffle_keys) {
                std::shuffle(keys.begin(), keys.end(), g);
              }

              const auto worker_fn{[&](const size_t w, size_t i, const size_t i1) -> void {
#if 1
                shift_prefetch_queue<key_t, prefetch_t, 2> queue;

                for (size_t j{}; j != queue.capacity; ++j) {
                  queue.prepare_lookup(map, keys[i++] * key_scale, optimisitic_prefetch);
                }

                for (; NVHM_LIKELY_(i != i1); ++i) {
                  const auto [k0, h0]{
                    queue.prepare_lookup(map, keys[i] * key_scale, optimisitic_prefetch)
                  };
                  const read_pos_t pos{map.lookup(k0, h0)};
                  CHECK_POS_VALUE_AND_META_(i - queue.capacity, k0);
                }

                i -= queue.capacity;
                for (size_t j{}; j != queue.capacity; ++j) {
                  const auto [k0, h0]{queue.pop()};
                  const read_pos_t pos{map.lookup(k0, h0)};
                  CHECK_POS_VALUE_AND_META_(i, k0);
                  ++i;
                }
#else
                key_t k0{keys[i++] * key_scale};
                key_t k1{keys[i++] * key_scale};

                prefetch_t h0{map.read_prefetch(k0)};
                prefetch_t h1{map.read_prefetch(k1)};

                for (; NVHM_LIKELY_(i != i1); ++i) {
                  const key_t k2{keys[i] * key_scale};
                  const prefetch_t h2{map.read_prefetch(k2)};
                  const read_pos_t pos{map.lookup(k0, h0)};
                  CHECK_POS_VALUE_AND_META_(i - 2, k0);
                  shift(k0, k1, k2);
                  shift(h0, h1, h2);
                }

                read_pos_t pos{map.lookup(k0, h0)};
                CHECK_POS_VALUE_AND_META_(i, k0);
                ++i;
                pos = map.lookup(k1, h1);
                CHECK_POS_VALUE_AND_META_(i, k1);
                ++i;
#endif
              }};
              if (!launch_and_time_workers(trial, worker_fn)) return false;
            }
            std::cout << " (prefetch = 2)\n";
          } break;
          case 3: {
            std::cout << map_type << ", " << std::setw(15) << "Find 100% hit (sq  3): ";
            std::cout.flush();
            for (size_t trial{}; trial != num_trials; ++trial) {
              if (shuffle_keys) {
                std::shuffle(keys.begin(), keys.end(), g);
              }

              const auto worker_fn{[&](const size_t w, size_t i, const size_t i1) -> void {
#if 1
                shift_prefetch_queue<key_t, prefetch_t, 3> queue;

                for (size_t j{}; j != queue.capacity; ++j) {
                  queue.prepare_lookup(map, keys[i++] * key_scale, optimisitic_prefetch);
                }

                for (; NVHM_LIKELY_(i != i1); ++i) {
                  const auto [k0, h0]{
                    queue.prepare_lookup(map, keys[i] * key_scale, optimisitic_prefetch)
                  };
                  const read_pos_t pos{map.lookup(k0, h0)};
                  CHECK_POS_VALUE_AND_META_(i - queue.capacity, k0);
                }

                i -= queue.capacity;
                for (size_t j{}; j != queue.capacity; ++j) {
                  const auto [k0, h0]{queue.pop()};
                  const read_pos_t pos{map.lookup(k0, h0)};
                  CHECK_POS_VALUE_AND_META_(i, k0);
                  ++i;
                }
#else
                key_t k0{keys[i++] * key_scale};
                key_t k1{keys[i++] * key_scale};
                key_t k2{keys[i++] * key_scale};

                prefetch_t h0{map.read_prefetch(k0)};
                prefetch_t h1{map.read_prefetch(k1)};
                prefetch_t h2{map.read_prefetch(k2)};

                for (; NVHM_LIKELY_(i != i1); ++i) {
                  const key_t k3{keys[i] * key_scale};
                  const prefetch_t h3{map.read_prefetch(k3)};
                  const read_pos_t pos{map.lookup(k0, h0)};
                  CHECK_POS_VALUE_AND_META_(i - 3, k0);
                  shift(k0, k1, k2, k3);
                  shift(h0, h1, h2, h3);
                }

                read_pos_t pos{map.lookup(k0, h0)};
                CHECK_POS_VALUE_AND_META_(i, k0);
                ++i;
                pos = map.lookup(k1, h1);
                CHECK_POS_VALUE_AND_META_(i, k1);
                ++i;
                pos = map.lookup(k2, h2);
                CHECK_POS_VALUE_AND_META_(i, k2);
                ++i;
#endif
              }};
              if (!launch_and_time_workers(trial, worker_fn)) return false;
            }
            std::cout << " (prefetch = 3)\n";
          } break;
          case 4: {
            std::cout << map_type << ", " << std::setw(15) << "Find 100% hit (sq  4): ";
            std::cout.flush();
            for (size_t trial{}; trial != num_trials; ++trial) {
              if (shuffle_keys) {
                std::shuffle(keys.begin(), keys.end(), g);
              }

              if (!launch_and_time_workers(
                    trial,
                    [&](const size_t w, size_t i, const size_t i1) -> void {
#if 1
                shift_prefetch_queue<key_t, prefetch_t, 4> queue;

                for (size_t j{}; j != queue.capacity; ++j) {
                  queue.prepare_lookup(map, keys[i++] * key_scale, optimisitic_prefetch);
                }

                for (; NVHM_LIKELY_(i != i1); ++i) {
                  const auto [k0, h0]{
                    queue.prepare_lookup(map, keys[i] * key_scale, optimisitic_prefetch)
                  };
                  const read_pos_t pos{map.lookup(k0, h0)};
                  CHECK_POS_VALUE_AND_META_(i - queue.capacity, k0);
                }

                i -= queue.capacity;
                for (size_t j{}; j != queue.capacity; ++j) {
                  const auto [k0, h0]{queue.pop()};
                  const read_pos_t pos{map.lookup(k0, h0)};
                  CHECK_POS_VALUE_AND_META_(i, k0);
                  ++i;
                }
#else
                key_t k0{keys[i++] * key_scale};
                key_t k1{keys[i++] * key_scale};
                key_t k2{keys[i++] * key_scale};
                key_t k3{keys[i++] * key_scale};

                prefetch_t h0{map.read_prefetch(k0)};
                prefetch_t h1{map.read_prefetch(k1)};
                prefetch_t h2{map.read_prefetch(k2)};
                prefetch_t h3{map.read_prefetch(k3)};

                for (; NVHM_LIKELY_(i != i1); ++i) {
                  const key_t k4{keys[i] * key_scale};
                  const prefetch_t h4{map.read_prefetch(k4)};
                  const read_pos_t pos{map.lookup(k0, h0)};
                  CHECK_POS_VALUE_AND_META_(i - 4, k0);
                  shift(k0, k1, k2, k3, k4);
                  shift(h0, h1, h2, h3, h4);
                }

                read_pos_t pos{map.lookup(k0, h0)};
                CHECK_POS_VALUE_AND_META_(i, k0);
                ++i;
                pos = map.lookup(k1, h1);
                CHECK_POS_VALUE_AND_META_(i, k1);
                ++i;
                pos = map.lookup(k2, h2);
                CHECK_POS_VALUE_AND_META_(i, k2);
                ++i;
                pos = map.lookup(k3, h3);
                CHECK_POS_VALUE_AND_META_(i, k3);
                ++i;
#endif
              }
                  ))
                return false;
            }
            std::cout << " (prefetch = 4)\n";
          } break;
          case 5: {
            std::cout << map_type << ", " << std::setw(15) << "Find 100% hit (sq  5): ";
            std::cout.flush();
            for (size_t trial{}; trial != num_trials; ++trial) {
              if (shuffle_keys) {
                std::shuffle(keys.begin(), keys.end(), g);
              }

              const auto worker_fn{[&](const size_t w, size_t i, const size_t i1) -> void {
#if 1
                shift_prefetch_queue<key_t, prefetch_t, 5> queue;

                for (size_t j{}; j != queue.capacity; ++j) {
                  queue.prepare_lookup(map, keys[i++] * key_scale, optimisitic_prefetch);
                }

                for (; NVHM_LIKELY_(i != i1); ++i) {
                  const auto [k0, h0]{
                    queue.prepare_lookup(map, keys[i] * key_scale, optimisitic_prefetch)
                  };
                  const read_pos_t pos{map.lookup(k0, h0)};
                  CHECK_POS_VALUE_AND_META_(i - queue.capacity, k0);
                }

                i -= queue.capacity;
                for (size_t j{}; j != queue.capacity; ++j) {
                  const auto [k0, h0]{queue.pop()};
                  const read_pos_t pos{map.lookup(k0, h0)};
                  CHECK_POS_VALUE_AND_META_(i, k0);
                  ++i;
                }
#else
                key_t k0{keys[i++] * key_scale};
                key_t k1{keys[i++] * key_scale};
                key_t k2{keys[i++] * key_scale};
                key_t k3{keys[i++] * key_scale};
                key_t k4{keys[i++] * key_scale};

                prefetch_t h0{map.read_prefetch(k0)};
                prefetch_t h1{map.read_prefetch(k1)};
                prefetch_t h2{map.read_prefetch(k2)};
                prefetch_t h3{map.read_prefetch(k3)};
                prefetch_t h4{map.read_prefetch(k4)};

                for (; NVHM_LIKELY_(i != i1); ++i) {
                  const key_t k5{keys[i] * key_scale};
                  const prefetch_t h5{map.read_prefetch(k5)};
                  const read_pos_t pos{map.lookup(k0, h0)};
                  CHECK_POS_VALUE_AND_META_(i - 5, k0);
                  shift(k0, k1, k2, k3, k4, k5);
                  shift(h0, h1, h2, h3, h4, h5);
                }

                read_pos_t pos{map.lookup(k0, h0)};
                CHECK_POS_VALUE_AND_META_(i, k0);
                ++i;
                pos = map.lookup(k1, h1);
                CHECK_POS_VALUE_AND_META_(i, k1);
                ++i;
                pos = map.lookup(k2, h2);
                CHECK_POS_VALUE_AND_META_(i, k2);
                ++i;
                pos = map.lookup(k3, h3);
                CHECK_POS_VALUE_AND_META_(i, k3);
                ++i;
                pos = map.lookup(k4, h4);
                CHECK_POS_VALUE_AND_META_(i, k4);
                ++i;
#endif
              }};
              if (!launch_and_time_workers(trial, worker_fn)) return false;
            }
            std::cout << " (prefetch = 5)\n";
          } break;
        }

        std::cout << map_type << ", " << std::setw(15) << "Find 100% hit (rq " << std::setw(2)
                  << find_prefetch << "): ";
        std::cout.flush();
        for (size_t trial{}; trial != num_trials; ++trial) {
          if (shuffle_keys) {
            std::shuffle(keys.begin(), keys.end(), g);
          }

          const auto worker_fn{[&](const size_t w, size_t i, const size_t i1) -> void {
#if 1
            ring_prefetch_queue<key_t, prefetch_t, max_prefetch_cache_size> queue;

            while (queue.size() != find_prefetch) {
              queue.prepare_lookup(map, keys[i++] * key_scale, optimisitic_prefetch);
            }

            for (; NVHM_LIKELY_(i != i1); ++i) {
              const auto [k0, h0]{
                queue.prepare_lookup(map, keys[i] * key_scale, optimisitic_prefetch).pop()
              };
              const read_pos_t pos{map.lookup(k0, h0)};
              CHECK_POS_VALUE_AND_META_(i - find_prefetch, k0);
            }

            i -= find_prefetch;
            for (; !queue.empty(); ++i) {
              const auto [k0, h0]{queue.pop()};
              const read_pos_t pos{map.lookup(k0, h0)};
              CHECK_POS_VALUE_AND_META_(i, k0);
            }
#else
            key_t k0{keys[i++] * key_scale};
            key_t k1{keys[i++] * key_scale};
            key_t k2{keys[i++] * key_scale};
            key_t k3{keys[i++] * key_scale};
            key_t k4{keys[i++] * key_scale};

            prefetch_t h0{map.read_prefetch(k0)};
            prefetch_t h1{map.read_prefetch(k1)};
            prefetch_t h2{map.read_prefetch(k2)};
            prefetch_t h3{map.read_prefetch(k3)};
            prefetch_t h4{map.read_prefetch(k4)};

            for (; NVHM_LIKELY_(i != i1); ++i) {
              const key_t k5{keys[i] * key_scale};
              const prefetch_t h5{map.read_prefetch(k5)};
              const read_pos_t pos{map.lookup(k0, h0)};
              CHECK_POS_VALUE_AND_META_(i - 5, k0);
              shift(k0, k1, k2, k3, k4, k5);
              shift(h0, h1, h2, h3, h4, h5);
            }

            read_pos_t pos{map.lookup(k0, h0)};
            CHECK_POS_VALUE_AND_META_(i, k0);
            ++i;
            pos = map.lookup(k1, h1);
            CHECK_POS_VALUE_AND_META_(i, k1);
            ++i;
            pos = map.lookup(k2, h2);
            CHECK_POS_VALUE_AND_META_(i, k2);
            ++i;
            pos = map.lookup(k3, h3);
            CHECK_POS_VALUE_AND_META_(i, k3);
            ++i;
            pos = map.lookup(k4, h4);
            CHECK_POS_VALUE_AND_META_(i, k4);
            ++i;
#endif
          }};
          if (!launch_and_time_workers(trial, worker_fn)) return false;
        }
        std::cout << " (prefetch = " << find_prefetch << ")\n";
      }
    }
  }

  {
    constexpr bool optimisitic_prefetch{false};
    const auto is_valid_key{[](const size_t i) { return i % 2 == 0; }};
    const auto make_next_key{[&](const size_t i) {
      return (keys[i] + (is_valid_key(i) ? 0 : static_cast<key_t>(keys.size()))) * key_scale;
    }};

    std::cout << map_type << ", " << std::setw(15) << "Find  50% hit:         ";
    std::cout.flush();
    for (size_t trial{}; trial != num_trials; ++trial) {
      if (shuffle_keys) {
        std::shuffle(keys.begin(), keys.end(), g);
      }

      if (!launch_and_time_workers(trial, [&](const size_t w, size_t i, const size_t i1) -> void {
        for (; i != i1; ++i) {
          const key_t k{make_next_key(i)};
          const read_pos_t pos{map.lookup(k)};
          CHECK_POS_VALUE_AND_META_(i, k);
        }
      }))
        return false;
    }
    std::cout << '\n';

#define SHIFT_PREFETCH_FIND_50_(_find_prefetch_)                                                  \
  do {                                                                                            \
    std::cout << map_type << ", " << std::setw(15) << "Find  50% hit (sq " << std::setw(2)        \
              << _find_prefetch_ << "): ";                                                        \
    std::cout.flush();                                                                            \
    for (size_t trial{}; trial != num_trials; ++trial) {                                          \
      if (shuffle_keys) {                                                                         \
        std::shuffle(keys.begin(), keys.end(), g);                                                \
      }                                                                                           \
                                                                                                  \
      const auto worker_fn{[&](const size_t w, size_t i, const size_t i1) -> void {               \
        shift_prefetch_queue<key_t, prefetch_t, _find_prefetch_> queue;                           \
                                                                                                  \
        for (size_t j{}; j != queue.capacity; ++j, ++i) {                                         \
          queue.prepare_lookup(map, make_next_key(i), optimisitic_prefetch);                      \
        }                                                                                         \
                                                                                                  \
        for (; NVHM_LIKELY_(i != i1); ++i) {                                                      \
          const auto [k0, h0]{queue.prepare_lookup(map, make_next_key(i), optimisitic_prefetch)}; \
          const read_pos_t pos{map.lookup(k0, h0)};                                               \
          CHECK_POS_VALUE_AND_META_(i - queue.capacity, k0);                                      \
        }                                                                                         \
                                                                                                  \
        i -= queue.capacity;                                                                      \
        for (size_t j{}; j != queue.capacity; ++j, ++i) {                                         \
          const auto [k0, h0]{queue.pop()};                                                       \
          const read_pos_t pos{map.lookup(k0, h0)};                                               \
          CHECK_POS_VALUE_AND_META_(i, k0);                                                       \
        }                                                                                         \
      }};                                                                                         \
      if (!launch_and_time_workers(trial, worker_fn)) return false;                               \
    }                                                                                             \
    std::cout << " (prefetch " << _find_prefetch_ << ")\n";                                       \
  } while (false)

    if (num_keys_per_worker >= max_find_prefetch) {
      for (size_t find_prefetch{1}; find_prefetch < max_find_prefetch; ++find_prefetch) {
        switch (find_prefetch) {
          case 1: SHIFT_PREFETCH_FIND_50_(1); break;
          case 2: SHIFT_PREFETCH_FIND_50_(2); break;
          case 3: SHIFT_PREFETCH_FIND_50_(3); break;
          case 4: SHIFT_PREFETCH_FIND_50_(4); break;
          case 5: SHIFT_PREFETCH_FIND_50_(5); break;
          case 6: SHIFT_PREFETCH_FIND_50_(6); break;
          case 7: SHIFT_PREFETCH_FIND_50_(7); break;
          case 8: SHIFT_PREFETCH_FIND_50_(8); break;
          case 9: SHIFT_PREFETCH_FIND_50_(9); break;
          case 10: SHIFT_PREFETCH_FIND_50_(10); break;
        }

        std::cout << map_type << ", " << std::setw(15) << "Find  50% hit (rq " << std::setw(2)
                  << find_prefetch << "): ";
        std::cout.flush();
        for (size_t trial{}; trial != num_trials; ++trial) {
          if (shuffle_keys) {
            std::shuffle(keys.begin(), keys.end(), g);
          }

          const auto worker_fn{[&](const size_t w, size_t i, const size_t i1) -> void {
            ring_prefetch_queue<key_t, prefetch_t, max_prefetch_cache_size> queue;

            for (; queue.size() != find_prefetch; ++i) {
              queue.prepare_lookup(map, make_next_key(i), optimisitic_prefetch);
            }

            for (; NVHM_LIKELY_(i != i1); ++i) {
              const auto [k0, h0]{
                queue.prepare_lookup(map, make_next_key(i), optimisitic_prefetch).pop()
              };
              const read_pos_t pos{map.lookup(k0, h0)};
              CHECK_POS_VALUE_AND_META_(i - find_prefetch, k0);
            }

            i -= find_prefetch;
            for (; !queue.empty(); ++i) {
              const auto [k0, h0]{queue.pop()};
              const read_pos_t pos{map.lookup(k0, h0)};
              CHECK_POS_VALUE_AND_META_(i, k0);
            }
          }};
          if (!launch_and_time_workers(trial, worker_fn)) return false;
        }
        std::cout << " (prefetch " << find_prefetch << ")\n";
      }
    }
  }

  {
    constexpr bool optimisitic_prefetch{false};
    const auto is_valid_key{[](const size_t i) { return i % 10 == 0; }};
    const auto make_next_key{[&](const size_t i) {
      return (keys[i] + (is_valid_key(i) ? 0 : static_cast<key_t>(keys.size()))) * key_scale;
    }};

    std::cout << map_type << ", " << std::setw(15) << "Find  10% hit:         ";
    std::cout.flush();
    for (size_t trial{}; trial != num_trials; ++trial) {
      if (shuffle_keys) {
        std::shuffle(keys.begin(), keys.end(), g);
      }

      const auto worker_fn{[&](const size_t w, size_t i, const size_t i1) -> void {
        for (; i != i1; ++i) {
          const key_t k{make_next_key(i)};
          const read_pos_t pos{map.lookup(k)};
          CHECK_POS_VALUE_AND_META_(i, k);
        }
      }};
      if (!launch_and_time_workers(trial, worker_fn)) return false;
    }
    std::cout << '\n';

#define SHIFT_PREFETCH_FIND_10_(_find_prefetch_)                                                  \
  do {                                                                                            \
    std::cout << map_type << ", " << std::setw(15) << "Find  10% hit (sq " << std::setw(2)        \
              << _find_prefetch_ << "): ";                                                        \
    std::cout.flush();                                                                            \
    for (size_t trial{}; trial != num_trials; ++trial) {                                          \
      if (shuffle_keys) {                                                                         \
        std::shuffle(keys.begin(), keys.end(), g);                                                \
      }                                                                                           \
                                                                                                  \
      const auto worker_fn{[&](const size_t w, size_t i, const size_t i1) -> void {               \
        shift_prefetch_queue<key_t, prefetch_t, _find_prefetch_> queue;                           \
                                                                                                  \
        for (size_t j{}; j != queue.capacity; ++j, ++i) {                                         \
          queue.prepare_lookup(map, make_next_key(i), optimisitic_prefetch);                      \
        }                                                                                         \
                                                                                                  \
        for (; NVHM_LIKELY_(i != i1); ++i) {                                                      \
          const auto [k0, h0]{queue.prepare_lookup(map, make_next_key(i), optimisitic_prefetch)}; \
          const read_pos_t pos{map.lookup(k0, h0)};                                               \
          CHECK_POS_VALUE_AND_META_(i - queue.capacity, k0);                                      \
        }                                                                                         \
                                                                                                  \
        i -= queue.capacity;                                                                      \
        for (size_t j{}; j != queue.capacity; ++j, ++i) {                                         \
          const auto [k0, h0]{queue.pop()};                                                       \
          const read_pos_t pos{map.lookup(k0, h0)};                                               \
          CHECK_POS_VALUE_AND_META_(i, k0);                                                       \
        }                                                                                         \
      }};                                                                                         \
      if (!launch_and_time_workers(trial, worker_fn)) return false;                               \
    }                                                                                             \
    std::cout << " (prefetch " << _find_prefetch_ << ")\n";                                       \
  } while (false)

    if (num_keys_per_worker >= max_find_prefetch) {
      for (size_t find_prefetch{1}; find_prefetch < max_find_prefetch; ++find_prefetch) {
        switch (find_prefetch) {
          case 1: SHIFT_PREFETCH_FIND_10_(1); break;
          case 2: SHIFT_PREFETCH_FIND_10_(2); break;
          case 3: SHIFT_PREFETCH_FIND_10_(3); break;
          case 4: SHIFT_PREFETCH_FIND_10_(4); break;
          case 5: SHIFT_PREFETCH_FIND_10_(5); break;
          case 6: SHIFT_PREFETCH_FIND_10_(6); break;
          case 7: SHIFT_PREFETCH_FIND_10_(7); break;
          case 8: SHIFT_PREFETCH_FIND_10_(8); break;
          case 9: SHIFT_PREFETCH_FIND_10_(9); break;
          case 10: SHIFT_PREFETCH_FIND_10_(10); break;
        }

        std::cout << map_type << ", " << std::setw(15) << "Find  10% hit (rq " << std::setw(2)
                  << find_prefetch << "): ";
        std::cout.flush();
        for (size_t trial{}; trial != num_trials; ++trial) {
          if (shuffle_keys) {
            std::shuffle(keys.begin(), keys.end(), g);
          }

          const auto worker_fn{[&](const size_t w, size_t i, const size_t i1) -> void {
            ring_prefetch_queue<key_t, prefetch_t, max_prefetch_cache_size> queue;

            for (; queue.size() != find_prefetch; ++i) {
              queue.prepare_lookup(map, make_next_key(i), optimisitic_prefetch);
            }

            for (; NVHM_LIKELY_(i != i1); ++i) {
              const auto [k0, h0]{
                queue.prepare_lookup(map, make_next_key(i), optimisitic_prefetch).pop()
              };
              const read_pos_t pos{map.lookup(k0, h0)};
              CHECK_POS_VALUE_AND_META_(i - find_prefetch, k0);
            }

            i -= find_prefetch;
            for (; !queue.empty(); ++i) {
              const auto [k0, h0]{queue.pop()};
              const read_pos_t pos{map.lookup(k0, h0)};
              CHECK_POS_VALUE_AND_META_(i, k0);
            }
          }};
          if (!launch_and_time_workers(trial, worker_fn)) return false;
        }
        std::cout << " (prefetch " << find_prefetch << ")\n";
      }
    }
  }

  {
    constexpr bool optimisitic_prefetch{false};
    const auto is_valid_key{[](const size_t i) { return i % 100 == 0; }};
    const auto make_next_key{[&](const size_t i) {
      return (keys[i] + (is_valid_key(i) ? 0 : static_cast<key_t>(keys.size()))) * key_scale;
    }};

    std::cout << map_type << ", " << std::setw(15) << "Find   1% hit:         ";
    std::cout.flush();
    for (size_t trial{}; trial != num_trials; ++trial) {
      if (shuffle_keys) {
        std::shuffle(keys.begin(), keys.end(), g);
      }

      const auto worker_fn{[&](const size_t w, size_t i, const size_t i1) -> void {
        for (; i != i1; ++i) {
          const key_t k{make_next_key(i)};
          const read_pos_t pos{map.lookup(k)};
          CHECK_POS_VALUE_AND_META_(i, k);
        }
      }};
      if (!launch_and_time_workers(trial, worker_fn)) return false;
    }
    std::cout << '\n';

#define SHIFT_PREFETCH_FIND_1_(_find_prefetch_)                                                   \
  do {                                                                                            \
    std::cout << map_type << ", " << std::setw(15) << "Find   1% hit (sq " << std::setw(2)        \
              << _find_prefetch_ << "): ";                                                        \
    std::cout.flush();                                                                            \
    for (size_t trial{}; trial != num_trials; ++trial) {                                          \
      if (shuffle_keys) {                                                                         \
        std::shuffle(keys.begin(), keys.end(), g);                                                \
      }                                                                                           \
                                                                                                  \
      const auto worker_fn{[&](const size_t w, size_t i, const size_t i1) -> void {               \
        shift_prefetch_queue<key_t, prefetch_t, _find_prefetch_> queue;                           \
                                                                                                  \
        for (size_t j{}; j != queue.capacity; ++j, ++i) {                                         \
          queue.prepare_lookup(map, make_next_key(i), optimisitic_prefetch);                      \
        }                                                                                         \
                                                                                                  \
        for (; NVHM_LIKELY_(i != i1); ++i) {                                                      \
          const auto [k0, h0]{queue.prepare_lookup(map, make_next_key(i), optimisitic_prefetch)}; \
          const read_pos_t pos{map.lookup(k0, h0)};                                               \
          CHECK_POS_VALUE_AND_META_(i - queue.capacity, k0);                                      \
        }                                                                                         \
                                                                                                  \
        i -= queue.capacity;                                                                      \
        for (size_t j{}; j != queue.capacity; ++j, ++i) {                                         \
          const auto [k0, h0]{queue.pop()};                                                       \
          const read_pos_t pos{map.lookup(k0, h0)};                                               \
          CHECK_POS_VALUE_AND_META_(i, k0);                                                       \
        }                                                                                         \
      }};                                                                                         \
      if (!launch_and_time_workers(trial, worker_fn)) return false;                               \
    }                                                                                             \
    std::cout << " (prefetch " << _find_prefetch_ << ")\n";                                       \
  } while (false)

    if (num_keys_per_worker >= max_find_prefetch) {
      for (size_t find_prefetch{1}; find_prefetch < max_find_prefetch; ++find_prefetch) {
        switch (find_prefetch) {
          case 1: SHIFT_PREFETCH_FIND_1_(1); break;
          case 2: SHIFT_PREFETCH_FIND_1_(2); break;
          case 3: SHIFT_PREFETCH_FIND_1_(3); break;
          case 4: SHIFT_PREFETCH_FIND_1_(4); break;
          case 5: SHIFT_PREFETCH_FIND_1_(5); break;
          case 6: SHIFT_PREFETCH_FIND_1_(6); break;
          case 7: SHIFT_PREFETCH_FIND_1_(7); break;
          case 8: SHIFT_PREFETCH_FIND_1_(8); break;
          case 9: SHIFT_PREFETCH_FIND_1_(9); break;
          case 10: SHIFT_PREFETCH_FIND_1_(10); break;
        }

        std::cout << map_type << ", " << std::setw(15) << "Find   1% hit (rq " << std::setw(2)
                  << find_prefetch << "): ";
        std::cout.flush();
        for (size_t trial{}; trial != num_trials; ++trial) {
          if (shuffle_keys) {
            std::shuffle(keys.begin(), keys.end(), g);
          }

          const auto worker_fn{[&](const size_t w, size_t i, const size_t i1) -> void {
            ring_prefetch_queue<key_t, prefetch_t, max_prefetch_cache_size> queue;

            for (; queue.size() != find_prefetch; ++i) {
              queue.prepare_lookup(map, make_next_key(i), optimisitic_prefetch);
            }

            for (; NVHM_LIKELY_(i != i1); ++i) {
              const auto [k0, h0]{
                queue.prepare_lookup(map, make_next_key(i), optimisitic_prefetch).pop()
              };
              const read_pos_t pos{map.lookup(k0, h0)};
              CHECK_POS_VALUE_AND_META_(i - find_prefetch, k0);
            }

            i -= find_prefetch;
            for (; !queue.empty(); ++i) {
              const auto [k0, h0]{queue.pop()};
              const read_pos_t pos{map.lookup(k0, h0)};
              CHECK_POS_VALUE_AND_META_(i, k0);
            }
          }};
          if (!launch_and_time_workers(trial, worker_fn)) return false;
        }
        std::cout << " (prefetch " << find_prefetch << ")\n";
      }
    }
  }

  for (size_t i{}; i < vbuffer.size(); ++i) {
    if (static_cast<size_t>(vbuffer[i]) != i % 100) {
      return false;
    }
  }
  return true;
}

template <typename T>
constexpr size_t size_of{sizeof(T)};

template <>
constexpr size_t size_of<void>{};

template <typename Map, typename MetaType, bool FillInsteadOfCopy, bool TestMeta>
bool __attribute__((noinline)) benchmark_10M_std(
  const size_t num_trials, const size_t num_keys, const bool shuffle_keys, const size_t value_size,
  const size_t num_workers
) {
  using map_t = Map;
  const std::string map_type{type_to_string<map_t>()};
  using key_t = typename map_t::key_type;
  using meta_t = MetaType;
  constexpr static size_t meta_size{size_of<meta_t>};
  const size_t vm_size{value_size + meta_size};
  const size_t value_alignment{1};
  const size_t vm_stride{(vm_size + value_alignment - 1) / value_alignment * value_alignment};

  std::vector<key_t> keys(num_keys);
  for (size_t i{}; i != keys.size(); ++i) {
    keys[i] = static_cast<key_t>(i);
  }
  std::random_device rd;
  std::mt19937 g(rd());
  std::vector<char> data(keys.size() * vm_stride);
  std::vector<char*> slots(keys.size());
  for (size_t i{}; i != slots.size(); ++i) {
    slots[i] = &data[i * vm_stride];
  }
  std::shuffle(slots.begin(), slots.end(), g);

  std::vector<char> vbuffer(value_size);
  for (size_t i{}; i != vbuffer.size(); ++i) {
    vbuffer[i] = static_cast<char>(i % 100);
  }
  map_t map;  //(keys.size());

  std::cout << map_type << ", " << std::setw(15) << "Insert: ";
  std::cout.flush();
  for (size_t trial{}; trial != num_trials; ++trial) {
    if (shuffle_keys) {
      std::shuffle(keys.begin(), keys.end(), g);
    }
    map.clear();
    const auto begin{std::chrono::high_resolution_clock::now()};
    for (const key_t k : keys) {
      char* const __restrict vm{slots[static_cast<size_t>(k)]};
      map.emplace(k * key_scale, vm);
      if (value_size) {
        if constexpr (FillInsteadOfCopy) {
          std::fill_n(vm, value_size, static_cast<char>(~k));
        } else {
          fast_copy(vm, vbuffer.data(), value_size);
        }
      }
      if constexpr (meta_size) {
        *reinterpret_cast<meta_t*>(&vm[value_size]) = -k;
      }
    }
    const auto end{std::chrono::high_resolution_clock::now()};

    std::cout << (trial ? ", " : "")
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms";
    std::cout.flush();
  }
  std::cout << ", size: " << map.size() << " / " << 'x' /*map.capacity()*/ << '\n';
  const auto map_end{map.end()};

  // Find.
  std::vector<std::thread> workers(num_workers);
  std::atomic_size_t worker_ready;
  std::vector<bool> worker_result(num_workers, true);
  std::vector<std::chrono::milliseconds> worker_times(num_workers);
  const size_t num_keys_per_worker{(keys.size() + num_workers - 1) / num_workers};

  const auto launch_and_time_workers{
    [&](const size_t trial, const std::function<void(size_t, size_t, size_t)>& worker_fn) -> bool {
    worker_ready.store(0, std::memory_order_relaxed);
    for (size_t w{}; w != num_workers; ++w) {
      workers[w] = std::thread([&, w]() {
        const size_t i0{std::min(w * num_keys_per_worker, keys.size())};
        const size_t i1{std::min(i0 + num_keys_per_worker, keys.size())};

        worker_ready.fetch_add(1, std::memory_order_relaxed);
        while (worker_ready.load(std::memory_order_relaxed) != num_workers) {
          // Spin until all threads are ready.
        }

        const auto begin{std::chrono::high_resolution_clock::now()};
        worker_fn(w, i0, i1);
        const auto end{std::chrono::high_resolution_clock::now()};
        worker_times[w] = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
      });
    }

    std::chrono::milliseconds sum_time{};
    for (size_t w{}; w != num_workers; ++w) {
      workers[w].join();
      if (!worker_result[w]) {
        std::cerr << "Worker #" << w << " failed!";
        return false;
      }
      sum_time += worker_times[w];
    }
    sum_time /= static_cast<int64_t>(num_workers);
    std::cout << (trial ? ", " : "") << std::setw(5) << sum_time.count() << "ms";
    std::cout.flush();
    return true;
  }
  };

  {
    const auto is_valid_key{[](const size_t) { return true; }};
    const auto make_next_key{[&](const size_t i) {
      return (keys[i] + (is_valid_key(i) ? 0 : static_cast<key_t>(keys.size()))) * key_scale;
    }};

    std::cout << map_type << ", " << std::setw(15) << "Find 100% hit:         ";
    std::cout.flush();
    for (size_t trial{}; trial != num_trials; ++trial) {
      if (shuffle_keys) {
        std::shuffle(keys.begin(), keys.end(), g);
      }

      const auto worker_fn{[&](const size_t w, size_t i, const size_t i1) -> void {
        for (; NVHM_LIKELY_(i != i1); ++i) {
          const key_t k{make_next_key(i)};
          const auto it{map.find(k)};
          CHECK_POS_VALUE_AND_META_STD_(i, k);
        }
      }};
      if (!launch_and_time_workers(trial, worker_fn)) return false;
    }
    std::cout << " (no prefetch)\n";
  }

  {
    const auto is_valid_key{[](const size_t i) { return i % 2 == 0; }};
    const auto make_next_key{[&](const size_t i) {
      return (keys[i] + (is_valid_key(i) ? 0 : static_cast<key_t>(keys.size()))) * key_scale;
    }};

    std::cout << map_type << ", " << std::setw(15) << "Find  50% hit:         ";
    std::cout.flush();
    for (size_t trial{}; trial != num_trials; ++trial) {
      if (shuffle_keys) {
        std::shuffle(keys.begin(), keys.end(), g);
      }

      if (!launch_and_time_workers(trial, [&](const size_t w, size_t i, const size_t i1) -> void {
        for (; i != i1; ++i) {
          const key_t k{make_next_key(i)};
          const auto it{map.find(k)};
          CHECK_POS_VALUE_AND_META_STD_(i, k);
        }
      }))
        return false;
    }
    std::cout << " (no prefetch)\n";
  }

  {
    const auto is_valid_key{[](const size_t i) { return i % 10 == 0; }};
    const auto make_next_key{[&](const size_t i) {
      return (keys[i] + (is_valid_key(i) ? 0 : static_cast<key_t>(keys.size()))) * key_scale;
    }};

    std::cout << map_type << ", " << std::setw(15) << "Find  10% hit:         ";
    std::cout.flush();
    for (size_t trial{}; trial != num_trials; ++trial) {
      if (shuffle_keys) {
        std::shuffle(keys.begin(), keys.end(), g);
      }

      const auto worker_fn{[&](const size_t w, size_t i, const size_t i1) -> void {
        for (; i != i1; ++i) {
          const key_t k{make_next_key(i)};
          const auto it{map.find(k)};
          CHECK_POS_VALUE_AND_META_STD_(i, k);
        }
      }};
      if (!launch_and_time_workers(trial, worker_fn)) return false;
    }
    std::cout << " (no prefetch)\n";
  }

  {
    const auto is_valid_key{[](const size_t i) { return i % 100 == 0; }};
    const auto make_next_key{[&](const size_t i) {
      return (keys[i] + (is_valid_key(i) ? 0 : static_cast<key_t>(keys.size()))) * key_scale;
    }};

    std::cout << map_type << ", " << std::setw(15) << "Find   1% hit:         ";
    std::cout.flush();
    for (size_t trial{}; trial != num_trials; ++trial) {
      if (shuffle_keys) {
        std::shuffle(keys.begin(), keys.end(), g);
      }

      const auto worker_fn{[&](const size_t w, size_t i, const size_t i1) -> void {
        for (; i != i1; ++i) {
          const key_t k{make_next_key(i)};
          const auto it{map.find(k)};
          CHECK_POS_VALUE_AND_META_STD_(i, k);
        }
      }};
      if (!launch_and_time_workers(trial, worker_fn)) return false;
    }
    std::cout << " (no prefetch)\n";
  }

  for (size_t i{}; i < vbuffer.size(); ++i) {
    if (static_cast<size_t>(vbuffer[i]) != i % 100) {
      return false;
    }
  }
  return true;
}

int main(int argc, [[maybe_unused]] const char* argv[]) {
  const size_t num_trials{4};
  // const size_t num_maps{8};
  const size_t num_keys{50'000'000};
  const bool shuffle_keys{true};
  const size_t value_size{256};
  // using probe_seq_t = linear_probe_seq;
  using probe_seq_t = quadratic_seq<>;
  const size_t num_workers{8};

  using key_t = int64_t;
  using meta_t = time_t;

  static constexpr bool fill_not_compare{false};
  const size_t max_insert_prefetch{static_cast<size_t>(argc) * 0};
  const size_t max_find_prefetch{static_cast<size_t>(argc) * 11};

  //if (!benchmark_10M<
  //      map<key_t, meta_t, char, default_kernel_t, probe_seq_t>, fill_not_compare, true>(
  //      num_trials, num_keys, shuffle_keys, value_size, num_workers, max_insert_prefetch,
  //      max_find_prefetch
  //    )) {
  //  std::cout << "Error!\n";
  //}

  if (!benchmark_10M<
        map<key_t, meta_t, char, array_kernel8_t, probe_seq_t>, fill_not_compare, true>(
        num_trials, num_keys, shuffle_keys, value_size, num_workers, max_insert_prefetch,
        max_find_prefetch
      )) {
    std::cout << "Error!\n";
  }
  if (!benchmark_10M<
        map<key_t, meta_t, char, array_kernel16_t, probe_seq_t>, fill_not_compare, true>(
        num_trials, num_keys, shuffle_keys, value_size, num_workers, max_insert_prefetch,
        max_find_prefetch
      )) {
    std::cout << "Error!\n";
  }
  if (!benchmark_10M<
        map<key_t, meta_t, char, array_kernel32_t, probe_seq_t>, fill_not_compare, true>(
        num_trials, num_keys, shuffle_keys, value_size, num_workers, max_insert_prefetch,
        max_find_prefetch
      )) {
    std::cout << "Error!\n";
  }
  if (!benchmark_10M<
        map<key_t, meta_t, char, array_kernel64_t, probe_seq_t>, fill_not_compare, true>(
        num_trials, num_keys, shuffle_keys, value_size, num_workers, max_insert_prefetch,
        max_find_prefetch
      )) {
    std::cout << "Error!\n";
  }
  if (!benchmark_10M<
        map<key_t, meta_t, char, array_kernel128_t, probe_seq_t>, fill_not_compare, true>(
        num_trials, num_keys, shuffle_keys, value_size, num_workers, max_insert_prefetch,
        max_find_prefetch
      )) {
    std::cout << "Error!\n";
  }
  if (!benchmark_10M<
        map<key_t, meta_t, char, array_kernel256_t, probe_seq_t>, fill_not_compare, true>(
        num_trials, num_keys, shuffle_keys, value_size, num_workers, max_insert_prefetch,
        max_find_prefetch
      )) {
    std::cout << "Error!\n";
  }
  if (!benchmark_10M<
        map<key_t, meta_t, char, array_kernel512_t, probe_seq_t>, fill_not_compare, true>(
        num_trials, num_keys, shuffle_keys, value_size, num_workers, max_insert_prefetch,
        max_find_prefetch
      )) {
    std::cout << "Error!\n";
  }
  if (!benchmark_10M<
        map<key_t, meta_t, char, array_kernel1024_t, probe_seq_t>, fill_not_compare, true>(
        num_trials, num_keys, shuffle_keys, value_size, num_workers, max_insert_prefetch,
        max_find_prefetch
      )) {
    std::cout << "Error!\n";
  }

  if (!benchmark_10M<map<key_t, meta_t, char, uint_kernel8_t, probe_seq_t>, fill_not_compare, true>(
        num_trials, num_keys, shuffle_keys, value_size, num_workers, max_insert_prefetch,
        max_find_prefetch
      )) {
    std::cout << "Error!\n";
  }
  if (!benchmark_10M<
        map<key_t, meta_t, char, uint_kernel16_t, probe_seq_t>, fill_not_compare, true>(
        num_trials, num_keys, shuffle_keys, value_size, num_workers, max_insert_prefetch,
        max_find_prefetch
      )) {
    std::cout << "Error!\n";
  }
  if (!benchmark_10M<
        map<key_t, meta_t, char, uint_kernel32_t, probe_seq_t>, fill_not_compare, true>(
        num_trials, num_keys, shuffle_keys, value_size, num_workers, max_insert_prefetch,
        max_find_prefetch
      )) {
    std::cout << "Error!\n";
  }
  if (!benchmark_10M<
        map<key_t, meta_t, char, uint_kernel64_t, probe_seq_t>, fill_not_compare, true>(
        num_trials, num_keys, shuffle_keys, value_size, num_workers, max_insert_prefetch,
        max_find_prefetch
      )) {
    std::cout << "Error!\n";
  }
  if (!benchmark_10M<
        map<key_t, meta_t, char, uint_kernel128_t, probe_seq_t>, fill_not_compare, true>(
        num_trials, num_keys, shuffle_keys, value_size, num_workers, max_insert_prefetch,
        max_find_prefetch
      )) {
    std::cout << "Error!\n";
  }

#if NVHM_WITH_SSE2
  if (!benchmark_10M<map<key_t, meta_t, char, sse_kernel_t, probe_seq_t>, fill_not_compare, true>(
        num_trials, num_keys, shuffle_keys, value_size, num_workers, max_insert_prefetch,
        max_find_prefetch
      )) {
    std::cout << "Error!\n";
  }
#endif
#if NVHM_WITH_AVX2
  if (!benchmark_10M<map<key_t, meta_t, char, avx_kernel_t, probe_seq_t>, fill_not_compare, true>(
        num_trials, num_keys, shuffle_keys, value_size, num_workers, max_insert_prefetch,
        max_find_prefetch
      )) {
    std::cout << "Error!\n";
  }
#endif
#if NVHM_WITH_AVX512
  if (!benchmark_10M<
        map<key_t, meta_t, char, avx512_kernel_t, probe_seq_t>, fill_not_compare, true>(
        num_trials, num_keys, shuffle_keys, value_size, num_workers, max_insert_prefetch,
        max_find_prefetch
      )) {
    std::cout << "Error!\n";
  }
#endif
#if NVHM_WITH_NEON
  if (!benchmark_10M<
        map<key_t, meta_t, char, neon_kernel64_t, probe_seq_t>, fill_not_compare, true>(
        num_trials, num_keys, shuffle_keys, value_size, num_workers, max_insert_prefetch,
        max_find_prefetch
      )) {
    std::cout << "Error!\n";
  }
  if (!benchmark_10M<
        map<key_t, meta_t, char, neon_kernel128_t, probe_seq_t>, fill_not_compare, true>(
        num_trials, num_keys, shuffle_keys, value_size, num_workers, max_insert_prefetch,
        max_find_prefetch
      )) {
    std::cout << "Error!\n";
  }
  if (!benchmark_10M<
        map<key_t, meta_t, char, neon_kernel256_t, probe_seq_t>, fill_not_compare, true>(
        num_trials, num_keys, shuffle_keys, value_size, num_workers, max_insert_prefetch,
        max_find_prefetch
      )) {
    std::cout << "Error!\n";
  }
  if (!benchmark_10M<
        map<key_t, meta_t, char, neon_kernel512_t, probe_seq_t>, fill_not_compare, true>(
        num_trials, num_keys, shuffle_keys, value_size, num_workers, max_insert_prefetch,
        max_find_prefetch
      )) {
    std::cout << "Error!\n";
  }
#endif
#if NVHM_WITH_SVE && __ARM_FEATURE_SVE_BITS
  if (!benchmark_10M<map<key_t, meta_t, char, sve_kernel8_t, probe_seq_t>, fill_not_compare, true>(
        num_trials, num_keys, shuffle_keys, value_size, num_workers, max_insert_prefetch,
        max_find_prefetch
      )) {
    std::cout << "Error!\n";
  }
  if (!benchmark_10M<map<key_t, meta_t, char, sve_kernel16_t, probe_seq_t>, fill_not_compare, true>(
        num_trials, num_keys, shuffle_keys, value_size, num_workers, max_insert_prefetch,
        max_find_prefetch
      )) {
    std::cout << "Error!\n";
  }
  if (!benchmark_10M<map<key_t, meta_t, char, sve_kernel32_t, probe_seq_t>, fill_not_compare, true>(
        num_trials, num_keys, shuffle_keys, value_size, num_workers, max_insert_prefetch,
        max_find_prefetch
      )) {
    std::cout << "Error!\n";
  }
  if (!benchmark_10M<map<key_t, meta_t, char, sve_kernel64_t, probe_seq_t>, fill_not_compare, true>(
        num_trials, num_keys, shuffle_keys, value_size, num_workers, max_insert_prefetch,
        max_find_prefetch
      )) {
    std::cout << "Error!\n";
  }
  if (!benchmark_10M<
        map<key_t, meta_t, char, sve_kernel128_t, probe_seq_t>, fill_not_compare, true>(
        num_trials, num_keys, shuffle_keys, value_size, num_workers, max_insert_prefetch,
        max_find_prefetch
      )) {
    std::cout << "Error!\n";
  }
#if __ARM_FEATURE_SVE_BITS >= 256
  if (!benchmark_10M<
        map<key_t, meta_t, char, sve_kernel256_t, probe_seq_t>, fill_not_compare, true>(
        num_trials, num_keys, shuffle_keys, value_size, num_workers, max_insert_prefetch,
        max_find_prefetch
      )) {
    std::cout << "Error!\n";
  }
#endif
#if __ARM_FEATURE_SVE_BITS >= 512
  if (!benchmark_10M<
        map<key_t, meta_t, char, sve_kernel512_t, probe_seq_t>, fill_not_compare, true>(
        num_trials, num_keys, shuffle_keys, value_size, num_workers, max_insert_prefetch,
        max_find_prefetch
      )) {
    std::cout << "Error!\n";
  }
#endif
#if __ARM_FEATURE_SVE_BITS >= 1024
  if (!benchmark_10M<
        map<key_t, meta_t, char, sve_kernel1024_t, probe_seq_t>, fill_not_compare, true>(
        num_trials, num_keys, shuffle_keys, value_size, num_workers, max_insert_prefetch,
        max_find_prefetch
      )) {
    std::cout << "Error!\n";
  }
#endif
#if __ARM_FEATURE_SVE_BITS >= 2048
  if (!benchmark_10M<
        map<key_t, meta_t, char, sve_kernel2048_t, probe_seq_t>, fill_not_compare, true>(
        num_trials, num_keys, shuffle_keys, value_size, num_workers, max_insert_prefetch,
        max_find_prefetch
      )) {
    std::cout << "Error!\n";
  }
#endif
#endif
  if (!benchmark_10M_std<std_map_shim<map<key_t, char*>>, meta_t, fill_not_compare, true>(
        num_trials, num_keys, shuffle_keys, value_size, num_workers
      )) {
    std::cout << "Error!\n";
  }
  if (!benchmark_10M_std<phmap::flat_hash_map<key_t, char*>, meta_t, fill_not_compare, true>(
        num_trials, num_keys, shuffle_keys, value_size, num_workers
      )) {
    std::cout << "Error!\n";
  }
  if (!benchmark_10M_std<absl::flat_hash_map<key_t, char*>, meta_t, fill_not_compare, true>(
        num_trials, num_keys, shuffle_keys, value_size, num_workers
      )) {
    std::cout << "Error!\n";
  }
  if (!benchmark_10M_std<std::unordered_map<key_t, char*>, meta_t, fill_not_compare, true>(
        num_trials, num_keys, shuffle_keys, value_size, num_workers
      )) {
    std::cout << "Error!\n";
  }
  return 0;
}
