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
#include <map>
#include <unordered_map>
#include <nvhashmap/map.hpp>
#include <nvhashmap/std_map_shim.hpp>
#include <nvhashmap/prefetch.hpp>
//#include <random>
#include <thread>
#include <CLI/CLI.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#include <phmap.h>
#pragma GCC diagnostic pop

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#include <absl/container/flat_hash_map.h>
#pragma GCC diagnostic pop

#if __cplusplus >= 202002L
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#include <folly/container/F14Map.h>
#pragma GCC diagnostic pop
#endif

using namespace nvhm;
#include "../utils.hpp"

/**
 * `nvhm_bench_compile_kernels` is used to control the level of compile-time optimizations.
 *    0: Default kernel only.
 * >=10: All default kernels.
 * >=20: All (available) platform kernels
 * >=30: All integer kernels.
 * >=40: All (fallback) array kernels.
 */
 constexpr static int_t nvhm_bench_compile_kernels{NVHM_BENCH_COMPILE_KERNELS};
 /**
  * `nvhm_bench_compile_probe_seqs` is used to control the level of compile-time optimizations.
  *    0: Default probe sequence only.
  * >=10: Unaligned probe squences.
  * >=20: Aligned probe sequences.
  */
 constexpr static int_t nvhm_bench_compile_probe_seqs{NVHM_BENCH_COMPILE_PROBE_SEQS};
 /**
  * `nvhm_bench_compile_map_types` is used to control the level of compile-time optimizations.
  *    0: Default map type only.
  * >=10: Std-library type and shim.
  * >=20: All platform map types.
  */
constexpr static int_t nvhm_bench_compile_map_types{NVHM_BENCH_COMPILE_MAP_TYPES};

class stopwatch {
  public:
   inline stopwatch() : begin_(std::chrono::high_resolution_clock::now()) {}
   
   inline auto elapsed() const {
     return std::chrono::high_resolution_clock::now() - begin_;
   }
 
   inline std::chrono::milliseconds elapsed_ms() const {
     return std::chrono::duration_cast<std::chrono::milliseconds>(elapsed());
   }
 
   inline friend std::ostream& operator<<(std::ostream& os, const stopwatch& c) {
     return os << c.elapsed_ms();
   }
 
  private:
   std::chrono::high_resolution_clock::time_point begin_;
 };
 
static std::atomic_int64_t num_workers_ready;

class worker {
 public:
  static constexpr int_t scratch_buf_size{2048};
 
  static void set_num_workers(int_t num_workers) {
    num_workers_ready.store(num_workers, std::memory_order_relaxed);
  }

  worker() = delete;
  worker(const worker&) = delete;
  worker& operator=(const worker&) = delete;
  inline worker(worker&&) noexcept = default;
  inline worker& operator=(worker&&) noexcept = default;

  inline worker(int_t index)
    : index_{index}, status_{true}, time_{std::chrono::milliseconds::zero()} {}

  void join() { return thread_.join(); }

  template <typename Func>
  void assign(const Func& work) {
    status_ = false;
    time_ = std::chrono::milliseconds::zero();

    thread_ = std::thread([&, work]() {
      num_workers_ready.fetch_add(-1, std::memory_order_relaxed);
      while (num_workers_ready.load(std::memory_order_relaxed) != 0) {
        // Spin lock until all workers are ready.
      }
      
      const stopwatch timer;
      std::tie(status_, count_) = work(*this);
      time_ = timer.elapsed_ms();
    });
  }

  constexpr int_t index() const { return index_; }
  constexpr bool status() const { return status_; }
  constexpr std::chrono::milliseconds time() const { return time_; }
  constexpr int_t count() const { return count_; }

  alignas(page_size) char scratch_buf[scratch_buf_size];

 private:
  int_t index_;
  bool status_;
  std::chrono::milliseconds time_;
  int_t count_;
  std::thread thread_;
};

template <typename Map, typename Key, typename PrefetchHint>
NVHM_ALWAYS_INLINE void insert_entry(Map& __restrict map, int_t /*i*/, const Key& __restrict k, PrefetchHint&& h, const char* __restrict blob, int_t blob_size) {
  using map_t = Map;
  using write_pos_t = typename map_t::write_pos;

  write_pos_t pos{[&](){
    if constexpr (std::is_same_v<PrefetchHint, std::monostate>) {
      return map.insert(k);
    } else {
      return map.insert(k, std::forward<PrefetchHint>(h));
    }
  }()};
  NVHM_ASSERT_(pos != npos);

  if constexpr (map_t::has_values) {
    map.value_at(pos) = -k;
  }
  
  if constexpr (map_t::has_blobs) {
    map.set_blob_at(pos, blob, blob_size);
  }
}

template <typename Queue, typename Map, typename Key, typename PrefetchHint = typename Map::prefetch_hint>
NVHM_NO_INLINE std::pair<bool, int_t> do_insert(worker& __restrict w, int_t batch_size, Map& __restrict map, const std::vector<Key>& __restrict keys, const std::vector<char>& __restrict blobs, int_t blob_size, int_t queue_len) {
  const int_t i0{std::min((w.index() + 0) * batch_size, to_int(keys.size()))};
  const int_t i1{std::min((w.index() + 1) * batch_size, to_int(keys.size()))};

  int_t i{i0};

  if constexpr (std::is_same_v<Queue, void>) {
    if (queue_len != 0) {
      throw std::runtime_error("queue_len != 0");
    }

    for (; i < i1; ++i) {
      insert_entry(map, i, keys[to_uint(i)], std::monostate{}, &blobs[to_uint(i * blob_size)], blob_size);
    }
  } else {
    if (queue_len > Queue::capacity) {
      throw std::runtime_error("queue_len > queue_t::capacity");
    }
    if (static_cast<int_t>(keys.size()) < Queue::capacity) {
      throw std::runtime_error("keys.size() < queue_t::capacity");
    }
    Queue q;
    
    if constexpr (Queue::type == queue_t::shift) {
      if (queue_len != Queue::capacity) {
        throw std::runtime_error("queue_len != queue_t::capacity");
      }
      for (int_t j{}; j < queue_len; ++j) {
        q.prefill_write(j, map, keys[to_uint(i0 + j)]);
      }
    } else if constexpr (Queue::type == queue_t::ring) {
      for (int_t j{}; j < queue_len; ++j) {
        q.prefill_write(map, keys[to_uint(i0 + j)]);
      }
    } else {
      static_assert(dependent_false_v<Queue>, "Invalid queue type!");
    }

    for (; i < i1 - queue_len; ++i) {
      const auto [k0, h0]{q.push_write(map,  keys[to_uint(i + queue_len)])};
      insert_entry(map, i, k0, std::move(h0), &blobs[to_uint(i * blob_size)], blob_size);
    }

    for (int_t j{}; j < queue_len; ++i, ++j) {
      const auto [k0, h0]{q.pop()};
      insert_entry(map, i, k0, std::move(h0), &blobs[to_uint(i * blob_size)], blob_size);
    }
  }

  return {true, i - i0};
}

template <typename Map, typename Key, typename PrefetchHint>
NVHM_ALWAYS_INLINE bool find_and_verify(worker& __restrict w, const Map& __restrict map, int_t i, const Key& __restrict k, PrefetchHint&& h, int_t blob_size, bool should_exist, bool check_blobs) {
  using map_t = Map;
  using read_pos_t = typename map_t::read_pos;
  
  read_pos_t pos{[&](){
    if constexpr (std::is_same_v<PrefetchHint, std::monostate>) {
      return map.find(k);
    } else {
      return map.find(k, std::forward<PrefetchHint>(h));
    }
  }()};

  if (should_exist) {
    if (pos == npos) {
      std::cerr << "Unexpected find error! i = " << i << ", k = " << k << " (line: " << __LINE__ << ")\n";
      return false;
    }

    if constexpr (map_t::has_values) {
      if (map.value_at(pos) != -k) {
        std::cerr << "Value error! i = " << i << ", k = " << k << " (line: " << __LINE__ << ")\n";
        return false;
      }
    }

    if constexpr (map_t::has_blobs) {
      map.get_blob_at(pos, w.scratch_buf, blob_size);

      if (check_blobs) {
        for (int_t j{}; j < blob_size; ++j) {
          if (w.scratch_buf[j] != static_cast<char>(i + j + k)) {
            std::cerr << "Blob error! i = " << i << ", k = " << k << " (line: " << __LINE__ << ")\n";
            return false;
          }
        }
      }
    }
  } else if (pos != npos) {
    std::cerr << "Miss error! i = " << i << ", k = " << k << " (line: " << __LINE__ << ")\n";
    return false;
  }

  return true;
}

template <typename Queue, typename Map, typename Key, typename PrefetchHint = typename Map::prefetch_hint>
NVHM_NO_INLINE std::pair<bool, int_t> do_find(worker& __restrict w, int_t batch_size, const Map& __restrict map, const std::vector<Key>& __restrict keys, int_t blob_size, std::vector<bool>& __restrict should_exist, bool check_blobs, int_t queue_len) {
  const int_t i0{std::min((w.index() + 0) * batch_size, to_int(keys.size()))};
  const int_t i1{std::min((w.index() + 1) * batch_size, to_int(keys.size()))};

  bool b{true};
  int_t n{};

  if constexpr (std::is_same_v<Queue, void>) {
    if (queue_len != 0) {
      throw std::runtime_error("queue_len != 0");
    }
    for (int_t i{i0}; i < i1; ++i) {
      b = should_exist[to_uint(i)];
      n += b;
      b = find_and_verify(w, map, i, keys[to_uint(i)], std::monostate{}, blob_size, b, check_blobs);
      if (!b) break;
    }
  } else {
    if (queue_len > Queue::capacity) {
      throw std::runtime_error("queue_len > queue_t::capacity");
    }
    if (static_cast<int_t>(keys.size()) < queue_len) {
      throw std::runtime_error("keys.size() < queue_len");
    }
    Queue q;

    if constexpr (Queue::type == queue_t::shift) {
      if (queue_len != Queue::capacity) {
        throw std::runtime_error("queue_len != queue_t::capacity");
      }
      for (int_t j{}; j < Queue::capacity; ++j) {
        q.prefill_read(j, map, keys[to_uint(i0 + j)]);
      }
    } else if constexpr (Queue::type == queue_t::ring) {
      for (int_t j{}; j < queue_len; ++j) {
        q.prefill_read(map, keys[to_uint(i0 + j)]);
      }
    } else {
      static_assert(dependent_false_v<Queue>, "Invalid queue type!");
    }

    int_t i{i0};
    for (; i < i1 - queue_len; ++i) {
      const auto [k0, h0]{q.push_read(map, keys[to_uint(i + queue_len)])};
      b = should_exist[to_uint(i)];
      n += b;
      b = find_and_verify(w, map, i, k0, std::move(h0), blob_size, b, check_blobs);
      if (!b) return {b, n};
    }
    
    for (int_t j{}; j < queue_len; ++i, ++j) {
      const auto [k0, h0]{q.pop()};
      b = should_exist[to_uint(i)];
      n += b;
      b = find_and_verify(w, map, i, k0, std::move(h0), blob_size, b, check_blobs);
      if (!b) break;
    }
  }

  return {b, n};
}

template <typename It>
NVHM_NO_INLINE std::pair<std::chrono::milliseconds, int_t> accumulate_workers(It begin, It last) {
  int_t num_workers{last - begin};

  bool success{true};
  std::chrono::milliseconds max{};
  int_t count{};
  for (; begin != last; ++begin) {
    begin->join();

    max = std::max(max, begin->time());
    count += begin->count();

    if (!begin->status()) {
      std::cerr << "Worker #" << begin->index() << " failed!";
      success = false;
    }
  }
  
  if (!success) {
    return {std::chrono::milliseconds::zero(), -1};
  }
  return {max / num_workers, count};
}

template <typename Map>
NVHM_NO_INLINE void bench_nvhm_map(
  const int_t num_keys, const bool shuffle_keys, const int_t blob_size, 
  const int_t num_workers, const int_t num_trials,
  const queue_t insert_queue_type, const int_t max_insert_queue_len,
  const int_t find_keep_perc, const queue_t find_queue_type, const int_t max_find_queue_len, const bool check_blobs,
  std::mt19937_64& rng
) {
  using map_t = Map;
  using conf_t = typename map_t::conf_type;
  using key_t = typename map_t::key_type;
  using hint_t = typename map_t::prefetch_hint;
  
  if (num_workers < 1 || num_workers > 1024) {
    throw std::runtime_error("`num_workers` is out of bounds!");
  }
  if (find_keep_perc < 0 || find_keep_perc > 100) {
    throw std::runtime_error("`prune_perc` is out of bounds!");
  }
  if (worker::scratch_buf_size < blob_size) {
    throw std::runtime_error("`scratch_buf_size` is too small!");
  }

  const std::string map_type{type_to_string<map_t>()};

  // Initialize key and data buffers.
  std::vector<worker> workers;
  workers.reserve(to_uint(num_workers));
  for (int_t i{}; i < num_workers; ++i) {
    workers.emplace_back(i);
  }
  
  std::vector<key_t> keys(to_uint(num_keys));
  std::iota(keys.begin(), keys.end(), 1337);

  if (shuffle_keys) {
    std::shuffle(keys.begin(), keys.end(), rng);
  }

  std::vector<char> blobs;
  conf_t conf{};
  if constexpr (map_t::has_blobs) {
    blobs.resize(to_uint(num_keys * blob_size));
    for (int_t i{}; i < num_keys; ++i) {
      for (int_t j{}; j < blob_size; ++j) {
        blobs[to_uint(i * blob_size + j)] += static_cast<char>(i + j + keys[to_uint(i)]);
      }
    }

    conf.set_blob(blob_size);
  }
  map_t map(conf);

  // Insert
  for (int_t queue_len{}; queue_len < 1 + max_insert_queue_len; ++queue_len) {
    std::cout << map_type << ", " << std::setw(10) << "Insert " << 100 << "% (" << insert_queue_type << ' ' << std::setw(2) << queue_len << "): ";
    std::cout.flush();

    int_t total_count{};
    for (int_t trial{}; trial < num_trials; ++trial) {
      const int_t batch_size{ceil_div<int_t>(num_keys, 1)};

      std::function<std::pair<bool, int_t>(worker&)> fn;
      if (queue_len == 0) {
        fn = [&](worker& w) { return do_insert<void>(w, batch_size, map, keys, blobs, blob_size, 0); };
      } else {
        switch (insert_queue_type) {
          case queue_t::shift:
            switch (queue_len) {
              case  1: fn = [&](worker& w) { return do_insert<shift_prefetch_queue<key_t, hint_t,  1>>(w, batch_size, map, keys, blobs, blob_size,  1); }; break;
              case  2: fn = [&](worker& w) { return do_insert<shift_prefetch_queue<key_t, hint_t,  2>>(w, batch_size, map, keys, blobs, blob_size,  2); }; break;
              case  3: fn = [&](worker& w) { return do_insert<shift_prefetch_queue<key_t, hint_t,  3>>(w, batch_size, map, keys, blobs, blob_size,  3); }; break;
              case  4: fn = [&](worker& w) { return do_insert<shift_prefetch_queue<key_t, hint_t,  4>>(w, batch_size, map, keys, blobs, blob_size,  4); }; break;
              case  5: fn = [&](worker& w) { return do_insert<shift_prefetch_queue<key_t, hint_t,  5>>(w, batch_size, map, keys, blobs, blob_size,  5); }; break;
              case  6: fn = [&](worker& w) { return do_insert<shift_prefetch_queue<key_t, hint_t,  6>>(w, batch_size, map, keys, blobs, blob_size,  6); }; break;
              case  7: fn = [&](worker& w) { return do_insert<shift_prefetch_queue<key_t, hint_t,  7>>(w, batch_size, map, keys, blobs, blob_size,  7); }; break;
              case  8: fn = [&](worker& w) { return do_insert<shift_prefetch_queue<key_t, hint_t,  8>>(w, batch_size, map, keys, blobs, blob_size,  8); }; break;
              case  9: fn = [&](worker& w) { return do_insert<shift_prefetch_queue<key_t, hint_t,  9>>(w, batch_size, map, keys, blobs, blob_size,  9); }; break;
              case 10: fn = [&](worker& w) { return do_insert<shift_prefetch_queue<key_t, hint_t, 10>>(w, batch_size, map, keys, blobs, blob_size, 10); }; break;
              case 11: fn = [&](worker& w) { return do_insert<shift_prefetch_queue<key_t, hint_t, 11>>(w, batch_size, map, keys, blobs, blob_size, 11); }; break;
              case 12: fn = [&](worker& w) { return do_insert<shift_prefetch_queue<key_t, hint_t, 12>>(w, batch_size, map, keys, blobs, blob_size, 12); }; break;
              case 13: fn = [&](worker& w) { return do_insert<shift_prefetch_queue<key_t, hint_t, 13>>(w, batch_size, map, keys, blobs, blob_size, 13); }; break;
              case 14: fn = [&](worker& w) { return do_insert<shift_prefetch_queue<key_t, hint_t, 14>>(w, batch_size, map, keys, blobs, blob_size, 14); }; break;
              case 15: fn = [&](worker& w) { return do_insert<shift_prefetch_queue<key_t, hint_t, 15>>(w, batch_size, map, keys, blobs, blob_size, 15); }; break;
              case 16: fn = [&](worker& w) { return do_insert<shift_prefetch_queue<key_t, hint_t, 16>>(w, batch_size, map, keys, blobs, blob_size, 16); }; break;
              default: throw std::runtime_error("`queue_len` is out of bounds!");
            }
            break;
          case queue_t::ring:
            if (queue_len <= 4) {
              fn = [&](worker& w) { return do_insert<ring_prefetch_queue<key_t, hint_t, 4>>(w, batch_size, map, keys, blobs, blob_size, queue_len); };
            } else if (queue_len <= 8) {
              fn = [&](worker& w) { return do_insert<ring_prefetch_queue<key_t, hint_t, 8>>(w, batch_size, map, keys, blobs, blob_size, queue_len); };
            } else if (queue_len <= 16) {
              fn = [&](worker& w) { return do_insert<ring_prefetch_queue<key_t, hint_t, 16>>(w, batch_size, map, keys, blobs, blob_size, queue_len); };
            } else if (queue_len <= 32) {
              fn = [&](worker& w) { return do_insert<ring_prefetch_queue<key_t, hint_t, 32>>(w, batch_size, map, keys, blobs, blob_size, queue_len); };
            } else if (queue_len <= 64) {
              fn = [&](worker& w) { return do_insert<ring_prefetch_queue<key_t, hint_t, 64>>(w, batch_size, map, keys, blobs, blob_size, queue_len); };
            } else {
              throw std::runtime_error("`queue_len` is out of bounds!");
            }
            break;
          default:
            throw std::runtime_error("Unsupported `queue_type`!");
        }
      }
      if (fn) {
        map.clear();
        worker::set_num_workers(1);
        workers[0].assign(fn);
        auto [ms, count]{accumulate_workers(workers.begin(), workers.begin() + 1)};
        std::cout << (trial ? ", " : "") << std::setw(5) << ms;
        std::cout.flush();
        total_count += count;
      }
    }
    std::cout << ", size: " << map.size() << " / " << map.capacity() << " [ " << total_count << " ]" << '\n';
  }
  if (!map.check_integrity()) {
    throw std::runtime_error("Map integrity check failed!");
  }

  // Set half of the bits.
  std::vector<size_t> valid_indexes(keys.size());
  std::iota(valid_indexes.begin(), valid_indexes.end(), 0);
  std::shuffle(valid_indexes.begin(), valid_indexes.end(), rng);

  std::cout << "map size after pruning: " << map.size();
  std::vector<bool> should_exist(keys.size());
  size_t i{};
  for (; i < valid_indexes.size(); ++i) {
    if (i < valid_indexes.size() * to_uint(find_keep_perc) / 100) {
      should_exist[valid_indexes[i]] = true;
    } else {
      should_exist[valid_indexes[i]] = false;
      map.erase(keys[valid_indexes[i]]);
    }
  }
  std::cout << " -> " << map.size() << '\n';
  if (!map.check_integrity()) {
    throw std::runtime_error("Map integrity check failed!");
  }

  // Find
  for (int_t queue_len{}; queue_len < 1 + max_find_queue_len; ++queue_len) {
    std::cout << map_type << ", " << std::setw(10) << "Find " << find_keep_perc << "% hit (" << find_queue_type << ' ' << std::setw(2) << queue_len << "): ";
    std::cout.flush();

    int_t total_count{};
    for (int_t trial{}; trial < num_trials; ++trial) {
      const int_t batch_size{ceil_div(num_keys, num_workers)};

      std::function<std::pair<bool, int_t>(worker&)> fn;
      if (queue_len == 0) {
        fn = [&](worker& w_ctx) { return do_find<void>(w_ctx, batch_size, map, keys, blob_size, should_exist, check_blobs, 0); };
      } else {
        switch (find_queue_type) {
          case queue_t::shift:
            switch (queue_len) {
              case  1: fn = [&](worker& w) { return do_find<shift_prefetch_queue<key_t, hint_t,  1>>(w, batch_size, map, keys, blob_size, should_exist, check_blobs,  1); }; break;
              case  2: fn = [&](worker& w) { return do_find<shift_prefetch_queue<key_t, hint_t,  2>>(w, batch_size, map, keys, blob_size, should_exist, check_blobs,  2); }; break;
              case  3: fn = [&](worker& w) { return do_find<shift_prefetch_queue<key_t, hint_t,  3>>(w, batch_size, map, keys, blob_size, should_exist, check_blobs,  3); }; break;
              case  4: fn = [&](worker& w) { return do_find<shift_prefetch_queue<key_t, hint_t,  4>>(w, batch_size, map, keys, blob_size, should_exist, check_blobs,  4); }; break;
              case  5: fn = [&](worker& w) { return do_find<shift_prefetch_queue<key_t, hint_t,  5>>(w, batch_size, map, keys, blob_size, should_exist, check_blobs,  5); }; break;
              case  6: fn = [&](worker& w) { return do_find<shift_prefetch_queue<key_t, hint_t,  6>>(w, batch_size, map, keys, blob_size, should_exist, check_blobs,  6); }; break;
              case  7: fn = [&](worker& w) { return do_find<shift_prefetch_queue<key_t, hint_t,  7>>(w, batch_size, map, keys, blob_size, should_exist, check_blobs,  7); }; break;
              case  8: fn = [&](worker& w) { return do_find<shift_prefetch_queue<key_t, hint_t,  8>>(w, batch_size, map, keys, blob_size, should_exist, check_blobs,  8); }; break;
              case  9: fn = [&](worker& w) { return do_find<shift_prefetch_queue<key_t, hint_t,  9>>(w, batch_size, map, keys, blob_size, should_exist, check_blobs,  9); }; break;
              case 10: fn = [&](worker& w) { return do_find<shift_prefetch_queue<key_t, hint_t, 10>>(w, batch_size, map, keys, blob_size, should_exist, check_blobs, 10); }; break;
              case 11: fn = [&](worker& w) { return do_find<shift_prefetch_queue<key_t, hint_t, 11>>(w, batch_size, map, keys, blob_size, should_exist, check_blobs, 11); }; break;
              case 12: fn = [&](worker& w) { return do_find<shift_prefetch_queue<key_t, hint_t, 12>>(w, batch_size, map, keys, blob_size, should_exist, check_blobs, 12); }; break;
              case 13: fn = [&](worker& w) { return do_find<shift_prefetch_queue<key_t, hint_t, 13>>(w, batch_size, map, keys, blob_size, should_exist, check_blobs, 13); }; break;
              case 14: fn = [&](worker& w) { return do_find<shift_prefetch_queue<key_t, hint_t, 14>>(w, batch_size, map, keys, blob_size, should_exist, check_blobs, 14); }; break;
              case 15: fn = [&](worker& w) { return do_find<shift_prefetch_queue<key_t, hint_t, 15>>(w, batch_size, map, keys, blob_size, should_exist, check_blobs, 15); }; break;
              case 16: fn = [&](worker& w) { return do_find<shift_prefetch_queue<key_t, hint_t, 16>>(w, batch_size, map, keys, blob_size, should_exist, check_blobs, 16); }; break;
              default: throw std::runtime_error("`queue_len` is out of bounds!");
            }
            break;
          case queue_t::ring:
            if (queue_len <= 4) {
              fn = [&](worker& w) { return do_find<ring_prefetch_queue<key_t, hint_t, 4>>(w, batch_size, map, keys, blob_size, should_exist, check_blobs, queue_len); };
            } else if (queue_len <= 8) {
              fn = [&](worker& w) { return do_find<ring_prefetch_queue<key_t, hint_t, 8>>(w, batch_size, map, keys, blob_size, should_exist, check_blobs, queue_len); };
            } else if (queue_len <= 16) {
              fn = [&](worker& w) { return do_find<ring_prefetch_queue<key_t, hint_t, 16>>(w, batch_size, map, keys, blob_size, should_exist, check_blobs, queue_len); };
            } else if (queue_len <= 32) {
              fn = [&](worker& w) { return do_find<ring_prefetch_queue<key_t, hint_t, 32>>(w, batch_size, map, keys, blob_size, should_exist, check_blobs, queue_len); };
            } else if (queue_len <= 64) {
              fn = [&](worker& w) { return do_find<ring_prefetch_queue<key_t, hint_t, 64>>(w, batch_size, map, keys, blob_size, should_exist, check_blobs, queue_len); };
            } else {
              throw std::runtime_error("`queue_len` is out of bounds!");
            }
            break;
          default:
            throw std::runtime_error("Unsupported `queue_type`!");
        }
      }
      if (fn) {
        worker::set_num_workers(num_workers);
        for (int_t w{}; w < num_workers; ++w) {
          workers[to_uint(w)].assign(fn);
        }
        auto [ms, count]{accumulate_workers(workers.begin(), workers.end())};
        std::cout << (trial ? ", " : "") << std::setw(5) << ms;
        std::cout.flush();
        total_count += count;
      }
    }
    std::cout << "; count: " << total_count << '\n';
  }
}

#define NVHM_BENCH_MAP_ARGS_()\
  num_keys, shuffle_keys, blob_size,\
  num_workers, num_trials,\
  insert_queue_type, max_insert_queue_len,\
  find_keep_perc, find_queue_type, max_find_queue_len, check_blobs,\
  rng

template <typename Key, typename Value, flags_t Flags, typename Kernel>
void run_bench_nvhm_map_5(
  int_t num_keys, bool shuffle_keys, int_t blob_size,
  int_t num_workers, int_t num_trials,
  queue_t insert_queue_type, int_t max_insert_queue_len,
  int_t find_keep_perc, queue_t find_queue_type, int_t max_find_queue_len, bool check_blobs,
  std::mt19937_64& rng,
  const std::string& probe_seq_type
) {
  if (probe_seq_type == "default") {
    return bench_nvhm_map<map<Key, Value, Flags, Kernel, default_seq_t>>(NVHM_BENCH_MAP_ARGS_());
  }
  if constexpr (nvhm_bench_compile_probe_seqs >= 10) {
    if (probe_seq_type == "linear") {
      return bench_nvhm_map<map<Key, Value, Flags, Kernel, linear_seq<0>>>(NVHM_BENCH_MAP_ARGS_());
    }
    if (probe_seq_type == "quadratic") {
      return bench_nvhm_map<map<Key, Value, Flags, Kernel, quadratic_seq<0>>>(NVHM_BENCH_MAP_ARGS_());
    }
  }
  if constexpr (nvhm_bench_compile_probe_seqs >= 20) {
    if (probe_seq_type == "aligned_linear") {
      return bench_nvhm_map<map<Key, Value, Flags, Kernel, linear_seq<cache_line_size>>>(NVHM_BENCH_MAP_ARGS_());
    }
    if (probe_seq_type == "aligned_quadratic") {
      return bench_nvhm_map<map<Key, Value, Flags, Kernel, quadratic_seq<cache_line_size>>>(NVHM_BENCH_MAP_ARGS_());
    }
  }
  throw std::runtime_error("Invalid probe sequence type: " + probe_seq_type + ". Did you forget to compile with `NVHM_BENCH_COMPILE_PROBE_SEQS`?");
}

#define NVHM_BENCH_MAP_ARGS_5_()\
  num_keys, shuffle_keys, blob_size,\
  num_workers, num_trials,\
  insert_queue_type, max_insert_queue_len,\
  find_keep_perc, find_queue_type, max_find_queue_len, check_blobs,\
  rng,\
  probe_seq_type

template <typename Key, typename Value, flags_t Flags>
void run_bench_nvhm_map_4(
  int_t num_keys, bool shuffle_keys, int_t blob_size,
  int_t num_workers, int_t num_trials,
  queue_t insert_queue_type, int_t max_insert_queue_len,
  int_t find_keep_perc, queue_t find_queue_type, int_t max_find_queue_len, bool check_blobs,
  std::mt19937_64& rng,
  const std::string& kernel_type, const std::string& probe_seq_type
) {
  if (kernel_type == "default") {
    return run_bench_nvhm_map_5<Key, Value, Flags, default_kernel_t<>>(NVHM_BENCH_MAP_ARGS_5_());
  }
  if constexpr (nvhm_bench_compile_kernels >= 10) {
    if (kernel_type == "default1") {
      return run_bench_nvhm_map_5<Key, Value, Flags, default_kernel_t<1>>(NVHM_BENCH_MAP_ARGS_5_());
    } else if (kernel_type == "default2") {
      return run_bench_nvhm_map_5<Key, Value, Flags, default_kernel_t<2>>(NVHM_BENCH_MAP_ARGS_5_());
    } else if (kernel_type == "default4") {
      return run_bench_nvhm_map_5<Key, Value, Flags, default_kernel_t<4>>(NVHM_BENCH_MAP_ARGS_5_());
    } else if (kernel_type == "default8") {
      return run_bench_nvhm_map_5<Key, Value, Flags, default_kernel_t<8>>(NVHM_BENCH_MAP_ARGS_5_());
    } else if (kernel_type == "default16") {
      return run_bench_nvhm_map_5<Key, Value, Flags, default_kernel_t<16>>(NVHM_BENCH_MAP_ARGS_5_());
    } else if (kernel_type == "default32") {
      return run_bench_nvhm_map_5<Key, Value, Flags, default_kernel_t<32>>(NVHM_BENCH_MAP_ARGS_5_());
    } else if (kernel_type == "default64") {
      return run_bench_nvhm_map_5<Key, Value, Flags, default_kernel_t<64>>(NVHM_BENCH_MAP_ARGS_5_());
    } else if (kernel_type == "default128") {
      return run_bench_nvhm_map_5<Key, Value, Flags, default_kernel_t<128>>(NVHM_BENCH_MAP_ARGS_5_());
    } else if (kernel_type == "default256") {
      return run_bench_nvhm_map_5<Key, Value, Flags, default_kernel_t<256>>(NVHM_BENCH_MAP_ARGS_5_());
    } else if (kernel_type == "default512") {
      return run_bench_nvhm_map_5<Key, Value, Flags, default_kernel_t<512>>(NVHM_BENCH_MAP_ARGS_5_());
    }
  }

  if constexpr (nvhm_bench_compile_kernels >= 20) {
    #if NVHM_WITH_SSE >= 2
    if (kernel_type == "sse") {
      return run_bench_nvhm_map_5<Key, Value, Flags, sse_kernel_t>(NVHM_BENCH_MAP_ARGS_5_());
    }
    #endif
    #if NVHM_WITH_AVX >= 2
    if (kernel_type == "avx") {
      return run_bench_nvhm_map_5<Key, Value, Flags, avx_kernel_t>(NVHM_BENCH_MAP_ARGS_5_());
    }
    #endif
    #if NVHM_WITH_AVX512
    if (kernel_type == "avx512") {
      return run_bench_nvhm_map_5<Key, Value, Flags, avx512_kernel_t>(NVHM_BENCH_MAP_ARGS_5_());
    }
    #endif
    #if NVHM_WITH_NEON
    if (kernel_type == "neon8") {
      return run_bench_nvhm_map_5<Key, Value, Flags, neon_kernel8_t>(NVHM_BENCH_MAP_ARGS_5_());
    }
    if (kernel_type == "neon16") {
      return run_bench_nvhm_map_5<Key, Value, Flags, neon_kernel16_t>(NVHM_BENCH_MAP_ARGS_5_());
    }
    if (kernel_type == "neon32") {
      return run_bench_nvhm_map_5<Key, Value, Flags, neon_kernel32_t>(NVHM_BENCH_MAP_ARGS_5_());
    }
    if (kernel_type == "neon64") {
      return run_bench_nvhm_map_5<Key, Value, Flags, neon_kernel64_t>(NVHM_BENCH_MAP_ARGS_5_());
    }
    #endif
    #if NVHM_WITH_SVE
    #if NVHM_WITH_SVE_SIZE >= 1
    if (kernel_type == "sve1") {
      return run_bench_nvhm_map_5<Key, Value, Flags, sve_kernel1_t>(NVHM_BENCH_MAP_ARGS_5_());
    }
    #endif
    #if NVHM_WITH_SVE_SIZE >= 2
    if (kernel_type == "sve2") {
      return run_bench_nvhm_map_5<Key, Value, Flags, sve_kernel2_t>(NVHM_BENCH_MAP_ARGS_5_());
    }
    #endif
    #if NVHM_WITH_SVE_SIZE >= 4
    if (kernel_type == "sve4") {
      return run_bench_nvhm_map_5<Key, Value, Flags, sve_kernel4_t>(NVHM_BENCH_MAP_ARGS_5_());
    }
    #endif
    #if NVHM_WITH_SVE_SIZE >= 8
    if (kernel_type == "sve8") {
      return run_bench_nvhm_map_5<Key, Value, Flags, sve_kernel8_t>(NVHM_BENCH_MAP_ARGS_5_());
    }
    #endif
    #if NVHM_WITH_SVE_SIZE >= 16
    if (kernel_type == "sve16") {
      return run_bench_nvhm_map_5<Key, Value, Flags, sve_kernel16_t>(NVHM_BENCH_MAP_ARGS_5_());
    }
    #endif
    #if NVHM_WITH_SVE_SIZE >= 32
    if (kernel_type == "sve32") {
      return run_bench_nvhm_map_5<Key, Value, Flags, sve_kernel32_t>(NVHM_BENCH_MAP_ARGS_5_());
    }
    #endif
    #if NVHM_WITH_SVE_SIZE >= 64
    if (kernel_type == "sve64") {
      return run_bench_nvhm_map_5<Key, Value, Flags, sve_kernel64_t>(NVHM_BENCH_MAP_ARGS_5_());
    }
    #endif
    #if NVHM_WITH_SVE_SIZE >= 128
    if (kernel_type == "sve128") {
      return run_bench_nvhm_map_5<Key, Value, Flags, sve_kernel128_t>(NVHM_BENCH_MAP_ARGS_5_());
    }
    #endif
    #if NVHM_WITH_SVE_SIZE >= 256
    if (kernel_type == "sve256") {
      return run_bench_nvhm_map_5<Key, Value, Flags, sve_kernel256_t>(NVHM_BENCH_MAP_ARGS_5_());
    }
    #endif
    #endif
  }

  if constexpr (nvhm_bench_compile_kernels >= 30) {
    if (kernel_type == "uint1") {
      return run_bench_nvhm_map_5<Key, Value, Flags, uint_kernel1_t>(NVHM_BENCH_MAP_ARGS_5_());
    } else if (kernel_type == "uint2") {
      return run_bench_nvhm_map_5<Key, Value, Flags, uint_kernel2_t>(NVHM_BENCH_MAP_ARGS_5_());
    } else if (kernel_type == "uint4") {
      return run_bench_nvhm_map_5<Key, Value, Flags, uint_kernel4_t>(NVHM_BENCH_MAP_ARGS_5_());
    } else if (kernel_type == "uint8") {
      return run_bench_nvhm_map_5<Key, Value, Flags, uint_kernel8_t>(NVHM_BENCH_MAP_ARGS_5_());
    } else if (kernel_type == "uint16") {
      return run_bench_nvhm_map_5<Key, Value, Flags, uint_kernel16_t>(NVHM_BENCH_MAP_ARGS_5_());
    }

    if (kernel_type == "fast_uint1") {
      return run_bench_nvhm_map_5<Key, Value, Flags, fast_uint_kernel1_t>(NVHM_BENCH_MAP_ARGS_5_());
    } else if (kernel_type == "fast_uint2") {
      return run_bench_nvhm_map_5<Key, Value, Flags, fast_uint_kernel2_t>(NVHM_BENCH_MAP_ARGS_5_());
    } else if (kernel_type == "fast_uint4") {
      return run_bench_nvhm_map_5<Key, Value, Flags, fast_uint_kernel4_t>(NVHM_BENCH_MAP_ARGS_5_());
    } else if (kernel_type == "fast_uint8") {
      return run_bench_nvhm_map_5<Key, Value, Flags, fast_uint_kernel8_t>(NVHM_BENCH_MAP_ARGS_5_());
    } else if (kernel_type == "fast_uint16") {
      return run_bench_nvhm_map_5<Key, Value, Flags, fast_uint_kernel16_t>(NVHM_BENCH_MAP_ARGS_5_());
    }
  }

  if constexpr (nvhm_bench_compile_kernels >= 40) {
    if (kernel_type == "array1") {
      return run_bench_nvhm_map_5<Key, Value, Flags, array_kernel1_t>(NVHM_BENCH_MAP_ARGS_5_());
    } else if (kernel_type == "array2") {
      return run_bench_nvhm_map_5<Key, Value, Flags, array_kernel2_t>(NVHM_BENCH_MAP_ARGS_5_());
    } else if (kernel_type == "array4") {
      return run_bench_nvhm_map_5<Key, Value, Flags, array_kernel4_t>(NVHM_BENCH_MAP_ARGS_5_());
    } else if (kernel_type == "array8") {
      return run_bench_nvhm_map_5<Key, Value, Flags, array_kernel8_t>(NVHM_BENCH_MAP_ARGS_5_());
    } else if (kernel_type == "array16") {
      return run_bench_nvhm_map_5<Key, Value, Flags, array_kernel16_t>(NVHM_BENCH_MAP_ARGS_5_());
    } else if (kernel_type == "array32") {
      return run_bench_nvhm_map_5<Key, Value, Flags, array_kernel32_t>(NVHM_BENCH_MAP_ARGS_5_());
    } else if (kernel_type == "array64") {
      return run_bench_nvhm_map_5<Key, Value, Flags, array_kernel64_t>(NVHM_BENCH_MAP_ARGS_5_());
    } else if (kernel_type == "array128") {
      return run_bench_nvhm_map_5<Key, Value, Flags, array_kernel128_t>(NVHM_BENCH_MAP_ARGS_5_());
    } else if (kernel_type == "array256") {
      return run_bench_nvhm_map_5<Key, Value, Flags, array_kernel256_t>(NVHM_BENCH_MAP_ARGS_5_());
    } else if (kernel_type == "array512") {
      return run_bench_nvhm_map_5<Key, Value, Flags, array_kernel512_t>(NVHM_BENCH_MAP_ARGS_5_());
    }
  }

  throw std::runtime_error("Invalid kernel type: " + kernel_type + ". Did you forget to compile with `NVHM_BENCH_COMPILE_KERNELS`?");
}

#define NVHM_BENCH_MAP_ARGS_3_AND_4_()\
  num_keys, shuffle_keys, blob_size,\
  num_workers, num_trials,\
  insert_queue_type, max_insert_queue_len,\
  find_keep_perc, find_queue_type, max_find_queue_len, check_blobs,\
  rng,\
  kernel_type, probe_seq_type

template <typename Key, typename Value, flags_t Flags>
void run_bench_nvhm_map_3(
  int_t num_keys, bool shuffle_keys, int_t blob_size,
  int_t num_workers, int_t num_trials,
  queue_t insert_queue_type, int_t max_insert_queue_len,
  int_t find_keep_perc, queue_t find_queue_type, int_t max_find_queue_len, bool check_blobs,
  std::mt19937_64& rng,
  const std::string& kernel_type, const std::string& probe_seq_type) {
  if (blob_size > 0) {
    run_bench_nvhm_map_4<Key, Value, Flags | flags_t::blobs>(NVHM_BENCH_MAP_ARGS_3_AND_4_());
  } else {
    run_bench_nvhm_map_4<Key, Value, Flags>(NVHM_BENCH_MAP_ARGS_3_AND_4_());
  }
}

template <typename Key, typename Value>
void run_bench_nvhm_map_2(
  int_t num_keys, bool shuffle_keys, bool aggressive_prefetch, int_t blob_size,
  int_t num_workers, int_t num_trials,
  queue_t insert_queue_type, int_t max_insert_queue_len,
  int_t find_keep_perc, queue_t find_queue_type, int_t max_find_queue_len, bool check_blobs,
  std::mt19937_64& rng,
  const std::string& kernel_type, const std::string& probe_seq_type) {
  if (aggressive_prefetch) {
    run_bench_nvhm_map_3<Key, Value, flags_t::aggressive_prefetch>(NVHM_BENCH_MAP_ARGS_3_AND_4_());
  } else {
    run_bench_nvhm_map_3<Key, Value, flags_t::none>(NVHM_BENCH_MAP_ARGS_3_AND_4_());
  }
}

#define NVHM_BENCH_MAP_ARGS_2_()\
  num_keys, shuffle_keys, aggressive_prefetch, blob_size,\
  num_workers, num_trials,\
  insert_queue_type, max_insert_queue_len,\
  find_keep_perc, find_queue_type, max_find_queue_len, check_blobs,\
  rng,\
  kernel_type, probe_seq_type

template <typename Key>
void run_bench_nvhm_map_1(
  int_t num_keys, bool shuffle_keys, bool aggressive_prefetch, const std::string& value_type, int_t blob_size,
  int_t num_workers, int_t num_trials,
  queue_t insert_queue_type, int_t max_insert_queue_len,
  int_t find_keep_perc, queue_t find_queue_type, int_t max_find_queue_len, bool check_blobs,
  std::mt19937_64& rng,
  const std::string& kernel_type, const std::string& probe_seq_type) {
  if (value_type == "time") {
    run_bench_nvhm_map_2<Key, time_t>(NVHM_BENCH_MAP_ARGS_2_());
  } else if (value_type == "void") {
    run_bench_nvhm_map_2<Key, void>(NVHM_BENCH_MAP_ARGS_2_());
  } else {
    throw std::runtime_error("Invalid value type: " + value_type);
  }
}

#define NVHM_BENCH_MAP_ARGS_1_()\
  num_keys, shuffle_keys, aggressive_prefetch, value_type, blob_size,\
  num_workers, num_trials,\
  insert_queue_type, max_insert_queue_len,\
  find_keep_perc, find_queue_type, max_find_queue_len, check_blobs,\
  rng,\
  kernel_type, probe_seq_type

void run_bench_nvhm_map_0(
  const std::string& key_type, int_t num_keys, bool shuffle_keys, bool aggressive_prefetch, const std::string& value_type, int_t blob_size,
  int_t num_workers, int_t num_trials,
  queue_t insert_queue_type, int_t max_insert_queue_len,
  int_t find_keep_perc, queue_t find_queue_type, int_t max_find_queue_len, bool check_blobs,
  std::mt19937_64& rng,
  const std::string& kernel_type, const std::string& probe_seq_type) {
  if (key_type == "int32") {
    run_bench_nvhm_map_1<int32_t>(NVHM_BENCH_MAP_ARGS_1_());
  } else if (key_type == "int64") {
    run_bench_nvhm_map_1<int64_t>(NVHM_BENCH_MAP_ARGS_1_());
  } else {
    throw std::runtime_error("Invalid key type: " + key_type);
  }
}

template <typename Value, bool HasBlobs, bool SerializeCopy, typename Map, typename Key>
NVHM_ALWAYS_INLINE void insert_entry_std(
  Map& __restrict map, int_t /*i*/, const Key& __restrict k, std::byte* __restrict slot,
  const char* __restrict blob, int_t blob_size) {
  auto [it, success]{map.emplace(k, slot)};
  NVHM_ASSERT_(success);
  if constexpr (SerializeCopy) {
    slot = it->second;
  }

  if constexpr (HasBlobs) {
    std::memcpy(slot, blob, to_uint(blob_size));
  }

  if constexpr (std::is_same_v<Value, void>) {
  } else if constexpr (std::is_same_v<Value, time_t>) {
    *reinterpret_cast<time_t*>(&slot[blob_size]) = -k;
  } else {
    static_assert(dependent_false_v<Value>, "Unsupported `Value` type!");
  }
}

template <typename Value, bool HasBlobs, bool SerializeCopy, typename Map, typename Key>
NVHM_NO_INLINE std::pair<bool, int_t> do_insert_std(
  worker& __restrict w, int_t batch_size, Map& __restrict map, const std::vector<Key>& __restrict keys, 
  const std::vector<std::byte*>& __restrict slots, const std::vector<char>& __restrict blobs, int_t blob_size) {
  const int_t i0{std::min((w.index() + 0) * batch_size, to_int(keys.size()))};
  const int_t i1{std::min((w.index() + 1) * batch_size, to_int(keys.size()))};

  int_t i{i0};
  for (; i < i1; ++i) {
    insert_entry_std<Value, HasBlobs, SerializeCopy>(map, i, keys[to_uint(i)], slots[to_uint(i)], &blobs[to_uint(i * blob_size)], blob_size);
  }

  return {true, i - i0};
}

template <typename Value, bool HasBlobs, typename Map, typename It, typename Key>
NVHM_ALWAYS_INLINE bool find_and_verify_std(
  worker& __restrict w, const Map& __restrict map, const It& __restrict map_end,
  int_t i, const Key& __restrict k, int_t blob_size, bool should_exist, bool check_blobs) {
  auto it{map.find(k)};

  if (should_exist) {
    if (it == map_end) {
      std::cerr << "Unexpected find error! i = " << i << ", k = " << k << " (line: " << __LINE__ << ")\n";
      return false;
    }
    std::byte* __restrict slot{it->second};
   
    if constexpr (HasBlobs) {
      std::memcpy(w.scratch_buf, slot, to_uint(blob_size));

      if (check_blobs) {
        for (int_t j{}; j < blob_size; ++j) {
          if (w.scratch_buf[j] != static_cast<char>(i + j + k)) {
            std::cerr << "Blob error! i = " << i << ", k = " << k << " (line: " << __LINE__ << ")\n";
            return false;
          }
        }
      }
    }

    if constexpr (std::is_same_v<Value, void>) {
    } else if constexpr (std::is_same_v<Value, time_t>) {
      if (*reinterpret_cast<time_t*>(&slot[blob_size]) != -k) {
        std::cerr << "Value error! i = " << i << ", k = " << k << " (line: " << __LINE__ << ")\n";
        return false;
      }
    } else {
      static_assert(dependent_false_v<Value>, "Unsupported `Value` type!");
    }
  } else if (it != map_end) {
    std::cerr << "Miss error! i = " << i << ", k = " << k << " (line: " << __LINE__ << ")\n";
    return false;
  }

  return true;
}

template <typename Value, bool HasBlobs, typename Map, typename It, typename Key>
NVHM_NO_INLINE std::pair<bool, int_t> do_find_std(
  worker& __restrict w, int_t batch_size, const Map& __restrict map, const It& __restrict map_end,
  const std::vector<Key>& __restrict keys, int_t blob_size, std::vector<bool>& __restrict should_exist, bool check_blobs) {
  const int_t i0{std::min((w.index() + 0) * batch_size, to_int(keys.size()))};
  const int_t i1{std::min((w.index() + 1) * batch_size, to_int(keys.size()))};

  bool b{true};
  int_t n{};

  for (int_t i{i0}; i < i1; ++i) {
    b = should_exist[to_uint(i)];
    n += b;
    b = find_and_verify_std<Value, HasBlobs>(w, map, map_end, i, keys[to_uint(i)], blob_size, b, check_blobs);
    if (!b) break;
  }

  return {b, n};
}

template <typename Map, typename Value, bool HasBlobs, bool SerializeCopy>
void bench_std_map(
  const int_t num_keys, const bool shuffle_keys, const int_t blob_size, 
  const int_t num_workers, const int_t num_trials,
  const queue_t, const int_t max_insert_queue_len,
  const int_t find_keep_perc, const queue_t, const int_t max_find_queue_len, const bool check_blobs,
  std::mt19937_64& rng) {
  using map_t = Map;
  using key_t = typename map_t::key_type;
  using ptr_t = typename map_t::value_type;
  static_assert(std::is_same_v<ptr_t, std::pair<const key_t, std::byte*>>, "`Value` must be `std::byte*`!");
  using value_t = Value;
  constexpr static bool has_values{!std::is_same_v<value_t, void>};
  constexpr static bool has_blobs{HasBlobs};
  constexpr static bool serialize_copy{SerializeCopy};

  if (num_workers < 1 || num_workers > 1024) {
    throw std::runtime_error("`num_workers` is out of bounds!");
  }
  if (has_blobs && blob_size < 1) {
    throw std::runtime_error("`blobs_size` is too small!");
  }
  if (worker::scratch_buf_size < blob_size) {
    throw std::runtime_error("`scratch_buf_size` is too small!");
  }

  const std::string map_type{type_to_string<map_t>()};

  // Initialize key and data buffers.
  std::vector<worker> workers;
  workers.reserve(to_uint(num_workers));
  for (int_t i{}; i < num_workers; ++i) {
    workers.emplace_back(i);
  }
  
  std::vector<key_t> keys(to_uint(num_keys));
  std::iota(keys.begin(), keys.end(), 1337);

  if (shuffle_keys) {
    std::shuffle(keys.begin(), keys.end(), rng);
  }

  std::vector<char> blobs;
  if constexpr (has_blobs) {
    blobs.resize(to_uint(num_keys * blob_size));

    for (int_t i{}; i < num_keys; ++i) {
      for (int_t j{}; j < blob_size; ++j) {
        blobs[to_uint(i * blob_size + j)] += static_cast<char>(i + j + keys[to_uint(i)]);
      }
    }
  }

  constexpr static int_t entry_align{32};
  const static int_t entry_size{num_bytes_v<value_t> + blob_size};
  const static int_t entry_stride{round_up(entry_size, entry_align)};
  std::vector<std::byte> entries;
  std::vector<std::byte*> slots;
  if constexpr (has_values || has_blobs) {
    entries.resize(to_uint(num_keys * entry_stride));
    slots.resize(to_uint(num_keys));
   
    for (int_t i{}; i < num_keys; ++i) {
      std::byte* p{&entries[to_uint(i * entry_stride)]};
      slots[to_uint(i)] = p;
    }
  }
  // Need to shuffle the slots to so key indexes are decorrelated.
  std::shuffle(slots.begin(), slots.end(), rng);

  map_t map;

  for (int_t queue_len{}; queue_len < 1 + max_insert_queue_len; ++queue_len) {
    std::cout << map_type << ", " << std::setw(10) << "Insert " << 100 << "%: ";
    std::cout.flush();

    int_t total_count{};
    for (int_t trial{}; trial < num_trials; ++trial) {
      const int_t batch_size{ceil_div<int_t>(num_keys, 1)};

      std::function<std::pair<bool, int_t>(worker&)> fn;
      if (queue_len == 0) {
        fn = [&](worker& w) { return do_insert_std<value_t, has_blobs, serialize_copy>(w, batch_size, map, keys, slots, blobs, blob_size); };
      } else {
        throw std::runtime_error("Unsupported `queue_len`!");
      }
      if (fn) {
        map.clear();
        worker::set_num_workers(1);
        workers[0].assign(fn);
        auto [ms, count]{accumulate_workers(workers.begin(), workers.begin() + 1)};
        std::cout << (trial ? ", " : "") << std::setw(5) << ms;
        std::cout.flush();
        total_count += count;
      }
    }
    std::cout << ", size: " << map.size() << "[ " << total_count << " ]" << '\n';
  }
  
  // Set half of the bits.
  std::vector<size_t> valid_indexes(keys.size());
  std::iota(valid_indexes.begin(), valid_indexes.end(), 0);
  std::shuffle(valid_indexes.begin(), valid_indexes.end(), rng);

  std::cout << "map size after pruning: " << map.size();
  std::vector<bool> should_exist(keys.size());
  size_t i{};
  for (; i < valid_indexes.size(); ++i) {
    if (i < valid_indexes.size() * to_uint(find_keep_perc) / 100) {
      should_exist[valid_indexes[i]] = true;
    } else {
      should_exist[valid_indexes[i]] = false;
      map.erase(keys[valid_indexes[i]]);
    }
  }
  std::cout << " -> " << map.size() << '\n';

  // Find
  for (int_t queue_len{}; queue_len < 1 + max_find_queue_len; ++queue_len) {
    std::cout << map_type << ", " << std::setw(10) << "Find " << find_keep_perc << "% hit: ";
    std::cout.flush();

    int_t total_count{};
    for (int_t trial{}; trial < num_trials; ++trial) {
      const int_t batch_size{ceil_div(num_keys, num_workers)};

      std::function<std::pair<bool, int_t>(worker&)> fn;
      if (queue_len == 0) {
        fn = [&](worker& w) { return do_find_std<value_t, has_blobs>(w, batch_size, map, map.end(), keys, blob_size, should_exist, check_blobs); };
      } else {
        throw std::runtime_error("Unsupported `queue_len`!");
      }
      if (fn) {
        worker::set_num_workers(num_workers);
        for (int_t w{}; w < num_workers; ++w) {
          workers[to_uint(w)].assign(fn);
        }
        auto [ms, count]{accumulate_workers(workers.begin(), workers.end())};
        std::cout << (trial ? ", " : "") << std::setw(5) << ms;
        std::cout.flush();
        total_count += count;
      }
    }
    std::cout << "; count: " << total_count << '\n';
  }
}

#define NVHM_BENCH_STD_MAP_ARGS_()\
  num_keys, shuffle_keys, blob_size,\
  num_workers, num_trials,\
  insert_queue_type, max_insert_queue_len,\
  find_keep_perc, find_queue_type, max_find_queue_len, check_blobs,\
  rng
  
template <typename Key, typename Value, bool HasBlobs, bool SerializeCopy>
void run_bench_std_map_4(
  const int_t num_keys, const bool shuffle_keys, const int_t blob_size, 
  const int_t num_workers, const int_t num_trials,
  const queue_t insert_queue_type, const int_t max_insert_queue_len,
  const int_t find_keep_perc, const queue_t find_queue_type, const int_t max_find_queue_len, const bool check_blobs,
  std::mt19937_64& rng,
  const std::string& map_type) {
  if constexpr (nvhm_bench_compile_map_types >= 10) {
    if (map_type == "nvhm_std_map_shim") {
      return bench_std_map<std_map_shim<map<Key, std::byte*>>, Value, HasBlobs, SerializeCopy>(NVHM_BENCH_STD_MAP_ARGS_());
    }
    if (map_type == "std_unordered_map") {
      return bench_std_map<std::unordered_map<Key, std::byte*>, Value, HasBlobs, SerializeCopy>(NVHM_BENCH_STD_MAP_ARGS_());
    }
  }

  if constexpr (nvhm_bench_compile_map_types >= 20) {
    if (map_type == "absl_flat_hash_map") {
      return bench_std_map<absl::flat_hash_map<Key, std::byte*>, Value, HasBlobs, SerializeCopy>(NVHM_BENCH_STD_MAP_ARGS_());
    }
    #if __cplusplus >= 202002L
    if (map_type == "folly_f14_value_map") {
      return bench_std_map<folly::F14ValueMap<Key, std::byte*>, Value, HasBlobs, SerializeCopy>(NVHM_BENCH_STD_MAP_ARGS_());
    }
    #endif
    if (map_type == "phmap_flat_hash_map") {
      return bench_std_map<phmap::flat_hash_map<Key, std::byte*>, Value, HasBlobs, SerializeCopy>(NVHM_BENCH_STD_MAP_ARGS_());
    }
  }

  throw std::runtime_error("Unsupported `map_type`: " + map_type + ". Did you forget to compile with `NVHM_BENCH_COMPILE_MAP_TYPES`?");
}

#define NVHM_BENCH_STD_MAP_4_ARGS_()\
  num_keys, shuffle_keys, blob_size,\
  num_workers, num_trials,\
  insert_queue_type, max_insert_queue_len,\
  find_keep_perc, find_queue_type, max_find_queue_len, check_blobs,\
  rng, map_type

template <typename Map, typename Value, bool HasBlobs>
void run_bench_std_map_3(
  const int_t num_keys, const bool shuffle_keys, const int_t blob_size, 
  const int_t num_workers, const int_t num_trials,
  const queue_t insert_queue_type, const int_t max_insert_queue_len, bool serialize_copy,
  const int_t find_keep_perc, const queue_t find_queue_type, const int_t max_find_queue_len, const bool check_blobs,
  std::mt19937_64& rng,
  const std::string& map_type) {
  if (serialize_copy) {
    run_bench_std_map_4<Map, Value, HasBlobs, true>(NVHM_BENCH_STD_MAP_4_ARGS_());
  } else {
    run_bench_std_map_4<Map, Value, HasBlobs, false>(NVHM_BENCH_STD_MAP_4_ARGS_());
  }
}

#define NVHM_BENCH_STD_MAP_2_AND_3_ARGS_()\
  num_keys, shuffle_keys, blob_size,\
  num_workers, num_trials,\
  insert_queue_type, max_insert_queue_len, serialize_copy,\
  find_keep_perc, find_queue_type, max_find_queue_len, check_blobs,\
  rng, map_type

template <typename Map, typename Value>
void run_bench_std_map_2(
  const int_t num_keys, const bool shuffle_keys, const int_t blob_size, 
  const int_t num_workers, const int_t num_trials,
  const queue_t insert_queue_type, const int_t max_insert_queue_len, bool serialize_copy,
  const int_t find_keep_perc, const queue_t find_queue_type, const int_t max_find_queue_len, const bool check_blobs,
  std::mt19937_64& rng,
  const std::string& map_type) {
  if (blob_size > 0) {
    run_bench_std_map_3<Map, Value, true>(NVHM_BENCH_STD_MAP_2_AND_3_ARGS_());
  } else {
    run_bench_std_map_3<Map, Value, false>(NVHM_BENCH_STD_MAP_2_AND_3_ARGS_());
  }
}

template <typename Key>
void run_bench_std_map_1(
  const int_t num_keys, const bool shuffle_keys, const std::string& value_type, const int_t blob_size, 
  const int_t num_workers, const int_t num_trials,
  const queue_t insert_queue_type, const int_t max_insert_queue_len, bool serialize_copy,
  const int_t find_keep_perc, const queue_t find_queue_type, const int_t max_find_queue_len, const bool check_blobs,
  std::mt19937_64& rng,
  const std::string& map_type) {
  if (value_type == "time") {
    run_bench_std_map_2<Key, time_t>(NVHM_BENCH_STD_MAP_2_AND_3_ARGS_());
  } else if (value_type == "void") {
    run_bench_std_map_2<Key, void>(NVHM_BENCH_STD_MAP_2_AND_3_ARGS_());
  } else {
    throw std::runtime_error("Invalid value type: " + value_type);
  }
}

#define NVHM_BENCH_STD_MAP_1_ARGS_()\
  num_keys, shuffle_keys, value_type, blob_size,\
  num_workers, num_trials,\
  insert_queue_type, max_insert_queue_len, serialize_copy,\
  find_keep_perc, find_queue_type, max_find_queue_len, check_blobs,\
  rng, map_type

void run_bench_std_map_0(
  const std::string& key_type, const int_t num_keys, const bool shuffle_keys, const std::string& value_type, const int_t blob_size, 
  const int_t num_workers, const int_t num_trials,
  const queue_t insert_queue_type, const int_t max_insert_queue_len, bool serialize_copy,
  const int_t find_keep_perc, const queue_t find_queue_type, const int_t max_find_queue_len, const bool check_blobs,
  std::mt19937_64& rng,
  const std::string& map_type) {
  if (key_type == "int32" ) {
    run_bench_std_map_1<int32_t>(NVHM_BENCH_STD_MAP_1_ARGS_());
  } else if (key_type == "int64") {
    run_bench_std_map_1<int64_t>(NVHM_BENCH_STD_MAP_1_ARGS_());
  } else {
    throw std::runtime_error("Unsupported `key_type`!");
  }
}

template <typename T>
CLI::Validator make_set_validator(std::string name, const std::set<T>& values) {
  std::ostringstream os;

  os << name << '{';
  const char* sep{""};
  for (const auto& v : values) {
    os << sep << v;
    sep = "|";
  }
  os << '}';

  return {
    [values, name](const std::string& x) -> std::string {
      if (values.find(x) != values.end()) {
        return "";
      }
      return "Value " + x + " not in allowed for " + name + "!";
    },
    os.str()
  };
}

int main(int argc, char* argv[]) {
  CLI::App app{"NVHashmap Benchmark"};
  
  std::string key_type{"int64"};
  int_t num_keys{50'000'000};
  bool shuffle_keys{true};
  bool aggressive_prefetch{true};
  std::string value_type{"time"};
  int_t blob_size{256};
  int_t num_workers{8};
  int_t num_trials{4};
  queue_t insert_queue_type{queue_t::ring};
  int_t max_insert_queue_len{0};
  int_t find_keep_perc{100};
  queue_t find_queue_type{queue_t::ring};
  int_t max_find_queue_len{0};
  bool check_blobs{false};
  std::size_t seed{rd()};
  std::string map_type{"nvhm_map"};
  std::string kernel_type{"default"};
  std::string probe_seq_type{"default"};
  bool serialize_copy{false};

  const std::map<std::string, queue_t> str_to_queue{
    {to_string(queue_t::shift), queue_t::shift},
    {to_string(queue_t::ring), queue_t::ring}
  };

  const std::set<std::string> kernel_types{
    #if NVHM_BENCH_COMPILE_KERNELS >= 10
    "default1", "default2", "default4", "default8", "default16", "default32", "default64", "default128", "default256", "default512",
    #endif
    #if NVHM_BENCH_COMPILE_KERNELS >= 20
    #if NVHM_WITH_SSE >= 2
    "sse",
    #endif
    #if NVHM_WITH_AVX >= 2
    "avx",
    #endif
    #if NVHM_WITH_AVX512
    "avx512",
    #endif
    #if NVHM_WITH_NEON
    "neon8", "neon16", "neon32", "neon64",
    #endif
    #if NVHM_WITH_SVE
    #if NVHM_WITH_SVE_SIZE >= 1
    "sve1",
    #endif
    #if NVHM_WITH_SVE_SIZE >= 2
    "sve2",
    #endif
    #if NVHM_WITH_SVE_SIZE >= 4
    "sve4",
    #endif
    #if NVHM_WITH_SVE_SIZE >= 8
    "sve8",
    #endif
    #if NVHM_WITH_SVE_SIZE >= 16
    "sve16",
    #endif
    #if NVHM_WITH_SVE_SIZE >= 32
    "sve32",
    #endif
    #if NVHM_WITH_SVE_SIZE >= 64
    "sve64",
    #endif
    #if NVHM_WITH_SVE_SIZE >= 128
    "sve128",
    #endif
    #if NVHM_WITH_SVE_SIZE >= 256
    "sve256",
    #endif
    #endif
    #endif
    #if NVHM_BENCH_COMPILE_KERNELS >= 30
    "uint1", "uint2", "uint4", "uint8", "uint16",
    "fast_uint1", "fast_uint2", "fast_uint4", "fast_uint8", "fast_uint16",
    #endif
    #if NVHM_BENCH_COMPILE_KERNELS >= 40
    "array1", "array2", "array4", "array8", "array16", "array32", "array64", "array128", "array256", "array512",
    #endif
    "default"
  };

  const std::set<std::string> probe_seq_types{
    #if NVHM_BENCH_COMPILE_PROBE_SEQS >= 10
    "linear", "quadratic",
    #endif
    #if NVHM_BENCH_COMPILE_PROBE_SEQS >= 20
    "aligned_linear", "aligned_quadratic",
    #endif
    "default"
  };

  const std::set<std::string> map_types{
    #if NVHM_BENCH_COMPILE_MAP_TYPES >= 10
    "nvhm_std_map_shim",
    "std_unordered_map",
    #endif
    #if NVHM_BENCH_COMPILE_MAP_TYPES >= 20
    "absl_flat_hash_map",
    #if __cplusplus >= 202002L
    "folly_f14_value_map",
    #endif
    "phmap_flat_hash_map",
    #endif
    "nvhm_map"
  };
  
  app.add_option("--key_type", key_type, "Key type (int32 | int64)")->default_val(key_type);
  app.add_option("--num_keys", num_keys, "Number of keys")->default_val(num_keys)->check(CLI::Validator(CLI::PositiveNumber));
  app.add_option("--shuffle_keys", shuffle_keys, "Shuffle keys before insertion")->default_val(shuffle_keys);
  app.add_option("--aggressive_prefetch", aggressive_prefetch, "Aggressive prefetch")->default_val(aggressive_prefetch);
  app.add_option("--value_type", value_type, "Value type (time | void)")->default_val(value_type);
  app.add_option("--blob_size", blob_size, "Blob size")->default_val(blob_size)->check(CLI::Validator(CLI::NonNegativeNumber));
  app.add_option("--num_workers", num_workers, "Number of workers")->default_val(num_workers)->check(CLI::Validator(CLI::PositiveNumber));
  app.add_option("--num_trials", num_trials, "Number of trials")->default_val(num_trials)->check(CLI::Validator(CLI::PositiveNumber));
  app.add_option("--insert_queue_type", insert_queue_type, "Insert queue type")->default_str(to_string(insert_queue_type))->transform(CLI::CheckedTransformer(str_to_queue, CLI::ignore_case));
  app.add_option("--max_insert_queue_len", max_insert_queue_len, "Max insert queue length")->default_val(max_insert_queue_len)->check(CLI::Validator(CLI::NonNegativeNumber));
  app.add_option("--find_keep_perc", find_keep_perc, "Find keep percentage")->default_val(find_keep_perc)->check(CLI::Validator(CLI::NonNegativeNumber));
  app.add_option("--find_queue_type", find_queue_type, "Find queue type")->default_str(to_string(find_queue_type))->transform(CLI::CheckedTransformer(str_to_queue, CLI::ignore_case));
  app.add_option("--max_find_queue_len", max_find_queue_len, "Max find queue length")->default_val(max_find_queue_len)->check(CLI::Validator(CLI::NonNegativeNumber));
  app.add_option("--check_blobs", check_blobs, "Check blobs")->default_val(check_blobs);
  app.add_option("--seed", seed, "Randomizer seed")->default_str("random");
  app.add_option("--map_type", map_type, "Map type")->default_str(map_type)->check(make_set_validator("MAP_TYPE", map_types));
  app.add_option("--kernel_type", kernel_type, "Kernel type")->default_str(kernel_type)->check(make_set_validator("KERNEL_TYPE", kernel_types));
  app.add_option("--probe_seq_type", probe_seq_type, "Probe sequence type")->default_str(probe_seq_type)->check(make_set_validator("PROBE_SEQ_TYPE", probe_seq_types));
  app.add_option("--serialize_copy", serialize_copy, "Serialize copy")->default_val(serialize_copy);

  argv = app.ensure_utf8(argv);
  CLI11_PARSE(app, argc, argv);

  std::cout << argv[0] << '\n';
  std::cout << "  --key_type " << key_type << '\n';
  std::cout << "  --num_keys " << num_keys << '\n';
  std::cout << "  --shuffle_keys " << shuffle_keys << '\n';
  std::cout << "  --aggressive_prefetch " << aggressive_prefetch << '\n';
  std::cout << "  --value_type " << value_type << '\n';
  std::cout << "  --blob_size " << blob_size << '\n';
  std::cout << "  --num_workers " << num_workers << '\n';
  std::cout << "  --num_trials " << num_trials << '\n';
  std::cout << "  --insert_queue_type " << insert_queue_type << '\n';
  std::cout << "  --max_insert_queue_len " << max_insert_queue_len << '\n';
  std::cout << "  --find_keep_perc " << find_keep_perc << '\n';
  std::cout << "  --find_queue_type " << find_queue_type << '\n';
  std::cout << "  --max_find_queue_len " << max_find_queue_len << '\n';
  std::cout << "  --check_blobs " << check_blobs << '\n';
  std::cout << "  --seed = " << seed << '\n';
  std::cout << "  --map_type " << map_type << '\n';
  std::cout << "  --kernel_type " << kernel_type << '\n';
  std::cout << "  --probe_seq_type " << probe_seq_type << '\n';
  std::cout << "  --serialize_copy " << serialize_copy << '\n';

  std::mt19937_64 rng(seed);

  if (map_type == "nvhm_map") {
    run_bench_nvhm_map_0(
      key_type, num_keys, shuffle_keys, aggressive_prefetch, value_type, blob_size,
      num_workers, num_trials,
      insert_queue_type, max_insert_queue_len,
      find_keep_perc, find_queue_type, max_find_queue_len, check_blobs,
      rng, kernel_type, probe_seq_type
    );
  } else {
    run_bench_std_map_0(
      key_type, num_keys, shuffle_keys, value_type, blob_size,
      num_workers, num_trials,
      insert_queue_type, max_insert_queue_len, serialize_copy,
      find_keep_perc, find_queue_type, max_find_queue_len, check_blobs,
      rng, map_type
    );
  }

  return 0;
}
