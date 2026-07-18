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

enum class statistic_t {
  max,
  mean,
  sum
};

constexpr const char* to_string(statistic_t s) {
  switch (s) {
    case statistic_t::max: return "max";
    case statistic_t::sum: return "sum";
    case statistic_t::mean: return "mean";
  }
  return "error";
}

inline std::ostream& operator<<(std::ostream& os, statistic_t s) {
  return os << to_string(s);
}

static statistic_t stat{statistic_t::max};

template <typename It>
NVHM_NO_INLINE std::pair<std::chrono::milliseconds, int_t> accumulate_workers(It begin, It last) {
  int_t num_workers{last - begin};

  bool success{true};
  std::chrono::milliseconds time{};
  int_t count{};
  for (; begin != last; ++begin) {
    begin->join();

    if (stat == statistic_t::max) {
      time = std::max(time, begin->time());
    } else {
      time += begin->time();
    }
    count += begin->count();

    if (!begin->status()) {
      std::cerr << "Worker #" << begin->index() << " failed!";
      success = false;
    }
  }
  
  if (!success) {
    return {std::chrono::milliseconds::zero(), -1};
  }
  switch (stat) {
    case statistic_t::max:
      return {time, count};
    case statistic_t::sum:
      return {time, count};
    case statistic_t::mean:
      return {time / num_workers, count};
  }
}

static int_t blob_size{256};

template <typename Map, typename Key, typename PrefetchHint>
NVHM_ALWAYS_INLINE void insert_entry(Map& __restrict map, int_t /*i*/, const Key& __restrict k, PrefetchHint&& h, const char* __restrict blob) {
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

template <typename Queue, typename Map, typename Key, typename PrefetchHint = typename Map::prefetch_hint>
NVHM_NO_INLINE std::pair<bool, int_t> do_insert(worker& __restrict w, int_t batch_size, Map& __restrict map, const std::vector<Key>& __restrict keys, const std::vector<char>& __restrict blobs, int_t queue_len) {
  const int_t i0{std::min((w.index() + 0) * batch_size, to_int(keys.size()))};
  const int_t i1{std::min((w.index() + 1) * batch_size, to_int(keys.size()))};

  int_t i{i0};

  if constexpr (std::is_same_v<Queue, void>) {
    if (queue_len != 0) {
      throw std::runtime_error("queue_len != 0");
    }

    for (; i < i1; ++i) {
      insert_entry(map, i, keys[to_uint(i)], std::monostate{}, &blobs[to_uint(i * blob_size)]);
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
    }

    for (int_t j{}; j < queue_len; ++j) {
      if constexpr (Queue::type == queue_t::shift) {
        q.prefill_write(j, map, keys[to_uint(i0 + j)]);
      } else if constexpr (Queue::type == queue_t::ring) {
        q.prefill_write(map, keys[to_uint(i0 + j)]);
      } else {
        static_assert(dependent_false_v<Queue>, "Invalid queue type!");
      }
    } 

    for (; i < i1 - queue_len; ++i) {
      const auto [k0, h0]{q.push_write(map,  keys[to_uint(i + queue_len)])};
      insert_entry(map, i, k0, std::move(h0), &blobs[to_uint(i * blob_size)]);
    }

    for (int_t j{}; j < queue_len; ++i, ++j) {
      const auto [k0, h0]{q.pop()};
      insert_entry(map, i, k0, std::move(h0), &blobs[to_uint(i * blob_size)]);
    }
  }

  return {true, i - i0};
}

template <typename Value, bool HasBlobs, bool SerializeCopy, typename Map, typename Key>
NVHM_NO_INLINE std::pair<bool, int_t> do_insert_std(worker& __restrict w, int_t batch_size, Map& __restrict map, const std::vector<Key>& __restrict keys,  const std::vector<std::byte*>& __restrict slots, const std::vector<char>& __restrict blobs) {
  const int_t i0{std::min((w.index() + 0) * batch_size, to_int(keys.size()))};
  const int_t i1{std::min((w.index() + 1) * batch_size, to_int(keys.size()))};

  int_t i{i0};
  for (; i < i1; ++i) {
    insert_entry_std<Value, HasBlobs, SerializeCopy>(map, i, keys[to_uint(i)], slots[to_uint(i)], &blobs[to_uint(i * blob_size)], blob_size);
  }

  return {true, i - i0};
}
static bool check_blobs{false};

template <typename Map, typename Key, typename PrefetchHint>
NVHM_ALWAYS_INLINE bool find_and_verify(worker& __restrict w, const Map& __restrict map, int_t i, const Key& __restrict k, PrefetchHint&& h, bool should_exist) {
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

template <typename Value, bool HasBlobs, typename Map, typename It, typename Key>
NVHM_ALWAYS_INLINE bool find_and_verify_std(worker& __restrict w, const Map& __restrict map, const It& __restrict map_end, int_t i, const Key& __restrict k, bool should_exist) {
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

template <typename Queue, typename Map, typename Key, typename PrefetchHint = typename Map::prefetch_hint>
NVHM_NO_INLINE std::pair<bool, int_t> do_find(worker& __restrict w, int_t batch_size, const Map& __restrict map, const std::vector<int_t>& __restrict indexes, const std::vector<Key>& __restrict keys, std::vector<bool>& __restrict should_exist, int_t queue_len) {
  const int_t i0{std::min((w.index() + 0) * batch_size, to_int(keys.size()))};
  const int_t i1{std::min((w.index() + 1) * batch_size, to_int(keys.size()))};

  bool b{true};
  int_t n{};

  if constexpr (std::is_same_v<Queue, void>) {
    if (queue_len != 0) {
      throw std::runtime_error("queue_len != 0");
    }
    for (int_t i{i0}; i < i1; ++i) {
      int_t idx{indexes[to_uint(i)]};
      b = should_exist[to_uint(idx)];
      n += b;
      b = find_and_verify(w, map, idx, keys[to_uint(idx)], std::monostate{}, b);
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
    }

    for (int_t j{}; j < queue_len; ++j) {
      int_t idx{indexes[to_uint(i0 + j)]};
      if constexpr (Queue::type == queue_t::shift) {
        q.prefill_read(j, map, keys[to_uint(idx)]);
      } else if constexpr (Queue::type == queue_t::ring) {
        q.prefill_read(map, keys[to_uint(idx)]);
      } else {
        static_assert(dependent_false_v<Queue>, "Invalid queue type!");
      }
    }

    int_t i{i0};
    for (; i < i1 - queue_len; ++i) {
      int_t idx{indexes[to_uint(i + queue_len)]};
      const auto [k0, h0]{q.push_read(map, keys[to_uint(idx)])};
      idx = indexes[to_uint(i)];
      b = should_exist[to_uint(idx)];
      n += b;
      b = find_and_verify(w, map, idx, k0, std::move(h0), b);
      if (!b) return {b, n};
    }
    
    for (int_t j{}; j < queue_len; ++i, ++j) {
      const auto [k0, h0]{q.pop()};
      int_t idx{indexes[to_uint(i)]};
      b = should_exist[to_uint(idx)];
      n += b;
      b = find_and_verify(w, map, idx, k0, std::move(h0), b);
      if (!b) break;
    }
  }

  return {b, n};
}

template <typename Value, bool HasBlobs, typename Map, typename It, typename Key>
NVHM_NO_INLINE std::pair<bool, int_t> do_find_std(worker& __restrict w, int_t batch_size, const Map& __restrict map, const It& __restrict map_end, const std::vector<int_t>& __restrict indexes, const std::vector<Key>& __restrict keys, std::vector<bool>& __restrict should_exist) {
  const int_t i0{std::min((w.index() + 0) * batch_size, to_int(keys.size()))};
  const int_t i1{std::min((w.index() + 1) * batch_size, to_int(keys.size()))};

  bool b{true};
  int_t n{};

  for (int_t i{i0}; i < i1; ++i) {
    int_t idx{indexes[to_uint(i)]};
    b = should_exist[to_uint(idx)];
    n += b;
    b = find_and_verify_std<Value, HasBlobs>(w, map, map_end, idx, keys[to_uint(idx)], b);
    if (!b) break;
  }

  return {b, n};
}

enum class key_source_t {
  polynomial,
  random
};

constexpr const char* to_string(key_source_t kd) {
  switch (kd) {
    case key_source_t::polynomial: return "polynomial";
    case key_source_t::random: return "random";
  }
  return "error";
}

inline std::ostream& operator<<(std::ostream& os, key_source_t kd) {
  return os << to_string(kd);
}

static int_t num_keys{50'000'000};
static key_source_t key_source{key_source_t::polynomial};
static int_t key_c[]{13, 3, 7};
static int_t num_workers{8};
static int_t num_insert_trials{5};
static queue_t insert_queue_type{queue_t::ring};
static int_t min_insert_queue_len{0};
static int_t max_insert_queue_len{0};
static int_t num_find_trials{5};
static int_t find_keep_perc{100};
static queue_t find_queue_type{queue_t::ring};
static int_t min_find_queue_len{0};
static int_t max_find_queue_len{0};
static std::size_t seed{rd()};

template <typename Map>
NVHM_NO_INLINE void bench_nvhm_map() {
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
 
  std::string map_type{type_to_string<map_t>()};
  map_type.erase(std::remove_if(map_type.begin(), map_type.end(), [](unsigned char c){ return std::isspace(c); }), map_type.end());
  std::replace(map_type.begin(), map_type.end(), ',', '|');
  std::mt19937_64 rng{seed};
 
  // Initialize key and data buffers.
  std::vector<worker> workers;
  workers.reserve(to_uint(num_workers));
  for (int_t i{}; i < num_workers; ++i) {
    workers.emplace_back(i);
  }
  
  std::vector<key_t> keys(to_uint(num_keys));
  std::uniform_int_distribution<key_t> uniform_dist;
  for (int_t i{}; i < num_keys; ++i) {
    switch (key_source) {
      case key_source_t::polynomial:
        keys[to_uint(i)] = static_cast<key_t>(key_c[2] * i * i + key_c[1] * i + key_c[0]);
        break;
      case key_source_t::random:
        keys[to_uint(i)] = uniform_dist(rng);
        break;
      default:
        throw std::runtime_error("Unsupported `key_source`!");
    }
  }
  if (key_source != key_source_t::random) {
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
  for (int_t queue_len{min_insert_queue_len}; queue_len < max_insert_queue_len; ++queue_len) {
    if (num_insert_trials > 0) {
      std::cout << map_type << ", " << std::setw(10) << "Worker " << 1 << ", Insert " << 100 << "% (" << insert_queue_type << ' ' << std::setw(2) << queue_len << "): ";
      std::cout.flush();
    }
 
    int_t total_count{};
    for (int_t trial{}; trial < std::max<int_t>(1, num_insert_trials); ++trial) {
      const int_t batch_size{ceil_div<int_t>(num_keys, 1)};
 
      std::function<std::pair<bool, int_t>(worker&)> fn;
      if (queue_len == 0) {
        fn = [&](worker& w) { return do_insert<void>(w, batch_size, map, keys, blobs, 0); };
      } else {
        switch (insert_queue_type) {
          case queue_t::shift:
            switch (queue_len) {
              case  1: fn = [&](worker& w) { return do_insert<shift_prefetch_queue<key_t, hint_t,  1>>(w, batch_size, map, keys, blobs,  1); }; break;
              case  2: fn = [&](worker& w) { return do_insert<shift_prefetch_queue<key_t, hint_t,  2>>(w, batch_size, map, keys, blobs,  2); }; break;
              case  3: fn = [&](worker& w) { return do_insert<shift_prefetch_queue<key_t, hint_t,  3>>(w, batch_size, map, keys, blobs,  3); }; break;
              case  4: fn = [&](worker& w) { return do_insert<shift_prefetch_queue<key_t, hint_t,  4>>(w, batch_size, map, keys, blobs,  4); }; break;
              case  5: fn = [&](worker& w) { return do_insert<shift_prefetch_queue<key_t, hint_t,  5>>(w, batch_size, map, keys, blobs,  5); }; break;
              case  6: fn = [&](worker& w) { return do_insert<shift_prefetch_queue<key_t, hint_t,  6>>(w, batch_size, map, keys, blobs,  6); }; break;
              case  7: fn = [&](worker& w) { return do_insert<shift_prefetch_queue<key_t, hint_t,  7>>(w, batch_size, map, keys, blobs,  7); }; break;
              case  8: fn = [&](worker& w) { return do_insert<shift_prefetch_queue<key_t, hint_t,  8>>(w, batch_size, map, keys, blobs,  8); }; break;
              case  9: fn = [&](worker& w) { return do_insert<shift_prefetch_queue<key_t, hint_t,  9>>(w, batch_size, map, keys, blobs,  9); }; break;
              case 10: fn = [&](worker& w) { return do_insert<shift_prefetch_queue<key_t, hint_t, 10>>(w, batch_size, map, keys, blobs, 10); }; break;
              case 11: fn = [&](worker& w) { return do_insert<shift_prefetch_queue<key_t, hint_t, 11>>(w, batch_size, map, keys, blobs, 11); }; break;
              case 12: fn = [&](worker& w) { return do_insert<shift_prefetch_queue<key_t, hint_t, 12>>(w, batch_size, map, keys, blobs, 12); }; break;
              case 13: fn = [&](worker& w) { return do_insert<shift_prefetch_queue<key_t, hint_t, 13>>(w, batch_size, map, keys, blobs, 13); }; break;
              case 14: fn = [&](worker& w) { return do_insert<shift_prefetch_queue<key_t, hint_t, 14>>(w, batch_size, map, keys, blobs, 14); }; break;
              case 15: fn = [&](worker& w) { return do_insert<shift_prefetch_queue<key_t, hint_t, 15>>(w, batch_size, map, keys, blobs, 15); }; break;
              case 16: fn = [&](worker& w) { return do_insert<shift_prefetch_queue<key_t, hint_t, 16>>(w, batch_size, map, keys, blobs, 16); }; break;
              default: throw std::runtime_error("`queue_len` is out of bounds!");
            }
            break;
          case queue_t::ring:
            if (queue_len <= 4) {
              fn = [&](worker& w) { return do_insert<ring_prefetch_queue<key_t, hint_t, 4>>(w, batch_size, map, keys, blobs, queue_len); };
            } else if (queue_len <= 8) {
              fn = [&](worker& w) { return do_insert<ring_prefetch_queue<key_t, hint_t, 8>>(w, batch_size, map, keys, blobs, queue_len); };
            } else if (queue_len <= 16) {
              fn = [&](worker& w) { return do_insert<ring_prefetch_queue<key_t, hint_t, 16>>(w, batch_size, map, keys, blobs, queue_len); };
            } else if (queue_len <= 32) {
              fn = [&](worker& w) { return do_insert<ring_prefetch_queue<key_t, hint_t, 32>>(w, batch_size, map, keys, blobs, queue_len); };
            } else if (queue_len <= 64) {
              fn = [&](worker& w) { return do_insert<ring_prefetch_queue<key_t, hint_t, 64>>(w, batch_size, map, keys, blobs, queue_len); };
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
        if (num_insert_trials > 0) {
          std::cout << (trial ? ", " : "") << std::setw(5) << ms;
          std::cout.flush();
        }
        total_count += count;
      }
    }
    if (num_insert_trials > 0) {
      std::cout << ", size: " << map.size() << " / " << map.capacity() << " [ " << total_count << " ]" << '\n';
    }
  }
  if (!map.check_integrity()) {
    throw std::runtime_error("Map integrity check failed!");
  }
  if (num_find_trials <= 0) return;
 
  // Set half of the bits.
  std::vector<int_t> indexes(keys.size());
  std::iota(indexes.begin(), indexes.end(), 0);
  std::shuffle(indexes.begin(), indexes.end(), rng);
 
  std::cerr << "\nmap size after pruning: " << map.size();
  std::vector<bool> should_exist(keys.size());
  for (int_t i{}; i < to_int(indexes.size()); ++i) {
    int_t idx{indexes[to_uint(i)]};
    if (i < to_int(indexes.size()) * find_keep_perc / 100) {
      should_exist[to_uint(idx)] = true;
    } else {
      should_exist[to_uint(idx)] = false;
      map.erase(keys[to_uint(idx)]);
    }
  }
  std::cerr << " -> " << map.size() << '\n';
  if (!map.check_integrity()) {
    throw std::runtime_error("Map integrity check failed!");
  }
 
  // Shuffle once more to decorrelate insert and find order.
  std::shuffle(indexes.begin(), indexes.end(), rng);

  // Find
  for (int_t queue_len{min_find_queue_len}; queue_len < max_find_queue_len; ++queue_len) {
    if (num_find_trials > 0) {
      std::cout << map_type << ", " << std::setw(10) << "Worker " << num_workers << ", Find " << find_keep_perc << "% hit (" << find_queue_type << ' ' << std::setw(2) << queue_len << "): ";
      std::cout.flush();
    }

    int_t total_count{};
    for (int_t trial{}; trial < num_find_trials; ++trial) {
      const int_t batch_size{ceil_div(num_keys, num_workers)};
 
      std::function<std::pair<bool, int_t>(worker&)> fn;
      if (queue_len == 0) {
        fn = [&](worker& w) { return do_find<void>(w, batch_size, map, indexes, keys, should_exist, 0); };
      } else {
        switch (find_queue_type) {
          case queue_t::shift:
            switch (queue_len) {
              case  1: fn = [&](worker& w) { return do_find<shift_prefetch_queue<key_t, hint_t,  1>>(w, batch_size, map, indexes, keys, should_exist,  1); }; break;
              case  2: fn = [&](worker& w) { return do_find<shift_prefetch_queue<key_t, hint_t,  2>>(w, batch_size, map, indexes, keys, should_exist,  2); }; break;
              case  3: fn = [&](worker& w) { return do_find<shift_prefetch_queue<key_t, hint_t,  3>>(w, batch_size, map, indexes, keys, should_exist,  3); }; break;
              case  4: fn = [&](worker& w) { return do_find<shift_prefetch_queue<key_t, hint_t,  4>>(w, batch_size, map, indexes, keys, should_exist,  4); }; break;
              case  5: fn = [&](worker& w) { return do_find<shift_prefetch_queue<key_t, hint_t,  5>>(w, batch_size, map, indexes, keys, should_exist,  5); }; break;
              case  6: fn = [&](worker& w) { return do_find<shift_prefetch_queue<key_t, hint_t,  6>>(w, batch_size, map, indexes, keys, should_exist,  6); }; break;
              case  7: fn = [&](worker& w) { return do_find<shift_prefetch_queue<key_t, hint_t,  7>>(w, batch_size, map, indexes, keys, should_exist,  7); }; break;
              case  8: fn = [&](worker& w) { return do_find<shift_prefetch_queue<key_t, hint_t,  8>>(w, batch_size, map, indexes, keys, should_exist,  8); }; break;
              case  9: fn = [&](worker& w) { return do_find<shift_prefetch_queue<key_t, hint_t,  9>>(w, batch_size, map, indexes, keys, should_exist,  9); }; break;
              case 10: fn = [&](worker& w) { return do_find<shift_prefetch_queue<key_t, hint_t, 10>>(w, batch_size, map, indexes, keys, should_exist, 10); }; break;
              case 11: fn = [&](worker& w) { return do_find<shift_prefetch_queue<key_t, hint_t, 11>>(w, batch_size, map, indexes, keys, should_exist, 11); }; break;
              case 12: fn = [&](worker& w) { return do_find<shift_prefetch_queue<key_t, hint_t, 12>>(w, batch_size, map, indexes, keys, should_exist, 12); }; break;
              case 13: fn = [&](worker& w) { return do_find<shift_prefetch_queue<key_t, hint_t, 13>>(w, batch_size, map, indexes, keys, should_exist, 13); }; break;
              case 14: fn = [&](worker& w) { return do_find<shift_prefetch_queue<key_t, hint_t, 14>>(w, batch_size, map, indexes, keys, should_exist, 14); }; break;
              case 15: fn = [&](worker& w) { return do_find<shift_prefetch_queue<key_t, hint_t, 15>>(w, batch_size, map, indexes, keys, should_exist, 15); }; break;
              case 16: fn = [&](worker& w) { return do_find<shift_prefetch_queue<key_t, hint_t, 16>>(w, batch_size, map, indexes, keys, should_exist, 16); }; break;
              default: throw std::runtime_error("`queue_len` is out of bounds!");
            }
            break;
          case queue_t::ring:
            if (queue_len <= 4) {
              fn = [&](worker& w) { return do_find<ring_prefetch_queue<key_t, hint_t, 4>>(w, batch_size, map, indexes, keys, should_exist, queue_len); };
            } else if (queue_len <= 8) {
              fn = [&](worker& w) { return do_find<ring_prefetch_queue<key_t, hint_t, 8>>(w, batch_size, map, indexes, keys, should_exist, queue_len); };
            } else if (queue_len <= 16) {
              fn = [&](worker& w) { return do_find<ring_prefetch_queue<key_t, hint_t, 16>>(w, batch_size, map, indexes, keys, should_exist, queue_len); };
            } else if (queue_len <= 32) {
              fn = [&](worker& w) { return do_find<ring_prefetch_queue<key_t, hint_t, 32>>(w, batch_size, map, indexes, keys, should_exist, queue_len); };
            } else if (queue_len <= 64) {
              fn = [&](worker& w) { return do_find<ring_prefetch_queue<key_t, hint_t, 64>>(w, batch_size, map, indexes, keys, should_exist, queue_len); };
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
    if (num_find_trials > 0) {
      std::cout << " [ " << total_count << " ]\n";
    }
  }
}

template <typename Map, typename Value, bool HasBlobs, bool SerializeCopy>
void bench_std_map() {
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

  std::string map_type{type_to_string<map_t>()};
  map_type.erase(std::remove_if(map_type.begin(), map_type.end(), [](unsigned char c){ return std::isspace(c); }), map_type.end());
  std::replace(map_type.begin(), map_type.end(), ',', '|');
  std::mt19937_64 rng{seed};

  // Initialize key and data buffers.
  std::vector<worker> workers;
  workers.reserve(to_uint(num_workers));
  for (int_t i{}; i < num_workers; ++i) {
    workers.emplace_back(i);
  }
  
  std::vector<key_t> keys(to_uint(num_keys));
  std::uniform_int_distribution<key_t> uniform_dist;
  for (int_t i{}; i < num_keys; ++i) {
    switch (key_source) {
      case key_source_t::polynomial:
        keys[to_uint(i)] = static_cast<key_t>(key_c[2] * i * i + key_c[1] * i + key_c[0]);
        break;
      case key_source_t::random:
        keys[to_uint(i)] = uniform_dist(rng);
        break;
      default:
        throw std::runtime_error("Unsupported `key_source`!");
    }
  }
  if (key_source != key_source_t::random) {
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

  for (int_t queue_len{min_insert_queue_len}; queue_len < max_insert_queue_len; ++queue_len) {
    if (num_insert_trials > 0) {
      std::cout << map_type << ", " << std::setw(10) << "Worker " << 1 << ", Insert " << 100 << "%: ";
      std::cout.flush();
    }

    int_t total_count{};
    for (int_t trial{}; trial < std::max<int_t>(1, num_insert_trials); ++trial) {
      const int_t batch_size{ceil_div<int_t>(num_keys, 1)};

      std::function<std::pair<bool, int_t>(worker&)> fn;
      if (queue_len == 0) {
        fn = [&](worker& w) { return do_insert_std<value_t, has_blobs, serialize_copy>(w, batch_size, map, keys, slots, blobs); };
      } else {
        throw std::runtime_error("Unsupported `queue_len`!");
      }
      if (fn) {
        map.clear();
        worker::set_num_workers(1);
        workers[0].assign(fn);
        auto [ms, count]{accumulate_workers(workers.begin(), workers.begin() + 1)};
        if (num_insert_trials > 0) {
          std::cout << (trial ? ", " : "") << std::setw(5) << ms;
          std::cout.flush();
        }
        total_count += count;
      }
    }
    if (num_insert_trials > 0) {
      std::cout << ", size: " << map.size() << " [ " << total_count << " ]" << '\n';
    }
  }
  if (num_find_trials <= 0) return;
  
  // Set half of the bits.
  std::vector<int_t> indexes(keys.size());
  std::iota(indexes.begin(), indexes.end(), 0);
  std::shuffle(indexes.begin(), indexes.end(), rng);

  std::cerr << "\nmap size after pruning: " << map.size();
  std::vector<bool> should_exist(keys.size());
  for (int_t i{}; i < to_int(indexes.size()); ++i) {
    int_t idx{indexes[to_uint(i)]};
    if (i < to_int(indexes.size()) * find_keep_perc / 100) {
      should_exist[to_uint(idx)] = true;
    } else {
      should_exist[to_uint(idx)] = false;
      map.erase(keys[to_uint(idx)]);
    }
  }
  std::cerr << " -> " << map.size() << '\n';

  // Shuffle once more to decorrelate insert and find order.
  std::shuffle(indexes.begin(), indexes.end(), rng);

  // Find
  for (int_t queue_len{min_find_queue_len}; queue_len < max_find_queue_len; ++queue_len) {
    std::cout << map_type << ", " << std::setw(10) << "Worker " << num_workers << ", Find " << find_keep_perc << "% hit: ";
    std::cout.flush();

    int_t total_count{};
    for (int_t trial{}; trial < num_find_trials; ++trial) {
      const int_t batch_size{ceil_div(num_keys, num_workers)};

      std::function<std::pair<bool, int_t>(worker&)> fn;
      if (queue_len == 0) {
        fn = [&](worker& w) { return do_find_std<value_t, has_blobs>(w, batch_size, map, map.end(), indexes, keys, should_exist); };
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
    std::cout << " [ " << total_count << " ]\n";
  }
}

template <typename Key, typename Value, flags_t Flags, typename Kernel>
void run_bench_nvhm_map_5(const std::string& probe_seq_type) {
  if (probe_seq_type == "default") {
    return bench_nvhm_map<map<Key, Value, Flags, Kernel, default_seq_t>>();
  }
  if constexpr (nvhm_bench_compile_probe_seqs >= 10) {
    if (probe_seq_type == "linear") {
      return bench_nvhm_map<map<Key, Value, Flags, Kernel, linear_seq<0>>>();
    }
    if (probe_seq_type == "quadratic") {
      return bench_nvhm_map<map<Key, Value, Flags, Kernel, quadratic_seq<0>>>();
    }
  }
  if constexpr (nvhm_bench_compile_probe_seqs >= 20) {
    if (probe_seq_type == "aligned_linear") {
      return bench_nvhm_map<map<Key, Value, Flags, Kernel, linear_seq<cache_line_size>>>();
    }
    if (probe_seq_type == "aligned_quadratic") {
      return bench_nvhm_map<map<Key, Value, Flags, Kernel, quadratic_seq<cache_line_size>>>();
    }
  }
  throw std::runtime_error("Invalid probe sequence type: " + probe_seq_type + ". Did you forget to compile with `NVHM_BENCH_COMPILE_PROBE_SEQS`?");
}

template <typename Key, typename Value, flags_t Flags>
void run_bench_nvhm_map_4(const std::string& kernel_type, const std::string& probe_seq_type) {
  if (kernel_type == "default") {
    return run_bench_nvhm_map_5<Key, Value, Flags, default_kernel_t<>>(probe_seq_type);
  }
  if constexpr (nvhm_bench_compile_kernels >= 10) {
    if (kernel_type == "default1") {
      return run_bench_nvhm_map_5<Key, Value, Flags, default_kernel_t<1>>(probe_seq_type);
    } else if (kernel_type == "default2") {
      return run_bench_nvhm_map_5<Key, Value, Flags, default_kernel_t<2>>(probe_seq_type);
    } else if (kernel_type == "default4") {
      return run_bench_nvhm_map_5<Key, Value, Flags, default_kernel_t<4>>(probe_seq_type);
    } else if (kernel_type == "default8") {
      return run_bench_nvhm_map_5<Key, Value, Flags, default_kernel_t<8>>(probe_seq_type);
    } else if (kernel_type == "default16") {
      return run_bench_nvhm_map_5<Key, Value, Flags, default_kernel_t<16>>(probe_seq_type);
    } else if (kernel_type == "default32") {
      return run_bench_nvhm_map_5<Key, Value, Flags, default_kernel_t<32>>(probe_seq_type);
    } else if (kernel_type == "default64") {
      return run_bench_nvhm_map_5<Key, Value, Flags, default_kernel_t<64>>(probe_seq_type);
    } else if (kernel_type == "default128") {
      return run_bench_nvhm_map_5<Key, Value, Flags, default_kernel_t<128>>(probe_seq_type);
    } else if (kernel_type == "default256") {
      return run_bench_nvhm_map_5<Key, Value, Flags, default_kernel_t<256>>(probe_seq_type);
    } else if (kernel_type == "default512") {
      return run_bench_nvhm_map_5<Key, Value, Flags, default_kernel_t<512>>(probe_seq_type);
    }
  }

  if constexpr (nvhm_bench_compile_kernels >= 20) {
    #if NVHM_WITH_SSE >= 2
    if (kernel_type == "sse") {
      return run_bench_nvhm_map_5<Key, Value, Flags, sse_kernel_t>(probe_seq_type);
    }
    #endif
    #if NVHM_WITH_AVX >= 2
    if (kernel_type == "avx") {
      return run_bench_nvhm_map_5<Key, Value, Flags, avx_kernel_t>(probe_seq_type);
    }
    #endif
    #if NVHM_WITH_AVX512
    if (kernel_type == "avx512") {
      return run_bench_nvhm_map_5<Key, Value, Flags, avx512_kernel_t>(probe_seq_type);
    }
    #endif
    #if NVHM_WITH_NEON
    if (kernel_type == "neon8") {
      return run_bench_nvhm_map_5<Key, Value, Flags, neon_kernel8_t>(probe_seq_type);
    }
    if (kernel_type == "neon16") {
      return run_bench_nvhm_map_5<Key, Value, Flags, neon_kernel16_t>(probe_seq_type);
    }
    if (kernel_type == "neon32") {
      return run_bench_nvhm_map_5<Key, Value, Flags, neon_kernel32_t>(probe_seq_type);
    }
    if (kernel_type == "neon64") {
      return run_bench_nvhm_map_5<Key, Value, Flags, neon_kernel64_t>(probe_seq_type);
    }
    #endif
    #if NVHM_WITH_SVE
    #if NVHM_WITH_SVE_SIZE >= 1
    if (kernel_type == "sve1") {
      return run_bench_nvhm_map_5<Key, Value, Flags, sve_kernel1_t>(probe_seq_type);
    }
    #endif
    #if NVHM_WITH_SVE_SIZE >= 2
    if (kernel_type == "sve2") {
      return run_bench_nvhm_map_5<Key, Value, Flags, sve_kernel2_t>(probe_seq_type);
    }
    #endif
    #if NVHM_WITH_SVE_SIZE >= 4
    if (kernel_type == "sve4") {
      return run_bench_nvhm_map_5<Key, Value, Flags, sve_kernel4_t>(probe_seq_type);
    }
    #endif
    #if NVHM_WITH_SVE_SIZE >= 8
    if (kernel_type == "sve8") {
      return run_bench_nvhm_map_5<Key, Value, Flags, sve_kernel8_t>(probe_seq_type);
    }
    #endif
    #if NVHM_WITH_SVE_SIZE >= 16
    if (kernel_type == "sve16") {
      return run_bench_nvhm_map_5<Key, Value, Flags, sve_kernel16_t>(probe_seq_type);
    }
    #endif
    #if NVHM_WITH_SVE_SIZE >= 32
    if (kernel_type == "sve32") {
      return run_bench_nvhm_map_5<Key, Value, Flags, sve_kernel32_t>(probe_seq_type);
    }
    #endif
    #if NVHM_WITH_SVE_SIZE >= 64
    if (kernel_type == "sve64") {
      return run_bench_nvhm_map_5<Key, Value, Flags, sve_kernel64_t>(probe_seq_type);
    }
    #endif
    #if NVHM_WITH_SVE_SIZE >= 128
    if (kernel_type == "sve128") {
      return run_bench_nvhm_map_5<Key, Value, Flags, sve_kernel128_t>(probe_seq_type);
    }
    #endif
    #if NVHM_WITH_SVE_SIZE >= 256
    if (kernel_type == "sve256") {
      return run_bench_nvhm_map_5<Key, Value, Flags, sve_kernel256_t>(probe_seq_type);
    }
    #endif
    #endif
  }

  if constexpr (nvhm_bench_compile_kernels >= 30) {
    if (kernel_type == "uint1") {
      return run_bench_nvhm_map_5<Key, Value, Flags, uint_kernel1_t>(probe_seq_type);
    } else if (kernel_type == "uint2") {
      return run_bench_nvhm_map_5<Key, Value, Flags, uint_kernel2_t>(probe_seq_type);
    } else if (kernel_type == "uint4") {
      return run_bench_nvhm_map_5<Key, Value, Flags, uint_kernel4_t>(probe_seq_type);
    } else if (kernel_type == "uint8") {
      return run_bench_nvhm_map_5<Key, Value, Flags, uint_kernel8_t>(probe_seq_type);
    } else if (kernel_type == "uint16") {
      return run_bench_nvhm_map_5<Key, Value, Flags, uint_kernel16_t>(probe_seq_type);
    }

    if (kernel_type == "fast_uint1") {
      return run_bench_nvhm_map_5<Key, Value, Flags, fast_uint_kernel1_t>(probe_seq_type);
    } else if (kernel_type == "fast_uint2") {
      return run_bench_nvhm_map_5<Key, Value, Flags, fast_uint_kernel2_t>(probe_seq_type);
    } else if (kernel_type == "fast_uint4") {
      return run_bench_nvhm_map_5<Key, Value, Flags, fast_uint_kernel4_t>(probe_seq_type);
    } else if (kernel_type == "fast_uint8") {
      return run_bench_nvhm_map_5<Key, Value, Flags, fast_uint_kernel8_t>(probe_seq_type);
    } else if (kernel_type == "fast_uint16") {
      return run_bench_nvhm_map_5<Key, Value, Flags, fast_uint_kernel16_t>(probe_seq_type);
    }
  }

  if constexpr (nvhm_bench_compile_kernels >= 40) {
    if (kernel_type == "array1") {
      return run_bench_nvhm_map_5<Key, Value, Flags, array_kernel1_t>(probe_seq_type);
    } else if (kernel_type == "array2") {
      return run_bench_nvhm_map_5<Key, Value, Flags, array_kernel2_t>(probe_seq_type);
    } else if (kernel_type == "array4") {
      return run_bench_nvhm_map_5<Key, Value, Flags, array_kernel4_t>(probe_seq_type);
    } else if (kernel_type == "array8") {
      return run_bench_nvhm_map_5<Key, Value, Flags, array_kernel8_t>(probe_seq_type);
    } else if (kernel_type == "array16") {
      return run_bench_nvhm_map_5<Key, Value, Flags, array_kernel16_t>(probe_seq_type);
    } else if (kernel_type == "array32") {
      return run_bench_nvhm_map_5<Key, Value, Flags, array_kernel32_t>(probe_seq_type);
    } else if (kernel_type == "array64") {
      return run_bench_nvhm_map_5<Key, Value, Flags, array_kernel64_t>(probe_seq_type);
    } else if (kernel_type == "array128") {
      return run_bench_nvhm_map_5<Key, Value, Flags, array_kernel128_t>(probe_seq_type);
    } else if (kernel_type == "array256") {
      return run_bench_nvhm_map_5<Key, Value, Flags, array_kernel256_t>(probe_seq_type);
    } else if (kernel_type == "array512") {
      return run_bench_nvhm_map_5<Key, Value, Flags, array_kernel512_t>(probe_seq_type);
    }
  }

  throw std::runtime_error("Invalid kernel type: " + kernel_type + ". Did you forget to compile with `NVHM_BENCH_COMPILE_KERNELS`?");
}

template <typename Key, typename Value, flags_t Flags>
void run_bench_nvhm_map_3(const std::string& kernel_type, const std::string& probe_seq_type) {
  if (blob_size > 0) {
    run_bench_nvhm_map_4<Key, Value, Flags | flags_t::blobs>(kernel_type, probe_seq_type);
  } else {
    run_bench_nvhm_map_4<Key, Value, Flags>(kernel_type, probe_seq_type);
  }
}

template <typename Key, typename Value>
void run_bench_nvhm_map_2(bool aggressive_prefetch, const std::string& kernel_type, const std::string& probe_seq_type) {
  if (aggressive_prefetch) {
    run_bench_nvhm_map_3<Key, Value, flags_t::aggressive_prefetch>(kernel_type, probe_seq_type);
  } else {
    run_bench_nvhm_map_3<Key, Value, flags_t::none>(kernel_type, probe_seq_type);
  }
}

template <typename Key>
void run_bench_nvhm_map_1(bool aggressive_prefetch, const std::string& value_type, const std::string& kernel_type, const std::string& probe_seq_type) {
  if (value_type == "time") {
    run_bench_nvhm_map_2<Key, time_t>(aggressive_prefetch, kernel_type, probe_seq_type);
  } else if (value_type == "void") {
    run_bench_nvhm_map_2<Key, void>(aggressive_prefetch, kernel_type, probe_seq_type);
  } else {
    throw std::runtime_error("Invalid value type: " + value_type);
  }
}

void run_bench_nvhm_map_0(const std::string& key_type, bool aggressive_prefetch, const std::string& value_type, const std::string& kernel_type, const std::string& probe_seq_type) {
  if (key_type == "int32") {
    run_bench_nvhm_map_1<int32_t>(aggressive_prefetch, value_type, kernel_type, probe_seq_type);
  } else if (key_type == "int64") {
    run_bench_nvhm_map_1<int64_t>(aggressive_prefetch, value_type, kernel_type, probe_seq_type);
  } else {
    throw std::runtime_error("Invalid key type: " + key_type);
  }
}

template <typename Key, typename Value, bool HasBlobs, bool SerializeCopy>
void run_bench_std_map_4(const std::string& map_type) {
  if constexpr (nvhm_bench_compile_map_types >= 10) {
    if (map_type == "nvhm_std_map_shim") {
      return bench_std_map<std_map_shim<map<Key, std::byte*>>, Value, HasBlobs, SerializeCopy>();
    }
    if (map_type == "std_unordered_map") {
      return bench_std_map<std::unordered_map<Key, std::byte*>, Value, HasBlobs, SerializeCopy>();
    }
  }

  if constexpr (nvhm_bench_compile_map_types >= 20) {
    if (map_type == "absl_flat_hash_map") {
      return bench_std_map<absl::flat_hash_map<Key, std::byte*>, Value, HasBlobs, SerializeCopy>();
    }
    #if __cplusplus >= 202002L
    if (map_type == "folly_f14_value_map") {
      return bench_std_map<folly::F14ValueMap<Key, std::byte*>, Value, HasBlobs, SerializeCopy>();
    }
    #endif
    if (map_type == "phmap_flat_hash_map") {
      return bench_std_map<phmap::flat_hash_map<Key, std::byte*>, Value, HasBlobs, SerializeCopy>();
    }
  }

  throw std::runtime_error("Unsupported `map_type`: " + map_type + ". Did you forget to compile with `NVHM_BENCH_COMPILE_MAP_TYPES`?");
}

template <typename Map, typename Value, bool HasBlobs>
void run_bench_std_map_3(bool serialize_copy, const std::string& map_type) {
  if (serialize_copy) {
    run_bench_std_map_4<Map, Value, HasBlobs, true>(map_type);
  } else {
    run_bench_std_map_4<Map, Value, HasBlobs, false>(map_type);
  }
}

template <typename Map, typename Value>
void run_bench_std_map_2(bool serialize_copy, const std::string& map_type) {
  if (blob_size > 0) {
    run_bench_std_map_3<Map, Value, true>(serialize_copy, map_type);
  } else {
    run_bench_std_map_3<Map, Value, false>(serialize_copy, map_type);
  }
}

template <typename Key>
void run_bench_std_map_1(const std::string& value_type, bool serialize_copy, const std::string& map_type) {
  if (value_type == "time") {
    run_bench_std_map_2<Key, time_t>(serialize_copy, map_type);
  } else if (value_type == "void") {
    run_bench_std_map_2<Key, void>(serialize_copy, map_type);
  } else {
    throw std::runtime_error("Invalid value type: " + value_type);
  }
}

void run_bench_std_map_0(
  const std::string& key_type, const std::string& value_type, bool serialize_copy, const std::string& map_type) {
  if (key_type == "int32" ) {
    run_bench_std_map_1<int32_t>(value_type, serialize_copy, map_type);
  } else if (key_type == "int64") {
    run_bench_std_map_1<int64_t>(value_type, serialize_copy, map_type);
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

  const std::map<std::string, statistic_t> str_to_statistic{
    {to_string(statistic_t::max), statistic_t::max},
    {to_string(statistic_t::sum), statistic_t::sum},
    {to_string(statistic_t::mean), statistic_t::mean}
  };

  const std::map<std::string, key_source_t> str_to_key_source{
    {to_string(key_source_t::polynomial), key_source_t::polynomial},
    {to_string(key_source_t::random), key_source_t::random}
  };
  
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
  std::string key_type{"int64"};
  bool aggressive_prefetch{true};
  std::string value_type{"time"};
  std::string map_type{"nvhm_map"};
  std::string kernel_type{"default"};
  std::string probe_seq_type{"default"};
  bool serialize_copy{false};

  app.add_option("--stat", stat, "The statistic to use for reporting")->default_str(to_string(stat))->transform(CLI::CheckedTransformer(str_to_statistic, CLI::ignore_case));
  app.add_option("--key_type", key_type, "Key type (int32 | int64)")->default_val(key_type);
  app.add_option("--num_keys", num_keys, "Number of keys")->default_val(num_keys)->check(CLI::Validator(CLI::PositiveNumber));
  app.add_option("--key_source", key_source, "Key source")->default_str(to_string(key_source))->transform(CLI::CheckedTransformer(str_to_key_source, CLI::ignore_case));
  app.add_option("--key_c0", key_c[0], "Key coefficient 0 (key space density control)")->default_val(key_c[0]);
  app.add_option("--key_c1", key_c[1], "Key coefficient 1 (key space density control)")->default_val(key_c[1]);
  app.add_option("--key_c2", key_c[2], "Key coefficient 2 (key space density control)")->default_val(key_c[2]);
  app.add_option("--aggressive_prefetch", aggressive_prefetch, "Aggressive prefetch")->default_val(aggressive_prefetch);
  app.add_option("--value_type", value_type, "Value type (time | void)")->default_val(value_type);
  app.add_option("--blob_size", blob_size, "Blob size")->default_val(blob_size)->check(CLI::Validator(CLI::NonNegativeNumber));
  app.add_option("--num_workers", num_workers, "Number of workers")->default_val(num_workers)->check(CLI::Validator(CLI::PositiveNumber));
  app.add_option("--num_insert_trials", num_insert_trials, "Number of trials for insert")->default_val(num_insert_trials)->check(CLI::Validator(CLI::NonNegativeNumber));
  app.add_option("--insert_queue_type", insert_queue_type, "Insert queue type")->default_str(to_string(insert_queue_type))->transform(CLI::CheckedTransformer(str_to_queue, CLI::ignore_case));
  app.add_option("--min_insert_queue_len", min_insert_queue_len, "Min insert queue length")->default_val(min_insert_queue_len)->check(CLI::Validator(CLI::NonNegativeNumber));
  app.add_option("--max_insert_queue_len", max_insert_queue_len, "Max insert queue length")->default_val(max_insert_queue_len)->check(CLI::Validator(CLI::NonNegativeNumber));
  app.add_option("--num_find_trials", num_find_trials, "Number of trials for find")->default_val(num_find_trials)->check(CLI::Validator(CLI::NonNegativeNumber));
  app.add_option("--find_keep_perc", find_keep_perc, "Find keep percentage")->default_val(find_keep_perc)->check(CLI::Validator(CLI::NonNegativeNumber));
  app.add_option("--find_queue_type", find_queue_type, "Find queue type")->default_str(to_string(find_queue_type))->transform(CLI::CheckedTransformer(str_to_queue, CLI::ignore_case));
  app.add_option("--min_find_queue_len", min_find_queue_len, "Min find queue length")->default_val(min_find_queue_len)->check(CLI::Validator(CLI::NonNegativeNumber));
  app.add_option("--max_find_queue_len", max_find_queue_len, "Max find queue length")->default_val(max_find_queue_len)->check(CLI::Validator(CLI::NonNegativeNumber));
  app.add_option("--check_blobs", check_blobs, "Check blobs")->default_val(check_blobs);
  app.add_option("--seed", seed, "Randomizer seed")->default_str("random");
  app.add_option("--map_type", map_type, "Map type")->default_str(map_type)->check(make_set_validator("MAP_TYPE", map_types));
  app.add_option("--kernel_type", kernel_type, "Kernel type")->default_str(kernel_type)->check(make_set_validator("KERNEL_TYPE", kernel_types));
  app.add_option("--probe_seq_type", probe_seq_type, "Probe sequence type")->default_str(probe_seq_type)->check(make_set_validator("PROBE_SEQ_TYPE", probe_seq_types));
  app.add_option("--serialize_copy", serialize_copy, "Serialize copy")->default_val(serialize_copy);

  argv = app.ensure_utf8(argv);
  CLI11_PARSE(app, argc, argv);

  std::cerr << argv[0] << " \\\n";
  std::cerr << "  --stat " << stat << " \\\n";
  std::cerr << "  --key_type " << key_type << " \\\n";
  std::cerr << "  --num_keys " << num_keys << " \\\n";
  std::cerr << "  --key_source " << key_source << " \\\n";
  std::cerr << "  --key_c0 " << key_c[0] << " \\\n";
  std::cerr << "  --key_c1 " << key_c[1] << " \\\n";
  std::cerr << "  --key_c2 " << key_c[2] << " \\\n";
  std::cerr << "  --aggressive_prefetch " << aggressive_prefetch << " \\\n";
  std::cerr << "  --value_type " << value_type << " \\\n";
  std::cerr << "  --blob_size " << blob_size << " \\\n";
  std::cerr << "  --num_workers " << num_workers << " \\\n";
  std::cerr << "  --num_insert_trials " << num_insert_trials << " \\\n";
  std::cerr << "  --insert_queue_type " << insert_queue_type << " \\\n";
  std::cerr << "  --min_insert_queue_len " << min_insert_queue_len << " \\\n";
  std::cerr << "  --max_insert_queue_len " << max_insert_queue_len << " \\\n";
  std::cerr << "  --num_find_trials " << num_find_trials << " \\\n";
  std::cerr << "  --find_keep_perc " << find_keep_perc << " \\\n";
  std::cerr << "  --find_queue_type " << find_queue_type << " \\\n";
  std::cerr << "  --min_find_queue_len " << min_find_queue_len << " \\\n";
  std::cerr << "  --max_find_queue_len " << max_find_queue_len << " \\\n";
  std::cerr << "  --check_blobs " << check_blobs << " \\\n";
  std::cerr << "  --seed " << seed << " \\\n";
  std::cerr << "  --map_type " << map_type << " \\\n";
  std::cerr << "  --kernel_type " << kernel_type << " \\\n";
  std::cerr << "  --probe_seq_type " << probe_seq_type << " \\\n";
  std::cerr << "  --serialize_copy " << serialize_copy << " \\\n";

  if (min_insert_queue_len > max_insert_queue_len) {
    throw std::runtime_error("`min_insert_queue_len` must be less than `max_insert_queue_len`!");
  }
  ++max_insert_queue_len;

  if (min_find_queue_len > max_find_queue_len) {
    throw std::runtime_error("`min_find_queue_len` must be less than `max_find_queue_len`!");
  }
  ++max_find_queue_len;

  if (max_insert_queue_len - min_insert_queue_len < 1) {
    throw std::runtime_error("`max_insert_queue_len - min_insert_queue_len` must be = 1!");
  }

  if (map_type == "nvhm_map") {
    run_bench_nvhm_map_0(key_type, aggressive_prefetch, value_type, kernel_type, probe_seq_type);
  } else {
    run_bench_std_map_0(key_type, value_type, serialize_copy, map_type);
  }

  return 0;
}
