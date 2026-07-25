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

#include <nvhashmap/map.hpp>
#include <nvhashmap/std_map_shim.hpp>
#include <nvhashmap/prefetch.hpp>
#include <chrono>
#include <thread>
#include <unordered_map>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
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
#include "../tools_common.hpp"

std::atomic<int_t> num_workers_ready;

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

NVHM_MAKE_ENUM_WITH_VALIDATOR_(statistic_t,
  max,
  mean,
  sum
);

statistic_t stat{statistic_t::max};

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
  
  if (success) {
    switch (stat) {
      case statistic_t::max: return {time, count};
      case statistic_t::sum: return {time, count};
      case statistic_t::mean: return {time / num_workers, count};
    }
  }
  return {std::chrono::milliseconds::zero(), -1};
}

int_t blob_size{120};

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

bool check_blobs{false};

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

int_t num_keys{50'000'000};
key_source_t key_source{key_source_t::polynomial};
std::array<int_t, 3> key_poly{13, 3, 7};
int_t num_workers{8};
int_t num_insert_trials{5};
queue_t insert_queue_type{queue_t::ring};
int_t min_insert_queue_len{0};
int_t max_insert_queue_len{0};
int_t num_find_trials{5};
int_t find_hit_perc{100};
queue_t find_queue_type{queue_t::ring};
int_t min_find_queue_len{0};
int_t max_find_queue_len{0};
std::size_t seed{rd()};
int_t map_type_print_len{120};

template <typename Map>
NVHM_NO_INLINE void bench_nvhm_map() {
  using map_t = Map;
  using conf_t = typename map_t::conf_type;
  using key_t = typename map_t::key_type;
  using hint_t = typename map_t::prefetch_hint;
  
  if (num_workers < 1 || num_workers > 1024) {
    throw std::runtime_error("`num_workers` is out of bounds!");
  }
  if (find_hit_perc < 0 || find_hit_perc > 100) {
    throw std::runtime_error("`prune_perc` is out of bounds!");
  }
  if (worker::scratch_buf_size < blob_size) {
    throw std::runtime_error("`scratch_buf_size` is too small!");
  }
 
  static std::string map_type{type_to_string<map_t>()};
  if (map_type.size() > to_uint(map_type_print_len)) {
    map_type.resize(to_uint(map_type_print_len));
  }
  std::mt19937_64 rng{seed};
 
  // Initialize key and data buffers.
  std::vector<worker> workers;
  workers.reserve(to_uint(num_workers));
  for (int_t i{}; i < num_workers; ++i) {
    workers.emplace_back(i);
  }
  
  const std::vector<key_t> keys{make_keys<key_t>(num_keys, key_source, key_poly, rng)};

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
      std::cout
        << "| "
        << std::left << std::setw(static_cast<int>(map_type_print_len)) << map_type << " | "
        << std::right << std::setw(4) << 1 << " | "
        << std::left << std::setw(6) << "insert" << " | "
        << std::right << std::setw(4) << 100 << " | " << std::setw(5) << insert_queue_type << " | " << std::setw(4) << queue_len
        << std::flush;
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
          std::cout << " | " << std::setw(5) << ms.count() << std::flush;
        }
        total_count += count;
      }
    }
    if (num_insert_trials > 0) {
      std::cout << " | " << std::setw(10) << map.size() << " | " << std::setw(10) << map.capacity() << " | " << std::setw(12) << total_count << " |\n";
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
  
  std::vector<bool> should_exist(keys.size());
  for (int_t i{}; i < to_int(indexes.size()); ++i) {
    int_t idx{indexes[to_uint(i)]};
    if (i < to_int(indexes.size()) * find_hit_perc / 100) {
      should_exist[to_uint(idx)] = true;
    } else {
      should_exist[to_uint(idx)] = false;
      map.erase(keys[to_uint(idx)]);
    }
  }
  if (!map.check_integrity()) {
    std::cerr << "\nmap size after pruning: " << map.size() << " -> " << map.size() << '\n' << std::flush;
    throw std::runtime_error("Map integrity check failed!");
  }
 
  // Shuffle once more to decorrelate insert and find order.
  std::shuffle(indexes.begin(), indexes.end(), rng);

  // Find
  for (int_t queue_len{min_find_queue_len}; queue_len < max_find_queue_len; ++queue_len) {
    if (num_find_trials > 0) {
      std::cout
        << "| "
        << std::left << std::setw(static_cast<int>(map_type_print_len)) << map_type << " | "
        << std::right << std::setw(4) << num_workers << " | "
        << std::left << std::setw(6) << "find" << " | "
        << std::right << std::setw(4) << find_hit_perc << " | " << std::setw(5) << find_queue_type << " | " << std::setw(4) << queue_len
        << std::flush;
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
        std::cout << " | " << std::setw(5) << ms.count() << std::flush;
        total_count += count;
      }
    }
    if (num_find_trials > 0) {
      std::cout << " | " << std::setw(10) << map.size() << " | " << std::setw(10) << map.capacity() << " | " << std::setw(12) << total_count << " |\n";
    }
  }
}

template <typename Map, typename Value, bool HasBlobs, bool SerializeCopy>
NVHM_NO_INLINE void bench_std_map() {
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

  static std::string map_type{type_to_string<map_t>()};
  if (map_type.size() > to_uint(map_type_print_len)) {
    map_type.resize(to_uint(map_type_print_len));
  }
  std::mt19937_64 rng{seed};

  // Initialize key and data buffers.
  std::vector<worker> workers;
  workers.reserve(to_uint(num_workers));
  for (int_t i{}; i < num_workers; ++i) {
    workers.emplace_back(i);
  }
  
  const std::vector<key_t> keys{make_keys<key_t>(num_keys, key_source, key_poly, rng)};

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
      std::cout
        << "| "
        << std::left << std::setw(static_cast<int>(map_type_print_len)) << map_type << " | "
        << std::right << std::setw(4) << 1 << " | "
        << std::left << std::setw(6) << "insert" << " | "
        << std::right << std::setw(4) << 100 << " | " << std::setw(5) << "" << " | " << std::setw(4) << ""
        << std::flush;
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
          std::cout << " | " << std::setw(5) << ms.count() << std::flush;
        }
        total_count += count;
      }
    }
    if (num_insert_trials > 0) {
      std::cout << " | " << std::setw(10) << map.size() << " | " << std::setw(10) << "" << " | " << std::setw(12) << total_count << " |\n";
    }
  }
  if (num_find_trials <= 0) return;
  
  // Set half of the bits.
  std::vector<int_t> indexes(keys.size());
  std::iota(indexes.begin(), indexes.end(), 0);
  std::shuffle(indexes.begin(), indexes.end(), rng);

  std::vector<bool> should_exist(keys.size());
  for (int_t i{}; i < to_int(indexes.size()); ++i) {
    int_t idx{indexes[to_uint(i)]};
    if (i < to_int(indexes.size()) * find_hit_perc / 100) {
      should_exist[to_uint(idx)] = true;
    } else {
      should_exist[to_uint(idx)] = false;
      map.erase(keys[to_uint(idx)]);
    }
  }
  if constexpr (false) {
    std::cerr << "\nmap size after pruning: " << map.size() << " -> " << map.size() << '\n' << std::flush;
  }

  // Shuffle once more to decorrelate insert and find order.
  std::shuffle(indexes.begin(), indexes.end(), rng);

  // Find
  for (int_t queue_len{min_find_queue_len}; queue_len < max_find_queue_len; ++queue_len) {
    if (num_find_trials > 0) {
      std::cout
        << "| "
        << std::left << std::setw(static_cast<int>(map_type_print_len)) << map_type << " | "
        << std::right << std::setw(4) << num_workers << " | "
        << std::left << std::setw(6) << "find" << " | "
        << std::right << std::setw(4) << find_hit_perc << " | " << std::setw(5) << "" << " | " << std::setw(4) << ""
        << std::flush;
    }

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
        std::cout << " | " << std::setw(5) << ms.count() << std::flush;
        total_count += count;
      }
    }
    if (num_find_trials > 0) {
      std::cout << " | " << std::setw(10) << map.size() << " | " << std::setw(10) << "" << " | " << std::setw(12) << total_count << " |\n";
    }
  }
}

NVHM_MAKE_ENUM_WITH_VALIDATOR_(map_type_t,
  #if NVHM_TOOLS_COMPILE_MAP_TYPES >= 10
  nvhm_std_map_shim,
  std_unordered_map,
  #endif
  #if NVHM_TOOLS_COMPILE_MAP_TYPES >= 20
  absl_flat_hash_map,
  #if __cplusplus >= 202002L
  folly_f14_value_map,
  #endif
  phmap_flat_hash_map,
  #endif
  nvhm_map
);

template <typename Key, typename Value, flags_t Flags, typename Kernel>
void run_bench_nvhm_map_5(probe_seq_type_t probe_seq_type) {
  switch (probe_seq_type) {
    case probe_seq_type_t::default_: return bench_nvhm_map<map<Key, Value, Flags, Kernel, default_seq_t>>();
    #if NVHM_TOOLS_COMPILE_PROBE_SEQS >= 10
    case probe_seq_type_t::linear: return bench_nvhm_map<map<Key, Value, Flags, Kernel, linear_seq<0>>>();
    case probe_seq_type_t::quadratic: return bench_nvhm_map<map<Key, Value, Flags, Kernel, quadratic_seq<0>>>();
    #endif
    #if NVHM_TOOLS_COMPILE_PROBE_SEQS >= 20
    case probe_seq_type_t::aligned_linear: return bench_nvhm_map<map<Key, Value, Flags, Kernel, linear_seq<cache_line_size>>>();
    case probe_seq_type_t::aligned_quadratic: return bench_nvhm_map<map<Key, Value, Flags, Kernel, quadratic_seq<cache_line_size>>>();
    #endif
  }
 
  std::ostringstream os;
  os << "Invalid probe sequence type: " << probe_seq_type << ". Did you forget to compile with `NVHM_TOOLS_COMPILE_PROBE_SEQS`?";
  throw std::runtime_error(os.str());
}

template <typename Key, typename Value, flags_t Flags>
void run_bench_nvhm_map_4(kernel_type_t kernel_type, probe_seq_type_t probe_seq_type) {
  switch (kernel_type) {
    case kernel_type_t::default_: return run_bench_nvhm_map_5<Key, Value, Flags, default_kernel_t<>>(probe_seq_type); 
    #if NVHM_TOOLS_COMPILE_KERNELS >= 10
    case kernel_type_t::default1: return run_bench_nvhm_map_5<Key, Value, Flags, default_kernel_t<1>>(probe_seq_type);
    case kernel_type_t::default2: return run_bench_nvhm_map_5<Key, Value, Flags, default_kernel_t<2>>(probe_seq_type);
    case kernel_type_t::default4: return run_bench_nvhm_map_5<Key, Value, Flags, default_kernel_t<4>>(probe_seq_type);
    case kernel_type_t::default8: return run_bench_nvhm_map_5<Key, Value, Flags, default_kernel_t<8>>(probe_seq_type);
    case kernel_type_t::default16: return run_bench_nvhm_map_5<Key, Value, Flags, default_kernel_t<16>>(probe_seq_type);
    case kernel_type_t::default32: return run_bench_nvhm_map_5<Key, Value, Flags, default_kernel_t<32>>(probe_seq_type);
    case kernel_type_t::default64: return run_bench_nvhm_map_5<Key, Value, Flags, default_kernel_t<64>>(probe_seq_type);
    case kernel_type_t::default128: return run_bench_nvhm_map_5<Key, Value, Flags, default_kernel_t<128>>(probe_seq_type);
    case kernel_type_t::default256: return run_bench_nvhm_map_5<Key, Value, Flags, default_kernel_t<256>>(probe_seq_type);
    case kernel_type_t::default512: return run_bench_nvhm_map_5<Key, Value, Flags, default_kernel_t<512>>(probe_seq_type);
    #endif
    #if NVHM_TOOLS_COMPILE_KERNELS >= 20
    #if NVHM_WITH_SSE >= 2
    case kernel_type_t::sse: return run_bench_nvhm_map_5<Key, Value, Flags, sse_kernel_t>(probe_seq_type);
    #endif
    #if NVHM_WITH_AVX >= 2
    case kernel_type_t::avx: return run_bench_nvhm_map_5<Key, Value, Flags, avx_kernel_t>(probe_seq_type);
    #endif
    #if NVHM_WITH_AVX512
    case kernel_type_t::avx512: return run_bench_nvhm_map_5<Key, Value, Flags, avx512_kernel_t>(probe_seq_type);
    #endif
    #if NVHM_WITH_NEON
    case kernel_type_t::neon8: return run_bench_nvhm_map_5<Key, Value, Flags, neon_kernel8_t>(probe_seq_type);
    case kernel_type_t::neon16: return run_bench_nvhm_map_5<Key, Value, Flags, neon_kernel16_t>(probe_seq_type);
    case kernel_type_t::neon32: return run_bench_nvhm_map_5<Key, Value, Flags, neon_kernel32_t>(probe_seq_type);
    case kernel_type_t::neon64: return run_bench_nvhm_map_5<Key, Value, Flags, neon_kernel64_t>(probe_seq_type);
    #endif
    #if NVHM_WITH_SVE
    #if NVHM_WITH_SVE_SIZE >= 1
    case kernel_type_t::sve1: return run_bench_nvhm_map_5<Key, Value, Flags, sve_kernel1_t>(probe_seq_type);
    #endif
    #if NVHM_WITH_SVE_SIZE >= 2
    case kernel_type_t::sve2: return run_bench_nvhm_map_5<Key, Value, Flags, sve_kernel2_t>(probe_seq_type);
    #endif
    #if NVHM_WITH_SVE_SIZE >= 4
    case kernel_type_t::sve4: return run_bench_nvhm_map_5<Key, Value, Flags, sve_kernel4_t>(probe_seq_type);
    #endif
    #if NVHM_WITH_SVE_SIZE >= 8
    case kernel_type_t::sve8: return run_bench_nvhm_map_5<Key, Value, Flags, sve_kernel8_t>(probe_seq_type);
    #endif
    #if NVHM_WITH_SVE_SIZE >= 16
    case kernel_type_t::sve16: return run_bench_nvhm_map_5<Key, Value, Flags, sve_kernel16_t>(probe_seq_type);
    #endif
    #if NVHM_WITH_SVE_SIZE >= 32
    case kernel_type_t::sve32: return run_bench_nvhm_map_5<Key, Value, Flags, sve_kernel32_t>(probe_seq_type);
    #endif
    #if NVHM_WITH_SVE_SIZE >= 64
    case kernel_type_t::sve64: return run_bench_nvhm_map_5<Key, Value, Flags, sve_kernel64_t>(probe_seq_type);
    #endif
    #if NVHM_WITH_SVE_SIZE >= 128
    case kernel_type_t::sve128: return run_bench_nvhm_map_5<Key, Value, Flags, sve_kernel128_t>(probe_seq_type);
    #endif
    #if NVHM_WITH_SVE_SIZE >= 256
    case kernel_type_t::sve256: return run_bench_nvhm_map_5<Key, Value, Flags, sve_kernel256_t>(probe_seq_type);
    #endif
    #endif
    #endif
    #if NVHM_TOOLS_COMPILE_KERNELS >= 30
    case kernel_type_t::uint1: return run_bench_nvhm_map_5<Key, Value, Flags, uint_kernel1_t>(probe_seq_type);
    case kernel_type_t::uint2: return run_bench_nvhm_map_5<Key, Value, Flags, uint_kernel2_t>(probe_seq_type);
    case kernel_type_t::uint4: return run_bench_nvhm_map_5<Key, Value, Flags, uint_kernel4_t>(probe_seq_type);
    case kernel_type_t::uint8: return run_bench_nvhm_map_5<Key, Value, Flags, uint_kernel8_t>(probe_seq_type);
    case kernel_type_t::uint16: return run_bench_nvhm_map_5<Key, Value, Flags, uint_kernel16_t>(probe_seq_type);
    case kernel_type_t::fast_uint1: return run_bench_nvhm_map_5<Key, Value, Flags, fast_uint_kernel1_t>(probe_seq_type);
    case kernel_type_t::fast_uint2: return run_bench_nvhm_map_5<Key, Value, Flags, fast_uint_kernel2_t>(probe_seq_type);
    case kernel_type_t::fast_uint4: return run_bench_nvhm_map_5<Key, Value, Flags, fast_uint_kernel4_t>(probe_seq_type);
    case kernel_type_t::fast_uint8: return run_bench_nvhm_map_5<Key, Value, Flags, fast_uint_kernel8_t>(probe_seq_type);
    case kernel_type_t::fast_uint16: return run_bench_nvhm_map_5<Key, Value, Flags, fast_uint_kernel16_t>(probe_seq_type);
    #endif
    #if NVHM_TOOLS_COMPILE_KERNELS >= 40
    case kernel_type_t::array1: return run_bench_nvhm_map_5<Key, Value, Flags, array_kernel1_t>(probe_seq_type);
    case kernel_type_t::array2: return run_bench_nvhm_map_5<Key, Value, Flags, array_kernel2_t>(probe_seq_type);
    case kernel_type_t::array4: return run_bench_nvhm_map_5<Key, Value, Flags, array_kernel4_t>(probe_seq_type);
    case kernel_type_t::array8: return run_bench_nvhm_map_5<Key, Value, Flags, array_kernel8_t>(probe_seq_type);
    case kernel_type_t::array16: return run_bench_nvhm_map_5<Key, Value, Flags, array_kernel16_t>(probe_seq_type);
    case kernel_type_t::array32: return run_bench_nvhm_map_5<Key, Value, Flags, array_kernel32_t>(probe_seq_type);
    case kernel_type_t::array64: return run_bench_nvhm_map_5<Key, Value, Flags, array_kernel64_t>(probe_seq_type);
    case kernel_type_t::array128: return run_bench_nvhm_map_5<Key, Value, Flags, array_kernel128_t>(probe_seq_type);
    case kernel_type_t::array256: return run_bench_nvhm_map_5<Key, Value, Flags, array_kernel256_t>(probe_seq_type);
    case kernel_type_t::array512: return run_bench_nvhm_map_5<Key, Value, Flags, array_kernel512_t>(probe_seq_type);
    #endif
  }

  std::ostringstream os;
  os << "Invalid kernel type: " << kernel_type << ". Did you forget to compile with `NVHM_TOOLS_COMPILE_KERNELS`?";
  throw std::runtime_error(os.str());
}

template <typename Key, typename Value, flags_t Flags>
void run_bench_nvhm_map_3(kernel_type_t kernel_type, probe_seq_type_t probe_seq_type) {
  if (blob_size > 0) {
    run_bench_nvhm_map_4<Key, Value, Flags | flags_t::blobs>(kernel_type, probe_seq_type);
  } else {
    run_bench_nvhm_map_4<Key, Value, Flags>(kernel_type, probe_seq_type);
  }
}

template <typename Key, typename Value>
void run_bench_nvhm_map_2(bool aggressive_prefetch, kernel_type_t kernel_type, probe_seq_type_t probe_seq_type) {
  if (aggressive_prefetch) {
    run_bench_nvhm_map_3<Key, Value, flags_t::aggressive_prefetch>(kernel_type, probe_seq_type);
  } else {
    run_bench_nvhm_map_3<Key, Value, flags_t::none>(kernel_type, probe_seq_type);
  }
}

template <typename Key>
void run_bench_nvhm_map_1(bool aggressive_prefetch, const std::string& value_type, kernel_type_t kernel_type, probe_seq_type_t probe_seq_type) {
  if (value_type == "time") {
    run_bench_nvhm_map_2<Key, time_t>(aggressive_prefetch, kernel_type, probe_seq_type);
  } else if (value_type == "void") {
    run_bench_nvhm_map_2<Key, void>(aggressive_prefetch, kernel_type, probe_seq_type);
  } else {
    throw std::runtime_error("Invalid value type: " + value_type);
  }
}

void run_bench_nvhm_map_0(key_type_t key_type, bool aggressive_prefetch, const std::string& value_type, kernel_type_t kernel_type, probe_seq_type_t probe_seq_type) {
  switch (key_type) {
    case key_type_t::int32: return run_bench_nvhm_map_1<int32_t>(aggressive_prefetch, value_type, kernel_type, probe_seq_type);
    case key_type_t::int64: return run_bench_nvhm_map_1<int64_t>(aggressive_prefetch, value_type, kernel_type, probe_seq_type);
  }
  throw std::runtime_error("Unsupported `key_type`!");
}

template <typename Key, typename Value, bool HasBlobs, bool SerializeCopy>
void run_bench_std_map_4(map_type_t map_type) {
  switch (map_type) {
    case map_type_t::nvhm_map: break;
    #if NVHM_TOOLS_COMPILE_MAP_TYPES >= 10
    case map_type_t::nvhm_std_map_shim: return bench_std_map<std_map_shim<map<Key, std::byte*>>, Value, HasBlobs, SerializeCopy>();
    case map_type_t::std_unordered_map: return bench_std_map<std::unordered_map<Key, std::byte*>, Value, HasBlobs, SerializeCopy>();
    #endif
    #if NVHM_TOOLS_COMPILE_MAP_TYPES >= 20
    case map_type_t::absl_flat_hash_map: return bench_std_map<absl::flat_hash_map<Key, std::byte*>, Value, HasBlobs, SerializeCopy>();
    #if __cplusplus >= 202002L
    case map_type_t::folly_f14_value_map: return bench_std_map<folly::F14ValueMap<Key, std::byte*>, Value, HasBlobs, SerializeCopy>();
    #endif
    case map_type_t::phmap_flat_hash_map: return bench_std_map<phmap::flat_hash_map<Key, std::byte*>, Value, HasBlobs, SerializeCopy>();
    #endif
  }

  std::ostringstream os;
  os << "Invalid map type: " << map_type << ". Did you forget to compile with `NVHM_TOOLS_COMPILE_MAP_TYPES`?";
  throw std::runtime_error(os.str());
}

template <typename Map, typename Value, bool HasBlobs>
void run_bench_std_map_3(bool serialize_copy, map_type_t map_type) {
  if (serialize_copy) {
    run_bench_std_map_4<Map, Value, HasBlobs, true>(map_type);
  } else {
    run_bench_std_map_4<Map, Value, HasBlobs, false>(map_type);
  }
}

template <typename Map, typename Value>
void run_bench_std_map_2(bool serialize_copy, map_type_t map_type) {
  if (blob_size > 0) {
    run_bench_std_map_3<Map, Value, true>(serialize_copy, map_type);
  } else {
    run_bench_std_map_3<Map, Value, false>(serialize_copy, map_type);
  }
}

template <typename Key>
void run_bench_std_map_1(const std::string& value_type, bool serialize_copy, map_type_t map_type) {
  if (value_type == "time") {
    run_bench_std_map_2<Key, time_t>(serialize_copy, map_type);
  } else if (value_type == "void") {
    run_bench_std_map_2<Key, void>(serialize_copy, map_type);
  } else {
    throw std::runtime_error("Invalid value type: " + value_type);
  }
}

void run_bench_std_map_0(
  key_type_t key_type, const std::string& value_type, bool serialize_copy, map_type_t map_type) {
  switch (key_type) {
    case key_type_t::int32: return run_bench_std_map_1<int32_t>(value_type, serialize_copy, map_type);
    case key_type_t::int64: return run_bench_std_map_1<int64_t>(value_type, serialize_copy, map_type);
  }
  throw std::runtime_error("Unsupported `key_type`!");
}

int main(int argc, char* argv[]) {
  CLI::App app{"NVHashmap Benchmark"};

  bool print_config{false};
  key_type_t key_type{key_type_t::int64};
  bool aggressive_prefetch{true};
  std::string value_type{"time"};
  map_type_t map_type{map_type_t::nvhm_map};
  kernel_type_t kernel_type{kernel_type_t::default_};
  probe_seq_type_t probe_seq_type{probe_seq_type_t::default_};
  bool serialize_copy{false};
  bool print_header{true};

  app.add_option("--print_config", print_config, "Print config")->capture_default_str();
  app.add_option("--stat", stat, "The statistic to use for reporting")->capture_default_str()->transform(statistic_t_validator);
  app.add_option("--key_type", key_type, "Key type")->capture_default_str()->transform(key_type_t_validator);
  app.add_option("--num_keys", num_keys, "Number of keys")->capture_default_str()->check(CLI::Validator(CLI::PositiveNumber));
  app.add_option("--key_source", key_source, "Key source")->capture_default_str()->transform(key_source_t_validator);
  app.add_option("--key_poly", key_poly, "Key coefficients for polynomial key source\n(c0 + c1 * x + c2 * x^2)")->delimiter(',')->capture_default_str();
  app.add_option("--aggressive_prefetch", aggressive_prefetch, "Aggressive prefetch")->capture_default_str();
  app.add_option("--value_type", value_type, "Value type (time | void)")->capture_default_str();
  app.add_option("--blob_size", blob_size, "Blob size")->capture_default_str()->check(CLI::Validator(CLI::NonNegativeNumber));
  app.add_option("--num_workers", num_workers, "Number of workers")->capture_default_str()->check(CLI::Validator(CLI::PositiveNumber));
  app.add_option("--num_insert_trials", num_insert_trials, "Number of trials for insert")->capture_default_str()->check(CLI::Validator(CLI::NonNegativeNumber));
  app.add_option("--insert_queue_type", insert_queue_type, "Insert queue type")->capture_default_str()->transform(
    make_enum_validator<0, queue_t::shift, queue_t::ring>());
  app.add_option("--min_insert_queue_len", min_insert_queue_len, "Min insert queue length")->capture_default_str()->check(CLI::Validator(CLI::NonNegativeNumber));
  app.add_option("--max_insert_queue_len", max_insert_queue_len, "Max insert queue length")->capture_default_str()->check(CLI::Validator(CLI::NonNegativeNumber));
  app.add_option("--num_find_trials", num_find_trials, "Number of trials for find")->capture_default_str()->check(CLI::Validator(CLI::NonNegativeNumber));
  app.add_option("--find_hit_perc", find_hit_perc, "Find hit %")->capture_default_str()->check(CLI::Validator(CLI::NonNegativeNumber));
  app.add_option("--find_queue_type", find_queue_type, "Find queue type")->capture_default_str()->transform(
    make_enum_validator<0, queue_t::shift, queue_t::ring>());
  app.add_option("--min_find_queue_len", min_find_queue_len, "Min find queue length")->capture_default_str()->check(CLI::Validator(CLI::NonNegativeNumber));
  app.add_option("--max_find_queue_len", max_find_queue_len, "Max find queue length")->capture_default_str()->check(CLI::Validator(CLI::NonNegativeNumber));
  app.add_option("--check_blobs", check_blobs, "Check blobs")->capture_default_str();
  app.add_option("--seed", seed, "Randomizer seed")->default_str("random");
  app.add_option("--map_type", map_type, "Map type")->capture_default_str()->transform(map_type_t_validator);
  app.add_option("--kernel_type", kernel_type, "Kernel type")->capture_default_str()->transform(kernel_type_t_validator);
  app.add_option("--probe_seq_type", probe_seq_type, "Probe sequence type")->capture_default_str()->transform(probe_seq_type_t_validator);
  app.add_option("--serialize_copy", serialize_copy, "Serialize copy")->capture_default_str();
  app.add_option("--print_header", print_header, "Print header")->capture_default_str();
  app.add_option("--map_type_print_len", map_type_print_len, "Map type print length")->capture_default_str()->check(CLI::Range(32, 256));

  argv = app.ensure_utf8(argv);
  CLI11_PARSE(app, argc, argv);

  if (print_config) {
    std::cerr << argv[0] << " \\\n";
    std::cerr << "  --stat " << stat << " \\\n";
    std::cerr << "  --key_type " << key_type << " \\\n";
    std::cerr << "  --num_keys " << num_keys << " \\\n";
    std::cerr << "  --key_source " << key_source << " \\\n";
    std::cerr << "  --key_poly " << key_poly[0] << ',' << key_poly[1] << ',' << key_poly[2] << " \\\n";
    std::cerr << "  --aggressive_prefetch " << aggressive_prefetch << " \\\n";
    std::cerr << "  --value_type " << value_type << " \\\n";
    std::cerr << "  --blob_size " << blob_size << " \\\n";
    std::cerr << "  --num_workers " << num_workers << " \\\n";
    std::cerr << "  --num_insert_trials " << num_insert_trials << " \\\n";
    std::cerr << "  --insert_queue_type " << insert_queue_type << " \\\n";
    std::cerr << "  --min_insert_queue_len " << min_insert_queue_len << " \\\n";
    std::cerr << "  --max_insert_queue_len " << max_insert_queue_len << " \\\n";
    std::cerr << "  --num_find_trials " << num_find_trials << " \\\n";
    std::cerr << "  --find_hit_perc " << find_hit_perc << " \\\n";
    std::cerr << "  --find_queue_type " << find_queue_type << " \\\n";
    std::cerr << "  --min_find_queue_len " << min_find_queue_len << " \\\n";
    std::cerr << "  --max_find_queue_len " << max_find_queue_len << " \\\n";
    std::cerr << "  --check_blobs " << check_blobs << " \\\n";
    std::cerr << "  --seed " << seed << " \\\n";
    std::cerr << "  --map_type " << map_type << " \\\n";
    std::cerr << "  --kernel_type " << kernel_type << " \\\n";
    std::cerr << "  --probe_seq_type " << probe_seq_type << " \\\n";
    std::cerr << "  --serialize_copy " << serialize_copy << " \\\n";
    std::cerr << "  --print_config " << print_config << " \\\n";
    std::cerr << "  --print_header " << print_header << " \\\n";
    std::cerr << "  --map_type_print_len " << map_type_print_len << " \\\n";
  }

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

  const int num_trials{static_cast<int>(std::max(num_insert_trials, num_find_trials))};
  const int num_bench_cols{3 * (num_trials - 1) + 5 * num_trials};
  const int bench_str_cols{static_cast<int>(strlen("benchmark times in ms"))};
  const int bench_not_str_cols{std::max(num_bench_cols - bench_str_cols, 0)};
  const int bench_str_left_pad{bench_not_str_cols / 2};
  const int bench_str_right_pad{bench_not_str_cols - bench_str_left_pad};

  if (print_header) {
    std::cout
      << "| "
      << std::left << std::setw(static_cast<int>(map_type_print_len)) << "map_count" << " | "
      << std::right << std::setw(4) << "#wrk" << " | "
      << std::left << std::setw(6) << "op" << " | "
      << std::right << std::setw(4) << "hit%" << " | " << std::setw(5) << "queue" << " | " << std::setw(4) << "qlen"
      << " | " << std::setw(bench_str_left_pad) << "" << "benchmark times in ms" << std::setw(bench_str_right_pad) << ""
      << " | " << std::setw(10) << "size" << " | " << std::setw(10) << "capacity" << " | " << std::setw(12) << "transactions"
      << " |\n" << std::flush;

    std::cout
      << "| "
      << std::left << std::setw(static_cast<int>(map_type_print_len)) << "" << " | "
      << std::right << std::setw(4) << "" << " | "
      << std::left << std::setw(6) << "" << " | "
      << std::right << std::setw(4) << "" << " | " << std::setw(5) << "" << " | " << std::setw(4) << "";
    for (int_t i{}; i < std::max(num_insert_trials, num_find_trials); ++i) {
      std::cout << " | " << std::setw(5) << to_string('#', i);
    }
    std::cout
      << " | " << std::setw(10) << "" << " | " << std::setw(10) << "" << " | " << std::setw(12) << ""
      << " |\n" << std::flush;

    std::cout
      << "| " << std::setfill('-')
      << std::left << std::setw(static_cast<int>(map_type_print_len)) << "" << " | "
      << std::right << std::setw(4) << "" << " | "
      << std::left << std::setw(6) << "" << " | "
      << std::right << std::setw(4) << "" << " | " << std::setw(5) << "" << " | " << std::setw(4) << "";
    for (int_t i{}; i < std::max(num_insert_trials, num_find_trials); ++i) {
      std::cout << " | " << std::setw(5) << "";
    }
    std::cout
      << " | " << std::setw(10) << "" << " | " << std::setw(10) << "" << " | " << std::setw(12) << ""
      << " |\n" << std::setfill(' ') << std::flush;
  }

  if (map_type == map_type_t::nvhm_map) {
    run_bench_nvhm_map_0(key_type, aggressive_prefetch, value_type, kernel_type, probe_seq_type);
  } else {
    run_bench_std_map_0(key_type, value_type, serialize_copy, map_type);
  }

  return 0;
}
