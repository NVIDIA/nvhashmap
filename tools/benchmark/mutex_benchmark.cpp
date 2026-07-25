/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <nvhashmap/mutex.hpp>

using namespace nvhm;
#include "../tools_common.hpp"

#include <thread>

using namespace nvhm;

// ~100 ns of dependent local work (xor-shift chain).
NVHM_ALWAYS_INLINE int_t local_work(int_t x, int_t n) noexcept {
  for (int_t i{}; i < n; ++i) {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
  }
  return x;
}

NVHM_MAKE_ENUM_WITH_VALIDATOR_(scenario_t,
  high_contention, // N threads, exclusive lock/unlock back-to-back.
  low_contention,  // N threads, short critical section, ~90% of the time spent outside the lock.
  read_mostly      // N threads, `read_perc`% shared reads, rest exclusive writes.
);

scenario_t scenario{scenario_t::high_contention};
int_t read_perc{95};
int_t cs_work{16};

template <typename Mutex>
NVHM_NO_INLINE std::tuple<double, int_t, int_t, bool> run(const int_t num_workers, const std::chrono::milliseconds test_duration) {
  using mutex_t = Mutex;
  alignas(cache_line_size) mutex_t m;
  alignas(cache_line_size) std::pair<int_t, int_t> c;

  std::atomic<int_t> ready{};
  std::atomic<bool> go{};
  std::atomic<bool> stop{};

  std::vector<std::thread> threads;
  threads.reserve(to_uint(num_workers));

  std::vector<int_t> per_thread_ops(to_uint(num_workers));
  std::vector<int_t> per_thread_sinks(to_uint(num_workers));

  for (int_t i{}; i < num_workers; ++i) {
    threads.emplace_back([&, i]() {
      int_t x{i * 2654435761 + 1};
      int_t local_ops{};
      int_t drift{};

      ready.fetch_add(1, std::memory_order_relaxed);
      while (!go.load(std::memory_order_acquire)) {}

      while (!stop.load(std::memory_order_relaxed)) {
        switch (scenario) {
          case scenario_t::high_contention: {
            std::unique_lock l{m};
            ++c.first;
            x = local_work(x, cs_work);
          } break;
          case scenario_t::low_contention: {
            {
              std::unique_lock l{m};
              ++c.first;
              x = local_work(x, cs_work);
            }
            x = local_work(x, 32);
          } break;
          case scenario_t::read_mostly: {
            if (local_ops % 100 < read_perc) {
              // Read
              using lock_t = std::conditional_t<std::is_same_v<mutex_t, std::mutex>, std::unique_lock<mutex_t>, std::shared_lock<mutex_t>>;
              lock_t l{m};
              drift += c.first - c.second;  // Always 0 under the invariant.
              x = local_work(x, cs_work);
            } else {
              // Write
              std::unique_lock l{m};
              ++c.first;
              ++c.second;
              x = local_work(x, cs_work);
            }
          } break;
        }
        ++local_ops;
      }

      per_thread_ops[to_uint(i)] = local_ops;
      per_thread_sinks[to_uint(i)] = x + drift;
    });
  }

  while (ready.load(std::memory_order_relaxed) != num_workers) {}
  stopwatch clock;
  go.store(true, std::memory_order_release);

  std::this_thread::sleep_for(test_duration);
  stop.store(true, std::memory_order_relaxed);

  const double dur{std::chrono::duration<double>(clock.elapsed()).count()};

  for (std::thread& t : threads) {
    t.join();
  }

  int_t ops{};
  int_t sink{};  // Dummy to prevent certain compiler optimizations.
  for (int_t i{}; i < num_workers; ++i) {
    ops += per_thread_ops[to_uint(i)];
    sink += per_thread_sinks[to_uint(i)];
  }

  // Integrity check:
  // Every exclusive op incremented `n0` exactly once; readers saw `n0 == n1`.
  // This also prevents the optimizer from pruning some parts of the ops.
  bool ok{false};
  switch (scenario) {
    case scenario_t::high_contention:
    case scenario_t::low_contention:
      ok = c.first == ops;
      break;
    case scenario_t::read_mostly:
      ok = c.first == c.second;
      break;
  }

  return {dur, ops, sink, ok};
}

int_t duration_ms{300};
std::vector<int_t> num_workers{1, 2, 4, 8, 16, 32, 64};

constexpr double mops_per_sec(double duration, int_t ops) noexcept {
  return static_cast<double>(ops) / (duration * 1e6);
}

constexpr double ns_per_op(double duration, int_t ops) noexcept {
  return duration / static_cast<double>(ops) * 1e9;
}
template <typename Mutex>
void run() {
  using mutex_t = Mutex;

  const auto fn{[&](const int_t num_workers) {
    (void)run<mutex_t>(num_workers, std::chrono::milliseconds(duration_ms / 4));  // Warm-up.
    auto [dur, ops, sink, ok]{run<mutex_t>(num_workers, std::chrono::milliseconds(duration_ms))};

    std::cout
      << "| " << std::left
      << std::setw(18) << scenario << " | "
      << std::setw(50) << type_to_string<mutex_t>() << " | "
      << std::right << std::setw(8) << num_workers << " | "
      << std::fixed
      << std::setw(12) << std::setprecision(2) << mops_per_sec(dur, ops) << " | "
      << std::setw(12) << std::setprecision(1) << ns_per_op(dur, ops) * static_cast<double>(num_workers) << " | "
      << std::left
      << std::setw(8) << (ok ? "ok" : "fail") << " |\n"
      << std::flush;
  }};
  
  std::for_each(num_workers.begin(), num_workers.end(), fn);
}

NVHM_MAKE_ENUM_WITH_VALIDATOR_(mutex_type_t,
  nvhm_spin_wait_mutex,
  std_mutex,
  std_shared_mutex
);

void run(mutex_type_t mutex_type, int_t spin_count, bool use_wait) {
  switch (mutex_type) {
    case mutex_type_t::nvhm_spin_wait_mutex:
      if (use_wait) {
        switch (spin_count) {
          #if defined(__cpp_lib_atomic_wait)
          case   0: return run<spin_wait_mutex<int32_t,   0, true>>();
          case   1: return run<spin_wait_mutex<int32_t,   1, true>>();
          case   2: return run<spin_wait_mutex<int32_t,   2, true>>();
          case   4: return run<spin_wait_mutex<int32_t,   4, true>>();
          case   8: return run<spin_wait_mutex<int32_t,   8, true>>();
          case  16: return run<spin_wait_mutex<int32_t,  16, true>>();
          case  32: return run<spin_wait_mutex<int32_t,  32, true>>();
          case  64: return run<spin_wait_mutex<int32_t,  64, true>>();
          case 128: return run<spin_wait_mutex<int32_t, 128, true>>();
          case 256: return run<spin_wait_mutex<int32_t, 256, true>>();
          #endif
        }
        throw std::runtime_error("This feature needs atomic_wait, which is not supported prior to C++20!");
      } else {
        return run<spin_wait_mutex<int32_t, 1, false>>();
      }

    case mutex_type_t::std_mutex:
      return run<std::mutex>();

    case mutex_type_t::std_shared_mutex:
      return run<std::shared_mutex>();
  }

  throw std::runtime_error("Unsupported combination of `mutex_type`, `spin_count` and `use_wait`!");
}

int main(int argc, char* argv[]) {
  CLI::App app{"NVHashmap Mutex Benchmark"};

  mutex_type_t mutex_type{mutex_type_t::nvhm_spin_wait_mutex};
  int_t spin_count{mutex_default_spin_iterations};
  bool use_wait{mutex_default_use_wait};
  bool print_header{true};

  app.add_option("--scenario", scenario, "The scenario to test")->capture_default_str()->transform(scenario_t_validator);
  app.add_option("--read_perc", read_perc, "Percentage of shared reads in the `read_mostly` scenario")->capture_default_str()->check(CLI::Range(0, 100));
  app.add_option("--cs_work", cs_work, "Amount of work to conduct within the critical section (~3 ns each on Grace)")->capture_default_str()->check(CLI::Validator(CLI::NonNegativeNumber));
  app.add_option("--duration_ms", duration_ms, "Measured duration per cell in milliseconds")->capture_default_str()->check(CLI::Validator(CLI::NonNegativeNumber));
  app.add_option("--num_workers", num_workers, "Worker counts to sweep")->delimiter(',')->option_text("INT[,INT...] [1,2,4,8,16,32,64]");
  app.add_option("--mutex_type", mutex_type, "Mutex type to test")->capture_default_str()->transform(mutex_type_t_validator);
  app.add_option("--spin_count", spin_count, "Spin-count for NVHM mutex")->capture_default_str()->check(CLI::Validator(CLI::NonNegativeNumber));
  app.add_option("--use_wait", use_wait, "Whether or not use atomic wait for NVHM mutex")->capture_default_str();
  app.add_option("--print_header", print_header, "Print header")->capture_default_str();

  CLI11_PARSE(app, argc, argv);

  if (print_header) {  
    std::cout
      << "| " << std::left
      << std::setw(18) << "Scenario" << " | "
      << std::setw(50) << "Mutex Type" << " | "
      << std::right << std::setw(8) << "Threads" << " | "
      << std::setw(12) << "MOPS" << " | "
      << std::setw(12) << "ns/op" << " | "
      << std::left
      << std::setw(8) << "Status" << " |\n"
      << std::flush;
    
    std::cout
      << "| " << std::left << std::setfill('-')
      << std::setw(18) << "" << " | "
      << std::setw(50) << "" << " | "
      << std::right << std::setw(8) << "" << " | "
      << std::setw(12) << "" << " | "
      << std::setw(12) << "" << " | "
      << std::left
      << std::setw(8) << "" << " |\n"
      << std::setfill(' ') << std::flush;
  }

  run(mutex_type, spin_count, use_wait);

  return 0;
}
