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

#pragma once

#include "common.hpp"
#include <atomic>
#include <mutex>
#include <optional>
#include <random>
#include <shared_mutex>

namespace nvhm {

/**
 * Consume `unique_lock` and construct a `shared_lock` through downgrading.
 *
 * @param lock The `unique_lock`.
 * @returns The `shared_lock`.
 */
template <typename Mutex>
[[nodiscard]] NVHM_ALWAYS_INLINE std::shared_lock<Mutex> downgrade(std::unique_lock<Mutex>&& lock) noexcept {
  Mutex* m{lock.mutex()};
  if (!m) {
    return {};
  }

  bool owns_lock{lock.owns_lock()};
  lock.release();

  if (!owns_lock) {
    return {*m, std::defer_lock};
  }

  m->downgrade();
  return {*m, std::adopt_lock};
}

/**
 * Consume `shared_lock` and construct a `unique_lock` through upgrading.
 *
 * Note, this is a best effort. It will try to convert the shared lock. But you could do
 * this with multiple threads in parallel, which would cause a deadlock. So, the fallback
 * is to unlock the shared lock and re-acquire the lock as a unique lock. In that case,
 * changes to the lock-guarded data structure may occur while the lock is not held.
 * 
 * @param lock The `shared_lock`.
 * @returns The `unique_lock`.
 */
template <typename Mutex>
[[nodiscard]] NVHM_ALWAYS_INLINE std::unique_lock<Mutex> upgrade(std::shared_lock<Mutex>&& lock) noexcept {
  Mutex* m{lock.mutex()};
  if (!m) {
    return {};
  }

  bool owns_lock{lock.owns_lock()};
  lock.release();

  if (!owns_lock) {
    return {*m, std::defer_lock};
  }

  if (!m->try_upgrade()) {
    NVHM_LOG_(log_level_t::info, "Direct lock upgrade failed. Attempting to re-lock.");
    m->unlock_shared();
    m->lock();
  }
  return {*m, std::adopt_lock};
}

/**
 * Tries to consume `shared_lock` and construct a `unique_lock` through upgrading.
 *
 * @param lock The `shared_lock`.
 * @returns The `unique_lock`, or std::nullopt, if the conversion was unsuccessful. In this case the original `shared_lock` remains valid.
 */
template <typename Mutex>
[[nodiscard]] NVHM_ALWAYS_INLINE std::optional<std::unique_lock<Mutex>> try_upgrade(std::shared_lock<Mutex>& lock) noexcept {
  Mutex* m{lock.mutex()};
  if (m && lock.owns_lock() && m->try_upgrade()) {
    lock.release();
    return std::unique_lock<Mutex>{*m, std::adopt_lock};
  }

  return std::nullopt;
}

namespace detail {

inline thread_local std::random_device rd;
inline thread_local std::default_random_engine rng{rd()};

template <int_t MinBackOff, int_t MaxBackOff>
NVHM_ALWAYS_INLINE void backoff() noexcept {
  static_assert(MinBackOff >= 8 && has_single_bit(MinBackOff));
  static_assert(MaxBackOff >= MinBackOff && has_single_bit(MaxBackOff));

  const int_t n{std::uniform_int_distribution<int_t>{MinBackOff, MaxBackOff}(rng)};
  for (int_t i{}; i < n; ++i) {
    #if defined(__x86_64__) || defined(__i386__)
    asm volatile("pause" : : : "memory");
    #elif defined(__aarch64__) || defined(__arm__)
    // The consensus online is that `isb` is better suited for this task than `yield`.
    asm volatile("isb" : : : "memory");
    #elif defined(__riscv)
    // https://sourceware.org/git/?p=glibc.git;f=sysdeps/riscv/atomic-machine.h;hb=HEAD#l49
    asm volatile(".insn i 0x0f, 0, x0, x0, 0x010" : : : "memory");
    #else
    #error Unsupported architecture!
    #endif
  }
}

}  // namespace detail

constexpr static int_t mutex_default_spin_iterations{32};
constexpr static int_t mutex_default_min_back_off{8};
#if defined(__cpp_lib_atomic_wait)
constexpr static int_t mutex_default_use_wait{true};
#else
constexpr static int_t mutex_default_use_wait{false};
#endif
constexpr static int_t mutex_default_max_back_off{32};

/**
 * A mutex that uses a spin loop to acquire locks. If the lock is not acquired after a certain
 * number of iterations (default: ~8 us on Grace), it will fallback to a futex wait queue
 * (~10 us on Grace) if `UseWait = true`.
 */
template <
  typename StateType = int32_t,
  int_t SpinIterations = mutex_default_spin_iterations, bool UseWait = mutex_default_use_wait,
  int_t MinBackOff = mutex_default_min_back_off, int_t MaxBackOff = mutex_default_max_back_off>
class spin_wait_mutex {
 public:
  using state_type = StateType;
  static_assert(std::is_integral_v<state_type>, "`state_type` must be integral!");
  static_assert(std::is_signed_v<state_type>, "`state_type` must be signed!");
  static_assert(num_bytes_v<state_type> <= sizeof(int_t));

  constexpr static state_type max_num_readers{std::numeric_limits<state_type>::max()};
  constexpr static int_t spin_iterations{SpinIterations};
  static_assert(spin_iterations >= 0);
  constexpr static bool use_wait{UseWait};
  static_assert(spin_iterations > 0 || use_wait);
  #if !defined(__cpp_lib_atomic_wait)
  static_assert(!use_wait, "Waiting for atomics is not supported before C++-20!");
  #endif

  constexpr static int_t min_backoff{MinBackOff};
  constexpr static int_t max_backoff{MaxBackOff};

  constexpr spin_wait_mutex() noexcept : state_{0} {}
  spin_wait_mutex(const spin_wait_mutex&) = delete;
  spin_wait_mutex(spin_wait_mutex&&) = delete;
  spin_wait_mutex& operator=(const spin_wait_mutex&) = delete;
  spin_wait_mutex& operator=(spin_wait_mutex&&) = delete;

  [[nodiscard]] NVHM_ALWAYS_INLINE bool is_locked() const noexcept {
    return state_.load(std::memory_order_relaxed) < 0;
  }

  NVHM_ALWAYS_INLINE void lock() noexcept {
    for (int_t i{};; ++i) {
      // Redundant but faster because unlike CAS, it will not implicitly invalidate the cache line for other spinners.
      state_type s{state_.load(std::memory_order_relaxed)};

      // Spin for a while.
      if (i < spin_iterations || !use_wait) {
        // If empty, try claiming the mutex.
        if (s == 0) {
          if (state_.compare_exchange_weak(s, -1, std::memory_order_acquire, std::memory_order_relaxed)) {
            return;
          }
        }

        detail::backoff<min_backoff, max_backoff>();
        continue;
      }
      
      // Fallback to wait queue.
      if constexpr (use_wait) {
        #if defined(__cpp_lib_atomic_wait)
        if (s == 0) {
          // If empty, try claiming the mutex.
          if (state_.compare_exchange_strong(s, -1, std::memory_order_acquire, std::memory_order_relaxed)) {
            return;
          }
        }

        // Await notification from whoever won the race.
        state_.wait(s, std::memory_order_relaxed);

        // Revert to spinning and repeat.
        i = 0;
        #endif
      }
    }
  }

  NVHM_ALWAYS_INLINE bool try_lock() noexcept {
    state_type s{};
    return state_.compare_exchange_strong(s, -1, std::memory_order_acquire, std::memory_order_relaxed);
  }
  
  NVHM_ALWAYS_INLINE void unlock() noexcept {
    NVHM_ASSERT_(state_.load(std::memory_order_relaxed) < 0, "Mutex was not write-locked!");
    state_.store(0, std::memory_order_release);

    if constexpr (use_wait) {
      #if defined(__cpp_lib_atomic_wait)
      state_.notify_all();
      #endif
    }
  }

  NVHM_ALWAYS_INLINE void downgrade() noexcept {
    state_type s{state_.exchange(1, std::memory_order_release)};
    NVHM_ASSERT_(s < 0, "Mutex was not write-locked!");

    if constexpr (use_wait) {
      #if defined(__cpp_lib_atomic_wait)
      state_.notify_all();
      #endif
    }
  }

  [[nodiscard]] NVHM_ALWAYS_INLINE int_t num_shared_locks() const noexcept {
    return std::max(state_.load(std::memory_order_relaxed), {});
  }

  NVHM_ALWAYS_INLINE int_t lock_shared() noexcept {
    for (int_t i{};; ++i) {
      // Redundant but faster because unlike CAS, it will not implicitly invalidate the cache line for other spinners.
      state_type s{state_.load(std::memory_order_relaxed)};

      // Spin for a while.
      if (i < spin_iterations || !use_wait) {
        if (s >= 0) {
          // If no writer, try claiming the mutex.
          NVHM_ASSERT_(s < max_num_readers, "Mutex reader overflow!");
          if (state_.compare_exchange_weak(s, s + 1, std::memory_order_acquire, std::memory_order_relaxed)) {
            return s + 1;
          }
        }

        detail::backoff<min_backoff, max_backoff>();
        continue;
      }

      // Fallback to wait queue.
      if constexpr (use_wait) {
        #if defined(__cpp_lib_atomic_wait)
        if (s >= 0) {
          // If no writer present, try claiming the lock.
          NVHM_ASSERT_(s < max_num_readers, "Mutex reader overflow!");
          if (state_.compare_exchange_weak(s, s + 1, std::memory_order_acquire, std::memory_order_relaxed)) {
            return s + 1;
          }
        } else {
          // Await notification from the writer.
          state_.wait(s, std::memory_order_relaxed);
        }

        // Revert to spinning and repeat.
        i = 0;
        #endif
      }
    }
  }

  NVHM_ALWAYS_INLINE bool try_lock_shared() noexcept {
    state_type s{state_.load(std::memory_order_relaxed)};
    while (s >= 0) {
      NVHM_ASSERT_(s < max_num_readers, "Mutex reader overflow!");
      if (state_.compare_exchange_weak(s, s + 1, std::memory_order_acquire, std::memory_order_relaxed)) {
        return true;
      }
    }
    return false;
  }

  NVHM_ALWAYS_INLINE int_t unlock_shared() noexcept {
    state_type s{state_.fetch_sub(1, std::memory_order_release)};
    NVHM_ASSERT_(s > 0, "Mutex was not read-locked!");

    if constexpr (use_wait) {
      #if defined(__cpp_lib_atomic_wait)
      if (s == 1) {
        // Wake up a queued writer (if exists).
        this->state_.notify_one();
      }
      #endif
    }

    return s - 1;
  }

  NVHM_ALWAYS_INLINE bool try_upgrade() noexcept {
    state_type s{1};
    return state_.compare_exchange_strong(s, -1, std::memory_order_acquire, std::memory_order_relaxed);
  }

 protected:
  std::atomic<state_type> state_;
};

using spin_wait_mutex_t = spin_wait_mutex<>;

}  // namespace nvhm