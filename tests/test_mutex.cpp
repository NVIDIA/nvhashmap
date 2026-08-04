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
#include "test_common.hpp"
#include <gtest/gtest.h>
#include <chrono>
#include <thread>

using namespace nvhm;

template <typename Mutex>
void test_mutex_traits() {
  using mutex_t = Mutex;

  static_assert(!std::is_copy_constructible_v<mutex_t>);
  static_assert(!std::is_move_constructible_v<mutex_t>);
  static_assert(!std::is_copy_assignable_v<mutex_t>);
  static_assert(!std::is_move_assignable_v<mutex_t>);

  EXPECT_GT(mutex_t::max_num_readers, 0);
}

template <typename Mutex>
void test_mutex_lock_unlock(const int_t num_iter = 1024, const int_t num_readers = 16) {
  using mutex_t = Mutex;

  mutex_t m;

  EXPECT_FALSE(m.is_locked());
  EXPECT_EQ(m.num_shared_locks(), 0);

  for (int_t i{}; i < num_iter; ++i) {
    m.lock();
    EXPECT_TRUE(m.is_locked());
    EXPECT_EQ(m.num_shared_locks(), 0);

    m.unlock();
    EXPECT_FALSE(m.is_locked());
    EXPECT_EQ(m.num_shared_locks(), 0);

    int_t r{};
    while (r < num_readers) {
      EXPECT_EQ(m.lock_shared(), ++r);
      EXPECT_FALSE(m.is_locked());
      EXPECT_EQ(m.num_shared_locks(), r);
    }
    while (r > 0) {
      EXPECT_EQ(m.unlock_shared(), --r);
      EXPECT_FALSE(m.is_locked());
      EXPECT_EQ(m.num_shared_locks(), r);
    }

    EXPECT_FALSE(m.is_locked());
    EXPECT_EQ(m.num_shared_locks(), 0);
  }
}

template <typename Mutex>
void test_mutex_try_lock(const int_t num_iter = 1024, const int_t num_readers = 16) {
  using mutex_t = Mutex;

  mutex_t m;

  for (int_t i{}; i < num_iter; ++i) {
    EXPECT_TRUE(m.try_lock());
    EXPECT_TRUE(m.is_locked());
    EXPECT_EQ(m.num_shared_locks(), 0);

    EXPECT_FALSE(m.try_lock());
    EXPECT_FALSE(m.try_lock_shared());

    m.unlock();
    EXPECT_FALSE(m.is_locked());
    EXPECT_EQ(m.num_shared_locks(), 0);

    int_t r{};
    while (r < num_readers) {
      EXPECT_TRUE(m.try_lock_shared());
      EXPECT_FALSE(m.is_locked());
      EXPECT_EQ(m.num_shared_locks(), ++r);
    }
    while (r > 0) {
      EXPECT_EQ(m.unlock_shared(), --r);
      EXPECT_FALSE(m.is_locked());
      EXPECT_EQ(m.num_shared_locks(), r);
    }

    EXPECT_FALSE(m.is_locked());
    EXPECT_EQ(m.num_shared_locks(), 0);
  }
}

template <typename Mutex>
void test_mutex_downgrade(const int_t num_iter = 1024) {
  using mutex_t = Mutex;

  mutex_t m;

  for (int_t i{}; i < num_iter; ++i) {
    m.lock();
    EXPECT_TRUE(m.is_locked());
    EXPECT_EQ(m.num_shared_locks(), 0);

    m.downgrade();
    EXPECT_FALSE(m.is_locked());
    EXPECT_EQ(m.num_shared_locks(), 1);

    // Other readers may join after the downgrade, writers may not.
    EXPECT_TRUE(m.try_lock_shared());
    EXPECT_FALSE(m.try_lock());

    EXPECT_EQ(m.unlock_shared(), 1);
    EXPECT_EQ(m.unlock_shared(), 0);
  }
}

template <typename Mutex>
void test_mutex_try_upgrade(const int_t num_iter = 1024) {
  using mutex_t = Mutex;

  mutex_t m;

  for (int_t i{}; i < num_iter; ++i) {
    m.lock_shared();
    EXPECT_FALSE(m.is_locked());
    EXPECT_EQ(m.num_shared_locks(), 1);

    EXPECT_TRUE(m.try_upgrade());
    EXPECT_TRUE(m.is_locked());
    EXPECT_EQ(m.num_shared_locks(), 0);

    m.unlock();
    EXPECT_FALSE(m.is_locked());
    EXPECT_EQ(m.num_shared_locks(), 0);

    // A free mutex cannot be upgraded.
    EXPECT_FALSE(m.try_upgrade());
    EXPECT_FALSE(m.is_locked());
    EXPECT_EQ(m.num_shared_locks(), 0);

    // With more than one reader, the upgrade fails and changes nothing.
    m.lock_shared();
    m.lock_shared();
    EXPECT_FALSE(m.is_locked());
    EXPECT_EQ(m.num_shared_locks(), 2);

    EXPECT_FALSE(m.try_upgrade());
    EXPECT_FALSE(m.is_locked());
    EXPECT_EQ(m.num_shared_locks(), 2);

    EXPECT_EQ(m.unlock_shared(), 1);
    EXPECT_EQ(m.unlock_shared(), 0);
  }
}

template <typename Mutex>
void test_mutex_std_locks(const int_t num_iter = 1024) {
  using mutex_t = Mutex;

  mutex_t m;

  for (int_t i{}; i < num_iter; ++i) {
    std::lock_guard<mutex_t> lock{m};
    EXPECT_TRUE(m.is_locked());
    EXPECT_EQ(m.num_shared_locks(), 0);
  }
  EXPECT_FALSE(m.is_locked());

  for (int_t i{}; i < num_iter; ++i) {
    std::unique_lock<mutex_t> lock{m};
    EXPECT_TRUE(m.is_locked());
    EXPECT_EQ(m.num_shared_locks(), 0);
    EXPECT_TRUE(lock.owns_lock());

    lock.unlock();
    EXPECT_FALSE(m.is_locked());
    EXPECT_EQ(m.num_shared_locks(), 0);
    EXPECT_FALSE(lock.owns_lock());

    lock.lock();
    EXPECT_TRUE(m.is_locked());
    EXPECT_EQ(m.num_shared_locks(), 0);
    EXPECT_TRUE(lock.owns_lock());
    EXPECT_FALSE(m.try_lock());
    
    lock.unlock();
    EXPECT_FALSE(m.is_locked());
    EXPECT_EQ(m.num_shared_locks(), 0);
    EXPECT_FALSE(lock.owns_lock());

    EXPECT_TRUE(lock.try_lock());
    EXPECT_TRUE(m.is_locked());
    EXPECT_EQ(m.num_shared_locks(), 0);
    EXPECT_TRUE(lock.owns_lock());
  }
  EXPECT_FALSE(m.is_locked());
  EXPECT_EQ(m.num_shared_locks(), 0);

  for (int_t i{}; i < num_iter; ++i) {
    std::shared_lock<mutex_t> lock{m};
    EXPECT_FALSE(m.is_locked());
    EXPECT_EQ(m.num_shared_locks(), 1);
    EXPECT_TRUE(lock.owns_lock());

    lock.unlock();
    EXPECT_FALSE(m.is_locked());
    EXPECT_EQ(m.num_shared_locks(), 0);
    EXPECT_FALSE(lock.owns_lock());

    lock.lock();
    EXPECT_FALSE(m.is_locked());
    EXPECT_EQ(m.num_shared_locks(), 1);
    EXPECT_TRUE(lock.owns_lock());
    
    lock.unlock();
    EXPECT_FALSE(m.is_locked());
    EXPECT_EQ(m.num_shared_locks(), 0);
    EXPECT_FALSE(lock.owns_lock());

    EXPECT_TRUE(lock.try_lock());
    EXPECT_FALSE(m.is_locked());
    EXPECT_EQ(m.num_shared_locks(), 1);
    EXPECT_TRUE(lock.owns_lock());
  }
  EXPECT_FALSE(m.is_locked());
  EXPECT_EQ(m.num_shared_locks(), 0);

  for (int_t i{}; i < num_iter; ++i) {
    std::shared_lock<mutex_t> l0{m}, l1{m}, l2{m};
    EXPECT_FALSE(m.is_locked());
    EXPECT_EQ(m.num_shared_locks(), 3);
    EXPECT_TRUE(l0.owns_lock());
    EXPECT_TRUE(l1.owns_lock());
    EXPECT_TRUE(l2.owns_lock());

    l1.unlock();
    EXPECT_FALSE(m.is_locked());
    EXPECT_EQ(m.num_shared_locks(), 2);
  }
  EXPECT_FALSE(m.is_locked());
  EXPECT_EQ(m.num_shared_locks(), 0);

  for (int_t i{}; i < num_iter; ++i) {
    std::shared_lock<mutex_t> l0{m, std::defer_lock};
    std::shared_lock<mutex_t> l1{m, std::defer_lock};
    EXPECT_FALSE(m.is_locked());
    EXPECT_EQ(m.num_shared_locks(), 0);
    EXPECT_FALSE(l0.owns_lock());
    EXPECT_FALSE(l1.owns_lock());

    std::lock(l0, l1);
    EXPECT_FALSE(m.is_locked());
    EXPECT_EQ(m.num_shared_locks(), 2);
    EXPECT_TRUE(l0.owns_lock());
    EXPECT_TRUE(l1.owns_lock());
  }
  EXPECT_FALSE(m.is_locked());
  EXPECT_EQ(m.num_shared_locks(), 0);

  for (int_t i{}; i < num_iter; ++i) {
    m.lock_shared();
    m.lock_shared();
    EXPECT_FALSE(m.is_locked());
    EXPECT_EQ(m.num_shared_locks(), 2);
    
    std::shared_lock<mutex_t> l0{m, std::adopt_lock};
    std::shared_lock<mutex_t> l1{m, std::adopt_lock};
    EXPECT_TRUE(l0.owns_lock());
    EXPECT_TRUE(l1.owns_lock());
  }
  EXPECT_FALSE(m.is_locked());
  EXPECT_EQ(m.num_shared_locks(), 0);
}

template <typename Mutex>
void test_mutex_std_lock_downgrade(const int_t num_iter = 1024) {
  using mutex_t = Mutex;

  // Empty lock, becomes empty lock.
  for (int_t i{}; i < num_iter; ++i) {
    std::unique_lock<mutex_t> l0{};
    EXPECT_EQ(l0.mutex(), nullptr);
    EXPECT_FALSE(l0.owns_lock());

    std::shared_lock<mutex_t> l1{downgrade(std::move(l0))};
    EXPECT_EQ(l0.mutex(), nullptr);
    EXPECT_FALSE(l0.owns_lock());
    EXPECT_EQ(l1.mutex(), nullptr);
    EXPECT_FALSE(l1.owns_lock());
  }

  mutex_t m;

  // Deferred lock becomes deferred lock.
  for (int_t i{}; i < num_iter; ++i) {
    std::unique_lock<mutex_t> l0{m, std::defer_lock};
    EXPECT_FALSE(m.is_locked());
    EXPECT_EQ(m.num_shared_locks(), 0);
    EXPECT_EQ(l0.mutex(), &m);
    EXPECT_FALSE(l0.owns_lock());

    std::shared_lock<mutex_t> l1{downgrade(std::move(l0))};
    EXPECT_FALSE(m.is_locked());
    EXPECT_EQ(m.num_shared_locks(), 0);
    EXPECT_EQ(l0.mutex(), nullptr);
    EXPECT_FALSE(l0.owns_lock());
    EXPECT_EQ(l1.mutex(), &m);
    EXPECT_FALSE(l1.owns_lock());
  }
  EXPECT_FALSE(m.is_locked());
  EXPECT_EQ(m.num_shared_locks(), 0);

  // Actual lock becomes actual lock.
  for (int_t i{}; i < num_iter; ++i) {
    std::unique_lock<mutex_t> l0{m};
    EXPECT_TRUE(m.is_locked());
    EXPECT_EQ(m.num_shared_locks(), 0);
    EXPECT_EQ(l0.mutex(), &m);
    EXPECT_TRUE(l0.owns_lock());

    std::shared_lock<mutex_t> l1{downgrade(std::move(l0))};
    EXPECT_FALSE(m.is_locked());
    EXPECT_EQ(m.num_shared_locks(), 1);
    EXPECT_EQ(l0.mutex(), nullptr);
    EXPECT_FALSE(l0.owns_lock());
    EXPECT_EQ(l1.mutex(), &m);
    EXPECT_TRUE(l1.owns_lock());
  }
  EXPECT_FALSE(m.is_locked());
  EXPECT_EQ(m.num_shared_locks(), 0);
}

template <typename Mutex>
void test_mutex_std_lock_try_upgrade(const int_t num_iter = 1024) {
  using mutex_t = Mutex;

  // 0 readers: Empty locks cannot be upgraded.
  for (int_t i{}; i < num_iter; ++i) {
    std::shared_lock<mutex_t> l0;
    EXPECT_EQ(l0.mutex(), nullptr);
    EXPECT_FALSE(l0.owns_lock());

    std::optional<std::unique_lock<mutex_t>> l1{try_upgrade(l0)};
    EXPECT_FALSE(l1.has_value());
  }

  mutex_t m;

  // 0 readers: Deferred locks cannot be upgraded. Original lock remains untouched.
  for (int_t i{}; i < num_iter; ++i) {
    std::shared_lock l0{m, std::defer_lock};
    EXPECT_FALSE(m.is_locked());
    EXPECT_EQ(m.num_shared_locks(), 0);
    EXPECT_EQ(l0.mutex(), &m);
    EXPECT_FALSE(l0.owns_lock());

    std::optional<std::unique_lock<mutex_t>> l1{try_upgrade(l0)};
    EXPECT_FALSE(m.is_locked());
    EXPECT_EQ(m.num_shared_locks(), 0);
    EXPECT_EQ(l0.mutex(), &m);
    EXPECT_FALSE(l0.owns_lock());
    EXPECT_FALSE(l1.has_value());
  }
  EXPECT_FALSE(m.is_locked());
  EXPECT_EQ(m.num_shared_locks(), 0);

  // 1 readers: Upgrade succeeds atomically and consumes the shared lock.
  for (int_t i{}; i < num_iter; ++i) {
    std::shared_lock<mutex_t> l0{m};
    EXPECT_FALSE(m.is_locked());
    EXPECT_EQ(m.num_shared_locks(), 1);
    EXPECT_EQ(l0.mutex(), &m);
    EXPECT_TRUE(l0.owns_lock());

    std::optional<std::unique_lock<mutex_t>> l1{try_upgrade(l0)};
    EXPECT_TRUE(m.is_locked());
    EXPECT_EQ(m.num_shared_locks(), 0);
    EXPECT_EQ(l0.mutex(), nullptr);
    EXPECT_FALSE(l0.owns_lock());
    EXPECT_TRUE(l1.has_value());
    EXPECT_EQ(l1->mutex(), &m);
    EXPECT_TRUE(l1->owns_lock());
  }
  EXPECT_FALSE(m.is_locked());
  EXPECT_EQ(m.num_shared_locks(), 0);

  // 2 readers: The upgrade fails and shared lock remains untouched.
  for (int_t i{}; i < num_iter; ++i) {
    std::shared_lock l0{m}, l1{m};
    EXPECT_FALSE(m.is_locked());
    EXPECT_EQ(m.num_shared_locks(), 2);
    EXPECT_EQ(l0.mutex(), &m);
    EXPECT_TRUE(l0.owns_lock());
    EXPECT_EQ(l1.mutex(), &m);
    EXPECT_TRUE(l1.owns_lock());

    std::optional<std::unique_lock<mutex_t>> l2{try_upgrade(l0)};
    EXPECT_FALSE(m.is_locked());
    EXPECT_EQ(m.num_shared_locks(), 2);
    EXPECT_EQ(l0.mutex(), &m);
    EXPECT_TRUE(l0.owns_lock());
    EXPECT_EQ(l1.mutex(), &m);
    EXPECT_TRUE(l1.owns_lock());
    EXPECT_FALSE(l2.has_value());
  }
  EXPECT_FALSE(m.is_locked());
  EXPECT_EQ(m.num_shared_locks(), 0);
}

template <typename Mutex>
void test_mutex_std_lock_upgrade(const int_t num_iter = 8) {
  using mutex_t = Mutex;

  // 0 readers: Empty lock upgrades to empty lock.
  for (int_t i{}; i < num_iter; ++i) {
    std::shared_lock<mutex_t> l0;
    EXPECT_EQ(l0.mutex(), nullptr);
    EXPECT_FALSE(l0.owns_lock());

    std::unique_lock<mutex_t> l1{upgrade_or_relock(std::move(l0))};
    EXPECT_EQ(l0.mutex(), nullptr);
    EXPECT_FALSE(l0.owns_lock());
    EXPECT_EQ(l1.mutex(), nullptr);
    EXPECT_FALSE(l1.owns_lock());
  }

  mutex_t m;

  // 0 readers: Deferred locks upgrades to deferred lock.
  for (int_t i{}; i < num_iter; ++i) {
    std::shared_lock l0{m, std::defer_lock};
    EXPECT_FALSE(m.is_locked());
    EXPECT_EQ(m.num_shared_locks(), 0);
    EXPECT_EQ(l0.mutex(), &m);
    EXPECT_FALSE(l0.owns_lock());

    std::unique_lock<mutex_t> l1{upgrade_or_relock(std::move(l0))};
    EXPECT_FALSE(m.is_locked());
    EXPECT_EQ(m.num_shared_locks(), 0);
    EXPECT_EQ(l0.mutex(), nullptr);
    EXPECT_FALSE(l0.owns_lock());
    EXPECT_EQ(l1.mutex(), &m);
    EXPECT_FALSE(l1.owns_lock());
  }
  EXPECT_FALSE(m.is_locked());
  EXPECT_EQ(m.num_shared_locks(), 0);

  // 1 readers: Consumes the shared lock, and turn it into an upgraded lock.
  for (int_t i{}; i < num_iter; ++i) {
    std::shared_lock l0{m};
    EXPECT_FALSE(m.is_locked());
    EXPECT_EQ(m.num_shared_locks(), 1);
    EXPECT_EQ(l0.mutex(), &m);
    EXPECT_TRUE(l0.owns_lock());

    std::unique_lock<mutex_t> l1{upgrade_or_relock(std::move(l0))};
    EXPECT_TRUE(m.is_locked());
    EXPECT_EQ(m.num_shared_locks(), 0);
    EXPECT_EQ(l0.mutex(), nullptr);
    EXPECT_FALSE(l0.owns_lock());
    EXPECT_EQ(l1.mutex(), &m);
    EXPECT_TRUE(l1.owns_lock());
  }
  EXPECT_FALSE(m.is_locked());
  EXPECT_EQ(m.num_shared_locks(), 0);

  // 2 readers: Upgrade unlocks shared lock, and relocks as unique lock.
  for (int_t i{}; i < num_iter; ++i) {
    std::shared_lock l0{m}, l1{m};
    EXPECT_FALSE(m.is_locked());
    EXPECT_EQ(m.num_shared_locks(), 2);
    EXPECT_EQ(l0.mutex(), &m);
    EXPECT_TRUE(l0.owns_lock());
    EXPECT_EQ(l1.mutex(), &m);
    EXPECT_TRUE(l1.owns_lock());

    std::unique_lock<mutex_t> l2;
    std::thread t{[&]() {
      l2 = upgrade_or_relock(std::move(l0));
      EXPECT_EQ(l0.mutex(), nullptr);
      EXPECT_FALSE(l0.owns_lock());
      EXPECT_EQ(l2.mutex(), &m);
      EXPECT_TRUE(l2.owns_lock());
    }};

    std::this_thread::sleep_for(std::chrono::milliseconds(25));
    l1.unlock();
    EXPECT_EQ(l1.mutex(), &m);
    EXPECT_FALSE(l1.owns_lock());

    t.join();
    EXPECT_EQ(l0.mutex(), nullptr);
    EXPECT_FALSE(l0.owns_lock());
    EXPECT_EQ(l2.mutex(), &m);
    EXPECT_TRUE(l2.owns_lock());
  }
  EXPECT_FALSE(m.is_locked());
  EXPECT_EQ(m.num_shared_locks(), 0);
}

/**
 * A blocked writer must not acquire the lock before the holder releases it, and must
 * acquire it afterwards.
 */
template <typename Mutex>
void test_mutex_writer_blocks(const int_t num_iter = 8) {
  using mutex_t = Mutex;

  std::atomic<int_t> n;
  mutex_t m;

  for (int_t i{}; i < num_iter; ++i) {
    m.lock();
    n.store(0, std::memory_order_relaxed);

    std::thread t0{[&]() {
      std::unique_lock<mutex_t> l{m};
      EXPECT_TRUE(m.is_locked());
      EXPECT_EQ(m.num_shared_locks(), 0);
      EXPECT_EQ(l.mutex(), &m);
      EXPECT_TRUE(l.owns_lock());
      n.fetch_add(1, std::memory_order_relaxed);
    }};

    std::thread t1{[&]() {
      std::unique_lock<mutex_t> l{m};
      EXPECT_TRUE(m.is_locked());
      EXPECT_EQ(m.num_shared_locks(), 0);
      EXPECT_EQ(l.mutex(), &m);
      EXPECT_TRUE(l.owns_lock());
      n.fetch_add(1, std::memory_order_relaxed);
    }};

    std::this_thread::sleep_for(std::chrono::milliseconds(25));
    EXPECT_TRUE(m.is_locked());
    EXPECT_EQ(m.num_shared_locks(), 0);
    EXPECT_EQ(n.load(std::memory_order_relaxed), 0);

    m.unlock();
    t0.join();
    t1.join();
    EXPECT_FALSE(m.is_locked());
    EXPECT_EQ(m.num_shared_locks(), 0);
    EXPECT_EQ(n.load(std::memory_order_relaxed), 2);
  }
}

/**
 * Blocked readers must not acquire the lock before the writer releases it, and must
 * acquire it afterwards. Readers must not block each other.
 */
template <typename Mutex>
void test_mutex_reader_blocks(const int_t num_iter = 8) {
  using mutex_t = Mutex;

  std::atomic<int_t> n;
  mutex_t m;

  for (int_t i{}; i < num_iter; ++i) {
    m.lock();
    n.store(0, std::memory_order_relaxed);

    std::thread t0{[&]() {
      std::shared_lock<mutex_t> l{m};
      EXPECT_FALSE(m.is_locked());
      EXPECT_GE(m.num_shared_locks(), 1);
      EXPECT_EQ(l.mutex(), &m);
      EXPECT_TRUE(l.owns_lock());
      n.fetch_add(1, std::memory_order_relaxed);
    }};
    std::thread t1{[&]() {
      std::shared_lock<mutex_t> l{m};
      EXPECT_FALSE(m.is_locked());
      EXPECT_GE(m.num_shared_locks(), 1);
      EXPECT_EQ(l.mutex(), &m);
      EXPECT_TRUE(l.owns_lock());
      n.fetch_add(1, std::memory_order_relaxed);
    }};

    std::this_thread::sleep_for(std::chrono::milliseconds(25));
    EXPECT_TRUE(m.is_locked());
    EXPECT_EQ(m.num_shared_locks(), 0);
    EXPECT_EQ(n.load(std::memory_order_relaxed), 0);

    m.unlock();
    t0.join();
    t1.join();
    EXPECT_FALSE(m.is_locked());
    EXPECT_EQ(m.num_shared_locks(), 0);
    EXPECT_EQ(n.load(std::memory_order_relaxed), 2);
  }
}

/**
 * Increment a non-atomic counter from many threads. Any mutual exclusion violation
 * makes the final count fall short.
 */
template <typename Mutex>
void test_mutex_writer_stress(const int_t num_iter = 16384, const int_t num_threads = 4) {
  using mutex_t = Mutex;

  std::vector<std::thread> threads;
  threads.reserve(to_uint(num_threads));

  int_t n{};
  mutex_t m;

  m.lock();
  for (int_t j{}; j < num_threads; ++j) {
    threads.emplace_back([&]() {
      for (int_t i{}; i < num_iter; ++i) {
        std::lock_guard<mutex_t> l{m};
        ++n;
      }
    });
  }

  m.unlock();
  for (auto& t : threads) {
    t.join();
  }
  EXPECT_FALSE(m.is_locked());
  EXPECT_EQ(m.num_shared_locks(), 0);
  EXPECT_EQ(n, num_threads * num_iter);
}

/**
 * Writers keep two counters equal under a write lock; readers must never observe
 * them out of sync.
 */
 template <typename Mutex>
void test_mutex_reader_writer_stress(const int_t num_iter = 16384, const int_t num_writers = 2, const int_t num_readers = 4) {
  using mutex_t = Mutex;

  std::vector<std::thread> threads;
  threads.reserve(to_uint(num_writers + num_readers));

  int_t n0{}, n1{};
  std::atomic<int_t> errors{};
  mutex_t m;

  m.lock();
  for (int_t j{}; j < num_writers; ++j) {
    threads.emplace_back([&]() {
      for (int_t i{}; i < num_iter; ++i) {
        std::lock_guard<mutex_t> l{m};
        ++n0;
        ++n1;
      }
    });
  }
  for (int_t j{}; j < num_readers; ++j) {
    threads.emplace_back([&]() {
      for (int_t i{}; i < num_iter; ++i) {
        std::shared_lock<mutex_t> l{m};
        errors.fetch_add(n0 != n1, std::memory_order_relaxed);
      }
    });
  }

  m.unlock();
  for (auto& t : threads) {
    t.join();
  }
  EXPECT_FALSE(m.is_locked());
  EXPECT_EQ(m.num_shared_locks(), 0);
  EXPECT_EQ(n0, n1);
  EXPECT_EQ(n0, num_writers * num_iter);
  EXPECT_EQ(errors.load(std::memory_order_relaxed), 0);
}

/**
 * Concurrent shared-to-unique upgrades and back. Both the atomic path (`try_upgrade`)
 * and the manual fallback must preserve mutual exclusion, so the final count is exact.
 */
 template <typename Mutex>
void test_mutex_upgrade_downgrade_stress(const int_t num_iter = 16384, const int_t num_threads = 3) {
  using mutex_t = Mutex;

  std::vector<std::thread> threads;
  threads.reserve(to_uint(num_threads));

  int_t n{};
  std::atomic<int_t> errors{};
  mutex_t m;

  m.lock();
  for (int_t j{}; j < num_threads; ++j) {
    threads.emplace_back([&]() {
      for (int_t i{}; i < num_iter; ++i) {
        std::shared_lock<mutex_t> l0{m};
        std::optional<std::unique_lock<mutex_t>> l1{try_upgrade(l0)};
        if (l1) {
          ++n;
          l0 = downgrade(std::move(*l1));
          
          errors.fetch_add(!l0.owns_lock(), std::memory_order_relaxed);
          errors.fetch_add(l1->owns_lock(), std::memory_order_relaxed);
          errors.fetch_add(n <= 0, std::memory_order_relaxed);
        } else {
          errors.fetch_add(!l0.owns_lock(), std::memory_order_relaxed);
          l0.unlock();

          std::lock_guard<mutex_t> l1{m};
          ++n;
        }
      }
    });
  }

  m.unlock();
  for (auto& t : threads) {
    t.join();
  }

  EXPECT_FALSE(m.is_locked());
  EXPECT_EQ(m.num_shared_locks(), 0);
  EXPECT_EQ(n, num_threads * num_iter);
  EXPECT_EQ(errors.load(std::memory_order_relaxed), 0);
}

#define EVAL_MUTEX_TESTS_(_X_) \
  TEST(test_mutex_traits, _X_) { test_mutex_traits<_X_>(); }\
  TEST(test_mutex_lock_unlock, _X_) { test_mutex_lock_unlock<_X_>(); }\
  TEST(test_mutex_try_lock, _X_) { test_mutex_try_lock<_X_>(); }\
  TEST(test_mutex_downgrade, _X_) { test_mutex_downgrade<_X_>(); }\
  TEST(test_mutex_try_upgrade, _X_) { test_mutex_try_upgrade<_X_>(); }\
  TEST(test_mutex_std_locks, _X_) { test_mutex_std_locks<_X_>(); }\
  TEST(test_mutex_std_lock_downgrade, _X_) { test_mutex_std_lock_downgrade<_X_>(); }\
  TEST(test_mutex_std_lock_try_upgrade, _X_) { test_mutex_std_lock_try_upgrade<_X_>(); }\
  TEST(test_mutex_std_lock_upgrade, _X_) { test_mutex_std_lock_upgrade<_X_>(); }\
  TEST(test_mutex_writer_blocks, _X_) { test_mutex_writer_blocks<_X_>(); }\
  TEST(test_mutex_reader_blocks, _X_) { test_mutex_reader_blocks<_X_>(); }\
  TEST(test_mutex_writer_stress, _X_) { test_mutex_writer_stress<_X_>(); }\
  TEST(test_mutex_reader_writer_stress, _X_) { test_mutex_reader_writer_stress<_X_>(); }\
  TEST(test_mutex_upgrade_downgrade_stress, _X_) { test_mutex_upgrade_downgrade_stress<_X_>(); }

using spin_wait_mutex_1_t = spin_wait_mutex<int8_t>;
using spin_wait_mutex_2_t = spin_wait_mutex<int16_t>;
using spin_wait_mutex_4_t = spin_wait_mutex<int32_t>;
using spin_wait_mutex_8_t = spin_wait_mutex<int64_t>;

using spin_mutex_1_t = spin_wait_mutex<int8_t, 1, false>;
using spin_mutex_2_t = spin_wait_mutex<int16_t, 1, false>;
using spin_mutex_4_t = spin_wait_mutex<int32_t, 1, false>;
using spin_mutex_8_t = spin_wait_mutex<int64_t, 1, false>;

#if defined(__cpp_lib_atomic_wait)
using wait_mutex_1_t = spin_wait_mutex<int8_t, 0, true>;
using wait_mutex_2_t = spin_wait_mutex<int16_t, 0, true>;
using wait_mutex_4_t = spin_wait_mutex<int32_t, 0, true>;
using wait_mutex_8_t = spin_wait_mutex<int64_t, 0, true>;

// These spin, but then take the futex path almost immediately.
using spin_wait_mutex_spin_1_t = spin_wait_mutex<int8_t, 1, true>;
using spin_wait_mutex_spin_2_t = spin_wait_mutex<int16_t, 1, true>;
using spin_wait_mutex_spin_4_t = spin_wait_mutex<int32_t, 1, true>;
using spin_wait_mutex_spin_8_t = spin_wait_mutex<int64_t, 1, true>;
#endif

NVHM_FOR_EACH_(
  EVAL_MUTEX_TESTS_,
  spin_wait_mutex_1_t,
  spin_wait_mutex_2_t,
  spin_wait_mutex_4_t,
  spin_wait_mutex_8_t,
  spin_mutex_1_t,
  spin_mutex_2_t,
  spin_mutex_4_t,
  spin_mutex_8_t
  #if defined(__cpp_lib_atomic_wait)
  ,
  wait_mutex_1_t,
  wait_mutex_2_t,
  wait_mutex_4_t,
  wait_mutex_8_t,
  spin_wait_mutex_spin_1_t,
  spin_wait_mutex_spin_2_t,
  spin_wait_mutex_spin_4_t,
  spin_wait_mutex_spin_8_t
  #endif
);
