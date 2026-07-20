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

int main() {
  using mutex_t = spin_wait_mutex<>;

  mutex_t m;
  bool b;
  int_t n;

  b = m.is_locked();
  m.lock();
  b = m.try_lock();
  m.unlock();
  m.downgrade();

  n = m.num_shared_locks();
  n = m.lock_shared();
  b = m.try_lock_shared();
  n = m.unlock_shared();
  b = m.try_upgrade();

  std::unique_lock<mutex_t> ul{m};
  std::shared_lock<mutex_t> sl{m};

  ul.lock();
  b = ul.try_lock();
  ul.unlock();
  ul.release();
  sl = downgrade(std::move(ul));
  
  sl.lock();
  b = sl.try_lock();
  sl.unlock();
  sl.release();
  ul = upgrade(std::move(sl));
  auto opt_ul{try_upgrade(sl)};

  return b || n != 0 || opt_ul.has_value();
}
