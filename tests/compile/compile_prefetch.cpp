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

#include <nvhashmap/prefetch.hpp>
// clang-format off
#include "compile_common.hpp"
#include <nvhashmap/map.hpp>
// clang-format on

using namespace nvhm;
 
int main() {
  using map_t = map<dummy_key, dummy_value, flags_t::all>;
  using conf_t = typename map_t::conf_type;
  using prefetch_hint_t = typename map_t::prefetch_hint;

  int_t i{};
  bool b{};
  dummy_key k;
  prefetch_hint_t h;

  map_t map{conf_t{}.set_capacity(1024).set_blob(128)};

  using shift_queue_t = shift_prefetch_queue<dummy_key, prefetch_hint_t, 8>;
  {
    using const_entry_t = typename shift_queue_t::const_entry_type;
    using entry_t = typename shift_queue_t::entry_type;

    shift_queue_t q;
    
    entry_t e;

    const_entry_t ce0{q.front()};
    const_entry_t ce1{q.back()};

    e = q.push(k, h);
    e = q.push(k, std::move(h));
    e = q.push(std::move(k), h);
    e = q.push(std::move(k), std::move(h));

    e = q.push_read(map, k);
    e = q.push_read(map, std::move(k));

    e = q.push_write(map, k);
    e = q.push_write(map, std::move(k));

    e = q.pop();

    q = q.prefill(i, k, h);
    q = q.prefill(i, k, std::move(h));
    q = q.prefill(i, std::move(k), h);
    q = q.prefill(i, std::move(k), std::move(h));
    
    q = q.skip();

    q = q.prefill_read(i, map, k);
    q = q.prefill_read(i, map, std::move(k));

    q = q.prefill_write(i, map, k);
    q = q.prefill_write(i, map, std::move(k));

    b = ce0 == ce1;
  }

  using ring_queue_t = ring_prefetch_queue<dummy_key, prefetch_hint_t, 8>;
  {
    using const_entry_t = typename shift_queue_t::const_entry_type;
    using entry_t = typename shift_queue_t::entry_type;

    entry_t e;

    ring_queue_t q;

    const_entry_t ce0{q.front()};
    const_entry_t ce1{q.back()};
    
    i = q.size();
    b = q.empty();
    b = q.full();

    e = q.push(k, h);
    e = q.push(k, std::move(h));
    e = q.push(std::move(k), h);
    e = q.push(std::move(k), std::move(h));

    e = q.push_read(map, k);
    e = q.push_read(map, std::move(k));

    e = q.push_write(map, k);
    e = q.push_write(map, std::move(k));
   
    e = q.pop();
    
    q = q.skip();
  
    q = q.prefill(k, h);
    q = q.prefill(k, std::move(h));
    q = q.prefill(std::move(k), h);
    q = q.prefill(std::move(k), std::move(h));
    
    q = q.prefill_read(map, k);
    q = q.prefill_read(map, std::move(k));
    
    q = q.prefill_write(map, k);
    q = q.prefill_write(map, std::move(k));

    b = ce0 == ce1;
  }
  
  return b;
}