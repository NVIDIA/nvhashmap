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

#include <nvhashmap/std_map_shim.hpp>
// clang-format off
#include <nvhashmap/map.hpp>
#include "compile_common.hpp"
// clang-format on

using namespace nvhm;

int main() {
  using inner_t = map<dummy_key, dummy_value, flags_t::all & ~flags_t::duplicates>;
  using conf_t = typename inner_t::conf_type;

  using map_t = std_map_shim<inner_t>;
  using size_t = typename map_t::size_type;
  using const_iterator_t = typename map_t::const_iterator;
  using iterator_t = typename map_t::iterator;
  using const_mapped_t = typename map_t::const_mapped_type;
  using mapped_t = typename map_t::mapped_type;

  map_t map{conf_t{}.set_capacity(1024).set_blob(128)};
  const map_t& cmap{map};

  bool b;
  int_t i{};
  size_t u;
  float f;
  dummy_key k;
  dummy_value v;

  const_mapped_t cm{cmap.at(k)};
  mapped_t m{map.at(k)};
  map.at(k) = m;

  const_iterator_t cit{cmap.begin()};
  iterator_t it{map.begin()};

  u = cmap.capacity();
  u = map.capacity();

  cit = cmap.cbegin();
  cit = map.cbegin();
  cit = cmap.cend();
  cit = map.cend();

  map.clear();

  b = cmap.contains(k);
  b = map.contains(k);

  u = cmap.count(k);
  u = map.count(k);

  std::tie(it, b) = map.emplace(k, m);
  std::tie(it, b) = map.emplace(k, std::move(m));
  std::tie(it, b) = map.emplace(std::move(k), m);
  std::tie(it, b) = map.emplace(std::move(k), std::move(m));

  b = cmap.empty();
  b = map.empty();

  cit = cmap.end();
  it = map.end();

  std::tie(cit, cit) = cmap.equal_range(k);
  std::tie(it, it) = map.equal_range(k);

  it = map.erase(k);
  it = map.erase(std::move(k));
  map.erase(it);
  it = map.erase(std::move(it), it);
  it = map.erase(std::move(it), cit);
  it = map.erase(std::move(it), std::move(it));
  it = map.erase(std::move(it), std::move(cit));

  cit = cmap.find(k);
  it = map.find(k);

  std::tie(it, b) = map.insert(k, m);
  std::tie(it, b) = map.insert(k, std::move(m));
  std::tie(it, b) = map.insert(std::move(k), m);
  std::tie(it, b) = map.insert(std::move(k), std::move(m));

  std::tie(it, b) = map.insert_or_assign(k, m);
  std::tie(it, b) = map.insert_or_assign(k, std::move(m));
  std::tie(it, b) = map.insert_or_assign(std::move(k), m);
  std::tie(it, b) = map.insert_or_assign(std::move(k), std::move(m));

  f = cmap.load_factor();
  f = map.load_factor();

  f = cmap.max_load_factor();
  f = map.max_load_factor();

  map.max_load_factor(f);

  u = cmap.max_size();
  u = map.max_size();

  map.rehash(u);

  map.reserve(u);

  u = cmap.size();
  u = map.size();

  std::tie(it, b) = map.try_emplace(k);
  std::tie(it, b) = map.try_emplace(k, v);
  std::tie(it, b) = map.try_emplace(std::move(k));
  std::tie(it, b) = map.try_emplace(std::move(k), v);
  std::tie(it, b) = map.try_emplace(std::move(k), std::move(v));

  m = map[k];
  map[k] = m;

  b = map == map;
  b = map != map;
  b = map == cmap;
  b = map != cmap;
  b = cmap == map;
  b = cmap != map;
  b = cmap == cmap;
  b = cmap != cmap;

  swap(map, map);

  using const_iterator_value_t = typename const_iterator_t::value_type;
  using iterator_value_t = typename iterator_t::value_type;

  ++cit;
  ++it;
  --cit;
  --it;
  
  cit++;
  it++;
  cit--;
  it--;

  cit += i;
  it += i;
  cit -= i;
  it -= i;

  i = cit - cit;
  i = cit - it;
  i = it - cit;
  i = it - it;

  const_iterator_value_t ce{cit[i]};
  iterator_value_t e{it[i]};
  std::cout << reinterpret_cast<const void*>(&ce) << reinterpret_cast<const void*>(&e);

  b = cit == cit;
  b = cit == it;
  b = it == cit;
  b = it == it;
  b = cit != cit;
  b = cit != it;
  b = it != cit;
  b = it != it;
  b = cit < cit;
  b = cit < it;
  b = it < cit;
  b = it < it;
  b = cit > cit;
  b = cit > it;
  b = it > cit;
  b = it > it;
  b = cit <= cit;
  b = cit <= it;
  b = it <= cit;
  b = it <= it;
  b = cit >= cit;
  b = cit >= it;
  b = it >= cit;
  b = it >= it;

  std::cout << u << reinterpret_cast<const void*>(&cm);
  return 0;
}