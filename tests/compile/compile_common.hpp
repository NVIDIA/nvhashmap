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

#pragma once

#include <nvhashmap/container.hpp>

class dummy_key {
 public:
  constexpr dummy_key() noexcept = default;

  constexpr friend bool operator==(const dummy_key& lhs, const dummy_key& rhs) noexcept { return &lhs == &rhs; }
  constexpr friend bool operator!=(const dummy_key& lhs, const dummy_key& rhs) noexcept { return !(lhs == rhs); }

  constexpr friend std::ostream& operator<<(std::ostream& os, const dummy_key&) noexcept { return os; }
};

class dummy_value {
 public:
  constexpr dummy_value() noexcept = default;

  constexpr friend bool operator==(const dummy_value& lhs, const dummy_value& rhs) noexcept { return &lhs == &rhs; }
  constexpr friend bool operator!=(const dummy_value& lhs, const dummy_value& rhs) noexcept { return !(lhs == rhs); }

  constexpr friend std::ostream& operator<<(std::ostream& os, const dummy_value&) noexcept { return os; }
};

namespace std {

template <>
struct hash<dummy_key> {
  std::size_t operator()(const dummy_key& v) const noexcept {
    return reinterpret_cast<std::size_t>(&v);
  }
};

}

template <bool ThreadSafe, typename Map, typename... Args>
int compile_map(Args&&... args) {
  using namespace nvhm;

  using map_t = Map;
  using conf_t = typename map_t::conf_type;
  using key_t = typename map_t::key_type;
  using value_t = typename map_t::value_type;
  using write_pos_t = typename map_t::write_pos;
  using read_pos_t = typename map_t::read_pos;
  using prefetch_hint_t = typename map_t::prefetch_hint;
  using const_iterator_t = typename map_t::const_iterator;
  using iterator_t = typename map_t::iterator;
  using const_entry_t = typename map_t::const_entry_type;
  using entry_t = typename map_t::entry_type;
  using const_mapped_t = typename map_t::const_mapped_type;
  using mapped_t = typename map_t::mapped_type;
  using probe_seq_t = typename map_t::probe_seq_type;
  
  conf_t c;
  bool b;
  double d;
  int_t i;
  const blob_t* cbl;
  blob_t* bl{nullptr};
  key_t k;
  state_t s;
  lru_t l;
  insert_op_t op;
  value_t v;
  value_t& rv{v};
  std::string str;
  std::vector<key_t> keys;
  std::vector<state_t> states;
  std::vector<lru_t> lrus;
  std::vector<value_t> vals;
  std::vector<blob_t> blobs;
  std::vector<read_pos_t> r_poses;
  std::vector<write_pos_t> w_poses;
  std::array<int_t, map_t::kernel_size> counts;

  map_t map{args..., conf_t{}.set_capacity(1024).set_blob(128)};
  const map_t& cmap{map};

  write_pos_t w_pos{map.insert(k)};
  prefetch_hint_t hint{map.read_prefetch(k)};
  read_pos_t r_pos{map.find(k)};
  const_iterator_t cit{cmap.begin()};
  iterator_t it{map.begin()};
  probe_seq_t seq{std::get<2>(map.erase_first(k))};

  blobs = cmap.all_blobs_for(k);
  blobs = map.all_blobs_for(k);

  vals = cmap.all_values_for(k);
  vals = map.all_values_for(k);

  const_entry_t ce0{cmap.at(r_pos)};
  const_entry_t ce1{map.at(r_pos)};
  entry_t e0{map.at(w_pos)};
  std::get<1>(map.at(w_pos)) = v;
  std::get<1>(map.at(w_pos)) = std::move(v);
  *std::get<2>(map.at(w_pos)) = *bl;

  cit = cmap.begin();
  it = map.begin();

  cbl = cmap.blob_at(r_pos);
  cbl = map.blob_at(r_pos);
  bl = map.blob_at(w_pos);
  *map.blob_at(w_pos) = *bl;
  blobs = cmap.blobs();
  blobs = map.blobs();

  i = cmap.bucket_size_at(r_pos);
  i = map.bucket_size_at(r_pos);
  i = map.bucket_size_at(w_pos);
  i = cmap.bucket_num_empty_slots_at(r_pos);
  i = map.bucket_num_empty_slots_at(r_pos);
  i = map.bucket_num_empty_slots_at(w_pos);
  i = cmap.bucket_num_tombstone_slots_at(r_pos);
  i = map.bucket_num_tombstone_slots_at(r_pos);
  i = map.bucket_num_tombstone_slots_at(w_pos);

  i = cmap.capacity();
  i = map.capacity();

  cit = cmap.cbegin();
  cit = map.cbegin();

  b = cmap.check_integrity();
  b = map.check_integrity();

  i = map.clear();
  i = map.clear(i);

  c = cmap.conf();
  c = map.conf();

  b = cmap.contains(k);
  b = map.contains(k);
  b = cmap.contains(k, hint);
  b = map.contains(k, hint);
  b = cmap.contains_at(r_pos);
  b = map.contains_at(r_pos);
  b = map.contains_at(w_pos);

  i = cmap.count(k);
  i = map.count(k);
  i = cmap.count_if([](const read_pos_t& p) { return p != npos; });
  i = map.count_if([](const read_pos_t& p) { return p != npos; });

  cmap.count_state_collisions(counts);
  map.count_state_collisions(counts);

  cit = cmap.end();
  it = map.end();

  b                       = map.erase(k);
  b                       = map.erase(k, hint);
  std::tie(b, w_pos, seq) = map.erase_first(k);
  std::tie(b, w_pos, seq) = map.erase_first(k, hint);
  std::tie(b, w_pos, seq) = map.erase_next(std::move(w_pos), seq, k);
  std::tie(b, w_pos, seq) = map.erase_next(std::move(w_pos), seq, k, hint);
  std::tie(b, w_pos) = map.erase_at(std::move(w_pos));
  i = map.erase_if([](const write_pos_t& p) { return p != npos; });
  i = map.erase_range(keys.begin(), keys.end());
  i = map.erase_all(k);

  r_pos = cmap.find(k);
  r_pos = map.find(k);
  r_pos = cmap.find(k, hint);
  r_pos = map.find(k, hint);
  std::tie(r_pos, seq) = cmap.find_first(k);
  std::tie(r_pos, seq) = map.find_first(k);
  std::tie(r_pos, seq) = cmap.find_first(k, hint);
  std::tie(r_pos, seq) = map.find_first(k, hint);
  std::tie(r_pos, seq) = cmap.find_next(std::move(r_pos), seq, k);
  std::tie(r_pos, seq) = map.find_next(std::move(r_pos), seq, k);
  std::tie(r_pos, seq) = map.find_next(std::move(w_pos), seq, k);
  std::tie(r_pos, seq) = cmap.find_next(std::move(r_pos), seq, k, hint);
  std::tie(r_pos, seq) = map.find_next(std::move(r_pos), seq, k, hint);
  std::tie(r_pos, seq) = map.find_next(std::move(w_pos), seq, k, hint);
  r_pos = cmap.find_if([](const read_pos_t& p) { return p != npos; });
  r_pos = map.find_if([](const read_pos_t& p) { return p != npos; });
  i = cmap.find_range(keys.begin(), keys.end());
  i = map.find_range(keys.begin(), keys.end());
  r_poses = cmap.find_all(k);
  r_poses = map.find_all(k);

  cmap.for_each([](const read_pos_t&) {});
  map.for_each([](const read_pos_t&) {});
  map.for_each([](const write_pos_t&) {});
  cmap.for_each(k, [&](const read_pos_t&, const probe_seq_t&) {});
  map.for_each(k, [&](const read_pos_t&, const probe_seq_t&) {});
  map.for_each(k, [&](const write_pos_t&, const probe_seq_t&) {});

  cmap.for_each_blob([](const blob_t*) {});
  map.for_each_blob([](blob_t*) {});
  cmap.for_each_entry([](const key_t&, const value_t&, const blob_t*) {});
  map.for_each_entry([](const key_t&, value_t&, blob_t*) {});
  cmap.for_each_key([](const key_t&) {});
  map.for_each_key([](const key_t&) {});
  cmap.for_each_lru([](const lru_t&) {});
  map.for_each_lru([](const lru_t&) {});
  cmap.for_each_state([](const state_t&) {});
  map.for_each_state([](const state_t&) {});
  cmap.for_each_value([](const value_t&) {});
  map.for_each_value([](value_t&) {});

  cmap.get_blob_at(r_pos, nullptr);
  map.get_blob_at(r_pos, nullptr);
  map.get_blob_at(w_pos, nullptr);
  cmap.get_blob_at(r_pos, nullptr, 1);
  map.get_blob_at(r_pos, nullptr, 1);
  map.get_blob_at(w_pos, nullptr, 1);

  i = map.grow();

  w_pos = map.insert(k);
  w_pos = map.insert(std::move(k));
  std::tie(w_pos, seq, op) = map.insert_ex(k);
  std::tie(w_pos, seq, op) = map.insert_ex(std::move(k));
  i = map.insert_range(keys.begin(), keys.end());

  b = cmap.is_empty();
  b = map.is_empty();
  b = cmap.is_full();
  b = map.is_full();

  k = cmap.key_at(r_pos);
  k = map.key_at(r_pos);
  k = map.key_at(w_pos);
  keys = cmap.keys();
  keys = map.keys();

  d = cmap.load_factor();
  d = map.load_factor();

  l = cmap.lru_at(r_pos);
  l = map.lru_at(r_pos);
  l = map.lru_at(w_pos);
  lrus = cmap.lrus();
  lrus = map.lrus();

  const_mapped_t cm0{cmap.mapped_at(r_pos)};
  const_mapped_t cm1{map.mapped_at(r_pos)};
  mapped_t m{map.mapped_at(w_pos)};
  map.mapped_at(w_pos) = m;
  map.mapped_at(w_pos) = std::move(m);

  d = cmap.max_load_factor();
  d = map.max_load_factor();

  i = cmap.num_empty_slots();
  i = map.num_empty_slots();
  i = cmap.num_not_hash_slots();
  i = map.num_not_hash_slots();
  i = cmap.num_tombstone_slots();
  i = map.num_tombstone_slots();

  r_pos = map.read_npos();

  hint = map.read_prefetch(k);
  map.read_prefetch(k, hint);
  cmap.read_prefetch_value_at(r_pos);
  map.read_prefetch_value_at(r_pos);
  cmap.read_prefetch_value_at(w_pos);
  map.read_prefetch_value_at(w_pos);
  cmap.read_prefetch_blob_at(r_pos);
  map.read_prefetch_blob_at(r_pos);
  cmap.read_prefetch_blob_at(w_pos);
  map.read_prefetch_blob_at(w_pos);
  
  cmap.render(std::cout);
  map.render(std::cout);
  cmap.render(std::cout, true);
  map.render(std::cout, true);
  cmap.render(std::cout, true, blob_render_t::full);
  map.render(std::cout, true, blob_render_t::full);

  i = map.reserve(1024);

  i = map.resize(1024);

  i = map.scrub();
  i = map.scrub(max_lru);

  map.set_blob_at(w_pos, nullptr);
  map.set_blob_at(w_pos, nullptr, 1);

  map.set_value_at(w_pos, v);

  i = cmap.size();
  i = map.size();

  i = map.shrink();

  s = cmap.state_at(r_pos);
  s = map.state_at(r_pos);
  s = map.state_at(w_pos);
  states = cmap.states();
  states = map.states();
  counts = cmap.state_collisions();
  counts = map.state_collisions();

  cit = cmap.to_iterator(std::move(r_pos));
  it = map.to_iterator(std::move(w_pos));
  cit = cmap.to_citerator(std::move(r_pos));
  cit = cmap.to_citerator(std::move(w_pos));

  str = cmap.to_string();
  str = map.to_string();

  w_pos = map.update(k);
  w_pos = map.update(k, hint);
  std::tie(w_pos, seq) = map.update_first(k);
  std::tie(w_pos, seq) = map.update_first(k, hint);
  std::tie(w_pos, seq) = map.update_next(std::move(w_pos), seq, k);
  std::tie(w_pos, seq) = map.update_next(std::move(w_pos), seq, k, hint);
  map.update_if([](const write_pos_t& p) { return p != npos; });
  w_poses = map.update_all(k);

  v = cmap.value_at(r_pos);
  v = map.value_at(r_pos);
  rv = map.value_at(w_pos);
  map.value_at(w_pos) = v;
  map.value_at(w_pos) = std::move(v);

  w_pos = map.write_npos();

  hint = map.write_prefetch(k);
  map.write_prefetch(k, hint);
  map.write_prefetch_value_at(w_pos);
  map.write_prefetch_blob_at(w_pos);
  
  m = map[k];
  map[k] = m;

  std::cout << map;

  if constexpr (!test_flags(map_t::flags, flags_t::duplicates)) {
    b = map == map;
    b = map != map;
    b = map == cmap;
    b = map != cmap;
    b = cmap == map;
    b = cmap != map;
    b = cmap == cmap;
    b = cmap != cmap;
  }
  
  swap(map, map);


  using const_entry_ptr_t = typename const_iterator_t::entry_ptr_type;
  using entry_ptr_t = typename iterator_t::entry_ptr_type;

  cbl = cit.blob();
  bl = it.blob();
  const_entry_t ce2{cit.entry()};
  entry_t e1{it.entry()};
  k = cit.key();
  k = it.key();
  l = cit.lru();
  l = it.lru();
  b = cit.query();
  b = it.query();
  v = cit.value();
  v = it.value();
  const_mapped_t cm2{cit.mapped()};
  m = it.mapped();
  const_entry_t ce3{*cit};
  entry_t e2{*it};
  const_entry_ptr_t cp{*cit.operator->()};
  entry_ptr_t p{*it.operator->()};

  b = cit.is_left();
  b = it.is_left();
  b = cit.is_right();
  b = it.is_right();

  ++cit;
  ++it;
  --cit;
  --it;
  
  if constexpr (!ThreadSafe) {
    cit++;
    it++;
    cit--;
    it--;
  }

  cit += i;
  it += i;
  cit -= i;
  it -= i;

  i = cit - cit;
  i = cit - it;
  i = it - cit;
  i = it - it;

  if constexpr (!ThreadSafe) {
    const_entry_t ce4{cit[i]};
    entry_t e3{it[i]};
    b = &ce4 == &ce4 && &e3 == &e3;
  }

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

  std::cout << b << d << i << cbl << bl << k << l << s;
  return &e0 == &e1 && &e2 == &e2 && &ce0 == &ce1 && &ce2 == &ce3 && &m == &m && &cm0 == &cm1 && &cm0 == &cm2 && &cp == &cp && &p == &p;
}
