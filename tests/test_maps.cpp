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

#include "test_common.hpp"
#include <gtest/gtest.h>
#include <nvhashmap/guarded.hpp>
#include <nvhashmap/map.hpp>
#include <numeric>
#include <nvhashmap/cache.hpp>
#include <nvhashmap/guarded.hpp>
#include <nvhashmap/sharded.hpp>
#include <set>

using namespace nvhm;

using string_t = std::string;

template <typename Key, typename Kernel>
void test_conf() {
  using set_t = set<Key, flags_t::none, Kernel>;
  using conf_t = typename set_t::conf_type;

  conf_t conf;
  EXPECT_EQ(conf.min_capacity(), conf_t::min_capacity_limit(conf.grow_bias()));
  EXPECT_EQ(conf.init_capacity(), conf_t::default_capacity(conf.grow_bias()));

  EXPECT_EQ(conf.blob_size(), 0);
  EXPECT_EQ(conf.blob_stride(), 0);

  EXPECT_EQ(conf.scrub_bias(), conf_t::default_scrub_bias);
  EXPECT_EQ(conf.grow_bias(), conf_t::default_grow_bias);
  EXPECT_EQ(conf.shrink_bias(), conf_t::default_shrink_bias);
}

template <typename Key, typename Kernel>
void test_init() {
  using set_t = set<Key, flags_t::none, Kernel>;
  using conf_t = typename set_t::conf_type;

  conf_t conf;
  set_t s;

  EXPECT_EQ(conf.init_capacity(), conf_t::default_capacity(conf.grow_bias()));
  EXPECT_EQ(s.capacity(), conf.init_capacity());
  EXPECT_EQ(s.size(), 0);
  EXPECT_TRUE(s.is_empty());
  EXPECT_FALSE(s.is_full());
  EXPECT_EQ(s.num_empty_slots(), conf.init_capacity());
  EXPECT_EQ(s.num_tombstone_slots(), 0);

  s = conf.set_capacity(0);

  EXPECT_EQ(conf.init_capacity(), conf_t::min_capacity_limit(conf.grow_bias()));
  EXPECT_EQ(s.capacity(), conf.init_capacity());
  EXPECT_EQ(s.size(), 0);
  EXPECT_TRUE(s.is_empty());
  EXPECT_FALSE(s.is_full());
  EXPECT_EQ(s.num_empty_slots(), conf.init_capacity());
  EXPECT_EQ(s.num_tombstone_slots(), 0);

  s = set_t(conf.set_capacity(999));
  EXPECT_GE(conf.init_capacity(), 1024);
  EXPECT_EQ(s.capacity(), conf.init_capacity());
  EXPECT_EQ(s.size(), 0);
  EXPECT_TRUE(s.is_empty());
  EXPECT_FALSE(s.is_full());
  EXPECT_EQ(s.num_empty_slots(), conf.init_capacity());
  EXPECT_EQ(s.num_tombstone_slots(), 0);
}

template <typename K, int_t N>
inline void make_key_sequence(int32_t start_key, K (&keys)[N]) noexcept {
  int32_t int_keys[N];
  std::iota(std::begin(int_keys), std::end(int_keys), start_key);

  std::transform(std::begin(int_keys), std::end(int_keys), std::begin(keys), [](int32_t key) {
    if constexpr (std::is_same_v<K, string_t>) {
      return std::to_string(key);
    } else {
      return key;
    }
  });
}

template <typename Key>
float key_to_float(const Key& key) {
  if constexpr (std::is_same_v<Key, string_t>) {
    return static_cast<float>(std::stoi(key));
  } else {
    return static_cast<float>(key);
  }
}

template <typename Key, typename Kernel>
void test_insert() {
  using map_t = map<Key, float, flags_t::blobs, Kernel>;
  using conf_t = typename map_t::conf_type;
  using key_t = typename map_t::key_type;
  using read_pos_t = typename map_t::read_pos;

  constexpr int_t num_keys{1000};
  key_t keys[num_keys];
  make_key_sequence<key_t>(1337, keys);

  const auto conf{conf_t{}.set_capacity(0).set_blob(sizeof(float)).auto_adjust()};
  map_t map(conf);

  constexpr int_t n{map_t::kernel_size * 10 / 16};

  for (int_t i{}; i < n; ++i) {
    auto [pos, seq, op]{map.insert_ex(keys[i])};
    EXPECT_GE(pos, 0);
    EXPECT_LT(pos, map.capacity());
    EXPECT_EQ(seq.psl(), 0);
    EXPECT_EQ(op, insert_op_t::insert);

    float value{key_to_float(keys[i])};
    map.value_at(pos) = value;
    map.set_blob_at(pos, &value);
  }

  // Should lead to `found` situation.
  for (int_t i{}; i < n; ++i) {
    auto [pos, seq, op]{map.insert_ex(keys[i])};
    EXPECT_GE(pos, 0);
    EXPECT_LT(pos, map.capacity());
    EXPECT_EQ(seq.psl(), 0);
    EXPECT_EQ(op, insert_op_t::found);

    EXPECT_EQ(map.value_at(pos), key_to_float(keys[i]));
    EXPECT_EQ(*reinterpret_cast<const float*>(map.blob_at(pos)), key_to_float(keys[i]));
  }

  EXPECT_EQ(map.capacity(), conf.init_capacity());
  EXPECT_EQ(map.size(), n);
  EXPECT_EQ(map.is_empty(), map_t::kernel_size == 1);
  EXPECT_EQ(map.is_full(), false);
  EXPECT_EQ(map.num_empty_slots(), conf.init_capacity() - n);
  EXPECT_EQ(map.num_tombstone_slots(), 0);

  EXPECT_EQ(map.grow_threshold(), std::max(map_t::kernel_size / 8, {1}));

  // See if the growth happens as expected.
  for (int_t i{n}; i < num_keys; ++i) {
    int_t grow_threshold{map.grow_threshold()};
    int_t before_capacity{map.capacity()};
    bool should_grow{map.num_empty_slots() <= grow_threshold};

    EXPECT_EQ(map.find(keys[i]), npos);
    {
      auto [pos, seq, op]{map.insert_ex(keys[i])};
      EXPECT_GE(pos, 0);
      EXPECT_LT(pos, map.capacity());
      EXPECT_EQ(op, insert_op_t::insert);

      float value{key_to_float(keys[i])};
      map.value_at(pos) = value;
      map.set_blob_at(pos, &value);
    }
    {
    read_pos_t pos{map.find(keys[i])};
      EXPECT_GE(pos, 0);
      EXPECT_LT(pos, map.capacity());
      EXPECT_EQ(map.key_at(pos), keys[i]);
      EXPECT_EQ(map.value_at(pos), key_to_float(keys[i]));
      EXPECT_EQ(*reinterpret_cast<const float*>(map.blob_at(pos)), key_to_float(keys[i]));
    }

    if (should_grow) {
      EXPECT_EQ(map.capacity(), before_capacity << 1);
      EXPECT_GT(map.num_empty_slots(), map.grow_threshold());
    }
  }

  // Ensure all keys still exist.
  for (int_t i{}; i < num_keys; ++i) {
    EXPECT_NE(map.find(keys[i]), npos) << "keys[" << i << "] = " << keys[i] << " found at " << map.find(keys[i]);
  }
}

template <typename Key, typename Kernel>
void test_iterators() {
  using set_t = set<Key, flags_t::none, Kernel>;
  using key_t = typename set_t::key_type;
  using read_pos_t = typename set_t::read_pos;

  constexpr int_t num_keys{100};
  key_t keys[num_keys];
  make_key_sequence<key_t>(1337, keys);

  set_t set;
  EXPECT_EQ(set.insert_range(std::begin(keys), std::end(keys)), num_keys);

  std::set<key_t> keys_found;
  set.for_each([&](const read_pos_t& pos) {
    EXPECT_GE(pos, 0);
    EXPECT_LT(pos, set.capacity());
    const key_t& key{set.key_at(pos)};
    EXPECT_NE(std::find(std::begin(keys), std::end(keys), key), std::end(keys)) << "key = " << key << " not found in keys";
    keys_found.insert(key);
  });
  EXPECT_EQ(to_int(keys_found.size()), num_keys);

  keys_found.clear();
  for (auto it{set.begin()}; it != set.end(); ++it) {
    const key_t& key{it.key()};
    EXPECT_NE(std::find(std::begin(keys), std::end(keys), key), std::end(keys)) << "key = " << key << " not found in keys";
    keys_found.insert(key);
  }
  EXPECT_EQ(to_int(keys_found.size()), num_keys);

  keys_found.clear();
  for (const key_t& key : set) {
    EXPECT_NE(std::find(std::begin(keys), std::end(keys), key), std::end(keys)) << "key = " << key << " not found in keys";
    keys_found.insert(key);
  }
  EXPECT_EQ(to_int(keys_found.size()), num_keys);
}

template <typename Key, typename Kernel>
void test_map_iterators() {
  using map_t = map<Key, float, flags_t::blobs, Kernel>;
  using conf_t = typename map_t::conf_type;
  using key_t = typename map_t::key_type;
  using read_pos_t = typename map_t::read_pos;
  using write_pos_t = typename map_t::write_pos;

  constexpr int_t num_keys{100};
  key_t keys[num_keys];
  make_key_sequence<key_t>(1337, keys);

  map_t map(conf_t{}.set_blob(sizeof(float)).auto_adjust());
  for (const key_t& key : keys) {
    write_pos_t pos{map.insert(key)};
    EXPECT_NE(pos, npos);
    float value{key_to_float(key)};
    map.value_at(pos) = value;
    map.set_blob_at(pos, &value);
  }

  std::set<key_t> keys_found;
  map.for_each([&](const read_pos_t& pos) {
    EXPECT_GE(pos, 0);
    EXPECT_LT(pos, map.capacity());
    const key_t& key{map.key_at(pos)};
    EXPECT_NE(std::find(std::begin(keys), std::end(keys), key), std::end(keys)) << "key = " << key << " not found in keys";
    EXPECT_EQ(map.value_at(pos), key_to_float(key));
    EXPECT_EQ(*reinterpret_cast<const float*>(map.blob_at(pos)), key_to_float(key));
    keys_found.insert(key);
  });
  EXPECT_EQ(to_int(keys_found.size()), num_keys);

  keys_found.clear();
  for (auto it{map.begin()}; it != map.end(); ++it) {
    const key_t& key{it.key()};
    EXPECT_NE(std::find(std::begin(keys), std::end(keys), key), std::end(keys)) << "key = " << key << " not found in keys";
    EXPECT_EQ(it.value(), key_to_float(key));
    EXPECT_EQ(*reinterpret_cast<const float*>(it.blob()), key_to_float(key));
    keys_found.insert(key);
  }
  EXPECT_EQ(to_int(keys_found.size()), num_keys);

  keys_found.clear();
  for (const auto& [key, value, blob] : map) {
    EXPECT_NE(std::find(std::begin(keys), std::end(keys), key), std::end(keys)) << "key = " << key << " not found in keys";
    EXPECT_EQ(value, key_to_float(key));
    EXPECT_EQ(*reinterpret_cast<const float*>(blob), key_to_float(key));
    keys_found.insert(key);
  }
  EXPECT_EQ(to_int(keys_found.size()), num_keys);
}

template <typename Key, typename Kernel>
void test_copy_move_compare() {
  using map_t = map<Key, float, flags_t::blobs, Kernel>;
  using conf_t = typename map_t::conf_type;
  using key_t = typename map_t::key_type;
  using write_pos_t = typename map_t::write_pos;

  constexpr int_t num_keys{100};
  key_t keys[num_keys];
  make_key_sequence<key_t>(1337, keys);

  const auto conf{conf_t{}.set_capacity(0).set_blob(sizeof(float))};
  map_t map0(conf), map1(conf);

  EXPECT_EQ(map0.conf(), map1.conf());
  EXPECT_EQ(map0.size(), map1.size());
  EXPECT_EQ(map0.capacity(), map1.capacity());
  EXPECT_EQ(map0.num_empty_slots(), map1.num_empty_slots());
  EXPECT_EQ(map0.num_tombstone_slots(), map1.num_tombstone_slots());
  EXPECT_EQ(map0.is_empty(), map1.is_empty());
  EXPECT_EQ(map0.is_full(), map1.is_full());
  EXPECT_EQ(map0.load_factor(), map1.load_factor());
  EXPECT_EQ(map0.grow_threshold(), map1.grow_threshold());
  EXPECT_EQ(map0, map1);

  for (const key_t& key : keys) {
    write_pos_t pos{map0.insert(key)};
    EXPECT_NE(pos, npos);
    map0.value_at(pos) = key_to_float(key);
    map0.set_blob_at(pos, &map0.value_at(pos));
    
    pos = map1.insert(key);
    EXPECT_NE(pos, npos);
    map1.value_at(pos) = key_to_float(key);
    map1.set_blob_at(pos, &map0.value_at(pos));
  }

  EXPECT_EQ(map0.size(), map1.size());
  EXPECT_EQ(map0.capacity(), map1.capacity());
  EXPECT_EQ(map0.num_empty_slots(), map1.num_empty_slots());
  EXPECT_EQ(map0.num_tombstone_slots(), map1.num_tombstone_slots());
  EXPECT_EQ(map0.is_empty(), map1.is_empty());
  EXPECT_EQ(map0.is_full(), map1.is_full());
  EXPECT_EQ(map0.load_factor(), map1.load_factor());
  EXPECT_EQ(map0.grow_threshold(), map1.grow_threshold());
  EXPECT_EQ(map0, map1);

  map0.erase(keys[40]);

  EXPECT_NE(map0.size(), map1.size());
  EXPECT_EQ(map0.capacity(), map1.capacity());
  EXPECT_NE(map0, map1);

  map0 = map1;
  EXPECT_EQ(map0.size(), map1.size());
  EXPECT_EQ(map0.capacity(), map1.capacity());
  EXPECT_EQ(map0.num_empty_slots(), map1.num_empty_slots());
  EXPECT_EQ(map0.num_tombstone_slots(), map1.num_tombstone_slots());
  EXPECT_EQ(map0.is_empty(), map1.is_empty());
  EXPECT_EQ(map0.is_full(), map1.is_full());
  EXPECT_EQ(map0.load_factor(), map1.load_factor());
  EXPECT_EQ(map0.grow_threshold(), map1.grow_threshold());
  EXPECT_EQ(map0, map1);

  map_t map2{map0};
  EXPECT_EQ(map2.size(), map1.size());
  EXPECT_EQ(map2.capacity(), map1.capacity());
  EXPECT_EQ(map2.num_empty_slots(), map1.num_empty_slots());
  EXPECT_EQ(map2.num_tombstone_slots(), map1.num_tombstone_slots());
  EXPECT_EQ(map2.is_empty(), map1.is_empty());
  EXPECT_EQ(map2.is_full(), map1.is_full());
  EXPECT_EQ(map2.load_factor(), map1.load_factor());
  EXPECT_EQ(map2.grow_threshold(), map1.grow_threshold());
  EXPECT_EQ(map2, map1);

  EXPECT_EQ(map2, map0);
  map2.value_at(map2.update(keys[42])) = 47.0f;
  EXPECT_NE(map2, map0);

  std::swap(map2, map1);
  EXPECT_EQ(map2, map0);
  EXPECT_NE(map1, map0);
  
  map2.blob_at(map2.update(keys[42]))[0] = std::byte{255};
  EXPECT_NE(map2, map0);
}

template <typename Key, typename Kernel>
void test_reserve_resize_clear() {
  using set_t = set<Key, flags_t::none, Kernel>;
  using conf_t = typename set_t::conf_type;
  using key_t = typename set_t::key_type;

  constexpr int_t num_keys{100};
  key_t keys[num_keys];
  make_key_sequence<key_t>(1337, keys);

  set_t set(conf_t{}.set_capacity(0).auto_adjust());
  EXPECT_EQ(set.insert_range(std::begin(keys), std::end(keys)), num_keys);
  EXPECT_EQ(set.size(), num_keys);

  int_t prev_capacity{set.capacity()};
  set.reserve(111);
  EXPECT_EQ(set.capacity(), prev_capacity);
  EXPECT_EQ(set.size(), num_keys);

  set.reserve(102333);
  EXPECT_NE(set.capacity(), prev_capacity);
  EXPECT_EQ(set.size(), num_keys);

  prev_capacity = set.capacity();
  set.reserve(3333);
  EXPECT_EQ(set.capacity(), prev_capacity);
  EXPECT_EQ(set.size(), num_keys);

  set.resize(111);
  EXPECT_NE(set.capacity(), prev_capacity);
  EXPECT_EQ(set.size(), num_keys);

  prev_capacity = set.capacity();
  set.resize(34);
  EXPECT_EQ(set.capacity(), prev_capacity);
  EXPECT_EQ(set.size(), num_keys);

  set.clear();
  EXPECT_EQ(set.capacity(), prev_capacity);
  EXPECT_EQ(set.size(), 0);

  set.resize(0);
  EXPECT_EQ(set.capacity(), set.conf().min_capacity());
  EXPECT_EQ(set.size(), 0);

  set.grow();
  EXPECT_EQ(set.capacity(), set.conf().min_capacity() * 2);
  EXPECT_EQ(set.size(), 0);
}

template <typename Key, typename Kernel>
void test_erase() {
  using set_t = set<Key, flags_t::none, Kernel>;
  using conf_t = typename set_t::conf_type;
  using key_t = typename set_t::key_type;
  using read_pos_t = typename set_t::read_pos;
  
  // Low congestion case.
  {
    constexpr int_t num_keys{6144};
    key_t keys[num_keys];
    make_key_sequence<key_t>(1337, keys);

    const auto conf{conf_t{}.set_capacity(0).auto_adjust()};
    set_t set(conf);
    set.reserve(num_keys);
    EXPECT_EQ(set.insert_range(std::begin(keys), std::end(keys)), num_keys);

    for (const key_t& key : keys) {
      read_pos_t pos{set.find(key)};
      EXPECT_NE(pos, npos);

      raw_pos_t raw_pos{pos.raw()};
      if (raw_pos % set_t::kernel_size != set_t::kernel_size - 1) continue;

      int_t size{set.size()};
      int_t num_empty{set.num_empty_slots()};
      int_t num_tombstone{set.num_tombstone_slots()};
      int_t num_empty_in_bucket{set.bucket_num_empty_slots_at(pos)};
      EXPECT_TRUE(set.erase(key));

      EXPECT_EQ(set.size(), size - 1);
      if (num_empty_in_bucket != 0) {
        // Erase keys that have adjacent empty slots.
        EXPECT_EQ(set.num_empty_slots(), num_empty + 1);
        EXPECT_EQ(set.num_tombstone_slots(), num_tombstone);
      } else {
        // Erase keys that have adjacent empty slots.
        EXPECT_EQ(set.num_empty_slots(), num_empty);
        EXPECT_EQ(set.num_tombstone_slots(), num_tombstone + 1);
      }
    }
  }

  
  // High congestion case.
  {
    constexpr int_t num_keys{8060};
    key_t keys[num_keys];
    make_key_sequence<key_t>(1337, keys);

    const auto conf{conf_t{}.set_capacity(0).set_grow_bias(conf_t::max_grow_bias).auto_adjust()};
    set_t set(conf);
    set.reserve(num_keys);
    EXPECT_EQ(set.insert_range(std::begin(keys), std::end(keys)), num_keys);

    for (const key_t& key : keys) {
      read_pos_t pos{set.find(key)};
      EXPECT_NE(pos, npos);

      raw_pos_t raw_pos{pos.raw()};
      if (raw_pos % set_t::kernel_size != set_t::kernel_size - 1) continue;

      int_t size{set.size()};
      int_t num_empty{set.num_empty_slots()};
      int_t num_tombstone{set.num_tombstone_slots()};
      int_t num_empty_in_bucket{set.bucket_num_empty_slots_at(pos)};
      EXPECT_TRUE(set.erase(key));

      EXPECT_EQ(set.size(), size - 1);
      if (num_empty_in_bucket != 0) {
        // Erase keys that have adjacent empty slots.
        EXPECT_EQ(set.num_empty_slots(), num_empty + 1);
        EXPECT_EQ(set.num_tombstone_slots(), num_tombstone);
      } else {
        // Erase keys that have adjacent empty slots.
        EXPECT_EQ(set.num_empty_slots(), num_empty);
        EXPECT_EQ(set.num_tombstone_slots(), num_tombstone + 1);
      }
    }
  }
}

template <typename Key, typename Kernel>
void test_erase_auto_shrink() {
  using set_t = set<Key, flags_t::auto_shrink, Kernel>;
  using conf_t = typename set_t::conf_type;
  using key_t = typename set_t::key_type;

  constexpr int_t num_keys{8000};
  key_t keys[num_keys];
  make_key_sequence<key_t>(1337, keys);

  const auto conf{conf_t{}.set_capacity(0).set_grow_bias(conf_t::max_grow_bias).set_shrink_bias(4).auto_adjust()};
  set_t set(conf);
  set.reserve(num_keys);
  EXPECT_EQ(set.insert_range(std::begin(keys), std::end(keys)), num_keys);

  EXPECT_EQ(set.capacity(), 8192);
  EXPECT_EQ(set.size(), num_keys);

  int_t i{};
  for (; set.size() > 2048; ++i) {
    EXPECT_TRUE(set.erase(keys[i]));
  }
  EXPECT_EQ(set.capacity(), 4096);

  for (; set.size() > 1024; ++i) {
    EXPECT_TRUE(set.erase(keys[i]));
  }
  EXPECT_EQ(set.capacity(), 2048);

  for (; set.size() > 512; ++i) {
    EXPECT_TRUE(set.erase(keys[i]));
  }
  EXPECT_EQ(set.capacity(), 1024);

  for (; set.size() > 256; ++i) {
    EXPECT_TRUE(set.erase(keys[i]));
  }
  EXPECT_EQ(set.capacity(), std::max(set_t::kernel_size, {512}));

  for (; set.size() > 128; ++i) {
    EXPECT_TRUE(set.erase(keys[i]));
  }
  EXPECT_EQ(set.capacity(), std::max(set_t::kernel_size, {256}));

  for (; set.size() > 64; ++i) {
    EXPECT_TRUE(set.erase(keys[i]));
  }
  EXPECT_EQ(set.capacity(), std::max(set_t::kernel_size, {128}));

  for (; set.size() > 32; ++i) {
    EXPECT_TRUE(set.erase(keys[i]));
  }
  EXPECT_EQ(set.capacity(), std::max(set_t::kernel_size, {64}));

  for (; set.size() > 16; ++i) {
    EXPECT_TRUE(set.erase(keys[i]));
  }
  EXPECT_EQ(set.capacity(), set.conf().min_capacity());

  for (; set.size() > 8; ++i) {
    EXPECT_TRUE(set.erase(keys[i]));
  }
  EXPECT_EQ(set.capacity(), set.conf().min_capacity());
}

template <typename Key, typename Kernel>
void test_erase_auto_scrub() {
  using set_t = set<Key, flags_t::auto_scrub, Kernel>;
  using conf_t = typename set_t::conf_type;
  using key_t = typename set_t::key_type;

  constexpr int_t num_keys{1000};
  key_t keys[num_keys];
  make_key_sequence<key_t>(1337, keys);

  set_t set(conf_t{}.set_capacity(0).set_grow_bias(conf_t::max_grow_bias).set_scrub_bias(4).auto_adjust());
  set.reserve(num_keys);
  EXPECT_EQ(set.insert_range(std::begin(keys), std::end(keys)), num_keys);

  int_t i{};
  for (; set.size() > 512; ++i) {
    EXPECT_TRUE(set.erase(keys[i]));
  }
  EXPECT_EQ(set.capacity(), 1024);

  for (; i < num_keys; ++i) {
    EXPECT_TRUE(set.contains(keys[i]));
  }
}

template <typename Key, typename Kernel>
void test_erase_auto_reclaim() {
  using set_t = set<Key, flags_t::none, Kernel>;
  using conf_t = typename set_t::conf_type;
  using key_t = typename set_t::key_type;

  //if (set_t::kernel_size == 1) return;

  constexpr int_t num_keys{set_t::kernel_size == 1 ? 7500 : 8000};
  key_t keys[num_keys];
  make_key_sequence<key_t>(1337, keys);

  set_t set(conf_t{}.set_capacity(0).set_grow_bias(conf_t::max_grow_bias).auto_adjust());
  set.reserve(num_keys);
  EXPECT_EQ(set.insert_range(std::begin(keys), std::end(keys)), num_keys);

  // Produce some tombstones.
  EXPECT_EQ(set.erase_range(std::begin(keys), std::end(keys)), num_keys);
  EXPECT_EQ(set.size(), 0);

  int_t prev_capacity{set.capacity()};
  int_t prev_num_empty{set.num_empty_slots()};
  int_t prev_num_tombstone{set.num_tombstone_slots()};
  if (prev_num_tombstone == 0) {
    GTEST_SKIP() << "No tombstones to reclaim...";
    return;
  }

  make_key_sequence<key_t>(13337, keys);
  EXPECT_EQ(set.find_range(std::begin(keys), std::end(keys)), 0);

  for (int_t i{}; i < num_keys / 2; ++i) {
    EXPECT_NE(set.insert(keys[i]), npos);
  }

  EXPECT_EQ(set.capacity(), prev_capacity);
  EXPECT_LT(set.num_empty_slots(), prev_num_empty);
  EXPECT_LT(set.num_tombstone_slots(), prev_num_tombstone);
}

template <typename Key, typename Kernel>
void test_no_mulitmap_insert() {
  using set_t = set<Key, flags_t::none, Kernel>;
  using conf_t = typename set_t::conf_type;
  using key_t = typename set_t::key_type;

  const auto conf{conf_t{}.set_capacity(0).auto_adjust()};
  set_t set(conf);

  constexpr int_t num_keys{666};
  key_t keys[num_keys];
  make_key_sequence<key_t>(1337, keys);
  EXPECT_EQ(set.insert_range(std::begin(keys), std::end(keys)), num_keys);
  EXPECT_EQ(set.size(), num_keys);
  
  EXPECT_EQ(set.insert_range(std::begin(keys), std::end(keys)), num_keys);
  EXPECT_EQ(set.size(), num_keys);
}

template <typename Key, typename Kernel>
void test_mulitmap_insert() {
  using set_t = set<Key, flags_t::duplicates, Kernel>;
  using conf_t = typename set_t::conf_type;
  using key_t = typename set_t::key_type;

  const auto conf{conf_t{}.set_capacity(0).auto_adjust()};
  set_t set(conf);

  constexpr int_t num_keys{666};
  key_t keys[num_keys];
  make_key_sequence<key_t>(1337, keys);
  EXPECT_EQ(set.insert_range(std::begin(keys), std::end(keys)), num_keys);
  EXPECT_EQ(set.size(), num_keys);
  
  EXPECT_EQ(set.insert_range(std::begin(keys), std::end(keys)), num_keys);
  EXPECT_EQ(set.size(), num_keys * 2);
}

template <typename Key, typename Kernel>
void test_mulitmap_find() {
  using map_t = map<Key, double, flags_t::blobs | flags_t::duplicates, Kernel>;
  using conf_t = typename map_t::conf_type;
  using key_t = typename map_t::key_type;
  using write_pos_t = typename map_t::write_pos;

  constexpr int_t num_keys{200};
  key_t keys[num_keys];
  make_key_sequence<key_t>(1337, keys);

  const auto conf{conf_t{}.set_capacity(0).set_grow_bias(conf_t::max_grow_bias).set_blob(3).auto_adjust()};
  map_t map(conf);
  map.reserve(num_keys * 4);
  const int_t prev_capacity{map.capacity()};

  for (const key_t& key : keys) {
    write_pos_t pos{map.insert(key)};
    EXPECT_NE(pos, npos);

    map.set_value_at(pos, 1.0);
    EXPECT_EQ(map.value_at(pos), 1.0);
  }
  EXPECT_EQ(map.capacity(), prev_capacity);
  EXPECT_EQ(map.size(), num_keys);

  for (const key_t& key : keys) {
    auto [pos, seq]{map.find_first(key)};
    EXPECT_NE(pos, npos);
    EXPECT_EQ(map.value_at(pos), 1.0);

    std::tie(pos, seq) = map.find_next(pos, seq, key);
    EXPECT_EQ(pos, npos);
  }

  for (const key_t& key : keys) {
    auto [pos, seq]{map.update_first(key)};
    EXPECT_NE(pos, npos);
    EXPECT_EQ(map.value_at(pos), 1.0);
    
    std::tie(pos, seq) = map.update_next(std::move(pos), seq, key);
    EXPECT_EQ(pos, npos);
  }

  // Insert keys again. Now we should find a successor for each key.
  for (const key_t& key : keys) {
    write_pos_t pos{map.insert(key)};
    EXPECT_NE(pos, npos);
    map.set_value_at(pos, 2.0);
    EXPECT_EQ(map.value_at(pos), 2.0);

    pos = map.insert(key);
    EXPECT_NE(pos, npos);
    map.set_value_at(pos, 3.0);
    EXPECT_EQ(map.value_at(pos), 3.0);

    pos = map.insert(key);
    EXPECT_NE(pos, npos);
    map.set_value_at(pos, 4.0);
    EXPECT_EQ(map.value_at(pos), 4.0);

    pos = map.insert(key);
    EXPECT_NE(pos, npos);
    map.set_value_at(pos, 5.0);
    EXPECT_EQ(map.value_at(pos), 5.0);
  }
  EXPECT_EQ(map.capacity(), prev_capacity);
  EXPECT_EQ(map.size(), num_keys * 5);

  for (const key_t& key : keys) {
    auto [pos, seq]{map.find_first(key)};
    EXPECT_NE(pos, npos);
    EXPECT_EQ(map.value_at(pos), 1.0);

    std::tie(pos, seq) = map.find_next(pos, seq, key);
    EXPECT_NE(pos, npos);
    EXPECT_EQ(map.value_at(pos), 2.0);

    std::tie(pos, seq) = map.find_next(pos, seq, key);
    EXPECT_NE(pos, npos);
    EXPECT_EQ(map.value_at(pos), 3.0);

    std::tie(pos, seq) = map.find_next(pos, seq, key);
    EXPECT_NE(pos, npos);
    EXPECT_EQ(map.value_at(pos), 4.0);

    std::tie(pos, seq) = map.find_next(pos, seq, key);
    EXPECT_NE(pos, npos);
    EXPECT_EQ(map.value_at(pos), 5.0);

    std::tie(pos, seq) = map.find_next(pos, seq, key);
    EXPECT_EQ(pos, npos);
  }

  for (const key_t& key : keys) {
    auto [pos, seq]{map.update_first(key)};
    EXPECT_NE(pos, npos);
    EXPECT_EQ(map.value_at(pos), 1.0);
    
    std::tie(pos, seq) = map.update_next(std::move(pos), seq, key);
    EXPECT_NE(pos, npos);
    EXPECT_EQ(map.value_at(pos), 2.0);

    std::tie(pos, seq) = map.update_next(std::move(pos), seq, key);
    EXPECT_NE(pos, npos);
    EXPECT_EQ(map.value_at(pos), 3.0);

    std::tie(pos, seq) = map.update_next(std::move(pos), seq, key);
    EXPECT_NE(pos, npos);
    EXPECT_EQ(map.value_at(pos), 4.0);

    std::tie(pos, seq) = map.update_next(std::move(pos), seq, key);
    EXPECT_NE(pos, npos);
    EXPECT_EQ(map.value_at(pos), 5.0);

    std::tie(pos, seq) = map.update_next(std::move(pos), seq, key);
    EXPECT_EQ(pos, npos);
  }

  // `_all` functions`.
  for (const key_t& key : keys) {
    EXPECT_EQ(map.count(key), 5);
    EXPECT_EQ(map.find_all(key).size(), 5);
    EXPECT_EQ(map.update_all(key).size(), 5);
    EXPECT_EQ(map.all_values_for(key), (std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0}));
    EXPECT_EQ(map.all_blobs_for(key).size(), 5 * map.conf().blob_size());
  }
}

template <typename Key, typename Kernel>
void test_mulitmap_erase() {
  using map_t = map<Key, double, flags_t::duplicates, Kernel>;
  using conf_t = typename map_t::conf_type;
  using key_t = typename map_t::key_type;
  using write_pos_t = typename map_t::write_pos;

  constexpr int_t num_keys{200};
  key_t keys[num_keys];
  make_key_sequence<key_t>(1337, keys);

  const auto conf{conf_t{}.set_capacity(0).set_grow_bias(conf_t::max_grow_bias).auto_adjust()};
  map_t map(conf);
  map.reserve(num_keys * 4);
  const int_t prev_capacity{map.capacity()};

  // Insert keys.
  for (const key_t& key : keys) {
    write_pos_t pos{map.insert(key)};
    EXPECT_NE(pos, npos);

    map.set_value_at(pos, 1.0);
    EXPECT_EQ(map.value_at(pos), 1.0);
  }
  EXPECT_EQ(map.capacity(), prev_capacity);
  EXPECT_EQ(map.size(), num_keys);

  // Insert keys again.
  for (const key_t& key : keys) {
    write_pos_t pos{map.insert(key)};
    EXPECT_NE(pos, npos);
    map.set_value_at(pos, 2.0);
    EXPECT_EQ(map.value_at(pos), 2.0);

    pos = map.insert(key);
    EXPECT_NE(pos, npos);
    map.set_value_at(pos, 3.0);
    EXPECT_EQ(map.value_at(pos), 3.0);

    pos = map.insert(key);
    EXPECT_NE(pos, npos);
    map.set_value_at(pos, 4.0);
    EXPECT_EQ(map.value_at(pos), 4.0);

    pos = map.insert(key);
    EXPECT_NE(pos, npos);
    map.set_value_at(pos, 5.0);
    EXPECT_EQ(map.value_at(pos), 5.0);
  }
  EXPECT_EQ(map.capacity(), prev_capacity);
  EXPECT_EQ(map.size(), num_keys * 5);

  // Ensure we can find each key duplicate.
  for (const key_t& key : keys) {
    auto [pos, seq]{map.find_first(key)};
    EXPECT_NE(pos, npos);
    EXPECT_EQ(map.value_at(pos), 1.0);

    std::tie(pos, seq) = map.find_next(pos, seq, key);
    EXPECT_NE(pos, npos);
    EXPECT_EQ(map.value_at(pos), 2.0);

    std::tie(pos, seq) = map.find_next(pos, seq, key);
    EXPECT_NE(pos, npos);
    EXPECT_EQ(map.value_at(pos), 3.0);

    std::tie(pos, seq) = map.find_next(pos, seq, key);
    EXPECT_NE(pos, npos);
    EXPECT_EQ(map.value_at(pos), 4.0);

    std::tie(pos, seq) = map.find_next(pos, seq, key);
    EXPECT_NE(pos, npos);
    EXPECT_EQ(map.value_at(pos), 5.0);

    std::tie(pos, seq) = map.find_next(pos, seq, key);
    EXPECT_EQ(pos, npos);
  }

  // Erase the 1st and 3nd key.
  for (const key_t& key : keys) {
    auto [found, pos, seq]{map.erase_first(key)};
    EXPECT_TRUE(found);
    EXPECT_NE(pos, npos);
    //EXPECT_EQ(map.value_at(pos), 1.0);

    std::tie(pos, seq) = map.update_next(std::move(pos), seq, key);
    EXPECT_NE(pos, npos);
    EXPECT_EQ(map.value_at(pos), 2.0);

    std::tie(found, pos, seq) = map.erase_next(std::move(pos), seq, key);
    EXPECT_TRUE(found);
    EXPECT_NE(pos, npos);
    //EXPECT_EQ(map.value_at(pos), 3.0);

    std::tie(pos, seq) = map.update_next(std::move(pos), seq, key);
    EXPECT_NE(pos, npos);
    EXPECT_EQ(map.value_at(pos), 4.0);

    std::tie(pos, seq) = map.update_next(std::move(pos), seq, key);
    EXPECT_NE(pos, npos);
    EXPECT_EQ(map.value_at(pos), 5.0);

    std::tie(found, pos, seq) = map.erase_next(std::move(pos), seq, key);
    EXPECT_FALSE(found);
    EXPECT_EQ(pos, npos);
  }

  // Now should only be able to find the 2nd, 4th and 5th key.
  for (const key_t& key : keys) {
    auto [pos, seq]{map.find_first(key)};
    EXPECT_NE(pos, npos);
    EXPECT_EQ(map.value_at(pos), 2.0);

    std::tie(pos, seq) = map.find_next(pos, seq, key);
    EXPECT_NE(pos, npos);
    EXPECT_EQ(map.value_at(pos), 4.0);

    std::tie(pos, seq) = map.find_next(pos, seq, key);
    EXPECT_NE(pos, npos);
    EXPECT_EQ(map.value_at(pos), 5.0);

    std::tie(pos, seq) = map.find_next(pos, seq, key);
    EXPECT_EQ(pos, npos);
  }

  // `erase_all` should get rid of all the remaining keys.
  for (const key_t& key : keys) {
    EXPECT_EQ(map.erase_all(key), 3);
  }
  EXPECT_EQ(map.size(), 0);
}

#define TEST_MAPS_2_(_X_, _Y_) \
  TEST(test_set_conf, _X_##_##_Y_) { test_conf<_Y_, _X_>(); }\
  TEST(test_init, _X_##_##_Y_) { test_init<_Y_, _X_>(); }\
  TEST(test_insert, _X_##_##_Y_) { test_insert<_Y_, _X_>(); }\
  TEST(test_set_iterators, _X_##_##_Y_) { test_iterators<_Y_, _X_>(); }\
  TEST(test_map_iterators, _X_##_##_Y_) { test_map_iterators<_Y_, _X_>(); }\
  TEST(test_set_copy_move_compare, _X_##_##_Y_) { test_copy_move_compare<_Y_, _X_>(); }\
  TEST(test_reserve_resize_clear, _X_##_##_Y_) { test_reserve_resize_clear<_Y_, _X_>(); }\
  TEST(test_erase, _X_##_##_Y_) { test_erase<_Y_, _X_>(); }\
  TEST(test_erase_auto_shrink, _X_##_##_Y_) { test_erase_auto_shrink<_Y_, _X_>(); }\
  TEST(test_erase_auto_scrub, _X_##_##_Y_) { test_erase_auto_scrub<_Y_, _X_>(); }\
  TEST(test_erase_auto_reclaim, _X_##_##_Y_) { test_erase_auto_reclaim<_Y_, _X_>(); }\
  TEST(test_no_mulitmap_insert, _X_##_##_Y_) { test_no_mulitmap_insert<_Y_, _X_>(); }\
  TEST(test_mulitmap_insert, _X_##_##_Y_) { test_mulitmap_insert<_Y_, _X_>(); }\
  TEST(test_mulitmap_find, _X_##_##_Y_) { test_mulitmap_find<_Y_, _X_>(); }\
  TEST(test_mulitmap_erase, _X_##_##_Y_) { test_mulitmap_erase<_Y_, _X_>(); }

#define TEST_MAPS_(_X_) \
  TEST_MAPS_2_(_X_, int32_t)\
  TEST_MAPS_2_(_X_, int64_t)\
  TEST_MAPS_2_(_X_, string_t)

// clang-format off
NVHM_FOR_EACH_(
  TEST_MAPS_,
  array_kernel1_t,
  array_kernel2_t,
  array_kernel4_t,
  array_kernel8_t,
  array_kernel16_t,
  array_kernel32_t,
  array_kernel64_t,
  array_kernel128_t,
  array_kernel256_t,
  array_kernel512_t
)

NVHM_FOR_EACH_(
  TEST_MAPS_,
  uint_kernel1_t,
  uint_kernel2_t,
  uint_kernel4_t,
  uint_kernel8_t,
  uint_kernel16_t
)

NVHM_FOR_EACH_(
  TEST_MAPS_,
  fast_uint_kernel1_t,
  fast_uint_kernel2_t,
  fast_uint_kernel4_t,
  fast_uint_kernel8_t,
  fast_uint_kernel16_t
)

#if NVHM_WITH_SSE >= 2
NVHM_FOR_EACH_(TEST_MAPS_, sse_kernel_t)
#endif

#if NVHM_WITH_AVX >= 2
NVHM_FOR_EACH_(TEST_MAPS_, avx_kernel_t)
#endif

#if NVHM_WITH_AVX512
NVHM_FOR_EACH_(TEST_MAPS_, avx512_kernel_t)
#endif

#if NVHM_WITH_NEON
NVHM_FOR_EACH_(
  TEST_MAPS_,
  neon_kernel8_t,
  neon_kernel16_t,
  neon_kernel32_t,
  neon_kernel64_t
)
#endif

#if NVHM_WITH_SVE
NVHM_FOR_EACH_(
  TEST_MAPS_
#if NVHM_WITH_SVE_SIZE >= 1
  , sve_kernel1_t
#endif
#if NVHM_WITH_SVE_SIZE >= 2
  , sve_kernel2_t
#endif
#if NVHM_WITH_SVE_SIZE >= 4
  , sve_kernel4_t
#endif
#if NVHM_WITH_SVE_SIZE >= 8
  , sve_kernel8_t
#endif
#if NVHM_WITH_SVE_SIZE >= 16
  , sve_kernel16_t
#endif
#if NVHM_WITH_SVE_SIZE >= 32
  , sve_kernel32_t
#endif
#if NVHM_WITH_SVE_SIZE >= 64
  , sve_kernel64_t
#endif
#if NVHM_WITH_SVE_SIZE >= 128
  , sve_kernel128_t
#endif
#if NVHM_WITH_SVE_SIZE >= 256
  , sve_kernel256_t
#endif
)
#endif
// clang-format on