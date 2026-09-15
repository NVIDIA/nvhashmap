/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>
#include <nvhashmap/map.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <iterator>
#include <vector>

using namespace nvhm;

template <typename Kernel, bool MinimizePSL>
using reuse_map_t = map<int64_t, void, char, Kernel, default_seq_t, MinimizePSL, false>;

template <typename Map>
class TombstoneReuseTest : public ::testing::Test {
 protected:
  using kernel_t = typename Map::kernel_type;
  static constexpr size_t row_bytes{68};
  Map table{1024, row_bytes, 32};

  std::vector<int64_t> colliding_keys(size_t count) const {
    std::vector<int64_t> keys;
    const size_t bucket_mask{table.capacity() - kernel_t::size};
    // Fill one probe block completely so erase must leave a tombstone.
    for (int64_t key{1}; keys.size() < count; ++key) {
      if (hash_to_pos<kernel_t>(key_to_hash(key), bucket_mask) == 0) {
        keys.push_back(key);
      }
    }
    return keys;
  }

  static std::array<char, row_bytes> value(int64_t key) {
    std::array<char, row_bytes> bytes{};
    std::memcpy(bytes.data(), &key, sizeof(key));
    bytes.back() = 42;
    return bytes;
  }

  void insert(int64_t key, size_t& inserted) {
    const auto pos{table.upsert(key, inserted)};
    const auto bytes{value(key)};
    table.set_raw_values_at(pos, bytes.data(), bytes.size());
  }

  void expect_contents(std::vector<int64_t> expected) const {
    std::vector<int64_t> actual;
    table.keys(std::back_inserter(actual));
    std::sort(actual.begin(), actual.end());
    std::sort(expected.begin(), expected.end());
    EXPECT_EQ(actual, expected);
    EXPECT_EQ(table.size(), actual.size());
    for (int64_t key : expected) {
      const auto pos{table.lookup(key)};
      ASSERT_NE(pos, npos);
      const auto bytes{value(key)};
      EXPECT_EQ(std::memcmp(table.raw_values_at(pos), bytes.data(), bytes.size()), 0);
    }
  }
};

using ReuseMaps = ::testing::Types<
  reuse_map_t<uint_kernel64_t, true>, reuse_map_t<array_kernel128_t, true>,
  reuse_map_t<default_kernel_t, true>, reuse_map_t<uint_kernel64_t, false>,
  reuse_map_t<array_kernel128_t, false>, reuse_map_t<default_kernel_t, false>>;
TYPED_TEST_SUITE(TombstoneReuseTest, ReuseMaps);

TYPED_TEST(TombstoneReuseTest, ReuseCountsEachNewKeyOnceAndSurvivesGrowth) {
  constexpr size_t block_size{TypeParam::kernel_type::size};
  const auto keys{this->colliding_keys(block_size + 4)};
  size_t inserted{};
  for (size_t i{}; i < block_size + 2; ++i)
    this->insert(keys[i], inserted);
  ASSERT_EQ(this->table.size(), block_size + 2);
  ASSERT_TRUE(this->table.erase(keys[0]));
  ASSERT_TRUE(this->table.erase(keys[1]));
  ASSERT_FALSE(this->table.erase(keys[0]));
  ASSERT_EQ(this->table.num_tombstones(), 2);
  ASSERT_EQ(this->table.size(), block_size);
  const size_t used_before{this->table.num_used()};

  // An existing key in the next block must be found past the tombstones.
  this->insert(keys[block_size], inserted);
  EXPECT_EQ(inserted, block_size + 2);
  EXPECT_EQ(this->table.size(), block_size);
  EXPECT_EQ(this->table.num_tombstones(), 2);

  for (size_t i{}; i < 2; ++i) {
    this->insert(keys[block_size + 2 + i], inserted);
    EXPECT_EQ(inserted, block_size + 3 + i);
    ASSERT_EQ(this->table.size(), block_size + 1 + i);
    if constexpr (TypeParam::minimize_psl) {
      EXPECT_EQ(this->table.num_used(), used_before);
      EXPECT_EQ(this->table.num_tombstones(), 1 - i);
    } else {
      EXPECT_EQ(this->table.num_used(), used_before + 1 + i);
      EXPECT_EQ(this->table.num_tombstones(), 2);
    }
  }

  const std::vector<int64_t> expected(keys.begin() + 2, keys.end());
  this->expect_contents(expected);
  const size_t old_capacity{this->table.capacity()};
  this->table.grow();
  EXPECT_GT(this->table.capacity(), old_capacity);
  EXPECT_EQ(this->table.num_tombstones(), 0);
  this->expect_contents(expected);
  for (int64_t key : expected)
    EXPECT_TRUE(this->table.erase(key));
  EXPECT_EQ(this->table.size(), 0);
  EXPECT_TRUE(this->table.empty());
}

TYPED_TEST(TombstoneReuseTest, RepeatedEraseReinsertAndReplayKeepExactSize) {
  constexpr size_t block_size{TypeParam::kernel_type::size};
  const auto keys{this->colliding_keys(block_size + 1)};
  size_t inserted{};
  for (int64_t key : keys)
    this->insert(key, inserted);
  ASSERT_EQ(this->table.size(), keys.size());

  for (size_t cycle{}; cycle < 8; ++cycle) {
    SCOPED_TRACE(cycle);
    ASSERT_TRUE(this->table.erase(keys[0]));
    ASSERT_EQ(this->table.size(), keys.size() - 1);
    this->insert(keys[0], inserted);
    EXPECT_EQ(inserted, keys.size() + cycle + 1);
    ASSERT_EQ(this->table.size(), keys.size());
    // Replaying an already-applied update must not increment either count.
    this->insert(keys[0], inserted);
    EXPECT_EQ(inserted, keys.size() + cycle + 1);
    EXPECT_EQ(this->table.size(), keys.size());
    this->expect_contents(keys);
  }
}

TYPED_TEST(TombstoneReuseTest, ReclaimedEmptySlotCountsNewKeyOnce) {
  const auto keys{this->colliding_keys(2)};
  size_t inserted{};
  this->insert(keys[0], inserted);
  ASSERT_TRUE(this->table.erase(keys[0]));
  ASSERT_EQ(this->table.num_tombstones(), 0);
  ASSERT_EQ(this->table.num_used(), 0);
  this->insert(keys[1], inserted);
  EXPECT_EQ(inserted, 2);
  EXPECT_EQ(this->table.num_used(), 1);
  EXPECT_EQ(this->table.size(), 1);
  this->expect_contents({keys[1]});
}
