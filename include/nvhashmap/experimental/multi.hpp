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

#include "hash.hpp"
#include <algorithm>
#include <memory>

namespace nvhm { namespace experimental {

template <typename Key>
constexpr size_t key_to_shard(const Key& k, const size_t shard_mask) noexcept {
  hash_t h{hasher<Key>(k)};

  // TODO: Not yet configurable because we may want to change that later.
  constexpr uint64_t c{UINT64_C(0x63a0'8307'3c4c'e4a1)};
  const auto h128{static_cast<__uint128_t>(h) * c};
  h = static_cast<hash_t>(h128 >> 64) + static_cast<hash_t>(h128);

  return h & shard_mask;
}

template <typename Shard>
class multi {
 public:
  using shard_type = Shard;
  using key_type = typename shard_type::key_type;
  using value_type = typename shard_type::value_type;
  constexpr static bool has_values{shard_type::has_values};
  using raw_value_type = typename shard_type::raw_value_type;

  using kernel_type = typename shard_type::kernel_type;
  using mask_type = typename shard_type::mask_type;
  using probe_seq_type = typename shard_type::probe_seq_type;
  using prefetch_type = std::tuple<size_t, typename shard_type::prefetch_type>;

  struct read_pos final {
    using pos_type = typename shard_type::read_pos_type;

    read_pos() = delete;

    inline operator pos_type() const noexcept { return pos; }

    const shard_type* const shard;
    const pos_type pos;
  };

  struct write_pos final {
    using pos_type = typename shard_type::write_pos_type;

    write_pos() = delete;

    inline operator pos_type() const noexcept { return pos; }

    shard_type* const shard;
    const pos_type pos;
  };

  using read_pos_type = read_pos;
  using write_pos_type = write_pos;

  constexpr static size_t max_num_shards{32'768};
  static_assert(has_single_bit(max_num_shards));
  constexpr static size_t num_shard_bits{countr_zero(max_num_shards)};
  static_assert(num_shard_bits >= 4 && num_shard_bits <= 16);

  multi() = delete;

  template <typename... ShardArgs>
  inline multi(const size_t num_shards = 1, ShardArgs&&... args) {
    if (num_shards >= 1 && num_shards <= max_num_shards && has_single_bit(num_shards)) {
      throw std::out_of_range(
        "Number of shards must be within [1, max_num_shards] and a power of 2."
      );
    }

    shard_mask_ = num_shards - 1;
    shards_.resize(num_shards);
    for (size_t i{}; i != num_shards; ++i) {
      shards_[i] = std::make_unique<shard_type>(args...);
    }
  }

  inline multi(const multi&) = default;
  inline multi(multi&&) = default;
  inline multi& operator=(const multi&) = default;
  inline multi& operator=(multi&&) = default;

  inline ~multi() = default;

  inline void swap(multi& that) noexcept {
    std::swap(shard_mask_, that.shard_mask_);
    std::swap(shards_, that.shards_);
  }

  inline size_t raw_value_size() const noexcept { return shards_.front().raw_value_size(); }

  inline size_t capacity() const noexcept {
    size_t res{};
    for (const auto& __restrict s : shards_) {
      res += s->capacity();
    }
    return res;
  }

  inline size_t size() const noexcept {
    size_t res{};
    for (const auto& __restrict s : shards_) {
      res += s->size();
    }
    return res;
  }

  inline bool empty() const noexcept {
    bool res{true};
    for (const auto& __restrict s : shards_) {
      res &= s->empty();
    }
    return res;
  }

  inline bool full() const noexcept {
    bool res{false};
    for (const auto& __restrict s : shards_) {
      res |= s->full();
    }
    return res;
  }

  inline std::vector<size_t> shard_capacities() const noexcept {
    std::vector<size_t> res(shards_.size());
    std::transform(shards_.begin(), shards_.end(), res.begin(), [](const auto& __restrict s) {
      return s->capacity();
    });
    return res;
  }

  inline std::vector<size_t> shard_sizes() const noexcept {
    std::vector<size_t> res(shards_.size());
    std::transform(shards_.begin(), shards_.end(), res.begin(), [](const auto& __restrict s) {
      return s->size();
    });
    return res;
  }

 public:
  inline void clear() {
    for (auto& __restrict s : shards_) {
      s->clear();
    }
  }

  inline bool contains(const key_type& key) const noexcept {
    return shards_[key_to_shard(key, shard_mask_)].contains(key);
  }

  inline bool contains(const key_type& key, const prefetch_type& pre) const noexcept {
    return shards_[key_to_shard(key, shard_mask_)].contains(key, pre);
  }

  inline bool erase(const key_type& key) {
    return shards_[key_to_shard(key, shard_mask_)].erase(key);
  }

  inline bool erase(const key_type& key, const prefetch_type& pre) {
    return shards_[key_to_shard(key, shard_mask_)].erase(key, pre);
  }

  inline bool erase_at(const write_pos_type& pos) noexcept {
    NVHM_ASSERT_(std::find(shards_.begin(), shards_.end(), pos.shard) != shards_.end());
    return pos.shard->erase_at(pos.pos);
  }

  inline read_pos_type lookup(const key_type& key) const noexcept {
    const shard_type& __restrict s{shards_[key_to_shard(key, shard_mask_)]};
    return {&s, s.lookup(key)};
  }

  inline read_pos_type lookup(const key_type& key, const prefetch_type& pre) const noexcept {
    const shard_type& __restrict s{shards_[std::get<0>(pre)]};
    return {&s, s.lookup(key, std::get<1>(pre))};
  }

  inline write_pos_type update(const key_type& key) noexcept {
    shard_type& __restrict s{shards_[key_to_shard(key, shard_mask_)]};
    return {&s, s.update(key)};
  }

  inline write_pos_type update(const key_type& key, const prefetch_type& pre) noexcept {
    shard_type& __restrict s{shards_[std::get<0>(pre)]};
    return {&s, s.update(key, std::get<1>(pre))};
  }

  inline write_pos_type upsert(const key_type& key) {
    shard_type& __restrict s{shards_[key_to_shard(key, shard_mask_)]};
    return {&s, s.upsert(key)};
  }

  inline write_pos_type upsert(const key_type& key, const prefetch_type& pre) {
    shard_type& __restrict s{shards_[std::get<0>(pre)]};
    return {&s, s.upsert(key, std::get<1>(pre))};
  }

  template <typename It>
  inline write_pos_type upsert(const key_type& key, It& it) {
    shard_type& __restrict s{shards_[key_to_shard(key, shard_mask_)]};
    return {&s, s.upsert(key, it)};
  }

  template <typename It>
  inline write_pos_type upsert(const key_type& key, It& it, const prefetch_type& pre) {
    shard_type& __restrict s{shards_[std::get<0>(pre)]};
    return {&s, s.upsert(key, it, std::get<1>(pre))};
  }

  inline prefetch_type read_prefetch(const key_type& key, const bool optimistic) const noexcept {
    const size_t idx{key_to_shard(key, shard_mask_)};
    return {idx, shards_[idx].read_prefetch(key, optimistic)};
  }

  inline prefetch_type write_prefetch(const key_type& key, const bool optimistic) noexcept {
    const size_t idx{key_to_shard(key, shard_mask_)};
    return {idx, shards_[idx].write_prefetch(key, optimistic)};
  }

 public:
  inline read_pos_type first() const noexcept {
    using pos_type = typename read_pos_type::pos_type;

    for (const auto& __restrict s : shards_) {
      const pos_type pos{s->first()};
      if (pos != npos) {
        return {s.get(), pos};
      }
    }
    return {nullptr, npos};
  }

  inline write_pos_type first() noexcept {
    using pos_type = typename write_pos_type::pos_type;

    for (const auto& __restrict s : shards_) {
      const pos_type pos{s->first()};
      if (pos != npos) {
        return {s.get(), pos};
      }
    }
    return {nullptr, npos};
  }

  template <typename OutIt>
  inline OutIt keys(OutIt it) const noexcept {
    for (const auto& __restrict s : shards_) {
      it = s->keys(it);
    }
    return it;
  }

  template <typename OutIt>
  inline OutIt keys_and_values(OutIt it) const noexcept {
    for (const auto& __restrict s : shards_) {
      it = s->keys_and_values(it);
    }
    return it;
  }

  inline void transform_values(const std::function<void(value_type& __restrict)>& __restrict f) {
    for (const auto& __restrict s : shards_) {
      s.transform_values(f);
    }
  }

  inline void transform_raw_values(
    const std::function<raw_value_type(raw_value_type)>& __restrict f
  ) {
    for (const auto& __restrict s : shards_) {
      s.transform_raw_values(f);
    }
  }

 public:
  inline bool is_occupied(const read_pos_type& pos) const noexcept {
    NVHM_ASSERT_(std::find(shards_.begin(), shards_.end(), pos.shard) != shards_.end());
    return pos.shard->is_occupied(pos.pos);
  }

  inline state_t state_at(const read_pos_type& pos) const noexcept {
    NVHM_ASSERT_(std::find(shards_.begin(), shards_.end(), pos.shard) != shards_.end());
    return pos.shard->state_at(pos.pos);
  }

  inline const key_type& key_at(const read_pos_type& pos) const noexcept {
    NVHM_ASSERT_(std::find(shards_.begin(), shards_.end(), pos.shard) != shards_.end());
    return pos.shard->key_at(pos.pos);
  }

  inline const value_type& value_at(const read_pos_type& pos) const noexcept {
    NVHM_ASSERT_(std::find(shards_.begin(), shards_.end(), pos.shard) != shards_.end());
    return pos.shard->value_at(pos.pos);
  }

  inline value_type& value_at(const write_pos_type& pos) noexcept {
    NVHM_ASSERT_(std::find(shards_.begin(), shards_.end(), pos.shard) != shards_.end());
    return pos.shard->value_at(pos.pos);
  }

  inline const raw_value_type* raw_values_at(const read_pos_type& pos) const noexcept {
    NVHM_ASSERT_(std::find(shards_.begin(), shards_.end(), pos.shard) != shards_.end());
    return pos.shard->raw_values_at(pos.pos);
  }

  inline raw_value_type* raw_values_at(const write_pos_type& pos) noexcept {
    NVHM_ASSERT_(std::find(shards_.begin(), shards_.end(), pos.shard) != shards_.end());
    return pos.shard->raw_values_at(pos.pos);
  }

  inline const raw_value_type* get_raw_values(
    const read_pos_type& pos, raw_value_type* const dst, const size_t n
  ) const noexcept {
    NVHM_ASSERT_(std::find(shards_.begin(), shards_.end(), pos.shard) != shards_.end());
    return pos.shard->get_raw_values(pos.pos, dst, n);
  }

  inline void set_raw_values(
    const write_pos_type& pos, const raw_value_type* const src, const size_t n
  ) noexcept {
    NVHM_ASSERT_(std::find(shards_.begin(), shards_.end(), pos.shard) != shards_.end());
    return pos.shard->set_raw_values(pos.pos, src, n);
  }

 public:
  inline size_t psl(const key_type& key) const noexcept {
    return shards_[key_to_shard(key, shard_mask_)].psl(key);
  }

  inline std::array<size_t, kernel_type::size> state_collisions() const noexcept {
    std::array<size_t, kernel_type::size> res{};
    for (const auto& __restrict s : shards_) {
      const std::array<size_t, kernel_type::size> tmp{s->state_collisions()};
      for (size_t i{}; i != kernel_type::size; ++i) {
        res[i] += tmp[i];
      }
    }
    return res;
  }

  inline std::vector<std::array<size_t, kernel_type::size>> shard_state_collisions(
  ) const noexcept {
    std::vector<std::array<size_t, kernel_type::size>> res(shards_.size());
    std::transform(shards_.begin(), shards_.end(), res.begin(), [](const auto& __restrict s) {
      return s->state_collisions();
    });
    return res;
  }

 protected:
  size_t shard_mask_;
  std::vector<std::unique_ptr<shard_type>> shards_;
};

}}
