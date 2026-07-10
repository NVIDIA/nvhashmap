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

#include <nvhashmap/allocator.hpp>
#include <nvhashmap/kernel.hpp>

#include "test_common.hpp"
#include <gtest/gtest.h>
#include <random>
#include <iomanip>

using namespace nvhm;

template <typename Kernel>
const static std::function<state_t()> state_gen{[]() {
  using kern_t = Kernel;
  static std::uniform_int_distribution<state_t> dist(-20, 127);
  state_t s{dist(rng)};
  if (s < -10) {
    s = kern_t::empty;
  } else if (s < 0) {
    s = kern_t::tombstone;
  }
  return s;
}};

template <typename Kernel>
void test_kernel_load_store(const int_t num_iter = 1024) {
  using kern_t = Kernel;

  alignas(kern_t::size) state_t s0[kern_t::size];
  alignas(kern_t::size) state_t s1[kern_t::size];

  for (int_t i{}; i < num_iter; ++i) {
    std::generate(std::begin(s0), std::end(s0), state_gen<kern_t>);

    kern_t::store(kern_t::load(s0), s1);

    for (int_t j{}; j < kern_t::size; ++j) {
      EXPECT_EQ(s0[j], s1[j]);
    }
  }
}

template <typename Kernel>
void test_kernel_at(const int_t num_iter = 1024) {
  using kern_t = Kernel;
  using repr_t = typename kern_t::repr_type;

  alignas(kern_t::size) state_t s[kern_t::size];

  for (int_t i{}; i < num_iter; ++i) {
    std::generate(std::begin(s), std::end(s), state_gen<kern_t>);

    repr_t k{kern_t::load(s)};

    for (int_t j{}; j < kern_t::size; ++j) {
      EXPECT_EQ(kern_t::at(k, j), s[j]);
    }
  }
}

template <typename Kernel>
void test_kernel_mask(const int_t num_iter = 1024) {
  using kern_t = Kernel;
  using repr_t = typename kern_t::repr_type;
  using mask_t = typename kern_t::mask_type;
  using mask_repr_t = typename mask_t::repr_type;

  mask_repr_t m;
  alignas(kern_t::size) state_t s[kern_t::size];

  for (int_t i{}; i < num_iter; ++i) {
    std::generate(std::begin(s), std::end(s), state_gen<kern_t>);

    repr_t k{kern_t::load(s)};

    // Mask existing values.
    for (int_t j{}; j < kern_t::size; ++j) {
      m = kern_t::mask(k, s[j]);
      EXPECT_TRUE(mask_t::at(m, j));

      m = kern_t::mask_not(k, s[j]);
      EXPECT_FALSE(mask_t::at(m, j));
    }

    // Mask all possible values.
    if constexpr (!kern_t::produces_false_positives) {
      for (int_t n{std::numeric_limits<state_t>::min()}; n <= std::numeric_limits<state_t>::max(); ++n) {
        m = kern_t::mask(k, static_cast<state_t>(n));
        for (int_t j{}; j < kern_t::size; ++j) {
          EXPECT_EQ(mask_t::at(m, j), s[j] == n);
        }

        m = kern_t::mask_not(k, static_cast<state_t>(n));
        for (int_t j{}; j < kern_t::size; ++j) {
          EXPECT_NE(mask_t::at(m, j), s[j] == n);
        }
      }
    }
  }
}

template <typename Kernel>
void test_kernel_mask_hash(const int_t num_iter = 1024) {
  using kern_t = Kernel;
  using repr_t = typename kern_t::repr_type;
  using mask_t = typename kern_t::mask_type;
  using mask_repr_t = typename mask_t::repr_type;

  alignas(kern_t::size) state_t s[kern_t::size];

  for (int_t i{}; i < num_iter; ++i) {
    std::generate(std::begin(s), std::end(s), state_gen<kern_t>);

    repr_t k{kern_t::load(s)};
    mask_repr_t m{kern_t::mask_hash(k)};

    for (int_t j{}; j < kern_t::size; ++j) {
      EXPECT_EQ(mask_t::at(m, j), is_hash(s[j]));
    }
  }
}

template <typename Kernel>
void test_kernel_mask_not_hash(const int_t num_iter = 1024) {
  using kern_t = Kernel;
  using repr_t = typename kern_t::repr_type;
  using mask_t = typename kern_t::mask_type;
  using mask_repr_t = typename mask_t::repr_type;

  alignas(kern_t::size) state_t s[kern_t::size];

  for (int_t i{}; i < num_iter; ++i) {
    std::generate(std::begin(s), std::end(s), state_gen<kern_t>);

    repr_t k{kern_t::load(s)};
    mask_repr_t m{kern_t::mask_not_hash(k)};

    for (int_t j{}; j < kern_t::size; ++j) {
      EXPECT_EQ(mask_t::at(m, j), is_not_hash(s[j]));
    }
  }
}

template <typename Kernel>
void test_kernel_mask_empty(const int_t num_iter = 1024) {
  using kern_t = Kernel;
  using repr_t = typename kern_t::repr_type;
  using mask_t = typename kern_t::mask_type;
  using mask_repr_t = typename mask_t::repr_type;

  repr_t k;
  mask_repr_t m;
  alignas(kern_t::size) state_t s[kern_t::size];

  for (int_t i{}; i < num_iter; ++i) {
    std::generate(std::begin(s), std::end(s), state_gen<kern_t>);

    k = kern_t::load(s);

    m = kern_t::mask_empty(k);
    for (int_t j{}; j < kern_t::size; ++j) {
      EXPECT_EQ(mask_t::at(m, j), s[j] == kern_t::empty);
    }

    m = kern_t::mask_not_empty(k);
    for (int_t j{}; j < kern_t::size; ++j) {
      EXPECT_EQ(mask_t::at(m, j), s[j] != kern_t::empty);
    }
  }
}

template <typename Kernel>
void test_kernel_mask_tombstone(const int_t num_iter = 1024) {
  using kern_t = Kernel;
  using repr_t = typename kern_t::repr_type;
  using mask_t = typename kern_t::mask_type;
  using mask_repr_t = typename mask_t::repr_type;

  repr_t k;
  mask_repr_t m;
  alignas(kern_t::size) state_t s[kern_t::size];

  for (int_t i{}; i < num_iter; ++i) {
    std::generate(std::begin(s), std::end(s), state_gen<kern_t>);

    k = kern_t::load(s);

    m = kern_t::mask_tombstone(k);
    for (int_t j{}; j < kern_t::size; ++j) {
      EXPECT_EQ(mask_t::at(m, j), s[j] == kern_t::tombstone);
    }

    m = kern_t::mask_not_tombstone(k);
    for (int_t j{}; j < kern_t::size; ++j) {
      EXPECT_EQ(mask_t::at(m, j), s[j] != kern_t::tombstone);
    }
  }
}

template <typename Kernel>
void test_kernel_mask_count_equal(const int_t num_iter = 1024) {
  using kern_t = Kernel;
  using repr_t = typename kern_t::repr_type;
  using mask_t = typename kern_t::mask_type;
  using mask_repr_t = typename mask_t::repr_type;

  alignas(kern_t::size) state_t s0[kern_t::size];
  alignas(kern_t::size) state_t s1[kern_t::size];

  repr_t k0, k1;
  mask_repr_t m;

  for (int_t i{}; i < num_iter; ++i) {
    std::generate(std::begin(s0), std::end(s0), state_gen<kern_t>);
    std::copy(std::begin(s0), std::end(s0), std::begin(s1));
    
    k0 = kern_t::load(s0);
    m = kern_t::mask_equal(k0, k0);
    EXPECT_EQ(mask_t::count(m), kern_t::size);
    EXPECT_EQ(kern_t::count_equal(k0, k0), kern_t::size);

    k1 = kern_t::load(s1);
    m = kern_t::mask_equal(k0, k1);
    EXPECT_EQ(mask_t::count(m), kern_t::size);
    EXPECT_EQ(kern_t::count_equal(k0, k1), kern_t::size);

    for (int j{}; j < kern_t::size; ++j) {
      ++s1[j];
      k1 = kern_t::load(s1);
      m = kern_t::mask_equal(k0, k1);
      EXPECT_EQ(mask_t::count(m), kern_t::size - 1);
      EXPECT_FALSE(mask_t::at(m, j));
      EXPECT_EQ(kern_t::count_equal(k0, k1), kern_t::size - 1);
      --s1[j];
    }

    std::generate(std::begin(s1), std::end(s1), state_gen<kern_t>);
    for (int j{}; j < kern_t::size; ++j) {
      k1 = kern_t::load(s1);
      m = kern_t::mask_equal(k0, k1);

      for (int k{}; k < kern_t::size; ++k) {
        EXPECT_EQ(mask_t::at(m, k), s0[k] == s1[k]);
      }

      int_t n{};
      for (int k{}; k < kern_t::size; ++k) {
        n += s0[k] == s1[k];
      }
      EXPECT_EQ(kern_t::count_equal(k0, k1), n);
    }
  }
}

template <typename Kernel>
void test_kernel_mask_count_not_equal(const int_t num_iter = 1024) {
  using kern_t = Kernel;
  using repr_t = typename kern_t::repr_type;
  using mask_t = typename kern_t::mask_type;
  using mask_repr_t = typename mask_t::repr_type;

  alignas(kern_t::size) state_t s0[kern_t::size];
  alignas(kern_t::size) state_t s1[kern_t::size];

  repr_t k0, k1;
  mask_repr_t m;

  for (int_t i{}; i < num_iter; ++i) {
    std::generate(std::begin(s0), std::end(s0), state_gen<kern_t>);
    std::copy(std::begin(s0), std::end(s0), std::begin(s1));
    
    k0 = kern_t::load(s0);
    m = kern_t::mask_not_equal(k0, k0);
    EXPECT_FALSE(mask_t::has_next(m));
    EXPECT_EQ(kern_t::count_not_equal(k0, k0), 0);

    k1 = kern_t::load(s1);
    m = kern_t::mask_not_equal(k0, k1);
    EXPECT_FALSE(mask_t::has_next(m));
    EXPECT_EQ(kern_t::count_not_equal(k0, k1), 0);

    for (int j{}; j < kern_t::size; ++j) {
      ++s1[j];
      k1 = kern_t::load(s1);
      m = kern_t::mask_not_equal(k0, k1);
      EXPECT_TRUE(mask_t::has_next(m));
      EXPECT_EQ(mask_t::next(m), j);
      EXPECT_EQ(kern_t::count_not_equal(k0, k1), 1);
      --s1[j];
    }

    std::generate(std::begin(s1), std::end(s1), state_gen<kern_t>);
    for (int j{}; j < kern_t::size; ++j) {
      k1 = kern_t::load(s1);
      m = kern_t::mask_not_equal(k0, k1);

      for (int k{}; k < kern_t::size; ++k) {
        EXPECT_EQ(mask_t::at(m, k), s0[k] != s1[k]);
      }

      int_t n{};
      for (int k{}; k < kern_t::size; ++k) {
        n += s0[k] != s1[k];
      }
      EXPECT_EQ(kern_t::count_not_equal(k0, k1), n);
    }
  }
}

template <typename Kernel>
void test_kernel_has_hash(const int_t num_iter = 1024) {
  using kern_t = Kernel;
  using repr_t = typename kern_t::repr_type;

  alignas(kern_t::size) state_t s[kern_t::size];

  for (int_t i{}; i < num_iter; ++i) {
    std::generate(std::begin(s), std::end(s), state_gen<kern_t>);

    bool b{std::any_of(std::begin(s), std::end(s), is_hash)};
    repr_t k{kern_t::load(s)};

    EXPECT_EQ(kern_t::has_hash(k), b);
  }
}

template <typename Kernel>
void test_kernel_has_not_hash(const int_t num_iter = 1024) {
  using kern_t = Kernel;
  using repr_t = typename kern_t::repr_type;

  alignas(kern_t::size) state_t s[kern_t::size];

  for (int_t i{}; i < num_iter; ++i) {
    std::generate(std::begin(s), std::end(s), state_gen<kern_t>);

    bool b{std::any_of(std::begin(s), std::end(s), is_not_hash)};
    repr_t k{kern_t::load(s)};

    EXPECT_EQ(kern_t::has_not_hash(k), b);
  }
}

template <typename Kernel>
void test_kernel_has_empty(const int_t num_iter = 1024) {
  using kern_t = Kernel;
  using repr_t = typename kern_t::repr_type;

  bool b;
  repr_t k;
  alignas(kern_t::size) state_t s[kern_t::size];

  for (int_t i{}; i < num_iter; ++i) {
    std::generate(std::begin(s), std::end(s), state_gen<kern_t>);

    k = kern_t::load(s);

    b = std::find(std::begin(s), std::end(s), kern_t::empty) != std::end(s);
    EXPECT_EQ(kern_t::has_empty(k), b);

    b = std::find_if(std::begin(s), std::end(s), [](state_t s) { return s != kern_t::empty; }) != std::end(s);
    EXPECT_EQ(kern_t::has_not_empty(k), b);
  }
}

template <typename Kernel>
void test_kernel_has_tombstone(const int_t num_iter = 1024) {
  using kern_t = Kernel;
  using repr_t = typename kern_t::repr_type;

  bool b;
  repr_t k;
  alignas(kern_t::size) state_t s[kern_t::size];

  for (int_t i{}; i < num_iter; ++i) {
    std::generate(std::begin(s), std::end(s), state_gen<kern_t>);

    k = kern_t::load(s);

    b = std::find(std::begin(s), std::end(s), kern_t::tombstone) != std::end(s);
    EXPECT_EQ(kern_t::has_tombstone(k), b);
    
    b = std::find_if(std::begin(s), std::end(s), [](state_t s) { return s != kern_t::tombstone; }) != std::end(s);
    EXPECT_EQ(kern_t::has_not_tombstone(k), b);
  }
}

template <typename Kernel>
void test_kernel_hash_to_tombstone(const int_t num_iter = 1024) {
  using kern_t = Kernel;
  using repr_t = typename kern_t::repr_type;

  alignas(kern_t::size) state_t s[kern_t::size];

  for (int_t i{}; i < num_iter; ++i) {
    std::generate(std::begin(s), std::end(s), state_gen<kern_t>);

    repr_t k{kern_t::load(s)};
    k = kern_t::hash_to_tombstone(k);

    for (int_t j{}; j < kern_t::size; ++j) {
      EXPECT_EQ(kern_t::at(k, j), s[j] == kern_t::empty ? kern_t::empty : kern_t::tombstone);
    }
  }
}

template <typename Kernel>
void test_kernel_not_hash_to_empty(const int_t num_iter = 1024) {
  using kern_t = Kernel;
  using repr_t = typename kern_t::repr_type;

  alignas(kern_t::size) state_t s[kern_t::size];

  for (int_t i{}; i < num_iter; ++i) {
    std::generate(std::begin(s), std::end(s), state_gen<kern_t>);

    repr_t k{kern_t::load(s)};
    k = kern_t::not_hash_to_empty(k);

    for (int_t j{}; j < kern_t::size; ++j) {
      EXPECT_EQ(kern_t::at(k, j), is_hash(s[j]) ? s[j] : kern_t::empty);
    }
  }
}

template <typename Kernel>
const static std::function<int_t()> size_gen{[]() {
  using kern_t = Kernel;
  static std::uniform_int_distribution<int_t> dist(0, kern_t::size);
  return dist(rng);
}};

template <typename Kernel>
const static std::function<int_t()> idx_gen{[]() {
  using kern_t = Kernel;
  static std::uniform_int_distribution<int_t> dist(0, kern_t::size - 1);
  return dist(rng);
}};

template <typename Kernel>
void test_kernel_to_empty(const int_t num_iter = 1024) {
  using kern_t = Kernel;
  using repr_t = typename kern_t::repr_type;
  using mask_t = typename kern_t::mask_type;
  using mask_repr_t = typename mask_t::repr_type;

  alignas(kern_t::size) state_t s0[kern_t::size];
  alignas(kern_t::size) state_t s1[kern_t::size];
  repr_t k;
  mask_repr_t m;

  for (int_t i{}; i < num_iter; ++i) {
    std::generate(std::begin(s0), std::end(s0), state_gen<kern_t>);

    k = kern_t::load(s0);
    k = kern_t::to_empty(k, mask_t::empty());
    kern_t::store(k, s1);
    for (int_t j{}; j < kern_t::size; ++j) {
      EXPECT_EQ(s0[j], s1[j]);
    }

    k = kern_t::load(s0);
    k = kern_t::to_empty(k, mask_t::full());
    kern_t::store(k, s1);
    for (state_t sj : s1) {
      EXPECT_EQ(sj, kern_t::empty);
    }

    m = mask_t::empty();
    const int_t n{size_gen<kern_t>()};
    for (int_t j{}; j < n; ++j) {
      m = mask_t::join(m, mask_t::single(idx_gen<kern_t>()));
    }

    k = kern_t::load(s0);
    k = kern_t::to_empty(k, m);
    kern_t::store(k, s1);
    for (int_t j{}; j < kern_t::size; ++j) {
      EXPECT_EQ(s1[j], mask_t::at(m, j) ? kern_t::empty : s0[j]);
    }
  }
}

template <typename Kernel>
const static std::function<lru_t()> lru_gen{[]() {
  static std::uniform_int_distribution<lru_t> dist(0, max_lru);
  return dist(rng);
}};

template <typename Kernel>
void test_kernel_load_store_lru(const int_t num_iter = 1024) {
  using kern_t = Kernel;
  using lru_repr_t = typename kern_t::lru_repr_type;

  alignas(kern_t::size) lru_t n0[kern_t::size];
  alignas(kern_t::size) lru_t n1[kern_t::size];

  for (int_t i{}; i < num_iter; ++i) {
    std::generate(std::begin(n0), std::end(n0), lru_gen<kern_t>);

    lru_repr_t l{kern_t::load_lru(n0)};
    kern_t::store_lru(l, n1);

    for (int_t j{}; j < kern_t::size; ++j) {
      EXPECT_EQ(n0[j], n1[j]);
    }
  }
}

template <typename Kernel>
void test_kernel_lru_at(const int_t num_iter = 1024) {
  using kern_t = Kernel;
  using lru_repr_t = typename kern_t::lru_repr_type;

  alignas(kern_t::size) lru_t n[kern_t::size];

  for (int_t i{}; i < num_iter; ++i) {
    std::generate(std::begin(n), std::end(n), lru_gen<kern_t>);

    lru_repr_t l{kern_t::load_lru(n)};

    for (int_t j{}; j < kern_t::size; ++j) {
      EXPECT_EQ(kern_t::lru_at(l, j), n[j]);
    }
  }
}

template <typename Kernel>
void test_kernel_min_lru(const int_t num_iter = 1024) {
  using kern_t = Kernel;
  using lru_repr_t = typename kern_t::lru_repr_type;

  alignas(kern_t::size) lru_t n[kern_t::size];
  lru_t r_min, l_min;
  int_t r_idx, l_idx;
  lru_repr_t l;

  for (int_t i{}; i < num_iter; ++i) {
    std::generate(std::begin(n), std::end(n), lru_gen<kern_t>);
    r_min = *std::min_element(std::begin(n), std::end(n));
    r_idx = std::find(std::begin(n), std::end(n), r_min) - std::begin(n);

    l = kern_t::load_lru(n);
    std::tie(l_min, l_idx) = kern_t::min_lru(l);

    EXPECT_EQ(l_min, r_min);
    EXPECT_EQ(l_idx, r_idx);
  }
}

template <typename Kernel>
void test_kernel_mask_min_lru(const int_t num_iter = 1024) {
  using kern_t = Kernel;
  using lru_repr_t = typename kern_t::lru_repr_type;
  using mask_t = typename kern_t::mask_type;
  using mask_repr_t = typename mask_t::repr_type;

  alignas(kern_t::size) lru_t n[kern_t::size];
  lru_t n_min;
  lru_repr_t l;
  mask_repr_t m;

  for (int_t i{}; i < num_iter; ++i) {
    std::generate(std::begin(n), std::end(n), lru_gen<kern_t>);
    n_min = *std::min_element(std::begin(n), std::end(n));

    l = kern_t::load_lru(n);
    m = kern_t::mask_min_lru(l);

    for (int_t j{}; j < kern_t::size; ++j) {
      EXPECT_EQ(mask_t::at(m, j), n[j] == n_min);
      if (n[j] == n_min) break;
    }
  }
}

template <typename Kernel>
void test_kernel_mask_lru_below(const int_t num_iter = 1024) {
  using kern_t = Kernel;
  using lru_repr_t = typename kern_t::lru_repr_type;
  using mask_t = typename kern_t::mask_type;
  using mask_repr_t = typename mask_t::repr_type;

  alignas(kern_t::size) lru_t n[kern_t::size];
  lru_repr_t l;
  mask_repr_t m;

  for (int_t i{}; i < num_iter; ++i) {
    std::generate(std::begin(n), std::end(n), lru_gen<kern_t>);
    
    l = kern_t::load_lru(n);
    m = kern_t::mask_lru_below(l, 0);
    EXPECT_FALSE(mask_t::has_next(m));

    for (lru_t t{}; t < max_lru; ++t) {
      m = kern_t::mask_lru_below(l, t + 1);
      for (int_t j{}; j < kern_t::size; ++j) {
        EXPECT_EQ(mask_t::at(m, j), n[j] <= t);
      }
    }
  }
}

template <typename Kernel>
void test_kernel_update_lru(const int_t num_iter = 256, const int_t num_iter2 = 256) {
  using kern_t = Kernel;
  using lru_repr_t = typename kern_t::lru_repr_type;
  using mask_t = typename kern_t::mask_type;
  using mask_repr_t = typename mask_t::repr_type;
  
  alignas(kern_t::size) lru_t n0[kern_t::size];
  alignas(kern_t::size) lru_t n1[kern_t::size];

  // Test all same
  for (uint_t lru{}; lru <= max_lru; ++lru) {
    for (int_t j{}; j < kern_t::size; ++j) {
      std::fill(std::begin(n0), std::end(n0), lru);

      lru_repr_t l{kern_t::load_lru(n0)};

      for (int_t i{}; i < num_iter; ++i) {
        if (n0[j] == max_lru) {
          for (lru_t& nj : n0) nj /= 2;
        }
        ++n0[j];
      
        l = kern_t::update_lru(l, mask_t::single(j));
        kern_t::store_lru(l, n1);

        for (int_t j{}; j < kern_t::size; ++j) {
          EXPECT_EQ(n0[j], n1[j]);
        }
      }
    }
  }

  // Test random permutations.
  for (int_t i{}; i < num_iter; ++i) {
    std::generate(std::begin(n0), std::end(n0), lru_gen<kern_t>);
    
    lru_repr_t l{kern_t::load_lru(n0)};

    for (int_t i2{}; i2 < num_iter2; ++i2) {
      int_t j{idx_gen<kern_t>()};
      if (n0[j] == max_lru) {
        for (lru_t& nj : n0) nj /= 2;
      }
      ++n0[j];

      mask_repr_t m{mask_t::single(j)};
      l = kern_t::update_lru(l, m);
      kern_t::store_lru(l, n1);

      for (int_t j{}; j < kern_t::size; ++j) {
        EXPECT_EQ(n0[j], n1[j]);
      }
    }
  }
}

template <typename Kernel>
void test_kernel_to_empty_if_lru_below(const int_t num_iter = 1024) {
  using kern_t = Kernel;
  using repr_t = typename kern_t::repr_type;
  using lru_repr_t = typename kern_t::lru_repr_type;

  alignas(kern_t::size) state_t s0[kern_t::size];
  alignas(kern_t::size) state_t s1[kern_t::size];
  repr_t k;
  alignas(kern_t::size) lru_t n[kern_t::size];
  lru_repr_t l;

  for (int_t i{}; i < num_iter; ++i) {
    std::generate(std::begin(s0), std::end(s0), state_gen<kern_t>);
    std::generate(std::begin(n), std::end(n), lru_gen<kern_t>);

    k = kern_t::load(s0);
    l = kern_t::load_lru(n);
    k = kern_t::to_empty_if_lru_below(k, l, 0);
    kern_t::store(k, s1);
    for (int_t j{}; j < kern_t::size; ++j) {
      EXPECT_EQ(s0[j], s1[j]);
    }

    for (lru_t t{}; t < max_lru; ++t) {
      k = kern_t::load(s0);
      l = kern_t::load_lru(n);
      k = kern_t::to_empty_if_lru_below(k, l, t + 1);
      kern_t::store(k, s1);
      for (int_t j{}; j < kern_t::size; ++j) {
        EXPECT_EQ(s1[j], n[j] <= t ? kern_t::empty : s0[j]);
      }
    }
  }
}

#define EVAL_KERNELS_(_X_) \
  TEST(test_kernel_load_store, _X_) { test_kernel_load_store<_X_>(); } \
  TEST(test_kernel_at, _X_) { test_kernel_at<_X_>(); } \
  TEST(test_kernel_mask, _X_) { test_kernel_mask<_X_>(); } \
  TEST(test_kernel_mask_hash, _X_) { test_kernel_mask_hash<_X_>(); } \
  TEST(test_kernel_mask_not_hash, _X_) { test_kernel_mask_not_hash<_X_>(); } \
  TEST(test_kernel_mask_empty, _X_) { test_kernel_mask_empty<_X_>(); } \
  TEST(test_kernel_mask_tombstone, _X_) { test_kernel_mask_tombstone<_X_>(); } \
  TEST(test_kernel_mask_count_equal, _X_) { test_kernel_mask_count_equal<_X_>(); } \
  TEST(test_kernel_mask_count_not_equal, _X_) { test_kernel_mask_count_not_equal<_X_>(); } \
  TEST(test_kernel_has_hash, _X_) { test_kernel_has_hash<_X_>(); } \
  TEST(test_kernel_has_not_hash, _X_) { test_kernel_has_not_hash<_X_>(); } \
  TEST(test_kernel_has_empty, _X_) { test_kernel_has_empty<_X_>(); } \
  TEST(test_kernel_has_tombstone, _X_) { test_kernel_has_tombstone<_X_>(); } \
  TEST(test_kernel_hash_to_tombstone, _X_) { test_kernel_hash_to_tombstone<_X_>(); } \
  TEST(test_kernel_not_hash_to_empty, _X_) { test_kernel_not_hash_to_empty<_X_>(); } \
  TEST(test_kernel_to_empty, _X_) { test_kernel_to_empty<_X_>(); } \
  TEST(test_kernel_load_store_lru, _X_) { test_kernel_load_store_lru<_X_>(); } \
  TEST(test_kernel_lru_at, _X_) { test_kernel_lru_at<_X_>(); } \
  TEST(test_kernel_min_lru, _X_) { test_kernel_min_lru<_X_>(); } \
  TEST(test_kernel_mask_min_lru, _X_) { test_kernel_mask_min_lru<_X_>(); } \
  TEST(test_kernel_mask_lru_below, _X_) { test_kernel_mask_lru_below<_X_>(); } \
  TEST(test_kernel_update_lru, _X_) { test_kernel_update_lru<_X_>(); } \
  TEST(test_kernel_to_empty_if_lru_below, _X_) { test_kernel_to_empty_if_lru_below<_X_>(); }

// clang-format off
NVHM_FOR_EACH_(
  EVAL_KERNELS_,
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
  EVAL_KERNELS_,
  uint_kernel1_t,
  uint_kernel2_t,
  uint_kernel4_t,
  uint_kernel8_t,
  uint_kernel16_t
)

NVHM_FOR_EACH_(
  EVAL_KERNELS_,
  fast_uint_kernel1_t,
  fast_uint_kernel2_t,
  fast_uint_kernel4_t,
  fast_uint_kernel8_t,
  fast_uint_kernel16_t
)

#if NVHM_WITH_SSE >= 2
NVHM_FOR_EACH_(EVAL_KERNELS_, sse_kernel_t)
#endif

#if NVHM_WITH_AVX >= 2
NVHM_FOR_EACH_(EVAL_KERNELS_, avx_kernel_t)
#endif

#if NVHM_WITH_AVX512
NVHM_FOR_EACH_(EVAL_KERNELS_, avx512_kernel_t)
#endif

#if NVHM_WITH_NEON
NVHM_FOR_EACH_(
  EVAL_KERNELS_,
  neon_kernel8_t,
  neon_kernel16_t,
  neon_kernel32_t,
  neon_kernel64_t
)
#endif

#if NVHM_WITH_SVE
NVHM_FOR_EACH_(
  EVAL_KERNELS_
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