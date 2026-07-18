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

#include <algorithm>
#include <charconv>
#include <random>
#include <string>
#include <map>
#include <set>
#include <vector>
#include <CLI/CLI.hpp>

template <typename T>
CLI::Validator make_set_validator(std::string name, const std::set<T>& values) {
  std::ostringstream os;

  os << name << '{';
  const char* sep{""};
  for (const auto& v : values) {
    os << sep << v;
    sep = "|";
  }
  os << '}';

  return {
    [values, name](const std::string& x) -> std::string {
      if (values.find(x) != values.end()) {
        return "";
      }
      return "Value " + x + " not in allowed for " + name + "!";
    },
    os.str()
  };
}

enum class key_source_t {
  polynomial,
  random
};

constexpr const char* to_string(key_source_t kd) {
  switch (kd) {
    case key_source_t::polynomial: return "polynomial";
    case key_source_t::random: return "random";
  }
  return "error";
}

inline std::ostream& operator<<(std::ostream& os, key_source_t kd) {
  return os << to_string(kd);
}

const std::map<std::string, key_source_t> str_to_key_source{
  {to_string(key_source_t::polynomial), key_source_t::polynomial},
  {to_string(key_source_t::random), key_source_t::random}
};

template <typename Key>
inline std::vector<Key> make_keys(int_t num_keys, key_source_t key_source, int_t key_c[3], std::mt19937_64& __restrict rng) {
  using key_t = Key;
  std::uniform_int_distribution<key_t> uniform_dist;

  std::vector<key_t> keys;
  keys.resize(to_uint(num_keys));
  for (int_t i{}; i < num_keys; ++i) {
    switch (key_source) {
      case key_source_t::polynomial:
        keys[to_uint(i)] = static_cast<key_t>(key_c[2] * i * i + key_c[1] * i + key_c[0]);
        break;
      case key_source_t::random:
        keys[to_uint(i)] = uniform_dist(rng);
        break;
      default:
        throw std::runtime_error("Unsupported `key_source`!");
    }
  }

  if (key_source != key_source_t::random) {
    std::shuffle(keys.begin(), keys.end(), rng);
  }

  return keys;
}

inline int rendered_length(int_t value, int base = 10) {
  char buf[96];
  auto [ptr, ec]{std::to_chars(std::begin(buf), std::end(buf), value, base)};
  if (ec != std::errc()) {
    throw std::runtime_error("Cannot convert number to string!");
  }
  return static_cast<int>(ptr - buf);
}

enum class key_type_t {
  int32,
  int64
};

constexpr const char* to_string(key_type_t kt) {
  switch (kt) {
    case key_type_t::int32: return "int32";
    case key_type_t::int64: return "int64";
  }
  return "error";
}

inline std::ostream& operator<<(std::ostream& os, key_type_t kt) {
  return os << to_string(kt);
}

const std::map<std::string, key_type_t> str_to_key_type{
  {to_string(key_type_t::int32), key_type_t::int32},
  {to_string(key_type_t::int64), key_type_t::int64}
};

enum class kernel_type_t {
  #if NVHM_TOOLS_COMPILE_KERNELS >= 10
  default1, default2, default4, default8, default16, default32, default64, default128, default256, default512,
  #endif
  #if NVHM_TOOLS_COMPILE_KERNELS >= 20
  #if NVHM_WITH_SSE >= 2
  sse,
  #endif
  #if NVHM_WITH_AVX >= 2
  avx,
  #endif
  #if NVHM_WITH_AVX512
  avx512,
  #endif
  #if NVHM_WITH_NEON
  neon8, neon16, neon32, neon64,
  #endif
  #if NVHM_WITH_SVE
  #if NVHM_WITH_SVE_SIZE >= 1
  sve1,
  #endif
  #if NVHM_WITH_SVE_SIZE >= 2
  sve2,
  #endif
  #if NVHM_WITH_SVE_SIZE >= 4
  sve4,
  #endif
  #if NVHM_WITH_SVE_SIZE >= 8
  sve8,
  #endif
  #if NVHM_WITH_SVE_SIZE >= 16
  sve16,
  #endif
  #if NVHM_WITH_SVE_SIZE >= 32
  sve32,
  #endif
  #if NVHM_WITH_SVE_SIZE >= 64
  sve64,
  #endif
  #if NVHM_WITH_SVE_SIZE >= 128
  sve128,
  #endif
  #if NVHM_WITH_SVE_SIZE >= 256
  sve256,
  #endif
  #endif
  #endif
  #if NVHM_TOOLS_COMPILE_KERNELS >= 30
  uint1, uint2, uint4, uint8, uint16,
  fast_uint1, fast_uint2, fast_uint4, fast_uint8, fast_uint16,
  #endif
  #if NVHM_TOOLS_COMPILE_KERNELS >= 40
  array1, array2, array4, array8, array16, array32, array64, array128, array256, array512,
  #endif
  default_
};

const char* to_string(kernel_type_t kt) {
  switch (kt) {
    case kernel_type_t::default_: return "default";
    #if NVHM_TOOLS_COMPILE_KERNELS >= 10
    case kernel_type_t::default1: return "default1";
    case kernel_type_t::default2: return "default2";
    case kernel_type_t::default4: return "default4";
    case kernel_type_t::default8: return "default8";
    case kernel_type_t::default16: return "default16";
    case kernel_type_t::default32: return "default32";
    case kernel_type_t::default64: return "default64";
    case kernel_type_t::default128: return "default128";
    case kernel_type_t::default256: return "default256";
    case kernel_type_t::default512: return "default512";
    #endif
    #if NVHM_TOOLS_COMPILE_KERNELS >= 20
    #if NVHM_WITH_SSE >= 2
    case kernel_type_t::sse: return "sse";
    #endif
    #if NVHM_WITH_AVX >= 2
    case kernel_type_t::avx: return "avx";
    #endif
    #if NVHM_WITH_AVX512
    case kernel_type_t::avx512: return "avx512";
    #endif
    #if NVHM_WITH_NEON
    case kernel_type_t::neon8: return "neon8";
    case kernel_type_t::neon16: return "neon16";
    case kernel_type_t::neon32: return "neon32";
    case kernel_type_t::neon64: return "neon64";
    #endif
    #if NVHM_WITH_SVE
    #if NVHM_WITH_SVE_SIZE >= 1
    case kernel_type_t::sve1: return "sve1";
    #endif
    #if NVHM_WITH_SVE_SIZE >= 2
    case kernel_type_t::sve2: return "sve2";
    #endif
    #if NVHM_WITH_SVE_SIZE >= 4
    case kernel_type_t::sve4: return "sve4";
    #endif
    #if NVHM_WITH_SVE_SIZE >= 8
    case kernel_type_t::sve8: return "sve8";
    #endif
    #if NVHM_WITH_SVE_SIZE >= 16
    case kernel_type_t::sve16: return "sve16";
    #endif
    #if NVHM_WITH_SVE_SIZE >= 32
    case kernel_type_t::sve32: return "sve32";
    #endif
    #if NVHM_WITH_SVE_SIZE >= 64
    case kernel_type_t::sve64: return "sve64";
    #endif
    #if NVHM_WITH_SVE_SIZE >= 128
    case kernel_type_t::sve128: return "sve128";
    #endif
    #if NVHM_WITH_SVE_SIZE >= 256
    case kernel_type_t::sve256: return "sve256";
    #endif
    #endif
    #endif
    #if NVHM_TOOLS_COMPILE_KERNELS >= 30
    case kernel_type_t::uint1: return "uint1";
    case kernel_type_t::uint2: return "uint2";
    case kernel_type_t::uint4: return "uint4";
    case kernel_type_t::uint8: return "uint8";
    case kernel_type_t::uint16: return "uint16";
    case kernel_type_t::fast_uint1: return "fast_uint1";
    case kernel_type_t::fast_uint2: return "fast_uint2";
    case kernel_type_t::fast_uint4: return "fast_uint4";
    case kernel_type_t::fast_uint8: return "fast_uint8";
    case kernel_type_t::fast_uint16: return "fast_uint16";
    #endif
    #if NVHM_TOOLS_COMPILE_KERNELS >= 40
    case kernel_type_t::array1: return "array1";
    case kernel_type_t::array2: return "array2";
    case kernel_type_t::array4: return "array4";
    case kernel_type_t::array8: return "array8";
    case kernel_type_t::array16: return "array16";
    case kernel_type_t::array32: return "array32";
    case kernel_type_t::array64: return "array64";
    case kernel_type_t::array128: return "array128";
    case kernel_type_t::array256: return "array256";
    case kernel_type_t::array512: return "array512";
    #endif
  };

  return "error";
};

inline std::ostream& operator<<(std::ostream& os, kernel_type_t kt) {
  return os << to_string(kt);
}

const std::map<std::string, kernel_type_t> str_to_kernel_type{
  #if NVHM_TOOLS_COMPILE_KERNELS >= 10
  {to_string(kernel_type_t::default1), kernel_type_t::default1},
  {to_string(kernel_type_t::default2), kernel_type_t::default2},
  {to_string(kernel_type_t::default4), kernel_type_t::default4},
  {to_string(kernel_type_t::default8), kernel_type_t::default8},
  {to_string(kernel_type_t::default16), kernel_type_t::default16},
  {to_string(kernel_type_t::default32), kernel_type_t::default32},
  {to_string(kernel_type_t::default64), kernel_type_t::default64},
  {to_string(kernel_type_t::default128), kernel_type_t::default128},
  {to_string(kernel_type_t::default256), kernel_type_t::default256},
  {to_string(kernel_type_t::default512), kernel_type_t::default512},
  #endif
  #if NVHM_TOOLS_COMPILE_KERNELS >= 20
  #if NVHM_WITH_SSE >= 2
  {to_string(kernel_type_t::sse), kernel_type_t::sse},
  #endif
  #if NVHM_WITH_AVX >= 2
  {to_string(kernel_type_t::avx), kernel_type_t::avx},
  #endif
  #if NVHM_WITH_AVX512
  {to_string(kernel_type_t::avx512), kernel_type_t::avx512},
  #endif
  #if NVHM_WITH_NEON
  {to_string(kernel_type_t::neon8), kernel_type_t::neon8},
  {to_string(kernel_type_t::neon16), kernel_type_t::neon16},
  {to_string(kernel_type_t::neon32), kernel_type_t::neon32},
  {to_string(kernel_type_t::neon64), kernel_type_t::neon64},
  #endif
  #if NVHM_WITH_SVE
  #if NVHM_WITH_SVE_SIZE >= 1
  {to_string(kernel_type_t::sve1), kernel_type_t::sve1},
  #endif
  #if NVHM_WITH_SVE_SIZE >= 2
  {to_string(kernel_type_t::sve2), kernel_type_t::sve2},
  #endif
  #if NVHM_WITH_SVE_SIZE >= 4
  {to_string(kernel_type_t::sve4), kernel_type_t::sve4},
  #endif
  #if NVHM_WITH_SVE_SIZE >= 8
  {to_string(kernel_type_t::sve8), kernel_type_t::sve8},
  #endif
  #if NVHM_WITH_SVE_SIZE >= 16
  {to_string(kernel_type_t::sve16), kernel_type_t::sve16},
  #endif
  #if NVHM_WITH_SVE_SIZE >= 32
  {to_string(kernel_type_t::sve32), kernel_type_t::sve32},
  #endif
  #if NVHM_WITH_SVE_SIZE >= 64
  {to_string(kernel_type_t::sve64), kernel_type_t::sve64},
  #endif
  #if NVHM_WITH_SVE_SIZE >= 128
  {to_string(kernel_type_t::sve128), kernel_type_t::sve128},
  #endif
  #if NVHM_WITH_SVE_SIZE >= 256
  {to_string(kernel_type_t::sve256), kernel_type_t::sve256},
  #endif
  #endif
  #endif
  #if NVHM_TOOLS_COMPILE_KERNELS >= 30
  {to_string(kernel_type_t::uint1), kernel_type_t::uint1},
  {to_string(kernel_type_t::uint2), kernel_type_t::uint2},
  {to_string(kernel_type_t::uint4), kernel_type_t::uint4},
  {to_string(kernel_type_t::uint8), kernel_type_t::uint8},
  {to_string(kernel_type_t::uint16), kernel_type_t::uint16},
  {to_string(kernel_type_t::fast_uint1), kernel_type_t::fast_uint1},
  {to_string(kernel_type_t::fast_uint2), kernel_type_t::fast_uint2},
  {to_string(kernel_type_t::fast_uint4), kernel_type_t::fast_uint4},
  {to_string(kernel_type_t::fast_uint8), kernel_type_t::fast_uint8},
  {to_string(kernel_type_t::fast_uint16), kernel_type_t::fast_uint16},
  #endif
  #if NVHM_TOOLS_COMPILE_KERNELS >= 40
  {to_string(kernel_type_t::array1), kernel_type_t::array1},
  {to_string(kernel_type_t::array2), kernel_type_t::array2},
  {to_string(kernel_type_t::array4), kernel_type_t::array4},
  {to_string(kernel_type_t::array8), kernel_type_t::array8},
  {to_string(kernel_type_t::array16), kernel_type_t::array16},
  {to_string(kernel_type_t::array32), kernel_type_t::array32},
  {to_string(kernel_type_t::array64), kernel_type_t::array64},
  {to_string(kernel_type_t::array128), kernel_type_t::array128},
  {to_string(kernel_type_t::array256), kernel_type_t::array256},
  {to_string(kernel_type_t::array512), kernel_type_t::array512},
  #endif
  {to_string(kernel_type_t::default_), kernel_type_t::default_}
};

enum class probe_seq_type_t {
  #if NVHM_TOOLS_COMPILE_PROBE_SEQS >= 10
  linear, quadratic,
  #endif
  #if NVHM_TOOLS_COMPILE_PROBE_SEQS >= 20
  aligned_linear, aligned_quadratic,
  #endif
  default_
};

const char* to_string(probe_seq_type_t ps) {
  switch (ps) {
    case probe_seq_type_t::default_: return "default";
    #if NVHM_TOOLS_COMPILE_PROBE_SEQS >= 10
    case probe_seq_type_t::linear: return "linear";
    case probe_seq_type_t::quadratic: return "quadratic";
    #endif
    #if NVHM_TOOLS_COMPILE_PROBE_SEQS >= 20
    case probe_seq_type_t::aligned_linear: return "aligned_linear";
    case probe_seq_type_t::aligned_quadratic: return "aligned_quadratic";
    #endif
  }
  return "error";
}

inline std::ostream& operator<<(std::ostream& os, probe_seq_type_t ps) {
  return os << to_string(ps);
}

const std::map<std::string, probe_seq_type_t> str_to_probe_seq_type{
  #if NVHM_TOOLS_COMPILE_PROBE_SEQS >= 10
  {to_string(probe_seq_type_t::linear), probe_seq_type_t::linear},
  {to_string(probe_seq_type_t::quadratic), probe_seq_type_t::quadratic},
  #endif
  #if NVHM_TOOLS_COMPILE_PROBE_SEQS >= 20
  {to_string(probe_seq_type_t::aligned_linear), probe_seq_type_t::aligned_linear},
  {to_string(probe_seq_type_t::aligned_quadratic), probe_seq_type_t::aligned_quadratic},
  #endif
  {to_string(probe_seq_type_t::default_), probe_seq_type_t::default_}
};

enum class nvhm_type_t { map, cache };

constexpr const char* to_string(nvhm_type_t nt) {
  switch (nt) {
    case nvhm_type_t::map: return "map";
    case nvhm_type_t::cache: return "cache";
  }
  return "error";
}

inline std::ostream& operator<<(std::ostream& os, nvhm_type_t nt) {
  return os << to_string(nt);
}

const std::map<std::string, nvhm_type_t> str_to_nvhm_type{
  {to_string(nvhm_type_t::map), nvhm_type_t::map},
  {to_string(nvhm_type_t::cache), nvhm_type_t::cache}
};

enum class test_trigger_t : uint_t {
  interval,
  load_perc,
};

constexpr const char* to_string(test_trigger_t tt) {
  switch (tt) {
    case test_trigger_t::interval:
      return "interval";
    case test_trigger_t::load_perc:
      return "load_perc";
  }

  return "error";
}

inline std::ostream& operator<<(std::ostream& os, test_trigger_t tt) {
  return os << to_string(tt);
}

const std::map<std::string, test_trigger_t> str_to_test_trigger{
  {to_string(test_trigger_t::interval), test_trigger_t::interval},
  {to_string(test_trigger_t::load_perc), test_trigger_t::load_perc}
};

std::random_device rd;

#include <cxxabi.h>

template <typename T>
std::string type_to_string() {
  const char* type_name{typeid(T).name()};

#if defined(__GNUC__) || defined(__clang__)
  int status{};
  std::unique_ptr<char, void(*)(void*)> ptr{
    abi::__cxa_demangle(type_name, nullptr, nullptr, &status),
    &std::free
  };
  if (status == 0 && ptr) {
    return ptr.get();
  }
#endif

  return type_name;
}
