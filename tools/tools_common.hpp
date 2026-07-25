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
#include <chrono>
#include <cxxabi.h>
#include <random>
#include <string>
#include <map>
#include <set>
#include <vector>
#include <CLI/CLI.hpp>

#define NVHM_MAKE_ENUM_TO_ENUM_CLASS_(_type_name_, _value_) , _type_name_::_value_

#define NVHM_MAKE_ENUM_WITH_VALIDATOR_(_type_name_, ...)                                                       \
  NVHM_MAKE_ENUM_(_type_name_, __VA_ARGS__);                                                                   \
                                                                                                               \
  inline const auto _type_name_##_validator{                                                                   \
    make_enum_validator<0 NVHM_MAKE_ENUM_FOR_EACH_(NVHM_MAKE_ENUM_TO_ENUM_CLASS_, _type_name_, __VA_ARGS__)>() \
  }

class stopwatch {
 public:
  using time_point_type = std::chrono::steady_clock::time_point;
  using duration_type = std::chrono::steady_clock::duration;
  using tick_length = std::chrono::steady_clock::period;

  inline stopwatch() noexcept : begin_(std::chrono::steady_clock::now()) {}
  
  inline duration_type elapsed() const noexcept {
    return std::chrono::steady_clock::now() - begin_;
  }

  inline std::chrono::milliseconds elapsed_ms() const noexcept {
    return std::chrono::duration_cast<std::chrono::milliseconds>(elapsed());
  }

  inline std::chrono::seconds elapsed_s() const noexcept {
   return std::chrono::duration_cast<std::chrono::seconds>(elapsed());
  }

 private:
  time_point_type begin_;
};

template <int, auto Arg0, decltype(Arg0)... Args>
CLI::Validator make_enum_validator() {
  using underlying_t = std::underlying_type_t<decltype(Arg0)>;

  const static std::string desc{[&]() {
    std::ostringstream os;
    os << "in {" << Arg0;
    ((os << '|' << Args), ...);
    os << '}';
    return os.str();
  }()};

  const static std::map<std::string_view, underlying_t> map{
    {to_string_view(Arg0), static_cast<underlying_t>(Arg0)},
    {to_string_view(Args), static_cast<underlying_t>(Args)}...
  };

  return {
    [&](std::string& input) -> std::string {
      auto it{map.find(input)};
      if (it == map.end()) {
        std::ostringstream os;
        os << "Not a valid ENUM: " << input << " (valid values are " << Arg0;
        ((os << '|' << Args), ...);
        os << ')';
        return os.str();
      }
      input = std::to_string(it->second);
      return {};
    },
    desc, "ENUM"
  };
}

NVHM_MAKE_ENUM_WITH_VALIDATOR_(key_source_t,
  polynomial,
  random
);

template <typename Key>
inline std::vector<Key> make_keys(int_t num_keys, key_source_t key_source, const std::array<int_t, 3>& key_poly, std::mt19937_64& __restrict rng) {
  using key_t = Key;
  std::uniform_int_distribution<key_t> uniform_dist;

  std::vector<key_t> keys;
  keys.resize(to_uint(num_keys));
  for (int_t i{}; i < num_keys; ++i) {
    switch (key_source) {
      case key_source_t::polynomial:
        keys[to_uint(i)] = static_cast<key_t>(key_poly[0] + key_poly[1] * i + key_poly[2] * i * i);
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

NVHM_MAKE_ENUM_WITH_VALIDATOR_(key_type_t,
  int32,
  int64
);

NVHM_MAKE_ENUM_WITH_VALIDATOR_(kernel_type_t,
  default_
  #if NVHM_TOOLS_COMPILE_KERNELS >= 10
  , default1, default2, default4, default8, default16, default32, default64, default128, default256, default512
  #endif
  #if NVHM_TOOLS_COMPILE_KERNELS >= 20
  #if NVHM_WITH_SSE >= 2
  , sse
  #endif
  #if NVHM_WITH_AVX >= 2
  , avx
  #endif
  #if NVHM_WITH_AVX512
  , avx512
  #endif
  #if NVHM_WITH_NEON
  , neon8, neon16, neon32, neon64
  #endif
  #if NVHM_WITH_SVE
  #if NVHM_WITH_SVE_SIZE >= 1
  , sve1
  #endif
  #if NVHM_WITH_SVE_SIZE >= 2
  , sve2
  #endif
  #if NVHM_WITH_SVE_SIZE >= 4
  , sve4
  #endif
  #if NVHM_WITH_SVE_SIZE >= 8
  , sve8
  #endif
  #if NVHM_WITH_SVE_SIZE >= 16
  , sve16
  #endif
  #if NVHM_WITH_SVE_SIZE >= 32
  , sve32
  #endif
  #if NVHM_WITH_SVE_SIZE >= 64
  , sve64
  #endif
  #if NVHM_WITH_SVE_SIZE >= 128
  , sve128
  #endif
  #if NVHM_WITH_SVE_SIZE >= 256
  , sve256
  #endif
  #endif
  #endif
  #if NVHM_TOOLS_COMPILE_KERNELS >= 30
  , uint1, uint2, uint4, uint8, uint16
  , fast_uint1, fast_uint2, fast_uint4, fast_uint8, fast_uint16
  #endif
  #if NVHM_TOOLS_COMPILE_KERNELS >= 40
  , array1, array2, array4, array8, array16, array32, array64, array128, array256, array512
  #endif
);

NVHM_MAKE_ENUM_WITH_VALIDATOR_(probe_seq_type_t,
  default_
  #if NVHM_TOOLS_COMPILE_PROBE_SEQS >= 10
  , linear, quadratic
  #endif
  #if NVHM_TOOLS_COMPILE_PROBE_SEQS >= 20
  , aligned_linear, aligned_quadratic
  #endif
);

NVHM_MAKE_ENUM_WITH_VALIDATOR_(nvhm_type_t,
  map,
  cache
);

NVHM_MAKE_ENUM_WITH_VALIDATOR_(test_trigger_t,
  interval,
  load_perc
);

template <typename T>
std::string type_to_string() {
  const char* p{typeid(T).name()};

  #if defined(__GNUC__) || defined(__clang__)
  int status{};
  std::unique_ptr<char, void(*)(void*)> ptr{
    abi::__cxa_demangle(p, nullptr, nullptr, &status),
    &std::free
  };
  
  std::string s;
  if (status == 0 && ptr) {
    s = ptr.get();
  } else
  #endif  
  {
    s = p;
  }

  s.erase(std::remove_if(s.begin(), s.end(), [](unsigned char c){ return std::isspace(c); }), s.end());
  //std::replace(s.begin(), s.end(), ',', '|');
  return s;
}

template <typename... Args>
inline std::string to_string(Args&&... args) {
  std::ostringstream os;
  render_args(os, std::forward<Args>(args)...);
  return os.str();
}

inline std::random_device rd;
