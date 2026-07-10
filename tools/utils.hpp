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
#include <vector>

template <typename Key>
inline std::vector<Key> make_keys(int_t num_keys, bool shuffle_keys, std::size_t seed) {
  using key_t = Key;

  std::vector<key_t> keys;
  keys.reserve(to_uint(num_keys));
  for (int_t i{}; i < num_keys; ++i) {
    keys.emplace_back(static_cast<key_t>(i));
  }

  if (shuffle_keys) {
    std::mt19937_64 g(seed);
    std::shuffle(keys.begin(), keys.end(), g);
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

static std::random_device rd;


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
