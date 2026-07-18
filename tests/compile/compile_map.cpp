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

#include <nvhashmap/map.hpp>
#include "compile_common.hpp"

using namespace nvhm;

template <flags_t Flags>
int compile_map() {
  using map_t = map<dummy_key, dummy_value, Flags>;
  return compile_map<false, map_t>();
}

int main() {
  int i{};
  i += compile_map<flags_t::none>();
  i += compile_map<flags_t::blobs>();
  i += compile_map<flags_t::duplicates>();
  i += compile_map<flags_t::aggressive_prefetch>();
  i += compile_map<flags_t::auto_scrub>();
  i += compile_map<flags_t::auto_shrink>();
  i += compile_map<flags_t::all>();
  return i;
}