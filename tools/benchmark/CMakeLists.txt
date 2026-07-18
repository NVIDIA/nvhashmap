# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.12)

include_directories(
  ${PROJECT_SOURCE_DIR}/third_party/parallel-hashmap/parallel_hashmap
  ${PROJECT_SOURCE_DIR}/third_party/abseil-cpp
)

# `NVHM_TOOLS_COMPILE_MAP_TYPES`: Controls the amount of map type variants compiled.
#    0: Default map type only.
# >=10: Std-library type and shim.
# >=20: All platform map types.

add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark absl::hash absl::raw_hash_set)
target_compile_definitions(benchmark PRIVATE
  NVHM_TOOLS_COMPILE_KERNELS=0
  NVHM_TOOLS_COMPILE_PROBE_SEQS=0
  NVHM_TOOLS_COMPILE_MAP_TYPES=20
)
if (DEFINED CMAKE_CXX_STANDARD)
  if (${CMAKE_CXX_STANDARD} GREATER_EQUAL 20)
    target_include_directories(benchmark PRIVATE
      ${PROJECT_SOURCE_DIR}/third_party/folly
      ${CMAKE_CURRENT_BINARY_DIR}/../third_party/folly
    )
    target_link_libraries(benchmark folly)
  endif()
endif()