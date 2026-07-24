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

# `NVHM_BENCH_COMPILE_KERNELS`: Controls the the amount of kernel variants compiled.
#    0: Default kernel only.
# >=10: All default kernels.
# >=20: All (available) platform kernels
# >=30: All integer kernels.
# >=40: All (fallback) array kernels.
set(NVHM_BENCH_COMPILE_KERNELS "0" CACHE STRING "Controls what kernel variants are compiled.")

# `NVHM_BENCH_COMPILE_PROBE_SEQS`: Controls the amount of probe sequence variants compiled.
#    0: Default probe sequence only.
# >=10: Unaligned probe squences.
# >=20: Aligned probe sequences.
set(NVHM_BENCH_COMPILE_PROBE_SEQS "0" CACHE STRING "Controls what probe sequence variants are compiled.")

# `NVHM_BENCH_COMPILE_MAP_TYPES`: Controls the amount of map type variants compiled.
#    0: Default map type only.
# >=10: Std-library type and shim.
# >=20: All platform map types.
set(NVHM_BENCH_COMPILE_MAP_TYPES "20" CACHE STRING "Controls what map type variants are compiled.")

message(STATUS "NVHM_BENCH_COMPILE_KERNELS    = ${NVHM_BENCH_COMPILE_KERNELS}")
message(STATUS "NVHM_BENCH_COMPILE_PROBE_SEQS = ${NVHM_BENCH_COMPILE_PROBE_SEQS}")
message(STATUS "NVHM_BENCH_COMPILE_MAP_TYPES  = ${NVHM_BENCH_COMPILE_MAP_TYPES}")

add_executable(map_benchmark map_benchmark.cpp)
target_link_libraries(map_benchmark absl::hash absl::raw_hash_set)
target_compile_definitions(map_benchmark PRIVATE "NVHM_TOOLS_COMPILE_KERNELS=${NVHM_BENCH_COMPILE_KERNELS}")
target_compile_definitions(map_benchmark PRIVATE "NVHM_TOOLS_COMPILE_PROBE_SEQS=${NVHM_BENCH_COMPILE_PROBE_SEQS}")
target_compile_definitions(map_benchmark PRIVATE "NVHM_TOOLS_COMPILE_MAP_TYPES=${NVHM_BENCH_COMPILE_MAP_TYPES}")
if (DEFINED CMAKE_CXX_STANDARD)
  if (${CMAKE_CXX_STANDARD} GREATER_EQUAL 20)
    target_include_directories(map_benchmark PRIVATE
      ${PROJECT_SOURCE_DIR}/third_party/folly
      ${CMAKE_CURRENT_BINARY_DIR}/../third_party/folly
    )
    target_link_libraries(map_benchmark folly)
  endif()
endif()

add_executable(mutex_benchmark mutex_benchmark.cpp)
target_link_libraries(mutex_benchmark)