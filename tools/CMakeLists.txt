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
  ${PROJECT_SOURCE_DIR}/third_party/cli11/include
)

# `NVHM_TOOLS_COMPILE_KERNELS`: Controls the the amount of kernel variants compiled.
#    0: Default kernel only.
# >=10: All default kernels.
# >=20: All (available) platform kernels
# >=30: All integer kernels.
# >=40: All (fallback) array kernels.

# `NVHM_TOOLS_COMPILE_PROBE_SEQS`: Controls the amount of probe sequence variants compiled.
#    0: Default probe sequence only.
# >=10: Unaligned probe squences.
# >=20: Aligned probe sequences.

# `NVHM_TOOLS_COMPILE_MAP_TYPES`: Controls the amount of map type variants compiled.
#    0: Default map type only.
# >=10: Std-library type and shim.
# >=20: All platform map types.

add_executable(coll_count coll_count.cpp)
target_compile_definitions(coll_count PRIVATE
  NVHM_TOOLS_COMPILE_KERNELS=10
  NVHM_TOOLS_COMPILE_PROBE_SEQS=0
)

add_executable(find_psl find_psl.cpp)
target_compile_definitions(find_psl PRIVATE
  NVHM_TOOLS_COMPILE_KERNELS=10
  NVHM_TOOLS_COMPILE_PROBE_SEQS=0
)

add_executable(kernel_pop kernel_pop.cpp)
target_compile_definitions(kernel_pop PRIVATE
  NVHM_TOOLS_COMPILE_KERNELS=10
  NVHM_TOOLS_COMPILE_PROBE_SEQS=0
)

if (NVHM_BUILD_BENCH)
  add_subdirectory(benchmark)
endif()
