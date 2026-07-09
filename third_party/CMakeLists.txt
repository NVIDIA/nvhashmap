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

function(add_googletest)
  set(BUILD_GMOCK OFF)
  set(INSTALL_GTEST OFF)
  add_subdirectory(googletest)
endfunction(add_googletest)

function(add_abseil)
  set(BUILD_TESTING OFF)
  set(ABSL_BUILD_TESTING OFF)
  set(ABSL_BUILD_TEST_HELPERS OFF)
  set(ABSL_USE_GOOGLETEST_HEAD OFF)
  add_subdirectory(abseil-cpp)
endfunction(add_abseil)

function(add_folly)
  set(BUILD_TESTS OFF)
  set(BUILD_BENCHMARKS OFF)
  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv9-a -mtune=neoverse-v2")
  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=native")
  set(CMAKE_CXX_FLAGS "${NVHM_CXX_BASE_FLAGS}")
  add_compile_definitions(FMT_CONSTEVAL=)
  add_subdirectory(folly)
endfunction(add_folly)


if (NVHM_BUILD_TESTS)
  add_googletest()
endif()

if (NVHM_BUILD_BENCH)
  add_abseil()
  if (DEFINED CMAKE_CXX_STANDARD)
    if (${CMAKE_CXX_STANDARD} GREATER_EQUAL 20)
      add_folly()
    endif()
  endif()
endif()