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

if(NVHM_HAVE_LIBSTDCXX)
  set(NVHM_DEBUG_HARDENING _GLIBCXX_ASSERTIONS)
elseif(NVHM_HAVE_LIBCXX_HARDENING)
  set(NVHM_DEBUG_HARDENING _LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_EXTENSIVE)
elseif(NVHM_HAVE_LIBCXX)
  set(NVHM_DEBUG_HARDENING _LIBCPP_ENABLE_ASSERTIONS=1)
else()
  set(NVHM_DEBUG_HARDENING "")
endif()
message(STATUS "NVHM_DEBUG_HARDENING = ${NVHM_DEBUG_HARDENING}")

if(NVHM_DEBUG_HARDENING)
  add_compile_definitions(
    $<$<CONFIG:Debug>:${NVHM_DEBUG_HARDENING}>
  )
endif()

add_subdirectory(compile)

add_executable(test_basics test_basics.cpp)
target_link_libraries(test_basics GTest::gtest_main)

add_executable(test_masks test_masks.cpp)
target_link_libraries(test_masks GTest::gtest_main)

add_executable(test_kernels test_kernels.cpp)
target_link_libraries(test_kernels GTest::gtest_main)

add_executable(test_memory test_memory.cpp)
target_link_libraries(test_memory GTest::gtest_main)

add_executable(test_mutex test_mutex.cpp)
target_link_libraries(test_mutex GTest::gtest_main)

add_executable(test_probe_seq test_probe_seq.cpp)
target_link_libraries(test_probe_seq GTest::gtest_main)

add_executable(test_maps test_maps.cpp)
target_link_libraries(test_maps GTest::gtest_main)
