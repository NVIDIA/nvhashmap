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

cmake_minimum_required(VERSION 3.15)

#set(CMAKE_C_COMPILER clang-20)
#set(CMAKE_CXX_COMPILER clang++-20)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(nvhashmap LANGUAGES CXX)

# Get third party modules compiled before doing anything else.
option(NVHM_BUILD_TESTS "Build the tests" ON)
option(NVHM_BUILD_BENCH "Build the benchmark" ON)
option(NVHM_BUILD_TOOLS "Build the tools" ON)
add_subdirectory(third_party)

# Default to release build.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()
message(STATUS "Build mode: ${CMAKE_BUILD_TYPE}")

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(NVHM_TARGET "-print-target-triple")
else()
  # Assume GNU.
  set(NVHM_TARGET "-print-multiarch")
endif()
execute_process(
  COMMAND "${CMAKE_CXX_COMPILER}" "${NVHM_TARGET}"
  OUTPUT_VARIABLE NVHM_TARGET
  COMMAND_ERROR_IS_FATAL ANY
)
string(REGEX MATCH "^[^-]+" NVHM_TARGET "${NVHM_TARGET}")
message(STATUS "Target architecture: ${NVHM_TARGET}")

# For testing.
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv9-a -mtune=neoverse-v2")
if("${CMAKE_CXX_FLAGS}" STREQUAL "")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=native")
else()
  message(STATUS "CMAKE_CXX_FLAGS provided as '${CMAKE_CXX_FLAGS}'. Disabling custom settings.")
endif()

# Enable warnings.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wconversion")

# Determine system cache line size.
if (${CMAKE_HOST_UNIX})
  execute_process(
    COMMAND getconf LEVEL1_DCACHE_LINESIZE
    OUTPUT_VARIABLE NVHM_HAVE_CACHE_LINE_SIZE
    COMMAND_ERROR_IS_FATAL ANY
  )
  string(STRIP "${NVHM_HAVE_CACHE_LINE_SIZE}" NVHM_HAVE_CACHE_LINE_SIZE)
  message(STATUS "Detected cache line size: ${NVHM_HAVE_CACHE_LINE_SIZE} B")

  execute_process(
    COMMAND getconf PAGE_SIZE
    OUTPUT_VARIABLE NVHM_HAVE_PAGE_SIZE
    COMMAND_ERROR_IS_FATAL ANY
  )
  string(STRIP "${NVHM_HAVE_PAGE_SIZE}" NVHM_HAVE_PAGE_SIZE)
  message(STATUS "Detected page size: ${NVHM_HAVE_PAGE_SIZE} B")

  execute_process(
    COMMAND grep "Hugepagesize:" /proc/meminfo
    OUTPUT_VARIABLE NVHM_HAVE_HUGEPAGE_SIZE
    RESULT_VARIABLE NVHM_RESULT
  )
  if(NVHM_RESULT AND NOT NVHM_RESULT EQUAL 0)
    message(WARNING "Probing `/proc/meminfo` failed!")
    set(NVHM_HAVE_HUGEPAGE_SIZE "")
  endif()
  unset(NVHM_RESULT)
  string(REGEX MATCH "[0-9]+" NVHM_HAVE_HUGEPAGE_SIZE "${NVHM_HAVE_HUGEPAGE_SIZE}")
  if(NVHM_HAVE_HUGEPAGE_SIZE STREQUAL "")
    set(NVHM_HAVE_HUGEPAGE_SIZE "2048")
  endif()
  message(STATUS "Detected hugepage size: ${NVHM_HAVE_HUGEPAGE_SIZE} kiB")

  execute_process(
    COMMAND cat /sys/kernel/mm/hugepages/hugepages-${NVHM_HAVE_HUGEPAGE_SIZE}kB/nr_hugepages
    OUTPUT_VARIABLE NVHM_HAVE_NUM_HUGEPAGES
    RESULT_VARIABLE NVHM_RESULT
  )
  if(NVHM_RESULT AND NOT NVHM_RESULT EQUAL 0)
    message(WARNING "Probing `/sys/kernel/mm/hugepages/hugepages-${NVHM_HAVE_HUGEPAGE_SIZE}kB/nr_hugepages` failed!")
    set(NVHM_HAVE_NUM_HUGEPAGES "0")
  endif()
  unset(NVHM_RESULT)
  
  string(STRIP "${NVHM_HAVE_NUM_HUGEPAGES}" NVHM_HAVE_NUM_HUGEPAGES)
  message(STATUS "Detected hugepage amount: ${NVHM_HAVE_NUM_HUGEPAGES} pages")
  math(EXPR NVHM_HAVE_HUGEPAGE_SIZE "${NVHM_HAVE_HUGEPAGE_SIZE} * 1024")
  
  execute_process(
    COMMAND cat /sys/kernel/mm/transparent_hugepage/enabled
    OUTPUT_VARIABLE NVHM_HAVE_TRANSPARENT_HUGEPAGES
    RESULT_VARIABLE NVHM_RESULT
  )
  if(NVHM_RESULT AND NOT NVHM_RESULT EQUAL 0)
    message(WARNING "Probing /sys/kernel/mm/transparent_hugepage/enabled failed!")
    set(NVHM_HAVE_TRANSPARENT_HUGEPAGES "[madvise]")
  endif()
  unset(NVHM_RESULT)
  string(STRIP "${NVHM_HAVE_TRANSPARENT_HUGEPAGES}" NVHM_HAVE_TRANSPARENT_HUGEPAGES)
  message(STATUS "Detected transparent hugepage setting: ${NVHM_HAVE_TRANSPARENT_HUGEPAGES}")
  string(REPLACE " " ";" NVHM_HAVE_TRANSPARENT_HUGEPAGES "${NVHM_HAVE_TRANSPARENT_HUGEPAGES}")
  if("[never]" IN_LIST NVHM_HAVE_TRANSPARENT_HUGEPAGES)
    set(NVHM_HAVE_TRANSPARENT_HUGEPAGES OFF)
  else()
    set(NVHM_HAVE_TRANSPARENT_HUGEPAGES ON)
  endif()
else()
  set(NVHM_HAVE_CACHE_LINE_SIZE "64")
  set(NVHM_HAVE_PAGE_SIZE "4096")
  set(NVHM_HAVE_HUGEPAGE_SIZE "2097152")
  set(NVHM_HAVE_NUM_HUGEPAGES "0")
  set(NVHM_HAVE_TRANSPARENT_HUGEPAGES ON)
endif()

# Determine CPU features.
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_FLAGS "${CMAKE_CXX_FLAGS}")
check_cxx_source_compiles("
  #if __SSE__
  int main() { return 0; }
  #else
  #error SSE is not available!
  #endif" NVHM_HAVE_SSE
)
check_cxx_source_compiles("
  #if __SSE2__
  int main() { return 0; }
  #else
  #error SSE2 is not available!
  #endif" NVHM_HAVE_SSE2
)
check_cxx_source_compiles("
  #if __SSSE3__
  int main() { return 0; }
  #else
  #error SSSE3 is not available!
  #endif" NVHM_HAVE_SSSE3
)
check_cxx_source_compiles("
  #if __SSE4_1__
  int main() { return 0; }
  #else
  #error SSE4.1 is not available!
  #endif" NVHM_HAVE_SSE4_1
)
check_cxx_source_compiles("
  #if __AVX2__
  int main() { return 0; }
  #else
  #error AVX2 is not available!
  #endif" NVHM_HAVE_AVX2
)
check_cxx_source_compiles("
  #if __AVX512F__
  int main() { return 0; }
  #else
  #error AVX-512 F is not available!
  #endif" NVHM_HAVE_AVX512_F
)
check_cxx_source_compiles("
  #if __AVX512BW__
  int main() { return 0; }
  #else
  #error AVX-512 BW is not available!
  #endif" NVHM_HAVE_AVX512_BW
)
check_cxx_source_compiles("
  #if __AVX512VL__
  int main() { return 0; }
  #else
  #error AVX-512 VL is not available!
  #endif" NVHM_HAVE_AVX512_VL
)
check_cxx_source_compiles("
  #if __AVX512VBMI__
  int main() { return 0; }
  #else
  #error AVX-512 VBMI is not available!
  #endif" NVHM_HAVE_AVX512_VBMI
)
check_cxx_source_compiles("
  #if __ARM_NEON
  int main() { return 0; }
  #else
  #error NEON is not available!
  #endif" NVHM_HAVE_NEON
)
check_cxx_source_compiles("
  #if __ARM_FEATURE_SVE
  int main() { return 0; }
  #else
  #error SVE is not available!
  #endif" NVHM_HAVE_SVE
)
if(${NVHM_HAVE_SVE})
  set(NVHM_SVE_SIZE_TOOL "${CMAKE_BINARY_DIR}/third_party/sve_num_bits")
  if(NOT EXISTS ${NVHM_SVE_SIZE_TOOL})
    execute_process(
      COMMAND echo -e "
      #include <stdio.h>
      #include <arm_sve.h>
      int main() { printf(\"%lu\", svcntb() * 8); return 0; }"
      COMMAND ${CMAKE_CXX_COMPILER} -march=native -o ${NVHM_SVE_SIZE_TOOL} -xc -
      COMMAND_ERROR_IS_FATAL ANY
    )
  endif()
  execute_process(
    COMMAND "${NVHM_SVE_SIZE_TOOL}"
    OUTPUT_VARIABLE NVHM_HAVE_SVE_MAX_BITS
    COMMAND_ERROR_IS_FATAL ANY
  )
  message(STATUS "Detected SVE register size: ${NVHM_HAVE_SVE_MAX_BITS}")
else()
  set(NVHM_HAVE_SVE_MAX_BITS "0")
endif()
set(CMAKE_REQUIRED_FLAGS "")

message(STATUS "NVHM_HAVE_CACHE_LINE_SIZE       = ${NVHM_HAVE_CACHE_LINE_SIZE}"      )
message(STATUS "NVHM_HAVE_PAGE_SIZE             = ${NVHM_HAVE_PAGE_SIZE}"            )
message(STATUS "NVHM_HAVE_HUGEPAGE_SIZE         = ${NVHM_HAVE_HUGEPAGE_SIZE}"        )
message(STATUS "NVHM_HAVE_NUM_HUGEPAGES         = ${NVHM_HAVE_NUM_HUGEPAGES}"        )
message(STATUS "NVHM_HAVE_TRANSPARENT_HUGEPAGES = ${NVHM_HAVE_TRANSPARENT_HUGEPAGES}")
message(STATUS "NVHM_HAVE_SSE                   = ${NVHM_HAVE_SSE}"                  )
message(STATUS "NVHM_HAVE_SSE2                  = ${NVHM_HAVE_SSE2}"                 )
message(STATUS "NVHM_HAVE_SSSE3                 = ${NVHM_HAVE_SSSE3}"                )
message(STATUS "NVHM_HAVE_SSE4_1                = ${NVHM_HAVE_SSE4_1}"               )
message(STATUS "NVHM_HAVE_AVX2                  = ${NVHM_HAVE_AVX2}"                 )
message(STATUS "NVHM_HAVE_AVX512_F              = ${NVHM_HAVE_AVX512_F}"             )
message(STATUS "NVHM_HAVE_AVX512_BW             = ${NVHM_HAVE_AVX512_BW}"            )
message(STATUS "NVHM_HAVE_AVX512_VL             = ${NVHM_HAVE_AVX512_VL}"            )
message(STATUS "NVHM_HAVE_AVX512_VBMI           = ${NVHM_HAVE_AVX512_VBMI}"          )
message(STATUS "NVHM_HAVE_NEON                  = ${NVHM_HAVE_NEON}"                 )
message(STATUS "NVHM_HAVE_SVE                   = ${NVHM_HAVE_SVE}"                  )
message(STATUS "NVHM_HAVE_SVE_MAX_BITS          = ${NVHM_HAVE_SVE_MAX_BITS}"         )

option(NVHM_WANT_ALL "Enable all features" OFF)
if(${NVHM_WANT_ALL})
  set(NVHM_HAVE_HUGEPAGES ON)
  set(NVHM_HAVE_SSE ON)
  set(NVHM_HAVE_SSE2 ON)
  set(NVHM_HAVE_SSSE3 ON)
  set(NVHM_HAVE_SSE4_1 ON)
  set(NVHM_HAVE_AVX2 ON)
  set(NVHM_HAVE_AVX512_F ON)
  set(NVHM_HAVE_AVX512_BW ON)
  set(NVHM_HAVE_AVX512_VL ON)
  set(NVHM_HAVE_AVX512_VBMI ON)
  set(NVHM_HAVE_NEON ON)
  set(NVHM_HAVE_SVE ON)
endif()

set(NVHM_WANT_CACHE_LINE_SIZE "${NVHM_HAVE_CACHE_LINE_SIZE}" CACHE STRING "Cache line size")
set(NVHM_WANT_PAGE_SIZE "${NVHM_HAVE_PAGE_SIZE}" CACHE STRING "Page size")
set(NVHM_WANT_HUGEPAGE_SIZE "${NVHM_HAVE_HUGEPAGE_SIZE}" CACHE STRING "Desired hugepage size")
option(NVHM_WANT_HUGEPAGES "Enable hugepages" "${NVHM_HAVE_TRANSPARENT_HUGEPAGES}")
option(NVHM_WANT_TRANSPARENT_HUGEPAGES "Prefer using transparent hugepages" "${NVHM_WANT_HUGEPAGES}")

if((${NVHM_HAVE_SSE}) OR (${NVHM_HAVE_SSE2}) OR (${NVHM_HAVE_SSSE3}) OR (${NVHM_HAVE_SSE4_1}))
  option(NVHM_WANT_SSE          "Enable SSE"          ON )
else()
  option(NVHM_WANT_SSE          "Enable SSE"          OFF)
endif()

option(NVHM_WANT_SSE_PREFETCH "Enable SSE prefetch" ${NVHM_WANT_SSE}   )

if(${NVHM_HAVE_SSE} AND ${NVHM_HAVE_SSE2})
  option(NVHM_WANT_SSE2         "Enable SSE2"         ON )
else()
  option(NVHM_WANT_SSE2         "Enable SSE2"         OFF)
endif()

option(NVHM_WANT_SSE3         "Enable SSSE3"        ${NVHM_HAVE_SSSE3} )
option(NVHM_WANT_SSE4         "Enable SSE4.1"       ${NVHM_HAVE_SSE4_1})
option(NVHM_WANT_AVX2         "Enable AVX2"         ${NVHM_HAVE_AVX2}  )

if(${NVHM_HAVE_AVX512_F} AND ${NVHM_HAVE_AVX512_BW})
  option(NVHM_WANT_AVX512       "Enable AVX512"       ON )
else()
  option(NVHM_WANT_AVX512       "Enable AVX512"       OFF)
endif()

if(${NVHM_HAVE_AVX512_F} AND ${NVHM_HAVE_AVX512_VL})
  option(NVHM_WANT_AVX_FVL      "Enable AVX FVL"      ON )
else()
  option(NVHM_WANT_AVX_FVL      "Enable AVX FVL"      OFF)
endif()

if(${NVHM_HAVE_AVX512_BW} AND ${NVHM_HAVE_AVX512_VL})
  option(NVHM_WANT_AVX_BWVL     "Enable AVX BWVL"     ON )
else()
  option(NVHM_WANT_AVX_BWVL     "Enable AVX BWVL"     OFF)
endif()

option(NVHM_WANT_AVX_VBMI     "Enable AVX VBMI"     ${NVHM_HAVE_AVX512_VBMI})
option(NVHM_WANT_NEON         "Enable NEON"         ${NVHM_HAVE_NEON}       )

if("${NVHM_HAVE_SVE_MAX_BITS}" STREQUAL "0")
  set(NVHM_HAVE_SVE_MAX_BITS "128")
endif()
set(NVHM_WANT_SVE_MAX_BITS "${NVHM_HAVE_SVE_MAX_BITS}" CACHE STRING "Enable SVE number of bits")
if((${NVHM_HAVE_SVE}) AND NOT ("${NVHM_WANT_SVE_MAX_BITS}" STREQUAL "0"))
  option(NVHM_WANT_SVE          "Enable SVE"          ON )
else()
  option(NVHM_WANT_SVE          "Enable SVE"          OFF)
endif()

if(${NVHM_WANT_SVE})
  option(NVHM_WANT_SVE_OVER_NEON "Prefer SVE over NEON" ON )
else()
  option(NVHM_WANT_SVE_OVER_NEON "Prefer SVE over NEON" OFF)
endif()

set(NVHM_WANT_PREFETCH_CACHE_LEVEL "2" CACHE STRING "Cache level to aim for when issuing prefetches")

option(NVHM_WANT_RANGE_PREFETCH "Use scoped SVE prefetches (experimental feature, not for production use)" OFF)

message(STATUS "NVHM_WANT_CACHE_LINE_SIZE       = ${NVHM_WANT_CACHE_LINE_SIZE}"      )
message(STATUS "NVHM_WANT_PAGE_SIZE             = ${NVHM_WANT_PAGE_SIZE}"            )
message(STATUS "NVHM_WANT_HUGEPAGE_SIZE         = ${NVHM_WANT_HUGEPAGE_SIZE}"        )
message(STATUS "NVHM_WANT_HUGEPAGES             = ${NVHM_WANT_HUGEPAGES}"            )
message(STATUS "NVHM_WANT_TRANSPARENT_HUGEPAGES = ${NVHM_WANT_TRANSPARENT_HUGEPAGES}")
message(STATUS "NVHM_WANT_SSE                   = ${NVHM_WANT_SSE}"                  )
message(STATUS "NVHM_WANT_SSE_PREFETCH          = ${NVHM_WANT_SSE_PREFETCH}"         )
message(STATUS "NVHM_WANT_SSE2                  = ${NVHM_WANT_SSE2}"                 )
message(STATUS "NVHM_WANT_SSE3                  = ${NVHM_WANT_SSE3}"                 )
message(STATUS "NVHM_WANT_SSE4                  = ${NVHM_WANT_SSE4}"                 )
message(STATUS "NVHM_WANT_AVX2                  = ${NVHM_WANT_AVX2}"                 )
message(STATUS "NVHM_WANT_AVX512                = ${NVHM_WANT_AVX512}"               )
message(STATUS "NVHM_WANT_AVX_FVL               = ${NVHM_WANT_AVX_FVL}"              )
message(STATUS "NVHM_WANT_AVX_BWVL              = ${NVHM_WANT_AVX_BWVL}"             )
message(STATUS "NVHM_WANT_AVX_VBMI              = ${NVHM_WANT_AVX_VBMI}"             )
message(STATUS "NVHM_WANT_NEON                  = ${NVHM_WANT_NEON}"                 )
message(STATUS "NVHM_WANT_SVE                   = ${NVHM_WANT_SVE}"                  )
message(STATUS "NVHM_WANT_SVE_MAX_BITS          = ${NVHM_WANT_SVE_MAX_BITS}"         )
message(STATUS "NVHM_WANT_SVE_OVER_NEON         = ${NVHM_WANT_SVE_OVER_NEON}"        )
message(STATUS "NVHM_WANT_RANGE_PREFETCH        = ${NVHM_WANT_RANGE_PREFETCH}"       )

if ("${NVHM_TARGET}" STREQUAL "x86_64")
  if(${NVHM_WANT_SSE})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse")
  endif()
  if(${NVHM_WANT_SSE2})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2")
  endif()
  if(${NVHM_WANT_SSE3})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mssse3")
  endif()
  if(${NVHM_WANT_SSE4})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1")
  endif()
  if(${NVHM_WANT_AVX2})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
  endif()
  if(${NVHM_WANT_AVX512} OR ${NVHM_WANT_AVX_FVL})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f")
  endif()
  if(${NVHM_WANT_AVX512} OR ${NVHM_WANT_AVX_BWVL})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512bw")
  endif()
  if(${NVHM_WANT_AVX_BWVL} OR ${NVHM_WANT_AVX_FVL})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512vl")
  endif()
  if(${NVHM_WANT_AVX_VBMI})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512vbmi")
  endif()
elseif("${NVHM_TARGET}" STREQUAL "aarch64")
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    if(${NVHM_WANT_NEON})
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xclang -target-feature -Xclang +neon")
    endif()
    if(${NVHM_WANT_SVE})
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xclang -target-feature -Xclang +sve")
    endif()
  else()
    string(REGEX MATCH "-march=([^ \t\r\n]*)" NVHM_MARCH "${CMAKE_CXX_FLAGS}")
    set(NVHM_MARCH_OLD "${NVHM_MARCH}")
    string(REGEX REPLACE "^-march=" "" NVHM_MARCH "${NVHM_MARCH}")

    if("${NVHM_MARCH}" STREQUAL "")
      set(NVHM_MARCH "armv8-a")
    endif()

    string(REGEX MATCH "^armv[0-9]+" NVHM_AARCH_MAJOR "${NVHM_MARCH}")
    string(REPLACE "armv" "" NVHM_AARCH_MAJOR "${NVHM_AARCH_MAJOR}")
    if("${NVHM_AARCH_MAJOR}" STREQUAL "")
      set(NVHM_AARCH_MAJOR "99")
    endif()
    message(STATUS "AARCH64 major version: ${NVHM_AARCH_MAJOR}")

    set(NVHM_IMPLICT_NEON OFF)
    if("${NVHM_AARCH_MAJOR}" STRGREATER "7")
      set(NVHM_IMPLICT_NEON ON)
    elseif("${NVHM_MARCH}" MATCHES "\\+simd")
      set(NVHM_IMPLICT_NEON ON)
    endif()

    set(NVHM_IMPLICT_SVE OFF)
    if("${NVHM_AARCH_MAJOR}" STRGREATER "8")
      set(NVHM_IMPLICT_SVE ON)
    elseif("${NVHM_MARCH}" MATCHES "\\+sve")
      set(NVHM_IMPLICT_SVE ON)
    endif()

    unset(NVHM_AARCH_MAJOR)

    # https://gcc.gnu.org/onlinedocs/gcc/AArch64-Options.html#aarch64-feature-modifiers
    set(NVHM_MARCH "-march=${NVHM_MARCH}")
    if((${NVHM_WANT_NEON}) AND NOT (${NVHM_IMPLICT_NEON}))
      set(NVHM_MARCH "${NVHM_MARCH}+simd")
    endif()
    if((${NVHM_WANT_SVE}) AND NOT (${NVHM_IMPLICT_SVE}))
      set(NVHM_MARCH "${NVHM_MARCH}+sve")
    endif()
    unset(NVHM_IMPLICT_NEON)
    unset(NVHM_IMPLICT_SVE)

    #message(STATUS "NVHM_MARCH: ${NVHM_MARCH_OLD} => ${NVHM_MARCH}")
    #message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
    if("${NVHM_MARCH_OLD}" STREQUAL "")
      string(CONCAT CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" " ${NVHM_MARCH}")
    else()
      string(REPLACE "${NVHM_MARCH_OLD}" "${NVHM_MARCH}" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    endif()
    #message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
    unset(NVHM_MARCH)
    unset(NVHM_MARCH_OLD)
  endif()
  if(${NVHM_WANT_SVE})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msve-vector-bits=${NVHM_WANT_SVE_MAX_BITS}")
  endif()
endif()

option(NVHM_ALLOW_FALSE_POSITIVE_MATCHES "Allows false positives in masks" ON)
option(NVHM_ALLOW_FAST_LRU_RESCALE "Allow using faster, but numerically incosistent rescale methods" ON)

# Write configuration header file.
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/include")
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/include/nvhashmap_conf.hpp" CONTENT
"/**
 * Generated during build CMake process by ${PROJECT_SOURCE_DIR}/CMakeLists.txt.
 */
#pragma once

#define NVHM_WITH_SSE           ($<BOOL:${NVHM_WANT_SSE}> && __SSE__)
#define NVHM_WITH_SSE_PREFETCH  ($<BOOL:${NVHM_WITH_SSE_PREFETCH}> && __SSE__)
#define NVHM_WITH_SSE2          ($<BOOL:${NVHM_WANT_SSE2}> && __SSE2__)
#define NVHM_WITH_SSE3          ($<BOOL:${NVHM_WANT_SSE3}> && __SSSE3__)
#define NVHM_WITH_SSE4          ($<BOOL:${NVHM_WANT_SSE4}> && __SSE4_1__)
#define NVHM_WITH_AVX2          ($<BOOL:${NVHM_WANT_AVX2}> && __AVX2__)
#define NVHM_WITH_AVX512        ($<BOOL:${NVHM_WANT_AVX512}> && __AVX512F__ && __AVX512BW__)
#define NVHM_WITH_AVX_FVL       ($<BOOL:${NVHM_WANT_AVX_FVL}> && __AVX512F__ && __AVX512VL__)
#define NVHM_WITH_AVX_BWVL      ($<BOOL:${NVHM_WANT_AVX_BWVL}> && __AVX512BW__ && __AVX512VL__)
#define NVHM_WITH_AVX_VBMI      ($<BOOL:${NVHM_WANT_AVX_VBMI}> && __AVX512VBMI__)
#define NVHM_WITH_NEON          ($<BOOL:${NVHM_WANT_NEON}> && __ARM_NEON)
#define NVHM_WITH_SVE           ($<BOOL:${NVHM_WANT_SVE}> && __ARM_FEATURE_SVE && __ARM_FEATURE_SVE_BITS >= 128)

namespace nvhm {

/**
 * In situations where both SVE and NEON are available, prefer SVE over NEON.
 */
constexpr static bool prefer_sve_over_neon{$<BOOL:${NVHM_WANT_SVE_OVER_NEON}>};

/**
 * Assumed cache line size. For peformance reasons, we allocate buffers at cache line sized
 * boundaries. Also determines the smallest size of a table.
 */
constexpr static size_t cache_line_size{${NVHM_WANT_CACHE_LINE_SIZE}};
static_assert(has_single_bit(cache_line_size));

/**
 * Page size.
 */
constexpr static size_t page_size{${NVHM_WANT_PAGE_SIZE}};
static_assert(page_size >= cache_line_size && has_single_bit(page_size));

/**
 * Hugepage size.
 */
constexpr static size_t hugepage_size{${NVHM_WANT_HUGEPAGE_SIZE}};
static_assert(hugepage_size >= hugepage_size && has_single_bit(page_size));

/**
 * Enable hugepage allocation.
 */
constexpr static bool use_hugepages{$<BOOL:${NVHM_WANT_HUGEPAGES}>};

/**
 * Allow using transparent hugepages.
 */
constexpr static bool use_transparent_hugepages{$<BOOL:${NVHM_WANT_TRANSPARENT_HUGEPAGES}>};

/**
 * Allow false mask-function to return false positives. This can make masking is faster, but may cause uncessary subsequent checks.
 */
constexpr static bool allow_false_positive_matches{$<BOOL:${NVHM_ALLOW_FALSE_POSITIVE_MATCHES}>};

/**
 * Allow LRU rescales to happen faster (but results might be not comparable between kernels).
 */
constexpr static bool allow_fast_lru_rescale{$<BOOL:${NVHM_ALLOW_FAST_LRU_RESCALE}>};

/**
 * Prefetch locality.
 */
constexpr static int prefetch_cache_level{${NVHM_WANT_PREFETCH_CACHE_LEVEL}};
static_assert(prefetch_cache_level >= 1 && prefetch_cache_level <= 5);

/**
 * Use range prefetch instructions if available (ARM64 only,experimental feature, not recommended for production use)!
 */
constexpr static bool use_range_prefetch{$<BOOL:${NVHM_WANT_RANGE_PREFETCH}>};

/**
 * Hasher to use.
 */
template <typename Key>
constexpr static std::hash<Key> hasher;

enum class hash_mixer_t {
  none,
  marsaglia
};

/**
 * Hash mixer to use.
 */
constexpr static hash_mixer_t hash_mixer{hash_mixer_t::marsaglia};

enum class hash_layout_t {
  traditional,
  monkey_wrench
};

/**
 * Hash layout to use.
 */
constexpr static hash_layout_t hash_layout{hash_layout_t::monkey_wrench};

}  // namespace nvhm")

# All done. Let's build!
include_directories(
  "${PROJECT_SOURCE_DIR}/include"
  "${CMAKE_BINARY_DIR}/include"
)

message(STATUS "CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")

if (NVHM_BUILD_TESTS)
  add_subdirectory(tests)
endif()

if (NVHM_BUILD_BENCH)
  add_subdirectory(benchmark)
endif()

if (NVHM_BUILD_TOOLS)
  add_subdirectory(tools)
endif()
