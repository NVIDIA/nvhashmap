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

#set(CMAKE_C_COMPILER clang-21)
#set(CMAKE_CXX_COMPILER clang++-21)

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()
if(CMAKE_CXX_STANDARD MATCHES "^(98|11|14)$")
  message(FATAL_ERROR "nvHashMap requires at least C++ 17")
endif()
message(STATUS "Compiling for C++ ${CMAKE_CXX_STANDARD}")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(nvhashmap LANGUAGES CXX)

# Get third party modules compiled before doing anything else.
option(NVHM_BUILD_TESTS "Build the tests" ON)
option(NVHM_BUILD_BENCH "Build the benchmark" ON)
option(NVHM_BUILD_TOOLS "Build the tools" ON)

# Default to release build.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()
message(STATUS "Build flags: ${CMAKE_BUILD_TYPE}")

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

# Determine system cache line size.
set(NVHM_HAVE_CACHE_LINE_SIZE "")
if(NOT NVHM_HAVE_CACHE_LINE_SIZE MATCHES "^[1-9][0-9]*$")
  set(NVHM_TMP "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size")
  if(EXISTS "${NVHM_TMP}")
    file(READ "${NVHM_TMP}" NVHM_HAVE_CACHE_LINE_SIZE)
    string(STRIP "${NVHM_HAVE_CACHE_LINE_SIZE}" NVHM_HAVE_CACHE_LINE_SIZE)
  endif()
  unset(NVHM_TMP)
endif()
if(NOT NVHM_HAVE_CACHE_LINE_SIZE MATCHES "^[1-9][0-9]*$")
  execute_process(
    COMMAND sysctl -n hw.cachelinesize
    OUTPUT_VARIABLE NVHM_HAVE_CACHE_LINE_SIZE
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
endif()
if(NOT NVHM_HAVE_CACHE_LINE_SIZE MATCHES "^[1-9][0-9]*$")
  execute_process(
    COMMAND getconf LEVEL1_DCACHE_LINESIZE
    OUTPUT_VARIABLE NVHM_HAVE_CACHE_LINE_SIZE
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
endif()
if(NOT NVHM_HAVE_CACHE_LINE_SIZE MATCHES "^[1-9][0-9]*$")
  message(WARNING "Could not determine cache line size. Assuming 64 bytes.")
  set(NVHM_HAVE_CACHE_LINE_SIZE "64")
endif()
message(STATUS "Detected cache line size: ${NVHM_HAVE_CACHE_LINE_SIZE} B")

# Determine system memory page size.
set(NVHM_HAVE_PAGE_SIZE "")
if(NOT NVHM_HAVE_PAGE_SIZE MATCHES "^[1-9][0-9]*$")
  execute_process(
    COMMAND sysctl -n hw.pagesize
    OUTPUT_VARIABLE NVHM_HAVE_PAGE_SIZE
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
endif()
if(NOT NVHM_HAVE_PAGE_SIZE MATCHES "^[1-9][0-9]*$")
  execute_process(
    COMMAND getconf PAGE_SIZE
    OUTPUT_VARIABLE NVHM_HAVE_PAGE_SIZE
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
endif()
if(NOT NVHM_HAVE_PAGE_SIZE MATCHES "^[1-9][0-9]*$")
  execute_process(
    COMMAND getconf PAGESIZE
    OUTPUT_VARIABLE NVHM_HAVE_PAGE_SIZE
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
endif()
if(NOT NVHM_HAVE_PAGE_SIZE MATCHES "^[1-9][0-9]*$")
  message(WARNING "Could not determine page size. Assuming 4096 bytes.")
  set(NVHM_HAVE_PAGE_SIZE "4096")
endif()
message(STATUS "Detected page size: ${NVHM_HAVE_PAGE_SIZE} B")

# Determine system memory hugepage properties.
set(NVHM_HAVE_TRANSPARENT_HUGEPAGES OFF)
if(NOT NVHM_HAVE_TRANSPARENT_HUGEPAGES)
  set(NVHM_TMP "/sys/kernel/mm/transparent_hugepage/enabled")
  if(EXISTS "${NVHM_TMP}")
    file(READ "${NVHM_TMP}" NVHM_TMP)
    string(STRIP "${NVHM_TMP}" NVHM_TMP)
    message(STATUS "System's hugepage setting: ${NVHM_TMP}")
    if(NVHM_TMP MATCHES "\\[([a-z]+)\\]")
      set(NVHM_TMP "${CMAKE_MATCH_1}")
      if(NOT NVHM_TMP STREQUAL "never")
        set(NVHM_HAVE_TRANSPARENT_HUGEPAGES ON)
      endif()
    endif()
  endif()
  unset(NVHM_TMP)
endif()
message(STATUS "Detected transparent hugepages: ${NVHM_HAVE_TRANSPARENT_HUGEPAGES}")

set(NVHM_HAVE_HUGEPAGE_SIZE "")
if(NOT NVHM_HAVE_HUGEPAGE_SIZE MATCHES "^[1-9][0-9]*$")
  if(EXISTS "/proc/meminfo")
    file(READ "/proc/meminfo" NVHM_TMP)
    if(NVHM_TMP MATCHES "Hugepagesize:[ \t]*([1-9][0-9]*)[ \t]*kB")
      math(EXPR NVHM_HAVE_HUGEPAGE_SIZE "${CMAKE_MATCH_1} * 1024")
    endif()
    unset(NVHM_TMP)
  endif()
endif()
if(NOT NVHM_HAVE_HUGEPAGE_SIZE MATCHES "^[1-9][0-9]*$")
  set(NVHM_TMP "/sys/kernel/mm/transparent_hugepage/hpage_pmd_size")
  if(EXISTS "${NVHM_TMP}")
    file(READ "${NVHM_TMP}" NVHM_HAVE_HUGEPAGE_SIZE)
    string(STRIP "${NVHM_HAVE_HUGEPAGE_SIZE}" NVHM_HAVE_HUGEPAGE_SIZE)
  endif()
  unset(NVHM_TMP)
endif()
if(NOT NVHM_HAVE_HUGEPAGE_SIZE MATCHES "^[1-9][0-9]*$")
  message(WARNING "Could not determine hugepage size. Assuming 2097152 bytes.")
  set(NVHM_HAVE_HUGEPAGE_SIZE "2097152")
endif()
message(STATUS "Detected hugepage size: ${NVHM_HAVE_HUGEPAGE_SIZE} B")

set(NVHM_HAVE_NUM_HUGEPAGES "")
message(STATUS "NVHM_HAVE_HUGEPAGE_SIZE1: ${NVHM_HAVE_HUGEPAGE_SIZE}")
if(NOT NVHM_HAVE_NUM_HUGEPAGES MATCHES "^[0-9]+$")
  if(EXISTS "/proc/meminfo")
    file(READ "/proc/meminfo" NVHM_TMP)
    if(NVHM_TMP MATCHES "HugePages_Total:[ \t]*([0-9]+)")
      set(NVHM_HAVE_NUM_HUGEPAGES "${CMAKE_MATCH_1}")
    endif()
    unset(NVHM_TMP)
  endif()
endif()
message(STATUS "NVHM_HAVE_HUGEPAGE_SIZE2: ${NVHM_HAVE_HUGEPAGE_SIZE}")
if(NOT NVHM_HAVE_NUM_HUGEPAGES MATCHES "^[0-9]+$")
  math(EXPR NVHM_TMP "${NVHM_HAVE_HUGEPAGE_SIZE} / 1024")
  set(NVHM_TMP "/sys/kernel/mm/hugepages/hugepages-${NVHM_TMP}kB/nr_hugepages")
  if(EXISTS "${NVHM_TMP}")
    file(READ "${NVHM_TMP}" NVHM_HAVE_NUM_HUGEPAGES)
    string(STRIP "${NVHM_HAVE_NUM_HUGEPAGES}" NVHM_HAVE_NUM_HUGEPAGES)
  endif()
  unset(NVHM_TMP)
endif()
message(STATUS "NVHM_HAVE_HUGEPAGE_SIZE3: ${NVHM_HAVE_HUGEPAGE_SIZE}")
if(NOT NVHM_HAVE_NUM_HUGEPAGES MATCHES "^[0-9]+$")
  message(WARNING "Could not determine hugepage amount. Assuming 0 pages.")
  set(NVHM_HAVE_NUM_HUGEPAGES "0")
endif()
message(STATUS "Detected hugepage amount: ${NVHM_HAVE_NUM_HUGEPAGES} pages")

# Determine CPU features.
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_FLAGS "${CMAKE_CXX_FLAGS}")

check_cxx_source_compiles("
  #include <cstddef>
  #if defined(__GLIBCXX__)
  int main() { return 0; }
  #else
  #error Not libstdc++!
  #endif" NVHM_HAVE_LIBSTDCXX
)
check_cxx_source_compiles("
  #include <cstddef>
  #if defined(_LIBCPP_VERSION)
  int main() { return 0; }
  #else
  #error Not libc++!
  #endif" NVHM_HAVE_LIBCXX
)
check_cxx_source_compiles("
  #include <cstddef>
  #if defined(_LIBCPP_HARDENING_MODE_EXTENSIVE)
  int main() { return 0; }
  #else
  #error libc++ hardening modes not available!
  #endif" NVHM_HAVE_LIBCXX_HARDENING
)

check_cxx_source_compiles("
  #include <sys/mman.h>
  #if defined(MADV_HUGEPAGE)
  int main() { return 0; }
  #else
  #error madvise hugepages are not available!
  #endif" NVHM_HAVE_MADVISE_HUGEPAGE
)
check_cxx_source_compiles("
  #include <sys/mman.h>
  #if defined(MAP_HUGETLB) && defined(MAP_HUGE_SHIFT)
  int main() { return 0; }
  #else
  #error Hugetable mmap flags are not available!
  #endif" NVHM_HAVE_MMAP_HUGETLB
)

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
  #if __AVX__
  int main() { return 0; }
  #else
  #error AVX is not available!
  #endif" NVHM_HAVE_AVX
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
check_cxx_source_compiles("
  #if __ARM_FEATURE_SVE2
  int main() { return 0; }
  #else
  #error SVE2 is not available!
  #endif" NVHM_HAVE_SVE2
)
if(${NVHM_HAVE_SVE})
  set(NVHM_SVE_SIZE_TOOL "${CMAKE_BINARY_DIR}/sve_num_bits")
  if(NOT EXISTS ${NVHM_SVE_SIZE_TOOL})
    message(STATUS "Building SVE size tool...")
    execute_process(
      COMMAND echo -e "
      #include <stdio.h>
      #include <arm_sve.h>
      int main() { printf(\"%lu\", svcntb()); return 0; }"
      COMMAND ${CMAKE_CXX_COMPILER} -march=native -o ${NVHM_SVE_SIZE_TOOL} -xc -
      COMMAND_ERROR_IS_FATAL ANY
    )
  endif()
  execute_process(
    COMMAND "${NVHM_SVE_SIZE_TOOL}"
    OUTPUT_VARIABLE NVHM_HAVE_SVE_SIZE
    COMMAND_ERROR_IS_FATAL ANY
  )
  message(STATUS "Detected SVE register size: ${NVHM_HAVE_SVE_SIZE}")
  if(("${NVHM_HAVE_SVE_SIZE}" STREQUAL "0") OR ("${NVHM_HAVE_SVE_SIZE}" STREQUAL ""))
    set(NVHM_HAVE_SVE_SIZE "128")
  endif()
else()
  set(NVHM_HAVE_SVE_SIZE "128")
endif()
set(CMAKE_REQUIRED_FLAGS "")

message(STATUS "NVHM_HAVE_CACHE_LINE_SIZE       = ${NVHM_HAVE_CACHE_LINE_SIZE}"      )
message(STATUS "NVHM_HAVE_PAGE_SIZE             = ${NVHM_HAVE_PAGE_SIZE}"            )
message(STATUS "NVHM_HAVE_TRANSPARENT_HUGEPAGES = ${NVHM_HAVE_TRANSPARENT_HUGEPAGES}")
message(STATUS "NVHM_HAVE_HUGEPAGE_SIZE         = ${NVHM_HAVE_HUGEPAGE_SIZE}"        )
message(STATUS "NVHM_HAVE_NUM_HUGEPAGES         = ${NVHM_HAVE_NUM_HUGEPAGES}"        )
message(STATUS "NVHM_HAVE_LIBSTDCXX             = ${NVHM_HAVE_LIBSTDCXX}"            )
message(STATUS "NVHM_HAVE_LIBCXX                = ${NVHM_HAVE_LIBCXX}"               )
message(STATUS "NVHM_HAVE_LIBCXX_HARDENING      = ${NVHM_HAVE_LIBCXX_HARDENING}"     )
message(STATUS "NVHM_HAVE_MADVISE_HUGEPAGE      = ${NVHM_HAVE_MADVISE_HUGEPAGE}"     )
message(STATUS "NVHM_HAVE_MMAP_HUGETLB          = ${NVHM_HAVE_MMAP_HUGETLB}"         )
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
message(STATUS "NVHM_HAVE_SVE_SIZE              = ${NVHM_HAVE_SVE_SIZE}"             )
message(STATUS "NVHM_HAVE_SVE2                  = ${NVHM_HAVE_SVE2}"                 )

set(NVHM_WANT_CACHE_LINE_SIZE "${NVHM_HAVE_CACHE_LINE_SIZE}" CACHE STRING "Cache line size")
set(NVHM_WANT_PAGE_SIZE "${NVHM_HAVE_PAGE_SIZE}" CACHE STRING "Page size")
set(NVHM_WANT_HUGEPAGE_SIZE "${NVHM_HAVE_HUGEPAGE_SIZE}" CACHE STRING "Desired hugepage size")

option(NVHM_WANT_PAGE_ALIGN "Enable using page aligned allocations by default" ON)
option(NVHM_WANT_TRANSPARENT_HUGEPAGES "Prefer using transparent hugepages" "${NVHM_HAVE_TRANSPARENT_HUGEPAGES}")
if(((${NVHM_HAVE_NUM_HUGEPAGES} GREATER 0) AND (${NVHM_HAVE_MMAP_HUGETLB})) OR
   ((${NVHM_WANT_TRANSPARENT_HUGEPAGES}) AND (${NVHM_HAVE_MADVISE_HUGEPAGE})))
  option(NVHM_WANT_HUGEPAGE_ALIGN "Enable using hugepages by default" ON )
else()
  option(NVHM_WANT_HUGEPAGE_ALIGN "Enable using hugepages by default" OFF)
endif()

if((${NVHM_HAVE_SSE}) OR (${NVHM_HAVE_SSE2}) OR (${NVHM_HAVE_SSSE3}) OR (${NVHM_HAVE_SSE4_1}))
  option(NVHM_WANT_SSE          "Enable SSE"          ON )
else()
  option(NVHM_WANT_SSE          "Enable SSE"          OFF)
endif()

option(NVHM_WANT_SSE_PREFETCH "Enable SSE prefetch" ${NVHM_WANT_SSE})

if(((${NVHM_HAVE_SSE}) AND (${NVHM_HAVE_SSE2})) OR (${NVHM_HAVE_SSSE3}) OR (${NVHM_HAVE_SSE4_1}))
  option(NVHM_WANT_SSE2         "Enable SSE2"         ON )
else()
  option(NVHM_WANT_SSE2         "Enable SSE2"         OFF)
endif()

if(((${NVHM_HAVE_SSE}) AND (${NVHM_HAVE_SSE2}) AND (${NVHM_HAVE_SSSE3})) OR (${NVHM_HAVE_SSE4_1}))
  option(NVHM_WANT_SSE3         "Enable SSSE3"        ON )
else()
  option(NVHM_WANT_SSE3         "Enable SSSE3"        OFF)
endif()

if((${NVHM_HAVE_SSE}) AND (${NVHM_HAVE_SSE2}) AND (${NVHM_HAVE_SSSE3}) AND (${NVHM_HAVE_SSE4_1}))
  option(NVHM_WANT_SSE4         "Enable SSE4.1"       ON )
else()
  option(NVHM_WANT_SSE4         "Enable SSE4.1"       OFF)
endif()

option(NVHM_WANT_AVX          "Enable AVX"          ${NVHM_HAVE_AVX})

if((${NVHM_HAVE_AVX}) AND (${NVHM_HAVE_AVX2}))
  option(NVHM_WANT_AVX2         "Enable AVX2"         ON )
else()
  option(NVHM_WANT_AVX2         "Enable AVX2"         OFF)
endif()

if((${NVHM_HAVE_AVX512_F}) AND (${NVHM_HAVE_AVX512_BW}))
  option(NVHM_WANT_AVX512       "Enable AVX512"       ON )
else()
  option(NVHM_WANT_AVX512       "Enable AVX512"       OFF)
endif()

if((${NVHM_HAVE_AVX512_F}) AND (${NVHM_HAVE_AVX512_VL}))
  option(NVHM_WANT_AVX_FVL      "Enable AVX FVL"      ON )
else()
  option(NVHM_WANT_AVX_FVL      "Enable AVX FVL"      OFF)
endif()

if((${NVHM_HAVE_AVX512_BW}) AND (${NVHM_HAVE_AVX512_VL}))
  option(NVHM_WANT_AVX_BWVL     "Enable AVX BWVL"     ON )
else()
  option(NVHM_WANT_AVX_BWVL     "Enable AVX BWVL"     OFF)
endif()

option(NVHM_WANT_AVX_VBMI     "Enable AVX VBMI"     ${NVHM_HAVE_AVX512_VBMI})
option(NVHM_WANT_NEON         "Enable NEON"         ${NVHM_HAVE_NEON}       )

set(NVHM_WANT_SVE_SIZE "${NVHM_HAVE_SVE_SIZE}" CACHE STRING "Enable SVE for sizes up to this value (in bytes)")
if((${NVHM_HAVE_SVE}) AND NOT ("${NVHM_WANT_SVE_SIZE}" STREQUAL "0"))
  option(NVHM_WANT_SVE          "Enable SVE"          ON )
else()
  option(NVHM_WANT_SVE          "Enable SVE"          OFF)
endif()
if((${NVHM_HAVE_SVE}) AND (${NVHM_HAVE_SVE2}) AND NOT ("${NVHM_WANT_SVE_SIZE}" STREQUAL "0"))
  option(NVHM_WANT_SVE2         "Enable SVE2"         ON )
else()
  option(NVHM_WANT_SVE2         "Enable SVE2"         OFF)
endif()

set(NVHM_WANT_PREFETCH_CACHE_LEVEL "1" CACHE STRING "Cache level to aim for when issuing prefetches")

option(NVHM_WANT_RANGE_PREFETCH "Use scoped SVE prefetches (experimental feature, not for production use)" OFF)

message(STATUS "NVHM_WANT_CACHE_LINE_SIZE       = ${NVHM_WANT_CACHE_LINE_SIZE}"      )
message(STATUS "NVHM_WANT_PAGE_SIZE             = ${NVHM_WANT_PAGE_SIZE}"            )
message(STATUS "NVHM_WANT_PAGE_ALIGN            = ${NVHM_WANT_PAGE_ALIGN}"           )
message(STATUS "NVHM_WANT_HUGEPAGE_SIZE         = ${NVHM_WANT_HUGEPAGE_SIZE}"        )
message(STATUS "NVHM_WANT_HUGEPAGE_ALIGN        = ${NVHM_WANT_HUGEPAGE_ALIGN}"       )
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
message(STATUS "NVHM_WANT_SVE_SIZE              = ${NVHM_WANT_SVE_SIZE}"             )
message(STATUS "NVHM_WANT_SVE2                  = ${NVHM_WANT_SVE2}"                 )
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
  if(${NVHM_WANT_AVX})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
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
    if(${NVHM_WANT_SVE2})
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xclang -target-feature -Xclang +sve2")
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

    # https://gcc.gnu.org/onlinedocs/gcc/AArch64-Options.html#aarch64-feature-modifiers
    set(NVHM_IMPLICT OFF)
    if("${NVHM_AARCH_MAJOR}" GREATER "7")
      set(NVHM_IMPLICT ON)
    elseif("${NVHM_MARCH}" MATCHES "\\+simd")
      set(NVHM_IMPLICT ON)
    endif()
    set(NVHM_MARCH "-march=${NVHM_MARCH}")
    if((${NVHM_WANT_NEON}) AND NOT (${NVHM_IMPLICT}))
      set(NVHM_MARCH "${NVHM_MARCH}+simd")
    endif()
    unset(NVHM_IMPLICT)

    set(NVHM_IMPLICT OFF)
    if("${NVHM_AARCH_MAJOR}" GREATER "8")
      set(NVHM_IMPLICT ON)
    elseif("${NVHM_MARCH}" MATCHES "\\+sve")
      set(NVHM_IMPLICT ON)
    endif()
    if((${NVHM_WANT_SVE}) AND NOT (${NVHM_IMPLICT}))
      set(NVHM_MARCH "${NVHM_MARCH}+sve")
    endif()
    unset(NVHM_IMPLICT)

    set(NVHM_IMPLICT OFF)
    if("${NVHM_AARCH_MAJOR}" GREATER "8")
      set(NVHM_IMPLICT ON)
    elseif("${NVHM_MARCH}" MATCHES "\\+sve2")
      set(NVHM_IMPLICT ON)
    endif()
    if((${NVHM_WANT_SVE2}) AND NOT (${NVHM_IMPLICT}))
      set(NVHM_MARCH "${NVHM_MARCH}+sve2")
    endif()
    unset(NVHM_IMPLICT)

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
    math(EXPR NVHM_WANT_SVE_NUM_BITS "${NVHM_WANT_SVE_SIZE} * 8")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msve-vector-bits=${NVHM_WANT_SVE_NUM_BITS}")
    unset(NVHM_WANT_SVE_NUM_BITS)
  endif()
endif()

# Write configuration header file.
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/include")
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/include/nvhashmap_conf.hpp" CONTENT
"/**
 * Generated during build CMake process by ${PROJECT_SOURCE_DIR}/CMakeLists.txt.
 */
#pragma once

#if   $<BOOL:${NVHM_WANT_SSE4}> && __SSE__ && __SSE2__ && __SSSE3__ && __SSE4_1__
#define NVHM_WITH_SSE (4)
#elif $<BOOL:${NVHM_WANT_SSE3}> && __SSE__ && __SSE2__ && __SSSE3__
#define NVHM_WITH_SSE (3)
#elif $<BOOL:${NVHM_WANT_SSE2}> && __SSE__ && __SSE2__
#define NVHM_WITH_SSE (2)
#elif $<BOOL:${NVHM_WANT_SSE}>  && __SSE__
#define NVHM_WITH_SSE (1)
#else
#define NVHM_WITH_SSE (0)
#endif

#if   $<BOOL:${NVHM_WANT_AVX2}> && __AVX__ && __AVX2__
#define NVHM_WITH_AVX (2)
#elif $<BOOL:${NVHM_WANT_AVX}>  && __AVX__
#define NVHM_WITH_AVX (1)
#else
#define NVHM_WITH_AVX (0)
#endif

#define NVHM_WITH_AVX512   ($<BOOL:${NVHM_WANT_AVX512}> && __AVX512F__ && __AVX512BW__)
#define NVHM_WITH_AVX_FVL  ($<BOOL:${NVHM_WANT_AVX_FVL}> && __AVX512F__ && __AVX512VL__)
#define NVHM_WITH_AVX_BWVL ($<BOOL:${NVHM_WANT_AVX_BWVL}> && __AVX512BW__ && __AVX512VL__)
#define NVHM_WITH_AVX_VBMI ($<BOOL:${NVHM_WANT_AVX_VBMI}> && __AVX512VBMI__)

#define NVHM_WITH_NEON     ($<BOOL:${NVHM_WANT_NEON}> && __ARM_NEON)

#if $<BOOL:${NVHM_WANT_SVE2}> && __ARM_FEATURE_SVE2 && __ARM_FEATURE_SVE_BITS
#define NVHM_WITH_SVE      (2)
#elif $<BOOL:${NVHM_WANT_SVE}> && __ARM_FEATURE_SVE && __ARM_FEATURE_SVE_BITS
#define NVHM_WITH_SVE      (1)
#else
#define NVHM_WITH_SVE      (0)
#endif
#define NVHM_WITH_SVE_SIZE (${NVHM_WANT_SVE_SIZE})

namespace nvhm {

/**
 * Assumed cache line size. For peformance reasons, we allocate buffers at cache line sized
 * boundaries. Also determines the smallest size of a table.
 */
constexpr static int_t cache_line_size{${NVHM_WANT_CACHE_LINE_SIZE}};
static_assert(has_single_bit(cache_line_size));

/**
 * Page size.
 */
constexpr static int_t page_size{${NVHM_WANT_PAGE_SIZE}};
static_assert(has_single_bit(page_size));
static_assert(page_size >= cache_line_size);

constexpr static bool page_align_by_default{$<BOOL:${NVHM_WANT_PAGE_ALIGN}>};

/**
 * Hugepage size.
 */
constexpr static int_t hugepage_size{${NVHM_WANT_HUGEPAGE_SIZE}};
static_assert(has_single_bit(hugepage_size));
static_assert(hugepage_size >= page_size);

/**
 * Enable hugepage allocation.
 */
constexpr static bool use_hugepages_by_default{$<BOOL:${NVHM_WANT_HUGEPAGE_ALIGN}>};

/**
 * Allow using transparent hugepages.
 */
constexpr static bool prefer_madvice_hugepages{$<BOOL:${NVHM_WANT_TRANSPARENT_HUGEPAGES}> || ($<BOOL:${NVHM_HAVE_NUM_HUGEPAGES}> == 0)};

/**
 * Prefetch locality.
 */
constexpr static int prefetch_cache_level{${NVHM_WANT_PREFETCH_CACHE_LEVEL}};
static_assert(prefetch_cache_level >= 1 && prefetch_cache_level <= 5);

/**
 * Prefer SSE prefetch over builtin prefetch.
 */
constexpr static bool use_sse_prefetch{$<BOOL:${NVHM_WANT_SSE_PREFETCH}>};

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

message(STATUS "CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")
add_subdirectory(third_party)

# Enable warnings.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wconversion")

# All done. Let's build!
include_directories(
  "${PROJECT_SOURCE_DIR}/include"
  "${CMAKE_BINARY_DIR}/include"
)

if (NVHM_BUILD_TESTS)
  add_subdirectory(tests)
endif()

if (NVHM_BUILD_TOOLS)
  add_subdirectory(tools)
endif()
