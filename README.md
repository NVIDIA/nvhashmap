<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# nvHashMap - High Performance Hash-Map for NVIDIA CPUs

[![Unit Tests](https://github.com/NVIDIA/nvhashmap/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/NVIDIA/nvhashmap/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![C++](https://img.shields.io/badge/C%2B%2B-17%20%2F%2020%20%2F%2023-blue.svg)](#requirements)
![Status](https://img.shields.io/badge/status-actively%20maintained-brightgreen.svg)

*nvHashMap* provides highly optimized CPU hash-maps and associative container implementations. It is optimized for both NVIDIA CPUs and CPUs from other vendors.

nvHashMap provides optimized kernels to maximize hash-map lookup and insert performance on
* [NVIDIA Vera](https://www.nvidia.com/en-us/data-center/vera-cpu),
* [NVIDIA Grace](https://www.nvidia.com/en-us/data-center/grace-cpu),
* 64-bit Arm CPUs that support the NEON or SVE instruction sets *(i.e., armv8-a or newer, with up to 2048-bit register size)*, and
* x86 CPUs that support at least SSE2 (additional optimizations are enabled if SSE3 / SSE4 / AVX2 / AVX512 / VBMI are available).

Further, nvHashMap provides C++-17 standard compatible fallback implementations that should compile and be runnable on every 64-bit little-endian CPU.


## Requirements

|                  | Requirement                                                |
| ---------------- | ---------------------------------------------------------- |
| Compiler         | GCC or Clang                                               |
| C++ standard     | C++-17 minimum, C++-20 by default                          |
| CMake            | 3.15 or newer                                              |
| Architecture     | ARM64 *(AArch64)*, x86-64, other 64-bit little-endian CPUs |

**Compiler**:
CI builds and tests with the GCC and Clang versions shipped with Ubuntu 26.04; older releases are expected to work, but are not covered by automated testing.

**C++ standard**:
nvHashMap requires C++-17 or later. `cmake` will select C++-20 unless you pass `-DCMAKE_CXX_STANDARD=<N>`. Note that C++-17 lacks `std::atomic<T>::wait`, so the `spin_wait_mutex` cannot fall back to a futex wait queue and keeps spinning instead. Under heavy contention, a C++-20 build therefore behaves noticeably better.


### Key and value types

Floating point keys are supported, but `NaN` keys are rejected because they are not comparable by definition.


### Optional dependencies

nvHashMap itself is a header-only library and has no dependencies beyond the C++ standard library. The bundled submodules are only needed to build tests, tools, and benchmark utilities. If you configure with `-DNVHM_BUILD_TESTS=OFF -DNVHM_BUILD_TOOLS=OFF -DNVHM_BUILD_BENCH=OFF`, you can skip `git submodule update` entirely.


## Prerequisites

### Debian/Ubuntu + Clang
```shell
sudo apt-get update
sudo apt-get install -y cmake clang
export CC="/usr/bin/clang"
export CXX="/usr/bin/clang++"
```

To cross compile for x86_64 from ARM64:
```shell
export CC="clang --target=x86_64-linux-gnu"   
export CXX="clang++ --target=x86_64-linux-gnu"
```

To cross compile for ARM64 from x86_64:
```shell
export CC="clang --target=aarch64-linux-gnu"   
export CXX="clang++ --target=aarch64-linux-gnu"
```

### Debian/Ubuntu + g++
```shell
sudo apt-get update
sudo apt-get install -y cmake g++
export CC="gcc"
export CXX="g++"
```

To cross compile for x86_64 from ARM64:
```shell
sudo apt-get update
sudo apt-get install -y cmake g++-x86-64-linux-gnu
export CC="x86_64-linux-gnu-gcc"
export CXX="x86_64-linux-gnu-g++"
```

To cross compile for ARM64 from x86_64:
```shell
sudo apt-get update
sudo apt-get install -y cmake g++-aarch64-linux-gnu
export CC="aarch64-linux-gnu-gcc"
export CXX="aarch64-linux-gnu-g++"
```


## Building from source
nvHashMap is a header-only library. However, to maximize performance we generate a configuration header file in your build directory under `${CMAKE_BINARY_DIR}/include` while running `cmake`. To use nvHashMap in your project, just add nvHashMap using `add_subdirectory` to your `cmake` configuration, and expand your C++ compiler's search path to include `${CMAKE_BINARY_DIR}/include` and the `include` directory where nvHashMap itself lives.

### Obtaining the code
```shell
git clone https://github.com/NVIDIA/nvhashmap.git
cd nvhashmap
git submodule update --init --recursive
```

### For NVIDIA Vera & NVIDIA Grace
```shell
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-mcpu=grace -msve-vector-bits=128" ..
```

### For other CPUs
```shell
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
```

The `cmake` command may have to be adjusted depending on the capabilities of your target machine. If you are compiling directly on your target machine, we will detect the capabilities of your system and adjust the configuration automatically.

If the automatic process fails for you, or if you want to compile cross-platform, you can toggle individual optimizations as follows.

**General**:
If you cross-compile, we cannot make any assumptions regarding your target hardware and will rely on compiler defaults. For optimal performance, you should either set `-mcpu`, or `-march` and `-mtune` via `-DCMAKE_CXX_FLAGS`.

**ARM64 specific compilation options**:
1. `-DNVHM_WANT_NEON=ON` enables ARM NEON codepath, which provides kernels `neon_kernel8_t`, `neon_kernel16_t`, `neon_kernel32_t`, and `neon_kernel64_t` *(the NEON 128-bit kernel becomes the `default_kernel_t` on ARM64 targets)*.
2. `-DNVHM_WANT_SVE=ON` enables ARM SVE codepath, which provides `sve_kernelX_t` kernels, where `X` is any power of 2 between `[1, NVHM_WANT_SVE_SIZE]` *(if enabled, the SVE 128-bit SVE kernel `sve_kernel16_t` replaces the 128-bit NEON kernel as the `default_kernel_t<>` on ARM64 targets)*.
3. `-DNVHM_WANT_SVE_SIZE=<N>` allows you specify the maximum SVE kernel size (in bytes) that can be used. If not specified, we will try to automatically determine and set this property based on the capabilities of the build machine.
4. `-DNVHM_WANT_SVE2=ON` enables ARM SVE2 codepath to enhance the SVE kernel.

**x86/AMD64 specific compilation options**:
1. `-DNVHM_WANT_SSE2=ON` enables x86 SSE2 codepath, which provides the `sse_kernel_t` kernel *(the SSE2 128-bit kernel becomes the `default_kernel_t` on x86/AMD64 targets)*.
2. `-DNVHM_WANT_SSE3=ON` enables x86 SSE3 codepaths to enhance the SSE2 kernel.
3. `-DNVHM_WANT_SSE4=ON` enables x86 SSE4 codepaths to enhance the SSE2 kernel.
4. `-DNVHM_WANT_AVX2=ON` enables x86 AVX2 codepath, which provides the optional `avx_kernel_t` kernel *(enabling this feature also enhances the SSE kernel)*.
5. `-DNVHM_WANT_AVX512=ON` enables x86 AVX-512 codepath, which provides the optional `avx512_kernel_t` kernel *(enabling this feature also enhances the SSE2 and AVX2 kernels)*.
6. `-DNVHM_WANT_AVX_FVL=ON` enhances SSE2, AVX2 and AVX512 kernels, if CPU supports the AVX-512 F and VL extensions.
7. `-DNVHM_WANT_AVX_BWVL=ON` enhances SSE2, AVX2 and AVX512 kernels, if CPU supports the AVX-512 BW and VL extensions.
8. `-DNVHM_WANT_AVX_VBMI=ON` enhances SSE2, AVX2 and AVX512 kernels, if CPU supports the AVX-512 VBMI extension.


## Using nvHashMap within another `cmake` project

Typically, you want to use nvHashMap from within your project. There are multiple ways to achieve this. For example, by adding it as a `git submodule`, and then using `add_subdirectory` to execute the above configuration step as a nested `cmake` run. In this case you have to pass the previously discussed configuration options via the `cmake` environment.

An alternative, is to use the `FetchContent` plugin. In the following we provide a `cmake`-snippet that will download and make nvHashMap's headers available in your current `cmake` scope.

```cmake
include(FetchContent)
set(FETCHCONTENT_QUIET FALSE)

FetchContent_Declare(
  nvhashmap
  DOWNLOAD_COMMAND git clone
    --branch main
    --depth 1
    --progress "https://github.com/NVIDIA/nvhashmap.git"
    ${CMAKE_BINARY_DIR}/_deps/nvhashmap-src
)
FetchContent_Populate(nvhashmap)

execute_process(WORKING_DIRECTORY ${nvhashmap_BINARY_DIR}
  COMMAND cmake
    -DCMAKE_BUILD_TYPE=Release
    -DNVHM_BUILD_TESTS=OFF
    -DNVHM_BUILD_TOOLS=OFF
    -DNVHM_BUILD_BENCH=OFF
    ${nvhashmap_SOURCE_DIR}
  COMMAND_ERROR_IS_FATAL ANY
)

include_directories(
  ${nvhashmap_SOURCE_DIR}/include
  ${nvhashmap_BINARY_DIR}/include
)
```


## Building tests and the benchmark program

Assuming you have run `cmake` as discussed above, you can compile the code as follows:
```shell
make -j
```


## Using
*TODO*; however, `src/map_benchmark.cpp` provides a good example that shows how NV Hash-Map can be used and how it differs from `std::unordered_map`.


## License

```text
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
