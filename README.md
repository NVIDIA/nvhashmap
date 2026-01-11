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

# NVIDIA Hash-Map

NV Hash-Map provides highly optimized CPU hash-maps and associative container implementations for NVIDIA and other CPUs. It is optimized for both NVIDIA and CPUs from other vendors.

NV Hash-Map provides optimized kernels to maximize hashmap lookup and insert performance on
* NVIDIA Grace [https://www.nvidia.com/en-us/data-center/grace-cpu],
* ARM64 CPUs that support the NEON or SVE instruction sets *(i.e., armv8-a or newer, with up to 2048-bit register size)*, and
* x86 CPUs that support at least SSE2 (additional optimizations are enabled if SSE3 / SSE4 / AVX2 / AVX512 are available).

Further, NV Hash-Map provides C++-17 standard compatible fallback implentations that should be compile and runnable on every 64-bit little-endian CPU.


## Building
NV Hash-Map is a header-only library. However, to maximize performance we generate a configuration header file using CMake.


### Prerequisites (Debian/Ubuntu + GCC)
```shell
sudo apt-get update
sudo apt-get install -y cmake g++
export CC="gcc"
export CXX="g++"
```

We are not aware of any compilation issues with `gcc` / `g++` versions. Please file an issue if you come across a non-working configuration.


### Prerequisites (Debian/Ubuntu + Clang)
```shell
sudo apt-get update
sudo apt-get install -y cmake clang-20
export CC="clang-20"
export CXX="clang++-20"
```

Assuming you start with a minimal Ubuntu docker image without any C++ compiler, you may need to register Clang as a the default alternative for `/usr/lib/c++` to satisfy `cmake`:
```shell
sudo update-alternatives --install /usr/bin/c++ c++ /usr/bin/clang++-16 100
```

**Known issues**:
A LLVM AST parsing issues in `clang++-17`, `clang++-18`, and `clang++-19` may prevent successful compilation in some distributions.  The only wo workaround is to use compilers that do not have this bug.  The problem has been [made known](https://github.com/llvm/llvm-project/issues/107609) to Clang team by myself. It has recently been marked as fixed. However, distributions are slow to acquire compiler updates. For Ubuntu, known compatible LLVM compilers are `clang++-16` as shipped with Ubuntu 24.04, and `clang++-20` as shipped with Ubuntu 25.04.


### Prerequisites (Debian/Ubuntu cross-platform compile)
If targeting **ARM64** (e.g., NVIDIA Grace):
```shell
sudo apt-get update
sudo apt-get install -y cmake g++-aarch64-linux-gnu
export CC="aarch64-linux-gnu-gcc"
export CXX="aarch64-linux-gnu-g++"
```

If targeting **x86/AMD64** (e.g., most Intel+AMD CPUs):
```shell
sudo apt-get update
sudo apt-get install -y cmake g++-x86-64-linux-gnu
export CC="x86_64-linux-gnu-gcc"
export CXX="x86_64-linux-gnu-g++"
```

We are not aware of any compilation issues with `gcc` / `g++` versions. Please file an issue if you come across a non-working configuration.


## Generating the configuration


### For NVIDIA Grace
```shell
git clone https://gitlab-master.nvidia.com/mlanger/monkey-wrench.git
cd monkey-wrench
git submodule update --init --recursive
mkdir build
cd build
cmake -DNVHM_WANT_ALL=ON -DCMAKE_CXX_FLAGS="-march=armv9-a -mtune=neoverse-v2" ..
```


### For other CPUs
```shell
git clone https://gitlab-master.nvidia.com/mlanger/monkey-wrench.git
cd monkey-wrench
git submodule update --init --recursive
mkdir build
cd build
cmake ..
```

The `cmake` command will depend on the capabilties of your target. If you are compiling directly on your target machine, we will detect the capabilitites of your system and adjust the configuration automatically.

If the automatic process fails for you, or you want to compile cross-platform, you can toggle individual optimizations as follows.

**General**:
1. When compiling on your target system, we strongly urge you to add `-DCMAKE_CXX_FLAGS="-march=native -mtune=native"` when calling `cmake` to ensure the compiler can generate the best possible code.
2. If you cross-compile, we cannot make any assumptions regarding your target hardware and will rely to compiler defaults. For optimal performance, you should set `-march` and `-mtune` via `-DCMAKE_CXX_FLAGS`.

**ARM64 specific**:
1. `-DNVHM_WANT_ALL=ON` equivalent to setting all the following options to `ON`.
2. `-DNVHM_WANT_NEON=ON` enables ARM NEON codepath, which provides kernels `arm_kernel64_t` and `arm_kernel18_t` *(the NEON 128-bit kernel becomes the `default_kernel_t` on ARM64 targets)*.
3. `-DNVHM_WANT_SVE=ON` enables ARM SVE codepath, which provides `arm_sve_kernelX_t` kernels, where `X` is any power of 2 between `[1, NVHM_WANT_SVE_MAX_BITS]` *(if enabled, the SVE 128-bit SVE kernel `arm_sve_kernel128_t` replaces the 128-bit NEON kernel as the `default_kernel_t` on ARM64 targets)*.
4. `-DNVHM_WANT_SVE_MAX_BITS=<N>` allows you specify the maximum SVE kernel size that can be used. If not specified, we will try to automatically determine and set this propertye on the build machine.

**x86/AMD64 specific**:
1. `-DNVHM_WANT_ALL=ON` equivalent to setting all the following options to `ON`.
2. `-DNVHM_WANT_SSE2=ON` enables x86 SSE2 codepath, which provides the `x86_kernel128_t` kernel *(the SSE2 128-bit kernel becomes the `default_kernel_t` on x86/AMD64 targets)*.
3. `-DNVHM_WANT_SSE3=ON` enables x86 SSE3 codepath, which enhandes the SSE2 kernel *(also enables `NVHM_WANT_SSE2`)*.
4. `-DNVHM_WANT_SSE4=ON` enables x86 SSE4 codepath, which enhandes the SSE2 kernel *(also enables `NVHM_WANT_SSE2`)*.
5. `-DNVHM_WANT_AVX2=ON` enables x86 AVX2 codepath, which provides the optional `x86_kernel256_t` kernel *(enabling this my also enhance the SSE kernel)*.
6. `-DNVHM_WANT_AVX512=ON` enables x86 AVX-512 codepath, which provides the optional `x86_kernel512_t` kernel *(enabling this my also enhance the SSE2 and AVX2 kernels)*.
7. `-DNVHM_WANT_AVX_FVL=ON` enhances SSE2, AVX2 and AVX512 kernels, if CPU supports the AVX-512 F and VL extensions.
8. `-DNVHM_WANT_AVX_BWVL=ON` enhances SSE2, AVX2 and AVX512 kernels, if CPU supports the AVX-512 BW and VL extensions.
9. `-DNVHM_WANT_AVX_VBMI=ON` enhances SSE2, AVX2 and AVX512 kernels, if CPU supports the AVX-512 VBMI extension.


## Using NV Hash-Map within another `cmake` project

Typically, you want to use NV Hash-Map from within your project. There are multiple ways to achieve this. For example, by adding it as a `git submodule using`, and then using `add_subdirectory` to execute the above configuration step as a nested `cmake` run. In this case you have to pass the previously discussed configuration options via the `cmake` environment.

An alternative, is to use the `FetchContent` plugin. In the following we provide a `cmake`-snippot that will download and make NV Hash-Map's headers available in your current `cmake` scope.

```cmake
include(FetchContent)
set(FETCHCONTENT_QUIET FALSE)

FetchContent_Declare(
  monkey-wrench
  DOWNLOAD_COMMAND git clone
    --branch main
    --depth 1
    --progress "https://gitlab-master.nvidia.com/mlanger/monkey-wrench.git"
    ${CMAKE_BINARY_DIR}/_deps/monkey-wrench-src
)
FetchContent_Populate(monkey-wrench)

execute_process(WORKING_DIRECTORY ${monkey-wrench_BINARY_DIR}
  COMMAND cmake
    -DCMAKE_BUILD_TYPE=Release
    -DBUILD_TESTS=OFF
    -DBUILD_BENCH=OFF
    ${monkey-wrench_SOURCE_DIR}
  COMMAND_ERROR_IS_FATAL ANY
)

include_directories(
  ${monkey-wrench_SOURCE_DIR}/include
  ${monkey-wrench_BINARY_DIR}/include
)
```


## Building tests and the benchmark program

Assuming you have run `cmake` as discussed in the previous section, there is nothing you need to do
```shell
make -j
```


## Using
*TODO*, for the moment `src/benchmark.cpp` provides a good example that shows how NV Hash-Map can be used and how it differs from `std::unordered_map`.


## Project status
Actively maintained.


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
