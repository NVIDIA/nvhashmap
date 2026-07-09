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


# These applications are pointless. They are just here to ensure each header is self-contained.

add_executable(compile_allocator      compile_allocator.cpp)
add_executable(compile_cache          compile_cache.cpp)
add_executable(compile_common         compile_common.cpp)
add_executable(compile_conf           compile_conf.cpp)
add_executable(compile_container      compile_container.cpp)
add_executable(compile_guarded        compile_guarded.cpp)
add_executable(compile_hash           compile_hash.cpp)
add_executable(compile_kernel         compile_kernel.cpp)
add_executable(compile_map            compile_map.cpp)
add_executable(compile_mask           compile_mask.cpp)
add_executable(compile_memory         compile_memory.cpp)
add_executable(compile_prefetch       compile_prefetch.cpp)
add_executable(compile_probe_seq      compile_probe_seq.cpp)
add_executable(compile_raw_map_base   compile_raw_map_base.cpp)
add_executable(compile_sharded        compile_sharded.cpp)
add_executable(compile_stdlib_ext     compile_stdlib_ext.cpp)
add_executable(compile_std_map_shim   compile_std_map_shim.cpp)
add_executable(compile_swiss_map_base compile_swiss_map_base.cpp)