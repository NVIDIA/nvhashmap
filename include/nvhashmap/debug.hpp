/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#ifdef NVHM_ASSERT_
#error NVHM_ASSERT_ was defined elsewhere. Something is wrong.
#endif

#ifdef NVHM_DBG_LOG_
#error NVHM_DBG_LOG_ was defined elsewhere. Something is wrong.
#endif

#if defined(NDEBUG)
#define NVHM_ASSERT_(...) \
  do {                    \
  } while (false)

#define NVHM_DBG_LOG_(...) \
  do {                     \
  } while (false)

#else
#define NVHM_ASSERT_(_expr_, ...)                                                            \
  do {                                                                                       \
    if (NVHM_UNLIKELY_(!(_expr_))) {                                                         \
      nvhm::write_args(                                                                      \
        std::cerr, "\nAssertion: ", #_expr_, " failed!", "\nLocation: ", __FUNCTION__, " (", \
        __FILE__, ':', __LINE__, ')', "\nContext: ", ##__VA_ARGS__, "\n\n"                   \
      );                                                                                     \
      std::cerr.flush();                                                                     \
      std::abort();                                                                          \
    }                                                                                        \
  } while (false)

#define NVHM_DBG_LOG_(...)                                                                        \
  do {                                                                                            \
    nvhm::write_args(std::cerr, __FUNCTION__, " (", __FILE__, ':', __LINE__, "): ", __VA_ARGS__); \
  } while (false)

#include <iostream>
#include <iomanip>

namespace nvhm {

template <typename... Args>
inline void write_args(std::ostream& o, Args&&... args) {
  (o << ... << args);
}

template <typename Mask>
inline std::string render_mask(typename Mask::repr_type m) {
  using mask_t = Mask;

  std::ostringstream os;
  for (size_t i{}; mask_t::has_next(m); m = mask_t::step(m)) {
    os << (i++ ? ", " : "") << mask_t::next(m);
  }
  return os.str();
}

template <typename Kernel>
inline std::string render_kernel(const typename Kernel::repr_type& k) {
  using kernel_t = Kernel;

  std::ostringstream os;
  os << std::hex;
  for (size_t i{}; i < kernel_t::size; ++i) {
    os << (i == 0 ? "" : ", ");
    os << std::setw(2) << (kernel_t::at(k, i) & 0xff);
  }
  return os.str();
}

}  // namespace nvhm
#endif
