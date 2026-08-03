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

#include "stdlib_ext.hpp"

namespace nvhm {

#if INTPTR_MAX == INT32_MAX
using int_t = std::int32_t;
using uint_t = std::uint32_t;
#elif INTPTR_MAX == INT64_MAX
using int_t = std::int64_t;
using uint_t = std::uint64_t;
#else
static_assert(false, "Unsupported platform!");
#endif
using bitmask_t = uint_t;

constexpr int_t to_int(uint_t x) noexcept { return static_cast<int_t>(x); }
constexpr uint_t to_uint(int_t x) noexcept { return static_cast<uint_t>(x); }

}  // namespace nvhm

#include <functional>

/**
 * `nvhashmap_conf.hpp` is generated during CMake in your build directory under
 * `${CMAKE_BINARY_DIR}/include`. If your compiler fails locating it, something in your
 * configuration is odd. Try adding the it to the search path manually, by ensuring that
 * `${CMAKE_BINARY_DIR}/include` is part in your include file search path.
 */
#include <nvhashmap_conf.hpp>

#ifdef NVHM_ASSERT_
#error NVHM_ASSERT_ was defined elsewhere. Something is wrong.
#endif
#define NVHM_ASSERT_(_expr_, ...)                                            \
  do {                                                                       \
    if constexpr (nvhm::max_check_level >= nvhm::check_level_t::prototype) { \
      if (NVHM_UNLIKELY_(!(_expr_))) {                                       \
        std::ostream& os{nvhm::debug_stream()};                              \
        nvhm::render_args(os,                                                \
          "\nAssertion: ", #_expr_, " failed!",                              \
          "\nLocation: ", __FUNCTION__, " (", __FILE__, ':', __LINE__, ')',  \
          "\nContext: ", ##__VA_ARGS__, "\n\n"                               \
        );                                                                   \
        if constexpr (nvhm::max_check_level >= nvhm::check_level_t::debug) { \
          os.flush();                                                        \
        }                                                                    \
        std::abort();                                                        \
      }                                                                      \
    }                                                                        \
  } while (false)

#ifdef NVHM_THROW_
#error NVHM_THROW_ was defined elsewhere. Something is wrong.
#endif
#define NVHM_THROW_(_class_, ...)                                                               \
  do {                                                                                          \
    throw _class_(nvhm::render_args_to_string(                                                  \
      #_class_, ": ", ##__VA_ARGS__, " [", __FUNCTION__, " @ ",  __FILE__, ':', __LINE__,']')); \
  } while (false)

#ifdef NVHM_ASSUME_
#error NVHM_ASSUME_ was defined elsewhere. Something is wrong.
#endif
#define NVHM_ASSUME_(_expr_, ...)                                                           \
  do {                                                                                      \
    if constexpr (nvhm::max_check_level >= nvhm::check_level_t::release) {                  \
      if (NVHM_UNLIKELY_(!(_expr_))) {                                                      \
        NVHM_THROW_(std::logic_error, "Assumption(", #_expr_, ") failed! ", ##__VA_ARGS__); \
      }                                                                                     \
    }                                                                                       \
  } while (false)

#ifdef NVHM_LOG_
#error NVHM_LOG_ was defined elsewhere. Something is wrong.
#endif
#define NVHM_LOG_(_level_, ...)                                                                 \
  do {                                                                                          \
    if constexpr ((_level_) <= nvhm::max_log_level) {                                           \
      std::ostream& os{nvhm::debug_stream()};                                                   \
      nvhm::render_args(os, __FUNCTION__, " (", __FILE__, ':', __LINE__, "): ", ##__VA_ARGS__); \
    }                                                                                           \
  } while (false)

#ifdef NVHM_LIKELY_
#error NVHM_LIKELY_ was defined elsewhere. Something is wrong.
#endif
#define NVHM_LIKELY_(_expr_) __builtin_expect((_expr_), 1)

#ifdef NVHM_UNLIKELY_
#error NVHM_UNLIKELY_ was defined elsewhere. Something is wrong.
#endif
#define NVHM_UNLIKELY_(_expr_) __builtin_expect((_expr_), 0)

#ifdef NVHM_MAKE_NOT_INSTANTIABLE_
#error NVHM_MAKE_NOT_INSTANTIABLE_ was defined elsewhere. Something is wrong.
#endif
#define NVHM_MAKE_NOT_INSTANTIABLE_(_class_)   \
  _class_() = delete;                          \
  _class_(const _class_&) = delete;            \
  _class_(_class_&&) = delete;                 \
  _class_& operator=(const _class_&) = delete; \
  _class_& operator=(_class_&&) = delete;

#ifdef NVHM_ALWAYS_INLINE
#error NVHM_ALWAYS_INLINE was defined elsewhere. Something is wrong.
#endif
#if defined(__clang__)
#define NVHM_ALWAYS_INLINE [[clang::always_inline]] inline
#elif defined(__GNUC__)
#define NVHM_ALWAYS_INLINE [[gnu::always_inline]] inline
#else
#define NVHM_ALWAYS_INLINE inline
#endif

#ifdef NVHM_NO_INLINE
#error NVHM_NO_INLINE was defined elsewhere. Something is wrong.
#endif
#if defined(__clang__)
#define NVHM_NO_INLINE [[clang::noinline]]
#elif defined(__GNUC__)
#define NVHM_NO_INLINE [[gnu::noinline]]
#else
#define NVHM_NO_INLINE
#endif

#ifdef NVHM_NO_INLINE_CALL
#error NVHM_NO_INLINE_CALL was defined elsewhere. Something is wrong.
#endif
#if defined(__clang__)
#define NVHM_NO_INLINE_CALL [[clang::noinline]]
#else
#define NVHM_NO_INLINE_CALL
#endif

// clang-format off
#ifdef NVHM_ARG_SELECT_
#error NVHM_ARG_SELECT_ was defined elsewhere. Something is wrong.
#endif
#define NVHM_ARG_SELECT_(                                                                         \
  _00_, _01_, _02_, _03_, _04_, _05_, _06_, _07_, _08_, _09_, _0A_, _0B_, _0C_, _0D_, _0E_, _0F_, \
  _10_, _11_, _12_, _13_, _14_, _15_, _16_, _17_, _18_, _19_, _1A_, _1B_, _1C_, _1D_, _1E_, _1F_, \
  _20_, _21_, _22_, _23_, _24_, _25_, _26_, _27_, _28_, _29_, _2A_, _2B_, _2C_, _2D_, _2E_, _2F_, \
  _N_, ...) _N_

#if                                                                                                                                                               \
  defined(NVHM_MAKE_ENUM_FOR_EACH_00_) || defined(NVHM_MAKE_ENUM_FOR_EACH_01_) || defined(NVHM_MAKE_ENUM_FOR_EACH_02_) || defined(NVHM_MAKE_ENUM_FOR_EACH_03_) || \
  defined(NVHM_MAKE_ENUM_FOR_EACH_04_) || defined(NVHM_MAKE_ENUM_FOR_EACH_05_) || defined(NVHM_MAKE_ENUM_FOR_EACH_06_) || defined(NVHM_MAKE_ENUM_FOR_EACH_07_) || \
  defined(NVHM_MAKE_ENUM_FOR_EACH_08_) || defined(NVHM_MAKE_ENUM_FOR_EACH_09_) || defined(NVHM_MAKE_ENUM_FOR_EACH_0A_) || defined(NVHM_MAKE_ENUM_FOR_EACH_0B_) || \
  defined(NVHM_MAKE_ENUM_FOR_EACH_0C_) || defined(NVHM_MAKE_ENUM_FOR_EACH_0D_) || defined(NVHM_MAKE_ENUM_FOR_EACH_0E_) || defined(NVHM_MAKE_ENUM_FOR_EACH_0F_) || \
  defined(NVHM_MAKE_ENUM_FOR_EACH_10_) || defined(NVHM_MAKE_ENUM_FOR_EACH_11_) || defined(NVHM_MAKE_ENUM_FOR_EACH_12_) || defined(NVHM_MAKE_ENUM_FOR_EACH_13_) || \
  defined(NVHM_MAKE_ENUM_FOR_EACH_14_) || defined(NVHM_MAKE_ENUM_FOR_EACH_15_) || defined(NVHM_MAKE_ENUM_FOR_EACH_16_) || defined(NVHM_MAKE_ENUM_FOR_EACH_17_) || \
  defined(NVHM_MAKE_ENUM_FOR_EACH_18_) || defined(NVHM_MAKE_ENUM_FOR_EACH_19_) || defined(NVHM_MAKE_ENUM_FOR_EACH_1A_) || defined(NVHM_MAKE_ENUM_FOR_EACH_1B_) || \
  defined(NVHM_MAKE_ENUM_FOR_EACH_1C_) || defined(NVHM_MAKE_ENUM_FOR_EACH_1D_) || defined(NVHM_MAKE_ENUM_FOR_EACH_1E_) || defined(NVHM_MAKE_ENUM_FOR_EACH_1F_) || \
  defined(NVHM_MAKE_ENUM_FOR_EACH_20_) || defined(NVHM_MAKE_ENUM_FOR_EACH_21_) || defined(NVHM_MAKE_ENUM_FOR_EACH_22_) || defined(NVHM_MAKE_ENUM_FOR_EACH_23_) || \
  defined(NVHM_MAKE_ENUM_FOR_EACH_24_) || defined(NVHM_MAKE_ENUM_FOR_EACH_25_) || defined(NVHM_MAKE_ENUM_FOR_EACH_26_) || defined(NVHM_MAKE_ENUM_FOR_EACH_27_) || \
  defined(NVHM_MAKE_ENUM_FOR_EACH_28_) || defined(NVHM_MAKE_ENUM_FOR_EACH_29_) || defined(NVHM_MAKE_ENUM_FOR_EACH_2A_) || defined(NVHM_MAKE_ENUM_FOR_EACH_2B_) || \
  defined(NVHM_MAKE_ENUM_FOR_EACH_2C_) || defined(NVHM_MAKE_ENUM_FOR_EACH_2D_) || defined(NVHM_MAKE_ENUM_FOR_EACH_2E_) || defined(NVHM_MAKE_ENUM_FOR_EACH_2F_)
#error NVHM_MAKE_ENUM_FOR_..._ was defined elsewhere. Something is wrong.
#endif
#define NVHM_MAKE_ENUM_FOR_EACH_00_(_F_, _type_name_, _value_) _F_(_type_name_, _value_)
#define NVHM_MAKE_ENUM_FOR_EACH_01_(_F_, _type_name_, _value_, ...) _F_(_type_name_, _value_) NVHM_MAKE_ENUM_FOR_EACH_00_(_F_, _type_name_, __VA_ARGS__)
#define NVHM_MAKE_ENUM_FOR_EACH_02_(_F_, _type_name_, _value_, ...) _F_(_type_name_, _value_) NVHM_MAKE_ENUM_FOR_EACH_01_(_F_, _type_name_, __VA_ARGS__)
#define NVHM_MAKE_ENUM_FOR_EACH_03_(_F_, _type_name_, _value_, ...) _F_(_type_name_, _value_) NVHM_MAKE_ENUM_FOR_EACH_02_(_F_, _type_name_, __VA_ARGS__)
#define NVHM_MAKE_ENUM_FOR_EACH_04_(_F_, _type_name_, _value_, ...) _F_(_type_name_, _value_) NVHM_MAKE_ENUM_FOR_EACH_03_(_F_, _type_name_, __VA_ARGS__)
#define NVHM_MAKE_ENUM_FOR_EACH_05_(_F_, _type_name_, _value_, ...) _F_(_type_name_, _value_) NVHM_MAKE_ENUM_FOR_EACH_04_(_F_, _type_name_, __VA_ARGS__)
#define NVHM_MAKE_ENUM_FOR_EACH_06_(_F_, _type_name_, _value_, ...) _F_(_type_name_, _value_) NVHM_MAKE_ENUM_FOR_EACH_05_(_F_, _type_name_, __VA_ARGS__)
#define NVHM_MAKE_ENUM_FOR_EACH_07_(_F_, _type_name_, _value_, ...) _F_(_type_name_, _value_) NVHM_MAKE_ENUM_FOR_EACH_06_(_F_, _type_name_, __VA_ARGS__)
#define NVHM_MAKE_ENUM_FOR_EACH_08_(_F_, _type_name_, _value_, ...) _F_(_type_name_, _value_) NVHM_MAKE_ENUM_FOR_EACH_07_(_F_, _type_name_, __VA_ARGS__)
#define NVHM_MAKE_ENUM_FOR_EACH_09_(_F_, _type_name_, _value_, ...) _F_(_type_name_, _value_) NVHM_MAKE_ENUM_FOR_EACH_08_(_F_, _type_name_, __VA_ARGS__)
#define NVHM_MAKE_ENUM_FOR_EACH_0A_(_F_, _type_name_, _value_, ...) _F_(_type_name_, _value_) NVHM_MAKE_ENUM_FOR_EACH_09_(_F_, _type_name_, __VA_ARGS__)
#define NVHM_MAKE_ENUM_FOR_EACH_0B_(_F_, _type_name_, _value_, ...) _F_(_type_name_, _value_) NVHM_MAKE_ENUM_FOR_EACH_0A_(_F_, _type_name_, __VA_ARGS__)
#define NVHM_MAKE_ENUM_FOR_EACH_0C_(_F_, _type_name_, _value_, ...) _F_(_type_name_, _value_) NVHM_MAKE_ENUM_FOR_EACH_0B_(_F_, _type_name_, __VA_ARGS__)
#define NVHM_MAKE_ENUM_FOR_EACH_0D_(_F_, _type_name_, _value_, ...) _F_(_type_name_, _value_) NVHM_MAKE_ENUM_FOR_EACH_0C_(_F_, _type_name_, __VA_ARGS__)
#define NVHM_MAKE_ENUM_FOR_EACH_0E_(_F_, _type_name_, _value_, ...) _F_(_type_name_, _value_) NVHM_MAKE_ENUM_FOR_EACH_0D_(_F_, _type_name_, __VA_ARGS__)
#define NVHM_MAKE_ENUM_FOR_EACH_0F_(_F_, _type_name_, _value_, ...) _F_(_type_name_, _value_) NVHM_MAKE_ENUM_FOR_EACH_0E_(_F_, _type_name_, __VA_ARGS__)
#define NVHM_MAKE_ENUM_FOR_EACH_10_(_F_, _type_name_, _value_, ...) _F_(_type_name_, _value_) NVHM_MAKE_ENUM_FOR_EACH_0F_(_F_, _type_name_, __VA_ARGS__)
#define NVHM_MAKE_ENUM_FOR_EACH_11_(_F_, _type_name_, _value_, ...) _F_(_type_name_, _value_) NVHM_MAKE_ENUM_FOR_EACH_10_(_F_, _type_name_, __VA_ARGS__)
#define NVHM_MAKE_ENUM_FOR_EACH_12_(_F_, _type_name_, _value_, ...) _F_(_type_name_, _value_) NVHM_MAKE_ENUM_FOR_EACH_11_(_F_, _type_name_, __VA_ARGS__)
#define NVHM_MAKE_ENUM_FOR_EACH_13_(_F_, _type_name_, _value_, ...) _F_(_type_name_, _value_) NVHM_MAKE_ENUM_FOR_EACH_12_(_F_, _type_name_, __VA_ARGS__)
#define NVHM_MAKE_ENUM_FOR_EACH_14_(_F_, _type_name_, _value_, ...) _F_(_type_name_, _value_) NVHM_MAKE_ENUM_FOR_EACH_13_(_F_, _type_name_, __VA_ARGS__)
#define NVHM_MAKE_ENUM_FOR_EACH_15_(_F_, _type_name_, _value_, ...) _F_(_type_name_, _value_) NVHM_MAKE_ENUM_FOR_EACH_14_(_F_, _type_name_, __VA_ARGS__)
#define NVHM_MAKE_ENUM_FOR_EACH_16_(_F_, _type_name_, _value_, ...) _F_(_type_name_, _value_) NVHM_MAKE_ENUM_FOR_EACH_15_(_F_, _type_name_, __VA_ARGS__)
#define NVHM_MAKE_ENUM_FOR_EACH_17_(_F_, _type_name_, _value_, ...) _F_(_type_name_, _value_) NVHM_MAKE_ENUM_FOR_EACH_16_(_F_, _type_name_, __VA_ARGS__)
#define NVHM_MAKE_ENUM_FOR_EACH_18_(_F_, _type_name_, _value_, ...) _F_(_type_name_, _value_) NVHM_MAKE_ENUM_FOR_EACH_17_(_F_, _type_name_, __VA_ARGS__)
#define NVHM_MAKE_ENUM_FOR_EACH_19_(_F_, _type_name_, _value_, ...) _F_(_type_name_, _value_) NVHM_MAKE_ENUM_FOR_EACH_18_(_F_, _type_name_, __VA_ARGS__)
#define NVHM_MAKE_ENUM_FOR_EACH_1A_(_F_, _type_name_, _value_, ...) _F_(_type_name_, _value_) NVHM_MAKE_ENUM_FOR_EACH_19_(_F_, _type_name_, __VA_ARGS__)
#define NVHM_MAKE_ENUM_FOR_EACH_1B_(_F_, _type_name_, _value_, ...) _F_(_type_name_, _value_) NVHM_MAKE_ENUM_FOR_EACH_1A_(_F_, _type_name_, __VA_ARGS__)
#define NVHM_MAKE_ENUM_FOR_EACH_1C_(_F_, _type_name_, _value_, ...) _F_(_type_name_, _value_) NVHM_MAKE_ENUM_FOR_EACH_1B_(_F_, _type_name_, __VA_ARGS__)
#define NVHM_MAKE_ENUM_FOR_EACH_1D_(_F_, _type_name_, _value_, ...) _F_(_type_name_, _value_) NVHM_MAKE_ENUM_FOR_EACH_1C_(_F_, _type_name_, __VA_ARGS__)
#define NVHM_MAKE_ENUM_FOR_EACH_1E_(_F_, _type_name_, _value_, ...) _F_(_type_name_, _value_) NVHM_MAKE_ENUM_FOR_EACH_1D_(_F_, _type_name_, __VA_ARGS__)
#define NVHM_MAKE_ENUM_FOR_EACH_1F_(_F_, _type_name_, _value_, ...) _F_(_type_name_, _value_) NVHM_MAKE_ENUM_FOR_EACH_1E_(_F_, _type_name_, __VA_ARGS__)
#define NVHM_MAKE_ENUM_FOR_EACH_20_(_F_, _type_name_, _value_, ...) _F_(_type_name_, _value_) NVHM_MAKE_ENUM_FOR_EACH_1F_(_F_, _type_name_, __VA_ARGS__)
#define NVHM_MAKE_ENUM_FOR_EACH_21_(_F_, _type_name_, _value_, ...) _F_(_type_name_, _value_) NVHM_MAKE_ENUM_FOR_EACH_20_(_F_, _type_name_, __VA_ARGS__)
#define NVHM_MAKE_ENUM_FOR_EACH_22_(_F_, _type_name_, _value_, ...) _F_(_type_name_, _value_) NVHM_MAKE_ENUM_FOR_EACH_21_(_F_, _type_name_, __VA_ARGS__)
#define NVHM_MAKE_ENUM_FOR_EACH_23_(_F_, _type_name_, _value_, ...) _F_(_type_name_, _value_) NVHM_MAKE_ENUM_FOR_EACH_22_(_F_, _type_name_, __VA_ARGS__)
#define NVHM_MAKE_ENUM_FOR_EACH_24_(_F_, _type_name_, _value_, ...) _F_(_type_name_, _value_) NVHM_MAKE_ENUM_FOR_EACH_23_(_F_, _type_name_, __VA_ARGS__)
#define NVHM_MAKE_ENUM_FOR_EACH_25_(_F_, _type_name_, _value_, ...) _F_(_type_name_, _value_) NVHM_MAKE_ENUM_FOR_EACH_24_(_F_, _type_name_, __VA_ARGS__)
#define NVHM_MAKE_ENUM_FOR_EACH_26_(_F_, _type_name_, _value_, ...) _F_(_type_name_, _value_) NVHM_MAKE_ENUM_FOR_EACH_25_(_F_, _type_name_, __VA_ARGS__)
#define NVHM_MAKE_ENUM_FOR_EACH_27_(_F_, _type_name_, _value_, ...) _F_(_type_name_, _value_) NVHM_MAKE_ENUM_FOR_EACH_26_(_F_, _type_name_, __VA_ARGS__)
#define NVHM_MAKE_ENUM_FOR_EACH_28_(_F_, _type_name_, _value_, ...) _F_(_type_name_, _value_) NVHM_MAKE_ENUM_FOR_EACH_27_(_F_, _type_name_, __VA_ARGS__)
#define NVHM_MAKE_ENUM_FOR_EACH_29_(_F_, _type_name_, _value_, ...) _F_(_type_name_, _value_) NVHM_MAKE_ENUM_FOR_EACH_28_(_F_, _type_name_, __VA_ARGS__)
#define NVHM_MAKE_ENUM_FOR_EACH_2A_(_F_, _type_name_, _value_, ...) _F_(_type_name_, _value_) NVHM_MAKE_ENUM_FOR_EACH_29_(_F_, _type_name_, __VA_ARGS__)
#define NVHM_MAKE_ENUM_FOR_EACH_2B_(_F_, _type_name_, _value_, ...) _F_(_type_name_, _value_) NVHM_MAKE_ENUM_FOR_EACH_2A_(_F_, _type_name_, __VA_ARGS__)
#define NVHM_MAKE_ENUM_FOR_EACH_2C_(_F_, _type_name_, _value_, ...) _F_(_type_name_, _value_) NVHM_MAKE_ENUM_FOR_EACH_2B_(_F_, _type_name_, __VA_ARGS__)
#define NVHM_MAKE_ENUM_FOR_EACH_2D_(_F_, _type_name_, _value_, ...) _F_(_type_name_, _value_) NVHM_MAKE_ENUM_FOR_EACH_2C_(_F_, _type_name_, __VA_ARGS__)
#define NVHM_MAKE_ENUM_FOR_EACH_2E_(_F_, _type_name_, _value_, ...) _F_(_type_name_, _value_) NVHM_MAKE_ENUM_FOR_EACH_2D_(_F_, _type_name_, __VA_ARGS__)
#define NVHM_MAKE_ENUM_FOR_EACH_2F_(_F_, _type_name_, _value_, ...) _F_(_type_name_, _value_) NVHM_MAKE_ENUM_FOR_EACH_2E_(_F_, _type_name_, __VA_ARGS__)

#ifdef NVHM_MAKE_ENUM_FOR_EACH_
#error NVHM_MAKE_ENUM_FOR_EACH_ was defined elsewhere. Something is wrong.
#endif
#define NVHM_MAKE_ENUM_FOR_EACH_(_F_, _type_name_, ...)                                                                 \
  NVHM_ARG_SELECT_(__VA_ARGS__,                                                                                         \
    NVHM_MAKE_ENUM_FOR_EACH_2F_, NVHM_MAKE_ENUM_FOR_EACH_2E_, NVHM_MAKE_ENUM_FOR_EACH_2D_, NVHM_MAKE_ENUM_FOR_EACH_2C_, \
    NVHM_MAKE_ENUM_FOR_EACH_2B_, NVHM_MAKE_ENUM_FOR_EACH_2A_, NVHM_MAKE_ENUM_FOR_EACH_29_, NVHM_MAKE_ENUM_FOR_EACH_28_, \
    NVHM_MAKE_ENUM_FOR_EACH_27_, NVHM_MAKE_ENUM_FOR_EACH_26_, NVHM_MAKE_ENUM_FOR_EACH_25_, NVHM_MAKE_ENUM_FOR_EACH_24_, \
    NVHM_MAKE_ENUM_FOR_EACH_23_, NVHM_MAKE_ENUM_FOR_EACH_22_, NVHM_MAKE_ENUM_FOR_EACH_21_, NVHM_MAKE_ENUM_FOR_EACH_20_, \
    NVHM_MAKE_ENUM_FOR_EACH_1F_, NVHM_MAKE_ENUM_FOR_EACH_1E_, NVHM_MAKE_ENUM_FOR_EACH_1D_, NVHM_MAKE_ENUM_FOR_EACH_1C_, \
    NVHM_MAKE_ENUM_FOR_EACH_1B_, NVHM_MAKE_ENUM_FOR_EACH_1A_, NVHM_MAKE_ENUM_FOR_EACH_19_, NVHM_MAKE_ENUM_FOR_EACH_18_, \
    NVHM_MAKE_ENUM_FOR_EACH_17_, NVHM_MAKE_ENUM_FOR_EACH_16_, NVHM_MAKE_ENUM_FOR_EACH_15_, NVHM_MAKE_ENUM_FOR_EACH_14_, \
    NVHM_MAKE_ENUM_FOR_EACH_13_, NVHM_MAKE_ENUM_FOR_EACH_12_, NVHM_MAKE_ENUM_FOR_EACH_11_, NVHM_MAKE_ENUM_FOR_EACH_10_, \
    NVHM_MAKE_ENUM_FOR_EACH_0F_, NVHM_MAKE_ENUM_FOR_EACH_0E_, NVHM_MAKE_ENUM_FOR_EACH_0D_, NVHM_MAKE_ENUM_FOR_EACH_0C_, \
    NVHM_MAKE_ENUM_FOR_EACH_0B_, NVHM_MAKE_ENUM_FOR_EACH_0A_, NVHM_MAKE_ENUM_FOR_EACH_09_, NVHM_MAKE_ENUM_FOR_EACH_08_, \
    NVHM_MAKE_ENUM_FOR_EACH_07_, NVHM_MAKE_ENUM_FOR_EACH_06_, NVHM_MAKE_ENUM_FOR_EACH_05_, NVHM_MAKE_ENUM_FOR_EACH_04_, \
    NVHM_MAKE_ENUM_FOR_EACH_03_, NVHM_MAKE_ENUM_FOR_EACH_02_, NVHM_MAKE_ENUM_FOR_EACH_01_, NVHM_MAKE_ENUM_FOR_EACH_00_  \
  )(_F_, _type_name_, __VA_ARGS__)

#ifdef NVHM_MAKE_ENUM_TO_STR_CASE_
#error NVHM_MAKE_ENUM_TO_STR_CASE_ was defined elsewhere. Something is wrong.
#endif
#define NVHM_MAKE_ENUM_TO_STR_CASE_(_type_name_, _value_) \
  case _type_name_::_value_:                              \
    if (array_equal(#_value_, "default_")) {              \
      return "default";                                   \
    }                                                     \
    return #_value_;

#ifdef NVHM_MAKE_ENUM_
#error NVHM_MAKE_ENUM_ was defined elsewhere. Something is wrong.
#endif
#define NVHM_MAKE_ENUM_(_type_name_, ...)                                              \
  enum class _type_name_ : int32_t {                                                   \
    __VA_ARGS__                                                                        \
  };                                                                                   \
                                                                                       \
  constexpr std::string_view to_string_view(const _type_name_ v) noexcept {            \
    switch (v) {                                                                       \
      NVHM_MAKE_ENUM_FOR_EACH_(NVHM_MAKE_ENUM_TO_STR_CASE_, _type_name_, __VA_ARGS__); \
    }                                                                                  \
    return "error";                                                                    \
  }                                                                                    \
                                                                                       \
  inline std::string to_string(_type_name_ v) noexcept {                               \
    return std::string{to_string_view(v)};                                             \
  }                                                                                    \
                                                                                       \
  inline std::ostream& operator<<(std::ostream& os, _type_name_ v) {                   \
    return os << to_string_view(v);                                                    \
  }

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

namespace nvhm {

template <int_t N, int_t M>
constexpr bool array_equal(const char (&a)[N], const char (&b)[M]) noexcept {
  if constexpr (N != M) return false;
  for (int_t i{}; i < N; ++i) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

template <typename T>
constexpr static int_t num_bytes_v{sizeof(T)};

template <>
constexpr int_t num_bytes_v<void>{};

template <typename T>
constexpr static int_t num_bits_v{num_bytes_v<T> * 8};

template <typename>
constexpr static bool dependent_false_v{false};

NVHM_MAKE_ENUM_(log_level_t,
  error,    // 0
  warning,  // 1
  info,     // 2
  debug     // 3
);

NVHM_MAKE_ENUM_(check_level_t,
  none,       // 0
  release,    // 1
  prototype,  // 2
  debug       // 3
);

#if defined(NDEBUG)
constexpr static bool is_debug_build{false};
constexpr static check_level_t max_check_level{check_level_t::none};
constexpr static log_level_t max_log_level{log_level_t::error};
#else
constexpr static bool is_debug_build{true};
constexpr static check_level_t max_check_level{check_level_t::debug};
constexpr static log_level_t max_log_level{log_level_t::debug};
#endif

constexpr std::ostream& debug_stream() noexcept { return std::cerr; }

template <typename... Args>
constexpr std::ostream& render_args(std::ostream& os, Args&&... args) {
  (os << ... << args);
  return os;
}

template <typename... Args>
inline std::string render_args_to_string(Args&&... args) {
  std::ostringstream os;
  render_args(os, std::forward<Args>(args)...);
  return os.str();
}

constexpr bitmask_t make_aligned_mask(int_t n, int_t alignment) noexcept {
  NVHM_ASSERT_(n > 0 && has_single_bit(n));
  NVHM_ASSERT_(alignment > 0 && has_single_bit(alignment));
  NVHM_ASSERT_(n >= alignment, "n = ", n, ", alignment = ", alignment);
  NVHM_ASSERT_(n % alignment == 0, "n = ", n, ", alignment = ", alignment);
  return to_uint(n - alignment);
}

template <int_t Alignment>
constexpr bitmask_t make_aligned_mask(int_t n) noexcept {
  static_assert(Alignment > 0 && has_single_bit(Alignment));
  return make_aligned_mask(n, Alignment);
}

template <int_t N, int_t Alignment>
constexpr static bitmask_t aligned_mask_v{[]() {
  static_assert(N > 0 && has_single_bit(N));
  static_assert(Alignment > 0 && has_single_bit(Alignment));
  static_assert(N >= Alignment && N % Alignment == 0);
  return N - Alignment;
}()};

constexpr bitmask_t make_size_mask(int_t n) noexcept {
  NVHM_ASSERT_(n > 0 && has_single_bit(n));
  return make_aligned_mask(n, 1);
}

template <int_t Size>
constexpr static bitmask_t size_mask_v{aligned_mask_v<Size, 1>};

constexpr bool is_size_mask(bitmask_t m) noexcept {
  m += 1;
  return !m || has_single_bit(m);
}

constexpr bool is_contiguous_mask(bitmask_t m) noexcept {
  if (m) {
    m += bitmask_t{1} << countr_zero(m);
  }
  return !m || has_single_bit(m);
}

constexpr static bitmask_t cache_line_mask{size_mask_v<cache_line_size>};

constexpr int_t aligned_mask_to_capacity(bitmask_t mask, int_t alignment) noexcept {
  NVHM_ASSERT_(has_single_bit(alignment));
  NVHM_ASSERT_(!mask || countr_zero(mask) == countr_zero(alignment), "mask = ", std::hex, mask, std::dec, ", alignment = ", alignment);

  int_t capacity{to_int(mask) + alignment};
  NVHM_ASSERT_(has_single_bit(capacity));
  NVHM_ASSERT_(capacity % alignment == 0, "mask = ", std::hex, mask, std::dec, ", capacity = ", capacity, ", alignment = ", alignment);

  return capacity;
}

template <int_t Alignment>
constexpr int_t aligned_mask_to_capacity(bitmask_t mask) noexcept {
  static_assert(has_single_bit(Alignment));
  return aligned_mask_to_capacity(mask, Alignment);
}

template <typename T>
constexpr static T ceil_div(T x, T n) noexcept {
  static_assert(std::is_integral_v<T>);
  return (x + n - 1) / n;
}

template <typename T>
constexpr static T round_up(T x, T n) noexcept {
  return ceil_div(x, n) * n;
}

template <int_t N>
constexpr static int_t round_up(int_t x) noexcept {
  if constexpr (has_single_bit(N)) {
    return (x + N - 1) & ~(N - 1);
  } else {
    return round_up(x, N);
  }
}

template <typename>
struct enum_bitmask_operators : std::false_type {};

template <typename T>
constexpr static bool enum_bitmask_operators_v{enum_bitmask_operators<T>::value};

template <typename T>
constexpr std::enable_if_t<enum_bitmask_operators_v<T>, T> operator~(T x) noexcept {
  using U = std::underlying_type_t<T>;
  return static_cast<T>(~static_cast<U>(x));
}

template <typename T>
constexpr std::enable_if_t<enum_bitmask_operators_v<T>, T> operator&(T a, T b) noexcept {
  using U = std::underlying_type_t<T>;
  return static_cast<T>(static_cast<U>(a) & static_cast<U>(b));
}

template <typename T>
constexpr std::enable_if_t<enum_bitmask_operators_v<T>, T> operator|(T a, T b) noexcept {
  using U = std::underlying_type_t<T>;
  return static_cast<T>(static_cast<U>(a) | static_cast<U>(b));
}

template <typename T>
constexpr std::enable_if_t<enum_bitmask_operators_v<T>, T> operator^(T a, T b) noexcept {
  using U = std::underlying_type_t<T>;
  return static_cast<T>(static_cast<U>(a) ^ static_cast<U>(b));
}

template <typename T>
constexpr std::enable_if_t<enum_bitmask_operators_v<T>, T&> operator&=(T& a, T b) noexcept {
  a = a & b;
  return a;
}

template <typename T>
constexpr std::enable_if_t<enum_bitmask_operators_v<T>, T&> operator|=(T& a, T b) noexcept {
  a = a | b;
  return a;
}

template <typename T>
constexpr std::enable_if_t<enum_bitmask_operators_v<T>, T&> operator^=(T& a, T b) noexcept {
  a = a ^ b;
  return a;
}

template <typename>
struct is_std_tuple : std::false_type {};

template <typename... Ts>
struct is_std_tuple<std::tuple<Ts...>> : std::true_type {};

template <typename T>
constexpr static bool is_std_tuple_v{is_std_tuple<T>::value};

static_assert(!is_std_tuple_v<int>);
static_assert(is_std_tuple_v<std::tuple<int>>);
static_assert(is_std_tuple_v<std::tuple<int, double>>);

template <typename T, typename = void>
struct is_ostreamable : std::false_type {};

template <typename T>
struct is_ostreamable<T, std::enable_if_t<std::is_same_v<decltype(std::declval<std::ostream&>() << std::declval<const T&>()), std::ostream&>>>
  : std::true_type {};

template <typename T>
constexpr static bool is_ostreamable_v{is_ostreamable<T>::value};

/**
 * Data type used for slots. Slots either contain a hash or `empty` or `tombstone`.
 * `empty` means that the slot was never used since the last rehash, or it was reclaimed.
 * `tombstone` means that the slot was used, then freed, but couldn't be reclaimed.
 */
using state_t = int8_t;
static_assert(std::is_signed_v<state_t> && sizeof(state_t) == 1);

constexpr static int_t num_state_bits{num_bits_v<state_t> - 1};
static_assert(num_state_bits == 7, "If this ever changes, you need to check everything else!");
constexpr static bitmask_t state_mask{size_mask_v<1 << num_state_bits>};

constexpr bool is_hash(state_t s) noexcept { return s >= 0; }
constexpr bool is_not_hash(state_t s) noexcept { return s < 0; }

using lru_t = uint8_t;
static_assert(std::is_unsigned_v<lru_t> && sizeof(lru_t) == sizeof(state_t));

constexpr static lru_t max_lru{std::numeric_limits<lru_t>::max()};
constexpr static lru_t default_lru{max_lru / 2};
static_assert(default_lru + 1 < max_lru);

using raw_pos_t = intptr_t;
constexpr static raw_pos_t npos{-1};

constexpr raw_pos_t align_pos(raw_pos_t pos, bitmask_t mask) noexcept {
  NVHM_ASSERT_(is_contiguous_mask(mask));
  return pos & static_cast<raw_pos_t>(mask);
}

template <bitmask_t Mask>
constexpr raw_pos_t align_pos(raw_pos_t pos) noexcept {
  static_assert(is_contiguous_mask(Mask));
  return pos & static_cast<raw_pos_t>(Mask);
}

using psl_t = int_t;

constexpr static psl_t inf_psl{-1};

template <typename InOutIt, typename Size, typename UnaryOp>
constexpr InOutIt transform_n(InOutIt first, Size n, UnaryOp f) {
  return std::transform(first, first + n, first, f);
}

template <typename InIt0, typename InIt1, typename Size>
constexpr bool equal_n(InIt0 first0, InIt1 first1, Size n) {
  return std::equal(first0, first0 + n, first1);
}

enum class flags_t : uint_t {
  none = 0,
  blobs = 1,                // The map allocates and maintains a blob storage.
  duplicates = 2,           // The map/set allows duplicate keys.
  aggressive_prefetch = 4,  // Be slightly more aggressive wehn prefetching.
  auto_scrub = 8,           // Occasional stop-the-world `scrub` upon `erase` to speedup `find`.
  auto_shrink = 16,         // Occasional stop-the-world `shrink` in `erase` to speedup `find`.
  all = blobs | duplicates | aggressive_prefetch | auto_scrub | auto_shrink
};
static_assert(has_single_bit(static_cast<uint_t>(flags_t::blobs)));
static_assert(has_single_bit(static_cast<uint_t>(flags_t::duplicates)));
static_assert(has_single_bit(static_cast<uint_t>(flags_t::aggressive_prefetch)));
static_assert(has_single_bit(static_cast<uint_t>(flags_t::auto_scrub)));
static_assert(has_single_bit(static_cast<uint_t>(flags_t::auto_shrink)));

template <>
struct enum_bitmask_operators<flags_t> : std::true_type {};

constexpr static bool test_flags(flags_t f, flags_t q) noexcept { return (f & q) == q; }

inline std::ostream& operator<<(std::ostream& os, flags_t f) {
  const char* sep{""};
  if (test_flags(f, flags_t::blobs)) {
    os << sep << "blobs";
    sep = " | ";
  }
  if (test_flags(f, flags_t::duplicates)) {
    os << sep << "duplicates";
    sep = " | ";
  }
  if (test_flags(f, flags_t::aggressive_prefetch)) {
    os << sep << "aggressive_prefetch";
    sep = " | ";
  }
  if (test_flags(f, flags_t::auto_scrub)) {
    os << sep << "auto_scrub";
    sep = " | ";
  }
  if (test_flags(f, flags_t::auto_shrink)) {
    os << sep << "auto_shrink";
    sep = " | ";
  }
  return os;
}

template <typename T>
inline std::ostream& render_n(const T* __restrict v, int_t n, std::ostream& os) noexcept {
  const char* sep{""};
  for (int_t i{}; i < n; ++i) {
    os << sep << v[i];
    sep = " ";
  }
  return os;
}

template <>
inline std::ostream& render_n<std::byte>(const std::byte* __restrict v, int_t n, std::ostream& os) noexcept {
  const std::ios_base::fmtflags fmt_flags{os.flags()};
  const char old_fill{os.fill()};

  os << std::hex;
  os.fill('0');

  const char* sep{""};
  for (int_t i{}; i < n; ++i) {
    os << sep << std::setw(2) << std::to_integer<int>(v[i]);
    sep = " ";
  }

  os.flags(fmt_flags);
  os.fill(old_fill);

  return os;
}

inline std::ostream& operator<<(std::ostream& os, std::byte b) noexcept {
  return render_n(&b, 1, os);
}

/**
 * Implements the CRTP (Curiously Recurring Template Pattern) pattern to allow compile
 * time resolution of the self type.
 */
template <typename Self>
class self_aware {
 public:
  using self_type = Self;

  constexpr self_aware() = default;

  constexpr self_type* self() noexcept { return static_cast<self_type*>(this); }
  constexpr const self_type* self() const noexcept { return static_cast<const self_type*>(this); }

  constexpr const self_type* cself() const noexcept { return self(); }
};

template <typename T>
using value_t = std::conditional_t<std::is_void_v<T>, std::monostate, T>;

template <typename T>
using const_view_t = std::conditional_t<std::is_scalar_v<T>, T, const T&>;

template <typename T>
constexpr static bool is_value_v{!std::is_same_v<value_t<T>, value_t<void>>};

using blob_t = std::byte;
static_assert(sizeof(blob_t) == 1, "Must check lot's of places if this changes!");

template <typename Key, typename Value, bool HasBlobs>
using const_entry_t = std::conditional_t<is_value_v<Value> && HasBlobs,
  std::tuple<const_view_t<Key>, const_view_t<Value>, const blob_t*>,
  std::conditional_t<is_value_v<Value>,
    std::pair<const_view_t<Key>, const_view_t<Value>>,
    std::conditional_t<HasBlobs,
      std::pair<const_view_t<Key>, const blob_t*>,
      const_view_t<Key>
    >
  >
>;

template <typename Key, typename Value, bool HasBlobs>
using entry_t = std::conditional_t<is_value_v<Value> && HasBlobs,
  std::tuple<const_view_t<Key>, Value&, blob_t*>,
  std::conditional_t<is_value_v<Value>,
    std::pair<const_view_t<Key>, Value&>,
    std::conditional_t<HasBlobs,
      std::pair<const_view_t<Key>, blob_t*>,
      const_view_t<Key>
    >
  >
>;

template <typename Value, bool HasBlobs>
using const_mapped_t = std::conditional_t<is_value_v<Value> && HasBlobs,
  std::pair<const_view_t<Value>, const blob_t*>,
  std::conditional_t<is_value_v<Value>,
    const_view_t<Value>, std::conditional_t<HasBlobs, const blob_t*, void>
  >
>;

template <typename Value, bool HasBlobs>
using mapped_t = std::conditional_t<is_value_v<Value> && HasBlobs,
  std::pair<Value&, blob_t*>,
  std::conditional_t<is_value_v<Value>,
    Value&, std::conditional_t<HasBlobs, blob_t*, void>
  >
>;

// We use function traits to peel apart callable signatures.
template <typename F>
struct func_traits : func_traits<decltype(&F::operator())> {};

// For a `const` member `operator()`.
template <typename C, typename Ret, typename... Args>
struct func_traits<Ret (C::*)(Args...) const> {
  using result_type = Ret;

  template <size_t I>
  struct arg {
    static_assert(I < sizeof...(Args), "Argument index out of range!");

    using type = typename std::tuple_element<I, std::tuple<Args...>>::type;
  };
};

// For a non-`const member `operator()`
template <typename C, typename Ret, typename... Args>
struct func_traits<Ret (C::*)(Args...)> {
  using result_type = Ret;

  template <size_t I>
  struct arg {
    static_assert(I < sizeof...(Args), "Argument index out of range!");

    using type = typename std::tuple_element<I, std::tuple<Args...>>::type;
  };
};

// For free functions and function pointers.
template <typename Ret, typename... Args>
struct func_traits<Ret (*)(Args...)> {
  using result_type = Ret;

  template <size_t I>
  struct arg {
    static_assert(I < sizeof...(Args), "Argument index out of range!");

    using type = typename std::tuple_element<I, std::tuple<Args...>>::type;
  };
};

template <typename F, size_t I>
using arg_n_t = typename func_traits<std::decay_t<F>>::template arg<I>::type;

NVHM_MAKE_ENUM_(arg_type_t,
  value,
  lvalue_ref,
  const_lvalue_ref,
  rvalue_ref
);

template <typename T>
struct arg_type {
  constexpr static arg_type_t value{arg_type_t::value};
};

template <typename T>
struct arg_type<T&> {
  constexpr static arg_type_t value{arg_type_t::lvalue_ref};
};

template <typename T>
struct arg_type<const T&> {
  constexpr static arg_type_t value{arg_type_t::const_lvalue_ref};
};

template <typename T>
struct arg_type<T&&> {
  constexpr static arg_type_t value{arg_type_t::rvalue_ref};
};

template <typename T>
constexpr static arg_type_t arg_type_v{arg_type<T>::value};

}  // namespace nvhm
