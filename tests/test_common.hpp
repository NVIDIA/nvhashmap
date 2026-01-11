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

#define NVHM_FOR_EACH_01_(_F_, _X_) _F_(_X_)
#define NVHM_FOR_EACH_02_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_01_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_03_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_02_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_04_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_03_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_05_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_04_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_06_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_05_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_07_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_06_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_08_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_07_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_09_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_08_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_10_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_09_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_11_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_10_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_12_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_11_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_13_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_12_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_14_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_13_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_15_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_14_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_16_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_15_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_17_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_16_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_18_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_17_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_19_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_18_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_20_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_19_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_21_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_20_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_22_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_21_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_23_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_22_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_24_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_23_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_25_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_24_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_26_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_25_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_27_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_26_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_28_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_27_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_29_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_28_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_30_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_29_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_31_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_30_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_32_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_31_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_33_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_32_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_34_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_33_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_35_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_34_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_36_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_35_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_37_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_36_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_38_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_37_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_39_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_38_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_40_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_39_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_41_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_40_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_42_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_41_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_43_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_42_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_44_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_43_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_45_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_44_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_46_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_45_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_47_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_46_(_F_, __VA_ARGS__)

// clang-format off
#define NVHM_GET_MACRO_48_(\
  _0_, _1_, _2_, _3_, _4_, _5_, _6_, _7_,\
  _8_, _9_, _10_, _11_, _12_, _13_, _14_, _15_,\
  _16_, _17_, _18_, _19_, _20_, _21_, _22_, _23_,\
  _24_, _25_, _26_, _27_, _28_, _29_, _30_, _31_,\
  _32_, _33_, _34_, _35_, _36_, _37_, _38_, _39_,\
  _40_, _41_, _42_, _43_, _44_, _45_, _46_, _47_,\
  _MACRO_NAME_, ...\
) _MACRO_NAME_

#define NVHM_FOR_EACH_( _F_, ...)\
  NVHM_GET_MACRO_48_(\
    _0_, __VA_ARGS__,\
    NVHM_FOR_EACH_47_, NVHM_FOR_EACH_46_, NVHM_FOR_EACH_45_, NVHM_FOR_EACH_44_,\
    NVHM_FOR_EACH_43_, NVHM_FOR_EACH_42_, NVHM_FOR_EACH_41_, NVHM_FOR_EACH_40_,\
    NVHM_FOR_EACH_39_, NVHM_FOR_EACH_38_, NVHM_FOR_EACH_37_, NVHM_FOR_EACH_36_,\
    NVHM_FOR_EACH_35_, NVHM_FOR_EACH_34_, NVHM_FOR_EACH_33_, NVHM_FOR_EACH_32_,\
    NVHM_FOR_EACH_31_, NVHM_FOR_EACH_30_, NVHM_FOR_EACH_29_, NVHM_FOR_EACH_28_,\
    NVHM_FOR_EACH_27_, NVHM_FOR_EACH_26_, NVHM_FOR_EACH_25_, NVHM_FOR_EACH_24_,\
    NVHM_FOR_EACH_23_, NVHM_FOR_EACH_22_, NVHM_FOR_EACH_21_, NVHM_FOR_EACH_20_,\
    NVHM_FOR_EACH_19_, NVHM_FOR_EACH_18_, NVHM_FOR_EACH_17_, NVHM_FOR_EACH_16_,\
    NVHM_FOR_EACH_15_, NVHM_FOR_EACH_14_, NVHM_FOR_EACH_13_, NVHM_FOR_EACH_12_,\
    NVHM_FOR_EACH_11_, NVHM_FOR_EACH_10_, NVHM_FOR_EACH_09_, NVHM_FOR_EACH_08_,\
    NVHM_FOR_EACH_07_, NVHM_FOR_EACH_06_, NVHM_FOR_EACH_05_, NVHM_FOR_EACH_04_,\
    NVHM_FOR_EACH_03_, NVHM_FOR_EACH_02_, NVHM_FOR_EACH_01_, NVHM_FOR_EACH_00_\
  )( _F_, __VA_ARGS__ )
// clang-format on
