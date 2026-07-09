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
#define NVHM_FOR_EACH_48_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_47_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_49_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_48_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_50_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_49_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_51_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_50_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_52_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_51_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_53_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_52_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_54_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_53_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_55_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_54_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_56_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_55_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_57_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_56_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_58_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_57_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_59_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_58_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_60_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_59_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_61_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_60_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_62_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_61_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_63_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_62_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_64_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_63_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_65_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_64_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_66_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_65_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_67_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_66_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_68_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_67_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_69_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_68_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_70_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_69_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_71_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_70_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_72_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_71_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_73_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_72_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_74_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_73_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_75_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_74_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_76_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_75_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_77_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_76_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_78_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_77_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_79_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_78_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_80_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_79_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_81_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_80_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_82_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_81_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_83_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_82_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_84_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_83_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_85_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_84_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_86_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_85_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_87_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_86_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_88_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_87_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_89_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_88_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_90_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_89_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_91_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_90_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_92_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_91_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_93_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_92_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_94_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_93_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_95_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_94_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_96_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_95_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_97_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_96_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_98_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_97_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_99_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_98_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_100_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_99_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_101_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_100_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_102_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_101_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_103_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_102_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_104_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_103_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_105_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_104_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_106_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_105_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_107_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_106_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_108_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_107_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_109_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_108_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_110_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_109_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_111_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_110_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_112_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_111_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_113_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_112_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_114_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_113_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_115_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_114_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_116_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_115_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_117_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_116_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_118_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_117_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_119_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_118_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_120_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_119_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_121_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_120_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_122_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_121_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_123_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_122_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_124_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_123_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_125_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_124_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_126_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_125_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_127_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_126_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_128_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_127_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_129_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_128_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_130_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_129_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_131_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_130_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_132_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_131_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_133_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_132_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_134_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_133_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_135_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_134_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_136_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_135_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_137_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_136_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_138_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_137_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_139_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_138_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_140_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_139_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_141_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_140_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_142_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_141_(_F_, __VA_ARGS__)
#define NVHM_FOR_EACH_143_(_F_, _X_, ...) _F_(_X_) NVHM_FOR_EACH_142_(_F_, __VA_ARGS__)

// clang-format off
#define NVHM_GET_FOR_EACH_MACRO_(\
  _0_, _1_, _2_, _3_, _4_, _5_, _6_, _7_,\
  _8_, _9_, _10_, _11_, _12_, _13_, _14_, _15_,\
  _16_, _17_, _18_, _19_, _20_, _21_, _22_, _23_,\
  _24_, _25_, _26_, _27_, _28_, _29_, _30_, _31_,\
  _32_, _33_, _34_, _35_, _36_, _37_, _38_, _39_,\
  _40_, _41_, _42_, _43_, _44_, _45_, _46_, _47_,\
  _48_, _49_, _50_, _51_, _52_, _53_, _54_, _55_,\
  _56_, _57_, _58_, _59_, _60_, _61_, _62_, _63_,\
  _64_, _65_, _66_, _67_, _68_, _69_, _70_, _71_,\
  _72_, _73_, _74_, _75_, _76_, _77_, _78_, _79_,\
  _80_, _81_, _82_, _83_, _84_, _85_, _86_, _87_,\
  _88_, _89_, _90_, _91_, _92_, _93_, _94_, _95_,\
  _96_, _97_, _98_, _99_, _100_, _101_, _102_, _103_,\
  _104_, _105_, _106_, _107_, _108_, _109_, _110_, _111_,\
  _112_, _113_, _114_, _115_, _116_, _117_, _118_, _119_,\
  _120_, _121_, _122_, _123_, _124_, _125_, _126_, _127_,\
  _128_, _129_, _130_, _131_, _132_, _133_, _134_, _135_,\
  _136_, _137_, _138_, _139_, _140_, _141_, _142_, _143_,\
  _MACRO_NAME_, ...\
) _MACRO_NAME_

#define NVHM_FOR_EACH_( _F_, ...)\
  NVHM_GET_FOR_EACH_MACRO_(\
    _0_, __VA_ARGS__,\
    NVHM_FOR_EACH_143_, NVHM_FOR_EACH_142_, NVHM_FOR_EACH_141_, NVHM_FOR_EACH_140_,\
    NVHM_FOR_EACH_139_, NVHM_FOR_EACH_138_, NVHM_FOR_EACH_137_, NVHM_FOR_EACH_136_,\
    NVHM_FOR_EACH_135_, NVHM_FOR_EACH_134_, NVHM_FOR_EACH_133_, NVHM_FOR_EACH_132_,\
    NVHM_FOR_EACH_131_, NVHM_FOR_EACH_130_, NVHM_FOR_EACH_129_, NVHM_FOR_EACH_128_,\
    NVHM_FOR_EACH_127_, NVHM_FOR_EACH_126_, NVHM_FOR_EACH_125_, NVHM_FOR_EACH_124_,\
    NVHM_FOR_EACH_123_, NVHM_FOR_EACH_122_, NVHM_FOR_EACH_121_, NVHM_FOR_EACH_120_,\
    NVHM_FOR_EACH_119_, NVHM_FOR_EACH_118_, NVHM_FOR_EACH_117_, NVHM_FOR_EACH_116_,\
    NVHM_FOR_EACH_115_, NVHM_FOR_EACH_114_, NVHM_FOR_EACH_113_, NVHM_FOR_EACH_112_,\
    NVHM_FOR_EACH_111_, NVHM_FOR_EACH_110_, NVHM_FOR_EACH_109_, NVHM_FOR_EACH_108_,\
    NVHM_FOR_EACH_107_, NVHM_FOR_EACH_106_, NVHM_FOR_EACH_105_, NVHM_FOR_EACH_104_,\
    NVHM_FOR_EACH_103_, NVHM_FOR_EACH_102_, NVHM_FOR_EACH_101_, NVHM_FOR_EACH_100_,\
    NVHM_FOR_EACH_99_, NVHM_FOR_EACH_98_, NVHM_FOR_EACH_97_, NVHM_FOR_EACH_96_,\
    NVHM_FOR_EACH_95_, NVHM_FOR_EACH_94_, NVHM_FOR_EACH_93_, NVHM_FOR_EACH_92_,\
    NVHM_FOR_EACH_91_, NVHM_FOR_EACH_90_, NVHM_FOR_EACH_89_, NVHM_FOR_EACH_88_,\
    NVHM_FOR_EACH_87_, NVHM_FOR_EACH_86_, NVHM_FOR_EACH_85_, NVHM_FOR_EACH_84_,\
    NVHM_FOR_EACH_83_, NVHM_FOR_EACH_82_, NVHM_FOR_EACH_81_, NVHM_FOR_EACH_80_,\
    NVHM_FOR_EACH_79_, NVHM_FOR_EACH_78_, NVHM_FOR_EACH_77_, NVHM_FOR_EACH_76_,\
    NVHM_FOR_EACH_75_, NVHM_FOR_EACH_74_, NVHM_FOR_EACH_73_, NVHM_FOR_EACH_72_,\
    NVHM_FOR_EACH_71_, NVHM_FOR_EACH_70_, NVHM_FOR_EACH_69_, NVHM_FOR_EACH_68_,\
    NVHM_FOR_EACH_67_, NVHM_FOR_EACH_66_, NVHM_FOR_EACH_65_, NVHM_FOR_EACH_64_,\
    NVHM_FOR_EACH_63_, NVHM_FOR_EACH_62_, NVHM_FOR_EACH_61_, NVHM_FOR_EACH_60_,\
    NVHM_FOR_EACH_59_, NVHM_FOR_EACH_58_, NVHM_FOR_EACH_57_, NVHM_FOR_EACH_56_,\
    NVHM_FOR_EACH_55_, NVHM_FOR_EACH_54_, NVHM_FOR_EACH_53_, NVHM_FOR_EACH_52_,\
    NVHM_FOR_EACH_51_, NVHM_FOR_EACH_50_, NVHM_FOR_EACH_49_, NVHM_FOR_EACH_48_,\
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

#include <iomanip>
#include <random>

inline std::ostream& operator<<(std::ostream& os, __uint128_t v) {
  std::ios_base::fmtflags f{os.flags()};

  size_t base;
  if ((f & std::ios_base::basefield) == std::ios_base::oct) {
    base = 8;
  } else if ((f & std::ios_base::basefield) == std::ios_base::hex) {
    base = 16;
  } else {
    base = 10;
  }
  bool uppercase{(f & std::ios_base::uppercase) != 0};

  if (f & std::ios_base::showbase) {
    if (base != 10) {
      os << '0';
    }
    if (base == 16) {
      os << (uppercase ? 'X' : 'x');
    }
  }

  char buffer[64];
  int n{};
  for (; v != 0; v /= base) {
    __uint128_t r{v % base};
    if (r < 10) {
      buffer[n++] = static_cast<char>('0' + r);
    } else {
      buffer[n++] = static_cast<char>((uppercase ? 'A' : 'a') + (r - 10));
    }
  }

  while (n--) {
    os << buffer[n];
  }

  return os;
}

static std::random_device rd;
static std::mt19937_64 rng(rd());
