/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <nvhashmap/map.hpp>
#include <nvhashmap/cache.hpp>
#include <nvhashmap/probe_seq.hpp>

using namespace nvhm;
#include "utils.hpp"

int_t init_capacity{1};
int_t num_keys{1'000'000};
key_source_t key_source{key_source_t::polynomial};
int_t key_c[]{13, 3, 7};
test_trigger_t test_trigger{test_trigger_t::interval};
int_t test_interval{50'000};
int_t test_load_perc{85};
int precision{1};
std::size_t seed{rd()};

template <typename Set>
NVHM_NO_INLINE void fill_and_count_populations() {
  using set_t = Set;
  using conf_t = typename set_t::conf_type;
  using key_t = typename set_t::key_type;

  std::mt19937_64 rng{seed};

  std::vector<key_t> keys{make_keys<key_t>(num_keys, key_source, key_c, rng)};
  const int num_align{rendered_length(num_keys)};
  
  std::cout << type_to_string<set_t>() << ", populations @ ";
  switch (test_trigger) {
    case test_trigger_t::interval:
      std::cout << test_interval << " interval";
      break;
    case test_trigger_t::load_perc:
      std::cout << test_load_perc << '%';
      break;
  }
  std::cout << '\n';
  
  set_t set{conf_t{}.set_capacity(init_capacity, 1).auto_adjust()};

  std::cout << "| " << std::setw(num_align) << "size";
  std::cout << " | " << std::setw(num_align) << "cap.";
  for (int_t i{}; i <= set_t::kernel_size; ++i) {
    std::cout << " | " << std::setw(num_align) << i;
  }
  std::cout << " |\n";

  std::cout << std::setfill('-');
  const char* sep = "| ";
  for (int_t i{}; i <= 2 + set_t::kernel_size; ++i) {
    std::cout << sep << std::setw(num_align) << "";
    sep = " | ";
  }
  std::cout << " |\n" << std::setfill(' ');

  int_t prev_capacity{};
  for (int_t i{}; i < num_keys;) {
    set.insert(keys[to_uint(i++)]);
    switch (test_trigger) {
      case test_trigger_t::interval:
        if (i % test_interval != 0) continue;
        break;
      case test_trigger_t::load_perc:
        if (prev_capacity == set.capacity()) continue;
        if (set.size() * 100 / set.capacity() < test_load_perc) continue;
        prev_capacity = set.capacity();
        break;
    }

    const double num_kernels{static_cast<double>(set.capacity() / set_t::kernel_size)};

    auto pops{set.kernel_populations()};

    std::cout << "| " << std::setw(num_align) << set.size();
    std::cout << " | " << std::setw(num_align) << set.capacity();
    std::cout << std::fixed << std::setprecision(precision);
    for (int_t pop : pops) {
      std::cout << " | " << std::setw(num_align);
      if (pop) {
        std::cout << static_cast<double>(pop * 100) / num_kernels;
      } else {
        std::cout << "";
      }
    }
    std::cout << " |\n";
  }
  
  std::cout << '\n';
}

template <typename Key, typename Kernel, typename ProbeSeq>
void run_3(nvhm_type_t nvhm_type) {
  switch (nvhm_type) {
    case nvhm_type_t::map:
      return fill_and_count_populations<set<Key, flags_t::none, Kernel, ProbeSeq>>();
    case nvhm_type_t::cache:
      return fill_and_count_populations<cache_set<Key, flags_t::none, Kernel>>();
  }

  std::ostringstream os;
  os << "Invalid set type: " << nvhm_type << '!';
  throw std::runtime_error(os.str());
}

template <typename Key, typename Kernel>
void run_2(probe_seq_type_t probe_seq_type, nvhm_type_t nvhm_type) {
  switch (probe_seq_type) {
    case probe_seq_type_t::default_: return run_3<Key, Kernel, default_seq_t>(nvhm_type);
    #if NVHM_TOOLS_COMPILE_PROBE_SEQS >= 10
    case probe_seq_type_t::linear: return run_3<Key, Kernel, linear_seq<0>>(nvhm_type);
    case probe_seq_type_t::quadratic: return run_3<Key, Kernel, quadratic_seq<0>>(nvhm_type);
    #endif
    #if NVHM_TOOLS_COMPILE_PROBE_SEQS >= 20
    case probe_seq_type_t::aligned_linear: return run_3<Key, Kernel, linear_seq<cache_line_size>>(nvhm_type);
    case probe_seq_type_t::aligned_quadratic: return run_3<Key, Kernel, quadratic_seq<cache_line_size>>(nvhm_type);
    #endif
  }

  std::ostringstream os;
  os << "Invalid probe sequence type: " << probe_seq_type << ". Did you forget to compile with `NVHM_TOOLS_COMPILE_PROBE_SEQS`?";
  throw std::runtime_error(os.str());
}

template <typename Key>
void run_1(kernel_type_t kernel_type, probe_seq_type_t probe_seq_type, nvhm_type_t nvhm_type) {
  switch (kernel_type) {
    case kernel_type_t::default_: return run_2<Key, default_kernel_t<>>(probe_seq_type, nvhm_type);
    #if NVHM_TOOLS_COMPILE_KERNELS >= 10
    case kernel_type_t::default1: return run_2<Key, default_kernel_t<1>>(probe_seq_type, nvhm_type);
    case kernel_type_t::default2: return run_2<Key, default_kernel_t<2>>(probe_seq_type, nvhm_type);
    case kernel_type_t::default4: return run_2<Key, default_kernel_t<4>>(probe_seq_type, nvhm_type);
    case kernel_type_t::default8: return run_2<Key, default_kernel_t<8>>(probe_seq_type, nvhm_type);
    case kernel_type_t::default16: return run_2<Key, default_kernel_t<16>>(probe_seq_type, nvhm_type);
    case kernel_type_t::default32: return run_2<Key, default_kernel_t<32>>(probe_seq_type, nvhm_type);
    case kernel_type_t::default64: return run_2<Key, default_kernel_t<64>>(probe_seq_type, nvhm_type);
    case kernel_type_t::default128: return run_2<Key, default_kernel_t<128>>(probe_seq_type, nvhm_type);
    case kernel_type_t::default256: return run_2<Key, default_kernel_t<256>>(probe_seq_type, nvhm_type);
    case kernel_type_t::default512: return run_2<Key, default_kernel_t<512>>(probe_seq_type, nvhm_type);
    #endif
    #if NVHM_TOOLS_COMPILE_KERNELS >= 20
    #if NVHM_WITH_SSE >= 2
    case kernel_type_t::sse: return run_2<Key, sse_kernel_t>(probe_seq_type, nvhm_type);
    #endif
    #if NVHM_WITH_AVX >= 2
    case kernel_type_t::avx: return run_2<Key, avx_kernel_t>(probe_seq_type, nvhm_type);
    #endif
    #if NVHM_WITH_AVX512
    case kernel_type_t::avx512: return run_2<Key, avx512_kernel_t>(probe_seq_type, nvhm_type);
    #endif
    #if NVHM_WITH_NEON
    case kernel_type_t::neon8: return run_2<Key, neon_kernel8_t>(probe_seq_type, nvhm_type);
    case kernel_type_t::neon16: return run_2<Key, neon_kernel16_t>(probe_seq_type, nvhm_type);
    case kernel_type_t::neon32: return run_2<Key, neon_kernel32_t>(probe_seq_type, nvhm_type);
    case kernel_type_t::neon64: return run_2<Key, neon_kernel64_t>(probe_seq_type, nvhm_type);
    #endif
    #if NVHM_WITH_SVE
    #if NVHM_WITH_SVE_SIZE >= 1
    case kernel_type_t::sve1: return run_2<Key, sve_kernel1_t>(probe_seq_type, nvhm_type);
    #endif
    #if NVHM_WITH_SVE_SIZE >= 2
    case kernel_type_t::sve2: return run_2<Key, sve_kernel2_t>(probe_seq_type, nvhm_type);
    #endif
    #if NVHM_WITH_SVE_SIZE >= 4
    case kernel_type_t::sve4: return run_2<Key, sve_kernel4_t>(probe_seq_type, nvhm_type);
    #endif
    #if NVHM_WITH_SVE_SIZE >= 8
    case kernel_type_t::sve8: return run_2<Key, sve_kernel8_t>(probe_seq_type, nvhm_type);
    #endif
    #if NVHM_WITH_SVE_SIZE >= 16
    case kernel_type_t::sve16: return run_2<Key, sve_kernel16_t>(probe_seq_type, nvhm_type);
    #endif
    #if NVHM_WITH_SVE_SIZE >= 32
    case kernel_type_t::sve32: return run_2<Key, sve_kernel32_t>(probe_seq_type, nvhm_type);
    #endif
    #if NVHM_WITH_SVE_SIZE >= 64
    case kernel_type_t::sve64: return run_2<Key, sve_kernel64_t>(probe_seq_type, nvhm_type);
    #endif
    #if NVHM_WITH_SVE_SIZE >= 128
    case kernel_type_t::sve128: return run_2<Key, sve_kernel128_t>(probe_seq_type, nvhm_type);
    #endif
    #if NVHM_WITH_SVE_SIZE >= 256
    case kernel_type_t::sve256: return run_2<Key, sve_kernel256_t>(probe_seq_type, nvhm_type);
    #endif
    #endif
    #endif
    #if NVHM_TOOLS_COMPILE_KERNELS >= 30
    case kernel_type_t::uint1: return run_2<Key, uint_kernel1_t>(probe_seq_type, nvhm_type);
    case kernel_type_t::uint2: return run_2<Key, uint_kernel2_t>(probe_seq_type, nvhm_type);
    case kernel_type_t::uint4: return run_2<Key, uint_kernel4_t>(probe_seq_type, nvhm_type);
    case kernel_type_t::uint8: return run_2<Key, uint_kernel8_t>(probe_seq_type, nvhm_type);
    case kernel_type_t::uint16: return run_2<Key, uint_kernel16_t>(probe_seq_type, nvhm_type);
    case kernel_type_t::fast_uint1: return run_2<Key, fast_uint_kernel1_t>(probe_seq_type, nvhm_type);
    case kernel_type_t::fast_uint2: return run_2<Key, fast_uint_kernel2_t>(probe_seq_type, nvhm_type);
    case kernel_type_t::fast_uint4: return run_2<Key, fast_uint_kernel4_t>(probe_seq_type, nvhm_type);
    case kernel_type_t::fast_uint8: return run_2<Key, fast_uint_kernel8_t>(probe_seq_type, nvhm_type);
    case kernel_type_t::fast_uint16: return run_2<Key, fast_uint_kernel16_t>(probe_seq_type, nvhm_type);
    #endif
    #if NVHM_TOOLS_COMPILE_KERNELS >= 40
    case kernel_type_t::array1: return run_2<Key, array_kernel1_t>(probe_seq_type, nvhm_type);
    case kernel_type_t::array2: return run_2<Key, array_kernel2_t>(probe_seq_type, nvhm_type);
    case kernel_type_t::array4: return run_2<Key, array_kernel4_t>(probe_seq_type, nvhm_type);
    case kernel_type_t::array8: return run_2<Key, array_kernel8_t>(probe_seq_type, nvhm_type);
    case kernel_type_t::array16: return run_2<Key, array_kernel16_t>(probe_seq_type, nvhm_type);
    case kernel_type_t::array32: return run_2<Key, array_kernel32_t>(probe_seq_type, nvhm_type);
    case kernel_type_t::array64: return run_2<Key, array_kernel64_t>(probe_seq_type, nvhm_type);
    case kernel_type_t::array128: return run_2<Key, array_kernel128_t>(probe_seq_type, nvhm_type);
    case kernel_type_t::array256: return run_2<Key, array_kernel256_t>(probe_seq_type, nvhm_type);
    case kernel_type_t::array512: return run_2<Key, array_kernel512_t>(probe_seq_type, nvhm_type);
    #endif
  }

  std::ostringstream os;
  os << "Invalid kernel type: " << kernel_type << ". Did you forget to compile with `NVHM_TOOLS_COMPILE_KERNELS`?";
  throw std::runtime_error(os.str());
}

void run_0(key_type_t key_type, kernel_type_t kernel_type, probe_seq_type_t probe_seq_type, nvhm_type_t nvhm_type) {
  switch (key_type) {
    case key_type_t::int32: return run_1<int32_t>(kernel_type, probe_seq_type, nvhm_type);
    case key_type_t::int64: return run_1<int64_t>(kernel_type, probe_seq_type, nvhm_type);
  }
  throw std::runtime_error("Unsupported `key_type`!");
}

int main(int argc, char* argv[]) {
  CLI::App app{"Kernel population analysis tool"};
  
  key_type_t key_type{key_type_t::int64};
  kernel_type_t kernel_type{kernel_type_t::default_};
  nvhm_type_t nvhm_type{nvhm_type_t::map};
  probe_seq_type_t probe_seq_type{probe_seq_type_t::default_};

  app.add_option("--init_capacity", init_capacity, "Initial capacity")->default_val(init_capacity)->check(CLI::Validator(CLI::PositiveNumber));
  app.add_option("--key_type", key_type, "Key type")->default_str(to_string(key_type))->transform(CLI::CheckedTransformer(str_to_key_type, CLI::ignore_case));
  app.add_option("--num_keys", num_keys, "Number of keys")->default_val(num_keys)->check(CLI::Validator(CLI::PositiveNumber));
  app.add_option("--key_source", key_source, "Key source")->default_str(to_string(key_source))->transform(CLI::CheckedTransformer(str_to_key_source, CLI::ignore_case));
  app.add_option("--key_c0", key_c[0], "Key coefficient 0 (key space density control)")->default_val(key_c[0]);
  app.add_option("--key_c1", key_c[1], "Key coefficient 1 (key space density control)")->default_val(key_c[1]);
  app.add_option("--key_c2", key_c[2], "Key coefficient 2 (key space density control)")->default_val(key_c[2]);
  app.add_option("--test_trigger", test_trigger, "Test trigger")->default_str(to_string(test_trigger))->transform(CLI::CheckedTransformer(str_to_test_trigger, CLI::ignore_case));
  app.add_option("--test_interval", test_interval, "Interval in which to test the probe sequence lengths")->default_val(test_interval);
  app.add_option("--precision", precision, "Precision of the output")->default_val(precision)->check(CLI::Validator(CLI::NonNegativeNumber));
  app.add_option("--nvhm_type", nvhm_type, "NVHM collection type")->default_str(to_string(nvhm_type))->transform(CLI::CheckedTransformer(str_to_nvhm_type, CLI::ignore_case));
  app.add_option("--kernel_type", kernel_type, "Kernel type")->default_str(to_string(kernel_type))->transform(CLI::CheckedTransformer(str_to_kernel_type, CLI::ignore_case));
  app.add_option("--probe_seq_type", probe_seq_type, "Probe sequence type")->default_str(to_string(probe_seq_type))->transform(CLI::CheckedTransformer(str_to_probe_seq_type));
  app.add_option("--seed", seed, "Randomizer seed")->default_str("random");

  argv = app.ensure_utf8(argv);
  CLI11_PARSE(app, argc, argv);

  run_0(key_type, kernel_type, probe_seq_type, nvhm_type);
  return 0;
}