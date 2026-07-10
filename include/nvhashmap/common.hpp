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

} // namespace nvhm

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
#define NVHM_ASSERT_(_expr_, ...)                                           \
  do {                                                                      \
    if constexpr (nvhm::max_check_level >= nvhm::check_level_t::prototype) {    \
      if (NVHM_UNLIKELY_(!(_expr_))) {                                      \
        std::ostream& os{nvhm::debug_stream()};                             \
        nvhm::render_args(os,                                               \
          "\nAssertion: ", #_expr_, " failed!",                             \
          "\nLocation: ", __FUNCTION__, " (", __FILE__, ':', __LINE__, ')', \
          "\nContext: ", ##__VA_ARGS__, "\n\n");                            \
          if constexpr (nvhm::max_check_level >= nvhm::check_level_t::debug) { \
            os.flush();                                                        \
          }                                                                    \
        std::abort();                                                       \
      }                                                                     \
    }                                                                       \
  } while (false)

#ifdef NVHM_ASSUME_
#error NVHM_ASSUME_ was defined elsewhere. Something is wrong.
#endif
#define NVHM_ASSUME_(_expr_, ...) \
  do {\
    if constexpr (nvhm::max_check_level >= nvhm::check_level_t::release) {\
      if (NVHM_UNLIKELY_(!(_expr_))) {\
        throw nvhm::logic_error("Assumption failed", __FILE__, __LINE__, __FUNCTION__, #_expr_, ##__VA_ARGS__);\
      }\
    }\
  } while (false)

#ifdef NVHM_LOG_
#error NVHM_LOG_ was defined elsewhere. Something is wrong.
#endif
#define NVHM_LOG_(_level_, ...) \
  do {                        \
    if constexpr ((_level_) <= nvhm::max_log_level) {\
      std::ostream& os{nvhm::debug_stream()};                           \
      nvhm::render_args(os, __FUNCTION__, " (", __FILE__, ':', __LINE__, "): ", ##__VA_ARGS__);\
    }\
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
#define NVHM_MAKE_NOT_INSTANTIABLE_(_class_) \
  _class_() = delete; \
  _class_(const _class_&) = delete; \
  _class_(_class_&&) = delete; \
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

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <utility>
#include <variant>

namespace nvhm {

template <typename T>
constexpr static int_t num_bytes_v{sizeof(T)};

template <>
constexpr int_t num_bytes_v<void>{};

template <typename T>
constexpr static int_t num_bits_v{num_bytes_v<T> * 8};

template <typename>
constexpr static bool dependent_false_v{false};

enum class log_level_t : int_t {
  error = 0,
  warning = 1,
  info = 2,
  debug = 3,
};

enum class check_level_t : int_t {
  none = 0,
  release = 1,
  prototype = 2,
  debug = 3
};

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

class logic_error : public std::logic_error {
 public:
  template <typename... Args>
  logic_error(const char reason[], const char file[], int_t line,  const char func[], const char expr[], Args&&... args)
    : std::logic_error(build_message(reason, file, line, func, expr, std::forward<Args>(args)...)),
    reason_(reason), file_(file), line_(line), func_(func), expr_(expr) {}

  const char* reason() const noexcept { return reason_; }
  const char* file() const noexcept { return file_; }
  int_t line() const noexcept { return line_; }
  const char* function() const noexcept { return func_; }
  const char* expression() const noexcept { return expr_; }

  template <typename... Args>
  static std::string build_message(const char reason[], const char file[], int_t line,  const char func[], const char expr[], Args&&... args) {
    std::ostringstream os;
    render_args(os, reason, " [", file, ':', line, " @ ", func, " -> ", expr, ']');
    if constexpr (sizeof...(Args) > 0) {
      render_args(os, ": ", args...);
    }
    return os.str();
  }

 protected:
  const char* reason_;
  const char* file_;
  int_t line_;
  const char* func_;
  const char* expr_;
};

constexpr bitmask_t make_aligned_mask(int_t n, int_t alignment) noexcept {
  NVHM_ASSERT_(n > 0 && has_single_bit(n));
  NVHM_ASSERT_(n > 0 && has_single_bit(alignment));
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
constexpr static T round_up(T x, T n) noexcept { return ceil_div(x, n) * n; }

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

constexpr static psl_t inf_psl{std::numeric_limits<psl_t>::max()};

template<class InOutIt, class Size, class UnaryOp>
constexpr InOutIt transform_n(InOutIt first, Size n, UnaryOp f) {
  return std::transform(first, first + n, first, f);
}

template<class InIt0, class InIt1, class Size>
constexpr bool equal_n(InIt0 first0, InIt1 first1, Size n) {
  return std::equal(first0, first0 + n, first1);
}

enum class flags_t : uint_t {
  none = 0,
  blobs = 1,  // The map allocates and maintains a blob storage.
  duplicates = 2,  // The map/set allows duplicate keys.
  aggressive_prefetch = 4,  // Be slightly more aggressive wehn prefetching.
  auto_scrub = 8,  // Occasional stop-the-world `scrub` upon `erase` to speedup `find`.
  auto_shrink = 16,  // Occasional stop-the-world `shrink` in `erase` to speedup `find`.
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
struct func_traits<Ret(C::*)(Args...) const> {
  using result_type = Ret;

  template <size_t I>
  struct arg {
    static_assert(I < sizeof...(Args), "Argument index out of range!");

    using type = typename std::tuple_element<I, std::tuple<Args...>>::type;
  };
};

// For a non-`const member `operator()`
template <typename C, typename Ret, typename... Args>
struct func_traits<Ret(C::*)(Args...)> {
  using result_type = Ret;

  template <size_t I>
  struct arg {
    static_assert(I < sizeof...(Args), "Argument index out of range!");

    using type = typename std::tuple_element<I, std::tuple<Args...>>::type;
  };
};

// For free functions and function pointers.
template <typename Ret, typename... Args>
struct func_traits<Ret(*)(Args...)> {
  using result_type = Ret;

  template <size_t I>
  struct arg {
    static_assert(I < sizeof...(Args), "Argument index out of range!");

    using type = typename std::tuple_element<I, std::tuple<Args...>>::type;
  };
};

template <typename F, size_t I>
using arg_n_t = typename func_traits<std::decay_t<F>>::template arg<I>::type;

enum arg_type_t {
  value,
  lvalue_ref,
  const_lvalue_ref,
  rvalue_ref
};

constexpr const char* to_string(arg_type_t at) {
  switch (at) {
    case arg_type_t::value:
      return "value";
    case arg_type_t::lvalue_ref:
      return "lvalue_ref";
    case arg_type_t::const_lvalue_ref:
      return "const_lvalue_ref";
    case arg_type_t::rvalue_ref:
      return "rvalue_ref";
  }
  return "error";
}

inline std::ostream& operator<<(std::ostream& os, arg_type_t at) {
  return os << to_string(at);
}

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
