/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#ifndef DESUL_ATOMICS_GENERIC_HPP_
#define DESUL_ATOMICS_GENERIC_HPP_

#include "desul/atomics/Common.hpp"
#include "desul/atomics/Compare_Exchange.hpp"
#include "desul/atomics/Lock_Array.hpp"
#include "desul/atomics/Macros.hpp"

#include <type_traits>

// Combination operands to be used in an Compare and Exchange based atomic
// operation
namespace desul {
namespace Impl {

template <class Scalar1, class Scalar2>
struct MaxOper {
  DESUL_FORCEINLINE_FUNCTION
  static Scalar1 apply(const Scalar1& val1, const Scalar2& val2) {
    return (val1 > val2 ? val1 : val2);
  }
  DESUL_FORCEINLINE_FUNCTION
  static constexpr bool check_early_exit(Scalar1 const& val1, Scalar2 const& val2) {
    return val1 > val2;
  }
};

template <class Scalar1, class Scalar2>
struct MinOper {
  DESUL_FORCEINLINE_FUNCTION
  static Scalar1 apply(const Scalar1& val1, const Scalar2& val2) {
    return (val1 < val2 ? val1 : val2);
  }
  DESUL_FORCEINLINE_FUNCTION
  static constexpr bool check_early_exit(Scalar1 const& val1, Scalar2 const& val2) {
    return val1 < val2;
  }
};

template <typename Op, typename Scalar1, typename Scalar2, typename = bool>
struct may_exit_early : std::false_type {};

// This exit early optimization causes weird compiler errors with MSVC 2019
#ifndef DESUL_HAVE_MSVC_ATOMICS
template <typename Op, typename Scalar1, typename Scalar2>
struct may_exit_early<Op,
                      Scalar1,
                      Scalar2,
                      decltype(Op::check_early_exit(std::declval<Scalar1 const&>(),
                                                    std::declval<Scalar2 const&>()))>
    : std::true_type {};
#endif

template <typename Op, typename Scalar1, typename Scalar2>
constexpr DESUL_FUNCTION
    typename std::enable_if<may_exit_early<Op, Scalar1, Scalar2>::value, bool>::type
    check_early_exit(Op const&, Scalar1 const& val1, Scalar2 const& val2) {
  return Op::check_early_exit(val1, val2);
}

template <typename Op, typename Scalar1, typename Scalar2>
constexpr DESUL_FUNCTION
    typename std::enable_if<!may_exit_early<Op, Scalar1, Scalar2>::value, bool>::type
    check_early_exit(Op const&, Scalar1 const&, Scalar2 const&) {
  return false;
}

template <class Scalar1, class Scalar2>
struct AddOper {
  DESUL_FORCEINLINE_FUNCTION
  static Scalar1 apply(const Scalar1& val1, const Scalar2& val2) { return val1 + val2; }
};

template <class Scalar1, class Scalar2>
struct SubOper {
  DESUL_FORCEINLINE_FUNCTION
  static Scalar1 apply(const Scalar1& val1, const Scalar2& val2) { return val1 - val2; }
};

template <class Scalar1, class Scalar2>
struct MulOper {
  DESUL_FORCEINLINE_FUNCTION
  static Scalar1 apply(const Scalar1& val1, const Scalar2& val2) { return val1 * val2; }
};

template <class Scalar1, class Scalar2>
struct DivOper {
  DESUL_FORCEINLINE_FUNCTION
  static Scalar1 apply(const Scalar1& val1, const Scalar2& val2) { return val1 / val2; }
};

template <class Scalar1, class Scalar2>
struct ModOper {
  DESUL_FORCEINLINE_FUNCTION
  static Scalar1 apply(const Scalar1& val1, const Scalar2& val2) { return val1 % val2; }
};

template <class Scalar1, class Scalar2>
struct AndOper {
  DESUL_FORCEINLINE_FUNCTION
  static Scalar1 apply(const Scalar1& val1, const Scalar2& val2) { return val1 & val2; }
};

template <class Scalar1, class Scalar2>
struct OrOper {
  DESUL_FORCEINLINE_FUNCTION
  static Scalar1 apply(const Scalar1& val1, const Scalar2& val2) { return val1 | val2; }
};

template <class Scalar1, class Scalar2>
struct XorOper {
  DESUL_FORCEINLINE_FUNCTION
  static Scalar1 apply(const Scalar1& val1, const Scalar2& val2) { return val1 ^ val2; }
};

template <class Scalar1, class Scalar2>
struct NandOper {
  DESUL_FORCEINLINE_FUNCTION
  static Scalar1 apply(const Scalar1& val1, const Scalar2& val2) {
    return ~(val1 & val2);
  }
};

template <class Scalar1, class Scalar2>
struct LShiftOper {
  DESUL_FORCEINLINE_FUNCTION
  static Scalar1 apply(const Scalar1& val1, const Scalar2& val2) {
    return val1 << val2;
  }
};

template <class Scalar1, class Scalar2>
struct RShiftOper {
  DESUL_FORCEINLINE_FUNCTION
  static Scalar1 apply(const Scalar1& val1, const Scalar2& val2) {
    return val1 >> val2;
  }
};

template <class Scalar1, class Scalar2>
struct IncModOper {
  DESUL_FORCEINLINE_FUNCTION
  static Scalar1 apply(const Scalar1& val1, const Scalar2& val2) {
    return ((val1 >= val2) ? Scalar1(0) : val1 + Scalar1(1));
  }
};

template <class Scalar1, class Scalar2>
struct DecModOper {
  DESUL_FORCEINLINE_FUNCTION
  static Scalar1 apply(const Scalar1& val1, const Scalar2& val2) {
    return (((val1 == Scalar1(0)) | (val1 > val2)) ? val2 : (val1 - Scalar1(1)));
  }
};

template <class Scalar1, class Scalar2>
struct StoreOper {
  DESUL_FORCEINLINE_FUNCTION
  static Scalar1 apply(const Scalar1&, const Scalar2& val2) { return val2; }
};

template <class Scalar1, class Scalar2>
struct LoadOper {
  DESUL_FORCEINLINE_FUNCTION
  static Scalar1 apply(const Scalar1& val1, const Scalar2&) { return val1; }
};

}  // namespace Impl
}  // namespace desul

#include <desul/atomics/Generic_Device.hpp>
#include <desul/atomics/Generic_Host.hpp>
#ifdef DESUL_HAVE_CUDA_ATOMICS
#include <desul/atomics/CUDA.hpp>
#endif
#include <desul/atomics/GCC.hpp>
#ifdef DESUL_HAVE_HIP_ATOMICS
#include <desul/atomics/HIP.hpp>
#endif
#include <desul/atomics/OpenMP.hpp>
#include <desul/atomics/SYCL.hpp>

namespace desul {

template <class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION void atomic_thread_fence(MemoryOrder order, MemoryScope scope) {
  DESUL_IF_ON_DEVICE(return Impl::device_atomic_thread_fence(order, scope);)
  DESUL_IF_ON_HOST(return Impl::host_atomic_thread_fence(order, scope);)
}
template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T
atomic_exchange(T* dest, T val, MemoryOrder order, MemoryScope scope) {
  DESUL_IF_ON_DEVICE(return Impl::device_atomic_exchange(dest, val, order, scope);)
  DESUL_IF_ON_HOST(return Impl::host_atomic_exchange(dest, val, order, scope);)
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T
atomic_compare_exchange(T* dest, T cmp, T val, MemoryOrder order, MemoryScope scope) {
  DESUL_IF_ON_DEVICE(
      return Impl::device_atomic_compare_exchange(dest, cmp, val, order, scope);)
  DESUL_IF_ON_HOST(
      return Impl::host_atomic_compare_exchange(dest, cmp, val, order, scope);)
}

// Fetch_Oper atomics: return value before operation
template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T
atomic_fetch_add(T* const dest, const T val, MemoryOrder order, MemoryScope scope) {
  DESUL_IF_ON_DEVICE(return Impl::device_atomic_fetch_add(dest, val, order, scope);)
  DESUL_IF_ON_HOST(return Impl::host_atomic_fetch_add(dest, val, order, scope);)
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T
atomic_fetch_sub(T* const dest, const T val, MemoryOrder order, MemoryScope scope) {
  DESUL_IF_ON_DEVICE(return Impl::device_atomic_fetch_sub(dest, val, order, scope);)
  DESUL_IF_ON_HOST(return Impl::host_atomic_fetch_sub(dest, val, order, scope);)
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T
atomic_fetch_max(T* const dest, const T val, MemoryOrder order, MemoryScope scope) {
  DESUL_IF_ON_DEVICE(return Impl::device_atomic_fetch_max(dest, val, order, scope);)
  DESUL_IF_ON_HOST(return Impl::host_atomic_fetch_max(dest, val, order, scope);)
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T
atomic_fetch_min(T* const dest, const T val, MemoryOrder order, MemoryScope scope) {
  DESUL_IF_ON_DEVICE(return Impl::device_atomic_fetch_min(dest, val, order, scope);)
  DESUL_IF_ON_HOST(return Impl::host_atomic_fetch_min(dest, val, order, scope);)
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T
atomic_fetch_mul(T* const dest, const T val, MemoryOrder order, MemoryScope scope) {
  DESUL_IF_ON_DEVICE(return Impl::device_atomic_fetch_mul(dest, val, order, scope);)
  DESUL_IF_ON_HOST(return Impl::host_atomic_fetch_mul(dest, val, order, scope);)
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T
atomic_fetch_div(T* const dest, const T val, MemoryOrder order, MemoryScope scope) {
  DESUL_IF_ON_DEVICE(return Impl::device_atomic_fetch_div(dest, val, order, scope);)
  DESUL_IF_ON_HOST(return Impl::host_atomic_fetch_div(dest, val, order, scope);)
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T
atomic_fetch_mod(T* const dest, const T val, MemoryOrder order, MemoryScope scope) {
  DESUL_IF_ON_DEVICE(return Impl::device_atomic_fetch_mod(dest, val, order, scope);)
  DESUL_IF_ON_HOST(return Impl::host_atomic_fetch_mod(dest, val, order, scope);)
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T
atomic_fetch_and(T* const dest, const T val, MemoryOrder order, MemoryScope scope) {
  DESUL_IF_ON_DEVICE(return Impl::device_atomic_fetch_and(dest, val, order, scope);)
  DESUL_IF_ON_HOST(return Impl::host_atomic_fetch_and(dest, val, order, scope);)
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T
atomic_fetch_or(T* const dest, const T val, MemoryOrder order, MemoryScope scope) {
  DESUL_IF_ON_DEVICE(return Impl::device_atomic_fetch_or(dest, val, order, scope);)
  DESUL_IF_ON_HOST(return Impl::host_atomic_fetch_or(dest, val, order, scope);)
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T
atomic_fetch_xor(T* const dest, const T val, MemoryOrder order, MemoryScope scope) {
  DESUL_IF_ON_DEVICE(return Impl::device_atomic_fetch_xor(dest, val, order, scope);)
  DESUL_IF_ON_HOST(return Impl::host_atomic_fetch_xor(dest, val, order, scope);)
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T
atomic_fetch_nand(T* const dest, const T val, MemoryOrder order, MemoryScope scope) {
  DESUL_IF_ON_DEVICE(return Impl::device_atomic_fetch_nand(dest, val, order, scope);)
  DESUL_IF_ON_HOST(return Impl::host_atomic_fetch_nand(dest, val, order, scope);)
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T atomic_fetch_lshift(T* const dest,
                                            const unsigned int val,
                                            MemoryOrder order,
                                            MemoryScope scope) {
  DESUL_IF_ON_DEVICE(return Impl::device_atomic_fetch_lshift(dest, val, order, scope);)
  DESUL_IF_ON_HOST(return Impl::host_atomic_fetch_lshift(dest, val, order, scope);)
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T atomic_fetch_rshift(T* const dest,
                                            const unsigned int val,
                                            MemoryOrder order,
                                            MemoryScope scope) {
  DESUL_IF_ON_DEVICE(return Impl::device_atomic_fetch_rshift(dest, val, order, scope);)
  DESUL_IF_ON_HOST(return Impl::host_atomic_fetch_rshift(dest, val, order, scope);)
}

// Oper Fetch atomics: return value after operation
template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T
atomic_add_fetch(T* const dest, const T val, MemoryOrder order, MemoryScope scope) {
  DESUL_IF_ON_DEVICE(return Impl::device_atomic_and_fetch(dest, val, order, scope);)
  DESUL_IF_ON_HOST(return Impl::host_atomic_and_fetch(dest, val, order, scope);)
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T
atomic_sub_fetch(T* const dest, const T val, MemoryOrder order, MemoryScope scope) {
  DESUL_IF_ON_DEVICE(return Impl::device_atomic_sub_fetch(dest, val, order, scope);)
  DESUL_IF_ON_HOST(return Impl::host_atomic_sub_fetch(dest, val, order, scope);)
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T
atomic_max_fetch(T* const dest, const T val, MemoryOrder order, MemoryScope scope) {
  DESUL_IF_ON_DEVICE(return Impl::device_atomic_max_fetch(dest, val, order, scope);)
  DESUL_IF_ON_HOST(return Impl::host_atomic_max_fetch(dest, val, order, scope);)
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T
atomic_min_fetch(T* const dest, const T val, MemoryOrder order, MemoryScope scope) {
  DESUL_IF_ON_DEVICE(return Impl::device_atomic_min_fetch(dest, val, order, scope);)
  DESUL_IF_ON_HOST(return Impl::host_atomic_min_fetch(dest, val, order, scope);)
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T
atomic_mul_fetch(T* const dest, const T val, MemoryOrder order, MemoryScope scope) {
  DESUL_IF_ON_DEVICE(return Impl::device_atomic_mul_fetch(dest, val, order, scope);)
  DESUL_IF_ON_HOST(return Impl::host_atomic_mul_fetch(dest, val, order, scope);)
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T
atomic_div_fetch(T* const dest, const T val, MemoryOrder order, MemoryScope scope) {
  DESUL_IF_ON_DEVICE(return Impl::device_atomic_div_fetch(dest, val, order, scope);)
  DESUL_IF_ON_HOST(return Impl::host_atomic_div_fetch(dest, val, order, scope);)
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T
atomic_mod_fetch(T* const dest, const T val, MemoryOrder order, MemoryScope scope) {
  DESUL_IF_ON_DEVICE(return Impl::device_atomic_mod_fetch(dest, val, order, scope);)
  DESUL_IF_ON_HOST(return Impl::host_atomic_mod_fetch(dest, val, order, scope);)
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T
atomic_and_fetch(T* const dest, const T val, MemoryOrder order, MemoryScope scope) {
  DESUL_IF_ON_DEVICE(return Impl::device_atomic_and_fetch(dest, val, order, scope);)
  DESUL_IF_ON_HOST(return Impl::host_atomic_and_fetch(dest, val, order, scope);)
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T
atomic_or_fetch(T* const dest, const T val, MemoryOrder order, MemoryScope scope) {
  DESUL_IF_ON_DEVICE(return Impl::device_atomic_or_fetch(dest, val, order, scope);)
  DESUL_IF_ON_HOST(return Impl::host_atomic_or_fetch(dest, val, order, scope);)
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T
atomic_xor_fetch(T* const dest, const T val, MemoryOrder order, MemoryScope scope) {
  DESUL_IF_ON_DEVICE(return Impl::device_atomic_xor_fetch(dest, val, order, scope);)
  DESUL_IF_ON_HOST(return Impl::host_atomic_xor_fetch(dest, val, order, scope);)
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T
atomic_nand_fetch(T* const dest, const T val, MemoryOrder order, MemoryScope scope) {
  DESUL_IF_ON_DEVICE(return Impl::device_atomic_nand_fetch(dest, val, order, scope);)
  DESUL_IF_ON_HOST(return Impl::host_atomic_nand_fetch(dest, val, order, scope);)
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T atomic_lshift_fetch(T* const dest,
                                            const unsigned int val,
                                            MemoryOrder order,
                                            MemoryScope scope) {
  DESUL_IF_ON_DEVICE(return Impl::device_atomic_lshift_fetch(dest, val, order, scope);)
  DESUL_IF_ON_HOST(return Impl::host_atomic_lshift_fetch(dest, val, order, scope);)
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T atomic_rshift_fetch(T* const dest,
                                            const unsigned int val,
                                            MemoryOrder order,
                                            MemoryScope scope) {
  DESUL_IF_ON_DEVICE(return Impl::device_atomic_rshift_fetch(dest, val, order, scope);)
  DESUL_IF_ON_HOST(return Impl::host_atomic_rshift_fetch(dest, val, order, scope);)
}

// Other atomics

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T atomic_load(const T* const dest,
                                    MemoryOrder order,
                                    MemoryScope scope) {
  DESUL_IF_ON_DEVICE(return Impl::device_atomic_load(dest, order, scope);)
  DESUL_IF_ON_HOST(return Impl::host_atomic_load(dest, order, scope);)
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION void atomic_store(T* const dest,
                                        const T val,
                                        MemoryOrder order,
                                        MemoryScope scope) {
  DESUL_IF_ON_DEVICE(return Impl::device_atomic_store(dest, val, order, scope);)
  DESUL_IF_ON_HOST(return Impl::host_atomic_store(dest, val, order, scope);)
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION void atomic_add(T* const dest,
                                      const T val,
                                      MemoryOrder order,
                                      MemoryScope scope) {
  DESUL_IF_ON_DEVICE(return Impl::device_atomic_add(dest, val, order, scope);)
  DESUL_IF_ON_HOST(return Impl::host_atomic_add(dest, val, order, scope);)
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION void atomic_sub(T* const dest,
                                      const T val,
                                      MemoryOrder order,
                                      MemoryScope scope) {
  DESUL_IF_ON_DEVICE(return Impl::device_atomic_sub(dest, val, order, scope);)
  DESUL_IF_ON_HOST(return Impl::host_atomic_sub(dest, val, order, scope);)
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION void atomic_mul(T* const dest,
                                      const T val,
                                      MemoryOrder order,
                                      MemoryScope scope) {
  DESUL_IF_ON_DEVICE(return Impl::device_atomic_mul(dest, val, order, scope);)
  DESUL_IF_ON_HOST(return Impl::host_atomic_mul(dest, val, order, scope);)
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION void atomic_div(T* const dest,
                                      const T val,
                                      MemoryOrder order,
                                      MemoryScope scope) {
  DESUL_IF_ON_DEVICE(return Impl::device_atomic_div(dest, val, order, scope);)
  DESUL_IF_ON_HOST(return Impl::host_atomic_div(dest, val, order, scope);)
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION void atomic_min(T* const dest,
                                      const T val,
                                      MemoryOrder order,
                                      MemoryScope scope) {
  DESUL_IF_ON_DEVICE(return Impl::device_atomic_min(dest, val, order, scope);)
  DESUL_IF_ON_HOST(return Impl::host_atomic_min(dest, val, order, scope);)
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION void atomic_max(T* const dest,
                                      const T val,
                                      MemoryOrder order,
                                      MemoryScope scope) {
  DESUL_IF_ON_DEVICE(return Impl::device_atomic_max(dest, val, order, scope);)
  DESUL_IF_ON_HOST(return Impl::host_atomic_max(dest, val, order, scope);)
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T atomic_inc_fetch(T* const dest,
                                         MemoryOrder order,
                                         MemoryScope scope) {
  DESUL_IF_ON_DEVICE(return Impl::device_atomic_inc_fetch(dest, order, scope);)
  DESUL_IF_ON_HOST(return Impl::host_atomic_inc_fetch(dest, order, scope);)
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T atomic_dec_fetch(T* const dest,
                                         MemoryOrder order,
                                         MemoryScope scope) {
  DESUL_IF_ON_DEVICE(return Impl::device_atomic_dec_fetch(dest, order, scope);)
  DESUL_IF_ON_HOST(return Impl::host_atomic_dec_fetch(dest, order, scope);)
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T atomic_fetch_inc(T* const dest,
                                         MemoryOrder order,
                                         MemoryScope scope) {
  DESUL_IF_ON_DEVICE(return Impl::device_atomic_fetch_inc(dest, order, scope);)
  DESUL_IF_ON_HOST(return Impl::host_atomic_fetch_inc(dest, order, scope);)
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T
atomic_fetch_inc_mod(T* const dest, T val, MemoryOrder order, MemoryScope scope) {
  DESUL_IF_ON_DEVICE(return Impl::device_atomic_fetch_inc_mod(dest, val, order, scope);)
  DESUL_IF_ON_HOST(return Impl::host_atomic_fetch_inc_mod(dest, val, order, scope);)
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T atomic_fetch_dec(T* const dest,
                                         MemoryOrder order,
                                         MemoryScope scope) {
  DESUL_IF_ON_DEVICE(return Impl::device_atomic_fetch_dec(dest, order, scope);)
  DESUL_IF_ON_HOST(return Impl::host_atomic_fetch_dec(dest, order, scope);)
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T
atomic_fetch_dec_mod(T* const dest, T val, MemoryOrder order, MemoryScope scope) {
  DESUL_IF_ON_DEVICE(return Impl::device_atomic_fetch_dec_mod(dest, val, order, scope);)
  DESUL_IF_ON_HOST(return Impl::host_atomic_fetch_dec_mod(dest, val, order, scope);)
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION void atomic_inc(T* const dest,
                                      MemoryOrder order,
                                      MemoryScope scope) {
  DESUL_IF_ON_DEVICE(return Impl::device_atomic_inc(dest, order, scope);)
  DESUL_IF_ON_HOST(return Impl::host_atomic_inc(dest, order, scope);)
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION void atomic_dec(T* const dest,
                                      MemoryOrder order,
                                      MemoryScope scope) {
  DESUL_IF_ON_DEVICE(return Impl::device_atomic_dec(dest, order, scope);)
  DESUL_IF_ON_HOST(return Impl::host_atomic_dec(dest, order, scope);)
}

// FIXME
template <typename T,
          class SuccessMemoryOrder,
          class FailureMemoryOrder,
          class MemoryScope>
DESUL_INLINE_FUNCTION bool atomic_compare_exchange_strong(
    T* const dest,
    T& expected,
    T desired,
    SuccessMemoryOrder success,
    FailureMemoryOrder /*failure*/,
    MemoryScope scope) {
  T const old = atomic_compare_exchange(dest, expected, desired, success, scope);
  if (old != expected) {
    expected = old;
    return false;
  } else {
    return true;
  }
}

template <typename T,
          class SuccessMemoryOrder,
          class FailureMemoryOrder,
          class MemoryScope>
DESUL_INLINE_FUNCTION bool atomic_compare_exchange_weak(T* const dest,
                                                        T& expected,
                                                        T desired,
                                                        SuccessMemoryOrder success,
                                                        FailureMemoryOrder failure,
                                                        MemoryScope scope) {
  return atomic_compare_exchange_strong(
      dest, expected, desired, success, failure, scope);
}

}  // namespace desul

#if defined(__GNUC__) && (!defined(__clang__))
#pragma GCC diagnostic pop
#endif
#endif
