/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#ifndef DESUL_ATOMICS_GENERIC_DEVICE_HPP_
#define DESUL_ATOMICS_GENERIC_DEVICE_HPP_

#include <type_traits>
#if defined(__GNUC__) && (!defined(__clang__))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

namespace desul {
namespace Impl {

template <class Oper,
          typename T,
          class MemoryOrder,
          class MemoryScope,
          // equivalent to:
          //   requires device_atomic_always_lock_free(sizeof(T))
          std::enable_if_t<atomic_always_lock_free(sizeof(T)), int> = 0>
DESUL_INLINE_FUNCTION T
device_atomic_fetch_oper(const Oper& op,
                         T* const dest,
                         dont_deduce_this_parameter_t<const T> val,
                         MemoryOrder order,
                         MemoryScope scope) {
  using cas_t = typename atomic_compare_exchange_type<sizeof(T)>::type;
  cas_t oldval = reinterpret_cast<cas_t&>(*dest);
  cas_t assume = oldval;

  do {
    if (Impl::check_early_exit(op, reinterpret_cast<T&>(oldval), val))
      return reinterpret_cast<T&>(oldval);
    assume = oldval;
    T newval = op.apply(reinterpret_cast<T&>(assume), val);
    oldval = device_atomic_compare_exchange(reinterpret_cast<cas_t*>(dest),
                                            assume,
                                            reinterpret_cast<cas_t&>(newval),
                                            order,
                                            scope);
  } while (assume != oldval);

  return reinterpret_cast<T&>(oldval);
}

template <class Oper,
          typename T,
          class MemoryOrder,
          class MemoryScope,
          // equivalent to:
          //   requires device_atomic_always_lock_free(sizeof(T))
          std::enable_if_t<atomic_always_lock_free(sizeof(T)), int> = 0>
DESUL_INLINE_FUNCTION T
device_atomic_oper_fetch(const Oper& op,
                         T* const dest,
                         dont_deduce_this_parameter_t<const T> val,
                         MemoryOrder order,
                         MemoryScope scope) {
  using cas_t = typename atomic_compare_exchange_type<sizeof(T)>::type;
  cas_t oldval = reinterpret_cast<cas_t&>(*dest);
  T newval = val;
  cas_t assume = oldval;
  do {
    if (Impl::check_early_exit(op, reinterpret_cast<T&>(oldval), val))
      return reinterpret_cast<T&>(oldval);
    assume = oldval;
    newval = op.apply(reinterpret_cast<T&>(assume), val);
    oldval = device_atomic_compare_exchange(reinterpret_cast<cas_t*>(dest),
                                            assume,
                                            reinterpret_cast<cas_t&>(newval),
                                            order,
                                            scope);
  } while (assume != oldval);

  return newval;
}

template <class Oper,
          typename T,
          class MemoryOrder,
          class MemoryScope,
          // equivalent to:
          //   requires !atomic_always_lock_free(sizeof(T))
          std::enable_if_t<!atomic_always_lock_free(sizeof(T)), int> = 0>
DESUL_INLINE_FUNCTION T
device_atomic_fetch_oper(const Oper& op,
                         T* const dest,
                         dont_deduce_this_parameter_t<const T> val,
                         MemoryOrder /*order*/,
                         MemoryScope scope) {
  // This is a way to avoid dead lock in a warp or wave front
  T return_val;
  int done = 0;
#ifdef __HIPCC__
  unsigned long long int active = DESUL_IMPL_BALLOT_MASK(1);
  unsigned long long int done_active = 0;
  while (active != done_active) {
    if (!done) {
      if (Impl::lock_address_hip((void*)dest, scope)) {
        device_atomic_thread_fence(MemoryOrderAcquire(), scope);
        return_val = *dest;
        *dest = op.apply(return_val, val);
        device_atomic_thread_fence(MemoryOrderRelease(), scope);
        Impl::unlock_address_hip((void*)dest, scope);
        done = 1;
      }
    }
    done_active = DESUL_IMPL_BALLOT_MASK(done);
  }
  return return_val;
// FIXME_SYCL not implemented
#elif defined(__SYCL_DEVICE_ONLY__)
  (void)op;
  (void)dest;
  (void)scope;
  (void)return_val;
  (void)done;

  assert(false);
  return val;
#else
  unsigned int mask = DESUL_IMPL_ACTIVEMASK;
  unsigned int active = DESUL_IMPL_BALLOT_MASK(mask, 1);
  unsigned int done_active = 0;
  while (active != done_active) {
    if (!done) {
      if (Impl::lock_address_cuda((void*)dest, scope)) {
        device_atomic_thread_fence(MemoryOrderAcquire(), scope);
        return_val = *dest;
        *dest = op.apply(return_val, val);
        device_atomic_thread_fence(MemoryOrderRelease(), scope);
        Impl::unlock_address_cuda((void*)dest, scope);
        done = 1;
      }
    }
    done_active = DESUL_IMPL_BALLOT_MASK(mask, done);
  }
  return return_val;
#endif
}

template <class Oper,
          typename T,
          class MemoryOrder,
          class MemoryScope,
          // equivalent to:
          //   requires !atomic_always_lock_free(sizeof(T))
          std::enable_if_t<!atomic_always_lock_free(sizeof(T)), int> = 0>
DESUL_INLINE_FUNCTION T
device_atomic_oper_fetch(const Oper& op,
                         T* const dest,
                         dont_deduce_this_parameter_t<const T> val,
                         MemoryOrder /*order*/,
                         MemoryScope scope) {
  // This is a way to avoid dead lock in a warp or wave front
  T return_val;
  int done = 0;
#ifdef __HIPCC__
  unsigned long long int active = DESUL_IMPL_BALLOT_MASK(1);
  unsigned long long int done_active = 0;
  while (active != done_active) {
    if (!done) {
      if (Impl::lock_address_hip((void*)dest, scope)) {
        device_atomic_thread_fence(MemoryOrderAcquire(), scope);
        return_val = op.apply(*dest, val);
        *dest = return_val;
        device_atomic_thread_fence(MemoryOrderRelease(), scope);
        Impl::unlock_address_hip((void*)dest, scope);
        done = 1;
      }
    }
    done_active = DESUL_IMPL_BALLOT_MASK(done);
  }
  return return_val;
  // FIXME_SYCL not implemented
#elif defined(__SYCL_DEVICE_ONLY__)
  (void)op;
  (void)dest;
  (void)scope;
  (void)done;

  assert(false);
  return val;
#else
  unsigned int mask = DESUL_IMPL_ACTIVEMASK;
  unsigned int active = DESUL_IMPL_BALLOT_MASK(mask, 1);
  unsigned int done_active = 0;
  while (active != done_active) {
    if (!done) {
      if (Impl::lock_address_cuda((void*)dest, scope)) {
        device_atomic_thread_fence(MemoryOrderAcquire(), scope);
        return_val = op.apply(*dest, val);
        *dest = return_val;
        device_atomic_thread_fence(MemoryOrderRelease(), scope);
        Impl::unlock_address_cuda((void*)dest, scope);
        done = 1;
      }
    }
    done_active = DESUL_IMPL_BALLOT_MASK(mask, done);
  }
  return return_val;
#endif
}

template <class Oper, typename T, class MemoryOrder>
DESUL_INLINE_FUNCTION T
device_atomic_fetch_oper(const Oper& op,
                         T* const dest,
                         dont_deduce_this_parameter_t<const T> val,
                         MemoryOrder /*order*/,
                         MemoryScopeCaller /*scope*/) {
  T oldval = *dest;
  *dest = op.apply(oldval, val);
  return oldval;
}

template <class Oper, typename T, class MemoryOrder>
DESUL_INLINE_FUNCTION T
device_atomic_oper_fetch(const Oper& op,
                         T* const dest,
                         dont_deduce_this_parameter_t<const T> val,
                         MemoryOrder /*order*/,
                         MemoryScopeCaller /*scope*/) {
  T oldval = *dest;
  T newval = op.apply(oldval, val);
  *dest = newval;
  return newval;
}

// Fetch_Oper atomics: return value before operation
template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T device_atomic_fetch_add(T* const dest,
                                                const T val,
                                                MemoryOrder order,
                                                MemoryScope scope) {
  return device_atomic_fetch_oper(Impl::AddOper<T, const T>(), dest, val, order, scope);
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T device_atomic_fetch_sub(T* const dest,
                                                const T val,
                                                MemoryOrder order,
                                                MemoryScope scope) {
  return device_atomic_fetch_oper(Impl::SubOper<T, const T>(), dest, val, order, scope);
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T device_atomic_fetch_max(T* const dest,
                                                const T val,
                                                MemoryOrder order,
                                                MemoryScope scope) {
  return device_atomic_fetch_oper(Impl::MaxOper<T, const T>(), dest, val, order, scope);
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T device_atomic_fetch_min(T* const dest,
                                                const T val,
                                                MemoryOrder order,
                                                MemoryScope scope) {
  return device_atomic_fetch_oper(Impl::MinOper<T, const T>(), dest, val, order, scope);
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T device_atomic_fetch_mul(T* const dest,
                                                const T val,
                                                MemoryOrder order,
                                                MemoryScope scope) {
  return device_atomic_fetch_oper(Impl::MulOper<T, const T>(), dest, val, order, scope);
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T device_atomic_fetch_div(T* const dest,
                                                const T val,
                                                MemoryOrder order,
                                                MemoryScope scope) {
  return device_atomic_fetch_oper(Impl::DivOper<T, const T>(), dest, val, order, scope);
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T device_atomic_fetch_mod(T* const dest,
                                                const T val,
                                                MemoryOrder order,
                                                MemoryScope scope) {
  return device_atomic_fetch_oper(Impl::ModOper<T, const T>(), dest, val, order, scope);
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T device_atomic_fetch_and(T* const dest,
                                                const T val,
                                                MemoryOrder order,
                                                MemoryScope scope) {
  return device_atomic_fetch_oper(Impl::AndOper<T, const T>(), dest, val, order, scope);
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T device_atomic_fetch_or(T* const dest,
                                               const T val,
                                               MemoryOrder order,
                                               MemoryScope scope) {
  return device_atomic_fetch_oper(Impl::OrOper<T, const T>(), dest, val, order, scope);
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T device_atomic_fetch_xor(T* const dest,
                                                const T val,
                                                MemoryOrder order,
                                                MemoryScope scope) {
  return device_atomic_fetch_oper(Impl::XorOper<T, const T>(), dest, val, order, scope);
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T device_atomic_fetch_nand(T* const dest,
                                                 const T val,
                                                 MemoryOrder order,
                                                 MemoryScope scope) {
  return device_atomic_fetch_oper(
      Impl::NandOper<T, const T>(), dest, val, order, scope);
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T device_atomic_fetch_lshift(T* const dest,
                                                   const unsigned int val,
                                                   MemoryOrder order,
                                                   MemoryScope scope) {
  return device_atomic_fetch_oper(
      Impl::LShiftOper<T, const unsigned int>(), dest, val, order, scope);
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T device_atomic_fetch_rshift(T* const dest,
                                                   const unsigned int val,
                                                   MemoryOrder order,
                                                   MemoryScope scope) {
  return device_atomic_fetch_oper(
      Impl::RShiftOper<T, const unsigned int>(), dest, val, order, scope);
}

// Oper Fetch atomics: return value after operation
template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T device_atomic_add_fetch(T* const dest,
                                                const T val,
                                                MemoryOrder order,
                                                MemoryScope scope) {
  return device_atomic_oper_fetch(Impl::AddOper<T, const T>(), dest, val, order, scope);
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T device_atomic_sub_fetch(T* const dest,
                                                const T val,
                                                MemoryOrder order,
                                                MemoryScope scope) {
  return device_atomic_oper_fetch(Impl::SubOper<T, const T>(), dest, val, order, scope);
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T device_atomic_max_fetch(T* const dest,
                                                const T val,
                                                MemoryOrder order,
                                                MemoryScope scope) {
  return device_atomic_oper_fetch(Impl::MaxOper<T, const T>(), dest, val, order, scope);
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T device_atomic_min_fetch(T* const dest,
                                                const T val,
                                                MemoryOrder order,
                                                MemoryScope scope) {
  return device_atomic_oper_fetch(Impl::MinOper<T, const T>(), dest, val, order, scope);
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T device_atomic_mul_fetch(T* const dest,
                                                const T val,
                                                MemoryOrder order,
                                                MemoryScope scope) {
  return device_atomic_oper_fetch(Impl::MulOper<T, const T>(), dest, val, order, scope);
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T device_atomic_div_fetch(T* const dest,
                                                const T val,
                                                MemoryOrder order,
                                                MemoryScope scope) {
  return device_atomic_oper_fetch(Impl::DivOper<T, const T>(), dest, val, order, scope);
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T device_atomic_mod_fetch(T* const dest,
                                                const T val,
                                                MemoryOrder order,
                                                MemoryScope scope) {
  return device_atomic_oper_fetch(Impl::ModOper<T, const T>(), dest, val, order, scope);
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T device_atomic_and_fetch(T* const dest,
                                                const T val,
                                                MemoryOrder order,
                                                MemoryScope scope) {
  return device_atomic_oper_fetch(Impl::AndOper<T, const T>(), dest, val, order, scope);
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T device_atomic_or_fetch(T* const dest,
                                               const T val,
                                               MemoryOrder order,
                                               MemoryScope scope) {
  return device_atomic_oper_fetch(Impl::OrOper<T, const T>(), dest, val, order, scope);
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T device_atomic_xor_fetch(T* const dest,
                                                const T val,
                                                MemoryOrder order,
                                                MemoryScope scope) {
  return device_atomic_oper_fetch(Impl::XorOper<T, const T>(), dest, val, order, scope);
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T device_atomic_nand_fetch(T* const dest,
                                                 const T val,
                                                 MemoryOrder order,
                                                 MemoryScope scope) {
  return device_atomic_oper_fetch(
      Impl::NandOper<T, const T>(), dest, val, order, scope);
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T device_atomic_lshift_fetch(T* const dest,
                                                   const unsigned int val,
                                                   MemoryOrder order,
                                                   MemoryScope scope) {
  return device_atomic_oper_fetch(
      Impl::LShiftOper<T, const unsigned int>(), dest, val, order, scope);
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T device_atomic_rshift_fetch(T* const dest,
                                                   const unsigned int val,
                                                   MemoryOrder order,
                                                   MemoryScope scope) {
  return device_atomic_oper_fetch(
      Impl::RShiftOper<T, const unsigned int>(), dest, val, order, scope);
}

// Other atomics

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T device_atomic_load(const T* const dest,
                                           MemoryOrder order,
                                           MemoryScope scope) {
  return device_atomic_fetch_oper(
      Impl::LoadOper<T, const T>(), const_cast<T*>(dest), T(), order, scope);
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION void device_atomic_store(T* const dest,
                                               const T val,
                                               MemoryOrder order,
                                               MemoryScope scope) {
  (void)device_atomic_fetch_oper(
      Impl::StoreOper<T, const T>(), dest, val, order, scope);
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION void device_atomic_add(T* const dest,
                                             const T val,
                                             MemoryOrder order,
                                             MemoryScope scope) {
  (void)device_atomic_fetch_add(dest, val, order, scope);
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION void device_atomic_sub(T* const dest,
                                             const T val,
                                             MemoryOrder order,
                                             MemoryScope scope) {
  (void)device_atomic_fetch_sub(dest, val, order, scope);
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION void device_atomic_mul(T* const dest,
                                             const T val,
                                             MemoryOrder order,
                                             MemoryScope scope) {
  (void)device_atomic_fetch_mul(dest, val, order, scope);
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION void device_atomic_div(T* const dest,
                                             const T val,
                                             MemoryOrder order,
                                             MemoryScope scope) {
  (void)device_atomic_fetch_div(dest, val, order, scope);
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION void device_atomic_min(T* const dest,
                                             const T val,
                                             MemoryOrder order,
                                             MemoryScope scope) {
  (void)device_atomic_fetch_min(dest, val, order, scope);
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION void device_atomic_max(T* const dest,
                                             const T val,
                                             MemoryOrder order,
                                             MemoryScope scope) {
  (void)device_atomic_fetch_max(dest, val, order, scope);
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T device_atomic_inc_fetch(T* const dest,
                                                MemoryOrder order,
                                                MemoryScope scope) {
  return device_atomic_add_fetch(dest, T(1), order, scope);
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T device_atomic_dec_fetch(T* const dest,
                                                MemoryOrder order,
                                                MemoryScope scope) {
  return device_atomic_sub_fetch(dest, T(1), order, scope);
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T device_atomic_fetch_inc(T* const dest,
                                                MemoryOrder order,
                                                MemoryScope scope) {
  return device_atomic_fetch_add(dest, T(1), order, scope);
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T device_atomic_fetch_inc_mod(T* const dest,
                                                    T val,
                                                    MemoryOrder order,
                                                    MemoryScope scope) {
  static_assert(std::is_unsigned<T>::value,
                "Signed types not supported by device_atomic_fetch_inc_mod.");
  return device_atomic_fetch_oper(
      Impl::IncModOper<T, const T>(), dest, val, order, scope);
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T device_atomic_fetch_dec(T* const dest,
                                                MemoryOrder order,
                                                MemoryScope scope) {
  return device_atomic_fetch_sub(dest, T(1), order, scope);
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION T device_atomic_fetch_dec_mod(T* const dest,
                                                    T val,
                                                    MemoryOrder order,
                                                    MemoryScope scope) {
  static_assert(std::is_unsigned<T>::value,
                "Signed types not supported by device_atomic_fetch_dec_mod.");
  return device_atomic_fetch_oper(
      Impl::DecModOper<T, const T>(), dest, val, order, scope);
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION void device_atomic_inc(T* const dest,
                                             MemoryOrder order,
                                             MemoryScope scope) {
  return device_atomic_add(dest, T(1), order, scope);
}

template <typename T, class MemoryOrder, class MemoryScope>
DESUL_INLINE_FUNCTION void device_atomic_dec(T* const dest,
                                             MemoryOrder order,
                                             MemoryScope scope) {
  return device_atomic_sub(dest, T(1), order, scope);
}

// FIXME
template <typename T,
          class SuccessMemoryOrder,
          class FailureMemoryOrder,
          class MemoryScope>
DESUL_INLINE_FUNCTION bool device_atomic_compare_exchange_strong(
    T* const dest,
    T& expected,
    T desired,
    SuccessMemoryOrder success,
    FailureMemoryOrder /*failure*/,
    MemoryScope scope) {
  T const old = device_atomic_compare_exchange(dest, expected, desired, success, scope);
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
DESUL_INLINE_FUNCTION bool device_atomic_compare_exchange_weak(
    T* const dest,
    T& expected,
    T desired,
    SuccessMemoryOrder success,
    FailureMemoryOrder failure,
    MemoryScope scope) {
  return device_atomic_compare_exchange_strong(
      dest, expected, desired, success, failure, scope);
}

}  // namespace Impl
}  // namespace desul

#if defined(__GNUC__) && (!defined(__clang__))
#pragma GCC diagnostic pop
#endif
#endif
