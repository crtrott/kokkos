/* 
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/
#ifndef DESUL_ATOMICS_COMPARE_EXCHANGE_OPENMP_HPP_
#define DESUL_ATOMICS_COMPARE_EXCHANGE_OPENMP_HPP_
#include "desul/atomics/Common.hpp"
#include <cstdio>
#include <omp.h>

#ifdef DESUL_HAVE_OPENMP_ATOMICS
namespace desul {

#if _OPENMP > 201800
// atomic_thread_fence for Core Scope
inline void atomic_thread_fence(MemoryOrderSeqCst, MemoryScopeCore) {
  // There is no seq_cst flush in OpenMP, isn't it the same anyway for fence?
  #pragma omp flush acq_rel
}
inline void atomic_thread_fence(MemoryOrderAcqRel, MemoryScopeCore) {
  #pragma omp flush acq_rel
}
inline void atomic_thread_fence(MemoryOrderRelease, MemoryScopeCore) {
  #pragma omp flush release
}
inline void atomic_thread_fence(MemoryOrderAcquire, MemoryScopeCore) {
  #pragma omp flush acquire
}
// atomic_thread_fence for Device Scope
inline void atomic_thread_fence(MemoryOrderSeqCst, MemoryScopeDevice) {
  // There is no seq_cst flush in OpenMP, isn't it the same anyway for fence?
  #pragma omp flush acq_rel
}
inline void atomic_thread_fence(MemoryOrderAcqRel, MemoryScopeDevice) {
  #pragma omp flush acq_rel
}
inline void atomic_thread_fence(MemoryOrderRelease, MemoryScopeDevice) {
  #pragma omp flush release
}
inline void atomic_thread_fence(MemoryOrderAcquire, MemoryScopeDevice) {
  #pragma omp flush acquire
}
#else
// atomic_thread_fence for Core Scope
inline void atomic_thread_fence(MemoryOrderSeqCst, MemoryScopeCore) {
  #pragma omp flush
}
inline void atomic_thread_fence(MemoryOrderAcqRel, MemoryScopeCore) {
  #pragma omp flush
}
inline void atomic_thread_fence(MemoryOrderRelease, MemoryScopeCore) {
  #pragma omp flush
}
inline void atomic_thread_fence(MemoryOrderAcquire, MemoryScopeCore) {
  #pragma omp flush
}
// atomic_thread_fence for Device Scope
inline void atomic_thread_fence(MemoryOrderSeqCst, MemoryScopeDevice) {
  #pragma omp flush
}
inline void atomic_thread_fence(MemoryOrderAcqRel, MemoryScopeDevice) {
  #pragma omp flush
}
inline void atomic_thread_fence(MemoryOrderRelease, MemoryScopeDevice) {
  #pragma omp flush
}
inline void atomic_thread_fence(MemoryOrderAcquire, MemoryScopeDevice) {
  #pragma omp flush
}
#endif

template <typename T, class MemoryOrder, class MemoryScope>
T atomic_exchange(
    T* dest, std::enable_if_t<Impl::atomic_always_lock_free(sizeof(T)),T> value, MemoryOrder, MemoryScope) {
  T return_val;
  if(!std::is_same<MemoryOrder,MemoryOrderRelaxed>::value)
    atomic_thread_fence(MemoryOrderAcquire(),MemoryScope());
  using cas_t = typename Impl::atomic_compare_exchange_type<sizeof(T)>::type;
  cas_t* dest_c = reinterpret_cast<cas_t*>(dest);
  cas_t& value_c = reinterpret_cast<cas_t&>(value);
  cas_t return_val_c;
  cas_t& x = *dest_c;
  #pragma omp atomic capture
  { return_val_c = x; x = value_c; }
  if(!std::is_same<MemoryOrder,MemoryOrderRelaxed>::value)
    atomic_thread_fence(MemoryOrderRelease(),MemoryScope());
  return reinterpret_cast<T&>(return_val_c);
}

// OpenMP doesn't have compare exchange, so we use build-ins and rely on testing that this works
// Note that means we test this in OpenMPTarget offload regions!
template <typename T, class MemoryOrder, class MemoryScope>
std::enable_if_t<Impl::atomic_always_lock_free(sizeof(T)),T> atomic_compare_exchange(
    T* dest, T compare, T value, MemoryOrder, MemoryScope) {
  using cas_t = typename Impl::atomic_compare_exchange_type<sizeof(T)>::type;
  cas_t retval = __sync_val_compare_and_swap(
     reinterpret_cast<volatile cas_t*>(dest), 
     reinterpret_cast<cas_t&>(compare), 
     reinterpret_cast<cas_t&>(value));
  return reinterpret_cast<T&>(retval);
}

} // namespace desul
#if defined(__clang__) && (__clang_major__>=7)
// Disable warning for large atomics on clang 7 and up (checked with godbolt)
// error: large atomic operation may incur significant performance penalty [-Werror,-Watomic-alignment]
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Watomic-alignment"
#endif

// Locks need lockfree compare_exchange, but non-lock-free compare_exchange needs the locks

#include <desul/atomics/Lock_Array_OpenMP.hpp>
namespace desul {

#pragma omp begin declare variant match(device = {kind(nohost)})
template <typename T, class MemoryOrder, class MemoryScope>
std::enable_if_t<!Impl::atomic_always_lock_free(sizeof(T)), T> atomic_compare_exchange(
    T* const dest, T compare, T value, MemoryOrder, MemoryScope scope) {
  // This is a way to avoid dead lock in a warp or wave front
  T return_val;
  unsigned int done = 0;
  unsigned int mask = Impl::openmp_mask();
  unsigned int done_active = 0;
  while (mask != done_active) {
    if (!done) {
      if (Impl::lock_address_openmp((void*)dest, scope)) {
        if(std::is_same<MemoryOrder,MemoryOrderSeqCst>::value) atomic_thread_fence(MemoryOrderRelease(),scope);
        atomic_thread_fence(MemoryOrderAcquire(),scope);
        return_val = *dest;
        if(return_val == compare) {
          *dest = value;
          atomic_thread_fence(MemoryOrderRelease(),scope);
        }
        Impl::unlock_address_openmp((void*)dest, scope);
        done = 1;
      }
    }
    done_active = Impl::openmp_ballot_mask(mask, done);
  }
  return return_val;
}
template <typename T, class MemoryOrder, class MemoryScope>
std::enable_if_t<!Impl::atomic_always_lock_free(sizeof(T)), T> atomic_exchange(
    T* const dest, T value, MemoryOrder, MemoryScope scope) {
  // This is a way to avoid dead lock in a warp or wave front
  T return_val;
  unsigned int done = 0;
  unsigned int mask = Impl::openmp_mask();
  unsigned int done_active = 0;
  while (mask != done_active) {
    if (!done) {
      if (Impl::lock_address_openmp((void*)dest, scope)) {
        if(std::is_same<MemoryOrder,MemoryOrderSeqCst>::value) atomic_thread_fence(MemoryOrderRelease(),scope);
        atomic_thread_fence(MemoryOrderAcquire(),scope);
        return_val = *dest;
        *dest = value;
        atomic_thread_fence(MemoryOrderRelease(),scope);
        Impl::unlock_address_openmp((void*)dest, scope);
        done = 1;
      }
    }
    done_active = Impl::openmp_ballot_mask(mask, done);
  }
  return return_val;
}
#pragma omp end declare variant

#pragma omp begin declare variant match(device = {kind(host)})
template <typename T, class MemoryOrder, class MemoryScope>
std::enable_if_t<!Impl::atomic_always_lock_free(sizeof(T)), T>
atomic_compare_exchange(T* dest, T compare, T value, MemoryOrder, MemoryScope) {
/*  (void)__atomic_compare_exchange(dest,
                                  &compare,
                                  &value,
                                  false,
                                  GCCMemoryOrder<MemoryOrder>::value,
                                  GCCMemoryOrder<MemoryOrder>::value);*/
  return compare;
}
#pragma omp end declare variant

#pragma omp begin declare variant match(device = {kind(host)})
template <typename T, class MemoryOrder, class MemoryScope>
std::enable_if_t<!Impl::atomic_always_lock_free(sizeof(T)), T>
atomic_exchange(T* dest, T value, MemoryOrder, MemoryScope) {
  T return_val;
/*  __atomic_exchange(dest,
                                  &value,
                                  &return_val,
                                  GCCMemoryOrder<MemoryOrder>::value);*/
  return return_val;
}
#pragma omp end declare variant

#if defined(__clang__) && (__clang_major__>=7)
#pragma GCC diagnostic pop
#endif

}  // namespace desul
#endif
#endif
