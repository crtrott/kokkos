/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#ifndef DESUL_ATOMICS_COMPARE_EXCHANGE_CUDA_HPP_
#define DESUL_ATOMICS_COMPARE_EXCHANGE_CUDA_HPP_

#include <type_traits>

#include "desul/atomics/Common.hpp"
#include "desul/atomics/Lock_Array_Cuda.hpp"

namespace desul {
namespace Impl {

// clang-format off
inline __device__ void device_atomic_thread_fence(MemoryOrderRelease, MemoryScopeDevice) { __threadfence(); }
inline __device__ void device_atomic_thread_fence(MemoryOrderAcquire, MemoryScopeDevice) { __threadfence(); }
inline __device__ void device_atomic_thread_fence(MemoryOrderAcqRel,  MemoryScopeDevice) { __threadfence(); }
inline __device__ void device_atomic_thread_fence(MemoryOrderSeqCst,  MemoryScopeDevice) { __threadfence(); }
inline __device__ void device_atomic_thread_fence(MemoryOrderRelease, MemoryScopeCore)   { __threadfence_block(); }
inline __device__ void device_atomic_thread_fence(MemoryOrderAcquire, MemoryScopeCore)   { __threadfence_block(); }
inline __device__ void device_atomic_thread_fence(MemoryOrderAcqRel,  MemoryScopeCore)   { __threadfence_block(); }
inline __device__ void device_atomic_thread_fence(MemoryOrderSeqCst,  MemoryScopeCore)   { __threadfence_block(); }
#ifndef DESUL_CUDA_ARCH_IS_PRE_PASCAL
inline __device__ void device_atomic_thread_fence(MemoryOrderRelease, MemoryScopeNode)   { __threadfence_system(); }
inline __device__ void device_atomic_thread_fence(MemoryOrderAcquire, MemoryScopeNode)   { __threadfence_system(); }
inline __device__ void device_atomic_thread_fence(MemoryOrderAcqRel,  MemoryScopeNode)   { __threadfence_system(); }
inline __device__ void device_atomic_thread_fence(MemoryOrderSeqCst,  MemoryScopeNode)   { __threadfence_system(); }
#endif
// clang-format on

}  // namespace Impl
}  // namespace desul

#ifdef DESUL_CUDA_ARCH_IS_PRE_VOLTA

namespace desul {
namespace Impl {

template <typename T, class MemoryScope>
__device__ typename ::std::enable_if<sizeof(T) == 4, T>::type
device_atomic_compare_exchange(
    T* const dest, T compare, T value, MemoryOrderRelaxed, MemoryScope) {
  static_assert(sizeof(unsigned int) == 4,
                "this function assumes an unsigned int is 32-bit");
  unsigned int return_val = atomicCAS(reinterpret_cast<unsigned int*>(dest),
                                      reinterpret_cast<unsigned int&>(compare),
                                      reinterpret_cast<unsigned int&>(value));
  return reinterpret_cast<T&>(return_val);
}
template <typename T, class MemoryScope>
__device__ typename ::std::enable_if<sizeof(T) == 8, T>::type
device_atomic_compare_exchange(
    T* const dest, T compare, T value, MemoryOrderRelaxed, MemoryScope) {
  static_assert(sizeof(unsigned long long int) == 8,
                "this function assumes an unsigned long long is 64-bit");
  unsigned long long int return_val =
      atomicCAS(reinterpret_cast<unsigned long long int*>(dest),
                reinterpret_cast<unsigned long long int&>(compare),
                reinterpret_cast<unsigned long long int&>(value));
  return reinterpret_cast<T&>(return_val);
}

template <typename T, class MemoryScope>
__device__ typename ::std::enable_if<sizeof(T) == 4 || sizeof(T) == 8, T>::type
device_atomic_compare_exchange(
    T* const dest, T compare, T value, MemoryOrderRelease, MemoryScope) {
  T return_val = device_atomic_compare_exchange(
      dest, compare, value, MemoryOrderRelaxed(), MemoryScope());
  device_atomic_thread_fence(MemoryOrderRelease(), MemoryScope());
  return return_val;
}

template <typename T, class MemoryScope>
__device__ typename ::std::enable_if<sizeof(T) == 4 || sizeof(T) == 8, T>::type
device_atomic_compare_exchange(
    T* const dest, T compare, T value, MemoryOrderAcquire, MemoryScope) {
  device_atomic_thread_fence(MemoryOrderAcquire(), MemoryScope());
  T return_val = device_atomic_compare_exchange(
      dest, compare, value, MemoryOrderRelaxed(), MemoryScope());
  return return_val;
}

template <typename T, class MemoryScope>
__device__ typename ::std::enable_if<sizeof(T) == 4 || sizeof(T) == 8, T>::type
device_atomic_compare_exchange(
    T* const dest, T compare, T value, MemoryOrderAcqRel, MemoryScope) {
  device_atomic_thread_fence(MemoryOrderAcquire(), MemoryScope());
  T return_val = device_atomic_compare_exchange(
      dest, compare, value, MemoryOrderRelaxed(), MemoryScope());
  device_atomic_thread_fence(MemoryOrderRelease(), MemoryScope());
  return return_val;
}

template <typename T, class MemoryScope>
__device__ typename ::std::enable_if<sizeof(T) == 4, T>::type device_atomic_exchange(
    T* const dest, T value, MemoryOrderRelaxed, MemoryScope) {
  static_assert(sizeof(unsigned int) == 4,
                "this function assumes an unsigned int is 32-bit");
  unsigned int return_val = atomicExch(reinterpret_cast<unsigned int*>(dest),
                                       reinterpret_cast<unsigned int&>(value));
  return reinterpret_cast<T&>(return_val);
}
template <typename T, class MemoryScope>
__device__ typename ::std::enable_if<sizeof(T) == 8, T>::type device_atomic_exchange(
    T* const dest, T value, MemoryOrderRelaxed, MemoryScope) {
  static_assert(sizeof(unsigned long long int) == 8,
                "this function assumes an unsigned long long is 64-bit");
  unsigned long long int return_val =
      atomicExch(reinterpret_cast<unsigned long long int*>(dest),
                 reinterpret_cast<unsigned long long int&>(value));
  return reinterpret_cast<T&>(return_val);
}

template <typename T, class MemoryScope>
__device__ typename ::std::enable_if<sizeof(T) == 4 || sizeof(T) == 8, T>::type
device_atomic_exchange(T* const dest, T value, MemoryOrderRelease, MemoryScope) {
  T return_val =
      device_atomic_exchange(dest, value, MemoryOrderRelaxed(), MemoryScope());
  device_atomic_thread_fence(MemoryOrderRelease(), MemoryScope());
  return reinterpret_cast<T&>(return_val);
}

template <typename T, class MemoryScope>
__device__ typename ::std::enable_if<sizeof(T) == 4 || sizeof(T) == 8, T>::type
device_atomic_exchange(T* const dest, T value, MemoryOrderAcquire, MemoryScope) {
  device_atomic_thread_fence(MemoryOrderAcquire(), MemoryScope());
  T return_val =
      device_atomic_exchange(dest, value, MemoryOrderRelaxed(), MemoryScope());
  return reinterpret_cast<T&>(return_val);
}

template <typename T, class MemoryScope>
__device__ typename ::std::enable_if<sizeof(T) == 4 || sizeof(T) == 8, T>::type
device_atomic_exchange(T* const dest, T value, MemoryOrderAcqRel, MemoryScope) {
  device_atomic_thread_fence(MemoryOrderAcquire(), MemoryScope());
  T return_val =
      device_atomic_exchange(dest, value, MemoryOrderRelaxed(), MemoryScope());
  device_atomic_thread_fence(MemoryOrderRelease(), MemoryScope());
  return reinterpret_cast<T&>(return_val);
}
}  // namespace Impl
}  // namespace desul

#endif

// Including CUDA ptx based exchange atomics
// When building with clang we need to include the device functions always
// since clang must see a consistent overload set in both device and host compilation
// but that means we need to know on the host what to make visible, i.e. we need
// a host side compile knowledge of architecture.
// We simply can say DESUL proper doesn't support clang CUDA build pre Volta,
// Kokkos has that knowledge and so I use it here, allowing in Kokkos to use
// clang with pre Volta as CUDA compiler
#ifndef DESUL_CUDA_ARCH_IS_PRE_VOLTA
#include <desul/atomics/cuda/CUDA_asm_exchange.hpp>
#endif

// SeqCst is not directly supported by PTX, need the additional fences:

#if defined(__CUDA_ARCH__) || !defined(__NVCC__)
namespace desul {
namespace Impl {
template <typename T, class MemoryScope>
__device__ typename ::std::enable_if<sizeof(T) == 4, T>::type device_atomic_exchange(
    T* const dest, T value, MemoryOrderSeqCst, MemoryScope) {
  device_atomic_thread_fence(MemoryOrderAcquire(), MemoryScope());
  T return_val =
      device_atomic_exchange(dest, value, MemoryOrderRelaxed(), MemoryScope());
  device_atomic_thread_fence(MemoryOrderRelease(), MemoryScope());
  return return_val;
}
template <typename T, class MemoryScope>
__device__ typename ::std::enable_if<sizeof(T) == 8, T>::type device_atomic_exchange(
    T* const dest, T value, MemoryOrderSeqCst, MemoryScope) {
  device_atomic_thread_fence(MemoryOrderAcquire(), MemoryScope());
  T return_val =
      device_atomic_exchange(dest, value, MemoryOrderRelaxed(), MemoryScope());
  device_atomic_thread_fence(MemoryOrderRelease(), MemoryScope());
  return return_val;
}
template <typename T, class MemoryScope>
__device__ typename ::std::enable_if<sizeof(T) == 4, T>::type
device_atomic_compare_exchange(
    T* const dest, T compare, T value, MemoryOrderSeqCst, MemoryScope) {
  device_atomic_thread_fence(MemoryOrderAcquire(), MemoryScope());
  T return_val = device_atomic_compare_exchange(
      dest, compare, value, MemoryOrderRelaxed(), MemoryScope());
  device_atomic_thread_fence(MemoryOrderRelease(), MemoryScope());
  return return_val;
}
template <typename T, class MemoryScope>
__device__ typename ::std::enable_if<sizeof(T) == 8, T>::type
device_atomic_compare_exchange(
    T* const dest, T compare, T value, MemoryOrderSeqCst, MemoryScope) {
  device_atomic_thread_fence(MemoryOrderAcquire(), MemoryScope());
  T return_val = device_atomic_compare_exchange(
      dest, compare, value, MemoryOrderRelaxed(), MemoryScope());
  device_atomic_thread_fence(MemoryOrderRelease(), MemoryScope());
  return return_val;
}

template <typename T, class MemoryOrder, class MemoryScope>
__device__ typename ::std::enable_if<(sizeof(T) != 8) && (sizeof(T) != 4), T>::type
device_atomic_compare_exchange(
    T* const dest, T compare, T value, MemoryOrder, MemoryScope scope) {
  // This is a way to avoid dead lock in a warp or wave front
  T return_val;
  int done = 0;
  unsigned int mask = DESUL_IMPL_ACTIVEMASK;
  unsigned int active = DESUL_IMPL_BALLOT_MASK(mask, 1);
  unsigned int done_active = 0;
  while (active != done_active) {
    if (!done) {
      if (Impl::lock_address_cuda((void*)dest, scope)) {
        if (std::is_same<MemoryOrder, MemoryOrderSeqCst>::value)
          device_atomic_thread_fence(MemoryOrderRelease(), scope);
        device_atomic_thread_fence(MemoryOrderAcquire(), scope);
        return_val = *dest;
        if (return_val == compare) {
          *dest = value;
          device_atomic_thread_fence(MemoryOrderRelease(), scope);
        }
        Impl::unlock_address_cuda((void*)dest, scope);
        done = 1;
      }
    }
    done_active = DESUL_IMPL_BALLOT_MASK(mask, done);
  }
  return return_val;
}
template <typename T, class MemoryOrder, class MemoryScope>
__device__ typename ::std::enable_if<(sizeof(T) != 8) && (sizeof(T) != 4), T>::type
device_atomic_exchange(T* const dest, T value, MemoryOrder, MemoryScope scope) {
  // This is a way to avoid dead lock in a warp or wave front
  T return_val;
  int done = 0;
  unsigned int mask = DESUL_IMPL_ACTIVEMASK;
  unsigned int active = DESUL_IMPL_BALLOT_MASK(mask, 1);
  unsigned int done_active = 0;
  while (active != done_active) {
    if (!done) {
      if (Impl::lock_address_cuda((void*)dest, scope)) {
        if (std::is_same<MemoryOrder, MemoryOrderSeqCst>::value)
          device_atomic_thread_fence(MemoryOrderRelease(), scope);
        device_atomic_thread_fence(MemoryOrderAcquire(), scope);
        return_val = *dest;
        *dest = value;
        device_atomic_thread_fence(MemoryOrderRelease(), scope);
        Impl::unlock_address_cuda((void*)dest, scope);
        done = 1;
      }
    }
    done_active = DESUL_IMPL_BALLOT_MASK(mask, done);
  }
  return return_val;
}
}  // namespace Impl
}  // namespace desul
#endif

#endif
