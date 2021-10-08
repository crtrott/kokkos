/* 
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#ifndef DESUL_ATOMICS_LOCK_ARRAY_OPENMP_HPP_
#define DESUL_ATOMICS_LOCK_ARRAY_OPENMP_HPP_

#include "desul/atomics/Macros.hpp"
#include "desul/atomics/Common.hpp"

#ifdef DESUL_HAVE_OPENMP_ATOMICS

#include <cstdint>



namespace desul {
namespace Impl {

#pragma omp begin declare variant match(device = {kind(nohost)})
inline unsigned openmp_mask() {
  unsigned mask;
  asm volatile("activemask.b32 %0;" :"=r"(mask)::);
  return mask;
}

inline unsigned openmp_ballot_mask(unsigned mask, unsigned value) {
  unsigned result;
  asm volatile("{\n\t" \
               ".reg .pred p;\n\t" \
               "setp.gt.u32 p, %1, 0x0;\n\t" \
               "vote.sync.ballot.b32 %0,p,%2;" \
               "}" :"=r"(result):"r"(value),"r"(mask):);
  return result;
}
#pragma omp end declare variant

/// \brief This global variable in Host space is the central definition
///        of these arrays.
extern int32_t* OPENMP_SPACE_ATOMIC_LOCKS_DEVICE_h;
//extern int32_t* OPENMP_SPACE_ATOMIC_LOCKS_NODE_h;


/// \brief After this call, the g_host_cuda_lock_arrays variable has
///        valid, initialized arrays.
///
/// This call is idempotent.
/// The function is templated to make it a weak symbol to deal with Kokkos/RAJA
///   snapshotted version while also linking against pure Desul
template<typename /*AlwaysInt*/ = int>
void init_lock_arrays_openmp();

/// \brief After this call, the g_host_cuda_lock_arrays variable has
///        all null pointers, and all array memory has been freed.
///
/// This call is idempotent.
/// The function is templated to make it a weak symbol to deal with Kokkos/RAJA
///   snappshotted version while also linking against pure Desul
template<typename T = int>
void finalize_lock_arrays_openmp();

}  // namespace Impl
}  // namespace desul


#pragma omp begin declare target
extern int32_t* OPENMP_SPACE_ATOMIC_LOCKS_DEVICE;
#pragma omp end declare target
namespace desul {
namespace Impl {

/// \brief This global variable in OpenMP Target space is what kernels use
///        to get access to the lock arrays.


#define OPENMP_SPACE_ATOMIC_MASK 0x3FFFF

/// \brief Acquire a lock for the address
///
/// This function tries to acquire the lock for the hash value derived
/// from the provided ptr. If the lock is successfully acquired the
/// function returns true. Otherwise it returns false.
#pragma omp begin declare target device_type(nohost)
inline bool lock_address_openmp(void* ptr, desul::MemoryScopeDevice) {
  size_t offset = size_t(ptr);
  offset = offset >> 2;
  offset = offset & OPENMP_SPACE_ATOMIC_MASK;
  return (0 == desul::atomic_compare_exchange(&OPENMP_SPACE_ATOMIC_LOCKS_DEVICE[offset], 0, 1, desul::MemoryOrderAcquire(), desul::MemoryScopeDevice()));
}

/// \brief Release lock for the address
///
/// This function releases the lock for the hash value derived
/// from the provided ptr. This function should only be called
/// after previously successfully acquiring a lock with
/// lock_address.
inline void unlock_address_openmp(void* ptr, desul::MemoryScopeDevice) {
  size_t offset = size_t(ptr);
  offset = offset >> 2;
  offset = offset & OPENMP_SPACE_ATOMIC_MASK;
  desul::atomic_exchange(&OPENMP_SPACE_ATOMIC_LOCKS_DEVICE[offset], 0,  desul::MemoryOrderRelease(), desul::MemoryScopeDevice());
}
#pragma omp end declare target
}  // namespace Impl
}  // namespace desul
#endif
#endif
