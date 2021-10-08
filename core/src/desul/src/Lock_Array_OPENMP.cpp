/* 
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/
#define DESUL_LOCK_ARRAY_OPENMP_CPP_
#define DESUL_HAVE_OPENMP_ATOMICS
#include <desul/atomics/Macros.hpp>
#include <desul/atomics/Lock_Array.hpp>
#include <cinttypes>
#include <string>
#include <sstream>
#include <omp.h>

//namespace desul {
//namespace Impl {
#pragma omp begin declare target
int32_t* OPENMP_SPACE_ATOMIC_LOCKS_DEVICE;
//int32_t* OPENMP_SPACE_ATOMIC_LOCKS_NODE = nullptr;
#pragma omp end declare target
//}
//}  // namespace desul

namespace desul {

namespace Impl {


int32_t* OPENMP_SPACE_ATOMIC_LOCKS_DEVICE_h = nullptr;
//int32_t* OPENMP_SPACE_ATOMIC_LOCKS_NODE_h = nullptr;

// define functions
template<typename T>
void init_lock_arrays_openmp() {
  size_t alloc_size = sizeof(int32_t) * (OPENMP_SPACE_ATOMIC_MASK + 1);
  OPENMP_SPACE_ATOMIC_LOCKS_DEVICE_h = (int32_t*) omp_target_alloc(alloc_size, omp_get_default_device());

  printf("Set Host: %p\n",OPENMP_SPACE_ATOMIC_LOCKS_DEVICE_h);
  #pragma omp target is_device_ptr(OPENMP_SPACE_ATOMIC_LOCKS_DEVICE_h)
  {
    OPENMP_SPACE_ATOMIC_LOCKS_DEVICE = OPENMP_SPACE_ATOMIC_LOCKS_DEVICE_h;
    printf("Device Ptr: %p\n",OPENMP_SPACE_ATOMIC_LOCKS_DEVICE);
  }
  int N = OPENMP_SPACE_ATOMIC_MASK + 1;
  printf("Init Lock Array: %i\n",N);
  #pragma omp target teams distribute parallel for
  for(int i=0; i<N; i++) {
    OPENMP_SPACE_ATOMIC_LOCKS_DEVICE[i] = 0;
  }
  printf("Init Done Array:\n");
}

template<typename T>
void finalize_lock_arrays_openmp() {
  if (OPENMP_SPACE_ATOMIC_LOCKS_DEVICE_h == nullptr) return;
  OPENMP_SPACE_ATOMIC_LOCKS_DEVICE_h = nullptr;

  #pragma omp target
  {
    OPENMP_SPACE_ATOMIC_LOCKS_DEVICE = nullptr;
  }
}

// Instantiate functions
template void init_lock_arrays_openmp<int>();
template void finalize_lock_arrays_openmp<int>();

}  // namespace Impl

}  // namespace desul
