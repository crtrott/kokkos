// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_CUDA_KERNELLAUNCHTILE_HPP
#define KOKKOS_CUDA_KERNELLAUNCHTILE_HPP

#include <Kokkos_Macros.hpp>

#include <Cuda/Kokkos_Cuda_Instance.hpp>

namespace Kokkos::Impl {

template <class DriverType>
__tile_global__ static void cuda_tile_launch_global_memory(
    const DriverType* driver) {
  driver->operator()();
}

// CUDA Tile Kernels neither support constant cache nor
// struct arguments to the __tile_global__ functions.
// So for now we only support GlobalLaunch
template <class DriverType, CudaLaunchMechanism LaunchMechanism>
struct CudaParallelLaunchTileKernelInvoker;

template <class DriverType>
struct CudaParallelLaunchTileKernelInvoker<DriverType,
                                           CudaLaunchMechanism::GlobalMemory> {
  static void invoke_kernel(DriverType const& driver, dim3 const& grid,
                            CudaInternal const* cuda_instance) {
    DriverType* driver_ptr = reinterpret_cast<DriverType*>(
        cuda_instance->scratch_functor(sizeof(DriverType)));

    KOKKOS_IMPL_CUDA_SAFE_CALL((cuda_instance->cuda_memcpy_async_wrapper(
        driver_ptr, &driver, sizeof(DriverType), cudaMemcpyDefault)));

    // The 1 might be able to go away
    cuda_tile_launch_global_memory<<<grid, 1, 0, cuda_instance->m_stream>>>(
        driver_ptr);
  }
};
}  // namespace Kokkos::Impl

#endif
