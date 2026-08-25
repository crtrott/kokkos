
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_CUDA

// #include <mutex>
// #include <cstdint>
// #include <cmath>
// #include <Kokkos_Parallel.hpp>
// #include <impl/Kokkos_Error.hpp>
// #include <Cuda/Kokkos_Cuda_abort.hpp>
// #include <Cuda/Kokkos_Cuda_Error.hpp>
#include <Cuda/Kokkos_Cuda_Instance.hpp>

namespace Kokkos::Impl {

template <class DriverType>
__tile_global__ static void cuda_tile_launch_global_memory(
    const DriverType* driver) {
  driver->operator()();
}

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

    // Set cuda device before launching kernel
    cuda_instance->set_cuda_device();

    // The 1 might be able to go away
    cuda_tile_launch_global_memory<<<grid, 1>>>(driver_ptr);
  }
};
}  // namespace Kokkos::Impl
#endif
