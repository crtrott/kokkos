// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_CUDA_PARALLEL_MDRANGETILE_HPP
#define KOKKOS_CUDA_PARALLEL_MDRANGETILE_HPP

#include <algorithm>

#include <Kokkos_Macros.hpp>
#include <Kokkos_Parallel.hpp>
#include <KokkosExp_MDRangePolicy.hpp>
#include <Cuda/Kokkos_Cuda_KernelLaunchTile.hpp>

#include <cuda_tile.h>

namespace Kokkos::Impl {

template <class FunctorType, class... Traits>
class ParallelFor<FunctorType,
                  Kokkos::Experimental::TileMDRangePolicy<Traits...>,
                  Kokkos::Cuda> {
 public:
  using Policy       = Kokkos::Experimental::TileMDRangePolicy<Traits...>;
  using functor_type = FunctorType;

 private:
  using array_index_type = typename Policy::array_index_type;
  using index_type       = typename Policy::index_type;
  using MaxGridSize      = Kokkos::Array<array_index_type, 3>;
  using array_type       = typename Policy::point_type;

  const FunctorType m_functor;
  const Policy m_policy;
  const MaxGridSize m_max_grid_size;

  array_type m_lower;
  array_type m_upper;
  array_type m_extent;

 public:
  Policy const& get_policy() const { return m_policy; }

  inline __tile__ void operator()() const {
    if constexpr (Policy::rank() == 1) {
      m_functor(cuda::tiles::bid().x + m_lower[0]);
    } else if constexpr (Policy::rank() == 2) {
      m_functor(cuda::tiles::bid().x + m_lower[0],
                cuda::tiles::bid().y + m_lower[1]);
    } else if constexpr (Policy::rank() == 3) {
      m_functor(cuda::tiles::bid().x + m_lower[0],
                cuda::tiles::bid().y + m_lower[1],
                cuda::tiles::bid().z + m_lower[2]);
    }
  }

  inline void execute() const {
    // make sure the grid dimensions don't exceed the max number of blocks
    // allowed
    const auto check_grid_sizes = [&]([[maybe_unused]] const dim3& grid) {
      KOKKOS_ASSERT(grid.x > 0 &&
                    grid.x <= static_cast<unsigned int>(m_max_grid_size[0]));
      KOKKOS_ASSERT(grid.y > 0 &&
                    grid.y <= static_cast<unsigned int>(m_max_grid_size[1]));
      KOKKOS_ASSERT(grid.z > 0 &&
                    grid.z <= static_cast<unsigned int>(m_max_grid_size[2]));
    };

    dim3 grid;
    if constexpr (Policy::rank() == 1) {
      grid = dim3{static_cast<unsigned>(m_extent[0]), 1u, 1u};
    } else if constexpr (Policy::rank() == 2) {
      grid = dim3{static_cast<unsigned>(m_extent[0]),
                  static_cast<unsigned>(m_extent[1]), 1u};
    } else if constexpr (Policy::rank() == 3) {
      grid = dim3{static_cast<unsigned>(m_extent[0]),
                  static_cast<unsigned>(m_extent[1]),
                  static_cast<unsigned>(m_extent[2])};
    } else {
      // Will not happen since we static assert in the TileMDRangePolicy
      Kokkos::abort("Unsupported TileMDRangePolicy Rank");
    }
    // ensure we don't exceed the capability of the device
    check_grid_sizes(grid);

    using impl_launch_invoker =
        Kokkos::Impl::CudaParallelLaunchTileKernelInvoker<
            ParallelFor, Kokkos::Impl::CudaLaunchMechanism::GlobalMemory>;

    impl_launch_invoker::invoke_kernel(
        *this, grid, m_policy.space().impl_internal_space_instance());
  }  // end execute

  //  inline
  ParallelFor(const FunctorType& arg_functor, Policy arg_policy)
      : m_functor(arg_functor),
        m_policy(arg_policy),
        m_max_grid_size({
            m_policy.space().cuda_device_prop().maxGridSize[0],
            m_policy.space().cuda_device_prop().maxGridSize[1],
            m_policy.space().cuda_device_prop().maxGridSize[2],
        }) {
    // Initialize begins and ends based on layout
    // Swap the fastest indexes to x dimension
    for (array_index_type i = 0; i < Policy::rank(); ++i) {
      m_lower[i]  = m_policy.lower(i);
      m_upper[i]  = m_policy.upper(i);
      m_extent[i] = m_upper[i] - m_lower[i];
    }
  }
};

}  // namespace Kokkos::Impl

#endif  // KOKKOS_CUDA_PARALLEL_MDRANGETILE_HPP
