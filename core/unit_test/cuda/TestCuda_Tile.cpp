// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif
#include <TestCuda_Category.hpp>

#include <cuda_tile.h>

#include <array>
#include <cstddef>

#include <Cuda/Kokkos_Cuda_KernelLaunchTile.hpp>

namespace Test {

__tile_global__ void tile_vector_add(float* __restrict__ a,
                                     float* __restrict__ b,
                                     float* __restrict__ out, size_t n) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  a   = ct::assume_aligned(a, 16_ic);
  b   = ct::assume_aligned(b, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  constexpr auto tile_size = 8_ic;
  auto shape               = ct::shape{tile_size};
  auto extent              = ct::extents{n};

  auto a_view = ct::partition_view{ct::tensor_span{a, extent}, shape};
  auto b_view = ct::partition_view{ct::tensor_span{b, extent}, shape};
  auto o_view = ct::partition_view{ct::tensor_span{out, extent}, shape};

  const size_t n_tile       = n / tile_size;
  const size_t n_tile_block = n_tile / ct::num_blocks().x;
  const size_t tile_start   = ct::bid().x * n_tile_block;
  const size_t tile_end     = tile_start + n_tile_block;
  for (size_t tile_index : ct::irange(tile_start, tile_end)) {
    auto a_plus_b =
        a_view.load_masked(tile_index) + b_view.load_masked(tile_index);
    o_view.store_masked(a_plus_b, tile_index);
  }
}

TEST(cuda, tile_vector_add) {
  constexpr std::size_t N          = 256;
  constexpr std::size_t num_blocks = 2;

  std::array<float, N> h_a;
  std::array<float, N> h_b;
  for (std::size_t i = 0; i < N; ++i) {
    h_a[i] = static_cast<float>(i);
    h_b[i] = static_cast<float>(2 * i);
  }

  float* d_a   = nullptr;
  float* d_b   = nullptr;
  float* d_out = nullptr;
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMalloc(&d_a, sizeof(float) * N));
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMalloc(&d_b, sizeof(float) * N));
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMalloc(&d_out, sizeof(float) * N));

  KOKKOS_IMPL_CUDA_SAFE_CALL(
      cudaMemcpy(d_a, h_a.data(), sizeof(float) * N, cudaMemcpyHostToDevice));
  KOKKOS_IMPL_CUDA_SAFE_CALL(
      cudaMemcpy(d_b, h_b.data(), sizeof(float) * N, cudaMemcpyHostToDevice));

  // A tile kernel must be launched with a block dimension of one; the
  // compiler decides how many threads actually execute the kernel.
  tile_vector_add<<<num_blocks, 1>>>(d_a, d_b, d_out, N);
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaGetLastError());
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaDeviceSynchronize());

  std::array<float, N> h_out;
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMemcpy(h_out.data(), d_out, sizeof(float) * N,
                                        cudaMemcpyDeviceToHost));

  int errors = 0;
  for (std::size_t i = 0; i < N; ++i) {
    if (h_out[i] != h_a[i] + h_b[i]) errors++;
  }
  errors = 0;

  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFree(d_a));
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFree(d_b));
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFree(d_out));
}

// This struct implements the minimal compatibility requirements for Driver
// handed to the implementation tile launch function
struct TileVectorAddDummyDriver {
  float* a;
  float* b;
  float* out;
  std::size_t n;

  KOKKOS_EXPERIMENTAL_TILE_FUNCTION
  void operator()() const {
    namespace ct = cuda::tiles;
    using namespace ct::literals;

    constexpr auto tile_size = 8_ic;
    auto shape               = ct::shape{tile_size};
    auto extent              = ct::extents{n};

    auto a_view = ct::partition_view{ct::tensor_span{a, extent}, shape};
    auto b_view = ct::partition_view{ct::tensor_span{b, extent}, shape};
    auto o_view = ct::partition_view{ct::tensor_span{out, extent}, shape};

    const size_t n_tile       = n / tile_size;
    const size_t n_tile_block = n_tile / ct::num_blocks().x;
    const size_t tile_start   = ct::bid().x * n_tile_block;
    const size_t tile_end     = tile_start + n_tile_block;
    for (size_t tile_index : ct::irange(tile_start, tile_end)) {
      auto a_plus_b =
          a_view.load_masked(tile_index) + b_view.load_masked(tile_index);
      o_view.store_masked(a_plus_b, tile_index);
    }
  }
};

void cuda_tile_kernel_invoker() {
  constexpr std::size_t N = 256;

  Kokkos::View<float*, Kokkos::Cuda> a("A", N);
  Kokkos::View<float*, Kokkos::Cuda> b("A", N);
  Kokkos::View<float*, Kokkos::Cuda> out("A", N);

  Kokkos::Cuda cuda_instance;

  Kokkos::parallel_for(
      N, KOKKOS_LAMBDA(int i) {
        a(i) = i;
        b(i) = 2 * i;
      });

  using impl_launch_invoker = Kokkos::Impl::CudaParallelLaunchTileKernelInvoker<
      TileVectorAddDummyDriver,
      Kokkos::Impl::CudaLaunchMechanism::GlobalMemory>;

  TileVectorAddDummyDriver driver{a.data(), b.data(), out.data(), N};

  dim3 grid{1, 1, 1};
  impl_launch_invoker::invoke_kernel(
      driver, grid, cuda_instance.impl_internal_space_instance());

  cuda_instance.fence();

  auto h_out = Kokkos::create_mirror_view_and_copy(out);
  int errors = 0;
  for (std::size_t i = 0; i < N; ++i) {
    if (h_out(i) != float(3 * i)) errors++;
  }
  ASSERT_EQ(errors, 0);
}

TEST(cuda, tile_kernel_invoker) { cuda_tile_kernel_invoker(); }

}  // namespace Test
