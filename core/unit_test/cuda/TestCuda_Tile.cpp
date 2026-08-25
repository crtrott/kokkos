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

namespace Test {

__tile_global__ void tile_vector_add(float* a, float* b, float* out,
                                     std::size_t n) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  a   = ct::assume_aligned(a, 16_ic);
  b   = ct::assume_aligned(b, 16_ic);
  out = ct::assume_aligned(out, 16_ic);

  auto shape  = ct::shape{8_ic};
  auto extent = ct::extents{n};

  auto a_view = ct::partition_view{ct::tensor_span{a, extent}, shape};
  auto b_view = ct::partition_view{ct::tensor_span{b, extent}, shape};
  auto o_view = ct::partition_view{ct::tensor_span{out, extent}, shape};

  int bx = ct::bid().x;
  o_view.store_masked(a_view.load_masked(bx) + b_view.load_masked(bx), bx);
}

TEST(cuda, tile_vector_add) {
  constexpr std::size_t N = 8;

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
  tile_vector_add<<<1, 1>>>(d_a, d_b, d_out, N);
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaGetLastError());
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaDeviceSynchronize());

  std::array<float, N> h_out;
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMemcpy(h_out.data(), d_out, sizeof(float) * N,
                                        cudaMemcpyDeviceToHost));

  for (std::size_t i = 0; i < N; ++i) {
    ASSERT_FLOAT_EQ(h_out[i], h_a[i] + h_b[i]);
  }

  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFree(d_a));
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFree(d_b));
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFree(d_out));
}

}  // namespace Test
