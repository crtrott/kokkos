// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif

#include <TestDefaultDeviceType_Category.hpp>

namespace Test {
template<class Exec, class V>
void vector_add(Exec exec, V a, V b, V c) {
  Kokkos::parallel_for(Kokkos::RangePolicy(exec, 0, a.extent(0)), KOKKOS_LAMBDA(int i) {
   c(i) = a(i) + b(i);
  });
  Kokkos::fence();
}

double* get_ptr(int N) {
  double* ptr;
  cudaMallocHost(&ptr, 8*N);
  return ptr;
}

template<class D, class S>
requires(Kokkos::is_view_v<D> && std::is_same_v<typename D::memory_space, Kokkos::SharedSpace>)
void deep_copy2(const D& d, const S& s) {
  auto exec = Kokkos::Cuda();
  if(d.data() == s.data()) {
    if constexpr (std::is_same_v<typename D::execution_space, Kokkos::DefaultExecutionSpace>) {
      printf("Prefetch Device\n");
      Kokkos::Impl::cuda_prefetch_pointer(exec.cuda_stream(), d.data(), d.mapping().required_span_size() * sizeof(double), true);
    } else {
      printf("Prefetch Host\n");
      Kokkos::Impl::cuda_prefetch_pointer(exec.cuda_stream(), d.data(), d.mapping().required_span_size() * sizeof(double), false);
    }
  } 
}

void foo() {
  int N = 100000000;
  Kokkos::View<double*, Kokkos::SharedSpace> a("A", N), b("B", N), c("C", N);
  //Kokkos::View<double*, Kokkos::SharedSpace> a(get_ptr(N), N), b(get_ptr(N), N), c(get_ptr(N), N);
  Kokkos::deep_copy(a, 1);
  Kokkos::deep_copy(b, 1);
  Kokkos::deep_copy(c, 1);

  auto dev = Kokkos::DefaultExecutionSpace();
  auto host = Kokkos::DefaultHostExecutionSpace();
  // GPU Warmup
  vector_add(dev, a, b, c);
  vector_add(host, a, b, c);
  vector_add(dev, a, b, c);

  Kokkos::Timer timer;
  // GPU Time1
  vector_add(dev, a, b, c);
  double time_dev1 = timer.seconds();

  timer.reset();
  // CPU Time1
  vector_add(host, a, b, c);
  double time_host1 = timer.seconds();

  timer.reset();
  // CPU Time2
  vector_add(host, a, b, c);
  double time_host2 = timer.seconds();

  timer.reset();
  // GPU Time2
  vector_add(dev, a, b, c);
  double time_dev2 = timer.seconds();

  timer.reset();
  // GPU Time3
  vector_add(dev, a, b, c);
  double time_dev3 = timer.seconds();

  double GB = 1. * N * 8 * 3 / 1024 / 1024 / 1024;

  //printf("%lf %lf %lf %lf %lf\n",GB/time_dev1, GB/time_host1, GB/time_host2, GB/time_dev2, GB/time_dev3);
  printf("%lf %lf %lf %lf %lf %lf\n",GB/time_dev1, GB/time_host1, GB/time_host2, GB/time_dev2, GB/time_dev3, time_dev1+time_host1+time_host2+time_dev2+time_dev3);
}

void foo2() {
  int N = 100000000;
  Kokkos::View<double*, Kokkos::SharedSpace> a("A", N), b("B", N), c("C", N);
  Kokkos::deep_copy(a, 1);
  Kokkos::deep_copy(b, 1);
  Kokkos::deep_copy(c, 1);

  //auto h_a = Kokkos::create_mirror_view(Kokkos::SharedHostPinnedSpace(), a);
  //auto h_b = Kokkos::create_mirror_view(Kokkos::SharedHostPinnedSpace(), b);
  //auto h_c = Kokkos::create_mirror_view(Kokkos::SharedHostPinnedSpace(), c);

  auto h_a = create_mirror_view(a);
  auto h_b = create_mirror_view(b);
  auto h_c = create_mirror_view(c);
  auto dev = Kokkos::DefaultExecutionSpace();
  auto host = Kokkos::DefaultHostExecutionSpace();
  // GPU Warmup
  vector_add(dev, a, b, c);
  vector_add(host, h_a,h_b, h_c);
  vector_add(dev, a, b, c);

  Kokkos::Timer timer;
  // GPU Time1
  vector_add(dev, a, b, c);
  double time_dev1 = timer.seconds();

  timer.reset();
  // CPU Time1
  #if 0
  Kokkos::parallel_for(N, KOKKOS_LAMBDA(int i) {
    h_a(i) = a(i);
    h_b(i) = b(i);
    h_c(i) = c(i);
  });
  Kokkos::fence();
  #else
  deep_copy2(h_a, a);
  deep_copy2(h_b, b);
  deep_copy2(h_c, c);
  #endif
  vector_add(host, h_a, h_b, h_c);
  Kokkos::fence();

  double time_host1 = timer.seconds();

  timer.reset();
  // CPU Time2
  vector_add(host, h_a, h_b, h_c);
  double time_host2 = timer.seconds();

  timer.reset();
  // GPU Time2
  #if 0
  Kokkos::parallel_for(N, KOKKOS_LAMBDA(int i) {
    a(i) = h_a(i);
    b(i) = h_b(i);
    c(i) = h_c(i);
  });
  Kokkos::fence();
  #else
  deep_copy2(a, h_a);
  deep_copy2(b, h_b);
  deep_copy2(c, h_c);
  #endif
  vector_add(dev, a, b, c);
  double time_dev2 = timer.seconds();

  timer.reset();
  // GPU Time3
  vector_add(dev, a, b, c);
  double time_dev3 = timer.seconds();

  double GB = 1. * N * 8 * 3 / 1024 / 1024 / 1024;
  //printf("%lf %lf %lf %lf %lf\n",GB/time_dev1, 2.*GB/time_host1, GB/time_host2, 2.*GB/time_dev2, GB/time_dev3);
  printf("%lf %lf %lf %lf %lf %lf\n",GB/time_dev1, GB/time_host1, GB/time_host2, GB/time_dev2, GB/time_dev3, time_dev1+time_host1+time_host2+time_dev2+time_dev3);
}

TEST(defaultdevicetype, development_test) { foo(); foo2(); }

}  // namespace Test
