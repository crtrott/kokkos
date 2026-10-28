// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif

#include <Kokkos_Random.hpp>
#include <TestDefaultDeviceType_Category.hpp>

namespace Test {

void stream(Kokkos::View<double*, Kokkos::AnonymousSpace> a,
   	    Kokkos::View<double*, Kokkos::AnonymousSpace> b,
	    Kokkos::View<double*, Kokkos::AnonymousSpace> c) {
  Kokkos::parallel_for(a.extent(0), KOKKOS_LAMBDA(int i) {
    c(i) += a(i) + b(i);
  });
}
void random(Kokkos::View<double*, Kokkos::AnonymousSpace> a,
   	    Kokkos::View<double*, Kokkos::AnonymousSpace> b,
	    Kokkos::View<double*, Kokkos::AnonymousSpace> c,
	    Kokkos::View<int*, Kokkos::AnonymousSpace> idx) {
  Kokkos::parallel_for(a.extent(0), KOKKOS_LAMBDA(int i) {
    c(i) += a(idx(i)) + b(idx(i));
  });
}

template<class MemSpace>
void bar(int N, int R) {
  Kokkos::View<double*, MemSpace> a("A", N);
  Kokkos::View<double*, MemSpace> b("B", N);
  Kokkos::View<double*, MemSpace> c("C", N);
  
  stream(a,b,c);

  Kokkos::fence();
  Kokkos::Timer timer;
  for(int r=0; r<R; r++) {
    stream(a,b,c);
  }
  Kokkos::fence();
  double time = timer.seconds();
  printf("Stream Time: %lf ms Space: %s\n",time*1000, MemSpace::name());

  Kokkos::View<int*, MemSpace> idx("idx", N);

  Kokkos::Random_XorShift64_Pool<TEST_EXECSPACE> generator(32178);
  Kokkos::fill_random(idx, generator, N);

  random(a,b,c,idx);
  Kokkos::fence();
  timer.reset();
  for(int r=0; r<R; r++) {
    random(a,b,c,idx);
  }
  Kokkos::fence();
  time = timer.seconds();
  printf("Random Time: %lf ms Space: %s\n",time*1000, MemSpace::name());

}

TEST(defaultdevicetype, development_test) {
  int N = 100000000;
  int R = 100;
  bar<Kokkos::DefaultMemorySpace>(N,R);
  bar<Kokkos::DefaultHostMemorySpace>(N,R);
}
}  // namespace Test
