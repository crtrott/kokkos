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

template<class T, size_t Rank>
struct ViewBuilder;

template<class T>
struct ViewBuilder<T, 1> {
  template<class ... Sizes>
  static auto get(Sizes ... sizes) { return Kokkos::View<T*>("A", sizes...); }
};

template<class T>
struct ViewBuilder<T, 2> {
  template<class ... Sizes>
  static auto get(Sizes ... sizes) { return Kokkos::View<T**>("A", sizes...); }
};

template<class T>
struct ViewBuilder<T, 3> {
  template<class ... Sizes>
  static auto get(Sizes ... sizes) { return Kokkos::View<T***>("A", sizes...); }
};

template<class T>
struct ViewBuilder<T, 4> {
  template<class ... Sizes>
  static auto get(Sizes ... sizes) { return Kokkos::View<T****>("A", sizes...); }
};

template<class T>
struct ViewBuilder<T, 5> {
  template<class ... Sizes>
  static auto get(Sizes ... sizes) { return Kokkos::View<T*****>("A", sizes...); }
};

template<class T>
struct ViewBuilder<T, 6> {
  template<class ... Sizes>
  static auto get(Sizes ... sizes) { return Kokkos::View<T******>("A", sizes...); }
};

using E = double;

template<class ... Sizes>
void triad(int R, Sizes ... sizes) {
  constexpr size_t rank = sizeof...(Sizes);
  using index_type = std::common_type_t<Sizes...>;
#if 1
  using view_t = 
    Kokkos::View<E, Kokkos::dextents<index_type, rank>, Kokkos::layout_right,
                 Kokkos::Experimental::Accessor<E, Kokkos::HostSpace, Kokkos::MemoryTraits<>>>;
  
  view_t x(Kokkos::view_alloc("X", Kokkos::WithoutInitializing), sizes...);
  view_t y(Kokkos::view_alloc("X", Kokkos::WithoutInitializing), sizes...);
  view_t z(Kokkos::view_alloc("X", Kokkos::WithoutInitializing), sizes...);
#else
  auto x = ViewBuilder<E, sizeof...(Sizes)>::get(sizes...);  
  auto y = ViewBuilder<E, sizeof...(Sizes)>::get(sizes...);  
  auto z = ViewBuilder<E, sizeof...(Sizes)>::get(sizes...);  
#endif


  Kokkos::Timer timer;
  for(int r=0; r < R+1; r++) {
    if(r == 1) { Kokkos::fence(); timer.reset(); }
    Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<rank>, Kokkos::IndexType<index_type>>(
      {(sizes,0)...},{sizes...}),
      [=]<class... Args>(Args...args) {
      z(args...) = x(args...) + y(args...);
    });
  }
  Kokkos::fence();
  double time = timer.seconds();
  printf("%lu %i %lu (%lu) %e %lf\n",rank,R, (size_t)x.size(), (size_t) x.extent(0), time, 1.0e-9 * R * x.size() * sizeof(E) * 3 /time);
}

TEST(defaultdevicetype, development_test) {
  int R = 5;
  double size = std::pow(32., 6.);
  int n1 = size;
  int n2 = std::pow(size, 1./2) + 1;
  int n3 = std::pow(size, 1./3) + 1;
  int n4 = std::pow(size, 1./4) + 1;
  int n5 = std::pow(size, 1./5) + 1;
  int n6 = std::pow(size, 1./6) + 1;
  printf("Element Size: %lu\n",sizeof(E));
  printf("Rank Repeats Size (1dim) Time GB/s \n");
  triad(R, n1);
  triad(R, n2, n2); 
  triad(R, n3, n3, n3);
  triad(R, n4, n4, n4, n4);
  triad(R, n5, n5, n5, n5, n5);
  triad(R, n6, n6, n6, n6, n6, n6);
}

}  // namespace Test
