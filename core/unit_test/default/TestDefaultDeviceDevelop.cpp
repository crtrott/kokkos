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

template<class Driver>
__global__ void kernel(__grid_constant__ const Driver f) {
  f();
}

template<class Lambda, int Rank, int Order>
struct Functor {
  Lambda f;
  Kokkos::Array<int, Rank> begins, ends, strides = {0,0,0};
  void execute() {

    // Really the next piece needs to be something like 
    // dim3 block = get_block(std::integral_constant<int, Rank>);
    // dim3 grid = get_grid(std::integral_constant<int, Rank>, block);
    dim3 block = {32, 4, 2};  
    dim3 grid;
    grid.x = (ends[2] - begins[2] + block.x-1)/block.x;
    grid.y = (ends[1] - begins[1] + block.y-1)/block.y;
    grid.z = (ends[0] - begins[0] + block.z-1)/block.z;
    strides[0] = grid.z * block.z;
    strides[1] = grid.y * block.y;
    strides[2] = grid.x * block.x;

    kernel<<<grid, block>>>(*this);
  }

  template<unsigned R, class ... Idxs>
  KOKKOS_INLINE_FUNCTION
  void iterate(std::integral_constant<unsigned, R>, const Kokkos::Array<unsigned, Rank>& mybegin, Idxs ... idxs) const {
    for(int idx = mybegin[R]; idx < ends[R]; idx += strides[R]) {
      if constexpr (Order == 0) 
        iterate(std::integral_constant<unsigned, R+1>(), mybegin, idxs..., idx);
      else
        iterate(std::integral_constant<unsigned, R+1>(), mybegin, idx, idxs...);
    }
  }

  template<class ... Idxs>
  KOKKOS_INLINE_FUNCTION
  void iterate(std::integral_constant<unsigned, Rank>, const Kokkos::Array<unsigned, Rank>&, Idxs ... idxs) const {
    f(idxs...);
  }

  KOKKOS_INLINE_FUNCTION
  void operator() () const {
    // this needs really function get_begins that has overloads for various ranks. Note that it would be order independent though!
    Kokkos::Array<unsigned, Rank> mybegins = //get_begins(std::integral_constant<unsigned, Rank>, std::integral_constant<unsigned, Order>); 
      {blockIdx.z * blockDim.z + threadIdx.z + begins[0], blockIdx.y * blockDim.y + threadIdx.y + begins[1], blockIdx.x * blockDim.x + threadIdx.x + begins[2]}; 
    iterate(std::integral_constant<unsigned, 0u>(), mybegins);
  }
};

template<unsigned R, class Lambda>
void parallel_left(Kokkos::Array<int, R> begins_in, Kokkos::Array<int, R> ends_in, Lambda lambda) {
  Kokkos::Array<int, R> begins, ends;
  for(int r=0; r<R; r++) { begins[r] = begins_in[R-1-r]; ends[r] = ends_in[R-1-r]; }
  Functor<Lambda, R, 1> f{lambda, begins, ends};
  f.execute();
}
template<unsigned R, class Lambda>
void parallel_right(Kokkos::Array<int, R> begins, Kokkos::Array<int, R> ends, Lambda lambda) {
  Functor<Lambda, R, 0> f{lambda, begins, ends};
  f.execute();
}

void foo_left() {
  Kokkos::View<double***, Kokkos::LayoutLeft> a("A",500,500,500), b("B",500,500,500);
  Kokkos::deep_copy(a,1.0);
  Kokkos::deep_copy(b,2.0);
  auto lambda = KOKKOS_LAMBDA(int i, int j, int k) { a(i,j,k) += b(i,j,k); };

  auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>>({0,0,0},{500,500,500}, {32, 8, 2});
  Kokkos::parallel_for(policy, lambda);
  Kokkos::fence();
  Kokkos::Timer timer;
  Kokkos::parallel_for(policy, lambda);
  Kokkos::fence();
  double time = timer.seconds();
  int errors = 0;
  Kokkos::parallel_reduce(policy, KOKKOS_LAMBDA(int i, int j, int k, int& err) { if(a(i,j,k) != 5.0) err++; }, errors);
  printf("Left: %lf %i\n", time*1.e6, errors);

}

void foo_left_new() {
  Kokkos::View<double***, Kokkos::LayoutLeft> a("A",500,500,500), b("B",500,500,500);
  Kokkos::deep_copy(a,1.0);
  Kokkos::deep_copy(b,2.0);
  auto lambda = KOKKOS_LAMBDA(int i, int j, int k) { a(i,j,k) += b(i,j,k); };

  parallel_left<3>(Kokkos::Array<int, 3>{0,0,0}, Kokkos::Array<int, 3>{500,500,500}, lambda);
  Kokkos::fence();
  Kokkos::Timer timer;
  parallel_left<3>(Kokkos::Array<int, 3>{0,0,0}, Kokkos::Array<int, 3>{500,500,500}, lambda);
  Kokkos::fence();
  double time = timer.seconds();
  int errors = 0;
  auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>>({0,0,0},{500,500,500}, {32, 8, 2});
  Kokkos::parallel_reduce(policy, KOKKOS_LAMBDA(int i, int j, int k, int& err) { if(a(i,j,k) != 5.0) err++; }, errors);
  printf("LeftNew: %lf %i\n", time*1.e6, errors);

}
void foo_right() {
  Kokkos::View<double***, Kokkos::LayoutRight> a("A",500,500,500), b("B",500,500,500);
  Kokkos::deep_copy(a,1.0);
  Kokkos::deep_copy(b,2.0);

  auto lambda = KOKKOS_LAMBDA(int i, int j, int k) { a(i,j,k) += b(i,j,k); };

  auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<3, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>({0,0,0},{500,500,500}, {32, 8, 2});
  Kokkos::parallel_for(policy, lambda);
  Kokkos::fence();
  Kokkos::Timer timer;
  Kokkos::parallel_for(policy, lambda);
  Kokkos::fence();
  double time = timer.seconds();
  int errors = 0;
  Kokkos::parallel_reduce(policy, KOKKOS_LAMBDA(int i, int j, int k, int& err) { if(a(i,j,k) != 5.0) err++; }, errors);
  printf("Right %lf %i\n", time*1.e6, errors);
}
void foo_right_new() {
  Kokkos::View<double***, Kokkos::LayoutRight> a("A",500,500,500), b("B",500,500,500);
  Kokkos::deep_copy(a,1.0);
  Kokkos::deep_copy(b,2.0);
  auto lambda = KOKKOS_LAMBDA(int i, int j, int k) { a(i,j,k) += b(i,j,k); };

  parallel_right<3>(Kokkos::Array<int, 3>{0,0,0}, Kokkos::Array<int, 3>{500,500,500}, lambda);
  Kokkos::fence();
  Kokkos::Timer timer;
  parallel_right<3>(Kokkos::Array<int, 3>{0,0,0}, Kokkos::Array<int, 3>{500,500,500}, lambda);
  Kokkos::fence();
  double time = timer.seconds();
  int errors = 0;
  auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>>({0,0,0},{500,500,500}, {32, 8, 2});
  Kokkos::parallel_reduce(policy, KOKKOS_LAMBDA(int i, int j, int k, int& err) { if(a(i,j,k) != 5.0) err++; }, errors);
  printf("RightNew: %lf %i\n", time*1.e6, errors);

}
TEST(defaultdevicetype, development_test) {foo_left_new(); foo_left(); foo_right(); foo_right_new();}

}  // namespace Test
