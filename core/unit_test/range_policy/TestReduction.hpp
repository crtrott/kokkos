// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Core.hpp>

namespace Test {

void test_range_policy_parallel_reduce_sharedspace_view() {
  Kokkos::View<int, Kokkos::Device<TEST_EXECSPACE, Kokkos::SharedSpace>>
    result("result");
  int N = 1037;
  Kokkos::parallel_reduce("TestParReduce_To_SharedSpace",
    Kokkos::RangePolicy<TEST_EXECSPACE>(0,N), KOKKOS_LAMBDA(int i, int& lsum) {
    lsum += i;
  },result);
  Kokkos::fence();
  ASSERT_EQ(result(), N*(N-1)/2);
}

TEST(TEST_CATEGORY, range_policy_parallel_reduce_sharedspace_view) {
  test_range_policy_parallel_reduce_sharedspace_view();
}

}  // namespace Test
