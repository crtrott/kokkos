// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Core.hpp>

namespace Test {

template <class Exec, class X, class Y>
KOKKOS_INLINE_FUNCTION void sum_views(const Exec& exec, const X& x,
                                      const Y& y) {
  auto policy = Kokkos::RangePolicy(exec, 0, x.extent(0));
  Kokkos::parallel_for(
      policy, KOKKOS_LAMBDA(const int& i) { x(i) += y(i); });
}

void test_self_similar_range() {
  int N         = 7;
  int num_teams = 5;

  Kokkos::View<float*> v_x("v_x", N), v_y("v_y", N);
  Kokkos::View<float**> M_x("M_x", num_teams, N), M_y("M_y", num_teams, N);
  Kokkos::deep_copy(v_x, 1);
  Kokkos::deep_copy(v_y, 2);
  Kokkos::deep_copy(M_x, 1);
  Kokkos::deep_copy(M_y, 2);

  // Call sum_views(ExecSpace)
  sum_views(Kokkos::DefaultExecutionSpace(), v_x, v_y);

  // call sum_views(TeamHandle)
  using team_t = typename Kokkos::TeamPolicy<>::member_type;
  Kokkos::parallel_for(
      "apxyFromTeam", Kokkos::TeamPolicy(num_teams, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const team_t& team) {
        sum_views(team, Kokkos::subview(M_x, team.league_rank(), Kokkos::ALL()),
                  Kokkos::subview(M_y, team.league_rank(), Kokkos::ALL()));
      });

  // check
  size_t result = 0;
  Kokkos::parallel_reduce(
      "Check1", v_x.extent(0),
      KOKKOS_LAMBDA(int i, size_t& val) { val += v_x(i); }, result);
  ASSERT_EQ(result, size_t(3) * v_x.extent(0));
  Kokkos::parallel_reduce(
      "Check2", M_x.extent(0),
      KOKKOS_LAMBDA(int i, size_t& val) {
        for (int j = 0; j < M_x.extent_int(1); j++) val += M_x(i, j);
      },
      result);
  ASSERT_EQ(result, size_t(3) * M_x.extent(0) * M_x.extent(1));
}

TEST(TEST_CATEGORY, self_similar_range_policy) { test_self_similar_range(); }

}  // namespace Test
