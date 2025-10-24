// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Core.hpp>

namespace Test {
template <class Policy, class ExpectedExecType, class ExpectedIndex>
constexpr bool check_compile_time_inputs() {
  static_assert(
      std::same_as<typename Policy::execution_type, ExpectedExecType>);
  static_assert(std::same_as<typename Policy::index_type, ExpectedIndex>);
  return true;
}

using TeamPolicy = Kokkos::TeamPolicy<>;
using TeamHandle = TeamPolicy::member_type;

using DefaultIndex    = typename TeamHandle::execution_space::size_type;
using LongIndex       = Kokkos::IndexType<long>;
using DynamicSchedule = Kokkos::Schedule<Kokkos::Dynamic>;
struct SomeTag {};

// clang-format off
static_assert(check_compile_time_inputs<Kokkos::RangePolicy<TeamHandle                                           >, TeamHandle, DefaultIndex>());
static_assert(check_compile_time_inputs<Kokkos::RangePolicy<TeamHandle                                           >, TeamHandle, DefaultIndex>());
static_assert(check_compile_time_inputs<Kokkos::RangePolicy<TeamHandle, DynamicSchedule                          >, TeamHandle, DefaultIndex>());
static_assert(check_compile_time_inputs<Kokkos::RangePolicy<TeamHandle, DynamicSchedule, SomeTag                 >, TeamHandle, DefaultIndex>());
static_assert(check_compile_time_inputs<Kokkos::RangePolicy<TeamHandle, LongIndex                                >, TeamHandle, long>());
static_assert(check_compile_time_inputs<Kokkos::RangePolicy<TeamHandle, DynamicSchedule, LongIndex               >, TeamHandle, long>());
static_assert(check_compile_time_inputs<Kokkos::RangePolicy<TeamHandle, LongIndex,       DynamicSchedule         >, TeamHandle, long>());
static_assert(check_compile_time_inputs<Kokkos::RangePolicy<TeamHandle, LongIndex,       DynamicSchedule, SomeTag>, TeamHandle, long>());
static_assert(check_compile_time_inputs<Kokkos::RangePolicy<TeamHandle, DynamicSchedule, LongIndex,       SomeTag>, TeamHandle, long>());
// clang-format on

template <class ExecType, class IndexType>
KOKKOS_INLINE_FUNCTION int check_runtime_inputs(
    const ExecType& exec, const IndexType beg, const IndexType end,
    const IndexType chunk_size = 0) {
  auto p    = Kokkos::RangePolicy(exec, beg, end);
  int nerrs = 0;

  if (p.begin() != beg) ++nerrs;
  if (p.end() != end) ++nerrs;

  auto p2 = p.set_chunk_size(chunk_size);
  if constexpr (Kokkos::ExecutionSpace<ExecType>)
    if (p2.chunk_size() != chunk_size) ++nerrs;

  return nerrs;
}

void test_self_similar_range_policy_runtime() {
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  using IndexType = ExecSpace::size_type;

  IndexType beg        = 5;
  IndexType end        = 15;
  IndexType chunk_size = 10;

  auto nerrs_exec_space =
      check_runtime_inputs(ExecSpace(), beg, end, chunk_size);
  ASSERT_EQ(nerrs_exec_space, 0);

  int nerrs_team_handle;
  using team_t = typename Kokkos::TeamPolicy<>::member_type;
  Kokkos::parallel_reduce(
      "check_runtime", Kokkos::TeamPolicy(1, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const team_t& team, int& nerrs) {
        nerrs = check_runtime_inputs(team, beg, end);
      },
      nerrs_team_handle);
  ASSERT_EQ(nerrs_team_handle, 0);
}

template <class Exec, class X, class Y>
KOKKOS_INLINE_FUNCTION void sum_views(const Exec& exec, const X& x,
                                      const Y& y) {
  auto policy = Kokkos::RangePolicy(exec, 0, x.extent(0));
  Kokkos::parallel_for(
      policy, KOKKOS_LAMBDA(const int& i) { x(i) += y(i); });
}

void test_self_similar_range_policy_computation() {
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

TEST(TEST_CATEGORY, self_similar_range_policy_runtime) {
  test_self_similar_range_policy_runtime();
}

TEST(TEST_CATEGORY, self_similar_range_policy_computation) {
  test_self_similar_range_policy_computation();
}

}  // namespace Test
