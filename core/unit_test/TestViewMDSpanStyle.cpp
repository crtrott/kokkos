// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

namespace {

// Test that mdspan-style View template parameters work
TEST(ViewMDSpanStyle, BasicDeclaration) {
  // Classical style: View<double**, LayoutLeft, Serial>
  using ViewClassical = Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::Serial>;

  // mdspan style: View<double, dextents<size_t, 2>, layout_left, accessor>
  using ViewMDSpan = Kokkos::View<
      double, Kokkos::dextents<size_t, 2>, Kokkos::layout_left,
      Kokkos::Impl::SpaceAwareAccessor<
          Kokkos::Serial::memory_space,
          Kokkos::default_accessor<double>>>;

  // Both should have compatible traits
  static_assert(ViewClassical::rank == ViewMDSpan::rank,
                "Ranks should match");
  static_assert(ViewClassical::rank == 2, "Rank should be 2");
  static_assert(std::is_same_v<typename ViewClassical::array_layout,
                               typename ViewMDSpan::array_layout>,
                "Layouts should match");
  static_assert(std::is_same_v<typename ViewClassical::memory_space,
                               typename ViewMDSpan::memory_space>,
                "Memory spaces should match");
}

// Test with LayoutRight
TEST(ViewMDSpanStyle, LayoutRight) {
  using ViewClassical = Kokkos::View<float***, Kokkos::LayoutRight, Kokkos::Serial>;

  using ViewMDSpan = Kokkos::View<
      float, Kokkos::dextents<size_t, 3>, Kokkos::layout_right,
      Kokkos::Impl::SpaceAwareAccessor<
          Kokkos::Serial::memory_space,
          Kokkos::default_accessor<float>>>;

  static_assert(ViewClassical::rank == ViewMDSpan::rank,
                "Ranks should match");
  static_assert(ViewClassical::rank == 3, "Rank should be 3");
  static_assert(std::is_same_v<typename ViewClassical::array_layout,
                               typename ViewMDSpan::array_layout>,
                "Layouts should match");
}

// Test with static extents
TEST(ViewMDSpanStyle, StaticExtents) {
  using ViewClassical = Kokkos::View<int[10][20], Kokkos::LayoutLeft, Kokkos::Serial>;

  using ViewMDSpan = Kokkos::View<
      int, Kokkos::extents<size_t, 10, 20>, Kokkos::layout_left,
      Kokkos::Impl::SpaceAwareAccessor<
          Kokkos::Serial::memory_space,
          Kokkos::default_accessor<int>>>;

  static_assert(ViewClassical::rank == ViewMDSpan::rank,
                "Ranks should match");
  static_assert(ViewClassical::rank == 2, "Rank should be 2");
  static_assert(ViewClassical::rank_dynamic == 0,
                "Should have no dynamic dimensions");
  static_assert(ViewMDSpan::rank_dynamic == 0,
                "Should have no dynamic dimensions");
}

// Test with mixed static/dynamic extents
TEST(ViewMDSpanStyle, MixedExtents) {
  using ViewClassical = Kokkos::View<double*[5], Kokkos::LayoutLeft, Kokkos::Serial>;

  using ViewMDSpan = Kokkos::View<
      double, Kokkos::extents<size_t, Kokkos::dynamic_extent, 5>,
      Kokkos::layout_left,
      Kokkos::Impl::SpaceAwareAccessor<
          Kokkos::Serial::memory_space,
          Kokkos::default_accessor<double>>>;

  static_assert(ViewClassical::rank == ViewMDSpan::rank,
                "Ranks should match");
  static_assert(ViewClassical::rank == 2, "Rank should be 2");
  static_assert(ViewClassical::rank_dynamic == 1,
                "Should have 1 dynamic dimension");
  static_assert(ViewMDSpan::rank_dynamic == 1,
                "Should have 1 dynamic dimension");
}

}  // namespace
