// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

// Simple compile test to verify mdspan-style View template parameters work

#include <Kokkos_Core.hpp>

int main() {
  Kokkos::initialize();

  // Classical style
  Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::Serial> view_classical(
      "classical", 10, 20);

  // mdspan style with dynamic extents
  Kokkos::View<double, Kokkos::dextents<size_t, 2>, Kokkos::layout_left,
               Kokkos::Impl::SpaceAwareAccessor<
                   Kokkos::Serial::memory_space,
                   Kokkos::default_accessor<double>>>
      view_mdspan("mdspan", 10, 20);

  // Verify they work the same way
  static_assert(decltype(view_classical)::rank == decltype(view_mdspan)::rank,
                "Ranks should match");
  static_assert(decltype(view_classical)::rank == 2, "Rank should be 2");

  // mdspan style with static extents
  Kokkos::View<int, Kokkos::extents<size_t, 10, 20>, Kokkos::layout_left,
               Kokkos::Impl::SpaceAwareAccessor<
                   Kokkos::Serial::memory_space,
                   Kokkos::default_accessor<int>>>
      view_static("static");

  static_assert(decltype(view_static)::rank == 2, "Rank should be 2");
  static_assert(decltype(view_static)::rank_dynamic == 0,
                "Should have no dynamic dimensions");

  // mdspan style with layout_right
  Kokkos::View<float, Kokkos::dextents<size_t, 3>, Kokkos::layout_right,
               Kokkos::Impl::SpaceAwareAccessor<
                   Kokkos::Serial::memory_space,
                   Kokkos::default_accessor<float>>>
      view_right("right", 5, 10, 15);

  static_assert(decltype(view_right)::rank == 3, "Rank should be 3");

  Kokkos::finalize();
  return 0;
}
