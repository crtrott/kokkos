// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

// Syntax check: Verify that mdspan-style View declarations compile

#include <Kokkos_Core.hpp>

// Test that both classical and mdspan-style declarations work
namespace {

// Classical style
using ViewClassical1 = Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::Serial>;
using ViewClassical2 = Kokkos::View<float*[10], Kokkos::LayoutRight, Kokkos::Serial>;
using ViewClassical3 = Kokkos::View<int[5][10], Kokkos::LayoutLeft, Kokkos::Serial>;

// mdspan style with dynamic extents
using ViewMDSpan1 = Kokkos::View<
    double,  
    Kokkos::dextents<size_t, 2>,
    Kokkos::layout_left,
    Kokkos::Impl::SpaceAwareAccessor<
        Kokkos::Serial::memory_space,
        Kokkos::default_accessor<double>>>;

// mdspan style with mixed static/dynamic extents
using ViewMDSpan2 = Kokkos::View<
    float,
    Kokkos::extents<size_t, Kokkos::dynamic_extent, 10>,
    Kokkos::layout_right,
    Kokkos::Impl::SpaceAwareAccessor<
        Kokkos::Serial::memory_space,
        Kokkos::default_accessor<float>>>;

// mdspan style with all static extents
using ViewMDSpan3 = Kokkos::View<
    int,
    Kokkos::extents<size_t, 5, 10>,
    Kokkos::layout_left,
    Kokkos::Impl::SpaceAwareAccessor<
        Kokkos::Serial::memory_space,
        Kokkos::default_accessor<int>>>;

// Verify trait compatibility
static_assert(ViewClassical1::rank == ViewMDSpan1::rank, "Ranks should match");
static_assert(ViewClassical1::rank == 2, "Rank should be 2");

static_assert(ViewClassical2::rank == ViewMDSpan2::rank, "Ranks should match");
static_assert(ViewClassical2::rank == 2, "Rank should be 2");
static_assert(ViewClassical2::rank_dynamic == ViewMDSpan2::rank_dynamic,
              "Dynamic ranks should match");
static_assert(ViewMDSpan2::rank_dynamic == 1, "Should have 1 dynamic dimension");

static_assert(ViewClassical3::rank == ViewMDSpan3::rank, "Ranks should match");
static_assert(ViewClassical3::rank == 2, "Rank should be 2");
static_assert(ViewClassical3::rank_dynamic == ViewMDSpan3::rank_dynamic,
              "Dynamic ranks should match");
static_assert(ViewMDSpan3::rank_dynamic == 0, "Should have 0 dynamic dimensions");

// Verify layouts match
static_assert(std::is_same_v<typename ViewClassical1::array_layout,
                             typename ViewMDSpan1::array_layout>,
              "Layouts should match");
static_assert(std::is_same_v<typename ViewClassical1::array_layout,
                             Kokkos::LayoutLeft>,
              "Should be LayoutLeft");

static_assert(std::is_same_v<typename ViewClassical2::array_layout,
                             typename ViewMDSpan2::array_layout>,
              "Layouts should match");
static_assert(std::is_same_v<typename ViewClassical2::array_layout,
                             Kokkos::LayoutRight>,
              "Should be LayoutRight");

// Verify memory spaces match
static_assert(std::is_same_v<typename ViewClassical1::memory_space,
                             typename ViewMDSpan1::memory_space>,
              "Memory spaces should match");
static_assert(std::is_same_v<typename ViewClassical1::memory_space,
                             Kokkos::Serial::memory_space>,
              "Should be Serial memory space");

}  // namespace

int main() {
  // Just a syntax check - if this compiles, the feature works
  return 0;
}
