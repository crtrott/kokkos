// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_UNITTEST_LAYOUTTILED_HPP
#define KOKKOS_UNITTEST_LAYOUTTILED_HPP

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif
#include <gtest/gtest.h>

#ifdef KOKKOS_ENABLE_IMPL_MDSPAN

namespace {

// ----------------------------------------------------------------------------
// Compile-time properties
// ----------------------------------------------------------------------------
TEST(TEST_CATEGORY, layout_tiled_static_properties) {
  using mapping_t =
      Kokkos::layout_tiled<4, 8>::mapping<Kokkos::dextents<int, 2>>;

  EXPECT_TRUE(mapping_t::is_always_unique());
  EXPECT_FALSE(mapping_t::is_always_exhaustive());
  EXPECT_FALSE(mapping_t::is_always_strided());
}

// ----------------------------------------------------------------------------
// required_span_size
// ----------------------------------------------------------------------------
TEST(TEST_CATEGORY, layout_tiled_required_span_size) {
  // Exact multiple of tile size – no padding required
  {
    Kokkos::layout_tiled<4, 8>::mapping<Kokkos::dextents<int, 2>> m(
        Kokkos::dextents<int, 2>{8, 16});
    EXPECT_EQ(m.required_span_size(), 8 * 16);
  }
  // Non-multiple – span must be padded up to complete tiles
  {
    Kokkos::layout_tiled<4, 8>::mapping<Kokkos::dextents<int, 2>> m(
        Kokkos::dextents<int, 2>{5, 10});
    // ceil(5/4)*4 = 8, ceil(10/8)*8 = 16  ->  8*16 = 128
    EXPECT_EQ(m.required_span_size(), 128);
  }
  // Single-element extents
  {
    Kokkos::layout_tiled<4, 8>::mapping<Kokkos::dextents<int, 2>> m(
        Kokkos::dextents<int, 2>{1, 1});
    // ceil(1/4)*4 * ceil(1/8)*8 = 4*8 = 32
    EXPECT_EQ(m.required_span_size(), 32);
  }
}

// ----------------------------------------------------------------------------
// is_exhaustive
// ----------------------------------------------------------------------------
TEST(TEST_CATEGORY, layout_tiled_is_exhaustive) {
  {
    Kokkos::layout_tiled<4, 8>::mapping<Kokkos::dextents<int, 2>> m(
        Kokkos::dextents<int, 2>{8, 16});
    EXPECT_TRUE(m.is_exhaustive());
  }
  {
    Kokkos::layout_tiled<4, 8>::mapping<Kokkos::dextents<int, 2>> m(
        Kokkos::dextents<int, 2>{5, 10});
    EXPECT_FALSE(m.is_exhaustive());
  }
}

// ----------------------------------------------------------------------------
// Index mapping correctness
// ----------------------------------------------------------------------------
TEST(TEST_CATEGORY, layout_tiled_index_mapping) {
  // 2×2 tiles of size 2×2 in a 4×4 matrix
  // Tiles (row-major):
  //   tile (0,0): elements (0-1, 0-1)  -> offsets 0-3
  //   tile (0,1): elements (0-1, 2-3)  -> offsets 4-7
  //   tile (1,0): elements (2-3, 0-1)  -> offsets 8-11
  //   tile (1,1): elements (2-3, 2-3)  -> offsets 12-15
  Kokkos::layout_tiled<2, 2>::mapping<Kokkos::dextents<int, 2>> m(
      Kokkos::dextents<int, 2>{4, 4});

  // tile (0,0)
  EXPECT_EQ(m(0, 0), 0);
  EXPECT_EQ(m(0, 1), 1);
  EXPECT_EQ(m(1, 0), 2);
  EXPECT_EQ(m(1, 1), 3);
  // tile (0,1)
  EXPECT_EQ(m(0, 2), 4);
  EXPECT_EQ(m(0, 3), 5);
  EXPECT_EQ(m(1, 2), 6);
  EXPECT_EQ(m(1, 3), 7);
  // tile (1,0)
  EXPECT_EQ(m(2, 0), 8);
  EXPECT_EQ(m(2, 1), 9);
  EXPECT_EQ(m(3, 0), 10);
  EXPECT_EQ(m(3, 1), 11);
  // tile (1,1)
  EXPECT_EQ(m(2, 2), 12);
  EXPECT_EQ(m(2, 3), 13);
  EXPECT_EQ(m(3, 2), 14);
  EXPECT_EQ(m(3, 3), 15);
}

// ----------------------------------------------------------------------------
// Uniqueness: every distinct (i0,i1) maps to a distinct offset
// ----------------------------------------------------------------------------
TEST(TEST_CATEGORY, layout_tiled_unique_mapping) {
  constexpr int M = 6, N = 10;
  Kokkos::layout_tiled<3, 5>::mapping<Kokkos::dextents<int, 2>> m(
      Kokkos::dextents<int, 2>{M, N});

  std::vector<int> seen(m.required_span_size(), 0);
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      auto idx = m(i, j);
      EXPECT_GE(idx, 0);
      EXPECT_LT(idx, m.required_span_size());
      EXPECT_EQ(seen[idx], 0) << "duplicate offset " << idx << " at (" << i
                               << "," << j << ")";
      seen[idx]++;
    }
  }
}

// ----------------------------------------------------------------------------
// mdspan construction from problem statement
// ----------------------------------------------------------------------------
TEST(TEST_CATEGORY, layout_tiled_mdspan_construction) {
  constexpr int M = 1000, N = 2000;
  // Allocate enough storage (padded span)
  using layout_t = Kokkos::layout_tiled<4, 8>;
  using extents_t = Kokkos::dextents<int, 2>;
  layout_t::mapping<extents_t> m(extents_t{M, N});
  std::vector<int> storage(m.required_span_size(), 0);
  int* ptr = storage.data();

  Kokkos::mdspan<int, Kokkos::dextents<int, 2>, Kokkos::layout_tiled<4, 8>> a(
      ptr, M, N);

  EXPECT_EQ(a.extent(0), M);
  EXPECT_EQ(a.extent(1), N);
  // Verify a write round-trip through the tiled mdspan
  a(42, 77) = 123;
  EXPECT_EQ(a(42, 77), 123);
}

// ----------------------------------------------------------------------------
// mapping equality
// ----------------------------------------------------------------------------
TEST(TEST_CATEGORY, layout_tiled_mapping_equality) {
  Kokkos::layout_tiled<4, 8>::mapping<Kokkos::dextents<int, 2>> m1(
      Kokkos::dextents<int, 2>{8, 16});
  Kokkos::layout_tiled<4, 8>::mapping<Kokkos::dextents<int, 2>> m2(
      Kokkos::dextents<int, 2>{8, 16});
  Kokkos::layout_tiled<4, 8>::mapping<Kokkos::dextents<int, 2>> m3(
      Kokkos::dextents<int, 2>{4, 8});

  EXPECT_EQ(m1, m2);
  EXPECT_NE(m1, m3);
}

}  // namespace

#endif  // KOKKOS_ENABLE_IMPL_MDSPAN

#endif  // KOKKOS_UNITTEST_LAYOUTTILED_HPP
