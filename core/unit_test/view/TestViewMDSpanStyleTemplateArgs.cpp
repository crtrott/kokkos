// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif

#include <type_traits>

// Compile-only test that verifies Views defined using mdspan-style template
// arguments (scalartype, extents, layout, accessor) produce the same
// mdspan_type as Views defined using traditional Kokkos style (datatype,
// layout, space, memorytraits).
//
// These tests use Kokkos::Experimental::Accessor to create accessors from
// traditional template arguments (scalar_type, space, MemoryTraits).

namespace {

// Helper to generate rank asterisks for traditional View syntax
template <size_t Rank>
struct rank_to_asterisks;

template <>
struct rank_to_asterisks<1> {
  template <class T>
  using type = T*;
};
template <>
struct rank_to_asterisks<2> {
  template <class T>
  using type = T**;
};
template <>
struct rank_to_asterisks<3> {
  template <class T>
  using type = T***;
};
template <>
struct rank_to_asterisks<4> {
  template <class T>
  using type = T****;
};
template <>
struct rank_to_asterisks<5> {
  template <class T>
  using type = T*****;
};
template <>
struct rank_to_asterisks<6> {
  template <class T>
  using type = T******;
};

// Helper to map Kokkos::LayoutLeft/Right to mdspan layout
template <class Layout>
struct layout_to_mdspan;

template <>
struct layout_to_mdspan<Kokkos::LayoutLeft> {
  using type = Kokkos::Experimental::layout_left_padded<Kokkos::dynamic_extent>;
};

template <>
struct layout_to_mdspan<Kokkos::LayoutRight> {
  using type =
      Kokkos::Experimental::layout_right_padded<Kokkos::dynamic_extent>;
};

// Helper struct to construct and compare View types
// ElementType: scalar type (e.g., double, const int)
// Rank: number of dimensions
// Layout: Kokkos::LayoutLeft or Kokkos::LayoutRight
// Space: execution/memory space
// MemTraits: Kokkos::MemoryTraits<...>
template <class ElementType, size_t Rank, class Layout, class Space,
          class MemTraits = Kokkos::MemoryTraits<>>
struct test_mdspan_view_equivalence {
  // Traditional View: View<T*, Layout, Space, MemTraits>
  using traditional_view =
      Kokkos::View<typename rank_to_asterisks<Rank>::template type<ElementType>,
                   Layout, Space, MemTraits>;

  // mdspan-style View: View<T, dextents<size_t, Rank>, layout_*_padded,
  // Accessor>
  using mdspan_style_view = Kokkos::View<
      ElementType, Kokkos::dextents<size_t, Rank>,
      typename layout_to_mdspan<Layout>::type,
      Kokkos::Experimental::Accessor<ElementType, Space, MemTraits>>;

  // Verify mdspan_type equivalence
  static constexpr bool value =
      std::is_same_v<typename traditional_view::mdspan_type,
                     typename mdspan_style_view::mdspan_type>;
};

}  // namespace

// Test 1: Basic 1D dynamic array with default traits
static_assert(
    test_mdspan_view_equivalence<double, 1, Kokkos::LayoutRight,
                                 Kokkos::DefaultExecutionSpace>::value);

// Test 2: 2D dynamic array with default traits
static_assert(
    test_mdspan_view_equivalence<int, 2, Kokkos::LayoutRight,
                                 Kokkos::DefaultExecutionSpace>::value);

// Test 3: 3D dynamic array with LayoutLeft
static_assert(
    test_mdspan_view_equivalence<float, 3, Kokkos::LayoutLeft,
                                 Kokkos::DefaultExecutionSpace>::value);

// Test 4: 4D dynamic array with Unmanaged memory trait
static_assert(test_mdspan_view_equivalence<
              double, 4, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace,
              Kokkos::MemoryTraits<Kokkos::Unmanaged>>::value);

// Test 5: 2D array with Atomic memory trait
static_assert(test_mdspan_view_equivalence<
              int, 2, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace,
              Kokkos::MemoryTraits<Kokkos::Atomic>>::value);

// Test 6: 3D array with combined Unmanaged and Atomic traits
static_assert(test_mdspan_view_equivalence<
              float, 3, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace,
              Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::Atomic>>::value);

// Test 7: Const value type
static_assert(
    test_mdspan_view_equivalence<const double, 2, Kokkos::LayoutRight,
                                 Kokkos::DefaultExecutionSpace>::value);

// Test 8: 5D array with LayoutLeft and Unmanaged
static_assert(test_mdspan_view_equivalence<
              long, 5, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace,
              Kokkos::MemoryTraits<Kokkos::Unmanaged>>::value);

// Test 9: 1D array with RandomAccess trait
static_assert(test_mdspan_view_equivalence<
              double, 1, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace,
              Kokkos::MemoryTraits<Kokkos::RandomAccess>>::value);

// Test 10: 6D array with all supported dimensions
static_assert(
    test_mdspan_view_equivalence<int, 6, Kokkos::LayoutRight,
                                 Kokkos::DefaultExecutionSpace>::value);

// Test 11: 2D array with LayoutRight explicitly specified
static_assert(
    test_mdspan_view_equivalence<short, 2, Kokkos::LayoutRight,
                                 Kokkos::DefaultExecutionSpace>::value);

// Test 12: 3D array with Restrict trait
static_assert(test_mdspan_view_equivalence<
              double, 3, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace,
              Kokkos::MemoryTraits<Kokkos::Restrict>>::value);

// Test 13: Verify HostSpace also works
static_assert(test_mdspan_view_equivalence<float, 2, Kokkos::LayoutRight,
                                           Kokkos::HostSpace>::value);

// Test 14: HostSpace with LayoutLeft
static_assert(test_mdspan_view_equivalence<int, 3, Kokkos::LayoutLeft,
                                           Kokkos::HostSpace>::value);

// Test 15: Complex type with multiple traits
static_assert(
    test_mdspan_view_equivalence<
        const long, 4, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace,
        Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess>>::value);
