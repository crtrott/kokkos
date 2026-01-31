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

// Test 1: Basic 1D dynamic array with default traits
static_assert(
    std::is_same_v<
        Kokkos::View<double*, Kokkos::DefaultExecutionSpace>::mdspan_type,
        Kokkos::View<
            double, Kokkos::dextents<size_t, 1>,
            Kokkos::Experimental::layout_right_padded<Kokkos::dynamic_extent>,
            Kokkos::Experimental::Accessor<
                double, Kokkos::DefaultExecutionSpace,
                Kokkos::MemoryTraits<>>>::mdspan_type>);

// Test 2: 2D dynamic array with default traits
static_assert(
    std::is_same_v<
        Kokkos::View<int**, Kokkos::DefaultExecutionSpace>::mdspan_type,
        Kokkos::View<
            int, Kokkos::dextents<size_t, 2>,
            Kokkos::Experimental::layout_right_padded<Kokkos::dynamic_extent>,
            Kokkos::Experimental::Accessor<int, Kokkos::DefaultExecutionSpace,
                                           Kokkos::MemoryTraits<>>>::
            mdspan_type>);

// Test 3: 3D dynamic array with LayoutLeft
static_assert(
    std::is_same_v<Kokkos::View<float***, Kokkos::LayoutLeft,
                                Kokkos::DefaultExecutionSpace>::mdspan_type,
                   Kokkos::View<float, Kokkos::dextents<size_t, 3>,
                                Kokkos::Experimental::layout_left_padded<
                                    Kokkos::dynamic_extent>,
                                Kokkos::Experimental::Accessor<
                                    float, Kokkos::DefaultExecutionSpace,
                                    Kokkos::MemoryTraits<>>>::mdspan_type>);

// Test 4: 4D dynamic array with Unmanaged memory trait
static_assert(
    std::is_same_v<
        Kokkos::View<double****, Kokkos::DefaultExecutionSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>::mdspan_type,
        Kokkos::View<
            double, Kokkos::dextents<size_t, 4>,
            Kokkos::Experimental::layout_right_padded<Kokkos::dynamic_extent>,
            Kokkos::Experimental::Accessor<
                double, Kokkos::DefaultExecutionSpace,
                Kokkos::MemoryTraits<Kokkos::Unmanaged>>>::mdspan_type>);

// Test 5: 2D array with Atomic memory trait
static_assert(
    std::is_same_v<
        Kokkos::View<int**, Kokkos::DefaultExecutionSpace,
                     Kokkos::MemoryTraits<Kokkos::Atomic>>::mdspan_type,
        Kokkos::View<
            int, Kokkos::dextents<size_t, 2>,
            Kokkos::Experimental::layout_right_padded<Kokkos::dynamic_extent>,
            Kokkos::Experimental::Accessor<
                int, Kokkos::DefaultExecutionSpace,
                Kokkos::MemoryTraits<Kokkos::Atomic>>>::mdspan_type>);

// Test 6: 3D array with combined Unmanaged and Atomic traits
static_assert(
    std::is_same_v<
        Kokkos::View<float***, Kokkos::LayoutLeft,
                     Kokkos::DefaultExecutionSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::Atomic>>::
            mdspan_type,
        Kokkos::View<
            float, Kokkos::dextents<size_t, 3>,
            Kokkos::Experimental::layout_left_padded<Kokkos::dynamic_extent>,
            Kokkos::Experimental::Accessor<
                float, Kokkos::DefaultExecutionSpace,
                Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::Atomic>>>::
            mdspan_type>);

// Test 7: Const value type
static_assert(
    std::is_same_v<Kokkos::View<const double**,
                                Kokkos::DefaultExecutionSpace>::mdspan_type,
                   Kokkos::View<const double, Kokkos::dextents<size_t, 2>,
                                Kokkos::Experimental::layout_right_padded<
                                    Kokkos::dynamic_extent>,
                                Kokkos::Experimental::Accessor<
                                    const double, Kokkos::DefaultExecutionSpace,
                                    Kokkos::MemoryTraits<>>>::mdspan_type>);

// Test 8: 5D array with LayoutLeft and Unmanaged
static_assert(
    std::is_same_v<
        Kokkos::View<long*****, Kokkos::LayoutLeft,
                     Kokkos::DefaultExecutionSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>::mdspan_type,
        Kokkos::View<
            long, Kokkos::dextents<size_t, 5>,
            Kokkos::Experimental::layout_left_padded<Kokkos::dynamic_extent>,
            Kokkos::Experimental::Accessor<
                long, Kokkos::DefaultExecutionSpace,
                Kokkos::MemoryTraits<Kokkos::Unmanaged>>>::mdspan_type>);

// Test 9: 1D array with RandomAccess trait
static_assert(
    std::is_same_v<
        Kokkos::View<double*, Kokkos::DefaultExecutionSpace,
                     Kokkos::MemoryTraits<Kokkos::RandomAccess>>::mdspan_type,
        Kokkos::View<
            double, Kokkos::dextents<size_t, 1>,
            Kokkos::Experimental::layout_right_padded<Kokkos::dynamic_extent>,
            Kokkos::Experimental::Accessor<
                double, Kokkos::DefaultExecutionSpace,
                Kokkos::MemoryTraits<Kokkos::RandomAccess>>>::mdspan_type>);

// Test 10: 6D array with all supported dimensions
static_assert(
    std::is_same_v<
        Kokkos::View<int******, Kokkos::DefaultExecutionSpace>::mdspan_type,
        Kokkos::View<
            int, Kokkos::dextents<size_t, 6>,
            Kokkos::Experimental::layout_right_padded<Kokkos::dynamic_extent>,
            Kokkos::Experimental::Accessor<int, Kokkos::DefaultExecutionSpace,
                                           Kokkos::MemoryTraits<>>>::
            mdspan_type>);

// Test 11: 2D array with LayoutRight explicitly specified
static_assert(
    std::is_same_v<Kokkos::View<short**, Kokkos::LayoutRight,
                                Kokkos::DefaultExecutionSpace>::mdspan_type,
                   Kokkos::View<short, Kokkos::dextents<size_t, 2>,
                                Kokkos::Experimental::layout_right_padded<
                                    Kokkos::dynamic_extent>,
                                Kokkos::Experimental::Accessor<
                                    short, Kokkos::DefaultExecutionSpace,
                                    Kokkos::MemoryTraits<>>>::mdspan_type>);

// Test 12: 3D array with Restrict trait
static_assert(
    std::is_same_v<
        Kokkos::View<double***, Kokkos::DefaultExecutionSpace,
                     Kokkos::MemoryTraits<Kokkos::Restrict>>::mdspan_type,
        Kokkos::View<
            double, Kokkos::dextents<size_t, 3>,
            Kokkos::Experimental::layout_right_padded<Kokkos::dynamic_extent>,
            Kokkos::Experimental::Accessor<
                double, Kokkos::DefaultExecutionSpace,
                Kokkos::MemoryTraits<Kokkos::Restrict>>>::mdspan_type>);

// Test 13: Verify HostSpace also works
static_assert(
    std::is_same_v<Kokkos::View<float**, Kokkos::HostSpace>::mdspan_type,
                   Kokkos::View<float, Kokkos::dextents<size_t, 2>,
                                Kokkos::Experimental::layout_right_padded<
                                    Kokkos::dynamic_extent>,
                                Kokkos::Experimental::Accessor<
                                    float, Kokkos::HostSpace,
                                    Kokkos::MemoryTraits<>>>::mdspan_type>);

// Test 14: HostSpace with LayoutLeft
static_assert(
    std::is_same_v<
        Kokkos::View<int***, Kokkos::LayoutLeft,
                     Kokkos::HostSpace>::mdspan_type,
        Kokkos::View<
            int, Kokkos::dextents<size_t, 3>,
            Kokkos::Experimental::layout_left_padded<Kokkos::dynamic_extent>,
            Kokkos::Experimental::Accessor<
                int, Kokkos::HostSpace, Kokkos::MemoryTraits<>>>::mdspan_type>);

// Test 15: Complex type with multiple traits
static_assert(
    std::is_same_v<
        Kokkos::View<const long****, Kokkos::LayoutLeft,
                     Kokkos::DefaultExecutionSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged |
                                          Kokkos::RandomAccess>>::mdspan_type,
        Kokkos::View<
            const long, Kokkos::dextents<size_t, 4>,
            Kokkos::Experimental::layout_left_padded<Kokkos::dynamic_extent>,
            Kokkos::Experimental::Accessor<
                const long, Kokkos::DefaultExecutionSpace,
                Kokkos::MemoryTraits<Kokkos::Unmanaged |
                                     Kokkos::RandomAccess>>>::mdspan_type>);
