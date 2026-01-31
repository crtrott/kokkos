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

// Test mdspan-style template arguments in Kokkos::View
// Comparing traditional style (datatype, layout, space, memorytraits)
// with mdspan style (scalartype, extents, layout, accessor)

// Test 1: Basic 1D dynamic array with default traits
static_assert(
    std::is_same_v<
        Kokkos::View<double*, TEST_EXECSPACE>::mdspan_type,
        Kokkos::View<
            double, Kokkos::dextents<size_t, 1>,
            Kokkos::Experimental::layout_right_padded<Kokkos::dynamic_extent>,
            Kokkos::Experimental::Accessor<double, TEST_EXECSPACE,
                                           Kokkos::MemoryTraits<>>>::
            mdspan_type>);

// Test 2: 2D dynamic array with default traits
static_assert(
    std::is_same_v<
        Kokkos::View<int**, TEST_EXECSPACE>::mdspan_type,
        Kokkos::View<
            int, Kokkos::dextents<size_t, 2>,
            Kokkos::Experimental::layout_right_padded<Kokkos::dynamic_extent>,
            Kokkos::Experimental::Accessor<int, TEST_EXECSPACE,
                                           Kokkos::MemoryTraits<>>>::
            mdspan_type>);

// Test 3: 3D dynamic array with LayoutLeft
static_assert(
    std::is_same_v<
        Kokkos::View<float***, Kokkos::LayoutLeft, TEST_EXECSPACE>::mdspan_type,
        Kokkos::View<
            float, Kokkos::dextents<size_t, 3>,
            Kokkos::Experimental::layout_left_padded<Kokkos::dynamic_extent>,
            Kokkos::Experimental::Accessor<float, TEST_EXECSPACE,
                                           Kokkos::MemoryTraits<>>>::
            mdspan_type>);

// Test 4: 4D dynamic array with Unmanaged memory trait
static_assert(
    std::is_same_v<
        Kokkos::View<double****, TEST_EXECSPACE,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>::mdspan_type,
        Kokkos::View<
            double, Kokkos::dextents<size_t, 4>,
            Kokkos::Experimental::layout_right_padded<Kokkos::dynamic_extent>,
            Kokkos::Experimental::Accessor<
                double, TEST_EXECSPACE,
                Kokkos::MemoryTraits<Kokkos::Unmanaged>>>::mdspan_type>);

// Test 5: 2D array with Atomic memory trait
static_assert(
    std::is_same_v<
        Kokkos::View<int**, TEST_EXECSPACE,
                     Kokkos::MemoryTraits<Kokkos::Atomic>>::mdspan_type,
        Kokkos::View<
            int, Kokkos::dextents<size_t, 2>,
            Kokkos::Experimental::layout_right_padded<Kokkos::dynamic_extent>,
            Kokkos::Experimental::Accessor<
                int, TEST_EXECSPACE, Kokkos::MemoryTraits<Kokkos::Atomic>>>::
            mdspan_type>);

// Test 6: 3D array with combined Unmanaged and Atomic traits
static_assert(
    std::is_same_v<
        Kokkos::View<float***, Kokkos::LayoutLeft, TEST_EXECSPACE,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::Atomic>>::
            mdspan_type,
        Kokkos::View<
            float, Kokkos::dextents<size_t, 3>,
            Kokkos::Experimental::layout_left_padded<Kokkos::dynamic_extent>,
            Kokkos::Experimental::Accessor<
                float, TEST_EXECSPACE,
                Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::Atomic>>>::
            mdspan_type>);

// Test 7: Const value type
static_assert(
    std::is_same_v<
        Kokkos::View<const double**, TEST_EXECSPACE>::mdspan_type,
        Kokkos::View<
            const double, Kokkos::dextents<size_t, 2>,
            Kokkos::Experimental::layout_right_padded<Kokkos::dynamic_extent>,
            Kokkos::Experimental::Accessor<const double, TEST_EXECSPACE,
                                           Kokkos::MemoryTraits<>>>::
            mdspan_type>);

// Test 8: 5D array with LayoutLeft and Unmanaged
static_assert(
    std::is_same_v<
        Kokkos::View<long*****, Kokkos::LayoutLeft, TEST_EXECSPACE,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>::mdspan_type,
        Kokkos::View<
            long, Kokkos::dextents<size_t, 5>,
            Kokkos::Experimental::layout_left_padded<Kokkos::dynamic_extent>,
            Kokkos::Experimental::Accessor<
                long, TEST_EXECSPACE,
                Kokkos::MemoryTraits<Kokkos::Unmanaged>>>::mdspan_type>);

// Test 9: 1D array with RandomAccess trait
static_assert(
    std::is_same_v<
        Kokkos::View<double*, TEST_EXECSPACE,
                     Kokkos::MemoryTraits<Kokkos::RandomAccess>>::mdspan_type,
        Kokkos::View<
            double, Kokkos::dextents<size_t, 1>,
            Kokkos::Experimental::layout_right_padded<Kokkos::dynamic_extent>,
            Kokkos::Experimental::Accessor<
                double, TEST_EXECSPACE,
                Kokkos::MemoryTraits<Kokkos::RandomAccess>>>::mdspan_type>);

// Test 10: 6D array with all supported dimensions
static_assert(
    std::is_same_v<
        Kokkos::View<int******, TEST_EXECSPACE>::mdspan_type,
        Kokkos::View<
            int, Kokkos::dextents<size_t, 6>,
            Kokkos::Experimental::layout_right_padded<Kokkos::dynamic_extent>,
            Kokkos::Experimental::Accessor<int, TEST_EXECSPACE,
                                           Kokkos::MemoryTraits<>>>::
            mdspan_type>);

TEST(defaultdevicetype, development_test) {}

}  // namespace Test
