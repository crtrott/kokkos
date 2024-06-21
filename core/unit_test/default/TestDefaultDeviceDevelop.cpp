//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include <TestDefaultDeviceType_Category.hpp>

namespace Test {

TEST(defaultdevicetype, development_test) {

  Kokkos::View<float*> a("A", 100);

  static_assert(std::is_same_v<typename Kokkos::Impl::SpaceAwareAccessor<Kokkos::HostSpace, Kokkos::default_accessor<float>>::reference, float&>);
  static_assert(std::is_same_v<decltype(Kokkos::Impl::SpaceAwareAccessor<Kokkos::HostSpace, Kokkos::default_accessor<float>>().access(a.data(),0)), float&>);
  static_assert(std::is_same_v<decltype((typename Kokkos::View<float*>::mdspan_type{a.data(), 100})(0)), float&>);
  static_assert(std::is_same_v<decltype(a(0)), float&>);

  for(int i=0; i<100; i++) {
    a(i) = i;
    ASSERT_EQ(a(i), float(i));
    ASSERT_EQ(&a(i), a.data() + i);
  }
}

}  // namespace Test
