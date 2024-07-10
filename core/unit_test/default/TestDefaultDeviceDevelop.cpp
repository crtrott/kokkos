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
  using acc_t = Kokkos::Impl::SpaceAwareAccessor<
                  typename Kokkos::DefaultExecutionSpace::memory_space,
                  Kokkos::default_accessor<int>>;
  Kokkos::BasicView<int, Kokkos::dextents<int, 1>, Kokkos::layout_right, acc_t> a("A", 5);
  Kokkos::View<int*> b("B", 5);
  auto prop = Kokkos::view_alloc("C");
  Kokkos::View<float*, Kokkos::LayoutRight> c(prop, Kokkos::LayoutRight(5));
}
}  // namespace Test
