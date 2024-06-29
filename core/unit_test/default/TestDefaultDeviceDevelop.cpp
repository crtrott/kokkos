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

  Kokkos::View<float*, Kokkos::LayoutLeft> a("A", 100);
  Kokkos::View<float[100], Kokkos::LayoutLeft> b;

  printf("Assignable: %i static: %i dyn: %i\n", Kokkos::is_assignable(b,a)?1:0, (int)b.static_extent(0), (int)a.extent(0));
  using DstTraits = typename decltype(b)::traits;
  using SrcTraits = typename decltype(a)::traits;
  using mapping_type =
      Kokkos::Impl::ViewMapping<DstTraits, SrcTraits,
                                typename DstTraits::specialize>;
  printf("Map: %i %i\n",mapping_type::is_assignable, (int) DstTraits::dimension::rank_dynamic);
  for(int r=0; r<8; r++) {
    printf("%i %i\n",(int)b.static_extent(r),(int)a.extent(r));
  }
}

}  // namespace Test
