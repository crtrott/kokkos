
/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include <TestDefaultDeviceType_Category.hpp>

namespace Test {

TEST(defaultdevicetype, development_test) {
  // Instantiate default constructor
  auto v = Kokkos::View<int*>{};
  // Instantiate with label and size
  auto v2 = Kokkos::View<int*>{"hello", 42};
  // Instantiate with raw pointer
  int data[]   = {1, 2, 3, 4};
  auto vstatic = Kokkos::View<int[4]>{Kokkos::view_wrap(data)};
  // Instantiate with view_alloc
  auto v4 = Kokkos::View<int*>{
      Kokkos::view_alloc("hello", Kokkos::WithoutInitializing), 42};
  // Instantiate with view_alloc allow padding (TODO make this actually work)
  // TODO this should fail if the layout doesn't support padding
  auto v5 =
      Kokkos::View<int*>{Kokkos::view_alloc("hello", Kokkos::AllowPadding), 42};

  // Conversion to compatible accessor
  using view_atomic =
      Kokkos::View<int*, Kokkos::MemoryTraits<Kokkos::Atomic>>::basic_view_type;
  auto vatomic = view_atomic{v2};

  // Conversion to dynamic
  auto vdyn = Kokkos::View<int*>{vstatic};

  auto vstatic2 = Kokkos::View<int*[4]>{"hello2", 10};
  auto vstatic3 = Kokkos::View<int[3][4]>{Kokkos::view_wrap(data)};
  auto vdyn2 = Kokkos::View<int**>{vstatic2};
  auto vdyn3 = Kokkos::View<int**>{vstatic3};
  // TODO converting assignment operator
  // vstatic2 = vstatic3;

  // Instantiate default 2d ctor
  auto v2d = Kokkos::View<int**>{};
  auto v2static1 = Kokkos::View<int*[3]>{};
  auto v2static2 = Kokkos::View<int[7][3]>{};
  auto v2staticr1 = Kokkos::View<int*[3], Kokkos::LayoutRight>{};
}

}  // namespace Test
