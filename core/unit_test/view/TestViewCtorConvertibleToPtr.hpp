// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
import kokkos.core_impl;
#else
#include <Kokkos_Core.hpp>
#endif

namespace {

struct ArrayLike {
  double data[3];
  KOKKOS_FUNCTION
  operator double*() { return data; }
  KOKKOS_FUNCTION
  operator const double*() const { return data; }
};

// clang-format off
// clang-format gets confused, and thinks the lambda closure brace is the namespace
void from_array_like() {
  Kokkos::View<ArrayLike*, TEST_EXECSPACE> a("A", 2);
  int errors = 0;
  Kokkos::parallel_reduce(Kokkos::RangePolicy(TEST_EXECSPACE(), 0, 1), KOKKOS_LAMBDA(int, int& err) {
    {
      Kokkos::View<double*> b(a(0), 3);
      if(b.data() != &a(0).data[0]) err++;
    }
    {
      Kokkos::View<double*, Kokkos::MemoryUnmanaged> b(a(0), 3);
      if (b.data() != &a(0).data[0]) err++;
    }
    {
      Kokkos::View<const double*> b(a(0), 3);
      if (b.data() != &a(0).data[0]) err++;
    }
    {
      Kokkos::View<const double*, Kokkos::MemoryUnmanaged> b(a(0), 3);
      if (b.data() != &a(0).data[0]) err++;
    }
  }, errors);
  ASSERT_EQ(errors, 0);
}
// clang-format on

void from_carray() {
  int errors = 0;
  Kokkos::parallel_reduce(
      Kokkos::RangePolicy(TEST_EXECSPACE(), 0, 1),
      KOKKOS_LAMBDA(int, int& err) {
        double a[3];
        {
          Kokkos::View<double*> b(a, 3);
          if (b.data() != &a[0]) err++;
        }
        {
          Kokkos::View<double*, Kokkos::MemoryUnmanaged> b(a, 3);
          if (b.data() != &a[0]) err++;
        }
        {
          Kokkos::View<const double*> b(a, 3);
          if (b.data() != &a[0]) err++;
        }
        {
          Kokkos::View<const double*, Kokkos::MemoryUnmanaged> b(a, 3);
          if (b.data() != &a[0]) err++;
        }
      },
      errors);
  ASSERT_EQ(errors, 0);
}

struct CustomPtr {
  double* ptr;
  KOKKOS_INLINE_FUNCTION
  operator double*() const { return ptr; }
};

void from_custom_ptr() {
  int errors = 0;
  Kokkos::parallel_reduce(
      Kokkos::RangePolicy(TEST_EXECSPACE(), 0, 1),
      KOKKOS_LAMBDA(int, int& err) {
        double data[3];
        CustomPtr a{&data[0]};
        {
          Kokkos::View<double*> b(a, 3);
          if (b.data() != &a[0]) err++;
        }
        {
          Kokkos::View<double*, Kokkos::MemoryUnmanaged> b(a, 3);
          if (b.data() != &a[0]) err++;
        }
        {
          Kokkos::View<const double*> b(a, 3);
          if (b.data() != &a[0]) err++;
        }
        {
          Kokkos::View<const double*, Kokkos::MemoryUnmanaged> b(a, 3);
          if (b.data() != &a[0]) err++;
        }
      },
      errors);
  ASSERT_EQ(errors, 0);
}

void from_ptr() {
  int errors = 0;
  Kokkos::parallel_reduce(
      Kokkos::RangePolicy(TEST_EXECSPACE(), 0, 1),
      KOKKOS_LAMBDA(int, int& err) {
        double a[3];
        {
          Kokkos::View<double*> b(&a[0], 3);
          if (b.data() != &a[0]) err++;
        }
        {
          Kokkos::View<double*, Kokkos::MemoryUnmanaged> b(&a[0], 3);
          if (b.data() != &a[0]) err++;
        }
        {
          Kokkos::View<const double*> b(&a[0], 3);
          if (b.data() != &a[0]) err++;
        }
        {
          Kokkos::View<const double*, Kokkos::MemoryUnmanaged> b(&a[0], 3);
          if (b.data() != &a[0]) err++;
        }
      },
      errors);
  ASSERT_EQ(errors, 0);
}

TEST(TEST_CATEGORY, view_ctor_ptr_convertible) {
  from_array_like();
  from_carray();
  from_custom_ptr();
  from_ptr();
}

}  // namespace
