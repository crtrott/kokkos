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
#include <concepts>
namespace Test {

template <class T, class... Args>
struct Base1View : public Kokkos::View<T, Args...> {
 private:
  using base_t = Kokkos::View<T, Args...>;

 public:
  using base_t::base_t;
  template <class... Idx>
  KOKKOS_FUNCTION typename base_t::reference operator()(
      const Idx... idx) const {
    static_assert(sizeof...(Idx) == base_t::rank());
    return base_t::operator()((idx - 1)...);
  }
};

template <class T, class... Args>
KOKKOS_FUNCTION auto Base1View_from_View(Kokkos::View<T, Args...>) {
  return Base1View<T, Args...>();
}

KOKKOS_INLINE_FUNCTION
auto base_1_slice(std::integral auto i) { return i - 1; }

template <class T1, class T2>
KOKKOS_FUNCTION auto base_1_slice(Kokkos::pair<T1, T2> p) {
  return Kokkos::pair<T1, T2>(p.first - 1, p.second - 1);
}

KOKKOS_INLINE_FUNCTION
auto base_1_slice(Kokkos::ALL_t) { return Kokkos::full_extent; }

template <class T, class... Args, class... Slices>
KOKKOS_FUNCTION auto subview(const Base1View<T, Args...>& org,
                             Slices... slices) {
  using return_type = decltype(Base1View_from_View(
      Kokkos::subview(Kokkos::View<T, Args...>(), slices...)));
  return return_type(
      Kokkos::submdspan(org.to_mdspan(), base_1_slice(slices)...));
}

TEST(defaultdevicetype, development_test) {
  Base1View<double**> a("A", 5, 7);
  ASSERT_EQ((&a(1, 1)), (a.data()));
  for (int i = 1; i <= a.extent_int(0); i++) a(i, 0) = i;
  auto s = subview(a, Kokkos::pair{3, 6}, 4);
  ASSERT_EQ((&a(3, 4)), (s.data()));
  ASSERT_EQ(s.extent(0), 3);
}

}  // namespace Test
