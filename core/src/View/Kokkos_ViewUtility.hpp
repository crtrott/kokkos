
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

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#include <Kokkos_Macros.hpp>
static_assert(false,
              "Including non-public Kokkos header files is not allowed.");
#endif
#ifndef KOKKOS_VIEW_UTILITY_HPP
#define KOKKOS_VIEW_UTILITY_HPP

#include <impl/Kokkos_ViewCtor.hpp>

namespace Kokkos {
/** \brief  Create View allocation parameter bundle from argument list.
 *
 *  Valid argument list members are:
 *    1) label as a "string" or std::string
 *    2) memory space instance of the View::memory_space type
 *    3) execution space instance compatible with the View::memory_space
 *    4) Kokkos::WithoutInitializing to bypass initialization
 *    4) Kokkos::AllowPadding to allow allocation to pad dimensions for memory
 * alignment
 */
template <class... Args>
inline Impl::ViewCtorProp<typename Impl::ViewCtorProp<void, Args>::type...>
view_alloc(Args const&... args) {
  using return_type =
      Impl::ViewCtorProp<typename Impl::ViewCtorProp<void, Args>::type...>;

  static_assert(!return_type::has_pointer,
                "Cannot give pointer-to-memory for view allocation");

  return return_type(args...);
}

template <class... Args>
KOKKOS_INLINE_FUNCTION
    Impl::ViewCtorProp<typename Impl::ViewCtorProp<void, Args>::type...>
    view_wrap(Args const&... args) {
  using return_type =
      Impl::ViewCtorProp<typename Impl::ViewCtorProp<void, Args>::type...>;

  static_assert(!return_type::has_memory_space &&
                    !return_type::has_execution_space &&
                    !return_type::has_label && return_type::has_pointer,
                "Must only give pointer-to-memory for view wrapping");

  return return_type(args...);
}

} /* namespace Kokkos */

#endif  // KOKKOS_VIEW_UTILITY_HPP
