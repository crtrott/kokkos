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

#ifndef KOKKOS_KOKKOS_MDSPANLAYOUT_HPP
#define KOKKOS_KOKKOS_MDSPANLAYOUT_HPP

#include <Kokkos_Macros.hpp>

#include <View/Kokkos_ExtractExtents.hpp>  // RemoveFirstExtent
#include <impl/Kokkos_EBO.hpp>
#include <impl/Kokkos_Error.hpp>  // KOKKOS_EXPECTS

#include <Kokkos_Layout.hpp>    // LayoutLeft, LayoutRight
#include <Kokkos_Concepts.hpp>  // is_array_layout

#include <experimental/mdspan>
#include <utility>  // std::declval

namespace Kokkos {
namespace Impl {

//==============================================================================
// <editor-fold desc="Call the with_padding_for_type customization point"> {{{1

// All of this is basically just boilerplate for invoking a customization point
// called `with_padding_for_type` on layouts that support runtime addition
// of padding (which does not include the standard layouts, for instance, but
// includes all Kokkos layouts before the refactor).
// This can be simplified a bit once we have detection idiom in Kokkos

template <class Layout, class ValueType, class SFINAESafeDetectionSlot = void>
struct ApplyPaddingToLayout {
  static constexpr auto layout_supports_padding = false;
};

template <class Layout, class ValueType>
struct ApplyPaddingToLayout<
    Layout, ValueType,
    void_t<decltype(std::declval<Layout&>()
                        .template with_padding_for_type<ValueType>())>> {
  static constexpr auto layout_supports_padding = true;
  static constexpr void apply(Layout& layout) {
    layout = layout.template with_padding_for_type<ValueType>();
  }
};

template <bool AllowPadding, class ValueType>
struct HandleLayoutPadding;

template <class ValueType>
struct HandleLayoutPadding</* AllowPadding = */ true, ValueType> {
  // TODO @mdspan informative error message with incomplete type instantiations
  //      to show the typenames of the broken layout in the compiler error
  //      message
  template <class Layout>
  static void apply_padding(Layout& layout) {
    using apply_layout_helper_t = ApplyPaddingToLayout<Layout, ValueType>;
    static_assert(
        apply_layout_helper_t::layout_supports_padding,
        "Requested AllowPadding for a layout that doesn't have the "
        " with_padding_for_type() customization point method implemented");
    apply_layout_helper_t::apply(layout);
  }
};

template <class ValueType>
struct HandleLayoutPadding</* AllowPadding = */ false, ValueType> {
  // We don't have to be able to apply padding if the constructor that asks
  // for it is never instantiated
  template <class Layout>
  static void apply_padding(Layout&) {
    // TODO @mdspan should we invoke the copy constructor of layout here anyway
    //              so that the number of copy constructions of the layout
    //              customization point doesn't depend on the padding? For
    //              trivial cases, the optimizer will remove it anyway
  }
};

// TODO @mdspan implement padding in the common layouts

// </editor-fold> end Call the with_padding_for_type customization point }}}1
//==============================================================================

// template <class Traits, class Layout>
// struct AdaptMDSpanLayoutForKokkosLayout;
//
////==============================================================================
//// <editor-fold desc="Convert Kokkos layouts to leading/trailing extents">
///{{{1
//
//// TODO @mdspan make sure a non-recursive version of this isn't faster
// template <class FinalExtentsType, std::size_t FirstIdx,
//          class = FinalExtentsType>
// struct _extract_extents_from_offset_in_layout;
//
// template <class FinalExtentsType, std::ptrdiff_t Extent,
//          std::ptrdiff_t... Extents, std::size_t Idx>
// struct _extract_extents_from_offset_in_layout<
//    FinalExtentsType, Idx, std::experimental::extents<Extent, Extents...> > {
//  template <class Layout, class... DynExtents>
//  KOKKOS_INLINE_FUNCTION static constexpr auto extract(Layout const& l,
//                                                       DynExtents... exts) {
//    // Layout objects take all of the extents, but we only need to pass the
//    // dynamic ones on to the extents constructor
//    using next_t = _extract_extents_from_offset_in_layout<
//        FinalExtentsType, Idx + 1, std::experimental::extents<Extents...> >;
//    return next_t::extract(l, exts...);
//  }
//};
//
// template <class FinalExtentsType, std::ptrdiff_t... Extents, std::size_t Idx>
// struct _extract_extents_from_offset_in_layout<
//    FinalExtentsType, Idx,
//    std::experimental::extents<std::experimental::dynamic_extent, Extents...>
//    >
//    {
//  template <class Layout, class... DynExtents>
//  KOKKOS_INLINE_FUNCTION static constexpr auto extract(Layout const& l,
//                                                       DynExtents... exts) {
//    using next_t = _extract_extents_from_offset_in_layout<
//        FinalExtentsType, Idx + 1, std::experimental::extents<Extents...> >;
//    return next_t::extract(l, exts..., l.dimension[Idx]);
//  }
//};
//
// template <class FinalExtentsType, std::size_t Idx>
// struct _extract_extents_from_offset_in_layout<FinalExtentsType, Idx,
//                                              std::experimental::extents<> > {
//  template <class Layout, class... DynExtents>
//  KOKKOS_INLINE_FUNCTION static constexpr auto extract(Layout const&,
//                                                       DynExtents... exts) {
//    return FinalExtentsType{exts...};
//  }
//};
//
//// </editor-fold> end Convert Kokkos layouts to leading/trailing extents }}}1
////==============================================================================
//
//// typename FirstExtentOnly<typename Traits::mdspan_extents_type>::type;
//
//// TODO @mdspan avoid instantiating this for every View type
//
// template <class Traits, class LeadingLayout, class TrailingLayout>
// struct SplitLayoutImpl {
// private:
//  using leading_extents_t =
//      decltype(std::declval<LeadingLayout const&>().extents());
//  using trailing_extents_t =
//    decltype(std::declval<TrailingLayout const&>().extents());
//  using leading_layout_t = LeadingLayout;
//  using trailing_layout_t = TrailingLayout;
//  using extents_type = typename Traits::mdspan_extents_type;
//
//  // TODO @mdspan no unique address (EBO) optimization
//  typename Traits::size_type m_stride = {1};
//
//  leading_layout_t m_leading_layout;
//  trailing_layout_t m_trailing_layout;
//
// public:
//  template <class KokkosLayout>
//  KOKKOS_FUNCTION
//  constexpr explicit SplitLayoutImpl(KokkosLayout const& ll)
//      : m_stride(1),
//        m_leading_layout(
//            _extract_extents_from_offset_in_layout<leading_extents_t,
//                                                   0>::extract(ll)),
//        m_trailing_layout(
//            _extract_extents_from_offset_in_layout<trailing_extents_t,
//                                                   1>::extract(ll)) {}
//
//  template <class IntegralType, class... IntegralTypes>
//  KOKKOS_FORCEINLINE_FUNCTION constexpr
//      typename Traits::mdspan_extents_type::index_type
//      map_with_stride_like_layout_left(IntegralType first_idx,
//                                       IntegralTypes... last_idxs) const {
//    return m_leading_layout(first_idx) +
//           m_stride * m_trailing_layout(last_idxs...);
//  }
//
//  template <class IntegralType, class... IntegralTypes>
//  KOKKOS_FORCEINLINE_FUNCTION constexpr
//      typename Traits::mdspan_extents_type::index_type
//      map_with_stride_like_layout_right(IntegralType last_idx,
//                                        IntegralTypes... first_idxs) const {
//    return m_leading_layout(first_idxs...) * m_stride +
//           m_trailing_layout(last_idx);
//  }
//};
//
// template <class Traits, class T>
// struct AdaptKokkosLayoutToMDSpanLayout;
//
// template <class Traits>
// struct AdaptKokkosLayoutToMDSpanLayout<Traits, Kokkos::LayoutLeft>
//  : SplitLayoutImpl<Traits
//
//{
//
//};

namespace {

// We'll use odd offsets from min to be layout left dimension strides and
// even offsets from min to be layout right dimension strides
constexpr std::ptrdiff_t stride_as_fixed_layout_tag_offset =
    std::numeric_limits<std::ptrdiff_t>::min();
// constexpr std::ptrdiff_t stride_as_layout_left_tag =
//    stride_as_fixed_layout_tag_offset + 1;
// constexpr std::ptrdiff_t stride_as_layout_right_tag =
//    stride_as_fixed_layout_tag_offset + 2;

template <std::ptrdiff_t StaticStride>
constexpr bool is_static_dimension_stride =
    (StaticStride < 0) && StaticStride != std::experimental::dynamic_extent &&
    ((StaticStride - stride_as_fixed_layout_tag_offset) % 2 == 0);
template <std::ptrdiff_t StaticStride>
constexpr bool extract_static_dimension_stride =
    StaticStride - stride_as_fixed_layout_tag_offset;
template <std::ptrdiff_t DimensionStride = 1>
constexpr std::ptrdiff_t fixed_layout_dimension_stride =
    stride_as_fixed_layout_tag_offset + DimensionStride;

// template <std::ptrdiff_t StaticStride>
// constexpr bool is_layout_left_with_dimension_stride =
//    (StaticStride < 0) && StaticStride != std::experimental::dynamic_extent &&
//    ((StaticStride - stride_as_layout_left_tag) % 2 == 0);
// template <std::ptrdiff_t StaticStride>
// constexpr bool extract_static_dimension_stride_for_layout_left =
//    (StaticStride - stride_as_layout_left_tag) / 2;
//
// template <std::ptrdiff_t StaticStride>
// constexpr bool is_layout_right_with_dimension_stride =
//    (StaticStride < 0) && StaticStride != std::experimental::dynamic_extent &&
//    ((StaticStride - stride_as_layout_right_tag) % 2 == 0);
// template <std::ptrdiff_t StaticStride>
// constexpr bool extract_static_dimension_stride_for_layout_right =
//    (StaticStride - stride_as_layout_right_tag) / 2;

}  // end anonymous namespace

namespace {

// There are still too many compiler bugs to use a constexpr function here :-(
// template <std::ptrdiff_t DimensionStride = 1>
// struct stride_as_layout_left_with_dimension_stride_t {
//  static_assert(DimensionStride > 0, "");
//  // Make an odd offset from std::numeric_limits<std::ptrdiff_t>::min()
//  static constexpr std::ptrdiff_t value =
//      stride_as_layout_left_tag + 2 * DimensionStride;
//};
//
// template <std::ptrdiff_t DimensionStride = 1>
// constexpr std::ptrdiff_t stride_as_layout_left_with_dimension_stride =
//    stride_as_layout_left_with_dimension_stride_t<DimensionStride>::value;
//
// template <std::ptrdiff_t DimensionStride = 1>
// struct stride_as_layout_right_with_dimension_stride_t {
//  static_assert(DimensionStride > 0, "");
//  // Make an odd offset from std::numeric_limits<std::ptrdiff_t>::min()
//  static constexpr std::ptrdiff_t value =
//      stride_as_layout_right_tag + 2 * DimensionStride;
//};
//
// template <std::ptrdiff_t DimensionStride = 1>
// constexpr std::ptrdiff_t stride_as_layout_right_with_dimension_stride =
//    stride_as_layout_right_with_dimension_stride_t<DimensionStride>::value;

}  // end anonymous namespace

//==============================================================================

template <std::ptrdiff_t StaticStride, std::ptrdiff_t Extent,
          std::ptrdiff_t Idx, class StaticStrides, class Extents, class IdxPack,
          bool LayoutLeftDerived, class Enable = void>
struct stride_storage_impl;

template <bool LayoutLeftBased, class StaticStrides, class Extents,
          class IdxPack = std::make_index_sequence<Extents::rank()>>
struct layout_stride_general_impl;

//==============================================================================

// TODO @mdspan move this to utilities
template <std::size_t Idx, class IPack>
struct _cheap_index_pack_at;
template <std::size_t Idx, class IPack, class Idxs>
struct _cheap_index_pack_at_impl;

template <std::size_t Idx, class T, T... Vals, std::size_t... Idxs>
struct _cheap_index_pack_at_impl<Idx, std::integer_sequence<T, Vals...>,
                                 std::integer_sequence<std::size_t, Idxs...>> {
  static constexpr auto value =
      _MDSPAN_FOLD_PLUS_RIGHT((Idx == Idxs) ? Vals : 0, /* + ... + */ 0);
};

template <std::size_t Idx, class T, T... Vals>
struct _cheap_index_pack_at<Idx, std::integer_sequence<T, Vals...>>
    : _cheap_index_pack_at_impl<Idx, std::integer_sequence<T, Vals...>,
                                std::make_index_sequence<sizeof...(Vals)>> {};

template <std::size_t Idx, std::ptrdiff_t... Vals>
struct _cheap_index_pack_at<Idx, std::experimental::extents<Vals...>>
    : _cheap_index_pack_at_impl<Idx,
                                std::integer_sequence<std::ptrdiff_t, Vals...>,
                                std::make_index_sequence<sizeof...(Vals)>> {};

//==============================================================================

struct no_more_strides_tag {
  KOKKOS_FORCEINLINE_FUNCTION constexpr std::ptrdiff_t _get_element_stride()
      const noexcept {
    return 1;
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr bool _is_contiguous() const noexcept {
    return true;
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr bool _is_always_contiguous()
      const noexcept {
    return true;
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr std::ptrdiff_t _static_contiguous_size()
      const noexcept {
    return 1;
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr std::ptrdiff_t _static_size()
      const noexcept {
    return 1;
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr std::ptrdiff_t _extent()
      const noexcept {
    return 1;
  }
};

//==============================================================================

template <std::size_t Idx, std::ptrdiff_t Extent, class Strides, class Exts,
          class Idxs, bool LayoutLeftBased>
struct stride_storage_common;
template <std::size_t Idx, std::ptrdiff_t Extent,
          std::ptrdiff_t... StaticStrides, std::ptrdiff_t... Exts,
          std::size_t... Idxs, bool LayoutLeftBased>
struct stride_storage_common<
    Idx, Extent, std::integer_sequence<std::ptrdiff_t, StaticStrides...>,
    std::experimental::extents<Exts...>,
    std::integer_sequence<std::size_t, Idxs...>, LayoutLeftBased> {
 protected:
  using extents_t = std::experimental::extents<Exts...>;
  using strides_t = std::integer_sequence<std::ptrdiff_t, StaticStrides...>;
  using idxs_t    = std::integer_sequence<std::size_t, Idxs...>;
  using derived_t =
      layout_stride_general_impl<LayoutLeftBased, extents_t, strides_t, idxs_t>;
  static constexpr std::size_t rank = sizeof...(Exts);

  // Lazy instantiation here shoud save on compilation time significantly
  struct lazy_instantiate_left {
    template <class = void>
    struct instantiate {
      using type =
          stride_storage_impl<_cheap_index_pack_at<Idx - 1, strides_t>::value,
                              _cheap_index_pack_at<Idx - 1, extents_t>::value,
                              Idx - 1, extents_t, strides_t, idxs_t,
                              LayoutLeftBased>;
    };
  };
  struct lazy_instantiate_right {
    template <class = void>
    struct instantiate {
      using type =
          stride_storage_impl<_cheap_index_pack_at<Idx + 1, strides_t>::value,
                              _cheap_index_pack_at<Idx + 1, extents_t>::value,
                              Idx + 1, extents_t, strides_t, idxs_t,
                              LayoutLeftBased>;
    };
  };
  struct lazy_instantiate_no_strides {
    template <class = void>
    struct instantiate {
      using type = no_more_strides_tag;
    };
  };

  template <class _lazy_instantiate_tag = void>
  KOKKOS_FORCEINLINE_FUNCTION constexpr auto const& self_as_next_left_storage()
      const noexcept {
    using next_stride_storage_left_t =
        typename std::conditional_t<Idx == 0, lazy_instantiate_no_strides,
                                    lazy_instantiate_left>::
            template instantiate<_lazy_instantiate_tag>::type;
    return static_cast<next_stride_storage_left_t const&>(
        static_cast<derived_t const&>(*this));
  }

  template <class _lazy_instantiate_tag = void>
  KOKKOS_FORCEINLINE_FUNCTION constexpr auto const& self_as_next_right_storage()
      const noexcept {
    using next_stride_storage_right_t = typename std::conditional_t<
        Idx == rank - 1, lazy_instantiate_no_strides, lazy_instantiate_right>::
        template instantiate<_lazy_instantiate_tag>::type;
    return static_cast<next_stride_storage_right_t const&>(
        static_cast<derived_t const&>(*this));
  }

  template <class _lazy_instantiate_tag = void>
  KOKKOS_FORCEINLINE_FUNCTION constexpr auto const& self_as_next_storage_impl(
      std::true_type /* LayoutLeftBased */) const noexcept {
    return self_as_next_left_storage();
  }
  template <class _lazy_instantiate_tag = void>
  KOKKOS_FORCEINLINE_FUNCTION constexpr auto const& self_as_next_storage_impl(
      std::false_type /* LayoutRightBased */) const noexcept {
    return self_as_next_right_storage();
  }

  template <class _lazy_instantiate_tag = void>
  KOKKOS_FORCEINLINE_FUNCTION constexpr auto const& self_as_next_storage()
      const noexcept {
    return self_as_next_storage_impl(
        std::integral_constant<bool, LayoutLeftBased>{});
  }

  KOKKOS_FORCEINLINE_FUNCTION constexpr std::ptrdiff_t _static_contiguous_size()
      const {
    auto static_extent =
        (Extent == std::experimental::dynamic_extent) ? 0 : Extent;
    return static_extent * self_as_next_storage()._static_contiguous_size();
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr std::ptrdiff_t _contiguous_size()
      const {
    return static_cast<derived_t const&>(*this).extents().extent(Idx) *
           self_as_next_storage()._contiguous_size();
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr std::ptrdiff_t _extent() const {
#ifndef KOKKOS_IMPL_USE_STANDARD_MDSPAN
    return static_cast<derived_t const&>(*this)
        .extents()
        .template __extent<Idx>();
#else
    return static_cast<derived_t const&>(*this).extents().extent(Idx);
#endif
  }
};

//------------------------------------------------------------------------------

// Integer compile-time constant stride case
template <std::ptrdiff_t StaticStride, std::ptrdiff_t Extent,
          std::ptrdiff_t Idx, class Strides, class Extents, class Idxs,
          bool LayoutLeftBased>
struct stride_storage_impl<
    StaticStride, Extent, Idx, Strides, Extents, Idxs, LayoutLeftBased,
    std::enable_if_t<(StaticStride > 0) &&
                     StaticStride != std::experimental::dynamic_extent>>
    : stride_storage_common<Idx, Extent, Strides, Extents, Idxs,
                            LayoutLeftBased> {
 private:
  using base_t = stride_storage_common<Idx, Extent, Strides, Extents, Idxs,
                                       LayoutLeftBased>;
  static constexpr std::ptrdiff_t m_stride = StaticStride;

 public:
  KOKKOS_FORCEINLINE_FUNCTION
  constexpr std::ptrdiff_t _get_element_stride() const noexcept {
    return m_stride;
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr bool _is_always_contiguous() const {
    auto const& next = this->base_t::self_as_next_storage();
    return (m_stride == next._static_contiguous_size()) &&
           next._is_always_contiguous();
  }

  KOKKOS_FORCEINLINE_FUNCTION constexpr bool _is_contiguous() const {
    auto const& next = this->base_t::self_as_next_storage();
    return (m_stride == next._contiguous_size()) && next._is_contiguous();
  }
};

//------------------------------------------------------------------------------
// Fixed-layout-like with constant stride along a dimension
template <std::ptrdiff_t StaticStride, std::ptrdiff_t Extent,
          std::ptrdiff_t Idx, class Strides, class Extents, class Idxs,
          bool LayoutLeftBased>
struct stride_storage_impl<
    StaticStride, Extent, Idx, Strides, Extents, Idxs, LayoutLeftBased,
    std::enable_if_t<is_static_dimension_stride<StaticStride>>>
    : stride_storage_common<Idx, Extent, Strides, Extents, Idxs,
                            LayoutLeftBased> {
 private:
  static constexpr std::ptrdiff_t m_dimension_stride =
      extract_static_dimension_stride<StaticStride>;
  using base_t = stride_storage_common<Idx, Extent, Strides, Extents, Idxs,
                                       LayoutLeftBased>;

 public:
  // The stride between two consecutive elements in the dimension (i.e., _not_
  // the dimension stride that's independent of the strides around it). This
  // is what we usually call stride in, e.g., mdspan
  KOKKOS_FORCEINLINE_FUNCTION constexpr std::ptrdiff_t _get_element_stride()
      const noexcept {
    return m_dimension_stride * this->base_t::self_as_next_storage()._extent() *
           this->base_t::self_as_next_storage()._get_element_stride();
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr bool _is_always_contiguous() const {
    return m_dimension_stride == 1 &&
           this->base_t::self_as_next_storage()._is_always_contiguous();
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr bool _is_contiguous() const {
    return (m_dimension_stride == 1) &&
           this->base_t::self_as_next_storage()._is_contiguous();
  }
};

// Dynamic stride
template <std::ptrdiff_t Extent, std::ptrdiff_t Idx, class StaticStrides,
          class Extents, class Idxs, bool LayoutLeftBased>
struct stride_storage_impl<std::experimental::dynamic_extent, Extent, Idx,
                           StaticStrides, Extents, Idxs, LayoutLeftBased>
    : stride_storage_common<Idx, Extent, StaticStrides, Extents, Idxs,
                            LayoutLeftBased> {
 private:
  std::ptrdiff_t m_stride = {0};

 protected:
  // The stride between two consecutive elements in the dimension (i.e., _not_
  // the dimension stride that's independent of the strides around it). This
  // is what we usually call stride in, e.g., mdspan
  KOKKOS_FORCEINLINE_FUNCTION
  constexpr std::ptrdiff_t _get_element_stride() const noexcept {
    return m_stride;
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr bool _is_always_contiguous() const {
    return false;
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr bool _is_contiguous() const {
    auto const& next = this->base_t::self_as_next_storage();
    return (m_stride == next._contiguous_size()) && next._is_contiguous();
  }
  KOKKOS_FUNCTION
  KOKKOS_CONSTEXPR_14 inline void _set_stride(
      std::ptrdiff_t new_stride) noexcept {
    m_stride = new_stride;
  }
};

//==============================================================================

//==============================================================================
// <editor-fold desc="layout_stride_general_impl"> {{{1

template <bool LayoutLeftBased, std::ptrdiff_t... StaticStrides,
          std::ptrdiff_t... Extents, std::ptrdiff_t... Idxs>
struct KOKKOS_IMPL_ENFORCE_EMPTY_BASE_OPTIMIZATION layout_stride_general_impl<
    LayoutLeftBased, std::integer_sequence<std::ptrdiff_t, StaticStrides...>,
    std::experimental::extents<Extents...>,
    std::integer_sequence<std::size_t, Idxs...>>
    : NoUniqueAddressMemberEmulation<std::experimental::extents<Extents...>>,
      stride_storage_impl<
          StaticStrides, Extents, Idxs,
          std::integer_sequence<std::ptrdiff_t, StaticStrides...>,
          std::experimental::extents<Extents...>,
          std::integer_sequence<std::size_t, Idxs...>, LayoutLeftBased>...,
      no_more_strides_tag {
 private:
  using extents_storage_base_t =
      NoUniqueAddressMemberEmulation<std::experimental::extents<Extents...>>;
  using strides_t = std::integer_sequence<std::ptrdiff_t, StaticStrides...>;
  using extents_t = std::experimental::extents<Extents...>;
  using idxs_t    = std::integer_sequence<std::size_t, Idxs...>;
  template <std::size_t Idx>
  using stride_storage_base_for_idx =
      stride_storage_impl<_cheap_index_pack_at<Idx, strides_t>::value,
                          _cheap_index_pack_at<Idx, extents_t>::value, Idx,
                          strides_t, extents_t, idxs_t, LayoutLeftBased>;
  // The "first" stride to start recursion on is different for LayoutLeft and
  // LayoutRight
  // Start from the right if we're layout left, and start from the left if we're
  // layout right.
  using first_stride_storage_t =
      std::conditional_t<LayoutLeftBased,
                         stride_storage_base_for_idx<sizeof...(Extents) - 1>,
                         stride_storage_base_for_idx<0>>;

  // If we're getting here, something has gone wrong internally
  static_assert(sizeof...(Idxs) >= 2, "");

 public:
  KOKKOS_FORCEINLINE_FUNCTION constexpr auto extents() const noexcept {
    return this->extents_storage_base_t::no_unique_address_data_member();
  }
  template <class... IntegralTypes>
  KOKKOS_FORCEINLINE_FUNCTION constexpr auto operator()(
      IntegralTypes... idxs) const noexcept {
    return _MDSPAN_FOLD_PLUS_RIGHT(
        (this->stride_storage_base_for_idx<Idxs>::_get_element_stride() * idxs),
        /* + ... + */ 0);
  }
  KOKKOS_FUNCTION constexpr bool is_contiguous() const noexcept {
    return this->first_stride_storage_t::_is_contiguous();
  }
  KOKKOS_FUNCTION constexpr bool is_always_contiguous() const noexcept {
    return this->first_stride_storage_t::_is_always_contiguous();
  }
  KOKKOS_FUNCTION constexpr auto required_span_size() const noexcept {
    return _MDSPAN_FOLD_PLUS_RIGHT(
        (this->stride_storage_base_for_idx<Idxs>::_get_element_stride() *
         this->extents().template __extent<Idxs>()),
        /* + ... + */ 0);
  }
  KOKKOS_FUNCTION constexpr auto size() const noexcept {
    return _MDSPAN_FOLD_TIMES_RIGHT(this->extents().template __extent<Idxs>(),
                                    /* + ... + */ 0);
  }

  //----------------------------------------------------------------------------
  // <editor-fold desc="Ctor, destructor, and assignment"> {{{2

  layout_stride_general_impl() = default;

  layout_stride_general_impl(layout_stride_general_impl const&) = default;

  layout_stride_general_impl(layout_stride_general_impl&&) = default;

  layout_stride_general_impl& operator=(layout_stride_general_impl const&) =
      default;

  layout_stride_general_impl& operator=(layout_stride_general_impl&&) = default;

  ~layout_stride_general_impl() = default;

  // TODO @mdspan compatible layout conversion
  // TODO default values for strides

  KOKKOS_INLINE_FUNCTION
  constexpr explicit layout_stride_general_impl(extents_t const& exts)
      : extents_storage_base_t(exts), stride_storage_base_for_idx<Idxs>()... {}

  // </editor-fold> end Ctor, destructor, and assignment }}}2
  //----------------------------------------------------------------------------
};

// </editor-fold> end layout_stride_general_impl }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="Contiguous 1D layout"> {{{1

// All contiguous 1D layouts work the same way, so we can reduce compile-time
// by having exactly one of these for most of the requirements and one per
// extents type for the few things that depend on it
struct contiguous_mapping_1d_common {
  KOKKOS_FUNCTION constexpr bool is_unique() const noexcept { return true; }
  KOKKOS_FUNCTION constexpr bool is_always_unique() const noexcept {
    return true;
  }
  KOKKOS_FUNCTION constexpr bool is_strided() const noexcept { return true; }
  KOKKOS_FUNCTION constexpr bool is_always_strided() const noexcept {
    return true;
  }
  KOKKOS_FUNCTION constexpr bool is_contiguous() const noexcept { return true; }
  KOKKOS_FUNCTION constexpr bool is_always_contiguous() const noexcept {
    return true;
  }
  // TODO @mdspan a constexpr macro that goes away when the assertion is made?
  // this needs to be a template to avoid narrowing warnings
  template <class IntegralType>
  KOKKOS_FUNCTION constexpr std::ptrdiff_t stride(
      IntegralType /* r */) const noexcept {
    static_assert(
        std::is_integral<IntegralType>::value,
        "1D layout mapping argument type to stride() needs to be integral");
    // Can't do this while preserving constexpr:
    // KOKKOS_EXPECTS(r == 0);
    return 1;
  }
  // this needs to be a template to avoid narrowing warnings
  template <class IntegralType>
  KOKKOS_FORCEINLINE_FUNCTION constexpr std::ptrdiff_t operator()(
      IntegralType i) const noexcept {
    return std::ptrdiff_t(i);
  }
};

template <class Extents>
struct contiguous_mapping_1d;

// static extent case
template <std::ptrdiff_t Extent>
struct contiguous_mapping_1d<std::experimental::extents<Extent>>
    : contiguous_mapping_1d_common {
 private:
  using base_t = contiguous_mapping_1d_common;
  using base_t::base_t;
  using extents_t = std::experimental::extents<Extent>;

 public:
  // The analog of this isn't explicit in P0009, so I didn't do that here
  KOKKOS_INLINE_FUNCTION
  constexpr contiguous_mapping_1d(extents_t const&) noexcept {}
  // TODO @mdspan a constexpr macro that goes away when the assertion active?
  KOKKOS_INLINE_FUNCTION
  contiguous_mapping_1d(
      std::experimental::extents<std::experimental::dynamic_extent> const&
          arg_ext) noexcept {
    KOKKOS_EXPECTS(arg_ext.extent(0) == Extent);
    (void)arg_ext;  // handle unused argument warnings
  }
  // We want to force inline this because it's on the path of the view element
  // access operator evaluation
  KOKKOS_FORCEINLINE_FUNCTION constexpr extents_t extents() const noexcept {
    return extents_t{};
  }
  KOKKOS_FUNCTION constexpr typename extents_t::index_type required_span_size()
      const noexcept {
    return Extent;
  }
};

// static extent case
template <>
struct contiguous_mapping_1d<
    std::experimental::extents<std::experimental::dynamic_extent>>
    : contiguous_mapping_1d_common {
  using base_t = contiguous_mapping_1d_common;
  using base_t::base_t;
  using extents_t =
      std::experimental::extents<std::experimental::dynamic_extent>;
  extents_t m_extents;

 public:
  // The analog of this isn't explicit in P0009, so I didn't do that here
  KOKKOS_INLINE_FUNCTION
  constexpr contiguous_mapping_1d(extents_t const&) noexcept {}
  // TODO @mdspan a constexpr macro that goes away when the assertion active?
  template <std::ptrdiff_t StaticExtent,
            //----------------------------------------
            /* requires
             *   (StaticExtent != dynamic_extent)
             * This should be fine and not conflict with the other constructor,
             * but there are a lot of compiler bugs around this, so SFINAE just
             * to be safe here
             */
            std::enable_if_t<StaticExtent != std::experimental::dynamic_extent,
                             int> = 0
            //----------------------------------------
            >
  KOKKOS_INLINE_FUNCTION contiguous_mapping_1d(
      std::experimental::extents<StaticExtent> const& arg_ext) noexcept
      : m_extents(arg_ext) {}
  // We want to force inline this because it's on the path of the view element
  // access operator evaluation
  KOKKOS_FORCEINLINE_FUNCTION constexpr extents_t extents() const noexcept {
    return m_extents;
  }
  KOKKOS_FUNCTION constexpr typename extents_t::index_type required_span_size()
      const noexcept {
    // NOTE: nonstandard mdspan extension
    return m_extents.template __extent<0>();
  }
};

// </editor-fold> end Contiguous 1D layout }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="Special 0D layout"> {{{1

// All reasonable 0D layouts work exactly the same way, so we can reduce
// compile-time by having exactly one of these
struct mapping_0d {
  KOKKOS_FUNCTION constexpr bool is_unique() const noexcept { return true; }
  KOKKOS_FUNCTION constexpr bool is_always_unique() const noexcept {
    return true;
  }
  KOKKOS_FUNCTION constexpr bool is_strided() const noexcept { return true; }
  KOKKOS_FUNCTION constexpr bool is_always_strided() const noexcept {
    return true;
  }
  KOKKOS_FUNCTION constexpr bool is_contiguous() const noexcept { return true; }
  KOKKOS_FUNCTION constexpr bool is_always_contiguous() const noexcept {
    return true;
  }

  // Note: stride function isn't required because there's no valid way to call
  // it (i.e., all arguments would be invalid

  KOKKOS_FORCEINLINE_FUNCTION constexpr std::ptrdiff_t operator()()
      const noexcept {
    return 0;
  }
};

// </editor-fold> end Special 0D layout }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="Convert legacy Kokkos Layouts to extents objects"> {{{1

template <class Exts>
struct extents_as_integer_sequence;

template <std::ptrdiff_t... Exts>
struct extents_as_integer_sequence<std::experimental::extents<Exts...>>
    : identity<std::integer_sequence<std::ptrdiff_t, Exts...>> {};

// We need to build up a parameter pack that only includes the dynamic extents
// We use an integer_sequence for the extents rather than using the extents
// type directly to avoid instantiating more versions of
// std::experimental::extents than is absolutely necessary, since that's more
// expensive at compile time.
template <class ExtentsToConstruct,
          class ExtsSeq =
              typename extents_as_integer_sequence<ExtentsToConstruct>::type,
          class = std::make_index_sequence<ExtsSeq::size()>>
struct construct_extents_from_kokkos_layout;

// The static extent case
template <class ExtentsToConstruct, std::ptrdiff_t Ext, std::ptrdiff_t... Exts,
          std::size_t Idx, std::size_t... Idxs>
struct construct_extents_from_kokkos_layout<
    ExtentsToConstruct, std::integer_sequence<std::ptrdiff_t, Ext, Exts...>,
    std::integer_sequence<std::size_t, Idx, Idxs...>> {
  using next_t = construct_extents_from_kokkos_layout<
      ExtentsToConstruct, std::integer_sequence<std::ptrdiff_t, Exts...>,
      std::integer_sequence<std::size_t, Idxs...>>;
  template <class Layout, class... CtorArgs>
  KOKKOS_INLINE_FUNCTION static auto apply(Layout const& kokkos_layout,
                                           CtorArgs... values) {
    // Since construction of std::extents only takes dynamic extents as
    // argument but legacy Kokkos layouts take all of the extents, we
    // need to just skip adding the parameter to the constructor arguments pack.
    // We can still assert that the dynamic value given to the legacy Kokkos
    // layout matches the static one specified as part of the extents type.
    KOKKOS_EXPECTS(Ext == kokkos_layout.dimension[Idx]);
    return next_t::apply(kokkos_layout, values...);
  }
};

// The dynamic extent case
template <class ExtentsToConstruct, std::ptrdiff_t... Exts, std::size_t Idx,
          std::size_t... Idxs>
struct construct_extents_from_kokkos_layout<
    ExtentsToConstruct,
    std::integer_sequence<std::ptrdiff_t, std::experimental::dynamic_extent,
                          Exts...>,
    std::integer_sequence<std::size_t, Idx, Idxs...>> {
  using next_t = construct_extents_from_kokkos_layout<
      ExtentsToConstruct, std::integer_sequence<std::ptrdiff_t, Exts...>,
      std::integer_sequence<std::size_t, Idxs...>>;
  template <class Layout, class... CtorArgs>
  KOKKOS_INLINE_FUNCTION static auto apply(Layout const& kokkos_layout,
                                           CtorArgs... values) {
    return next_t::apply(kokkos_layout, values...,
                         kokkos_layout.dimension[Idx]);
  }
};

// The recursive base case
template <class ExtentsToConstruct>
struct construct_extents_from_kokkos_layout<
    ExtentsToConstruct, std::integer_sequence<std::ptrdiff_t>,
    std::integer_sequence<std::size_t>> {
  using next_t = construct_extents_from_kokkos_layout<
      ExtentsToConstruct, std::integer_sequence<std::ptrdiff_t>,
      std::integer_sequence<std::size_t>>;
  template <class Layout, class... CtorArgs>
  KOKKOS_INLINE_FUNCTION static auto apply(Layout const& kokkos_layout,
                                           CtorArgs... values) {
    return ExtentsToConstruct{values...};
  }
};

// </editor-fold> end Convert legacy Kokkos Layouts to extents objects }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="LayoutLeft"> {{{1

//----------------------------------------------------------------------------
// <editor-fold desc="make layout left from extents"> {{{2

// A helper type that gets the layout as a nested type. This is so that
// we don't have to repeat ourselves for derived-class members. Also
// helps reduce the number of template specializations generated

template <class Extents,
          class IdxSeq = std::make_index_sequence<Extents::rank()>>
struct MakeKokkosLayoutLeftFromExtents;

template <class Extents, std::size_t... RestIdxs>
struct MakeKokkosLayoutLeftFromExtents<
    Extents, std::integer_sequence<std::size_t, 0, 1, RestIdxs...>>
    : identity<layout_stride_general_impl<
          // LayoutLeftBased
          true,
          // Strides
          std::integer_sequence<
              std::ptrdiff_t, 1,                 /* first stride is one */
              std::experimental::dynamic_extent, /* second stride is runtime */
              /* rest are layout-left-like with "dimension stride" of 1 */
              repeated_value<std::ptrdiff_t, fixed_layout_dimension_stride<1>,
                             RestIdxs>::value...>,
          // Extents
          Extents,
          // Indices
          std::integer_sequence<std::size_t, 0, 1, RestIdxs...>>> {};

// Special case for the 1D-layout
template <class Extents>
struct MakeKokkosLayoutLeftFromExtents<Extents,
                                       std::integer_sequence<std::size_t, 0>>
    : identity<contiguous_mapping_1d<Extents>> {};

// Special case for the 0D-layout
template <>
struct MakeKokkosLayoutLeftFromExtents<std::experimental::extents<>,
                                       std::integer_sequence<std::size_t>>
    : identity<mapping_0d> {};

// </editor-fold> end make layout left from extents }}}2
//----------------------------------------------------------------------------

template <class Extents,
          class IdxSeq = std::make_index_sequence<Extents::rank()>>
struct MDSpanLayoutForLayoutLeftImpl;

template <class Extents, std::size_t... Idxs>
struct MDSpanLayoutForLayoutLeftImpl<
    Extents, std::integer_sequence<std::size_t, Idxs...>>
    : MakeKokkosLayoutLeftFromExtents<Extents>::type {
 private:
  using extents_t = Extents;
  using base_t    = typename MakeKokkosLayoutLeftFromExtents<extents_t>::type;

  // Deduce extents here to avoid instantiating _set_stride when it's not
  // available
  template <std::ptrdiff_t Ext0, std::ptrdiff_t Ext1,
      std::ptrdiff_t... ExtsDeduced>
  void _setup_dynamic_stride(
      std::experimental::extents<Ext0, Ext1, ExtsDeduced...> const& exts)
  const {
    // Note: Nonstandard mdspan extension
    // set the dynamic stride to be the extent of the last dimension by default
    // Since we only have one dynamic stride, this should be unambiguous
    this->_set_stride(exts.template __extent<0>());
  }

  // Note: if for some reason this doesn't work on some compiler in the future,
  // just make special overloads that deduce 0 and 1 extents, respectively,
  // for a std::extents template specialiation
  template <class ExtentsDeduced>
  void _setup_dynamic_stride(ExtentsDeduced const&) const {
    static_assert(ExtentsDeduced::rank() < 2, "");
  }

 public:
  using base_t::base_t;
  KOKKOS_INLINE_FUNCTION
  // TODO @mdspan should this be explicit?
  MDSpanLayoutForLayoutLeftImpl(Kokkos::LayoutLeft const& ll)
      : base_t(construct_extents_from_kokkos_layout<extents_t>::apply(ll)) {
    // Since we only have one dynamic stride, this should be unambiguous
    _setup_dynamic_stride(this->extents());
  }
};

// </editor-fold> end LayoutLeft }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="LayoutRight"> {{{1

template <class Extents,
          class IdxSeq = std::make_index_sequence<Extents::rank()>>
struct MakeKokkosLayoutRightFromExtents
    // LayoutLeftBased doesn't matter for the 1D or 0D cases, that don't match
    // the partial specialization below, so in the interest of fewer
    // instantiations just re-use the layout left versions
    : MakeKokkosLayoutLeftFromExtents<Extents> {};

template <class Extents, std::size_t... RestIdxs>
struct MakeKokkosLayoutRightFromExtents<
    Extents, std::integer_sequence<std::size_t, 0, 1, RestIdxs...>>
    : identity<layout_stride_general_impl<
          // LayoutLeftBased
          false,
          // Strides
          std::integer_sequence<
              std::ptrdiff_t,
              /* first n-2 are layout-right-like with "dimension stride" of 1 */
              repeated_value<std::ptrdiff_t, fixed_layout_dimension_stride<1>,
                             RestIdxs>::value...,
              std::experimental::dynamic_extent, /* second-to-last stride is
                                                    runtime */
              1>,                                /* last stride is one */
          // Extents
          Extents,
          // Indices
          std::integer_sequence<std::size_t, 0, 1, RestIdxs...>>> {};

template <class Extents>
struct MDSpanLayoutForLayoutRightImpl
    : MakeKokkosLayoutRightFromExtents<Extents>::type {
 private:
  using extents_t = Extents;
  using base_t    = typename MakeKokkosLayoutRightFromExtents<extents_t>::type;

  // Deduce extents here to avoid instantiating _set_stride when it's not
  // available
  template <std::ptrdiff_t Ext0, std::ptrdiff_t Ext1,
            std::ptrdiff_t... ExtsDeduced>
  void _setup_dynamic_stride(
      std::experimental::extents<Ext0, Ext1, ExtsDeduced...> const& exts)
      const {
    // Note: Nonstandard mdspan extension
    // set the dynamic stride to be the extent of the last dimension by default
    // Since we only have one dynamic stride, this should be unambiguous
    this->_set_stride(exts.template __extent<sizeof...(ExtsDeduced) + 1>());
  }

  // Note: if for some reason this doesn't work on some compiler in the future,
  // just make special overloads that deduce 0 and 1 extents, respectively,
  // for a std::extents template specialiation
  template <class ExtentsDeduced>
  void _setup_dynamic_stride(ExtentsDeduced const&) const {
    static_assert(ExtentsDeduced::rank() < 2, "");
  }

 public:
  using base_t::base_t;
  KOKKOS_INLINE_FUNCTION
  // TODO @mdspan should this be explicit?
  MDSpanLayoutForLayoutRightImpl(Kokkos::LayoutRight const& ll)
      : base_t(construct_extents_from_kokkos_layout<extents_t>::apply(ll)) {
    _setup_dynamic_stride(this->extents());
  }
};

// </editor-fold> end LayoutRight }}}1
//==============================================================================

template <class Traits, class Layout>
struct MDSpanLayoutFromKokkosLayout;

template <class Traits>
struct MDSpanLayoutFromKokkosLayout<Traits, Kokkos::LayoutLeft> {
  template <class Extents>
  using mapping = MDSpanLayoutForLayoutLeftImpl<Extents>;
};

template <class Traits>
struct MDSpanLayoutFromKokkosLayout<Traits, Kokkos::LayoutRight> {
  template <class Extents>
  using mapping = MDSpanLayoutForLayoutRightImpl<Extents>;
};

// TODO @mdspan layout stride

// </editor-fold> end MDSpanLayoutFromKokkosLayout }}}1
//==============================================================================

}  // end namespace Impl
}  // end namespace Kokkos

#endif  // KOKKOS_KOKKOS_MDSPANLAYOUT_HPP
