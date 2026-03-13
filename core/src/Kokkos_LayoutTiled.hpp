// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

/// \file Kokkos_LayoutTiled.hpp
/// \brief Declaration of the \c layout_tiled mdspan layout policy.

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#include <Kokkos_Macros.hpp>
static_assert(false,
              "Including non-public Kokkos header files is not allowed.");
#endif

#ifndef KOKKOS_LAYOUT_TILED_HPP
#define KOKKOS_LAYOUT_TILED_HPP

#include <Kokkos_Macros.hpp>

#ifdef KOKKOS_ENABLE_IMPL_MDSPAN
#include <View/MDSpan/Kokkos_MDSpan_Header.hpp>

namespace Kokkos {

/// \struct layout_tiled
/// \brief  An mdspan layout policy implementing a rank-2 tiled mapping with
///         compile-time tile sizes.
///
/// Elements are arranged in row-major order both within each tile and
/// across tiles.  For a matrix with extents (M, N) and tile sizes
/// (TileSize0, TileSize1), element (i0, i1) maps to:
///
///   tile_row  = i0 / TileSize0
///   tile_col  = i1 / TileSize1
///   num_tiles_1 = ceil(N / TileSize1)
///   offset    = (tile_row * num_tiles_1 + tile_col) * TileSize0 * TileSize1
///             + (i0 % TileSize0) * TileSize1
///             + (i1 % TileSize1)
///
/// The required span size is always a multiple of TileSize0 * TileSize1
/// (i.e., padded up to the next full tile in each dimension).
template <std::size_t TileSize0, std::size_t TileSize1>
struct layout_tiled {
  static_assert(TileSize0 > 0, "layout_tiled: TileSize0 must be positive");
  static_assert(TileSize1 > 0, "layout_tiled: TileSize1 must be positive");

  template <class Extents>
  class mapping {
   public:
    static_assert(Extents::rank() == 2,
                  "layout_tiled requires rank-2 extents");

    using extents_type = Extents;
    using index_type   = typename extents_type::index_type;
    using size_type    = typename extents_type::size_type;
    using rank_type    = typename extents_type::rank_type;
    using layout_type  = layout_tiled;

    KOKKOS_DEFAULTED_FUNCTION constexpr mapping() noexcept = default;
    KOKKOS_DEFAULTED_FUNCTION constexpr mapping(const mapping&) noexcept =
        default;
    KOKKOS_DEFAULTED_FUNCTION constexpr mapping& operator=(
        const mapping&) noexcept = default;

    KOKKOS_FUNCTION explicit constexpr mapping(
        const extents_type& exts) noexcept
        : m_extents(exts) {}

    // Conversion from a mapping with different extents (if constructible)
    template <class OtherExtents,
              std::enable_if_t<
                  std::is_constructible_v<extents_type, OtherExtents>, int> = 0>
    KOKKOS_FUNCTION explicit constexpr mapping(
        const mapping<OtherExtents>& other) noexcept
        : m_extents(other.extents()) {}

    KOKKOS_FUNCTION
    constexpr const extents_type& extents() const noexcept {
      return m_extents;
    }

    /// Returns the number of elements in the backing storage (including
    /// padding from incomplete tiles).
    KOKKOS_FUNCTION
    constexpr index_type required_span_size() const noexcept {
      const index_type n0 = m_extents.extent(0);
      const index_type n1 = m_extents.extent(1);
      const index_type num_tiles_0 =
          (n0 + static_cast<index_type>(TileSize0) - 1) /
          static_cast<index_type>(TileSize0);
      const index_type num_tiles_1 =
          (n1 + static_cast<index_type>(TileSize1) - 1) /
          static_cast<index_type>(TileSize1);
      return num_tiles_0 * static_cast<index_type>(TileSize0) * num_tiles_1 *
             static_cast<index_type>(TileSize1);
    }

    /// Maps a rank-2 index (i0, i1) to a linear offset.
    template <class I0, class I1>
    KOKKOS_FUNCTION constexpr index_type operator()(I0 i0,
                                                     I1 i1) const noexcept {
      const index_type idx0       = static_cast<index_type>(i0);
      const index_type idx1       = static_cast<index_type>(i1);
      const index_type tile_row   = idx0 / static_cast<index_type>(TileSize0);
      const index_type tile_col   = idx1 / static_cast<index_type>(TileSize1);
      const index_type within_row = idx0 % static_cast<index_type>(TileSize0);
      const index_type within_col = idx1 % static_cast<index_type>(TileSize1);
      const index_type num_tiles_1 =
          (m_extents.extent(1) + static_cast<index_type>(TileSize1) - 1) /
          static_cast<index_type>(TileSize1);
      return (tile_row * num_tiles_1 + tile_col) *
                 static_cast<index_type>(TileSize0 * TileSize1) +
             within_row * static_cast<index_type>(TileSize1) + within_col;
    }

    KOKKOS_INLINE_FUNCTION static constexpr bool is_always_unique() noexcept {
      return true;
    }
    /// Not always exhaustive: the layout pads each dimension up to the next
    /// full tile, so elements beyond the logical extents exist in the span.
    KOKKOS_INLINE_FUNCTION static constexpr bool
    is_always_exhaustive() noexcept {
      return false;
    }
    KOKKOS_INLINE_FUNCTION static constexpr bool
    is_always_strided() noexcept {
      return false;
    }

    KOKKOS_INLINE_FUNCTION constexpr bool is_unique() const noexcept {
      return true;
    }
    /// Exhaustive iff both extents are exact multiples of the tile sizes.
    KOKKOS_INLINE_FUNCTION constexpr bool is_exhaustive() const noexcept {
      return (m_extents.extent(0) % static_cast<index_type>(TileSize0) == 0) &&
             (m_extents.extent(1) % static_cast<index_type>(TileSize1) == 0);
    }
    KOKKOS_INLINE_FUNCTION constexpr bool is_strided() const noexcept {
      return false;
    }

    template <class OtherExtents>
    KOKKOS_FUNCTION friend constexpr bool operator==(
        const mapping& lhs, const mapping<OtherExtents>& rhs) noexcept {
      return lhs.extents() == rhs.extents();
    }

   private:
    extents_type m_extents{};
  };
};

}  // namespace Kokkos

#endif  // KOKKOS_ENABLE_IMPL_MDSPAN

#endif  // KOKKOS_LAYOUT_TILED_HPP
