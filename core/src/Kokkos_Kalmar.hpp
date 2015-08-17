/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
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
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact  H. Carter Edwards (hcedwar@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOS_KALMAR_HPP
#define KOKKOS_KALMAR_HPP

#include <Kokkos_Core_fwd.hpp>

#if defined( KOKKOS_HAVE_KALMAR )

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#include <cstddef>
#include <iosfwd>
#include <Kokkos_HostSpace.hpp>
#include <Kokkos_ScratchSpace.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_Layout.hpp>
#include <impl/Kokkos_Tags.hpp>

/*--------------------------------------------------------------------------*/

#include <hc.hpp>

namespace Kokkos {

/// \class Kalmar
/// \brief Kokkos device for multicore processors in the host memory space.
class Kalmar {
public:
  //------------------------------------
  //! \name Type declarations that all Kokkos devices must provide.
  //@{

  //! Tag this class as a kokkos execution space
  typedef Kalmar                execution_space ;
  typedef HostSpace             memory_space ;
  //! This execution space preferred device_type
  typedef Kokkos::Device<execution_space,memory_space> device_type;

  typedef LayoutRight           array_layout ;
  typedef HostSpace::size_type  size_type ;

  typedef ScratchMemorySpace< Serial > scratch_memory_space ;

  //@}
  //------------------------------------
  //! \name Functions that all Kokkos devices must implement.
  //@{

  inline static bool in_parallel() restrict(cpu) { return false ; }
  inline static bool in_parallel() restrict(amp) { return true ; }

  /** \brief  Set the device in a "sleep" state. A noop for OpenMP.  */
  static bool sleep() {return true;}

  /** \brief Wake the device from the 'sleep' state. A noop for OpenMP. */
  static bool wake() {return true;}

  /** \brief Wait until all dispatched functors complete. A noop for OpenMP. */
  static void fence() {}

  /// \brief Print configuration information to the given output stream.
  static void print_configuration( std::ostream & , const bool detail = false ) {}

  /// \brief Free any resources being consumed by the device.
  static void finalize() {}

  /** \brief  Initialize the device.
   *
   *  1) If the hardware locality library is enabled and OpenMP has not
   *     already bound threads then bind OpenMP threads to maximize
   *     core utilization and group for memory hierarchy locality.
   *
   *  2) Allocate a HostThread for each OpenMP thread to hold its
   *     topology and fan in/out data.
   */
  static void initialize( ) {}

  static int is_initialized() {return 1;}
};

} // namespace Kokkos


#include <Kalmar/Kokkos_Kalmar_Parallel.hpp>

#endif
#endif


