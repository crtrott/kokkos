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

#include <Kalmar/Kokkos_Kalmar_Reduce.hpp>

namespace Kokkos {
namespace Impl {

template< class FunctorType , class Arg0 , class Arg1 , class Arg2 >
class ParallelFor< FunctorType
                 , Kokkos::RangePolicy< Arg0 , Arg1 , Arg2 , Kokkos::Kalmar > >
{
private:

  typedef Kokkos::RangePolicy< Arg0 , Arg1 , Arg2 , Kokkos::Kalmar > Policy ;

  FunctorType m_functor ;
  typename Policy::member_type m_offset ;

  inline
  void operator()( const concurrency::index<1> & idx ) const restrict(amp)
    {
       m_functor( idx[0] + m_offset);
    }

public:

  inline
  ParallelFor( const FunctorType & functor
             , const Policy      & policy )
     : m_functor( functor ),
       m_offset( policy.begin() )
    {

      auto make_lambda = [this]( const concurrency::index<1> & idx ) restrict(amp) {
        this->operator() (idx);
      };
      concurrency::parallel_for_each( concurrency::extent<1>(
         policy.end()-policy.begin()) , make_lambda);

    }
};

template< class FunctorType , class Arg0 , class Arg1 , class Arg2 >
class ParallelReduce<
  FunctorType , Kokkos::RangePolicy< Arg0 , Arg1 , Arg2 , Kokkos::Kalmar > >
{
public:

  typedef Kokkos::RangePolicy< Arg0 , Arg1 , Arg2 , Kokkos::Kalmar > Policy ;

  template< class ViewType >
  inline
  ParallelReduce( typename Impl::enable_if<
                    ( Impl::is_view< ViewType >::value &&
                      Impl::is_same< typename ViewType::memory_space , HostSpace >::value
                    ), const FunctorType & >::type functor
                , const Policy    & policy
                , const ViewType  & result_view )
    {
      Kokkos::Impl::reduce_enqueue
        ( policy.end() - policy.begin()
        , functor
        , result_view()
        );
    }
};

}
}

