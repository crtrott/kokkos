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

  struct kalmar_team_member_type {
    enum {TEAM_SIZE = 256 }; 
    typedef TeamPolicy<Kokkos::Kalmar,void,Kokkos::Kalmar> TeamPolicy;
    KOKKOS_INLINE_FUNCTION int league_rank() const ;
    KOKKOS_INLINE_FUNCTION int league_size() const { return m_league_size ; }
    KOKKOS_INLINE_FUNCTION int team_rank() const ;
    KOKKOS_INLINE_FUNCTION int team_size() const { return TEAM_SIZE ; }

    KOKKOS_INLINE_FUNCTION
    kalmar_team_member_type( const TeamPolicy & arg_policy
               , const hc::tiled_index< 1 > & arg_idx )
      : m_league_size( arg_policy.league_size() )
      , m_league_rank( arg_idx.tile[0]  )
      , m_team_rank( arg_idx.local[0] )
      {}

  private:
    int m_league_size ;
    int m_league_rank ;
    int m_team_rank ;
  };

template< class Arg0 , class Arg1 >
class TeamPolicy< Arg0 , Arg1 , Kokkos::Kalmar > {
public:
  enum { TEAM_SIZE = 256 };
  int m_league_size ;

  using execution_policy = TeamPolicy ;
  using execution_space  = Kokkos::Kalmar ;
  using work_tag         = void ;

  TeamPolicy( const int arg_league_size
            , const int arg_team_size )
    : m_league_size( arg_league_size )
    {}

  KOKKOS_INLINE_FUNCTION int team_size() const { return TEAM_SIZE ; }
  KOKKOS_INLINE_FUNCTION int league_size() const { return m_league_size ; }

  //This is again a reference thing from other module error 
  //We used auto last time to work around it.
  /*
  struct member_type {
    KOKKOS_INLINE_FUNCTION int league_rank() const ;
    KOKKOS_INLINE_FUNCTION int league_size() const { return m_league_size ; }
    KOKKOS_INLINE_FUNCTION int team_rank() const ;
    KOKKOS_INLINE_FUNCTION int team_size() const { return TEAM_SIZE ; }

    KOKKOS_INLINE_FUNCTION
    member_type( const TeamPolicy & arg_policy
               , const hc::tiled_index< TEAM_SIZE > & arg_idx )
      : m_league_size( arg_policy.league_size() )
      , m_league_rank( arg_idx.tile[0]  )
      , m_team_rank( arg_idx.local[0] )
      {}

  private:
    int m_league_size ;
    int m_league_rank ;
    int m_team_rank ;
  };
  */
  typedef kalmar_team_member_type member_type;
};

} // namespace Kokkos

namespace Kokkos {
namespace Impl {

//----------------------------------------------------------------------------

template< class FunctorType , class Arg0 , class Arg1 , class Arg2 >
class ParallelFor< FunctorType
                 , Kokkos::RangePolicy< Arg0 , Arg1 , Arg2 , Kokkos::Kalmar > >
{
private:

  typedef Kokkos::RangePolicy< Arg0 , Arg1 , Arg2 , Kokkos::Kalmar > Policy ;

  const FunctorType& m_functor ;
  typename Policy::member_type m_offset ;

public:
  template<typename Tag>
  KOKKOS_INLINE_FUNCTION
  static
  void driver(const FunctorType& functor,
              typename std::enable_if< std::is_same<Tag, void>::value,
                                       typename Policy::member_type const & >::type index) { functor(index); }

  template<typename Tag>
  KOKKOS_INLINE_FUNCTION
  static
  void driver(const FunctorType& functor,
              typename std::enable_if< !std::is_same<Tag, void>::value,
                                       typename Policy::member_type const & >::type index) { functor(Tag(), index); }

  KOKKOS_INLINE_FUNCTION
  void operator()( const hc::index<1> & idx ) const
    {
       ParallelFor::template driver<typename Policy::work_tag> (m_functor, idx[0] + m_offset);
    }

  inline
  ParallelFor( const FunctorType & functor
             , const Policy      & policy )
     : m_functor( functor ),
       m_offset( policy.begin() )
    {

#if 0
      auto make_lambda = [this]( const hc::index<1> & idx ) restrict(amp) {
        this->operator() (idx);
      };
      hc::parallel_for_each( hc::extent<1>(
         policy.end()-policy.begin()) , make_lambda);
#else
      hc::completion_future fut = hc::parallel_for_each( hc::extent<1>(
         policy.end()-policy.begin()) , *this);
      fut.wait();
#endif

    }
};

//----------------------------------------------------------------------------

template< class FunctorType , class Arg0 , class Arg1 >
class ParallelFor< FunctorType
                 , Kokkos::TeamPolicy< Arg0 , Arg1 , Kokkos::Kalmar > >
{
  using Policy = Kokkos::TeamPolicy< Arg0 , Arg1 , Kokkos::Kalmar > ;
  const FunctorType& m_functor ;
public:
  inline
  ParallelFor( const FunctorType & functor
             , const Policy      & policy )
    :m_functor(functor)
    {
#if 0
      auto make_lambda =
        [&]( const hc::tiled_index< 1 > & idx ) restrict(amp)
      {
        using member_type = typename Policy::member_type ;
        
        this->m_functor( kalmar_team_member_type( policy , idx ) );
      };

      hc::extent< 1 >
        flat_extent( policy.league_size() * 256 );

      hc::tiled_extent< 1 > team_extent =
        flat_extent.tile(256);

      hc::parallel_for_each( team_extent , make_lambda );
#else
      hc::extent< 1 >
        flat_extent( policy.league_size() * 256 );

      hc::tiled_extent< 1 > team_extent =
        flat_extent.tile(256);

      hc::completion_future fut = hc::parallel_for_each( team_extent , *this );
      fut.wait();
#endif
    }
};

//----------------------------------------------------------------------------

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

