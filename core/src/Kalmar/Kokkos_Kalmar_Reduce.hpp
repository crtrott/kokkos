/***************************************************************************
*   © 2012,2014 Advanced Micro Devices, Inc. All rights reserved.
*
*   Licensed under the Apache License, Version 2.0 (the "License");
*   you may not use this file except in compliance with the License.
*   You may obtain a copy of the License at
*
*       http://www.apache.org/licenses/LICENSE-2.0
*
*   Unless required by applicable law or agreed to in writing, software
*   distributed under the License is distributed on an "AS IS" BASIS,
*   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*   See the License for the specific language governing permissions and
*   limitations under the License.

***************************************************************************/

///////////////////////////////////////////////////////////////////////////////
// AMP REDUCE
//////////////////////////////////////////////////////////////////////////////

#if !defined( KOKKOS_KALMAR_AMP_REDUCE_INL )
#define KOKKOS_KALMAR_AMP_REDUCE_INL


#if 1
// Issue: taking the address of a 'tile_static' variable
// may not dereference properly ???
#define REDUCE_WAVEFRONT_SIZE 256 //64
#define _REDUCE_STEP(_LENGTH, _IDX, _W) \
if ((_IDX < _W) && ((_IDX + _W) < _LENGTH)) {\
      ValueJoin::join( functor , & scratch[_IDX] , & scratch[ _IDX + _W ] ); \
}\
    t_idx.barrier.wait();
#else

#define REDUCE_WAVEFRONT_SIZE 256 //64
#define _REDUCE_STEP(_LENGTH, _IDX, _W) \
if ((_IDX < _W) && ((_IDX + _W) < _LENGTH)) {\
        scratch[_IDX] += scratch[ _IDX + _W ] ; \
}\
    t_idx.barrier.wait();
#endif

#include <algorithm>
#include <type_traits>

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace Kokkos {
namespace Impl {

// This is the base implementation of reduction that is called by all of the convenience wrappers below.
// first and last must be iterators from a DeviceVector
template< class FunctorType , typename T >
void reduce_enqueue(
  const int           szElements ,
  const FunctorType & functor ,
  T& output_result )
{
  using ValueInit = Kokkos::Impl::FunctorValueInit< FunctorType , void > ;
  using ValueJoin = Kokkos::Impl::FunctorValueJoin< FunctorType , void > ;

  int max_ComputeUnits = 32;
  int numTiles = max_ComputeUnits*32;			/* Max no. of WG for Tahiti(32 compute Units) and 32 is the tuning factor that gives good performance*/

  int length = (REDUCE_WAVEFRONT_SIZE*numTiles);	

  length = szElements < length ? szElements : length;
  unsigned int residual = length % REDUCE_WAVEFRONT_SIZE;
  length = residual ? (length + REDUCE_WAVEFRONT_SIZE - residual): length ;

  const int numTilesMax =( szElements + REDUCE_WAVEFRONT_SIZE - 1 ) / REDUCE_WAVEFRONT_SIZE ;

  if ( numTilesMax < numTiles ) numTiles = numTilesMax ;
		
  // For storing tiles' contributions:
  //long * const terms  = new long[szElements];
  T * const result = new T[numTiles];

  //for ( int i =0 ; i <numTiles ; ++i ) result[i] = 99000 + i ;

  hc::extent< 1 > inputExtent(length);
  hc::tiled_extent< 1 >
    tiledExtentReduce = inputExtent.tile(REDUCE_WAVEFRONT_SIZE);

  // AMP doesn't have APIs to get CU capacity. Launchable size is great though.
/*
  printf("reduce_enqueue szElements %d tiledExtentReduce %d length %d\n"
        , szElements
        , tiledExtentReduce.tile_dim0
        , length
        );
*/
  try
  {
    hc::completion_future fut = hc::parallel_for_each
      ( tiledExtentReduce
      , [ = , & functor]
        ( hc::tiled_index<1> t_idx ) restrict(amp)
        {
          tile_static T scratch[REDUCE_WAVEFRONT_SIZE];

          int gx = t_idx.global[0];
          int gloId = gx;
          //  Initialize local data store
          //  Index of this member in its work group.
          unsigned int tileIndex = t_idx.local[0];

          T tmp; 
          T accumulator = ValueInit::init(functor,&tmp) ;

          // Shared memory within this tile:
          // ValueInit::init( m_functor , & accumulator );

          for ( ; gx < szElements ; gx += length ) {
            // functor( gx , accumulator );
            // terms[gx] = accumulator ;
            // terms[gx] = 1 ;
            {
#if 0
              // correct accumulator and terms
              accumulator += ( terms[gx] = gx + 1 );
#elif 0
              // correct accumulator and terms
              terms[gx] = gx + 1 ;
              accumulator += terms[gx] ;
#elif 0
              // error accumulator and correct terms
              terms[gx] = functor.value( gx );
              terms[gx] += 1 ;
              // accumulator += terms[gx] ;
#else
              functor(gx,accumulator);
              //terms[gx] = accumulator;
#endif
            }
          }

          scratch[tileIndex] = accumulator;
          t_idx.barrier.wait();

          unsigned int tail = szElements - (t_idx.tile[0] * REDUCE_WAVEFRONT_SIZE);

          // Reduce within this tile:
#if 1
          _REDUCE_STEP(tail, tileIndex, 128);
          _REDUCE_STEP(tail, tileIndex, 64);
          _REDUCE_STEP(tail, tileIndex, 32);
          _REDUCE_STEP(tail, tileIndex, 16);
          _REDUCE_STEP(tail, tileIndex, 8);
          _REDUCE_STEP(tail, tileIndex, 4);
          _REDUCE_STEP(tail, tileIndex, 2);
          _REDUCE_STEP(tail, tileIndex, 1);
#endif


          //  Abort threads that are passed the end of the input vector
          if (gloId >= szElements)
          	return;

          //  Write only the single reduced value for the entire workgroup
          if (tileIndex == 0)
          {
            result[t_idx.tile[ 0 ]] = scratch[0];
          }

       });
       // End of hc::parallel_for_each
       fut.wait();
       T acc = result[0];

       // ValueInit::init( m_functor , & acc );

       //for(int i = 0; i < szElements; ++i)
       //    printf("terms[%d] = %ld\n",i,long(terms[i]));

       for(int i = 1; i < numTiles; ++i)
         {
           //printf("result[%d] = %ld\n",i,long(result[i]));
           ValueJoin::join( functor , & acc, result + i );
         }

       delete[] result ;

       output_result = acc ;

       // TBD: apply functor's final operations
  }
  catch(std::exception &e)
  {
    throw e ;
  }


}

}} //end of namespace Kokkos::Impl

#endif /* #if !defined( KOKKOS_KALMAR_AMP_REDUCE_INL ) */

