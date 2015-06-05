#include <amp.h>
#if 0
//#ifdef KOKKOS_USE_ATOMICS_KALMAR_GPU
namespace Kokkos {
  //Kalmar can do:
  //Types int/unsigned int
  //variants: atomic_exchange/compare_exchange/fetch_add/fetch_sub/fetch_max/fetch_min/fetch_and/fetch_or/fetch_xor/fetch_inc/fetch_dec 


  KOKKOS_INLINE_FUNCTION
  int atomic_compare_exchange(int* dest, int compare, int& val) {
    #ifdef __GPU__
    ::atomic_compare_exchange(dest,&compare,val);
    #endif 
    return compare;
  }

  KOKKOS_INLINE_FUNCTION
  unsigned int atomic_compare_exchange(unsigned int* dest, unsigned int compare, unsigned int& val) {
    #ifdef __GPU__
    ::atomic_compare_exchange(dest,&compare,val);
    #endif 
    return compare;
   
  }

  template<class T>
  KOKKOS_INLINE_FUNCTION
  T atomic_compare_echange(T* dest, T& compare , typename std::enable_if<sizeof(T) == sizeof(int),T>::type & val) {
    union U {
      int i ;
      T t ;
      KOKKOS_INLINE_FUNCTION U() {};
    } assume , oldval, newval ;

    assume.t = compare;
    oldval.t = compare;

    Kokkos::atomic_compare_exchange( (int*)dest , & assume.i , newval.i );

    return assume.t ;
  }

  KOKKOS_INLINE_FUNCTION
  int atomic_fetch_add(int* dest, int& val) {
    return ::atomic_fetch_add(dest,val);
  }
  
  KOKKOS_INLINE_FUNCTION
  unsigned int atomic_fetch_add(unsigned int* dest, unsigned int& val) {
    return ::atomic_fetch_add(dest,val);
  }

  template<class T>
  KOKKOS_INLINE_FUNCTION
  T atomic_fetch_add(T* dest, typename std::enable_if<sizeof(T) == sizeof(int),T>::type & val) {
    union U {
      int i ;
      T t ;
      KOKKOS_INLINE_FUNCTION U() {};
    } assume , oldval , newval ;

    oldval.t = *dest ;

    do {
      assume.i = oldval.i ;
      newval.t = assume.t + val ;
      oldval.i = Kokkos::atomic_compare_exchange( (int*)dest , assume.i , newval.i );
    } while ( assume.i != oldval.i );

    return oldval.t ;    
  }

  KOKKOS_INLINE_FUNCTION
  int atomic_fetch_sub(int* dest, int val) {
    #ifdef __GPU__
    return ::atomic_fetch_sub(dest,val);
    #endif
  }

  KOKKOS_INLINE_FUNCTION
  unsigned int atomic_fetch_sub(unsigned int* dest, unsigned int& val) {
    #ifdef __GPU__ 
    return ::atomic_fetch_sub(dest,val);
    #endif
  }

  template<class T>
  KOKKOS_INLINE_FUNCTION
  T atomic_fetch_sub(T* dest, typename std::enable_if<sizeof(T) == sizeof(int),T>::type & val) {
    union U {
      int i ;
      T t ;
      KOKKOS_INLINE_FUNCTION U() {};
    } assume , oldval , newval ;

    oldval.t = *dest ;

    do {
      assume.i = oldval.i ;
      newval.t = assume.t - val ;
      oldval.i = Kokkos::atomic_compare_exchange( (int*)dest , assume.i , newval.i );
    } while ( assume.i != oldval.i );

    return oldval.t ;
  }
}
#endif

