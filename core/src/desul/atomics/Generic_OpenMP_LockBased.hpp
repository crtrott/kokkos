
#pragma omp begin declare variant match(device = {kind(host)})
template <class Oper, typename T, class MemoryOrder, class MemoryScope,
  // equivalent to:
  //   requires !atomic_always_lock_free(sizeof(T))
  std::enable_if_t<!atomic_always_lock_free(sizeof(T)), int> = 0
>
DESUL_INLINE_FUNCTION T
atomic_fetch_oper(const Oper& op,
                  T* const dest,
                  dont_deduce_this_parameter_t<const T> val,
                  MemoryOrder /*order*/,
                  MemoryScope scope) {
  // Acquire a lock for the address
  while (!Impl::lock_address((void*)dest, scope)) {}

  atomic_thread_fence(MemoryOrderAcquire(),scope);
  T return_val = *dest;
  *dest = op.apply(return_val, val);
  atomic_thread_fence(MemoryOrderRelease(),scope);
  Impl::unlock_address((void*)dest, scope);
  return return_val;
}
#pragma omp end declare variant

#pragma omp begin declare variant match(device = {kind(nohost)})
template <class Oper, typename T, class MemoryOrder, class MemoryScope,
  // equivalent to:
  //   requires !atomic_always_lock_free(sizeof(T))
  std::enable_if_t<!atomic_always_lock_free(sizeof(T)), int> = 0
>
DESUL_INLINE_FUNCTION T
atomic_fetch_oper(const Oper& op,
                  T* const dest,
                  dont_deduce_this_parameter_t<const T> val,
                  MemoryOrder /*order*/,
                  MemoryScope scope) {
  // This is a way to avoid dead lock in a warp or wave front
  T return_val;
  unsigned int done = 0;
  unsigned int mask = openmp_mask();
  unsigned int done_active = 0;
  while (mask != done_active) {
    if (!done) {
      if (Impl::lock_address_openmp((void*)dest, scope)) {
        atomic_thread_fence(MemoryOrderAcquire(), scope);
        return_val = *dest;
        *dest = op.apply(return_val, val);
        atomic_thread_fence(MemoryOrderRelease(), scope);
        Impl::unlock_address_openmp((void*)dest, scope);
        done = 1;
      }
    }
    done_active = openmp_ballot_mask(mask,done);
  }
  return return_val;
}
#pragma omp end declare variant

#pragma omp begin declare variant match(device = {kind(host)})
template <class Oper, typename T, class MemoryOrder, class MemoryScope,
  // equivalent to:
  //   requires !atomic_always_lock_free(sizeof(T))
  std::enable_if_t<!atomic_always_lock_free(sizeof(T)), int> = 0
>
DESUL_INLINE_FUNCTION T
atomic_oper_fetch(const Oper& op,
                  T* const dest,
                  dont_deduce_this_parameter_t<const T> val,
                  MemoryOrder /*order*/,
                  MemoryScope scope) {
  // Acquire a lock for the address
  while (!Impl::lock_address((void*)dest, scope)) {}

  atomic_thread_fence(MemoryOrderAcquire(),scope);
  T return_val = op.apply(*dest, val);
  *dest = return_val;
  atomic_thread_fence(MemoryOrderRelease(),scope);
  Impl::unlock_address((void*)dest, scope);
  return return_val;
}
#pragma omp end declare variant

#pragma omp begin declare variant match(device = {kind(nohost)})
template <class Oper, typename T, class MemoryOrder, class MemoryScope,
  // equivalent to:
  //   requires !atomic_always_lock_free(sizeof(T))
  std::enable_if_t<!atomic_always_lock_free(sizeof(T)), int> = 0
>
DESUL_INLINE_FUNCTION T
atomic_oper_fetch(const Oper& op,
                  T* const dest,
                  dont_deduce_this_parameter_t<const T> val,
                  MemoryOrder /*order*/,
                  MemoryScope scope) {
  T return_val;
  unsigned int done = 0;
  unsigned int mask = openmp_mask();
  unsigned int done_active = 0;
  while (mask != done_active) {
    if (!done) {
      if (Impl::lock_address_openmp((void*)dest, scope)) {
        atomic_thread_fence(MemoryOrderAcquire(),scope);
        return_val = op.apply(*dest, val);
        *dest = return_val;
        atomic_thread_fence(MemoryOrderRelease(),scope);
        Impl::unlock_address_openmp((void*)dest, scope);
        done = 1;
      }
    }
    done_active = openmp_ballot_mask(mask, done);
  }
}
#pragma omp end declare variant
