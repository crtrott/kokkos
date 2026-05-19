#include <limits>

namespace desul {
namespace Impl {

#include <desul/atomics/cuda/cuda_cc7_asm_loadstore.inc>

#ifdef DESUL_HAVE_16BYTE_LOCK_FREE_ATOMICS_DEVICE
//#include <desul/atomics/cuda/cuda_cc9_asm_loadstore.inc>
#endif
}  // namespace Impl
}  // namespace desul
