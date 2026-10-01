// Compile-time test for src/fp16.h's device-pass-gated section, the
// device-compile half layer. Unlike the .cppm wwr* tests beside it, that section
// defines __device__ functions over vendor types that only a device pass can
// name, so -- like cooperative_groups.cu -- the test is a device TU and building
// it under the selected backend IS the assertion. The kernel is never launched:
// every name below just has to compile through the one include switch on both
// backends (nvcc and clang's -x hip disagree on more than the include path).
// Reached through wwr.device, exactly as a real device consumer reaches the header.
#include "fp16.h"

// Defining a __global__ kernel needs the launch runtime (hipLaunchKernel under
// HIP). fp16.h's vendor headers happen to pull it transitively, but this
// kernel TU names it itself rather than lean on that. nvcc supplies
// <cuda_runtime.h> for a .cu implicitly; spell both for a self-contained TU.
// WWR_SELECTED_* comes from fp16.h (via selected_backend.h).
#if defined(WWR_SELECTED_CUDA)
#include <cuda_runtime.h>
#else
#include <hip/hip_runtime.h>
#endif

#include <type_traits>

namespace wwr {
namespace {

// ---------------------------------------------------------------------------
// Layout parity: the header claims a host-allocated buffer and a kernel
// parameter named here agree on type, so their sizes must match what the module
// side and a caller assume -- 16 bits, on both backends. A vendor changing it
// breaks that agreement silently.
// ---------------------------------------------------------------------------
static_assert(sizeof(wwrHalf) == 2, "wwrHalf is expected to be 16 bits");

} // namespace

// Half conversions, both directions, plus the operators. The forward conversion
// and the widening one close the round trip -- a store-as-half functor converts
// one way and reads back the other. Both directions are __device__ and spelled
// identically by the vendors, so no #if reaches this TU; this kernel is a device
// context, which is where the conversions are callable.
__global__ void wwr_fp_conversions(float *out) {
  const wwrHalf h = wwrFloat2Half(1.5f);
  float acc = wwrHalf2Float(h);

  // §3's other half: half DOES carry operators on both backends, so a functor
  // accumulating in reduced precision needs no wrapper -- this is already
  // backend-neutral code naming no vendor symbol. Exercised here so a vendor
  // dropping the operators is caught too.
  const wwrHalf hsum = h + wwrFloat2Half(2.0f);
  acc += wwrHalf2Float(hsum);

  out[0] = acc;
}

} // namespace wwr
