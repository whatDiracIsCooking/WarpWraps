// Compile-time test for src/fp8.cuh, the device-compile half layer. Unlike the
// .cppm wwr* tests beside it, this header defines __device__ functions over
// vendor types that only a device pass can name, so -- like fp16.cu / bf16.cu --
// the test is a device TU and building it under the selected backend IS the
// assertion. The kernel is never launched: every name below just has to compile
// through the one include switch on both backends (nvcc and clang's -x hip
// disagree on more than the include path). Reached through wwr.device, exactly
// as a real device consumer reaches the header.
#include "fp8.cuh"

// Defining a __global__ kernel needs the launch runtime (hipLaunchKernel under
// HIP). fp8.cuh's vendor headers happen to pull it transitively, but this
// kernel TU names it itself rather than lean on that. nvcc supplies
// <cuda_runtime.h> for a .cu implicitly; spell both for a self-contained TU.
// WWR_SELECTED_* comes from fp8.cuh (via device_guard.h).
#if defined(WWR_SELECTED_CUDA)
#include <cuda_runtime.h>
#else
#include <hip/hip_runtime.h>
#endif

namespace wwr {
namespace {

// ---------------------------------------------------------------------------
// Layout parity: the header claims a host-allocated buffer and a kernel
// parameter named here agree on type, so their widths must match what the
// module side and a caller assume -- one byte per scalar, on both backends. A
// vendor changing it breaks that agreement silently. The same widths the
// module-side test/gpu/fp8.cppm pins.
// ---------------------------------------------------------------------------
static_assert(sizeof(wwrFp8Storage) == 1);
static_assert(sizeof(wwrFp8x2Storage) == 2);
static_assert(sizeof(wwrFp8x4Storage) == 4);
static_assert(sizeof(wwrFp8E4m3) == 1);
static_assert(sizeof(wwrFp8E5m2) == 1);

} // namespace

// The narrowing forwarders, both float and double, in both OCP formats -- the
// one thing this header adds that names a divergent vendor symbol. Every name
// is __device__ and spelled identically by the vendors through the header's
// switch, so no #if reaches this TU; this kernel is a device context, which is
// where the conversions are callable.
__global__ void wwr_fp8_conversions(float *out) {
  const wwrFp8Storage e4 = wwrFloat2Fp8(1.5f, wwrSatfinite, wwrE4m3);
  const wwrFp8Storage e5 = wwrDouble2Fp8(2.5, wwrNosat, wwrE5m2);

  // §3's other half: ctor-narrowing (wwrFp8E4m3{x}) and widening
  // (static_cast<float>) ride the scalar type's own members -- no wrapper, no
  // vendor symbol -- so once the alias exists this is already backend-neutral
  // code. Exercised here so a vendor dropping either is caught too.
  const wwrFp8E4m3 v4{1.5f};
  const wwrFp8E5m2 v5{2.5f};
  float acc = static_cast<float>(v4) + static_cast<float>(v5);

  // Fold the raw storage bytes in too, so the narrowing forwarders above are
  // named and not elided. They are integer storage, so this is a plain
  // widening of the byte value -- the fp8-aware widening is the scalar path.
  acc += static_cast<float>(e4) + static_cast<float>(e5);

  out[0] = acc;
}

} // namespace wwr
