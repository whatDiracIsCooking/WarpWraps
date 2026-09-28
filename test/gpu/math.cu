// Compile-time test for src/wrappers/math/math.cuh, the templated device math
// layer.
// Like fp16.cu / atomics.cu beside it, this header's functions are __device__
// templates over vendor intrinsics that only a device pass can name, so the
// test is a device TU and building it under the selected backend IS the
// assertion. The kernel is never launched: every instantiation below just has
// to compile through the one include switch on both backends (nvcc and clang's
// -x hip disagree on more than the include path). Reached through wwr.device,
// exactly as a real device consumer reaches the header.
#include "wrappers/math/math.cuh"

// Defining a __global__ kernel needs the launch runtime. math.cuh's vendor
// runtime header happens to pull it, but this kernel TU names it itself rather
// than lean on that -- the same self-containment fp16.cu keeps. WWR_SELECTED_*
// comes from math.cuh (via device_guard.h).
#if defined(WWR_SELECTED_CUDA)
#include <cuda_runtime.h>
#else
#include <hip/hip_runtime.h>
#endif

namespace wwr {
namespace {

// Instantiate every wrapper for one element type, so each template body is
// forced through the float/double branch it dispatches to. A functor doing
// real element-wise math reaches exactly these names.
template <typename T> __device__ T exercise(const T x, const T y) {
  T acc = exp(x) + exp2(x) + expm1(x);
  acc += log(x) + log2(x) + log10(x) + log1p(x);
  acc += sqrt(x) + rsqrt(x);
  acc += sin(x) + cos(x) + tan(x);
  acc += asin(x) + acos(x) + atan(x);
  acc += sinh(x) + cosh(x) + tanh(x);
  acc += floor(x) + ceil(x) + trunc(x) + round(x) + rint(x);
  acc += fabs(x);
  acc += pow(x, y) + atan2(x, y) + hypot(x, y) + fmod(x, y);
  acc += copysign(x, y) + fmin(x, y) + fmax(x, y);
  acc += fma(x, y, acc);
  // Predicates return bool; fold them in so their instantiations are forced too.
  if (isnan(x) || isinf(x) || !isfinite(x)) {
    acc += y;
  }
  return acc;
}

} // namespace

// Both precisions in one kernel: float and double take different vendor entry
// points (::expf vs ::exp, ...), so instantiating both is what proves the
// compile-time dispatch resolves on each. Spelled identically on both backends,
// so no #if reaches this body.
__global__ void wwr_math(float *fout, double *dout) {
  fout[0] = exercise(fout[0], fout[1]);
  dout[0] = exercise(dout[0], dout[1]);
}

} // namespace wwr
