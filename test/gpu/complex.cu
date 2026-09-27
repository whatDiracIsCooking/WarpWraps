// Compile-time test for src/complex.cuh, the device-compile complex layer.
// Unlike the .cppm gpu* tests beside it, this header defines __device__
// functions over vendor types that only a device pass can name, so -- like
// cooperative_groups.cu -- the test is a device TU and building it under the
// selected backend IS the assertion. The kernels are never launched: every name
// below just has to compile through the one include switch on both backends
// (nvcc and clang's -x hip disagree on more than the include path). Reached
// through gpumod.device, exactly as a real device consumer reaches the
// header.
//
// The static_asserts turn the §3 divergence complex.cuh is built around into a
// compile-time tripwire: cuComplex is an operator-less float2 aggregate where
// hipComplex is a class, which is *why* complex arithmetic goes through gpuC*
// functions rather than operators. Pinning the aggregate/class split per backend
// makes a vendor closing that gap -- or this header switching to a complex type
// with different layout -- fail the build instead of drifting silently. Backend
// is selected on WWR_SELECTED_CUDA, the device-pass macro the header itself
// switches on, consistent across both of HIP's compile passes.
#include "complex.cuh"

// Defining a __global__ kernel needs the launch runtime (hipLaunchKernel under
// HIP). complex.cuh pulls only the complex type header, not the runtime, so this
// kernel TU names it itself rather than lean on a transitive include. nvcc
// supplies <cuda_runtime.h> for a .cu implicitly; spell both for a self-contained
// TU. WWR_SELECTED_* comes from complex.cuh (via device_guard.h).
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
// side and a caller assume -- two floats / two doubles, on both backends. A
// vendor changing either breaks that agreement silently.
// ---------------------------------------------------------------------------
static_assert(sizeof(gpuFloatComplex) == 2 * sizeof(float),
              "gpuFloatComplex is expected to be two floats");
static_assert(sizeof(gpuDoubleComplex) == 2 * sizeof(double),
              "gpuDoubleComplex is expected to be two doubles");

// The complex types must stay trivially copyable on both backends -- that is
// what lets a value cross the host/device boundary as raw bytes.
static_assert(std::is_trivially_copyable_v<gpuFloatComplex>,
              "gpuFloatComplex is expected to be trivially copyable");
static_assert(std::is_trivially_copyable_v<gpuDoubleComplex>,
              "gpuDoubleComplex is expected to be trivially copyable");

// §3, the divergence this header exists to absorb: cuComplex is a plain float2
// aggregate with no operators, hipComplex is a class that defines them. Pinning
// it per backend documents why gpuC* arithmetic is functional and traps a
// vendor changing the type category out from under that choice.
#if defined(WWR_SELECTED_CUDA)
static_assert(std::is_aggregate_v<gpuFloatComplex>,
              "CUDA cuFloatComplex is expected to be an aggregate (float2) with "
              "no arithmetic operators");
#else
static_assert(!std::is_aggregate_v<gpuFloatComplex>,
              "HIP hipFloatComplex is expected to be a class type that carries "
              "its own operators");
#endif

} // namespace

// Single-precision complex: construction, both accessors, magnitude, conjugate
// and the four arithmetic operations -- the whole gpuC*f surface, which is the
// only portable spelling because cuFloatComplex has no operator* (see the
// static_assert above).
__global__ void wwr_fp_complex_float(float *out) {
  const gpuFloatComplex a = make_gpuFloatComplex(1.0f, 2.0f);
  const gpuFloatComplex b = make_gpuFloatComplex(3.0f, -1.0f);

  const gpuFloatComplex sum = gpuCaddf(a, b);
  const gpuFloatComplex dif = gpuCsubf(a, b);
  const gpuFloatComplex prod = gpuCmulf(a, b);
  const gpuFloatComplex quot = gpuCdivf(a, b);
  const gpuFloatComplex conj = gpuConjf(a);

  float acc = gpuCrealf(sum) + gpuCimagf(dif);
  acc += gpuCrealf(prod) + gpuCimagf(quot);
  acc += gpuCrealf(conj) + gpuCimagf(conj);
  acc += gpuCabsf(a);
  out[0] = acc;
}

// Double-precision complex: the same surface over gpuDoubleComplex.
__global__ void wwr_fp_complex_double(double *out) {
  const gpuDoubleComplex a = make_gpuDoubleComplex(1.0, 2.0);
  const gpuDoubleComplex b = make_gpuDoubleComplex(3.0, -1.0);

  const gpuDoubleComplex sum = gpuCadd(a, b);
  const gpuDoubleComplex dif = gpuCsub(a, b);
  const gpuDoubleComplex prod = gpuCmul(a, b);
  const gpuDoubleComplex quot = gpuCdiv(a, b);
  const gpuDoubleComplex conj = gpuConj(a);

  double acc = gpuCreal(sum) + gpuCimag(dif);
  acc += gpuCreal(prod) + gpuCimag(quot);
  acc += gpuCreal(conj) + gpuCimag(conj);
  acc += gpuCabs(a);
  out[0] = acc;
}

// The Complex-named surface: the make_gpuComplex alias and the two precision
// conversions -- the portable functions the vendors name with the bare Complex
// token. FloatToDouble widens, DoubleToFloat narrows; the round trip exercises
// both, fed by the alias constructor.
__global__ void wwr_fp_complex_convert(double *out) {
  const gpuComplex c = make_gpuComplex(1.5f, -2.5f);
  const gpuDoubleComplex wide = gpuComplexFloatToDouble(c);
  const gpuFloatComplex narrow = gpuComplexDoubleToFloat(wide);

  double acc = gpuCreal(wide) + gpuCimag(wide);
  acc += static_cast<double>(gpuCrealf(narrow) + gpuCimagf(narrow));
  out[0] = acc;
}

} // namespace wwr
