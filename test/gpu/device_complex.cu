// Compile-time test for src/wrappers/complex/device_complex.cuh, the device
// mirror of wwr.wrappers.complex. Like test/gpu/complex.cu and math.cu, this is
// a device TU whose *building* under the selected backend IS the assertion: the
// wwr::complex<T> struct, its operators and both conversion directions only have
// meaning in a device pass (they route through complex.cuh's __device__
// make_gpu*Complex / wwrCreal*), and the claim they can break is that the whole
// surface instantiates for float and double, through one include switch, on both
// backends. The kernels below are never launched.
//
// The static_asserts pin the design invariants the header depends on: complex<T>
// stays an AGGREGATE (so complex<float>{1,2} is aggregate init) despite carrying
// a conversion operator, hidden-friend operators and compound-assignment
// members; it stays trivially copyable and layout-parallel with the vendor type
// (two T), which is what lets a value cross the host/device boundary and convert
// without a reinterpret_cast. Reached through wwr.device, the include path a real
// device consumer uses.
#include "wrappers/complex/device_complex.cuh"

// Defining a __global__ kernel needs the launch runtime; device_complex.cuh
// pulls only the complex headers, so name the runtime here. WWR_SELECTED_* comes
// from device_complex.cuh (via complex.cuh -> device_guard.h).
#if defined(WWR_SELECTED_CUDA)
#include <cuda_runtime.h>
#else
#include <hip/hip_runtime.h>
#endif

#include <type_traits>

namespace wwr {
namespace {

// The core design claims, per precision.
static_assert(std::is_aggregate_v<complex<float>>,
              "complex<float> must stay an aggregate -- conversion operator, "
              "hidden-friend operators and compound-assignment members must not "
              "introduce a user-declared constructor");
static_assert(std::is_aggregate_v<complex<double>>,
              "complex<double> must stay an aggregate");

static_assert(std::is_trivially_copyable_v<complex<float>>,
              "complex<float> must be trivially copyable to cross the boundary");
static_assert(std::is_trivially_copyable_v<complex<double>>,
              "complex<double> must be trivially copyable");

// Layout parity with the vendor type it converts to -- two floats / two doubles.
static_assert(sizeof(complex<float>) == 2 * sizeof(float));
static_assert(sizeof(complex<double>) == 2 * sizeof(double));
static_assert(sizeof(complex<float>) == sizeof(wwrFloatComplex),
              "complex<float> must match wwrFloatComplex in size");
static_assert(sizeof(complex<double>) == sizeof(wwrDoubleComplex),
              "complex<double> must match wwrDoubleComplex in size");

} // namespace

// Single precision: aggregate init, every operator, both scalar orders, the
// compound assignments, equality, and the round trip complex<float> ->
// wwrFloatComplex -> complex<float>.
__global__ void wwr_wrapper_complex_float(wwrFloatComplex *out, int *flag) {
  const complex<float> a{1.0f, 2.0f};
  const complex<float> b{3.0f, -1.0f};

  // Device-context constant evaluation: proves the operators are usable in a
  // constant expression on the device, not merely that constexpr parses.
  // (1+2i)*(3+4i) = -5+10i.
  constexpr complex<float> ce = complex<float>{1.0f, 2.0f} * complex<float>{3.0f, 4.0f};
  static_assert(ce.re == -5.0f && ce.im == 10.0f, "device constexpr complex product");

  complex<float> c = a + b;
  c = a - b;
  c = a * b;
  c = a / b;
  c = -a;

  c = a * 2.0f;
  c = 2.0f * a;
  c = a / 2.0f;
  c = a + 1.0f;
  c = 1.0f + a;
  c = a - 1.0f;
  c = 1.0f - a;

  c += b;
  c -= b;
  c *= b;
  c /= b;
  c += 1.0f;
  c -= 1.0f;
  c *= 2.0f;
  c /= 2.0f;

  const bool eq = (a == b);
  const bool ne = (a != b);

  const wwrFloatComplex v = c;              // implicit conversion to the vendor type
  const complex<float> back = to_complex(v); // and back
  *out = back;                               // implicit conversion again, on store
  *flag = (eq || ne) ? 1 : 0;
}

// Double precision: the same surface over complex<double> / wwrDoubleComplex.
__global__ void wwr_wrapper_complex_double(wwrDoubleComplex *out, int *flag) {
  const complex<double> a{1.0, 2.0};
  const complex<double> b{3.0, -1.0};

  complex<double> c = a + b;
  c = a - b;
  c = a * b;
  c = a / b;
  c = -a;

  c = a * 2.0;
  c = 2.0 * a;
  c = a / 2.0;
  c = a + 1.0;
  c = 1.0 + a;
  c = a - 1.0;
  c = 1.0 - a;

  c += b;
  c -= b;
  c *= b;
  c /= b;
  c += 1.0;
  c -= 1.0;
  c *= 2.0;
  c /= 2.0;

  const bool eq = (a == b);
  const bool ne = (a != b);

  const wwrDoubleComplex v = c;
  const complex<double> back = to_complex(v);
  *out = back;
  *flag = (eq || ne) ? 1 : 0;
}

} // namespace wwr
