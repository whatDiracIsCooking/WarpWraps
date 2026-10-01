// Compile-time test for src/vector_types.h's device-pass-gated section, the
// device-compile vector-types layer. Unlike the .cppm wwr* test beside it, that
// section defines __device__ make_* constructors that only a device pass can
// name, so -- like complex.cu / fp16.cu -- the test is a device TU and building
// it under the selected backend IS the assertion. The kernel is never launched:
// every name below just has to compile through the one include switch on both
// backends (nvcc and clang's -x hip disagree on more than the include path).
// Reached through wwr.device, exactly as a real device consumer reaches the
// header.
#include "vector_types.h"

// Defining a __global__ kernel needs the launch runtime (hipLaunchKernel under
// HIP). vector_types.h pulls only the vector-type header, not the runtime, so
// this kernel TU names it itself rather than lean on a transitive include. nvcc
// supplies <cuda_runtime.h> for a .cu implicitly; spell both for a
// self-contained TU. WWR_SELECTED_* comes from vector_types.h (via
// selected_backend.h).
#if defined(WWR_SELECTED_CUDA)
#include <cuda_runtime.h>
#else
#include <hip/hip_runtime.h>
#endif

#include <type_traits>

// Inside namespace wwr: an unqualified type name (char1, ...) resolves to the
// wwr:: alias, and an unqualified make_ call resolves to wwr::make_ and stops
// there -- never reaching the vendor's global ::make_ (scalar args carry no
// ADL), so there is nothing to disambiguate.
namespace wwr {
namespace {

// ---------------------------------------------------------------------------
// Layout parity, for every aliased type: a host-allocated buffer and a kernel
// parameter named here agree on type, so an N-vector must be exactly its N
// elements with no padding (true on both backends, where the alignment the
// vendors stamp on -- 8 for float2, 16 for float4 -- is already a multiple of
// N*sizeof(element)), and must stay trivially copyable so a value crosses the
// host/device boundary as raw bytes. A vendor changing either breaks that
// agreement silently. The macro pins both for one type; the element type E
// matches the vendor's own.
// ---------------------------------------------------------------------------
#define WWR_VT_LAYOUT(T, E, N)                                                                     \
  static_assert(sizeof(T) == (N) * sizeof(E), #T " is expected to be " #N " x " #E);               \
  static_assert(std::is_trivially_copyable_v<T>, #T " is expected to be trivially copyable");

WWR_VT_LAYOUT(char1, signed char, 1)
WWR_VT_LAYOUT(char2, signed char, 2)
WWR_VT_LAYOUT(char3, signed char, 3)
WWR_VT_LAYOUT(char4, signed char, 4)
WWR_VT_LAYOUT(uchar1, unsigned char, 1)
WWR_VT_LAYOUT(uchar2, unsigned char, 2)
WWR_VT_LAYOUT(uchar3, unsigned char, 3)
WWR_VT_LAYOUT(uchar4, unsigned char, 4)

WWR_VT_LAYOUT(short1, short, 1)
WWR_VT_LAYOUT(short2, short, 2)
WWR_VT_LAYOUT(short3, short, 3)
WWR_VT_LAYOUT(short4, short, 4)
WWR_VT_LAYOUT(ushort1, unsigned short, 1)
WWR_VT_LAYOUT(ushort2, unsigned short, 2)
WWR_VT_LAYOUT(ushort3, unsigned short, 3)
WWR_VT_LAYOUT(ushort4, unsigned short, 4)

WWR_VT_LAYOUT(int1, int, 1)
WWR_VT_LAYOUT(int2, int, 2)
WWR_VT_LAYOUT(int3, int, 3)
WWR_VT_LAYOUT(int4, int, 4)
WWR_VT_LAYOUT(uint1, unsigned int, 1)
WWR_VT_LAYOUT(uint2, unsigned int, 2)
WWR_VT_LAYOUT(uint3, unsigned int, 3)
WWR_VT_LAYOUT(uint4, unsigned int, 4)

WWR_VT_LAYOUT(long1, long int, 1)
WWR_VT_LAYOUT(long2, long int, 2)
WWR_VT_LAYOUT(long3, long int, 3)
WWR_VT_LAYOUT(ulong1, unsigned long int, 1)
WWR_VT_LAYOUT(ulong2, unsigned long int, 2)
WWR_VT_LAYOUT(ulong3, unsigned long int, 3)

WWR_VT_LAYOUT(longlong1, long long int, 1)
WWR_VT_LAYOUT(longlong2, long long int, 2)
WWR_VT_LAYOUT(longlong3, long long int, 3)
WWR_VT_LAYOUT(ulonglong1, unsigned long long int, 1)
WWR_VT_LAYOUT(ulonglong2, unsigned long long int, 2)
WWR_VT_LAYOUT(ulonglong3, unsigned long long int, 3)

WWR_VT_LAYOUT(float1, float, 1)
WWR_VT_LAYOUT(float2, float, 2)
WWR_VT_LAYOUT(float3, float, 3)
WWR_VT_LAYOUT(float4, float, 4)

WWR_VT_LAYOUT(double1, double, 1)
WWR_VT_LAYOUT(double2, double, 2)
WWR_VT_LAYOUT(double3, double, 3)

#undef WWR_VT_LAYOUT

} // namespace

// The device make_ constructors, one per macro arm (MAKE1..4) and across every
// scalar base, plus component read-back (.x/.y/.z/.w, spelled identically on a
// CUDA aggregate and a HIP HIP_vector_type). Building this is the claim that the
// whole device section resolves through the one include switch; the values are
// accumulated only so nothing is optimised away before the store.
__global__ void wwr_vector_types_make(float *out) {
  float acc = 0.0f;

  // Rank 1 -- the MAKE1 arm, over several bases.
  acc += static_cast<float>(make_char1(1).x);
  acc += static_cast<float>(make_ushort1(2).x);
  acc += static_cast<float>(make_ulong1(3).x);
  acc += static_cast<float>(make_ulonglong1(4).x);
  acc += make_float1(5.0f).x;
  acc += static_cast<float>(make_double1(6.0).x);

  // Rank 2 -- the MAKE2 arm.
  const int2 i2 = make_int2(1, 2);
  acc += static_cast<float>(i2.x + i2.y);
  const uchar2 uc2 = make_uchar2(3, 4);
  acc += static_cast<float>(uc2.x + uc2.y);
  const long2 l2 = make_long2(5, 6);
  acc += static_cast<float>(l2.x + l2.y);
  const longlong2 ll2 = make_longlong2(7, 8);
  acc += static_cast<float>(ll2.x + ll2.y);
  const float2 f2 = make_float2(1.0f, 2.0f);
  acc += f2.x + f2.y;
  const double2 d2 = make_double2(3.0, 4.0);
  acc += static_cast<float>(d2.x + d2.y);

  // Rank 3 -- the MAKE3 arm.
  const short3 s3 = make_short3(1, 2, 3);
  acc += static_cast<float>(s3.x + s3.y + s3.z);
  const uint3 u3 = make_uint3(4, 5, 6);
  acc += static_cast<float>(u3.x + u3.y + u3.z);
  const ulong3 ul3 = make_ulong3(7, 8, 9);
  acc += static_cast<float>(ul3.x + ul3.y + ul3.z);
  const float3 f3 = make_float3(1.0f, 2.0f, 3.0f);
  acc += f3.x + f3.y + f3.z;
  const double3 d3 = make_double3(4.0, 5.0, 6.0);
  acc += static_cast<float>(d3.x + d3.y + d3.z);

  // Rank 4 -- the MAKE4 arm (only the non-deprecated bases; see
  // src/vector_types.h for why the 64-bit 4-vectors are absent).
  const char4 c4 = make_char4(1, 2, 3, 4);
  acc += static_cast<float>(c4.x + c4.y + c4.z + c4.w);
  const ushort4 us4 = make_ushort4(5, 6, 7, 8);
  acc += static_cast<float>(us4.x + us4.y + us4.z + us4.w);
  const int4 i4 = make_int4(1, 2, 3, 4);
  acc += static_cast<float>(i4.x + i4.y + i4.z + i4.w);
  const uint4 u4 = make_uint4(5, 6, 7, 8);
  acc += static_cast<float>(u4.x + u4.y + u4.z + u4.w);
  const float4 f4 = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
  acc += f4.x + f4.y + f4.z + f4.w;

  out[0] = acc;
}

} // namespace wwr
