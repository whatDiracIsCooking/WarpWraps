/**
 * @file vector_types.h
 * @brief The vendor vector types (float2, int4, ...), plus the make_* device
 *        constructors, for the vector-types layer
 *
 * float2 / int4 / uchar3 / ... re-exported into namespace wwr under the
 * vendors' OWN names -- the names a host-allocated buffer and a kernel parameter
 * must agree on. It carries two things:
 *   - the type aliases (always), which vector_types.cppm re-exports from its
 *     global module fragment and any TU naming the type includes;
 *   - the make_* constructors as __device__ __forceinline__, in a section gated
 *     behind the device-pass macros -- the device half, so a device .cu includes
 *     this one neutral header. Link wwr.device.
 *
 * No wwr* prefix, and that is the point. Every other module here invents a
 * neutral name (wwrFloatComplex, wwrHalf) because the backends spell the vendor
 * type differently and code above src must not pick a side. These types carry
 * NO such divergence: CUDA and HIP spell every one of them identically (float2
 * is ::float2 on both, int4 is ::int4 on both), so there is nothing to translate
 * and a parallel vocabulary would be pure noise. The layer keeps the vendors'
 * names and only moves them into namespace wwr, so code above src names them the
 * one way everyone already knows -- wwr::float2 IS ::float2. (complex is the
 * contrast: cuFloatComplex and hipFloatComplex genuinely differ, so a neutral
 * wwrFloatComplex is mandatory there.) The make_* constructors keep the vendor
 * spelling too (make_float2, make_int4).
 *
 * What differs underneath is only the category: on CUDA a vector type is a C
 * aggregate (struct float2 { float x, y; }); on HIP it is the class
 * HIP_vector_type<T, N>. Both accept N-argument brace construction (float2{x, y})
 * on host and device, which is the whole reason the make_* wrappers below can be
 * written once with no vendor call.
 *
 * That brace construction is also why this layer, unlike complex, imports and
 * needs NO raw vendor module. A vector type has no portable arithmetic to wrap
 * -- CUDA's aggregates carry no operators where HIP's class does, the same split
 * complex absorbs -- and the vendors expose none for it (there is no cuVadd);
 * the ONLY operation is construction, and construction is the portable brace.
 * The vendors' own make_float2 / ... are static-inline (internal linkage) and so
 * could not be re-exported by a module anyway (docs/architecture.md §12, the
 * rule that forces complex through its raw module); brace construction sidesteps
 * that entirely. So vector_types.cppm imports nothing -- it re-exports these
 * aliases and defines the host make_* as the same brace. (A device TU that
 * includes this header sees both wwr::make_float2 here and the vendor's global
 * ::make_float2; they never clash, because unqualified lookup inside namespace
 * wwr stops at wwr and the scalar arguments carry no ADL.)
 *
 * Scope -- the shared surface, MINUS one cross-backend gap:
 *   - the 12 scalar bases char/uchar/short/ushort/int/uint/long/ulong/longlong/
 *     ulonglong/float/double, in ranks 1-3, for both backends;
 *   - rank 4 only for char/uchar/short/ushort/int/uint/float. The 64-bit-element
 *     4-vectors long4/ulong4/longlong4/ulonglong4/double4 are [[deprecated]] on
 *     CUDA 13 (NVIDIA steers callers to long4_16a / long4_32a, which pin the
 *     alignment the plain name left ambiguous) and those replacements do not
 *     exist on HIP -- so no spelling of a 64-bit 4-vector is portable, and none
 *     is offered. Add them the day a portable spelling exists.
 *
 * dim3 is not here: it is launch geometry (a defaulted constructor, a uint3
 * conversion), not a data-carrying vector, and today only the HIP raw module
 * exposes it (wwr::hip::dim3) -- a neutral dim3 belongs with a launch/runtime
 * wrapper, not this value layer.
 *
 * This is the `.h` + `.cppm` shape for a vendor header with both host and device
 * symbols, the same as fp16.h / complex.h. See docs/architecture.md, section 3,
 * and src/README.md.
 */

#pragma once

// WWR_SELECTED_CUDA / WWR_SELECTED_HIP, from the compiler's device macro in a
// device pass or from WWR_GPU_BACKEND_* in a host compile. Directly, not via
// device_guard.h: this header is host-safe and must not #error.
#include "selected_backend.h"

#if defined(WWR_SELECTED_CUDA)

// NOT <vector_types.h>: that is the CUDA vendor header's OWN name, and this file
// shares it. src/ is on the include path (wwr::backend), so a bare
// `#include <vector_types.h>` here re-finds THIS file -- already `#pragma once`d
// -- and the vendor types never arrive. <vector_functions.h> is the vendor
// header that pairs those types with their make_* constructors; its own
// `#include "vector_types.h"` resolves relative to the toolkit directory, so it
// pulls in the real vector types regardless of include-path order. (The make_*
// it also brings are vendor static-inline and simply go unused -- the ones below
// are a portable brace, not a vendor call.) complex.h / fp16.h never hit this:
// their vendor headers (cuComplex.h, cuda_fp16.h) are named differently from the
// wrapper.
#include <vector_functions.h>

#else

// Load-bearing, and must stay before the HIP header: host_defines.h (pulled in
// transitively by amd_detail/amd_hip_vector_types.h) poisons __noinline__ for
// libc++'s __config, so a HIP header reached before <array> makes __config fail
// to compile. Same pre-include the src/hip global module fragments carry, and
// the one complex.h / fp16.h carry for the same reason. docs/architecture.md,
// section 9.
#include <array>

#include <hip/hip_vector_types.h>

#endif

namespace wwr {

// ========================================================================
// Types -- re-exported into namespace wwr under the vendors' own names, which
// are identical on both backends (::float2 is ::float2 on CUDA and HIP), so no
// backend #if. wwr::float2 IS ::float2.
// ========================================================================

// Signed / unsigned char
using char1 = ::char1;
using char2 = ::char2;
using char3 = ::char3;
using char4 = ::char4;
using uchar1 = ::uchar1;
using uchar2 = ::uchar2;
using uchar3 = ::uchar3;
using uchar4 = ::uchar4;

// Signed / unsigned short
using short1 = ::short1;
using short2 = ::short2;
using short3 = ::short3;
using short4 = ::short4;
using ushort1 = ::ushort1;
using ushort2 = ::ushort2;
using ushort3 = ::ushort3;
using ushort4 = ::ushort4;

// Signed / unsigned int
using int1 = ::int1;
using int2 = ::int2;
using int3 = ::int3;
using int4 = ::int4;
using uint1 = ::uint1;
using uint2 = ::uint2;
using uint3 = ::uint3;
using uint4 = ::uint4;

// Signed / unsigned long -- rank 4 omitted (see file header: deprecated on CUDA 13)
using long1 = ::long1;
using long2 = ::long2;
using long3 = ::long3;
using ulong1 = ::ulong1;
using ulong2 = ::ulong2;
using ulong3 = ::ulong3;

// Signed / unsigned long long -- rank 4 omitted (see file header)
using longlong1 = ::longlong1;
using longlong2 = ::longlong2;
using longlong3 = ::longlong3;
using ulonglong1 = ::ulonglong1;
using ulonglong2 = ::ulonglong2;
using ulonglong3 = ::ulonglong3;

// Single precision
using float1 = ::float1;
using float2 = ::float2;
using float3 = ::float3;
using float4 = ::float4;

// Double precision -- rank 4 omitted (see file header)
using double1 = ::double1;
using double2 = ::double2;
using double3 = ::double3;

} // namespace wwr

// ========================================================================
// Device constructors -- present only in a device-compile pass
//
// make_<type>, the __device__ __forceinline__ counterpart to the host
// constructors in vector_types.cppm, gated behind the compiler's own device-pass
// macros so the type aliases above still compile in a host TU (where __device__
// is not a keyword, so these must be ABSENT rather than #error). A .cu that
// #includes vector_types.h gets them; vector_types.cppm, compiled as host C++,
// does not -- its global module fragment includes this header with the section
// gated out, which is also what keeps these from colliding with the module's own
// host definitions. Link wwr.device.
//
// Each is one brace construction, T{...}, the portable spelling on both backends
// (a CUDA aggregate, a HIP HIP_vector_type constructor) -- see this file's
// header for why no vendor make_* is called. The local macros expand to one
// constructor apiece and are #undef'd at the end of the section; the element
// type E matches the vendor's own make_* parameter type, so no brace narrows.
// ========================================================================

#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)

namespace wwr {

#define WWR_VT_MAKE1(T, E)                                                                         \
  __device__ __forceinline__ T make_##T(const E x) { return T{x}; }
#define WWR_VT_MAKE2(T, E)                                                                         \
  __device__ __forceinline__ T make_##T(const E x, const E y) { return T{x, y}; }
#define WWR_VT_MAKE3(T, E)                                                                         \
  __device__ __forceinline__ T make_##T(const E x, const E y, const E z) { return T{x, y, z}; }
#define WWR_VT_MAKE4(T, E)                                                                         \
  __device__ __forceinline__ T make_##T(const E x, const E y, const E z, const E w) {              \
    return T{x, y, z, w};                                                                          \
  }

WWR_VT_MAKE1(char1, signed char)
WWR_VT_MAKE2(char2, signed char)
WWR_VT_MAKE3(char3, signed char)
WWR_VT_MAKE4(char4, signed char)
WWR_VT_MAKE1(uchar1, unsigned char)
WWR_VT_MAKE2(uchar2, unsigned char)
WWR_VT_MAKE3(uchar3, unsigned char)
WWR_VT_MAKE4(uchar4, unsigned char)

WWR_VT_MAKE1(short1, short)
WWR_VT_MAKE2(short2, short)
WWR_VT_MAKE3(short3, short)
WWR_VT_MAKE4(short4, short)
WWR_VT_MAKE1(ushort1, unsigned short)
WWR_VT_MAKE2(ushort2, unsigned short)
WWR_VT_MAKE3(ushort3, unsigned short)
WWR_VT_MAKE4(ushort4, unsigned short)

WWR_VT_MAKE1(int1, int)
WWR_VT_MAKE2(int2, int)
WWR_VT_MAKE3(int3, int)
WWR_VT_MAKE4(int4, int)
WWR_VT_MAKE1(uint1, unsigned int)
WWR_VT_MAKE2(uint2, unsigned int)
WWR_VT_MAKE3(uint3, unsigned int)
WWR_VT_MAKE4(uint4, unsigned int)

WWR_VT_MAKE1(long1, long int)
WWR_VT_MAKE2(long2, long int)
WWR_VT_MAKE3(long3, long int)
WWR_VT_MAKE1(ulong1, unsigned long int)
WWR_VT_MAKE2(ulong2, unsigned long int)
WWR_VT_MAKE3(ulong3, unsigned long int)

WWR_VT_MAKE1(longlong1, long long int)
WWR_VT_MAKE2(longlong2, long long int)
WWR_VT_MAKE3(longlong3, long long int)
WWR_VT_MAKE1(ulonglong1, unsigned long long int)
WWR_VT_MAKE2(ulonglong2, unsigned long long int)
WWR_VT_MAKE3(ulonglong3, unsigned long long int)

WWR_VT_MAKE1(float1, float)
WWR_VT_MAKE2(float2, float)
WWR_VT_MAKE3(float3, float)
WWR_VT_MAKE4(float4, float)

WWR_VT_MAKE1(double1, double)
WWR_VT_MAKE2(double2, double)
WWR_VT_MAKE3(double3, double)

#undef WWR_VT_MAKE1
#undef WWR_VT_MAKE2
#undef WWR_VT_MAKE3
#undef WWR_VT_MAKE4

} // namespace wwr

#endif // device-compile pass
