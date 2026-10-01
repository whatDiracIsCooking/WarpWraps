/**
 * @file vector_types.cppm
 * @brief Backend-neutral vector types and constructors: the CUDA / HIP vector
 *        types (float2, int4, ...) re-exported into namespace wwr
 *
 * The host-module counterpart to vector_types.h's device-pass-gated section:
 * the float2 / int4 / ... aliases and the make_* constructors, for whichever
 * backend this build is configured for. Companion to fp16.cppm and
 * complex.cppm. See vector_types.h for the full rationale -- why there is no
 * wwr* prefix (the backends spell these types identically, so the layer keeps
 * the vendors' own names and only moves them into namespace wwr), the scope (the
 * shared surface minus the CUDA-13-deprecated 64-bit 4-vectors), and why dim3 is
 * not here.
 *
 * Usage:
 *   import wwr.vector_types;
 *
 *   wwr::float2 v = wwr::make_float2(1.0f, 2.0f);
 *
 * Like fp16.cppm and unlike complex.cppm, this module imports no raw vendor
 * module. A vector type has no portable arithmetic to wrap, so the only
 * operation is construction, and construction is a portable brace -- float2{x, y}
 * is a CUDA aggregate init and a HIP HIP_vector_type constructor on either
 * backend. The make_* wrappers below are that one brace, so they name no vendor
 * symbol and need nothing imported (the vendors' own static-inline make_float2 /
 * ... could not be re-exported by a module regardless -- docs/architecture.md
 * §12). The types come from vector_types.h, in the GMF.
 *
 * The constructors duplicate vector_types.h's device section on purpose, the
 * same way fp16.cppm duplicates fp16.h's: a host TU reaches them by importing
 * this module, a device TU reaches them by including vector_types.h, and neither
 * can use the other's (a module cannot be #included into a kernel, and the
 * header's wrappers are gated to a device pass). The two copies are the same
 * brace over the same vendor types, so a host-built value and a kernel-built one
 * agree.
 */

module;

#include "backend.h"

// The vector-type aliases (re-exported below). Compiled as host C++,
// vector_types.h's device-pass-gated make_* section is absent; a device TU gets
// those constructors instead, and its absence here is also what lets this module
// define the host constructors below without colliding with the header.
#include "vector_types.h"

export module wwr.vector_types;

export namespace wwr {

// ========================================================================
// Types -- re-exported from vector_types.h (the global-module aliases the GMF
// #include brought in), so importers of wwr.vector_types see them. The vendor
// names are identical on both backends, so there is nothing to translate; the
// aliases live in vector_types.h.
// ========================================================================

using wwr::char1;
using wwr::char2;
using wwr::char3;
using wwr::char4;
using wwr::uchar1;
using wwr::uchar2;
using wwr::uchar3;
using wwr::uchar4;

using wwr::short1;
using wwr::short2;
using wwr::short3;
using wwr::short4;
using wwr::ushort1;
using wwr::ushort2;
using wwr::ushort3;
using wwr::ushort4;

using wwr::int1;
using wwr::int2;
using wwr::int3;
using wwr::int4;
using wwr::uint1;
using wwr::uint2;
using wwr::uint3;
using wwr::uint4;

using wwr::long1;
using wwr::long2;
using wwr::long3;
using wwr::ulong1;
using wwr::ulong2;
using wwr::ulong3;

using wwr::longlong1;
using wwr::longlong2;
using wwr::longlong3;
using wwr::ulonglong1;
using wwr::ulonglong2;
using wwr::ulonglong3;

using wwr::float1;
using wwr::float2;
using wwr::float3;
using wwr::float4;

using wwr::double1;
using wwr::double2;
using wwr::double3;

// ========================================================================
// Constructors
//
// One brace construction apiece, T{...} -- the host counterpart to
// vector_types.h's device section (see this file's header for the deliberate
// duplication). The local macros expand to one constructor each and are
// #undef'd below; E matches the vendor's own make_* parameter type so no brace
// narrows.
// ========================================================================

#define WWR_VT_MAKE1(T, E)                                                                         \
  inline T make_##T(const E x) { return T{x}; }
#define WWR_VT_MAKE2(T, E)                                                                         \
  inline T make_##T(const E x, const E y) { return T{x, y}; }
#define WWR_VT_MAKE3(T, E)                                                                         \
  inline T make_##T(const E x, const E y, const E z) { return T{x, y, z}; }
#define WWR_VT_MAKE4(T, E)                                                                         \
  inline T make_##T(const E x, const E y, const E z, const E w) { return T{x, y, z, w}; }

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
