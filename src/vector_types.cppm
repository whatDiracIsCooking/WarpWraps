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
 * The constructors are not restated here: they are the one fragment every path
 * shares, detail/vector_types_names.h, pasted into this module's purview with the
 * host `inline` qualifier exactly as fp16.cppm pastes detail/fp16_names.h. A host
 * TU reaches them by importing this module, a device TU by including
 * vector_types.h (which pastes the same list with __device__ __forceinline__),
 * and a non-module host TU through wwr/vector_types.h -- one brace over the same
 * vendor types in every case, so a host-built value and a kernel-built one agree.
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
// vector_types.h's device section. The list itself is NOT restated here: it is
// the one fragment every path shares, detail/vector_types_names.h, pasted below
// with the host `inline` qualifier (vector_types.h's device section pastes the
// same list with __device__ __forceinline__; wwr/vector_types.h is the
// non-module host twin). Add a constructor there, once, and every path gains it.
// ========================================================================

#define WWR_VT_FN inline
#include "detail/vector_types_names.h"
#undef WWR_VT_FN

} // namespace wwr
