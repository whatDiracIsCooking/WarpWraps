// vector_types.cppm - Compile-time tests for wwr.vector_types
//
// Each name in namespace wwr is the backend's own identically-spelled vector
// type, and each exported make_ constructor links across the import. See
// gpu_check_macros.h.
//
// Unlike the other wwr* tests here, this one imports no raw vendor module:
// wwr.vector_types wraps no module (there is none for the vendor vector types),
// so the backend entity each alias must equal is the vendor's own global type,
// which the vendor header in the global module fragment supplies. The device
// make_ constructors are proved separately by vector_types.cu.

module;

#include "gpu_check_macros.h"

// The vendor global vector types (::float2, ::int4, ...), to pin each wwr alias
// against. On CUDA this is <vector_functions.h>, NOT <vector_types.h>: src/ is
// on this test's include path (wwr::backend, via the module it links), and the
// vendor header shares src/vector_types.h's name -- the same shadow
// src/vector_types.h documents and sidesteps the same way. HIP's header name
// does not collide.
#if defined(WWR_GPU_BACKEND_CUDA)
#include <vector_functions.h>
#else
// <array> before the HIP header, for the libc++ __config / __noinline__ reason
// src/vector_types.h carries.
#include <array>
#include <hip/hip_vector_types.h>
#endif

export module wwr.test.gpu.vector_types;

import std;
import wwr.vector_types;

namespace wwr::test {

using namespace wwr;

// Each wwr alias is exactly the vendor's identically-spelled global type -- one
// spelling on both backends, so the wwr name and the expected backend name are
// the same token. The macro pins the alias and link-checks its constructor in
// one line; the constructors are brace builders, not WWR_FUNCTION reference
// bindings, so &gpu != &backend and WWR_SAME_FUNCTION cannot apply -- a bare
// WWR_LINK_CHECK is the build-time claim that each exported wrapper is reachable
// by name across the import (unqualified lookup here stops at wwr, so it finds
// wwr::make_, not the vendor's global ::make_).
#define WWR_VT_CHECK(T)                                                                            \
  WWR_SAME_TYPE(wwr::T, ::T)                                                                        \
  WWR_LINK_CHECK(make_##T)

WWR_VT_CHECK(char1)
WWR_VT_CHECK(char2)
WWR_VT_CHECK(char3)
WWR_VT_CHECK(char4)
WWR_VT_CHECK(uchar1)
WWR_VT_CHECK(uchar2)
WWR_VT_CHECK(uchar3)
WWR_VT_CHECK(uchar4)

WWR_VT_CHECK(short1)
WWR_VT_CHECK(short2)
WWR_VT_CHECK(short3)
WWR_VT_CHECK(short4)
WWR_VT_CHECK(ushort1)
WWR_VT_CHECK(ushort2)
WWR_VT_CHECK(ushort3)
WWR_VT_CHECK(ushort4)

WWR_VT_CHECK(int1)
WWR_VT_CHECK(int2)
WWR_VT_CHECK(int3)
WWR_VT_CHECK(int4)
WWR_VT_CHECK(uint1)
WWR_VT_CHECK(uint2)
WWR_VT_CHECK(uint3)
WWR_VT_CHECK(uint4)

WWR_VT_CHECK(long1)
WWR_VT_CHECK(long2)
WWR_VT_CHECK(long3)
WWR_VT_CHECK(ulong1)
WWR_VT_CHECK(ulong2)
WWR_VT_CHECK(ulong3)

WWR_VT_CHECK(longlong1)
WWR_VT_CHECK(longlong2)
WWR_VT_CHECK(longlong3)
WWR_VT_CHECK(ulonglong1)
WWR_VT_CHECK(ulonglong2)
WWR_VT_CHECK(ulonglong3)

WWR_VT_CHECK(float1)
WWR_VT_CHECK(float2)
WWR_VT_CHECK(float3)
WWR_VT_CHECK(float4)

WWR_VT_CHECK(double1)
WWR_VT_CHECK(double2)
WWR_VT_CHECK(double3)

#undef WWR_VT_CHECK

} // namespace wwr::test
