// vector_types_host.cpp - the non-module HOST #include path for the vector_types
// surface
//
// The counterpart to vector_types.cppm's module test, through the OTHER path a
// consumer has -- #include "wwr/vector_types.h" from a plain, non-module TU --
// proving that path yields the identical surface (the float2 / int4 / ... aliases
// and the host make_* constructors) bound to the identical backend entities. The
// device constructors in vector_types.h's device section are proved separately by
// vector_types.cu; the full host surface is pinned by the module test, and this
// shares detail/vector_types_names.h with it, so what is under test here is the
// include PATH.
//
// A plain .cpp, not a .cppm: that is the whole point. It imports nothing and
// links wwr::vector_types::host, the header-only target, which carries
// WWR_GPU_BACKEND_* via wwr::backend.

#include <type_traits>

#include "wwr/vector_types.h"

#include "gpu_check_macros.h"

// Inside namespace wwr (as the module test uses wwr::test): an unqualified
// make_ call resolves to wwr::make_ in this enclosing namespace and lookup stops
// there -- it never reaches the vendor's identically-named global ::make_, so
// WWR_LINK_CHECK's &make_ is unambiguous. At global scope a `using namespace wwr`
// would make both visible at once and &make_ would be an overloaded-function
// type (unlike fp16's wwrFloat2Half, which has no vendor-global twin).
namespace wwr::test {

using namespace wwr;

// Each wwr alias is exactly the vendor's identically-spelled global type (one
// spelling on both backends, so no #if), and each make_ constructor is reachable
// by name through the #include path. The constructors are brace builders, not
// WWR_FUNCTION reference bindings, so &gpu != &backend and WWR_SAME_FUNCTION
// cannot apply -- a bare WWR_LINK_CHECK is the build-time claim. This mirrors the
// module test's WWR_VT_CHECK, line for line, so the two paths are pinned alike.
#define WWR_VT_CHECK(T)                                                                             \
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

int main() { return 0; }
