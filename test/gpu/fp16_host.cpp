// fp16_host.cpp - the non-module HOST #include path for the fp16 surface
//
// The counterpart to fp16.cppm's module test, through the OTHER path a consumer
// has -- #include "wwr/fp16.h" from a plain, non-module TU -- proving that path
// yields the identical surface bound to the identical backend entities. The
// device conversions in fp16.h's device section are proved separately by fp16.cu;
// the full host surface is pinned by the module test, and this shares
// detail/fp16_names.h with it, so what is under test here is the include PATH.
//
// A plain .cpp, not a .cppm: that is the whole point. It imports nothing and
// links wwr::fp16::host, the header-only target, which carries WWR_GPU_BACKEND_*
// via wwr::backend.

#include <type_traits>

#include "wwr/fp16.h"

#include "gpu_check_macros.h"

using namespace wwr;

// wwrHalf is the backend's own __half (spelled the same on both backends, so no
// #if here -- unlike bf16, whose type diverges).
WWR_SAME_TYPE(wwrHalf, ::__half)

// The conversion wrappers are forwarding inline functions, not WWR_FUNCTION
// reference bindings, so &gpu != &backend and WWR_SAME_FUNCTION cannot apply. A
// bare WWR_LINK_CHECK is the build-time claim: the wrapper is reachable by name
// through the #include path and links.
WWR_LINK_CHECK(wwrFloat2Half)
WWR_LINK_CHECK(wwrHalf2Float)

int main() { return 0; }
