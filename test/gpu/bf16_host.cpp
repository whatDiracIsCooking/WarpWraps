// bf16_host.cpp - the non-module HOST #include path for the bf16 surface
//
// The fp16_host.cpp twin, one layer over the bf16 surface: #include "wwr/bf16.h"
// from a plain, non-module TU, proving that path yields the identical surface
// bound to the identical backend entities. The device conversions in bf16.h's
// device section are proved separately by bf16.cu; this shares detail/bf16_names.h
// with the module test, so what is under test here is the include PATH.
//
// A plain .cpp that imports nothing and links wwr::bf16::host, the header-only
// target, which carries WWR_GPU_BACKEND_* via wwr::backend.

#include <type_traits>

#include "wwr/bf16.h"

#include "gpu_check_macros.h"

using namespace wwr;

// wwrBfloat16 is the backend's own bfloat16 -- the type DIVERGES by backend
// (__nv_bfloat16 vs __hip_bfloat16), so it is asserted per backend.
#if defined(WWR_GPU_BACKEND_CUDA)
WWR_SAME_TYPE(wwrBfloat16, ::__nv_bfloat16)
#else
WWR_SAME_TYPE(wwrBfloat16, ::__hip_bfloat16)
#endif

// Forwarding inline functions (see fp16_host.cpp), so WWR_LINK_CHECK, not
// WWR_SAME_FUNCTION.
WWR_LINK_CHECK(wwrFloat2Bfloat16)
WWR_LINK_CHECK(wwrBfloat162Float)

int main() { return 0; }
