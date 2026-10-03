/**
 * @file blaslt.h
 * @brief The single cuBLASLt / hipBLASLt vendor-include point for the blaslt
 *        layer
 *
 * A src/-root header that brings in the vendor "Lt" GEMM header for whichever
 * backend was selected, and nothing else: the wwrblasLt* surface itself is the
 * shared fragment detail/blaslt_names.h, bound to the `::cublasLt*` /
 * `::hipblasLt*` declarations this header supplies. It is the "rand.h shape" for
 * a host-only vendor library -- blaslt.cppm draws its whole surface from here
 * (binding wwr* references straight to the declarations with the _RAW macros,
 * importing no raw vendor module), and wwr/blaslt.h (the non-module #include
 * path) does the same, so this is the one place the vendor header enters the
 * blaslt layer. The raw module wwr.cuda.cublasLt / wwr.hip.hipblaslt stays as a
 * single-vendor surface for the much larger private API (algo introspection,
 * logger, tile/stages enums and the *_EXT attributes), off this path. See
 * src/blaslt.cppm and src/README.md.
 *
 * HOST only -- cuBLASLt/hipBLASLt has no device surface, so unlike rand.h /
 * complex.h this header carries no device-pass-gated section. It reaches
 * selected_backend.h directly, not device_guard.h, so it compiles in the host
 * TUs that include it.
 */

#pragma once

// WWR_SELECTED_CUDA / WWR_SELECTED_HIP, from WWR_GPU_BACKEND_* in a host compile.
#include "selected_backend.h"

#if defined(WWR_SELECTED_CUDA)

// cublasLt.h pulls in cublas_api.h, which carries the cublasStatus_t /
// cublasComputeType_t declarations the two inherited-type renames resolve
// against.
#include <cublasLt.h>

#else

// Load-bearing, and must stay before the HIP header: host_defines.h (pulled in
// transitively by the hipBLASLt headers) poisons __noinline__ for libc++'s
// __config, so a HIP header reached before <array> makes __config fail to
// compile. Same pre-include src/hip/hipblaslt.cppm's global module fragment
// carries. docs/architecture.md, section 9.
#include <array>

#include <hipblaslt/hipblaslt.h>

#endif
