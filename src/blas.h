/**
 * @file blas.h
 * @brief The single cuBLAS / hipBLAS vendor-include point for the blas layer
 *
 * A src/-root header that brings in the vendor BLAS header for whichever backend
 * was selected, and nothing else: the wwrblas* surface itself is the shared
 * fragment detail/blas_names.h, bound to the `::cublas*_v2` / `::hipblas*`
 * declarations this header supplies. It is the "rand.h shape" for a host-only
 * vendor library -- blas.cppm draws its whole surface from here (binding wwr*
 * references straight to the declarations with the _RAW macros, importing no raw
 * vendor module), and wwr/blas.h (the non-module #include path) does the same, so
 * this is the one place the vendor header enters the blas layer. The raw module
 * wwr.cuda.cublas_v2 / wwr.hip.hipblas stays as a single-vendor surface for the
 * cuBLAS-only extras, off this path. See src/blas.cppm and src/README.md.
 *
 * HOST only -- cuBLAS/hipBLAS has no device surface, so unlike rand.h / complex.h
 * this header carries no device-pass-gated section. It reaches selected_backend.h
 * directly, not device_guard.h, so it compiles in the host TUs that include it.
 */

#pragma once

// WWR_SELECTED_CUDA / WWR_SELECTED_HIP, from WWR_GPU_BACKEND_* in a host compile.
#include "selected_backend.h"

#if defined(WWR_SELECTED_CUDA)

#include <cublas_v2.h>
#include <cuda_runtime.h>

#else

// Load-bearing, and must stay before the HIP header: host_defines.h (pulled in
// transitively by the hipBLAS headers) poisons __noinline__ for libc++'s
// __config, so a HIP header reached before <array> makes __config fail to
// compile. Same pre-include src/hip/hipblas.cppm's global module fragment carries.
// docs/architecture.md, section 9.
#include <array>

#include <hipblas/hipblas.h>

#endif
