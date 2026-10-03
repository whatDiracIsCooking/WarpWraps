/**
 * @file solver.h
 * @brief The single cuSOLVER Dense / hipSOLVER vendor-include point for the
 *        solver layer
 *
 * A src/-root header that brings in the vendor dense-solver header for whichever
 * backend was selected, plus library_types.h for the cudaDataType / hipDataType
 * enumerators the data-type constants name, and nothing else: the wwrsolver*
 * surface itself is the shared fragment detail/solver_names.h, bound to the
 * `::cusolverDn*` / `::hipsolverDn*` declarations this header supplies. It is the
 * "rand.h shape" for a host-only vendor library -- solver.cppm draws its whole
 * surface from here (binding wwr* references straight to the declarations with
 * the _RAW macros, importing no raw vendor module), and wwr/solver.h (the
 * non-module #include path) does the same, so this is the one place the vendor
 * header enters the solver layer. The raw module wwr.cuda.cusolverDn /
 * wwr.hip.hipsolver stays as a single-vendor surface for the CUDA-only modern
 * eigen/SVD extras, off this path. See src/solver.cppm and src/README.md.
 *
 * HOST only -- cuSOLVER/hipSOLVER has no device surface, so unlike rand.h /
 * complex.h this header carries no device-pass-gated section. It reaches
 * selected_backend.h directly, not device_guard.h, so it compiles in the host
 * TUs that include it.
 */

#pragma once

// WWR_SELECTED_CUDA / WWR_SELECTED_HIP, from WWR_GPU_BACKEND_* in a host compile.
#include "selected_backend.h"

#if defined(WWR_SELECTED_CUDA)

#include <cusolverDn.h>
#include <library_types.h>

#else

// Load-bearing, and must stay before the HIP headers: host_defines.h (pulled in
// transitively by the hipSOLVER headers) poisons __noinline__ for libc++'s
// __config, so a HIP header reached before <array> makes __config fail to
// compile. Same pre-include src/hip/hipsolver.cppm's global module fragment
// carries. docs/architecture.md, section 9.
#include <array>

#include <hip/library_types.h>
#include <hipsolver/hipsolver.h>

#endif
