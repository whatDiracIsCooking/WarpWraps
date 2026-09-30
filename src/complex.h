/**
 * @file complex.h
 * @brief The complex types shared by complex.cuh and complex.cppm
 *
 * wwrFloatComplex / wwrDoubleComplex / wwrComplex, aliased to the vendor's
 * cuComplex / hipComplex for whichever backend was selected. The one definition
 * both sides name: complex.cuh includes it for device code, and complex.cppm
 * includes it in its global module fragment and re-exports the three names, so a
 * host-allocated buffer and a kernel parameter agree on the same vendor type.
 *
 * Only the types live here. The arithmetic and construction wrappers do not: a
 * device TU needs them __device__ __forceinline__ (complex.cuh), a host TU needs
 * them forwarded to the raw module's external-linkage wrappers (complex.cppm) --
 * the vendors' own functions are static inline and cannot be exposed from an
 * exported inline. Only the types are spelled identically on both paths, so only
 * the types are shared. See docs/architecture.md, section 3.
 */

#pragma once

// WWR_SELECTED_CUDA / WWR_SELECTED_HIP, from the compiler's device macro in a
// device pass or from WWR_GPU_BACKEND_* in complex.cppm's host compile.
#include "selected_backend.h"

#if defined(WWR_SELECTED_CUDA)

#include <cuComplex.h>

#else

// Load-bearing, and must stay before the HIP header: host_defines.h (pulled in
// transitively by the HIP vendor headers) poisons __noinline__ for libc++'s
// __config, so a HIP header reached before <array> makes __config fail to
// compile. Same pre-include the src/hip global module fragments carry.
// docs/architecture.md, section 9.
#include <array>

#include <hip/hip_complex.h>

#endif

namespace wwr {

// ========================================================================
// Types -- the same ones complex.cppm exports and complex.cuh exposes
// ========================================================================

#if defined(WWR_SELECTED_CUDA)

using wwrFloatComplex = ::cuFloatComplex;
using wwrDoubleComplex = ::cuDoubleComplex;
using wwrComplex = ::cuComplex;

#else

using wwrFloatComplex = ::hipFloatComplex;
using wwrDoubleComplex = ::hipDoubleComplex;
using wwrComplex = ::hipComplex;

#endif

} // namespace wwr
