/**
 * @file sparse.h
 * @brief The single cuSPARSE / hipSPARSE vendor-include point for the sparse
 *        layer
 *
 * A src/-root header that brings in the vendor sparse header for whichever
 * backend was selected, plus <cstddef> for the std::size_t* the two
 * *_bufferSize shims name, and nothing else: the wwrsparse* surface itself is
 * the shared fragment detail/sparse_names.h, bound to the `::cusparse*` /
 * `::hipsparse*` declarations this header supplies. It is the "rand.h shape"
 * for a host-only vendor library -- sparse.cppm draws its whole surface from
 * here (binding wwr* references straight to the declarations with the _RAW
 * macros, importing no raw vendor module), and wwr/sparse.h (the non-module
 * #include path) does the same, so this is the one place the vendor header
 * enters the sparse layer. The raw module wwr.cuda.cusparse / wwr.hip.hipsparse
 * stays as a single-vendor surface for the modern generic API (SpMV, SpMM, ...)
 * and the single-vendor extras, off this path. See src/sparse.cppm and
 * src/README.md.
 *
 * std::size_t (not the complex types) is the one std-library name the fragment
 * reaches from here: the CUDA *_bufferSize shims declare a std::size_t* out
 * parameter, so <cstddef> here keeps the fragment self-sufficient on both host
 * paths -- wwr/sparse.h cannot import anything, and sparse.cppm's own `import
 * std` is there for a different reason (sealing hipsparse-bfloat16.h's textual
 * <ostream> against a consumer's import std; see sparse.cppm), not for size_t.
 * The complex types the shims name (wwrFloatComplex / wwrDoubleComplex) come the
 * other way, as in blas: import wwr.complex on the module side, #include
 * "complex.h" in wwr/sparse.h.
 *
 * HOST only -- cuSPARSE/hipSPARSE has no device surface, so unlike rand.h /
 * complex.h this header carries no device-pass-gated section. It reaches
 * selected_backend.h directly, not device_guard.h, so it compiles in the host
 * TUs that include it.
 */

#pragma once

// std::size_t for the two *_bufferSize shims' out parameter (see the file
// header). <cstddef> on both host paths, so the fragment needs no import std.
#include <cstddef>

// WWR_SELECTED_CUDA / WWR_SELECTED_HIP, from WWR_GPU_BACKEND_* in a host compile.
#include "selected_backend.h"

#if defined(WWR_SELECTED_CUDA)

#include <cusparse.h>

#else

// Load-bearing, and must stay before the HIP header: host_defines.h (pulled in
// transitively by the hipSPARSE headers) poisons __noinline__ for libc++'s
// __config, so a HIP header reached before <array> makes __config fail to
// compile. Same pre-include src/hip/hipsparse.cppm's global module fragment
// carries. docs/architecture.md, section 9.
#include <array>

#include <hipsparse/hipsparse.h>

#endif
