/**
 * @file wwr/blas.h
 * @brief The backend-neutral cuBLAS / hipBLAS surface as a single,
 *        self-contained HOST #include, for a TU that does not use C++ modules
 *
 * The non-module counterpart to `import wwr.blas;`: a plain .cpp that #includes
 * this one header gets the same wwrblas* / WWRBLAS_* surface the module exports --
 * the handle/status/op types, the status / op / fill / diag / side / pointer-mode
 * constants, the level 1/2/3 and extension functions, and the const-correct
 * getrs/getri batched forwarders -- bound to the selected backend's ::cublas*_v2 /
 * ::hipblas* entry points.
 *
 * The wwrblas* name list is NOT restated here: it is the one fragment the host
 * paths share, detail/blas_names.h, pasted below inside `namespace wwr` exactly as
 * blas.cppm's purview pastes it. Add a name there, once, and this path gains it.
 * What this file supplies before that #include is what the fragment documents it
 * needs: the vendor header (via blas.h, the layer's one vendor-include point), the
 * complex types the getri/getrs forwarders name (via complex.h), and the
 * reference-alias binding macros (the module gets those from backend.h; here they
 * are defined locally and #undef'd at the end so none leaks into the including TU).
 *
 * Backend selection and the include root arrive by linking wwr::blas::host. HOST
 * only -- cuBLAS/hipBLAS has no device surface. See src/blas.cppm, src/blas.h and
 * docs/architecture.md, section 5.
 */

#pragma once

// HOST only -- see the file header. cuBLAS/hipBLAS has no device surface, but the
// guard keeps this path consistent with the other wwr/*.h headers.
#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)
#error                                                                                              \
    "wwr/blas.h is the HOST #include path for the blas surface. See src/README.md."
#endif

// The single vendor-include point for the blas layer (the vendor header the
// bindings below resolve against) and selected_backend.h's WWR_SELECTED_*. Angle
// brackets, not quotes: this file is itself src/wwr/blas.h, so a quoted "blas.h"
// would resolve to this file (quotes search the current directory first) and
// #pragma-once to nothing; <blas.h> skips that search and resolves to src/blas.h
// on the include root.
#include <blas.h>

// The complex types the getri/getrs batched forwarders name. The module reaches
// these by import wwr.complex; a non-module TU reaches the same wwr* aliases from
// complex.h (the src/-root shared-type header), reached root-relative with no
// self-collision (there is no src/wwr/complex.h).
#include "complex.h"

// ---------------------------------------------------------------------------
// Binding macros, reference-alias form -- the same shape backend.h gives the
// module, defined locally here because a non-module TU imports nothing. Keyed on
// WWR_SELECTED_*, set by selected_backend.h (via blas.h). #undef'd at the end so
// none leaks into the including TU.
// ---------------------------------------------------------------------------
#if defined(WWR_SELECTED_CUDA)
#define WWR_SELECT_RAW(cuda_name, hip_name) ::cuda_name
#else
#define WWR_SELECT_RAW(cuda_name, hip_name) ::hip_name
#endif
#define WWR_TYPE_RAW(wwr_name, cuda_name, hip_name)                                                 \
  using wwr_name = WWR_SELECT_RAW(cuda_name, hip_name);
#define WWR_VALUE_RAW(wwr_name, cuda_name, hip_name)                                                \
  inline constexpr auto wwr_name = WWR_SELECT_RAW(cuda_name, hip_name);
#define WWR_FUNCTION_RAW(wwr_name, cuda_name, hip_name)                                             \
  inline constexpr auto &wwr_name = WWR_SELECT_RAW(cuda_name, hip_name);

// ---------------------------------------------------------------------------
// The wwrblas* surface, pasted from the one fragment the host paths share. The
// macros above plus the vendor header, wwrFloatComplex / wwrDoubleComplex and
// WWR_SELECTED_* are exactly what detail/blas_names.h documents it needs in scope.
// ---------------------------------------------------------------------------
namespace wwr {
#include "detail/blas_names.h"
} // namespace wwr

#undef WWR_SELECT_RAW
#undef WWR_TYPE_RAW
#undef WWR_VALUE_RAW
#undef WWR_FUNCTION_RAW
