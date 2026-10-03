/**
 * @file wwr/solver.h
 * @brief The backend-neutral cuSOLVER Dense / hipSOLVER surface as a single,
 *        self-contained HOST #include, for a TU that does not use C++ modules
 *
 * The non-module counterpart to `import wwr.solver;`: a plain .cpp that #includes
 * this one header gets the same wwrsolver* / WWRSOLVER_* surface the module
 * exports -- the handle/params/info/status/eig types, the status / eig-mode /
 * eig-range / eig-type / data-type constants, the two hand-written
 * status-to-string switches, and the legacy and modern dense-solver functions --
 * bound to the selected backend's ::cusolverDn* / ::hipsolverDn* entry points.
 *
 * The wwrsolver* name list is NOT restated here: it is the one fragment the host
 * paths share, detail/solver_names.h, pasted below inside `namespace wwr` exactly
 * as solver.cppm's purview pastes it. Add a name there, once, and this path gains
 * it. What this file supplies before that #include is what the fragment documents
 * it needs: the vendor header and library_types.h (via solver.h, the layer's one
 * vendor-include point) and the reference-alias binding macros (the module gets
 * those from backend.h; here they are defined locally and #undef'd at the end so
 * none leaks into the including TU). Unlike wwr/blas.h this path pulls in no
 * complex.h: the fragment names no complex type.
 *
 * Backend selection and the include root arrive by linking wwr::solver::host.
 * HOST only -- cuSOLVER/hipSOLVER has no device surface. See src/solver.cppm,
 * src/solver.h and docs/architecture.md, section 5.
 */

#pragma once

// HOST only -- see the file header. cuSOLVER/hipSOLVER has no device surface, but
// the guard keeps this path consistent with the other wwr/*.h headers.
#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)
#error                                                                                              \
    "wwr/solver.h is the HOST #include path for the solver surface. See src/README.md."
#endif

// The single vendor-include point for the solver layer (the vendor header the
// bindings below resolve against) and selected_backend.h's WWR_SELECTED_*. Angle
// brackets, not quotes: this file is itself src/wwr/solver.h, so a quoted
// "solver.h" would resolve to this file (quotes search the current directory
// first) and #pragma-once to nothing; <solver.h> skips that search and resolves
// to src/solver.h on the include root.
#include <solver.h>

// ---------------------------------------------------------------------------
// Binding macros, reference-alias form -- the same shape backend.h gives the
// module, defined locally here because a non-module TU imports nothing. Keyed on
// WWR_SELECTED_*, set by selected_backend.h (via solver.h). #undef'd at the end
// so none leaks into the including TU.
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
// The wwrsolver* surface, pasted from the one fragment the host paths share. The
// macros above plus the vendor header, library_types.h and WWR_SELECTED_* are
// exactly what detail/solver_names.h documents it needs in scope.
// ---------------------------------------------------------------------------
namespace wwr {
#include "detail/solver_names.h"
} // namespace wwr

#undef WWR_SELECT_RAW
#undef WWR_TYPE_RAW
#undef WWR_VALUE_RAW
#undef WWR_FUNCTION_RAW
