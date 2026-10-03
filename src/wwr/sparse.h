/**
 * @file wwr/sparse.h
 * @brief The backend-neutral cuSPARSE / hipSPARSE surface as a single,
 *        self-contained HOST #include, for a TU that does not use C++ modules
 *
 * The non-module counterpart to `import wwr.sparse;`: a plain .cpp that #includes
 * this one header gets the same wwrsparse* / WWRSPARSE_* surface the module
 * exports -- the handle/status/descriptor/operation/direction types, the status
 * / operation / direction / action / index-base / matrix-type / fill / diag /
 * pointer-mode constants, the handle, stream, pointer-mode, error-string and
 * matrix-descriptor helpers, the typed BSR / tridiagonal / pentadiagonal /
 * csrgeam2 / conversion entry points, and the std::size_t* buffer-size shims --
 * bound to the selected backend's ::cusparse* / ::hipsparse* entry points.
 *
 * The wwrsparse* name list is NOT restated here: it is the one fragment the host
 * paths share, detail/sparse_names.h, pasted below inside `namespace wwr`
 * exactly as sparse.cppm's purview pastes it. Add a name there, once, and this
 * path gains it. What this file supplies before that #include is what the
 * fragment documents it needs: the vendor header and std::size_t (via sparse.h,
 * the layer's one vendor-include point, which pulls in <cstddef>), the complex
 * types the *_bufferSize shims name (via complex.h), and the reference-alias
 * binding macros (the module gets those from backend.h; here they are defined
 * locally and #undef'd at the end so none leaks into the including TU).
 *
 * Backend selection and the include root arrive by linking wwr::sparse::host.
 * HOST only -- cuSPARSE/hipSPARSE has no device surface. See src/sparse.cppm,
 * src/sparse.h and docs/architecture.md, section 5.
 */

#pragma once

// HOST only -- see the file header. cuSPARSE/hipSPARSE has no device surface, but
// the guard keeps this path consistent with the other wwr/*.h headers.
#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)
#error                                                                                              \
    "wwr/sparse.h is the HOST #include path for the sparse surface. See src/README.md."
#endif

// The single vendor-include point for the sparse layer (the vendor header the
// bindings below resolve against, plus <cstddef> for the shims' std::size_t) and
// selected_backend.h's WWR_SELECTED_*. Angle brackets, not quotes: this file is
// itself src/wwr/sparse.h, so a quoted "sparse.h" would resolve to this file
// (quotes search the current directory first) and #pragma-once to nothing;
// <sparse.h> skips that search and resolves to src/sparse.h on the include root.
#include <sparse.h>

// The complex types the C/Z *_bufferSize shims name. The module reaches these by
// import wwr.complex; a non-module TU reaches the same wwr* aliases from
// complex.h (the src/-root shared-type header), reached root-relative with no
// self-collision (there is no src/wwr/complex.h).
#include "complex.h"

// ---------------------------------------------------------------------------
// Binding macros, reference-alias form -- the same shape backend.h gives the
// module, defined locally here because a non-module TU imports nothing. Keyed on
// WWR_SELECTED_*, set by selected_backend.h (via sparse.h). #undef'd at the end
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
// The wwrsparse* surface, pasted from the one fragment the host paths share. The
// macros above plus the vendor header, std::size_t, wwrFloatComplex /
// wwrDoubleComplex and WWR_SELECTED_* are exactly what detail/sparse_names.h
// documents it needs in scope.
// ---------------------------------------------------------------------------
namespace wwr {
#include "detail/sparse_names.h"
} // namespace wwr

#undef WWR_SELECT_RAW
#undef WWR_TYPE_RAW
#undef WWR_VALUE_RAW
#undef WWR_FUNCTION_RAW
