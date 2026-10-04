/**
 * @file wwr/comp.h
 * @brief The backend-neutral nvCOMP / hipCOMP batched-compression surface as a
 *        single, self-contained HOST #include, for a TU that does not use
 *        C++ modules
 *
 * The non-module counterpart to `import wwr.comp;`: a plain .cpp that #includes
 * this one header gets the same wwrcomp* / WWRCOMP_* surface the module exports --
 * the status/type/stream types, the status and data-type constants, the neutral
 * LZ4 / Snappy / Cascaded option structs, and the batched compress/decompress
 * forwarders -- bound to the selected backend's `::nvcomp*` / `::hipcomp*` entry
 * points.
 *
 * The wwrcomp* name list is NOT restated here: it is the one fragment the host
 * paths share, detail/comp_names.h, pasted below inside `namespace wwr` exactly as
 * comp.cppm's purview pastes it. Add a name there, once, and this path gains it.
 * What this file supplies before that #include is what the fragment documents it
 * needs: the vendor headers and std::size_t (via comp.h, the layer's one
 * vendor-include point) and the reference-alias binding macros (the module gets
 * those from backend.h; here they are defined locally and #undef'd at the end so
 * none leaks into the including TU).
 *
 * Backend selection and the include root arrive by linking wwr::comp::host. HOST
 * only -- nvCOMP/hipCOMP's batched LLIF is a host-launched API. See src/comp.cppm,
 * src/comp.h and docs/architecture.md, section 5.
 */

#pragma once

// HOST only -- see the file header. nvCOMP/hipCOMP's batched LLIF is host-side,
// and the guard keeps this path consistent with the other wwr/*.h headers.
#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)
#error                                                                                              \
    "wwr/comp.h is the HOST #include path for the comp surface. See src/README.md."
#endif

// The single vendor-include point for the comp layer (the vendor headers the
// bindings below resolve against, plus <cstddef> for the fragment's std::size_t)
// and selected_backend.h's WWR_SELECTED_*. Angle brackets, not quotes: this file
// is itself src/wwr/comp.h, so a quoted "comp.h" would resolve to this file
// (quotes search the current directory first) and #pragma-once to nothing;
// <comp.h> skips that search and resolves to src/comp.h on the include root.
#include <comp.h>

// ---------------------------------------------------------------------------
// Binding macros, reference-alias form -- the same shape backend.h gives the
// module, defined locally here because a non-module TU imports nothing. Keyed on
// WWR_SELECTED_*, set by selected_backend.h (via comp.h). Only the TYPE / VALUE
// variants the fragment uses are defined (comp binds no function references --
// its functions are hand-written shims). #undef'd at the end so none leaks into
// the including TU.
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

// ---------------------------------------------------------------------------
// The wwrcomp* surface, pasted from the one fragment the host paths share. The
// macros above plus the vendor headers, std::size_t and WWR_SELECTED_* are
// exactly what detail/comp_names.h documents it needs in scope.
// ---------------------------------------------------------------------------
namespace wwr {
#include "detail/comp_names.h"
} // namespace wwr

#undef WWR_SELECT_RAW
#undef WWR_TYPE_RAW
#undef WWR_VALUE_RAW
