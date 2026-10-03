/**
 * @file wwr/rtc.h
 * @brief The backend-neutral NVRTC / hipRTC surface as a single, self-contained
 *        HOST #include, for a TU that does not use C++ modules
 *
 * The non-module counterpart to `import wwr.rtc;`: a plain .cpp that #includes
 * this one header gets the same wwrrtc* / WWRRTC_* surface the module exports --
 * the program / result types, the 12 shared result codes, and the version,
 * error-string, program-lifecycle, compilation, compiled-output, log and
 * name-expression functions -- bound to the selected backend's ::nvrtc* /
 * ::hiprtc* entry points.
 *
 * The wwrrtc* name list is NOT restated here: it is the one fragment the host
 * paths share, detail/rtc_names.h, pasted below inside `namespace wwr` exactly
 * as rtc.cppm's purview pastes it. Add a name there, once, and this path gains
 * it. What this file supplies before that #include is what the fragment
 * documents it needs: the vendor header (via rtc.h, the layer's one
 * vendor-include point) and the reference-alias binding macros (the module gets
 * those from backend.h; here they are defined locally and #undef'd at the end so
 * none leaks into the including TU).
 *
 * Backend selection and the include root arrive by linking wwr::rtc::host. HOST
 * only -- NVRTC / hipRTC has no device surface. See src/rtc.cppm, src/rtc.h and
 * docs/architecture.md, section 5.
 */

#pragma once

// HOST only -- see the file header. NVRTC / hipRTC has no device surface, but
// the guard keeps this path consistent with the other wwr/*.h headers.
#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)
#error                                                                                              \
    "wwr/rtc.h is the HOST #include path for the rtc surface. See src/README.md."
#endif

// The single vendor-include point for the rtc layer (the vendor header the
// bindings below resolve against) and selected_backend.h's WWR_SELECTED_*. Angle
// brackets, not quotes: this file is itself src/wwr/rtc.h, so a quoted "rtc.h"
// would resolve to this file (quotes search the current directory first) and
// #pragma-once to nothing; <rtc.h> skips that search and resolves to src/rtc.h
// on the include root.
#include <rtc.h>

// ---------------------------------------------------------------------------
// Binding macros, reference-alias form -- the same shape backend.h gives the
// module, defined locally here because a non-module TU imports nothing. Keyed on
// WWR_SELECTED_*, set by selected_backend.h (via rtc.h). #undef'd at the end so
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
// The wwrrtc* surface, pasted from the one fragment the host paths share. The
// macros above plus the vendor header are exactly what detail/rtc_names.h
// documents it needs in scope.
// ---------------------------------------------------------------------------
namespace wwr {
#include "detail/rtc_names.h"
} // namespace wwr

#undef WWR_SELECT_RAW
#undef WWR_TYPE_RAW
#undef WWR_VALUE_RAW
#undef WWR_FUNCTION_RAW
