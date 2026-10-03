/**
 * @file wwr/fft.h
 * @brief The backend-neutral cuFFT / hipFFT surface as a single, self-contained
 *        HOST #include, for a TU that does not use C++ modules
 *
 * The non-module counterpart to `import wwr.fft;`: a plain .cpp that #includes
 * this one header gets the same wwrfft* / WWRFFT_* surface the module exports --
 * the handle/result/type/real/complex types, the direction / transform-type /
 * result-code constants, the two hand-written status-to-string switches, and the
 * lifecycle, plan, work-size and exec functions -- bound to the selected
 * backend's ::cufft* / ::hipfft* entry points.
 *
 * The wwrfft* name list is NOT restated here: it is the one fragment the host
 * paths share, detail/fft_names.h, pasted below inside `namespace wwr` exactly as
 * fft.cppm's purview pastes it. Add a name there, once, and this path gains it.
 * What this file supplies before that #include is what the fragment documents it
 * needs: the vendor header (via fft.h, the layer's one vendor-include point) and
 * the reference-alias binding macros (the module gets those from backend.h; here
 * they are defined locally and #undef'd at the end so none leaks into the
 * including TU). Unlike wwr/blas.h this path pulls in no complex.h: the fragment
 * names no complex type beyond the vendor's own.
 *
 * Backend selection and the include root arrive by linking wwr::fft::host.
 * HOST only -- the cuFFT/hipFFT plan and exec API is host-side. See src/fft.cppm,
 * src/fft.h and docs/architecture.md, section 5.
 */

#pragma once

// HOST only -- see the file header. The cuFFT/hipFFT plan and exec API is
// host-side, but the guard keeps this path consistent with the other wwr/*.h
// headers.
#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)
#error                                                                                              \
    "wwr/fft.h is the HOST #include path for the fft surface. See src/README.md."
#endif

// The single vendor-include point for the fft layer (the vendor header the
// bindings below resolve against) and selected_backend.h's WWR_SELECTED_*. Angle
// brackets, not quotes: this file is itself src/wwr/fft.h, so a quoted "fft.h"
// would resolve to this file (quotes search the current directory first) and
// #pragma-once to nothing; <fft.h> skips that search and resolves to src/fft.h on
// the include root.
#include <fft.h>

// ---------------------------------------------------------------------------
// Binding macros, reference-alias form -- the same shape backend.h gives the
// module, defined locally here because a non-module TU imports nothing. Keyed on
// WWR_SELECTED_*, set by selected_backend.h (via fft.h). #undef'd at the end so
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
// The wwrfft* surface, pasted from the one fragment the host paths share. The
// macros above plus the vendor header and WWR_SELECTED_* are exactly what
// detail/fft_names.h documents it needs in scope.
// ---------------------------------------------------------------------------
namespace wwr {
#include "detail/fft_names.h"
} // namespace wwr

#undef WWR_SELECT_RAW
#undef WWR_TYPE_RAW
#undef WWR_VALUE_RAW
#undef WWR_FUNCTION_RAW
