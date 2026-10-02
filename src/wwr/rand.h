/**
 * @file wwr/rand.h
 * @brief The backend-neutral cuRAND / hipRAND HOST API as a single,
 *        self-contained HOST #include, for a TU that does not use C++ modules
 *
 * The non-module counterpart to `import wwr.rand;`: a plain .cpp that #includes
 * this one header gets the same wwrrand* / WWRRAND_* HOST surface the module
 * exports -- the generator and status types, the status / RNG-type / ordering /
 * direction-vector constants, the management and generation functions, and the
 * two scramble-constant forwarders -- bound to the selected backend's ::curand* /
 * ::hiprand* entry points. The device generator STATE TYPES (wwrrandState, ...)
 * come along too: they live in rand.h, which this header includes, and a
 * non-module TU names them straight from there.
 *
 * The wwrrand* HOST name list is NOT restated here: it is the one fragment the
 * host paths share, detail/rand_names.h, pasted below inside `namespace wwr`
 * exactly as rand.cppm's purview pastes it. Add a host name there, once, and this
 * path gains it too. What this file supplies before that #include is only what the
 * fragment's header documents it needs: the vendor host header (via rand.h) and
 * the reference-alias binding macros (the module keeps those macros in backend.h;
 * here they are defined locally and #undef'd at the end so none leaks into the
 * including TU).
 *
 * The __device__ generator FUNCTIONS are deliberately NOT here -- they are a
 * device-only surface in rand.h's device-pass-gated section. A device .cu reaches
 * them by #including rand.h directly.
 *
 * Backend selection is shared too: rand.h pulls in selected_backend.h. The
 * include root, the WWR_SELECTED_* / WWR_GPU_BACKEND_* defines, and the RNG link
 * library arrive by linking wwr::rand::host. HOST only. See src/rand.cppm,
 * src/rand.h and docs/architecture.md, sections 1 and 6.
 */

#pragma once

// HOST only -- see the file header. A device .cu/.cuh gets the generator
// functions from rand.h's device-pass-gated section.
#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)
#error                                                                                              \
    "wwr/rand.h is the HOST #include path; a device .cu/.cuh gets the rand device generators from rand.h's device section. See src/README.md."
#endif

// The single vendor-include point for the rand layer: the host header (curand.h /
// hiprand.h) the bindings below resolve against, the device generator state types
// (named straight from here by a non-module TU), and selected_backend.h's
// WWR_SELECTED_*. Its device-pass-gated generators are absent in this host TU.
//
// Angle brackets, not quotes: this file is itself src/wwr/rand.h, so a quoted
// "rand.h" would resolve to this file (quotes search the current file's directory
// first) and #pragma-once to nothing; <rand.h> skips that search and resolves to
// src/rand.h on the include root. rand.h stays the layer's one vendor-include
// point -- this path goes through it rather than re-including the vendor headers.
#include <rand.h>

// ---------------------------------------------------------------------------
// Binding macros, reference-alias form -- the same shape backend.h gives the
// module, defined locally here because a non-module TU imports nothing. Keyed on
// WWR_SELECTED_*, set by selected_backend.h (via rand.h). #undef'd at the end so
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
// The wwrrand* HOST surface -- types, constants, functions and the two
// scramble-constant forwarders -- pasted from the one fragment the host paths
// share. The macros above plus the vendor host header and WWR_SELECTED_* (both
// via rand.h) are exactly what detail/rand_names.h documents it needs in scope.
// ---------------------------------------------------------------------------
namespace wwr {
#include "detail/rand_names.h"
} // namespace wwr

#undef WWR_SELECT_RAW
#undef WWR_TYPE_RAW
#undef WWR_VALUE_RAW
#undef WWR_FUNCTION_RAW
