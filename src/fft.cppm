/**
 * @file fft.cppm
 * @brief Backend-neutral FFT: wwrfft* names for cuFFT / hipFFT
 *
 * wwrfft<name> stands for cufft<name> on a CUDA build and hipfft<name> on a HIP
 * build. Only the base (single-GPU) API is wrapped here -- the multi-GPU
 * eXtended surface is reached through wwr.cuda.cufftXt / wwr.hip.hipfftXt
 * directly. See backend.h.
 *
 * Backend differences resolved here, not above:
 *
 * - Direction flags: WWRFFT_FORWARD / WWRFFT_INVERSE follow cuFFT's spelling
 *   (hipFFT spells the inverse HIPFFT_BACKWARD), as the wwr* layer does throughout.
 * - Result codes: only the 14 codes both backends export get a WWRFFT_* alias;
 *   a backend-specific code is reached through the raw module.
 * - Status strings: wwrfftGetStatusName / wwrfftGetStatusString are hand-written
 *   per backend, neither vendor shipping one (see docs/architecture.md §5).
 *
 * wwrfftGetProperty is omitted (the vendors disagree on the property-type enum);
 * reach it through the raw module.
 *
 * The wwrfft* surface itself is NOT restated here: it is the one fragment the two
 * host paths share, detail/fft_names.h, pasted below inside the exported
 * `namespace wwr` exactly as wwr/fft.h (the non-module #include path) pastes it.
 * Add a name there, once, and both paths gain it. The vendor header is drawn from
 * fft.h, the single vendor-include point (the "rand.h shape"): this module binds
 * wwr* references straight to the `::cufft*` / `::hipfft*` declarations with the
 * _RAW macros and imports no raw vendor module -- the host API is a real
 * external-linkage library, so a reference needs only the declaration. No import
 * is needed for the surface: its only hand-written functions are the
 * status-string switches, which name no non-vendor type.
 *
 * Usage:
 *   import wwr.fft;
 *
 *   wwrfftHandle plan;
 *   wwrfftCreate(&plan);
 */

module;

#include "backend.h"

// The single vendor-include point for the fft layer: cufft.h / hipfft.h, whose
// `::cufft*` / `::hipfft*` declarations the _RAW bindings in detail/fft_names.h
// resolve against. No raw vendor module is imported -- the host API is
// external-linkage, so a reference or type alias needs only these declarations.
// See fft.h and backend.h.
#include "fft.h"

export module wwr.fft;

export namespace wwr {

// The whole neutral FFT surface -- types, constants, the status-string switches
// and the lifecycle, plan, work-size and exec functions -- lives in
// detail/fft_names.h, the one list both host paths share: this module and
// wwr/fft.h (the non-module #include path). The WWR_*_RAW macros from backend.h
// and WWR_SELECTED_* and the vendor header from fft.h's #include are exactly what
// that fragment's header documents it needs in scope.
#include "detail/fft_names.h"

} // namespace wwr
