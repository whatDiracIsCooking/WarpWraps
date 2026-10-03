/**
 * @file rtc.cppm
 * @brief Backend-neutral runtime compilation: wwrrtc* names for NVRTC / hipRTC
 *
 * wwrrtc<name> stands for nvrtc<name> on a CUDA build and hiprtc<name> on a HIP
 * build. Only the surface the two libraries share by name is wrapped here -- the
 * measured intersection (devtools/header_intersection.py --cuda nvrtc.json --hip
 * hiprtc.json): the program lifecycle, compilation, the log and name-expression
 * queries, the result enum, and the version/error-string helpers. See backend.h.
 *
 * Backend difference resolved here, not above:
 *
 * - The compiled-output getter: NVRTC emits PTX (nvrtcGetPTX / nvrtcGetPTXSize),
 *   hipRTC emits a code object (hiprtcGetCode / hiprtcGetCodeSize). One neutral
 *   spelling, wwrrtcGetCode / wwrrtcGetCodeSize, maps to each backend's own.
 *
 * Single-backend names are reached through the raw module, not aliased here:
 * NVRTC's CUBIN / LTO IR / OptiX IR getters, its PCH and flow-callback surface
 * and its supported-arch query; hipRTC's linking surface (hiprtcLink*, with the
 * hipJitOption / hipJitInputType enums the hiprtcJIT_option / hiprtcJITInputType
 * macro aliases expand to) and its bitcode getters. The result codes likewise:
 * only the 12 both enums share get a WWRRTC_* alias -- an NVRTC-only code
 * (NVRTC_ERROR_CANCELLED, the PCH/time-trace codes, ...) or an hipRTC-only one
 * (HIPRTC_ERROR_LINKING) is reached through the raw module.
 *
 * The wwrrtc* surface itself is NOT restated here: it is the one fragment the
 * two host paths share, detail/rtc_names.h, pasted below inside the exported
 * `namespace wwr` exactly as wwr/rtc.h (the non-module #include path) pastes it.
 * Add a name there, once, and both paths gain it. The vendor header is drawn
 * from rtc.h, the single vendor-include point (the "rand.h shape"): this module
 * binds wwr* references straight to the `::nvrtc*` / `::hiprtc*` declarations
 * with the _RAW macros and imports no raw vendor module -- the runtime-compilation
 * API is a real external-linkage library, so a reference needs only the
 * declaration.
 *
 * Usage:
 *   import wwr.rtc;
 *
 *   wwrrtcProgram prog;
 *   wwrrtcCreateProgram(&prog, src, "k.cu", 0, nullptr, nullptr);
 */

module;

#include "backend.h"

// The single vendor-include point for the rtc layer: nvrtc.h / hip/hiprtc.h,
// whose `::nvrtc*` / `::hiprtc*` declarations the _RAW bindings in
// detail/rtc_names.h resolve against. No raw vendor module is imported -- the
// runtime-compilation API is external-linkage, so a reference or type alias
// needs only these declarations. See rtc.h and backend.h.
#include "rtc.h"

export module wwr.rtc;

export namespace wwr {

// The whole neutral runtime-compilation surface -- the two types, the 12 shared
// result codes and the functions -- lives in detail/rtc_names.h, the one list
// both host paths share: this module and wwr/rtc.h (the non-module #include
// path). The WWR_*_RAW macros from backend.h and the vendor header from rtc.h's
// #include are exactly what that fragment's header documents it needs in scope.
#include "detail/rtc_names.h"

} // namespace wwr
