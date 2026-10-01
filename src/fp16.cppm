/**
 * @file fp16.cppm
 * @brief Backend-neutral half-precision type and host conversions: wwr* for __half
 *
 * The host-module counterpart to fp16.h's device-pass-gated section: wwrHalf and
 * the float<->half conversions. Both backends spell the type __half (cuda_fp16.h /
 * hip_fp16.h); the alias exists so code above src names it one way.
 *
 * Unlike complex.cppm, this module imports no raw vendor module. The float<->half
 * conversions have external linkage (docs/architecture.md section 12 lists only
 * the fp4/fp6/fp8 conversion families as static-inline, not fp16's), so the host
 * wrappers below bind to ::__float2half / ::__half2float directly -- the vendor
 * header, in the GMF via fp16.h, preserves their linkage (section 14). The type
 * comes from fp16.h too, so nothing here needs an import. See fp16.h.
 *
 * Usage:
 *   import wwr.fp16;
 */

module;

#include "backend.h"

// The wwrHalf type (re-exported below) and the vendor header whose __float2half /
// __half2float the host wrappers call directly. Compiled as host C++, fp16.h's
// device-pass-gated section is absent; a device TU gets those wrappers instead.
#include "fp16.h"

export module wwr.fp16;

export namespace wwr {

using wwr::wwrHalf;

/// @brief Convert a float to half precision (round to nearest even)
inline wwrHalf wwrFloat2Half(const float value) {
  return ::__float2half(value);
}

/// @brief Widen a half-precision value back to float (exact)
inline float wwrHalf2Float(const wwrHalf value) {
  return ::__half2float(value);
}

} // namespace wwr
