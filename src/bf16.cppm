/**
 * @file bf16.cppm
 * @brief Backend-neutral bfloat16 type and host conversions: wwr* for
 *        __nv_bfloat16 / __hip_bfloat16
 *
 * The host-module counterpart to bf16.h's device-pass-gated section: wwrBfloat16
 * and the float<->bfloat16 conversions. The type DIVERGES by backend
 * (__nv_bfloat16 vs __hip_bfloat16); the alias names it one way.
 *
 * Unlike complex.cppm, this module imports no raw vendor module. The
 * float<->bfloat16 conversions have external linkage (docs/architecture.md
 * section 12 lists only the fp4/fp6/fp8 conversion families as static-inline, not
 * bf16's), so the host wrappers below bind to ::__float2bfloat16 /
 * ::__bfloat162float directly -- the vendor header, in the GMF via bf16.h,
 * preserves their linkage (section 14). The type comes from bf16.h too, so nothing
 * here needs an import. (bfloat16's arithmetic OPERATORS are static-inline on HIP
 * and wrapped in the raw module, but this neutral layer wraps only the
 * conversions, which are not.) See bf16.h.
 *
 * Usage:
 *   import wwr.bf16;
 */

module;

#include "backend.h"

// The wwrBfloat16 type (re-exported below) and the vendor header whose
// __float2bfloat16 / __bfloat162float the host wrappers call directly. Compiled as
// host C++, bf16.h's device-pass-gated section is absent; a device TU gets those
// wrappers instead.
#include "bf16.h"

export module wwr.bf16;

export namespace wwr {

using wwr::wwrBfloat16;

/// @brief Convert a float to bfloat16 (round to nearest even)
inline wwrBfloat16 wwrFloat2Bfloat16(const float value) {
  return ::__float2bfloat16(value);
}

/// @brief Widen a bfloat16 value back to float (exact)
inline float wwrBfloat162Float(const wwrBfloat16 value) {
  return ::__bfloat162float(value);
}

} // namespace wwr
