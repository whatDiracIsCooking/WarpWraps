/**
 * @file fp8.cppm
 * @brief Backend-neutral 8-bit float scalar types: wwrFp8* for __nv_fp8_* / __hip_fp8_*
 *
 * Exports the surface CUDA's cuda_fp8.h and HIP's hip_fp8.h share -- the OCP
 * E4M3 and E5M2 scalar types (plus their x2/x4 packed variants), the storage
 * typedefs, the saturation and interpretation enums, and the float/double
 * narrowing conversions -- under one set of wwr* names in namespace wwr. See
 * backend.h for the switch. Companion to fp6.cppm and fp4.cppm.
 *
 * SCOPED TO THE INTERSECTION. HIP defines four fp8 formats (OCP e4m3/e5m2 and
 * AMD fnuz-encoded e4m3/e5m2); CUDA defines three (e4m3/e5m2 plus an e8m0
 * scaling-factor format). Only OCP e4m3/e5m2 and the __NV_/__HIP_ E4M3/E5M2
 * interpretation enumerators exist on both, so those are all this neutral layer
 * names -- the fnuz and e8m0 surfaces stay reachable only through the raw
 * modules (wwr.hip.hip_fp8 / wwr.cuda.cuda_fp8), which carry them for the
 * backend that has them.
 *
 * Only NARROWING (float/double -> storage) is wrapped: neither vendor has a
 * direct storage -> float widening call (widening goes through __half_raw,
 * fp16's domain), so a value read back to float travels through the scalar
 * struct types' own operator float / operator double, which both backends carry
 * on the aliased types. The packed (x2) and __half_raw / bfloat16-raw
 * conversions likewise remain in the raw modules.
 *
 * Unlike fp16.cppm, this module includes no vendor header in its GMF: the raw
 * wwr.hip.hip_fp8 / wwr.cuda.cuda_fp8 modules already export host wrappers for
 * the static-inline __nv_cvt_* / __hip_cvt_* conversions (docs/architecture.md
 * section 12), so the wwr* forwarders below reach them through the import via
 * WWR_SELECT, exactly as the types do. Importing rather than #including also
 * means docs/architecture.md section 10 (hip_fp8.h needs <algorithm> before it)
 * is handled once in the raw module and never recurs here.
 *
 * Usage:
 *   import wwr.fp8;
 *
 *   wwrFp8Storage x = wwrFloat2Fp8(1.5f, wwrSatfinite, wwrE4m3);
 *   wwrFp8E4m3 v{1.5f};              // narrow via the scalar type's own ctor
 *   float back = static_cast<float>(v);  // widen via its operator float
 */

module;

#include "backend.h"

export module wwr.fp8;

#if defined(WWR_GPU_BACKEND_CUDA)
import wwr.cuda.cuda_fp8;
#else
import wwr.hip.hip_fp8;
#endif

export namespace wwr {

// ========================================================================
// Storage typedefs -- the unsigned integer holding 1/2/4 packed fp8 values
// ========================================================================

WWR_TYPE(wwrFp8Storage, __nv_fp8_storage_t, __hip_fp8_storage_t)
WWR_TYPE(wwrFp8x2Storage, __nv_fp8x2_storage_t, __hip_fp8x2_storage_t)
WWR_TYPE(wwrFp8x4Storage, __nv_fp8x4_storage_t, __hip_fp8x4_storage_t)

// ========================================================================
// Saturation mode (used when narrowing to an fp8 destination)
// ========================================================================

WWR_TYPE(wwrSaturation, __nv_saturation_t, __hip_saturation_t)
WWR_VALUE(wwrNosat, __NV_NOSAT, __HIP_NOSAT)
WWR_VALUE(wwrSatfinite, __NV_SATFINITE, __HIP_SATFINITE)

// ========================================================================
// Interpretation (format kind) of an fp8 value -- the shared OCP formats
// ========================================================================

WWR_TYPE(wwrFp8Interpretation, __nv_fp8_interpretation_t, __hip_fp8_interpretation_t)
WWR_VALUE(wwrE4m3, __NV_E4M3, __HIP_E4M3)
WWR_VALUE(wwrE5m2, __NV_E5M2, __HIP_E5M2)

// ========================================================================
// Scalar struct types (OCP E4M3: 1 sign, 4 exponent, 3 mantissa bits)
// ========================================================================

WWR_TYPE(wwrFp8E4m3, __nv_fp8_e4m3, __hip_fp8_e4m3)
WWR_TYPE(wwrFp8x2E4m3, __nv_fp8x2_e4m3, __hip_fp8x2_e4m3)
WWR_TYPE(wwrFp8x4E4m3, __nv_fp8x4_e4m3, __hip_fp8x4_e4m3)

// ========================================================================
// Scalar struct types (OCP E5M2: 1 sign, 5 exponent, 2 mantissa bits)
// ========================================================================

WWR_TYPE(wwrFp8E5m2, __nv_fp8_e5m2, __hip_fp8_e5m2)
WWR_TYPE(wwrFp8x2E5m2, __nv_fp8x2_e5m2, __hip_fp8x2_e5m2)
WWR_TYPE(wwrFp8x4E5m2, __nv_fp8x4_e5m2, __hip_fp8x4_e5m2)

// ========================================================================
// Narrowing conversions (float/double -> fp8 storage)
//
// Forwarding functions, not WWR_FUNCTION reference bindings: they route
// through the raw module's own host wrappers, which WWR_SELECT names, the same
// way complex.cppm's make_wwr*Complex do. The argument order -- (value,
// saturation, interpretation) -- is the one both vendors declare.
// ========================================================================

/// @brief Narrow a float to an fp8 value in the given format (round to nearest)
inline wwrFp8Storage wwrFloat2Fp8(const float value, const wwrSaturation saturate,
                                  const wwrFp8Interpretation interpretation) {
  return WWR_SELECT(__nv_cvt_float_to_fp8, __hip_cvt_float_to_fp8)(value, saturate, interpretation);
}

/// @brief Narrow a double to an fp8 value in the given format (round to nearest)
inline wwrFp8Storage wwrDouble2Fp8(const double value, const wwrSaturation saturate,
                                   const wwrFp8Interpretation interpretation) {
  return WWR_SELECT(__nv_cvt_double_to_fp8, __hip_cvt_double_to_fp8)(value, saturate, interpretation);
}

} // namespace wwr
