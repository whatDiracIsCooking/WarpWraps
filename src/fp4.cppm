/**
 * @file fp4.cppm
 * @brief Backend-neutral 4-bit float scalar types: wwrFp4* for __nv_fp4_* / __hip_fp4_*
 *
 * Exports the surface CUDA's cuda_fp4.h and HIP's hip_fp4.h share -- the E2M1
 * scalar type (plus its x2/x4 packed variants), the storage typedefs, the
 * interpretation enum, and the float/double narrowing conversions -- under one
 * set of wwr* names in namespace wwr. See backend.h for the switch.
 * Companion to fp8.cppm and fp6.cppm.
 *
 * Both vendors define the one fp4 format (E2M1), so the whole scalar surface
 * intersects; the divergence is only in spelling, which the wwr* names absorb.
 *
 * BLOCKED ON THE HIP BACKEND, so this module is NOT registered in
 * src/CMakeLists.txt (nor its test in test/gpu/CMakeLists.txt) -- it exists as
 * complete, reviewed source that flips on the day its HIP raw dependency does.
 * wwr.hip.hip_fp4 does not build with this toolchain: its vendor header pulls
 * in amd_detail/amd_hip_ocp_types.h, which #errors outside real `-x hip` device
 * mode or a genuine GCC >= 13 front end. See src/hip/README.md, "wwr.hip.hip_fp4
 * / wwr.hip.hip_fp6 -- blocked, not built", and the BLOCKED comments in
 * src/CMakeLists.txt / src/hip/CMakeLists.txt. The CUDA raw module
 * (wwr.cuda.cuda_fp4) builds, so the block is HIP-only; fp8.cppm, whose HIP raw
 * module is live, IS wired in.
 *
 * Only NARROWING (float/double -> storage) is wrapped, for the reason fp8.cppm
 * gives: neither vendor has a direct storage -> float widening call, so a value
 * read back to float travels through the scalar struct types' own operator
 * float / operator double. The packed (x2) and __half_raw conversions remain in
 * the raw modules (wwr.hip.hip_fp4 / wwr.cuda.cuda_fp4).
 *
 * ONE MODULE PER TYPE, ON PURPOSE. fp4 and fp6 are kept as separate modules --
 * never a combined fp_narrow -- because docs/architecture.md section 11: the raw
 * hip_fp4.h and hip_fp6.h headers each define the same internal:: helpers as
 * non-inline statics and cannot share a translation unit, so their raw modules
 * cannot either. This neutral module imports rather than #includes, so it never
 * puts hip_fp4.h in a TU and so stays clear of section 11 directly -- but the
 * separation is preserved so the layering matches the raw one it sits on. The
 * rounding-mode enum is exported under fp4-specific names (wwrFp4RoundMode,
 * wwrFp4Round*) rather than a bare wwrRoundMode shared with fp6.cppm, so a TU
 * may import both wwr.fp4 and wwr.fp6 without an ODR clash on a module-attached
 * inline constant -- see fp6.cppm's header for the full reason.
 *
 * Like complex.cppm, this module includes no vendor header in its GMF: the raw
 * modules already export host wrappers for the static-inline __nv_cvt_* /
 * __hip_cvt_* conversions (docs/architecture.md section 12), so the wwr*
 * forwarders reach them through the import via WWR_SELECT.
 *
 * Usage:
 *   import wwr.fp4;
 *
 *   wwrFp4Storage x = wwrFloat2Fp4(1.5f, wwrE2m1, wwrFp4RoundNearest);
 *   wwrFp4E2m1 v{1.5f};                  // narrow via the scalar type's ctor
 *   float back = static_cast<float>(v);  // widen via its operator float
 */

module;

#include "backend.h"

export module wwr.fp4;

#if defined(WWR_GPU_BACKEND_CUDA)
import wwr.cuda.cuda_fp4;
#else
import wwr.hip.hip_fp4;
#endif

export namespace wwr {

// ========================================================================
// Storage typedefs -- the unsigned integer holding 1/2/4 packed fp4 values
// ========================================================================

WWR_TYPE(wwrFp4Storage, __nv_fp4_storage_t, __hip_fp4_storage_t)
WWR_TYPE(wwrFp4x2Storage, __nv_fp4x2_storage_t, __hip_fp4x2_storage_t)
WWR_TYPE(wwrFp4x4Storage, __nv_fp4x4_storage_t, __hip_fp4x4_storage_t)

// ========================================================================
// Interpretation (format kind) of an fp4 value -- E2M1 is the only one
// ========================================================================

WWR_TYPE(wwrFp4Interpretation, __nv_fp4_interpretation_t, __hip_fp4_interpretation_t)
WWR_VALUE(wwrE2m1, __NV_E2M1, __HIP_E2M1)

// ========================================================================
// Rounding mode taken by the fp4 narrowing conversions
// (fp4-specific names -- see the file header on why not a shared wwrRoundMode)
// ========================================================================

WWR_TYPE(wwrFp4RoundMode, cudaRoundMode, hipRoundMode)
WWR_VALUE(wwrFp4RoundNearest, cudaRoundNearest, hipRoundNearest)
WWR_VALUE(wwrFp4RoundZero, cudaRoundZero, hipRoundZero)
WWR_VALUE(wwrFp4RoundPosInf, cudaRoundPosInf, hipRoundPosInf)
WWR_VALUE(wwrFp4RoundMinInf, cudaRoundMinInf, hipRoundMinInf)

// ========================================================================
// Scalar struct types (E2M1: 2 exponent, 1 mantissa bit)
// ========================================================================

WWR_TYPE(wwrFp4E2m1, __nv_fp4_e2m1, __hip_fp4_e2m1)
WWR_TYPE(wwrFp4x2E2m1, __nv_fp4x2_e2m1, __hip_fp4x2_e2m1)
WWR_TYPE(wwrFp4x4E2m1, __nv_fp4x4_e2m1, __hip_fp4x4_e2m1)

// ========================================================================
// Narrowing conversions (float/double -> fp4 storage)
//
// Forwarding functions routing through the raw module's host wrappers via
// WWR_SELECT, as in complex.cppm. fp4 has no saturation parameter (out-of-range
// always saturates to MAXNORM); the argument order both vendors declare is
// (value, interpretation, rounding).
// ========================================================================

/// @brief Narrow a float to an fp4 value in the given format and rounding
inline wwrFp4Storage wwrFloat2Fp4(const float value, const wwrFp4Interpretation interpretation,
                                  const wwrFp4RoundMode rounding) {
  return WWR_SELECT(__nv_cvt_float_to_fp4, __hip_cvt_float_to_fp4)(value, interpretation, rounding);
}

/// @brief Narrow a double to an fp4 value in the given format and rounding
inline wwrFp4Storage wwrDouble2Fp4(const double value, const wwrFp4Interpretation interpretation,
                                   const wwrFp4RoundMode rounding) {
  return WWR_SELECT(__nv_cvt_double_to_fp4, __hip_cvt_double_to_fp4)(value, interpretation, rounding);
}

} // namespace wwr
