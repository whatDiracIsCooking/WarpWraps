/**
 * @file fp6.cppm
 * @brief Backend-neutral 6-bit float scalar types: wwrFp6* for __nv_fp6_* / __hip_fp6_*
 *
 * Exports the surface CUDA's cuda_fp6.h and HIP's hip_fp6.h share -- the E3M2
 * and E2M3 scalar types (plus their x2/x4 packed variants), the storage
 * typedefs, the interpretation enum, and the float/double narrowing
 * conversions -- under one set of gpu* names in namespace wwr. See
 * backend.h for the switch. Companion to fp8.cppm and fp4.cppm.
 *
 * Both vendors define the same two fp6 formats (E3M2, E2M3), so the whole
 * scalar surface intersects; the divergence is only in spelling, which the
 * gpu* names absorb.
 *
 * BLOCKED ON THE HIP BACKEND, so this module is NOT registered in
 * src/CMakeLists.txt (nor its test in test/gpu/CMakeLists.txt) -- it exists as
 * complete, reviewed source that flips on the day its HIP raw dependency does.
 * wwr.hip.hip_fp6 does not build with this toolchain: its vendor header pulls
 * in amd_detail/amd_hip_ocp_types.h, which #errors outside real `-x hip` device
 * mode or a genuine GCC >= 13 front end. See src/hip/README.md, "wwr.hip.hip_fp4
 * / wwr.hip.hip_fp6 -- blocked, not built", and the BLOCKED comments in
 * src/CMakeLists.txt / src/hip/CMakeLists.txt. The CUDA raw module
 * (wwr.cuda.cuda_fp6) builds, so the block is HIP-only; fp8.cppm, whose HIP raw
 * module is live, IS wired in.
 *
 * Only NARROWING (float/double -> storage) is wrapped, for the reason fp8.cppm
 * gives: neither vendor has a direct storage -> float widening call, so a value
 * read back to float travels through the scalar struct types' own operator
 * float / operator double. The packed (x2) and __half_raw conversions remain in
 * the raw modules (wwr.hip.hip_fp6 / wwr.cuda.cuda_fp6).
 *
 * The rounding-mode enum is exported here under fp6-specific names
 * (wwrFp6RoundMode and wwrFp6Round*), NOT a bare wwrRoundMode shared with
 * fp4.cppm. The raw modules can each re-export the one global ::hipRoundMode /
 * ::cudaRoundMode with a `using` and collide on nothing, but a WWR_VALUE here
 * *defines* a module-attached inline constant, so a single wwrRoundMode /
 * wwrRoundNearest defined in both wwr.fp6 and wwr.fp4 would be two entities of
 * the same name -- an ODR clash the moment a TU imports both. Per-type names
 * keep fp4 and fp6 independently importable, in the spirit of
 * docs/architecture.md section 11.
 *
 * Like complex.cppm, this module includes no vendor header in its GMF: the raw
 * modules already export host wrappers for the static-inline __nv_cvt_* /
 * __hip_cvt_* conversions (docs/architecture.md section 12), so the gpu*
 * forwarders reach them through the import via WWR_SELECT. Importing rather
 * than #including also keeps this module clear of section 11 entirely -- it
 * never puts the hip_fp6.h header in a TU, so it can never share one with
 * hip_fp4.h.
 *
 * Usage:
 *   import wwr.fp6;
 *
 *   wwrFp6Storage x = wwrFloat2Fp6(1.5f, wwrE3m2, wwrFp6RoundNearest);
 *   wwrFp6E3m2 v{1.5f};                  // narrow via the scalar type's ctor
 *   float back = static_cast<float>(v);  // widen via its operator float
 */

module;

#include "backend.h"

export module wwr.fp6;

#if defined(WWR_GPU_BACKEND_CUDA)
import wwr.cuda.cuda_fp6;
#else
import wwr.hip.hip_fp6;
#endif

export namespace wwr {

// ========================================================================
// Storage typedefs -- the unsigned integer holding 1/2/4 packed fp6 values
// ========================================================================

WWR_TYPE(wwrFp6Storage, __nv_fp6_storage_t, __hip_fp6_storage_t)
WWR_TYPE(wwrFp6x2Storage, __nv_fp6x2_storage_t, __hip_fp6x2_storage_t)
WWR_TYPE(wwrFp6x4Storage, __nv_fp6x4_storage_t, __hip_fp6x4_storage_t)

// ========================================================================
// Interpretation (format kind) of an fp6 value
// ========================================================================

WWR_TYPE(wwrFp6Interpretation, __nv_fp6_interpretation_t, __hip_fp6_interpretation_t)
WWR_VALUE(wwrE3m2, __NV_E3M2, __HIP_E3M2)
WWR_VALUE(wwrE2m3, __NV_E2M3, __HIP_E2M3)

// ========================================================================
// Rounding mode taken by the fp6 narrowing conversions
// (fp6-specific names -- see the file header on why not a shared wwrRoundMode)
// ========================================================================

WWR_TYPE(wwrFp6RoundMode, cudaRoundMode, hipRoundMode)
WWR_VALUE(wwrFp6RoundNearest, cudaRoundNearest, hipRoundNearest)
WWR_VALUE(wwrFp6RoundZero, cudaRoundZero, hipRoundZero)
WWR_VALUE(wwrFp6RoundPosInf, cudaRoundPosInf, hipRoundPosInf)
WWR_VALUE(wwrFp6RoundMinInf, cudaRoundMinInf, hipRoundMinInf)

// ========================================================================
// Scalar struct types (E3M2: 3 exponent, 2 mantissa bits)
// ========================================================================

WWR_TYPE(wwrFp6E3m2, __nv_fp6_e3m2, __hip_fp6_e3m2)
WWR_TYPE(wwrFp6x2E3m2, __nv_fp6x2_e3m2, __hip_fp6x2_e3m2)
WWR_TYPE(wwrFp6x4E3m2, __nv_fp6x4_e3m2, __hip_fp6x4_e3m2)

// ========================================================================
// Scalar struct types (E2M3: 2 exponent, 3 mantissa bits)
// ========================================================================

WWR_TYPE(wwrFp6E2m3, __nv_fp6_e2m3, __hip_fp6_e2m3)
WWR_TYPE(wwrFp6x2E2m3, __nv_fp6x2_e2m3, __hip_fp6x2_e2m3)
WWR_TYPE(wwrFp6x4E2m3, __nv_fp6x4_e2m3, __hip_fp6x4_e2m3)

// ========================================================================
// Narrowing conversions (float/double -> fp6 storage)
//
// Forwarding functions routing through the raw module's host wrappers via
// WWR_SELECT, as in complex.cppm. fp6 has no saturation parameter (out-of-range
// always saturates to MAXNORM); the argument order both vendors declare is
// (value, interpretation, rounding).
// ========================================================================

/// @brief Narrow a float to an fp6 value in the given format and rounding
inline wwrFp6Storage wwrFloat2Fp6(const float value, const wwrFp6Interpretation interpretation,
                                  const wwrFp6RoundMode rounding) {
  return WWR_SELECT(__nv_cvt_float_to_fp6, __hip_cvt_float_to_fp6)(value, interpretation, rounding);
}

/// @brief Narrow a double to an fp6 value in the given format and rounding
inline wwrFp6Storage wwrDouble2Fp6(const double value, const wwrFp6Interpretation interpretation,
                                   const wwrFp6RoundMode rounding) {
  return WWR_SELECT(__nv_cvt_double_to_fp6, __hip_cvt_double_to_fp6)(value, interpretation, rounding);
}

} // namespace wwr
