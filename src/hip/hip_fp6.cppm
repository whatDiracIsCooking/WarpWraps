/**
 * @file hip_fp6.cppm
 * @brief HIP FP6 Floating-Point API module wrapper
 *
 * Wraps hip/hip_fp6.h for C++23 module-based code: 6-bit floating-point types
 * (__hip_fp6_e2m3, __hip_fp6_e3m2, plus x2/x4 variants), enumerations, storage
 * typedefs and host-accessible conversion functions. CUDA counterpart:
 * gpumod.cuda.cuda_fp6.
 *
 * IMPORTANT: this module and gpumod.hip.hip_fp4 must never appear in the same
 * translation unit's global module fragment -- both vendor headers define the
 * same internal:: helpers as non-inline statics. See docs/architecture.md,
 * section 11. This .cppm includes only hip_fp6.h; keep it that way.
 *
 * Like hip_fp4, the conversions take HIP's `enum hipRoundMode`, exported here
 * independently of hip_fp4.cppm's own copy -- two distinct modules each
 * re-exporting one global-namespace enum, no conflict.
 *
 * The C-style __hip_cvt_* functions are `static inline` and so get forwarding
 * wrappers rather than `using` declarations -- docs/architecture.md, section 12.
 *
 * Usage:
 *   import gpumod.hip.hip_fp6;
 */

module;

// Load-bearing, and must stay before the HIP header: host_defines.h poisons
// __noinline__ for libc++'s __config. docs/architecture.md, section 9.
#include <array>
#include <hip/hip_fp6.h>

export module gpumod.hip.hip_fp6;

export namespace gpumod::hip {

// ========================================================================
// Storage Typedefs
// ========================================================================

using ::__hip_fp6_storage_t;
using ::__hip_fp6x2_storage_t;
using ::__hip_fp6x4_storage_t;

// ========================================================================
// Enumerations
// ========================================================================

// Interpretation (format kind) of an fp6 value
using ::__HIP_E2M3;
using ::__HIP_E3M2;
using ::__hip_fp6_interpretation_t;

// Rounding mode parameter shared by the fp4/fp6 conversion functions
// (see file header note above -- currently ignored on AMD GPUs)
using ::hipRoundMinInf;
using ::hipRoundMode;
using ::hipRoundNearest;
using ::hipRoundPosInf;
using ::hipRoundZero;

// ========================================================================
// C++ FP6 Struct Types (E3M2 format)
// ========================================================================

using ::__hip_fp6_e3m2;
using ::__hip_fp6x2_e3m2;
using ::__hip_fp6x4_e3m2;

// ========================================================================
// C++ FP6 Struct Types (E2M3 format)
// ========================================================================

using ::__hip_fp6_e2m3;
using ::__hip_fp6x2_e2m3;
using ::__hip_fp6x4_e2m3;

// ========================================================================
// Dependent Raw Types (from amd_hip_fp8.h / amd_hip_fp16.h / amd_hip_bf16.h)
// ========================================================================

using ::double2;
using ::float2;

using ::__half2_raw;
using ::__half_raw;

using ::__hip_bfloat162_raw;
using ::__hip_bfloat16_raw;

// ========================================================================
// C-Style Conversion Function Wrappers
// ========================================================================
// The following functions are defined as static inline in the global
// namespace in amd_hip_fp6.h and cannot be re-exported via using
// declarations. Thin inline wrappers are provided here so that callers
// importing this module can access them by name within gpumod::hip.

// -- Narrowing conversions (to fp6 storage) -------------------------------

inline __hip_fp6_storage_t __hip_cvt_double_to_fp6(const double x,
                                                   const __hip_fp6_interpretation_t interp,
                                                   const hipRoundMode rounding) {
  return ::__hip_cvt_double_to_fp6(x, interp, rounding);
}

inline __hip_fp6x2_storage_t __hip_cvt_double2_to_fp6x2(const double2 x,
                                                        const __hip_fp6_interpretation_t interp,
                                                        const hipRoundMode rounding) {
  return ::__hip_cvt_double2_to_fp6x2(x, interp, rounding);
}

inline __hip_fp6_storage_t __hip_cvt_float_to_fp6(const float x,
                                                  const __hip_fp6_interpretation_t interp,
                                                  const hipRoundMode rounding) {
  return ::__hip_cvt_float_to_fp6(x, interp, rounding);
}

inline __hip_fp6x2_storage_t __hip_cvt_float2_to_fp6x2(const float2 x,
                                                       const __hip_fp6_interpretation_t interp,
                                                       const hipRoundMode rounding) {
  return ::__hip_cvt_float2_to_fp6x2(x, interp, rounding);
}

inline __hip_fp6_storage_t __hip_cvt_halfraw_to_fp6(const __half_raw x,
                                                    const __hip_fp6_interpretation_t interp,
                                                    const hipRoundMode rounding) {
  return ::__hip_cvt_halfraw_to_fp6(x, interp, rounding);
}

inline __hip_fp6x2_storage_t __hip_cvt_halfraw2_to_fp6x2(const __half2_raw x,
                                                         const __hip_fp6_interpretation_t interp,
                                                         const hipRoundMode rounding) {
  return ::__hip_cvt_halfraw2_to_fp6x2(x, interp, rounding);
}

inline __hip_fp6_storage_t __hip_cvt_bfloat16raw_to_fp6(const __hip_bfloat16_raw x,
                                                        const __hip_fp6_interpretation_t interp,
                                                        const hipRoundMode rounding) {
  return ::__hip_cvt_bfloat16raw_to_fp6(x, interp, rounding);
}

inline __hip_fp6x2_storage_t
__hip_cvt_bfloat16raw2_to_fp6x2(const __hip_bfloat162_raw x,
                                const __hip_fp6_interpretation_t interp,
                                const hipRoundMode rounding) {
  return ::__hip_cvt_bfloat16raw2_to_fp6x2(x, interp, rounding);
}

// -- Widening conversions (from fp6 storage) ------------------------------

inline __half_raw __hip_cvt_fp6_to_halfraw(const __hip_fp6_storage_t x,
                                           const __hip_fp6_interpretation_t interp) {
  return ::__hip_cvt_fp6_to_halfraw(x, interp);
}

inline __half2_raw __hip_cvt_fp6x2_to_halfraw2(const __hip_fp6x2_storage_t x,
                                               const __hip_fp6_interpretation_t interp) {
  return ::__hip_cvt_fp6x2_to_halfraw2(x, interp);
}

} // namespace gpumod::hip
