/**
 * @file hip_fp4.cppm
 * @brief HIP FP4 Floating-Point API module wrapper
 *
 * Wraps hip/hip_fp4.h for C++23 module-based code: 4-bit floating-point types,
 * enumerations, storage typedefs and host-accessible conversion functions.
 * CUDA counterpart: gpumod.cuda.cuda_fp4.
 *
 * IMPORTANT: this module and gpumod.hip.hip_fp6 must never appear in the same
 * translation unit's global module fragment -- both vendor headers define the
 * same internal:: helpers as non-inline statics. See docs/architecture.md,
 * section 11. This .cppm includes only hip_fp4.h; keep it that way.
 *
 * The conversion functions take HIP's own `enum hipRoundMode` where CUDA's
 * take cudaRoundMode, so hipRoundMode is exported here too. AMD GPUs do not
 * currently honor it, but it is part of the call signature.
 *
 * The C-style __hip_cvt_* functions are `static inline` and so get forwarding
 * wrappers rather than `using` declarations -- docs/architecture.md, section
 * 12. The struct types' converting constructors work via ADL once exported.
 *
 * Usage:
 *   import gpumod.hip.hip_fp4;
 */

module;

// Load-bearing, and must stay before the HIP header: host_defines.h poisons
// __noinline__ for libc++'s __config. docs/architecture.md, section 9.
#include <array>
#include <hip/hip_fp4.h>

export module gpumod.hip.hip_fp4;

export namespace wwr::hip {

// ========================================================================
// Storage Typedefs
// ========================================================================

using ::__hip_fp4_storage_t;
using ::__hip_fp4x2_storage_t;
using ::__hip_fp4x4_storage_t;

// ========================================================================
// Enumerations
// ========================================================================

// Interpretation (format kind) of an fp4 value -- E2M1 is the only one HIP defines
using ::__HIP_E2M1;
using ::__hip_fp4_interpretation_t;

// Rounding mode parameter shared by the fp4/fp6 conversion functions
// (see file header note above -- currently ignored on AMD GPUs)
using ::hipRoundMinInf;
using ::hipRoundMode;
using ::hipRoundNearest;
using ::hipRoundPosInf;
using ::hipRoundZero;

// ========================================================================
// C++ FP4 Struct Types (E2M1 format)
// ========================================================================

using ::__hip_fp4_e2m1;
using ::__hip_fp4x2_e2m1;
using ::__hip_fp4x4_e2m1;

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
// namespace in amd_hip_fp4.h and cannot be re-exported via using
// declarations. Thin inline wrappers are provided here so that callers
// importing this module can access them by name within wwr::hip.

// -- Narrowing conversions (to fp4 storage) -------------------------------

inline __hip_fp4_storage_t __hip_cvt_double_to_fp4(const double x,
                                                   const __hip_fp4_interpretation_t interp,
                                                   const hipRoundMode rounding) {
  return ::__hip_cvt_double_to_fp4(x, interp, rounding);
}

inline __hip_fp4x2_storage_t __hip_cvt_double2_to_fp4x2(const double2 x,
                                                        const __hip_fp4_interpretation_t interp,
                                                        const hipRoundMode rounding) {
  return ::__hip_cvt_double2_to_fp4x2(x, interp, rounding);
}

inline __hip_fp4_storage_t __hip_cvt_float_to_fp4(const float x,
                                                  const __hip_fp4_interpretation_t interp,
                                                  const hipRoundMode rounding) {
  return ::__hip_cvt_float_to_fp4(x, interp, rounding);
}

inline __hip_fp4x2_storage_t __hip_cvt_float2_to_fp4x2(const float2 x,
                                                       const __hip_fp4_interpretation_t interp,
                                                       const hipRoundMode rounding) {
  return ::__hip_cvt_float2_to_fp4x2(x, interp, rounding);
}

inline __hip_fp4_storage_t __hip_cvt_halfraw_to_fp4(const __half_raw x,
                                                    const __hip_fp4_interpretation_t interp,
                                                    const hipRoundMode rounding) {
  return ::__hip_cvt_halfraw_to_fp4(x, interp, rounding);
}

inline __hip_fp4x2_storage_t __hip_cvt_halfraw2_to_fp4x2(const __half2_raw x,
                                                         const __hip_fp4_interpretation_t interp,
                                                         const hipRoundMode rounding) {
  return ::__hip_cvt_halfraw2_to_fp4x2(x, interp, rounding);
}

inline __hip_fp4_storage_t __hip_cvt_bfloat16raw_to_fp4(const __hip_bfloat16_raw x,
                                                        const __hip_fp4_interpretation_t interp,
                                                        const hipRoundMode rounding) {
  return ::__hip_cvt_bfloat16raw_to_fp4(x, interp, rounding);
}

inline __hip_fp4x2_storage_t
__hip_cvt_bfloat16raw2_to_fp4x2(const __hip_bfloat162_raw x,
                                const __hip_fp4_interpretation_t interp,
                                const hipRoundMode rounding) {
  return ::__hip_cvt_bfloat16raw2_to_fp4x2(x, interp, rounding);
}

// -- Widening conversions (from fp4 storage) ------------------------------

inline __half_raw __hip_cvt_fp4_to_halfraw(const __hip_fp4_storage_t x,
                                           const __hip_fp4_interpretation_t interp) {
  return ::__hip_cvt_fp4_to_halfraw(x, interp);
}

inline __half2_raw __hip_cvt_fp4x2_to_halfraw2(const __hip_fp4x2_storage_t x,
                                               const __hip_fp4_interpretation_t interp) {
  return ::__hip_cvt_fp4x2_to_halfraw2(x, interp);
}

} // namespace wwr::hip
