/**
 * @file hip_fp8.cppm
 * @brief HIP FP8 Floating-Point API module wrapper
 *
 * Wraps hip/hip_fp8.h for C++23 module-based code: 8-bit floating-point types,
 * enumerations, storage typedefs and host-accessible conversion functions.
 * CUDA counterpart: wwr.cuda.cuda_fp8.
 *
 * HIP defines FOUR struct formats where CUDA's cuda_fp8.h has three: OCP
 * e4m3/e5m2 and AMD's original fnuz-encoded e4m3/e5m2, each with x2/x4 vector
 * variants. There is no e8m0 scaling-factor format.
 * __hip_fp8_interpretation_t carries all four interpretations, and
 * __hip_saturation_t is a 2-value enum distinct from CUDA's __nv_saturation_t.
 *
 * The C-style __hip_cvt_* functions are `static inline` and so get forwarding
 * wrappers rather than `using` declarations -- docs/architecture.md, section
 * 12. The struct types' converting constructors work via ADL once exported.
 *
 * This module needs BOTH <array> and <algorithm> pre-included, for two
 * different reasons -- docs/architecture.md, sections 9 and 10.
 *
 * Usage:
 *   import wwr.hip.hip_fp8;
 */

module;

// Load-bearing, and must stay before the HIP header: host_defines.h poisons
// __noinline__ for libc++'s __config. docs/architecture.md, section 9.
#include <array>
// Also load-bearing, for a second reason: device_library_decls.h poisons
// __local for libc++'s <algorithm>. docs/architecture.md, section 10.
#include <algorithm>
#include <hip/hip_fp8.h>

export module wwr.hip.hip_fp8;

export namespace wwr::hip {

// ========================================================================
// Storage Typedefs
// ========================================================================

using ::__hip_fp8_storage_t;
using ::__hip_fp8x2_storage_t;
using ::__hip_fp8x4_storage_t;

// ========================================================================
// Enumerations
// ========================================================================

// Saturation mode used when narrowing to fp8 destination types
using ::__HIP_NOSAT;
using ::__HIP_SATFINITE;
using ::__hip_saturation_t;

// Interpretation (format kind) of an fp8 value
using ::__HIP_E4M3;
using ::__HIP_E4M3_FNUZ;
using ::__HIP_E5M2;
using ::__HIP_E5M2_FNUZ;
using ::__hip_fp8_interpretation_t;

// ========================================================================
// C++ FP8 Struct Types (OCP E4M3 format)
// ========================================================================

using ::__hip_fp8_e4m3;
using ::__hip_fp8x2_e4m3;
using ::__hip_fp8x4_e4m3;

// ========================================================================
// C++ FP8 Struct Types (OCP E5M2 format)
// ========================================================================

using ::__hip_fp8_e5m2;
using ::__hip_fp8x2_e5m2;
using ::__hip_fp8x4_e5m2;

// ========================================================================
// C++ FP8 Struct Types (AMD fnuz E4M3 format)
// ========================================================================

using ::__hip_fp8_e4m3_fnuz;
using ::__hip_fp8x2_e4m3_fnuz;
using ::__hip_fp8x4_e4m3_fnuz;

// ========================================================================
// C++ FP8 Struct Types (AMD fnuz E5M2 format)
// ========================================================================

using ::__hip_fp8_e5m2_fnuz;
using ::__hip_fp8x2_e5m2_fnuz;
using ::__hip_fp8x4_e5m2_fnuz;

// ========================================================================
// Dependent Raw Types (from amd_hip_fp16.h / amd_hip_bf16.h / vector_types.h)
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
// namespace in amd_hip_fp8.h and cannot be re-exported via using
// declarations. Thin inline wrappers are provided here so that callers
// importing this module can access them by name within wwr::hip.

// -- Narrowing conversions (to fp8 storage) -------------------------------

inline __hip_fp8_storage_t __hip_cvt_float_to_fp8(const float f, const __hip_saturation_t sat,
                                                  const __hip_fp8_interpretation_t interp) {
  return ::__hip_cvt_float_to_fp8(f, sat, interp);
}

inline __hip_fp8x2_storage_t __hip_cvt_float2_to_fp8x2(const float2 f2,
                                                       const __hip_saturation_t sat,
                                                       const __hip_fp8_interpretation_t interp) {
  return ::__hip_cvt_float2_to_fp8x2(f2, sat, interp);
}

inline __hip_fp8_storage_t __hip_cvt_double_to_fp8(const double d, const __hip_saturation_t sat,
                                                   const __hip_fp8_interpretation_t interp) {
  return ::__hip_cvt_double_to_fp8(d, sat, interp);
}

inline __hip_fp8x2_storage_t __hip_cvt_double2_to_fp8x2(const double2 d2,
                                                        const __hip_saturation_t sat,
                                                        const __hip_fp8_interpretation_t interp) {
  return ::__hip_cvt_double2_to_fp8x2(d2, sat, interp);
}

inline __hip_fp8_storage_t __hip_cvt_bfloat16raw_to_fp8(const __hip_bfloat16_raw hr,
                                                        const __hip_saturation_t sat,
                                                        const __hip_fp8_interpretation_t interp) {
  return ::__hip_cvt_bfloat16raw_to_fp8(hr, sat, interp);
}

inline __hip_fp8x2_storage_t
__hip_cvt_bfloat16raw2_to_fp8x2(const __hip_bfloat162_raw hr, const __hip_saturation_t sat,
                                const __hip_fp8_interpretation_t interp) {
  return ::__hip_cvt_bfloat16raw2_to_fp8x2(hr, sat, interp);
}

inline __hip_fp8_storage_t __hip_cvt_halfraw_to_fp8(const __half_raw x,
                                                    const __hip_saturation_t sat,
                                                    const __hip_fp8_interpretation_t interp) {
  return ::__hip_cvt_halfraw_to_fp8(x, sat, interp);
}

inline __hip_fp8x2_storage_t __hip_cvt_halfraw2_to_fp8x2(const __half2_raw x,
                                                         const __hip_saturation_t sat,
                                                         const __hip_fp8_interpretation_t interp) {
  return ::__hip_cvt_halfraw2_to_fp8x2(x, sat, interp);
}

// -- Widening conversions (from fp8 storage) ------------------------------

inline __half_raw __hip_cvt_fp8_to_halfraw(const __hip_fp8_storage_t x,
                                           const __hip_fp8_interpretation_t interp) {
  return ::__hip_cvt_fp8_to_halfraw(x, interp);
}

inline __half2_raw __hip_cvt_fp8x2_to_halfraw2(const __hip_fp8x2_storage_t x,
                                               const __hip_fp8_interpretation_t interp) {
  return ::__hip_cvt_fp8x2_to_halfraw2(x, interp);
}

} // namespace wwr::hip
