/**
 * @file cuda_fp8.cppm
 * @brief CUDA FP8 Floating-Point API module wrapper
 *
 * This module wraps the cuda_fp8.h header for use in C++20/C++23 module-based code.
 * It exports 8-bit floating-point types, enumerations, storage typedefs, and
 * host-accessible conversion functions.
 *
 * Usage:
 *   import gpumod.cuda.cuda_fp8;
 *
 * Note: cuda_fp8.h includes cuda_fp16.h and cuda_bf16.h internally, so the
 * raw storage types from those headers (__half_raw, __nv_bfloat16_raw, etc.)
 * are available to the conversion functions. This module exports the fp8-specific
 * types and enumerations. The C-style conversion functions (e.g.,
 * __nv_cvt_float_to_fp8) are declared as static inline and therefore cannot be
 * re-exported via using declarations; they are available to code that has
 * imported this module through the global module fragment's #include.
 *
 * The C++ struct types (__nv_fp8_e4m3, __nv_fp8_e5m2, __nv_fp8_e8m0, and
 * their x2/x4 vector variants) are fully exported and usable on the host.
 *
 * Operators and converting constructors work automatically via Argument-Dependent
 * Lookup (ADL) when using the exported types.
 */

module;

#include <cuda_fp8.h>

export module gpumod.cuda.cuda_fp8;

import std;

export namespace gpumod::cuda {

// ========================================================================
// Storage Typedefs
// ========================================================================

// Underlying unsigned integer storage for a single fp8 value (1 byte)
using ::__nv_fp8_storage_t;

// Underlying unsigned integer storage for a pair of fp8 values (2 bytes)
using ::__nv_fp8x2_storage_t;

// Underlying unsigned integer storage for four fp8 values (4 bytes)
using ::__nv_fp8x4_storage_t;

// ========================================================================
// Enumerations
// ========================================================================

// Saturation mode used when narrowing to fp8 destination types
using ::__nv_saturation_t;

// Enumerators of __nv_saturation_t
using ::__NV_NOSAT;
using ::__NV_SATFINITE;

// Interpretation (format kind) of an fp8 value
using ::__nv_fp8_interpretation_t;

// Enumerators of __nv_fp8_interpretation_t
using ::__NV_E4M3;
using ::__NV_E5M2;

// ========================================================================
// C++ FP8 Struct Types (E5M2 format)
// ========================================================================

// Single 8-bit float (E5M2: 5 exponent bits, 2 mantissa bits)
using ::__nv_fp8_e5m2;

// Vector of two E5M2 fp8 values packed into 2 bytes
using ::__nv_fp8x2_e5m2;

// Vector of four E5M2 fp8 values packed into 4 bytes
using ::__nv_fp8x4_e5m2;

// ========================================================================
// C++ FP8 Struct Types (E4M3 format)
// ========================================================================

// Single 8-bit float (E4M3: 4 exponent bits, 3 mantissa bits)
using ::__nv_fp8_e4m3;

// Vector of two E4M3 fp8 values packed into 2 bytes
using ::__nv_fp8x2_e4m3;

// Vector of four E4M3 fp8 values packed into 4 bytes
using ::__nv_fp8x4_e4m3;

// ========================================================================
// C++ FP8 Struct Types (E8M0 scaling factor format)
// ========================================================================

// Single 8-bit scaling factor (E8M0: 8 exponent bits, 0 mantissa bits)
using ::__nv_fp8_e8m0;

// Vector of two E8M0 scaling factors packed into 2 bytes
using ::__nv_fp8x2_e8m0;

// Vector of four E8M0 scaling factors packed into 4 bytes
using ::__nv_fp8x4_e8m0;

// ========================================================================
// Dependent Raw Types (from cuda_fp16.h / cuda_bf16.h / vector_types.h)
// ========================================================================

// Vector types used as parameters in the conversion functions below
using ::double2;
using ::float2;

// Raw half-precision storage types (from cuda_fp16.h)
using ::__half2_raw;
using ::__half_raw;

// Raw bfloat16 storage types (from cuda_bf16.h)
using ::__nv_bfloat162_raw;
using ::__nv_bfloat16_raw;

// ========================================================================
// Rounding Mode (from device_types.h, included transitively via cuda_fp8.h)
// ========================================================================

// Rounding mode used by e8m0 scaling factor conversion functions
using ::cudaRoundMode;

// Enumerators of cudaRoundMode
using ::cudaRoundMinInf;
using ::cudaRoundNearest;
using ::cudaRoundPosInf;
using ::cudaRoundZero;

// ========================================================================
// C-Style Conversion Function Wrappers
// ========================================================================
// The following functions are defined as static inline in the global namespace
// in cuda_fp8.h and cannot be re-exported via using declarations. Thin inline
// wrappers are provided here so that callers importing this module can access
// them by name within the gpumod namespace.

// ── Narrowing conversions (to fp8 storage) ───────────────────────────────

inline __nv_fp8_storage_t
__nv_cvt_double_to_fp8(const double x, const __nv_saturation_t saturate,
                       const __nv_fp8_interpretation_t fp8_interpretation) {
  return ::__nv_cvt_double_to_fp8(x, saturate, fp8_interpretation);
}

inline __nv_fp8x2_storage_t
__nv_cvt_double2_to_fp8x2(const double2 x, const __nv_saturation_t saturate,
                          const __nv_fp8_interpretation_t fp8_interpretation) {
  return ::__nv_cvt_double2_to_fp8x2(x, saturate, fp8_interpretation);
}

inline __nv_fp8_storage_t
__nv_cvt_float_to_fp8(const float x, const __nv_saturation_t saturate,
                      const __nv_fp8_interpretation_t fp8_interpretation) {
  return ::__nv_cvt_float_to_fp8(x, saturate, fp8_interpretation);
}

inline __nv_fp8x2_storage_t
__nv_cvt_float2_to_fp8x2(const float2 x, const __nv_saturation_t saturate,
                         const __nv_fp8_interpretation_t fp8_interpretation) {
  return ::__nv_cvt_float2_to_fp8x2(x, saturate, fp8_interpretation);
}

inline __nv_fp8_storage_t
__nv_cvt_halfraw_to_fp8(const __half_raw x, const __nv_saturation_t saturate,
                        const __nv_fp8_interpretation_t fp8_interpretation) {
  return ::__nv_cvt_halfraw_to_fp8(x, saturate, fp8_interpretation);
}

inline __nv_fp8x2_storage_t
__nv_cvt_halfraw2_to_fp8x2(const __half2_raw x, const __nv_saturation_t saturate,
                           const __nv_fp8_interpretation_t fp8_interpretation) {
  return ::__nv_cvt_halfraw2_to_fp8x2(x, saturate, fp8_interpretation);
}

inline __nv_fp8_storage_t
__nv_cvt_bfloat16raw_to_fp8(const __nv_bfloat16_raw x, const __nv_saturation_t saturate,
                            const __nv_fp8_interpretation_t fp8_interpretation) {
  return ::__nv_cvt_bfloat16raw_to_fp8(x, saturate, fp8_interpretation);
}

inline __nv_fp8x2_storage_t
__nv_cvt_bfloat16raw2_to_fp8x2(const __nv_bfloat162_raw x, const __nv_saturation_t saturate,
                               const __nv_fp8_interpretation_t fp8_interpretation) {
  return ::__nv_cvt_bfloat16raw2_to_fp8x2(x, saturate, fp8_interpretation);
}

// ── Widening conversions (from fp8 storage) ──────────────────────────────

inline __half_raw __nv_cvt_fp8_to_halfraw(const __nv_fp8_storage_t x,
                                          const __nv_fp8_interpretation_t fp8_interpretation) {
  return ::__nv_cvt_fp8_to_halfraw(x, fp8_interpretation);
}

inline __half2_raw __nv_cvt_fp8x2_to_halfraw2(const __nv_fp8x2_storage_t x,
                                              const __nv_fp8_interpretation_t fp8_interpretation) {
  return ::__nv_cvt_fp8x2_to_halfraw2(x, fp8_interpretation);
}

// ── E8M0 scaling factor conversions ──────────────────────────────────────
// Note: these functions take an additional cudaRoundMode rounding parameter
// compared to the fp8 narrowing conversions above.

inline __nv_fp8_storage_t __nv_cvt_bfloat16raw_to_e8m0(const __nv_bfloat16_raw x,
                                                       const __nv_saturation_t saturate,
                                                       const enum cudaRoundMode rounding) {
  return ::__nv_cvt_bfloat16raw_to_e8m0(x, saturate, rounding);
}

inline __nv_fp8x2_storage_t __nv_cvt_bfloat162raw_to_e8m0x2(const __nv_bfloat162_raw x,
                                                            const __nv_saturation_t saturate,
                                                            const enum cudaRoundMode rounding) {
  return ::__nv_cvt_bfloat162raw_to_e8m0x2(x, saturate, rounding);
}

inline __nv_fp8_storage_t __nv_cvt_float_to_e8m0(const float x, const __nv_saturation_t saturate,
                                                 const enum cudaRoundMode rounding) {
  return ::__nv_cvt_float_to_e8m0(x, saturate, rounding);
}

inline __nv_fp8x2_storage_t __nv_cvt_float2_to_e8m0x2(const float2 x,
                                                      const __nv_saturation_t saturate,
                                                      const enum cudaRoundMode rounding) {
  return ::__nv_cvt_float2_to_e8m0x2(x, saturate, rounding);
}

inline __nv_fp8_storage_t __nv_cvt_double_to_e8m0(const double x, const __nv_saturation_t saturate,
                                                  const enum cudaRoundMode rounding) {
  return ::__nv_cvt_double_to_e8m0(x, saturate, rounding);
}

inline __nv_fp8x2_storage_t __nv_cvt_double2_to_e8m0x2(const double2 x,
                                                       const __nv_saturation_t saturate,
                                                       const enum cudaRoundMode rounding) {
  return ::__nv_cvt_double2_to_e8m0x2(x, saturate, rounding);
}

inline __nv_bfloat16_raw __nv_cvt_e8m0_to_bf16raw(const __nv_fp8_storage_t x) {
  return ::__nv_cvt_e8m0_to_bf16raw(x);
}

inline __nv_bfloat162_raw __nv_cvt_e8m0x2_to_bf162raw(const __nv_fp8x2_storage_t x) {
  return ::__nv_cvt_e8m0x2_to_bf162raw(x);
}

} // namespace gpumod::cuda
