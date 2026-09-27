/**
 * @file cuda_fp6.cppm
 * @brief CUDA FP6 Floating-Point API module wrapper
 *
 * This module wraps the cuda_fp6.h header for use in C++20/C++23 module-based code.
 * It exports 6-bit floating-point types, enumerations, storage typedefs, and
 * host-accessible conversion functions.
 *
 * Usage:
 *   import wwr.cuda.cuda_fp6;
 *
 * Note: cuda_fp6.h includes cuda_fp8.h internally (which in turn includes
 * cuda_fp16.h and cuda_bf16.h), so the raw storage types from those headers
 * (__half_raw, __nv_bfloat16_raw, etc.) are available to the conversion
 * functions. This module exports the fp6-specific types and enumerations.
 * The C-style conversion functions (e.g., __nv_cvt_float_to_fp6) are declared
 * as static inline and therefore cannot be re-exported via using declarations;
 * they are wrapped as thin inline forwarders in the export namespace.
 *
 * The C++ struct types (__nv_fp6_e3m2, __nv_fp6_e2m3, and their x2/x4 vector
 * variants) are fully exported and usable on the host.
 *
 * Operators and converting constructors work automatically via Argument-Dependent
 * Lookup (ADL) when using the exported types.
 */

module;

#include <cuda_fp6.h>

export module wwr.cuda.cuda_fp6;

import std;

export namespace wwr::cuda {

// ========================================================================
// Storage Typedefs
// ========================================================================

// Underlying unsigned integer storage for a single fp6 value (1 byte)
using ::__nv_fp6_storage_t;

// Underlying unsigned integer storage for a pair of fp6 values (2 bytes)
using ::__nv_fp6x2_storage_t;

// Underlying unsigned integer storage for four fp6 values (4 bytes)
using ::__nv_fp6x4_storage_t;

// ========================================================================
// Enumerations
// ========================================================================

// Interpretation (format kind) of an fp6 value
using ::__nv_fp6_interpretation_t;

// Enumerators of __nv_fp6_interpretation_t
using ::__NV_E2M3;
using ::__NV_E3M2;

// ========================================================================
// C++ FP6 Struct Types (E3M2 format)
// ========================================================================

// Single 6-bit float (E3M2: 3 exponent bits, 2 mantissa bits), stored in 1 byte
using ::__nv_fp6_e3m2;

// Vector of two E3M2 fp6 values packed into 2 bytes
using ::__nv_fp6x2_e3m2;

// Vector of four E3M2 fp6 values packed into 4 bytes
using ::__nv_fp6x4_e3m2;

// ========================================================================
// C++ FP6 Struct Types (E2M3 format)
// ========================================================================

// Single 6-bit float (E2M3: 2 exponent bits, 3 mantissa bits), stored in 1 byte
using ::__nv_fp6_e2m3;

// Vector of two E2M3 fp6 values packed into 2 bytes
using ::__nv_fp6x2_e2m3;

// Vector of four E2M3 fp6 values packed into 4 bytes
using ::__nv_fp6x4_e2m3;

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
// Rounding Mode (from device_types.h, included transitively via cuda_fp6.h)
// ========================================================================

// Rounding mode used by fp6 conversion functions
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
// in cuda_fp6.h and cannot be re-exported via using declarations. Thin inline
// wrappers are provided here so that callers importing this module can access
// them by name within the wwr namespace.
//
// Note: Unlike the fp8 narrowing conversions, all fp6 narrowing conversions
// require an explicit cudaRoundMode rounding parameter. The fp6 header does
// not define a saturation parameter — out-of-range values always saturate to
// MAXNORM.

// ── Narrowing conversions (to fp6 storage) ───────────────────────────────

inline __nv_fp6_storage_t __nv_cvt_double_to_fp6(const double x,
                                                 const __nv_fp6_interpretation_t fp6_interpretation,
                                                 const enum cudaRoundMode rounding) {
  return ::__nv_cvt_double_to_fp6(x, fp6_interpretation, rounding);
}

inline __nv_fp6x2_storage_t
__nv_cvt_double2_to_fp6x2(const double2 x, const __nv_fp6_interpretation_t fp6_interpretation,
                          const enum cudaRoundMode rounding) {
  return ::__nv_cvt_double2_to_fp6x2(x, fp6_interpretation, rounding);
}

inline __nv_fp6_storage_t __nv_cvt_float_to_fp6(const float x,
                                                const __nv_fp6_interpretation_t fp6_interpretation,
                                                const enum cudaRoundMode rounding) {
  return ::__nv_cvt_float_to_fp6(x, fp6_interpretation, rounding);
}

inline __nv_fp6x2_storage_t
__nv_cvt_float2_to_fp6x2(const float2 x, const __nv_fp6_interpretation_t fp6_interpretation,
                         const enum cudaRoundMode rounding) {
  return ::__nv_cvt_float2_to_fp6x2(x, fp6_interpretation, rounding);
}

inline __nv_fp6_storage_t
__nv_cvt_halfraw_to_fp6(const __half_raw x, const __nv_fp6_interpretation_t fp6_interpretation,
                        const enum cudaRoundMode rounding) {
  return ::__nv_cvt_halfraw_to_fp6(x, fp6_interpretation, rounding);
}

inline __nv_fp6x2_storage_t
__nv_cvt_halfraw2_to_fp6x2(const __half2_raw x, const __nv_fp6_interpretation_t fp6_interpretation,
                           const enum cudaRoundMode rounding) {
  return ::__nv_cvt_halfraw2_to_fp6x2(x, fp6_interpretation, rounding);
}

inline __nv_fp6_storage_t
__nv_cvt_bfloat16raw_to_fp6(const __nv_bfloat16_raw x,
                            const __nv_fp6_interpretation_t fp6_interpretation,
                            const enum cudaRoundMode rounding) {
  return ::__nv_cvt_bfloat16raw_to_fp6(x, fp6_interpretation, rounding);
}

inline __nv_fp6x2_storage_t
__nv_cvt_bfloat16raw2_to_fp6x2(const __nv_bfloat162_raw x,
                               const __nv_fp6_interpretation_t fp6_interpretation,
                               const enum cudaRoundMode rounding) {
  return ::__nv_cvt_bfloat16raw2_to_fp6x2(x, fp6_interpretation, rounding);
}

// ── Widening conversions (from fp6 storage) ──────────────────────────────

inline __half_raw __nv_cvt_fp6_to_halfraw(const __nv_fp6_storage_t x,
                                          const __nv_fp6_interpretation_t fp6_interpretation) {
  return ::__nv_cvt_fp6_to_halfraw(x, fp6_interpretation);
}

inline __half2_raw __nv_cvt_fp6x2_to_halfraw2(const __nv_fp6x2_storage_t x,
                                              const __nv_fp6_interpretation_t fp6_interpretation) {
  return ::__nv_cvt_fp6x2_to_halfraw2(x, fp6_interpretation);
}

} // namespace wwr::cuda
