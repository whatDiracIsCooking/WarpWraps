/**
 * @file cuda_fp4.cppm
 * @brief CUDA FP4 Floating-Point API module wrapper
 *
 * This module wraps the cuda_fp4.h header for use in C++20/C++23 module-based code.
 * It exports 4-bit floating-point types, enumerations, storage typedefs, and
 * host-accessible conversion functions.
 *
 * Usage:
 *   import gpumod.cuda.cuda_fp4;
 *
 * Note: cuda_fp4.h includes cuda_fp6.h which includes cuda_fp8.h, so the raw
 * storage and half-precision types (__half_raw, __nv_bfloat16_raw, etc.) are
 * available to the conversion functions through transitive includes. This module
 * exports the fp4-specific types and enumerations along with re-exports of the
 * dependent raw types needed to call the conversion wrappers.
 *
 * The C-style conversion functions (e.g., __nv_cvt_float_to_fp4) are declared
 * as static inline and therefore cannot be re-exported via using declarations;
 * they are wrapped as thin inline forwarders in the export namespace.
 *
 * The C++ struct types (__nv_fp4_e2m1, __nv_fp4x2_e2m1, __nv_fp4x4_e2m1) are
 * fully exported and usable on the host.
 *
 * Operators and converting constructors work automatically via Argument-Dependent
 * Lookup (ADL) when using the exported types.
 */

module;

#include <cuda_fp4.h>

export module gpumod.cuda.cuda_fp4;

import std;

export namespace wwr::cuda {

// ========================================================================
// Storage Typedefs
// ========================================================================

// Underlying unsigned integer storage for a single fp4 value (1 byte).
// Two fp4 values are packed into the upper and lower nibbles of this byte.
using ::__nv_fp4_storage_t;

// Underlying unsigned integer storage for a pair of fp4 values (1 byte).
// The two fp4 values occupy the upper nibble (bits [7:4]) and lower nibble
// (bits [3:0]) of the byte.
using ::__nv_fp4x2_storage_t;

// Underlying unsigned integer storage for four fp4 values (2 bytes).
// Pairs of nibbles are packed across two bytes.
using ::__nv_fp4x4_storage_t;

// ========================================================================
// Enumerations
// ========================================================================

// Interpretation (format kind) of an fp4 value
using ::__nv_fp4_interpretation_t;

// Enumerators of __nv_fp4_interpretation_t
// __NV_E2M1: fp4 numbers with 1 sign bit, 2 exponent bits, 1 mantissa bit
using ::__NV_E2M1;

// ========================================================================
// C++ FP4 Struct Types (E2M1 format)
// ========================================================================

// Single 4-bit float (E2M1: 1 sign, 2 exponent bits, 1 mantissa bit).
// Stored in the lower nibble of a 1-byte __nv_fp4_storage_t.
using ::__nv_fp4_e2m1;

// Vector of two E2M1 fp4 values packed into 1 byte (upper and lower nibbles).
using ::__nv_fp4x2_e2m1;

// Vector of four E2M1 fp4 values packed into 2 bytes.
using ::__nv_fp4x4_e2m1;

// ========================================================================
// Dependent Raw Types (from cuda_fp8.h / cuda_fp16.h / cuda_bf16.h)
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

// Rounding mode used by fp4 conversion functions
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
// in cuda_fp4.h (via cuda_fp4.hpp) and cannot be re-exported via using
// declarations. Thin inline wrappers are provided here so that callers
// importing this module can access them by name within the wwr namespace.
//
// Note: Unlike the fp8 narrowing conversion functions (which take
// __nv_saturation_t), the fp4 conversion functions take a cudaRoundMode
// rounding parameter. The fp4 format always saturates out-of-range values to
// MAXNORM — saturation is implicit.

// ── Narrowing conversions (to fp4 storage) ───────────────────────────────

inline __nv_fp4_storage_t __nv_cvt_double_to_fp4(const double x,
                                                 const __nv_fp4_interpretation_t fp4_interpretation,
                                                 const enum cudaRoundMode rounding) {
  return ::__nv_cvt_double_to_fp4(x, fp4_interpretation, rounding);
}

inline __nv_fp4x2_storage_t
__nv_cvt_double2_to_fp4x2(const double2 x, const __nv_fp4_interpretation_t fp4_interpretation,
                          const enum cudaRoundMode rounding) {
  return ::__nv_cvt_double2_to_fp4x2(x, fp4_interpretation, rounding);
}

inline __nv_fp4_storage_t __nv_cvt_float_to_fp4(const float x,
                                                const __nv_fp4_interpretation_t fp4_interpretation,
                                                const enum cudaRoundMode rounding) {
  return ::__nv_cvt_float_to_fp4(x, fp4_interpretation, rounding);
}

inline __nv_fp4x2_storage_t
__nv_cvt_float2_to_fp4x2(const float2 x, const __nv_fp4_interpretation_t fp4_interpretation,
                         const enum cudaRoundMode rounding) {
  return ::__nv_cvt_float2_to_fp4x2(x, fp4_interpretation, rounding);
}

inline __nv_fp4_storage_t
__nv_cvt_halfraw_to_fp4(const __half_raw x, const __nv_fp4_interpretation_t fp4_interpretation,
                        const enum cudaRoundMode rounding) {
  return ::__nv_cvt_halfraw_to_fp4(x, fp4_interpretation, rounding);
}

inline __nv_fp4x2_storage_t
__nv_cvt_halfraw2_to_fp4x2(const __half2_raw x, const __nv_fp4_interpretation_t fp4_interpretation,
                           const enum cudaRoundMode rounding) {
  return ::__nv_cvt_halfraw2_to_fp4x2(x, fp4_interpretation, rounding);
}

inline __nv_fp4_storage_t
__nv_cvt_bfloat16raw_to_fp4(const __nv_bfloat16_raw x,
                            const __nv_fp4_interpretation_t fp4_interpretation,
                            const enum cudaRoundMode rounding) {
  return ::__nv_cvt_bfloat16raw_to_fp4(x, fp4_interpretation, rounding);
}

inline __nv_fp4x2_storage_t
__nv_cvt_bfloat16raw2_to_fp4x2(const __nv_bfloat162_raw x,
                               const __nv_fp4_interpretation_t fp4_interpretation,
                               const enum cudaRoundMode rounding) {
  return ::__nv_cvt_bfloat16raw2_to_fp4x2(x, fp4_interpretation, rounding);
}

// ── Widening conversions (from fp4 storage) ──────────────────────────────

inline __half_raw __nv_cvt_fp4_to_halfraw(const __nv_fp4_storage_t x,
                                          const __nv_fp4_interpretation_t fp4_interpretation) {
  return ::__nv_cvt_fp4_to_halfraw(x, fp4_interpretation);
}

inline __half2_raw __nv_cvt_fp4x2_to_halfraw2(const __nv_fp4x2_storage_t x,
                                              const __nv_fp4_interpretation_t fp4_interpretation) {
  return ::__nv_cvt_fp4x2_to_halfraw2(x, fp4_interpretation);
}

} // namespace wwr::cuda
