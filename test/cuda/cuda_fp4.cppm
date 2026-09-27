// cuda_fp4.cppm - Compile-time tests for wwr.cuda.cuda_fp4

module;

#include "test/shared/link_check.h"

export module wwr.test.cuda.cuda_fp4;

import std;
import wwr.cuda.cuda_fp4;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time tests for wwr.cuda.cuda_fp4
//
// We verify at compile-time that:
//   1. Storage typedef sizes match documented byte widths
//   2. Storage typedefs are integral unsigned types
//   3. Enum type is recognized as an enum (std::is_enum_v)
//   4. Enum enumerator value matches the documented integer assignment
//   5. C++ struct types are classes of the correct byte size
//   6. C++ struct types are standard layout (C/CUDA interop guarantee)
//   7. Exported inline wrapper functions are callable with the correct
//      argument types (std::is_invocable_v checks)
//   8. Return types of key wrapper functions match the expected storage types
//
// Note: The conversion wrappers (__nv_cvt_*) are exported as inline
// functions in the wwr namespace. Because they are inline (not
// separately linkable symbols), WWR_LINK_CHECK is not applicable; instead
// std::is_invocable_v verifies the signatures at compile time.
//
// Note: The C++ fp4 struct types (__nv_fp4_e2m1, etc.) have user-defined
// constructors and therefore are NOT trivially constructible. They are,
// however, standard layout.
//
// Note: __nv_fp4_storage_t and __nv_fp4x2_storage_t are both typedef'd to
// unsigned char (1 byte each). Two fp4 values share one byte via nibble
// packing, so both single and x2 storage types are 1 byte. The x4 storage
// type (__nv_fp4x4_storage_t) is typedef'd to unsigned short (2 bytes).
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace wwr::cuda::test {

using namespace wwr::cuda;

// ────────────────────────────────────────────────────────────────────────
// Storage typedef sizes
// ────────────────────────────────────────────────────────────────────────

// __nv_fp4_storage_t   == unsigned char  (1 byte, holds one fp4 value in lower nibble)
static_assert(sizeof(__nv_fp4_storage_t) == 1);
// __nv_fp4x2_storage_t == unsigned char  (1 byte, holds two fp4 values in upper/lower nibbles)
static_assert(sizeof(__nv_fp4x2_storage_t) == 1);
// __nv_fp4x4_storage_t == unsigned short (2 bytes, holds four fp4 values across two nibble pairs)
static_assert(sizeof(__nv_fp4x4_storage_t) == 2);

// Storage typedefs are integral (unsigned) types
static_assert(std::is_integral_v<__nv_fp4_storage_t>);
static_assert(std::is_integral_v<__nv_fp4x2_storage_t>);
static_assert(std::is_integral_v<__nv_fp4x4_storage_t>);
static_assert(!std::is_signed_v<__nv_fp4_storage_t>);
static_assert(!std::is_signed_v<__nv_fp4x2_storage_t>);
static_assert(!std::is_signed_v<__nv_fp4x4_storage_t>);

// ────────────────────────────────────────────────────────────────────────
// Enum type check
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_enum_v<__nv_fp4_interpretation_t>);

// ────────────────────────────────────────────────────────────────────────
// Enum values: __nv_fp4_interpretation_t
// ────────────────────────────────────────────────────────────────────────

// __NV_E2M1 is the sole enumerator, defined as 0
static_assert(static_cast<int>(__NV_E2M1) == 0);

// ────────────────────────────────────────────────────────────────────────
// C++ fp4 struct types: byte sizes
//
// __nv_fp4_e2m1   occupies 1 byte  (single fp4 value, lower nibble of storage)
// __nv_fp4x2_e2m1 occupies 1 byte  (two fp4 values packed in upper/lower nibbles)
// __nv_fp4x4_e2m1 occupies 2 bytes (four fp4 values packed across two bytes)
// ────────────────────────────────────────────────────────────────────────

static_assert(sizeof(__nv_fp4_e2m1) == 1);
static_assert(sizeof(__nv_fp4x2_e2m1) == 1);
static_assert(sizeof(__nv_fp4x4_e2m1) == 2);

// ────────────────────────────────────────────────────────────────────────
// C++ fp4 struct types: are class types
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_class_v<__nv_fp4_e2m1>);
static_assert(std::is_class_v<__nv_fp4x2_e2m1>);
static_assert(std::is_class_v<__nv_fp4x4_e2m1>);

// ────────────────────────────────────────────────────────────────────────
// C++ fp4 struct types: standard layout
//
// Standard layout is required for C/CUDA ABI interop. The structs contain
// a single __nv_fp4_storage_t (or x2/x4) data member with no virtual
// functions, so standard layout must hold.
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_standard_layout_v<__nv_fp4_e2m1>);
static_assert(std::is_standard_layout_v<__nv_fp4x2_e2m1>);
static_assert(std::is_standard_layout_v<__nv_fp4x4_e2m1>);

// ────────────────────────────────────────────────────────────────────────
// Inline wrapper function invocability checks
//
// Each static_assert confirms that the exported inline wrapper in the
// wwr namespace is callable with the expected argument types.
// WWR_LINK_CHECK is not applicable for inline functions (no external symbol).
//
// All fp4 narrowing conversions take (value, interpretation, rounding) —
// note the rounding parameter (cudaRoundMode) instead of the saturation
// parameter (__nv_saturation_t) used by fp8.  Saturation to MAXNORM is
// always implicit in the fp4 conversion functions.
// ────────────────────────────────────────────────────────────────────────

// Narrowing conversions (to fp4 storage)
static_assert(std::is_invocable_v<decltype(__nv_cvt_double_to_fp4), double,
                                  __nv_fp4_interpretation_t, cudaRoundMode>);

static_assert(std::is_invocable_v<decltype(__nv_cvt_double2_to_fp4x2), double2,
                                  __nv_fp4_interpretation_t, cudaRoundMode>);

static_assert(std::is_invocable_v<decltype(__nv_cvt_float_to_fp4), float, __nv_fp4_interpretation_t,
                                  cudaRoundMode>);

static_assert(std::is_invocable_v<decltype(__nv_cvt_float2_to_fp4x2), float2,
                                  __nv_fp4_interpretation_t, cudaRoundMode>);

static_assert(std::is_invocable_v<decltype(__nv_cvt_halfraw_to_fp4), __half_raw,
                                  __nv_fp4_interpretation_t, cudaRoundMode>);

static_assert(std::is_invocable_v<decltype(__nv_cvt_halfraw2_to_fp4x2), __half2_raw,
                                  __nv_fp4_interpretation_t, cudaRoundMode>);

static_assert(std::is_invocable_v<decltype(__nv_cvt_bfloat16raw_to_fp4), __nv_bfloat16_raw,
                                  __nv_fp4_interpretation_t, cudaRoundMode>);

static_assert(std::is_invocable_v<decltype(__nv_cvt_bfloat16raw2_to_fp4x2), __nv_bfloat162_raw,
                                  __nv_fp4_interpretation_t, cudaRoundMode>);

// Widening conversions (from fp4 storage)
static_assert(std::is_invocable_v<decltype(__nv_cvt_fp4_to_halfraw), __nv_fp4_storage_t,
                                  __nv_fp4_interpretation_t>);

static_assert(std::is_invocable_v<decltype(__nv_cvt_fp4x2_to_halfraw2), __nv_fp4x2_storage_t,
                                  __nv_fp4_interpretation_t>);

// ────────────────────────────────────────────────────────────────────────
// Return type checks for narrowing wrappers
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_same_v<std::invoke_result_t<decltype(__nv_cvt_float_to_fp4), float,
                                                  __nv_fp4_interpretation_t, cudaRoundMode>,
                             __nv_fp4_storage_t>);

static_assert(std::is_same_v<std::invoke_result_t<decltype(__nv_cvt_float2_to_fp4x2), float2,
                                                  __nv_fp4_interpretation_t, cudaRoundMode>,
                             __nv_fp4x2_storage_t>);

static_assert(std::is_same_v<std::invoke_result_t<decltype(__nv_cvt_double_to_fp4), double,
                                                  __nv_fp4_interpretation_t, cudaRoundMode>,
                             __nv_fp4_storage_t>);

static_assert(std::is_same_v<std::invoke_result_t<decltype(__nv_cvt_double2_to_fp4x2), double2,
                                                  __nv_fp4_interpretation_t, cudaRoundMode>,
                             __nv_fp4x2_storage_t>);

// ────────────────────────────────────────────────────────────────────────
// Return type checks for widening wrappers
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_same_v<std::invoke_result_t<decltype(__nv_cvt_fp4_to_halfraw),
                                                  __nv_fp4_storage_t, __nv_fp4_interpretation_t>,
                             __half_raw>);

static_assert(std::is_same_v<std::invoke_result_t<decltype(__nv_cvt_fp4x2_to_halfraw2),
                                                  __nv_fp4x2_storage_t, __nv_fp4_interpretation_t>,
                             __half2_raw>);

} // namespace wwr::cuda::test
