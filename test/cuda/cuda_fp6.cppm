// cuda_fp6.cppm - Compile-time tests for wwr.cuda.cuda_fp6

module;

#include "test/shared/link_check.h"

export module wwr.test.cuda.cuda_fp6;

import std;
import wwr.cuda.cuda_fp6;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time tests for wwr.cuda.cuda_fp6
//
// We verify at compile-time that:
//   1. Storage typedef sizes match documented byte widths
//   2. Storage typedefs are unsigned integral types
//   3. Enum types are recognized as enums (std::is_enum_v)
//   4. Enum enumerator values match documented integer assignments
//   5. C++ struct types are classes of the correct byte size
//   6. C++ struct types are standard layout (C/CUDA interop guarantee)
//   7. Exported inline wrapper functions are callable with the correct
//      argument types (std::is_invocable_v checks)
//   8. Return types of inline wrapper functions are correct
//
// Note: The conversion wrappers (__nv_cvt_*) are exported as inline
// functions in the wwr namespace. Because they are inline (not
// separately linkable symbols), WWR_LINK_CHECK is not applicable; instead
// std::is_invocable_v verifies the signatures at compile time.
//
// Note: The C++ fp6 struct types (__nv_fp6_e3m2, etc.) have user-defined
// constructors and therefore are NOT trivially constructible. They are,
// however, standard layout.
//
// Note: Unlike fp8, all fp6 narrowing conversions require a cudaRoundMode
// parameter. There is no saturation parameter — out-of-range values always
// saturate to MAXNORM. The fp6 interpretation enum (__nv_fp6_interpretation_t)
// has two enumerators: __NV_E2M3 (0) and __NV_E3M2 (1).
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace wwr::cuda::test {

using namespace wwr::cuda;

// ────────────────────────────────────────────────────────────────────────
// Storage typedef sizes
// ────────────────────────────────────────────────────────────────────────

// __nv_fp6_storage_t  == unsigned char  (1 byte, holds one fp6 value)
static_assert(sizeof(__nv_fp6_storage_t) == 1);
// __nv_fp6x2_storage_t == unsigned short (2 bytes, holds two fp6 values)
static_assert(sizeof(__nv_fp6x2_storage_t) == 2);
// __nv_fp6x4_storage_t == unsigned int   (4 bytes, holds four fp6 values)
static_assert(sizeof(__nv_fp6x4_storage_t) == 4);

// Storage typedefs are integral (unsigned) types
static_assert(std::is_integral_v<__nv_fp6_storage_t>);
static_assert(std::is_integral_v<__nv_fp6x2_storage_t>);
static_assert(std::is_integral_v<__nv_fp6x4_storage_t>);
static_assert(!std::is_signed_v<__nv_fp6_storage_t>);
static_assert(!std::is_signed_v<__nv_fp6x2_storage_t>);
static_assert(!std::is_signed_v<__nv_fp6x4_storage_t>);

// ────────────────────────────────────────────────────────────────────────
// Enum type checks
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_enum_v<__nv_fp6_interpretation_t>);

// ────────────────────────────────────────────────────────────────────────
// Enum values: __nv_fp6_interpretation_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(__NV_E2M3) == 0);
static_assert(static_cast<int>(__NV_E3M2) == 1);

// ────────────────────────────────────────────────────────────────────────
// C++ fp6 struct types: byte sizes
//
// Each scalar type is stored in exactly 1 byte (the upper 2 bits of the
// storage byte are unused; 6 bits hold the fp6 value).
// x2 types occupy exactly 2 bytes (2 fp6 values packed).
// x4 types occupy exactly 4 bytes (4 fp6 values packed).
// ────────────────────────────────────────────────────────────────────────

static_assert(sizeof(__nv_fp6_e3m2) == 1);
static_assert(sizeof(__nv_fp6x2_e3m2) == 2);
static_assert(sizeof(__nv_fp6x4_e3m2) == 4);

static_assert(sizeof(__nv_fp6_e2m3) == 1);
static_assert(sizeof(__nv_fp6x2_e2m3) == 2);
static_assert(sizeof(__nv_fp6x4_e2m3) == 4);

// ────────────────────────────────────────────────────────────────────────
// C++ fp6 struct types: are class types
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_class_v<__nv_fp6_e3m2>);
static_assert(std::is_class_v<__nv_fp6x2_e3m2>);
static_assert(std::is_class_v<__nv_fp6x4_e3m2>);

static_assert(std::is_class_v<__nv_fp6_e2m3>);
static_assert(std::is_class_v<__nv_fp6x2_e2m3>);
static_assert(std::is_class_v<__nv_fp6x4_e2m3>);

// ────────────────────────────────────────────────────────────────────────
// C++ fp6 struct types: standard layout
//
// Standard layout is required for C/CUDA ABI interop. The structs contain
// a single __nv_fp6_storage_t (or x2/x4) data member with no virtual
// functions, so standard layout must hold.
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_standard_layout_v<__nv_fp6_e3m2>);
static_assert(std::is_standard_layout_v<__nv_fp6x2_e3m2>);
static_assert(std::is_standard_layout_v<__nv_fp6x4_e3m2>);

static_assert(std::is_standard_layout_v<__nv_fp6_e2m3>);
static_assert(std::is_standard_layout_v<__nv_fp6x2_e2m3>);
static_assert(std::is_standard_layout_v<__nv_fp6x4_e2m3>);

// ────────────────────────────────────────────────────────────────────────
// Inline wrapper function invocability checks
//
// Each static_assert confirms that the exported inline wrapper in the
// wwr namespace is callable with the expected argument types.
// WWR_LINK_CHECK is not applicable for inline functions (no external symbol).
//
// All narrowing conversions take: (source_type, __nv_fp6_interpretation_t,
//   cudaRoundMode) — note no saturation parameter (unlike fp8).
// Widening conversions take: (storage_type, __nv_fp6_interpretation_t).
// ────────────────────────────────────────────────────────────────────────

// Narrowing conversions (to fp6 storage)
static_assert(std::is_invocable_v<decltype(__nv_cvt_double_to_fp6), double,
                                  __nv_fp6_interpretation_t, cudaRoundMode>);

static_assert(std::is_invocable_v<decltype(__nv_cvt_double2_to_fp6x2), double2,
                                  __nv_fp6_interpretation_t, cudaRoundMode>);

static_assert(std::is_invocable_v<decltype(__nv_cvt_float_to_fp6), float, __nv_fp6_interpretation_t,
                                  cudaRoundMode>);

static_assert(std::is_invocable_v<decltype(__nv_cvt_float2_to_fp6x2), float2,
                                  __nv_fp6_interpretation_t, cudaRoundMode>);

static_assert(std::is_invocable_v<decltype(__nv_cvt_halfraw_to_fp6), __half_raw,
                                  __nv_fp6_interpretation_t, cudaRoundMode>);

static_assert(std::is_invocable_v<decltype(__nv_cvt_halfraw2_to_fp6x2), __half2_raw,
                                  __nv_fp6_interpretation_t, cudaRoundMode>);

static_assert(std::is_invocable_v<decltype(__nv_cvt_bfloat16raw_to_fp6), __nv_bfloat16_raw,
                                  __nv_fp6_interpretation_t, cudaRoundMode>);

static_assert(std::is_invocable_v<decltype(__nv_cvt_bfloat16raw2_to_fp6x2), __nv_bfloat162_raw,
                                  __nv_fp6_interpretation_t, cudaRoundMode>);

// Widening conversions (from fp6 storage)
static_assert(std::is_invocable_v<decltype(__nv_cvt_fp6_to_halfraw), __nv_fp6_storage_t,
                                  __nv_fp6_interpretation_t>);

static_assert(std::is_invocable_v<decltype(__nv_cvt_fp6x2_to_halfraw2), __nv_fp6x2_storage_t,
                                  __nv_fp6_interpretation_t>);

// ────────────────────────────────────────────────────────────────────────
// Return type checks for narrowing wrappers
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_same_v<std::invoke_result_t<decltype(__nv_cvt_float_to_fp6), float,
                                                  __nv_fp6_interpretation_t, cudaRoundMode>,
                             __nv_fp6_storage_t>);

static_assert(std::is_same_v<std::invoke_result_t<decltype(__nv_cvt_float2_to_fp6x2), float2,
                                                  __nv_fp6_interpretation_t, cudaRoundMode>,
                             __nv_fp6x2_storage_t>);

static_assert(std::is_same_v<std::invoke_result_t<decltype(__nv_cvt_double_to_fp6), double,
                                                  __nv_fp6_interpretation_t, cudaRoundMode>,
                             __nv_fp6_storage_t>);

static_assert(std::is_same_v<std::invoke_result_t<decltype(__nv_cvt_double2_to_fp6x2), double2,
                                                  __nv_fp6_interpretation_t, cudaRoundMode>,
                             __nv_fp6x2_storage_t>);

// ────────────────────────────────────────────────────────────────────────
// Return type checks for widening wrappers
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_same_v<std::invoke_result_t<decltype(__nv_cvt_fp6_to_halfraw),
                                                  __nv_fp6_storage_t, __nv_fp6_interpretation_t>,
                             __half_raw>);

static_assert(std::is_same_v<std::invoke_result_t<decltype(__nv_cvt_fp6x2_to_halfraw2),
                                                  __nv_fp6x2_storage_t, __nv_fp6_interpretation_t>,
                             __half2_raw>);

} // namespace wwr::cuda::test
