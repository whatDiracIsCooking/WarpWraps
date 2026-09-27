// hip_fp8.cppm - Compile-time tests for wwr.hip.hip_fp8

module;

#include "test/shared/link_check.h"

export module wwr.test.hip.hip_fp8;

import std;
import wwr.hip.hip_fp8;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time tests for wwr.hip.hip_fp8
//
// Mirrors test/cuda/cuda_fp8.cppm's structure. Unlike CUDA (e4m3/e5m2/e8m0),
// HIP defines four struct formats: OCP e4m3/e5m2 and AMD fnuz-encoded
// e4m3/e5m2, each with x2/x4 vector variants -- no e8m0 counterpart exists
// here. The __hip_cvt_* conversion wrappers are exported as inline functions
// (not separately linkable symbols), so std::is_invocable_v verifies their
// signatures at compile time instead of WWR_LINK_CHECK.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace wwr::hip::test {

using namespace wwr::hip;

// ────────────────────────────────────────────────────────────────────────
// Storage typedef sizes
// ────────────────────────────────────────────────────────────────────────

static_assert(sizeof(__hip_fp8_storage_t) == 1);
static_assert(sizeof(__hip_fp8x2_storage_t) == 2);
static_assert(sizeof(__hip_fp8x4_storage_t) == 4);

static_assert(std::is_integral_v<__hip_fp8_storage_t>);
static_assert(std::is_integral_v<__hip_fp8x2_storage_t>);
static_assert(std::is_integral_v<__hip_fp8x4_storage_t>);
static_assert(!std::is_signed_v<__hip_fp8_storage_t>);
static_assert(!std::is_signed_v<__hip_fp8x2_storage_t>);
static_assert(!std::is_signed_v<__hip_fp8x4_storage_t>);

// ────────────────────────────────────────────────────────────────────────
// Enum type checks
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_enum_v<__hip_saturation_t>);
static_assert(std::is_enum_v<__hip_fp8_interpretation_t>);

// ────────────────────────────────────────────────────────────────────────
// Enum values: __hip_saturation_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(__HIP_NOSAT) == 0);
static_assert(static_cast<int>(__HIP_SATFINITE) == 1);

// ────────────────────────────────────────────────────────────────────────
// Enum values: __hip_fp8_interpretation_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(__HIP_E4M3) == 0);
static_assert(static_cast<int>(__HIP_E5M2) == 1);
static_assert(static_cast<int>(__HIP_E4M3_FNUZ) == 2);
static_assert(static_cast<int>(__HIP_E5M2_FNUZ) == 3);

// ────────────────────────────────────────────────────────────────────────
// C++ fp8 struct types: byte sizes
// ────────────────────────────────────────────────────────────────────────

static_assert(sizeof(__hip_fp8_e4m3) == 1);
static_assert(sizeof(__hip_fp8x2_e4m3) == 2);
static_assert(sizeof(__hip_fp8x4_e4m3) == 4);

static_assert(sizeof(__hip_fp8_e5m2) == 1);
static_assert(sizeof(__hip_fp8x2_e5m2) == 2);
static_assert(sizeof(__hip_fp8x4_e5m2) == 4);

static_assert(sizeof(__hip_fp8_e4m3_fnuz) == 1);
static_assert(sizeof(__hip_fp8x2_e4m3_fnuz) == 2);
static_assert(sizeof(__hip_fp8x4_e4m3_fnuz) == 4);

static_assert(sizeof(__hip_fp8_e5m2_fnuz) == 1);
static_assert(sizeof(__hip_fp8x2_e5m2_fnuz) == 2);
static_assert(sizeof(__hip_fp8x4_e5m2_fnuz) == 4);

// ────────────────────────────────────────────────────────────────────────
// C++ fp8 struct types: are class types, standard layout
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_class_v<__hip_fp8_e4m3>);
static_assert(std::is_class_v<__hip_fp8x2_e4m3>);
static_assert(std::is_class_v<__hip_fp8x4_e4m3>);
static_assert(std::is_class_v<__hip_fp8_e5m2>);
static_assert(std::is_class_v<__hip_fp8x2_e5m2>);
static_assert(std::is_class_v<__hip_fp8x4_e5m2>);
static_assert(std::is_class_v<__hip_fp8_e4m3_fnuz>);
static_assert(std::is_class_v<__hip_fp8x2_e4m3_fnuz>);
static_assert(std::is_class_v<__hip_fp8x4_e4m3_fnuz>);
static_assert(std::is_class_v<__hip_fp8_e5m2_fnuz>);
static_assert(std::is_class_v<__hip_fp8x2_e5m2_fnuz>);
static_assert(std::is_class_v<__hip_fp8x4_e5m2_fnuz>);

static_assert(std::is_standard_layout_v<__hip_fp8_e4m3>);
static_assert(std::is_standard_layout_v<__hip_fp8x2_e4m3>);
static_assert(std::is_standard_layout_v<__hip_fp8x4_e4m3>);
static_assert(std::is_standard_layout_v<__hip_fp8_e5m2>);
static_assert(std::is_standard_layout_v<__hip_fp8x2_e5m2>);
static_assert(std::is_standard_layout_v<__hip_fp8x4_e5m2>);
static_assert(std::is_standard_layout_v<__hip_fp8_e4m3_fnuz>);
static_assert(std::is_standard_layout_v<__hip_fp8x2_e4m3_fnuz>);
static_assert(std::is_standard_layout_v<__hip_fp8x4_e4m3_fnuz>);
static_assert(std::is_standard_layout_v<__hip_fp8_e5m2_fnuz>);
static_assert(std::is_standard_layout_v<__hip_fp8x2_e5m2_fnuz>);
static_assert(std::is_standard_layout_v<__hip_fp8x4_e5m2_fnuz>);

// ────────────────────────────────────────────────────────────────────────
// Inline wrapper function invocability checks
// WWR_LINK_CHECK is not applicable for inline functions (no external symbol).
// ────────────────────────────────────────────────────────────────────────

// Narrowing conversions (to fp8 storage)
static_assert(std::is_invocable_v<decltype(__hip_cvt_double_to_fp8), double, __hip_saturation_t,
                                  __hip_fp8_interpretation_t>);

static_assert(std::is_invocable_v<decltype(__hip_cvt_double2_to_fp8x2), double2, __hip_saturation_t,
                                  __hip_fp8_interpretation_t>);

static_assert(std::is_invocable_v<decltype(__hip_cvt_float_to_fp8), float, __hip_saturation_t,
                                  __hip_fp8_interpretation_t>);

static_assert(std::is_invocable_v<decltype(__hip_cvt_float2_to_fp8x2), float2, __hip_saturation_t,
                                  __hip_fp8_interpretation_t>);

static_assert(std::is_invocable_v<decltype(__hip_cvt_halfraw_to_fp8), __half_raw,
                                  __hip_saturation_t, __hip_fp8_interpretation_t>);

static_assert(std::is_invocable_v<decltype(__hip_cvt_halfraw2_to_fp8x2), __half2_raw,
                                  __hip_saturation_t, __hip_fp8_interpretation_t>);

static_assert(std::is_invocable_v<decltype(__hip_cvt_bfloat16raw_to_fp8), __hip_bfloat16_raw,
                                  __hip_saturation_t, __hip_fp8_interpretation_t>);

static_assert(std::is_invocable_v<decltype(__hip_cvt_bfloat16raw2_to_fp8x2), __hip_bfloat162_raw,
                                  __hip_saturation_t, __hip_fp8_interpretation_t>);

// Widening conversions (from fp8 storage)
static_assert(std::is_invocable_v<decltype(__hip_cvt_fp8_to_halfraw), __hip_fp8_storage_t,
                                  __hip_fp8_interpretation_t>);

static_assert(std::is_invocable_v<decltype(__hip_cvt_fp8x2_to_halfraw2), __hip_fp8x2_storage_t,
                                  __hip_fp8_interpretation_t>);

// Verify return types of narrowing wrappers
static_assert(std::is_same_v<std::invoke_result_t<decltype(__hip_cvt_float_to_fp8), float,
                                                  __hip_saturation_t, __hip_fp8_interpretation_t>,
                             __hip_fp8_storage_t>);

static_assert(std::is_same_v<std::invoke_result_t<decltype(__hip_cvt_float2_to_fp8x2), float2,
                                                  __hip_saturation_t, __hip_fp8_interpretation_t>,
                             __hip_fp8x2_storage_t>);

// Verify return types of widening wrappers
static_assert(std::is_same_v<std::invoke_result_t<decltype(__hip_cvt_fp8_to_halfraw),
                                                  __hip_fp8_storage_t, __hip_fp8_interpretation_t>,
                             __half_raw>);

static_assert(
    std::is_same_v<std::invoke_result_t<decltype(__hip_cvt_fp8x2_to_halfraw2),
                                        __hip_fp8x2_storage_t, __hip_fp8_interpretation_t>,
                   __half2_raw>);

} // namespace wwr::hip::test
