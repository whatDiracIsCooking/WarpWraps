// hip_fp4.cppm - Compile-time tests for gpumod.hip.hip_fp4
//
// IMPORTANT: do not add `import gpumod.hip.hip_fp6;` to this file -- see
// src/hip/hip_fp4.cppm's file header and src/hip/README.md for why hip_fp4
// and hip_fp6 must never land in the same translation unit.

module;

#include "test/shared/link_check.h"

export module gpumod.test.hip.hip_fp4;

import std;
import gpumod.hip.hip_fp4;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time tests for gpumod.hip.hip_fp4
//
// Mirrors test/cuda/cuda_fp4.cppm. HIP's fp4 conversion functions take an
// enum hipRoundMode parameter (not cudaRoundMode) -- see src/hip/hip_fp4.cppm.
// __hip_fp4_storage_t/__hip_fp4x2_storage_t are typedef'd to
// __hip_fp8_storage_t (unsigned char, 1 byte); __hip_fp4x4_storage_t is
// typedef'd to __hip_fp8x2_storage_t (unsigned short, 2 bytes) -- same shape
// as CUDA's fp4 storage sizes.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace wwr::hip::test {

using namespace wwr::hip;

// ────────────────────────────────────────────────────────────────────────
// Storage typedef sizes
// ────────────────────────────────────────────────────────────────────────

static_assert(sizeof(__hip_fp4_storage_t) == 1);
static_assert(sizeof(__hip_fp4x2_storage_t) == 1);
static_assert(sizeof(__hip_fp4x4_storage_t) == 2);

static_assert(std::is_integral_v<__hip_fp4_storage_t>);
static_assert(std::is_integral_v<__hip_fp4x2_storage_t>);
static_assert(std::is_integral_v<__hip_fp4x4_storage_t>);
static_assert(!std::is_signed_v<__hip_fp4_storage_t>);
static_assert(!std::is_signed_v<__hip_fp4x2_storage_t>);
static_assert(!std::is_signed_v<__hip_fp4x4_storage_t>);

// ────────────────────────────────────────────────────────────────────────
// Enum type checks
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_enum_v<__hip_fp4_interpretation_t>);
static_assert(std::is_enum_v<hipRoundMode>);

// ────────────────────────────────────────────────────────────────────────
// Enum values
// ────────────────────────────────────────────────────────────────────────

// __HIP_E2M1 is the sole enumerator, defined as 0
static_assert(static_cast<int>(__HIP_E2M1) == 0);

static_assert(static_cast<int>(hipRoundNearest) == 0);
static_assert(static_cast<int>(hipRoundZero) == 1);
static_assert(static_cast<int>(hipRoundPosInf) == 2);
static_assert(static_cast<int>(hipRoundMinInf) == 3);

// ────────────────────────────────────────────────────────────────────────
// C++ fp4 struct types: byte sizes, class-ness, standard layout
// ────────────────────────────────────────────────────────────────────────

static_assert(sizeof(__hip_fp4_e2m1) == 1);
static_assert(sizeof(__hip_fp4x2_e2m1) == 1);
static_assert(sizeof(__hip_fp4x4_e2m1) == 2);

static_assert(std::is_class_v<__hip_fp4_e2m1>);
static_assert(std::is_class_v<__hip_fp4x2_e2m1>);
static_assert(std::is_class_v<__hip_fp4x4_e2m1>);

static_assert(std::is_standard_layout_v<__hip_fp4_e2m1>);
static_assert(std::is_standard_layout_v<__hip_fp4x2_e2m1>);
static_assert(std::is_standard_layout_v<__hip_fp4x4_e2m1>);

// ────────────────────────────────────────────────────────────────────────
// Inline wrapper function invocability checks
// WWR_LINK_CHECK is not applicable for inline functions (no external symbol).
// ────────────────────────────────────────────────────────────────────────

// Narrowing conversions (to fp4 storage)
static_assert(std::is_invocable_v<decltype(__hip_cvt_double_to_fp4), double,
                                  __hip_fp4_interpretation_t, hipRoundMode>);

static_assert(std::is_invocable_v<decltype(__hip_cvt_double2_to_fp4x2), double2,
                                  __hip_fp4_interpretation_t, hipRoundMode>);

static_assert(std::is_invocable_v<decltype(__hip_cvt_float_to_fp4), float,
                                  __hip_fp4_interpretation_t, hipRoundMode>);

static_assert(std::is_invocable_v<decltype(__hip_cvt_float2_to_fp4x2), float2,
                                  __hip_fp4_interpretation_t, hipRoundMode>);

static_assert(std::is_invocable_v<decltype(__hip_cvt_halfraw_to_fp4), __half_raw,
                                  __hip_fp4_interpretation_t, hipRoundMode>);

static_assert(std::is_invocable_v<decltype(__hip_cvt_halfraw2_to_fp4x2), __half2_raw,
                                  __hip_fp4_interpretation_t, hipRoundMode>);

static_assert(std::is_invocable_v<decltype(__hip_cvt_bfloat16raw_to_fp4), __hip_bfloat16_raw,
                                  __hip_fp4_interpretation_t, hipRoundMode>);

static_assert(std::is_invocable_v<decltype(__hip_cvt_bfloat16raw2_to_fp4x2), __hip_bfloat162_raw,
                                  __hip_fp4_interpretation_t, hipRoundMode>);

// Widening conversions (from fp4 storage)
static_assert(std::is_invocable_v<decltype(__hip_cvt_fp4_to_halfraw), __hip_fp4_storage_t,
                                  __hip_fp4_interpretation_t>);

static_assert(std::is_invocable_v<decltype(__hip_cvt_fp4x2_to_halfraw2), __hip_fp4x2_storage_t,
                                  __hip_fp4_interpretation_t>);

// ────────────────────────────────────────────────────────────────────────
// Return type checks
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_same_v<std::invoke_result_t<decltype(__hip_cvt_float_to_fp4), float,
                                                  __hip_fp4_interpretation_t, hipRoundMode>,
                             __hip_fp4_storage_t>);

static_assert(std::is_same_v<std::invoke_result_t<decltype(__hip_cvt_float2_to_fp4x2), float2,
                                                  __hip_fp4_interpretation_t, hipRoundMode>,
                             __hip_fp4x2_storage_t>);

static_assert(std::is_same_v<std::invoke_result_t<decltype(__hip_cvt_fp4_to_halfraw),
                                                  __hip_fp4_storage_t, __hip_fp4_interpretation_t>,
                             __half_raw>);

static_assert(
    std::is_same_v<std::invoke_result_t<decltype(__hip_cvt_fp4x2_to_halfraw2),
                                        __hip_fp4x2_storage_t, __hip_fp4_interpretation_t>,
                   __half2_raw>);

} // namespace wwr::hip::test
