// hip_fp6.cppm - Compile-time tests for gpumod.hip.hip_fp6
//
// IMPORTANT: do not add `import gpumod.hip.hip_fp4;` to this file -- see
// src/hip/hip_fp6.cppm's file header and src/hip/README.md for why hip_fp4
// and hip_fp6 must never land in the same translation unit.

module;

#include "test/shared/link_check.h"

export module gpumod.test.hip.hip_fp6;

import std;
import gpumod.hip.hip_fp6;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time tests for gpumod.hip.hip_fp6
//
// Mirrors test/cuda/cuda_fp6.cppm. HIP's fp6 conversion functions take an
// enum hipRoundMode parameter (not cudaRoundMode) -- see src/hip/hip_fp6.cppm.
// __hip_fp6_storage_t/__hip_fp6x2_storage_t/__hip_fp6x4_storage_t are
// typedef'd to __hip_fp8_storage_t / __hip_fp8x2_storage_t / __hip_fp8x4_storage_t
// -- same 1/2/4-byte shape as CUDA's fp6 storage sizes. The fp6 interpretation
// enum (__hip_fp6_interpretation_t) has two enumerators: __HIP_E3M2 (0) and
// __HIP_E2M3 (1) -- note the reversed numbering vs CUDA's __NV_E2M3 (0) /
// __NV_E3M2 (1), verified against amd_hip_fp6.h directly.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace gpumod::hip::test {

using namespace gpumod::hip;

// ────────────────────────────────────────────────────────────────────────
// Storage typedef sizes
// ────────────────────────────────────────────────────────────────────────

static_assert(sizeof(__hip_fp6_storage_t) == 1);
static_assert(sizeof(__hip_fp6x2_storage_t) == 2);
static_assert(sizeof(__hip_fp6x4_storage_t) == 4);

static_assert(std::is_integral_v<__hip_fp6_storage_t>);
static_assert(std::is_integral_v<__hip_fp6x2_storage_t>);
static_assert(std::is_integral_v<__hip_fp6x4_storage_t>);
static_assert(!std::is_signed_v<__hip_fp6_storage_t>);
static_assert(!std::is_signed_v<__hip_fp6x2_storage_t>);
static_assert(!std::is_signed_v<__hip_fp6x4_storage_t>);

// ────────────────────────────────────────────────────────────────────────
// Enum type checks
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_enum_v<__hip_fp6_interpretation_t>);
static_assert(std::is_enum_v<hipRoundMode>);

// ────────────────────────────────────────────────────────────────────────
// Enum values
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(__HIP_E3M2) == 0);
static_assert(static_cast<int>(__HIP_E2M3) == 1);

static_assert(static_cast<int>(hipRoundNearest) == 0);
static_assert(static_cast<int>(hipRoundZero) == 1);
static_assert(static_cast<int>(hipRoundPosInf) == 2);
static_assert(static_cast<int>(hipRoundMinInf) == 3);

// ────────────────────────────────────────────────────────────────────────
// C++ fp6 struct types: byte sizes, class-ness, standard layout
// ────────────────────────────────────────────────────────────────────────

static_assert(sizeof(__hip_fp6_e3m2) == 1);
static_assert(sizeof(__hip_fp6x2_e3m2) == 2);
static_assert(sizeof(__hip_fp6x4_e3m2) == 4);

static_assert(sizeof(__hip_fp6_e2m3) == 1);
static_assert(sizeof(__hip_fp6x2_e2m3) == 2);
static_assert(sizeof(__hip_fp6x4_e2m3) == 4);

static_assert(std::is_class_v<__hip_fp6_e3m2>);
static_assert(std::is_class_v<__hip_fp6x2_e3m2>);
static_assert(std::is_class_v<__hip_fp6x4_e3m2>);
static_assert(std::is_class_v<__hip_fp6_e2m3>);
static_assert(std::is_class_v<__hip_fp6x2_e2m3>);
static_assert(std::is_class_v<__hip_fp6x4_e2m3>);

static_assert(std::is_standard_layout_v<__hip_fp6_e3m2>);
static_assert(std::is_standard_layout_v<__hip_fp6x2_e3m2>);
static_assert(std::is_standard_layout_v<__hip_fp6x4_e3m2>);
static_assert(std::is_standard_layout_v<__hip_fp6_e2m3>);
static_assert(std::is_standard_layout_v<__hip_fp6x2_e2m3>);
static_assert(std::is_standard_layout_v<__hip_fp6x4_e2m3>);

// ────────────────────────────────────────────────────────────────────────
// Inline wrapper function invocability checks
// GPUMOD_LINK_CHECK is not applicable for inline functions (no external symbol).
// ────────────────────────────────────────────────────────────────────────

// Narrowing conversions (to fp6 storage)
static_assert(std::is_invocable_v<decltype(__hip_cvt_double_to_fp6), double,
                                  __hip_fp6_interpretation_t, hipRoundMode>);

static_assert(std::is_invocable_v<decltype(__hip_cvt_double2_to_fp6x2), double2,
                                  __hip_fp6_interpretation_t, hipRoundMode>);

static_assert(std::is_invocable_v<decltype(__hip_cvt_float_to_fp6), float,
                                  __hip_fp6_interpretation_t, hipRoundMode>);

static_assert(std::is_invocable_v<decltype(__hip_cvt_float2_to_fp6x2), float2,
                                  __hip_fp6_interpretation_t, hipRoundMode>);

static_assert(std::is_invocable_v<decltype(__hip_cvt_halfraw_to_fp6), __half_raw,
                                  __hip_fp6_interpretation_t, hipRoundMode>);

static_assert(std::is_invocable_v<decltype(__hip_cvt_halfraw2_to_fp6x2), __half2_raw,
                                  __hip_fp6_interpretation_t, hipRoundMode>);

static_assert(std::is_invocable_v<decltype(__hip_cvt_bfloat16raw_to_fp6), __hip_bfloat16_raw,
                                  __hip_fp6_interpretation_t, hipRoundMode>);

static_assert(std::is_invocable_v<decltype(__hip_cvt_bfloat16raw2_to_fp6x2), __hip_bfloat162_raw,
                                  __hip_fp6_interpretation_t, hipRoundMode>);

// Widening conversions (from fp6 storage)
static_assert(std::is_invocable_v<decltype(__hip_cvt_fp6_to_halfraw), __hip_fp6_storage_t,
                                  __hip_fp6_interpretation_t>);

static_assert(std::is_invocable_v<decltype(__hip_cvt_fp6x2_to_halfraw2), __hip_fp6x2_storage_t,
                                  __hip_fp6_interpretation_t>);

// ────────────────────────────────────────────────────────────────────────
// Return type checks
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_same_v<std::invoke_result_t<decltype(__hip_cvt_float_to_fp6), float,
                                                  __hip_fp6_interpretation_t, hipRoundMode>,
                             __hip_fp6_storage_t>);

static_assert(std::is_same_v<std::invoke_result_t<decltype(__hip_cvt_float2_to_fp6x2), float2,
                                                  __hip_fp6_interpretation_t, hipRoundMode>,
                             __hip_fp6x2_storage_t>);

static_assert(std::is_same_v<std::invoke_result_t<decltype(__hip_cvt_fp6_to_halfraw),
                                                  __hip_fp6_storage_t, __hip_fp6_interpretation_t>,
                             __half_raw>);

static_assert(
    std::is_same_v<std::invoke_result_t<decltype(__hip_cvt_fp6x2_to_halfraw2),
                                        __hip_fp6x2_storage_t, __hip_fp6_interpretation_t>,
                   __half2_raw>);

} // namespace gpumod::hip::test
