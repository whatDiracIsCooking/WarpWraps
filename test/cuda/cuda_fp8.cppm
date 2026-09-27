// cuda_fp8.cppm - Compile-time tests for gpumod.cuda.cuda_fp8

module;

#include "test/shared/link_check.h"

export module gpumod.test.cuda.cuda_fp8;

import std;
import gpumod.cuda.cuda_fp8;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time tests for gpumod.cuda.cuda_fp8
//
// We verify at compile-time that:
//   1. Storage typedef sizes match documented byte widths
//   2. Enum types are recognized as enums (std::is_enum_v)
//   3. Enum enumerator values match documented integer assignments
//   4. C++ struct types are classes of the correct byte size
//   5. C++ struct types are standard layout (C/CUDA interop guarantee)
//   6. Exported inline wrapper functions are callable with the correct
//      argument types (std::is_invocable_v checks)
//
// Note: The conversion wrappers (__nv_cvt_*) are exported as inline
// functions in the wwr namespace. Because they are inline (not
// separately linkable symbols), WWR_LINK_CHECK is not applicable; instead
// std::is_invocable_v verifies the signatures at compile time.
//
// Note: The C++ fp8 struct types (__nv_fp8_e4m3, etc.) have user-defined
// constructors and therefore are NOT trivially constructible. They are,
// however, standard layout.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace wwr::cuda::test {

using namespace wwr::cuda;

// ────────────────────────────────────────────────────────────────────────
// Storage typedef sizes
// ────────────────────────────────────────────────────────────────────────

// __nv_fp8_storage_t  == unsigned char  (1 byte, holds one fp8 value)
static_assert(sizeof(__nv_fp8_storage_t) == 1);
// __nv_fp8x2_storage_t == unsigned short (2 bytes, holds two fp8 values)
static_assert(sizeof(__nv_fp8x2_storage_t) == 2);
// __nv_fp8x4_storage_t == unsigned int   (4 bytes, holds four fp8 values)
static_assert(sizeof(__nv_fp8x4_storage_t) == 4);

// Storage typedefs are integral (unsigned) types
static_assert(std::is_integral_v<__nv_fp8_storage_t>);
static_assert(std::is_integral_v<__nv_fp8x2_storage_t>);
static_assert(std::is_integral_v<__nv_fp8x4_storage_t>);
static_assert(!std::is_signed_v<__nv_fp8_storage_t>);
static_assert(!std::is_signed_v<__nv_fp8x2_storage_t>);
static_assert(!std::is_signed_v<__nv_fp8x4_storage_t>);

// ────────────────────────────────────────────────────────────────────────
// Enum type checks
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_enum_v<__nv_saturation_t>);
static_assert(std::is_enum_v<__nv_fp8_interpretation_t>);

// ────────────────────────────────────────────────────────────────────────
// Enum values: __nv_saturation_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(__NV_NOSAT) == 0);
static_assert(static_cast<int>(__NV_SATFINITE) == 1);

// ────────────────────────────────────────────────────────────────────────
// Enum values: __nv_fp8_interpretation_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(__NV_E4M3) == 0);
static_assert(static_cast<int>(__NV_E5M2) == 1);

// ────────────────────────────────────────────────────────────────────────
// C++ fp8 struct types: byte sizes
//
// Each scalar type occupies exactly 1 byte (1 fp8 value).
// x2 types occupy exactly 2 bytes (2 fp8 values packed).
// x4 types occupy exactly 4 bytes (4 fp8 values packed).
// ────────────────────────────────────────────────────────────────────────

static_assert(sizeof(__nv_fp8_e5m2) == 1);
static_assert(sizeof(__nv_fp8x2_e5m2) == 2);
static_assert(sizeof(__nv_fp8x4_e5m2) == 4);

static_assert(sizeof(__nv_fp8_e4m3) == 1);
static_assert(sizeof(__nv_fp8x2_e4m3) == 2);
static_assert(sizeof(__nv_fp8x4_e4m3) == 4);

static_assert(sizeof(__nv_fp8_e8m0) == 1);
static_assert(sizeof(__nv_fp8x2_e8m0) == 2);
static_assert(sizeof(__nv_fp8x4_e8m0) == 4);

// ────────────────────────────────────────────────────────────────────────
// C++ fp8 struct types: are class types
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_class_v<__nv_fp8_e5m2>);
static_assert(std::is_class_v<__nv_fp8x2_e5m2>);
static_assert(std::is_class_v<__nv_fp8x4_e5m2>);

static_assert(std::is_class_v<__nv_fp8_e4m3>);
static_assert(std::is_class_v<__nv_fp8x2_e4m3>);
static_assert(std::is_class_v<__nv_fp8x4_e4m3>);

static_assert(std::is_class_v<__nv_fp8_e8m0>);
static_assert(std::is_class_v<__nv_fp8x2_e8m0>);
static_assert(std::is_class_v<__nv_fp8x4_e8m0>);

// ────────────────────────────────────────────────────────────────────────
// C++ fp8 struct types: standard layout
//
// Standard layout is required for C/CUDA ABI interop. The structs contain
// a single __nv_fp8_storage_t (or x2/x4) data member with no virtual
// functions, so standard layout must hold.
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_standard_layout_v<__nv_fp8_e5m2>);
static_assert(std::is_standard_layout_v<__nv_fp8x2_e5m2>);
static_assert(std::is_standard_layout_v<__nv_fp8x4_e5m2>);

static_assert(std::is_standard_layout_v<__nv_fp8_e4m3>);
static_assert(std::is_standard_layout_v<__nv_fp8x2_e4m3>);
static_assert(std::is_standard_layout_v<__nv_fp8x4_e4m3>);

static_assert(std::is_standard_layout_v<__nv_fp8_e8m0>);
static_assert(std::is_standard_layout_v<__nv_fp8x2_e8m0>);
static_assert(std::is_standard_layout_v<__nv_fp8x4_e8m0>);

// ────────────────────────────────────────────────────────────────────────
// Inline wrapper function invocability checks
//
// Each static_assert confirms that the exported inline wrapper in the
// wwr namespace is callable with the expected argument types.
// WWR_LINK_CHECK is not applicable for inline functions (no external symbol).
// ────────────────────────────────────────────────────────────────────────

// Narrowing conversions (to fp8 storage)
static_assert(std::is_invocable_v<decltype(__nv_cvt_double_to_fp8), double, __nv_saturation_t,
                                  __nv_fp8_interpretation_t>);

static_assert(std::is_invocable_v<decltype(__nv_cvt_double2_to_fp8x2), double2, __nv_saturation_t,
                                  __nv_fp8_interpretation_t>);

static_assert(std::is_invocable_v<decltype(__nv_cvt_float_to_fp8), float, __nv_saturation_t,
                                  __nv_fp8_interpretation_t>);

static_assert(std::is_invocable_v<decltype(__nv_cvt_float2_to_fp8x2), float2, __nv_saturation_t,
                                  __nv_fp8_interpretation_t>);

static_assert(std::is_invocable_v<decltype(__nv_cvt_halfraw_to_fp8), __half_raw, __nv_saturation_t,
                                  __nv_fp8_interpretation_t>);

static_assert(std::is_invocable_v<decltype(__nv_cvt_halfraw2_to_fp8x2), __half2_raw,
                                  __nv_saturation_t, __nv_fp8_interpretation_t>);

static_assert(std::is_invocable_v<decltype(__nv_cvt_bfloat16raw_to_fp8), __nv_bfloat16_raw,
                                  __nv_saturation_t, __nv_fp8_interpretation_t>);

static_assert(std::is_invocable_v<decltype(__nv_cvt_bfloat16raw2_to_fp8x2), __nv_bfloat162_raw,
                                  __nv_saturation_t, __nv_fp8_interpretation_t>);

// Widening conversions (from fp8 storage)
static_assert(std::is_invocable_v<decltype(__nv_cvt_fp8_to_halfraw), __nv_fp8_storage_t,
                                  __nv_fp8_interpretation_t>);

static_assert(std::is_invocable_v<decltype(__nv_cvt_fp8x2_to_halfraw2), __nv_fp8x2_storage_t,
                                  __nv_fp8_interpretation_t>);

// E8M0 scaling factor conversions (extra cudaRoundMode parameter)
static_assert(std::is_invocable_v<decltype(__nv_cvt_bfloat16raw_to_e8m0), __nv_bfloat16_raw,
                                  __nv_saturation_t, cudaRoundMode>);

static_assert(std::is_invocable_v<decltype(__nv_cvt_bfloat162raw_to_e8m0x2), __nv_bfloat162_raw,
                                  __nv_saturation_t, cudaRoundMode>);

static_assert(
    std::is_invocable_v<decltype(__nv_cvt_float_to_e8m0), float, __nv_saturation_t, cudaRoundMode>);

static_assert(std::is_invocable_v<decltype(__nv_cvt_float2_to_e8m0x2), float2, __nv_saturation_t,
                                  cudaRoundMode>);

static_assert(std::is_invocable_v<decltype(__nv_cvt_double_to_e8m0), double, __nv_saturation_t,
                                  cudaRoundMode>);

static_assert(std::is_invocable_v<decltype(__nv_cvt_double2_to_e8m0x2), double2, __nv_saturation_t,
                                  cudaRoundMode>);

static_assert(std::is_invocable_v<decltype(__nv_cvt_e8m0_to_bf16raw), __nv_fp8_storage_t>);

static_assert(std::is_invocable_v<decltype(__nv_cvt_e8m0x2_to_bf162raw), __nv_fp8x2_storage_t>);

// Verify return types of narrowing wrappers
static_assert(std::is_same_v<std::invoke_result_t<decltype(__nv_cvt_float_to_fp8), float,
                                                  __nv_saturation_t, __nv_fp8_interpretation_t>,
                             __nv_fp8_storage_t>);

static_assert(std::is_same_v<std::invoke_result_t<decltype(__nv_cvt_float2_to_fp8x2), float2,
                                                  __nv_saturation_t, __nv_fp8_interpretation_t>,
                             __nv_fp8x2_storage_t>);

// Verify return types of widening wrappers
static_assert(std::is_same_v<std::invoke_result_t<decltype(__nv_cvt_fp8_to_halfraw),
                                                  __nv_fp8_storage_t, __nv_fp8_interpretation_t>,
                             __half_raw>);

static_assert(std::is_same_v<std::invoke_result_t<decltype(__nv_cvt_fp8x2_to_halfraw2),
                                                  __nv_fp8x2_storage_t, __nv_fp8_interpretation_t>,
                             __half2_raw>);

// Verify return types of e8m0 wrappers
static_assert(
    std::is_same_v<std::invoke_result_t<decltype(__nv_cvt_e8m0_to_bf16raw), __nv_fp8_storage_t>,
                   __nv_bfloat16_raw>);

static_assert(std::is_same_v<
              std::invoke_result_t<decltype(__nv_cvt_e8m0x2_to_bf162raw), __nv_fp8x2_storage_t>,
              __nv_bfloat162_raw>);

} // namespace wwr::cuda::test
