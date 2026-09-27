// hipfft.cppm - Compile-time tests for wwr.hip.hipfft

module;

#include "test/shared/link_check.h"

export module wwr.test.hip.hipfft;

import std;
import wwr.hip.hipfft;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time tests for wwr.hip.hipfft
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace wwr::hip::test {

using namespace wwr::hip;

// ────────────────────────────────────────────────────────────────────────
// Constexpr call-site flag values
// ────────────────────────────────────────────────────────────────────────

static_assert(HIPFFT_FORWARD == -1);
static_assert(HIPFFT_BACKWARD == 1);

// ────────────────────────────────────────────────────────────────────────
// Enum type checks
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_enum_v<hipfftResult_t>);
static_assert(std::is_enum_v<hipfftType_t>);
static_assert(std::is_enum_v<hipfftLibraryPropertyType_t>);

// ────────────────────────────────────────────────────────────────────────
// Enum values: hipfftResult_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(HIPFFT_SUCCESS) == 0);
static_assert(static_cast<int>(HIPFFT_INVALID_PLAN) == 1);
static_assert(static_cast<int>(HIPFFT_ALLOC_FAILED) == 2);
static_assert(static_cast<int>(HIPFFT_INVALID_TYPE) == 3);
static_assert(static_cast<int>(HIPFFT_INVALID_VALUE) == 4);
static_assert(static_cast<int>(HIPFFT_INTERNAL_ERROR) == 5);
static_assert(static_cast<int>(HIPFFT_EXEC_FAILED) == 6);
static_assert(static_cast<int>(HIPFFT_SETUP_FAILED) == 7);
static_assert(static_cast<int>(HIPFFT_INVALID_SIZE) == 8);
static_assert(static_cast<int>(HIPFFT_UNALIGNED_DATA) == 9);
static_assert(static_cast<int>(HIPFFT_INCOMPLETE_PARAMETER_LIST) == 10);
static_assert(static_cast<int>(HIPFFT_INVALID_DEVICE) == 11);
static_assert(static_cast<int>(HIPFFT_PARSE_ERROR) == 12);
static_assert(static_cast<int>(HIPFFT_NO_WORKSPACE) == 13);
static_assert(static_cast<int>(HIPFFT_NOT_IMPLEMENTED) == 14);
static_assert(static_cast<int>(HIPFFT_NOT_SUPPORTED) == 16);

// ────────────────────────────────────────────────────────────────────────
// Enum values: hipfftType_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(HIPFFT_R2C) == 0x2a);
static_assert(static_cast<int>(HIPFFT_C2R) == 0x2c);
static_assert(static_cast<int>(HIPFFT_C2C) == 0x29);
static_assert(static_cast<int>(HIPFFT_D2Z) == 0x6a);
static_assert(static_cast<int>(HIPFFT_Z2D) == 0x6c);
static_assert(static_cast<int>(HIPFFT_Z2Z) == 0x69);

// ────────────────────────────────────────────────────────────────────────
// Enum values: hipfftLibraryPropertyType_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(HIPFFT_MAJOR_VERSION) == 0);
static_assert(static_cast<int>(HIPFFT_MINOR_VERSION) == 1);
static_assert(static_cast<int>(HIPFFT_PATCH_LEVEL) == 2);

// ────────────────────────────────────────────────────────────────────────
// WWR_LINK_CHECK: plan creation
// ────────────────────────────────────────────────────────────────────────

WWR_LINK_CHECK(hipfftPlan1d)
WWR_LINK_CHECK(hipfftPlan2d)
WWR_LINK_CHECK(hipfftPlan3d)
WWR_LINK_CHECK(hipfftPlanMany)

// ────────────────────────────────────────────────────────────────────────
// WWR_LINK_CHECK: plan make (two-step)
// ────────────────────────────────────────────────────────────────────────

WWR_LINK_CHECK(hipfftCreate)
WWR_LINK_CHECK(hipfftExtPlanScaleFactor)
WWR_LINK_CHECK(hipfftMakePlan1d)
WWR_LINK_CHECK(hipfftMakePlan2d)
WWR_LINK_CHECK(hipfftMakePlan3d)
WWR_LINK_CHECK(hipfftMakePlanMany)
WWR_LINK_CHECK(hipfftMakePlanMany64)

// ────────────────────────────────────────────────────────────────────────
// WWR_LINK_CHECK: work size estimation
// ────────────────────────────────────────────────────────────────────────

WWR_LINK_CHECK(hipfftEstimate1d)
WWR_LINK_CHECK(hipfftEstimate2d)
WWR_LINK_CHECK(hipfftEstimate3d)
WWR_LINK_CHECK(hipfftEstimateMany)

// ────────────────────────────────────────────────────────────────────────
// WWR_LINK_CHECK: work size query
// ────────────────────────────────────────────────────────────────────────

WWR_LINK_CHECK(hipfftGetSize1d)
WWR_LINK_CHECK(hipfftGetSize2d)
WWR_LINK_CHECK(hipfftGetSize3d)
WWR_LINK_CHECK(hipfftGetSizeMany)
WWR_LINK_CHECK(hipfftGetSizeMany64)
WWR_LINK_CHECK(hipfftGetSize)

// ────────────────────────────────────────────────────────────────────────
// WWR_LINK_CHECK: work area management
// ────────────────────────────────────────────────────────────────────────

WWR_LINK_CHECK(hipfftSetWorkArea)
WWR_LINK_CHECK(hipfftSetAutoAllocation)

// ────────────────────────────────────────────────────────────────────────
// WWR_LINK_CHECK: execution
// ────────────────────────────────────────────────────────────────────────

WWR_LINK_CHECK(hipfftExecC2C)
WWR_LINK_CHECK(hipfftExecR2C)
WWR_LINK_CHECK(hipfftExecC2R)
WWR_LINK_CHECK(hipfftExecZ2Z)
WWR_LINK_CHECK(hipfftExecD2Z)
WWR_LINK_CHECK(hipfftExecZ2D)

// ────────────────────────────────────────────────────────────────────────
// WWR_LINK_CHECK: utility / lifecycle
// ────────────────────────────────────────────────────────────────────────

WWR_LINK_CHECK(hipfftSetStream)
WWR_LINK_CHECK(hipfftDestroy)
WWR_LINK_CHECK(hipfftGetVersion)
WWR_LINK_CHECK(hipfftGetProperty)

} // namespace wwr::hip::test
