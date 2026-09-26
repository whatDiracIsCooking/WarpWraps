// hipfft.cppm - Compile-time tests for gpumod.hip.hipfft

module;

#include "test/shared/link_check.h"

export module gpumod.test.hip.hipfft;

import std;
import gpumod.hip.hipfft;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time tests for gpumod.hip.hipfft
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace gpumod::hip::test {

using namespace gpumod::hip;

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
// GPUMOD_LINK_CHECK: plan creation
// ────────────────────────────────────────────────────────────────────────

GPUMOD_LINK_CHECK(hipfftPlan1d)
GPUMOD_LINK_CHECK(hipfftPlan2d)
GPUMOD_LINK_CHECK(hipfftPlan3d)
GPUMOD_LINK_CHECK(hipfftPlanMany)

// ────────────────────────────────────────────────────────────────────────
// GPUMOD_LINK_CHECK: plan make (two-step)
// ────────────────────────────────────────────────────────────────────────

GPUMOD_LINK_CHECK(hipfftCreate)
GPUMOD_LINK_CHECK(hipfftExtPlanScaleFactor)
GPUMOD_LINK_CHECK(hipfftMakePlan1d)
GPUMOD_LINK_CHECK(hipfftMakePlan2d)
GPUMOD_LINK_CHECK(hipfftMakePlan3d)
GPUMOD_LINK_CHECK(hipfftMakePlanMany)
GPUMOD_LINK_CHECK(hipfftMakePlanMany64)

// ────────────────────────────────────────────────────────────────────────
// GPUMOD_LINK_CHECK: work size estimation
// ────────────────────────────────────────────────────────────────────────

GPUMOD_LINK_CHECK(hipfftEstimate1d)
GPUMOD_LINK_CHECK(hipfftEstimate2d)
GPUMOD_LINK_CHECK(hipfftEstimate3d)
GPUMOD_LINK_CHECK(hipfftEstimateMany)

// ────────────────────────────────────────────────────────────────────────
// GPUMOD_LINK_CHECK: work size query
// ────────────────────────────────────────────────────────────────────────

GPUMOD_LINK_CHECK(hipfftGetSize1d)
GPUMOD_LINK_CHECK(hipfftGetSize2d)
GPUMOD_LINK_CHECK(hipfftGetSize3d)
GPUMOD_LINK_CHECK(hipfftGetSizeMany)
GPUMOD_LINK_CHECK(hipfftGetSizeMany64)
GPUMOD_LINK_CHECK(hipfftGetSize)

// ────────────────────────────────────────────────────────────────────────
// GPUMOD_LINK_CHECK: work area management
// ────────────────────────────────────────────────────────────────────────

GPUMOD_LINK_CHECK(hipfftSetWorkArea)
GPUMOD_LINK_CHECK(hipfftSetAutoAllocation)

// ────────────────────────────────────────────────────────────────────────
// GPUMOD_LINK_CHECK: execution
// ────────────────────────────────────────────────────────────────────────

GPUMOD_LINK_CHECK(hipfftExecC2C)
GPUMOD_LINK_CHECK(hipfftExecR2C)
GPUMOD_LINK_CHECK(hipfftExecC2R)
GPUMOD_LINK_CHECK(hipfftExecZ2Z)
GPUMOD_LINK_CHECK(hipfftExecD2Z)
GPUMOD_LINK_CHECK(hipfftExecZ2D)

// ────────────────────────────────────────────────────────────────────────
// GPUMOD_LINK_CHECK: utility / lifecycle
// ────────────────────────────────────────────────────────────────────────

GPUMOD_LINK_CHECK(hipfftSetStream)
GPUMOD_LINK_CHECK(hipfftDestroy)
GPUMOD_LINK_CHECK(hipfftGetVersion)
GPUMOD_LINK_CHECK(hipfftGetProperty)

} // namespace gpumod::hip::test
