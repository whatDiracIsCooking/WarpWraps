// hipfftXt.cppm - Compile-time tests for gpumod.hip.hipfftXt

module;

#include "test/shared/link_check.h"

export module gpumod.test.hip.hipfftXt;

import std;
import gpumod.hip.hipfftXt;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time tests for gpumod.hip.hipfftXt
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace gpumod::hip::test {

using namespace gpumod::hip;

// ────────────────────────────────────────────────────────────────────────
// Enum type checks
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_enum_v<hiplibFormat_t>);
static_assert(std::is_enum_v<hipfftXtCopyType_t>);
static_assert(std::is_enum_v<hipfftXtCallbackType_t>);
static_assert(std::is_enum_v<hipfftXtSubFormat_t>);

// ────────────────────────────────────────────────────────────────────────
// Enum values: hiplibFormat_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(HIPLIB_FORMAT_HIPFFT) == 0x0);
static_assert(static_cast<int>(HIPLIB_FORMAT_UNDEFINED) == 0x1);

// ────────────────────────────────────────────────────────────────────────
// Enum values: hipfftXtCopyType_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(HIPFFT_COPY_HOST_TO_DEVICE) == 0x00);
static_assert(static_cast<int>(HIPFFT_COPY_DEVICE_TO_HOST) == 0x01);
static_assert(static_cast<int>(HIPFFT_COPY_DEVICE_TO_DEVICE) == 0x02);
static_assert(static_cast<int>(HIPFFT_COPY_UNDEFINED) == 0x03);

// ────────────────────────────────────────────────────────────────────────
// Enum values: hipfftXtCallbackType_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(HIPFFT_CB_LD_COMPLEX) == 0x0);
static_assert(static_cast<int>(HIPFFT_CB_LD_COMPLEX_DOUBLE) == 0x1);
static_assert(static_cast<int>(HIPFFT_CB_LD_REAL) == 0x2);
static_assert(static_cast<int>(HIPFFT_CB_LD_REAL_DOUBLE) == 0x3);
static_assert(static_cast<int>(HIPFFT_CB_ST_COMPLEX) == 0x4);
static_assert(static_cast<int>(HIPFFT_CB_ST_COMPLEX_DOUBLE) == 0x5);
static_assert(static_cast<int>(HIPFFT_CB_ST_REAL) == 0x6);
static_assert(static_cast<int>(HIPFFT_CB_ST_REAL_DOUBLE) == 0x7);
static_assert(static_cast<int>(HIPFFT_CB_UNDEFINED) == 0x8);

// ────────────────────────────────────────────────────────────────────────
// Enum values: hipfftXtSubFormat_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(HIPFFT_XT_FORMAT_INPUT) == 0x00);
static_assert(static_cast<int>(HIPFFT_XT_FORMAT_OUTPUT) == 0x01);
static_assert(static_cast<int>(HIPFFT_XT_FORMAT_INPLACE) == 0x02);
static_assert(static_cast<int>(HIPFFT_XT_FORMAT_INPLACE_SHUFFLED) == 0x03);
static_assert(static_cast<int>(HIPFFT_XT_FORMAT_1D_INPUT_SHUFFLED) == 0x04);
static_assert(static_cast<int>(HIPFFT_FORMAT_UNDEFINED) == 0x05);

// ────────────────────────────────────────────────────────────────────────
// Struct type traits
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_standard_layout_v<hipXtDesc_t>);
static_assert(std::is_standard_layout_v<hipLibXtDesc_t>);

// ────────────────────────────────────────────────────────────────────────
// GPUMOD_LINK_CHECK: callback registration
// ────────────────────────────────────────────────────────────────────────

GPUMOD_LINK_CHECK(hipfftXtSetCallback)
GPUMOD_LINK_CHECK(hipfftXtClearCallback)
GPUMOD_LINK_CHECK(hipfftXtSetCallbackSharedSize)

// ────────────────────────────────────────────────────────────────────────
// GPUMOD_LINK_CHECK: extended plan creation and size query
// ────────────────────────────────────────────────────────────────────────

GPUMOD_LINK_CHECK(hipfftXtMakePlanMany)
GPUMOD_LINK_CHECK(hipfftXtGetSizeMany)

// ────────────────────────────────────────────────────────────────────────
// GPUMOD_LINK_CHECK: generic execution
// ────────────────────────────────────────────────────────────────────────

GPUMOD_LINK_CHECK(hipfftXtExec)

// ────────────────────────────────────────────────────────────────────────
// GPUMOD_LINK_CHECK: multi-GPU setup
// ────────────────────────────────────────────────────────────────────────

GPUMOD_LINK_CHECK(hipfftXtSetGPUs)

// ────────────────────────────────────────────────────────────────────────
// GPUMOD_LINK_CHECK: multi-GPU memory management
// ────────────────────────────────────────────────────────────────────────

GPUMOD_LINK_CHECK(hipfftXtMalloc)
GPUMOD_LINK_CHECK(hipfftXtMemcpy)
GPUMOD_LINK_CHECK(hipfftXtFree)

// ────────────────────────────────────────────────────────────────────────
// GPUMOD_LINK_CHECK: multi-GPU execution (descriptor-based, typed)
// ────────────────────────────────────────────────────────────────────────

GPUMOD_LINK_CHECK(hipfftXtExecDescriptorC2C)
GPUMOD_LINK_CHECK(hipfftXtExecDescriptorR2C)
GPUMOD_LINK_CHECK(hipfftXtExecDescriptorC2R)
GPUMOD_LINK_CHECK(hipfftXtExecDescriptorZ2Z)
GPUMOD_LINK_CHECK(hipfftXtExecDescriptorD2Z)
GPUMOD_LINK_CHECK(hipfftXtExecDescriptorZ2D)
GPUMOD_LINK_CHECK(hipfftXtExecDescriptor)

} // namespace gpumod::hip::test
