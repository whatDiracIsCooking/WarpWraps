// hipfftXt.cppm - Compile-time tests for wwr.hip.hipfftXt

module;

#include "test/shared/link_check.h"

export module wwr.test.hip.hipfftXt;

import std;
import wwr.hip.hipfftXt;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time tests for wwr.hip.hipfftXt
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace wwr::hip::test {

using namespace wwr::hip;

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
// WWR_LINK_CHECK: callback registration
// ────────────────────────────────────────────────────────────────────────

WWR_LINK_CHECK(hipfftXtSetCallback)
WWR_LINK_CHECK(hipfftXtClearCallback)
WWR_LINK_CHECK(hipfftXtSetCallbackSharedSize)

// ────────────────────────────────────────────────────────────────────────
// WWR_LINK_CHECK: extended plan creation and size query
// ────────────────────────────────────────────────────────────────────────

WWR_LINK_CHECK(hipfftXtMakePlanMany)
WWR_LINK_CHECK(hipfftXtGetSizeMany)

// ────────────────────────────────────────────────────────────────────────
// WWR_LINK_CHECK: generic execution
// ────────────────────────────────────────────────────────────────────────

WWR_LINK_CHECK(hipfftXtExec)

// ────────────────────────────────────────────────────────────────────────
// WWR_LINK_CHECK: multi-GPU setup
// ────────────────────────────────────────────────────────────────────────

WWR_LINK_CHECK(hipfftXtSetGPUs)

// ────────────────────────────────────────────────────────────────────────
// WWR_LINK_CHECK: multi-GPU memory management
// ────────────────────────────────────────────────────────────────────────

WWR_LINK_CHECK(hipfftXtMalloc)
WWR_LINK_CHECK(hipfftXtMemcpy)
WWR_LINK_CHECK(hipfftXtFree)

// ────────────────────────────────────────────────────────────────────────
// WWR_LINK_CHECK: multi-GPU execution (descriptor-based, typed)
// ────────────────────────────────────────────────────────────────────────

WWR_LINK_CHECK(hipfftXtExecDescriptorC2C)
WWR_LINK_CHECK(hipfftXtExecDescriptorR2C)
WWR_LINK_CHECK(hipfftXtExecDescriptorC2R)
WWR_LINK_CHECK(hipfftXtExecDescriptorZ2Z)
WWR_LINK_CHECK(hipfftXtExecDescriptorD2Z)
WWR_LINK_CHECK(hipfftXtExecDescriptorZ2D)
WWR_LINK_CHECK(hipfftXtExecDescriptor)

} // namespace wwr::hip::test
