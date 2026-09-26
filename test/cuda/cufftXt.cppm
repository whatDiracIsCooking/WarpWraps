// cufftXt.cppm - Compile-time tests for gpumod.cuda.cufftXt

module;

#include "test/shared/link_check.h"

export module gpumod.test.cuda.cufftXt;

import std;
import gpumod.cuda.cufftXt;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time tests for gpumod.cuda.cufftXt
//
// The module is a pure re-export (using declarations only; no macros are
// used as call-site flags in cufftXt.h so no constexpr replacements are
// needed). We verify at compile-time that:
//   1. All enum types are indeed enum types
//   2. All enum enumerators carry the specified integer values
//   3. Descriptor structs satisfy standard-layout and trivial-copyability
//      where applicable
//   4. Callback function-pointer typedefs are plain pointer types
//   5. Every non-inline function symbol resolves at link time (GPUMOD_LINK_CHECK)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace gpumod::cuda::test {

using namespace gpumod::cuda;

// ────────────────────────────────────────────────────────────────────────
// Enum type checks
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_enum_v<cudaXtCopyType_t>);
static_assert(std::is_enum_v<libFormat_t>);
static_assert(std::is_enum_v<cufftXtSubFormat_t>);
static_assert(std::is_enum_v<cufftXtCopyType_t>);
static_assert(std::is_enum_v<cufftXtQueryType_t>);
static_assert(std::is_enum_v<cufftXtWorkAreaPolicy_t>);
static_assert(std::is_enum_v<cufftXtCallbackType_t>);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cudaXtCopyType_t (cudalibxt.h)
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(LIB_XT_COPY_HOST_TO_DEVICE) == 0);
static_assert(static_cast<int>(LIB_XT_COPY_DEVICE_TO_HOST) == 1);
static_assert(static_cast<int>(LIB_XT_COPY_DEVICE_TO_DEVICE) == 2);

// ────────────────────────────────────────────────────────────────────────
// Enum values: libFormat_t (cudalibxt.h)
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(LIB_FORMAT_CUFFT) == 0x0);
static_assert(static_cast<int>(LIB_FORMAT_UNDEFINED) == 0x1);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cufftXtSubFormat_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUFFT_XT_FORMAT_INPUT) == 0x00);
static_assert(static_cast<int>(CUFFT_XT_FORMAT_OUTPUT) == 0x01);
static_assert(static_cast<int>(CUFFT_XT_FORMAT_INPLACE) == 0x02);
static_assert(static_cast<int>(CUFFT_XT_FORMAT_INPLACE_SHUFFLED) == 0x03);
static_assert(static_cast<int>(CUFFT_XT_FORMAT_1D_INPUT_SHUFFLED) == 0x04);
static_assert(static_cast<int>(CUFFT_XT_FORMAT_DISTRIBUTED_INPUT) == 0x05);
static_assert(static_cast<int>(CUFFT_XT_FORMAT_DISTRIBUTED_OUTPUT) == 0x06);
static_assert(static_cast<int>(CUFFT_FORMAT_UNDEFINED) == 0x07);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cufftXtCopyType_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUFFT_COPY_HOST_TO_DEVICE) == 0x00);
static_assert(static_cast<int>(CUFFT_COPY_DEVICE_TO_HOST) == 0x01);
static_assert(static_cast<int>(CUFFT_COPY_DEVICE_TO_DEVICE) == 0x02);
static_assert(static_cast<int>(CUFFT_COPY_UNDEFINED) == 0x03);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cufftXtQueryType_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUFFT_QUERY_1D_FACTORS) == 0x00);
static_assert(static_cast<int>(CUFFT_QUERY_UNDEFINED) == 0x01);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cufftXtWorkAreaPolicy_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUFFT_WORKAREA_MINIMAL) == 0);
static_assert(static_cast<int>(CUFFT_WORKAREA_USER) == 1);
static_assert(static_cast<int>(CUFFT_WORKAREA_PERFORMANCE) == 2);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cufftXtCallbackType_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUFFT_CB_LD_COMPLEX) == 0x0);
static_assert(static_cast<int>(CUFFT_CB_LD_COMPLEX_DOUBLE) == 0x1);
static_assert(static_cast<int>(CUFFT_CB_LD_REAL) == 0x2);
static_assert(static_cast<int>(CUFFT_CB_LD_REAL_DOUBLE) == 0x3);
static_assert(static_cast<int>(CUFFT_CB_ST_COMPLEX) == 0x4);
static_assert(static_cast<int>(CUFFT_CB_ST_COMPLEX_DOUBLE) == 0x5);
static_assert(static_cast<int>(CUFFT_CB_ST_REAL) == 0x6);
static_assert(static_cast<int>(CUFFT_CB_ST_REAL_DOUBLE) == 0x7);
static_assert(static_cast<int>(CUFFT_CB_UNDEFINED) == 0x8);

// ────────────────────────────────────────────────────────────────────────
// Descriptor struct type traits
// cudaXtDesc and cudaLibXtDesc are plain C structs — standard layout and
// trivially copyable.
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_standard_layout_v<cudaXtDesc>);
static_assert(std::is_trivially_copyable_v<cudaXtDesc>);

static_assert(std::is_standard_layout_v<cudaLibXtDesc>);
static_assert(std::is_trivially_copyable_v<cudaLibXtDesc>);

static_assert(std::is_standard_layout_v<cufftXt1dFactors>);
static_assert(std::is_trivially_copyable_v<cufftXt1dFactors>);

// ────────────────────────────────────────────────────────────────────────
// Callback typedef checks — all must be non-null function pointer types
// ────────────────────────────────────────────────────────────────────────

// Legacy load callbacks (size_t offset)
static_assert(std::is_pointer_v<cufftCallbackLoadC>);
static_assert(std::is_pointer_v<cufftCallbackLoadZ>);
static_assert(std::is_pointer_v<cufftCallbackLoadR>);
static_assert(std::is_pointer_v<cufftCallbackLoadD>);

// Legacy store callbacks (size_t offset)
static_assert(std::is_pointer_v<cufftCallbackStoreC>);
static_assert(std::is_pointer_v<cufftCallbackStoreZ>);
static_assert(std::is_pointer_v<cufftCallbackStoreR>);
static_assert(std::is_pointer_v<cufftCallbackStoreD>);

// JIT/LTO load callbacks (unsigned long long offset)
static_assert(std::is_pointer_v<cufftJITCallbackLoadC>);
static_assert(std::is_pointer_v<cufftJITCallbackLoadZ>);
static_assert(std::is_pointer_v<cufftJITCallbackLoadR>);
static_assert(std::is_pointer_v<cufftJITCallbackLoadD>);

// JIT/LTO store callbacks (unsigned long long offset)
static_assert(std::is_pointer_v<cufftJITCallbackStoreC>);
static_assert(std::is_pointer_v<cufftJITCallbackStoreZ>);
static_assert(std::is_pointer_v<cufftJITCallbackStoreR>);
static_assert(std::is_pointer_v<cufftJITCallbackStoreD>);

// ────────────────────────────────────────────────────────────────────────
// Link-time symbol resolution
// Forces the linker to resolve every re-exported function symbol,
// catching missing or unresolvable exports that type-only checks miss.
// ────────────────────────────────────────────────────────────────────────

// Multi-GPU Setup
GPUMOD_LINK_CHECK(cufftXtSetGPUs)

// Multi-GPU Memory Management
GPUMOD_LINK_CHECK(cufftXtMalloc)
GPUMOD_LINK_CHECK(cufftXtMemcpy)
GPUMOD_LINK_CHECK(cufftXtFree)
GPUMOD_LINK_CHECK(cufftXtSetWorkArea)

// Descriptor-based typed execution
GPUMOD_LINK_CHECK(cufftXtExecDescriptorC2C)
GPUMOD_LINK_CHECK(cufftXtExecDescriptorR2C)
GPUMOD_LINK_CHECK(cufftXtExecDescriptorC2R)
GPUMOD_LINK_CHECK(cufftXtExecDescriptorZ2Z)
GPUMOD_LINK_CHECK(cufftXtExecDescriptorD2Z)
GPUMOD_LINK_CHECK(cufftXtExecDescriptorZ2D)

// Generic execution
GPUMOD_LINK_CHECK(cufftXtExec)
GPUMOD_LINK_CHECK(cufftXtExecDescriptor)

// Extended plan creation and size query
GPUMOD_LINK_CHECK(cufftXtMakePlanMany)
GPUMOD_LINK_CHECK(cufftXtGetSizeMany)

// Work area policy
GPUMOD_LINK_CHECK(cufftXtSetWorkAreaPolicy)

// Query
GPUMOD_LINK_CHECK(cufftXtQueryPlan)

// Callback registration
GPUMOD_LINK_CHECK(cufftXtSetCallback)
GPUMOD_LINK_CHECK(cufftXtClearCallback)
GPUMOD_LINK_CHECK(cufftXtSetCallbackSharedSize)
GPUMOD_LINK_CHECK(cufftXtSetJITCallback)

} // namespace gpumod::cuda::test
