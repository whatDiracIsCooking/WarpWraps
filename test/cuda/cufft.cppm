// cufft.cppm - Compile-time tests for gpumod.cuda.cufft

module;

#include "test/shared/link_check.h"

export module gpumod.test.cuda.cufft;

import std;
import gpumod.cuda.cufft;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time tests for gpumod.cuda.cufft
//
// The module is pure re-export (using declarations + constexpr flag values).
// Runtime tests for the underlying cuFFT API would just test cuFFT itself.
// We verify at compile-time that:
//   1. Constexpr call-site flags have the correct values
//   2. Key enum values with cuFFT-specified values are correct
//   3. Handle type traits (cufftHandle is a plain int, not a pointer)
//   4. Scalar/complex types satisfy trivial copyability and standard layout
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace gpumod::cuda::test {

using namespace gpumod::cuda;

// ────────────────────────────────────────────────────────────────────────
// Constexpr direction flags
// ────────────────────────────────────────────────────────────────────────

static_assert(CUFFT_FORWARD == -1);
static_assert(CUFFT_INVERSE == 1);

// ────────────────────────────────────────────────────────────────────────
// Enum type checks
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_enum_v<cufftResult_t>);
static_assert(std::is_enum_v<cufftType_t>);
static_assert(std::is_enum_v<cufftCompatibility_t>);
static_assert(std::is_enum_v<cufftProperty_t>);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cufftResult_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUFFT_SUCCESS) == 0x0);
static_assert(static_cast<int>(CUFFT_INVALID_PLAN) == 0x1);
static_assert(static_cast<int>(CUFFT_ALLOC_FAILED) == 0x2);
static_assert(static_cast<int>(CUFFT_INVALID_TYPE) == 0x3);
static_assert(static_cast<int>(CUFFT_INVALID_VALUE) == 0x4);
static_assert(static_cast<int>(CUFFT_INTERNAL_ERROR) == 0x5);
static_assert(static_cast<int>(CUFFT_EXEC_FAILED) == 0x6);
static_assert(static_cast<int>(CUFFT_SETUP_FAILED) == 0x7);
static_assert(static_cast<int>(CUFFT_INVALID_SIZE) == 0x8);
static_assert(static_cast<int>(CUFFT_UNALIGNED_DATA) == 0x9);
static_assert(static_cast<int>(CUFFT_INVALID_DEVICE) == 0xB);
static_assert(static_cast<int>(CUFFT_NO_WORKSPACE) == 0xD);
static_assert(static_cast<int>(CUFFT_NOT_IMPLEMENTED) == 0xE);
static_assert(static_cast<int>(CUFFT_NOT_SUPPORTED) == 0x10);
static_assert(static_cast<int>(CUFFT_MISSING_DEPENDENCY) == 0x11);
static_assert(static_cast<int>(CUFFT_NVRTC_FAILURE) == 0x12);
static_assert(static_cast<int>(CUFFT_NVJITLINK_FAILURE) == 0x13);
static_assert(static_cast<int>(CUFFT_NVSHMEM_FAILURE) == 0x14);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cufftType_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUFFT_R2C) == 0x2a);
static_assert(static_cast<int>(CUFFT_C2R) == 0x2c);
static_assert(static_cast<int>(CUFFT_C2C) == 0x29);
static_assert(static_cast<int>(CUFFT_D2Z) == 0x6a);
static_assert(static_cast<int>(CUFFT_Z2D) == 0x6c);
static_assert(static_cast<int>(CUFFT_Z2Z) == 0x69);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cufftCompatibility_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUFFT_COMPATIBILITY_FFTW_PADDING) == 0x01);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cufftProperty_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(NVFFT_PLAN_PROPERTY_INT64_PATIENT_JIT) == 0x1);
static_assert(static_cast<int>(NVFFT_PLAN_PROPERTY_INT64_MAX_NUM_HOST_THREADS) == 0x2);

// ────────────────────────────────────────────────────────────────────────
// Handle type traits
// cufftHandle is typedef'd as int — NOT a pointer (unlike CUDA stream handles)
// ────────────────────────────────────────────────────────────────────────

static_assert(!std::is_pointer_v<cufftHandle>);
static_assert(std::is_integral_v<cufftHandle>);
static_assert(std::is_same_v<cufftHandle, int>);
static_assert(sizeof(cufftHandle) == sizeof(int));

// ────────────────────────────────────────────────────────────────────────
// Scalar / complex type traits
// ────────────────────────────────────────────────────────────────────────

// cufftReal is float
static_assert(std::is_same_v<cufftReal, float>);
static_assert(sizeof(cufftReal) == 4);

// cufftDoubleReal is double
static_assert(std::is_same_v<cufftDoubleReal, double>);
static_assert(sizeof(cufftDoubleReal) == 8);

// cufftComplex is an interleaved float2 (re, im)
static_assert(sizeof(cufftComplex) == 2 * sizeof(float));
static_assert(std::is_trivially_copyable_v<cufftComplex>);
static_assert(std::is_standard_layout_v<cufftComplex>);

// cufftDoubleComplex is an interleaved double2 (re, im)
static_assert(sizeof(cufftDoubleComplex) == 2 * sizeof(double));
static_assert(std::is_trivially_copyable_v<cufftDoubleComplex>);
static_assert(std::is_standard_layout_v<cufftDoubleComplex>);

// ────────────────────────────────────────────────────────────────────────
// Link-time symbol resolution
// Forces the linker to resolve every re-exported function symbol,
// catching missing or unresolvable exports that type-only checks miss.
// ────────────────────────────────────────────────────────────────────────

// Plan Creation
GPUMOD_LINK_CHECK(cufftPlan1d)
GPUMOD_LINK_CHECK(cufftPlan2d)
GPUMOD_LINK_CHECK(cufftPlan3d)
GPUMOD_LINK_CHECK(cufftPlanMany)

// Plan Make (two-step)
GPUMOD_LINK_CHECK(cufftCreate)
GPUMOD_LINK_CHECK(cufftMakePlan1d)
GPUMOD_LINK_CHECK(cufftMakePlan2d)
GPUMOD_LINK_CHECK(cufftMakePlan3d)
GPUMOD_LINK_CHECK(cufftMakePlanMany)
GPUMOD_LINK_CHECK(cufftMakePlanMany64)

// Work Size Estimation
GPUMOD_LINK_CHECK(cufftEstimate1d)
GPUMOD_LINK_CHECK(cufftEstimate2d)
GPUMOD_LINK_CHECK(cufftEstimate3d)
GPUMOD_LINK_CHECK(cufftEstimateMany)

// Work Size Query
GPUMOD_LINK_CHECK(cufftGetSize1d)
GPUMOD_LINK_CHECK(cufftGetSize2d)
GPUMOD_LINK_CHECK(cufftGetSize3d)
GPUMOD_LINK_CHECK(cufftGetSizeMany)
GPUMOD_LINK_CHECK(cufftGetSizeMany64)
GPUMOD_LINK_CHECK(cufftGetSize)

// Work Area Management
GPUMOD_LINK_CHECK(cufftSetWorkArea)
GPUMOD_LINK_CHECK(cufftSetAutoAllocation)

// Execution
GPUMOD_LINK_CHECK(cufftExecC2C)
GPUMOD_LINK_CHECK(cufftExecR2C)
GPUMOD_LINK_CHECK(cufftExecC2R)
GPUMOD_LINK_CHECK(cufftExecZ2Z)
GPUMOD_LINK_CHECK(cufftExecD2Z)
GPUMOD_LINK_CHECK(cufftExecZ2D)

// Utility / Lifecycle
GPUMOD_LINK_CHECK(cufftSetStream)
GPUMOD_LINK_CHECK(cufftDestroy)
GPUMOD_LINK_CHECK(cufftGetVersion)
GPUMOD_LINK_CHECK(cufftGetProperty)

// Per-Plan Properties
GPUMOD_LINK_CHECK(cufftSetPlanPropertyInt64)
GPUMOD_LINK_CHECK(cufftGetPlanPropertyInt64)
GPUMOD_LINK_CHECK(cufftResetPlanProperty)

} // namespace gpumod::cuda::test
