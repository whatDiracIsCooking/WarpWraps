/**
 * @file cufft.cppm
 * @brief Primary interface for wwr.cuda.cufft
 *
 * This module wraps the native cuFFT API and exports types, constants,
 * and functions for cuFFT library management and FFT operations.
 *
 * Usage:
 *   import wwr.cuda.cufft;
 */

module;

#include <cufft.h>

// ========================================================================
// Validate call-site flag macros before #undef — these fire at parse time
// in the global module fragment, before any module-unit code is processed.
// ========================================================================

static_assert(CUFFT_FORWARD == -1, "CUFFT_FORWARD value mismatch");
static_assert(CUFFT_INVERSE == 1, "CUFFT_INVERSE value mismatch");

// Undefine call-site flag macros so we can declare constexpr replacements.
// Version/metadata macros (CUFFT_VER_*, CUFFT_VERSION, MAX_CUFFT_ERROR,
// MAX_SHIM_RANK) are left alone — they are not passed to any function and
// are not exported.
#undef CUFFT_FORWARD
#undef CUFFT_INVERSE

export module wwr.cuda.cufft;

// ========================================================================
// Export all cuFFT types and functions in wwr namespace
// ========================================================================

export namespace wwr::cuda {

// ========================================================================
// Constexpr wrappers for cuFFT call-site flag macros
// ========================================================================

// FFT direction flags (used as the `direction` argument to Exec* functions)
constexpr int CUFFT_FORWARD = -1;
constexpr int CUFFT_INVERSE = 1;

// ========================================================================
// Core scalar types
// ========================================================================
using ::cufftComplex;
using ::cufftDoubleComplex;
using ::cufftDoubleReal;
using ::cufftHandle;
using ::cufftReal;

// Dependent types from included headers that appear in cuFFT signatures
using ::cudaStream_t;
using ::libraryPropertyType;

// ========================================================================
// Result / status enum
// ========================================================================
using ::cufftResult;
using ::cufftResult_t;

// cufftResult_t enumerators
using ::CUFFT_ALLOC_FAILED;
using ::CUFFT_EXEC_FAILED;
using ::CUFFT_INTERNAL_ERROR;
using ::CUFFT_INVALID_DEVICE;
using ::CUFFT_INVALID_PLAN;
using ::CUFFT_INVALID_SIZE;
using ::CUFFT_INVALID_TYPE;
using ::CUFFT_INVALID_VALUE;
using ::CUFFT_MISSING_DEPENDENCY;
using ::CUFFT_NO_WORKSPACE;
using ::CUFFT_NOT_IMPLEMENTED;
using ::CUFFT_NOT_SUPPORTED;
using ::CUFFT_NVJITLINK_FAILURE;
using ::CUFFT_NVRTC_FAILURE;
using ::CUFFT_NVSHMEM_FAILURE;
using ::CUFFT_SETUP_FAILED;
using ::CUFFT_SUCCESS;
using ::CUFFT_UNALIGNED_DATA;

// ========================================================================
// Transform type enum
// ========================================================================
using ::cufftType;
using ::cufftType_t;

// cufftType_t enumerators
using ::CUFFT_C2C;
using ::CUFFT_C2R;
using ::CUFFT_D2Z;
using ::CUFFT_R2C;
using ::CUFFT_Z2D;
using ::CUFFT_Z2Z;

// ========================================================================
// Compatibility enum
// ========================================================================
using ::cufftCompatibility;
using ::cufftCompatibility_t;

// cufftCompatibility_t enumerators
using ::CUFFT_COMPATIBILITY_FFTW_PADDING;

// ========================================================================
// Plan property enum (per-plan configuration)
// ========================================================================
using ::cufftProperty;
using ::cufftProperty_t;

// cufftProperty_t enumerators
using ::NVFFT_PLAN_PROPERTY_INT64_MAX_NUM_HOST_THREADS;
using ::NVFFT_PLAN_PROPERTY_INT64_PATIENT_JIT;

// ========================================================================
// Plan Creation Functions
// ========================================================================
using ::cufftPlan1d;
using ::cufftPlan2d;
using ::cufftPlan3d;
using ::cufftPlanMany;

// ========================================================================
// Plan Make Functions (two-step: cufftCreate then MakePlan*)
// ========================================================================
using ::cufftCreate;
using ::cufftMakePlan1d;
using ::cufftMakePlan2d;
using ::cufftMakePlan3d;
using ::cufftMakePlanMany;
using ::cufftMakePlanMany64;

// ========================================================================
// Work Size Estimation Functions
// ========================================================================
using ::cufftEstimate1d;
using ::cufftEstimate2d;
using ::cufftEstimate3d;
using ::cufftEstimateMany;

// ========================================================================
// Work Size Query Functions
// ========================================================================
using ::cufftGetSize;
using ::cufftGetSize1d;
using ::cufftGetSize2d;
using ::cufftGetSize3d;
using ::cufftGetSizeMany;
using ::cufftGetSizeMany64;

// ========================================================================
// Work Area Management
// ========================================================================
using ::cufftSetAutoAllocation;
using ::cufftSetWorkArea;

// ========================================================================
// Execution Functions
// ========================================================================
using ::cufftExecC2C;
using ::cufftExecC2R;
using ::cufftExecD2Z;
using ::cufftExecR2C;
using ::cufftExecZ2D;
using ::cufftExecZ2Z;

// ========================================================================
// Utility / Lifecycle Functions
// ========================================================================
using ::cufftDestroy;
using ::cufftGetProperty;
using ::cufftGetVersion;
using ::cufftSetStream;

// ========================================================================
// Per-Plan Property Functions
// ========================================================================
using ::cufftGetPlanPropertyInt64;
using ::cufftResetPlanProperty;
using ::cufftSetPlanPropertyInt64;

} // namespace wwr::cuda
