/**
 * @file hipfft.cppm
 * @brief hipFFT API module wrapper for gpumod project
 *
 * Wraps hipfft/hipfft.h. CUDA counterpart: gpumod.cuda.cufft.
 *
 * hipFFT's call-site direction flags (HIPFFT_FORWARD / HIPFFT_BACKWARD) are
 * `#define`d plain ints, not enumerators -- parity with how cufft.cppm
 * handles CUFFT_FORWARD/CUFFT_INVERSE: validated by static_assert against
 * the macro before `#undef`, then re-declared as constexpr replacements.
 * Note the name is HIPFFT_BACKWARD, not CUFFT_INVERSE -- hipFFT's own
 * naming, kept as-is rather than renamed to match cuFFT.
 *
 * Usage:
 *   import gpumod.hip.hipfft;
 */

module;

// Pre-include <array> before the HIP header -- see src/hip/hip_complex.cppm's
// file header for why (amd_hip_vector_types.h, pulled in transitively via
// hip/hip_complex.h, #includes host_defines.h immediately before <array>,
// poisoning __has_attribute(__noinline__) for any later first-inclusion of
// <array> in the TU). Confirmed necessary here by direct experiment too.
// Load-bearing, and must stay before the HIP header: host_defines.h poisons
// __noinline__ for libc++'s __config. docs/architecture.md, section 9.
#include <array>
#include <hipfft/hipfft.h>

// Validate call-site flag macros before #undef -- these fire at parse time
// in the global module fragment, before any module-unit code is processed.
static_assert(HIPFFT_FORWARD == -1, "HIPFFT_FORWARD value mismatch");
static_assert(HIPFFT_BACKWARD == 1, "HIPFFT_BACKWARD value mismatch");

#undef HIPFFT_FORWARD
#undef HIPFFT_BACKWARD

export module gpumod.hip.hipfft;

import std;

export namespace gpumod::hip {

// ========================================================================
// Constexpr wrappers for hipFFT call-site flag macros
// ========================================================================
constexpr int HIPFFT_FORWARD = -1;
constexpr int HIPFFT_BACKWARD = 1;

// ========================================================================
// Core scalar types
// ========================================================================
using ::hipfftComplex;
using ::hipfftDoubleComplex;
using ::hipfftDoubleReal;
using ::hipfftHandle;
using ::hipfftReal;

// ========================================================================
// Result / status enum
// ========================================================================
using ::hipfftResult;
using ::hipfftResult_t;

using ::HIPFFT_ALLOC_FAILED;
using ::HIPFFT_EXEC_FAILED;
using ::HIPFFT_INCOMPLETE_PARAMETER_LIST;
using ::HIPFFT_INTERNAL_ERROR;
using ::HIPFFT_INVALID_DEVICE;
using ::HIPFFT_INVALID_PLAN;
using ::HIPFFT_INVALID_SIZE;
using ::HIPFFT_INVALID_TYPE;
using ::HIPFFT_INVALID_VALUE;
using ::HIPFFT_NO_WORKSPACE;
using ::HIPFFT_NOT_IMPLEMENTED;
using ::HIPFFT_NOT_SUPPORTED;
using ::HIPFFT_PARSE_ERROR;
using ::HIPFFT_SETUP_FAILED;
using ::HIPFFT_SUCCESS;
using ::HIPFFT_UNALIGNED_DATA;

// ========================================================================
// Transform type enum
// ========================================================================
using ::hipfftType;
using ::hipfftType_t;

using ::HIPFFT_C2C;
using ::HIPFFT_C2R;
using ::HIPFFT_D2Z;
using ::HIPFFT_R2C;
using ::HIPFFT_Z2D;
using ::HIPFFT_Z2Z;

// ========================================================================
// Library property type enum
// ========================================================================
using ::hipfftLibraryPropertyType;
using ::hipfftLibraryPropertyType_t;

using ::HIPFFT_MAJOR_VERSION;
using ::HIPFFT_MINOR_VERSION;
using ::HIPFFT_PATCH_LEVEL;

// ========================================================================
// Plan Creation Functions
// ========================================================================
using ::hipfftPlan1d;
using ::hipfftPlan2d;
using ::hipfftPlan3d;
using ::hipfftPlanMany;

// ========================================================================
// Plan Make Functions (two-step: hipfftCreate then MakePlan*)
// ========================================================================
using ::hipfftCreate;
using ::hipfftExtPlanScaleFactor;
using ::hipfftMakePlan1d;
using ::hipfftMakePlan2d;
using ::hipfftMakePlan3d;
using ::hipfftMakePlanMany;
using ::hipfftMakePlanMany64;

// ========================================================================
// Work Size Estimation Functions
// ========================================================================
using ::hipfftEstimate1d;
using ::hipfftEstimate2d;
using ::hipfftEstimate3d;
using ::hipfftEstimateMany;

// ========================================================================
// Work Size Query Functions
// ========================================================================
using ::hipfftGetSize;
using ::hipfftGetSize1d;
using ::hipfftGetSize2d;
using ::hipfftGetSize3d;
using ::hipfftGetSizeMany;
using ::hipfftGetSizeMany64;

// ========================================================================
// Work Area Management
// ========================================================================
using ::hipfftSetAutoAllocation;
using ::hipfftSetWorkArea;

// ========================================================================
// Execution Functions
// ========================================================================
using ::hipfftExecC2C;
using ::hipfftExecC2R;
using ::hipfftExecD2Z;
using ::hipfftExecR2C;
using ::hipfftExecZ2D;
using ::hipfftExecZ2Z;

// ========================================================================
// Utility / Lifecycle Functions
// ========================================================================
using ::hipfftDestroy;
using ::hipfftGetProperty;
using ::hipfftGetVersion;
using ::hipfftSetStream;

} // namespace gpumod::hip
