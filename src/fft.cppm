/**
 * @file fft.cppm
 * @brief Backend-neutral FFT: gpufft* names for cuFFT / hipFFT
 *
 * gpufft<name> stands for cufft<name> on a CUDA build and hipfft<name> on a HIP
 * build. Only the base (single-GPU) API is wrapped here -- the multi-GPU
 * eXtended surface is reached through gpumod.cuda.cufftXt / gpumod.hip.hipfftXt
 * directly. See gpu_backend.h.
 *
 * Backend differences resolved here, not above:
 *
 * - Direction flags: GPUFFT_FORWARD / GPUFFT_INVERSE follow cuFFT's spelling
 *   (hipFFT spells the inverse HIPFFT_BACKWARD), as the gpu* layer does throughout.
 * - Result codes: only the 14 codes both backends export get a GPUFFT_* alias;
 *   a backend-specific code is reached through the raw module.
 * - Status strings: gpufftGetStatusName / gpufftGetStatusString are hand-written
 *   per backend, neither vendor shipping one (see docs/architecture.md §5).
 *
 * gpufftGetProperty is omitted (the vendors disagree on the property-type enum);
 * reach it through the raw module.
 *
 * Usage:
 *   import gpumod.fft;
 *
 *   gpufftHandle plan;
 *   gpufftCreate(&plan);
 */

module;

#include "gpu_backend.h"

export module gpumod.fft;

#if defined(GPUMOD_GPU_BACKEND_CUDA)
import gpumod.cuda.cufft;
#else
import gpumod.hip.hipfft;
#endif

export namespace gpumod {

// ========================================================================
// Types
// ========================================================================

GPUMOD_TYPE(gpufftHandle, cufftHandle, hipfftHandle)
GPUMOD_TYPE(gpufftResult_t, cufftResult_t, hipfftResult_t)
GPUMOD_TYPE(gpufftType_t, cufftType_t, hipfftType_t)
GPUMOD_TYPE(gpufftReal, cufftReal, hipfftReal)
GPUMOD_TYPE(gpufftDoubleReal, cufftDoubleReal, hipfftDoubleReal)
GPUMOD_TYPE(gpufftComplex, cufftComplex, hipfftComplex)
GPUMOD_TYPE(gpufftDoubleComplex, cufftDoubleComplex, hipfftDoubleComplex)

// ========================================================================
// Constants
// ========================================================================

// FFT direction flags (the `direction` argument to gpufftExecC2C/Z2Z).
// cuFFT: CUFFT_FORWARD/CUFFT_INVERSE; hipFFT: HIPFFT_FORWARD/HIPFFT_BACKWARD.
GPUMOD_VALUE(GPUFFT_FORWARD, CUFFT_FORWARD, HIPFFT_FORWARD)
GPUMOD_VALUE(GPUFFT_INVERSE, CUFFT_INVERSE, HIPFFT_BACKWARD)

// Transform type (the `type` argument to the plan-creation functions).
GPUMOD_VALUE(GPUFFT_R2C, CUFFT_R2C, HIPFFT_R2C)
GPUMOD_VALUE(GPUFFT_C2R, CUFFT_C2R, HIPFFT_C2R)
GPUMOD_VALUE(GPUFFT_C2C, CUFFT_C2C, HIPFFT_C2C)
GPUMOD_VALUE(GPUFFT_D2Z, CUFFT_D2Z, HIPFFT_D2Z)
GPUMOD_VALUE(GPUFFT_Z2D, CUFFT_Z2D, HIPFFT_Z2D)
GPUMOD_VALUE(GPUFFT_Z2Z, CUFFT_Z2Z, HIPFFT_Z2Z)

// Result codes -- the 14 shared by both enums. cuFFT-only (MISSING_DEPENDENCY,
// NVRTC/NVJITLINK/NVSHMEM_FAILURE) and hipFFT-only (INCOMPLETE_PARAMETER_LIST,
// PARSE_ERROR) codes are reached through the raw module.
GPUMOD_VALUE(GPUFFT_SUCCESS, CUFFT_SUCCESS, HIPFFT_SUCCESS)
GPUMOD_VALUE(GPUFFT_INVALID_PLAN, CUFFT_INVALID_PLAN, HIPFFT_INVALID_PLAN)
GPUMOD_VALUE(GPUFFT_ALLOC_FAILED, CUFFT_ALLOC_FAILED, HIPFFT_ALLOC_FAILED)
GPUMOD_VALUE(GPUFFT_INVALID_TYPE, CUFFT_INVALID_TYPE, HIPFFT_INVALID_TYPE)
GPUMOD_VALUE(GPUFFT_INVALID_VALUE, CUFFT_INVALID_VALUE, HIPFFT_INVALID_VALUE)
GPUMOD_VALUE(GPUFFT_INTERNAL_ERROR, CUFFT_INTERNAL_ERROR, HIPFFT_INTERNAL_ERROR)
GPUMOD_VALUE(GPUFFT_EXEC_FAILED, CUFFT_EXEC_FAILED, HIPFFT_EXEC_FAILED)
GPUMOD_VALUE(GPUFFT_SETUP_FAILED, CUFFT_SETUP_FAILED, HIPFFT_SETUP_FAILED)
GPUMOD_VALUE(GPUFFT_INVALID_SIZE, CUFFT_INVALID_SIZE, HIPFFT_INVALID_SIZE)
GPUMOD_VALUE(GPUFFT_UNALIGNED_DATA, CUFFT_UNALIGNED_DATA, HIPFFT_UNALIGNED_DATA)
GPUMOD_VALUE(GPUFFT_INVALID_DEVICE, CUFFT_INVALID_DEVICE, HIPFFT_INVALID_DEVICE)
GPUMOD_VALUE(GPUFFT_NO_WORKSPACE, CUFFT_NO_WORKSPACE, HIPFFT_NO_WORKSPACE)
GPUMOD_VALUE(GPUFFT_NOT_IMPLEMENTED, CUFFT_NOT_IMPLEMENTED, HIPFFT_NOT_IMPLEMENTED)
GPUMOD_VALUE(GPUFFT_NOT_SUPPORTED, CUFFT_NOT_SUPPORTED, HIPFFT_NOT_SUPPORTED)

// ========================================================================
// Status strings: neither backend ships a status-to-string function, so
// each backend gets its own hand-written switch over the result codes its
// raw module exports (cuFFT 18, hipFFT 16). Default covers the rest.
// ========================================================================

inline const char *gpufftGetStatusName(gpufftResult_t error) noexcept {
#if defined(GPUMOD_GPU_BACKEND_CUDA)
  switch (error) {
  case CUFFT_SUCCESS:
    return "CUFFT_SUCCESS";
  case CUFFT_INVALID_PLAN:
    return "CUFFT_INVALID_PLAN";
  case CUFFT_ALLOC_FAILED:
    return "CUFFT_ALLOC_FAILED";
  case CUFFT_INVALID_TYPE:
    return "CUFFT_INVALID_TYPE";
  case CUFFT_INVALID_VALUE:
    return "CUFFT_INVALID_VALUE";
  case CUFFT_INTERNAL_ERROR:
    return "CUFFT_INTERNAL_ERROR";
  case CUFFT_EXEC_FAILED:
    return "CUFFT_EXEC_FAILED";
  case CUFFT_SETUP_FAILED:
    return "CUFFT_SETUP_FAILED";
  case CUFFT_INVALID_SIZE:
    return "CUFFT_INVALID_SIZE";
  case CUFFT_UNALIGNED_DATA:
    return "CUFFT_UNALIGNED_DATA";
  case CUFFT_INVALID_DEVICE:
    return "CUFFT_INVALID_DEVICE";
  case CUFFT_NO_WORKSPACE:
    return "CUFFT_NO_WORKSPACE";
  case CUFFT_NOT_IMPLEMENTED:
    return "CUFFT_NOT_IMPLEMENTED";
  case CUFFT_NOT_SUPPORTED:
    return "CUFFT_NOT_SUPPORTED";
  case CUFFT_MISSING_DEPENDENCY:
    return "CUFFT_MISSING_DEPENDENCY";
  case CUFFT_NVRTC_FAILURE:
    return "CUFFT_NVRTC_FAILURE";
  case CUFFT_NVJITLINK_FAILURE:
    return "CUFFT_NVJITLINK_FAILURE";
  case CUFFT_NVSHMEM_FAILURE:
    return "CUFFT_NVSHMEM_FAILURE";
  default:
    return "CUFFT_STATUS_UNKNOWN";
  }
#else
  switch (error) {
  case HIPFFT_SUCCESS:
    return "HIPFFT_SUCCESS";
  case HIPFFT_INVALID_PLAN:
    return "HIPFFT_INVALID_PLAN";
  case HIPFFT_ALLOC_FAILED:
    return "HIPFFT_ALLOC_FAILED";
  case HIPFFT_INVALID_TYPE:
    return "HIPFFT_INVALID_TYPE";
  case HIPFFT_INVALID_VALUE:
    return "HIPFFT_INVALID_VALUE";
  case HIPFFT_INTERNAL_ERROR:
    return "HIPFFT_INTERNAL_ERROR";
  case HIPFFT_EXEC_FAILED:
    return "HIPFFT_EXEC_FAILED";
  case HIPFFT_SETUP_FAILED:
    return "HIPFFT_SETUP_FAILED";
  case HIPFFT_INVALID_SIZE:
    return "HIPFFT_INVALID_SIZE";
  case HIPFFT_UNALIGNED_DATA:
    return "HIPFFT_UNALIGNED_DATA";
  case HIPFFT_INCOMPLETE_PARAMETER_LIST:
    return "HIPFFT_INCOMPLETE_PARAMETER_LIST";
  case HIPFFT_INVALID_DEVICE:
    return "HIPFFT_INVALID_DEVICE";
  case HIPFFT_PARSE_ERROR:
    return "HIPFFT_PARSE_ERROR";
  case HIPFFT_NO_WORKSPACE:
    return "HIPFFT_NO_WORKSPACE";
  case HIPFFT_NOT_IMPLEMENTED:
    return "HIPFFT_NOT_IMPLEMENTED";
  case HIPFFT_NOT_SUPPORTED:
    return "HIPFFT_NOT_SUPPORTED";
  default:
    return "HIPFFT_STATUS_UNKNOWN";
  }
#endif
}

inline const char *gpufftGetStatusString(gpufftResult_t error) noexcept {
#if defined(GPUMOD_GPU_BACKEND_CUDA)
  switch (error) {
  case CUFFT_SUCCESS:
    return "the operation completed successfully";
  case CUFFT_INVALID_PLAN:
    return "the plan handle is invalid";
  case CUFFT_ALLOC_FAILED:
    return "resource allocation failed";
  case CUFFT_INVALID_TYPE:
    return "an invalid transform type was requested";
  case CUFFT_INVALID_VALUE:
    return "an invalid value was provided as argument";
  case CUFFT_INTERNAL_ERROR:
    return "an internal operation failed";
  case CUFFT_EXEC_FAILED:
    return "the FFT failed to execute on the GPU";
  case CUFFT_SETUP_FAILED:
    return "the cuFFT library failed to initialize";
  case CUFFT_INVALID_SIZE:
    return "an invalid transform size was requested";
  case CUFFT_UNALIGNED_DATA:
    return "the input or output data is not correctly aligned";
  case CUFFT_INVALID_DEVICE:
    return "the plan was created on a different device";
  case CUFFT_NO_WORKSPACE:
    return "no workspace has been provided for the plan";
  case CUFFT_NOT_IMPLEMENTED:
    return "the requested functionality is not implemented";
  case CUFFT_NOT_SUPPORTED:
    return "the operation is not supported";
  case CUFFT_MISSING_DEPENDENCY:
    return "a required runtime dependency is missing";
  case CUFFT_NVRTC_FAILURE:
    return "the NVRTC runtime compilation failed";
  case CUFFT_NVJITLINK_FAILURE:
    return "the nvJitLink call failed";
  case CUFFT_NVSHMEM_FAILURE:
    return "an NVSHMEM operation failed";
  default:
    return "unknown cuFFT error";
  }
#else
  switch (error) {
  case HIPFFT_SUCCESS:
    return "the operation completed successfully";
  case HIPFFT_INVALID_PLAN:
    return "the plan handle is invalid";
  case HIPFFT_ALLOC_FAILED:
    return "resource allocation failed";
  case HIPFFT_INVALID_TYPE:
    return "an invalid transform type was requested";
  case HIPFFT_INVALID_VALUE:
    return "an invalid value was provided as argument";
  case HIPFFT_INTERNAL_ERROR:
    return "an internal operation failed";
  case HIPFFT_EXEC_FAILED:
    return "the FFT failed to execute on the GPU";
  case HIPFFT_SETUP_FAILED:
    return "the hipFFT library failed to initialize";
  case HIPFFT_INVALID_SIZE:
    return "an invalid transform size was requested";
  case HIPFFT_UNALIGNED_DATA:
    return "the input or output data is not correctly aligned";
  case HIPFFT_INCOMPLETE_PARAMETER_LIST:
    return "the parameter list is incomplete";
  case HIPFFT_INVALID_DEVICE:
    return "the plan was created on a different device";
  case HIPFFT_PARSE_ERROR:
    return "an internal plan-parse error occurred";
  case HIPFFT_NO_WORKSPACE:
    return "no workspace has been provided for the plan";
  case HIPFFT_NOT_IMPLEMENTED:
    return "the requested functionality is not implemented";
  case HIPFFT_NOT_SUPPORTED:
    return "the operation is not supported";
  default:
    return "unknown hipFFT error";
  }
#endif
}

// ========================================================================
// Lifecycle
// ========================================================================

GPUMOD_FUNCTION(gpufftCreate, cufftCreate, hipfftCreate)
GPUMOD_FUNCTION(gpufftDestroy, cufftDestroy, hipfftDestroy)
GPUMOD_FUNCTION(gpufftSetStream, cufftSetStream, hipfftSetStream)
GPUMOD_FUNCTION(gpufftGetVersion, cufftGetVersion, hipfftGetVersion)

// ========================================================================
// Plan creation (one-shot)
// ========================================================================

GPUMOD_FUNCTION(gpufftPlan1d, cufftPlan1d, hipfftPlan1d)
GPUMOD_FUNCTION(gpufftPlan2d, cufftPlan2d, hipfftPlan2d)
GPUMOD_FUNCTION(gpufftPlan3d, cufftPlan3d, hipfftPlan3d)
GPUMOD_FUNCTION(gpufftPlanMany, cufftPlanMany, hipfftPlanMany)

// ========================================================================
// Plan make (two-step: gpufftCreate, then MakePlan*)
// ========================================================================

GPUMOD_FUNCTION(gpufftMakePlan1d, cufftMakePlan1d, hipfftMakePlan1d)
GPUMOD_FUNCTION(gpufftMakePlan2d, cufftMakePlan2d, hipfftMakePlan2d)
GPUMOD_FUNCTION(gpufftMakePlan3d, cufftMakePlan3d, hipfftMakePlan3d)
GPUMOD_FUNCTION(gpufftMakePlanMany, cufftMakePlanMany, hipfftMakePlanMany)
GPUMOD_FUNCTION(gpufftMakePlanMany64, cufftMakePlanMany64, hipfftMakePlanMany64)

// ========================================================================
// Work size estimation
// ========================================================================

GPUMOD_FUNCTION(gpufftEstimate1d, cufftEstimate1d, hipfftEstimate1d)
GPUMOD_FUNCTION(gpufftEstimate2d, cufftEstimate2d, hipfftEstimate2d)
GPUMOD_FUNCTION(gpufftEstimate3d, cufftEstimate3d, hipfftEstimate3d)
GPUMOD_FUNCTION(gpufftEstimateMany, cufftEstimateMany, hipfftEstimateMany)

// ========================================================================
// Work size query
// ========================================================================

GPUMOD_FUNCTION(gpufftGetSize1d, cufftGetSize1d, hipfftGetSize1d)
GPUMOD_FUNCTION(gpufftGetSize2d, cufftGetSize2d, hipfftGetSize2d)
GPUMOD_FUNCTION(gpufftGetSize3d, cufftGetSize3d, hipfftGetSize3d)
GPUMOD_FUNCTION(gpufftGetSizeMany, cufftGetSizeMany, hipfftGetSizeMany)
GPUMOD_FUNCTION(gpufftGetSizeMany64, cufftGetSizeMany64, hipfftGetSizeMany64)
GPUMOD_FUNCTION(gpufftGetSize, cufftGetSize, hipfftGetSize)

// ========================================================================
// Work area management
// ========================================================================

GPUMOD_FUNCTION(gpufftSetWorkArea, cufftSetWorkArea, hipfftSetWorkArea)
GPUMOD_FUNCTION(gpufftSetAutoAllocation, cufftSetAutoAllocation, hipfftSetAutoAllocation)

// ========================================================================
// Execution
// ========================================================================

GPUMOD_FUNCTION(gpufftExecC2C, cufftExecC2C, hipfftExecC2C)
GPUMOD_FUNCTION(gpufftExecR2C, cufftExecR2C, hipfftExecR2C)
GPUMOD_FUNCTION(gpufftExecC2R, cufftExecC2R, hipfftExecC2R)
GPUMOD_FUNCTION(gpufftExecZ2Z, cufftExecZ2Z, hipfftExecZ2Z)
GPUMOD_FUNCTION(gpufftExecD2Z, cufftExecD2Z, hipfftExecD2Z)
GPUMOD_FUNCTION(gpufftExecZ2D, cufftExecZ2D, hipfftExecZ2D)

} // namespace gpumod
