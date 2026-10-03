/**
 * @file detail/fft_names.h
 * @brief The backend-neutral cuFFT / hipFFT surface, as a macro-driven include
 *        fragment shared by the module and the non-module #include path
 *
 * NOT a standalone header: it is the list of wwrfft* / WWRFFT_* names (types,
 * constants, the two hand-written status-string switches and the functions) with
 * NO namespace of its own and NO vendor #include. The includer supplies all of
 * that and pastes this inside its own `namespace wwr` -- so one list binds both
 * ways the surface is consumed: wwr.fft (the module, `export namespace wwr`) and
 * wwr/fft.h (the non-module #include path). Add a name here, once, and both paths
 * gain it.
 *
 * HOST only -- the cuFFT / hipFFT plan and exec API is a host API. The surface
 * binds straight to the vendor's external-linkage `::cufft*` / `::hipfft*`
 * declarations via the _RAW macros (the "rand.h shape"): no raw vendor module is
 * imported, so the same `::`-prefixed names resolve in the module (vendor header
 * in its GMF) and the #include path alike. The multi-GPU eXtended surface is
 * deliberately absent; reach it through wwr.cuda.cufftXt / wwr.hip.hipfftXt.
 *
 * Before including, the includer must have, in order:
 *   - the vendor header in scope (cufft.h / hipfft/hipfft.h), which fft.h pulls
 *     in;
 *   - WWR_SELECT_RAW(cuda, hip) plus WWR_TYPE_RAW / WWR_VALUE_RAW /
 *     WWR_FUNCTION_RAW on top of it -- keyed on WWR_GPU_BACKEND_* in the module
 *     (backend.h) or WWR_SELECTED_* in the #include path (wwr/fft.h);
 *   - WWR_SELECTED_CUDA / WWR_SELECTED_HIP (selected_backend.h, via fft.h) for
 *     the two status-string switches' backend conditional.
 *
 * The only hand-written functions are the status-string switches, which take a
 * result-code enum and return a `const char *`. See src/fft.cppm, src/fft.h,
 * src/wwr/fft.h and docs/architecture.md section 5.
 */

#pragma once

#ifndef WWR_FUNCTION_RAW
#error                                                                                             \
    "detail/fft_names.h is an include fragment, not a standalone header: define WWR_TYPE_RAW/VALUE_RAW/FUNCTION_RAW and WWR_SELECT_RAW, ensure the vendor header (via fft.h) and WWR_SELECTED_* are in scope, and #include it inside namespace wwr. See src/fft.h, src/fft.cppm and src/wwr/fft.h."
#endif

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables): each wwrfft*
// function below is a deliberate constexpr reference to the selected backend's
// entry point (via WWR_FUNCTION_RAW). A reference to a vendor function has no
// const form, so the check cannot be satisfied without abandoning the alias
// pattern -- see backend.h and detail/blas_names.h.

// ========================================================================
// Types
// ========================================================================

WWR_TYPE_RAW(wwrfftHandle, cufftHandle, hipfftHandle)
WWR_TYPE_RAW(wwrfftResult_t, cufftResult_t, hipfftResult_t)
WWR_TYPE_RAW(wwrfftType_t, cufftType_t, hipfftType_t)
WWR_TYPE_RAW(wwrfftReal, cufftReal, hipfftReal)
WWR_TYPE_RAW(wwrfftDoubleReal, cufftDoubleReal, hipfftDoubleReal)
WWR_TYPE_RAW(wwrfftComplex, cufftComplex, hipfftComplex)
WWR_TYPE_RAW(wwrfftDoubleComplex, cufftDoubleComplex, hipfftDoubleComplex)

// ========================================================================
// Constants
// ========================================================================

// FFT direction flags (the `direction` argument to wwrfftExecC2C/Z2Z).
// cuFFT: CUFFT_FORWARD/CUFFT_INVERSE; hipFFT: HIPFFT_FORWARD/HIPFFT_BACKWARD.
// These are `#define`d plain-int macros in the vendor header (not enumerators),
// so WWR_VALUE_RAW's `::` qualification does not apply -- `::CUFFT_FORWARD`
// expands to `::-1`. Written as a direct #if instead, the same way solver.cppm's
// cudaDataType / hipDataType note once did: bind straight to the live macro,
// whose value the raw module's static_asserts (cufft.cppm / hipfft.cppm) pin.
#if defined(WWR_SELECTED_CUDA)
inline constexpr int WWRFFT_FORWARD = CUFFT_FORWARD;
inline constexpr int WWRFFT_INVERSE = CUFFT_INVERSE;
#else
inline constexpr int WWRFFT_FORWARD = HIPFFT_FORWARD;
inline constexpr int WWRFFT_INVERSE = HIPFFT_BACKWARD;
#endif

// Transform type (the `type` argument to the plan-creation functions).
WWR_VALUE_RAW(WWRFFT_R2C, CUFFT_R2C, HIPFFT_R2C)
WWR_VALUE_RAW(WWRFFT_C2R, CUFFT_C2R, HIPFFT_C2R)
WWR_VALUE_RAW(WWRFFT_C2C, CUFFT_C2C, HIPFFT_C2C)
WWR_VALUE_RAW(WWRFFT_D2Z, CUFFT_D2Z, HIPFFT_D2Z)
WWR_VALUE_RAW(WWRFFT_Z2D, CUFFT_Z2D, HIPFFT_Z2D)
WWR_VALUE_RAW(WWRFFT_Z2Z, CUFFT_Z2Z, HIPFFT_Z2Z)

// Result codes -- the 14 shared by both enums. cuFFT-only (MISSING_DEPENDENCY,
// NVRTC/NVJITLINK/NVSHMEM_FAILURE) and hipFFT-only (INCOMPLETE_PARAMETER_LIST,
// PARSE_ERROR) codes are reached through the raw module.
WWR_VALUE_RAW(WWRFFT_SUCCESS, CUFFT_SUCCESS, HIPFFT_SUCCESS)
WWR_VALUE_RAW(WWRFFT_INVALID_PLAN, CUFFT_INVALID_PLAN, HIPFFT_INVALID_PLAN)
WWR_VALUE_RAW(WWRFFT_ALLOC_FAILED, CUFFT_ALLOC_FAILED, HIPFFT_ALLOC_FAILED)
WWR_VALUE_RAW(WWRFFT_INVALID_TYPE, CUFFT_INVALID_TYPE, HIPFFT_INVALID_TYPE)
WWR_VALUE_RAW(WWRFFT_INVALID_VALUE, CUFFT_INVALID_VALUE, HIPFFT_INVALID_VALUE)
WWR_VALUE_RAW(WWRFFT_INTERNAL_ERROR, CUFFT_INTERNAL_ERROR, HIPFFT_INTERNAL_ERROR)
WWR_VALUE_RAW(WWRFFT_EXEC_FAILED, CUFFT_EXEC_FAILED, HIPFFT_EXEC_FAILED)
WWR_VALUE_RAW(WWRFFT_SETUP_FAILED, CUFFT_SETUP_FAILED, HIPFFT_SETUP_FAILED)
WWR_VALUE_RAW(WWRFFT_INVALID_SIZE, CUFFT_INVALID_SIZE, HIPFFT_INVALID_SIZE)
WWR_VALUE_RAW(WWRFFT_UNALIGNED_DATA, CUFFT_UNALIGNED_DATA, HIPFFT_UNALIGNED_DATA)
WWR_VALUE_RAW(WWRFFT_INVALID_DEVICE, CUFFT_INVALID_DEVICE, HIPFFT_INVALID_DEVICE)
WWR_VALUE_RAW(WWRFFT_NO_WORKSPACE, CUFFT_NO_WORKSPACE, HIPFFT_NO_WORKSPACE)
WWR_VALUE_RAW(WWRFFT_NOT_IMPLEMENTED, CUFFT_NOT_IMPLEMENTED, HIPFFT_NOT_IMPLEMENTED)
WWR_VALUE_RAW(WWRFFT_NOT_SUPPORTED, CUFFT_NOT_SUPPORTED, HIPFFT_NOT_SUPPORTED)

// ========================================================================
// Status strings: neither backend ships a status-to-string function, so
// each backend gets its own hand-written switch over the result codes its
// enum exports (cuFFT 18, hipFFT 16). Default covers the rest.
// ========================================================================

inline const char *wwrfftGetStatusName(wwrfftResult_t error) noexcept {
#if defined(WWR_SELECTED_CUDA)
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

inline const char *wwrfftGetStatusString(wwrfftResult_t error) noexcept {
#if defined(WWR_SELECTED_CUDA)
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

WWR_FUNCTION_RAW(wwrfftCreate, cufftCreate, hipfftCreate)
WWR_FUNCTION_RAW(wwrfftDestroy, cufftDestroy, hipfftDestroy)
WWR_FUNCTION_RAW(wwrfftSetStream, cufftSetStream, hipfftSetStream)
WWR_FUNCTION_RAW(wwrfftGetVersion, cufftGetVersion, hipfftGetVersion)

// ========================================================================
// Plan creation (one-shot)
// ========================================================================

WWR_FUNCTION_RAW(wwrfftPlan1d, cufftPlan1d, hipfftPlan1d)
WWR_FUNCTION_RAW(wwrfftPlan2d, cufftPlan2d, hipfftPlan2d)
WWR_FUNCTION_RAW(wwrfftPlan3d, cufftPlan3d, hipfftPlan3d)
WWR_FUNCTION_RAW(wwrfftPlanMany, cufftPlanMany, hipfftPlanMany)

// ========================================================================
// Plan make (two-step: wwrfftCreate, then MakePlan*)
// ========================================================================

WWR_FUNCTION_RAW(wwrfftMakePlan1d, cufftMakePlan1d, hipfftMakePlan1d)
WWR_FUNCTION_RAW(wwrfftMakePlan2d, cufftMakePlan2d, hipfftMakePlan2d)
WWR_FUNCTION_RAW(wwrfftMakePlan3d, cufftMakePlan3d, hipfftMakePlan3d)
WWR_FUNCTION_RAW(wwrfftMakePlanMany, cufftMakePlanMany, hipfftMakePlanMany)
WWR_FUNCTION_RAW(wwrfftMakePlanMany64, cufftMakePlanMany64, hipfftMakePlanMany64)

// ========================================================================
// Work size estimation
// ========================================================================

WWR_FUNCTION_RAW(wwrfftEstimate1d, cufftEstimate1d, hipfftEstimate1d)
WWR_FUNCTION_RAW(wwrfftEstimate2d, cufftEstimate2d, hipfftEstimate2d)
WWR_FUNCTION_RAW(wwrfftEstimate3d, cufftEstimate3d, hipfftEstimate3d)
WWR_FUNCTION_RAW(wwrfftEstimateMany, cufftEstimateMany, hipfftEstimateMany)

// ========================================================================
// Work size query
// ========================================================================

WWR_FUNCTION_RAW(wwrfftGetSize1d, cufftGetSize1d, hipfftGetSize1d)
WWR_FUNCTION_RAW(wwrfftGetSize2d, cufftGetSize2d, hipfftGetSize2d)
WWR_FUNCTION_RAW(wwrfftGetSize3d, cufftGetSize3d, hipfftGetSize3d)
WWR_FUNCTION_RAW(wwrfftGetSizeMany, cufftGetSizeMany, hipfftGetSizeMany)
WWR_FUNCTION_RAW(wwrfftGetSizeMany64, cufftGetSizeMany64, hipfftGetSizeMany64)
WWR_FUNCTION_RAW(wwrfftGetSize, cufftGetSize, hipfftGetSize)

// ========================================================================
// Work area management
// ========================================================================

WWR_FUNCTION_RAW(wwrfftSetWorkArea, cufftSetWorkArea, hipfftSetWorkArea)
WWR_FUNCTION_RAW(wwrfftSetAutoAllocation, cufftSetAutoAllocation, hipfftSetAutoAllocation)

// ========================================================================
// Execution
// ========================================================================

WWR_FUNCTION_RAW(wwrfftExecC2C, cufftExecC2C, hipfftExecC2C)
WWR_FUNCTION_RAW(wwrfftExecR2C, cufftExecR2C, hipfftExecR2C)
WWR_FUNCTION_RAW(wwrfftExecC2R, cufftExecC2R, hipfftExecC2R)
WWR_FUNCTION_RAW(wwrfftExecZ2Z, cufftExecZ2Z, hipfftExecZ2Z)
WWR_FUNCTION_RAW(wwrfftExecD2Z, cufftExecD2Z, hipfftExecD2Z)
WWR_FUNCTION_RAW(wwrfftExecZ2D, cufftExecZ2D, hipfftExecZ2D)

// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)
