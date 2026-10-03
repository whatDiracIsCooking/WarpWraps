/**
 * @file detail/solver_names.h
 * @brief The backend-neutral cuSOLVER Dense / hipSOLVER surface, as a
 *        macro-driven include fragment shared by the module and the non-module
 *        #include path
 *
 * NOT a standalone header: it is the list of wwrsolver* / WWRSOLVER_* names
 * (types, constants, the two hand-written status-string switches and the
 * functions) with NO namespace of its own and NO vendor #include. The includer
 * supplies all of that and pastes this inside its own `namespace wwr` -- so one
 * list binds both ways the surface is consumed: wwr.solver (the module,
 * `export namespace wwr`) and wwr/solver.h (the non-module #include path). Add a
 * name here, once, and both paths gain it.
 *
 * HOST only -- cuSOLVER / hipSOLVER is a host API. The surface binds straight to
 * the vendor's external-linkage `::cusolverDn*` / `::hipsolverDn*` declarations
 * via the _RAW macros (the "rand.h shape"): no raw vendor module is imported, so
 * the same `::`-prefixed names resolve in the module (vendor header in its GMF)
 * and the #include path alike. The CUDA-only modern eigen/SVD surface (Xlarft,
 * Xsytrs, Xtrtri, the modern eigenvalue/SVD API) is deliberately absent; reach
 * it through wwr.cuda.cusolverDn.
 *
 * Before including, the includer must have, in order:
 *   - the vendor header in scope (cusolverDn.h / hipsolver.h) and library_types.h
 *     for the cudaDataType / hipDataType enumerators, which solver.h pulls in;
 *   - WWR_SELECT_RAW(cuda, hip) plus WWR_TYPE_RAW / WWR_VALUE_RAW /
 *     WWR_FUNCTION_RAW on top of it -- keyed on WWR_GPU_BACKEND_* in the module
 *     (backend.h) or WWR_SELECTED_* in the #include path (wwr/solver.h);
 *   - WWR_SELECTED_CUDA / WWR_SELECTED_HIP (selected_backend.h, via solver.h) for
 *     the two status-string switches' backend conditional.
 *
 * Unlike detail/blas_names.h this fragment names no complex type: its only
 * hand-written functions are the status-string switches, which take a status
 * enum and return a `const char *`. See src/solver.cppm, src/solver.h,
 * src/wwr/solver.h and docs/architecture.md section 5.
 */

#pragma once

#ifndef WWR_FUNCTION_RAW
#error                                                                                             \
    "detail/solver_names.h is an include fragment, not a standalone header: define WWR_TYPE_RAW/VALUE_RAW/FUNCTION_RAW and WWR_SELECT_RAW, ensure the vendor header and library_types.h (via solver.h) and WWR_SELECTED_* are in scope, and #include it inside namespace wwr. See src/solver.h, src/solver.cppm and src/wwr/solver.h."
#endif

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables): each wwrsolver*
// function below is a deliberate constexpr reference to the selected backend's
// entry point (via WWR_FUNCTION_RAW). A reference to a vendor function has no
// const form, so the check cannot be satisfied without abandoning the alias
// pattern -- see backend.h and detail/blas_names.h.

// ========================================================================
// Types
// ========================================================================

WWR_TYPE_RAW(wwrsolverDnHandle_t, cusolverDnHandle_t, hipsolverDnHandle_t)
WWR_TYPE_RAW(wwrsolverDnParams_t, cusolverDnParams_t, hipsolverDnParams_t)
WWR_TYPE_RAW(wwrsolverStatus_t, cusolverStatus_t, hipsolverStatus_t)
WWR_TYPE_RAW(wwrsolverEigMode_t, cusolverEigMode_t, hipsolverEigMode_t)
WWR_TYPE_RAW(wwrsolverEigType_t, cusolverEigType_t, hipsolverEigType_t)
WWR_TYPE_RAW(wwrsolverEigRange_t, cusolverEigRange_t, hipsolverEigRange_t)
WWR_TYPE_RAW(wwrsolverSyevjInfo_t, syevjInfo_t, hipsolverSyevjInfo_t)
WWR_TYPE_RAW(wwrsolverGesvdjInfo_t, gesvdjInfo_t, hipsolverGesvdjInfo_t)
WWR_TYPE_RAW(wwrsolverDataType_t, cudaDataType, hipDataType)

// ========================================================================
// Constants
// ========================================================================

WWR_VALUE_RAW(WWRSOLVER_STATUS_SUCCESS, CUSOLVER_STATUS_SUCCESS, HIPSOLVER_STATUS_SUCCESS)

WWR_VALUE_RAW(WWRSOLVER_EIG_MODE_NOVECTOR, CUSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_EIG_MODE_NOVECTOR)
WWR_VALUE_RAW(WWRSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_MODE_VECTOR, HIPSOLVER_EIG_MODE_VECTOR)

WWR_VALUE_RAW(WWRSOLVER_EIG_RANGE_ALL, CUSOLVER_EIG_RANGE_ALL, HIPSOLVER_EIG_RANGE_ALL)
WWR_VALUE_RAW(WWRSOLVER_EIG_RANGE_V, CUSOLVER_EIG_RANGE_V, HIPSOLVER_EIG_RANGE_V)
WWR_VALUE_RAW(WWRSOLVER_EIG_RANGE_I, CUSOLVER_EIG_RANGE_I, HIPSOLVER_EIG_RANGE_I)

WWR_VALUE_RAW(WWRSOLVER_EIG_TYPE_1, CUSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_TYPE_1)
WWR_VALUE_RAW(WWRSOLVER_EIG_TYPE_2, CUSOLVER_EIG_TYPE_2, HIPSOLVER_EIG_TYPE_2)
WWR_VALUE_RAW(WWRSOLVER_EIG_TYPE_3, CUSOLVER_EIG_TYPE_3, HIPSOLVER_EIG_TYPE_3)

// The cudaDataType / hipDataType enumerators, bound straight to the vendor's own
// (via WWR_VALUE_RAW, so each carries the backend enum type that is exactly
// wwrsolverDataType_t). They come from library_types.h (via solver.h), not from a
// raw module, which is what lets the _RAW binding resolve them here.
WWR_VALUE_RAW(WWRSOLVER_R_32F, CUDA_R_32F, HIP_R_32F)
WWR_VALUE_RAW(WWRSOLVER_R_64F, CUDA_R_64F, HIP_R_64F)
WWR_VALUE_RAW(WWRSOLVER_C_32F, CUDA_C_32F, HIP_C_32F)
WWR_VALUE_RAW(WWRSOLVER_C_64F, CUDA_C_64F, HIP_C_64F)

// ========================================================================
// Status strings: neither backend has an official status-to-string
// function, and the two enumerator sets differ (hipSOLVER has 14 values,
// including HIPSOLVER_STATUS_HANDLE_IS_NULLPTR/INVALID_ENUM; cuSOLVER has 26,
// including the IRS-refinement-specific codes hipSOLVER has no equivalent
// of), so each backend gets its own hand-written switch.
// ========================================================================

inline const char *wwrsolverGetStatusName(wwrsolverStatus_t error) noexcept {
#if defined(WWR_SELECTED_CUDA)
  switch (error) {
  case CUSOLVER_STATUS_SUCCESS:
    return "CUSOLVER_STATUS_SUCCESS";
  case CUSOLVER_STATUS_NOT_INITIALIZED:
    return "CUSOLVER_STATUS_NOT_INITIALIZED";
  case CUSOLVER_STATUS_ALLOC_FAILED:
    return "CUSOLVER_STATUS_ALLOC_FAILED";
  case CUSOLVER_STATUS_INVALID_VALUE:
    return "CUSOLVER_STATUS_INVALID_VALUE";
  case CUSOLVER_STATUS_ARCH_MISMATCH:
    return "CUSOLVER_STATUS_ARCH_MISMATCH";
  case CUSOLVER_STATUS_MAPPING_ERROR:
    return "CUSOLVER_STATUS_MAPPING_ERROR";
  case CUSOLVER_STATUS_EXECUTION_FAILED:
    return "CUSOLVER_STATUS_EXECUTION_FAILED";
  case CUSOLVER_STATUS_INTERNAL_ERROR:
    return "CUSOLVER_STATUS_INTERNAL_ERROR";
  case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
    return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
  case CUSOLVER_STATUS_NOT_SUPPORTED:
    return "CUSOLVER_STATUS_NOT_SUPPORTED";
  case CUSOLVER_STATUS_ZERO_PIVOT:
    return "CUSOLVER_STATUS_ZERO_PIVOT";
  case CUSOLVER_STATUS_INVALID_LICENSE:
    return "CUSOLVER_STATUS_INVALID_LICENSE";
  case CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED:
    return "CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED";
  case CUSOLVER_STATUS_IRS_PARAMS_INVALID:
    return "CUSOLVER_STATUS_IRS_PARAMS_INVALID";
  case CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC:
    return "CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC";
  case CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE:
    return "CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE";
  case CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER:
    return "CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER";
  case CUSOLVER_STATUS_IRS_INTERNAL_ERROR:
    return "CUSOLVER_STATUS_IRS_INTERNAL_ERROR";
  case CUSOLVER_STATUS_IRS_NOT_SUPPORTED:
    return "CUSOLVER_STATUS_IRS_NOT_SUPPORTED";
  case CUSOLVER_STATUS_IRS_OUT_OF_RANGE:
    return "CUSOLVER_STATUS_IRS_OUT_OF_RANGE";
  case CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES:
    return "CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES";
  case CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED:
    return "CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED";
  case CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED:
    return "CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED";
  case CUSOLVER_STATUS_IRS_MATRIX_SINGULAR:
    return "CUSOLVER_STATUS_IRS_MATRIX_SINGULAR";
  case CUSOLVER_STATUS_INVALID_WORKSPACE:
    return "CUSOLVER_STATUS_INVALID_WORKSPACE";
  default:
    return "CUSOLVER_STATUS_UNKNOWN";
  }
#else
  switch (error) {
  case HIPSOLVER_STATUS_SUCCESS:
    return "HIPSOLVER_STATUS_SUCCESS";
  case HIPSOLVER_STATUS_NOT_INITIALIZED:
    return "HIPSOLVER_STATUS_NOT_INITIALIZED";
  case HIPSOLVER_STATUS_ALLOC_FAILED:
    return "HIPSOLVER_STATUS_ALLOC_FAILED";
  case HIPSOLVER_STATUS_INVALID_VALUE:
    return "HIPSOLVER_STATUS_INVALID_VALUE";
  case HIPSOLVER_STATUS_MAPPING_ERROR:
    return "HIPSOLVER_STATUS_MAPPING_ERROR";
  case HIPSOLVER_STATUS_EXECUTION_FAILED:
    return "HIPSOLVER_STATUS_EXECUTION_FAILED";
  case HIPSOLVER_STATUS_INTERNAL_ERROR:
    return "HIPSOLVER_STATUS_INTERNAL_ERROR";
  case HIPSOLVER_STATUS_NOT_SUPPORTED:
    return "HIPSOLVER_STATUS_NOT_SUPPORTED";
  case HIPSOLVER_STATUS_ARCH_MISMATCH:
    return "HIPSOLVER_STATUS_ARCH_MISMATCH";
  case HIPSOLVER_STATUS_HANDLE_IS_NULLPTR:
    return "HIPSOLVER_STATUS_HANDLE_IS_NULLPTR";
  case HIPSOLVER_STATUS_INVALID_ENUM:
    return "HIPSOLVER_STATUS_INVALID_ENUM";
  case HIPSOLVER_STATUS_ZERO_PIVOT:
    return "HIPSOLVER_STATUS_ZERO_PIVOT";
  case HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
    return "HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
  default:
    return "HIPSOLVER_STATUS_UNKNOWN";
  }
#endif
}

inline const char *wwrsolverGetStatusString(wwrsolverStatus_t error) noexcept {
#if defined(WWR_SELECTED_CUDA)
  switch (error) {
  case CUSOLVER_STATUS_SUCCESS:
    return "the operation completed successfully";
  case CUSOLVER_STATUS_NOT_INITIALIZED:
    return "the library was not initialized";
  case CUSOLVER_STATUS_ALLOC_FAILED:
    return "resource allocation failed";
  case CUSOLVER_STATUS_INVALID_VALUE:
    return "an invalid value was provided as argument";
  case CUSOLVER_STATUS_ARCH_MISMATCH:
    return "the device architecture is not supported";
  case CUSOLVER_STATUS_MAPPING_ERROR:
    return "an error occurred while mapping memory";
  case CUSOLVER_STATUS_EXECUTION_FAILED:
    return "the GPU program failed to execute";
  case CUSOLVER_STATUS_INTERNAL_ERROR:
    return "an internal operation failed";
  case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
    return "the matrix type is not supported";
  case CUSOLVER_STATUS_NOT_SUPPORTED:
    return "the operation is not supported";
  case CUSOLVER_STATUS_ZERO_PIVOT:
    return "a zero pivot was encountered during factorization";
  case CUSOLVER_STATUS_INVALID_LICENSE:
    return "an invalid license was detected";
  case CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED:
    return "IRS parameters were not initialized";
  case CUSOLVER_STATUS_IRS_PARAMS_INVALID:
    return "IRS parameters are invalid";
  case CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC:
    return "IRS precision parameters are invalid";
  case CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE:
    return "IRS refinement parameters are invalid";
  case CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER:
    return "IRS maximum iteration parameters are invalid";
  case CUSOLVER_STATUS_IRS_INTERNAL_ERROR:
    return "an internal error occurred in IRS";
  case CUSOLVER_STATUS_IRS_NOT_SUPPORTED:
    return "the IRS operation is not supported";
  case CUSOLVER_STATUS_IRS_OUT_OF_RANGE:
    return "IRS parameter is out of valid range";
  case CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES:
    return "the number of right-hand sides is not supported for GMRES refinement";
  case CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED:
    return "IRS info structure was not initialized";
  case CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED:
    return "IRS info structure was not properly destroyed";
  case CUSOLVER_STATUS_IRS_MATRIX_SINGULAR:
    return "the matrix is singular in IRS solver";
  case CUSOLVER_STATUS_INVALID_WORKSPACE:
    return "the provided workspace is invalid";
  default:
    return "unknown cuSOLVER error";
  }
#else
  switch (error) {
  case HIPSOLVER_STATUS_SUCCESS:
    return "the operation completed successfully";
  case HIPSOLVER_STATUS_NOT_INITIALIZED:
    return "the library was not initialized";
  case HIPSOLVER_STATUS_ALLOC_FAILED:
    return "resource allocation failed";
  case HIPSOLVER_STATUS_INVALID_VALUE:
    return "an invalid value was provided as argument";
  case HIPSOLVER_STATUS_MAPPING_ERROR:
    return "an error occurred while mapping memory";
  case HIPSOLVER_STATUS_EXECUTION_FAILED:
    return "the GPU program failed to execute";
  case HIPSOLVER_STATUS_INTERNAL_ERROR:
    return "an internal operation failed";
  case HIPSOLVER_STATUS_NOT_SUPPORTED:
    return "the operation is not supported";
  case HIPSOLVER_STATUS_ARCH_MISMATCH:
    return "the device architecture is not supported";
  case HIPSOLVER_STATUS_HANDLE_IS_NULLPTR:
    return "the handle was null";
  case HIPSOLVER_STATUS_INVALID_ENUM:
    return "an invalid enum value was provided as argument";
  case HIPSOLVER_STATUS_ZERO_PIVOT:
    return "a zero pivot was encountered during factorization";
  case HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
    return "the matrix type is not supported";
  default:
    return "unknown hipSOLVER error";
  }
#endif
}

// ========================================================================
// Handle, params, Jacobi info objects
// ========================================================================

WWR_FUNCTION_RAW(wwrsolverDnCreate, cusolverDnCreate, hipsolverDnCreate)
WWR_FUNCTION_RAW(wwrsolverDnDestroy, cusolverDnDestroy, hipsolverDnDestroy)
WWR_FUNCTION_RAW(wwrsolverDnSetStream, cusolverDnSetStream, hipsolverDnSetStream)
WWR_FUNCTION_RAW(wwrsolverDnGetStream, cusolverDnGetStream, hipsolverDnGetStream)

WWR_FUNCTION_RAW(wwrsolverDnCreateParams, cusolverDnCreateParams, hipsolverDnCreateParams)
WWR_FUNCTION_RAW(wwrsolverDnDestroyParams, cusolverDnDestroyParams, hipsolverDnDestroyParams)

WWR_FUNCTION_RAW(wwrsolverDnCreateSyevjInfo, cusolverDnCreateSyevjInfo, hipsolverDnCreateSyevjInfo)
WWR_FUNCTION_RAW(wwrsolverDnDestroySyevjInfo, cusolverDnDestroySyevjInfo,
                hipsolverDnDestroySyevjInfo)
WWR_FUNCTION_RAW(wwrsolverDnCreateGesvdjInfo, cusolverDnCreateGesvdjInfo,
                hipsolverDnCreateGesvdjInfo)
WWR_FUNCTION_RAW(wwrsolverDnDestroyGesvdjInfo, cusolverDnDestroyGesvdjInfo,
                hipsolverDnDestroyGesvdjInfo)

// ────────────────────────────────────────────────────────────────────────
// Legacy linear solver API (int-based, pre-params) -- 32 functions, all
// shared between cuSOLVER and hipSOLVER
// ────────────────────────────────────────────────────────────────────────

WWR_FUNCTION_RAW(wwrsolverDnSpotrf_bufferSize, cusolverDnSpotrf_bufferSize,
                hipsolverDnSpotrf_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnDpotrf_bufferSize, cusolverDnDpotrf_bufferSize,
                hipsolverDnDpotrf_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnCpotrf_bufferSize, cusolverDnCpotrf_bufferSize,
                hipsolverDnCpotrf_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnZpotrf_bufferSize, cusolverDnZpotrf_bufferSize,
                hipsolverDnZpotrf_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnSpotrf, cusolverDnSpotrf, hipsolverDnSpotrf)
WWR_FUNCTION_RAW(wwrsolverDnDpotrf, cusolverDnDpotrf, hipsolverDnDpotrf)
WWR_FUNCTION_RAW(wwrsolverDnCpotrf, cusolverDnCpotrf, hipsolverDnCpotrf)
WWR_FUNCTION_RAW(wwrsolverDnZpotrf, cusolverDnZpotrf, hipsolverDnZpotrf)
WWR_FUNCTION_RAW(wwrsolverDnSpotrs, cusolverDnSpotrs, hipsolverDnSpotrs)
WWR_FUNCTION_RAW(wwrsolverDnDpotrs, cusolverDnDpotrs, hipsolverDnDpotrs)
WWR_FUNCTION_RAW(wwrsolverDnCpotrs, cusolverDnCpotrs, hipsolverDnCpotrs)
WWR_FUNCTION_RAW(wwrsolverDnZpotrs, cusolverDnZpotrs, hipsolverDnZpotrs)
WWR_FUNCTION_RAW(wwrsolverDnSpotri_bufferSize, cusolverDnSpotri_bufferSize,
                hipsolverDnSpotri_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnDpotri_bufferSize, cusolverDnDpotri_bufferSize,
                hipsolverDnDpotri_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnCpotri_bufferSize, cusolverDnCpotri_bufferSize,
                hipsolverDnCpotri_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnZpotri_bufferSize, cusolverDnZpotri_bufferSize,
                hipsolverDnZpotri_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnSpotri, cusolverDnSpotri, hipsolverDnSpotri)
WWR_FUNCTION_RAW(wwrsolverDnDpotri, cusolverDnDpotri, hipsolverDnDpotri)
WWR_FUNCTION_RAW(wwrsolverDnCpotri, cusolverDnCpotri, hipsolverDnCpotri)
WWR_FUNCTION_RAW(wwrsolverDnZpotri, cusolverDnZpotri, hipsolverDnZpotri)
WWR_FUNCTION_RAW(wwrsolverDnSgetrf_bufferSize, cusolverDnSgetrf_bufferSize,
                hipsolverDnSgetrf_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnDgetrf_bufferSize, cusolverDnDgetrf_bufferSize,
                hipsolverDnDgetrf_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnCgetrf_bufferSize, cusolverDnCgetrf_bufferSize,
                hipsolverDnCgetrf_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnZgetrf_bufferSize, cusolverDnZgetrf_bufferSize,
                hipsolverDnZgetrf_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnSgetrf, cusolverDnSgetrf, hipsolverDnSgetrf)
WWR_FUNCTION_RAW(wwrsolverDnDgetrf, cusolverDnDgetrf, hipsolverDnDgetrf)
WWR_FUNCTION_RAW(wwrsolverDnCgetrf, cusolverDnCgetrf, hipsolverDnCgetrf)
WWR_FUNCTION_RAW(wwrsolverDnZgetrf, cusolverDnZgetrf, hipsolverDnZgetrf)
WWR_FUNCTION_RAW(wwrsolverDnSgetrs, cusolverDnSgetrs, hipsolverDnSgetrs)
WWR_FUNCTION_RAW(wwrsolverDnDgetrs, cusolverDnDgetrs, hipsolverDnDgetrs)
WWR_FUNCTION_RAW(wwrsolverDnCgetrs, cusolverDnCgetrs, hipsolverDnCgetrs)
WWR_FUNCTION_RAW(wwrsolverDnZgetrs, cusolverDnZgetrs, hipsolverDnZgetrs)
WWR_FUNCTION_RAW(wwrsolverDnSgeqrf_bufferSize, cusolverDnSgeqrf_bufferSize,
                hipsolverDnSgeqrf_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnDgeqrf_bufferSize, cusolverDnDgeqrf_bufferSize,
                hipsolverDnDgeqrf_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnCgeqrf_bufferSize, cusolverDnCgeqrf_bufferSize,
                hipsolverDnCgeqrf_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnZgeqrf_bufferSize, cusolverDnZgeqrf_bufferSize,
                hipsolverDnZgeqrf_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnSgeqrf, cusolverDnSgeqrf, hipsolverDnSgeqrf)
WWR_FUNCTION_RAW(wwrsolverDnDgeqrf, cusolverDnDgeqrf, hipsolverDnDgeqrf)
WWR_FUNCTION_RAW(wwrsolverDnCgeqrf, cusolverDnCgeqrf, hipsolverDnCgeqrf)
WWR_FUNCTION_RAW(wwrsolverDnZgeqrf, cusolverDnZgeqrf, hipsolverDnZgeqrf)
WWR_FUNCTION_RAW(wwrsolverDnSormqr_bufferSize, cusolverDnSormqr_bufferSize,
                hipsolverDnSormqr_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnDormqr_bufferSize, cusolverDnDormqr_bufferSize,
                hipsolverDnDormqr_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnSormqr, cusolverDnSormqr, hipsolverDnSormqr)
WWR_FUNCTION_RAW(wwrsolverDnDormqr, cusolverDnDormqr, hipsolverDnDormqr)
WWR_FUNCTION_RAW(wwrsolverDnCunmqr_bufferSize, cusolverDnCunmqr_bufferSize,
                hipsolverDnCunmqr_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnZunmqr_bufferSize, cusolverDnZunmqr_bufferSize,
                hipsolverDnZunmqr_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnCunmqr, cusolverDnCunmqr, hipsolverDnCunmqr)
WWR_FUNCTION_RAW(wwrsolverDnZunmqr, cusolverDnZunmqr, hipsolverDnZunmqr)
WWR_FUNCTION_RAW(wwrsolverDnSSgels_bufferSize, cusolverDnSSgels_bufferSize,
                hipsolverDnSSgels_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnDDgels_bufferSize, cusolverDnDDgels_bufferSize,
                hipsolverDnDDgels_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnCCgels_bufferSize, cusolverDnCCgels_bufferSize,
                hipsolverDnCCgels_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnZZgels_bufferSize, cusolverDnZZgels_bufferSize,
                hipsolverDnZZgels_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnSSgels, cusolverDnSSgels, hipsolverDnSSgels)
WWR_FUNCTION_RAW(wwrsolverDnDDgels, cusolverDnDDgels, hipsolverDnDDgels)
WWR_FUNCTION_RAW(wwrsolverDnCCgels, cusolverDnCCgels, hipsolverDnCCgels)
WWR_FUNCTION_RAW(wwrsolverDnZZgels, cusolverDnZZgels, hipsolverDnZZgels)
WWR_FUNCTION_RAW(wwrsolverDnSSgesv_bufferSize, cusolverDnSSgesv_bufferSize,
                hipsolverDnSSgesv_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnDDgesv_bufferSize, cusolverDnDDgesv_bufferSize,
                hipsolverDnDDgesv_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnCCgesv_bufferSize, cusolverDnCCgesv_bufferSize,
                hipsolverDnCCgesv_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnZZgesv_bufferSize, cusolverDnZZgesv_bufferSize,
                hipsolverDnZZgesv_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnSSgesv, cusolverDnSSgesv, hipsolverDnSSgesv)
WWR_FUNCTION_RAW(wwrsolverDnDDgesv, cusolverDnDDgesv, hipsolverDnDDgesv)
WWR_FUNCTION_RAW(wwrsolverDnCCgesv, cusolverDnCCgesv, hipsolverDnCCgesv)
WWR_FUNCTION_RAW(wwrsolverDnZZgesv, cusolverDnZZgesv, hipsolverDnZZgesv)
WWR_FUNCTION_RAW(wwrsolverDnSpotrfBatched, cusolverDnSpotrfBatched, hipsolverDnSpotrfBatched)
WWR_FUNCTION_RAW(wwrsolverDnDpotrfBatched, cusolverDnDpotrfBatched, hipsolverDnDpotrfBatched)
WWR_FUNCTION_RAW(wwrsolverDnCpotrfBatched, cusolverDnCpotrfBatched, hipsolverDnCpotrfBatched)
WWR_FUNCTION_RAW(wwrsolverDnZpotrfBatched, cusolverDnZpotrfBatched, hipsolverDnZpotrfBatched)
WWR_FUNCTION_RAW(wwrsolverDnSpotrsBatched, cusolverDnSpotrsBatched, hipsolverDnSpotrsBatched)
WWR_FUNCTION_RAW(wwrsolverDnDpotrsBatched, cusolverDnDpotrsBatched, hipsolverDnDpotrsBatched)
WWR_FUNCTION_RAW(wwrsolverDnCpotrsBatched, cusolverDnCpotrsBatched, hipsolverDnCpotrsBatched)
WWR_FUNCTION_RAW(wwrsolverDnZpotrsBatched, cusolverDnZpotrsBatched, hipsolverDnZpotrsBatched)
WWR_FUNCTION_RAW(wwrsolverDnSsytrf_bufferSize, cusolverDnSsytrf_bufferSize,
                hipsolverDnSsytrf_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnDsytrf_bufferSize, cusolverDnDsytrf_bufferSize,
                hipsolverDnDsytrf_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnCsytrf_bufferSize, cusolverDnCsytrf_bufferSize,
                hipsolverDnCsytrf_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnZsytrf_bufferSize, cusolverDnZsytrf_bufferSize,
                hipsolverDnZsytrf_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnSsytrf, cusolverDnSsytrf, hipsolverDnSsytrf)
WWR_FUNCTION_RAW(wwrsolverDnDsytrf, cusolverDnDsytrf, hipsolverDnDsytrf)
WWR_FUNCTION_RAW(wwrsolverDnCsytrf, cusolverDnCsytrf, hipsolverDnCsytrf)
WWR_FUNCTION_RAW(wwrsolverDnZsytrf, cusolverDnZsytrf, hipsolverDnZsytrf)
WWR_FUNCTION_RAW(wwrsolverDnSgebrd_bufferSize, cusolverDnSgebrd_bufferSize,
                hipsolverDnSgebrd_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnDgebrd_bufferSize, cusolverDnDgebrd_bufferSize,
                hipsolverDnDgebrd_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnCgebrd_bufferSize, cusolverDnCgebrd_bufferSize,
                hipsolverDnCgebrd_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnZgebrd_bufferSize, cusolverDnZgebrd_bufferSize,
                hipsolverDnZgebrd_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnSgebrd, cusolverDnSgebrd, hipsolverDnSgebrd)
WWR_FUNCTION_RAW(wwrsolverDnDgebrd, cusolverDnDgebrd, hipsolverDnDgebrd)
WWR_FUNCTION_RAW(wwrsolverDnCgebrd, cusolverDnCgebrd, hipsolverDnCgebrd)
WWR_FUNCTION_RAW(wwrsolverDnZgebrd, cusolverDnZgebrd, hipsolverDnZgebrd)
WWR_FUNCTION_RAW(wwrsolverDnSorgqr_bufferSize, cusolverDnSorgqr_bufferSize,
                hipsolverDnSorgqr_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnDorgqr_bufferSize, cusolverDnDorgqr_bufferSize,
                hipsolverDnDorgqr_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnCungqr_bufferSize, cusolverDnCungqr_bufferSize,
                hipsolverDnCungqr_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnZungqr_bufferSize, cusolverDnZungqr_bufferSize,
                hipsolverDnZungqr_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnSorgqr, cusolverDnSorgqr, hipsolverDnSorgqr)
WWR_FUNCTION_RAW(wwrsolverDnDorgqr, cusolverDnDorgqr, hipsolverDnDorgqr)
WWR_FUNCTION_RAW(wwrsolverDnCungqr, cusolverDnCungqr, hipsolverDnCungqr)
WWR_FUNCTION_RAW(wwrsolverDnZungqr, cusolverDnZungqr, hipsolverDnZungqr)
WWR_FUNCTION_RAW(wwrsolverDnSorgbr_bufferSize, cusolverDnSorgbr_bufferSize,
                hipsolverDnSorgbr_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnDorgbr_bufferSize, cusolverDnDorgbr_bufferSize,
                hipsolverDnDorgbr_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnCungbr_bufferSize, cusolverDnCungbr_bufferSize,
                hipsolverDnCungbr_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnZungbr_bufferSize, cusolverDnZungbr_bufferSize,
                hipsolverDnZungbr_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnSorgbr, cusolverDnSorgbr, hipsolverDnSorgbr)
WWR_FUNCTION_RAW(wwrsolverDnDorgbr, cusolverDnDorgbr, hipsolverDnDorgbr)
WWR_FUNCTION_RAW(wwrsolverDnCungbr, cusolverDnCungbr, hipsolverDnCungbr)
WWR_FUNCTION_RAW(wwrsolverDnZungbr, cusolverDnZungbr, hipsolverDnZungbr)

// ────────────────────────────────────────────────────────────────────────
// Legacy eigenvalue/SVD solver API (int-based, pre-params) -- 48 functions,
// all shared between cuSOLVER and hipSOLVER
// ────────────────────────────────────────────────────────────────────────

WWR_FUNCTION_RAW(wwrsolverDnSgesvd_bufferSize, cusolverDnSgesvd_bufferSize,
                hipsolverDnSgesvd_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnDgesvd_bufferSize, cusolverDnDgesvd_bufferSize,
                hipsolverDnDgesvd_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnCgesvd_bufferSize, cusolverDnCgesvd_bufferSize,
                hipsolverDnCgesvd_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnZgesvd_bufferSize, cusolverDnZgesvd_bufferSize,
                hipsolverDnZgesvd_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnSgesvd, cusolverDnSgesvd, hipsolverDnSgesvd)
WWR_FUNCTION_RAW(wwrsolverDnDgesvd, cusolverDnDgesvd, hipsolverDnDgesvd)
WWR_FUNCTION_RAW(wwrsolverDnCgesvd, cusolverDnCgesvd, hipsolverDnCgesvd)
WWR_FUNCTION_RAW(wwrsolverDnZgesvd, cusolverDnZgesvd, hipsolverDnZgesvd)
WWR_FUNCTION_RAW(wwrsolverDnSsyevd_bufferSize, cusolverDnSsyevd_bufferSize,
                hipsolverDnSsyevd_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnDsyevd_bufferSize, cusolverDnDsyevd_bufferSize,
                hipsolverDnDsyevd_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnSsyevd, cusolverDnSsyevd, hipsolverDnSsyevd)
WWR_FUNCTION_RAW(wwrsolverDnDsyevd, cusolverDnDsyevd, hipsolverDnDsyevd)
WWR_FUNCTION_RAW(wwrsolverDnSsyevdx_bufferSize, cusolverDnSsyevdx_bufferSize,
                hipsolverDnSsyevdx_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnDsyevdx_bufferSize, cusolverDnDsyevdx_bufferSize,
                hipsolverDnDsyevdx_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnSsyevdx, cusolverDnSsyevdx, hipsolverDnSsyevdx)
WWR_FUNCTION_RAW(wwrsolverDnDsyevdx, cusolverDnDsyevdx, hipsolverDnDsyevdx)
WWR_FUNCTION_RAW(wwrsolverDnCheevd_bufferSize, cusolverDnCheevd_bufferSize,
                hipsolverDnCheevd_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnZheevd_bufferSize, cusolverDnZheevd_bufferSize,
                hipsolverDnZheevd_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnCheevd, cusolverDnCheevd, hipsolverDnCheevd)
WWR_FUNCTION_RAW(wwrsolverDnZheevd, cusolverDnZheevd, hipsolverDnZheevd)
WWR_FUNCTION_RAW(wwrsolverDnCheevdx_bufferSize, cusolverDnCheevdx_bufferSize,
                hipsolverDnCheevdx_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnZheevdx_bufferSize, cusolverDnZheevdx_bufferSize,
                hipsolverDnZheevdx_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnCheevdx, cusolverDnCheevdx, hipsolverDnCheevdx)
WWR_FUNCTION_RAW(wwrsolverDnZheevdx, cusolverDnZheevdx, hipsolverDnZheevdx)
WWR_FUNCTION_RAW(wwrsolverDnSsytrd_bufferSize, cusolverDnSsytrd_bufferSize,
                hipsolverDnSsytrd_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnDsytrd_bufferSize, cusolverDnDsytrd_bufferSize,
                hipsolverDnDsytrd_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnSsytrd, cusolverDnSsytrd, hipsolverDnSsytrd)
WWR_FUNCTION_RAW(wwrsolverDnDsytrd, cusolverDnDsytrd, hipsolverDnDsytrd)
WWR_FUNCTION_RAW(wwrsolverDnChetrd_bufferSize, cusolverDnChetrd_bufferSize,
                hipsolverDnChetrd_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnZhetrd_bufferSize, cusolverDnZhetrd_bufferSize,
                hipsolverDnZhetrd_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnChetrd, cusolverDnChetrd, hipsolverDnChetrd)
WWR_FUNCTION_RAW(wwrsolverDnZhetrd, cusolverDnZhetrd, hipsolverDnZhetrd)
WWR_FUNCTION_RAW(wwrsolverDnSorgtr_bufferSize, cusolverDnSorgtr_bufferSize,
                hipsolverDnSorgtr_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnDorgtr_bufferSize, cusolverDnDorgtr_bufferSize,
                hipsolverDnDorgtr_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnSorgtr, cusolverDnSorgtr, hipsolverDnSorgtr)
WWR_FUNCTION_RAW(wwrsolverDnDorgtr, cusolverDnDorgtr, hipsolverDnDorgtr)
WWR_FUNCTION_RAW(wwrsolverDnCungtr_bufferSize, cusolverDnCungtr_bufferSize,
                hipsolverDnCungtr_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnZungtr_bufferSize, cusolverDnZungtr_bufferSize,
                hipsolverDnZungtr_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnCungtr, cusolverDnCungtr, hipsolverDnCungtr)
WWR_FUNCTION_RAW(wwrsolverDnZungtr, cusolverDnZungtr, hipsolverDnZungtr)
WWR_FUNCTION_RAW(wwrsolverDnSormtr_bufferSize, cusolverDnSormtr_bufferSize,
                hipsolverDnSormtr_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnDormtr_bufferSize, cusolverDnDormtr_bufferSize,
                hipsolverDnDormtr_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnSormtr, cusolverDnSormtr, hipsolverDnSormtr)
WWR_FUNCTION_RAW(wwrsolverDnDormtr, cusolverDnDormtr, hipsolverDnDormtr)
WWR_FUNCTION_RAW(wwrsolverDnCunmtr_bufferSize, cusolverDnCunmtr_bufferSize,
                hipsolverDnCunmtr_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnZunmtr_bufferSize, cusolverDnZunmtr_bufferSize,
                hipsolverDnZunmtr_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnCunmtr, cusolverDnCunmtr, hipsolverDnCunmtr)
WWR_FUNCTION_RAW(wwrsolverDnZunmtr, cusolverDnZunmtr, hipsolverDnZunmtr)
WWR_FUNCTION_RAW(wwrsolverDnSsyevj_bufferSize, cusolverDnSsyevj_bufferSize,
                hipsolverDnSsyevj_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnDsyevj_bufferSize, cusolverDnDsyevj_bufferSize,
                hipsolverDnDsyevj_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnSsyevj, cusolverDnSsyevj, hipsolverDnSsyevj)
WWR_FUNCTION_RAW(wwrsolverDnDsyevj, cusolverDnDsyevj, hipsolverDnDsyevj)
WWR_FUNCTION_RAW(wwrsolverDnSsyevjBatched_bufferSize, cusolverDnSsyevjBatched_bufferSize,
                hipsolverDnSsyevjBatched_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnDsyevjBatched_bufferSize, cusolverDnDsyevjBatched_bufferSize,
                hipsolverDnDsyevjBatched_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnSsyevjBatched, cusolverDnSsyevjBatched, hipsolverDnSsyevjBatched)
WWR_FUNCTION_RAW(wwrsolverDnDsyevjBatched, cusolverDnDsyevjBatched, hipsolverDnDsyevjBatched)
WWR_FUNCTION_RAW(wwrsolverDnCheevj_bufferSize, cusolverDnCheevj_bufferSize,
                hipsolverDnCheevj_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnZheevj_bufferSize, cusolverDnZheevj_bufferSize,
                hipsolverDnZheevj_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnCheevj, cusolverDnCheevj, hipsolverDnCheevj)
WWR_FUNCTION_RAW(wwrsolverDnZheevj, cusolverDnZheevj, hipsolverDnZheevj)
WWR_FUNCTION_RAW(wwrsolverDnCheevjBatched_bufferSize, cusolverDnCheevjBatched_bufferSize,
                hipsolverDnCheevjBatched_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnZheevjBatched_bufferSize, cusolverDnZheevjBatched_bufferSize,
                hipsolverDnZheevjBatched_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnCheevjBatched, cusolverDnCheevjBatched, hipsolverDnCheevjBatched)
WWR_FUNCTION_RAW(wwrsolverDnZheevjBatched, cusolverDnZheevjBatched, hipsolverDnZheevjBatched)
WWR_FUNCTION_RAW(wwrsolverDnSgesvdj_bufferSize, cusolverDnSgesvdj_bufferSize,
                hipsolverDnSgesvdj_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnDgesvdj_bufferSize, cusolverDnDgesvdj_bufferSize,
                hipsolverDnDgesvdj_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnCgesvdj_bufferSize, cusolverDnCgesvdj_bufferSize,
                hipsolverDnCgesvdj_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnZgesvdj_bufferSize, cusolverDnZgesvdj_bufferSize,
                hipsolverDnZgesvdj_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnSgesvdj, cusolverDnSgesvdj, hipsolverDnSgesvdj)
WWR_FUNCTION_RAW(wwrsolverDnDgesvdj, cusolverDnDgesvdj, hipsolverDnDgesvdj)
WWR_FUNCTION_RAW(wwrsolverDnCgesvdj, cusolverDnCgesvdj, hipsolverDnCgesvdj)
WWR_FUNCTION_RAW(wwrsolverDnZgesvdj, cusolverDnZgesvdj, hipsolverDnZgesvdj)
WWR_FUNCTION_RAW(wwrsolverDnSgesvdjBatched_bufferSize, cusolverDnSgesvdjBatched_bufferSize,
                hipsolverDnSgesvdjBatched_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnDgesvdjBatched_bufferSize, cusolverDnDgesvdjBatched_bufferSize,
                hipsolverDnDgesvdjBatched_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnCgesvdjBatched_bufferSize, cusolverDnCgesvdjBatched_bufferSize,
                hipsolverDnCgesvdjBatched_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnZgesvdjBatched_bufferSize, cusolverDnZgesvdjBatched_bufferSize,
                hipsolverDnZgesvdjBatched_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnSgesvdjBatched, cusolverDnSgesvdjBatched, hipsolverDnSgesvdjBatched)
WWR_FUNCTION_RAW(wwrsolverDnDgesvdjBatched, cusolverDnDgesvdjBatched, hipsolverDnDgesvdjBatched)
WWR_FUNCTION_RAW(wwrsolverDnCgesvdjBatched, cusolverDnCgesvdjBatched, hipsolverDnCgesvdjBatched)
WWR_FUNCTION_RAW(wwrsolverDnZgesvdjBatched, cusolverDnZgesvdjBatched, hipsolverDnZgesvdjBatched)
WWR_FUNCTION_RAW(wwrsolverDnSsygvd_bufferSize, cusolverDnSsygvd_bufferSize,
                hipsolverDnSsygvd_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnDsygvd_bufferSize, cusolverDnDsygvd_bufferSize,
                hipsolverDnDsygvd_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnSsygvd, cusolverDnSsygvd, hipsolverDnSsygvd)
WWR_FUNCTION_RAW(wwrsolverDnDsygvd, cusolverDnDsygvd, hipsolverDnDsygvd)
WWR_FUNCTION_RAW(wwrsolverDnSsygvdx_bufferSize, cusolverDnSsygvdx_bufferSize,
                hipsolverDnSsygvdx_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnDsygvdx_bufferSize, cusolverDnDsygvdx_bufferSize,
                hipsolverDnDsygvdx_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnSsygvdx, cusolverDnSsygvdx, hipsolverDnSsygvdx)
WWR_FUNCTION_RAW(wwrsolverDnDsygvdx, cusolverDnDsygvdx, hipsolverDnDsygvdx)
WWR_FUNCTION_RAW(wwrsolverDnChegvd_bufferSize, cusolverDnChegvd_bufferSize,
                hipsolverDnChegvd_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnZhegvd_bufferSize, cusolverDnZhegvd_bufferSize,
                hipsolverDnZhegvd_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnChegvd, cusolverDnChegvd, hipsolverDnChegvd)
WWR_FUNCTION_RAW(wwrsolverDnZhegvd, cusolverDnZhegvd, hipsolverDnZhegvd)
WWR_FUNCTION_RAW(wwrsolverDnChegvdx_bufferSize, cusolverDnChegvdx_bufferSize,
                hipsolverDnChegvdx_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnZhegvdx_bufferSize, cusolverDnZhegvdx_bufferSize,
                hipsolverDnZhegvdx_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnChegvdx, cusolverDnChegvdx, hipsolverDnChegvdx)
WWR_FUNCTION_RAW(wwrsolverDnZhegvdx, cusolverDnZhegvdx, hipsolverDnZhegvdx)
WWR_FUNCTION_RAW(wwrsolverDnSsygvj_bufferSize, cusolverDnSsygvj_bufferSize,
                hipsolverDnSsygvj_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnDsygvj_bufferSize, cusolverDnDsygvj_bufferSize,
                hipsolverDnDsygvj_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnSsygvj, cusolverDnSsygvj, hipsolverDnSsygvj)
WWR_FUNCTION_RAW(wwrsolverDnDsygvj, cusolverDnDsygvj, hipsolverDnDsygvj)
WWR_FUNCTION_RAW(wwrsolverDnChegvj_bufferSize, cusolverDnChegvj_bufferSize,
                hipsolverDnChegvj_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnZhegvj_bufferSize, cusolverDnZhegvj_bufferSize,
                hipsolverDnZhegvj_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnChegvj, cusolverDnChegvj, hipsolverDnChegvj)
WWR_FUNCTION_RAW(wwrsolverDnZhegvj, cusolverDnZhegvj, hipsolverDnZhegvj)
WWR_FUNCTION_RAW(wwrsolverDnSgesvdaStridedBatched_bufferSize,
                cusolverDnSgesvdaStridedBatched_bufferSize,
                hipsolverDnSgesvdaStridedBatched_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnDgesvdaStridedBatched_bufferSize,
                cusolverDnDgesvdaStridedBatched_bufferSize,
                hipsolverDnDgesvdaStridedBatched_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnCgesvdaStridedBatched_bufferSize,
                cusolverDnCgesvdaStridedBatched_bufferSize,
                hipsolverDnCgesvdaStridedBatched_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnZgesvdaStridedBatched_bufferSize,
                cusolverDnZgesvdaStridedBatched_bufferSize,
                hipsolverDnZgesvdaStridedBatched_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnSgesvdaStridedBatched, cusolverDnSgesvdaStridedBatched,
                hipsolverDnSgesvdaStridedBatched)
WWR_FUNCTION_RAW(wwrsolverDnDgesvdaStridedBatched, cusolverDnDgesvdaStridedBatched,
                hipsolverDnDgesvdaStridedBatched)
WWR_FUNCTION_RAW(wwrsolverDnCgesvdaStridedBatched, cusolverDnCgesvdaStridedBatched,
                hipsolverDnCgesvdaStridedBatched)
WWR_FUNCTION_RAW(wwrsolverDnZgesvdaStridedBatched, cusolverDnZgesvdaStridedBatched,
                hipsolverDnZgesvdaStridedBatched)

// ────────────────────────────────────────────────────────────────────────
// Modern (X-prefixed, wwrsolverDnParams_t-based, int64_t-dimensioned) API --
// 8 functions shared between cuSOLVER and hipSOLVER (potrf, potrs, getrf,
// getrs, geqrf + _bufferSize where cuSOLVER has one). hipsolverDn has no
// X-prefixed counterpart for anything else cuSOLVER's modern API offers
// (larft, sytrs, trtri, and the entire modern eigenvalue/SVD surface); those
// are unwrapped -- reach them through wwr.cuda.cusolverDn directly.
// ────────────────────────────────────────────────────────────────────────

WWR_FUNCTION_RAW(wwrsolverDnXpotrf_bufferSize, cusolverDnXpotrf_bufferSize,
                hipsolverDnXpotrf_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnXpotrf, cusolverDnXpotrf, hipsolverDnXpotrf)
WWR_FUNCTION_RAW(wwrsolverDnXpotrs, cusolverDnXpotrs, hipsolverDnXpotrs)
WWR_FUNCTION_RAW(wwrsolverDnXgetrf_bufferSize, cusolverDnXgetrf_bufferSize,
                hipsolverDnXgetrf_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnXgetrf, cusolverDnXgetrf, hipsolverDnXgetrf)
WWR_FUNCTION_RAW(wwrsolverDnXgetrs, cusolverDnXgetrs, hipsolverDnXgetrs)
WWR_FUNCTION_RAW(wwrsolverDnXgeqrf_bufferSize, cusolverDnXgeqrf_bufferSize,
                hipsolverDnXgeqrf_bufferSize)
WWR_FUNCTION_RAW(wwrsolverDnXgeqrf, cusolverDnXgeqrf, hipsolverDnXgeqrf)

// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)
