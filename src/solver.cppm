/**
 * @file solver.cppm
 * @brief Backend-neutral dense solver: wwrsolver* names for cuSOLVER / hipSOLVER
 *
 * wwrsolverDn<X><basename> stands for cusolverDn<X><basename> /
 * hipsolverDn<X><basename> for the legacy (int-based) API, and
 * wwrsolverDnX<basename> for the modern (cusolverDnParams_t-based,
 * int64_t-dimensioned) one. Every function is written out in full, one line
 * each. See backend.h.
 *
 * Both backends' legacy linear-solver and eigen/SVD partitions match 1:1
 * (32 and 48 functions), so every legacy function is listed.
 * hipSOLVER's modern surface is far narrower: only Xpotrf, Xpotrs, Xgetrf,
 * Xgetrs, Xgeqrf and their _bufferSize functions exist, and those 8 are what is
 * listed. The rest -- Xlarft, Xsytrs, Xtrtri and the whole modern
 * eigenvalue/SVD API -- has no hipSOLVER counterpart; reach it through
 * wwr.cuda.cusolverDn on a CUDA build. wwrsolverGetStatusName/String are
 * hand-written switches per backend -- docs/architecture.md, section 5.
 *
 * Usage:
 *   import wwr.solver;
 *
 *   wwrsolverDnHandle_t handle;
 *   wwrsolverDnCreate(&handle);
 */

module;

#include "backend.h"

#if defined(WWR_GPU_BACKEND_CUDA)
#include <library_types.h>
#else
#include <hip/library_types.h>
#endif

export module wwr.solver;

import wwr.complex;
import wwr.blas;
#if defined(WWR_GPU_BACKEND_CUDA)
import wwr.cuda.cusolverDn;
#else
import wwr.hip.hipsolver;
#endif

export namespace wwr {

// ========================================================================
// Types
// ========================================================================

WWR_TYPE(wwrsolverDnHandle_t, cusolverDnHandle_t, hipsolverDnHandle_t)
WWR_TYPE(wwrsolverDnParams_t, cusolverDnParams_t, hipsolverDnParams_t)
WWR_TYPE(wwrsolverStatus_t, cusolverStatus_t, hipsolverStatus_t)
WWR_TYPE(wwrsolverEigMode_t, cusolverEigMode_t, hipsolverEigMode_t)
WWR_TYPE(wwrsolverEigType_t, cusolverEigType_t, hipsolverEigType_t)
WWR_TYPE(wwrsolverEigRange_t, cusolverEigRange_t, hipsolverEigRange_t)
WWR_TYPE(wwrsolverSyevjInfo_t, syevjInfo_t, hipsolverSyevjInfo_t)
WWR_TYPE(wwrsolverGesvdjInfo_t, gesvdjInfo_t, hipsolverGesvdjInfo_t)
WWR_TYPE(wwrsolverDataType_t, cudaDataType, hipDataType)

// ========================================================================
// Constants
// ========================================================================

WWR_VALUE(WWRSOLVER_STATUS_SUCCESS, CUSOLVER_STATUS_SUCCESS, HIPSOLVER_STATUS_SUCCESS)

WWR_VALUE(WWRSOLVER_EIG_MODE_NOVECTOR, CUSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_EIG_MODE_NOVECTOR)
WWR_VALUE(WWRSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_MODE_VECTOR, HIPSOLVER_EIG_MODE_VECTOR)

WWR_VALUE(WWRSOLVER_EIG_RANGE_ALL, CUSOLVER_EIG_RANGE_ALL, HIPSOLVER_EIG_RANGE_ALL)
WWR_VALUE(WWRSOLVER_EIG_RANGE_V, CUSOLVER_EIG_RANGE_V, HIPSOLVER_EIG_RANGE_V)
WWR_VALUE(WWRSOLVER_EIG_RANGE_I, CUSOLVER_EIG_RANGE_I, HIPSOLVER_EIG_RANGE_I)

WWR_VALUE(WWRSOLVER_EIG_TYPE_1, CUSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_TYPE_1)
WWR_VALUE(WWRSOLVER_EIG_TYPE_2, CUSOLVER_EIG_TYPE_2, HIPSOLVER_EIG_TYPE_2)
WWR_VALUE(WWRSOLVER_EIG_TYPE_3, CUSOLVER_EIG_TYPE_3, HIPSOLVER_EIG_TYPE_3)

// cudaDataType/hipDataType enumerators are plain C enum constants with no
// wwr-namespaced alias (see wwr.blas -- wwrFloatComplex
// etc. are the closest precedent, but those DO have a namespaced alias via
// WWR_COMPLEX_TYPE; these do not), so WWR_VALUE's ::wwr:: qualification
// does not apply here -- write the #if directly instead.
#if defined(WWR_GPU_BACKEND_CUDA)
inline constexpr wwrsolverDataType_t WWRSOLVER_R_32F = CUDA_R_32F;
inline constexpr wwrsolverDataType_t WWRSOLVER_R_64F = CUDA_R_64F;
inline constexpr wwrsolverDataType_t WWRSOLVER_C_32F = CUDA_C_32F;
inline constexpr wwrsolverDataType_t WWRSOLVER_C_64F = CUDA_C_64F;
#else
inline constexpr wwrsolverDataType_t WWRSOLVER_R_32F = HIP_R_32F;
inline constexpr wwrsolverDataType_t WWRSOLVER_R_64F = HIP_R_64F;
inline constexpr wwrsolverDataType_t WWRSOLVER_C_32F = HIP_C_32F;
inline constexpr wwrsolverDataType_t WWRSOLVER_C_64F = HIP_C_64F;
#endif

// ========================================================================
// Status strings: neither backend has an official status-to-string
// function, and the two enumerator sets differ (hipSOLVER has 14 values,
// including HIPSOLVER_STATUS_HANDLE_IS_NULLPTR/INVALID_ENUM; cuSOLVER has 26,
// including the IRS-refinement-specific codes hipSOLVER has no equivalent
// of), so each backend gets its own hand-written switch.
// ========================================================================

inline const char *wwrsolverGetStatusName(wwrsolverStatus_t error) noexcept {
#if defined(WWR_GPU_BACKEND_CUDA)
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
#if defined(WWR_GPU_BACKEND_CUDA)
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

WWR_FUNCTION(wwrsolverDnCreate, cusolverDnCreate, hipsolverDnCreate)
WWR_FUNCTION(wwrsolverDnDestroy, cusolverDnDestroy, hipsolverDnDestroy)
WWR_FUNCTION(wwrsolverDnSetStream, cusolverDnSetStream, hipsolverDnSetStream)
WWR_FUNCTION(wwrsolverDnGetStream, cusolverDnGetStream, hipsolverDnGetStream)

WWR_FUNCTION(wwrsolverDnCreateParams, cusolverDnCreateParams, hipsolverDnCreateParams)
WWR_FUNCTION(wwrsolverDnDestroyParams, cusolverDnDestroyParams, hipsolverDnDestroyParams)

WWR_FUNCTION(wwrsolverDnCreateSyevjInfo, cusolverDnCreateSyevjInfo, hipsolverDnCreateSyevjInfo)
WWR_FUNCTION(wwrsolverDnDestroySyevjInfo, cusolverDnDestroySyevjInfo,
                hipsolverDnDestroySyevjInfo)
WWR_FUNCTION(wwrsolverDnCreateGesvdjInfo, cusolverDnCreateGesvdjInfo,
                hipsolverDnCreateGesvdjInfo)
WWR_FUNCTION(wwrsolverDnDestroyGesvdjInfo, cusolverDnDestroyGesvdjInfo,
                hipsolverDnDestroyGesvdjInfo)

// ────────────────────────────────────────────────────────────────────────
// Legacy linear solver API (int-based, pre-params) -- 32 functions, all
// shared between cuSOLVER and hipSOLVER
// ────────────────────────────────────────────────────────────────────────

WWR_FUNCTION(wwrsolverDnSpotrf_bufferSize, cusolverDnSpotrf_bufferSize,
                hipsolverDnSpotrf_bufferSize)
WWR_FUNCTION(wwrsolverDnDpotrf_bufferSize, cusolverDnDpotrf_bufferSize,
                hipsolverDnDpotrf_bufferSize)
WWR_FUNCTION(wwrsolverDnCpotrf_bufferSize, cusolverDnCpotrf_bufferSize,
                hipsolverDnCpotrf_bufferSize)
WWR_FUNCTION(wwrsolverDnZpotrf_bufferSize, cusolverDnZpotrf_bufferSize,
                hipsolverDnZpotrf_bufferSize)
WWR_FUNCTION(wwrsolverDnSpotrf, cusolverDnSpotrf, hipsolverDnSpotrf)
WWR_FUNCTION(wwrsolverDnDpotrf, cusolverDnDpotrf, hipsolverDnDpotrf)
WWR_FUNCTION(wwrsolverDnCpotrf, cusolverDnCpotrf, hipsolverDnCpotrf)
WWR_FUNCTION(wwrsolverDnZpotrf, cusolverDnZpotrf, hipsolverDnZpotrf)
WWR_FUNCTION(wwrsolverDnSpotrs, cusolverDnSpotrs, hipsolverDnSpotrs)
WWR_FUNCTION(wwrsolverDnDpotrs, cusolverDnDpotrs, hipsolverDnDpotrs)
WWR_FUNCTION(wwrsolverDnCpotrs, cusolverDnCpotrs, hipsolverDnCpotrs)
WWR_FUNCTION(wwrsolverDnZpotrs, cusolverDnZpotrs, hipsolverDnZpotrs)
WWR_FUNCTION(wwrsolverDnSpotri_bufferSize, cusolverDnSpotri_bufferSize,
                hipsolverDnSpotri_bufferSize)
WWR_FUNCTION(wwrsolverDnDpotri_bufferSize, cusolverDnDpotri_bufferSize,
                hipsolverDnDpotri_bufferSize)
WWR_FUNCTION(wwrsolverDnCpotri_bufferSize, cusolverDnCpotri_bufferSize,
                hipsolverDnCpotri_bufferSize)
WWR_FUNCTION(wwrsolverDnZpotri_bufferSize, cusolverDnZpotri_bufferSize,
                hipsolverDnZpotri_bufferSize)
WWR_FUNCTION(wwrsolverDnSpotri, cusolverDnSpotri, hipsolverDnSpotri)
WWR_FUNCTION(wwrsolverDnDpotri, cusolverDnDpotri, hipsolverDnDpotri)
WWR_FUNCTION(wwrsolverDnCpotri, cusolverDnCpotri, hipsolverDnCpotri)
WWR_FUNCTION(wwrsolverDnZpotri, cusolverDnZpotri, hipsolverDnZpotri)
WWR_FUNCTION(wwrsolverDnSgetrf_bufferSize, cusolverDnSgetrf_bufferSize,
                hipsolverDnSgetrf_bufferSize)
WWR_FUNCTION(wwrsolverDnDgetrf_bufferSize, cusolverDnDgetrf_bufferSize,
                hipsolverDnDgetrf_bufferSize)
WWR_FUNCTION(wwrsolverDnCgetrf_bufferSize, cusolverDnCgetrf_bufferSize,
                hipsolverDnCgetrf_bufferSize)
WWR_FUNCTION(wwrsolverDnZgetrf_bufferSize, cusolverDnZgetrf_bufferSize,
                hipsolverDnZgetrf_bufferSize)
WWR_FUNCTION(wwrsolverDnSgetrf, cusolverDnSgetrf, hipsolverDnSgetrf)
WWR_FUNCTION(wwrsolverDnDgetrf, cusolverDnDgetrf, hipsolverDnDgetrf)
WWR_FUNCTION(wwrsolverDnCgetrf, cusolverDnCgetrf, hipsolverDnCgetrf)
WWR_FUNCTION(wwrsolverDnZgetrf, cusolverDnZgetrf, hipsolverDnZgetrf)
WWR_FUNCTION(wwrsolverDnSgetrs, cusolverDnSgetrs, hipsolverDnSgetrs)
WWR_FUNCTION(wwrsolverDnDgetrs, cusolverDnDgetrs, hipsolverDnDgetrs)
WWR_FUNCTION(wwrsolverDnCgetrs, cusolverDnCgetrs, hipsolverDnCgetrs)
WWR_FUNCTION(wwrsolverDnZgetrs, cusolverDnZgetrs, hipsolverDnZgetrs)
WWR_FUNCTION(wwrsolverDnSgeqrf_bufferSize, cusolverDnSgeqrf_bufferSize,
                hipsolverDnSgeqrf_bufferSize)
WWR_FUNCTION(wwrsolverDnDgeqrf_bufferSize, cusolverDnDgeqrf_bufferSize,
                hipsolverDnDgeqrf_bufferSize)
WWR_FUNCTION(wwrsolverDnCgeqrf_bufferSize, cusolverDnCgeqrf_bufferSize,
                hipsolverDnCgeqrf_bufferSize)
WWR_FUNCTION(wwrsolverDnZgeqrf_bufferSize, cusolverDnZgeqrf_bufferSize,
                hipsolverDnZgeqrf_bufferSize)
WWR_FUNCTION(wwrsolverDnSgeqrf, cusolverDnSgeqrf, hipsolverDnSgeqrf)
WWR_FUNCTION(wwrsolverDnDgeqrf, cusolverDnDgeqrf, hipsolverDnDgeqrf)
WWR_FUNCTION(wwrsolverDnCgeqrf, cusolverDnCgeqrf, hipsolverDnCgeqrf)
WWR_FUNCTION(wwrsolverDnZgeqrf, cusolverDnZgeqrf, hipsolverDnZgeqrf)
WWR_FUNCTION(wwrsolverDnSormqr_bufferSize, cusolverDnSormqr_bufferSize,
                hipsolverDnSormqr_bufferSize)
WWR_FUNCTION(wwrsolverDnDormqr_bufferSize, cusolverDnDormqr_bufferSize,
                hipsolverDnDormqr_bufferSize)
WWR_FUNCTION(wwrsolverDnSormqr, cusolverDnSormqr, hipsolverDnSormqr)
WWR_FUNCTION(wwrsolverDnDormqr, cusolverDnDormqr, hipsolverDnDormqr)
WWR_FUNCTION(wwrsolverDnCunmqr_bufferSize, cusolverDnCunmqr_bufferSize,
                hipsolverDnCunmqr_bufferSize)
WWR_FUNCTION(wwrsolverDnZunmqr_bufferSize, cusolverDnZunmqr_bufferSize,
                hipsolverDnZunmqr_bufferSize)
WWR_FUNCTION(wwrsolverDnCunmqr, cusolverDnCunmqr, hipsolverDnCunmqr)
WWR_FUNCTION(wwrsolverDnZunmqr, cusolverDnZunmqr, hipsolverDnZunmqr)
WWR_FUNCTION(wwrsolverDnSSgels_bufferSize, cusolverDnSSgels_bufferSize,
                hipsolverDnSSgels_bufferSize)
WWR_FUNCTION(wwrsolverDnDDgels_bufferSize, cusolverDnDDgels_bufferSize,
                hipsolverDnDDgels_bufferSize)
WWR_FUNCTION(wwrsolverDnCCgels_bufferSize, cusolverDnCCgels_bufferSize,
                hipsolverDnCCgels_bufferSize)
WWR_FUNCTION(wwrsolverDnZZgels_bufferSize, cusolverDnZZgels_bufferSize,
                hipsolverDnZZgels_bufferSize)
WWR_FUNCTION(wwrsolverDnSSgels, cusolverDnSSgels, hipsolverDnSSgels)
WWR_FUNCTION(wwrsolverDnDDgels, cusolverDnDDgels, hipsolverDnDDgels)
WWR_FUNCTION(wwrsolverDnCCgels, cusolverDnCCgels, hipsolverDnCCgels)
WWR_FUNCTION(wwrsolverDnZZgels, cusolverDnZZgels, hipsolverDnZZgels)
WWR_FUNCTION(wwrsolverDnSSgesv_bufferSize, cusolverDnSSgesv_bufferSize,
                hipsolverDnSSgesv_bufferSize)
WWR_FUNCTION(wwrsolverDnDDgesv_bufferSize, cusolverDnDDgesv_bufferSize,
                hipsolverDnDDgesv_bufferSize)
WWR_FUNCTION(wwrsolverDnCCgesv_bufferSize, cusolverDnCCgesv_bufferSize,
                hipsolverDnCCgesv_bufferSize)
WWR_FUNCTION(wwrsolverDnZZgesv_bufferSize, cusolverDnZZgesv_bufferSize,
                hipsolverDnZZgesv_bufferSize)
WWR_FUNCTION(wwrsolverDnSSgesv, cusolverDnSSgesv, hipsolverDnSSgesv)
WWR_FUNCTION(wwrsolverDnDDgesv, cusolverDnDDgesv, hipsolverDnDDgesv)
WWR_FUNCTION(wwrsolverDnCCgesv, cusolverDnCCgesv, hipsolverDnCCgesv)
WWR_FUNCTION(wwrsolverDnZZgesv, cusolverDnZZgesv, hipsolverDnZZgesv)
WWR_FUNCTION(wwrsolverDnSpotrfBatched, cusolverDnSpotrfBatched, hipsolverDnSpotrfBatched)
WWR_FUNCTION(wwrsolverDnDpotrfBatched, cusolverDnDpotrfBatched, hipsolverDnDpotrfBatched)
WWR_FUNCTION(wwrsolverDnCpotrfBatched, cusolverDnCpotrfBatched, hipsolverDnCpotrfBatched)
WWR_FUNCTION(wwrsolverDnZpotrfBatched, cusolverDnZpotrfBatched, hipsolverDnZpotrfBatched)
WWR_FUNCTION(wwrsolverDnSpotrsBatched, cusolverDnSpotrsBatched, hipsolverDnSpotrsBatched)
WWR_FUNCTION(wwrsolverDnDpotrsBatched, cusolverDnDpotrsBatched, hipsolverDnDpotrsBatched)
WWR_FUNCTION(wwrsolverDnCpotrsBatched, cusolverDnCpotrsBatched, hipsolverDnCpotrsBatched)
WWR_FUNCTION(wwrsolverDnZpotrsBatched, cusolverDnZpotrsBatched, hipsolverDnZpotrsBatched)
WWR_FUNCTION(wwrsolverDnSsytrf_bufferSize, cusolverDnSsytrf_bufferSize,
                hipsolverDnSsytrf_bufferSize)
WWR_FUNCTION(wwrsolverDnDsytrf_bufferSize, cusolverDnDsytrf_bufferSize,
                hipsolverDnDsytrf_bufferSize)
WWR_FUNCTION(wwrsolverDnCsytrf_bufferSize, cusolverDnCsytrf_bufferSize,
                hipsolverDnCsytrf_bufferSize)
WWR_FUNCTION(wwrsolverDnZsytrf_bufferSize, cusolverDnZsytrf_bufferSize,
                hipsolverDnZsytrf_bufferSize)
WWR_FUNCTION(wwrsolverDnSsytrf, cusolverDnSsytrf, hipsolverDnSsytrf)
WWR_FUNCTION(wwrsolverDnDsytrf, cusolverDnDsytrf, hipsolverDnDsytrf)
WWR_FUNCTION(wwrsolverDnCsytrf, cusolverDnCsytrf, hipsolverDnCsytrf)
WWR_FUNCTION(wwrsolverDnZsytrf, cusolverDnZsytrf, hipsolverDnZsytrf)
WWR_FUNCTION(wwrsolverDnSgebrd_bufferSize, cusolverDnSgebrd_bufferSize,
                hipsolverDnSgebrd_bufferSize)
WWR_FUNCTION(wwrsolverDnDgebrd_bufferSize, cusolverDnDgebrd_bufferSize,
                hipsolverDnDgebrd_bufferSize)
WWR_FUNCTION(wwrsolverDnCgebrd_bufferSize, cusolverDnCgebrd_bufferSize,
                hipsolverDnCgebrd_bufferSize)
WWR_FUNCTION(wwrsolverDnZgebrd_bufferSize, cusolverDnZgebrd_bufferSize,
                hipsolverDnZgebrd_bufferSize)
WWR_FUNCTION(wwrsolverDnSgebrd, cusolverDnSgebrd, hipsolverDnSgebrd)
WWR_FUNCTION(wwrsolverDnDgebrd, cusolverDnDgebrd, hipsolverDnDgebrd)
WWR_FUNCTION(wwrsolverDnCgebrd, cusolverDnCgebrd, hipsolverDnCgebrd)
WWR_FUNCTION(wwrsolverDnZgebrd, cusolverDnZgebrd, hipsolverDnZgebrd)
WWR_FUNCTION(wwrsolverDnSorgqr_bufferSize, cusolverDnSorgqr_bufferSize,
                hipsolverDnSorgqr_bufferSize)
WWR_FUNCTION(wwrsolverDnDorgqr_bufferSize, cusolverDnDorgqr_bufferSize,
                hipsolverDnDorgqr_bufferSize)
WWR_FUNCTION(wwrsolverDnCungqr_bufferSize, cusolverDnCungqr_bufferSize,
                hipsolverDnCungqr_bufferSize)
WWR_FUNCTION(wwrsolverDnZungqr_bufferSize, cusolverDnZungqr_bufferSize,
                hipsolverDnZungqr_bufferSize)
WWR_FUNCTION(wwrsolverDnSorgqr, cusolverDnSorgqr, hipsolverDnSorgqr)
WWR_FUNCTION(wwrsolverDnDorgqr, cusolverDnDorgqr, hipsolverDnDorgqr)
WWR_FUNCTION(wwrsolverDnCungqr, cusolverDnCungqr, hipsolverDnCungqr)
WWR_FUNCTION(wwrsolverDnZungqr, cusolverDnZungqr, hipsolverDnZungqr)
WWR_FUNCTION(wwrsolverDnSorgbr_bufferSize, cusolverDnSorgbr_bufferSize,
                hipsolverDnSorgbr_bufferSize)
WWR_FUNCTION(wwrsolverDnDorgbr_bufferSize, cusolverDnDorgbr_bufferSize,
                hipsolverDnDorgbr_bufferSize)
WWR_FUNCTION(wwrsolverDnCungbr_bufferSize, cusolverDnCungbr_bufferSize,
                hipsolverDnCungbr_bufferSize)
WWR_FUNCTION(wwrsolverDnZungbr_bufferSize, cusolverDnZungbr_bufferSize,
                hipsolverDnZungbr_bufferSize)
WWR_FUNCTION(wwrsolverDnSorgbr, cusolverDnSorgbr, hipsolverDnSorgbr)
WWR_FUNCTION(wwrsolverDnDorgbr, cusolverDnDorgbr, hipsolverDnDorgbr)
WWR_FUNCTION(wwrsolverDnCungbr, cusolverDnCungbr, hipsolverDnCungbr)
WWR_FUNCTION(wwrsolverDnZungbr, cusolverDnZungbr, hipsolverDnZungbr)

// ────────────────────────────────────────────────────────────────────────
// Legacy eigenvalue/SVD solver API (int-based, pre-params) -- 48 functions,
// all shared between cuSOLVER and hipSOLVER
// ────────────────────────────────────────────────────────────────────────

WWR_FUNCTION(wwrsolverDnSgesvd_bufferSize, cusolverDnSgesvd_bufferSize,
                hipsolverDnSgesvd_bufferSize)
WWR_FUNCTION(wwrsolverDnDgesvd_bufferSize, cusolverDnDgesvd_bufferSize,
                hipsolverDnDgesvd_bufferSize)
WWR_FUNCTION(wwrsolverDnCgesvd_bufferSize, cusolverDnCgesvd_bufferSize,
                hipsolverDnCgesvd_bufferSize)
WWR_FUNCTION(wwrsolverDnZgesvd_bufferSize, cusolverDnZgesvd_bufferSize,
                hipsolverDnZgesvd_bufferSize)
WWR_FUNCTION(wwrsolverDnSgesvd, cusolverDnSgesvd, hipsolverDnSgesvd)
WWR_FUNCTION(wwrsolverDnDgesvd, cusolverDnDgesvd, hipsolverDnDgesvd)
WWR_FUNCTION(wwrsolverDnCgesvd, cusolverDnCgesvd, hipsolverDnCgesvd)
WWR_FUNCTION(wwrsolverDnZgesvd, cusolverDnZgesvd, hipsolverDnZgesvd)
WWR_FUNCTION(wwrsolverDnSsyevd_bufferSize, cusolverDnSsyevd_bufferSize,
                hipsolverDnSsyevd_bufferSize)
WWR_FUNCTION(wwrsolverDnDsyevd_bufferSize, cusolverDnDsyevd_bufferSize,
                hipsolverDnDsyevd_bufferSize)
WWR_FUNCTION(wwrsolverDnSsyevd, cusolverDnSsyevd, hipsolverDnSsyevd)
WWR_FUNCTION(wwrsolverDnDsyevd, cusolverDnDsyevd, hipsolverDnDsyevd)
WWR_FUNCTION(wwrsolverDnSsyevdx_bufferSize, cusolverDnSsyevdx_bufferSize,
                hipsolverDnSsyevdx_bufferSize)
WWR_FUNCTION(wwrsolverDnDsyevdx_bufferSize, cusolverDnDsyevdx_bufferSize,
                hipsolverDnDsyevdx_bufferSize)
WWR_FUNCTION(wwrsolverDnSsyevdx, cusolverDnSsyevdx, hipsolverDnSsyevdx)
WWR_FUNCTION(wwrsolverDnDsyevdx, cusolverDnDsyevdx, hipsolverDnDsyevdx)
WWR_FUNCTION(wwrsolverDnCheevd_bufferSize, cusolverDnCheevd_bufferSize,
                hipsolverDnCheevd_bufferSize)
WWR_FUNCTION(wwrsolverDnZheevd_bufferSize, cusolverDnZheevd_bufferSize,
                hipsolverDnZheevd_bufferSize)
WWR_FUNCTION(wwrsolverDnCheevd, cusolverDnCheevd, hipsolverDnCheevd)
WWR_FUNCTION(wwrsolverDnZheevd, cusolverDnZheevd, hipsolverDnZheevd)
WWR_FUNCTION(wwrsolverDnCheevdx_bufferSize, cusolverDnCheevdx_bufferSize,
                hipsolverDnCheevdx_bufferSize)
WWR_FUNCTION(wwrsolverDnZheevdx_bufferSize, cusolverDnZheevdx_bufferSize,
                hipsolverDnZheevdx_bufferSize)
WWR_FUNCTION(wwrsolverDnCheevdx, cusolverDnCheevdx, hipsolverDnCheevdx)
WWR_FUNCTION(wwrsolverDnZheevdx, cusolverDnZheevdx, hipsolverDnZheevdx)
WWR_FUNCTION(wwrsolverDnSsytrd_bufferSize, cusolverDnSsytrd_bufferSize,
                hipsolverDnSsytrd_bufferSize)
WWR_FUNCTION(wwrsolverDnDsytrd_bufferSize, cusolverDnDsytrd_bufferSize,
                hipsolverDnDsytrd_bufferSize)
WWR_FUNCTION(wwrsolverDnSsytrd, cusolverDnSsytrd, hipsolverDnSsytrd)
WWR_FUNCTION(wwrsolverDnDsytrd, cusolverDnDsytrd, hipsolverDnDsytrd)
WWR_FUNCTION(wwrsolverDnChetrd_bufferSize, cusolverDnChetrd_bufferSize,
                hipsolverDnChetrd_bufferSize)
WWR_FUNCTION(wwrsolverDnZhetrd_bufferSize, cusolverDnZhetrd_bufferSize,
                hipsolverDnZhetrd_bufferSize)
WWR_FUNCTION(wwrsolverDnChetrd, cusolverDnChetrd, hipsolverDnChetrd)
WWR_FUNCTION(wwrsolverDnZhetrd, cusolverDnZhetrd, hipsolverDnZhetrd)
WWR_FUNCTION(wwrsolverDnSorgtr_bufferSize, cusolverDnSorgtr_bufferSize,
                hipsolverDnSorgtr_bufferSize)
WWR_FUNCTION(wwrsolverDnDorgtr_bufferSize, cusolverDnDorgtr_bufferSize,
                hipsolverDnDorgtr_bufferSize)
WWR_FUNCTION(wwrsolverDnSorgtr, cusolverDnSorgtr, hipsolverDnSorgtr)
WWR_FUNCTION(wwrsolverDnDorgtr, cusolverDnDorgtr, hipsolverDnDorgtr)
WWR_FUNCTION(wwrsolverDnCungtr_bufferSize, cusolverDnCungtr_bufferSize,
                hipsolverDnCungtr_bufferSize)
WWR_FUNCTION(wwrsolverDnZungtr_bufferSize, cusolverDnZungtr_bufferSize,
                hipsolverDnZungtr_bufferSize)
WWR_FUNCTION(wwrsolverDnCungtr, cusolverDnCungtr, hipsolverDnCungtr)
WWR_FUNCTION(wwrsolverDnZungtr, cusolverDnZungtr, hipsolverDnZungtr)
WWR_FUNCTION(wwrsolverDnSormtr_bufferSize, cusolverDnSormtr_bufferSize,
                hipsolverDnSormtr_bufferSize)
WWR_FUNCTION(wwrsolverDnDormtr_bufferSize, cusolverDnDormtr_bufferSize,
                hipsolverDnDormtr_bufferSize)
WWR_FUNCTION(wwrsolverDnSormtr, cusolverDnSormtr, hipsolverDnSormtr)
WWR_FUNCTION(wwrsolverDnDormtr, cusolverDnDormtr, hipsolverDnDormtr)
WWR_FUNCTION(wwrsolverDnCunmtr_bufferSize, cusolverDnCunmtr_bufferSize,
                hipsolverDnCunmtr_bufferSize)
WWR_FUNCTION(wwrsolverDnZunmtr_bufferSize, cusolverDnZunmtr_bufferSize,
                hipsolverDnZunmtr_bufferSize)
WWR_FUNCTION(wwrsolverDnCunmtr, cusolverDnCunmtr, hipsolverDnCunmtr)
WWR_FUNCTION(wwrsolverDnZunmtr, cusolverDnZunmtr, hipsolverDnZunmtr)
WWR_FUNCTION(wwrsolverDnSsyevj_bufferSize, cusolverDnSsyevj_bufferSize,
                hipsolverDnSsyevj_bufferSize)
WWR_FUNCTION(wwrsolverDnDsyevj_bufferSize, cusolverDnDsyevj_bufferSize,
                hipsolverDnDsyevj_bufferSize)
WWR_FUNCTION(wwrsolverDnSsyevj, cusolverDnSsyevj, hipsolverDnSsyevj)
WWR_FUNCTION(wwrsolverDnDsyevj, cusolverDnDsyevj, hipsolverDnDsyevj)
WWR_FUNCTION(wwrsolverDnSsyevjBatched_bufferSize, cusolverDnSsyevjBatched_bufferSize,
                hipsolverDnSsyevjBatched_bufferSize)
WWR_FUNCTION(wwrsolverDnDsyevjBatched_bufferSize, cusolverDnDsyevjBatched_bufferSize,
                hipsolverDnDsyevjBatched_bufferSize)
WWR_FUNCTION(wwrsolverDnSsyevjBatched, cusolverDnSsyevjBatched, hipsolverDnSsyevjBatched)
WWR_FUNCTION(wwrsolverDnDsyevjBatched, cusolverDnDsyevjBatched, hipsolverDnDsyevjBatched)
WWR_FUNCTION(wwrsolverDnCheevj_bufferSize, cusolverDnCheevj_bufferSize,
                hipsolverDnCheevj_bufferSize)
WWR_FUNCTION(wwrsolverDnZheevj_bufferSize, cusolverDnZheevj_bufferSize,
                hipsolverDnZheevj_bufferSize)
WWR_FUNCTION(wwrsolverDnCheevj, cusolverDnCheevj, hipsolverDnCheevj)
WWR_FUNCTION(wwrsolverDnZheevj, cusolverDnZheevj, hipsolverDnZheevj)
WWR_FUNCTION(wwrsolverDnCheevjBatched_bufferSize, cusolverDnCheevjBatched_bufferSize,
                hipsolverDnCheevjBatched_bufferSize)
WWR_FUNCTION(wwrsolverDnZheevjBatched_bufferSize, cusolverDnZheevjBatched_bufferSize,
                hipsolverDnZheevjBatched_bufferSize)
WWR_FUNCTION(wwrsolverDnCheevjBatched, cusolverDnCheevjBatched, hipsolverDnCheevjBatched)
WWR_FUNCTION(wwrsolverDnZheevjBatched, cusolverDnZheevjBatched, hipsolverDnZheevjBatched)
WWR_FUNCTION(wwrsolverDnSgesvdj_bufferSize, cusolverDnSgesvdj_bufferSize,
                hipsolverDnSgesvdj_bufferSize)
WWR_FUNCTION(wwrsolverDnDgesvdj_bufferSize, cusolverDnDgesvdj_bufferSize,
                hipsolverDnDgesvdj_bufferSize)
WWR_FUNCTION(wwrsolverDnCgesvdj_bufferSize, cusolverDnCgesvdj_bufferSize,
                hipsolverDnCgesvdj_bufferSize)
WWR_FUNCTION(wwrsolverDnZgesvdj_bufferSize, cusolverDnZgesvdj_bufferSize,
                hipsolverDnZgesvdj_bufferSize)
WWR_FUNCTION(wwrsolverDnSgesvdj, cusolverDnSgesvdj, hipsolverDnSgesvdj)
WWR_FUNCTION(wwrsolverDnDgesvdj, cusolverDnDgesvdj, hipsolverDnDgesvdj)
WWR_FUNCTION(wwrsolverDnCgesvdj, cusolverDnCgesvdj, hipsolverDnCgesvdj)
WWR_FUNCTION(wwrsolverDnZgesvdj, cusolverDnZgesvdj, hipsolverDnZgesvdj)
WWR_FUNCTION(wwrsolverDnSgesvdjBatched_bufferSize, cusolverDnSgesvdjBatched_bufferSize,
                hipsolverDnSgesvdjBatched_bufferSize)
WWR_FUNCTION(wwrsolverDnDgesvdjBatched_bufferSize, cusolverDnDgesvdjBatched_bufferSize,
                hipsolverDnDgesvdjBatched_bufferSize)
WWR_FUNCTION(wwrsolverDnCgesvdjBatched_bufferSize, cusolverDnCgesvdjBatched_bufferSize,
                hipsolverDnCgesvdjBatched_bufferSize)
WWR_FUNCTION(wwrsolverDnZgesvdjBatched_bufferSize, cusolverDnZgesvdjBatched_bufferSize,
                hipsolverDnZgesvdjBatched_bufferSize)
WWR_FUNCTION(wwrsolverDnSgesvdjBatched, cusolverDnSgesvdjBatched, hipsolverDnSgesvdjBatched)
WWR_FUNCTION(wwrsolverDnDgesvdjBatched, cusolverDnDgesvdjBatched, hipsolverDnDgesvdjBatched)
WWR_FUNCTION(wwrsolverDnCgesvdjBatched, cusolverDnCgesvdjBatched, hipsolverDnCgesvdjBatched)
WWR_FUNCTION(wwrsolverDnZgesvdjBatched, cusolverDnZgesvdjBatched, hipsolverDnZgesvdjBatched)
WWR_FUNCTION(wwrsolverDnSsygvd_bufferSize, cusolverDnSsygvd_bufferSize,
                hipsolverDnSsygvd_bufferSize)
WWR_FUNCTION(wwrsolverDnDsygvd_bufferSize, cusolverDnDsygvd_bufferSize,
                hipsolverDnDsygvd_bufferSize)
WWR_FUNCTION(wwrsolverDnSsygvd, cusolverDnSsygvd, hipsolverDnSsygvd)
WWR_FUNCTION(wwrsolverDnDsygvd, cusolverDnDsygvd, hipsolverDnDsygvd)
WWR_FUNCTION(wwrsolverDnSsygvdx_bufferSize, cusolverDnSsygvdx_bufferSize,
                hipsolverDnSsygvdx_bufferSize)
WWR_FUNCTION(wwrsolverDnDsygvdx_bufferSize, cusolverDnDsygvdx_bufferSize,
                hipsolverDnDsygvdx_bufferSize)
WWR_FUNCTION(wwrsolverDnSsygvdx, cusolverDnSsygvdx, hipsolverDnSsygvdx)
WWR_FUNCTION(wwrsolverDnDsygvdx, cusolverDnDsygvdx, hipsolverDnDsygvdx)
WWR_FUNCTION(wwrsolverDnChegvd_bufferSize, cusolverDnChegvd_bufferSize,
                hipsolverDnChegvd_bufferSize)
WWR_FUNCTION(wwrsolverDnZhegvd_bufferSize, cusolverDnZhegvd_bufferSize,
                hipsolverDnZhegvd_bufferSize)
WWR_FUNCTION(wwrsolverDnChegvd, cusolverDnChegvd, hipsolverDnChegvd)
WWR_FUNCTION(wwrsolverDnZhegvd, cusolverDnZhegvd, hipsolverDnZhegvd)
WWR_FUNCTION(wwrsolverDnChegvdx_bufferSize, cusolverDnChegvdx_bufferSize,
                hipsolverDnChegvdx_bufferSize)
WWR_FUNCTION(wwrsolverDnZhegvdx_bufferSize, cusolverDnZhegvdx_bufferSize,
                hipsolverDnZhegvdx_bufferSize)
WWR_FUNCTION(wwrsolverDnChegvdx, cusolverDnChegvdx, hipsolverDnChegvdx)
WWR_FUNCTION(wwrsolverDnZhegvdx, cusolverDnZhegvdx, hipsolverDnZhegvdx)
WWR_FUNCTION(wwrsolverDnSsygvj_bufferSize, cusolverDnSsygvj_bufferSize,
                hipsolverDnSsygvj_bufferSize)
WWR_FUNCTION(wwrsolverDnDsygvj_bufferSize, cusolverDnDsygvj_bufferSize,
                hipsolverDnDsygvj_bufferSize)
WWR_FUNCTION(wwrsolverDnSsygvj, cusolverDnSsygvj, hipsolverDnSsygvj)
WWR_FUNCTION(wwrsolverDnDsygvj, cusolverDnDsygvj, hipsolverDnDsygvj)
WWR_FUNCTION(wwrsolverDnChegvj_bufferSize, cusolverDnChegvj_bufferSize,
                hipsolverDnChegvj_bufferSize)
WWR_FUNCTION(wwrsolverDnZhegvj_bufferSize, cusolverDnZhegvj_bufferSize,
                hipsolverDnZhegvj_bufferSize)
WWR_FUNCTION(wwrsolverDnChegvj, cusolverDnChegvj, hipsolverDnChegvj)
WWR_FUNCTION(wwrsolverDnZhegvj, cusolverDnZhegvj, hipsolverDnZhegvj)
WWR_FUNCTION(wwrsolverDnSgesvdaStridedBatched_bufferSize,
                cusolverDnSgesvdaStridedBatched_bufferSize,
                hipsolverDnSgesvdaStridedBatched_bufferSize)
WWR_FUNCTION(wwrsolverDnDgesvdaStridedBatched_bufferSize,
                cusolverDnDgesvdaStridedBatched_bufferSize,
                hipsolverDnDgesvdaStridedBatched_bufferSize)
WWR_FUNCTION(wwrsolverDnCgesvdaStridedBatched_bufferSize,
                cusolverDnCgesvdaStridedBatched_bufferSize,
                hipsolverDnCgesvdaStridedBatched_bufferSize)
WWR_FUNCTION(wwrsolverDnZgesvdaStridedBatched_bufferSize,
                cusolverDnZgesvdaStridedBatched_bufferSize,
                hipsolverDnZgesvdaStridedBatched_bufferSize)
WWR_FUNCTION(wwrsolverDnSgesvdaStridedBatched, cusolverDnSgesvdaStridedBatched,
                hipsolverDnSgesvdaStridedBatched)
WWR_FUNCTION(wwrsolverDnDgesvdaStridedBatched, cusolverDnDgesvdaStridedBatched,
                hipsolverDnDgesvdaStridedBatched)
WWR_FUNCTION(wwrsolverDnCgesvdaStridedBatched, cusolverDnCgesvdaStridedBatched,
                hipsolverDnCgesvdaStridedBatched)
WWR_FUNCTION(wwrsolverDnZgesvdaStridedBatched, cusolverDnZgesvdaStridedBatched,
                hipsolverDnZgesvdaStridedBatched)

// ────────────────────────────────────────────────────────────────────────
// Modern (X-prefixed, wwrsolverDnParams_t-based, int64_t-dimensioned) API --
// 8 functions shared between cuSOLVER and hipSOLVER (potrf, potrs, getrf,
// getrs, geqrf + _bufferSize where cuSOLVER has one). hipsolverDn has no
// X-prefixed counterpart for anything else cuSOLVER's modern API offers
// (larft, sytrs, trtri, and the entire modern eigenvalue/SVD surface); those
// are unwrapped -- reach them through wwr.cuda.cusolverDn directly.
// ────────────────────────────────────────────────────────────────────────

WWR_FUNCTION(wwrsolverDnXpotrf_bufferSize, cusolverDnXpotrf_bufferSize,
                hipsolverDnXpotrf_bufferSize)
WWR_FUNCTION(wwrsolverDnXpotrf, cusolverDnXpotrf, hipsolverDnXpotrf)
WWR_FUNCTION(wwrsolverDnXpotrs, cusolverDnXpotrs, hipsolverDnXpotrs)
WWR_FUNCTION(wwrsolverDnXgetrf_bufferSize, cusolverDnXgetrf_bufferSize,
                hipsolverDnXgetrf_bufferSize)
WWR_FUNCTION(wwrsolverDnXgetrf, cusolverDnXgetrf, hipsolverDnXgetrf)
WWR_FUNCTION(wwrsolverDnXgetrs, cusolverDnXgetrs, hipsolverDnXgetrs)
WWR_FUNCTION(wwrsolverDnXgeqrf_bufferSize, cusolverDnXgeqrf_bufferSize,
                hipsolverDnXgeqrf_bufferSize)
WWR_FUNCTION(wwrsolverDnXgeqrf, cusolverDnXgeqrf, hipsolverDnXgeqrf)

} // namespace wwr
