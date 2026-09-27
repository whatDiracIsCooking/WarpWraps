/**
 * @file solver.cppm
 * @brief Backend-neutral dense solver: gpusolver* names for cuSOLVER / hipSOLVER
 *
 * gpusolverDn<X><basename> stands for cusolverDn<X><basename> /
 * hipsolverDn<X><basename> for the legacy (int-based) API, and
 * gpusolverDnX<basename> for the modern (cusolverDnParams_t-based,
 * int64_t-dimensioned) one. Every function is written out in full, one line
 * each. See gpu_backend.h.
 *
 * Both backends' legacy linear-solver and eigen/SVD partitions match 1:1
 * (32 and 48 functions), so every legacy function is listed.
 * hipSOLVER's modern surface is far narrower: only Xpotrf, Xpotrs, Xgetrf,
 * Xgetrs, Xgeqrf and their _bufferSize functions exist, and those 8 are what is
 * listed. The rest -- Xlarft, Xsytrs, Xtrtri and the whole modern
 * eigenvalue/SVD API -- has no hipSOLVER counterpart; reach it through
 * gpumod.cuda.cusolverDn on a CUDA build. gpusolverGetStatusName/String are
 * hand-written switches per backend -- docs/architecture.md, section 5.
 *
 * Usage:
 *   import gpumod.solver;
 *
 *   gpusolverDnHandle_t handle;
 *   gpusolverDnCreate(&handle);
 */

module;

#include "gpu_backend.h"

#if defined(WWR_GPU_BACKEND_CUDA)
#include <library_types.h>
#else
#include <hip/library_types.h>
#endif

export module gpumod.solver;

import gpumod.complex;
import gpumod.blas;
#if defined(WWR_GPU_BACKEND_CUDA)
import gpumod.cuda.cusolverDn;
#else
import gpumod.hip.hipsolver;
#endif

export namespace wwr {

// ========================================================================
// Types
// ========================================================================

WWR_TYPE(gpusolverDnHandle_t, cusolverDnHandle_t, hipsolverDnHandle_t)
WWR_TYPE(gpusolverDnParams_t, cusolverDnParams_t, hipsolverDnParams_t)
WWR_TYPE(gpusolverStatus_t, cusolverStatus_t, hipsolverStatus_t)
WWR_TYPE(gpusolverEigMode_t, cusolverEigMode_t, hipsolverEigMode_t)
WWR_TYPE(gpusolverEigType_t, cusolverEigType_t, hipsolverEigType_t)
WWR_TYPE(gpusolverEigRange_t, cusolverEigRange_t, hipsolverEigRange_t)
WWR_TYPE(gpusolverSyevjInfo_t, syevjInfo_t, hipsolverSyevjInfo_t)
WWR_TYPE(gpusolverGesvdjInfo_t, gesvdjInfo_t, hipsolverGesvdjInfo_t)
WWR_TYPE(gpusolverDataType_t, cudaDataType, hipDataType)

// ========================================================================
// Constants
// ========================================================================

WWR_VALUE(GPUSOLVER_STATUS_SUCCESS, CUSOLVER_STATUS_SUCCESS, HIPSOLVER_STATUS_SUCCESS)

WWR_VALUE(GPUSOLVER_EIG_MODE_NOVECTOR, CUSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_EIG_MODE_NOVECTOR)
WWR_VALUE(GPUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_MODE_VECTOR, HIPSOLVER_EIG_MODE_VECTOR)

WWR_VALUE(GPUSOLVER_EIG_RANGE_ALL, CUSOLVER_EIG_RANGE_ALL, HIPSOLVER_EIG_RANGE_ALL)
WWR_VALUE(GPUSOLVER_EIG_RANGE_V, CUSOLVER_EIG_RANGE_V, HIPSOLVER_EIG_RANGE_V)
WWR_VALUE(GPUSOLVER_EIG_RANGE_I, CUSOLVER_EIG_RANGE_I, HIPSOLVER_EIG_RANGE_I)

WWR_VALUE(GPUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_TYPE_1)
WWR_VALUE(GPUSOLVER_EIG_TYPE_2, CUSOLVER_EIG_TYPE_2, HIPSOLVER_EIG_TYPE_2)
WWR_VALUE(GPUSOLVER_EIG_TYPE_3, CUSOLVER_EIG_TYPE_3, HIPSOLVER_EIG_TYPE_3)

// cudaDataType/hipDataType enumerators are plain C enum constants with no
// wwr-namespaced alias (see gpumod.blas -- gpuFloatComplex
// etc. are the closest precedent, but those DO have a namespaced alias via
// WWR_COMPLEX_TYPE; these do not), so WWR_VALUE's ::wwr:: qualification
// does not apply here -- write the #if directly instead.
#if defined(WWR_GPU_BACKEND_CUDA)
inline constexpr gpusolverDataType_t GPUSOLVER_R_32F = CUDA_R_32F;
inline constexpr gpusolverDataType_t GPUSOLVER_R_64F = CUDA_R_64F;
inline constexpr gpusolverDataType_t GPUSOLVER_C_32F = CUDA_C_32F;
inline constexpr gpusolverDataType_t GPUSOLVER_C_64F = CUDA_C_64F;
#else
inline constexpr gpusolverDataType_t GPUSOLVER_R_32F = HIP_R_32F;
inline constexpr gpusolverDataType_t GPUSOLVER_R_64F = HIP_R_64F;
inline constexpr gpusolverDataType_t GPUSOLVER_C_32F = HIP_C_32F;
inline constexpr gpusolverDataType_t GPUSOLVER_C_64F = HIP_C_64F;
#endif

// ========================================================================
// Status strings: neither backend has an official status-to-string
// function, and the two enumerator sets differ (hipSOLVER has 14 values,
// including HIPSOLVER_STATUS_HANDLE_IS_NULLPTR/INVALID_ENUM; cuSOLVER has 26,
// including the IRS-refinement-specific codes hipSOLVER has no equivalent
// of), so each backend gets its own hand-written switch.
// ========================================================================

inline const char *gpusolverGetStatusName(gpusolverStatus_t error) noexcept {
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

inline const char *gpusolverGetStatusString(gpusolverStatus_t error) noexcept {
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

WWR_FUNCTION(gpusolverDnCreate, cusolverDnCreate, hipsolverDnCreate)
WWR_FUNCTION(gpusolverDnDestroy, cusolverDnDestroy, hipsolverDnDestroy)
WWR_FUNCTION(gpusolverDnSetStream, cusolverDnSetStream, hipsolverDnSetStream)
WWR_FUNCTION(gpusolverDnGetStream, cusolverDnGetStream, hipsolverDnGetStream)

WWR_FUNCTION(gpusolverDnCreateParams, cusolverDnCreateParams, hipsolverDnCreateParams)
WWR_FUNCTION(gpusolverDnDestroyParams, cusolverDnDestroyParams, hipsolverDnDestroyParams)

WWR_FUNCTION(gpusolverDnCreateSyevjInfo, cusolverDnCreateSyevjInfo, hipsolverDnCreateSyevjInfo)
WWR_FUNCTION(gpusolverDnDestroySyevjInfo, cusolverDnDestroySyevjInfo,
                hipsolverDnDestroySyevjInfo)
WWR_FUNCTION(gpusolverDnCreateGesvdjInfo, cusolverDnCreateGesvdjInfo,
                hipsolverDnCreateGesvdjInfo)
WWR_FUNCTION(gpusolverDnDestroyGesvdjInfo, cusolverDnDestroyGesvdjInfo,
                hipsolverDnDestroyGesvdjInfo)

// ────────────────────────────────────────────────────────────────────────
// Legacy linear solver API (int-based, pre-params) -- 32 functions, all
// shared between cuSOLVER and hipSOLVER
// ────────────────────────────────────────────────────────────────────────

WWR_FUNCTION(gpusolverDnSpotrf_bufferSize, cusolverDnSpotrf_bufferSize,
                hipsolverDnSpotrf_bufferSize)
WWR_FUNCTION(gpusolverDnDpotrf_bufferSize, cusolverDnDpotrf_bufferSize,
                hipsolverDnDpotrf_bufferSize)
WWR_FUNCTION(gpusolverDnCpotrf_bufferSize, cusolverDnCpotrf_bufferSize,
                hipsolverDnCpotrf_bufferSize)
WWR_FUNCTION(gpusolverDnZpotrf_bufferSize, cusolverDnZpotrf_bufferSize,
                hipsolverDnZpotrf_bufferSize)
WWR_FUNCTION(gpusolverDnSpotrf, cusolverDnSpotrf, hipsolverDnSpotrf)
WWR_FUNCTION(gpusolverDnDpotrf, cusolverDnDpotrf, hipsolverDnDpotrf)
WWR_FUNCTION(gpusolverDnCpotrf, cusolverDnCpotrf, hipsolverDnCpotrf)
WWR_FUNCTION(gpusolverDnZpotrf, cusolverDnZpotrf, hipsolverDnZpotrf)
WWR_FUNCTION(gpusolverDnSpotrs, cusolverDnSpotrs, hipsolverDnSpotrs)
WWR_FUNCTION(gpusolverDnDpotrs, cusolverDnDpotrs, hipsolverDnDpotrs)
WWR_FUNCTION(gpusolverDnCpotrs, cusolverDnCpotrs, hipsolverDnCpotrs)
WWR_FUNCTION(gpusolverDnZpotrs, cusolverDnZpotrs, hipsolverDnZpotrs)
WWR_FUNCTION(gpusolverDnSpotri_bufferSize, cusolverDnSpotri_bufferSize,
                hipsolverDnSpotri_bufferSize)
WWR_FUNCTION(gpusolverDnDpotri_bufferSize, cusolverDnDpotri_bufferSize,
                hipsolverDnDpotri_bufferSize)
WWR_FUNCTION(gpusolverDnCpotri_bufferSize, cusolverDnCpotri_bufferSize,
                hipsolverDnCpotri_bufferSize)
WWR_FUNCTION(gpusolverDnZpotri_bufferSize, cusolverDnZpotri_bufferSize,
                hipsolverDnZpotri_bufferSize)
WWR_FUNCTION(gpusolverDnSpotri, cusolverDnSpotri, hipsolverDnSpotri)
WWR_FUNCTION(gpusolverDnDpotri, cusolverDnDpotri, hipsolverDnDpotri)
WWR_FUNCTION(gpusolverDnCpotri, cusolverDnCpotri, hipsolverDnCpotri)
WWR_FUNCTION(gpusolverDnZpotri, cusolverDnZpotri, hipsolverDnZpotri)
WWR_FUNCTION(gpusolverDnSgetrf_bufferSize, cusolverDnSgetrf_bufferSize,
                hipsolverDnSgetrf_bufferSize)
WWR_FUNCTION(gpusolverDnDgetrf_bufferSize, cusolverDnDgetrf_bufferSize,
                hipsolverDnDgetrf_bufferSize)
WWR_FUNCTION(gpusolverDnCgetrf_bufferSize, cusolverDnCgetrf_bufferSize,
                hipsolverDnCgetrf_bufferSize)
WWR_FUNCTION(gpusolverDnZgetrf_bufferSize, cusolverDnZgetrf_bufferSize,
                hipsolverDnZgetrf_bufferSize)
WWR_FUNCTION(gpusolverDnSgetrf, cusolverDnSgetrf, hipsolverDnSgetrf)
WWR_FUNCTION(gpusolverDnDgetrf, cusolverDnDgetrf, hipsolverDnDgetrf)
WWR_FUNCTION(gpusolverDnCgetrf, cusolverDnCgetrf, hipsolverDnCgetrf)
WWR_FUNCTION(gpusolverDnZgetrf, cusolverDnZgetrf, hipsolverDnZgetrf)
WWR_FUNCTION(gpusolverDnSgetrs, cusolverDnSgetrs, hipsolverDnSgetrs)
WWR_FUNCTION(gpusolverDnDgetrs, cusolverDnDgetrs, hipsolverDnDgetrs)
WWR_FUNCTION(gpusolverDnCgetrs, cusolverDnCgetrs, hipsolverDnCgetrs)
WWR_FUNCTION(gpusolverDnZgetrs, cusolverDnZgetrs, hipsolverDnZgetrs)
WWR_FUNCTION(gpusolverDnSgeqrf_bufferSize, cusolverDnSgeqrf_bufferSize,
                hipsolverDnSgeqrf_bufferSize)
WWR_FUNCTION(gpusolverDnDgeqrf_bufferSize, cusolverDnDgeqrf_bufferSize,
                hipsolverDnDgeqrf_bufferSize)
WWR_FUNCTION(gpusolverDnCgeqrf_bufferSize, cusolverDnCgeqrf_bufferSize,
                hipsolverDnCgeqrf_bufferSize)
WWR_FUNCTION(gpusolverDnZgeqrf_bufferSize, cusolverDnZgeqrf_bufferSize,
                hipsolverDnZgeqrf_bufferSize)
WWR_FUNCTION(gpusolverDnSgeqrf, cusolverDnSgeqrf, hipsolverDnSgeqrf)
WWR_FUNCTION(gpusolverDnDgeqrf, cusolverDnDgeqrf, hipsolverDnDgeqrf)
WWR_FUNCTION(gpusolverDnCgeqrf, cusolverDnCgeqrf, hipsolverDnCgeqrf)
WWR_FUNCTION(gpusolverDnZgeqrf, cusolverDnZgeqrf, hipsolverDnZgeqrf)
WWR_FUNCTION(gpusolverDnSormqr_bufferSize, cusolverDnSormqr_bufferSize,
                hipsolverDnSormqr_bufferSize)
WWR_FUNCTION(gpusolverDnDormqr_bufferSize, cusolverDnDormqr_bufferSize,
                hipsolverDnDormqr_bufferSize)
WWR_FUNCTION(gpusolverDnSormqr, cusolverDnSormqr, hipsolverDnSormqr)
WWR_FUNCTION(gpusolverDnDormqr, cusolverDnDormqr, hipsolverDnDormqr)
WWR_FUNCTION(gpusolverDnCunmqr_bufferSize, cusolverDnCunmqr_bufferSize,
                hipsolverDnCunmqr_bufferSize)
WWR_FUNCTION(gpusolverDnZunmqr_bufferSize, cusolverDnZunmqr_bufferSize,
                hipsolverDnZunmqr_bufferSize)
WWR_FUNCTION(gpusolverDnCunmqr, cusolverDnCunmqr, hipsolverDnCunmqr)
WWR_FUNCTION(gpusolverDnZunmqr, cusolverDnZunmqr, hipsolverDnZunmqr)
WWR_FUNCTION(gpusolverDnSSgels_bufferSize, cusolverDnSSgels_bufferSize,
                hipsolverDnSSgels_bufferSize)
WWR_FUNCTION(gpusolverDnDDgels_bufferSize, cusolverDnDDgels_bufferSize,
                hipsolverDnDDgels_bufferSize)
WWR_FUNCTION(gpusolverDnCCgels_bufferSize, cusolverDnCCgels_bufferSize,
                hipsolverDnCCgels_bufferSize)
WWR_FUNCTION(gpusolverDnZZgels_bufferSize, cusolverDnZZgels_bufferSize,
                hipsolverDnZZgels_bufferSize)
WWR_FUNCTION(gpusolverDnSSgels, cusolverDnSSgels, hipsolverDnSSgels)
WWR_FUNCTION(gpusolverDnDDgels, cusolverDnDDgels, hipsolverDnDDgels)
WWR_FUNCTION(gpusolverDnCCgels, cusolverDnCCgels, hipsolverDnCCgels)
WWR_FUNCTION(gpusolverDnZZgels, cusolverDnZZgels, hipsolverDnZZgels)
WWR_FUNCTION(gpusolverDnSSgesv_bufferSize, cusolverDnSSgesv_bufferSize,
                hipsolverDnSSgesv_bufferSize)
WWR_FUNCTION(gpusolverDnDDgesv_bufferSize, cusolverDnDDgesv_bufferSize,
                hipsolverDnDDgesv_bufferSize)
WWR_FUNCTION(gpusolverDnCCgesv_bufferSize, cusolverDnCCgesv_bufferSize,
                hipsolverDnCCgesv_bufferSize)
WWR_FUNCTION(gpusolverDnZZgesv_bufferSize, cusolverDnZZgesv_bufferSize,
                hipsolverDnZZgesv_bufferSize)
WWR_FUNCTION(gpusolverDnSSgesv, cusolverDnSSgesv, hipsolverDnSSgesv)
WWR_FUNCTION(gpusolverDnDDgesv, cusolverDnDDgesv, hipsolverDnDDgesv)
WWR_FUNCTION(gpusolverDnCCgesv, cusolverDnCCgesv, hipsolverDnCCgesv)
WWR_FUNCTION(gpusolverDnZZgesv, cusolverDnZZgesv, hipsolverDnZZgesv)
WWR_FUNCTION(gpusolverDnSpotrfBatched, cusolverDnSpotrfBatched, hipsolverDnSpotrfBatched)
WWR_FUNCTION(gpusolverDnDpotrfBatched, cusolverDnDpotrfBatched, hipsolverDnDpotrfBatched)
WWR_FUNCTION(gpusolverDnCpotrfBatched, cusolverDnCpotrfBatched, hipsolverDnCpotrfBatched)
WWR_FUNCTION(gpusolverDnZpotrfBatched, cusolverDnZpotrfBatched, hipsolverDnZpotrfBatched)
WWR_FUNCTION(gpusolverDnSpotrsBatched, cusolverDnSpotrsBatched, hipsolverDnSpotrsBatched)
WWR_FUNCTION(gpusolverDnDpotrsBatched, cusolverDnDpotrsBatched, hipsolverDnDpotrsBatched)
WWR_FUNCTION(gpusolverDnCpotrsBatched, cusolverDnCpotrsBatched, hipsolverDnCpotrsBatched)
WWR_FUNCTION(gpusolverDnZpotrsBatched, cusolverDnZpotrsBatched, hipsolverDnZpotrsBatched)
WWR_FUNCTION(gpusolverDnSsytrf_bufferSize, cusolverDnSsytrf_bufferSize,
                hipsolverDnSsytrf_bufferSize)
WWR_FUNCTION(gpusolverDnDsytrf_bufferSize, cusolverDnDsytrf_bufferSize,
                hipsolverDnDsytrf_bufferSize)
WWR_FUNCTION(gpusolverDnCsytrf_bufferSize, cusolverDnCsytrf_bufferSize,
                hipsolverDnCsytrf_bufferSize)
WWR_FUNCTION(gpusolverDnZsytrf_bufferSize, cusolverDnZsytrf_bufferSize,
                hipsolverDnZsytrf_bufferSize)
WWR_FUNCTION(gpusolverDnSsytrf, cusolverDnSsytrf, hipsolverDnSsytrf)
WWR_FUNCTION(gpusolverDnDsytrf, cusolverDnDsytrf, hipsolverDnDsytrf)
WWR_FUNCTION(gpusolverDnCsytrf, cusolverDnCsytrf, hipsolverDnCsytrf)
WWR_FUNCTION(gpusolverDnZsytrf, cusolverDnZsytrf, hipsolverDnZsytrf)
WWR_FUNCTION(gpusolverDnSgebrd_bufferSize, cusolverDnSgebrd_bufferSize,
                hipsolverDnSgebrd_bufferSize)
WWR_FUNCTION(gpusolverDnDgebrd_bufferSize, cusolverDnDgebrd_bufferSize,
                hipsolverDnDgebrd_bufferSize)
WWR_FUNCTION(gpusolverDnCgebrd_bufferSize, cusolverDnCgebrd_bufferSize,
                hipsolverDnCgebrd_bufferSize)
WWR_FUNCTION(gpusolverDnZgebrd_bufferSize, cusolverDnZgebrd_bufferSize,
                hipsolverDnZgebrd_bufferSize)
WWR_FUNCTION(gpusolverDnSgebrd, cusolverDnSgebrd, hipsolverDnSgebrd)
WWR_FUNCTION(gpusolverDnDgebrd, cusolverDnDgebrd, hipsolverDnDgebrd)
WWR_FUNCTION(gpusolverDnCgebrd, cusolverDnCgebrd, hipsolverDnCgebrd)
WWR_FUNCTION(gpusolverDnZgebrd, cusolverDnZgebrd, hipsolverDnZgebrd)
WWR_FUNCTION(gpusolverDnSorgqr_bufferSize, cusolverDnSorgqr_bufferSize,
                hipsolverDnSorgqr_bufferSize)
WWR_FUNCTION(gpusolverDnDorgqr_bufferSize, cusolverDnDorgqr_bufferSize,
                hipsolverDnDorgqr_bufferSize)
WWR_FUNCTION(gpusolverDnCungqr_bufferSize, cusolverDnCungqr_bufferSize,
                hipsolverDnCungqr_bufferSize)
WWR_FUNCTION(gpusolverDnZungqr_bufferSize, cusolverDnZungqr_bufferSize,
                hipsolverDnZungqr_bufferSize)
WWR_FUNCTION(gpusolverDnSorgqr, cusolverDnSorgqr, hipsolverDnSorgqr)
WWR_FUNCTION(gpusolverDnDorgqr, cusolverDnDorgqr, hipsolverDnDorgqr)
WWR_FUNCTION(gpusolverDnCungqr, cusolverDnCungqr, hipsolverDnCungqr)
WWR_FUNCTION(gpusolverDnZungqr, cusolverDnZungqr, hipsolverDnZungqr)
WWR_FUNCTION(gpusolverDnSorgbr_bufferSize, cusolverDnSorgbr_bufferSize,
                hipsolverDnSorgbr_bufferSize)
WWR_FUNCTION(gpusolverDnDorgbr_bufferSize, cusolverDnDorgbr_bufferSize,
                hipsolverDnDorgbr_bufferSize)
WWR_FUNCTION(gpusolverDnCungbr_bufferSize, cusolverDnCungbr_bufferSize,
                hipsolverDnCungbr_bufferSize)
WWR_FUNCTION(gpusolverDnZungbr_bufferSize, cusolverDnZungbr_bufferSize,
                hipsolverDnZungbr_bufferSize)
WWR_FUNCTION(gpusolverDnSorgbr, cusolverDnSorgbr, hipsolverDnSorgbr)
WWR_FUNCTION(gpusolverDnDorgbr, cusolverDnDorgbr, hipsolverDnDorgbr)
WWR_FUNCTION(gpusolverDnCungbr, cusolverDnCungbr, hipsolverDnCungbr)
WWR_FUNCTION(gpusolverDnZungbr, cusolverDnZungbr, hipsolverDnZungbr)

// ────────────────────────────────────────────────────────────────────────
// Legacy eigenvalue/SVD solver API (int-based, pre-params) -- 48 functions,
// all shared between cuSOLVER and hipSOLVER
// ────────────────────────────────────────────────────────────────────────

WWR_FUNCTION(gpusolverDnSgesvd_bufferSize, cusolverDnSgesvd_bufferSize,
                hipsolverDnSgesvd_bufferSize)
WWR_FUNCTION(gpusolverDnDgesvd_bufferSize, cusolverDnDgesvd_bufferSize,
                hipsolverDnDgesvd_bufferSize)
WWR_FUNCTION(gpusolverDnCgesvd_bufferSize, cusolverDnCgesvd_bufferSize,
                hipsolverDnCgesvd_bufferSize)
WWR_FUNCTION(gpusolverDnZgesvd_bufferSize, cusolverDnZgesvd_bufferSize,
                hipsolverDnZgesvd_bufferSize)
WWR_FUNCTION(gpusolverDnSgesvd, cusolverDnSgesvd, hipsolverDnSgesvd)
WWR_FUNCTION(gpusolverDnDgesvd, cusolverDnDgesvd, hipsolverDnDgesvd)
WWR_FUNCTION(gpusolverDnCgesvd, cusolverDnCgesvd, hipsolverDnCgesvd)
WWR_FUNCTION(gpusolverDnZgesvd, cusolverDnZgesvd, hipsolverDnZgesvd)
WWR_FUNCTION(gpusolverDnSsyevd_bufferSize, cusolverDnSsyevd_bufferSize,
                hipsolverDnSsyevd_bufferSize)
WWR_FUNCTION(gpusolverDnDsyevd_bufferSize, cusolverDnDsyevd_bufferSize,
                hipsolverDnDsyevd_bufferSize)
WWR_FUNCTION(gpusolverDnSsyevd, cusolverDnSsyevd, hipsolverDnSsyevd)
WWR_FUNCTION(gpusolverDnDsyevd, cusolverDnDsyevd, hipsolverDnDsyevd)
WWR_FUNCTION(gpusolverDnSsyevdx_bufferSize, cusolverDnSsyevdx_bufferSize,
                hipsolverDnSsyevdx_bufferSize)
WWR_FUNCTION(gpusolverDnDsyevdx_bufferSize, cusolverDnDsyevdx_bufferSize,
                hipsolverDnDsyevdx_bufferSize)
WWR_FUNCTION(gpusolverDnSsyevdx, cusolverDnSsyevdx, hipsolverDnSsyevdx)
WWR_FUNCTION(gpusolverDnDsyevdx, cusolverDnDsyevdx, hipsolverDnDsyevdx)
WWR_FUNCTION(gpusolverDnCheevd_bufferSize, cusolverDnCheevd_bufferSize,
                hipsolverDnCheevd_bufferSize)
WWR_FUNCTION(gpusolverDnZheevd_bufferSize, cusolverDnZheevd_bufferSize,
                hipsolverDnZheevd_bufferSize)
WWR_FUNCTION(gpusolverDnCheevd, cusolverDnCheevd, hipsolverDnCheevd)
WWR_FUNCTION(gpusolverDnZheevd, cusolverDnZheevd, hipsolverDnZheevd)
WWR_FUNCTION(gpusolverDnCheevdx_bufferSize, cusolverDnCheevdx_bufferSize,
                hipsolverDnCheevdx_bufferSize)
WWR_FUNCTION(gpusolverDnZheevdx_bufferSize, cusolverDnZheevdx_bufferSize,
                hipsolverDnZheevdx_bufferSize)
WWR_FUNCTION(gpusolverDnCheevdx, cusolverDnCheevdx, hipsolverDnCheevdx)
WWR_FUNCTION(gpusolverDnZheevdx, cusolverDnZheevdx, hipsolverDnZheevdx)
WWR_FUNCTION(gpusolverDnSsytrd_bufferSize, cusolverDnSsytrd_bufferSize,
                hipsolverDnSsytrd_bufferSize)
WWR_FUNCTION(gpusolverDnDsytrd_bufferSize, cusolverDnDsytrd_bufferSize,
                hipsolverDnDsytrd_bufferSize)
WWR_FUNCTION(gpusolverDnSsytrd, cusolverDnSsytrd, hipsolverDnSsytrd)
WWR_FUNCTION(gpusolverDnDsytrd, cusolverDnDsytrd, hipsolverDnDsytrd)
WWR_FUNCTION(gpusolverDnChetrd_bufferSize, cusolverDnChetrd_bufferSize,
                hipsolverDnChetrd_bufferSize)
WWR_FUNCTION(gpusolverDnZhetrd_bufferSize, cusolverDnZhetrd_bufferSize,
                hipsolverDnZhetrd_bufferSize)
WWR_FUNCTION(gpusolverDnChetrd, cusolverDnChetrd, hipsolverDnChetrd)
WWR_FUNCTION(gpusolverDnZhetrd, cusolverDnZhetrd, hipsolverDnZhetrd)
WWR_FUNCTION(gpusolverDnSorgtr_bufferSize, cusolverDnSorgtr_bufferSize,
                hipsolverDnSorgtr_bufferSize)
WWR_FUNCTION(gpusolverDnDorgtr_bufferSize, cusolverDnDorgtr_bufferSize,
                hipsolverDnDorgtr_bufferSize)
WWR_FUNCTION(gpusolverDnSorgtr, cusolverDnSorgtr, hipsolverDnSorgtr)
WWR_FUNCTION(gpusolverDnDorgtr, cusolverDnDorgtr, hipsolverDnDorgtr)
WWR_FUNCTION(gpusolverDnCungtr_bufferSize, cusolverDnCungtr_bufferSize,
                hipsolverDnCungtr_bufferSize)
WWR_FUNCTION(gpusolverDnZungtr_bufferSize, cusolverDnZungtr_bufferSize,
                hipsolverDnZungtr_bufferSize)
WWR_FUNCTION(gpusolverDnCungtr, cusolverDnCungtr, hipsolverDnCungtr)
WWR_FUNCTION(gpusolverDnZungtr, cusolverDnZungtr, hipsolverDnZungtr)
WWR_FUNCTION(gpusolverDnSormtr_bufferSize, cusolverDnSormtr_bufferSize,
                hipsolverDnSormtr_bufferSize)
WWR_FUNCTION(gpusolverDnDormtr_bufferSize, cusolverDnDormtr_bufferSize,
                hipsolverDnDormtr_bufferSize)
WWR_FUNCTION(gpusolverDnSormtr, cusolverDnSormtr, hipsolverDnSormtr)
WWR_FUNCTION(gpusolverDnDormtr, cusolverDnDormtr, hipsolverDnDormtr)
WWR_FUNCTION(gpusolverDnCunmtr_bufferSize, cusolverDnCunmtr_bufferSize,
                hipsolverDnCunmtr_bufferSize)
WWR_FUNCTION(gpusolverDnZunmtr_bufferSize, cusolverDnZunmtr_bufferSize,
                hipsolverDnZunmtr_bufferSize)
WWR_FUNCTION(gpusolverDnCunmtr, cusolverDnCunmtr, hipsolverDnCunmtr)
WWR_FUNCTION(gpusolverDnZunmtr, cusolverDnZunmtr, hipsolverDnZunmtr)
WWR_FUNCTION(gpusolverDnSsyevj_bufferSize, cusolverDnSsyevj_bufferSize,
                hipsolverDnSsyevj_bufferSize)
WWR_FUNCTION(gpusolverDnDsyevj_bufferSize, cusolverDnDsyevj_bufferSize,
                hipsolverDnDsyevj_bufferSize)
WWR_FUNCTION(gpusolverDnSsyevj, cusolverDnSsyevj, hipsolverDnSsyevj)
WWR_FUNCTION(gpusolverDnDsyevj, cusolverDnDsyevj, hipsolverDnDsyevj)
WWR_FUNCTION(gpusolverDnSsyevjBatched_bufferSize, cusolverDnSsyevjBatched_bufferSize,
                hipsolverDnSsyevjBatched_bufferSize)
WWR_FUNCTION(gpusolverDnDsyevjBatched_bufferSize, cusolverDnDsyevjBatched_bufferSize,
                hipsolverDnDsyevjBatched_bufferSize)
WWR_FUNCTION(gpusolverDnSsyevjBatched, cusolverDnSsyevjBatched, hipsolverDnSsyevjBatched)
WWR_FUNCTION(gpusolverDnDsyevjBatched, cusolverDnDsyevjBatched, hipsolverDnDsyevjBatched)
WWR_FUNCTION(gpusolverDnCheevj_bufferSize, cusolverDnCheevj_bufferSize,
                hipsolverDnCheevj_bufferSize)
WWR_FUNCTION(gpusolverDnZheevj_bufferSize, cusolverDnZheevj_bufferSize,
                hipsolverDnZheevj_bufferSize)
WWR_FUNCTION(gpusolverDnCheevj, cusolverDnCheevj, hipsolverDnCheevj)
WWR_FUNCTION(gpusolverDnZheevj, cusolverDnZheevj, hipsolverDnZheevj)
WWR_FUNCTION(gpusolverDnCheevjBatched_bufferSize, cusolverDnCheevjBatched_bufferSize,
                hipsolverDnCheevjBatched_bufferSize)
WWR_FUNCTION(gpusolverDnZheevjBatched_bufferSize, cusolverDnZheevjBatched_bufferSize,
                hipsolverDnZheevjBatched_bufferSize)
WWR_FUNCTION(gpusolverDnCheevjBatched, cusolverDnCheevjBatched, hipsolverDnCheevjBatched)
WWR_FUNCTION(gpusolverDnZheevjBatched, cusolverDnZheevjBatched, hipsolverDnZheevjBatched)
WWR_FUNCTION(gpusolverDnSgesvdj_bufferSize, cusolverDnSgesvdj_bufferSize,
                hipsolverDnSgesvdj_bufferSize)
WWR_FUNCTION(gpusolverDnDgesvdj_bufferSize, cusolverDnDgesvdj_bufferSize,
                hipsolverDnDgesvdj_bufferSize)
WWR_FUNCTION(gpusolverDnCgesvdj_bufferSize, cusolverDnCgesvdj_bufferSize,
                hipsolverDnCgesvdj_bufferSize)
WWR_FUNCTION(gpusolverDnZgesvdj_bufferSize, cusolverDnZgesvdj_bufferSize,
                hipsolverDnZgesvdj_bufferSize)
WWR_FUNCTION(gpusolverDnSgesvdj, cusolverDnSgesvdj, hipsolverDnSgesvdj)
WWR_FUNCTION(gpusolverDnDgesvdj, cusolverDnDgesvdj, hipsolverDnDgesvdj)
WWR_FUNCTION(gpusolverDnCgesvdj, cusolverDnCgesvdj, hipsolverDnCgesvdj)
WWR_FUNCTION(gpusolverDnZgesvdj, cusolverDnZgesvdj, hipsolverDnZgesvdj)
WWR_FUNCTION(gpusolverDnSgesvdjBatched_bufferSize, cusolverDnSgesvdjBatched_bufferSize,
                hipsolverDnSgesvdjBatched_bufferSize)
WWR_FUNCTION(gpusolverDnDgesvdjBatched_bufferSize, cusolverDnDgesvdjBatched_bufferSize,
                hipsolverDnDgesvdjBatched_bufferSize)
WWR_FUNCTION(gpusolverDnCgesvdjBatched_bufferSize, cusolverDnCgesvdjBatched_bufferSize,
                hipsolverDnCgesvdjBatched_bufferSize)
WWR_FUNCTION(gpusolverDnZgesvdjBatched_bufferSize, cusolverDnZgesvdjBatched_bufferSize,
                hipsolverDnZgesvdjBatched_bufferSize)
WWR_FUNCTION(gpusolverDnSgesvdjBatched, cusolverDnSgesvdjBatched, hipsolverDnSgesvdjBatched)
WWR_FUNCTION(gpusolverDnDgesvdjBatched, cusolverDnDgesvdjBatched, hipsolverDnDgesvdjBatched)
WWR_FUNCTION(gpusolverDnCgesvdjBatched, cusolverDnCgesvdjBatched, hipsolverDnCgesvdjBatched)
WWR_FUNCTION(gpusolverDnZgesvdjBatched, cusolverDnZgesvdjBatched, hipsolverDnZgesvdjBatched)
WWR_FUNCTION(gpusolverDnSsygvd_bufferSize, cusolverDnSsygvd_bufferSize,
                hipsolverDnSsygvd_bufferSize)
WWR_FUNCTION(gpusolverDnDsygvd_bufferSize, cusolverDnDsygvd_bufferSize,
                hipsolverDnDsygvd_bufferSize)
WWR_FUNCTION(gpusolverDnSsygvd, cusolverDnSsygvd, hipsolverDnSsygvd)
WWR_FUNCTION(gpusolverDnDsygvd, cusolverDnDsygvd, hipsolverDnDsygvd)
WWR_FUNCTION(gpusolverDnSsygvdx_bufferSize, cusolverDnSsygvdx_bufferSize,
                hipsolverDnSsygvdx_bufferSize)
WWR_FUNCTION(gpusolverDnDsygvdx_bufferSize, cusolverDnDsygvdx_bufferSize,
                hipsolverDnDsygvdx_bufferSize)
WWR_FUNCTION(gpusolverDnSsygvdx, cusolverDnSsygvdx, hipsolverDnSsygvdx)
WWR_FUNCTION(gpusolverDnDsygvdx, cusolverDnDsygvdx, hipsolverDnDsygvdx)
WWR_FUNCTION(gpusolverDnChegvd_bufferSize, cusolverDnChegvd_bufferSize,
                hipsolverDnChegvd_bufferSize)
WWR_FUNCTION(gpusolverDnZhegvd_bufferSize, cusolverDnZhegvd_bufferSize,
                hipsolverDnZhegvd_bufferSize)
WWR_FUNCTION(gpusolverDnChegvd, cusolverDnChegvd, hipsolverDnChegvd)
WWR_FUNCTION(gpusolverDnZhegvd, cusolverDnZhegvd, hipsolverDnZhegvd)
WWR_FUNCTION(gpusolverDnChegvdx_bufferSize, cusolverDnChegvdx_bufferSize,
                hipsolverDnChegvdx_bufferSize)
WWR_FUNCTION(gpusolverDnZhegvdx_bufferSize, cusolverDnZhegvdx_bufferSize,
                hipsolverDnZhegvdx_bufferSize)
WWR_FUNCTION(gpusolverDnChegvdx, cusolverDnChegvdx, hipsolverDnChegvdx)
WWR_FUNCTION(gpusolverDnZhegvdx, cusolverDnZhegvdx, hipsolverDnZhegvdx)
WWR_FUNCTION(gpusolverDnSsygvj_bufferSize, cusolverDnSsygvj_bufferSize,
                hipsolverDnSsygvj_bufferSize)
WWR_FUNCTION(gpusolverDnDsygvj_bufferSize, cusolverDnDsygvj_bufferSize,
                hipsolverDnDsygvj_bufferSize)
WWR_FUNCTION(gpusolverDnSsygvj, cusolverDnSsygvj, hipsolverDnSsygvj)
WWR_FUNCTION(gpusolverDnDsygvj, cusolverDnDsygvj, hipsolverDnDsygvj)
WWR_FUNCTION(gpusolverDnChegvj_bufferSize, cusolverDnChegvj_bufferSize,
                hipsolverDnChegvj_bufferSize)
WWR_FUNCTION(gpusolverDnZhegvj_bufferSize, cusolverDnZhegvj_bufferSize,
                hipsolverDnZhegvj_bufferSize)
WWR_FUNCTION(gpusolverDnChegvj, cusolverDnChegvj, hipsolverDnChegvj)
WWR_FUNCTION(gpusolverDnZhegvj, cusolverDnZhegvj, hipsolverDnZhegvj)
WWR_FUNCTION(gpusolverDnSgesvdaStridedBatched_bufferSize,
                cusolverDnSgesvdaStridedBatched_bufferSize,
                hipsolverDnSgesvdaStridedBatched_bufferSize)
WWR_FUNCTION(gpusolverDnDgesvdaStridedBatched_bufferSize,
                cusolverDnDgesvdaStridedBatched_bufferSize,
                hipsolverDnDgesvdaStridedBatched_bufferSize)
WWR_FUNCTION(gpusolverDnCgesvdaStridedBatched_bufferSize,
                cusolverDnCgesvdaStridedBatched_bufferSize,
                hipsolverDnCgesvdaStridedBatched_bufferSize)
WWR_FUNCTION(gpusolverDnZgesvdaStridedBatched_bufferSize,
                cusolverDnZgesvdaStridedBatched_bufferSize,
                hipsolverDnZgesvdaStridedBatched_bufferSize)
WWR_FUNCTION(gpusolverDnSgesvdaStridedBatched, cusolverDnSgesvdaStridedBatched,
                hipsolverDnSgesvdaStridedBatched)
WWR_FUNCTION(gpusolverDnDgesvdaStridedBatched, cusolverDnDgesvdaStridedBatched,
                hipsolverDnDgesvdaStridedBatched)
WWR_FUNCTION(gpusolverDnCgesvdaStridedBatched, cusolverDnCgesvdaStridedBatched,
                hipsolverDnCgesvdaStridedBatched)
WWR_FUNCTION(gpusolverDnZgesvdaStridedBatched, cusolverDnZgesvdaStridedBatched,
                hipsolverDnZgesvdaStridedBatched)

// ────────────────────────────────────────────────────────────────────────
// Modern (X-prefixed, gpusolverDnParams_t-based, int64_t-dimensioned) API --
// 8 functions shared between cuSOLVER and hipSOLVER (potrf, potrs, getrf,
// getrs, geqrf + _bufferSize where cuSOLVER has one). hipsolverDn has no
// X-prefixed counterpart for anything else cuSOLVER's modern API offers
// (larft, sytrs, trtri, and the entire modern eigenvalue/SVD surface); those
// are unwrapped -- reach them through gpumod.cuda.cusolverDn directly.
// ────────────────────────────────────────────────────────────────────────

WWR_FUNCTION(gpusolverDnXpotrf_bufferSize, cusolverDnXpotrf_bufferSize,
                hipsolverDnXpotrf_bufferSize)
WWR_FUNCTION(gpusolverDnXpotrf, cusolverDnXpotrf, hipsolverDnXpotrf)
WWR_FUNCTION(gpusolverDnXpotrs, cusolverDnXpotrs, hipsolverDnXpotrs)
WWR_FUNCTION(gpusolverDnXgetrf_bufferSize, cusolverDnXgetrf_bufferSize,
                hipsolverDnXgetrf_bufferSize)
WWR_FUNCTION(gpusolverDnXgetrf, cusolverDnXgetrf, hipsolverDnXgetrf)
WWR_FUNCTION(gpusolverDnXgetrs, cusolverDnXgetrs, hipsolverDnXgetrs)
WWR_FUNCTION(gpusolverDnXgeqrf_bufferSize, cusolverDnXgeqrf_bufferSize,
                hipsolverDnXgeqrf_bufferSize)
WWR_FUNCTION(gpusolverDnXgeqrf, cusolverDnXgeqrf, hipsolverDnXgeqrf)

} // namespace wwr
