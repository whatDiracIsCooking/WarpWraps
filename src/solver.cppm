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

#if defined(GPUMOD_GPU_BACKEND_CUDA)
#include <library_types.h>
#else
#include <hip/library_types.h>
#endif

export module gpumod.solver;

import gpumod.complex;
import gpumod.blas;
#if defined(GPUMOD_GPU_BACKEND_CUDA)
import gpumod.cuda.cusolverDn;
#else
import gpumod.hip.hipsolver;
#endif

export namespace gpumod {

// ========================================================================
// Types
// ========================================================================

GPUMOD_TYPE(gpusolverDnHandle_t, cusolverDnHandle_t, hipsolverDnHandle_t)
GPUMOD_TYPE(gpusolverDnParams_t, cusolverDnParams_t, hipsolverDnParams_t)
GPUMOD_TYPE(gpusolverStatus_t, cusolverStatus_t, hipsolverStatus_t)
GPUMOD_TYPE(gpusolverEigMode_t, cusolverEigMode_t, hipsolverEigMode_t)
GPUMOD_TYPE(gpusolverEigType_t, cusolverEigType_t, hipsolverEigType_t)
GPUMOD_TYPE(gpusolverEigRange_t, cusolverEigRange_t, hipsolverEigRange_t)
GPUMOD_TYPE(gpusolverSyevjInfo_t, syevjInfo_t, hipsolverSyevjInfo_t)
GPUMOD_TYPE(gpusolverGesvdjInfo_t, gesvdjInfo_t, hipsolverGesvdjInfo_t)
GPUMOD_TYPE(gpusolverDataType_t, cudaDataType, hipDataType)

// ========================================================================
// Constants
// ========================================================================

GPUMOD_VALUE(GPUSOLVER_STATUS_SUCCESS, CUSOLVER_STATUS_SUCCESS, HIPSOLVER_STATUS_SUCCESS)

GPUMOD_VALUE(GPUSOLVER_EIG_MODE_NOVECTOR, CUSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_EIG_MODE_NOVECTOR)
GPUMOD_VALUE(GPUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_MODE_VECTOR, HIPSOLVER_EIG_MODE_VECTOR)

GPUMOD_VALUE(GPUSOLVER_EIG_RANGE_ALL, CUSOLVER_EIG_RANGE_ALL, HIPSOLVER_EIG_RANGE_ALL)
GPUMOD_VALUE(GPUSOLVER_EIG_RANGE_V, CUSOLVER_EIG_RANGE_V, HIPSOLVER_EIG_RANGE_V)
GPUMOD_VALUE(GPUSOLVER_EIG_RANGE_I, CUSOLVER_EIG_RANGE_I, HIPSOLVER_EIG_RANGE_I)

GPUMOD_VALUE(GPUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_TYPE_1)
GPUMOD_VALUE(GPUSOLVER_EIG_TYPE_2, CUSOLVER_EIG_TYPE_2, HIPSOLVER_EIG_TYPE_2)
GPUMOD_VALUE(GPUSOLVER_EIG_TYPE_3, CUSOLVER_EIG_TYPE_3, HIPSOLVER_EIG_TYPE_3)

// cudaDataType/hipDataType enumerators are plain C enum constants with no
// gpumod-namespaced alias (see gpumod.blas -- gpuFloatComplex
// etc. are the closest precedent, but those DO have a namespaced alias via
// GPUMOD_COMPLEX_TYPE; these do not), so GPUMOD_VALUE's ::gpumod:: qualification
// does not apply here -- write the #if directly instead.
#if defined(GPUMOD_GPU_BACKEND_CUDA)
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
#if defined(GPUMOD_GPU_BACKEND_CUDA)
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
#if defined(GPUMOD_GPU_BACKEND_CUDA)
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

GPUMOD_FUNCTION(gpusolverDnCreate, cusolverDnCreate, hipsolverDnCreate)
GPUMOD_FUNCTION(gpusolverDnDestroy, cusolverDnDestroy, hipsolverDnDestroy)
GPUMOD_FUNCTION(gpusolverDnSetStream, cusolverDnSetStream, hipsolverDnSetStream)
GPUMOD_FUNCTION(gpusolverDnGetStream, cusolverDnGetStream, hipsolverDnGetStream)

GPUMOD_FUNCTION(gpusolverDnCreateParams, cusolverDnCreateParams, hipsolverDnCreateParams)
GPUMOD_FUNCTION(gpusolverDnDestroyParams, cusolverDnDestroyParams, hipsolverDnDestroyParams)

GPUMOD_FUNCTION(gpusolverDnCreateSyevjInfo, cusolverDnCreateSyevjInfo, hipsolverDnCreateSyevjInfo)
GPUMOD_FUNCTION(gpusolverDnDestroySyevjInfo, cusolverDnDestroySyevjInfo,
                hipsolverDnDestroySyevjInfo)
GPUMOD_FUNCTION(gpusolverDnCreateGesvdjInfo, cusolverDnCreateGesvdjInfo,
                hipsolverDnCreateGesvdjInfo)
GPUMOD_FUNCTION(gpusolverDnDestroyGesvdjInfo, cusolverDnDestroyGesvdjInfo,
                hipsolverDnDestroyGesvdjInfo)

// ────────────────────────────────────────────────────────────────────────
// Legacy linear solver API (int-based, pre-params) -- 32 functions, all
// shared between cuSOLVER and hipSOLVER
// ────────────────────────────────────────────────────────────────────────

GPUMOD_FUNCTION(gpusolverDnSpotrf_bufferSize, cusolverDnSpotrf_bufferSize,
                hipsolverDnSpotrf_bufferSize)
GPUMOD_FUNCTION(gpusolverDnDpotrf_bufferSize, cusolverDnDpotrf_bufferSize,
                hipsolverDnDpotrf_bufferSize)
GPUMOD_FUNCTION(gpusolverDnCpotrf_bufferSize, cusolverDnCpotrf_bufferSize,
                hipsolverDnCpotrf_bufferSize)
GPUMOD_FUNCTION(gpusolverDnZpotrf_bufferSize, cusolverDnZpotrf_bufferSize,
                hipsolverDnZpotrf_bufferSize)
GPUMOD_FUNCTION(gpusolverDnSpotrf, cusolverDnSpotrf, hipsolverDnSpotrf)
GPUMOD_FUNCTION(gpusolverDnDpotrf, cusolverDnDpotrf, hipsolverDnDpotrf)
GPUMOD_FUNCTION(gpusolverDnCpotrf, cusolverDnCpotrf, hipsolverDnCpotrf)
GPUMOD_FUNCTION(gpusolverDnZpotrf, cusolverDnZpotrf, hipsolverDnZpotrf)
GPUMOD_FUNCTION(gpusolverDnSpotrs, cusolverDnSpotrs, hipsolverDnSpotrs)
GPUMOD_FUNCTION(gpusolverDnDpotrs, cusolverDnDpotrs, hipsolverDnDpotrs)
GPUMOD_FUNCTION(gpusolverDnCpotrs, cusolverDnCpotrs, hipsolverDnCpotrs)
GPUMOD_FUNCTION(gpusolverDnZpotrs, cusolverDnZpotrs, hipsolverDnZpotrs)
GPUMOD_FUNCTION(gpusolverDnSpotri_bufferSize, cusolverDnSpotri_bufferSize,
                hipsolverDnSpotri_bufferSize)
GPUMOD_FUNCTION(gpusolverDnDpotri_bufferSize, cusolverDnDpotri_bufferSize,
                hipsolverDnDpotri_bufferSize)
GPUMOD_FUNCTION(gpusolverDnCpotri_bufferSize, cusolverDnCpotri_bufferSize,
                hipsolverDnCpotri_bufferSize)
GPUMOD_FUNCTION(gpusolverDnZpotri_bufferSize, cusolverDnZpotri_bufferSize,
                hipsolverDnZpotri_bufferSize)
GPUMOD_FUNCTION(gpusolverDnSpotri, cusolverDnSpotri, hipsolverDnSpotri)
GPUMOD_FUNCTION(gpusolverDnDpotri, cusolverDnDpotri, hipsolverDnDpotri)
GPUMOD_FUNCTION(gpusolverDnCpotri, cusolverDnCpotri, hipsolverDnCpotri)
GPUMOD_FUNCTION(gpusolverDnZpotri, cusolverDnZpotri, hipsolverDnZpotri)
GPUMOD_FUNCTION(gpusolverDnSgetrf_bufferSize, cusolverDnSgetrf_bufferSize,
                hipsolverDnSgetrf_bufferSize)
GPUMOD_FUNCTION(gpusolverDnDgetrf_bufferSize, cusolverDnDgetrf_bufferSize,
                hipsolverDnDgetrf_bufferSize)
GPUMOD_FUNCTION(gpusolverDnCgetrf_bufferSize, cusolverDnCgetrf_bufferSize,
                hipsolverDnCgetrf_bufferSize)
GPUMOD_FUNCTION(gpusolverDnZgetrf_bufferSize, cusolverDnZgetrf_bufferSize,
                hipsolverDnZgetrf_bufferSize)
GPUMOD_FUNCTION(gpusolverDnSgetrf, cusolverDnSgetrf, hipsolverDnSgetrf)
GPUMOD_FUNCTION(gpusolverDnDgetrf, cusolverDnDgetrf, hipsolverDnDgetrf)
GPUMOD_FUNCTION(gpusolverDnCgetrf, cusolverDnCgetrf, hipsolverDnCgetrf)
GPUMOD_FUNCTION(gpusolverDnZgetrf, cusolverDnZgetrf, hipsolverDnZgetrf)
GPUMOD_FUNCTION(gpusolverDnSgetrs, cusolverDnSgetrs, hipsolverDnSgetrs)
GPUMOD_FUNCTION(gpusolverDnDgetrs, cusolverDnDgetrs, hipsolverDnDgetrs)
GPUMOD_FUNCTION(gpusolverDnCgetrs, cusolverDnCgetrs, hipsolverDnCgetrs)
GPUMOD_FUNCTION(gpusolverDnZgetrs, cusolverDnZgetrs, hipsolverDnZgetrs)
GPUMOD_FUNCTION(gpusolverDnSgeqrf_bufferSize, cusolverDnSgeqrf_bufferSize,
                hipsolverDnSgeqrf_bufferSize)
GPUMOD_FUNCTION(gpusolverDnDgeqrf_bufferSize, cusolverDnDgeqrf_bufferSize,
                hipsolverDnDgeqrf_bufferSize)
GPUMOD_FUNCTION(gpusolverDnCgeqrf_bufferSize, cusolverDnCgeqrf_bufferSize,
                hipsolverDnCgeqrf_bufferSize)
GPUMOD_FUNCTION(gpusolverDnZgeqrf_bufferSize, cusolverDnZgeqrf_bufferSize,
                hipsolverDnZgeqrf_bufferSize)
GPUMOD_FUNCTION(gpusolverDnSgeqrf, cusolverDnSgeqrf, hipsolverDnSgeqrf)
GPUMOD_FUNCTION(gpusolverDnDgeqrf, cusolverDnDgeqrf, hipsolverDnDgeqrf)
GPUMOD_FUNCTION(gpusolverDnCgeqrf, cusolverDnCgeqrf, hipsolverDnCgeqrf)
GPUMOD_FUNCTION(gpusolverDnZgeqrf, cusolverDnZgeqrf, hipsolverDnZgeqrf)
GPUMOD_FUNCTION(gpusolverDnSormqr_bufferSize, cusolverDnSormqr_bufferSize,
                hipsolverDnSormqr_bufferSize)
GPUMOD_FUNCTION(gpusolverDnDormqr_bufferSize, cusolverDnDormqr_bufferSize,
                hipsolverDnDormqr_bufferSize)
GPUMOD_FUNCTION(gpusolverDnSormqr, cusolverDnSormqr, hipsolverDnSormqr)
GPUMOD_FUNCTION(gpusolverDnDormqr, cusolverDnDormqr, hipsolverDnDormqr)
GPUMOD_FUNCTION(gpusolverDnCunmqr_bufferSize, cusolverDnCunmqr_bufferSize,
                hipsolverDnCunmqr_bufferSize)
GPUMOD_FUNCTION(gpusolverDnZunmqr_bufferSize, cusolverDnZunmqr_bufferSize,
                hipsolverDnZunmqr_bufferSize)
GPUMOD_FUNCTION(gpusolverDnCunmqr, cusolverDnCunmqr, hipsolverDnCunmqr)
GPUMOD_FUNCTION(gpusolverDnZunmqr, cusolverDnZunmqr, hipsolverDnZunmqr)
GPUMOD_FUNCTION(gpusolverDnSSgels_bufferSize, cusolverDnSSgels_bufferSize,
                hipsolverDnSSgels_bufferSize)
GPUMOD_FUNCTION(gpusolverDnDDgels_bufferSize, cusolverDnDDgels_bufferSize,
                hipsolverDnDDgels_bufferSize)
GPUMOD_FUNCTION(gpusolverDnCCgels_bufferSize, cusolverDnCCgels_bufferSize,
                hipsolverDnCCgels_bufferSize)
GPUMOD_FUNCTION(gpusolverDnZZgels_bufferSize, cusolverDnZZgels_bufferSize,
                hipsolverDnZZgels_bufferSize)
GPUMOD_FUNCTION(gpusolverDnSSgels, cusolverDnSSgels, hipsolverDnSSgels)
GPUMOD_FUNCTION(gpusolverDnDDgels, cusolverDnDDgels, hipsolverDnDDgels)
GPUMOD_FUNCTION(gpusolverDnCCgels, cusolverDnCCgels, hipsolverDnCCgels)
GPUMOD_FUNCTION(gpusolverDnZZgels, cusolverDnZZgels, hipsolverDnZZgels)
GPUMOD_FUNCTION(gpusolverDnSSgesv_bufferSize, cusolverDnSSgesv_bufferSize,
                hipsolverDnSSgesv_bufferSize)
GPUMOD_FUNCTION(gpusolverDnDDgesv_bufferSize, cusolverDnDDgesv_bufferSize,
                hipsolverDnDDgesv_bufferSize)
GPUMOD_FUNCTION(gpusolverDnCCgesv_bufferSize, cusolverDnCCgesv_bufferSize,
                hipsolverDnCCgesv_bufferSize)
GPUMOD_FUNCTION(gpusolverDnZZgesv_bufferSize, cusolverDnZZgesv_bufferSize,
                hipsolverDnZZgesv_bufferSize)
GPUMOD_FUNCTION(gpusolverDnSSgesv, cusolverDnSSgesv, hipsolverDnSSgesv)
GPUMOD_FUNCTION(gpusolverDnDDgesv, cusolverDnDDgesv, hipsolverDnDDgesv)
GPUMOD_FUNCTION(gpusolverDnCCgesv, cusolverDnCCgesv, hipsolverDnCCgesv)
GPUMOD_FUNCTION(gpusolverDnZZgesv, cusolverDnZZgesv, hipsolverDnZZgesv)
GPUMOD_FUNCTION(gpusolverDnSpotrfBatched, cusolverDnSpotrfBatched, hipsolverDnSpotrfBatched)
GPUMOD_FUNCTION(gpusolverDnDpotrfBatched, cusolverDnDpotrfBatched, hipsolverDnDpotrfBatched)
GPUMOD_FUNCTION(gpusolverDnCpotrfBatched, cusolverDnCpotrfBatched, hipsolverDnCpotrfBatched)
GPUMOD_FUNCTION(gpusolverDnZpotrfBatched, cusolverDnZpotrfBatched, hipsolverDnZpotrfBatched)
GPUMOD_FUNCTION(gpusolverDnSpotrsBatched, cusolverDnSpotrsBatched, hipsolverDnSpotrsBatched)
GPUMOD_FUNCTION(gpusolverDnDpotrsBatched, cusolverDnDpotrsBatched, hipsolverDnDpotrsBatched)
GPUMOD_FUNCTION(gpusolverDnCpotrsBatched, cusolverDnCpotrsBatched, hipsolverDnCpotrsBatched)
GPUMOD_FUNCTION(gpusolverDnZpotrsBatched, cusolverDnZpotrsBatched, hipsolverDnZpotrsBatched)
GPUMOD_FUNCTION(gpusolverDnSsytrf_bufferSize, cusolverDnSsytrf_bufferSize,
                hipsolverDnSsytrf_bufferSize)
GPUMOD_FUNCTION(gpusolverDnDsytrf_bufferSize, cusolverDnDsytrf_bufferSize,
                hipsolverDnDsytrf_bufferSize)
GPUMOD_FUNCTION(gpusolverDnCsytrf_bufferSize, cusolverDnCsytrf_bufferSize,
                hipsolverDnCsytrf_bufferSize)
GPUMOD_FUNCTION(gpusolverDnZsytrf_bufferSize, cusolverDnZsytrf_bufferSize,
                hipsolverDnZsytrf_bufferSize)
GPUMOD_FUNCTION(gpusolverDnSsytrf, cusolverDnSsytrf, hipsolverDnSsytrf)
GPUMOD_FUNCTION(gpusolverDnDsytrf, cusolverDnDsytrf, hipsolverDnDsytrf)
GPUMOD_FUNCTION(gpusolverDnCsytrf, cusolverDnCsytrf, hipsolverDnCsytrf)
GPUMOD_FUNCTION(gpusolverDnZsytrf, cusolverDnZsytrf, hipsolverDnZsytrf)
GPUMOD_FUNCTION(gpusolverDnSgebrd_bufferSize, cusolverDnSgebrd_bufferSize,
                hipsolverDnSgebrd_bufferSize)
GPUMOD_FUNCTION(gpusolverDnDgebrd_bufferSize, cusolverDnDgebrd_bufferSize,
                hipsolverDnDgebrd_bufferSize)
GPUMOD_FUNCTION(gpusolverDnCgebrd_bufferSize, cusolverDnCgebrd_bufferSize,
                hipsolverDnCgebrd_bufferSize)
GPUMOD_FUNCTION(gpusolverDnZgebrd_bufferSize, cusolverDnZgebrd_bufferSize,
                hipsolverDnZgebrd_bufferSize)
GPUMOD_FUNCTION(gpusolverDnSgebrd, cusolverDnSgebrd, hipsolverDnSgebrd)
GPUMOD_FUNCTION(gpusolverDnDgebrd, cusolverDnDgebrd, hipsolverDnDgebrd)
GPUMOD_FUNCTION(gpusolverDnCgebrd, cusolverDnCgebrd, hipsolverDnCgebrd)
GPUMOD_FUNCTION(gpusolverDnZgebrd, cusolverDnZgebrd, hipsolverDnZgebrd)
GPUMOD_FUNCTION(gpusolverDnSorgqr_bufferSize, cusolverDnSorgqr_bufferSize,
                hipsolverDnSorgqr_bufferSize)
GPUMOD_FUNCTION(gpusolverDnDorgqr_bufferSize, cusolverDnDorgqr_bufferSize,
                hipsolverDnDorgqr_bufferSize)
GPUMOD_FUNCTION(gpusolverDnCungqr_bufferSize, cusolverDnCungqr_bufferSize,
                hipsolverDnCungqr_bufferSize)
GPUMOD_FUNCTION(gpusolverDnZungqr_bufferSize, cusolverDnZungqr_bufferSize,
                hipsolverDnZungqr_bufferSize)
GPUMOD_FUNCTION(gpusolverDnSorgqr, cusolverDnSorgqr, hipsolverDnSorgqr)
GPUMOD_FUNCTION(gpusolverDnDorgqr, cusolverDnDorgqr, hipsolverDnDorgqr)
GPUMOD_FUNCTION(gpusolverDnCungqr, cusolverDnCungqr, hipsolverDnCungqr)
GPUMOD_FUNCTION(gpusolverDnZungqr, cusolverDnZungqr, hipsolverDnZungqr)
GPUMOD_FUNCTION(gpusolverDnSorgbr_bufferSize, cusolverDnSorgbr_bufferSize,
                hipsolverDnSorgbr_bufferSize)
GPUMOD_FUNCTION(gpusolverDnDorgbr_bufferSize, cusolverDnDorgbr_bufferSize,
                hipsolverDnDorgbr_bufferSize)
GPUMOD_FUNCTION(gpusolverDnCungbr_bufferSize, cusolverDnCungbr_bufferSize,
                hipsolverDnCungbr_bufferSize)
GPUMOD_FUNCTION(gpusolverDnZungbr_bufferSize, cusolverDnZungbr_bufferSize,
                hipsolverDnZungbr_bufferSize)
GPUMOD_FUNCTION(gpusolverDnSorgbr, cusolverDnSorgbr, hipsolverDnSorgbr)
GPUMOD_FUNCTION(gpusolverDnDorgbr, cusolverDnDorgbr, hipsolverDnDorgbr)
GPUMOD_FUNCTION(gpusolverDnCungbr, cusolverDnCungbr, hipsolverDnCungbr)
GPUMOD_FUNCTION(gpusolverDnZungbr, cusolverDnZungbr, hipsolverDnZungbr)

// ────────────────────────────────────────────────────────────────────────
// Legacy eigenvalue/SVD solver API (int-based, pre-params) -- 48 functions,
// all shared between cuSOLVER and hipSOLVER
// ────────────────────────────────────────────────────────────────────────

GPUMOD_FUNCTION(gpusolverDnSgesvd_bufferSize, cusolverDnSgesvd_bufferSize,
                hipsolverDnSgesvd_bufferSize)
GPUMOD_FUNCTION(gpusolverDnDgesvd_bufferSize, cusolverDnDgesvd_bufferSize,
                hipsolverDnDgesvd_bufferSize)
GPUMOD_FUNCTION(gpusolverDnCgesvd_bufferSize, cusolverDnCgesvd_bufferSize,
                hipsolverDnCgesvd_bufferSize)
GPUMOD_FUNCTION(gpusolverDnZgesvd_bufferSize, cusolverDnZgesvd_bufferSize,
                hipsolverDnZgesvd_bufferSize)
GPUMOD_FUNCTION(gpusolverDnSgesvd, cusolverDnSgesvd, hipsolverDnSgesvd)
GPUMOD_FUNCTION(gpusolverDnDgesvd, cusolverDnDgesvd, hipsolverDnDgesvd)
GPUMOD_FUNCTION(gpusolverDnCgesvd, cusolverDnCgesvd, hipsolverDnCgesvd)
GPUMOD_FUNCTION(gpusolverDnZgesvd, cusolverDnZgesvd, hipsolverDnZgesvd)
GPUMOD_FUNCTION(gpusolverDnSsyevd_bufferSize, cusolverDnSsyevd_bufferSize,
                hipsolverDnSsyevd_bufferSize)
GPUMOD_FUNCTION(gpusolverDnDsyevd_bufferSize, cusolverDnDsyevd_bufferSize,
                hipsolverDnDsyevd_bufferSize)
GPUMOD_FUNCTION(gpusolverDnSsyevd, cusolverDnSsyevd, hipsolverDnSsyevd)
GPUMOD_FUNCTION(gpusolverDnDsyevd, cusolverDnDsyevd, hipsolverDnDsyevd)
GPUMOD_FUNCTION(gpusolverDnSsyevdx_bufferSize, cusolverDnSsyevdx_bufferSize,
                hipsolverDnSsyevdx_bufferSize)
GPUMOD_FUNCTION(gpusolverDnDsyevdx_bufferSize, cusolverDnDsyevdx_bufferSize,
                hipsolverDnDsyevdx_bufferSize)
GPUMOD_FUNCTION(gpusolverDnSsyevdx, cusolverDnSsyevdx, hipsolverDnSsyevdx)
GPUMOD_FUNCTION(gpusolverDnDsyevdx, cusolverDnDsyevdx, hipsolverDnDsyevdx)
GPUMOD_FUNCTION(gpusolverDnCheevd_bufferSize, cusolverDnCheevd_bufferSize,
                hipsolverDnCheevd_bufferSize)
GPUMOD_FUNCTION(gpusolverDnZheevd_bufferSize, cusolverDnZheevd_bufferSize,
                hipsolverDnZheevd_bufferSize)
GPUMOD_FUNCTION(gpusolverDnCheevd, cusolverDnCheevd, hipsolverDnCheevd)
GPUMOD_FUNCTION(gpusolverDnZheevd, cusolverDnZheevd, hipsolverDnZheevd)
GPUMOD_FUNCTION(gpusolverDnCheevdx_bufferSize, cusolverDnCheevdx_bufferSize,
                hipsolverDnCheevdx_bufferSize)
GPUMOD_FUNCTION(gpusolverDnZheevdx_bufferSize, cusolverDnZheevdx_bufferSize,
                hipsolverDnZheevdx_bufferSize)
GPUMOD_FUNCTION(gpusolverDnCheevdx, cusolverDnCheevdx, hipsolverDnCheevdx)
GPUMOD_FUNCTION(gpusolverDnZheevdx, cusolverDnZheevdx, hipsolverDnZheevdx)
GPUMOD_FUNCTION(gpusolverDnSsytrd_bufferSize, cusolverDnSsytrd_bufferSize,
                hipsolverDnSsytrd_bufferSize)
GPUMOD_FUNCTION(gpusolverDnDsytrd_bufferSize, cusolverDnDsytrd_bufferSize,
                hipsolverDnDsytrd_bufferSize)
GPUMOD_FUNCTION(gpusolverDnSsytrd, cusolverDnSsytrd, hipsolverDnSsytrd)
GPUMOD_FUNCTION(gpusolverDnDsytrd, cusolverDnDsytrd, hipsolverDnDsytrd)
GPUMOD_FUNCTION(gpusolverDnChetrd_bufferSize, cusolverDnChetrd_bufferSize,
                hipsolverDnChetrd_bufferSize)
GPUMOD_FUNCTION(gpusolverDnZhetrd_bufferSize, cusolverDnZhetrd_bufferSize,
                hipsolverDnZhetrd_bufferSize)
GPUMOD_FUNCTION(gpusolverDnChetrd, cusolverDnChetrd, hipsolverDnChetrd)
GPUMOD_FUNCTION(gpusolverDnZhetrd, cusolverDnZhetrd, hipsolverDnZhetrd)
GPUMOD_FUNCTION(gpusolverDnSorgtr_bufferSize, cusolverDnSorgtr_bufferSize,
                hipsolverDnSorgtr_bufferSize)
GPUMOD_FUNCTION(gpusolverDnDorgtr_bufferSize, cusolverDnDorgtr_bufferSize,
                hipsolverDnDorgtr_bufferSize)
GPUMOD_FUNCTION(gpusolverDnSorgtr, cusolverDnSorgtr, hipsolverDnSorgtr)
GPUMOD_FUNCTION(gpusolverDnDorgtr, cusolverDnDorgtr, hipsolverDnDorgtr)
GPUMOD_FUNCTION(gpusolverDnCungtr_bufferSize, cusolverDnCungtr_bufferSize,
                hipsolverDnCungtr_bufferSize)
GPUMOD_FUNCTION(gpusolverDnZungtr_bufferSize, cusolverDnZungtr_bufferSize,
                hipsolverDnZungtr_bufferSize)
GPUMOD_FUNCTION(gpusolverDnCungtr, cusolverDnCungtr, hipsolverDnCungtr)
GPUMOD_FUNCTION(gpusolverDnZungtr, cusolverDnZungtr, hipsolverDnZungtr)
GPUMOD_FUNCTION(gpusolverDnSormtr_bufferSize, cusolverDnSormtr_bufferSize,
                hipsolverDnSormtr_bufferSize)
GPUMOD_FUNCTION(gpusolverDnDormtr_bufferSize, cusolverDnDormtr_bufferSize,
                hipsolverDnDormtr_bufferSize)
GPUMOD_FUNCTION(gpusolverDnSormtr, cusolverDnSormtr, hipsolverDnSormtr)
GPUMOD_FUNCTION(gpusolverDnDormtr, cusolverDnDormtr, hipsolverDnDormtr)
GPUMOD_FUNCTION(gpusolverDnCunmtr_bufferSize, cusolverDnCunmtr_bufferSize,
                hipsolverDnCunmtr_bufferSize)
GPUMOD_FUNCTION(gpusolverDnZunmtr_bufferSize, cusolverDnZunmtr_bufferSize,
                hipsolverDnZunmtr_bufferSize)
GPUMOD_FUNCTION(gpusolverDnCunmtr, cusolverDnCunmtr, hipsolverDnCunmtr)
GPUMOD_FUNCTION(gpusolverDnZunmtr, cusolverDnZunmtr, hipsolverDnZunmtr)
GPUMOD_FUNCTION(gpusolverDnSsyevj_bufferSize, cusolverDnSsyevj_bufferSize,
                hipsolverDnSsyevj_bufferSize)
GPUMOD_FUNCTION(gpusolverDnDsyevj_bufferSize, cusolverDnDsyevj_bufferSize,
                hipsolverDnDsyevj_bufferSize)
GPUMOD_FUNCTION(gpusolverDnSsyevj, cusolverDnSsyevj, hipsolverDnSsyevj)
GPUMOD_FUNCTION(gpusolverDnDsyevj, cusolverDnDsyevj, hipsolverDnDsyevj)
GPUMOD_FUNCTION(gpusolverDnSsyevjBatched_bufferSize, cusolverDnSsyevjBatched_bufferSize,
                hipsolverDnSsyevjBatched_bufferSize)
GPUMOD_FUNCTION(gpusolverDnDsyevjBatched_bufferSize, cusolverDnDsyevjBatched_bufferSize,
                hipsolverDnDsyevjBatched_bufferSize)
GPUMOD_FUNCTION(gpusolverDnSsyevjBatched, cusolverDnSsyevjBatched, hipsolverDnSsyevjBatched)
GPUMOD_FUNCTION(gpusolverDnDsyevjBatched, cusolverDnDsyevjBatched, hipsolverDnDsyevjBatched)
GPUMOD_FUNCTION(gpusolverDnCheevj_bufferSize, cusolverDnCheevj_bufferSize,
                hipsolverDnCheevj_bufferSize)
GPUMOD_FUNCTION(gpusolverDnZheevj_bufferSize, cusolverDnZheevj_bufferSize,
                hipsolverDnZheevj_bufferSize)
GPUMOD_FUNCTION(gpusolverDnCheevj, cusolverDnCheevj, hipsolverDnCheevj)
GPUMOD_FUNCTION(gpusolverDnZheevj, cusolverDnZheevj, hipsolverDnZheevj)
GPUMOD_FUNCTION(gpusolverDnCheevjBatched_bufferSize, cusolverDnCheevjBatched_bufferSize,
                hipsolverDnCheevjBatched_bufferSize)
GPUMOD_FUNCTION(gpusolverDnZheevjBatched_bufferSize, cusolverDnZheevjBatched_bufferSize,
                hipsolverDnZheevjBatched_bufferSize)
GPUMOD_FUNCTION(gpusolverDnCheevjBatched, cusolverDnCheevjBatched, hipsolverDnCheevjBatched)
GPUMOD_FUNCTION(gpusolverDnZheevjBatched, cusolverDnZheevjBatched, hipsolverDnZheevjBatched)
GPUMOD_FUNCTION(gpusolverDnSgesvdj_bufferSize, cusolverDnSgesvdj_bufferSize,
                hipsolverDnSgesvdj_bufferSize)
GPUMOD_FUNCTION(gpusolverDnDgesvdj_bufferSize, cusolverDnDgesvdj_bufferSize,
                hipsolverDnDgesvdj_bufferSize)
GPUMOD_FUNCTION(gpusolverDnCgesvdj_bufferSize, cusolverDnCgesvdj_bufferSize,
                hipsolverDnCgesvdj_bufferSize)
GPUMOD_FUNCTION(gpusolverDnZgesvdj_bufferSize, cusolverDnZgesvdj_bufferSize,
                hipsolverDnZgesvdj_bufferSize)
GPUMOD_FUNCTION(gpusolverDnSgesvdj, cusolverDnSgesvdj, hipsolverDnSgesvdj)
GPUMOD_FUNCTION(gpusolverDnDgesvdj, cusolverDnDgesvdj, hipsolverDnDgesvdj)
GPUMOD_FUNCTION(gpusolverDnCgesvdj, cusolverDnCgesvdj, hipsolverDnCgesvdj)
GPUMOD_FUNCTION(gpusolverDnZgesvdj, cusolverDnZgesvdj, hipsolverDnZgesvdj)
GPUMOD_FUNCTION(gpusolverDnSgesvdjBatched_bufferSize, cusolverDnSgesvdjBatched_bufferSize,
                hipsolverDnSgesvdjBatched_bufferSize)
GPUMOD_FUNCTION(gpusolverDnDgesvdjBatched_bufferSize, cusolverDnDgesvdjBatched_bufferSize,
                hipsolverDnDgesvdjBatched_bufferSize)
GPUMOD_FUNCTION(gpusolverDnCgesvdjBatched_bufferSize, cusolverDnCgesvdjBatched_bufferSize,
                hipsolverDnCgesvdjBatched_bufferSize)
GPUMOD_FUNCTION(gpusolverDnZgesvdjBatched_bufferSize, cusolverDnZgesvdjBatched_bufferSize,
                hipsolverDnZgesvdjBatched_bufferSize)
GPUMOD_FUNCTION(gpusolverDnSgesvdjBatched, cusolverDnSgesvdjBatched, hipsolverDnSgesvdjBatched)
GPUMOD_FUNCTION(gpusolverDnDgesvdjBatched, cusolverDnDgesvdjBatched, hipsolverDnDgesvdjBatched)
GPUMOD_FUNCTION(gpusolverDnCgesvdjBatched, cusolverDnCgesvdjBatched, hipsolverDnCgesvdjBatched)
GPUMOD_FUNCTION(gpusolverDnZgesvdjBatched, cusolverDnZgesvdjBatched, hipsolverDnZgesvdjBatched)
GPUMOD_FUNCTION(gpusolverDnSsygvd_bufferSize, cusolverDnSsygvd_bufferSize,
                hipsolverDnSsygvd_bufferSize)
GPUMOD_FUNCTION(gpusolverDnDsygvd_bufferSize, cusolverDnDsygvd_bufferSize,
                hipsolverDnDsygvd_bufferSize)
GPUMOD_FUNCTION(gpusolverDnSsygvd, cusolverDnSsygvd, hipsolverDnSsygvd)
GPUMOD_FUNCTION(gpusolverDnDsygvd, cusolverDnDsygvd, hipsolverDnDsygvd)
GPUMOD_FUNCTION(gpusolverDnSsygvdx_bufferSize, cusolverDnSsygvdx_bufferSize,
                hipsolverDnSsygvdx_bufferSize)
GPUMOD_FUNCTION(gpusolverDnDsygvdx_bufferSize, cusolverDnDsygvdx_bufferSize,
                hipsolverDnDsygvdx_bufferSize)
GPUMOD_FUNCTION(gpusolverDnSsygvdx, cusolverDnSsygvdx, hipsolverDnSsygvdx)
GPUMOD_FUNCTION(gpusolverDnDsygvdx, cusolverDnDsygvdx, hipsolverDnDsygvdx)
GPUMOD_FUNCTION(gpusolverDnChegvd_bufferSize, cusolverDnChegvd_bufferSize,
                hipsolverDnChegvd_bufferSize)
GPUMOD_FUNCTION(gpusolverDnZhegvd_bufferSize, cusolverDnZhegvd_bufferSize,
                hipsolverDnZhegvd_bufferSize)
GPUMOD_FUNCTION(gpusolverDnChegvd, cusolverDnChegvd, hipsolverDnChegvd)
GPUMOD_FUNCTION(gpusolverDnZhegvd, cusolverDnZhegvd, hipsolverDnZhegvd)
GPUMOD_FUNCTION(gpusolverDnChegvdx_bufferSize, cusolverDnChegvdx_bufferSize,
                hipsolverDnChegvdx_bufferSize)
GPUMOD_FUNCTION(gpusolverDnZhegvdx_bufferSize, cusolverDnZhegvdx_bufferSize,
                hipsolverDnZhegvdx_bufferSize)
GPUMOD_FUNCTION(gpusolverDnChegvdx, cusolverDnChegvdx, hipsolverDnChegvdx)
GPUMOD_FUNCTION(gpusolverDnZhegvdx, cusolverDnZhegvdx, hipsolverDnZhegvdx)
GPUMOD_FUNCTION(gpusolverDnSsygvj_bufferSize, cusolverDnSsygvj_bufferSize,
                hipsolverDnSsygvj_bufferSize)
GPUMOD_FUNCTION(gpusolverDnDsygvj_bufferSize, cusolverDnDsygvj_bufferSize,
                hipsolverDnDsygvj_bufferSize)
GPUMOD_FUNCTION(gpusolverDnSsygvj, cusolverDnSsygvj, hipsolverDnSsygvj)
GPUMOD_FUNCTION(gpusolverDnDsygvj, cusolverDnDsygvj, hipsolverDnDsygvj)
GPUMOD_FUNCTION(gpusolverDnChegvj_bufferSize, cusolverDnChegvj_bufferSize,
                hipsolverDnChegvj_bufferSize)
GPUMOD_FUNCTION(gpusolverDnZhegvj_bufferSize, cusolverDnZhegvj_bufferSize,
                hipsolverDnZhegvj_bufferSize)
GPUMOD_FUNCTION(gpusolverDnChegvj, cusolverDnChegvj, hipsolverDnChegvj)
GPUMOD_FUNCTION(gpusolverDnZhegvj, cusolverDnZhegvj, hipsolverDnZhegvj)
GPUMOD_FUNCTION(gpusolverDnSgesvdaStridedBatched_bufferSize,
                cusolverDnSgesvdaStridedBatched_bufferSize,
                hipsolverDnSgesvdaStridedBatched_bufferSize)
GPUMOD_FUNCTION(gpusolverDnDgesvdaStridedBatched_bufferSize,
                cusolverDnDgesvdaStridedBatched_bufferSize,
                hipsolverDnDgesvdaStridedBatched_bufferSize)
GPUMOD_FUNCTION(gpusolverDnCgesvdaStridedBatched_bufferSize,
                cusolverDnCgesvdaStridedBatched_bufferSize,
                hipsolverDnCgesvdaStridedBatched_bufferSize)
GPUMOD_FUNCTION(gpusolverDnZgesvdaStridedBatched_bufferSize,
                cusolverDnZgesvdaStridedBatched_bufferSize,
                hipsolverDnZgesvdaStridedBatched_bufferSize)
GPUMOD_FUNCTION(gpusolverDnSgesvdaStridedBatched, cusolverDnSgesvdaStridedBatched,
                hipsolverDnSgesvdaStridedBatched)
GPUMOD_FUNCTION(gpusolverDnDgesvdaStridedBatched, cusolverDnDgesvdaStridedBatched,
                hipsolverDnDgesvdaStridedBatched)
GPUMOD_FUNCTION(gpusolverDnCgesvdaStridedBatched, cusolverDnCgesvdaStridedBatched,
                hipsolverDnCgesvdaStridedBatched)
GPUMOD_FUNCTION(gpusolverDnZgesvdaStridedBatched, cusolverDnZgesvdaStridedBatched,
                hipsolverDnZgesvdaStridedBatched)

// ────────────────────────────────────────────────────────────────────────
// Modern (X-prefixed, gpusolverDnParams_t-based, int64_t-dimensioned) API --
// 8 functions shared between cuSOLVER and hipSOLVER (potrf, potrs, getrf,
// getrs, geqrf + _bufferSize where cuSOLVER has one). hipsolverDn has no
// X-prefixed counterpart for anything else cuSOLVER's modern API offers
// (larft, sytrs, trtri, and the entire modern eigenvalue/SVD surface); those
// are unwrapped -- reach them through gpumod.cuda.cusolverDn directly.
// ────────────────────────────────────────────────────────────────────────

GPUMOD_FUNCTION(gpusolverDnXpotrf_bufferSize, cusolverDnXpotrf_bufferSize,
                hipsolverDnXpotrf_bufferSize)
GPUMOD_FUNCTION(gpusolverDnXpotrf, cusolverDnXpotrf, hipsolverDnXpotrf)
GPUMOD_FUNCTION(gpusolverDnXpotrs, cusolverDnXpotrs, hipsolverDnXpotrs)
GPUMOD_FUNCTION(gpusolverDnXgetrf_bufferSize, cusolverDnXgetrf_bufferSize,
                hipsolverDnXgetrf_bufferSize)
GPUMOD_FUNCTION(gpusolverDnXgetrf, cusolverDnXgetrf, hipsolverDnXgetrf)
GPUMOD_FUNCTION(gpusolverDnXgetrs, cusolverDnXgetrs, hipsolverDnXgetrs)
GPUMOD_FUNCTION(gpusolverDnXgeqrf_bufferSize, cusolverDnXgeqrf_bufferSize,
                hipsolverDnXgeqrf_bufferSize)
GPUMOD_FUNCTION(gpusolverDnXgeqrf, cusolverDnXgeqrf, hipsolverDnXgeqrf)

} // namespace gpumod
