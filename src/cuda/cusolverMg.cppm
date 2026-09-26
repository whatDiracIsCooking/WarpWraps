/**
 * @file cusolverMg.cppm
 * @brief Primary interface for gpumod.cuda.cusolverMg
 *
 * This module wraps the cuSOLVER Multi-GPU API and exports types, constants,
 * and functions for multi-GPU dense linear algebra operations.
 *
 * Usage:
 *   import gpumod.cuda.cusolverMg;
 */

module;

#include <cusolverMg.h>

export module gpumod.cuda.cusolverMg;

import std;

// ========================================================================
// Export all cuSOLVER Multi-GPU types and functions in gpumod namespace
// ========================================================================

export namespace gpumod::cuda {

// ========================================================================
// Core Types
// ========================================================================

// Multi-GPU handle
using ::cusolverMgContext;
using ::cusolverMgHandle_t;

// Distributed grid and matrix descriptor (opaque void* typedefs)
using ::cudaLibMgGrid_t;
using ::cudaLibMgMatrixDesc_t;

// Status type (shared with cuSOLVER Dense, defined in cusolver_common.h)
using ::cusolverStatus_t;

// ========================================================================
// Status Codes (from cusolver_common.h via cusolverDn.h)
// ========================================================================
using ::CUSOLVER_STATUS_ALLOC_FAILED;
using ::CUSOLVER_STATUS_ARCH_MISMATCH;
using ::CUSOLVER_STATUS_EXECUTION_FAILED;
using ::CUSOLVER_STATUS_INTERNAL_ERROR;
using ::CUSOLVER_STATUS_INVALID_LICENSE;
using ::CUSOLVER_STATUS_INVALID_VALUE;
using ::CUSOLVER_STATUS_INVALID_WORKSPACE;
using ::CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED;
using ::CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED;
using ::CUSOLVER_STATUS_IRS_INTERNAL_ERROR;
using ::CUSOLVER_STATUS_IRS_MATRIX_SINGULAR;
using ::CUSOLVER_STATUS_IRS_NOT_SUPPORTED;
using ::CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES;
using ::CUSOLVER_STATUS_IRS_OUT_OF_RANGE;
using ::CUSOLVER_STATUS_IRS_PARAMS_INVALID;
using ::CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER;
using ::CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC;
using ::CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE;
using ::CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED;
using ::CUSOLVER_STATUS_MAPPING_ERROR;
using ::CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
using ::CUSOLVER_STATUS_NOT_INITIALIZED;
using ::CUSOLVER_STATUS_NOT_SUPPORTED;
using ::CUSOLVER_STATUS_SUCCESS;
using ::CUSOLVER_STATUS_ZERO_PIVOT;

// ========================================================================
// Enumerations
// ========================================================================

// Grid mapping: controls how 1D device IDs map to a 2D grid
using ::CUDALIBMG_GRID_MAPPING_COL_MAJOR;
using ::CUDALIBMG_GRID_MAPPING_ROW_MAJOR;
using ::cusolverMgGridMapping_t;

// Eigenvalue computation mode (from cusolverDn.h, used by cusolverMgSyevd)
using ::CUSOLVER_EIG_MODE_NOVECTOR;
using ::CUSOLVER_EIG_MODE_VECTOR;
using ::cusolverEigMode_t;

// Data type (from CUDA runtime, used by descriptor and solver functions)
using ::cudaDataType;

// ========================================================================
// Handle Management Functions
// ========================================================================
using ::cusolverMgCreate;
using ::cusolverMgDestroy;

// ========================================================================
// Device Selection
// ========================================================================
using ::cusolverMgDeviceSelect;

// ========================================================================
// Grid Management Functions
// ========================================================================
using ::cusolverMgCreateDeviceGrid;
using ::cusolverMgDestroyGrid;

// ========================================================================
// Matrix Descriptor Management Functions
// ========================================================================
using ::cusolverMgCreateMatrixDesc;
using ::cusolverMgDestroyMatrixDesc;

// ========================================================================
// Symmetric Eigenvalue Problem (SYEVD) - Multi-GPU
// ========================================================================
using ::cusolverMgSyevd;
using ::cusolverMgSyevd_bufferSize;

// ========================================================================
// LU Factorization (GETRF) - Multi-GPU
// ========================================================================
using ::cusolverMgGetrf;
using ::cusolverMgGetrf_bufferSize;

// ========================================================================
// LU Solve (GETRS) - Multi-GPU
// ========================================================================
using ::cusolverMgGetrs;
using ::cusolverMgGetrs_bufferSize;

// ========================================================================
// Cholesky Factorization (POTRF) - Multi-GPU
// ========================================================================
using ::cusolverMgPotrf;
using ::cusolverMgPotrf_bufferSize;

// ========================================================================
// Cholesky Solve (POTRS) - Multi-GPU
// ========================================================================
using ::cusolverMgPotrs;
using ::cusolverMgPotrs_bufferSize;

// ========================================================================
// Cholesky Inverse (POTRI) - Multi-GPU
// ========================================================================
using ::cusolverMgPotri;
using ::cusolverMgPotri_bufferSize;

} // namespace gpumod::cuda
