/**
 * @file cublasXt.cppm
 * @brief Primary interface for gpumod.cuda.cublasXt
 *
 * This module wraps the native cuBLASXt API and exports types, constants,
 * and functions for the cuBLASXt multi-GPU BLAS extension library.
 *
 * Usage:
 *   import gpumod.cuda.cublasXt;
 */

module;

#include <cublasXt.h>

export module gpumod.cuda.cublasXt;

// ========================================================================
// Export all cuBLASXt types and functions in gpumod namespace
// ========================================================================

export namespace gpumod::cuda {

// ========================================================================
// Opaque handle type
// ========================================================================
using ::cublasXtHandle_t;

// ========================================================================
// Dependent types from cublas_v2.h / cublas_api.h that appear in cublasXt
// function signatures
// ========================================================================

// Status / result type (from cublas_api.h via cublas_v2.h)
using ::CUBLAS_STATUS_ALLOC_FAILED;
using ::CUBLAS_STATUS_ARCH_MISMATCH;
using ::CUBLAS_STATUS_EXECUTION_FAILED;
using ::CUBLAS_STATUS_INTERNAL_ERROR;
using ::CUBLAS_STATUS_INVALID_VALUE;
using ::CUBLAS_STATUS_LICENSE_ERROR;
using ::CUBLAS_STATUS_MAPPING_ERROR;
using ::CUBLAS_STATUS_NOT_INITIALIZED;
using ::CUBLAS_STATUS_NOT_SUPPORTED;
using ::CUBLAS_STATUS_SUCCESS;
using ::cublasStatus_t;

// Matrix operation enum
using ::CUBLAS_OP_C;
using ::CUBLAS_OP_CONJG;
using ::CUBLAS_OP_HERMITAN;
using ::CUBLAS_OP_N;
using ::CUBLAS_OP_T;
using ::cublasOperation_t;

// Fill mode enum
using ::CUBLAS_FILL_MODE_FULL;
using ::CUBLAS_FILL_MODE_LOWER;
using ::CUBLAS_FILL_MODE_UPPER;
using ::cublasFillMode_t;

// Diagonal type enum
using ::CUBLAS_DIAG_NON_UNIT;
using ::CUBLAS_DIAG_UNIT;
using ::cublasDiagType_t;

// Side mode enum
using ::CUBLAS_SIDE_LEFT;
using ::CUBLAS_SIDE_RIGHT;
using ::cublasSideMode_t;

// Complex scalar types (from cuComplex.h)
using ::cuComplex;
using ::cuDoubleComplex;

// ========================================================================
// Pinned memory mode enum
// ========================================================================
using ::CUBLASXT_PINNING_DISABLED;
using ::CUBLASXT_PINNING_ENABLED;
using ::cublasXtPinnedMemMode_t;

// ========================================================================
// CPU BLAS operation type enum
// ========================================================================
using ::CUBLASXT_COMPLEX;
using ::CUBLASXT_DOUBLE;
using ::CUBLASXT_DOUBLECOMPLEX;
using ::CUBLASXT_FLOAT;
using ::cublasXtOpType_t;

// ========================================================================
// BLAS routine identifier enum
// ========================================================================
using ::CUBLASXT_GEMM;
using ::CUBLASXT_HEMM;
using ::CUBLASXT_HER2K;
using ::CUBLASXT_HERK;
using ::CUBLASXT_HERKX;
using ::CUBLASXT_ROUTINE_MAX;
using ::CUBLASXT_SPMM;
using ::CUBLASXT_SYMM;
using ::CUBLASXT_SYR2K;
using ::CUBLASXT_SYRK;
using ::CUBLASXT_SYRKX;
using ::CUBLASXT_TRMM;
using ::CUBLASXT_TRSM;
using ::cublasXtBlasOp_t;

// ========================================================================
// Context management functions
// ========================================================================
using ::cublasXtCreate;
using ::cublasXtDestroy;

// ========================================================================
// Device selection and configuration functions
// ========================================================================
using ::cublasXtDeviceSelect;
using ::cublasXtGetBlockDim;
using ::cublasXtGetNumBoards;
using ::cublasXtMaxBoards;
using ::cublasXtSetBlockDim;

// ========================================================================
// Pinned memory mode functions
// ========================================================================
using ::cublasXtGetPinningMemMode;
using ::cublasXtSetPinningMemMode;

// ========================================================================
// CPU BLAS offload functions
// ========================================================================
using ::cublasXtSetCpuRatio;
using ::cublasXtSetCpuRoutine;

// ========================================================================
// GEMM functions (General Matrix-Matrix multiply)
// ========================================================================
using ::cublasXtCgemm;
using ::cublasXtDgemm;
using ::cublasXtSgemm;
using ::cublasXtZgemm;

// ========================================================================
// SYRK functions (Symmetric Rank-K update)
// ========================================================================
using ::cublasXtCsyrk;
using ::cublasXtDsyrk;
using ::cublasXtSsyrk;
using ::cublasXtZsyrk;

// ========================================================================
// HERK functions (Hermitian Rank-K update)
// ========================================================================
using ::cublasXtCherk;
using ::cublasXtZherk;

// ========================================================================
// SYR2K functions (Symmetric Rank-2K update)
// ========================================================================
using ::cublasXtCsyr2k;
using ::cublasXtDsyr2k;
using ::cublasXtSsyr2k;
using ::cublasXtZsyr2k;

// ========================================================================
// HERKX functions (Hermitian Rank-K variant extension)
// ========================================================================
using ::cublasXtCherkx;
using ::cublasXtZherkx;

// ========================================================================
// TRSM functions (Triangular Solve with Multiple RHS)
// ========================================================================
using ::cublasXtCtrsm;
using ::cublasXtDtrsm;
using ::cublasXtStrsm;
using ::cublasXtZtrsm;

// ========================================================================
// SYMM functions (Symmetric Matrix-Matrix multiply)
// ========================================================================
using ::cublasXtCsymm;
using ::cublasXtDsymm;
using ::cublasXtSsymm;
using ::cublasXtZsymm;

// ========================================================================
// HEMM functions (Hermitian Matrix-Matrix multiply)
// ========================================================================
using ::cublasXtChemm;
using ::cublasXtZhemm;

// ========================================================================
// SYRKX functions (Symmetric Rank-K variant extension)
// ========================================================================
using ::cublasXtCsyrkx;
using ::cublasXtDsyrkx;
using ::cublasXtSsyrkx;
using ::cublasXtZsyrkx;

// ========================================================================
// HER2K functions (Hermitian Rank-2K update)
// ========================================================================
using ::cublasXtCher2k;
using ::cublasXtZher2k;

// ========================================================================
// SPMM functions (Symmetric Packed Matrix-Matrix multiply)
// ========================================================================
using ::cublasXtCspmm;
using ::cublasXtDspmm;
using ::cublasXtSspmm;
using ::cublasXtZspmm;

// ========================================================================
// TRMM functions (Triangular Matrix-Matrix multiply)
// ========================================================================
using ::cublasXtCtrmm;
using ::cublasXtDtrmm;
using ::cublasXtStrmm;
using ::cublasXtZtrmm;

} // namespace gpumod::cuda
