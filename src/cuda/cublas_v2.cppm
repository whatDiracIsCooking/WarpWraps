/**
 * @file interface.cppm
 * @brief Primary interface for wwr.cuda.cublas_v2
 *
 * This module wraps the native cuBLAS API and exports types, constants,
 * and functions for cuBLAS library management and operations.
 *
 * Usage:
 *   import wwr.cuda.cublas_v2;
 */

module;

#include <cublas_v2.h>
#include <cuda_runtime.h>

// Undefine cuBLAS macros that redirect non-_v2 names to _v2 versions
// so we can define our own wrapper functions
#undef cublasCreate
#undef cublasDestroy
#undef cublasGetVersion
#undef cublasSetStream
#undef cublasGetStream
#undef cublasSetPointerMode
#undef cublasGetPointerMode

export module wwr.cuda.cublas_v2;

import std;

// ========================================================================
// Export all cuBLAS types and functions in wwr namespace
// ========================================================================

export namespace wwr::cuda {

// ========================================================================
// Core Types
// ========================================================================
using ::cublasHandle_t;
using ::cublasStatus_t;
using ::cudaStream_t;

// Status enum values
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

// ========================================================================
// Enumerations
// ========================================================================
using ::cublasFillMode_t;
// cublasFillMode_t enum values
using ::CUBLAS_FILL_MODE_FULL;
using ::CUBLAS_FILL_MODE_LOWER;
using ::CUBLAS_FILL_MODE_UPPER;

using ::cublasDiagType_t;
// cublasDiagType_t enum values
using ::CUBLAS_DIAG_NON_UNIT;
using ::CUBLAS_DIAG_UNIT;

using ::cublasSideMode_t;
// cublasSideMode_t enum values
using ::CUBLAS_SIDE_LEFT;
using ::CUBLAS_SIDE_RIGHT;

using ::cublasOperation_t;
// cublasOperation_t enum values
using ::CUBLAS_OP_C;
using ::CUBLAS_OP_CONJG;
using ::CUBLAS_OP_HERMITAN;
using ::CUBLAS_OP_N;
using ::CUBLAS_OP_T;

using ::cublasPointerMode_t;
// cublasPointerMode_t enum values
using ::CUBLAS_POINTER_MODE_DEVICE;
using ::CUBLAS_POINTER_MODE_HOST;

using ::cublasAtomicsMode_t;
// cublasAtomicsMode_t enum values
using ::CUBLAS_ATOMICS_ALLOWED;
using ::CUBLAS_ATOMICS_NOT_ALLOWED;

using ::cublasGemmAlgo_t;
// Common cublasGemmAlgo_t enum values
using ::CUBLAS_GEMM_ALGO0;
using ::CUBLAS_GEMM_DEFAULT;
using ::CUBLAS_GEMM_DEFAULT_TENSOR_OP;

using ::cublasMath_t;
// cublasMath_t enum values
using ::CUBLAS_DEFAULT_MATH;
using ::CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION;
using ::CUBLAS_PEDANTIC_MATH;
using ::CUBLAS_TENSOR_OP_MATH;
using ::CUBLAS_TF32_TENSOR_OP_MATH;

using ::cublasComputeType_t;
using ::cublasDataType_t;
// cublasComputeType_t enum values
using ::CUBLAS_COMPUTE_16F;
using ::CUBLAS_COMPUTE_32F;
using ::CUBLAS_COMPUTE_32F_FAST_16F;
using ::CUBLAS_COMPUTE_32F_FAST_TF32;
using ::CUBLAS_COMPUTE_64F;

using ::cublasEmulationStrategy_t;
// cublasEmulationStrategy_t enum values
using ::CUBLAS_EMULATION_STRATEGY_DEFAULT;
using ::CUBLAS_EMULATION_STRATEGY_EAGER;
using ::CUBLAS_EMULATION_STRATEGY_PERFORMANT;

// Logging callback type
using ::cublasLogCallback;

// ========================================================================
// Helper Functions - Context Management
// ========================================================================
// Export _v2 functions for code that uses them directly
using ::cublasCreate_v2;
using ::cublasDestroy_v2;
using ::cublasGetCudartVersion;
using ::cublasGetProperty;
using ::cublasGetVersion_v2;

// Convenience wrappers without _v2 suffix
cublasStatus_t cublasCreate(cublasHandle_t *handle) {
  return ::cublasCreate_v2(handle);
}

cublasStatus_t cublasDestroy(cublasHandle_t handle) {
  return ::cublasDestroy_v2(handle);
}

cublasStatus_t cublasGetVersion(cublasHandle_t handle, int *version) {
  return ::cublasGetVersion_v2(handle, version);
}

// ========================================================================
// Helper Functions - Stream Management
// ========================================================================
// Export _v2 functions for code that uses them directly
using ::cublasGetStream_v2;
using ::cublasSetStream_v2;

cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t streamId) {
  return ::cublasSetStream_v2(handle, streamId);
}

cublasStatus_t cublasGetStream(cublasHandle_t handle, cudaStream_t *streamId) {
  return ::cublasGetStream_v2(handle, streamId);
}

// ========================================================================
// Helper Functions - Pointer Mode
// ========================================================================
// Export _v2 functions for code that uses them directly
using ::cublasGetPointerMode_v2;
using ::cublasSetPointerMode_v2;

cublasStatus_t cublasSetPointerMode(cublasHandle_t handle, cublasPointerMode_t mode) {
  return ::cublasSetPointerMode_v2(handle, mode);
}

cublasStatus_t cublasGetPointerMode(cublasHandle_t handle, cublasPointerMode_t *mode) {
  return ::cublasGetPointerMode_v2(handle, mode);
}

// ========================================================================
// Helper Functions - Atomics Mode
// ========================================================================
using ::cublasGetAtomicsMode;
using ::cublasSetAtomicsMode;

// ========================================================================
// Helper Functions - Math Mode
// ========================================================================
using ::cublasGetMathMode;
using ::cublasSetMathMode;

// ========================================================================
// Helper Functions - Workspace
// ========================================================================
using ::cublasSetWorkspace;

// ========================================================================
// Helper Functions - SM Count Target
// ========================================================================
using ::cublasGetSmCountTarget;
using ::cublasSetSmCountTarget;

// ========================================================================
// Helper Functions - Emulation Strategy
// ========================================================================
// Unlike the rest of cublas_api.h, these two are `static inline` rather than
// exported symbols, so they have internal linkage and `using ::name;` is
// ill-formed inside an exported namespace ("cannot be exported"). Forwarding
// wrappers are the same workaround cuComplex.cppm uses for the whole of
// cuComplex.h. They appeared in a CUDA 13.0 patch release -- a toolkit older
// than the one docker/Dockerfile.cuda pins will not have them.
cublasStatus_t cublasSetEmulationStrategy(cublasHandle_t handle, cudaEmulationStrategy strategy) {
  return ::cublasSetEmulationStrategy(handle, strategy);
}

cublasStatus_t cublasGetEmulationStrategy(cublasHandle_t handle, cudaEmulationStrategy *strategy) {
  return ::cublasGetEmulationStrategy(handle, strategy);
}

// ========================================================================
// Helper Functions - Error/Status Strings
// ========================================================================
using ::cublasGetStatusName;
using ::cublasGetStatusString;

// ========================================================================
// Helper Functions - Logging
// ========================================================================
using ::cublasGetLoggerCallback;
using ::cublasLoggerConfigure;
using ::cublasSetLoggerCallback;

// ========================================================================
// Helper Functions - Vector/Matrix Transfers (legacy helpers)
// ========================================================================
using ::cublasGetMatrix;
using ::cublasGetMatrixAsync;
using ::cublasGetVector;
using ::cublasGetVectorAsync;
using ::cublasSetMatrix;
using ::cublasSetMatrixAsync;
using ::cublasSetVector;
using ::cublasSetVectorAsync;

// ========================================================================
// Level 1 BLAS Functions
// ========================================================================
// Export the actual _v2 function names (not the macro aliases)

// Index of maximum/minimum absolute value
using ::cublasIcamax_v2;
using ::cublasIcamax_v2_64;
using ::cublasIcamin_v2;
using ::cublasIcamin_v2_64;
using ::cublasIdamax_v2;
using ::cublasIdamax_v2_64;
using ::cublasIdamin_v2;
using ::cublasIdamin_v2_64;
using ::cublasIsamax_v2;
using ::cublasIsamax_v2_64;
using ::cublasIsamin_v2;
using ::cublasIsamin_v2_64;
using ::cublasIzamax_v2;
using ::cublasIzamax_v2_64;
using ::cublasIzamin_v2;
using ::cublasIzamin_v2_64;

// Sum of absolute values
using ::cublasDasum_v2;
using ::cublasDasum_v2_64;
using ::cublasDzasum_v2;
using ::cublasDzasum_v2_64;
using ::cublasSasum_v2;
using ::cublasSasum_v2_64;
using ::cublasScasum_v2;
using ::cublasScasum_v2_64;

// Scalar-vector product and sum
using ::cublasCaxpy_v2;
using ::cublasCaxpy_v2_64;
using ::cublasDaxpy_v2;
using ::cublasDaxpy_v2_64;
using ::cublasSaxpy_v2;
using ::cublasSaxpy_v2_64;
using ::cublasZaxpy_v2;
using ::cublasZaxpy_v2_64;

// Vector copy
using ::cublasCcopy_v2;
using ::cublasCcopy_v2_64;
using ::cublasDcopy_v2;
using ::cublasDcopy_v2_64;
using ::cublasScopy_v2;
using ::cublasScopy_v2_64;
using ::cublasZcopy_v2;
using ::cublasZcopy_v2_64;

// Dot product
using ::cublasCdotc_v2;
using ::cublasCdotc_v2_64;
using ::cublasCdotu_v2;
using ::cublasCdotu_v2_64;
using ::cublasDdot_v2;
using ::cublasDdot_v2_64;
using ::cublasSdot_v2;
using ::cublasSdot_v2_64;
using ::cublasZdotc_v2;
using ::cublasZdotc_v2_64;
using ::cublasZdotu_v2;
using ::cublasZdotu_v2_64;

// Euclidean norm
using ::cublasDnrm2_v2;
using ::cublasDnrm2_v2_64;
using ::cublasDznrm2_v2;
using ::cublasDznrm2_v2_64;
using ::cublasScnrm2_v2;
using ::cublasScnrm2_v2_64;
using ::cublasSnrm2_v2;
using ::cublasSnrm2_v2_64;

// Givens rotation
using ::cublasCrot_v2;
using ::cublasCrot_v2_64;
using ::cublasCrotg_v2;
using ::cublasCsrot_v2;
using ::cublasCsrot_v2_64;
using ::cublasDrot_v2;
using ::cublasDrot_v2_64;
using ::cublasDrotg_v2;
using ::cublasSrot_v2;
using ::cublasSrot_v2_64;
using ::cublasSrotg_v2;
using ::cublasZdrot_v2;
using ::cublasZdrot_v2_64;
using ::cublasZrot_v2;
using ::cublasZrot_v2_64;
using ::cublasZrotg_v2;

// Modified Givens rotation
using ::cublasDrotm_v2;
using ::cublasDrotm_v2_64;
using ::cublasDrotmg_v2;
using ::cublasSrotm_v2;
using ::cublasSrotm_v2_64;
using ::cublasSrotmg_v2;

// Scalar multiplication
using ::cublasCscal_v2;
using ::cublasCscal_v2_64;
using ::cublasCsscal_v2;
using ::cublasCsscal_v2_64;
using ::cublasDscal_v2;
using ::cublasDscal_v2_64;
using ::cublasSscal_v2;
using ::cublasSscal_v2_64;
using ::cublasZdscal_v2;
using ::cublasZdscal_v2_64;
using ::cublasZscal_v2;
using ::cublasZscal_v2_64;

// Vector swap
using ::cublasCswap_v2;
using ::cublasCswap_v2_64;
using ::cublasDswap_v2;
using ::cublasDswap_v2_64;
using ::cublasSswap_v2;
using ::cublasSswap_v2_64;
using ::cublasZswap_v2;
using ::cublasZswap_v2_64;

// ========================================================================
// Level 2 BLAS Functions
// ========================================================================

// General matrix-vector multiplication
using ::cublasCgemv_v2;
using ::cublasCgemv_v2_64;
using ::cublasDgemv_v2;
using ::cublasDgemv_v2_64;
using ::cublasSgemv_v2;
using ::cublasSgemv_v2_64;
using ::cublasZgemv_v2;
using ::cublasZgemv_v2_64;

// General banded matrix-vector multiplication
using ::cublasCgbmv_v2;
using ::cublasCgbmv_v2_64;
using ::cublasDgbmv_v2;
using ::cublasDgbmv_v2_64;
using ::cublasSgbmv_v2;
using ::cublasSgbmv_v2_64;
using ::cublasZgbmv_v2;
using ::cublasZgbmv_v2_64;

// General rank-1 update
using ::cublasCgerc_v2;
using ::cublasCgerc_v2_64;
using ::cublasCgeru_v2;
using ::cublasCgeru_v2_64;
using ::cublasDger_v2;
using ::cublasDger_v2_64;
using ::cublasSger_v2;
using ::cublasSger_v2_64;
using ::cublasZgerc_v2;
using ::cublasZgerc_v2_64;
using ::cublasZgeru_v2;
using ::cublasZgeru_v2_64;

// Symmetric matrix-vector multiplication
using ::cublasDsymv_v2;
using ::cublasDsymv_v2_64;
using ::cublasSsymv_v2;
using ::cublasSsymv_v2_64;

// Symmetric rank-1 update
using ::cublasDsyr_v2;
using ::cublasDsyr_v2_64;
using ::cublasSsyr_v2;
using ::cublasSsyr_v2_64;

// Symmetric rank-2 update
using ::cublasDsyr2_v2;
using ::cublasDsyr2_v2_64;
using ::cublasSsyr2_v2;
using ::cublasSsyr2_v2_64;

// Symmetric banded matrix-vector multiplication
using ::cublasDsbmv_v2;
using ::cublasDsbmv_v2_64;
using ::cublasSsbmv_v2;
using ::cublasSsbmv_v2_64;

// Symmetric packed matrix-vector multiplication
using ::cublasDspmv_v2;
using ::cublasDspmv_v2_64;
using ::cublasSspmv_v2;
using ::cublasSspmv_v2_64;

// Symmetric packed rank-1 update
using ::cublasDspr_v2;
using ::cublasDspr_v2_64;
using ::cublasSspr_v2;
using ::cublasSspr_v2_64;

// Symmetric packed rank-2 update
using ::cublasDspr2_v2;
using ::cublasDspr2_v2_64;
using ::cublasSspr2_v2;
using ::cublasSspr2_v2_64;

// Triangular matrix-vector multiplication
using ::cublasCtrmv_v2;
using ::cublasCtrmv_v2_64;
using ::cublasDtrmv_v2;
using ::cublasDtrmv_v2_64;
using ::cublasStrmv_v2;
using ::cublasStrmv_v2_64;
using ::cublasZtrmv_v2;
using ::cublasZtrmv_v2_64;

// Triangular solve
using ::cublasCtrsv_v2;
using ::cublasCtrsv_v2_64;
using ::cublasDtrsv_v2;
using ::cublasDtrsv_v2_64;
using ::cublasStrsv_v2;
using ::cublasStrsv_v2_64;
using ::cublasZtrsv_v2;
using ::cublasZtrsv_v2_64;

// Triangular banded matrix-vector multiplication
using ::cublasCtbmv_v2;
using ::cublasCtbmv_v2_64;
using ::cublasDtbmv_v2;
using ::cublasDtbmv_v2_64;
using ::cublasStbmv_v2;
using ::cublasStbmv_v2_64;
using ::cublasZtbmv_v2;
using ::cublasZtbmv_v2_64;

// Triangular banded solve
using ::cublasCtbsv_v2;
using ::cublasCtbsv_v2_64;
using ::cublasDtbsv_v2;
using ::cublasDtbsv_v2_64;
using ::cublasStbsv_v2;
using ::cublasStbsv_v2_64;
using ::cublasZtbsv_v2;
using ::cublasZtbsv_v2_64;

// Triangular packed matrix-vector multiplication
using ::cublasCtpmv_v2;
using ::cublasCtpmv_v2_64;
using ::cublasDtpmv_v2;
using ::cublasDtpmv_v2_64;
using ::cublasStpmv_v2;
using ::cublasStpmv_v2_64;
using ::cublasZtpmv_v2;
using ::cublasZtpmv_v2_64;

// Triangular packed solve
using ::cublasCtpsv_v2;
using ::cublasCtpsv_v2_64;
using ::cublasDtpsv_v2;
using ::cublasDtpsv_v2_64;
using ::cublasStpsv_v2;
using ::cublasStpsv_v2_64;
using ::cublasZtpsv_v2;
using ::cublasZtpsv_v2_64;

// Hermitian matrix-vector multiplication
using ::cublasChemv_v2;
using ::cublasChemv_v2_64;
using ::cublasZhemv_v2;
using ::cublasZhemv_v2_64;

// Hermitian banded matrix-vector multiplication
using ::cublasChbmv_v2;
using ::cublasChbmv_v2_64;
using ::cublasZhbmv_v2;
using ::cublasZhbmv_v2_64;

// Hermitian packed matrix-vector multiplication
using ::cublasChpmv_v2;
using ::cublasChpmv_v2_64;
using ::cublasZhpmv_v2;
using ::cublasZhpmv_v2_64;

// Hermitian rank-1 update
using ::cublasCher_v2;
using ::cublasCher_v2_64;
using ::cublasZher_v2;
using ::cublasZher_v2_64;

// Hermitian rank-2 update
using ::cublasCher2_v2;
using ::cublasCher2_v2_64;
using ::cublasZher2_v2;
using ::cublasZher2_v2_64;

// Hermitian packed rank-1 update
using ::cublasChpr_v2;
using ::cublasChpr_v2_64;
using ::cublasZhpr_v2;
using ::cublasZhpr_v2_64;

// Hermitian packed rank-2 update
using ::cublasChpr2_v2;
using ::cublasChpr2_v2_64;
using ::cublasZhpr2_v2;
using ::cublasZhpr2_v2_64;

// Batched general matrix-vector multiplication
using ::cublasCgemvBatched;
using ::cublasCgemvBatched_64;
using ::cublasDgemvBatched;
using ::cublasDgemvBatched_64;
using ::cublasSgemvBatched;
using ::cublasSgemvBatched_64;
using ::cublasZgemvBatched;
using ::cublasZgemvBatched_64;

// Strided batched general matrix-vector multiplication
using ::cublasCgemvStridedBatched;
using ::cublasCgemvStridedBatched_64;
using ::cublasDgemvStridedBatched;
using ::cublasDgemvStridedBatched_64;
using ::cublasSgemvStridedBatched;
using ::cublasSgemvStridedBatched_64;
using ::cublasZgemvStridedBatched;
using ::cublasZgemvStridedBatched_64;

// Level 3 BLAS Functions
// ========================================================================

// General matrix-matrix multiplication
using ::cublasCgemm_v2;
using ::cublasCgemm_v2_64;
using ::cublasDgemm_v2;
using ::cublasDgemm_v2_64;
using ::cublasSgemm_v2;
using ::cublasSgemm_v2_64;
using ::cublasZgemm_v2;
using ::cublasZgemm_v2_64;

// GEMM with 3m algorithm (complex only)
using ::cublasCgemm3m;
using ::cublasCgemm3m_64;
using ::cublasZgemm3m;
using ::cublasZgemm3m_64;

// Batched GEMM
using ::cublasCgemmBatched;
using ::cublasCgemmBatched_64;
using ::cublasDgemmBatched;
using ::cublasDgemmBatched_64;
using ::cublasSgemmBatched;
using ::cublasSgemmBatched_64;
using ::cublasZgemmBatched;
using ::cublasZgemmBatched_64;

// Strided batched GEMM
using ::cublasCgemmStridedBatched;
using ::cublasCgemmStridedBatched_64;
using ::cublasDgemmStridedBatched;
using ::cublasDgemmStridedBatched_64;
using ::cublasSgemmStridedBatched;
using ::cublasSgemmStridedBatched_64;
using ::cublasZgemmStridedBatched;
using ::cublasZgemmStridedBatched_64;

// Grouped batched GEMM (real types only in CUDA 13.0)
using ::cublasDgemmGroupedBatched;
using ::cublasDgemmGroupedBatched_64;
using ::cublasSgemmGroupedBatched;
using ::cublasSgemmGroupedBatched_64;

// Symmetric matrix-matrix multiplication
using ::cublasCsymm_v2;
using ::cublasCsymm_v2_64;
using ::cublasDsymm_v2;
using ::cublasDsymm_v2_64;
using ::cublasSsymm_v2;
using ::cublasSsymm_v2_64;
using ::cublasZsymm_v2;
using ::cublasZsymm_v2_64;

// Symmetric rank-k update
using ::cublasCsyrk_v2;
using ::cublasCsyrk_v2_64;
using ::cublasDsyrk_v2;
using ::cublasDsyrk_v2_64;
using ::cublasSsyrk_v2;
using ::cublasSsyrk_v2_64;
using ::cublasZsyrk_v2;
using ::cublasZsyrk_v2_64;

// Symmetric rank-2k update
using ::cublasCsyr2k_v2;
using ::cublasCsyr2k_v2_64;
using ::cublasDsyr2k_v2;
using ::cublasDsyr2k_v2_64;
using ::cublasSsyr2k_v2;
using ::cublasSsyr2k_v2_64;
using ::cublasZsyr2k_v2;
using ::cublasZsyr2k_v2_64;

// Symmetric rank-k update variant
using ::cublasCsyrkx;
using ::cublasCsyrkx_64;
using ::cublasDsyrkx;
using ::cublasDsyrkx_64;
using ::cublasSsyrkx;
using ::cublasSsyrkx_64;
using ::cublasZsyrkx;
using ::cublasZsyrkx_64;

// Triangular matrix-matrix multiplication
using ::cublasCtrmm_v2;
using ::cublasCtrmm_v2_64;
using ::cublasDtrmm_v2;
using ::cublasDtrmm_v2_64;
using ::cublasStrmm_v2;
using ::cublasStrmm_v2_64;
using ::cublasZtrmm_v2;
using ::cublasZtrmm_v2_64;

// Triangular solve with multiple right-hand sides
using ::cublasCtrsm_v2;
using ::cublasCtrsm_v2_64;
using ::cublasDtrsm_v2;
using ::cublasDtrsm_v2_64;
using ::cublasStrsm_v2;
using ::cublasStrsm_v2_64;
using ::cublasZtrsm_v2;
using ::cublasZtrsm_v2_64;

// Batched triangular solve
using ::cublasCtrsmBatched;
using ::cublasCtrsmBatched_64;
using ::cublasDtrsmBatched;
using ::cublasDtrsmBatched_64;
using ::cublasStrsmBatched;
using ::cublasStrsmBatched_64;
using ::cublasZtrsmBatched;
using ::cublasZtrsmBatched_64;

// Hermitian matrix-matrix multiplication (complex only)
using ::cublasChemm_v2;
using ::cublasChemm_v2_64;
using ::cublasZhemm_v2;
using ::cublasZhemm_v2_64;

// Hermitian rank-k update (complex only)
using ::cublasCherk_v2;
using ::cublasCherk_v2_64;
using ::cublasZherk_v2;
using ::cublasZherk_v2_64;

// Hermitian rank-2k update (complex only)
using ::cublasCher2k_v2;
using ::cublasCher2k_v2_64;
using ::cublasZher2k_v2;
using ::cublasZher2k_v2_64;

// Hermitian rank-k update variant (complex only)
using ::cublasCherkx;
using ::cublasCherkx_64;
using ::cublasZherkx;
using ::cublasZherkx_64;

// ========================================================================
// BLAS-like Extension Functions
// ========================================================================

// Matrix addition/transposition
using ::cublasCgeam;
using ::cublasCgeam_64;
using ::cublasDgeam;
using ::cublasDgeam_64;
using ::cublasSgeam;
using ::cublasSgeam_64;
using ::cublasZgeam;
using ::cublasZgeam_64;

// Diagonal matrix multiplication
using ::cublasCdgmm;
using ::cublasCdgmm_64;
using ::cublasDdgmm;
using ::cublasDdgmm_64;
using ::cublasSdgmm;
using ::cublasSdgmm_64;
using ::cublasZdgmm;
using ::cublasZdgmm_64;

// Batched LU factorization
using ::cublasCgetrfBatched;
using ::cublasDgetrfBatched;
using ::cublasSgetrfBatched;
using ::cublasZgetrfBatched;

// Batched LU solver
using ::cublasCgetrsBatched;
using ::cublasDgetrsBatched;
using ::cublasSgetrsBatched;
using ::cublasZgetrsBatched;

// Batched matrix inversion
using ::cublasCgetriBatched;
using ::cublasDgetriBatched;
using ::cublasSgetriBatched;
using ::cublasZgetriBatched;

// Batched matrix inversion (small matrices)
using ::cublasCmatinvBatched;
using ::cublasDmatinvBatched;
using ::cublasSmatinvBatched;
using ::cublasZmatinvBatched;

// Batched QR factorization
using ::cublasCgeqrfBatched;
using ::cublasDgeqrfBatched;
using ::cublasSgeqrfBatched;
using ::cublasZgeqrfBatched;

// Batched least-squares solver
using ::cublasCgelsBatched;
using ::cublasDgelsBatched;
using ::cublasSgelsBatched;
using ::cublasZgelsBatched;

// Triangular packed to regular format
using ::cublasCtpttr;
using ::cublasDtpttr;
using ::cublasStpttr;
using ::cublasZtpttr;

// Triangular regular to packed format
using ::cublasCtrttp;
using ::cublasDtrttp;
using ::cublasStrttp;
using ::cublasZtrttp;

} // namespace wwr::cuda
