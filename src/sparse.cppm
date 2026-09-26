/**
 * @file sparse.cppm
 * @brief Backend-neutral sparse: gpusparse* names for cuSPARSE / hipSPARSE
 *
 * gpusparse<X><name> stands for cusparse<X><name> on a CUDA build and
 * hipsparse<X><name> on a HIP build. See gpu_backend.h.
 *
 * Only the names src/wrappers/sparse uses are listed, plus the handle, stream,
 * pointer-mode, error-string and matrix-descriptor helpers a caller needs. This
 * layer covers the legacy typed (S/D/C/Z) functions the two backends still share
 * and that cuSPARSE has not deprecated: the BSR matrix-vector multiply, the
 * tridiagonal/pentadiagonal batch solvers (gtsv2 / gpsvInterleavedBatch), CSR
 * matrix addition (csrgeam2) and a few format conversions (nnz, gebsr2gebsc,
 * csr2gebsr). The modern generic API (SpMV, SpMM, SpGEMM, ...) is not wrapped
 * here -- its element type is a runtime cudaDataType/hipDataType argument rather
 * than a name letter, so it needs no S/D/C/Z dispatch; reach it through
 * gpumod.cuda.cusparse / gpumod.hip.hipsparse.
 *
 * cuSPARSE-only functions (the Preview SpMMOp API, CreateSlicedEll) and the
 * legacy typed functions cuSPARSE removed but hipSPARSE keeps (csrmv, csrsv2,
 * csrmm, csrsm2, csric02, csrilu02, gemvi, the HYB path, ...) are absent -- this
 * module adapts what both vendors offer, same policy as gpu.blas / gpu.solver.
 *
 * Usage:
 *   import gpumod.sparse;
 *
 *   gpusparseHandle_t handle;
 *   gpusparseCreate(&handle);
 */

module;

#include "gpu_backend.h"

export module gpumod.sparse;

import std;
import gpumod.complex;
#if defined(GPUMOD_GPU_BACKEND_CUDA)
import gpumod.cuda.cusparse;
#else
import gpumod.hip.hipsparse;
#endif

export namespace gpumod {

// ========================================================================
// Types
// ========================================================================

GPUMOD_TYPE(gpusparseHandle_t, cusparseHandle_t, hipsparseHandle_t)
GPUMOD_TYPE(gpusparseStatus_t, cusparseStatus_t, hipsparseStatus_t)
GPUMOD_TYPE(gpusparseMatDescr_t, cusparseMatDescr_t, hipsparseMatDescr_t)
GPUMOD_TYPE(gpusparseOperation_t, cusparseOperation_t, hipsparseOperation_t)
GPUMOD_TYPE(gpusparseDirection_t, cusparseDirection_t, hipsparseDirection_t)
GPUMOD_TYPE(gpusparseAction_t, cusparseAction_t, hipsparseAction_t)
GPUMOD_TYPE(gpusparseIndexBase_t, cusparseIndexBase_t, hipsparseIndexBase_t)
GPUMOD_TYPE(gpusparseMatrixType_t, cusparseMatrixType_t, hipsparseMatrixType_t)
GPUMOD_TYPE(gpusparseFillMode_t, cusparseFillMode_t, hipsparseFillMode_t)
GPUMOD_TYPE(gpusparseDiagType_t, cusparseDiagType_t, hipsparseDiagType_t)
GPUMOD_TYPE(gpusparsePointerMode_t, cusparsePointerMode_t, hipsparsePointerMode_t)

// ========================================================================
// Constants
// ========================================================================

GPUMOD_VALUE(GPUSPARSE_STATUS_SUCCESS, CUSPARSE_STATUS_SUCCESS, HIPSPARSE_STATUS_SUCCESS)

GPUMOD_VALUE(GPUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
             HIPSPARSE_OPERATION_NON_TRANSPOSE)
GPUMOD_VALUE(GPUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
             HIPSPARSE_OPERATION_TRANSPOSE)
GPUMOD_VALUE(GPUSPARSE_OPERATION_CONJUGATE_TRANSPOSE, CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE,
             HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE)

GPUMOD_VALUE(GPUSPARSE_DIRECTION_ROW, CUSPARSE_DIRECTION_ROW, HIPSPARSE_DIRECTION_ROW)
GPUMOD_VALUE(GPUSPARSE_DIRECTION_COLUMN, CUSPARSE_DIRECTION_COLUMN, HIPSPARSE_DIRECTION_COLUMN)

GPUMOD_VALUE(GPUSPARSE_ACTION_SYMBOLIC, CUSPARSE_ACTION_SYMBOLIC, HIPSPARSE_ACTION_SYMBOLIC)
GPUMOD_VALUE(GPUSPARSE_ACTION_NUMERIC, CUSPARSE_ACTION_NUMERIC, HIPSPARSE_ACTION_NUMERIC)

GPUMOD_VALUE(GPUSPARSE_INDEX_BASE_ZERO, CUSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ZERO)
GPUMOD_VALUE(GPUSPARSE_INDEX_BASE_ONE, CUSPARSE_INDEX_BASE_ONE, HIPSPARSE_INDEX_BASE_ONE)

GPUMOD_VALUE(GPUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_MATRIX_TYPE_GENERAL,
             HIPSPARSE_MATRIX_TYPE_GENERAL)
GPUMOD_VALUE(GPUSPARSE_MATRIX_TYPE_SYMMETRIC, CUSPARSE_MATRIX_TYPE_SYMMETRIC,
             HIPSPARSE_MATRIX_TYPE_SYMMETRIC)
GPUMOD_VALUE(GPUSPARSE_MATRIX_TYPE_HERMITIAN, CUSPARSE_MATRIX_TYPE_HERMITIAN,
             HIPSPARSE_MATRIX_TYPE_HERMITIAN)
GPUMOD_VALUE(GPUSPARSE_MATRIX_TYPE_TRIANGULAR, CUSPARSE_MATRIX_TYPE_TRIANGULAR,
             HIPSPARSE_MATRIX_TYPE_TRIANGULAR)

GPUMOD_VALUE(GPUSPARSE_FILL_MODE_LOWER, CUSPARSE_FILL_MODE_LOWER, HIPSPARSE_FILL_MODE_LOWER)
GPUMOD_VALUE(GPUSPARSE_FILL_MODE_UPPER, CUSPARSE_FILL_MODE_UPPER, HIPSPARSE_FILL_MODE_UPPER)

GPUMOD_VALUE(GPUSPARSE_DIAG_TYPE_NON_UNIT, CUSPARSE_DIAG_TYPE_NON_UNIT,
             HIPSPARSE_DIAG_TYPE_NON_UNIT)
GPUMOD_VALUE(GPUSPARSE_DIAG_TYPE_UNIT, CUSPARSE_DIAG_TYPE_UNIT, HIPSPARSE_DIAG_TYPE_UNIT)

GPUMOD_VALUE(GPUSPARSE_POINTER_MODE_HOST, CUSPARSE_POINTER_MODE_HOST, HIPSPARSE_POINTER_MODE_HOST)
GPUMOD_VALUE(GPUSPARSE_POINTER_MODE_DEVICE, CUSPARSE_POINTER_MODE_DEVICE,
             HIPSPARSE_POINTER_MODE_DEVICE)

// ========================================================================
// Handle, stream, pointer mode, error strings
// ========================================================================

GPUMOD_FUNCTION(gpusparseCreate, cusparseCreate, hipsparseCreate)
GPUMOD_FUNCTION(gpusparseDestroy, cusparseDestroy, hipsparseDestroy)
GPUMOD_FUNCTION(gpusparseSetStream, cusparseSetStream, hipsparseSetStream)
GPUMOD_FUNCTION(gpusparseGetStream, cusparseGetStream, hipsparseGetStream)
GPUMOD_FUNCTION(gpusparseSetPointerMode, cusparseSetPointerMode, hipsparseSetPointerMode)
GPUMOD_FUNCTION(gpusparseGetPointerMode, cusparseGetPointerMode, hipsparseGetPointerMode)
GPUMOD_FUNCTION(gpusparseGetErrorName, cusparseGetErrorName, hipsparseGetErrorName)
GPUMOD_FUNCTION(gpusparseGetErrorString, cusparseGetErrorString, hipsparseGetErrorString)

// ========================================================================
// Matrix descriptor (legacy helper routines)
// ========================================================================

GPUMOD_FUNCTION(gpusparseCreateMatDescr, cusparseCreateMatDescr, hipsparseCreateMatDescr)
GPUMOD_FUNCTION(gpusparseDestroyMatDescr, cusparseDestroyMatDescr, hipsparseDestroyMatDescr)
GPUMOD_FUNCTION(gpusparseSetMatType, cusparseSetMatType, hipsparseSetMatType)
GPUMOD_FUNCTION(gpusparseGetMatType, cusparseGetMatType, hipsparseGetMatType)
GPUMOD_FUNCTION(gpusparseSetMatFillMode, cusparseSetMatFillMode, hipsparseSetMatFillMode)
GPUMOD_FUNCTION(gpusparseGetMatFillMode, cusparseGetMatFillMode, hipsparseGetMatFillMode)
GPUMOD_FUNCTION(gpusparseSetMatDiagType, cusparseSetMatDiagType, hipsparseSetMatDiagType)
GPUMOD_FUNCTION(gpusparseGetMatDiagType, cusparseGetMatDiagType, hipsparseGetMatDiagType)
GPUMOD_FUNCTION(gpusparseSetMatIndexBase, cusparseSetMatIndexBase, hipsparseSetMatIndexBase)
GPUMOD_FUNCTION(gpusparseGetMatIndexBase, cusparseGetMatIndexBase, hipsparseGetMatIndexBase)

// ────────────────────────────────────────────────────────────────────────
// Level 2 -- BSR matrix-vector multiply
// ────────────────────────────────────────────────────────────────────────

GPUMOD_FUNCTION(gpusparseSbsrmv, cusparseSbsrmv, hipsparseSbsrmv)
GPUMOD_FUNCTION(gpusparseDbsrmv, cusparseDbsrmv, hipsparseDbsrmv)
GPUMOD_FUNCTION(gpusparseCbsrmv, cusparseCbsrmv, hipsparseCbsrmv)
GPUMOD_FUNCTION(gpusparseZbsrmv, cusparseZbsrmv, hipsparseZbsrmv)

// ────────────────────────────────────────────────────────────────────────
// Tridiagonal / pentadiagonal batch solvers (gtsv2 / gpsvInterleavedBatch)
// ────────────────────────────────────────────────────────────────────────

GPUMOD_FUNCTION(gpusparseSgtsv2_bufferSizeExt, cusparseSgtsv2_bufferSizeExt,
                hipsparseSgtsv2_bufferSizeExt)
GPUMOD_FUNCTION(gpusparseDgtsv2_bufferSizeExt, cusparseDgtsv2_bufferSizeExt,
                hipsparseDgtsv2_bufferSizeExt)
GPUMOD_FUNCTION(gpusparseCgtsv2_bufferSizeExt, cusparseCgtsv2_bufferSizeExt,
                hipsparseCgtsv2_bufferSizeExt)
GPUMOD_FUNCTION(gpusparseZgtsv2_bufferSizeExt, cusparseZgtsv2_bufferSizeExt,
                hipsparseZgtsv2_bufferSizeExt)

GPUMOD_FUNCTION(gpusparseSgtsv2, cusparseSgtsv2, hipsparseSgtsv2)
GPUMOD_FUNCTION(gpusparseDgtsv2, cusparseDgtsv2, hipsparseDgtsv2)
GPUMOD_FUNCTION(gpusparseCgtsv2, cusparseCgtsv2, hipsparseCgtsv2)
GPUMOD_FUNCTION(gpusparseZgtsv2, cusparseZgtsv2, hipsparseZgtsv2)

GPUMOD_FUNCTION(gpusparseSgtsv2_nopivot_bufferSizeExt, cusparseSgtsv2_nopivot_bufferSizeExt,
                hipsparseSgtsv2_nopivot_bufferSizeExt)
GPUMOD_FUNCTION(gpusparseDgtsv2_nopivot_bufferSizeExt, cusparseDgtsv2_nopivot_bufferSizeExt,
                hipsparseDgtsv2_nopivot_bufferSizeExt)
GPUMOD_FUNCTION(gpusparseCgtsv2_nopivot_bufferSizeExt, cusparseCgtsv2_nopivot_bufferSizeExt,
                hipsparseCgtsv2_nopivot_bufferSizeExt)
GPUMOD_FUNCTION(gpusparseZgtsv2_nopivot_bufferSizeExt, cusparseZgtsv2_nopivot_bufferSizeExt,
                hipsparseZgtsv2_nopivot_bufferSizeExt)

GPUMOD_FUNCTION(gpusparseSgtsv2_nopivot, cusparseSgtsv2_nopivot, hipsparseSgtsv2_nopivot)
GPUMOD_FUNCTION(gpusparseDgtsv2_nopivot, cusparseDgtsv2_nopivot, hipsparseDgtsv2_nopivot)
GPUMOD_FUNCTION(gpusparseCgtsv2_nopivot, cusparseCgtsv2_nopivot, hipsparseCgtsv2_nopivot)
GPUMOD_FUNCTION(gpusparseZgtsv2_nopivot, cusparseZgtsv2_nopivot, hipsparseZgtsv2_nopivot)

GPUMOD_FUNCTION(gpusparseSgtsv2StridedBatch_bufferSizeExt, cusparseSgtsv2StridedBatch_bufferSizeExt,
                hipsparseSgtsv2StridedBatch_bufferSizeExt)
GPUMOD_FUNCTION(gpusparseDgtsv2StridedBatch_bufferSizeExt, cusparseDgtsv2StridedBatch_bufferSizeExt,
                hipsparseDgtsv2StridedBatch_bufferSizeExt)
GPUMOD_FUNCTION(gpusparseCgtsv2StridedBatch_bufferSizeExt, cusparseCgtsv2StridedBatch_bufferSizeExt,
                hipsparseCgtsv2StridedBatch_bufferSizeExt)
GPUMOD_FUNCTION(gpusparseZgtsv2StridedBatch_bufferSizeExt, cusparseZgtsv2StridedBatch_bufferSizeExt,
                hipsparseZgtsv2StridedBatch_bufferSizeExt)

GPUMOD_FUNCTION(gpusparseSgtsv2StridedBatch, cusparseSgtsv2StridedBatch,
                hipsparseSgtsv2StridedBatch)
GPUMOD_FUNCTION(gpusparseDgtsv2StridedBatch, cusparseDgtsv2StridedBatch,
                hipsparseDgtsv2StridedBatch)
GPUMOD_FUNCTION(gpusparseCgtsv2StridedBatch, cusparseCgtsv2StridedBatch,
                hipsparseCgtsv2StridedBatch)
GPUMOD_FUNCTION(gpusparseZgtsv2StridedBatch, cusparseZgtsv2StridedBatch,
                hipsparseZgtsv2StridedBatch)

GPUMOD_FUNCTION(gpusparseSgtsvInterleavedBatch_bufferSizeExt,
                cusparseSgtsvInterleavedBatch_bufferSizeExt,
                hipsparseSgtsvInterleavedBatch_bufferSizeExt)
GPUMOD_FUNCTION(gpusparseDgtsvInterleavedBatch_bufferSizeExt,
                cusparseDgtsvInterleavedBatch_bufferSizeExt,
                hipsparseDgtsvInterleavedBatch_bufferSizeExt)
GPUMOD_FUNCTION(gpusparseCgtsvInterleavedBatch_bufferSizeExt,
                cusparseCgtsvInterleavedBatch_bufferSizeExt,
                hipsparseCgtsvInterleavedBatch_bufferSizeExt)
GPUMOD_FUNCTION(gpusparseZgtsvInterleavedBatch_bufferSizeExt,
                cusparseZgtsvInterleavedBatch_bufferSizeExt,
                hipsparseZgtsvInterleavedBatch_bufferSizeExt)

GPUMOD_FUNCTION(gpusparseSgtsvInterleavedBatch, cusparseSgtsvInterleavedBatch,
                hipsparseSgtsvInterleavedBatch)
GPUMOD_FUNCTION(gpusparseDgtsvInterleavedBatch, cusparseDgtsvInterleavedBatch,
                hipsparseDgtsvInterleavedBatch)
GPUMOD_FUNCTION(gpusparseCgtsvInterleavedBatch, cusparseCgtsvInterleavedBatch,
                hipsparseCgtsvInterleavedBatch)
GPUMOD_FUNCTION(gpusparseZgtsvInterleavedBatch, cusparseZgtsvInterleavedBatch,
                hipsparseZgtsvInterleavedBatch)

GPUMOD_FUNCTION(gpusparseSgpsvInterleavedBatch_bufferSizeExt,
                cusparseSgpsvInterleavedBatch_bufferSizeExt,
                hipsparseSgpsvInterleavedBatch_bufferSizeExt)
GPUMOD_FUNCTION(gpusparseDgpsvInterleavedBatch_bufferSizeExt,
                cusparseDgpsvInterleavedBatch_bufferSizeExt,
                hipsparseDgpsvInterleavedBatch_bufferSizeExt)
GPUMOD_FUNCTION(gpusparseCgpsvInterleavedBatch_bufferSizeExt,
                cusparseCgpsvInterleavedBatch_bufferSizeExt,
                hipsparseCgpsvInterleavedBatch_bufferSizeExt)
GPUMOD_FUNCTION(gpusparseZgpsvInterleavedBatch_bufferSizeExt,
                cusparseZgpsvInterleavedBatch_bufferSizeExt,
                hipsparseZgpsvInterleavedBatch_bufferSizeExt)

GPUMOD_FUNCTION(gpusparseSgpsvInterleavedBatch, cusparseSgpsvInterleavedBatch,
                hipsparseSgpsvInterleavedBatch)
GPUMOD_FUNCTION(gpusparseDgpsvInterleavedBatch, cusparseDgpsvInterleavedBatch,
                hipsparseDgpsvInterleavedBatch)
GPUMOD_FUNCTION(gpusparseCgpsvInterleavedBatch, cusparseCgpsvInterleavedBatch,
                hipsparseCgpsvInterleavedBatch)
GPUMOD_FUNCTION(gpusparseZgpsvInterleavedBatch, cusparseZgpsvInterleavedBatch,
                hipsparseZgpsvInterleavedBatch)

// ────────────────────────────────────────────────────────────────────────
// Extra -- CSR matrix addition (csrgeam2)
// ────────────────────────────────────────────────────────────────────────

GPUMOD_FUNCTION(gpusparseScsrgeam2_bufferSizeExt, cusparseScsrgeam2_bufferSizeExt,
                hipsparseScsrgeam2_bufferSizeExt)
GPUMOD_FUNCTION(gpusparseDcsrgeam2_bufferSizeExt, cusparseDcsrgeam2_bufferSizeExt,
                hipsparseDcsrgeam2_bufferSizeExt)
GPUMOD_FUNCTION(gpusparseCcsrgeam2_bufferSizeExt, cusparseCcsrgeam2_bufferSizeExt,
                hipsparseCcsrgeam2_bufferSizeExt)
GPUMOD_FUNCTION(gpusparseZcsrgeam2_bufferSizeExt, cusparseZcsrgeam2_bufferSizeExt,
                hipsparseZcsrgeam2_bufferSizeExt)

GPUMOD_FUNCTION(gpusparseScsrgeam2, cusparseScsrgeam2, hipsparseScsrgeam2)
GPUMOD_FUNCTION(gpusparseDcsrgeam2, cusparseDcsrgeam2, hipsparseDcsrgeam2)
GPUMOD_FUNCTION(gpusparseCcsrgeam2, cusparseCcsrgeam2, hipsparseCcsrgeam2)
GPUMOD_FUNCTION(gpusparseZcsrgeam2, cusparseZcsrgeam2, hipsparseZcsrgeam2)

// ────────────────────────────────────────────────────────────────────────
// Conversion -- nnz, gebsr2gebsc, csr2gebsr
// ────────────────────────────────────────────────────────────────────────

GPUMOD_FUNCTION(gpusparseSnnz, cusparseSnnz, hipsparseSnnz)
GPUMOD_FUNCTION(gpusparseDnnz, cusparseDnnz, hipsparseDnnz)
GPUMOD_FUNCTION(gpusparseCnnz, cusparseCnnz, hipsparseCnnz)
GPUMOD_FUNCTION(gpusparseZnnz, cusparseZnnz, hipsparseZnnz)

GPUMOD_FUNCTION(gpusparseSgebsr2gebsc, cusparseSgebsr2gebsc, hipsparseSgebsr2gebsc)
GPUMOD_FUNCTION(gpusparseDgebsr2gebsc, cusparseDgebsr2gebsc, hipsparseDgebsr2gebsc)
GPUMOD_FUNCTION(gpusparseCgebsr2gebsc, cusparseCgebsr2gebsc, hipsparseCgebsr2gebsc)
GPUMOD_FUNCTION(gpusparseZgebsr2gebsc, cusparseZgebsr2gebsc, hipsparseZgebsr2gebsc)

GPUMOD_FUNCTION(gpusparseScsr2gebsr, cusparseScsr2gebsr, hipsparseScsr2gebsr)
GPUMOD_FUNCTION(gpusparseDcsr2gebsr, cusparseDcsr2gebsr, hipsparseDcsr2gebsr)
GPUMOD_FUNCTION(gpusparseCcsr2gebsr, cusparseCcsr2gebsr, hipsparseCcsr2gebsr)
GPUMOD_FUNCTION(gpusparseZcsr2gebsr, cusparseZcsr2gebsr, hipsparseZcsr2gebsr)

// ────────────────────────────────────────────────────────────────────────
// gebsr2gebsc_bufferSize / csr2gebsr_bufferSize: uniform std::size_t* buffer size
//
// cuSPARSE's *_bufferSize (not the *Ext form) writes the byte count as int*;
// hipSPARSE writes it as std::size_t*. These keep hipSPARSE's std::size_t* signature on
// both backends, forwarding through an int on CUDA -- the mirror of blas's
// getrsBatched const shims. A single GPUMOD_FUNCTION line per name (in the #else
// branch) names both vendor symbols so the dispatch check maps either backend's
// call back to the gpusparse* alias, cuSPARSE's inlined-shim call included.
// ────────────────────────────────────────────────────────────────────────

#if defined(GPUMOD_GPU_BACKEND_CUDA)

inline gpusparseStatus_t gpusparseSgebsr2gebsc_bufferSize(
    gpusparseHandle_t handle, int mb, int nb, int nnzb, const float *bsrVal, const int *bsrRowPtr,
    const int *bsrColInd, int rowBlockDim, int colBlockDim, std::size_t *pBufferSizeInBytes) {
  int bytes = 0;
  gpusparseStatus_t status = ::gpumod::cuda::cusparseSgebsr2gebsc_bufferSize(
      handle, mb, nb, nnzb, bsrVal, bsrRowPtr, bsrColInd, rowBlockDim, colBlockDim, &bytes);
  *pBufferSizeInBytes = static_cast<std::size_t>(bytes);
  return status;
}
inline gpusparseStatus_t gpusparseDgebsr2gebsc_bufferSize(
    gpusparseHandle_t handle, int mb, int nb, int nnzb, const double *bsrVal, const int *bsrRowPtr,
    const int *bsrColInd, int rowBlockDim, int colBlockDim, std::size_t *pBufferSizeInBytes) {
  int bytes = 0;
  gpusparseStatus_t status = ::gpumod::cuda::cusparseDgebsr2gebsc_bufferSize(
      handle, mb, nb, nnzb, bsrVal, bsrRowPtr, bsrColInd, rowBlockDim, colBlockDim, &bytes);
  *pBufferSizeInBytes = static_cast<std::size_t>(bytes);
  return status;
}
inline gpusparseStatus_t gpusparseCgebsr2gebsc_bufferSize(gpusparseHandle_t handle, int mb, int nb,
                                                          int nnzb, const gpuFloatComplex *bsrVal,
                                                          const int *bsrRowPtr,
                                                          const int *bsrColInd, int rowBlockDim,
                                                          int colBlockDim,
                                                          std::size_t *pBufferSizeInBytes) {
  int bytes = 0;
  gpusparseStatus_t status = ::gpumod::cuda::cusparseCgebsr2gebsc_bufferSize(
      handle, mb, nb, nnzb, bsrVal, bsrRowPtr, bsrColInd, rowBlockDim, colBlockDim, &bytes);
  *pBufferSizeInBytes = static_cast<std::size_t>(bytes);
  return status;
}
inline gpusparseStatus_t gpusparseZgebsr2gebsc_bufferSize(gpusparseHandle_t handle, int mb, int nb,
                                                          int nnzb, const gpuDoubleComplex *bsrVal,
                                                          const int *bsrRowPtr,
                                                          const int *bsrColInd, int rowBlockDim,
                                                          int colBlockDim,
                                                          std::size_t *pBufferSizeInBytes) {
  int bytes = 0;
  gpusparseStatus_t status = ::gpumod::cuda::cusparseZgebsr2gebsc_bufferSize(
      handle, mb, nb, nnzb, bsrVal, bsrRowPtr, bsrColInd, rowBlockDim, colBlockDim, &bytes);
  *pBufferSizeInBytes = static_cast<std::size_t>(bytes);
  return status;
}

inline gpusparseStatus_t
gpusparseScsr2gebsr_bufferSize(gpusparseHandle_t handle, gpusparseDirection_t dirA, int m, int n,
                               const gpusparseMatDescr_t descrA, const float *csrVal,
                               const int *csrRowPtr, const int *csrColInd, int rowBlockDim,
                               int colBlockDim, std::size_t *pBufferSizeInBytes) {
  int bytes = 0;
  gpusparseStatus_t status = ::gpumod::cuda::cusparseScsr2gebsr_bufferSize(
      handle, dirA, m, n, descrA, csrVal, csrRowPtr, csrColInd, rowBlockDim, colBlockDim, &bytes);
  *pBufferSizeInBytes = static_cast<std::size_t>(bytes);
  return status;
}
inline gpusparseStatus_t
gpusparseDcsr2gebsr_bufferSize(gpusparseHandle_t handle, gpusparseDirection_t dirA, int m, int n,
                               const gpusparseMatDescr_t descrA, const double *csrVal,
                               const int *csrRowPtr, const int *csrColInd, int rowBlockDim,
                               int colBlockDim, std::size_t *pBufferSizeInBytes) {
  int bytes = 0;
  gpusparseStatus_t status = ::gpumod::cuda::cusparseDcsr2gebsr_bufferSize(
      handle, dirA, m, n, descrA, csrVal, csrRowPtr, csrColInd, rowBlockDim, colBlockDim, &bytes);
  *pBufferSizeInBytes = static_cast<std::size_t>(bytes);
  return status;
}
inline gpusparseStatus_t
gpusparseCcsr2gebsr_bufferSize(gpusparseHandle_t handle, gpusparseDirection_t dirA, int m, int n,
                               const gpusparseMatDescr_t descrA, const gpuFloatComplex *csrVal,
                               const int *csrRowPtr, const int *csrColInd, int rowBlockDim,
                               int colBlockDim, std::size_t *pBufferSizeInBytes) {
  int bytes = 0;
  gpusparseStatus_t status = ::gpumod::cuda::cusparseCcsr2gebsr_bufferSize(
      handle, dirA, m, n, descrA, csrVal, csrRowPtr, csrColInd, rowBlockDim, colBlockDim, &bytes);
  *pBufferSizeInBytes = static_cast<std::size_t>(bytes);
  return status;
}
inline gpusparseStatus_t
gpusparseZcsr2gebsr_bufferSize(gpusparseHandle_t handle, gpusparseDirection_t dirA, int m, int n,
                               const gpusparseMatDescr_t descrA, const gpuDoubleComplex *csrVal,
                               const int *csrRowPtr, const int *csrColInd, int rowBlockDim,
                               int colBlockDim, std::size_t *pBufferSizeInBytes) {
  int bytes = 0;
  gpusparseStatus_t status = ::gpumod::cuda::cusparseZcsr2gebsr_bufferSize(
      handle, dirA, m, n, descrA, csrVal, csrRowPtr, csrColInd, rowBlockDim, colBlockDim, &bytes);
  *pBufferSizeInBytes = static_cast<std::size_t>(bytes);
  return status;
}

#else

GPUMOD_FUNCTION(gpusparseSgebsr2gebsc_bufferSize, cusparseSgebsr2gebsc_bufferSize,
                hipsparseSgebsr2gebsc_bufferSize)
GPUMOD_FUNCTION(gpusparseDgebsr2gebsc_bufferSize, cusparseDgebsr2gebsc_bufferSize,
                hipsparseDgebsr2gebsc_bufferSize)
GPUMOD_FUNCTION(gpusparseCgebsr2gebsc_bufferSize, cusparseCgebsr2gebsc_bufferSize,
                hipsparseCgebsr2gebsc_bufferSize)
GPUMOD_FUNCTION(gpusparseZgebsr2gebsc_bufferSize, cusparseZgebsr2gebsc_bufferSize,
                hipsparseZgebsr2gebsc_bufferSize)

GPUMOD_FUNCTION(gpusparseScsr2gebsr_bufferSize, cusparseScsr2gebsr_bufferSize,
                hipsparseScsr2gebsr_bufferSize)
GPUMOD_FUNCTION(gpusparseDcsr2gebsr_bufferSize, cusparseDcsr2gebsr_bufferSize,
                hipsparseDcsr2gebsr_bufferSize)
GPUMOD_FUNCTION(gpusparseCcsr2gebsr_bufferSize, cusparseCcsr2gebsr_bufferSize,
                hipsparseCcsr2gebsr_bufferSize)
GPUMOD_FUNCTION(gpusparseZcsr2gebsr_bufferSize, cusparseZcsr2gebsr_bufferSize,
                hipsparseZcsr2gebsr_bufferSize)

#endif

} // namespace gpumod
