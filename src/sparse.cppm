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
 * wwr.cuda.cusparse / wwr.hip.hipsparse.
 *
 * cuSPARSE-only functions (the Preview SpMMOp API, CreateSlicedEll) and the
 * legacy typed functions cuSPARSE removed but hipSPARSE keeps (csrmv, csrsv2,
 * csrmm, csrsm2, csric02, csrilu02, gemvi, the HYB path, ...) are absent -- this
 * module adapts what both vendors offer, same policy as gpu.blas / gpu.solver.
 *
 * Usage:
 *   import wwr.sparse;
 *
 *   gpusparseHandle_t handle;
 *   gpusparseCreate(&handle);
 */

module;

#include "gpu_backend.h"

export module wwr.sparse;

import std;
import wwr.complex;
#if defined(WWR_GPU_BACKEND_CUDA)
import wwr.cuda.cusparse;
#else
import wwr.hip.hipsparse;
#endif

export namespace wwr {

// ========================================================================
// Types
// ========================================================================

WWR_TYPE(gpusparseHandle_t, cusparseHandle_t, hipsparseHandle_t)
WWR_TYPE(gpusparseStatus_t, cusparseStatus_t, hipsparseStatus_t)
WWR_TYPE(gpusparseMatDescr_t, cusparseMatDescr_t, hipsparseMatDescr_t)
WWR_TYPE(gpusparseOperation_t, cusparseOperation_t, hipsparseOperation_t)
WWR_TYPE(gpusparseDirection_t, cusparseDirection_t, hipsparseDirection_t)
WWR_TYPE(gpusparseAction_t, cusparseAction_t, hipsparseAction_t)
WWR_TYPE(gpusparseIndexBase_t, cusparseIndexBase_t, hipsparseIndexBase_t)
WWR_TYPE(gpusparseMatrixType_t, cusparseMatrixType_t, hipsparseMatrixType_t)
WWR_TYPE(gpusparseFillMode_t, cusparseFillMode_t, hipsparseFillMode_t)
WWR_TYPE(gpusparseDiagType_t, cusparseDiagType_t, hipsparseDiagType_t)
WWR_TYPE(gpusparsePointerMode_t, cusparsePointerMode_t, hipsparsePointerMode_t)

// ========================================================================
// Constants
// ========================================================================

WWR_VALUE(GPUSPARSE_STATUS_SUCCESS, CUSPARSE_STATUS_SUCCESS, HIPSPARSE_STATUS_SUCCESS)

WWR_VALUE(GPUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
             HIPSPARSE_OPERATION_NON_TRANSPOSE)
WWR_VALUE(GPUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
             HIPSPARSE_OPERATION_TRANSPOSE)
WWR_VALUE(GPUSPARSE_OPERATION_CONJUGATE_TRANSPOSE, CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE,
             HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE)

WWR_VALUE(GPUSPARSE_DIRECTION_ROW, CUSPARSE_DIRECTION_ROW, HIPSPARSE_DIRECTION_ROW)
WWR_VALUE(GPUSPARSE_DIRECTION_COLUMN, CUSPARSE_DIRECTION_COLUMN, HIPSPARSE_DIRECTION_COLUMN)

WWR_VALUE(GPUSPARSE_ACTION_SYMBOLIC, CUSPARSE_ACTION_SYMBOLIC, HIPSPARSE_ACTION_SYMBOLIC)
WWR_VALUE(GPUSPARSE_ACTION_NUMERIC, CUSPARSE_ACTION_NUMERIC, HIPSPARSE_ACTION_NUMERIC)

WWR_VALUE(GPUSPARSE_INDEX_BASE_ZERO, CUSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ZERO)
WWR_VALUE(GPUSPARSE_INDEX_BASE_ONE, CUSPARSE_INDEX_BASE_ONE, HIPSPARSE_INDEX_BASE_ONE)

WWR_VALUE(GPUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_MATRIX_TYPE_GENERAL,
             HIPSPARSE_MATRIX_TYPE_GENERAL)
WWR_VALUE(GPUSPARSE_MATRIX_TYPE_SYMMETRIC, CUSPARSE_MATRIX_TYPE_SYMMETRIC,
             HIPSPARSE_MATRIX_TYPE_SYMMETRIC)
WWR_VALUE(GPUSPARSE_MATRIX_TYPE_HERMITIAN, CUSPARSE_MATRIX_TYPE_HERMITIAN,
             HIPSPARSE_MATRIX_TYPE_HERMITIAN)
WWR_VALUE(GPUSPARSE_MATRIX_TYPE_TRIANGULAR, CUSPARSE_MATRIX_TYPE_TRIANGULAR,
             HIPSPARSE_MATRIX_TYPE_TRIANGULAR)

WWR_VALUE(GPUSPARSE_FILL_MODE_LOWER, CUSPARSE_FILL_MODE_LOWER, HIPSPARSE_FILL_MODE_LOWER)
WWR_VALUE(GPUSPARSE_FILL_MODE_UPPER, CUSPARSE_FILL_MODE_UPPER, HIPSPARSE_FILL_MODE_UPPER)

WWR_VALUE(GPUSPARSE_DIAG_TYPE_NON_UNIT, CUSPARSE_DIAG_TYPE_NON_UNIT,
             HIPSPARSE_DIAG_TYPE_NON_UNIT)
WWR_VALUE(GPUSPARSE_DIAG_TYPE_UNIT, CUSPARSE_DIAG_TYPE_UNIT, HIPSPARSE_DIAG_TYPE_UNIT)

WWR_VALUE(GPUSPARSE_POINTER_MODE_HOST, CUSPARSE_POINTER_MODE_HOST, HIPSPARSE_POINTER_MODE_HOST)
WWR_VALUE(GPUSPARSE_POINTER_MODE_DEVICE, CUSPARSE_POINTER_MODE_DEVICE,
             HIPSPARSE_POINTER_MODE_DEVICE)

// ========================================================================
// Handle, stream, pointer mode, error strings
// ========================================================================

WWR_FUNCTION(gpusparseCreate, cusparseCreate, hipsparseCreate)
WWR_FUNCTION(gpusparseDestroy, cusparseDestroy, hipsparseDestroy)
WWR_FUNCTION(gpusparseSetStream, cusparseSetStream, hipsparseSetStream)
WWR_FUNCTION(gpusparseGetStream, cusparseGetStream, hipsparseGetStream)
WWR_FUNCTION(gpusparseSetPointerMode, cusparseSetPointerMode, hipsparseSetPointerMode)
WWR_FUNCTION(gpusparseGetPointerMode, cusparseGetPointerMode, hipsparseGetPointerMode)
WWR_FUNCTION(gpusparseGetErrorName, cusparseGetErrorName, hipsparseGetErrorName)
WWR_FUNCTION(gpusparseGetErrorString, cusparseGetErrorString, hipsparseGetErrorString)

// ========================================================================
// Matrix descriptor (legacy helper routines)
// ========================================================================

WWR_FUNCTION(gpusparseCreateMatDescr, cusparseCreateMatDescr, hipsparseCreateMatDescr)
WWR_FUNCTION(gpusparseDestroyMatDescr, cusparseDestroyMatDescr, hipsparseDestroyMatDescr)
WWR_FUNCTION(gpusparseSetMatType, cusparseSetMatType, hipsparseSetMatType)
WWR_FUNCTION(gpusparseGetMatType, cusparseGetMatType, hipsparseGetMatType)
WWR_FUNCTION(gpusparseSetMatFillMode, cusparseSetMatFillMode, hipsparseSetMatFillMode)
WWR_FUNCTION(gpusparseGetMatFillMode, cusparseGetMatFillMode, hipsparseGetMatFillMode)
WWR_FUNCTION(gpusparseSetMatDiagType, cusparseSetMatDiagType, hipsparseSetMatDiagType)
WWR_FUNCTION(gpusparseGetMatDiagType, cusparseGetMatDiagType, hipsparseGetMatDiagType)
WWR_FUNCTION(gpusparseSetMatIndexBase, cusparseSetMatIndexBase, hipsparseSetMatIndexBase)
WWR_FUNCTION(gpusparseGetMatIndexBase, cusparseGetMatIndexBase, hipsparseGetMatIndexBase)

// ────────────────────────────────────────────────────────────────────────
// Level 2 -- BSR matrix-vector multiply
// ────────────────────────────────────────────────────────────────────────

WWR_FUNCTION(gpusparseSbsrmv, cusparseSbsrmv, hipsparseSbsrmv)
WWR_FUNCTION(gpusparseDbsrmv, cusparseDbsrmv, hipsparseDbsrmv)
WWR_FUNCTION(gpusparseCbsrmv, cusparseCbsrmv, hipsparseCbsrmv)
WWR_FUNCTION(gpusparseZbsrmv, cusparseZbsrmv, hipsparseZbsrmv)

// ────────────────────────────────────────────────────────────────────────
// Tridiagonal / pentadiagonal batch solvers (gtsv2 / gpsvInterleavedBatch)
// ────────────────────────────────────────────────────────────────────────

WWR_FUNCTION(gpusparseSgtsv2_bufferSizeExt, cusparseSgtsv2_bufferSizeExt,
                hipsparseSgtsv2_bufferSizeExt)
WWR_FUNCTION(gpusparseDgtsv2_bufferSizeExt, cusparseDgtsv2_bufferSizeExt,
                hipsparseDgtsv2_bufferSizeExt)
WWR_FUNCTION(gpusparseCgtsv2_bufferSizeExt, cusparseCgtsv2_bufferSizeExt,
                hipsparseCgtsv2_bufferSizeExt)
WWR_FUNCTION(gpusparseZgtsv2_bufferSizeExt, cusparseZgtsv2_bufferSizeExt,
                hipsparseZgtsv2_bufferSizeExt)

WWR_FUNCTION(gpusparseSgtsv2, cusparseSgtsv2, hipsparseSgtsv2)
WWR_FUNCTION(gpusparseDgtsv2, cusparseDgtsv2, hipsparseDgtsv2)
WWR_FUNCTION(gpusparseCgtsv2, cusparseCgtsv2, hipsparseCgtsv2)
WWR_FUNCTION(gpusparseZgtsv2, cusparseZgtsv2, hipsparseZgtsv2)

WWR_FUNCTION(gpusparseSgtsv2_nopivot_bufferSizeExt, cusparseSgtsv2_nopivot_bufferSizeExt,
                hipsparseSgtsv2_nopivot_bufferSizeExt)
WWR_FUNCTION(gpusparseDgtsv2_nopivot_bufferSizeExt, cusparseDgtsv2_nopivot_bufferSizeExt,
                hipsparseDgtsv2_nopivot_bufferSizeExt)
WWR_FUNCTION(gpusparseCgtsv2_nopivot_bufferSizeExt, cusparseCgtsv2_nopivot_bufferSizeExt,
                hipsparseCgtsv2_nopivot_bufferSizeExt)
WWR_FUNCTION(gpusparseZgtsv2_nopivot_bufferSizeExt, cusparseZgtsv2_nopivot_bufferSizeExt,
                hipsparseZgtsv2_nopivot_bufferSizeExt)

WWR_FUNCTION(gpusparseSgtsv2_nopivot, cusparseSgtsv2_nopivot, hipsparseSgtsv2_nopivot)
WWR_FUNCTION(gpusparseDgtsv2_nopivot, cusparseDgtsv2_nopivot, hipsparseDgtsv2_nopivot)
WWR_FUNCTION(gpusparseCgtsv2_nopivot, cusparseCgtsv2_nopivot, hipsparseCgtsv2_nopivot)
WWR_FUNCTION(gpusparseZgtsv2_nopivot, cusparseZgtsv2_nopivot, hipsparseZgtsv2_nopivot)

WWR_FUNCTION(gpusparseSgtsv2StridedBatch_bufferSizeExt, cusparseSgtsv2StridedBatch_bufferSizeExt,
                hipsparseSgtsv2StridedBatch_bufferSizeExt)
WWR_FUNCTION(gpusparseDgtsv2StridedBatch_bufferSizeExt, cusparseDgtsv2StridedBatch_bufferSizeExt,
                hipsparseDgtsv2StridedBatch_bufferSizeExt)
WWR_FUNCTION(gpusparseCgtsv2StridedBatch_bufferSizeExt, cusparseCgtsv2StridedBatch_bufferSizeExt,
                hipsparseCgtsv2StridedBatch_bufferSizeExt)
WWR_FUNCTION(gpusparseZgtsv2StridedBatch_bufferSizeExt, cusparseZgtsv2StridedBatch_bufferSizeExt,
                hipsparseZgtsv2StridedBatch_bufferSizeExt)

WWR_FUNCTION(gpusparseSgtsv2StridedBatch, cusparseSgtsv2StridedBatch,
                hipsparseSgtsv2StridedBatch)
WWR_FUNCTION(gpusparseDgtsv2StridedBatch, cusparseDgtsv2StridedBatch,
                hipsparseDgtsv2StridedBatch)
WWR_FUNCTION(gpusparseCgtsv2StridedBatch, cusparseCgtsv2StridedBatch,
                hipsparseCgtsv2StridedBatch)
WWR_FUNCTION(gpusparseZgtsv2StridedBatch, cusparseZgtsv2StridedBatch,
                hipsparseZgtsv2StridedBatch)

WWR_FUNCTION(gpusparseSgtsvInterleavedBatch_bufferSizeExt,
                cusparseSgtsvInterleavedBatch_bufferSizeExt,
                hipsparseSgtsvInterleavedBatch_bufferSizeExt)
WWR_FUNCTION(gpusparseDgtsvInterleavedBatch_bufferSizeExt,
                cusparseDgtsvInterleavedBatch_bufferSizeExt,
                hipsparseDgtsvInterleavedBatch_bufferSizeExt)
WWR_FUNCTION(gpusparseCgtsvInterleavedBatch_bufferSizeExt,
                cusparseCgtsvInterleavedBatch_bufferSizeExt,
                hipsparseCgtsvInterleavedBatch_bufferSizeExt)
WWR_FUNCTION(gpusparseZgtsvInterleavedBatch_bufferSizeExt,
                cusparseZgtsvInterleavedBatch_bufferSizeExt,
                hipsparseZgtsvInterleavedBatch_bufferSizeExt)

WWR_FUNCTION(gpusparseSgtsvInterleavedBatch, cusparseSgtsvInterleavedBatch,
                hipsparseSgtsvInterleavedBatch)
WWR_FUNCTION(gpusparseDgtsvInterleavedBatch, cusparseDgtsvInterleavedBatch,
                hipsparseDgtsvInterleavedBatch)
WWR_FUNCTION(gpusparseCgtsvInterleavedBatch, cusparseCgtsvInterleavedBatch,
                hipsparseCgtsvInterleavedBatch)
WWR_FUNCTION(gpusparseZgtsvInterleavedBatch, cusparseZgtsvInterleavedBatch,
                hipsparseZgtsvInterleavedBatch)

WWR_FUNCTION(gpusparseSgpsvInterleavedBatch_bufferSizeExt,
                cusparseSgpsvInterleavedBatch_bufferSizeExt,
                hipsparseSgpsvInterleavedBatch_bufferSizeExt)
WWR_FUNCTION(gpusparseDgpsvInterleavedBatch_bufferSizeExt,
                cusparseDgpsvInterleavedBatch_bufferSizeExt,
                hipsparseDgpsvInterleavedBatch_bufferSizeExt)
WWR_FUNCTION(gpusparseCgpsvInterleavedBatch_bufferSizeExt,
                cusparseCgpsvInterleavedBatch_bufferSizeExt,
                hipsparseCgpsvInterleavedBatch_bufferSizeExt)
WWR_FUNCTION(gpusparseZgpsvInterleavedBatch_bufferSizeExt,
                cusparseZgpsvInterleavedBatch_bufferSizeExt,
                hipsparseZgpsvInterleavedBatch_bufferSizeExt)

WWR_FUNCTION(gpusparseSgpsvInterleavedBatch, cusparseSgpsvInterleavedBatch,
                hipsparseSgpsvInterleavedBatch)
WWR_FUNCTION(gpusparseDgpsvInterleavedBatch, cusparseDgpsvInterleavedBatch,
                hipsparseDgpsvInterleavedBatch)
WWR_FUNCTION(gpusparseCgpsvInterleavedBatch, cusparseCgpsvInterleavedBatch,
                hipsparseCgpsvInterleavedBatch)
WWR_FUNCTION(gpusparseZgpsvInterleavedBatch, cusparseZgpsvInterleavedBatch,
                hipsparseZgpsvInterleavedBatch)

// ────────────────────────────────────────────────────────────────────────
// Extra -- CSR matrix addition (csrgeam2)
// ────────────────────────────────────────────────────────────────────────

WWR_FUNCTION(gpusparseScsrgeam2_bufferSizeExt, cusparseScsrgeam2_bufferSizeExt,
                hipsparseScsrgeam2_bufferSizeExt)
WWR_FUNCTION(gpusparseDcsrgeam2_bufferSizeExt, cusparseDcsrgeam2_bufferSizeExt,
                hipsparseDcsrgeam2_bufferSizeExt)
WWR_FUNCTION(gpusparseCcsrgeam2_bufferSizeExt, cusparseCcsrgeam2_bufferSizeExt,
                hipsparseCcsrgeam2_bufferSizeExt)
WWR_FUNCTION(gpusparseZcsrgeam2_bufferSizeExt, cusparseZcsrgeam2_bufferSizeExt,
                hipsparseZcsrgeam2_bufferSizeExt)

WWR_FUNCTION(gpusparseScsrgeam2, cusparseScsrgeam2, hipsparseScsrgeam2)
WWR_FUNCTION(gpusparseDcsrgeam2, cusparseDcsrgeam2, hipsparseDcsrgeam2)
WWR_FUNCTION(gpusparseCcsrgeam2, cusparseCcsrgeam2, hipsparseCcsrgeam2)
WWR_FUNCTION(gpusparseZcsrgeam2, cusparseZcsrgeam2, hipsparseZcsrgeam2)

// ────────────────────────────────────────────────────────────────────────
// Conversion -- nnz, gebsr2gebsc, csr2gebsr
// ────────────────────────────────────────────────────────────────────────

WWR_FUNCTION(gpusparseSnnz, cusparseSnnz, hipsparseSnnz)
WWR_FUNCTION(gpusparseDnnz, cusparseDnnz, hipsparseDnnz)
WWR_FUNCTION(gpusparseCnnz, cusparseCnnz, hipsparseCnnz)
WWR_FUNCTION(gpusparseZnnz, cusparseZnnz, hipsparseZnnz)

WWR_FUNCTION(gpusparseSgebsr2gebsc, cusparseSgebsr2gebsc, hipsparseSgebsr2gebsc)
WWR_FUNCTION(gpusparseDgebsr2gebsc, cusparseDgebsr2gebsc, hipsparseDgebsr2gebsc)
WWR_FUNCTION(gpusparseCgebsr2gebsc, cusparseCgebsr2gebsc, hipsparseCgebsr2gebsc)
WWR_FUNCTION(gpusparseZgebsr2gebsc, cusparseZgebsr2gebsc, hipsparseZgebsr2gebsc)

WWR_FUNCTION(gpusparseScsr2gebsr, cusparseScsr2gebsr, hipsparseScsr2gebsr)
WWR_FUNCTION(gpusparseDcsr2gebsr, cusparseDcsr2gebsr, hipsparseDcsr2gebsr)
WWR_FUNCTION(gpusparseCcsr2gebsr, cusparseCcsr2gebsr, hipsparseCcsr2gebsr)
WWR_FUNCTION(gpusparseZcsr2gebsr, cusparseZcsr2gebsr, hipsparseZcsr2gebsr)

// ────────────────────────────────────────────────────────────────────────
// gebsr2gebsc_bufferSize / csr2gebsr_bufferSize: uniform std::size_t* buffer size
//
// cuSPARSE's *_bufferSize (not the *Ext form) writes the byte count as int*;
// hipSPARSE writes it as std::size_t*. These keep hipSPARSE's std::size_t* signature on
// both backends, forwarding through an int on CUDA -- the mirror of blas's
// getrsBatched const shims. A single WWR_FUNCTION line per name (in the #else
// branch) names both vendor symbols so the dispatch check maps either backend's
// call back to the gpusparse* alias, cuSPARSE's inlined-shim call included.
// ────────────────────────────────────────────────────────────────────────

#if defined(WWR_GPU_BACKEND_CUDA)

inline gpusparseStatus_t gpusparseSgebsr2gebsc_bufferSize(
    gpusparseHandle_t handle, int mb, int nb, int nnzb, const float *bsrVal, const int *bsrRowPtr,
    const int *bsrColInd, int rowBlockDim, int colBlockDim, std::size_t *pBufferSizeInBytes) {
  int bytes = 0;
  gpusparseStatus_t status = ::wwr::cuda::cusparseSgebsr2gebsc_bufferSize(
      handle, mb, nb, nnzb, bsrVal, bsrRowPtr, bsrColInd, rowBlockDim, colBlockDim, &bytes);
  *pBufferSizeInBytes = static_cast<std::size_t>(bytes);
  return status;
}
inline gpusparseStatus_t gpusparseDgebsr2gebsc_bufferSize(
    gpusparseHandle_t handle, int mb, int nb, int nnzb, const double *bsrVal, const int *bsrRowPtr,
    const int *bsrColInd, int rowBlockDim, int colBlockDim, std::size_t *pBufferSizeInBytes) {
  int bytes = 0;
  gpusparseStatus_t status = ::wwr::cuda::cusparseDgebsr2gebsc_bufferSize(
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
  gpusparseStatus_t status = ::wwr::cuda::cusparseCgebsr2gebsc_bufferSize(
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
  gpusparseStatus_t status = ::wwr::cuda::cusparseZgebsr2gebsc_bufferSize(
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
  gpusparseStatus_t status = ::wwr::cuda::cusparseScsr2gebsr_bufferSize(
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
  gpusparseStatus_t status = ::wwr::cuda::cusparseDcsr2gebsr_bufferSize(
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
  gpusparseStatus_t status = ::wwr::cuda::cusparseCcsr2gebsr_bufferSize(
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
  gpusparseStatus_t status = ::wwr::cuda::cusparseZcsr2gebsr_bufferSize(
      handle, dirA, m, n, descrA, csrVal, csrRowPtr, csrColInd, rowBlockDim, colBlockDim, &bytes);
  *pBufferSizeInBytes = static_cast<std::size_t>(bytes);
  return status;
}

#else

WWR_FUNCTION(gpusparseSgebsr2gebsc_bufferSize, cusparseSgebsr2gebsc_bufferSize,
                hipsparseSgebsr2gebsc_bufferSize)
WWR_FUNCTION(gpusparseDgebsr2gebsc_bufferSize, cusparseDgebsr2gebsc_bufferSize,
                hipsparseDgebsr2gebsc_bufferSize)
WWR_FUNCTION(gpusparseCgebsr2gebsc_bufferSize, cusparseCgebsr2gebsc_bufferSize,
                hipsparseCgebsr2gebsc_bufferSize)
WWR_FUNCTION(gpusparseZgebsr2gebsc_bufferSize, cusparseZgebsr2gebsc_bufferSize,
                hipsparseZgebsr2gebsc_bufferSize)

WWR_FUNCTION(gpusparseScsr2gebsr_bufferSize, cusparseScsr2gebsr_bufferSize,
                hipsparseScsr2gebsr_bufferSize)
WWR_FUNCTION(gpusparseDcsr2gebsr_bufferSize, cusparseDcsr2gebsr_bufferSize,
                hipsparseDcsr2gebsr_bufferSize)
WWR_FUNCTION(gpusparseCcsr2gebsr_bufferSize, cusparseCcsr2gebsr_bufferSize,
                hipsparseCcsr2gebsr_bufferSize)
WWR_FUNCTION(gpusparseZcsr2gebsr_bufferSize, cusparseZcsr2gebsr_bufferSize,
                hipsparseZcsr2gebsr_bufferSize)

#endif

} // namespace wwr
