/**
 * @file sparse.cppm
 * @brief Backend-neutral sparse: wwrsparse* names for cuSPARSE / hipSPARSE
 *
 * wwrsparse<X><name> stands for cusparse<X><name> on a CUDA build and
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
 *   wwrsparseHandle_t handle;
 *   wwrsparseCreate(&handle);
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

WWR_TYPE(wwrsparseHandle_t, cusparseHandle_t, hipsparseHandle_t)
WWR_TYPE(wwrsparseStatus_t, cusparseStatus_t, hipsparseStatus_t)
WWR_TYPE(wwrsparseMatDescr_t, cusparseMatDescr_t, hipsparseMatDescr_t)
WWR_TYPE(wwrsparseOperation_t, cusparseOperation_t, hipsparseOperation_t)
WWR_TYPE(wwrsparseDirection_t, cusparseDirection_t, hipsparseDirection_t)
WWR_TYPE(wwrsparseAction_t, cusparseAction_t, hipsparseAction_t)
WWR_TYPE(wwrsparseIndexBase_t, cusparseIndexBase_t, hipsparseIndexBase_t)
WWR_TYPE(wwrsparseMatrixType_t, cusparseMatrixType_t, hipsparseMatrixType_t)
WWR_TYPE(wwrsparseFillMode_t, cusparseFillMode_t, hipsparseFillMode_t)
WWR_TYPE(wwrsparseDiagType_t, cusparseDiagType_t, hipsparseDiagType_t)
WWR_TYPE(wwrsparsePointerMode_t, cusparsePointerMode_t, hipsparsePointerMode_t)

// ========================================================================
// Constants
// ========================================================================

WWR_VALUE(WWRSPARSE_STATUS_SUCCESS, CUSPARSE_STATUS_SUCCESS, HIPSPARSE_STATUS_SUCCESS)

WWR_VALUE(WWRSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
             HIPSPARSE_OPERATION_NON_TRANSPOSE)
WWR_VALUE(WWRSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
             HIPSPARSE_OPERATION_TRANSPOSE)
WWR_VALUE(WWRSPARSE_OPERATION_CONJUGATE_TRANSPOSE, CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE,
             HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE)

WWR_VALUE(WWRSPARSE_DIRECTION_ROW, CUSPARSE_DIRECTION_ROW, HIPSPARSE_DIRECTION_ROW)
WWR_VALUE(WWRSPARSE_DIRECTION_COLUMN, CUSPARSE_DIRECTION_COLUMN, HIPSPARSE_DIRECTION_COLUMN)

WWR_VALUE(WWRSPARSE_ACTION_SYMBOLIC, CUSPARSE_ACTION_SYMBOLIC, HIPSPARSE_ACTION_SYMBOLIC)
WWR_VALUE(WWRSPARSE_ACTION_NUMERIC, CUSPARSE_ACTION_NUMERIC, HIPSPARSE_ACTION_NUMERIC)

WWR_VALUE(WWRSPARSE_INDEX_BASE_ZERO, CUSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ZERO)
WWR_VALUE(WWRSPARSE_INDEX_BASE_ONE, CUSPARSE_INDEX_BASE_ONE, HIPSPARSE_INDEX_BASE_ONE)

WWR_VALUE(WWRSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_MATRIX_TYPE_GENERAL,
             HIPSPARSE_MATRIX_TYPE_GENERAL)
WWR_VALUE(WWRSPARSE_MATRIX_TYPE_SYMMETRIC, CUSPARSE_MATRIX_TYPE_SYMMETRIC,
             HIPSPARSE_MATRIX_TYPE_SYMMETRIC)
WWR_VALUE(WWRSPARSE_MATRIX_TYPE_HERMITIAN, CUSPARSE_MATRIX_TYPE_HERMITIAN,
             HIPSPARSE_MATRIX_TYPE_HERMITIAN)
WWR_VALUE(WWRSPARSE_MATRIX_TYPE_TRIANGULAR, CUSPARSE_MATRIX_TYPE_TRIANGULAR,
             HIPSPARSE_MATRIX_TYPE_TRIANGULAR)

WWR_VALUE(WWRSPARSE_FILL_MODE_LOWER, CUSPARSE_FILL_MODE_LOWER, HIPSPARSE_FILL_MODE_LOWER)
WWR_VALUE(WWRSPARSE_FILL_MODE_UPPER, CUSPARSE_FILL_MODE_UPPER, HIPSPARSE_FILL_MODE_UPPER)

WWR_VALUE(WWRSPARSE_DIAG_TYPE_NON_UNIT, CUSPARSE_DIAG_TYPE_NON_UNIT,
             HIPSPARSE_DIAG_TYPE_NON_UNIT)
WWR_VALUE(WWRSPARSE_DIAG_TYPE_UNIT, CUSPARSE_DIAG_TYPE_UNIT, HIPSPARSE_DIAG_TYPE_UNIT)

WWR_VALUE(WWRSPARSE_POINTER_MODE_HOST, CUSPARSE_POINTER_MODE_HOST, HIPSPARSE_POINTER_MODE_HOST)
WWR_VALUE(WWRSPARSE_POINTER_MODE_DEVICE, CUSPARSE_POINTER_MODE_DEVICE,
             HIPSPARSE_POINTER_MODE_DEVICE)

// ========================================================================
// Handle, stream, pointer mode, error strings
// ========================================================================

WWR_FUNCTION(wwrsparseCreate, cusparseCreate, hipsparseCreate)
WWR_FUNCTION(wwrsparseDestroy, cusparseDestroy, hipsparseDestroy)
WWR_FUNCTION(wwrsparseSetStream, cusparseSetStream, hipsparseSetStream)
WWR_FUNCTION(wwrsparseGetStream, cusparseGetStream, hipsparseGetStream)
WWR_FUNCTION(wwrsparseSetPointerMode, cusparseSetPointerMode, hipsparseSetPointerMode)
WWR_FUNCTION(wwrsparseGetPointerMode, cusparseGetPointerMode, hipsparseGetPointerMode)
WWR_FUNCTION(wwrsparseGetErrorName, cusparseGetErrorName, hipsparseGetErrorName)
WWR_FUNCTION(wwrsparseGetErrorString, cusparseGetErrorString, hipsparseGetErrorString)

// ========================================================================
// Matrix descriptor (legacy helper routines)
// ========================================================================

WWR_FUNCTION(wwrsparseCreateMatDescr, cusparseCreateMatDescr, hipsparseCreateMatDescr)
WWR_FUNCTION(wwrsparseDestroyMatDescr, cusparseDestroyMatDescr, hipsparseDestroyMatDescr)
WWR_FUNCTION(wwrsparseSetMatType, cusparseSetMatType, hipsparseSetMatType)
WWR_FUNCTION(wwrsparseGetMatType, cusparseGetMatType, hipsparseGetMatType)
WWR_FUNCTION(wwrsparseSetMatFillMode, cusparseSetMatFillMode, hipsparseSetMatFillMode)
WWR_FUNCTION(wwrsparseGetMatFillMode, cusparseGetMatFillMode, hipsparseGetMatFillMode)
WWR_FUNCTION(wwrsparseSetMatDiagType, cusparseSetMatDiagType, hipsparseSetMatDiagType)
WWR_FUNCTION(wwrsparseGetMatDiagType, cusparseGetMatDiagType, hipsparseGetMatDiagType)
WWR_FUNCTION(wwrsparseSetMatIndexBase, cusparseSetMatIndexBase, hipsparseSetMatIndexBase)
WWR_FUNCTION(wwrsparseGetMatIndexBase, cusparseGetMatIndexBase, hipsparseGetMatIndexBase)

// ────────────────────────────────────────────────────────────────────────
// Level 2 -- BSR matrix-vector multiply
// ────────────────────────────────────────────────────────────────────────

WWR_FUNCTION(wwrsparseSbsrmv, cusparseSbsrmv, hipsparseSbsrmv)
WWR_FUNCTION(wwrsparseDbsrmv, cusparseDbsrmv, hipsparseDbsrmv)
WWR_FUNCTION(wwrsparseCbsrmv, cusparseCbsrmv, hipsparseCbsrmv)
WWR_FUNCTION(wwrsparseZbsrmv, cusparseZbsrmv, hipsparseZbsrmv)

// ────────────────────────────────────────────────────────────────────────
// Tridiagonal / pentadiagonal batch solvers (gtsv2 / gpsvInterleavedBatch)
// ────────────────────────────────────────────────────────────────────────

WWR_FUNCTION(wwrsparseSgtsv2_bufferSizeExt, cusparseSgtsv2_bufferSizeExt,
                hipsparseSgtsv2_bufferSizeExt)
WWR_FUNCTION(wwrsparseDgtsv2_bufferSizeExt, cusparseDgtsv2_bufferSizeExt,
                hipsparseDgtsv2_bufferSizeExt)
WWR_FUNCTION(wwrsparseCgtsv2_bufferSizeExt, cusparseCgtsv2_bufferSizeExt,
                hipsparseCgtsv2_bufferSizeExt)
WWR_FUNCTION(wwrsparseZgtsv2_bufferSizeExt, cusparseZgtsv2_bufferSizeExt,
                hipsparseZgtsv2_bufferSizeExt)

WWR_FUNCTION(wwrsparseSgtsv2, cusparseSgtsv2, hipsparseSgtsv2)
WWR_FUNCTION(wwrsparseDgtsv2, cusparseDgtsv2, hipsparseDgtsv2)
WWR_FUNCTION(wwrsparseCgtsv2, cusparseCgtsv2, hipsparseCgtsv2)
WWR_FUNCTION(wwrsparseZgtsv2, cusparseZgtsv2, hipsparseZgtsv2)

WWR_FUNCTION(wwrsparseSgtsv2_nopivot_bufferSizeExt, cusparseSgtsv2_nopivot_bufferSizeExt,
                hipsparseSgtsv2_nopivot_bufferSizeExt)
WWR_FUNCTION(wwrsparseDgtsv2_nopivot_bufferSizeExt, cusparseDgtsv2_nopivot_bufferSizeExt,
                hipsparseDgtsv2_nopivot_bufferSizeExt)
WWR_FUNCTION(wwrsparseCgtsv2_nopivot_bufferSizeExt, cusparseCgtsv2_nopivot_bufferSizeExt,
                hipsparseCgtsv2_nopivot_bufferSizeExt)
WWR_FUNCTION(wwrsparseZgtsv2_nopivot_bufferSizeExt, cusparseZgtsv2_nopivot_bufferSizeExt,
                hipsparseZgtsv2_nopivot_bufferSizeExt)

WWR_FUNCTION(wwrsparseSgtsv2_nopivot, cusparseSgtsv2_nopivot, hipsparseSgtsv2_nopivot)
WWR_FUNCTION(wwrsparseDgtsv2_nopivot, cusparseDgtsv2_nopivot, hipsparseDgtsv2_nopivot)
WWR_FUNCTION(wwrsparseCgtsv2_nopivot, cusparseCgtsv2_nopivot, hipsparseCgtsv2_nopivot)
WWR_FUNCTION(wwrsparseZgtsv2_nopivot, cusparseZgtsv2_nopivot, hipsparseZgtsv2_nopivot)

WWR_FUNCTION(wwrsparseSgtsv2StridedBatch_bufferSizeExt, cusparseSgtsv2StridedBatch_bufferSizeExt,
                hipsparseSgtsv2StridedBatch_bufferSizeExt)
WWR_FUNCTION(wwrsparseDgtsv2StridedBatch_bufferSizeExt, cusparseDgtsv2StridedBatch_bufferSizeExt,
                hipsparseDgtsv2StridedBatch_bufferSizeExt)
WWR_FUNCTION(wwrsparseCgtsv2StridedBatch_bufferSizeExt, cusparseCgtsv2StridedBatch_bufferSizeExt,
                hipsparseCgtsv2StridedBatch_bufferSizeExt)
WWR_FUNCTION(wwrsparseZgtsv2StridedBatch_bufferSizeExt, cusparseZgtsv2StridedBatch_bufferSizeExt,
                hipsparseZgtsv2StridedBatch_bufferSizeExt)

WWR_FUNCTION(wwrsparseSgtsv2StridedBatch, cusparseSgtsv2StridedBatch,
                hipsparseSgtsv2StridedBatch)
WWR_FUNCTION(wwrsparseDgtsv2StridedBatch, cusparseDgtsv2StridedBatch,
                hipsparseDgtsv2StridedBatch)
WWR_FUNCTION(wwrsparseCgtsv2StridedBatch, cusparseCgtsv2StridedBatch,
                hipsparseCgtsv2StridedBatch)
WWR_FUNCTION(wwrsparseZgtsv2StridedBatch, cusparseZgtsv2StridedBatch,
                hipsparseZgtsv2StridedBatch)

WWR_FUNCTION(wwrsparseSgtsvInterleavedBatch_bufferSizeExt,
                cusparseSgtsvInterleavedBatch_bufferSizeExt,
                hipsparseSgtsvInterleavedBatch_bufferSizeExt)
WWR_FUNCTION(wwrsparseDgtsvInterleavedBatch_bufferSizeExt,
                cusparseDgtsvInterleavedBatch_bufferSizeExt,
                hipsparseDgtsvInterleavedBatch_bufferSizeExt)
WWR_FUNCTION(wwrsparseCgtsvInterleavedBatch_bufferSizeExt,
                cusparseCgtsvInterleavedBatch_bufferSizeExt,
                hipsparseCgtsvInterleavedBatch_bufferSizeExt)
WWR_FUNCTION(wwrsparseZgtsvInterleavedBatch_bufferSizeExt,
                cusparseZgtsvInterleavedBatch_bufferSizeExt,
                hipsparseZgtsvInterleavedBatch_bufferSizeExt)

WWR_FUNCTION(wwrsparseSgtsvInterleavedBatch, cusparseSgtsvInterleavedBatch,
                hipsparseSgtsvInterleavedBatch)
WWR_FUNCTION(wwrsparseDgtsvInterleavedBatch, cusparseDgtsvInterleavedBatch,
                hipsparseDgtsvInterleavedBatch)
WWR_FUNCTION(wwrsparseCgtsvInterleavedBatch, cusparseCgtsvInterleavedBatch,
                hipsparseCgtsvInterleavedBatch)
WWR_FUNCTION(wwrsparseZgtsvInterleavedBatch, cusparseZgtsvInterleavedBatch,
                hipsparseZgtsvInterleavedBatch)

WWR_FUNCTION(wwrsparseSgpsvInterleavedBatch_bufferSizeExt,
                cusparseSgpsvInterleavedBatch_bufferSizeExt,
                hipsparseSgpsvInterleavedBatch_bufferSizeExt)
WWR_FUNCTION(wwrsparseDgpsvInterleavedBatch_bufferSizeExt,
                cusparseDgpsvInterleavedBatch_bufferSizeExt,
                hipsparseDgpsvInterleavedBatch_bufferSizeExt)
WWR_FUNCTION(wwrsparseCgpsvInterleavedBatch_bufferSizeExt,
                cusparseCgpsvInterleavedBatch_bufferSizeExt,
                hipsparseCgpsvInterleavedBatch_bufferSizeExt)
WWR_FUNCTION(wwrsparseZgpsvInterleavedBatch_bufferSizeExt,
                cusparseZgpsvInterleavedBatch_bufferSizeExt,
                hipsparseZgpsvInterleavedBatch_bufferSizeExt)

WWR_FUNCTION(wwrsparseSgpsvInterleavedBatch, cusparseSgpsvInterleavedBatch,
                hipsparseSgpsvInterleavedBatch)
WWR_FUNCTION(wwrsparseDgpsvInterleavedBatch, cusparseDgpsvInterleavedBatch,
                hipsparseDgpsvInterleavedBatch)
WWR_FUNCTION(wwrsparseCgpsvInterleavedBatch, cusparseCgpsvInterleavedBatch,
                hipsparseCgpsvInterleavedBatch)
WWR_FUNCTION(wwrsparseZgpsvInterleavedBatch, cusparseZgpsvInterleavedBatch,
                hipsparseZgpsvInterleavedBatch)

// ────────────────────────────────────────────────────────────────────────
// Extra -- CSR matrix addition (csrgeam2)
// ────────────────────────────────────────────────────────────────────────

WWR_FUNCTION(wwrsparseScsrgeam2_bufferSizeExt, cusparseScsrgeam2_bufferSizeExt,
                hipsparseScsrgeam2_bufferSizeExt)
WWR_FUNCTION(wwrsparseDcsrgeam2_bufferSizeExt, cusparseDcsrgeam2_bufferSizeExt,
                hipsparseDcsrgeam2_bufferSizeExt)
WWR_FUNCTION(wwrsparseCcsrgeam2_bufferSizeExt, cusparseCcsrgeam2_bufferSizeExt,
                hipsparseCcsrgeam2_bufferSizeExt)
WWR_FUNCTION(wwrsparseZcsrgeam2_bufferSizeExt, cusparseZcsrgeam2_bufferSizeExt,
                hipsparseZcsrgeam2_bufferSizeExt)

WWR_FUNCTION(wwrsparseScsrgeam2, cusparseScsrgeam2, hipsparseScsrgeam2)
WWR_FUNCTION(wwrsparseDcsrgeam2, cusparseDcsrgeam2, hipsparseDcsrgeam2)
WWR_FUNCTION(wwrsparseCcsrgeam2, cusparseCcsrgeam2, hipsparseCcsrgeam2)
WWR_FUNCTION(wwrsparseZcsrgeam2, cusparseZcsrgeam2, hipsparseZcsrgeam2)

// ────────────────────────────────────────────────────────────────────────
// Conversion -- nnz, gebsr2gebsc, csr2gebsr
// ────────────────────────────────────────────────────────────────────────

WWR_FUNCTION(wwrsparseSnnz, cusparseSnnz, hipsparseSnnz)
WWR_FUNCTION(wwrsparseDnnz, cusparseDnnz, hipsparseDnnz)
WWR_FUNCTION(wwrsparseCnnz, cusparseCnnz, hipsparseCnnz)
WWR_FUNCTION(wwrsparseZnnz, cusparseZnnz, hipsparseZnnz)

WWR_FUNCTION(wwrsparseSgebsr2gebsc, cusparseSgebsr2gebsc, hipsparseSgebsr2gebsc)
WWR_FUNCTION(wwrsparseDgebsr2gebsc, cusparseDgebsr2gebsc, hipsparseDgebsr2gebsc)
WWR_FUNCTION(wwrsparseCgebsr2gebsc, cusparseCgebsr2gebsc, hipsparseCgebsr2gebsc)
WWR_FUNCTION(wwrsparseZgebsr2gebsc, cusparseZgebsr2gebsc, hipsparseZgebsr2gebsc)

WWR_FUNCTION(wwrsparseScsr2gebsr, cusparseScsr2gebsr, hipsparseScsr2gebsr)
WWR_FUNCTION(wwrsparseDcsr2gebsr, cusparseDcsr2gebsr, hipsparseDcsr2gebsr)
WWR_FUNCTION(wwrsparseCcsr2gebsr, cusparseCcsr2gebsr, hipsparseCcsr2gebsr)
WWR_FUNCTION(wwrsparseZcsr2gebsr, cusparseZcsr2gebsr, hipsparseZcsr2gebsr)

// ────────────────────────────────────────────────────────────────────────
// gebsr2gebsc_bufferSize / csr2gebsr_bufferSize: uniform std::size_t* buffer size
//
// cuSPARSE's *_bufferSize (not the *Ext form) writes the byte count as int*;
// hipSPARSE writes it as std::size_t*. These keep hipSPARSE's std::size_t* signature on
// both backends, forwarding through an int on CUDA -- the mirror of blas's
// getrsBatched const shims. A single WWR_FUNCTION line per name (in the #else
// branch) names both vendor symbols so the dispatch check maps either backend's
// call back to the wwrsparse* alias, cuSPARSE's inlined-shim call included.
// ────────────────────────────────────────────────────────────────────────

#if defined(WWR_GPU_BACKEND_CUDA)

inline wwrsparseStatus_t wwrsparseSgebsr2gebsc_bufferSize(
    wwrsparseHandle_t handle, int mb, int nb, int nnzb, const float *bsrVal, const int *bsrRowPtr,
    const int *bsrColInd, int rowBlockDim, int colBlockDim, std::size_t *pBufferSizeInBytes) {
  int bytes = 0;
  wwrsparseStatus_t status = ::wwr::cuda::cusparseSgebsr2gebsc_bufferSize(
      handle, mb, nb, nnzb, bsrVal, bsrRowPtr, bsrColInd, rowBlockDim, colBlockDim, &bytes);
  *pBufferSizeInBytes = static_cast<std::size_t>(bytes);
  return status;
}
inline wwrsparseStatus_t wwrsparseDgebsr2gebsc_bufferSize(
    wwrsparseHandle_t handle, int mb, int nb, int nnzb, const double *bsrVal, const int *bsrRowPtr,
    const int *bsrColInd, int rowBlockDim, int colBlockDim, std::size_t *pBufferSizeInBytes) {
  int bytes = 0;
  wwrsparseStatus_t status = ::wwr::cuda::cusparseDgebsr2gebsc_bufferSize(
      handle, mb, nb, nnzb, bsrVal, bsrRowPtr, bsrColInd, rowBlockDim, colBlockDim, &bytes);
  *pBufferSizeInBytes = static_cast<std::size_t>(bytes);
  return status;
}
inline wwrsparseStatus_t wwrsparseCgebsr2gebsc_bufferSize(wwrsparseHandle_t handle, int mb, int nb,
                                                          int nnzb, const wwrFloatComplex *bsrVal,
                                                          const int *bsrRowPtr,
                                                          const int *bsrColInd, int rowBlockDim,
                                                          int colBlockDim,
                                                          std::size_t *pBufferSizeInBytes) {
  int bytes = 0;
  wwrsparseStatus_t status = ::wwr::cuda::cusparseCgebsr2gebsc_bufferSize(
      handle, mb, nb, nnzb, bsrVal, bsrRowPtr, bsrColInd, rowBlockDim, colBlockDim, &bytes);
  *pBufferSizeInBytes = static_cast<std::size_t>(bytes);
  return status;
}
inline wwrsparseStatus_t wwrsparseZgebsr2gebsc_bufferSize(wwrsparseHandle_t handle, int mb, int nb,
                                                          int nnzb, const wwrDoubleComplex *bsrVal,
                                                          const int *bsrRowPtr,
                                                          const int *bsrColInd, int rowBlockDim,
                                                          int colBlockDim,
                                                          std::size_t *pBufferSizeInBytes) {
  int bytes = 0;
  wwrsparseStatus_t status = ::wwr::cuda::cusparseZgebsr2gebsc_bufferSize(
      handle, mb, nb, nnzb, bsrVal, bsrRowPtr, bsrColInd, rowBlockDim, colBlockDim, &bytes);
  *pBufferSizeInBytes = static_cast<std::size_t>(bytes);
  return status;
}

inline wwrsparseStatus_t
wwrsparseScsr2gebsr_bufferSize(wwrsparseHandle_t handle, wwrsparseDirection_t dirA, int m, int n,
                               const wwrsparseMatDescr_t descrA, const float *csrVal,
                               const int *csrRowPtr, const int *csrColInd, int rowBlockDim,
                               int colBlockDim, std::size_t *pBufferSizeInBytes) {
  int bytes = 0;
  wwrsparseStatus_t status = ::wwr::cuda::cusparseScsr2gebsr_bufferSize(
      handle, dirA, m, n, descrA, csrVal, csrRowPtr, csrColInd, rowBlockDim, colBlockDim, &bytes);
  *pBufferSizeInBytes = static_cast<std::size_t>(bytes);
  return status;
}
inline wwrsparseStatus_t
wwrsparseDcsr2gebsr_bufferSize(wwrsparseHandle_t handle, wwrsparseDirection_t dirA, int m, int n,
                               const wwrsparseMatDescr_t descrA, const double *csrVal,
                               const int *csrRowPtr, const int *csrColInd, int rowBlockDim,
                               int colBlockDim, std::size_t *pBufferSizeInBytes) {
  int bytes = 0;
  wwrsparseStatus_t status = ::wwr::cuda::cusparseDcsr2gebsr_bufferSize(
      handle, dirA, m, n, descrA, csrVal, csrRowPtr, csrColInd, rowBlockDim, colBlockDim, &bytes);
  *pBufferSizeInBytes = static_cast<std::size_t>(bytes);
  return status;
}
inline wwrsparseStatus_t
wwrsparseCcsr2gebsr_bufferSize(wwrsparseHandle_t handle, wwrsparseDirection_t dirA, int m, int n,
                               const wwrsparseMatDescr_t descrA, const wwrFloatComplex *csrVal,
                               const int *csrRowPtr, const int *csrColInd, int rowBlockDim,
                               int colBlockDim, std::size_t *pBufferSizeInBytes) {
  int bytes = 0;
  wwrsparseStatus_t status = ::wwr::cuda::cusparseCcsr2gebsr_bufferSize(
      handle, dirA, m, n, descrA, csrVal, csrRowPtr, csrColInd, rowBlockDim, colBlockDim, &bytes);
  *pBufferSizeInBytes = static_cast<std::size_t>(bytes);
  return status;
}
inline wwrsparseStatus_t
wwrsparseZcsr2gebsr_bufferSize(wwrsparseHandle_t handle, wwrsparseDirection_t dirA, int m, int n,
                               const wwrsparseMatDescr_t descrA, const wwrDoubleComplex *csrVal,
                               const int *csrRowPtr, const int *csrColInd, int rowBlockDim,
                               int colBlockDim, std::size_t *pBufferSizeInBytes) {
  int bytes = 0;
  wwrsparseStatus_t status = ::wwr::cuda::cusparseZcsr2gebsr_bufferSize(
      handle, dirA, m, n, descrA, csrVal, csrRowPtr, csrColInd, rowBlockDim, colBlockDim, &bytes);
  *pBufferSizeInBytes = static_cast<std::size_t>(bytes);
  return status;
}

#else

WWR_FUNCTION(wwrsparseSgebsr2gebsc_bufferSize, cusparseSgebsr2gebsc_bufferSize,
                hipsparseSgebsr2gebsc_bufferSize)
WWR_FUNCTION(wwrsparseDgebsr2gebsc_bufferSize, cusparseDgebsr2gebsc_bufferSize,
                hipsparseDgebsr2gebsc_bufferSize)
WWR_FUNCTION(wwrsparseCgebsr2gebsc_bufferSize, cusparseCgebsr2gebsc_bufferSize,
                hipsparseCgebsr2gebsc_bufferSize)
WWR_FUNCTION(wwrsparseZgebsr2gebsc_bufferSize, cusparseZgebsr2gebsc_bufferSize,
                hipsparseZgebsr2gebsc_bufferSize)

WWR_FUNCTION(wwrsparseScsr2gebsr_bufferSize, cusparseScsr2gebsr_bufferSize,
                hipsparseScsr2gebsr_bufferSize)
WWR_FUNCTION(wwrsparseDcsr2gebsr_bufferSize, cusparseDcsr2gebsr_bufferSize,
                hipsparseDcsr2gebsr_bufferSize)
WWR_FUNCTION(wwrsparseCcsr2gebsr_bufferSize, cusparseCcsr2gebsr_bufferSize,
                hipsparseCcsr2gebsr_bufferSize)
WWR_FUNCTION(wwrsparseZcsr2gebsr_bufferSize, cusparseZcsr2gebsr_bufferSize,
                hipsparseZcsr2gebsr_bufferSize)

#endif

} // namespace wwr
