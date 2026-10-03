/**
 * @file detail/sparse_names.h
 * @brief The backend-neutral cuSPARSE / hipSPARSE surface, as a macro-driven
 *        include fragment shared by the module and the non-module #include path
 *
 * NOT a standalone header: it is the list of wwrsparse* / WWRSPARSE_* names
 * (types, constants, functions and the std::size_t* buffer-size shims) with NO
 * namespace of its own and NO vendor #include. The includer supplies all of
 * that and pastes this inside its own `namespace wwr` -- so one list binds both
 * ways the surface is consumed: wwr.sparse (the module, `export namespace wwr`)
 * and wwr/sparse.h (the non-module #include path). Add a name here, once, and
 * both paths gain it.
 *
 * HOST only -- cuSPARSE/hipSPARSE is a host API. The surface binds straight to
 * the vendor's external-linkage `::cusparse*` / `::hipsparse*` declarations via
 * the _RAW macros (the "rand.h shape"): no raw vendor module is imported, so
 * the same `::`-prefixed names resolve in the module (vendor header in its GMF)
 * and the #include path alike. The modern generic API (SpMV, SpMM, SpGEMM) and
 * the single-vendor legacy functions are deliberately absent; reach those
 * through wwr.cuda.cusparse / wwr.hip.hipsparse.
 *
 * One backend difference is resolved here, not above: cuSPARSE's non-Ext
 * gebsr2gebsc_bufferSize / csr2gebsr_bufferSize write the byte count as int*
 * where hipSPARSE writes std::size_t*. These keep hipSPARSE's std::size_t*
 * signature on both backends, forwarding through an int on CUDA -- the mirror
 * of blas's getrsBatched const shims. See docs/architecture.md, section 5.
 *
 * Before including, the includer must have, in order:
 *   - the vendor header in scope (cusparse.h / hipsparse/hipsparse.h), which
 *     sparse.h pulls in;
 *   - std::size_t in scope (the CUDA *_bufferSize shims' out parameter) -- from
 *     <cstddef> via sparse.h on both paths (or import std on the module side);
 *   - wwrFloatComplex / wwrDoubleComplex in `namespace wwr` (the complex
 *     *_bufferSize shims name them) -- from import wwr.complex in the module, or
 *     #include "complex.h" in the #include path;
 *   - WWR_SELECT_RAW(cuda, hip) plus WWR_TYPE_RAW / WWR_VALUE_RAW /
 *     WWR_FUNCTION_RAW on top of it -- keyed on WWR_GPU_BACKEND_* in the module
 *     (backend.h) or WWR_SELECTED_* in the #include path (wwr/sparse.h);
 *   - WWR_SELECTED_CUDA / WWR_SELECTED_HIP (selected_backend.h, via sparse.h)
 *     for the *_bufferSize shims' one backend conditional.
 *
 * See src/sparse.cppm, src/sparse.h, src/wwr/sparse.h and docs/architecture.md
 * section 5.
 */

#pragma once

#ifndef WWR_FUNCTION_RAW
#error                                                                                             \
    "detail/sparse_names.h is an include fragment, not a standalone header: define WWR_TYPE_RAW/VALUE_RAW/FUNCTION_RAW and WWR_SELECT_RAW, ensure the vendor header (via sparse.h), std::size_t, wwrFloatComplex and WWR_SELECTED_* are in scope, and #include it inside namespace wwr. See src/sparse.h, src/sparse.cppm and src/wwr/sparse.h."
#endif

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables): each
// wwrsparse* function below is a deliberate constexpr reference to the selected
// backend's entry point (via WWR_FUNCTION_RAW). A reference to a vendor function
// has no const form, so the check cannot be satisfied without abandoning the
// alias pattern -- see backend.h and detail/blas_names.h.

// ========================================================================
// Types
// ========================================================================

WWR_TYPE_RAW(wwrsparseHandle_t, cusparseHandle_t, hipsparseHandle_t)
WWR_TYPE_RAW(wwrsparseStatus_t, cusparseStatus_t, hipsparseStatus_t)
WWR_TYPE_RAW(wwrsparseMatDescr_t, cusparseMatDescr_t, hipsparseMatDescr_t)
WWR_TYPE_RAW(wwrsparseOperation_t, cusparseOperation_t, hipsparseOperation_t)
WWR_TYPE_RAW(wwrsparseDirection_t, cusparseDirection_t, hipsparseDirection_t)
WWR_TYPE_RAW(wwrsparseAction_t, cusparseAction_t, hipsparseAction_t)
WWR_TYPE_RAW(wwrsparseIndexBase_t, cusparseIndexBase_t, hipsparseIndexBase_t)
WWR_TYPE_RAW(wwrsparseMatrixType_t, cusparseMatrixType_t, hipsparseMatrixType_t)
WWR_TYPE_RAW(wwrsparseFillMode_t, cusparseFillMode_t, hipsparseFillMode_t)
WWR_TYPE_RAW(wwrsparseDiagType_t, cusparseDiagType_t, hipsparseDiagType_t)
WWR_TYPE_RAW(wwrsparsePointerMode_t, cusparsePointerMode_t, hipsparsePointerMode_t)

// ========================================================================
// Constants
// ========================================================================

WWR_VALUE_RAW(WWRSPARSE_STATUS_SUCCESS, CUSPARSE_STATUS_SUCCESS, HIPSPARSE_STATUS_SUCCESS)

WWR_VALUE_RAW(WWRSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
             HIPSPARSE_OPERATION_NON_TRANSPOSE)
WWR_VALUE_RAW(WWRSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
             HIPSPARSE_OPERATION_TRANSPOSE)
WWR_VALUE_RAW(WWRSPARSE_OPERATION_CONJUGATE_TRANSPOSE, CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE,
             HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE)

WWR_VALUE_RAW(WWRSPARSE_DIRECTION_ROW, CUSPARSE_DIRECTION_ROW, HIPSPARSE_DIRECTION_ROW)
WWR_VALUE_RAW(WWRSPARSE_DIRECTION_COLUMN, CUSPARSE_DIRECTION_COLUMN, HIPSPARSE_DIRECTION_COLUMN)

WWR_VALUE_RAW(WWRSPARSE_ACTION_SYMBOLIC, CUSPARSE_ACTION_SYMBOLIC, HIPSPARSE_ACTION_SYMBOLIC)
WWR_VALUE_RAW(WWRSPARSE_ACTION_NUMERIC, CUSPARSE_ACTION_NUMERIC, HIPSPARSE_ACTION_NUMERIC)

WWR_VALUE_RAW(WWRSPARSE_INDEX_BASE_ZERO, CUSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ZERO)
WWR_VALUE_RAW(WWRSPARSE_INDEX_BASE_ONE, CUSPARSE_INDEX_BASE_ONE, HIPSPARSE_INDEX_BASE_ONE)

WWR_VALUE_RAW(WWRSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_MATRIX_TYPE_GENERAL,
             HIPSPARSE_MATRIX_TYPE_GENERAL)
WWR_VALUE_RAW(WWRSPARSE_MATRIX_TYPE_SYMMETRIC, CUSPARSE_MATRIX_TYPE_SYMMETRIC,
             HIPSPARSE_MATRIX_TYPE_SYMMETRIC)
WWR_VALUE_RAW(WWRSPARSE_MATRIX_TYPE_HERMITIAN, CUSPARSE_MATRIX_TYPE_HERMITIAN,
             HIPSPARSE_MATRIX_TYPE_HERMITIAN)
WWR_VALUE_RAW(WWRSPARSE_MATRIX_TYPE_TRIANGULAR, CUSPARSE_MATRIX_TYPE_TRIANGULAR,
             HIPSPARSE_MATRIX_TYPE_TRIANGULAR)

WWR_VALUE_RAW(WWRSPARSE_FILL_MODE_LOWER, CUSPARSE_FILL_MODE_LOWER, HIPSPARSE_FILL_MODE_LOWER)
WWR_VALUE_RAW(WWRSPARSE_FILL_MODE_UPPER, CUSPARSE_FILL_MODE_UPPER, HIPSPARSE_FILL_MODE_UPPER)

WWR_VALUE_RAW(WWRSPARSE_DIAG_TYPE_NON_UNIT, CUSPARSE_DIAG_TYPE_NON_UNIT,
             HIPSPARSE_DIAG_TYPE_NON_UNIT)
WWR_VALUE_RAW(WWRSPARSE_DIAG_TYPE_UNIT, CUSPARSE_DIAG_TYPE_UNIT, HIPSPARSE_DIAG_TYPE_UNIT)

WWR_VALUE_RAW(WWRSPARSE_POINTER_MODE_HOST, CUSPARSE_POINTER_MODE_HOST, HIPSPARSE_POINTER_MODE_HOST)
WWR_VALUE_RAW(WWRSPARSE_POINTER_MODE_DEVICE, CUSPARSE_POINTER_MODE_DEVICE,
             HIPSPARSE_POINTER_MODE_DEVICE)

// ========================================================================
// Handle, stream, pointer mode, error strings
// ========================================================================

WWR_FUNCTION_RAW(wwrsparseCreate, cusparseCreate, hipsparseCreate)
WWR_FUNCTION_RAW(wwrsparseDestroy, cusparseDestroy, hipsparseDestroy)
WWR_FUNCTION_RAW(wwrsparseSetStream, cusparseSetStream, hipsparseSetStream)
WWR_FUNCTION_RAW(wwrsparseGetStream, cusparseGetStream, hipsparseGetStream)
WWR_FUNCTION_RAW(wwrsparseSetPointerMode, cusparseSetPointerMode, hipsparseSetPointerMode)
WWR_FUNCTION_RAW(wwrsparseGetPointerMode, cusparseGetPointerMode, hipsparseGetPointerMode)
WWR_FUNCTION_RAW(wwrsparseGetErrorName, cusparseGetErrorName, hipsparseGetErrorName)
WWR_FUNCTION_RAW(wwrsparseGetErrorString, cusparseGetErrorString, hipsparseGetErrorString)

// ========================================================================
// Matrix descriptor (legacy helper routines)
// ========================================================================

WWR_FUNCTION_RAW(wwrsparseCreateMatDescr, cusparseCreateMatDescr, hipsparseCreateMatDescr)
WWR_FUNCTION_RAW(wwrsparseDestroyMatDescr, cusparseDestroyMatDescr, hipsparseDestroyMatDescr)
WWR_FUNCTION_RAW(wwrsparseSetMatType, cusparseSetMatType, hipsparseSetMatType)
WWR_FUNCTION_RAW(wwrsparseGetMatType, cusparseGetMatType, hipsparseGetMatType)
WWR_FUNCTION_RAW(wwrsparseSetMatFillMode, cusparseSetMatFillMode, hipsparseSetMatFillMode)
WWR_FUNCTION_RAW(wwrsparseGetMatFillMode, cusparseGetMatFillMode, hipsparseGetMatFillMode)
WWR_FUNCTION_RAW(wwrsparseSetMatDiagType, cusparseSetMatDiagType, hipsparseSetMatDiagType)
WWR_FUNCTION_RAW(wwrsparseGetMatDiagType, cusparseGetMatDiagType, hipsparseGetMatDiagType)
WWR_FUNCTION_RAW(wwrsparseSetMatIndexBase, cusparseSetMatIndexBase, hipsparseSetMatIndexBase)
WWR_FUNCTION_RAW(wwrsparseGetMatIndexBase, cusparseGetMatIndexBase, hipsparseGetMatIndexBase)

// ────────────────────────────────────────────────────────────────────────
// Level 2 -- BSR matrix-vector multiply
// ────────────────────────────────────────────────────────────────────────

WWR_FUNCTION_RAW(wwrsparseSbsrmv, cusparseSbsrmv, hipsparseSbsrmv)
WWR_FUNCTION_RAW(wwrsparseDbsrmv, cusparseDbsrmv, hipsparseDbsrmv)
WWR_FUNCTION_RAW(wwrsparseCbsrmv, cusparseCbsrmv, hipsparseCbsrmv)
WWR_FUNCTION_RAW(wwrsparseZbsrmv, cusparseZbsrmv, hipsparseZbsrmv)

// ────────────────────────────────────────────────────────────────────────
// Tridiagonal / pentadiagonal batch solvers (gtsv2 / gpsvInterleavedBatch)
// ────────────────────────────────────────────────────────────────────────

WWR_FUNCTION_RAW(wwrsparseSgtsv2_bufferSizeExt, cusparseSgtsv2_bufferSizeExt,
                hipsparseSgtsv2_bufferSizeExt)
WWR_FUNCTION_RAW(wwrsparseDgtsv2_bufferSizeExt, cusparseDgtsv2_bufferSizeExt,
                hipsparseDgtsv2_bufferSizeExt)
WWR_FUNCTION_RAW(wwrsparseCgtsv2_bufferSizeExt, cusparseCgtsv2_bufferSizeExt,
                hipsparseCgtsv2_bufferSizeExt)
WWR_FUNCTION_RAW(wwrsparseZgtsv2_bufferSizeExt, cusparseZgtsv2_bufferSizeExt,
                hipsparseZgtsv2_bufferSizeExt)

WWR_FUNCTION_RAW(wwrsparseSgtsv2, cusparseSgtsv2, hipsparseSgtsv2)
WWR_FUNCTION_RAW(wwrsparseDgtsv2, cusparseDgtsv2, hipsparseDgtsv2)
WWR_FUNCTION_RAW(wwrsparseCgtsv2, cusparseCgtsv2, hipsparseCgtsv2)
WWR_FUNCTION_RAW(wwrsparseZgtsv2, cusparseZgtsv2, hipsparseZgtsv2)

WWR_FUNCTION_RAW(wwrsparseSgtsv2_nopivot_bufferSizeExt, cusparseSgtsv2_nopivot_bufferSizeExt,
                hipsparseSgtsv2_nopivot_bufferSizeExt)
WWR_FUNCTION_RAW(wwrsparseDgtsv2_nopivot_bufferSizeExt, cusparseDgtsv2_nopivot_bufferSizeExt,
                hipsparseDgtsv2_nopivot_bufferSizeExt)
WWR_FUNCTION_RAW(wwrsparseCgtsv2_nopivot_bufferSizeExt, cusparseCgtsv2_nopivot_bufferSizeExt,
                hipsparseCgtsv2_nopivot_bufferSizeExt)
WWR_FUNCTION_RAW(wwrsparseZgtsv2_nopivot_bufferSizeExt, cusparseZgtsv2_nopivot_bufferSizeExt,
                hipsparseZgtsv2_nopivot_bufferSizeExt)

WWR_FUNCTION_RAW(wwrsparseSgtsv2_nopivot, cusparseSgtsv2_nopivot, hipsparseSgtsv2_nopivot)
WWR_FUNCTION_RAW(wwrsparseDgtsv2_nopivot, cusparseDgtsv2_nopivot, hipsparseDgtsv2_nopivot)
WWR_FUNCTION_RAW(wwrsparseCgtsv2_nopivot, cusparseCgtsv2_nopivot, hipsparseCgtsv2_nopivot)
WWR_FUNCTION_RAW(wwrsparseZgtsv2_nopivot, cusparseZgtsv2_nopivot, hipsparseZgtsv2_nopivot)

WWR_FUNCTION_RAW(wwrsparseSgtsv2StridedBatch_bufferSizeExt,
                cusparseSgtsv2StridedBatch_bufferSizeExt,
                hipsparseSgtsv2StridedBatch_bufferSizeExt)
WWR_FUNCTION_RAW(wwrsparseDgtsv2StridedBatch_bufferSizeExt,
                cusparseDgtsv2StridedBatch_bufferSizeExt,
                hipsparseDgtsv2StridedBatch_bufferSizeExt)
WWR_FUNCTION_RAW(wwrsparseCgtsv2StridedBatch_bufferSizeExt,
                cusparseCgtsv2StridedBatch_bufferSizeExt,
                hipsparseCgtsv2StridedBatch_bufferSizeExt)
WWR_FUNCTION_RAW(wwrsparseZgtsv2StridedBatch_bufferSizeExt,
                cusparseZgtsv2StridedBatch_bufferSizeExt,
                hipsparseZgtsv2StridedBatch_bufferSizeExt)

WWR_FUNCTION_RAW(wwrsparseSgtsv2StridedBatch, cusparseSgtsv2StridedBatch,
                hipsparseSgtsv2StridedBatch)
WWR_FUNCTION_RAW(wwrsparseDgtsv2StridedBatch, cusparseDgtsv2StridedBatch,
                hipsparseDgtsv2StridedBatch)
WWR_FUNCTION_RAW(wwrsparseCgtsv2StridedBatch, cusparseCgtsv2StridedBatch,
                hipsparseCgtsv2StridedBatch)
WWR_FUNCTION_RAW(wwrsparseZgtsv2StridedBatch, cusparseZgtsv2StridedBatch,
                hipsparseZgtsv2StridedBatch)

WWR_FUNCTION_RAW(wwrsparseSgtsvInterleavedBatch_bufferSizeExt,
                cusparseSgtsvInterleavedBatch_bufferSizeExt,
                hipsparseSgtsvInterleavedBatch_bufferSizeExt)
WWR_FUNCTION_RAW(wwrsparseDgtsvInterleavedBatch_bufferSizeExt,
                cusparseDgtsvInterleavedBatch_bufferSizeExt,
                hipsparseDgtsvInterleavedBatch_bufferSizeExt)
WWR_FUNCTION_RAW(wwrsparseCgtsvInterleavedBatch_bufferSizeExt,
                cusparseCgtsvInterleavedBatch_bufferSizeExt,
                hipsparseCgtsvInterleavedBatch_bufferSizeExt)
WWR_FUNCTION_RAW(wwrsparseZgtsvInterleavedBatch_bufferSizeExt,
                cusparseZgtsvInterleavedBatch_bufferSizeExt,
                hipsparseZgtsvInterleavedBatch_bufferSizeExt)

WWR_FUNCTION_RAW(wwrsparseSgtsvInterleavedBatch, cusparseSgtsvInterleavedBatch,
                hipsparseSgtsvInterleavedBatch)
WWR_FUNCTION_RAW(wwrsparseDgtsvInterleavedBatch, cusparseDgtsvInterleavedBatch,
                hipsparseDgtsvInterleavedBatch)
WWR_FUNCTION_RAW(wwrsparseCgtsvInterleavedBatch, cusparseCgtsvInterleavedBatch,
                hipsparseCgtsvInterleavedBatch)
WWR_FUNCTION_RAW(wwrsparseZgtsvInterleavedBatch, cusparseZgtsvInterleavedBatch,
                hipsparseZgtsvInterleavedBatch)

WWR_FUNCTION_RAW(wwrsparseSgpsvInterleavedBatch_bufferSizeExt,
                cusparseSgpsvInterleavedBatch_bufferSizeExt,
                hipsparseSgpsvInterleavedBatch_bufferSizeExt)
WWR_FUNCTION_RAW(wwrsparseDgpsvInterleavedBatch_bufferSizeExt,
                cusparseDgpsvInterleavedBatch_bufferSizeExt,
                hipsparseDgpsvInterleavedBatch_bufferSizeExt)
WWR_FUNCTION_RAW(wwrsparseCgpsvInterleavedBatch_bufferSizeExt,
                cusparseCgpsvInterleavedBatch_bufferSizeExt,
                hipsparseCgpsvInterleavedBatch_bufferSizeExt)
WWR_FUNCTION_RAW(wwrsparseZgpsvInterleavedBatch_bufferSizeExt,
                cusparseZgpsvInterleavedBatch_bufferSizeExt,
                hipsparseZgpsvInterleavedBatch_bufferSizeExt)

WWR_FUNCTION_RAW(wwrsparseSgpsvInterleavedBatch, cusparseSgpsvInterleavedBatch,
                hipsparseSgpsvInterleavedBatch)
WWR_FUNCTION_RAW(wwrsparseDgpsvInterleavedBatch, cusparseDgpsvInterleavedBatch,
                hipsparseDgpsvInterleavedBatch)
WWR_FUNCTION_RAW(wwrsparseCgpsvInterleavedBatch, cusparseCgpsvInterleavedBatch,
                hipsparseCgpsvInterleavedBatch)
WWR_FUNCTION_RAW(wwrsparseZgpsvInterleavedBatch, cusparseZgpsvInterleavedBatch,
                hipsparseZgpsvInterleavedBatch)

// ────────────────────────────────────────────────────────────────────────
// Extra -- CSR matrix addition (csrgeam2)
// ────────────────────────────────────────────────────────────────────────

WWR_FUNCTION_RAW(wwrsparseScsrgeam2_bufferSizeExt, cusparseScsrgeam2_bufferSizeExt,
                hipsparseScsrgeam2_bufferSizeExt)
WWR_FUNCTION_RAW(wwrsparseDcsrgeam2_bufferSizeExt, cusparseDcsrgeam2_bufferSizeExt,
                hipsparseDcsrgeam2_bufferSizeExt)
WWR_FUNCTION_RAW(wwrsparseCcsrgeam2_bufferSizeExt, cusparseCcsrgeam2_bufferSizeExt,
                hipsparseCcsrgeam2_bufferSizeExt)
WWR_FUNCTION_RAW(wwrsparseZcsrgeam2_bufferSizeExt, cusparseZcsrgeam2_bufferSizeExt,
                hipsparseZcsrgeam2_bufferSizeExt)

WWR_FUNCTION_RAW(wwrsparseScsrgeam2, cusparseScsrgeam2, hipsparseScsrgeam2)
WWR_FUNCTION_RAW(wwrsparseDcsrgeam2, cusparseDcsrgeam2, hipsparseDcsrgeam2)
WWR_FUNCTION_RAW(wwrsparseCcsrgeam2, cusparseCcsrgeam2, hipsparseCcsrgeam2)
WWR_FUNCTION_RAW(wwrsparseZcsrgeam2, cusparseZcsrgeam2, hipsparseZcsrgeam2)

// ────────────────────────────────────────────────────────────────────────
// Conversion -- nnz, gebsr2gebsc, csr2gebsr
// ────────────────────────────────────────────────────────────────────────

WWR_FUNCTION_RAW(wwrsparseSnnz, cusparseSnnz, hipsparseSnnz)
WWR_FUNCTION_RAW(wwrsparseDnnz, cusparseDnnz, hipsparseDnnz)
WWR_FUNCTION_RAW(wwrsparseCnnz, cusparseCnnz, hipsparseCnnz)
WWR_FUNCTION_RAW(wwrsparseZnnz, cusparseZnnz, hipsparseZnnz)

WWR_FUNCTION_RAW(wwrsparseSgebsr2gebsc, cusparseSgebsr2gebsc, hipsparseSgebsr2gebsc)
WWR_FUNCTION_RAW(wwrsparseDgebsr2gebsc, cusparseDgebsr2gebsc, hipsparseDgebsr2gebsc)
WWR_FUNCTION_RAW(wwrsparseCgebsr2gebsc, cusparseCgebsr2gebsc, hipsparseCgebsr2gebsc)
WWR_FUNCTION_RAW(wwrsparseZgebsr2gebsc, cusparseZgebsr2gebsc, hipsparseZgebsr2gebsc)

WWR_FUNCTION_RAW(wwrsparseScsr2gebsr, cusparseScsr2gebsr, hipsparseScsr2gebsr)
WWR_FUNCTION_RAW(wwrsparseDcsr2gebsr, cusparseDcsr2gebsr, hipsparseDcsr2gebsr)
WWR_FUNCTION_RAW(wwrsparseCcsr2gebsr, cusparseCcsr2gebsr, hipsparseCcsr2gebsr)
WWR_FUNCTION_RAW(wwrsparseZcsr2gebsr, cusparseZcsr2gebsr, hipsparseZcsr2gebsr)

// ────────────────────────────────────────────────────────────────────────
// gebsr2gebsc_bufferSize / csr2gebsr_bufferSize: uniform std::size_t* buffer size
//
// cuSPARSE's *_bufferSize (not the *Ext form) writes the byte count as int*;
// hipSPARSE writes it as std::size_t*. These keep hipSPARSE's std::size_t*
// signature on both backends, forwarding through an int on CUDA -- the mirror
// of blas's getrsBatched const shims. Under _RAW the hand-written CUDA bodies
// call the vendor global `::cusparse*` directly (as blas_names.h's HIP getrs
// forwarders call `::hipblas*`), not a `::wwr::cuda::` re-export, since no raw
// vendor module is imported. The HIP branch is a single WWR_FUNCTION_RAW line
// per name, which names both vendor symbols so the dispatch check maps either
// backend's call back to the wwrsparse* alias.
// ────────────────────────────────────────────────────────────────────────

#if defined(WWR_SELECTED_CUDA)

inline wwrsparseStatus_t wwrsparseSgebsr2gebsc_bufferSize(
    wwrsparseHandle_t handle, int mb, int nb, int nnzb, const float *bsrVal, const int *bsrRowPtr,
    const int *bsrColInd, int rowBlockDim, int colBlockDim, std::size_t *pBufferSizeInBytes) {
  int bytes = 0;
  wwrsparseStatus_t status = ::cusparseSgebsr2gebsc_bufferSize(
      handle, mb, nb, nnzb, bsrVal, bsrRowPtr, bsrColInd, rowBlockDim, colBlockDim, &bytes);
  *pBufferSizeInBytes = static_cast<std::size_t>(bytes);
  return status;
}
inline wwrsparseStatus_t wwrsparseDgebsr2gebsc_bufferSize(
    wwrsparseHandle_t handle, int mb, int nb, int nnzb, const double *bsrVal, const int *bsrRowPtr,
    const int *bsrColInd, int rowBlockDim, int colBlockDim, std::size_t *pBufferSizeInBytes) {
  int bytes = 0;
  wwrsparseStatus_t status = ::cusparseDgebsr2gebsc_bufferSize(
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
  wwrsparseStatus_t status = ::cusparseCgebsr2gebsc_bufferSize(
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
  wwrsparseStatus_t status = ::cusparseZgebsr2gebsc_bufferSize(
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
  wwrsparseStatus_t status = ::cusparseScsr2gebsr_bufferSize(
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
  wwrsparseStatus_t status = ::cusparseDcsr2gebsr_bufferSize(
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
  wwrsparseStatus_t status = ::cusparseCcsr2gebsr_bufferSize(
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
  wwrsparseStatus_t status = ::cusparseZcsr2gebsr_bufferSize(
      handle, dirA, m, n, descrA, csrVal, csrRowPtr, csrColInd, rowBlockDim, colBlockDim, &bytes);
  *pBufferSizeInBytes = static_cast<std::size_t>(bytes);
  return status;
}

#else

WWR_FUNCTION_RAW(wwrsparseSgebsr2gebsc_bufferSize, cusparseSgebsr2gebsc_bufferSize,
                hipsparseSgebsr2gebsc_bufferSize)
WWR_FUNCTION_RAW(wwrsparseDgebsr2gebsc_bufferSize, cusparseDgebsr2gebsc_bufferSize,
                hipsparseDgebsr2gebsc_bufferSize)
WWR_FUNCTION_RAW(wwrsparseCgebsr2gebsc_bufferSize, cusparseCgebsr2gebsc_bufferSize,
                hipsparseCgebsr2gebsc_bufferSize)
WWR_FUNCTION_RAW(wwrsparseZgebsr2gebsc_bufferSize, cusparseZgebsr2gebsc_bufferSize,
                hipsparseZgebsr2gebsc_bufferSize)

WWR_FUNCTION_RAW(wwrsparseScsr2gebsr_bufferSize, cusparseScsr2gebsr_bufferSize,
                hipsparseScsr2gebsr_bufferSize)
WWR_FUNCTION_RAW(wwrsparseDcsr2gebsr_bufferSize, cusparseDcsr2gebsr_bufferSize,
                hipsparseDcsr2gebsr_bufferSize)
WWR_FUNCTION_RAW(wwrsparseCcsr2gebsr_bufferSize, cusparseCcsr2gebsr_bufferSize,
                hipsparseCcsr2gebsr_bufferSize)
WWR_FUNCTION_RAW(wwrsparseZcsr2gebsr_bufferSize, cusparseZcsr2gebsr_bufferSize,
                hipsparseZcsr2gebsr_bufferSize)

#endif

// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)
