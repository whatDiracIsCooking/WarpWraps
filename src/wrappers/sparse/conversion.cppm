/**
 * @file conversion.cppm
 * @brief GPU sparse format conversion / structure routines
 *
 * Type-safe wrappers for the nonzero count (nnz), general-BSR to general-BSC
 * transpose (gebsr2gebsc) and CSR to general-BSR conversion (csr2gebsr), each
 * with its companion buffer-size query where one exists. (The untyped structure
 * queries Xcsr2gebsrNnz / Xcoo2csr / ... are called directly through the raw
 * module in wwr.sparse.)
 *
 * The two *_bufferSize queries take a size_t* byte count on the neutral API:
 * cuSPARSE spells this parameter int*, hipSPARSE size_t*, and wwr.sparse
 * reconciles the two -- see its file header.
 *
 * Usage:
 *   import wwr.wrappers.sparse;
 */

module;

#include "dispatch_macros.h"

export module wwr.wrappers.sparse:conversion;

import wwr.sparse;
import wwr.complex;
import wwr.wrappers.common;
import std;

export namespace wwr {

// ========================================================================
// nnz: count nonzeros per row/column and in total, from a dense matrix
// ========================================================================

template<usual_fp T>
wwrsparseStatus_t nnz(wwrsparseHandle_t handle, wwrsparseDirection_t dirA, int m, int n,
                      const wwrsparseMatDescr_t descrA, const T *A, int lda, int *nnzPerRowCol,
                      int *nnzTotalDevHostPtr) {
  WWR_USUAL_DISPATCH(T, nnz, handle, dirA, m, n, descrA, A, lda, nnzPerRowCol,
                        nnzTotalDevHostPtr);
}

// ========================================================================
// gebsr2gebsc: general-BSR to general-BSC (block transpose)
// ========================================================================

template<usual_fp T>
wwrsparseStatus_t gebsr2gebsc_bufferSize(wwrsparseHandle_t handle, int mb, int nb, int nnzb,
                                         const T *bsrVal, const int *bsrRowPtr,
                                         const int *bsrColInd, int rowBlockDim, int colBlockDim,
                                         size_t *pBufferSizeInBytes) {
  WWR_USUAL_DISPATCH(T, gebsr2gebsc_bufferSize, handle, mb, nb, nnzb, bsrVal, bsrRowPtr,
                        bsrColInd, rowBlockDim, colBlockDim, pBufferSizeInBytes);
}

template<usual_fp T>
wwrsparseStatus_t gebsr2gebsc(wwrsparseHandle_t handle, int mb, int nb, int nnzb, const T *bsrVal,
                              const int *bsrRowPtr, const int *bsrColInd, int rowBlockDim,
                              int colBlockDim, T *bscVal, int *bscRowInd, int *bscColPtr,
                              wwrsparseAction_t copyValues, wwrsparseIndexBase_t idxBase,
                              void *pBuffer) {
  WWR_USUAL_DISPATCH(T, gebsr2gebsc, handle, mb, nb, nnzb, bsrVal, bsrRowPtr, bsrColInd,
                        rowBlockDim, colBlockDim, bscVal, bscRowInd, bscColPtr, copyValues, idxBase,
                        pBuffer);
}

// ========================================================================
// csr2gebsr: CSR to general-BSR
// ========================================================================

template<usual_fp T>
wwrsparseStatus_t csr2gebsr_bufferSize(wwrsparseHandle_t handle, wwrsparseDirection_t dirA, int m,
                                       int n, const wwrsparseMatDescr_t descrA,
                                       const T *csrSortedValA, const int *csrSortedRowPtrA,
                                       const int *csrSortedColIndA, int rowBlockDim,
                                       int colBlockDim, size_t *pBufferSizeInBytes) {
  WWR_USUAL_DISPATCH(T, csr2gebsr_bufferSize, handle, dirA, m, n, descrA, csrSortedValA,
                        csrSortedRowPtrA, csrSortedColIndA, rowBlockDim, colBlockDim,
                        pBufferSizeInBytes);
}

template<usual_fp T>
wwrsparseStatus_t csr2gebsr(wwrsparseHandle_t handle, wwrsparseDirection_t dirA, int m, int n,
                            const wwrsparseMatDescr_t descrA, const T *csrSortedValA,
                            const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                            const wwrsparseMatDescr_t descrC, T *bsrSortedValC,
                            int *bsrSortedRowPtrC, int *bsrSortedColIndC, int rowBlockDim,
                            int colBlockDim, void *pBuffer) {
  WWR_USUAL_DISPATCH(T, csr2gebsr, handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                        csrSortedColIndA, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC,
                        rowBlockDim, colBlockDim, pBuffer);
}

// ==================== Explicit Template Instantiations ====================
// Matching `template` instantiations live in instantiations.cpp.

// Function: nnz
extern template wwrsparseStatus_t nnz<float>(wwrsparseHandle_t, wwrsparseDirection_t, int, int,
                                             const wwrsparseMatDescr_t, const float *, int, int *,
                                             int *);
extern template wwrsparseStatus_t nnz<double>(wwrsparseHandle_t, wwrsparseDirection_t, int, int,
                                              const wwrsparseMatDescr_t, const double *, int, int *,
                                              int *);
extern template wwrsparseStatus_t nnz<wwrFloatComplex>(wwrsparseHandle_t, wwrsparseDirection_t, int,
                                                       int, const wwrsparseMatDescr_t,
                                                       const wwrFloatComplex *, int, int *, int *);
extern template wwrsparseStatus_t nnz<wwrDoubleComplex>(wwrsparseHandle_t, wwrsparseDirection_t,
                                                        int, int, const wwrsparseMatDescr_t,
                                                        const wwrDoubleComplex *, int, int *,
                                                        int *);

// Function: gebsr2gebsc_bufferSize
extern template wwrsparseStatus_t gebsr2gebsc_bufferSize<float>(wwrsparseHandle_t, int, int, int,
                                                                const float *, const int *,
                                                                const int *, int, int, size_t *);
extern template wwrsparseStatus_t gebsr2gebsc_bufferSize<double>(wwrsparseHandle_t, int, int, int,
                                                                 const double *, const int *,
                                                                 const int *, int, int, size_t *);
extern template wwrsparseStatus_t
gebsr2gebsc_bufferSize<wwrFloatComplex>(wwrsparseHandle_t, int, int, int, const wwrFloatComplex *,
                                        const int *, const int *, int, int, size_t *);
extern template wwrsparseStatus_t
gebsr2gebsc_bufferSize<wwrDoubleComplex>(wwrsparseHandle_t, int, int, int, const wwrDoubleComplex *,
                                         const int *, const int *, int, int, size_t *);

// Function: gebsr2gebsc
extern template wwrsparseStatus_t gebsr2gebsc<float>(wwrsparseHandle_t, int, int, int,
                                                     const float *, const int *, const int *, int,
                                                     int, float *, int *, int *, wwrsparseAction_t,
                                                     wwrsparseIndexBase_t, void *);
extern template wwrsparseStatus_t
gebsr2gebsc<double>(wwrsparseHandle_t, int, int, int, const double *, const int *, const int *, int,
                    int, double *, int *, int *, wwrsparseAction_t, wwrsparseIndexBase_t, void *);
extern template wwrsparseStatus_t
gebsr2gebsc<wwrFloatComplex>(wwrsparseHandle_t, int, int, int, const wwrFloatComplex *, const int *,
                             const int *, int, int, wwrFloatComplex *, int *, int *,
                             wwrsparseAction_t, wwrsparseIndexBase_t, void *);
extern template wwrsparseStatus_t
gebsr2gebsc<wwrDoubleComplex>(wwrsparseHandle_t, int, int, int, const wwrDoubleComplex *,
                              const int *, const int *, int, int, wwrDoubleComplex *, int *, int *,
                              wwrsparseAction_t, wwrsparseIndexBase_t, void *);

// Function: csr2gebsr_bufferSize
extern template wwrsparseStatus_t csr2gebsr_bufferSize<float>(wwrsparseHandle_t,
                                                              wwrsparseDirection_t, int, int,
                                                              const wwrsparseMatDescr_t,
                                                              const float *, const int *,
                                                              const int *, int, int, size_t *);
extern template wwrsparseStatus_t csr2gebsr_bufferSize<double>(wwrsparseHandle_t,
                                                               wwrsparseDirection_t, int, int,
                                                               const wwrsparseMatDescr_t,
                                                               const double *, const int *,
                                                               const int *, int, int, size_t *);
extern template wwrsparseStatus_t
csr2gebsr_bufferSize<wwrFloatComplex>(wwrsparseHandle_t, wwrsparseDirection_t, int, int,
                                      const wwrsparseMatDescr_t, const wwrFloatComplex *,
                                      const int *, const int *, int, int, size_t *);
extern template wwrsparseStatus_t
csr2gebsr_bufferSize<wwrDoubleComplex>(wwrsparseHandle_t, wwrsparseDirection_t, int, int,
                                       const wwrsparseMatDescr_t, const wwrDoubleComplex *,
                                       const int *, const int *, int, int, size_t *);

// Function: csr2gebsr
extern template wwrsparseStatus_t csr2gebsr<float>(wwrsparseHandle_t, wwrsparseDirection_t, int,
                                                   int, const wwrsparseMatDescr_t, const float *,
                                                   const int *, const int *,
                                                   const wwrsparseMatDescr_t, float *, int *, int *,
                                                   int, int, void *);
extern template wwrsparseStatus_t csr2gebsr<double>(wwrsparseHandle_t, wwrsparseDirection_t, int,
                                                    int, const wwrsparseMatDescr_t, const double *,
                                                    const int *, const int *,
                                                    const wwrsparseMatDescr_t, double *, int *,
                                                    int *, int, int, void *);
extern template wwrsparseStatus_t
csr2gebsr<wwrFloatComplex>(wwrsparseHandle_t, wwrsparseDirection_t, int, int,
                           const wwrsparseMatDescr_t, const wwrFloatComplex *, const int *,
                           const int *, const wwrsparseMatDescr_t, wwrFloatComplex *, int *, int *,
                           int, int, void *);
extern template wwrsparseStatus_t
csr2gebsr<wwrDoubleComplex>(wwrsparseHandle_t, wwrsparseDirection_t, int, int,
                            const wwrsparseMatDescr_t, const wwrDoubleComplex *, const int *,
                            const int *, const wwrsparseMatDescr_t, wwrDoubleComplex *, int *,
                            int *, int, int, void *);

} // namespace wwr
