/**
 * @file conversion.cppm
 * @brief GPU sparse format conversion / structure routines
 *
 * Type-safe wrappers for the nonzero count (nnz), general-BSR to general-BSC
 * transpose (gebsr2gebsc) and CSR to general-BSR conversion (csr2gebsr), each
 * with its companion buffer-size query where one exists. (The untyped structure
 * queries Xcsr2gebsrNnz / Xcoo2csr / ... are called directly through the raw
 * module in gpumod.sparse.)
 *
 * The two *_bufferSize queries take a size_t* byte count on the neutral API:
 * cuSPARSE spells this parameter int*, hipSPARSE size_t*, and gpumod.sparse
 * reconciles the two -- see its file header.
 *
 * Usage:
 *   import gpumod.wrappers.sparse;
 */

module;

#include "dispatch_macros.h"

export module gpumod.wrappers.sparse:conversion;

import gpumod.sparse;
import gpumod.complex;
import gpumod.wrappers.common;
import std;

export namespace wwr {

// ========================================================================
// nnz: count nonzeros per row/column and in total, from a dense matrix
// ========================================================================

template<usual_fp T>
gpusparseStatus_t nnz(gpusparseHandle_t handle, gpusparseDirection_t dirA, int m, int n,
                      const gpusparseMatDescr_t descrA, const T *A, int lda, int *nnzPerRowCol,
                      int *nnzTotalDevHostPtr) {
  WWR_USUAL_DISPATCH(T, nnz, handle, dirA, m, n, descrA, A, lda, nnzPerRowCol,
                        nnzTotalDevHostPtr);
}

// ========================================================================
// gebsr2gebsc: general-BSR to general-BSC (block transpose)
// ========================================================================

template<usual_fp T>
gpusparseStatus_t gebsr2gebsc_bufferSize(gpusparseHandle_t handle, int mb, int nb, int nnzb,
                                         const T *bsrVal, const int *bsrRowPtr,
                                         const int *bsrColInd, int rowBlockDim, int colBlockDim,
                                         size_t *pBufferSizeInBytes) {
  WWR_USUAL_DISPATCH(T, gebsr2gebsc_bufferSize, handle, mb, nb, nnzb, bsrVal, bsrRowPtr,
                        bsrColInd, rowBlockDim, colBlockDim, pBufferSizeInBytes);
}

template<usual_fp T>
gpusparseStatus_t gebsr2gebsc(gpusparseHandle_t handle, int mb, int nb, int nnzb, const T *bsrVal,
                              const int *bsrRowPtr, const int *bsrColInd, int rowBlockDim,
                              int colBlockDim, T *bscVal, int *bscRowInd, int *bscColPtr,
                              gpusparseAction_t copyValues, gpusparseIndexBase_t idxBase,
                              void *pBuffer) {
  WWR_USUAL_DISPATCH(T, gebsr2gebsc, handle, mb, nb, nnzb, bsrVal, bsrRowPtr, bsrColInd,
                        rowBlockDim, colBlockDim, bscVal, bscRowInd, bscColPtr, copyValues, idxBase,
                        pBuffer);
}

// ========================================================================
// csr2gebsr: CSR to general-BSR
// ========================================================================

template<usual_fp T>
gpusparseStatus_t csr2gebsr_bufferSize(gpusparseHandle_t handle, gpusparseDirection_t dirA, int m,
                                       int n, const gpusparseMatDescr_t descrA,
                                       const T *csrSortedValA, const int *csrSortedRowPtrA,
                                       const int *csrSortedColIndA, int rowBlockDim,
                                       int colBlockDim, size_t *pBufferSizeInBytes) {
  WWR_USUAL_DISPATCH(T, csr2gebsr_bufferSize, handle, dirA, m, n, descrA, csrSortedValA,
                        csrSortedRowPtrA, csrSortedColIndA, rowBlockDim, colBlockDim,
                        pBufferSizeInBytes);
}

template<usual_fp T>
gpusparseStatus_t csr2gebsr(gpusparseHandle_t handle, gpusparseDirection_t dirA, int m, int n,
                            const gpusparseMatDescr_t descrA, const T *csrSortedValA,
                            const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                            const gpusparseMatDescr_t descrC, T *bsrSortedValC,
                            int *bsrSortedRowPtrC, int *bsrSortedColIndC, int rowBlockDim,
                            int colBlockDim, void *pBuffer) {
  WWR_USUAL_DISPATCH(T, csr2gebsr, handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                        csrSortedColIndA, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC,
                        rowBlockDim, colBlockDim, pBuffer);
}

// ==================== Explicit Template Instantiations ====================
// Matching `template` instantiations live in instantiations.cpp.

// Function: nnz
extern template gpusparseStatus_t nnz<float>(gpusparseHandle_t, gpusparseDirection_t, int, int,
                                             const gpusparseMatDescr_t, const float *, int, int *,
                                             int *);
extern template gpusparseStatus_t nnz<double>(gpusparseHandle_t, gpusparseDirection_t, int, int,
                                              const gpusparseMatDescr_t, const double *, int, int *,
                                              int *);
extern template gpusparseStatus_t nnz<gpuFloatComplex>(gpusparseHandle_t, gpusparseDirection_t, int,
                                                       int, const gpusparseMatDescr_t,
                                                       const gpuFloatComplex *, int, int *, int *);
extern template gpusparseStatus_t nnz<gpuDoubleComplex>(gpusparseHandle_t, gpusparseDirection_t,
                                                        int, int, const gpusparseMatDescr_t,
                                                        const gpuDoubleComplex *, int, int *,
                                                        int *);

// Function: gebsr2gebsc_bufferSize
extern template gpusparseStatus_t gebsr2gebsc_bufferSize<float>(gpusparseHandle_t, int, int, int,
                                                                const float *, const int *,
                                                                const int *, int, int, size_t *);
extern template gpusparseStatus_t gebsr2gebsc_bufferSize<double>(gpusparseHandle_t, int, int, int,
                                                                 const double *, const int *,
                                                                 const int *, int, int, size_t *);
extern template gpusparseStatus_t
gebsr2gebsc_bufferSize<gpuFloatComplex>(gpusparseHandle_t, int, int, int, const gpuFloatComplex *,
                                        const int *, const int *, int, int, size_t *);
extern template gpusparseStatus_t
gebsr2gebsc_bufferSize<gpuDoubleComplex>(gpusparseHandle_t, int, int, int, const gpuDoubleComplex *,
                                         const int *, const int *, int, int, size_t *);

// Function: gebsr2gebsc
extern template gpusparseStatus_t gebsr2gebsc<float>(gpusparseHandle_t, int, int, int,
                                                     const float *, const int *, const int *, int,
                                                     int, float *, int *, int *, gpusparseAction_t,
                                                     gpusparseIndexBase_t, void *);
extern template gpusparseStatus_t
gebsr2gebsc<double>(gpusparseHandle_t, int, int, int, const double *, const int *, const int *, int,
                    int, double *, int *, int *, gpusparseAction_t, gpusparseIndexBase_t, void *);
extern template gpusparseStatus_t
gebsr2gebsc<gpuFloatComplex>(gpusparseHandle_t, int, int, int, const gpuFloatComplex *, const int *,
                             const int *, int, int, gpuFloatComplex *, int *, int *,
                             gpusparseAction_t, gpusparseIndexBase_t, void *);
extern template gpusparseStatus_t
gebsr2gebsc<gpuDoubleComplex>(gpusparseHandle_t, int, int, int, const gpuDoubleComplex *,
                              const int *, const int *, int, int, gpuDoubleComplex *, int *, int *,
                              gpusparseAction_t, gpusparseIndexBase_t, void *);

// Function: csr2gebsr_bufferSize
extern template gpusparseStatus_t csr2gebsr_bufferSize<float>(gpusparseHandle_t,
                                                              gpusparseDirection_t, int, int,
                                                              const gpusparseMatDescr_t,
                                                              const float *, const int *,
                                                              const int *, int, int, size_t *);
extern template gpusparseStatus_t csr2gebsr_bufferSize<double>(gpusparseHandle_t,
                                                               gpusparseDirection_t, int, int,
                                                               const gpusparseMatDescr_t,
                                                               const double *, const int *,
                                                               const int *, int, int, size_t *);
extern template gpusparseStatus_t
csr2gebsr_bufferSize<gpuFloatComplex>(gpusparseHandle_t, gpusparseDirection_t, int, int,
                                      const gpusparseMatDescr_t, const gpuFloatComplex *,
                                      const int *, const int *, int, int, size_t *);
extern template gpusparseStatus_t
csr2gebsr_bufferSize<gpuDoubleComplex>(gpusparseHandle_t, gpusparseDirection_t, int, int,
                                       const gpusparseMatDescr_t, const gpuDoubleComplex *,
                                       const int *, const int *, int, int, size_t *);

// Function: csr2gebsr
extern template gpusparseStatus_t csr2gebsr<float>(gpusparseHandle_t, gpusparseDirection_t, int,
                                                   int, const gpusparseMatDescr_t, const float *,
                                                   const int *, const int *,
                                                   const gpusparseMatDescr_t, float *, int *, int *,
                                                   int, int, void *);
extern template gpusparseStatus_t csr2gebsr<double>(gpusparseHandle_t, gpusparseDirection_t, int,
                                                    int, const gpusparseMatDescr_t, const double *,
                                                    const int *, const int *,
                                                    const gpusparseMatDescr_t, double *, int *,
                                                    int *, int, int, void *);
extern template gpusparseStatus_t
csr2gebsr<gpuFloatComplex>(gpusparseHandle_t, gpusparseDirection_t, int, int,
                           const gpusparseMatDescr_t, const gpuFloatComplex *, const int *,
                           const int *, const gpusparseMatDescr_t, gpuFloatComplex *, int *, int *,
                           int, int, void *);
extern template gpusparseStatus_t
csr2gebsr<gpuDoubleComplex>(gpusparseHandle_t, gpusparseDirection_t, int, int,
                            const gpusparseMatDescr_t, const gpuDoubleComplex *, const int *,
                            const int *, const gpusparseMatDescr_t, gpuDoubleComplex *, int *,
                            int *, int, int, void *);

} // namespace wwr
