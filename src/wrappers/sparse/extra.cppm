/**
 * @file extra.cppm
 * @brief GPU sparse extra routines: CSR matrix addition (csrgeam2)
 *
 * Type-safe wrappers for C = alpha*A + beta*B on CSR matrices, with its
 * companion buffer-size query. (The structure query Xcsrgeam2Nnz is untyped and
 * so is called directly through wwr.sparse's raw module.)
 *
 * Usage:
 *   import wwr.wrappers.sparse;
 */

module;

#include "dispatch_macros.h"

export module wwr.wrappers.sparse:extra;

import wwr.sparse;
import wwr.complex;
import wwr.wrappers.common;
import std;

export namespace wwr {

// ========================================================================
// csrgeam2: C = alpha*A + beta*B (CSR)
// ========================================================================

template<usual_fp T>
gpusparseStatus_t csrgeam2_bufferSizeExt(
    gpusparseHandle_t handle, int m, int n, const T *alpha, const gpusparseMatDescr_t descrA,
    int nnzA, const T *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const T *beta, const gpusparseMatDescr_t descrB, int nnzB, const T *csrSortedValB,
    const int *csrSortedRowPtrB, const int *csrSortedColIndB, const gpusparseMatDescr_t descrC,
    const T *csrSortedValC, const int *csrSortedRowPtrC, const int *csrSortedColIndC,
    size_t *pBufferSizeInBytes) {
  WWR_USUAL_DISPATCH(T, csrgeam2_bufferSizeExt, handle, m, n, alpha, descrA, nnzA, csrSortedValA,
                        csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB,
                        csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC,
                        csrSortedColIndC, pBufferSizeInBytes);
}

template<usual_fp T>
gpusparseStatus_t
csrgeam2(gpusparseHandle_t handle, int m, int n, const T *alpha, const gpusparseMatDescr_t descrA,
         int nnzA, const T *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA,
         const T *beta, const gpusparseMatDescr_t descrB, int nnzB, const T *csrSortedValB,
         const int *csrSortedRowPtrB, const int *csrSortedColIndB, const gpusparseMatDescr_t descrC,
         T *csrSortedValC, int *csrSortedRowPtrC, int *csrSortedColIndC, void *pBuffer) {
  WWR_USUAL_DISPATCH(T, csrgeam2, handle, m, n, alpha, descrA, nnzA, csrSortedValA,
                        csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB,
                        csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC,
                        csrSortedColIndC, pBuffer);
}

// ==================== Explicit Template Instantiations ====================
// Matching `template` instantiations live in instantiations.cpp.

// Function: csrgeam2_bufferSizeExt
extern template gpusparseStatus_t
csrgeam2_bufferSizeExt<float>(gpusparseHandle_t, int, int, const float *, const gpusparseMatDescr_t,
                              int, const float *, const int *, const int *, const float *,
                              const gpusparseMatDescr_t, int, const float *, const int *,
                              const int *, const gpusparseMatDescr_t, const float *, const int *,
                              const int *, size_t *);
extern template gpusparseStatus_t
csrgeam2_bufferSizeExt<double>(gpusparseHandle_t, int, int, const double *,
                               const gpusparseMatDescr_t, int, const double *, const int *,
                               const int *, const double *, const gpusparseMatDescr_t, int,
                               const double *, const int *, const int *, const gpusparseMatDescr_t,
                               const double *, const int *, const int *, size_t *);
extern template gpusparseStatus_t csrgeam2_bufferSizeExt<gpuFloatComplex>(
    gpusparseHandle_t, int, int, const gpuFloatComplex *, const gpusparseMatDescr_t, int,
    const gpuFloatComplex *, const int *, const int *, const gpuFloatComplex *,
    const gpusparseMatDescr_t, int, const gpuFloatComplex *, const int *, const int *,
    const gpusparseMatDescr_t, const gpuFloatComplex *, const int *, const int *, size_t *);
extern template gpusparseStatus_t csrgeam2_bufferSizeExt<gpuDoubleComplex>(
    gpusparseHandle_t, int, int, const gpuDoubleComplex *, const gpusparseMatDescr_t, int,
    const gpuDoubleComplex *, const int *, const int *, const gpuDoubleComplex *,
    const gpusparseMatDescr_t, int, const gpuDoubleComplex *, const int *, const int *,
    const gpusparseMatDescr_t, const gpuDoubleComplex *, const int *, const int *, size_t *);

// Function: csrgeam2
extern template gpusparseStatus_t
csrgeam2<float>(gpusparseHandle_t, int, int, const float *, const gpusparseMatDescr_t, int,
                const float *, const int *, const int *, const float *, const gpusparseMatDescr_t,
                int, const float *, const int *, const int *, const gpusparseMatDescr_t, float *,
                int *, int *, void *);
extern template gpusparseStatus_t
csrgeam2<double>(gpusparseHandle_t, int, int, const double *, const gpusparseMatDescr_t, int,
                 const double *, const int *, const int *, const double *,
                 const gpusparseMatDescr_t, int, const double *, const int *, const int *,
                 const gpusparseMatDescr_t, double *, int *, int *, void *);
extern template gpusparseStatus_t
csrgeam2<gpuFloatComplex>(gpusparseHandle_t, int, int, const gpuFloatComplex *,
                          const gpusparseMatDescr_t, int, const gpuFloatComplex *, const int *,
                          const int *, const gpuFloatComplex *, const gpusparseMatDescr_t, int,
                          const gpuFloatComplex *, const int *, const int *,
                          const gpusparseMatDescr_t, gpuFloatComplex *, int *, int *, void *);
extern template gpusparseStatus_t
csrgeam2<gpuDoubleComplex>(gpusparseHandle_t, int, int, const gpuDoubleComplex *,
                           const gpusparseMatDescr_t, int, const gpuDoubleComplex *, const int *,
                           const int *, const gpuDoubleComplex *, const gpusparseMatDescr_t, int,
                           const gpuDoubleComplex *, const int *, const int *,
                           const gpusparseMatDescr_t, gpuDoubleComplex *, int *, int *, void *);

} // namespace wwr
