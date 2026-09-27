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
wwrsparseStatus_t csrgeam2_bufferSizeExt(
    wwrsparseHandle_t handle, int m, int n, const T *alpha, const wwrsparseMatDescr_t descrA,
    int nnzA, const T *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const T *beta, const wwrsparseMatDescr_t descrB, int nnzB, const T *csrSortedValB,
    const int *csrSortedRowPtrB, const int *csrSortedColIndB, const wwrsparseMatDescr_t descrC,
    const T *csrSortedValC, const int *csrSortedRowPtrC, const int *csrSortedColIndC,
    size_t *pBufferSizeInBytes) {
  WWR_USUAL_DISPATCH(T, csrgeam2_bufferSizeExt, handle, m, n, alpha, descrA, nnzA, csrSortedValA,
                        csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB,
                        csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC,
                        csrSortedColIndC, pBufferSizeInBytes);
}

template<usual_fp T>
wwrsparseStatus_t
csrgeam2(wwrsparseHandle_t handle, int m, int n, const T *alpha, const wwrsparseMatDescr_t descrA,
         int nnzA, const T *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA,
         const T *beta, const wwrsparseMatDescr_t descrB, int nnzB, const T *csrSortedValB,
         const int *csrSortedRowPtrB, const int *csrSortedColIndB, const wwrsparseMatDescr_t descrC,
         T *csrSortedValC, int *csrSortedRowPtrC, int *csrSortedColIndC, void *pBuffer) {
  WWR_USUAL_DISPATCH(T, csrgeam2, handle, m, n, alpha, descrA, nnzA, csrSortedValA,
                        csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB,
                        csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC,
                        csrSortedColIndC, pBuffer);
}

// ==================== Explicit Template Instantiations ====================
// Matching `template` instantiations live in instantiations.cpp.

// Function: csrgeam2_bufferSizeExt
extern template wwrsparseStatus_t
csrgeam2_bufferSizeExt<float>(wwrsparseHandle_t, int, int, const float *, const wwrsparseMatDescr_t,
                              int, const float *, const int *, const int *, const float *,
                              const wwrsparseMatDescr_t, int, const float *, const int *,
                              const int *, const wwrsparseMatDescr_t, const float *, const int *,
                              const int *, size_t *);
extern template wwrsparseStatus_t
csrgeam2_bufferSizeExt<double>(wwrsparseHandle_t, int, int, const double *,
                               const wwrsparseMatDescr_t, int, const double *, const int *,
                               const int *, const double *, const wwrsparseMatDescr_t, int,
                               const double *, const int *, const int *, const wwrsparseMatDescr_t,
                               const double *, const int *, const int *, size_t *);
extern template wwrsparseStatus_t csrgeam2_bufferSizeExt<wwrFloatComplex>(
    wwrsparseHandle_t, int, int, const wwrFloatComplex *, const wwrsparseMatDescr_t, int,
    const wwrFloatComplex *, const int *, const int *, const wwrFloatComplex *,
    const wwrsparseMatDescr_t, int, const wwrFloatComplex *, const int *, const int *,
    const wwrsparseMatDescr_t, const wwrFloatComplex *, const int *, const int *, size_t *);
extern template wwrsparseStatus_t csrgeam2_bufferSizeExt<wwrDoubleComplex>(
    wwrsparseHandle_t, int, int, const wwrDoubleComplex *, const wwrsparseMatDescr_t, int,
    const wwrDoubleComplex *, const int *, const int *, const wwrDoubleComplex *,
    const wwrsparseMatDescr_t, int, const wwrDoubleComplex *, const int *, const int *,
    const wwrsparseMatDescr_t, const wwrDoubleComplex *, const int *, const int *, size_t *);

// Function: csrgeam2
extern template wwrsparseStatus_t
csrgeam2<float>(wwrsparseHandle_t, int, int, const float *, const wwrsparseMatDescr_t, int,
                const float *, const int *, const int *, const float *, const wwrsparseMatDescr_t,
                int, const float *, const int *, const int *, const wwrsparseMatDescr_t, float *,
                int *, int *, void *);
extern template wwrsparseStatus_t
csrgeam2<double>(wwrsparseHandle_t, int, int, const double *, const wwrsparseMatDescr_t, int,
                 const double *, const int *, const int *, const double *,
                 const wwrsparseMatDescr_t, int, const double *, const int *, const int *,
                 const wwrsparseMatDescr_t, double *, int *, int *, void *);
extern template wwrsparseStatus_t
csrgeam2<wwrFloatComplex>(wwrsparseHandle_t, int, int, const wwrFloatComplex *,
                          const wwrsparseMatDescr_t, int, const wwrFloatComplex *, const int *,
                          const int *, const wwrFloatComplex *, const wwrsparseMatDescr_t, int,
                          const wwrFloatComplex *, const int *, const int *,
                          const wwrsparseMatDescr_t, wwrFloatComplex *, int *, int *, void *);
extern template wwrsparseStatus_t
csrgeam2<wwrDoubleComplex>(wwrsparseHandle_t, int, int, const wwrDoubleComplex *,
                           const wwrsparseMatDescr_t, int, const wwrDoubleComplex *, const int *,
                           const int *, const wwrDoubleComplex *, const wwrsparseMatDescr_t, int,
                           const wwrDoubleComplex *, const int *, const int *,
                           const wwrsparseMatDescr_t, wwrDoubleComplex *, int *, int *, void *);

} // namespace wwr
