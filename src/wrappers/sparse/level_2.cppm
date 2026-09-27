/**
 * @file level_2.cppm
 * @brief GPU sparse Level 2 (matrix-vector) operations
 *
 * Type-safe wrappers for the BSR matrix-vector multiply.
 *
 * Usage:
 *   import wwr.wrappers.sparse;
 */

module;

#include "dispatch_macros.h"

export module wwr.wrappers.sparse:level_2;

import wwr.sparse;
import wwr.complex;
import wwr.wrappers.common;
import std;

export namespace wwr {

// ========================================================================
// BSR matrix-vector multiply: y = alpha*op(A)*x + beta*y
// ========================================================================

template<usual_fp T>
wwrsparseStatus_t bsrmv(wwrsparseHandle_t handle, wwrsparseDirection_t dirA,
                        wwrsparseOperation_t transA, int mb, int nb, int nnzb, const T *alpha,
                        const wwrsparseMatDescr_t descrA, const T *bsrSortedValA,
                        const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
                        const T *x, const T *beta, T *y) {
  WWR_USUAL_DISPATCH(T, bsrmv, handle, dirA, transA, mb, nb, nnzb, alpha, descrA, bsrSortedValA,
                        bsrSortedRowPtrA, bsrSortedColIndA, blockDim, x, beta, y);
}

// ==================== Explicit Template Instantiations ====================
// Hand-written, one per type. Matching `template` instantiations live in
// instantiations.cpp.

// Function: bsrmv
extern template wwrsparseStatus_t bsrmv<float>(wwrsparseHandle_t, wwrsparseDirection_t,
                                               wwrsparseOperation_t, int, int, int, const float *,
                                               const wwrsparseMatDescr_t, const float *,
                                               const int *, const int *, int, const float *,
                                               const float *, float *);
extern template wwrsparseStatus_t bsrmv<double>(wwrsparseHandle_t, wwrsparseDirection_t,
                                                wwrsparseOperation_t, int, int, int, const double *,
                                                const wwrsparseMatDescr_t, const double *,
                                                const int *, const int *, int, const double *,
                                                const double *, double *);
extern template wwrsparseStatus_t
bsrmv<wwrFloatComplex>(wwrsparseHandle_t, wwrsparseDirection_t, wwrsparseOperation_t, int, int, int,
                       const wwrFloatComplex *, const wwrsparseMatDescr_t, const wwrFloatComplex *,
                       const int *, const int *, int, const wwrFloatComplex *,
                       const wwrFloatComplex *, wwrFloatComplex *);
extern template wwrsparseStatus_t
bsrmv<wwrDoubleComplex>(wwrsparseHandle_t, wwrsparseDirection_t, wwrsparseOperation_t, int, int,
                        int, const wwrDoubleComplex *, const wwrsparseMatDescr_t,
                        const wwrDoubleComplex *, const int *, const int *, int,
                        const wwrDoubleComplex *, const wwrDoubleComplex *, wwrDoubleComplex *);

} // namespace wwr
