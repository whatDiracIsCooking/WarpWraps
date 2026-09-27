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
gpusparseStatus_t bsrmv(gpusparseHandle_t handle, gpusparseDirection_t dirA,
                        gpusparseOperation_t transA, int mb, int nb, int nnzb, const T *alpha,
                        const gpusparseMatDescr_t descrA, const T *bsrSortedValA,
                        const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
                        const T *x, const T *beta, T *y) {
  WWR_USUAL_DISPATCH(T, bsrmv, handle, dirA, transA, mb, nb, nnzb, alpha, descrA, bsrSortedValA,
                        bsrSortedRowPtrA, bsrSortedColIndA, blockDim, x, beta, y);
}

// ==================== Explicit Template Instantiations ====================
// Hand-written, one per type. Matching `template` instantiations live in
// instantiations.cpp.

// Function: bsrmv
extern template gpusparseStatus_t bsrmv<float>(gpusparseHandle_t, gpusparseDirection_t,
                                               gpusparseOperation_t, int, int, int, const float *,
                                               const gpusparseMatDescr_t, const float *,
                                               const int *, const int *, int, const float *,
                                               const float *, float *);
extern template gpusparseStatus_t bsrmv<double>(gpusparseHandle_t, gpusparseDirection_t,
                                                gpusparseOperation_t, int, int, int, const double *,
                                                const gpusparseMatDescr_t, const double *,
                                                const int *, const int *, int, const double *,
                                                const double *, double *);
extern template gpusparseStatus_t
bsrmv<gpuFloatComplex>(gpusparseHandle_t, gpusparseDirection_t, gpusparseOperation_t, int, int, int,
                       const gpuFloatComplex *, const gpusparseMatDescr_t, const gpuFloatComplex *,
                       const int *, const int *, int, const gpuFloatComplex *,
                       const gpuFloatComplex *, gpuFloatComplex *);
extern template gpusparseStatus_t
bsrmv<gpuDoubleComplex>(gpusparseHandle_t, gpusparseDirection_t, gpusparseOperation_t, int, int,
                        int, const gpuDoubleComplex *, const gpusparseMatDescr_t,
                        const gpuDoubleComplex *, const int *, const int *, int,
                        const gpuDoubleComplex *, const gpuDoubleComplex *, gpuDoubleComplex *);

} // namespace wwr
