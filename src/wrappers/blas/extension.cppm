/**
 * @file extension.cppm
 * @brief GPU BLAS BLAS-like extension operations
 *
 * This module provides type-safe wrappers for GPU BLAS extension operations including:
 * - Matrix addition/transposition (geam)
 * - Diagonal matrix multiplication (dgmm)
 * - Batched factorizations and solvers
 * - Triangular format conversions
 *
 * Usage:
 *   import wwr.wrappers.blas;
 */

module;

#include "dispatch_macros.h"

export module wwr.wrappers.blas:extension;

import wwr.blas;
import wwr.complex;
import :type_traits;
import std;

export namespace wwr {

// ========================================================================
// Matrix addition/transposition: C = alpha*op(A) + beta*op(B)
// ========================================================================

template<usual_fp T, int_type IntT>
wwrblasStatus_t geam(wwrblasHandle_t handle, wwrblasOperation_t transa, wwrblasOperation_t transb,
                     IntT m, IntT n, const T *alpha, const T *A, IntT lda, const T *beta,
                     const T *B, IntT ldb, T *C, IntT ldc) {
  WWR_USUAL_DISPATCH_64(T, IntT, geam, handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb,
                           C, ldc);
}

// ========================================================================
// Diagonal matrix multiplication: C = A * diag(x) or C = diag(x) * A
// ========================================================================

template<usual_fp T, int_type IntT>
wwrblasStatus_t dgmm(wwrblasHandle_t handle, wwrblasSideMode_t mode, IntT m, IntT n, const T *A,
                     IntT lda, const T *x, IntT incx, T *C, IntT ldc) {
  WWR_USUAL_DISPATCH_64(T, IntT, dgmm, handle, mode, m, n, A, lda, x, incx, C, ldc);
}

// ========================================================================
// Batched LU factorization: A = P * L * U
// ========================================================================

template<usual_fp T>
wwrblasStatus_t getrfBatched(wwrblasHandle_t handle, int n, T *const A[], int lda, int *P,
                             int *info, int batchSize) {
  WWR_USUAL_DISPATCH(T, getrfBatched, handle, n, A, lda, P, info, batchSize);
}

// ========================================================================
// Batched LU solver: op(A) * X = B
// ========================================================================

template<usual_fp T>
wwrblasStatus_t getrsBatched(wwrblasHandle_t handle, wwrblasOperation_t trans, int n, int nrhs,
                             const T *const Aarray[], int lda, const int *devIpiv,
                             T *const Barray[], int ldb, int *info, int batchSize) {
  WWR_USUAL_DISPATCH(T, getrsBatched, handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb,
                        info, batchSize);
}

// ========================================================================
// Batched matrix inversion
// ========================================================================

template<usual_fp T>
wwrblasStatus_t getriBatched(wwrblasHandle_t handle, int n, const T *const A[], int lda,
                             const int *P, T *const C[], int ldc, int *info, int batchSize) {
  WWR_USUAL_DISPATCH(T, getriBatched, handle, n, A, lda, P, C, ldc, info, batchSize);
}

// ========================================================================
// Batched QR factorization
// ========================================================================

template<usual_fp T>
wwrblasStatus_t geqrfBatched(wwrblasHandle_t handle, int m, int n, T *const Aarray[], int lda,
                             T *const TauArray[], int *info, int batchSize) {
  WWR_USUAL_DISPATCH(T, geqrfBatched, handle, m, n, Aarray, lda, TauArray, info, batchSize);
}

// ========================================================================
// Batched least-squares solver
// ========================================================================

template<usual_fp T>
wwrblasStatus_t gelsBatched(wwrblasHandle_t handle, wwrblasOperation_t trans, int m, int n,
                            int nrhs, T *const Aarray[], int lda, T *const Carray[], int ldc,
                            int *info, int *devInfoArray, int batchSize) {
  WWR_USUAL_DISPATCH(T, gelsBatched, handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info,
                        devInfoArray, batchSize);
}

// ==================== Explicit Template Instantiations ====================
// Hand-written, one per (function, type, index width). Matching `template`
// instantiations live in instantiations.cpp -- see that file, and follow
// math/triple_gemm when adding a new function.

// Function: geam
extern template wwrblasStatus_t geam<float, int>(wwrblasHandle_t, wwrblasOperation_t,
                                                 wwrblasOperation_t, int, int, const float *,
                                                 const float *, int, const float *, const float *,
                                                 int, float *, int);
extern template wwrblasStatus_t geam<float, int64_t>(wwrblasHandle_t, wwrblasOperation_t,
                                                     wwrblasOperation_t, int64_t, int64_t,
                                                     const float *, const float *, int64_t,
                                                     const float *, const float *, int64_t, float *,
                                                     int64_t);
extern template wwrblasStatus_t geam<double, int>(wwrblasHandle_t, wwrblasOperation_t,
                                                  wwrblasOperation_t, int, int, const double *,
                                                  const double *, int, const double *,
                                                  const double *, int, double *, int);
extern template wwrblasStatus_t geam<double, int64_t>(wwrblasHandle_t, wwrblasOperation_t,
                                                      wwrblasOperation_t, int64_t, int64_t,
                                                      const double *, const double *, int64_t,
                                                      const double *, const double *, int64_t,
                                                      double *, int64_t);
extern template wwrblasStatus_t
geam<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasOperation_t, wwrblasOperation_t, int, int,
                           const wwrFloatComplex *, const wwrFloatComplex *, int,
                           const wwrFloatComplex *, const wwrFloatComplex *, int, wwrFloatComplex *,
                           int);
extern template wwrblasStatus_t
geam<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasOperation_t, wwrblasOperation_t, int64_t,
                               int64_t, const wwrFloatComplex *, const wwrFloatComplex *, int64_t,
                               const wwrFloatComplex *, const wwrFloatComplex *, int64_t,
                               wwrFloatComplex *, int64_t);
extern template wwrblasStatus_t
geam<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasOperation_t, wwrblasOperation_t, int, int,
                            const wwrDoubleComplex *, const wwrDoubleComplex *, int,
                            const wwrDoubleComplex *, const wwrDoubleComplex *, int,
                            wwrDoubleComplex *, int);
extern template wwrblasStatus_t
geam<wwrDoubleComplex, int64_t>(wwrblasHandle_t, wwrblasOperation_t, wwrblasOperation_t, int64_t,
                                int64_t, const wwrDoubleComplex *, const wwrDoubleComplex *,
                                int64_t, const wwrDoubleComplex *, const wwrDoubleComplex *,
                                int64_t, wwrDoubleComplex *, int64_t);

// Function: dgmm
extern template wwrblasStatus_t dgmm<float, int>(wwrblasHandle_t, wwrblasSideMode_t, int, int,
                                                 const float *, int, const float *, int, float *,
                                                 int);
extern template wwrblasStatus_t dgmm<float, int64_t>(wwrblasHandle_t, wwrblasSideMode_t, int64_t,
                                                     int64_t, const float *, int64_t, const float *,
                                                     int64_t, float *, int64_t);
extern template wwrblasStatus_t dgmm<double, int>(wwrblasHandle_t, wwrblasSideMode_t, int, int,
                                                  const double *, int, const double *, int,
                                                  double *, int);
extern template wwrblasStatus_t dgmm<double, int64_t>(wwrblasHandle_t, wwrblasSideMode_t, int64_t,
                                                      int64_t, const double *, int64_t,
                                                      const double *, int64_t, double *, int64_t);
extern template wwrblasStatus_t dgmm<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasSideMode_t, int,
                                                           int, const wwrFloatComplex *, int,
                                                           const wwrFloatComplex *, int,
                                                           wwrFloatComplex *, int);
extern template wwrblasStatus_t dgmm<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasSideMode_t,
                                                               int64_t, int64_t,
                                                               const wwrFloatComplex *, int64_t,
                                                               const wwrFloatComplex *, int64_t,
                                                               wwrFloatComplex *, int64_t);
extern template wwrblasStatus_t dgmm<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasSideMode_t, int,
                                                            int, const wwrDoubleComplex *, int,
                                                            const wwrDoubleComplex *, int,
                                                            wwrDoubleComplex *, int);
extern template wwrblasStatus_t dgmm<wwrDoubleComplex, int64_t>(wwrblasHandle_t, wwrblasSideMode_t,
                                                                int64_t, int64_t,
                                                                const wwrDoubleComplex *, int64_t,
                                                                const wwrDoubleComplex *, int64_t,
                                                                wwrDoubleComplex *, int64_t);

// Function: getrfBatched
extern template wwrblasStatus_t getrfBatched<float>(wwrblasHandle_t, int, float *const[], int,
                                                    int *, int *, int);
extern template wwrblasStatus_t getrfBatched<double>(wwrblasHandle_t, int, double *const[], int,
                                                     int *, int *, int);
extern template wwrblasStatus_t getrfBatched<wwrFloatComplex>(wwrblasHandle_t, int,
                                                              wwrFloatComplex *const[], int, int *,
                                                              int *, int);
extern template wwrblasStatus_t getrfBatched<wwrDoubleComplex>(wwrblasHandle_t, int,
                                                               wwrDoubleComplex *const[], int,
                                                               int *, int *, int);

// Function: getrsBatched
extern template wwrblasStatus_t getrsBatched<float>(wwrblasHandle_t, wwrblasOperation_t, int, int,
                                                    const float *const[], int, const int *,
                                                    float *const[], int, int *, int);
extern template wwrblasStatus_t getrsBatched<double>(wwrblasHandle_t, wwrblasOperation_t, int, int,
                                                     const double *const[], int, const int *,
                                                     double *const[], int, int *, int);
extern template wwrblasStatus_t getrsBatched<wwrFloatComplex>(wwrblasHandle_t, wwrblasOperation_t,
                                                              int, int,
                                                              const wwrFloatComplex *const[], int,
                                                              const int *, wwrFloatComplex *const[],
                                                              int, int *, int);
extern template wwrblasStatus_t
getrsBatched<wwrDoubleComplex>(wwrblasHandle_t, wwrblasOperation_t, int, int,
                               const wwrDoubleComplex *const[], int, const int *,
                               wwrDoubleComplex *const[], int, int *, int);

// Function: getriBatched
extern template wwrblasStatus_t getriBatched<float>(wwrblasHandle_t, int, const float *const[], int,
                                                    const int *, float *const[], int, int *, int);
extern template wwrblasStatus_t getriBatched<double>(wwrblasHandle_t, int, const double *const[],
                                                     int, const int *, double *const[], int, int *,
                                                     int);
extern template wwrblasStatus_t getriBatched<wwrFloatComplex>(wwrblasHandle_t, int,
                                                              const wwrFloatComplex *const[], int,
                                                              const int *, wwrFloatComplex *const[],
                                                              int, int *, int);
extern template wwrblasStatus_t
getriBatched<wwrDoubleComplex>(wwrblasHandle_t, int, const wwrDoubleComplex *const[], int,
                               const int *, wwrDoubleComplex *const[], int, int *, int);

// Function: geqrfBatched
extern template wwrblasStatus_t geqrfBatched<float>(wwrblasHandle_t, int, int, float *const[], int,
                                                    float *const[], int *, int);
extern template wwrblasStatus_t geqrfBatched<double>(wwrblasHandle_t, int, int, double *const[],
                                                     int, double *const[], int *, int);
extern template wwrblasStatus_t geqrfBatched<wwrFloatComplex>(wwrblasHandle_t, int, int,
                                                              wwrFloatComplex *const[], int,
                                                              wwrFloatComplex *const[], int *, int);
extern template wwrblasStatus_t geqrfBatched<wwrDoubleComplex>(wwrblasHandle_t, int, int,
                                                               wwrDoubleComplex *const[], int,
                                                               wwrDoubleComplex *const[], int *,
                                                               int);

// Function: gelsBatched
extern template wwrblasStatus_t gelsBatched<float>(wwrblasHandle_t, wwrblasOperation_t, int, int,
                                                   int, float *const[], int, float *const[], int,
                                                   int *, int *, int);
extern template wwrblasStatus_t gelsBatched<double>(wwrblasHandle_t, wwrblasOperation_t, int, int,
                                                    int, double *const[], int, double *const[], int,
                                                    int *, int *, int);
extern template wwrblasStatus_t gelsBatched<wwrFloatComplex>(wwrblasHandle_t, wwrblasOperation_t,
                                                             int, int, int,
                                                             wwrFloatComplex *const[], int,
                                                             wwrFloatComplex *const[], int, int *,
                                                             int *, int);
extern template wwrblasStatus_t gelsBatched<wwrDoubleComplex>(wwrblasHandle_t, wwrblasOperation_t,
                                                              int, int, int,
                                                              wwrDoubleComplex *const[], int,
                                                              wwrDoubleComplex *const[], int, int *,
                                                              int *, int);

} // namespace wwr
