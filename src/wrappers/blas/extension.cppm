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
 *   import gpumod.wrappers.blas;
 */

module;

#include "dispatch_macros.h"

export module gpumod.wrappers.blas:extension;

import gpumod.blas;
import gpumod.complex;
import :type_traits;
import std;

export namespace wwr {

// ========================================================================
// Matrix addition/transposition: C = alpha*op(A) + beta*op(B)
// ========================================================================

template<usual_fp T, int_type IntT>
gpublasStatus_t geam(gpublasHandle_t handle, gpublasOperation_t transa, gpublasOperation_t transb,
                     IntT m, IntT n, const T *alpha, const T *A, IntT lda, const T *beta,
                     const T *B, IntT ldb, T *C, IntT ldc) {
  WWR_USUAL_DISPATCH_64(T, IntT, geam, handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb,
                           C, ldc);
}

// ========================================================================
// Diagonal matrix multiplication: C = A * diag(x) or C = diag(x) * A
// ========================================================================

template<usual_fp T, int_type IntT>
gpublasStatus_t dgmm(gpublasHandle_t handle, gpublasSideMode_t mode, IntT m, IntT n, const T *A,
                     IntT lda, const T *x, IntT incx, T *C, IntT ldc) {
  WWR_USUAL_DISPATCH_64(T, IntT, dgmm, handle, mode, m, n, A, lda, x, incx, C, ldc);
}

// ========================================================================
// Batched LU factorization: A = P * L * U
// ========================================================================

template<usual_fp T>
gpublasStatus_t getrfBatched(gpublasHandle_t handle, int n, T *const A[], int lda, int *P,
                             int *info, int batchSize) {
  WWR_USUAL_DISPATCH(T, getrfBatched, handle, n, A, lda, P, info, batchSize);
}

// ========================================================================
// Batched LU solver: op(A) * X = B
// ========================================================================

template<usual_fp T>
gpublasStatus_t getrsBatched(gpublasHandle_t handle, gpublasOperation_t trans, int n, int nrhs,
                             const T *const Aarray[], int lda, const int *devIpiv,
                             T *const Barray[], int ldb, int *info, int batchSize) {
  WWR_USUAL_DISPATCH(T, getrsBatched, handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb,
                        info, batchSize);
}

// ========================================================================
// Batched matrix inversion
// ========================================================================

template<usual_fp T>
gpublasStatus_t getriBatched(gpublasHandle_t handle, int n, const T *const A[], int lda,
                             const int *P, T *const C[], int ldc, int *info, int batchSize) {
  WWR_USUAL_DISPATCH(T, getriBatched, handle, n, A, lda, P, C, ldc, info, batchSize);
}

// ========================================================================
// Batched QR factorization
// ========================================================================

template<usual_fp T>
gpublasStatus_t geqrfBatched(gpublasHandle_t handle, int m, int n, T *const Aarray[], int lda,
                             T *const TauArray[], int *info, int batchSize) {
  WWR_USUAL_DISPATCH(T, geqrfBatched, handle, m, n, Aarray, lda, TauArray, info, batchSize);
}

// ========================================================================
// Batched least-squares solver
// ========================================================================

template<usual_fp T>
gpublasStatus_t gelsBatched(gpublasHandle_t handle, gpublasOperation_t trans, int m, int n,
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
extern template gpublasStatus_t geam<float, int>(gpublasHandle_t, gpublasOperation_t,
                                                 gpublasOperation_t, int, int, const float *,
                                                 const float *, int, const float *, const float *,
                                                 int, float *, int);
extern template gpublasStatus_t geam<float, int64_t>(gpublasHandle_t, gpublasOperation_t,
                                                     gpublasOperation_t, int64_t, int64_t,
                                                     const float *, const float *, int64_t,
                                                     const float *, const float *, int64_t, float *,
                                                     int64_t);
extern template gpublasStatus_t geam<double, int>(gpublasHandle_t, gpublasOperation_t,
                                                  gpublasOperation_t, int, int, const double *,
                                                  const double *, int, const double *,
                                                  const double *, int, double *, int);
extern template gpublasStatus_t geam<double, int64_t>(gpublasHandle_t, gpublasOperation_t,
                                                      gpublasOperation_t, int64_t, int64_t,
                                                      const double *, const double *, int64_t,
                                                      const double *, const double *, int64_t,
                                                      double *, int64_t);
extern template gpublasStatus_t
geam<gpuFloatComplex, int>(gpublasHandle_t, gpublasOperation_t, gpublasOperation_t, int, int,
                           const gpuFloatComplex *, const gpuFloatComplex *, int,
                           const gpuFloatComplex *, const gpuFloatComplex *, int, gpuFloatComplex *,
                           int);
extern template gpublasStatus_t
geam<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasOperation_t, gpublasOperation_t, int64_t,
                               int64_t, const gpuFloatComplex *, const gpuFloatComplex *, int64_t,
                               const gpuFloatComplex *, const gpuFloatComplex *, int64_t,
                               gpuFloatComplex *, int64_t);
extern template gpublasStatus_t
geam<gpuDoubleComplex, int>(gpublasHandle_t, gpublasOperation_t, gpublasOperation_t, int, int,
                            const gpuDoubleComplex *, const gpuDoubleComplex *, int,
                            const gpuDoubleComplex *, const gpuDoubleComplex *, int,
                            gpuDoubleComplex *, int);
extern template gpublasStatus_t
geam<gpuDoubleComplex, int64_t>(gpublasHandle_t, gpublasOperation_t, gpublasOperation_t, int64_t,
                                int64_t, const gpuDoubleComplex *, const gpuDoubleComplex *,
                                int64_t, const gpuDoubleComplex *, const gpuDoubleComplex *,
                                int64_t, gpuDoubleComplex *, int64_t);

// Function: dgmm
extern template gpublasStatus_t dgmm<float, int>(gpublasHandle_t, gpublasSideMode_t, int, int,
                                                 const float *, int, const float *, int, float *,
                                                 int);
extern template gpublasStatus_t dgmm<float, int64_t>(gpublasHandle_t, gpublasSideMode_t, int64_t,
                                                     int64_t, const float *, int64_t, const float *,
                                                     int64_t, float *, int64_t);
extern template gpublasStatus_t dgmm<double, int>(gpublasHandle_t, gpublasSideMode_t, int, int,
                                                  const double *, int, const double *, int,
                                                  double *, int);
extern template gpublasStatus_t dgmm<double, int64_t>(gpublasHandle_t, gpublasSideMode_t, int64_t,
                                                      int64_t, const double *, int64_t,
                                                      const double *, int64_t, double *, int64_t);
extern template gpublasStatus_t dgmm<gpuFloatComplex, int>(gpublasHandle_t, gpublasSideMode_t, int,
                                                           int, const gpuFloatComplex *, int,
                                                           const gpuFloatComplex *, int,
                                                           gpuFloatComplex *, int);
extern template gpublasStatus_t dgmm<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasSideMode_t,
                                                               int64_t, int64_t,
                                                               const gpuFloatComplex *, int64_t,
                                                               const gpuFloatComplex *, int64_t,
                                                               gpuFloatComplex *, int64_t);
extern template gpublasStatus_t dgmm<gpuDoubleComplex, int>(gpublasHandle_t, gpublasSideMode_t, int,
                                                            int, const gpuDoubleComplex *, int,
                                                            const gpuDoubleComplex *, int,
                                                            gpuDoubleComplex *, int);
extern template gpublasStatus_t dgmm<gpuDoubleComplex, int64_t>(gpublasHandle_t, gpublasSideMode_t,
                                                                int64_t, int64_t,
                                                                const gpuDoubleComplex *, int64_t,
                                                                const gpuDoubleComplex *, int64_t,
                                                                gpuDoubleComplex *, int64_t);

// Function: getrfBatched
extern template gpublasStatus_t getrfBatched<float>(gpublasHandle_t, int, float *const[], int,
                                                    int *, int *, int);
extern template gpublasStatus_t getrfBatched<double>(gpublasHandle_t, int, double *const[], int,
                                                     int *, int *, int);
extern template gpublasStatus_t getrfBatched<gpuFloatComplex>(gpublasHandle_t, int,
                                                              gpuFloatComplex *const[], int, int *,
                                                              int *, int);
extern template gpublasStatus_t getrfBatched<gpuDoubleComplex>(gpublasHandle_t, int,
                                                               gpuDoubleComplex *const[], int,
                                                               int *, int *, int);

// Function: getrsBatched
extern template gpublasStatus_t getrsBatched<float>(gpublasHandle_t, gpublasOperation_t, int, int,
                                                    const float *const[], int, const int *,
                                                    float *const[], int, int *, int);
extern template gpublasStatus_t getrsBatched<double>(gpublasHandle_t, gpublasOperation_t, int, int,
                                                     const double *const[], int, const int *,
                                                     double *const[], int, int *, int);
extern template gpublasStatus_t getrsBatched<gpuFloatComplex>(gpublasHandle_t, gpublasOperation_t,
                                                              int, int,
                                                              const gpuFloatComplex *const[], int,
                                                              const int *, gpuFloatComplex *const[],
                                                              int, int *, int);
extern template gpublasStatus_t
getrsBatched<gpuDoubleComplex>(gpublasHandle_t, gpublasOperation_t, int, int,
                               const gpuDoubleComplex *const[], int, const int *,
                               gpuDoubleComplex *const[], int, int *, int);

// Function: getriBatched
extern template gpublasStatus_t getriBatched<float>(gpublasHandle_t, int, const float *const[], int,
                                                    const int *, float *const[], int, int *, int);
extern template gpublasStatus_t getriBatched<double>(gpublasHandle_t, int, const double *const[],
                                                     int, const int *, double *const[], int, int *,
                                                     int);
extern template gpublasStatus_t getriBatched<gpuFloatComplex>(gpublasHandle_t, int,
                                                              const gpuFloatComplex *const[], int,
                                                              const int *, gpuFloatComplex *const[],
                                                              int, int *, int);
extern template gpublasStatus_t
getriBatched<gpuDoubleComplex>(gpublasHandle_t, int, const gpuDoubleComplex *const[], int,
                               const int *, gpuDoubleComplex *const[], int, int *, int);

// Function: geqrfBatched
extern template gpublasStatus_t geqrfBatched<float>(gpublasHandle_t, int, int, float *const[], int,
                                                    float *const[], int *, int);
extern template gpublasStatus_t geqrfBatched<double>(gpublasHandle_t, int, int, double *const[],
                                                     int, double *const[], int *, int);
extern template gpublasStatus_t geqrfBatched<gpuFloatComplex>(gpublasHandle_t, int, int,
                                                              gpuFloatComplex *const[], int,
                                                              gpuFloatComplex *const[], int *, int);
extern template gpublasStatus_t geqrfBatched<gpuDoubleComplex>(gpublasHandle_t, int, int,
                                                               gpuDoubleComplex *const[], int,
                                                               gpuDoubleComplex *const[], int *,
                                                               int);

// Function: gelsBatched
extern template gpublasStatus_t gelsBatched<float>(gpublasHandle_t, gpublasOperation_t, int, int,
                                                   int, float *const[], int, float *const[], int,
                                                   int *, int *, int);
extern template gpublasStatus_t gelsBatched<double>(gpublasHandle_t, gpublasOperation_t, int, int,
                                                    int, double *const[], int, double *const[], int,
                                                    int *, int *, int);
extern template gpublasStatus_t gelsBatched<gpuFloatComplex>(gpublasHandle_t, gpublasOperation_t,
                                                             int, int, int,
                                                             gpuFloatComplex *const[], int,
                                                             gpuFloatComplex *const[], int, int *,
                                                             int *, int);
extern template gpublasStatus_t gelsBatched<gpuDoubleComplex>(gpublasHandle_t, gpublasOperation_t,
                                                              int, int, int,
                                                              gpuDoubleComplex *const[], int,
                                                              gpuDoubleComplex *const[], int, int *,
                                                              int *, int);

} // namespace wwr
