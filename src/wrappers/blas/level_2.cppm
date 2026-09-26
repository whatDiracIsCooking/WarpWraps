/**
 * @file level_2.cppm
 * @brief GPU BLAS Level 2 (matrix-vector) operations
 *
 * This module provides type-safe wrappers for GPU BLAS Level 2 BLAS operations.
 *
 * Usage:
 *   import gpumod.wrappers.blas;
 */

module;

#include "dispatch_macros.h"

export module gpumod.wrappers.blas:level_2;

import gpumod.blas;
import gpumod.complex;
import :type_traits;
import std;

export namespace gpumod {

// ========================================================================
// General matrix-vector multiplication: y = alpha*op(A)*x + beta*y
// ========================================================================

template<usual_fp T, int_type IntT>
gpublasStatus_t gemv(gpublasHandle_t handle, gpublasOperation_t trans, IntT m, IntT n,
                     const T *alpha, const T *A, IntT lda, const T *x, IntT incx, const T *beta,
                     T *y, IntT incy) {
  GPUMOD_USUAL_DISPATCH_64(T, IntT, gemv, handle, trans, m, n, alpha, A, lda, x, incx, beta, y,
                           incy);
}

// ========================================================================
// General banded matrix-vector multiplication: y = alpha*op(A)*x + beta*y
// ========================================================================

template<usual_fp T, int_type IntT>
gpublasStatus_t gbmv(gpublasHandle_t handle, gpublasOperation_t trans, IntT m, IntT n, IntT kl,
                     IntT ku, const T *alpha, const T *A, IntT lda, const T *x, IntT incx,
                     const T *beta, T *y, IntT incy) {
  GPUMOD_USUAL_DISPATCH_64(T, IntT, gbmv, handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta,
                           y, incy);
}

// ========================================================================
// General rank-1 update: A = alpha*x*y^T + A (real)
// ========================================================================

template<real_fp T, int_type IntT>
gpublasStatus_t ger(gpublasHandle_t handle, IntT m, IntT n, const T *alpha, const T *x, IntT incx,
                    const T *y, IntT incy, T *A, IntT lda) {
  GPUMOD_REAL_DISPATCH_64(T, IntT, gpublas, S, D, ger, handle, m, n, alpha, x, incx, y, incy, A,
                          lda);
}

// ========================================================================
// General rank-1 update (complex, unconjugated): A = alpha*x*y^T + A
// ========================================================================

template<complex_fp T, int_type IntT>
gpublasStatus_t geru(gpublasHandle_t handle, IntT m, IntT n, const T *alpha, const T *x, IntT incx,
                     const T *y, IntT incy, T *A, IntT lda) {
  GPUMOD_COMPLEX_DISPATCH_64(T, IntT, gpublas, C, Z, geru, handle, m, n, alpha, x, incx, y, incy, A,
                             lda);
}

// ========================================================================
// General rank-1 update (complex, conjugated): A = alpha*x*y^H + A
// ========================================================================

template<complex_fp T, int_type IntT>
gpublasStatus_t gerc(gpublasHandle_t handle, IntT m, IntT n, const T *alpha, const T *x, IntT incx,
                     const T *y, IntT incy, T *A, IntT lda) {
  GPUMOD_COMPLEX_DISPATCH_64(T, IntT, gpublas, C, Z, gerc, handle, m, n, alpha, x, incx, y, incy, A,
                             lda);
}

// ========================================================================
// Symmetric matrix-vector multiplication: y = alpha*A*x + beta*y
// ========================================================================

template<real_fp T, int_type IntT>
gpublasStatus_t symv(gpublasHandle_t handle, gpublasFillMode_t uplo, IntT n, const T *alpha,
                     const T *A, IntT lda, const T *x, IntT incx, const T *beta, T *y, IntT incy) {
  GPUMOD_REAL_DISPATCH_64(T, IntT, gpublas, S, D, symv, handle, uplo, n, alpha, A, lda, x, incx,
                          beta, y, incy);
}

// ========================================================================
// Symmetric rank-1 update: A = alpha*x*x^T + A
// ========================================================================

template<real_fp T, int_type IntT>
gpublasStatus_t syr(gpublasHandle_t handle, gpublasFillMode_t uplo, IntT n, const T *alpha,
                    const T *x, IntT incx, T *A, IntT lda) {
  GPUMOD_REAL_DISPATCH_64(T, IntT, gpublas, S, D, syr, handle, uplo, n, alpha, x, incx, A, lda);
}

// ========================================================================
// Symmetric rank-2 update: A = alpha*x*y^T + alpha*y*x^T + A
// ========================================================================

template<real_fp T, int_type IntT>
gpublasStatus_t syr2(gpublasHandle_t handle, gpublasFillMode_t uplo, IntT n, const T *alpha,
                     const T *x, IntT incx, const T *y, IntT incy, T *A, IntT lda) {
  GPUMOD_REAL_DISPATCH_64(T, IntT, gpublas, S, D, syr2, handle, uplo, n, alpha, x, incx, y, incy, A,
                          lda);
}

// ========================================================================
// Symmetric banded matrix-vector multiplication: y = alpha*A*x + beta*y
// ========================================================================

template<real_fp T, int_type IntT>
gpublasStatus_t sbmv(gpublasHandle_t handle, gpublasFillMode_t uplo, IntT n, IntT k, const T *alpha,
                     const T *A, IntT lda, const T *x, IntT incx, const T *beta, T *y, IntT incy) {
  GPUMOD_REAL_DISPATCH_64(T, IntT, gpublas, S, D, sbmv, handle, uplo, n, k, alpha, A, lda, x, incx,
                          beta, y, incy);
}

// ========================================================================
// Symmetric packed matrix-vector multiplication: y = alpha*A*x + beta*y
// ========================================================================

template<real_fp T, int_type IntT>
gpublasStatus_t spmv(gpublasHandle_t handle, gpublasFillMode_t uplo, IntT n, const T *alpha,
                     const T *AP, const T *x, IntT incx, const T *beta, T *y, IntT incy) {
  GPUMOD_REAL_DISPATCH_64(T, IntT, gpublas, S, D, spmv, handle, uplo, n, alpha, AP, x, incx, beta,
                          y, incy);
}

// ========================================================================
// Symmetric packed rank-1 update: A = alpha*x*x^T + A
// ========================================================================

template<real_fp T, int_type IntT>
gpublasStatus_t spr(gpublasHandle_t handle, gpublasFillMode_t uplo, IntT n, const T *alpha,
                    const T *x, IntT incx, T *AP) {
  GPUMOD_REAL_DISPATCH_64(T, IntT, gpublas, S, D, spr, handle, uplo, n, alpha, x, incx, AP);
}

// ========================================================================
// Symmetric packed rank-2 update: A = alpha*x*y^T + alpha*y*x^T + A
// ========================================================================

template<real_fp T, int_type IntT>
gpublasStatus_t spr2(gpublasHandle_t handle, gpublasFillMode_t uplo, IntT n, const T *alpha,
                     const T *x, IntT incx, const T *y, IntT incy, T *AP) {
  GPUMOD_REAL_DISPATCH_64(T, IntT, gpublas, S, D, spr2, handle, uplo, n, alpha, x, incx, y, incy,
                          AP);
}

// ========================================================================
// Triangular matrix-vector multiplication: x = op(A)*x
// ========================================================================

template<usual_fp T, int_type IntT>
gpublasStatus_t trmv(gpublasHandle_t handle, gpublasFillMode_t uplo, gpublasOperation_t trans,
                     gpublasDiagType_t diag, IntT n, const T *A, IntT lda, T *x, IntT incx) {
  GPUMOD_USUAL_DISPATCH_64(T, IntT, trmv, handle, uplo, trans, diag, n, A, lda, x, incx);
}

// ========================================================================
// Triangular solve: op(A)*x = b
// ========================================================================

template<usual_fp T, int_type IntT>
gpublasStatus_t trsv(gpublasHandle_t handle, gpublasFillMode_t uplo, gpublasOperation_t trans,
                     gpublasDiagType_t diag, IntT n, const T *A, IntT lda, T *x, IntT incx) {
  GPUMOD_USUAL_DISPATCH_64(T, IntT, trsv, handle, uplo, trans, diag, n, A, lda, x, incx);
}

// ========================================================================
// Triangular banded matrix-vector multiplication: x = op(A)*x
// ========================================================================

template<usual_fp T, int_type IntT>
gpublasStatus_t tbmv(gpublasHandle_t handle, gpublasFillMode_t uplo, gpublasOperation_t trans,
                     gpublasDiagType_t diag, IntT n, IntT k, const T *A, IntT lda, T *x,
                     IntT incx) {
  GPUMOD_USUAL_DISPATCH_64(T, IntT, tbmv, handle, uplo, trans, diag, n, k, A, lda, x, incx);
}

// ========================================================================
// Triangular banded solve: op(A)*x = b
// ========================================================================

template<usual_fp T, int_type IntT>
gpublasStatus_t tbsv(gpublasHandle_t handle, gpublasFillMode_t uplo, gpublasOperation_t trans,
                     gpublasDiagType_t diag, IntT n, IntT k, const T *A, IntT lda, T *x,
                     IntT incx) {
  GPUMOD_USUAL_DISPATCH_64(T, IntT, tbsv, handle, uplo, trans, diag, n, k, A, lda, x, incx);
}

// ========================================================================
// Triangular packed matrix-vector multiplication: x = op(A)*x
// ========================================================================

template<usual_fp T, int_type IntT>
gpublasStatus_t tpmv(gpublasHandle_t handle, gpublasFillMode_t uplo, gpublasOperation_t trans,
                     gpublasDiagType_t diag, IntT n, const T *AP, T *x, IntT incx) {
  GPUMOD_USUAL_DISPATCH_64(T, IntT, tpmv, handle, uplo, trans, diag, n, AP, x, incx);
}

// ========================================================================
// Triangular packed solve: op(A)*x = b
// ========================================================================

template<usual_fp T, int_type IntT>
gpublasStatus_t tpsv(gpublasHandle_t handle, gpublasFillMode_t uplo, gpublasOperation_t trans,
                     gpublasDiagType_t diag, IntT n, const T *AP, T *x, IntT incx) {
  GPUMOD_USUAL_DISPATCH_64(T, IntT, tpsv, handle, uplo, trans, diag, n, AP, x, incx);
}

// ========================================================================
// Hermitian matrix-vector multiplication: y = alpha*A*x + beta*y
// ========================================================================

template<complex_fp T, int_type IntT>
gpublasStatus_t hemv(gpublasHandle_t handle, gpublasFillMode_t uplo, IntT n, const T *alpha,
                     const T *A, IntT lda, const T *x, IntT incx, const T *beta, T *y, IntT incy) {
  GPUMOD_COMPLEX_DISPATCH_64(T, IntT, gpublas, C, Z, hemv, handle, uplo, n, alpha, A, lda, x, incx,
                             beta, y, incy);
}

// ========================================================================
// Hermitian banded matrix-vector multiplication: y = alpha*A*x + beta*y
// ========================================================================

template<complex_fp T, int_type IntT>
gpublasStatus_t hbmv(gpublasHandle_t handle, gpublasFillMode_t uplo, IntT n, IntT k, const T *alpha,
                     const T *A, IntT lda, const T *x, IntT incx, const T *beta, T *y, IntT incy) {
  GPUMOD_COMPLEX_DISPATCH_64(T, IntT, gpublas, C, Z, hbmv, handle, uplo, n, k, alpha, A, lda, x,
                             incx, beta, y, incy);
}

// ========================================================================
// Hermitian packed matrix-vector multiplication: y = alpha*A*x + beta*y
// ========================================================================

template<complex_fp T, int_type IntT>
gpublasStatus_t hpmv(gpublasHandle_t handle, gpublasFillMode_t uplo, IntT n, const T *alpha,
                     const T *AP, const T *x, IntT incx, const T *beta, T *y, IntT incy) {
  GPUMOD_COMPLEX_DISPATCH_64(T, IntT, gpublas, C, Z, hpmv, handle, uplo, n, alpha, AP, x, incx,
                             beta, y, incy);
}

// ========================================================================
// Hermitian rank-1 update: A = alpha*x*x^H + A
// ========================================================================

template<complex_fp T, int_type IntT>
gpublasStatus_t her(gpublasHandle_t handle, gpublasFillMode_t uplo, IntT n,
                    const ComplexToRealType<T> *alpha, const T *x, IntT incx, T *A, IntT lda) {
  GPUMOD_COMPLEX_DISPATCH_64(T, IntT, gpublas, C, Z, her, handle, uplo, n, alpha, x, incx, A, lda);
}

// ========================================================================
// Hermitian rank-2 update: A = alpha*x*y^H + conj(alpha)*y*x^H + A
// ========================================================================

template<complex_fp T, int_type IntT>
gpublasStatus_t her2(gpublasHandle_t handle, gpublasFillMode_t uplo, IntT n, const T *alpha,
                     const T *x, IntT incx, const T *y, IntT incy, T *A, IntT lda) {
  GPUMOD_COMPLEX_DISPATCH_64(T, IntT, gpublas, C, Z, her2, handle, uplo, n, alpha, x, incx, y, incy,
                             A, lda);
}

// ========================================================================
// Hermitian packed rank-1 update: A = alpha*x*x^H + A
// ========================================================================

template<complex_fp T, int_type IntT>
gpublasStatus_t hpr(gpublasHandle_t handle, gpublasFillMode_t uplo, IntT n,
                    const ComplexToRealType<T> *alpha, const T *x, IntT incx, T *AP) {
  GPUMOD_COMPLEX_DISPATCH_64(T, IntT, gpublas, C, Z, hpr, handle, uplo, n, alpha, x, incx, AP);
}

// ========================================================================
// Hermitian packed rank-2 update: A = alpha*x*y^H + conj(alpha)*y*x^H + A
// ========================================================================

template<complex_fp T, int_type IntT>
gpublasStatus_t hpr2(gpublasHandle_t handle, gpublasFillMode_t uplo, IntT n, const T *alpha,
                     const T *x, IntT incx, const T *y, IntT incy, T *AP) {
  GPUMOD_COMPLEX_DISPATCH_64(T, IntT, gpublas, C, Z, hpr2, handle, uplo, n, alpha, x, incx, y, incy,
                             AP);
}

// ========================================================================
// Batched general matrix-vector multiplication
// ========================================================================

template<usual_fp T, int_type IntT>
gpublasStatus_t gemvBatched(gpublasHandle_t handle, gpublasOperation_t trans, IntT m, IntT n,
                            const T *alpha, const T *const Aarray[], IntT lda,
                            const T *const xarray[], IntT incx, const T *beta, T *const yarray[],
                            IntT incy, IntT batchCount) {
  GPUMOD_USUAL_DISPATCH_64(T, IntT, gemvBatched, handle, trans, m, n, alpha, Aarray, lda, xarray,
                           incx, beta, yarray, incy, batchCount);
}

// ========================================================================
// Strided batched general matrix-vector multiplication
// ========================================================================

template<usual_fp T, int_type IntT>
gpublasStatus_t gemvStridedBatched(gpublasHandle_t handle, gpublasOperation_t trans, IntT m, IntT n,
                                   const T *alpha, const T *A, IntT lda, long long int strideA,
                                   const T *x, IntT incx, long long int stridex, const T *beta,
                                   T *y, IntT incy, long long int stridey, IntT batchCount) {
  GPUMOD_USUAL_DISPATCH_64(T, IntT, gemvStridedBatched, handle, trans, m, n, alpha, A, lda, strideA,
                           x, incx, stridex, beta, y, incy, stridey, batchCount);
}

// ==================== Explicit Template Instantiations ====================
// Matching `template` instantiations live in instantiations.cpp.

// Function: gemv
extern template gpublasStatus_t gemv<float, int>(gpublasHandle_t, gpublasOperation_t, int, int,
                                                 const float *, const float *, int, const float *,
                                                 int, const float *, float *, int);
extern template gpublasStatus_t gemv<float, int64_t>(gpublasHandle_t, gpublasOperation_t, int64_t,
                                                     int64_t, const float *, const float *, int64_t,
                                                     const float *, int64_t, const float *, float *,
                                                     int64_t);
extern template gpublasStatus_t gemv<double, int>(gpublasHandle_t, gpublasOperation_t, int, int,
                                                  const double *, const double *, int,
                                                  const double *, int, const double *, double *,
                                                  int);
extern template gpublasStatus_t gemv<double, int64_t>(gpublasHandle_t, gpublasOperation_t, int64_t,
                                                      int64_t, const double *, const double *,
                                                      int64_t, const double *, int64_t,
                                                      const double *, double *, int64_t);
extern template gpublasStatus_t
gemv<gpuFloatComplex, int>(gpublasHandle_t, gpublasOperation_t, int, int, const gpuFloatComplex *,
                           const gpuFloatComplex *, int, const gpuFloatComplex *, int,
                           const gpuFloatComplex *, gpuFloatComplex *, int);
extern template gpublasStatus_t
gemv<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasOperation_t, int64_t, int64_t,
                               const gpuFloatComplex *, const gpuFloatComplex *, int64_t,
                               const gpuFloatComplex *, int64_t, const gpuFloatComplex *,
                               gpuFloatComplex *, int64_t);
extern template gpublasStatus_t
gemv<gpuDoubleComplex, int>(gpublasHandle_t, gpublasOperation_t, int, int, const gpuDoubleComplex *,
                            const gpuDoubleComplex *, int, const gpuDoubleComplex *, int,
                            const gpuDoubleComplex *, gpuDoubleComplex *, int);
extern template gpublasStatus_t
gemv<gpuDoubleComplex, int64_t>(gpublasHandle_t, gpublasOperation_t, int64_t, int64_t,
                                const gpuDoubleComplex *, const gpuDoubleComplex *, int64_t,
                                const gpuDoubleComplex *, int64_t, const gpuDoubleComplex *,
                                gpuDoubleComplex *, int64_t);

// Function: gbmv
extern template gpublasStatus_t gbmv<float, int>(gpublasHandle_t, gpublasOperation_t, int, int, int,
                                                 int, const float *, const float *, int,
                                                 const float *, int, const float *, float *, int);
extern template gpublasStatus_t gbmv<float, int64_t>(gpublasHandle_t, gpublasOperation_t, int64_t,
                                                     int64_t, int64_t, int64_t, const float *,
                                                     const float *, int64_t, const float *, int64_t,
                                                     const float *, float *, int64_t);
extern template gpublasStatus_t gbmv<double, int>(gpublasHandle_t, gpublasOperation_t, int, int,
                                                  int, int, const double *, const double *, int,
                                                  const double *, int, const double *, double *,
                                                  int);
extern template gpublasStatus_t gbmv<double, int64_t>(gpublasHandle_t, gpublasOperation_t, int64_t,
                                                      int64_t, int64_t, int64_t, const double *,
                                                      const double *, int64_t, const double *,
                                                      int64_t, const double *, double *, int64_t);
extern template gpublasStatus_t gbmv<gpuFloatComplex, int>(gpublasHandle_t, gpublasOperation_t, int,
                                                           int, int, int, const gpuFloatComplex *,
                                                           const gpuFloatComplex *, int,
                                                           const gpuFloatComplex *, int,
                                                           const gpuFloatComplex *,
                                                           gpuFloatComplex *, int);
extern template gpublasStatus_t
gbmv<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasOperation_t, int64_t, int64_t, int64_t,
                               int64_t, const gpuFloatComplex *, const gpuFloatComplex *, int64_t,
                               const gpuFloatComplex *, int64_t, const gpuFloatComplex *,
                               gpuFloatComplex *, int64_t);
extern template gpublasStatus_t
gbmv<gpuDoubleComplex, int>(gpublasHandle_t, gpublasOperation_t, int, int, int, int,
                            const gpuDoubleComplex *, const gpuDoubleComplex *, int,
                            const gpuDoubleComplex *, int, const gpuDoubleComplex *,
                            gpuDoubleComplex *, int);
extern template gpublasStatus_t
gbmv<gpuDoubleComplex, int64_t>(gpublasHandle_t, gpublasOperation_t, int64_t, int64_t, int64_t,
                                int64_t, const gpuDoubleComplex *, const gpuDoubleComplex *,
                                int64_t, const gpuDoubleComplex *, int64_t,
                                const gpuDoubleComplex *, gpuDoubleComplex *, int64_t);

// Function: ger
extern template gpublasStatus_t ger<float, int>(gpublasHandle_t, int, int, const float *,
                                                const float *, int, const float *, int, float *,
                                                int);
extern template gpublasStatus_t ger<float, int64_t>(gpublasHandle_t, int64_t, int64_t,
                                                    const float *, const float *, int64_t,
                                                    const float *, int64_t, float *, int64_t);
extern template gpublasStatus_t ger<double, int>(gpublasHandle_t, int, int, const double *,
                                                 const double *, int, const double *, int, double *,
                                                 int);
extern template gpublasStatus_t ger<double, int64_t>(gpublasHandle_t, int64_t, int64_t,
                                                     const double *, const double *, int64_t,
                                                     const double *, int64_t, double *, int64_t);

// Function: geru
extern template gpublasStatus_t geru<gpuFloatComplex, int>(gpublasHandle_t, int, int,
                                                           const gpuFloatComplex *,
                                                           const gpuFloatComplex *, int,
                                                           const gpuFloatComplex *, int,
                                                           gpuFloatComplex *, int);
extern template gpublasStatus_t geru<gpuFloatComplex, int64_t>(gpublasHandle_t, int64_t, int64_t,
                                                               const gpuFloatComplex *,
                                                               const gpuFloatComplex *, int64_t,
                                                               const gpuFloatComplex *, int64_t,
                                                               gpuFloatComplex *, int64_t);
extern template gpublasStatus_t geru<gpuDoubleComplex, int>(gpublasHandle_t, int, int,
                                                            const gpuDoubleComplex *,
                                                            const gpuDoubleComplex *, int,
                                                            const gpuDoubleComplex *, int,
                                                            gpuDoubleComplex *, int);
extern template gpublasStatus_t geru<gpuDoubleComplex, int64_t>(gpublasHandle_t, int64_t, int64_t,
                                                                const gpuDoubleComplex *,
                                                                const gpuDoubleComplex *, int64_t,
                                                                const gpuDoubleComplex *, int64_t,
                                                                gpuDoubleComplex *, int64_t);

// Function: gerc
extern template gpublasStatus_t gerc<gpuFloatComplex, int>(gpublasHandle_t, int, int,
                                                           const gpuFloatComplex *,
                                                           const gpuFloatComplex *, int,
                                                           const gpuFloatComplex *, int,
                                                           gpuFloatComplex *, int);
extern template gpublasStatus_t gerc<gpuFloatComplex, int64_t>(gpublasHandle_t, int64_t, int64_t,
                                                               const gpuFloatComplex *,
                                                               const gpuFloatComplex *, int64_t,
                                                               const gpuFloatComplex *, int64_t,
                                                               gpuFloatComplex *, int64_t);
extern template gpublasStatus_t gerc<gpuDoubleComplex, int>(gpublasHandle_t, int, int,
                                                            const gpuDoubleComplex *,
                                                            const gpuDoubleComplex *, int,
                                                            const gpuDoubleComplex *, int,
                                                            gpuDoubleComplex *, int);
extern template gpublasStatus_t gerc<gpuDoubleComplex, int64_t>(gpublasHandle_t, int64_t, int64_t,
                                                                const gpuDoubleComplex *,
                                                                const gpuDoubleComplex *, int64_t,
                                                                const gpuDoubleComplex *, int64_t,
                                                                gpuDoubleComplex *, int64_t);

// Function: symv
extern template gpublasStatus_t symv<float, int>(gpublasHandle_t, gpublasFillMode_t, int,
                                                 const float *, const float *, int, const float *,
                                                 int, const float *, float *, int);
extern template gpublasStatus_t symv<float, int64_t>(gpublasHandle_t, gpublasFillMode_t, int64_t,
                                                     const float *, const float *, int64_t,
                                                     const float *, int64_t, const float *, float *,
                                                     int64_t);
extern template gpublasStatus_t symv<double, int>(gpublasHandle_t, gpublasFillMode_t, int,
                                                  const double *, const double *, int,
                                                  const double *, int, const double *, double *,
                                                  int);
extern template gpublasStatus_t symv<double, int64_t>(gpublasHandle_t, gpublasFillMode_t, int64_t,
                                                      const double *, const double *, int64_t,
                                                      const double *, int64_t, const double *,
                                                      double *, int64_t);

// Function: syr
extern template gpublasStatus_t syr<float, int>(gpublasHandle_t, gpublasFillMode_t, int,
                                                const float *, const float *, int, float *, int);
extern template gpublasStatus_t syr<float, int64_t>(gpublasHandle_t, gpublasFillMode_t, int64_t,
                                                    const float *, const float *, int64_t, float *,
                                                    int64_t);
extern template gpublasStatus_t syr<double, int>(gpublasHandle_t, gpublasFillMode_t, int,
                                                 const double *, const double *, int, double *,
                                                 int);
extern template gpublasStatus_t syr<double, int64_t>(gpublasHandle_t, gpublasFillMode_t, int64_t,
                                                     const double *, const double *, int64_t,
                                                     double *, int64_t);

// Function: syr2
extern template gpublasStatus_t syr2<float, int>(gpublasHandle_t, gpublasFillMode_t, int,
                                                 const float *, const float *, int, const float *,
                                                 int, float *, int);
extern template gpublasStatus_t syr2<float, int64_t>(gpublasHandle_t, gpublasFillMode_t, int64_t,
                                                     const float *, const float *, int64_t,
                                                     const float *, int64_t, float *, int64_t);
extern template gpublasStatus_t syr2<double, int>(gpublasHandle_t, gpublasFillMode_t, int,
                                                  const double *, const double *, int,
                                                  const double *, int, double *, int);
extern template gpublasStatus_t syr2<double, int64_t>(gpublasHandle_t, gpublasFillMode_t, int64_t,
                                                      const double *, const double *, int64_t,
                                                      const double *, int64_t, double *, int64_t);

// Function: sbmv
extern template gpublasStatus_t sbmv<float, int>(gpublasHandle_t, gpublasFillMode_t, int, int,
                                                 const float *, const float *, int, const float *,
                                                 int, const float *, float *, int);
extern template gpublasStatus_t sbmv<float, int64_t>(gpublasHandle_t, gpublasFillMode_t, int64_t,
                                                     int64_t, const float *, const float *, int64_t,
                                                     const float *, int64_t, const float *, float *,
                                                     int64_t);
extern template gpublasStatus_t sbmv<double, int>(gpublasHandle_t, gpublasFillMode_t, int, int,
                                                  const double *, const double *, int,
                                                  const double *, int, const double *, double *,
                                                  int);
extern template gpublasStatus_t sbmv<double, int64_t>(gpublasHandle_t, gpublasFillMode_t, int64_t,
                                                      int64_t, const double *, const double *,
                                                      int64_t, const double *, int64_t,
                                                      const double *, double *, int64_t);

// Function: spmv
extern template gpublasStatus_t spmv<float, int>(gpublasHandle_t, gpublasFillMode_t, int,
                                                 const float *, const float *, const float *, int,
                                                 const float *, float *, int);
extern template gpublasStatus_t spmv<float, int64_t>(gpublasHandle_t, gpublasFillMode_t, int64_t,
                                                     const float *, const float *, const float *,
                                                     int64_t, const float *, float *, int64_t);
extern template gpublasStatus_t spmv<double, int>(gpublasHandle_t, gpublasFillMode_t, int,
                                                  const double *, const double *, const double *,
                                                  int, const double *, double *, int);
extern template gpublasStatus_t spmv<double, int64_t>(gpublasHandle_t, gpublasFillMode_t, int64_t,
                                                      const double *, const double *,
                                                      const double *, int64_t, const double *,
                                                      double *, int64_t);

// Function: spr
extern template gpublasStatus_t spr<float, int>(gpublasHandle_t, gpublasFillMode_t, int,
                                                const float *, const float *, int, float *);
extern template gpublasStatus_t spr<float, int64_t>(gpublasHandle_t, gpublasFillMode_t, int64_t,
                                                    const float *, const float *, int64_t, float *);
extern template gpublasStatus_t spr<double, int>(gpublasHandle_t, gpublasFillMode_t, int,
                                                 const double *, const double *, int, double *);
extern template gpublasStatus_t spr<double, int64_t>(gpublasHandle_t, gpublasFillMode_t, int64_t,
                                                     const double *, const double *, int64_t,
                                                     double *);

// Function: spr2
extern template gpublasStatus_t spr2<float, int>(gpublasHandle_t, gpublasFillMode_t, int,
                                                 const float *, const float *, int, const float *,
                                                 int, float *);
extern template gpublasStatus_t spr2<float, int64_t>(gpublasHandle_t, gpublasFillMode_t, int64_t,
                                                     const float *, const float *, int64_t,
                                                     const float *, int64_t, float *);
extern template gpublasStatus_t spr2<double, int>(gpublasHandle_t, gpublasFillMode_t, int,
                                                  const double *, const double *, int,
                                                  const double *, int, double *);
extern template gpublasStatus_t spr2<double, int64_t>(gpublasHandle_t, gpublasFillMode_t, int64_t,
                                                      const double *, const double *, int64_t,
                                                      const double *, int64_t, double *);

// Function: trmv
extern template gpublasStatus_t trmv<float, int>(gpublasHandle_t, gpublasFillMode_t,
                                                 gpublasOperation_t, gpublasDiagType_t, int,
                                                 const float *, int, float *, int);
extern template gpublasStatus_t trmv<float, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                     gpublasOperation_t, gpublasDiagType_t, int64_t,
                                                     const float *, int64_t, float *, int64_t);
extern template gpublasStatus_t trmv<double, int>(gpublasHandle_t, gpublasFillMode_t,
                                                  gpublasOperation_t, gpublasDiagType_t, int,
                                                  const double *, int, double *, int);
extern template gpublasStatus_t trmv<double, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                      gpublasOperation_t, gpublasDiagType_t,
                                                      int64_t, const double *, int64_t, double *,
                                                      int64_t);
extern template gpublasStatus_t trmv<gpuFloatComplex, int>(gpublasHandle_t, gpublasFillMode_t,
                                                           gpublasOperation_t, gpublasDiagType_t,
                                                           int, const gpuFloatComplex *, int,
                                                           gpuFloatComplex *, int);
extern template gpublasStatus_t trmv<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                               gpublasOperation_t,
                                                               gpublasDiagType_t, int64_t,
                                                               const gpuFloatComplex *, int64_t,
                                                               gpuFloatComplex *, int64_t);
extern template gpublasStatus_t trmv<gpuDoubleComplex, int>(gpublasHandle_t, gpublasFillMode_t,
                                                            gpublasOperation_t, gpublasDiagType_t,
                                                            int, const gpuDoubleComplex *, int,
                                                            gpuDoubleComplex *, int);
extern template gpublasStatus_t trmv<gpuDoubleComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                                gpublasOperation_t,
                                                                gpublasDiagType_t, int64_t,
                                                                const gpuDoubleComplex *, int64_t,
                                                                gpuDoubleComplex *, int64_t);

// Function: trsv
extern template gpublasStatus_t trsv<float, int>(gpublasHandle_t, gpublasFillMode_t,
                                                 gpublasOperation_t, gpublasDiagType_t, int,
                                                 const float *, int, float *, int);
extern template gpublasStatus_t trsv<float, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                     gpublasOperation_t, gpublasDiagType_t, int64_t,
                                                     const float *, int64_t, float *, int64_t);
extern template gpublasStatus_t trsv<double, int>(gpublasHandle_t, gpublasFillMode_t,
                                                  gpublasOperation_t, gpublasDiagType_t, int,
                                                  const double *, int, double *, int);
extern template gpublasStatus_t trsv<double, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                      gpublasOperation_t, gpublasDiagType_t,
                                                      int64_t, const double *, int64_t, double *,
                                                      int64_t);
extern template gpublasStatus_t trsv<gpuFloatComplex, int>(gpublasHandle_t, gpublasFillMode_t,
                                                           gpublasOperation_t, gpublasDiagType_t,
                                                           int, const gpuFloatComplex *, int,
                                                           gpuFloatComplex *, int);
extern template gpublasStatus_t trsv<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                               gpublasOperation_t,
                                                               gpublasDiagType_t, int64_t,
                                                               const gpuFloatComplex *, int64_t,
                                                               gpuFloatComplex *, int64_t);
extern template gpublasStatus_t trsv<gpuDoubleComplex, int>(gpublasHandle_t, gpublasFillMode_t,
                                                            gpublasOperation_t, gpublasDiagType_t,
                                                            int, const gpuDoubleComplex *, int,
                                                            gpuDoubleComplex *, int);
extern template gpublasStatus_t trsv<gpuDoubleComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                                gpublasOperation_t,
                                                                gpublasDiagType_t, int64_t,
                                                                const gpuDoubleComplex *, int64_t,
                                                                gpuDoubleComplex *, int64_t);

// Function: tbmv
extern template gpublasStatus_t tbmv<float, int>(gpublasHandle_t, gpublasFillMode_t,
                                                 gpublasOperation_t, gpublasDiagType_t, int, int,
                                                 const float *, int, float *, int);
extern template gpublasStatus_t tbmv<float, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                     gpublasOperation_t, gpublasDiagType_t, int64_t,
                                                     int64_t, const float *, int64_t, float *,
                                                     int64_t);
extern template gpublasStatus_t tbmv<double, int>(gpublasHandle_t, gpublasFillMode_t,
                                                  gpublasOperation_t, gpublasDiagType_t, int, int,
                                                  const double *, int, double *, int);
extern template gpublasStatus_t tbmv<double, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                      gpublasOperation_t, gpublasDiagType_t,
                                                      int64_t, int64_t, const double *, int64_t,
                                                      double *, int64_t);
extern template gpublasStatus_t tbmv<gpuFloatComplex, int>(gpublasHandle_t, gpublasFillMode_t,
                                                           gpublasOperation_t, gpublasDiagType_t,
                                                           int, int, const gpuFloatComplex *, int,
                                                           gpuFloatComplex *, int);
extern template gpublasStatus_t tbmv<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                               gpublasOperation_t,
                                                               gpublasDiagType_t, int64_t, int64_t,
                                                               const gpuFloatComplex *, int64_t,
                                                               gpuFloatComplex *, int64_t);
extern template gpublasStatus_t tbmv<gpuDoubleComplex, int>(gpublasHandle_t, gpublasFillMode_t,
                                                            gpublasOperation_t, gpublasDiagType_t,
                                                            int, int, const gpuDoubleComplex *, int,
                                                            gpuDoubleComplex *, int);
extern template gpublasStatus_t tbmv<gpuDoubleComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                                gpublasOperation_t,
                                                                gpublasDiagType_t, int64_t, int64_t,
                                                                const gpuDoubleComplex *, int64_t,
                                                                gpuDoubleComplex *, int64_t);

// Function: tbsv
extern template gpublasStatus_t tbsv<float, int>(gpublasHandle_t, gpublasFillMode_t,
                                                 gpublasOperation_t, gpublasDiagType_t, int, int,
                                                 const float *, int, float *, int);
extern template gpublasStatus_t tbsv<float, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                     gpublasOperation_t, gpublasDiagType_t, int64_t,
                                                     int64_t, const float *, int64_t, float *,
                                                     int64_t);
extern template gpublasStatus_t tbsv<double, int>(gpublasHandle_t, gpublasFillMode_t,
                                                  gpublasOperation_t, gpublasDiagType_t, int, int,
                                                  const double *, int, double *, int);
extern template gpublasStatus_t tbsv<double, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                      gpublasOperation_t, gpublasDiagType_t,
                                                      int64_t, int64_t, const double *, int64_t,
                                                      double *, int64_t);
extern template gpublasStatus_t tbsv<gpuFloatComplex, int>(gpublasHandle_t, gpublasFillMode_t,
                                                           gpublasOperation_t, gpublasDiagType_t,
                                                           int, int, const gpuFloatComplex *, int,
                                                           gpuFloatComplex *, int);
extern template gpublasStatus_t tbsv<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                               gpublasOperation_t,
                                                               gpublasDiagType_t, int64_t, int64_t,
                                                               const gpuFloatComplex *, int64_t,
                                                               gpuFloatComplex *, int64_t);
extern template gpublasStatus_t tbsv<gpuDoubleComplex, int>(gpublasHandle_t, gpublasFillMode_t,
                                                            gpublasOperation_t, gpublasDiagType_t,
                                                            int, int, const gpuDoubleComplex *, int,
                                                            gpuDoubleComplex *, int);
extern template gpublasStatus_t tbsv<gpuDoubleComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                                gpublasOperation_t,
                                                                gpublasDiagType_t, int64_t, int64_t,
                                                                const gpuDoubleComplex *, int64_t,
                                                                gpuDoubleComplex *, int64_t);

// Function: tpmv
extern template gpublasStatus_t tpmv<float, int>(gpublasHandle_t, gpublasFillMode_t,
                                                 gpublasOperation_t, gpublasDiagType_t, int,
                                                 const float *, float *, int);
extern template gpublasStatus_t tpmv<float, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                     gpublasOperation_t, gpublasDiagType_t, int64_t,
                                                     const float *, float *, int64_t);
extern template gpublasStatus_t tpmv<double, int>(gpublasHandle_t, gpublasFillMode_t,
                                                  gpublasOperation_t, gpublasDiagType_t, int,
                                                  const double *, double *, int);
extern template gpublasStatus_t tpmv<double, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                      gpublasOperation_t, gpublasDiagType_t,
                                                      int64_t, const double *, double *, int64_t);
extern template gpublasStatus_t tpmv<gpuFloatComplex, int>(gpublasHandle_t, gpublasFillMode_t,
                                                           gpublasOperation_t, gpublasDiagType_t,
                                                           int, const gpuFloatComplex *,
                                                           gpuFloatComplex *, int);
extern template gpublasStatus_t tpmv<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                               gpublasOperation_t,
                                                               gpublasDiagType_t, int64_t,
                                                               const gpuFloatComplex *,
                                                               gpuFloatComplex *, int64_t);
extern template gpublasStatus_t tpmv<gpuDoubleComplex, int>(gpublasHandle_t, gpublasFillMode_t,
                                                            gpublasOperation_t, gpublasDiagType_t,
                                                            int, const gpuDoubleComplex *,
                                                            gpuDoubleComplex *, int);
extern template gpublasStatus_t tpmv<gpuDoubleComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                                gpublasOperation_t,
                                                                gpublasDiagType_t, int64_t,
                                                                const gpuDoubleComplex *,
                                                                gpuDoubleComplex *, int64_t);

// Function: tpsv
extern template gpublasStatus_t tpsv<float, int>(gpublasHandle_t, gpublasFillMode_t,
                                                 gpublasOperation_t, gpublasDiagType_t, int,
                                                 const float *, float *, int);
extern template gpublasStatus_t tpsv<float, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                     gpublasOperation_t, gpublasDiagType_t, int64_t,
                                                     const float *, float *, int64_t);
extern template gpublasStatus_t tpsv<double, int>(gpublasHandle_t, gpublasFillMode_t,
                                                  gpublasOperation_t, gpublasDiagType_t, int,
                                                  const double *, double *, int);
extern template gpublasStatus_t tpsv<double, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                      gpublasOperation_t, gpublasDiagType_t,
                                                      int64_t, const double *, double *, int64_t);
extern template gpublasStatus_t tpsv<gpuFloatComplex, int>(gpublasHandle_t, gpublasFillMode_t,
                                                           gpublasOperation_t, gpublasDiagType_t,
                                                           int, const gpuFloatComplex *,
                                                           gpuFloatComplex *, int);
extern template gpublasStatus_t tpsv<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                               gpublasOperation_t,
                                                               gpublasDiagType_t, int64_t,
                                                               const gpuFloatComplex *,
                                                               gpuFloatComplex *, int64_t);
extern template gpublasStatus_t tpsv<gpuDoubleComplex, int>(gpublasHandle_t, gpublasFillMode_t,
                                                            gpublasOperation_t, gpublasDiagType_t,
                                                            int, const gpuDoubleComplex *,
                                                            gpuDoubleComplex *, int);
extern template gpublasStatus_t tpsv<gpuDoubleComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                                gpublasOperation_t,
                                                                gpublasDiagType_t, int64_t,
                                                                const gpuDoubleComplex *,
                                                                gpuDoubleComplex *, int64_t);

// Function: hemv
extern template gpublasStatus_t
hemv<gpuFloatComplex, int>(gpublasHandle_t, gpublasFillMode_t, int, const gpuFloatComplex *,
                           const gpuFloatComplex *, int, const gpuFloatComplex *, int,
                           const gpuFloatComplex *, gpuFloatComplex *, int);
extern template gpublasStatus_t
hemv<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t, int64_t, const gpuFloatComplex *,
                               const gpuFloatComplex *, int64_t, const gpuFloatComplex *, int64_t,
                               const gpuFloatComplex *, gpuFloatComplex *, int64_t);
extern template gpublasStatus_t
hemv<gpuDoubleComplex, int>(gpublasHandle_t, gpublasFillMode_t, int, const gpuDoubleComplex *,
                            const gpuDoubleComplex *, int, const gpuDoubleComplex *, int,
                            const gpuDoubleComplex *, gpuDoubleComplex *, int);
extern template gpublasStatus_t hemv<gpuDoubleComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                                int64_t, const gpuDoubleComplex *,
                                                                const gpuDoubleComplex *, int64_t,
                                                                const gpuDoubleComplex *, int64_t,
                                                                const gpuDoubleComplex *,
                                                                gpuDoubleComplex *, int64_t);

// Function: hbmv
extern template gpublasStatus_t
hbmv<gpuFloatComplex, int>(gpublasHandle_t, gpublasFillMode_t, int, int, const gpuFloatComplex *,
                           const gpuFloatComplex *, int, const gpuFloatComplex *, int,
                           const gpuFloatComplex *, gpuFloatComplex *, int);
extern template gpublasStatus_t
hbmv<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t, int64_t, int64_t,
                               const gpuFloatComplex *, const gpuFloatComplex *, int64_t,
                               const gpuFloatComplex *, int64_t, const gpuFloatComplex *,
                               gpuFloatComplex *, int64_t);
extern template gpublasStatus_t
hbmv<gpuDoubleComplex, int>(gpublasHandle_t, gpublasFillMode_t, int, int, const gpuDoubleComplex *,
                            const gpuDoubleComplex *, int, const gpuDoubleComplex *, int,
                            const gpuDoubleComplex *, gpuDoubleComplex *, int);
extern template gpublasStatus_t
hbmv<gpuDoubleComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t, int64_t, int64_t,
                                const gpuDoubleComplex *, const gpuDoubleComplex *, int64_t,
                                const gpuDoubleComplex *, int64_t, const gpuDoubleComplex *,
                                gpuDoubleComplex *, int64_t);

// Function: hpmv
extern template gpublasStatus_t
hpmv<gpuFloatComplex, int>(gpublasHandle_t, gpublasFillMode_t, int, const gpuFloatComplex *,
                           const gpuFloatComplex *, const gpuFloatComplex *, int,
                           const gpuFloatComplex *, gpuFloatComplex *, int);
extern template gpublasStatus_t
hpmv<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t, int64_t, const gpuFloatComplex *,
                               const gpuFloatComplex *, const gpuFloatComplex *, int64_t,
                               const gpuFloatComplex *, gpuFloatComplex *, int64_t);
extern template gpublasStatus_t
hpmv<gpuDoubleComplex, int>(gpublasHandle_t, gpublasFillMode_t, int, const gpuDoubleComplex *,
                            const gpuDoubleComplex *, const gpuDoubleComplex *, int,
                            const gpuDoubleComplex *, gpuDoubleComplex *, int);
extern template gpublasStatus_t hpmv<gpuDoubleComplex, int64_t>(
    gpublasHandle_t, gpublasFillMode_t, int64_t, const gpuDoubleComplex *, const gpuDoubleComplex *,
    const gpuDoubleComplex *, int64_t, const gpuDoubleComplex *, gpuDoubleComplex *, int64_t);

// Function: her
extern template gpublasStatus_t
her<gpuFloatComplex, int>(gpublasHandle_t, gpublasFillMode_t, int,
                          const ComplexToRealType<gpuFloatComplex> *, const gpuFloatComplex *, int,
                          gpuFloatComplex *, int);
extern template gpublasStatus_t
her<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t, int64_t,
                              const ComplexToRealType<gpuFloatComplex> *, const gpuFloatComplex *,
                              int64_t, gpuFloatComplex *, int64_t);
extern template gpublasStatus_t
her<gpuDoubleComplex, int>(gpublasHandle_t, gpublasFillMode_t, int,
                           const ComplexToRealType<gpuDoubleComplex> *, const gpuDoubleComplex *,
                           int, gpuDoubleComplex *, int);
extern template gpublasStatus_t
her<gpuDoubleComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t, int64_t,
                               const ComplexToRealType<gpuDoubleComplex> *,
                               const gpuDoubleComplex *, int64_t, gpuDoubleComplex *, int64_t);

// Function: her2
extern template gpublasStatus_t her2<gpuFloatComplex, int>(gpublasHandle_t, gpublasFillMode_t, int,
                                                           const gpuFloatComplex *,
                                                           const gpuFloatComplex *, int,
                                                           const gpuFloatComplex *, int,
                                                           gpuFloatComplex *, int);
extern template gpublasStatus_t her2<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                               int64_t, const gpuFloatComplex *,
                                                               const gpuFloatComplex *, int64_t,
                                                               const gpuFloatComplex *, int64_t,
                                                               gpuFloatComplex *, int64_t);
extern template gpublasStatus_t her2<gpuDoubleComplex, int>(gpublasHandle_t, gpublasFillMode_t, int,
                                                            const gpuDoubleComplex *,
                                                            const gpuDoubleComplex *, int,
                                                            const gpuDoubleComplex *, int,
                                                            gpuDoubleComplex *, int);
extern template gpublasStatus_t her2<gpuDoubleComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                                int64_t, const gpuDoubleComplex *,
                                                                const gpuDoubleComplex *, int64_t,
                                                                const gpuDoubleComplex *, int64_t,
                                                                gpuDoubleComplex *, int64_t);

// Function: hpr
extern template gpublasStatus_t
hpr<gpuFloatComplex, int>(gpublasHandle_t, gpublasFillMode_t, int,
                          const ComplexToRealType<gpuFloatComplex> *, const gpuFloatComplex *, int,
                          gpuFloatComplex *);
extern template gpublasStatus_t
hpr<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t, int64_t,
                              const ComplexToRealType<gpuFloatComplex> *, const gpuFloatComplex *,
                              int64_t, gpuFloatComplex *);
extern template gpublasStatus_t
hpr<gpuDoubleComplex, int>(gpublasHandle_t, gpublasFillMode_t, int,
                           const ComplexToRealType<gpuDoubleComplex> *, const gpuDoubleComplex *,
                           int, gpuDoubleComplex *);
extern template gpublasStatus_t
hpr<gpuDoubleComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t, int64_t,
                               const ComplexToRealType<gpuDoubleComplex> *,
                               const gpuDoubleComplex *, int64_t, gpuDoubleComplex *);

// Function: hpr2
extern template gpublasStatus_t hpr2<gpuFloatComplex, int>(gpublasHandle_t, gpublasFillMode_t, int,
                                                           const gpuFloatComplex *,
                                                           const gpuFloatComplex *, int,
                                                           const gpuFloatComplex *, int,
                                                           gpuFloatComplex *);
extern template gpublasStatus_t hpr2<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                               int64_t, const gpuFloatComplex *,
                                                               const gpuFloatComplex *, int64_t,
                                                               const gpuFloatComplex *, int64_t,
                                                               gpuFloatComplex *);
extern template gpublasStatus_t hpr2<gpuDoubleComplex, int>(gpublasHandle_t, gpublasFillMode_t, int,
                                                            const gpuDoubleComplex *,
                                                            const gpuDoubleComplex *, int,
                                                            const gpuDoubleComplex *, int,
                                                            gpuDoubleComplex *);
extern template gpublasStatus_t hpr2<gpuDoubleComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                                int64_t, const gpuDoubleComplex *,
                                                                const gpuDoubleComplex *, int64_t,
                                                                const gpuDoubleComplex *, int64_t,
                                                                gpuDoubleComplex *);

// Function: gemvBatched
extern template gpublasStatus_t gemvBatched<float, int>(gpublasHandle_t, gpublasOperation_t, int,
                                                        int, const float *, const float *const[],
                                                        int, const float *const[], int,
                                                        const float *, float *const[], int, int);
extern template gpublasStatus_t
gemvBatched<float, int64_t>(gpublasHandle_t, gpublasOperation_t, int64_t, int64_t, const float *,
                            const float *const[], int64_t, const float *const[], int64_t,
                            const float *, float *const[], int64_t, int64_t);
extern template gpublasStatus_t gemvBatched<double, int>(gpublasHandle_t, gpublasOperation_t, int,
                                                         int, const double *, const double *const[],
                                                         int, const double *const[], int,
                                                         const double *, double *const[], int, int);
extern template gpublasStatus_t
gemvBatched<double, int64_t>(gpublasHandle_t, gpublasOperation_t, int64_t, int64_t, const double *,
                             const double *const[], int64_t, const double *const[], int64_t,
                             const double *, double *const[], int64_t, int64_t);
extern template gpublasStatus_t
gemvBatched<gpuFloatComplex, int>(gpublasHandle_t, gpublasOperation_t, int, int,
                                  const gpuFloatComplex *, const gpuFloatComplex *const[], int,
                                  const gpuFloatComplex *const[], int, const gpuFloatComplex *,
                                  gpuFloatComplex *const[], int, int);
extern template gpublasStatus_t gemvBatched<gpuFloatComplex, int64_t>(
    gpublasHandle_t, gpublasOperation_t, int64_t, int64_t, const gpuFloatComplex *,
    const gpuFloatComplex *const[], int64_t, const gpuFloatComplex *const[], int64_t,
    const gpuFloatComplex *, gpuFloatComplex *const[], int64_t, int64_t);
extern template gpublasStatus_t
gemvBatched<gpuDoubleComplex, int>(gpublasHandle_t, gpublasOperation_t, int, int,
                                   const gpuDoubleComplex *, const gpuDoubleComplex *const[], int,
                                   const gpuDoubleComplex *const[], int, const gpuDoubleComplex *,
                                   gpuDoubleComplex *const[], int, int);
extern template gpublasStatus_t gemvBatched<gpuDoubleComplex, int64_t>(
    gpublasHandle_t, gpublasOperation_t, int64_t, int64_t, const gpuDoubleComplex *,
    const gpuDoubleComplex *const[], int64_t, const gpuDoubleComplex *const[], int64_t,
    const gpuDoubleComplex *, gpuDoubleComplex *const[], int64_t, int64_t);

// Function: gemvStridedBatched
extern template gpublasStatus_t
gemvStridedBatched<float, int>(gpublasHandle_t, gpublasOperation_t, int, int, const float *,
                               const float *, int, long long int, const float *, int, long long int,
                               const float *, float *, int, long long int, int);
extern template gpublasStatus_t
gemvStridedBatched<float, int64_t>(gpublasHandle_t, gpublasOperation_t, int64_t, int64_t,
                                   const float *, const float *, int64_t, long long int,
                                   const float *, int64_t, long long int, const float *, float *,
                                   int64_t, long long int, int64_t);
extern template gpublasStatus_t
gemvStridedBatched<double, int>(gpublasHandle_t, gpublasOperation_t, int, int, const double *,
                                const double *, int, long long int, const double *, int,
                                long long int, const double *, double *, int, long long int, int);
extern template gpublasStatus_t
gemvStridedBatched<double, int64_t>(gpublasHandle_t, gpublasOperation_t, int64_t, int64_t,
                                    const double *, const double *, int64_t, long long int,
                                    const double *, int64_t, long long int, const double *,
                                    double *, int64_t, long long int, int64_t);
extern template gpublasStatus_t gemvStridedBatched<gpuFloatComplex, int>(
    gpublasHandle_t, gpublasOperation_t, int, int, const gpuFloatComplex *, const gpuFloatComplex *,
    int, long long int, const gpuFloatComplex *, int, long long int, const gpuFloatComplex *,
    gpuFloatComplex *, int, long long int, int);
extern template gpublasStatus_t gemvStridedBatched<gpuFloatComplex, int64_t>(
    gpublasHandle_t, gpublasOperation_t, int64_t, int64_t, const gpuFloatComplex *,
    const gpuFloatComplex *, int64_t, long long int, const gpuFloatComplex *, int64_t,
    long long int, const gpuFloatComplex *, gpuFloatComplex *, int64_t, long long int, int64_t);
extern template gpublasStatus_t gemvStridedBatched<gpuDoubleComplex, int>(
    gpublasHandle_t, gpublasOperation_t, int, int, const gpuDoubleComplex *,
    const gpuDoubleComplex *, int, long long int, const gpuDoubleComplex *, int, long long int,
    const gpuDoubleComplex *, gpuDoubleComplex *, int, long long int, int);
extern template gpublasStatus_t gemvStridedBatched<gpuDoubleComplex, int64_t>(
    gpublasHandle_t, gpublasOperation_t, int64_t, int64_t, const gpuDoubleComplex *,
    const gpuDoubleComplex *, int64_t, long long int, const gpuDoubleComplex *, int64_t,
    long long int, const gpuDoubleComplex *, gpuDoubleComplex *, int64_t, long long int, int64_t);
} // namespace gpumod
