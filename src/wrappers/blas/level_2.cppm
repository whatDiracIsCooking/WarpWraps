/**
 * @file level_2.cppm
 * @brief GPU BLAS Level 2 (matrix-vector) operations
 *
 * This module provides type-safe wrappers for GPU BLAS Level 2 BLAS operations.
 *
 * Usage:
 *   import wwr.wrappers.blas;
 */

module;

#include "dispatch_macros.h"

export module wwr.wrappers.blas:level_2;

import wwr.blas;
import wwr.complex;
import :type_traits;
import std;

export namespace wwr {

// ========================================================================
// General matrix-vector multiplication: y = alpha*op(A)*x + beta*y
// ========================================================================

template<usual_fp T, int_type IntT>
wwrblasStatus_t gemv(wwrblasHandle_t handle, wwrblasOperation_t trans, IntT m, IntT n,
                     const T *alpha, const T *A, IntT lda, const T *x, IntT incx, const T *beta,
                     T *y, IntT incy) {
  WWR_USUAL_DISPATCH_64(T, IntT, gemv, handle, trans, m, n, alpha, A, lda, x, incx, beta, y,
                           incy);
}

// ========================================================================
// General banded matrix-vector multiplication: y = alpha*op(A)*x + beta*y
// ========================================================================

template<usual_fp T, int_type IntT>
wwrblasStatus_t gbmv(wwrblasHandle_t handle, wwrblasOperation_t trans, IntT m, IntT n, IntT kl,
                     IntT ku, const T *alpha, const T *A, IntT lda, const T *x, IntT incx,
                     const T *beta, T *y, IntT incy) {
  WWR_USUAL_DISPATCH_64(T, IntT, gbmv, handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta,
                           y, incy);
}

// ========================================================================
// General rank-1 update: A = alpha*x*y^T + A (real)
// ========================================================================

template<real_fp T, int_type IntT>
wwrblasStatus_t ger(wwrblasHandle_t handle, IntT m, IntT n, const T *alpha, const T *x, IntT incx,
                    const T *y, IntT incy, T *A, IntT lda) {
  WWR_REAL_DISPATCH_64(T, IntT, wwrblas, S, D, ger, handle, m, n, alpha, x, incx, y, incy, A,
                          lda);
}

// ========================================================================
// General rank-1 update (complex, unconjugated): A = alpha*x*y^T + A
// ========================================================================

template<complex_fp T, int_type IntT>
wwrblasStatus_t geru(wwrblasHandle_t handle, IntT m, IntT n, const T *alpha, const T *x, IntT incx,
                     const T *y, IntT incy, T *A, IntT lda) {
  WWR_COMPLEX_DISPATCH_64(T, IntT, wwrblas, C, Z, geru, handle, m, n, alpha, x, incx, y, incy, A,
                             lda);
}

// ========================================================================
// General rank-1 update (complex, conjugated): A = alpha*x*y^H + A
// ========================================================================

template<complex_fp T, int_type IntT>
wwrblasStatus_t gerc(wwrblasHandle_t handle, IntT m, IntT n, const T *alpha, const T *x, IntT incx,
                     const T *y, IntT incy, T *A, IntT lda) {
  WWR_COMPLEX_DISPATCH_64(T, IntT, wwrblas, C, Z, gerc, handle, m, n, alpha, x, incx, y, incy, A,
                             lda);
}

// ========================================================================
// Symmetric matrix-vector multiplication: y = alpha*A*x + beta*y
// ========================================================================

template<real_fp T, int_type IntT>
wwrblasStatus_t symv(wwrblasHandle_t handle, wwrblasFillMode_t uplo, IntT n, const T *alpha,
                     const T *A, IntT lda, const T *x, IntT incx, const T *beta, T *y, IntT incy) {
  WWR_REAL_DISPATCH_64(T, IntT, wwrblas, S, D, symv, handle, uplo, n, alpha, A, lda, x, incx,
                          beta, y, incy);
}

// ========================================================================
// Symmetric rank-1 update: A = alpha*x*x^T + A
// ========================================================================

template<real_fp T, int_type IntT>
wwrblasStatus_t syr(wwrblasHandle_t handle, wwrblasFillMode_t uplo, IntT n, const T *alpha,
                    const T *x, IntT incx, T *A, IntT lda) {
  WWR_REAL_DISPATCH_64(T, IntT, wwrblas, S, D, syr, handle, uplo, n, alpha, x, incx, A, lda);
}

// ========================================================================
// Symmetric rank-2 update: A = alpha*x*y^T + alpha*y*x^T + A
// ========================================================================

template<real_fp T, int_type IntT>
wwrblasStatus_t syr2(wwrblasHandle_t handle, wwrblasFillMode_t uplo, IntT n, const T *alpha,
                     const T *x, IntT incx, const T *y, IntT incy, T *A, IntT lda) {
  WWR_REAL_DISPATCH_64(T, IntT, wwrblas, S, D, syr2, handle, uplo, n, alpha, x, incx, y, incy, A,
                          lda);
}

// ========================================================================
// Symmetric banded matrix-vector multiplication: y = alpha*A*x + beta*y
// ========================================================================

template<real_fp T, int_type IntT>
wwrblasStatus_t sbmv(wwrblasHandle_t handle, wwrblasFillMode_t uplo, IntT n, IntT k, const T *alpha,
                     const T *A, IntT lda, const T *x, IntT incx, const T *beta, T *y, IntT incy) {
  WWR_REAL_DISPATCH_64(T, IntT, wwrblas, S, D, sbmv, handle, uplo, n, k, alpha, A, lda, x, incx,
                          beta, y, incy);
}

// ========================================================================
// Symmetric packed matrix-vector multiplication: y = alpha*A*x + beta*y
// ========================================================================

template<real_fp T, int_type IntT>
wwrblasStatus_t spmv(wwrblasHandle_t handle, wwrblasFillMode_t uplo, IntT n, const T *alpha,
                     const T *AP, const T *x, IntT incx, const T *beta, T *y, IntT incy) {
  WWR_REAL_DISPATCH_64(T, IntT, wwrblas, S, D, spmv, handle, uplo, n, alpha, AP, x, incx, beta,
                          y, incy);
}

// ========================================================================
// Symmetric packed rank-1 update: A = alpha*x*x^T + A
// ========================================================================

template<real_fp T, int_type IntT>
wwrblasStatus_t spr(wwrblasHandle_t handle, wwrblasFillMode_t uplo, IntT n, const T *alpha,
                    const T *x, IntT incx, T *AP) {
  WWR_REAL_DISPATCH_64(T, IntT, wwrblas, S, D, spr, handle, uplo, n, alpha, x, incx, AP);
}

// ========================================================================
// Symmetric packed rank-2 update: A = alpha*x*y^T + alpha*y*x^T + A
// ========================================================================

template<real_fp T, int_type IntT>
wwrblasStatus_t spr2(wwrblasHandle_t handle, wwrblasFillMode_t uplo, IntT n, const T *alpha,
                     const T *x, IntT incx, const T *y, IntT incy, T *AP) {
  WWR_REAL_DISPATCH_64(T, IntT, wwrblas, S, D, spr2, handle, uplo, n, alpha, x, incx, y, incy,
                          AP);
}

// ========================================================================
// Triangular matrix-vector multiplication: x = op(A)*x
// ========================================================================

template<usual_fp T, int_type IntT>
wwrblasStatus_t trmv(wwrblasHandle_t handle, wwrblasFillMode_t uplo, wwrblasOperation_t trans,
                     wwrblasDiagType_t diag, IntT n, const T *A, IntT lda, T *x, IntT incx) {
  WWR_USUAL_DISPATCH_64(T, IntT, trmv, handle, uplo, trans, diag, n, A, lda, x, incx);
}

// ========================================================================
// Triangular solve: op(A)*x = b
// ========================================================================

template<usual_fp T, int_type IntT>
wwrblasStatus_t trsv(wwrblasHandle_t handle, wwrblasFillMode_t uplo, wwrblasOperation_t trans,
                     wwrblasDiagType_t diag, IntT n, const T *A, IntT lda, T *x, IntT incx) {
  WWR_USUAL_DISPATCH_64(T, IntT, trsv, handle, uplo, trans, diag, n, A, lda, x, incx);
}

// ========================================================================
// Triangular banded matrix-vector multiplication: x = op(A)*x
// ========================================================================

template<usual_fp T, int_type IntT>
wwrblasStatus_t tbmv(wwrblasHandle_t handle, wwrblasFillMode_t uplo, wwrblasOperation_t trans,
                     wwrblasDiagType_t diag, IntT n, IntT k, const T *A, IntT lda, T *x,
                     IntT incx) {
  WWR_USUAL_DISPATCH_64(T, IntT, tbmv, handle, uplo, trans, diag, n, k, A, lda, x, incx);
}

// ========================================================================
// Triangular banded solve: op(A)*x = b
// ========================================================================

template<usual_fp T, int_type IntT>
wwrblasStatus_t tbsv(wwrblasHandle_t handle, wwrblasFillMode_t uplo, wwrblasOperation_t trans,
                     wwrblasDiagType_t diag, IntT n, IntT k, const T *A, IntT lda, T *x,
                     IntT incx) {
  WWR_USUAL_DISPATCH_64(T, IntT, tbsv, handle, uplo, trans, diag, n, k, A, lda, x, incx);
}

// ========================================================================
// Triangular packed matrix-vector multiplication: x = op(A)*x
// ========================================================================

template<usual_fp T, int_type IntT>
wwrblasStatus_t tpmv(wwrblasHandle_t handle, wwrblasFillMode_t uplo, wwrblasOperation_t trans,
                     wwrblasDiagType_t diag, IntT n, const T *AP, T *x, IntT incx) {
  WWR_USUAL_DISPATCH_64(T, IntT, tpmv, handle, uplo, trans, diag, n, AP, x, incx);
}

// ========================================================================
// Triangular packed solve: op(A)*x = b
// ========================================================================

template<usual_fp T, int_type IntT>
wwrblasStatus_t tpsv(wwrblasHandle_t handle, wwrblasFillMode_t uplo, wwrblasOperation_t trans,
                     wwrblasDiagType_t diag, IntT n, const T *AP, T *x, IntT incx) {
  WWR_USUAL_DISPATCH_64(T, IntT, tpsv, handle, uplo, trans, diag, n, AP, x, incx);
}

// ========================================================================
// Hermitian matrix-vector multiplication: y = alpha*A*x + beta*y
// ========================================================================

template<complex_fp T, int_type IntT>
wwrblasStatus_t hemv(wwrblasHandle_t handle, wwrblasFillMode_t uplo, IntT n, const T *alpha,
                     const T *A, IntT lda, const T *x, IntT incx, const T *beta, T *y, IntT incy) {
  WWR_COMPLEX_DISPATCH_64(T, IntT, wwrblas, C, Z, hemv, handle, uplo, n, alpha, A, lda, x, incx,
                             beta, y, incy);
}

// ========================================================================
// Hermitian banded matrix-vector multiplication: y = alpha*A*x + beta*y
// ========================================================================

template<complex_fp T, int_type IntT>
wwrblasStatus_t hbmv(wwrblasHandle_t handle, wwrblasFillMode_t uplo, IntT n, IntT k, const T *alpha,
                     const T *A, IntT lda, const T *x, IntT incx, const T *beta, T *y, IntT incy) {
  WWR_COMPLEX_DISPATCH_64(T, IntT, wwrblas, C, Z, hbmv, handle, uplo, n, k, alpha, A, lda, x,
                             incx, beta, y, incy);
}

// ========================================================================
// Hermitian packed matrix-vector multiplication: y = alpha*A*x + beta*y
// ========================================================================

template<complex_fp T, int_type IntT>
wwrblasStatus_t hpmv(wwrblasHandle_t handle, wwrblasFillMode_t uplo, IntT n, const T *alpha,
                     const T *AP, const T *x, IntT incx, const T *beta, T *y, IntT incy) {
  WWR_COMPLEX_DISPATCH_64(T, IntT, wwrblas, C, Z, hpmv, handle, uplo, n, alpha, AP, x, incx,
                             beta, y, incy);
}

// ========================================================================
// Hermitian rank-1 update: A = alpha*x*x^H + A
// ========================================================================

template<complex_fp T, int_type IntT>
wwrblasStatus_t her(wwrblasHandle_t handle, wwrblasFillMode_t uplo, IntT n,
                    const ComplexToRealType<T> *alpha, const T *x, IntT incx, T *A, IntT lda) {
  WWR_COMPLEX_DISPATCH_64(T, IntT, wwrblas, C, Z, her, handle, uplo, n, alpha, x, incx, A, lda);
}

// ========================================================================
// Hermitian rank-2 update: A = alpha*x*y^H + conj(alpha)*y*x^H + A
// ========================================================================

template<complex_fp T, int_type IntT>
wwrblasStatus_t her2(wwrblasHandle_t handle, wwrblasFillMode_t uplo, IntT n, const T *alpha,
                     const T *x, IntT incx, const T *y, IntT incy, T *A, IntT lda) {
  WWR_COMPLEX_DISPATCH_64(T, IntT, wwrblas, C, Z, her2, handle, uplo, n, alpha, x, incx, y, incy,
                             A, lda);
}

// ========================================================================
// Hermitian packed rank-1 update: A = alpha*x*x^H + A
// ========================================================================

template<complex_fp T, int_type IntT>
wwrblasStatus_t hpr(wwrblasHandle_t handle, wwrblasFillMode_t uplo, IntT n,
                    const ComplexToRealType<T> *alpha, const T *x, IntT incx, T *AP) {
  WWR_COMPLEX_DISPATCH_64(T, IntT, wwrblas, C, Z, hpr, handle, uplo, n, alpha, x, incx, AP);
}

// ========================================================================
// Hermitian packed rank-2 update: A = alpha*x*y^H + conj(alpha)*y*x^H + A
// ========================================================================

template<complex_fp T, int_type IntT>
wwrblasStatus_t hpr2(wwrblasHandle_t handle, wwrblasFillMode_t uplo, IntT n, const T *alpha,
                     const T *x, IntT incx, const T *y, IntT incy, T *AP) {
  WWR_COMPLEX_DISPATCH_64(T, IntT, wwrblas, C, Z, hpr2, handle, uplo, n, alpha, x, incx, y, incy,
                             AP);
}

// ========================================================================
// Batched general matrix-vector multiplication
// ========================================================================

template<usual_fp T, int_type IntT>
wwrblasStatus_t gemvBatched(wwrblasHandle_t handle, wwrblasOperation_t trans, IntT m, IntT n,
                            const T *alpha, const T *const Aarray[], IntT lda,
                            const T *const xarray[], IntT incx, const T *beta, T *const yarray[],
                            IntT incy, IntT batchCount) {
  WWR_USUAL_DISPATCH_64(T, IntT, gemvBatched, handle, trans, m, n, alpha, Aarray, lda, xarray,
                           incx, beta, yarray, incy, batchCount);
}

// ========================================================================
// Strided batched general matrix-vector multiplication
// ========================================================================

template<usual_fp T, int_type IntT>
wwrblasStatus_t gemvStridedBatched(wwrblasHandle_t handle, wwrblasOperation_t trans, IntT m, IntT n,
                                   const T *alpha, const T *A, IntT lda, long long int strideA,
                                   const T *x, IntT incx, long long int stridex, const T *beta,
                                   T *y, IntT incy, long long int stridey, IntT batchCount) {
  WWR_USUAL_DISPATCH_64(T, IntT, gemvStridedBatched, handle, trans, m, n, alpha, A, lda, strideA,
                           x, incx, stridex, beta, y, incy, stridey, batchCount);
}

// ==================== Explicit Template Instantiations ====================
// Matching `template` instantiations live in instantiations.cpp.

// Function: gemv
extern template wwrblasStatus_t gemv<float, int>(wwrblasHandle_t, wwrblasOperation_t, int, int,
                                                 const float *, const float *, int, const float *,
                                                 int, const float *, float *, int);
extern template wwrblasStatus_t gemv<float, int64_t>(wwrblasHandle_t, wwrblasOperation_t, int64_t,
                                                     int64_t, const float *, const float *, int64_t,
                                                     const float *, int64_t, const float *, float *,
                                                     int64_t);
extern template wwrblasStatus_t gemv<double, int>(wwrblasHandle_t, wwrblasOperation_t, int, int,
                                                  const double *, const double *, int,
                                                  const double *, int, const double *, double *,
                                                  int);
extern template wwrblasStatus_t gemv<double, int64_t>(wwrblasHandle_t, wwrblasOperation_t, int64_t,
                                                      int64_t, const double *, const double *,
                                                      int64_t, const double *, int64_t,
                                                      const double *, double *, int64_t);
extern template wwrblasStatus_t
gemv<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasOperation_t, int, int, const wwrFloatComplex *,
                           const wwrFloatComplex *, int, const wwrFloatComplex *, int,
                           const wwrFloatComplex *, wwrFloatComplex *, int);
extern template wwrblasStatus_t
gemv<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasOperation_t, int64_t, int64_t,
                               const wwrFloatComplex *, const wwrFloatComplex *, int64_t,
                               const wwrFloatComplex *, int64_t, const wwrFloatComplex *,
                               wwrFloatComplex *, int64_t);
extern template wwrblasStatus_t
gemv<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasOperation_t, int, int, const wwrDoubleComplex *,
                            const wwrDoubleComplex *, int, const wwrDoubleComplex *, int,
                            const wwrDoubleComplex *, wwrDoubleComplex *, int);
extern template wwrblasStatus_t
gemv<wwrDoubleComplex, int64_t>(wwrblasHandle_t, wwrblasOperation_t, int64_t, int64_t,
                                const wwrDoubleComplex *, const wwrDoubleComplex *, int64_t,
                                const wwrDoubleComplex *, int64_t, const wwrDoubleComplex *,
                                wwrDoubleComplex *, int64_t);

// Function: gbmv
extern template wwrblasStatus_t gbmv<float, int>(wwrblasHandle_t, wwrblasOperation_t, int, int, int,
                                                 int, const float *, const float *, int,
                                                 const float *, int, const float *, float *, int);
extern template wwrblasStatus_t gbmv<float, int64_t>(wwrblasHandle_t, wwrblasOperation_t, int64_t,
                                                     int64_t, int64_t, int64_t, const float *,
                                                     const float *, int64_t, const float *, int64_t,
                                                     const float *, float *, int64_t);
extern template wwrblasStatus_t gbmv<double, int>(wwrblasHandle_t, wwrblasOperation_t, int, int,
                                                  int, int, const double *, const double *, int,
                                                  const double *, int, const double *, double *,
                                                  int);
extern template wwrblasStatus_t gbmv<double, int64_t>(wwrblasHandle_t, wwrblasOperation_t, int64_t,
                                                      int64_t, int64_t, int64_t, const double *,
                                                      const double *, int64_t, const double *,
                                                      int64_t, const double *, double *, int64_t);
extern template wwrblasStatus_t gbmv<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasOperation_t, int,
                                                           int, int, int, const wwrFloatComplex *,
                                                           const wwrFloatComplex *, int,
                                                           const wwrFloatComplex *, int,
                                                           const wwrFloatComplex *,
                                                           wwrFloatComplex *, int);
extern template wwrblasStatus_t
gbmv<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasOperation_t, int64_t, int64_t, int64_t,
                               int64_t, const wwrFloatComplex *, const wwrFloatComplex *, int64_t,
                               const wwrFloatComplex *, int64_t, const wwrFloatComplex *,
                               wwrFloatComplex *, int64_t);
extern template wwrblasStatus_t
gbmv<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasOperation_t, int, int, int, int,
                            const wwrDoubleComplex *, const wwrDoubleComplex *, int,
                            const wwrDoubleComplex *, int, const wwrDoubleComplex *,
                            wwrDoubleComplex *, int);
extern template wwrblasStatus_t
gbmv<wwrDoubleComplex, int64_t>(wwrblasHandle_t, wwrblasOperation_t, int64_t, int64_t, int64_t,
                                int64_t, const wwrDoubleComplex *, const wwrDoubleComplex *,
                                int64_t, const wwrDoubleComplex *, int64_t,
                                const wwrDoubleComplex *, wwrDoubleComplex *, int64_t);

// Function: ger
extern template wwrblasStatus_t ger<float, int>(wwrblasHandle_t, int, int, const float *,
                                                const float *, int, const float *, int, float *,
                                                int);
extern template wwrblasStatus_t ger<float, int64_t>(wwrblasHandle_t, int64_t, int64_t,
                                                    const float *, const float *, int64_t,
                                                    const float *, int64_t, float *, int64_t);
extern template wwrblasStatus_t ger<double, int>(wwrblasHandle_t, int, int, const double *,
                                                 const double *, int, const double *, int, double *,
                                                 int);
extern template wwrblasStatus_t ger<double, int64_t>(wwrblasHandle_t, int64_t, int64_t,
                                                     const double *, const double *, int64_t,
                                                     const double *, int64_t, double *, int64_t);

// Function: geru
extern template wwrblasStatus_t geru<wwrFloatComplex, int>(wwrblasHandle_t, int, int,
                                                           const wwrFloatComplex *,
                                                           const wwrFloatComplex *, int,
                                                           const wwrFloatComplex *, int,
                                                           wwrFloatComplex *, int);
extern template wwrblasStatus_t geru<wwrFloatComplex, int64_t>(wwrblasHandle_t, int64_t, int64_t,
                                                               const wwrFloatComplex *,
                                                               const wwrFloatComplex *, int64_t,
                                                               const wwrFloatComplex *, int64_t,
                                                               wwrFloatComplex *, int64_t);
extern template wwrblasStatus_t geru<wwrDoubleComplex, int>(wwrblasHandle_t, int, int,
                                                            const wwrDoubleComplex *,
                                                            const wwrDoubleComplex *, int,
                                                            const wwrDoubleComplex *, int,
                                                            wwrDoubleComplex *, int);
extern template wwrblasStatus_t geru<wwrDoubleComplex, int64_t>(wwrblasHandle_t, int64_t, int64_t,
                                                                const wwrDoubleComplex *,
                                                                const wwrDoubleComplex *, int64_t,
                                                                const wwrDoubleComplex *, int64_t,
                                                                wwrDoubleComplex *, int64_t);

// Function: gerc
extern template wwrblasStatus_t gerc<wwrFloatComplex, int>(wwrblasHandle_t, int, int,
                                                           const wwrFloatComplex *,
                                                           const wwrFloatComplex *, int,
                                                           const wwrFloatComplex *, int,
                                                           wwrFloatComplex *, int);
extern template wwrblasStatus_t gerc<wwrFloatComplex, int64_t>(wwrblasHandle_t, int64_t, int64_t,
                                                               const wwrFloatComplex *,
                                                               const wwrFloatComplex *, int64_t,
                                                               const wwrFloatComplex *, int64_t,
                                                               wwrFloatComplex *, int64_t);
extern template wwrblasStatus_t gerc<wwrDoubleComplex, int>(wwrblasHandle_t, int, int,
                                                            const wwrDoubleComplex *,
                                                            const wwrDoubleComplex *, int,
                                                            const wwrDoubleComplex *, int,
                                                            wwrDoubleComplex *, int);
extern template wwrblasStatus_t gerc<wwrDoubleComplex, int64_t>(wwrblasHandle_t, int64_t, int64_t,
                                                                const wwrDoubleComplex *,
                                                                const wwrDoubleComplex *, int64_t,
                                                                const wwrDoubleComplex *, int64_t,
                                                                wwrDoubleComplex *, int64_t);

// Function: symv
extern template wwrblasStatus_t symv<float, int>(wwrblasHandle_t, wwrblasFillMode_t, int,
                                                 const float *, const float *, int, const float *,
                                                 int, const float *, float *, int);
extern template wwrblasStatus_t symv<float, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, int64_t,
                                                     const float *, const float *, int64_t,
                                                     const float *, int64_t, const float *, float *,
                                                     int64_t);
extern template wwrblasStatus_t symv<double, int>(wwrblasHandle_t, wwrblasFillMode_t, int,
                                                  const double *, const double *, int,
                                                  const double *, int, const double *, double *,
                                                  int);
extern template wwrblasStatus_t symv<double, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, int64_t,
                                                      const double *, const double *, int64_t,
                                                      const double *, int64_t, const double *,
                                                      double *, int64_t);

// Function: syr
extern template wwrblasStatus_t syr<float, int>(wwrblasHandle_t, wwrblasFillMode_t, int,
                                                const float *, const float *, int, float *, int);
extern template wwrblasStatus_t syr<float, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, int64_t,
                                                    const float *, const float *, int64_t, float *,
                                                    int64_t);
extern template wwrblasStatus_t syr<double, int>(wwrblasHandle_t, wwrblasFillMode_t, int,
                                                 const double *, const double *, int, double *,
                                                 int);
extern template wwrblasStatus_t syr<double, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, int64_t,
                                                     const double *, const double *, int64_t,
                                                     double *, int64_t);

// Function: syr2
extern template wwrblasStatus_t syr2<float, int>(wwrblasHandle_t, wwrblasFillMode_t, int,
                                                 const float *, const float *, int, const float *,
                                                 int, float *, int);
extern template wwrblasStatus_t syr2<float, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, int64_t,
                                                     const float *, const float *, int64_t,
                                                     const float *, int64_t, float *, int64_t);
extern template wwrblasStatus_t syr2<double, int>(wwrblasHandle_t, wwrblasFillMode_t, int,
                                                  const double *, const double *, int,
                                                  const double *, int, double *, int);
extern template wwrblasStatus_t syr2<double, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, int64_t,
                                                      const double *, const double *, int64_t,
                                                      const double *, int64_t, double *, int64_t);

// Function: sbmv
extern template wwrblasStatus_t sbmv<float, int>(wwrblasHandle_t, wwrblasFillMode_t, int, int,
                                                 const float *, const float *, int, const float *,
                                                 int, const float *, float *, int);
extern template wwrblasStatus_t sbmv<float, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, int64_t,
                                                     int64_t, const float *, const float *, int64_t,
                                                     const float *, int64_t, const float *, float *,
                                                     int64_t);
extern template wwrblasStatus_t sbmv<double, int>(wwrblasHandle_t, wwrblasFillMode_t, int, int,
                                                  const double *, const double *, int,
                                                  const double *, int, const double *, double *,
                                                  int);
extern template wwrblasStatus_t sbmv<double, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, int64_t,
                                                      int64_t, const double *, const double *,
                                                      int64_t, const double *, int64_t,
                                                      const double *, double *, int64_t);

// Function: spmv
extern template wwrblasStatus_t spmv<float, int>(wwrblasHandle_t, wwrblasFillMode_t, int,
                                                 const float *, const float *, const float *, int,
                                                 const float *, float *, int);
extern template wwrblasStatus_t spmv<float, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, int64_t,
                                                     const float *, const float *, const float *,
                                                     int64_t, const float *, float *, int64_t);
extern template wwrblasStatus_t spmv<double, int>(wwrblasHandle_t, wwrblasFillMode_t, int,
                                                  const double *, const double *, const double *,
                                                  int, const double *, double *, int);
extern template wwrblasStatus_t spmv<double, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, int64_t,
                                                      const double *, const double *,
                                                      const double *, int64_t, const double *,
                                                      double *, int64_t);

// Function: spr
extern template wwrblasStatus_t spr<float, int>(wwrblasHandle_t, wwrblasFillMode_t, int,
                                                const float *, const float *, int, float *);
extern template wwrblasStatus_t spr<float, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, int64_t,
                                                    const float *, const float *, int64_t, float *);
extern template wwrblasStatus_t spr<double, int>(wwrblasHandle_t, wwrblasFillMode_t, int,
                                                 const double *, const double *, int, double *);
extern template wwrblasStatus_t spr<double, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, int64_t,
                                                     const double *, const double *, int64_t,
                                                     double *);

// Function: spr2
extern template wwrblasStatus_t spr2<float, int>(wwrblasHandle_t, wwrblasFillMode_t, int,
                                                 const float *, const float *, int, const float *,
                                                 int, float *);
extern template wwrblasStatus_t spr2<float, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, int64_t,
                                                     const float *, const float *, int64_t,
                                                     const float *, int64_t, float *);
extern template wwrblasStatus_t spr2<double, int>(wwrblasHandle_t, wwrblasFillMode_t, int,
                                                  const double *, const double *, int,
                                                  const double *, int, double *);
extern template wwrblasStatus_t spr2<double, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, int64_t,
                                                      const double *, const double *, int64_t,
                                                      const double *, int64_t, double *);

// Function: trmv
extern template wwrblasStatus_t trmv<float, int>(wwrblasHandle_t, wwrblasFillMode_t,
                                                 wwrblasOperation_t, wwrblasDiagType_t, int,
                                                 const float *, int, float *, int);
extern template wwrblasStatus_t trmv<float, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                     wwrblasOperation_t, wwrblasDiagType_t, int64_t,
                                                     const float *, int64_t, float *, int64_t);
extern template wwrblasStatus_t trmv<double, int>(wwrblasHandle_t, wwrblasFillMode_t,
                                                  wwrblasOperation_t, wwrblasDiagType_t, int,
                                                  const double *, int, double *, int);
extern template wwrblasStatus_t trmv<double, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                      wwrblasOperation_t, wwrblasDiagType_t,
                                                      int64_t, const double *, int64_t, double *,
                                                      int64_t);
extern template wwrblasStatus_t trmv<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasFillMode_t,
                                                           wwrblasOperation_t, wwrblasDiagType_t,
                                                           int, const wwrFloatComplex *, int,
                                                           wwrFloatComplex *, int);
extern template wwrblasStatus_t trmv<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                               wwrblasOperation_t,
                                                               wwrblasDiagType_t, int64_t,
                                                               const wwrFloatComplex *, int64_t,
                                                               wwrFloatComplex *, int64_t);
extern template wwrblasStatus_t trmv<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasFillMode_t,
                                                            wwrblasOperation_t, wwrblasDiagType_t,
                                                            int, const wwrDoubleComplex *, int,
                                                            wwrDoubleComplex *, int);
extern template wwrblasStatus_t trmv<wwrDoubleComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                                wwrblasOperation_t,
                                                                wwrblasDiagType_t, int64_t,
                                                                const wwrDoubleComplex *, int64_t,
                                                                wwrDoubleComplex *, int64_t);

// Function: trsv
extern template wwrblasStatus_t trsv<float, int>(wwrblasHandle_t, wwrblasFillMode_t,
                                                 wwrblasOperation_t, wwrblasDiagType_t, int,
                                                 const float *, int, float *, int);
extern template wwrblasStatus_t trsv<float, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                     wwrblasOperation_t, wwrblasDiagType_t, int64_t,
                                                     const float *, int64_t, float *, int64_t);
extern template wwrblasStatus_t trsv<double, int>(wwrblasHandle_t, wwrblasFillMode_t,
                                                  wwrblasOperation_t, wwrblasDiagType_t, int,
                                                  const double *, int, double *, int);
extern template wwrblasStatus_t trsv<double, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                      wwrblasOperation_t, wwrblasDiagType_t,
                                                      int64_t, const double *, int64_t, double *,
                                                      int64_t);
extern template wwrblasStatus_t trsv<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasFillMode_t,
                                                           wwrblasOperation_t, wwrblasDiagType_t,
                                                           int, const wwrFloatComplex *, int,
                                                           wwrFloatComplex *, int);
extern template wwrblasStatus_t trsv<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                               wwrblasOperation_t,
                                                               wwrblasDiagType_t, int64_t,
                                                               const wwrFloatComplex *, int64_t,
                                                               wwrFloatComplex *, int64_t);
extern template wwrblasStatus_t trsv<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasFillMode_t,
                                                            wwrblasOperation_t, wwrblasDiagType_t,
                                                            int, const wwrDoubleComplex *, int,
                                                            wwrDoubleComplex *, int);
extern template wwrblasStatus_t trsv<wwrDoubleComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                                wwrblasOperation_t,
                                                                wwrblasDiagType_t, int64_t,
                                                                const wwrDoubleComplex *, int64_t,
                                                                wwrDoubleComplex *, int64_t);

// Function: tbmv
extern template wwrblasStatus_t tbmv<float, int>(wwrblasHandle_t, wwrblasFillMode_t,
                                                 wwrblasOperation_t, wwrblasDiagType_t, int, int,
                                                 const float *, int, float *, int);
extern template wwrblasStatus_t tbmv<float, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                     wwrblasOperation_t, wwrblasDiagType_t, int64_t,
                                                     int64_t, const float *, int64_t, float *,
                                                     int64_t);
extern template wwrblasStatus_t tbmv<double, int>(wwrblasHandle_t, wwrblasFillMode_t,
                                                  wwrblasOperation_t, wwrblasDiagType_t, int, int,
                                                  const double *, int, double *, int);
extern template wwrblasStatus_t tbmv<double, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                      wwrblasOperation_t, wwrblasDiagType_t,
                                                      int64_t, int64_t, const double *, int64_t,
                                                      double *, int64_t);
extern template wwrblasStatus_t tbmv<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasFillMode_t,
                                                           wwrblasOperation_t, wwrblasDiagType_t,
                                                           int, int, const wwrFloatComplex *, int,
                                                           wwrFloatComplex *, int);
extern template wwrblasStatus_t tbmv<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                               wwrblasOperation_t,
                                                               wwrblasDiagType_t, int64_t, int64_t,
                                                               const wwrFloatComplex *, int64_t,
                                                               wwrFloatComplex *, int64_t);
extern template wwrblasStatus_t tbmv<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasFillMode_t,
                                                            wwrblasOperation_t, wwrblasDiagType_t,
                                                            int, int, const wwrDoubleComplex *, int,
                                                            wwrDoubleComplex *, int);
extern template wwrblasStatus_t tbmv<wwrDoubleComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                                wwrblasOperation_t,
                                                                wwrblasDiagType_t, int64_t, int64_t,
                                                                const wwrDoubleComplex *, int64_t,
                                                                wwrDoubleComplex *, int64_t);

// Function: tbsv
extern template wwrblasStatus_t tbsv<float, int>(wwrblasHandle_t, wwrblasFillMode_t,
                                                 wwrblasOperation_t, wwrblasDiagType_t, int, int,
                                                 const float *, int, float *, int);
extern template wwrblasStatus_t tbsv<float, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                     wwrblasOperation_t, wwrblasDiagType_t, int64_t,
                                                     int64_t, const float *, int64_t, float *,
                                                     int64_t);
extern template wwrblasStatus_t tbsv<double, int>(wwrblasHandle_t, wwrblasFillMode_t,
                                                  wwrblasOperation_t, wwrblasDiagType_t, int, int,
                                                  const double *, int, double *, int);
extern template wwrblasStatus_t tbsv<double, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                      wwrblasOperation_t, wwrblasDiagType_t,
                                                      int64_t, int64_t, const double *, int64_t,
                                                      double *, int64_t);
extern template wwrblasStatus_t tbsv<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasFillMode_t,
                                                           wwrblasOperation_t, wwrblasDiagType_t,
                                                           int, int, const wwrFloatComplex *, int,
                                                           wwrFloatComplex *, int);
extern template wwrblasStatus_t tbsv<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                               wwrblasOperation_t,
                                                               wwrblasDiagType_t, int64_t, int64_t,
                                                               const wwrFloatComplex *, int64_t,
                                                               wwrFloatComplex *, int64_t);
extern template wwrblasStatus_t tbsv<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasFillMode_t,
                                                            wwrblasOperation_t, wwrblasDiagType_t,
                                                            int, int, const wwrDoubleComplex *, int,
                                                            wwrDoubleComplex *, int);
extern template wwrblasStatus_t tbsv<wwrDoubleComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                                wwrblasOperation_t,
                                                                wwrblasDiagType_t, int64_t, int64_t,
                                                                const wwrDoubleComplex *, int64_t,
                                                                wwrDoubleComplex *, int64_t);

// Function: tpmv
extern template wwrblasStatus_t tpmv<float, int>(wwrblasHandle_t, wwrblasFillMode_t,
                                                 wwrblasOperation_t, wwrblasDiagType_t, int,
                                                 const float *, float *, int);
extern template wwrblasStatus_t tpmv<float, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                     wwrblasOperation_t, wwrblasDiagType_t, int64_t,
                                                     const float *, float *, int64_t);
extern template wwrblasStatus_t tpmv<double, int>(wwrblasHandle_t, wwrblasFillMode_t,
                                                  wwrblasOperation_t, wwrblasDiagType_t, int,
                                                  const double *, double *, int);
extern template wwrblasStatus_t tpmv<double, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                      wwrblasOperation_t, wwrblasDiagType_t,
                                                      int64_t, const double *, double *, int64_t);
extern template wwrblasStatus_t tpmv<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasFillMode_t,
                                                           wwrblasOperation_t, wwrblasDiagType_t,
                                                           int, const wwrFloatComplex *,
                                                           wwrFloatComplex *, int);
extern template wwrblasStatus_t tpmv<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                               wwrblasOperation_t,
                                                               wwrblasDiagType_t, int64_t,
                                                               const wwrFloatComplex *,
                                                               wwrFloatComplex *, int64_t);
extern template wwrblasStatus_t tpmv<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasFillMode_t,
                                                            wwrblasOperation_t, wwrblasDiagType_t,
                                                            int, const wwrDoubleComplex *,
                                                            wwrDoubleComplex *, int);
extern template wwrblasStatus_t tpmv<wwrDoubleComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                                wwrblasOperation_t,
                                                                wwrblasDiagType_t, int64_t,
                                                                const wwrDoubleComplex *,
                                                                wwrDoubleComplex *, int64_t);

// Function: tpsv
extern template wwrblasStatus_t tpsv<float, int>(wwrblasHandle_t, wwrblasFillMode_t,
                                                 wwrblasOperation_t, wwrblasDiagType_t, int,
                                                 const float *, float *, int);
extern template wwrblasStatus_t tpsv<float, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                     wwrblasOperation_t, wwrblasDiagType_t, int64_t,
                                                     const float *, float *, int64_t);
extern template wwrblasStatus_t tpsv<double, int>(wwrblasHandle_t, wwrblasFillMode_t,
                                                  wwrblasOperation_t, wwrblasDiagType_t, int,
                                                  const double *, double *, int);
extern template wwrblasStatus_t tpsv<double, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                      wwrblasOperation_t, wwrblasDiagType_t,
                                                      int64_t, const double *, double *, int64_t);
extern template wwrblasStatus_t tpsv<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasFillMode_t,
                                                           wwrblasOperation_t, wwrblasDiagType_t,
                                                           int, const wwrFloatComplex *,
                                                           wwrFloatComplex *, int);
extern template wwrblasStatus_t tpsv<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                               wwrblasOperation_t,
                                                               wwrblasDiagType_t, int64_t,
                                                               const wwrFloatComplex *,
                                                               wwrFloatComplex *, int64_t);
extern template wwrblasStatus_t tpsv<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasFillMode_t,
                                                            wwrblasOperation_t, wwrblasDiagType_t,
                                                            int, const wwrDoubleComplex *,
                                                            wwrDoubleComplex *, int);
extern template wwrblasStatus_t tpsv<wwrDoubleComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                                wwrblasOperation_t,
                                                                wwrblasDiagType_t, int64_t,
                                                                const wwrDoubleComplex *,
                                                                wwrDoubleComplex *, int64_t);

// Function: hemv
extern template wwrblasStatus_t
hemv<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasFillMode_t, int, const wwrFloatComplex *,
                           const wwrFloatComplex *, int, const wwrFloatComplex *, int,
                           const wwrFloatComplex *, wwrFloatComplex *, int);
extern template wwrblasStatus_t
hemv<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, int64_t, const wwrFloatComplex *,
                               const wwrFloatComplex *, int64_t, const wwrFloatComplex *, int64_t,
                               const wwrFloatComplex *, wwrFloatComplex *, int64_t);
extern template wwrblasStatus_t
hemv<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasFillMode_t, int, const wwrDoubleComplex *,
                            const wwrDoubleComplex *, int, const wwrDoubleComplex *, int,
                            const wwrDoubleComplex *, wwrDoubleComplex *, int);
extern template wwrblasStatus_t hemv<wwrDoubleComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                                int64_t, const wwrDoubleComplex *,
                                                                const wwrDoubleComplex *, int64_t,
                                                                const wwrDoubleComplex *, int64_t,
                                                                const wwrDoubleComplex *,
                                                                wwrDoubleComplex *, int64_t);

// Function: hbmv
extern template wwrblasStatus_t
hbmv<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasFillMode_t, int, int, const wwrFloatComplex *,
                           const wwrFloatComplex *, int, const wwrFloatComplex *, int,
                           const wwrFloatComplex *, wwrFloatComplex *, int);
extern template wwrblasStatus_t
hbmv<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, int64_t, int64_t,
                               const wwrFloatComplex *, const wwrFloatComplex *, int64_t,
                               const wwrFloatComplex *, int64_t, const wwrFloatComplex *,
                               wwrFloatComplex *, int64_t);
extern template wwrblasStatus_t
hbmv<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasFillMode_t, int, int, const wwrDoubleComplex *,
                            const wwrDoubleComplex *, int, const wwrDoubleComplex *, int,
                            const wwrDoubleComplex *, wwrDoubleComplex *, int);
extern template wwrblasStatus_t
hbmv<wwrDoubleComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, int64_t, int64_t,
                                const wwrDoubleComplex *, const wwrDoubleComplex *, int64_t,
                                const wwrDoubleComplex *, int64_t, const wwrDoubleComplex *,
                                wwrDoubleComplex *, int64_t);

// Function: hpmv
extern template wwrblasStatus_t
hpmv<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasFillMode_t, int, const wwrFloatComplex *,
                           const wwrFloatComplex *, const wwrFloatComplex *, int,
                           const wwrFloatComplex *, wwrFloatComplex *, int);
extern template wwrblasStatus_t
hpmv<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, int64_t, const wwrFloatComplex *,
                               const wwrFloatComplex *, const wwrFloatComplex *, int64_t,
                               const wwrFloatComplex *, wwrFloatComplex *, int64_t);
extern template wwrblasStatus_t
hpmv<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasFillMode_t, int, const wwrDoubleComplex *,
                            const wwrDoubleComplex *, const wwrDoubleComplex *, int,
                            const wwrDoubleComplex *, wwrDoubleComplex *, int);
extern template wwrblasStatus_t hpmv<wwrDoubleComplex, int64_t>(
    wwrblasHandle_t, wwrblasFillMode_t, int64_t, const wwrDoubleComplex *, const wwrDoubleComplex *,
    const wwrDoubleComplex *, int64_t, const wwrDoubleComplex *, wwrDoubleComplex *, int64_t);

// Function: her
extern template wwrblasStatus_t
her<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasFillMode_t, int,
                          const ComplexToRealType<wwrFloatComplex> *, const wwrFloatComplex *, int,
                          wwrFloatComplex *, int);
extern template wwrblasStatus_t
her<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, int64_t,
                              const ComplexToRealType<wwrFloatComplex> *, const wwrFloatComplex *,
                              int64_t, wwrFloatComplex *, int64_t);
extern template wwrblasStatus_t
her<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasFillMode_t, int,
                           const ComplexToRealType<wwrDoubleComplex> *, const wwrDoubleComplex *,
                           int, wwrDoubleComplex *, int);
extern template wwrblasStatus_t
her<wwrDoubleComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, int64_t,
                               const ComplexToRealType<wwrDoubleComplex> *,
                               const wwrDoubleComplex *, int64_t, wwrDoubleComplex *, int64_t);

// Function: her2
extern template wwrblasStatus_t her2<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasFillMode_t, int,
                                                           const wwrFloatComplex *,
                                                           const wwrFloatComplex *, int,
                                                           const wwrFloatComplex *, int,
                                                           wwrFloatComplex *, int);
extern template wwrblasStatus_t her2<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                               int64_t, const wwrFloatComplex *,
                                                               const wwrFloatComplex *, int64_t,
                                                               const wwrFloatComplex *, int64_t,
                                                               wwrFloatComplex *, int64_t);
extern template wwrblasStatus_t her2<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasFillMode_t, int,
                                                            const wwrDoubleComplex *,
                                                            const wwrDoubleComplex *, int,
                                                            const wwrDoubleComplex *, int,
                                                            wwrDoubleComplex *, int);
extern template wwrblasStatus_t her2<wwrDoubleComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                                int64_t, const wwrDoubleComplex *,
                                                                const wwrDoubleComplex *, int64_t,
                                                                const wwrDoubleComplex *, int64_t,
                                                                wwrDoubleComplex *, int64_t);

// Function: hpr
extern template wwrblasStatus_t
hpr<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasFillMode_t, int,
                          const ComplexToRealType<wwrFloatComplex> *, const wwrFloatComplex *, int,
                          wwrFloatComplex *);
extern template wwrblasStatus_t
hpr<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, int64_t,
                              const ComplexToRealType<wwrFloatComplex> *, const wwrFloatComplex *,
                              int64_t, wwrFloatComplex *);
extern template wwrblasStatus_t
hpr<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasFillMode_t, int,
                           const ComplexToRealType<wwrDoubleComplex> *, const wwrDoubleComplex *,
                           int, wwrDoubleComplex *);
extern template wwrblasStatus_t
hpr<wwrDoubleComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, int64_t,
                               const ComplexToRealType<wwrDoubleComplex> *,
                               const wwrDoubleComplex *, int64_t, wwrDoubleComplex *);

// Function: hpr2
extern template wwrblasStatus_t hpr2<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasFillMode_t, int,
                                                           const wwrFloatComplex *,
                                                           const wwrFloatComplex *, int,
                                                           const wwrFloatComplex *, int,
                                                           wwrFloatComplex *);
extern template wwrblasStatus_t hpr2<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                               int64_t, const wwrFloatComplex *,
                                                               const wwrFloatComplex *, int64_t,
                                                               const wwrFloatComplex *, int64_t,
                                                               wwrFloatComplex *);
extern template wwrblasStatus_t hpr2<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasFillMode_t, int,
                                                            const wwrDoubleComplex *,
                                                            const wwrDoubleComplex *, int,
                                                            const wwrDoubleComplex *, int,
                                                            wwrDoubleComplex *);
extern template wwrblasStatus_t hpr2<wwrDoubleComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                                int64_t, const wwrDoubleComplex *,
                                                                const wwrDoubleComplex *, int64_t,
                                                                const wwrDoubleComplex *, int64_t,
                                                                wwrDoubleComplex *);

// Function: gemvBatched
extern template wwrblasStatus_t gemvBatched<float, int>(wwrblasHandle_t, wwrblasOperation_t, int,
                                                        int, const float *, const float *const[],
                                                        int, const float *const[], int,
                                                        const float *, float *const[], int, int);
extern template wwrblasStatus_t
gemvBatched<float, int64_t>(wwrblasHandle_t, wwrblasOperation_t, int64_t, int64_t, const float *,
                            const float *const[], int64_t, const float *const[], int64_t,
                            const float *, float *const[], int64_t, int64_t);
extern template wwrblasStatus_t gemvBatched<double, int>(wwrblasHandle_t, wwrblasOperation_t, int,
                                                         int, const double *, const double *const[],
                                                         int, const double *const[], int,
                                                         const double *, double *const[], int, int);
extern template wwrblasStatus_t
gemvBatched<double, int64_t>(wwrblasHandle_t, wwrblasOperation_t, int64_t, int64_t, const double *,
                             const double *const[], int64_t, const double *const[], int64_t,
                             const double *, double *const[], int64_t, int64_t);
extern template wwrblasStatus_t
gemvBatched<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasOperation_t, int, int,
                                  const wwrFloatComplex *, const wwrFloatComplex *const[], int,
                                  const wwrFloatComplex *const[], int, const wwrFloatComplex *,
                                  wwrFloatComplex *const[], int, int);
extern template wwrblasStatus_t gemvBatched<wwrFloatComplex, int64_t>(
    wwrblasHandle_t, wwrblasOperation_t, int64_t, int64_t, const wwrFloatComplex *,
    const wwrFloatComplex *const[], int64_t, const wwrFloatComplex *const[], int64_t,
    const wwrFloatComplex *, wwrFloatComplex *const[], int64_t, int64_t);
extern template wwrblasStatus_t
gemvBatched<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasOperation_t, int, int,
                                   const wwrDoubleComplex *, const wwrDoubleComplex *const[], int,
                                   const wwrDoubleComplex *const[], int, const wwrDoubleComplex *,
                                   wwrDoubleComplex *const[], int, int);
extern template wwrblasStatus_t gemvBatched<wwrDoubleComplex, int64_t>(
    wwrblasHandle_t, wwrblasOperation_t, int64_t, int64_t, const wwrDoubleComplex *,
    const wwrDoubleComplex *const[], int64_t, const wwrDoubleComplex *const[], int64_t,
    const wwrDoubleComplex *, wwrDoubleComplex *const[], int64_t, int64_t);

// Function: gemvStridedBatched
extern template wwrblasStatus_t
gemvStridedBatched<float, int>(wwrblasHandle_t, wwrblasOperation_t, int, int, const float *,
                               const float *, int, long long int, const float *, int, long long int,
                               const float *, float *, int, long long int, int);
extern template wwrblasStatus_t
gemvStridedBatched<float, int64_t>(wwrblasHandle_t, wwrblasOperation_t, int64_t, int64_t,
                                   const float *, const float *, int64_t, long long int,
                                   const float *, int64_t, long long int, const float *, float *,
                                   int64_t, long long int, int64_t);
extern template wwrblasStatus_t
gemvStridedBatched<double, int>(wwrblasHandle_t, wwrblasOperation_t, int, int, const double *,
                                const double *, int, long long int, const double *, int,
                                long long int, const double *, double *, int, long long int, int);
extern template wwrblasStatus_t
gemvStridedBatched<double, int64_t>(wwrblasHandle_t, wwrblasOperation_t, int64_t, int64_t,
                                    const double *, const double *, int64_t, long long int,
                                    const double *, int64_t, long long int, const double *,
                                    double *, int64_t, long long int, int64_t);
extern template wwrblasStatus_t gemvStridedBatched<wwrFloatComplex, int>(
    wwrblasHandle_t, wwrblasOperation_t, int, int, const wwrFloatComplex *, const wwrFloatComplex *,
    int, long long int, const wwrFloatComplex *, int, long long int, const wwrFloatComplex *,
    wwrFloatComplex *, int, long long int, int);
extern template wwrblasStatus_t gemvStridedBatched<wwrFloatComplex, int64_t>(
    wwrblasHandle_t, wwrblasOperation_t, int64_t, int64_t, const wwrFloatComplex *,
    const wwrFloatComplex *, int64_t, long long int, const wwrFloatComplex *, int64_t,
    long long int, const wwrFloatComplex *, wwrFloatComplex *, int64_t, long long int, int64_t);
extern template wwrblasStatus_t gemvStridedBatched<wwrDoubleComplex, int>(
    wwrblasHandle_t, wwrblasOperation_t, int, int, const wwrDoubleComplex *,
    const wwrDoubleComplex *, int, long long int, const wwrDoubleComplex *, int, long long int,
    const wwrDoubleComplex *, wwrDoubleComplex *, int, long long int, int);
extern template wwrblasStatus_t gemvStridedBatched<wwrDoubleComplex, int64_t>(
    wwrblasHandle_t, wwrblasOperation_t, int64_t, int64_t, const wwrDoubleComplex *,
    const wwrDoubleComplex *, int64_t, long long int, const wwrDoubleComplex *, int64_t,
    long long int, const wwrDoubleComplex *, wwrDoubleComplex *, int64_t, long long int, int64_t);
} // namespace wwr
