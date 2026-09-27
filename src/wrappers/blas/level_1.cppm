/**
 * @file level_1.cppm
 * @brief GPU BLAS Level 1 (vector-vector) operations
 *
 * This module provides type-safe wrappers for GPU BLAS Level 1 BLAS operations.
 *
 * Usage:
 *   import gpumod.wrappers.blas;
 */

module;

#include "dispatch_macros.h"

export module gpumod.wrappers.blas:level_1;

import gpumod.blas;
import gpumod.complex;
import :type_traits;
import std;

export namespace wwr {

// ========================================================================
// Index of maximum/minimum absolute value
// ========================================================================

template<usual_fp T, int_type IntT>
gpublasStatus_t iamax(gpublasHandle_t handle, IntT n, const T *x, IntT incx, IntT *result) {
  WWR_REAL_DISPATCH_64(T, IntT, gpublasI, s, d, amax, handle, n, x, incx, result);
  WWR_COMPLEX_DISPATCH_64(T, IntT, gpublasI, c, z, amax, handle, n, x, incx, result);
}

template<usual_fp T, int_type IntT>
gpublasStatus_t iamin(gpublasHandle_t handle, IntT n, const T *x, IntT incx, IntT *result) {
  WWR_REAL_DISPATCH_64(T, IntT, gpublasI, s, d, amin, handle, n, x, incx, result);
  WWR_COMPLEX_DISPATCH_64(T, IntT, gpublasI, c, z, amin, handle, n, x, incx, result);
}

// ========================================================================
// Sum of absolute values (returns real type even for complex)
// ========================================================================

template<usual_fp T, int_type IntT>
gpublasStatus_t asum(gpublasHandle_t handle, IntT n, const T *x, IntT incx,
                     ComplexToRealType<T> *result) {
  WWR_REAL_DISPATCH_64(T, IntT, gpublas, S, D, asum, handle, n, x, incx, result);
  WWR_COMPLEX_DISPATCH_64(T, IntT, gpublas, Sc, Dz, asum, handle, n, x, incx, result);
}

// ========================================================================
// Scalar-vector product and sum: y = alpha*x + y
// ========================================================================

template<usual_fp T, int_type IntT>
gpublasStatus_t axpy(gpublasHandle_t handle, IntT n, const T *alpha, const T *x, IntT incx, T *y,
                     IntT incy) {
  WWR_USUAL_DISPATCH_64(T, IntT, axpy, handle, n, alpha, x, incx, y, incy);
}

// ========================================================================
// Vector copy
// ========================================================================

template<usual_fp T, int_type IntT>
gpublasStatus_t copy(gpublasHandle_t handle, IntT n, const T *x, IntT incx, T *y, IntT incy) {
  WWR_USUAL_DISPATCH_64(T, IntT, copy, handle, n, x, incx, y, incy);
}

// ========================================================================
// Dot product
// ========================================================================

template<real_fp T, int_type IntT>
gpublasStatus_t dot(gpublasHandle_t handle, IntT n, const T *x, IntT incx, const T *y, IntT incy,
                    T *result) {
  WWR_REAL_DISPATCH_64(T, IntT, gpublas, S, D, dot, handle, n, x, incx, y, incy, result);
}

template<complex_fp T, int_type IntT>
gpublasStatus_t dotc(gpublasHandle_t handle, IntT n, const T *x, IntT incx, const T *y, IntT incy,
                     T *result) {
  WWR_COMPLEX_DISPATCH_64(T, IntT, gpublas, C, Z, dotc, handle, n, x, incx, y, incy, result);
}

template<complex_fp T, int_type IntT>
gpublasStatus_t dotu(gpublasHandle_t handle, IntT n, const T *x, IntT incx, const T *y, IntT incy,
                     T *result) {
  WWR_COMPLEX_DISPATCH_64(T, IntT, gpublas, C, Z, dotu, handle, n, x, incx, y, incy, result);
}

// ========================================================================
// Euclidean norm (returns real type even for complex)
// ========================================================================

template<usual_fp T, int_type IntT>
gpublasStatus_t nrm2(gpublasHandle_t handle, IntT n, const T *x, IntT incx,
                     ComplexToRealType<T> *result) {
  WWR_REAL_DISPATCH_64(T, IntT, gpublas, S, D, nrm2, handle, n, x, incx, result);
  WWR_COMPLEX_DISPATCH_64(T, IntT, gpublas, Sc, Dz, nrm2, handle, n, x, incx, result);
}

// ========================================================================
// Givens rotation: applies rotation matrix
// ========================================================================

template<real_fp T, int_type IntT>
gpublasStatus_t rot(gpublasHandle_t handle, IntT n, T *x, IntT incx, T *y, IntT incy, const T *c,
                    const T *s) {
  WWR_REAL_DISPATCH_64(T, IntT, gpublas, S, D, rot, handle, n, x, incx, y, incy, c, s);
}

template<complex_fp T, int_type IntT>
gpublasStatus_t rot(gpublasHandle_t handle, IntT n, T *x, IntT incx, T *y, IntT incy,
                    const ComplexToRealType<T> *c, const T *s) {
  WWR_COMPLEX_DISPATCH_64(T, IntT, gpublas, C, Z, rot, handle, n, x, incx, y, incy, c, s);
}

template<complex_fp T, int_type IntT>
gpublasStatus_t rot(gpublasHandle_t handle, IntT n, T *x, IntT incx, T *y, IntT incy,
                    const ComplexToRealType<T> *c, const ComplexToRealType<T> *s) {
  WWR_COMPLEX_DISPATCH_64(T, IntT, gpublas, Cs, Zd, rot, handle, n, x, incx, y, incy, c, s);
}

// ========================================================================
// Generate Givens rotation
// NOTE: rotg operates on scalars only (no integer size parameters).
// No _v2_64 variant exists - IntT template parameter is for API consistency.
// ========================================================================

template<real_fp T, int_type IntT = int>
gpublasStatus_t rotg(gpublasHandle_t handle, T *a, T *b, T *c, T *s) {
  WWR_REAL_DISPATCH(T, gpublas, S, D, rotg, handle, a, b, c, s);
}

template<complex_fp T, int_type IntT = int>
gpublasStatus_t rotg(gpublasHandle_t handle, T *a, T *b, ComplexToRealType<T> *c, T *s) {
  WWR_COMPLEX_DISPATCH(T, gpublas, C, Z, rotg, handle, a, b, c, s);
}

// ========================================================================
// Modified Givens rotation (real types only)
// ========================================================================

template<real_fp T, int_type IntT>
gpublasStatus_t rotm(gpublasHandle_t handle, IntT n, T *x, IntT incx, T *y, IntT incy,
                     const T *param) {
  WWR_REAL_DISPATCH_64(T, IntT, gpublas, S, D, rotm, handle, n, x, incx, y, incy, param);
}

// NOTE: rotmg operates on scalars only (no integer size parameters).
// No _v2_64 variant exists - IntT template parameter is for API consistency.
template<real_fp T, int_type IntT = int>
gpublasStatus_t rotmg(gpublasHandle_t handle, T *d1, T *d2, T *x1, const T *y1, T *param) {
  WWR_REAL_DISPATCH(T, gpublas, S, D, rotmg, handle, d1, d2, x1, y1, param);
}

// ========================================================================
// Scalar multiplication: x = alpha*x
// ========================================================================

template<usual_fp T, int_type IntT>
gpublasStatus_t scal(gpublasHandle_t handle, IntT n, const T *alpha, T *x, IntT incx) {
  WWR_USUAL_DISPATCH_64(T, IntT, scal, handle, n, alpha, x, incx);
}

template<complex_fp T, int_type IntT>
gpublasStatus_t scal(gpublasHandle_t handle, IntT n, const ComplexToRealType<T> *alpha, T *x,
                     IntT incx) {
  WWR_COMPLEX_DISPATCH_64(T, IntT, gpublas, Cs, Zd, scal, handle, n, alpha, x, incx);
}

// ========================================================================
// Vector swap
// ========================================================================

template<usual_fp T, int_type IntT>
gpublasStatus_t swap(gpublasHandle_t handle, IntT n, T *x, IntT incx, T *y, IntT incy) {
  WWR_USUAL_DISPATCH_64(T, IntT, swap, handle, n, x, incx, y, incy);
}

// ==================== Explicit Template Instantiations ====================
// Matching `template` instantiations live in instantiations.cpp.

// Function: iamax
extern template gpublasStatus_t iamax<float, int>(gpublasHandle_t, int, const float *, int, int *);
extern template gpublasStatus_t iamax<float, int64_t>(gpublasHandle_t, int64_t, const float *,
                                                      int64_t, int64_t *);
extern template gpublasStatus_t iamax<double, int>(gpublasHandle_t, int, const double *, int,
                                                   int *);
extern template gpublasStatus_t iamax<double, int64_t>(gpublasHandle_t, int64_t, const double *,
                                                       int64_t, int64_t *);
extern template gpublasStatus_t iamax<gpuFloatComplex, int>(gpublasHandle_t, int,
                                                            const gpuFloatComplex *, int, int *);
extern template gpublasStatus_t iamax<gpuFloatComplex, int64_t>(gpublasHandle_t, int64_t,
                                                                const gpuFloatComplex *, int64_t,
                                                                int64_t *);
extern template gpublasStatus_t iamax<gpuDoubleComplex, int>(gpublasHandle_t, int,
                                                             const gpuDoubleComplex *, int, int *);
extern template gpublasStatus_t iamax<gpuDoubleComplex, int64_t>(gpublasHandle_t, int64_t,
                                                                 const gpuDoubleComplex *, int64_t,
                                                                 int64_t *);

// Function: iamin
extern template gpublasStatus_t iamin<float, int>(gpublasHandle_t, int, const float *, int, int *);
extern template gpublasStatus_t iamin<float, int64_t>(gpublasHandle_t, int64_t, const float *,
                                                      int64_t, int64_t *);
extern template gpublasStatus_t iamin<double, int>(gpublasHandle_t, int, const double *, int,
                                                   int *);
extern template gpublasStatus_t iamin<double, int64_t>(gpublasHandle_t, int64_t, const double *,
                                                       int64_t, int64_t *);
extern template gpublasStatus_t iamin<gpuFloatComplex, int>(gpublasHandle_t, int,
                                                            const gpuFloatComplex *, int, int *);
extern template gpublasStatus_t iamin<gpuFloatComplex, int64_t>(gpublasHandle_t, int64_t,
                                                                const gpuFloatComplex *, int64_t,
                                                                int64_t *);
extern template gpublasStatus_t iamin<gpuDoubleComplex, int>(gpublasHandle_t, int,
                                                             const gpuDoubleComplex *, int, int *);
extern template gpublasStatus_t iamin<gpuDoubleComplex, int64_t>(gpublasHandle_t, int64_t,
                                                                 const gpuDoubleComplex *, int64_t,
                                                                 int64_t *);

// Function: asum
extern template gpublasStatus_t asum<float, int>(gpublasHandle_t, int, const float *, int,
                                                 ComplexToRealType<float> *);
extern template gpublasStatus_t asum<float, int64_t>(gpublasHandle_t, int64_t, const float *,
                                                     int64_t, ComplexToRealType<float> *);
extern template gpublasStatus_t asum<double, int>(gpublasHandle_t, int, const double *, int,
                                                  ComplexToRealType<double> *);
extern template gpublasStatus_t asum<double, int64_t>(gpublasHandle_t, int64_t, const double *,
                                                      int64_t, ComplexToRealType<double> *);
extern template gpublasStatus_t asum<gpuFloatComplex, int>(gpublasHandle_t, int,
                                                           const gpuFloatComplex *, int,
                                                           ComplexToRealType<gpuFloatComplex> *);
extern template gpublasStatus_t
asum<gpuFloatComplex, int64_t>(gpublasHandle_t, int64_t, const gpuFloatComplex *, int64_t,
                               ComplexToRealType<gpuFloatComplex> *);
extern template gpublasStatus_t asum<gpuDoubleComplex, int>(gpublasHandle_t, int,
                                                            const gpuDoubleComplex *, int,
                                                            ComplexToRealType<gpuDoubleComplex> *);
extern template gpublasStatus_t
asum<gpuDoubleComplex, int64_t>(gpublasHandle_t, int64_t, const gpuDoubleComplex *, int64_t,
                                ComplexToRealType<gpuDoubleComplex> *);

// Function: axpy
extern template gpublasStatus_t axpy<float, int>(gpublasHandle_t, int, const float *, const float *,
                                                 int, float *, int);
extern template gpublasStatus_t axpy<float, int64_t>(gpublasHandle_t, int64_t, const float *,
                                                     const float *, int64_t, float *, int64_t);
extern template gpublasStatus_t axpy<double, int>(gpublasHandle_t, int, const double *,
                                                  const double *, int, double *, int);
extern template gpublasStatus_t axpy<double, int64_t>(gpublasHandle_t, int64_t, const double *,
                                                      const double *, int64_t, double *, int64_t);
extern template gpublasStatus_t axpy<gpuFloatComplex, int>(gpublasHandle_t, int,
                                                           const gpuFloatComplex *,
                                                           const gpuFloatComplex *, int,
                                                           gpuFloatComplex *, int);
extern template gpublasStatus_t axpy<gpuFloatComplex, int64_t>(gpublasHandle_t, int64_t,
                                                               const gpuFloatComplex *,
                                                               const gpuFloatComplex *, int64_t,
                                                               gpuFloatComplex *, int64_t);
extern template gpublasStatus_t axpy<gpuDoubleComplex, int>(gpublasHandle_t, int,
                                                            const gpuDoubleComplex *,
                                                            const gpuDoubleComplex *, int,
                                                            gpuDoubleComplex *, int);
extern template gpublasStatus_t axpy<gpuDoubleComplex, int64_t>(gpublasHandle_t, int64_t,
                                                                const gpuDoubleComplex *,
                                                                const gpuDoubleComplex *, int64_t,
                                                                gpuDoubleComplex *, int64_t);

// Function: copy
extern template gpublasStatus_t copy<float, int>(gpublasHandle_t, int, const float *, int, float *,
                                                 int);
extern template gpublasStatus_t copy<float, int64_t>(gpublasHandle_t, int64_t, const float *,
                                                     int64_t, float *, int64_t);
extern template gpublasStatus_t copy<double, int>(gpublasHandle_t, int, const double *, int,
                                                  double *, int);
extern template gpublasStatus_t copy<double, int64_t>(gpublasHandle_t, int64_t, const double *,
                                                      int64_t, double *, int64_t);
extern template gpublasStatus_t copy<gpuFloatComplex, int>(gpublasHandle_t, int,
                                                           const gpuFloatComplex *, int,
                                                           gpuFloatComplex *, int);
extern template gpublasStatus_t copy<gpuFloatComplex, int64_t>(gpublasHandle_t, int64_t,
                                                               const gpuFloatComplex *, int64_t,
                                                               gpuFloatComplex *, int64_t);
extern template gpublasStatus_t copy<gpuDoubleComplex, int>(gpublasHandle_t, int,
                                                            const gpuDoubleComplex *, int,
                                                            gpuDoubleComplex *, int);
extern template gpublasStatus_t copy<gpuDoubleComplex, int64_t>(gpublasHandle_t, int64_t,
                                                                const gpuDoubleComplex *, int64_t,
                                                                gpuDoubleComplex *, int64_t);

// Function: dot
extern template gpublasStatus_t dot<float, int>(gpublasHandle_t, int, const float *, int,
                                                const float *, int, float *);
extern template gpublasStatus_t dot<float, int64_t>(gpublasHandle_t, int64_t, const float *,
                                                    int64_t, const float *, int64_t, float *);
extern template gpublasStatus_t dot<double, int>(gpublasHandle_t, int, const double *, int,
                                                 const double *, int, double *);
extern template gpublasStatus_t dot<double, int64_t>(gpublasHandle_t, int64_t, const double *,
                                                     int64_t, const double *, int64_t, double *);

// Function: dotc
extern template gpublasStatus_t dotc<gpuFloatComplex, int>(gpublasHandle_t, int,
                                                           const gpuFloatComplex *, int,
                                                           const gpuFloatComplex *, int,
                                                           gpuFloatComplex *);
extern template gpublasStatus_t dotc<gpuFloatComplex, int64_t>(gpublasHandle_t, int64_t,
                                                               const gpuFloatComplex *, int64_t,
                                                               const gpuFloatComplex *, int64_t,
                                                               gpuFloatComplex *);
extern template gpublasStatus_t dotc<gpuDoubleComplex, int>(gpublasHandle_t, int,
                                                            const gpuDoubleComplex *, int,
                                                            const gpuDoubleComplex *, int,
                                                            gpuDoubleComplex *);
extern template gpublasStatus_t dotc<gpuDoubleComplex, int64_t>(gpublasHandle_t, int64_t,
                                                                const gpuDoubleComplex *, int64_t,
                                                                const gpuDoubleComplex *, int64_t,
                                                                gpuDoubleComplex *);

// Function: dotu
extern template gpublasStatus_t dotu<gpuFloatComplex, int>(gpublasHandle_t, int,
                                                           const gpuFloatComplex *, int,
                                                           const gpuFloatComplex *, int,
                                                           gpuFloatComplex *);
extern template gpublasStatus_t dotu<gpuFloatComplex, int64_t>(gpublasHandle_t, int64_t,
                                                               const gpuFloatComplex *, int64_t,
                                                               const gpuFloatComplex *, int64_t,
                                                               gpuFloatComplex *);
extern template gpublasStatus_t dotu<gpuDoubleComplex, int>(gpublasHandle_t, int,
                                                            const gpuDoubleComplex *, int,
                                                            const gpuDoubleComplex *, int,
                                                            gpuDoubleComplex *);
extern template gpublasStatus_t dotu<gpuDoubleComplex, int64_t>(gpublasHandle_t, int64_t,
                                                                const gpuDoubleComplex *, int64_t,
                                                                const gpuDoubleComplex *, int64_t,
                                                                gpuDoubleComplex *);

// Function: nrm2
extern template gpublasStatus_t nrm2<float, int>(gpublasHandle_t, int, const float *, int,
                                                 ComplexToRealType<float> *);
extern template gpublasStatus_t nrm2<float, int64_t>(gpublasHandle_t, int64_t, const float *,
                                                     int64_t, ComplexToRealType<float> *);
extern template gpublasStatus_t nrm2<double, int>(gpublasHandle_t, int, const double *, int,
                                                  ComplexToRealType<double> *);
extern template gpublasStatus_t nrm2<double, int64_t>(gpublasHandle_t, int64_t, const double *,
                                                      int64_t, ComplexToRealType<double> *);
extern template gpublasStatus_t nrm2<gpuFloatComplex, int>(gpublasHandle_t, int,
                                                           const gpuFloatComplex *, int,
                                                           ComplexToRealType<gpuFloatComplex> *);
extern template gpublasStatus_t
nrm2<gpuFloatComplex, int64_t>(gpublasHandle_t, int64_t, const gpuFloatComplex *, int64_t,
                               ComplexToRealType<gpuFloatComplex> *);
extern template gpublasStatus_t nrm2<gpuDoubleComplex, int>(gpublasHandle_t, int,
                                                            const gpuDoubleComplex *, int,
                                                            ComplexToRealType<gpuDoubleComplex> *);
extern template gpublasStatus_t
nrm2<gpuDoubleComplex, int64_t>(gpublasHandle_t, int64_t, const gpuDoubleComplex *, int64_t,
                                ComplexToRealType<gpuDoubleComplex> *);

// Function: rot
extern template gpublasStatus_t rot<float, int>(gpublasHandle_t, int, float *, int, float *, int,
                                                const float *, const float *);
extern template gpublasStatus_t rot<float, int64_t>(gpublasHandle_t, int64_t, float *, int64_t,
                                                    float *, int64_t, const float *, const float *);
extern template gpublasStatus_t rot<double, int>(gpublasHandle_t, int, double *, int, double *, int,
                                                 const double *, const double *);
extern template gpublasStatus_t rot<double, int64_t>(gpublasHandle_t, int64_t, double *, int64_t,
                                                     double *, int64_t, const double *,
                                                     const double *);
extern template gpublasStatus_t
rot<gpuFloatComplex, int>(gpublasHandle_t, int, gpuFloatComplex *, int, gpuFloatComplex *, int,
                          const ComplexToRealType<gpuFloatComplex> *, const gpuFloatComplex *);
extern template gpublasStatus_t
rot<gpuFloatComplex, int64_t>(gpublasHandle_t, int64_t, gpuFloatComplex *, int64_t,
                              gpuFloatComplex *, int64_t,
                              const ComplexToRealType<gpuFloatComplex> *, const gpuFloatComplex *);
extern template gpublasStatus_t
rot<gpuDoubleComplex, int>(gpublasHandle_t, int, gpuDoubleComplex *, int, gpuDoubleComplex *, int,
                           const ComplexToRealType<gpuDoubleComplex> *, const gpuDoubleComplex *);
extern template gpublasStatus_t rot<gpuDoubleComplex, int64_t>(
    gpublasHandle_t, int64_t, gpuDoubleComplex *, int64_t, gpuDoubleComplex *, int64_t,
    const ComplexToRealType<gpuDoubleComplex> *, const gpuDoubleComplex *);
extern template gpublasStatus_t
rot<gpuFloatComplex, int>(gpublasHandle_t, int, gpuFloatComplex *, int, gpuFloatComplex *, int,
                          const ComplexToRealType<gpuFloatComplex> *,
                          const ComplexToRealType<gpuFloatComplex> *);
extern template gpublasStatus_t rot<gpuFloatComplex, int64_t>(
    gpublasHandle_t, int64_t, gpuFloatComplex *, int64_t, gpuFloatComplex *, int64_t,
    const ComplexToRealType<gpuFloatComplex> *, const ComplexToRealType<gpuFloatComplex> *);
extern template gpublasStatus_t
rot<gpuDoubleComplex, int>(gpublasHandle_t, int, gpuDoubleComplex *, int, gpuDoubleComplex *, int,
                           const ComplexToRealType<gpuDoubleComplex> *,
                           const ComplexToRealType<gpuDoubleComplex> *);
extern template gpublasStatus_t rot<gpuDoubleComplex, int64_t>(
    gpublasHandle_t, int64_t, gpuDoubleComplex *, int64_t, gpuDoubleComplex *, int64_t,
    const ComplexToRealType<gpuDoubleComplex> *, const ComplexToRealType<gpuDoubleComplex> *);

// Function: rotg
extern template gpublasStatus_t rotg<float, int>(gpublasHandle_t, float *, float *, float *,
                                                 float *);
extern template gpublasStatus_t rotg<double, int>(gpublasHandle_t, double *, double *, double *,
                                                  double *);
extern template gpublasStatus_t rotg<gpuFloatComplex, int>(gpublasHandle_t, gpuFloatComplex *,
                                                           gpuFloatComplex *,
                                                           ComplexToRealType<gpuFloatComplex> *,
                                                           gpuFloatComplex *);
extern template gpublasStatus_t rotg<gpuDoubleComplex, int>(gpublasHandle_t, gpuDoubleComplex *,
                                                            gpuDoubleComplex *,
                                                            ComplexToRealType<gpuDoubleComplex> *,
                                                            gpuDoubleComplex *);

// Function: rotm
extern template gpublasStatus_t rotm<float, int>(gpublasHandle_t, int, float *, int, float *, int,
                                                 const float *);
extern template gpublasStatus_t rotm<float, int64_t>(gpublasHandle_t, int64_t, float *, int64_t,
                                                     float *, int64_t, const float *);
extern template gpublasStatus_t rotm<double, int>(gpublasHandle_t, int, double *, int, double *,
                                                  int, const double *);
extern template gpublasStatus_t rotm<double, int64_t>(gpublasHandle_t, int64_t, double *, int64_t,
                                                      double *, int64_t, const double *);

// Function: rotmg
extern template gpublasStatus_t rotmg<float, int>(gpublasHandle_t, float *, float *, float *,
                                                  const float *, float *);
extern template gpublasStatus_t rotmg<double, int>(gpublasHandle_t, double *, double *, double *,
                                                   const double *, double *);

// Function: scal
extern template gpublasStatus_t scal<float, int>(gpublasHandle_t, int, const float *, float *, int);
extern template gpublasStatus_t scal<float, int64_t>(gpublasHandle_t, int64_t, const float *,
                                                     float *, int64_t);
extern template gpublasStatus_t scal<double, int>(gpublasHandle_t, int, const double *, double *,
                                                  int);
extern template gpublasStatus_t scal<double, int64_t>(gpublasHandle_t, int64_t, const double *,
                                                      double *, int64_t);
extern template gpublasStatus_t
scal<gpuFloatComplex, int>(gpublasHandle_t, int, const gpuFloatComplex *, gpuFloatComplex *, int);
extern template gpublasStatus_t scal<gpuFloatComplex, int64_t>(gpublasHandle_t, int64_t,
                                                               const gpuFloatComplex *,
                                                               gpuFloatComplex *, int64_t);
extern template gpublasStatus_t scal<gpuDoubleComplex, int>(gpublasHandle_t, int,
                                                            const gpuDoubleComplex *,
                                                            gpuDoubleComplex *, int);
extern template gpublasStatus_t scal<gpuDoubleComplex, int64_t>(gpublasHandle_t, int64_t,
                                                                const gpuDoubleComplex *,
                                                                gpuDoubleComplex *, int64_t);
extern template gpublasStatus_t
scal<gpuFloatComplex, int>(gpublasHandle_t, int, const ComplexToRealType<gpuFloatComplex> *,
                           gpuFloatComplex *, int);
extern template gpublasStatus_t
scal<gpuFloatComplex, int64_t>(gpublasHandle_t, int64_t, const ComplexToRealType<gpuFloatComplex> *,
                               gpuFloatComplex *, int64_t);
extern template gpublasStatus_t
scal<gpuDoubleComplex, int>(gpublasHandle_t, int, const ComplexToRealType<gpuDoubleComplex> *,
                            gpuDoubleComplex *, int);
extern template gpublasStatus_t
scal<gpuDoubleComplex, int64_t>(gpublasHandle_t, int64_t,
                                const ComplexToRealType<gpuDoubleComplex> *, gpuDoubleComplex *,
                                int64_t);

// Function: swap
extern template gpublasStatus_t swap<float, int>(gpublasHandle_t, int, float *, int, float *, int);
extern template gpublasStatus_t swap<float, int64_t>(gpublasHandle_t, int64_t, float *, int64_t,
                                                     float *, int64_t);
extern template gpublasStatus_t swap<double, int>(gpublasHandle_t, int, double *, int, double *,
                                                  int);
extern template gpublasStatus_t swap<double, int64_t>(gpublasHandle_t, int64_t, double *, int64_t,
                                                      double *, int64_t);
extern template gpublasStatus_t swap<gpuFloatComplex, int>(gpublasHandle_t, int, gpuFloatComplex *,
                                                           int, gpuFloatComplex *, int);
extern template gpublasStatus_t swap<gpuFloatComplex, int64_t>(gpublasHandle_t, int64_t,
                                                               gpuFloatComplex *, int64_t,
                                                               gpuFloatComplex *, int64_t);
extern template gpublasStatus_t
swap<gpuDoubleComplex, int>(gpublasHandle_t, int, gpuDoubleComplex *, int, gpuDoubleComplex *, int);
extern template gpublasStatus_t swap<gpuDoubleComplex, int64_t>(gpublasHandle_t, int64_t,
                                                                gpuDoubleComplex *, int64_t,
                                                                gpuDoubleComplex *, int64_t);

} // namespace wwr
