/**
 * @file level_1.cppm
 * @brief GPU BLAS Level 1 (vector-vector) operations
 *
 * This module provides type-safe wrappers for GPU BLAS Level 1 BLAS operations.
 *
 * Usage:
 *   import wwr.wrappers.blas;
 */

module;

#include "dispatch_macros.h"

export module wwr.wrappers.blas:level_1;

import wwr.blas;
import wwr.complex;
import :type_traits;
import std;

export namespace wwr {

// ========================================================================
// Index of maximum/minimum absolute value
// ========================================================================

template<usual_fp T, int_type IntT>
wwrblasStatus_t iamax(wwrblasHandle_t handle, IntT n, const T *x, IntT incx, IntT *result) {
  WWR_REAL_DISPATCH_64(T, IntT, wwrblasI, s, d, amax, handle, n, x, incx, result);
  WWR_COMPLEX_DISPATCH_64(T, IntT, wwrblasI, c, z, amax, handle, n, x, incx, result);
}

template<usual_fp T, int_type IntT>
wwrblasStatus_t iamin(wwrblasHandle_t handle, IntT n, const T *x, IntT incx, IntT *result) {
  WWR_REAL_DISPATCH_64(T, IntT, wwrblasI, s, d, amin, handle, n, x, incx, result);
  WWR_COMPLEX_DISPATCH_64(T, IntT, wwrblasI, c, z, amin, handle, n, x, incx, result);
}

// ========================================================================
// Sum of absolute values (returns real type even for complex)
// ========================================================================

template<usual_fp T, int_type IntT>
wwrblasStatus_t asum(wwrblasHandle_t handle, IntT n, const T *x, IntT incx,
                     ComplexToRealType<T> *result) {
  WWR_REAL_DISPATCH_64(T, IntT, wwrblas, S, D, asum, handle, n, x, incx, result);
  WWR_COMPLEX_DISPATCH_64(T, IntT, wwrblas, Sc, Dz, asum, handle, n, x, incx, result);
}

// ========================================================================
// Scalar-vector product and sum: y = alpha*x + y
// ========================================================================

template<usual_fp T, int_type IntT>
wwrblasStatus_t axpy(wwrblasHandle_t handle, IntT n, const T *alpha, const T *x, IntT incx, T *y,
                     IntT incy) {
  WWR_USUAL_DISPATCH_64(T, IntT, axpy, handle, n, alpha, x, incx, y, incy);
}

// ========================================================================
// Vector copy
// ========================================================================

template<usual_fp T, int_type IntT>
wwrblasStatus_t copy(wwrblasHandle_t handle, IntT n, const T *x, IntT incx, T *y, IntT incy) {
  WWR_USUAL_DISPATCH_64(T, IntT, copy, handle, n, x, incx, y, incy);
}

// ========================================================================
// Dot product
// ========================================================================

template<real_fp T, int_type IntT>
wwrblasStatus_t dot(wwrblasHandle_t handle, IntT n, const T *x, IntT incx, const T *y, IntT incy,
                    T *result) {
  WWR_REAL_DISPATCH_64(T, IntT, wwrblas, S, D, dot, handle, n, x, incx, y, incy, result);
}

template<complex_fp T, int_type IntT>
wwrblasStatus_t dotc(wwrblasHandle_t handle, IntT n, const T *x, IntT incx, const T *y, IntT incy,
                     T *result) {
  WWR_COMPLEX_DISPATCH_64(T, IntT, wwrblas, C, Z, dotc, handle, n, x, incx, y, incy, result);
}

template<complex_fp T, int_type IntT>
wwrblasStatus_t dotu(wwrblasHandle_t handle, IntT n, const T *x, IntT incx, const T *y, IntT incy,
                     T *result) {
  WWR_COMPLEX_DISPATCH_64(T, IntT, wwrblas, C, Z, dotu, handle, n, x, incx, y, incy, result);
}

// ========================================================================
// Euclidean norm (returns real type even for complex)
// ========================================================================

template<usual_fp T, int_type IntT>
wwrblasStatus_t nrm2(wwrblasHandle_t handle, IntT n, const T *x, IntT incx,
                     ComplexToRealType<T> *result) {
  WWR_REAL_DISPATCH_64(T, IntT, wwrblas, S, D, nrm2, handle, n, x, incx, result);
  WWR_COMPLEX_DISPATCH_64(T, IntT, wwrblas, Sc, Dz, nrm2, handle, n, x, incx, result);
}

// ========================================================================
// Givens rotation: applies rotation matrix
// ========================================================================

template<real_fp T, int_type IntT>
wwrblasStatus_t rot(wwrblasHandle_t handle, IntT n, T *x, IntT incx, T *y, IntT incy, const T *c,
                    const T *s) {
  WWR_REAL_DISPATCH_64(T, IntT, wwrblas, S, D, rot, handle, n, x, incx, y, incy, c, s);
}

template<complex_fp T, int_type IntT>
wwrblasStatus_t rot(wwrblasHandle_t handle, IntT n, T *x, IntT incx, T *y, IntT incy,
                    const ComplexToRealType<T> *c, const T *s) {
  WWR_COMPLEX_DISPATCH_64(T, IntT, wwrblas, C, Z, rot, handle, n, x, incx, y, incy, c, s);
}

template<complex_fp T, int_type IntT>
wwrblasStatus_t rot(wwrblasHandle_t handle, IntT n, T *x, IntT incx, T *y, IntT incy,
                    const ComplexToRealType<T> *c, const ComplexToRealType<T> *s) {
  WWR_COMPLEX_DISPATCH_64(T, IntT, wwrblas, Cs, Zd, rot, handle, n, x, incx, y, incy, c, s);
}

// ========================================================================
// Generate Givens rotation
// NOTE: rotg operates on scalars only (no integer size parameters).
// No _v2_64 variant exists - IntT template parameter is for API consistency.
// ========================================================================

template<real_fp T, int_type IntT = int>
wwrblasStatus_t rotg(wwrblasHandle_t handle, T *a, T *b, T *c, T *s) {
  WWR_REAL_DISPATCH(T, wwrblas, S, D, rotg, handle, a, b, c, s);
}

template<complex_fp T, int_type IntT = int>
wwrblasStatus_t rotg(wwrblasHandle_t handle, T *a, T *b, ComplexToRealType<T> *c, T *s) {
  WWR_COMPLEX_DISPATCH(T, wwrblas, C, Z, rotg, handle, a, b, c, s);
}

// ========================================================================
// Modified Givens rotation (real types only)
// ========================================================================

template<real_fp T, int_type IntT>
wwrblasStatus_t rotm(wwrblasHandle_t handle, IntT n, T *x, IntT incx, T *y, IntT incy,
                     const T *param) {
  WWR_REAL_DISPATCH_64(T, IntT, wwrblas, S, D, rotm, handle, n, x, incx, y, incy, param);
}

// NOTE: rotmg operates on scalars only (no integer size parameters).
// No _v2_64 variant exists - IntT template parameter is for API consistency.
template<real_fp T, int_type IntT = int>
wwrblasStatus_t rotmg(wwrblasHandle_t handle, T *d1, T *d2, T *x1, const T *y1, T *param) {
  WWR_REAL_DISPATCH(T, wwrblas, S, D, rotmg, handle, d1, d2, x1, y1, param);
}

// ========================================================================
// Scalar multiplication: x = alpha*x
// ========================================================================

template<usual_fp T, int_type IntT>
wwrblasStatus_t scal(wwrblasHandle_t handle, IntT n, const T *alpha, T *x, IntT incx) {
  WWR_USUAL_DISPATCH_64(T, IntT, scal, handle, n, alpha, x, incx);
}

template<complex_fp T, int_type IntT>
wwrblasStatus_t scal(wwrblasHandle_t handle, IntT n, const ComplexToRealType<T> *alpha, T *x,
                     IntT incx) {
  WWR_COMPLEX_DISPATCH_64(T, IntT, wwrblas, Cs, Zd, scal, handle, n, alpha, x, incx);
}

// ========================================================================
// Vector swap
// ========================================================================

template<usual_fp T, int_type IntT>
wwrblasStatus_t swap(wwrblasHandle_t handle, IntT n, T *x, IntT incx, T *y, IntT incy) {
  WWR_USUAL_DISPATCH_64(T, IntT, swap, handle, n, x, incx, y, incy);
}

// ==================== Explicit Template Instantiations ====================
// Matching `template` instantiations live in instantiations.cpp.

// Function: iamax
extern template wwrblasStatus_t iamax<float, int>(wwrblasHandle_t, int, const float *, int, int *);
extern template wwrblasStatus_t iamax<float, int64_t>(wwrblasHandle_t, int64_t, const float *,
                                                      int64_t, int64_t *);
extern template wwrblasStatus_t iamax<double, int>(wwrblasHandle_t, int, const double *, int,
                                                   int *);
extern template wwrblasStatus_t iamax<double, int64_t>(wwrblasHandle_t, int64_t, const double *,
                                                       int64_t, int64_t *);
extern template wwrblasStatus_t iamax<wwrFloatComplex, int>(wwrblasHandle_t, int,
                                                            const wwrFloatComplex *, int, int *);
extern template wwrblasStatus_t iamax<wwrFloatComplex, int64_t>(wwrblasHandle_t, int64_t,
                                                                const wwrFloatComplex *, int64_t,
                                                                int64_t *);
extern template wwrblasStatus_t iamax<wwrDoubleComplex, int>(wwrblasHandle_t, int,
                                                             const wwrDoubleComplex *, int, int *);
extern template wwrblasStatus_t iamax<wwrDoubleComplex, int64_t>(wwrblasHandle_t, int64_t,
                                                                 const wwrDoubleComplex *, int64_t,
                                                                 int64_t *);

// Function: iamin
extern template wwrblasStatus_t iamin<float, int>(wwrblasHandle_t, int, const float *, int, int *);
extern template wwrblasStatus_t iamin<float, int64_t>(wwrblasHandle_t, int64_t, const float *,
                                                      int64_t, int64_t *);
extern template wwrblasStatus_t iamin<double, int>(wwrblasHandle_t, int, const double *, int,
                                                   int *);
extern template wwrblasStatus_t iamin<double, int64_t>(wwrblasHandle_t, int64_t, const double *,
                                                       int64_t, int64_t *);
extern template wwrblasStatus_t iamin<wwrFloatComplex, int>(wwrblasHandle_t, int,
                                                            const wwrFloatComplex *, int, int *);
extern template wwrblasStatus_t iamin<wwrFloatComplex, int64_t>(wwrblasHandle_t, int64_t,
                                                                const wwrFloatComplex *, int64_t,
                                                                int64_t *);
extern template wwrblasStatus_t iamin<wwrDoubleComplex, int>(wwrblasHandle_t, int,
                                                             const wwrDoubleComplex *, int, int *);
extern template wwrblasStatus_t iamin<wwrDoubleComplex, int64_t>(wwrblasHandle_t, int64_t,
                                                                 const wwrDoubleComplex *, int64_t,
                                                                 int64_t *);

// Function: asum
extern template wwrblasStatus_t asum<float, int>(wwrblasHandle_t, int, const float *, int,
                                                 ComplexToRealType<float> *);
extern template wwrblasStatus_t asum<float, int64_t>(wwrblasHandle_t, int64_t, const float *,
                                                     int64_t, ComplexToRealType<float> *);
extern template wwrblasStatus_t asum<double, int>(wwrblasHandle_t, int, const double *, int,
                                                  ComplexToRealType<double> *);
extern template wwrblasStatus_t asum<double, int64_t>(wwrblasHandle_t, int64_t, const double *,
                                                      int64_t, ComplexToRealType<double> *);
extern template wwrblasStatus_t asum<wwrFloatComplex, int>(wwrblasHandle_t, int,
                                                           const wwrFloatComplex *, int,
                                                           ComplexToRealType<wwrFloatComplex> *);
extern template wwrblasStatus_t
asum<wwrFloatComplex, int64_t>(wwrblasHandle_t, int64_t, const wwrFloatComplex *, int64_t,
                               ComplexToRealType<wwrFloatComplex> *);
extern template wwrblasStatus_t asum<wwrDoubleComplex, int>(wwrblasHandle_t, int,
                                                            const wwrDoubleComplex *, int,
                                                            ComplexToRealType<wwrDoubleComplex> *);
extern template wwrblasStatus_t
asum<wwrDoubleComplex, int64_t>(wwrblasHandle_t, int64_t, const wwrDoubleComplex *, int64_t,
                                ComplexToRealType<wwrDoubleComplex> *);

// Function: axpy
extern template wwrblasStatus_t axpy<float, int>(wwrblasHandle_t, int, const float *, const float *,
                                                 int, float *, int);
extern template wwrblasStatus_t axpy<float, int64_t>(wwrblasHandle_t, int64_t, const float *,
                                                     const float *, int64_t, float *, int64_t);
extern template wwrblasStatus_t axpy<double, int>(wwrblasHandle_t, int, const double *,
                                                  const double *, int, double *, int);
extern template wwrblasStatus_t axpy<double, int64_t>(wwrblasHandle_t, int64_t, const double *,
                                                      const double *, int64_t, double *, int64_t);
extern template wwrblasStatus_t axpy<wwrFloatComplex, int>(wwrblasHandle_t, int,
                                                           const wwrFloatComplex *,
                                                           const wwrFloatComplex *, int,
                                                           wwrFloatComplex *, int);
extern template wwrblasStatus_t axpy<wwrFloatComplex, int64_t>(wwrblasHandle_t, int64_t,
                                                               const wwrFloatComplex *,
                                                               const wwrFloatComplex *, int64_t,
                                                               wwrFloatComplex *, int64_t);
extern template wwrblasStatus_t axpy<wwrDoubleComplex, int>(wwrblasHandle_t, int,
                                                            const wwrDoubleComplex *,
                                                            const wwrDoubleComplex *, int,
                                                            wwrDoubleComplex *, int);
extern template wwrblasStatus_t axpy<wwrDoubleComplex, int64_t>(wwrblasHandle_t, int64_t,
                                                                const wwrDoubleComplex *,
                                                                const wwrDoubleComplex *, int64_t,
                                                                wwrDoubleComplex *, int64_t);

// Function: copy
extern template wwrblasStatus_t copy<float, int>(wwrblasHandle_t, int, const float *, int, float *,
                                                 int);
extern template wwrblasStatus_t copy<float, int64_t>(wwrblasHandle_t, int64_t, const float *,
                                                     int64_t, float *, int64_t);
extern template wwrblasStatus_t copy<double, int>(wwrblasHandle_t, int, const double *, int,
                                                  double *, int);
extern template wwrblasStatus_t copy<double, int64_t>(wwrblasHandle_t, int64_t, const double *,
                                                      int64_t, double *, int64_t);
extern template wwrblasStatus_t copy<wwrFloatComplex, int>(wwrblasHandle_t, int,
                                                           const wwrFloatComplex *, int,
                                                           wwrFloatComplex *, int);
extern template wwrblasStatus_t copy<wwrFloatComplex, int64_t>(wwrblasHandle_t, int64_t,
                                                               const wwrFloatComplex *, int64_t,
                                                               wwrFloatComplex *, int64_t);
extern template wwrblasStatus_t copy<wwrDoubleComplex, int>(wwrblasHandle_t, int,
                                                            const wwrDoubleComplex *, int,
                                                            wwrDoubleComplex *, int);
extern template wwrblasStatus_t copy<wwrDoubleComplex, int64_t>(wwrblasHandle_t, int64_t,
                                                                const wwrDoubleComplex *, int64_t,
                                                                wwrDoubleComplex *, int64_t);

// Function: dot
extern template wwrblasStatus_t dot<float, int>(wwrblasHandle_t, int, const float *, int,
                                                const float *, int, float *);
extern template wwrblasStatus_t dot<float, int64_t>(wwrblasHandle_t, int64_t, const float *,
                                                    int64_t, const float *, int64_t, float *);
extern template wwrblasStatus_t dot<double, int>(wwrblasHandle_t, int, const double *, int,
                                                 const double *, int, double *);
extern template wwrblasStatus_t dot<double, int64_t>(wwrblasHandle_t, int64_t, const double *,
                                                     int64_t, const double *, int64_t, double *);

// Function: dotc
extern template wwrblasStatus_t dotc<wwrFloatComplex, int>(wwrblasHandle_t, int,
                                                           const wwrFloatComplex *, int,
                                                           const wwrFloatComplex *, int,
                                                           wwrFloatComplex *);
extern template wwrblasStatus_t dotc<wwrFloatComplex, int64_t>(wwrblasHandle_t, int64_t,
                                                               const wwrFloatComplex *, int64_t,
                                                               const wwrFloatComplex *, int64_t,
                                                               wwrFloatComplex *);
extern template wwrblasStatus_t dotc<wwrDoubleComplex, int>(wwrblasHandle_t, int,
                                                            const wwrDoubleComplex *, int,
                                                            const wwrDoubleComplex *, int,
                                                            wwrDoubleComplex *);
extern template wwrblasStatus_t dotc<wwrDoubleComplex, int64_t>(wwrblasHandle_t, int64_t,
                                                                const wwrDoubleComplex *, int64_t,
                                                                const wwrDoubleComplex *, int64_t,
                                                                wwrDoubleComplex *);

// Function: dotu
extern template wwrblasStatus_t dotu<wwrFloatComplex, int>(wwrblasHandle_t, int,
                                                           const wwrFloatComplex *, int,
                                                           const wwrFloatComplex *, int,
                                                           wwrFloatComplex *);
extern template wwrblasStatus_t dotu<wwrFloatComplex, int64_t>(wwrblasHandle_t, int64_t,
                                                               const wwrFloatComplex *, int64_t,
                                                               const wwrFloatComplex *, int64_t,
                                                               wwrFloatComplex *);
extern template wwrblasStatus_t dotu<wwrDoubleComplex, int>(wwrblasHandle_t, int,
                                                            const wwrDoubleComplex *, int,
                                                            const wwrDoubleComplex *, int,
                                                            wwrDoubleComplex *);
extern template wwrblasStatus_t dotu<wwrDoubleComplex, int64_t>(wwrblasHandle_t, int64_t,
                                                                const wwrDoubleComplex *, int64_t,
                                                                const wwrDoubleComplex *, int64_t,
                                                                wwrDoubleComplex *);

// Function: nrm2
extern template wwrblasStatus_t nrm2<float, int>(wwrblasHandle_t, int, const float *, int,
                                                 ComplexToRealType<float> *);
extern template wwrblasStatus_t nrm2<float, int64_t>(wwrblasHandle_t, int64_t, const float *,
                                                     int64_t, ComplexToRealType<float> *);
extern template wwrblasStatus_t nrm2<double, int>(wwrblasHandle_t, int, const double *, int,
                                                  ComplexToRealType<double> *);
extern template wwrblasStatus_t nrm2<double, int64_t>(wwrblasHandle_t, int64_t, const double *,
                                                      int64_t, ComplexToRealType<double> *);
extern template wwrblasStatus_t nrm2<wwrFloatComplex, int>(wwrblasHandle_t, int,
                                                           const wwrFloatComplex *, int,
                                                           ComplexToRealType<wwrFloatComplex> *);
extern template wwrblasStatus_t
nrm2<wwrFloatComplex, int64_t>(wwrblasHandle_t, int64_t, const wwrFloatComplex *, int64_t,
                               ComplexToRealType<wwrFloatComplex> *);
extern template wwrblasStatus_t nrm2<wwrDoubleComplex, int>(wwrblasHandle_t, int,
                                                            const wwrDoubleComplex *, int,
                                                            ComplexToRealType<wwrDoubleComplex> *);
extern template wwrblasStatus_t
nrm2<wwrDoubleComplex, int64_t>(wwrblasHandle_t, int64_t, const wwrDoubleComplex *, int64_t,
                                ComplexToRealType<wwrDoubleComplex> *);

// Function: rot
extern template wwrblasStatus_t rot<float, int>(wwrblasHandle_t, int, float *, int, float *, int,
                                                const float *, const float *);
extern template wwrblasStatus_t rot<float, int64_t>(wwrblasHandle_t, int64_t, float *, int64_t,
                                                    float *, int64_t, const float *, const float *);
extern template wwrblasStatus_t rot<double, int>(wwrblasHandle_t, int, double *, int, double *, int,
                                                 const double *, const double *);
extern template wwrblasStatus_t rot<double, int64_t>(wwrblasHandle_t, int64_t, double *, int64_t,
                                                     double *, int64_t, const double *,
                                                     const double *);
extern template wwrblasStatus_t
rot<wwrFloatComplex, int>(wwrblasHandle_t, int, wwrFloatComplex *, int, wwrFloatComplex *, int,
                          const ComplexToRealType<wwrFloatComplex> *, const wwrFloatComplex *);
extern template wwrblasStatus_t
rot<wwrFloatComplex, int64_t>(wwrblasHandle_t, int64_t, wwrFloatComplex *, int64_t,
                              wwrFloatComplex *, int64_t,
                              const ComplexToRealType<wwrFloatComplex> *, const wwrFloatComplex *);
extern template wwrblasStatus_t
rot<wwrDoubleComplex, int>(wwrblasHandle_t, int, wwrDoubleComplex *, int, wwrDoubleComplex *, int,
                           const ComplexToRealType<wwrDoubleComplex> *, const wwrDoubleComplex *);
extern template wwrblasStatus_t rot<wwrDoubleComplex, int64_t>(
    wwrblasHandle_t, int64_t, wwrDoubleComplex *, int64_t, wwrDoubleComplex *, int64_t,
    const ComplexToRealType<wwrDoubleComplex> *, const wwrDoubleComplex *);
extern template wwrblasStatus_t
rot<wwrFloatComplex, int>(wwrblasHandle_t, int, wwrFloatComplex *, int, wwrFloatComplex *, int,
                          const ComplexToRealType<wwrFloatComplex> *,
                          const ComplexToRealType<wwrFloatComplex> *);
extern template wwrblasStatus_t rot<wwrFloatComplex, int64_t>(
    wwrblasHandle_t, int64_t, wwrFloatComplex *, int64_t, wwrFloatComplex *, int64_t,
    const ComplexToRealType<wwrFloatComplex> *, const ComplexToRealType<wwrFloatComplex> *);
extern template wwrblasStatus_t
rot<wwrDoubleComplex, int>(wwrblasHandle_t, int, wwrDoubleComplex *, int, wwrDoubleComplex *, int,
                           const ComplexToRealType<wwrDoubleComplex> *,
                           const ComplexToRealType<wwrDoubleComplex> *);
extern template wwrblasStatus_t rot<wwrDoubleComplex, int64_t>(
    wwrblasHandle_t, int64_t, wwrDoubleComplex *, int64_t, wwrDoubleComplex *, int64_t,
    const ComplexToRealType<wwrDoubleComplex> *, const ComplexToRealType<wwrDoubleComplex> *);

// Function: rotg
extern template wwrblasStatus_t rotg<float, int>(wwrblasHandle_t, float *, float *, float *,
                                                 float *);
extern template wwrblasStatus_t rotg<double, int>(wwrblasHandle_t, double *, double *, double *,
                                                  double *);
extern template wwrblasStatus_t rotg<wwrFloatComplex, int>(wwrblasHandle_t, wwrFloatComplex *,
                                                           wwrFloatComplex *,
                                                           ComplexToRealType<wwrFloatComplex> *,
                                                           wwrFloatComplex *);
extern template wwrblasStatus_t rotg<wwrDoubleComplex, int>(wwrblasHandle_t, wwrDoubleComplex *,
                                                            wwrDoubleComplex *,
                                                            ComplexToRealType<wwrDoubleComplex> *,
                                                            wwrDoubleComplex *);

// Function: rotm
extern template wwrblasStatus_t rotm<float, int>(wwrblasHandle_t, int, float *, int, float *, int,
                                                 const float *);
extern template wwrblasStatus_t rotm<float, int64_t>(wwrblasHandle_t, int64_t, float *, int64_t,
                                                     float *, int64_t, const float *);
extern template wwrblasStatus_t rotm<double, int>(wwrblasHandle_t, int, double *, int, double *,
                                                  int, const double *);
extern template wwrblasStatus_t rotm<double, int64_t>(wwrblasHandle_t, int64_t, double *, int64_t,
                                                      double *, int64_t, const double *);

// Function: rotmg
extern template wwrblasStatus_t rotmg<float, int>(wwrblasHandle_t, float *, float *, float *,
                                                  const float *, float *);
extern template wwrblasStatus_t rotmg<double, int>(wwrblasHandle_t, double *, double *, double *,
                                                   const double *, double *);

// Function: scal
extern template wwrblasStatus_t scal<float, int>(wwrblasHandle_t, int, const float *, float *, int);
extern template wwrblasStatus_t scal<float, int64_t>(wwrblasHandle_t, int64_t, const float *,
                                                     float *, int64_t);
extern template wwrblasStatus_t scal<double, int>(wwrblasHandle_t, int, const double *, double *,
                                                  int);
extern template wwrblasStatus_t scal<double, int64_t>(wwrblasHandle_t, int64_t, const double *,
                                                      double *, int64_t);
extern template wwrblasStatus_t
scal<wwrFloatComplex, int>(wwrblasHandle_t, int, const wwrFloatComplex *, wwrFloatComplex *, int);
extern template wwrblasStatus_t scal<wwrFloatComplex, int64_t>(wwrblasHandle_t, int64_t,
                                                               const wwrFloatComplex *,
                                                               wwrFloatComplex *, int64_t);
extern template wwrblasStatus_t scal<wwrDoubleComplex, int>(wwrblasHandle_t, int,
                                                            const wwrDoubleComplex *,
                                                            wwrDoubleComplex *, int);
extern template wwrblasStatus_t scal<wwrDoubleComplex, int64_t>(wwrblasHandle_t, int64_t,
                                                                const wwrDoubleComplex *,
                                                                wwrDoubleComplex *, int64_t);
extern template wwrblasStatus_t
scal<wwrFloatComplex, int>(wwrblasHandle_t, int, const ComplexToRealType<wwrFloatComplex> *,
                           wwrFloatComplex *, int);
extern template wwrblasStatus_t
scal<wwrFloatComplex, int64_t>(wwrblasHandle_t, int64_t, const ComplexToRealType<wwrFloatComplex> *,
                               wwrFloatComplex *, int64_t);
extern template wwrblasStatus_t
scal<wwrDoubleComplex, int>(wwrblasHandle_t, int, const ComplexToRealType<wwrDoubleComplex> *,
                            wwrDoubleComplex *, int);
extern template wwrblasStatus_t
scal<wwrDoubleComplex, int64_t>(wwrblasHandle_t, int64_t,
                                const ComplexToRealType<wwrDoubleComplex> *, wwrDoubleComplex *,
                                int64_t);

// Function: swap
extern template wwrblasStatus_t swap<float, int>(wwrblasHandle_t, int, float *, int, float *, int);
extern template wwrblasStatus_t swap<float, int64_t>(wwrblasHandle_t, int64_t, float *, int64_t,
                                                     float *, int64_t);
extern template wwrblasStatus_t swap<double, int>(wwrblasHandle_t, int, double *, int, double *,
                                                  int);
extern template wwrblasStatus_t swap<double, int64_t>(wwrblasHandle_t, int64_t, double *, int64_t,
                                                      double *, int64_t);
extern template wwrblasStatus_t swap<wwrFloatComplex, int>(wwrblasHandle_t, int, wwrFloatComplex *,
                                                           int, wwrFloatComplex *, int);
extern template wwrblasStatus_t swap<wwrFloatComplex, int64_t>(wwrblasHandle_t, int64_t,
                                                               wwrFloatComplex *, int64_t,
                                                               wwrFloatComplex *, int64_t);
extern template wwrblasStatus_t
swap<wwrDoubleComplex, int>(wwrblasHandle_t, int, wwrDoubleComplex *, int, wwrDoubleComplex *, int);
extern template wwrblasStatus_t swap<wwrDoubleComplex, int64_t>(wwrblasHandle_t, int64_t,
                                                                wwrDoubleComplex *, int64_t,
                                                                wwrDoubleComplex *, int64_t);

} // namespace wwr
