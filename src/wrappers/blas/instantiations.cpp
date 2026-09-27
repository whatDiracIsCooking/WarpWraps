/**
 * @file instantiations.cpp
 * @brief Explicit template instantiations for the GPU BLAS wrappers
 *
 * This file contains all explicit template instantiations to avoid
 * code bloat from implicit instantiation at every call site.
 */

module wwr.wrappers.blas;

import wwr.blas;
import wwr.complex;
import wwr.wrappers.common;

namespace wwr {

// Function: iamax
template gpublasStatus_t iamax<float, int>(gpublasHandle_t, int, const float *, int, int *);
template gpublasStatus_t iamax<float, int64_t>(gpublasHandle_t, int64_t, const float *, int64_t,
                                               int64_t *);
template gpublasStatus_t iamax<double, int>(gpublasHandle_t, int, const double *, int, int *);
template gpublasStatus_t iamax<double, int64_t>(gpublasHandle_t, int64_t, const double *, int64_t,
                                                int64_t *);
template gpublasStatus_t iamax<gpuFloatComplex, int>(gpublasHandle_t, int, const gpuFloatComplex *,
                                                     int, int *);
template gpublasStatus_t iamax<gpuFloatComplex, int64_t>(gpublasHandle_t, int64_t,
                                                         const gpuFloatComplex *, int64_t,
                                                         int64_t *);
template gpublasStatus_t iamax<gpuDoubleComplex, int>(gpublasHandle_t, int,
                                                      const gpuDoubleComplex *, int, int *);
template gpublasStatus_t iamax<gpuDoubleComplex, int64_t>(gpublasHandle_t, int64_t,
                                                          const gpuDoubleComplex *, int64_t,
                                                          int64_t *);

// Function: iamin
template gpublasStatus_t iamin<float, int>(gpublasHandle_t, int, const float *, int, int *);
template gpublasStatus_t iamin<float, int64_t>(gpublasHandle_t, int64_t, const float *, int64_t,
                                               int64_t *);
template gpublasStatus_t iamin<double, int>(gpublasHandle_t, int, const double *, int, int *);
template gpublasStatus_t iamin<double, int64_t>(gpublasHandle_t, int64_t, const double *, int64_t,
                                                int64_t *);
template gpublasStatus_t iamin<gpuFloatComplex, int>(gpublasHandle_t, int, const gpuFloatComplex *,
                                                     int, int *);
template gpublasStatus_t iamin<gpuFloatComplex, int64_t>(gpublasHandle_t, int64_t,
                                                         const gpuFloatComplex *, int64_t,
                                                         int64_t *);
template gpublasStatus_t iamin<gpuDoubleComplex, int>(gpublasHandle_t, int,
                                                      const gpuDoubleComplex *, int, int *);
template gpublasStatus_t iamin<gpuDoubleComplex, int64_t>(gpublasHandle_t, int64_t,
                                                          const gpuDoubleComplex *, int64_t,
                                                          int64_t *);

// Function: asum
template gpublasStatus_t asum<float, int>(gpublasHandle_t, int, const float *, int,
                                          ComplexToRealType<float> *);
template gpublasStatus_t asum<float, int64_t>(gpublasHandle_t, int64_t, const float *, int64_t,
                                              ComplexToRealType<float> *);
template gpublasStatus_t asum<double, int>(gpublasHandle_t, int, const double *, int,
                                           ComplexToRealType<double> *);
template gpublasStatus_t asum<double, int64_t>(gpublasHandle_t, int64_t, const double *, int64_t,
                                               ComplexToRealType<double> *);
template gpublasStatus_t asum<gpuFloatComplex, int>(gpublasHandle_t, int, const gpuFloatComplex *,
                                                    int, ComplexToRealType<gpuFloatComplex> *);
template gpublasStatus_t asum<gpuFloatComplex, int64_t>(gpublasHandle_t, int64_t,
                                                        const gpuFloatComplex *, int64_t,
                                                        ComplexToRealType<gpuFloatComplex> *);
template gpublasStatus_t asum<gpuDoubleComplex, int>(gpublasHandle_t, int, const gpuDoubleComplex *,
                                                     int, ComplexToRealType<gpuDoubleComplex> *);
template gpublasStatus_t asum<gpuDoubleComplex, int64_t>(gpublasHandle_t, int64_t,
                                                         const gpuDoubleComplex *, int64_t,
                                                         ComplexToRealType<gpuDoubleComplex> *);

// Function: axpy
template gpublasStatus_t axpy<float, int>(gpublasHandle_t, int, const float *, const float *, int,
                                          float *, int);
template gpublasStatus_t axpy<float, int64_t>(gpublasHandle_t, int64_t, const float *,
                                              const float *, int64_t, float *, int64_t);
template gpublasStatus_t axpy<double, int>(gpublasHandle_t, int, const double *, const double *,
                                           int, double *, int);
template gpublasStatus_t axpy<double, int64_t>(gpublasHandle_t, int64_t, const double *,
                                               const double *, int64_t, double *, int64_t);
template gpublasStatus_t axpy<gpuFloatComplex, int>(gpublasHandle_t, int, const gpuFloatComplex *,
                                                    const gpuFloatComplex *, int, gpuFloatComplex *,
                                                    int);
template gpublasStatus_t axpy<gpuFloatComplex, int64_t>(gpublasHandle_t, int64_t,
                                                        const gpuFloatComplex *,
                                                        const gpuFloatComplex *, int64_t,
                                                        gpuFloatComplex *, int64_t);
template gpublasStatus_t axpy<gpuDoubleComplex, int>(gpublasHandle_t, int, const gpuDoubleComplex *,
                                                     const gpuDoubleComplex *, int,
                                                     gpuDoubleComplex *, int);
template gpublasStatus_t axpy<gpuDoubleComplex, int64_t>(gpublasHandle_t, int64_t,
                                                         const gpuDoubleComplex *,
                                                         const gpuDoubleComplex *, int64_t,
                                                         gpuDoubleComplex *, int64_t);

// Function: copy
template gpublasStatus_t copy<float, int>(gpublasHandle_t, int, const float *, int, float *, int);
template gpublasStatus_t copy<float, int64_t>(gpublasHandle_t, int64_t, const float *, int64_t,
                                              float *, int64_t);
template gpublasStatus_t copy<double, int>(gpublasHandle_t, int, const double *, int, double *,
                                           int);
template gpublasStatus_t copy<double, int64_t>(gpublasHandle_t, int64_t, const double *, int64_t,
                                               double *, int64_t);
template gpublasStatus_t copy<gpuFloatComplex, int>(gpublasHandle_t, int, const gpuFloatComplex *,
                                                    int, gpuFloatComplex *, int);
template gpublasStatus_t copy<gpuFloatComplex, int64_t>(gpublasHandle_t, int64_t,
                                                        const gpuFloatComplex *, int64_t,
                                                        gpuFloatComplex *, int64_t);
template gpublasStatus_t copy<gpuDoubleComplex, int>(gpublasHandle_t, int, const gpuDoubleComplex *,
                                                     int, gpuDoubleComplex *, int);
template gpublasStatus_t copy<gpuDoubleComplex, int64_t>(gpublasHandle_t, int64_t,
                                                         const gpuDoubleComplex *, int64_t,
                                                         gpuDoubleComplex *, int64_t);

// Function: dot
template gpublasStatus_t dot<float, int>(gpublasHandle_t, int, const float *, int, const float *,
                                         int, float *);
template gpublasStatus_t dot<float, int64_t>(gpublasHandle_t, int64_t, const float *, int64_t,
                                             const float *, int64_t, float *);
template gpublasStatus_t dot<double, int>(gpublasHandle_t, int, const double *, int, const double *,
                                          int, double *);
template gpublasStatus_t dot<double, int64_t>(gpublasHandle_t, int64_t, const double *, int64_t,
                                              const double *, int64_t, double *);

// Function: dotc
template gpublasStatus_t dotc<gpuFloatComplex, int>(gpublasHandle_t, int, const gpuFloatComplex *,
                                                    int, const gpuFloatComplex *, int,
                                                    gpuFloatComplex *);
template gpublasStatus_t dotc<gpuFloatComplex, int64_t>(gpublasHandle_t, int64_t,
                                                        const gpuFloatComplex *, int64_t,
                                                        const gpuFloatComplex *, int64_t,
                                                        gpuFloatComplex *);
template gpublasStatus_t dotc<gpuDoubleComplex, int>(gpublasHandle_t, int, const gpuDoubleComplex *,
                                                     int, const gpuDoubleComplex *, int,
                                                     gpuDoubleComplex *);
template gpublasStatus_t dotc<gpuDoubleComplex, int64_t>(gpublasHandle_t, int64_t,
                                                         const gpuDoubleComplex *, int64_t,
                                                         const gpuDoubleComplex *, int64_t,
                                                         gpuDoubleComplex *);

// Function: dotu
template gpublasStatus_t dotu<gpuFloatComplex, int>(gpublasHandle_t, int, const gpuFloatComplex *,
                                                    int, const gpuFloatComplex *, int,
                                                    gpuFloatComplex *);
template gpublasStatus_t dotu<gpuFloatComplex, int64_t>(gpublasHandle_t, int64_t,
                                                        const gpuFloatComplex *, int64_t,
                                                        const gpuFloatComplex *, int64_t,
                                                        gpuFloatComplex *);
template gpublasStatus_t dotu<gpuDoubleComplex, int>(gpublasHandle_t, int, const gpuDoubleComplex *,
                                                     int, const gpuDoubleComplex *, int,
                                                     gpuDoubleComplex *);
template gpublasStatus_t dotu<gpuDoubleComplex, int64_t>(gpublasHandle_t, int64_t,
                                                         const gpuDoubleComplex *, int64_t,
                                                         const gpuDoubleComplex *, int64_t,
                                                         gpuDoubleComplex *);

// Function: nrm2
template gpublasStatus_t nrm2<float, int>(gpublasHandle_t, int, const float *, int,
                                          ComplexToRealType<float> *);
template gpublasStatus_t nrm2<float, int64_t>(gpublasHandle_t, int64_t, const float *, int64_t,
                                              ComplexToRealType<float> *);
template gpublasStatus_t nrm2<double, int>(gpublasHandle_t, int, const double *, int,
                                           ComplexToRealType<double> *);
template gpublasStatus_t nrm2<double, int64_t>(gpublasHandle_t, int64_t, const double *, int64_t,
                                               ComplexToRealType<double> *);
template gpublasStatus_t nrm2<gpuFloatComplex, int>(gpublasHandle_t, int, const gpuFloatComplex *,
                                                    int, ComplexToRealType<gpuFloatComplex> *);
template gpublasStatus_t nrm2<gpuFloatComplex, int64_t>(gpublasHandle_t, int64_t,
                                                        const gpuFloatComplex *, int64_t,
                                                        ComplexToRealType<gpuFloatComplex> *);
template gpublasStatus_t nrm2<gpuDoubleComplex, int>(gpublasHandle_t, int, const gpuDoubleComplex *,
                                                     int, ComplexToRealType<gpuDoubleComplex> *);
template gpublasStatus_t nrm2<gpuDoubleComplex, int64_t>(gpublasHandle_t, int64_t,
                                                         const gpuDoubleComplex *, int64_t,
                                                         ComplexToRealType<gpuDoubleComplex> *);

// Function: rot
template gpublasStatus_t rot<float, int>(gpublasHandle_t, int, float *, int, float *, int,
                                         const float *, const float *);
template gpublasStatus_t rot<float, int64_t>(gpublasHandle_t, int64_t, float *, int64_t, float *,
                                             int64_t, const float *, const float *);
template gpublasStatus_t rot<double, int>(gpublasHandle_t, int, double *, int, double *, int,
                                          const double *, const double *);
template gpublasStatus_t rot<double, int64_t>(gpublasHandle_t, int64_t, double *, int64_t, double *,
                                              int64_t, const double *, const double *);
template gpublasStatus_t rot<gpuFloatComplex, int>(gpublasHandle_t, int, gpuFloatComplex *, int,
                                                   gpuFloatComplex *, int,
                                                   const ComplexToRealType<gpuFloatComplex> *,
                                                   const gpuFloatComplex *);
template gpublasStatus_t rot<gpuFloatComplex, int64_t>(gpublasHandle_t, int64_t, gpuFloatComplex *,
                                                       int64_t, gpuFloatComplex *, int64_t,
                                                       const ComplexToRealType<gpuFloatComplex> *,
                                                       const gpuFloatComplex *);
template gpublasStatus_t rot<gpuDoubleComplex, int>(gpublasHandle_t, int, gpuDoubleComplex *, int,
                                                    gpuDoubleComplex *, int,
                                                    const ComplexToRealType<gpuDoubleComplex> *,
                                                    const gpuDoubleComplex *);
template gpublasStatus_t rot<gpuDoubleComplex, int64_t>(gpublasHandle_t, int64_t,
                                                        gpuDoubleComplex *, int64_t,
                                                        gpuDoubleComplex *, int64_t,
                                                        const ComplexToRealType<gpuDoubleComplex> *,
                                                        const gpuDoubleComplex *);
template gpublasStatus_t rot<gpuFloatComplex, int>(gpublasHandle_t, int, gpuFloatComplex *, int,
                                                   gpuFloatComplex *, int,
                                                   const ComplexToRealType<gpuFloatComplex> *,
                                                   const ComplexToRealType<gpuFloatComplex> *);
template gpublasStatus_t rot<gpuFloatComplex, int64_t>(gpublasHandle_t, int64_t, gpuFloatComplex *,
                                                       int64_t, gpuFloatComplex *, int64_t,
                                                       const ComplexToRealType<gpuFloatComplex> *,
                                                       const ComplexToRealType<gpuFloatComplex> *);
template gpublasStatus_t rot<gpuDoubleComplex, int>(gpublasHandle_t, int, gpuDoubleComplex *, int,
                                                    gpuDoubleComplex *, int,
                                                    const ComplexToRealType<gpuDoubleComplex> *,
                                                    const ComplexToRealType<gpuDoubleComplex> *);
template gpublasStatus_t rot<gpuDoubleComplex, int64_t>(
    gpublasHandle_t, int64_t, gpuDoubleComplex *, int64_t, gpuDoubleComplex *, int64_t,
    const ComplexToRealType<gpuDoubleComplex> *, const ComplexToRealType<gpuDoubleComplex> *);

// Function: rotg
template gpublasStatus_t rotg<float, int>(gpublasHandle_t, float *, float *, float *, float *);
template gpublasStatus_t rotg<double, int>(gpublasHandle_t, double *, double *, double *, double *);
template gpublasStatus_t rotg<gpuFloatComplex, int>(gpublasHandle_t, gpuFloatComplex *,
                                                    gpuFloatComplex *,
                                                    ComplexToRealType<gpuFloatComplex> *,
                                                    gpuFloatComplex *);
template gpublasStatus_t rotg<gpuDoubleComplex, int>(gpublasHandle_t, gpuDoubleComplex *,
                                                     gpuDoubleComplex *,
                                                     ComplexToRealType<gpuDoubleComplex> *,
                                                     gpuDoubleComplex *);

// Function: rotm
template gpublasStatus_t rotm<float, int>(gpublasHandle_t, int, float *, int, float *, int,
                                          const float *);
template gpublasStatus_t rotm<float, int64_t>(gpublasHandle_t, int64_t, float *, int64_t, float *,
                                              int64_t, const float *);
template gpublasStatus_t rotm<double, int>(gpublasHandle_t, int, double *, int, double *, int,
                                           const double *);
template gpublasStatus_t rotm<double, int64_t>(gpublasHandle_t, int64_t, double *, int64_t,
                                               double *, int64_t, const double *);

// Function: rotmg
template gpublasStatus_t rotmg<float, int>(gpublasHandle_t, float *, float *, float *,
                                           const float *, float *);
template gpublasStatus_t rotmg<double, int>(gpublasHandle_t, double *, double *, double *,
                                            const double *, double *);

// Function: scal
template gpublasStatus_t scal<float, int>(gpublasHandle_t, int, const float *, float *, int);
template gpublasStatus_t scal<float, int64_t>(gpublasHandle_t, int64_t, const float *, float *,
                                              int64_t);
template gpublasStatus_t scal<double, int>(gpublasHandle_t, int, const double *, double *, int);
template gpublasStatus_t scal<double, int64_t>(gpublasHandle_t, int64_t, const double *, double *,
                                               int64_t);
template gpublasStatus_t scal<gpuFloatComplex, int>(gpublasHandle_t, int, const gpuFloatComplex *,
                                                    gpuFloatComplex *, int);
template gpublasStatus_t scal<gpuFloatComplex, int64_t>(gpublasHandle_t, int64_t,
                                                        const gpuFloatComplex *, gpuFloatComplex *,
                                                        int64_t);
template gpublasStatus_t scal<gpuDoubleComplex, int>(gpublasHandle_t, int, const gpuDoubleComplex *,
                                                     gpuDoubleComplex *, int);
template gpublasStatus_t scal<gpuDoubleComplex, int64_t>(gpublasHandle_t, int64_t,
                                                         const gpuDoubleComplex *,
                                                         gpuDoubleComplex *, int64_t);
template gpublasStatus_t scal<gpuFloatComplex, int>(gpublasHandle_t, int,
                                                    const ComplexToRealType<gpuFloatComplex> *,
                                                    gpuFloatComplex *, int);
template gpublasStatus_t scal<gpuFloatComplex, int64_t>(gpublasHandle_t, int64_t,
                                                        const ComplexToRealType<gpuFloatComplex> *,
                                                        gpuFloatComplex *, int64_t);
template gpublasStatus_t scal<gpuDoubleComplex, int>(gpublasHandle_t, int,
                                                     const ComplexToRealType<gpuDoubleComplex> *,
                                                     gpuDoubleComplex *, int);
template gpublasStatus_t
scal<gpuDoubleComplex, int64_t>(gpublasHandle_t, int64_t,
                                const ComplexToRealType<gpuDoubleComplex> *, gpuDoubleComplex *,
                                int64_t);

// Function: swap
template gpublasStatus_t swap<float, int>(gpublasHandle_t, int, float *, int, float *, int);
template gpublasStatus_t swap<float, int64_t>(gpublasHandle_t, int64_t, float *, int64_t, float *,
                                              int64_t);
template gpublasStatus_t swap<double, int>(gpublasHandle_t, int, double *, int, double *, int);
template gpublasStatus_t swap<double, int64_t>(gpublasHandle_t, int64_t, double *, int64_t,
                                               double *, int64_t);
template gpublasStatus_t swap<gpuFloatComplex, int>(gpublasHandle_t, int, gpuFloatComplex *, int,
                                                    gpuFloatComplex *, int);
template gpublasStatus_t swap<gpuFloatComplex, int64_t>(gpublasHandle_t, int64_t, gpuFloatComplex *,
                                                        int64_t, gpuFloatComplex *, int64_t);
template gpublasStatus_t swap<gpuDoubleComplex, int>(gpublasHandle_t, int, gpuDoubleComplex *, int,
                                                     gpuDoubleComplex *, int);
template gpublasStatus_t swap<gpuDoubleComplex, int64_t>(gpublasHandle_t, int64_t,
                                                         gpuDoubleComplex *, int64_t,
                                                         gpuDoubleComplex *, int64_t);

// Function: gemv
template gpublasStatus_t gemv<float, int>(gpublasHandle_t, gpublasOperation_t, int, int,
                                          const float *, const float *, int, const float *, int,
                                          const float *, float *, int);
template gpublasStatus_t gemv<float, int64_t>(gpublasHandle_t, gpublasOperation_t, int64_t, int64_t,
                                              const float *, const float *, int64_t, const float *,
                                              int64_t, const float *, float *, int64_t);
template gpublasStatus_t gemv<double, int>(gpublasHandle_t, gpublasOperation_t, int, int,
                                           const double *, const double *, int, const double *, int,
                                           const double *, double *, int);
template gpublasStatus_t gemv<double, int64_t>(gpublasHandle_t, gpublasOperation_t, int64_t,
                                               int64_t, const double *, const double *, int64_t,
                                               const double *, int64_t, const double *, double *,
                                               int64_t);
template gpublasStatus_t
gemv<gpuFloatComplex, int>(gpublasHandle_t, gpublasOperation_t, int, int, const gpuFloatComplex *,
                           const gpuFloatComplex *, int, const gpuFloatComplex *, int,
                           const gpuFloatComplex *, gpuFloatComplex *, int);
template gpublasStatus_t gemv<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasOperation_t,
                                                        int64_t, int64_t, const gpuFloatComplex *,
                                                        const gpuFloatComplex *, int64_t,
                                                        const gpuFloatComplex *, int64_t,
                                                        const gpuFloatComplex *, gpuFloatComplex *,
                                                        int64_t);
template gpublasStatus_t
gemv<gpuDoubleComplex, int>(gpublasHandle_t, gpublasOperation_t, int, int, const gpuDoubleComplex *,
                            const gpuDoubleComplex *, int, const gpuDoubleComplex *, int,
                            const gpuDoubleComplex *, gpuDoubleComplex *, int);
template gpublasStatus_t gemv<gpuDoubleComplex, int64_t>(gpublasHandle_t, gpublasOperation_t,
                                                         int64_t, int64_t, const gpuDoubleComplex *,
                                                         const gpuDoubleComplex *, int64_t,
                                                         const gpuDoubleComplex *, int64_t,
                                                         const gpuDoubleComplex *,
                                                         gpuDoubleComplex *, int64_t);

// Function: gbmv
template gpublasStatus_t gbmv<float, int>(gpublasHandle_t, gpublasOperation_t, int, int, int, int,
                                          const float *, const float *, int, const float *, int,
                                          const float *, float *, int);
template gpublasStatus_t gbmv<float, int64_t>(gpublasHandle_t, gpublasOperation_t, int64_t, int64_t,
                                              int64_t, int64_t, const float *, const float *,
                                              int64_t, const float *, int64_t, const float *,
                                              float *, int64_t);
template gpublasStatus_t gbmv<double, int>(gpublasHandle_t, gpublasOperation_t, int, int, int, int,
                                           const double *, const double *, int, const double *, int,
                                           const double *, double *, int);
template gpublasStatus_t gbmv<double, int64_t>(gpublasHandle_t, gpublasOperation_t, int64_t,
                                               int64_t, int64_t, int64_t, const double *,
                                               const double *, int64_t, const double *, int64_t,
                                               const double *, double *, int64_t);
template gpublasStatus_t gbmv<gpuFloatComplex, int>(gpublasHandle_t, gpublasOperation_t, int, int,
                                                    int, int, const gpuFloatComplex *,
                                                    const gpuFloatComplex *, int,
                                                    const gpuFloatComplex *, int,
                                                    const gpuFloatComplex *, gpuFloatComplex *,
                                                    int);
template gpublasStatus_t
gbmv<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasOperation_t, int64_t, int64_t, int64_t,
                               int64_t, const gpuFloatComplex *, const gpuFloatComplex *, int64_t,
                               const gpuFloatComplex *, int64_t, const gpuFloatComplex *,
                               gpuFloatComplex *, int64_t);
template gpublasStatus_t gbmv<gpuDoubleComplex, int>(gpublasHandle_t, gpublasOperation_t, int, int,
                                                     int, int, const gpuDoubleComplex *,
                                                     const gpuDoubleComplex *, int,
                                                     const gpuDoubleComplex *, int,
                                                     const gpuDoubleComplex *, gpuDoubleComplex *,
                                                     int);
template gpublasStatus_t
gbmv<gpuDoubleComplex, int64_t>(gpublasHandle_t, gpublasOperation_t, int64_t, int64_t, int64_t,
                                int64_t, const gpuDoubleComplex *, const gpuDoubleComplex *,
                                int64_t, const gpuDoubleComplex *, int64_t,
                                const gpuDoubleComplex *, gpuDoubleComplex *, int64_t);

// Function: ger
template gpublasStatus_t ger<float, int>(gpublasHandle_t, int, int, const float *, const float *,
                                         int, const float *, int, float *, int);
template gpublasStatus_t ger<float, int64_t>(gpublasHandle_t, int64_t, int64_t, const float *,
                                             const float *, int64_t, const float *, int64_t,
                                             float *, int64_t);
template gpublasStatus_t ger<double, int>(gpublasHandle_t, int, int, const double *, const double *,
                                          int, const double *, int, double *, int);
template gpublasStatus_t ger<double, int64_t>(gpublasHandle_t, int64_t, int64_t, const double *,
                                              const double *, int64_t, const double *, int64_t,
                                              double *, int64_t);

// Function: geru
template gpublasStatus_t geru<gpuFloatComplex, int>(gpublasHandle_t, int, int,
                                                    const gpuFloatComplex *,
                                                    const gpuFloatComplex *, int,
                                                    const gpuFloatComplex *, int, gpuFloatComplex *,
                                                    int);
template gpublasStatus_t geru<gpuFloatComplex, int64_t>(gpublasHandle_t, int64_t, int64_t,
                                                        const gpuFloatComplex *,
                                                        const gpuFloatComplex *, int64_t,
                                                        const gpuFloatComplex *, int64_t,
                                                        gpuFloatComplex *, int64_t);
template gpublasStatus_t geru<gpuDoubleComplex, int>(gpublasHandle_t, int, int,
                                                     const gpuDoubleComplex *,
                                                     const gpuDoubleComplex *, int,
                                                     const gpuDoubleComplex *, int,
                                                     gpuDoubleComplex *, int);
template gpublasStatus_t geru<gpuDoubleComplex, int64_t>(gpublasHandle_t, int64_t, int64_t,
                                                         const gpuDoubleComplex *,
                                                         const gpuDoubleComplex *, int64_t,
                                                         const gpuDoubleComplex *, int64_t,
                                                         gpuDoubleComplex *, int64_t);

// Function: gerc
template gpublasStatus_t gerc<gpuFloatComplex, int>(gpublasHandle_t, int, int,
                                                    const gpuFloatComplex *,
                                                    const gpuFloatComplex *, int,
                                                    const gpuFloatComplex *, int, gpuFloatComplex *,
                                                    int);
template gpublasStatus_t gerc<gpuFloatComplex, int64_t>(gpublasHandle_t, int64_t, int64_t,
                                                        const gpuFloatComplex *,
                                                        const gpuFloatComplex *, int64_t,
                                                        const gpuFloatComplex *, int64_t,
                                                        gpuFloatComplex *, int64_t);
template gpublasStatus_t gerc<gpuDoubleComplex, int>(gpublasHandle_t, int, int,
                                                     const gpuDoubleComplex *,
                                                     const gpuDoubleComplex *, int,
                                                     const gpuDoubleComplex *, int,
                                                     gpuDoubleComplex *, int);
template gpublasStatus_t gerc<gpuDoubleComplex, int64_t>(gpublasHandle_t, int64_t, int64_t,
                                                         const gpuDoubleComplex *,
                                                         const gpuDoubleComplex *, int64_t,
                                                         const gpuDoubleComplex *, int64_t,
                                                         gpuDoubleComplex *, int64_t);

// Function: symv
template gpublasStatus_t symv<float, int>(gpublasHandle_t, gpublasFillMode_t, int, const float *,
                                          const float *, int, const float *, int, const float *,
                                          float *, int);
template gpublasStatus_t symv<float, int64_t>(gpublasHandle_t, gpublasFillMode_t, int64_t,
                                              const float *, const float *, int64_t, const float *,
                                              int64_t, const float *, float *, int64_t);
template gpublasStatus_t symv<double, int>(gpublasHandle_t, gpublasFillMode_t, int, const double *,
                                           const double *, int, const double *, int, const double *,
                                           double *, int);
template gpublasStatus_t symv<double, int64_t>(gpublasHandle_t, gpublasFillMode_t, int64_t,
                                               const double *, const double *, int64_t,
                                               const double *, int64_t, const double *, double *,
                                               int64_t);

// Function: syr
template gpublasStatus_t syr<float, int>(gpublasHandle_t, gpublasFillMode_t, int, const float *,
                                         const float *, int, float *, int);
template gpublasStatus_t syr<float, int64_t>(gpublasHandle_t, gpublasFillMode_t, int64_t,
                                             const float *, const float *, int64_t, float *,
                                             int64_t);
template gpublasStatus_t syr<double, int>(gpublasHandle_t, gpublasFillMode_t, int, const double *,
                                          const double *, int, double *, int);
template gpublasStatus_t syr<double, int64_t>(gpublasHandle_t, gpublasFillMode_t, int64_t,
                                              const double *, const double *, int64_t, double *,
                                              int64_t);

// Function: syr2
template gpublasStatus_t syr2<float, int>(gpublasHandle_t, gpublasFillMode_t, int, const float *,
                                          const float *, int, const float *, int, float *, int);
template gpublasStatus_t syr2<float, int64_t>(gpublasHandle_t, gpublasFillMode_t, int64_t,
                                              const float *, const float *, int64_t, const float *,
                                              int64_t, float *, int64_t);
template gpublasStatus_t syr2<double, int>(gpublasHandle_t, gpublasFillMode_t, int, const double *,
                                           const double *, int, const double *, int, double *, int);
template gpublasStatus_t syr2<double, int64_t>(gpublasHandle_t, gpublasFillMode_t, int64_t,
                                               const double *, const double *, int64_t,
                                               const double *, int64_t, double *, int64_t);

// Function: sbmv
template gpublasStatus_t sbmv<float, int>(gpublasHandle_t, gpublasFillMode_t, int, int,
                                          const float *, const float *, int, const float *, int,
                                          const float *, float *, int);
template gpublasStatus_t sbmv<float, int64_t>(gpublasHandle_t, gpublasFillMode_t, int64_t, int64_t,
                                              const float *, const float *, int64_t, const float *,
                                              int64_t, const float *, float *, int64_t);
template gpublasStatus_t sbmv<double, int>(gpublasHandle_t, gpublasFillMode_t, int, int,
                                           const double *, const double *, int, const double *, int,
                                           const double *, double *, int);
template gpublasStatus_t sbmv<double, int64_t>(gpublasHandle_t, gpublasFillMode_t, int64_t, int64_t,
                                               const double *, const double *, int64_t,
                                               const double *, int64_t, const double *, double *,
                                               int64_t);

// Function: spmv
template gpublasStatus_t spmv<float, int>(gpublasHandle_t, gpublasFillMode_t, int, const float *,
                                          const float *, const float *, int, const float *, float *,
                                          int);
template gpublasStatus_t spmv<float, int64_t>(gpublasHandle_t, gpublasFillMode_t, int64_t,
                                              const float *, const float *, const float *, int64_t,
                                              const float *, float *, int64_t);
template gpublasStatus_t spmv<double, int>(gpublasHandle_t, gpublasFillMode_t, int, const double *,
                                           const double *, const double *, int, const double *,
                                           double *, int);
template gpublasStatus_t spmv<double, int64_t>(gpublasHandle_t, gpublasFillMode_t, int64_t,
                                               const double *, const double *, const double *,
                                               int64_t, const double *, double *, int64_t);

// Function: spr
template gpublasStatus_t spr<float, int>(gpublasHandle_t, gpublasFillMode_t, int, const float *,
                                         const float *, int, float *);
template gpublasStatus_t spr<float, int64_t>(gpublasHandle_t, gpublasFillMode_t, int64_t,
                                             const float *, const float *, int64_t, float *);
template gpublasStatus_t spr<double, int>(gpublasHandle_t, gpublasFillMode_t, int, const double *,
                                          const double *, int, double *);
template gpublasStatus_t spr<double, int64_t>(gpublasHandle_t, gpublasFillMode_t, int64_t,
                                              const double *, const double *, int64_t, double *);

// Function: spr2
template gpublasStatus_t spr2<float, int>(gpublasHandle_t, gpublasFillMode_t, int, const float *,
                                          const float *, int, const float *, int, float *);
template gpublasStatus_t spr2<float, int64_t>(gpublasHandle_t, gpublasFillMode_t, int64_t,
                                              const float *, const float *, int64_t, const float *,
                                              int64_t, float *);
template gpublasStatus_t spr2<double, int>(gpublasHandle_t, gpublasFillMode_t, int, const double *,
                                           const double *, int, const double *, int, double *);
template gpublasStatus_t spr2<double, int64_t>(gpublasHandle_t, gpublasFillMode_t, int64_t,
                                               const double *, const double *, int64_t,
                                               const double *, int64_t, double *);

// Function: trmv
template gpublasStatus_t trmv<float, int>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t,
                                          gpublasDiagType_t, int, const float *, int, float *, int);
template gpublasStatus_t trmv<float, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                              gpublasOperation_t, gpublasDiagType_t, int64_t,
                                              const float *, int64_t, float *, int64_t);
template gpublasStatus_t trmv<double, int>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t,
                                           gpublasDiagType_t, int, const double *, int, double *,
                                           int);
template gpublasStatus_t trmv<double, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                               gpublasOperation_t, gpublasDiagType_t, int64_t,
                                               const double *, int64_t, double *, int64_t);
template gpublasStatus_t trmv<gpuFloatComplex, int>(gpublasHandle_t, gpublasFillMode_t,
                                                    gpublasOperation_t, gpublasDiagType_t, int,
                                                    const gpuFloatComplex *, int, gpuFloatComplex *,
                                                    int);
template gpublasStatus_t trmv<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                        gpublasOperation_t, gpublasDiagType_t,
                                                        int64_t, const gpuFloatComplex *, int64_t,
                                                        gpuFloatComplex *, int64_t);
template gpublasStatus_t trmv<gpuDoubleComplex, int>(gpublasHandle_t, gpublasFillMode_t,
                                                     gpublasOperation_t, gpublasDiagType_t, int,
                                                     const gpuDoubleComplex *, int,
                                                     gpuDoubleComplex *, int);
template gpublasStatus_t trmv<gpuDoubleComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                         gpublasOperation_t, gpublasDiagType_t,
                                                         int64_t, const gpuDoubleComplex *, int64_t,
                                                         gpuDoubleComplex *, int64_t);

// Function: trsv
template gpublasStatus_t trsv<float, int>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t,
                                          gpublasDiagType_t, int, const float *, int, float *, int);
template gpublasStatus_t trsv<float, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                              gpublasOperation_t, gpublasDiagType_t, int64_t,
                                              const float *, int64_t, float *, int64_t);
template gpublasStatus_t trsv<double, int>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t,
                                           gpublasDiagType_t, int, const double *, int, double *,
                                           int);
template gpublasStatus_t trsv<double, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                               gpublasOperation_t, gpublasDiagType_t, int64_t,
                                               const double *, int64_t, double *, int64_t);
template gpublasStatus_t trsv<gpuFloatComplex, int>(gpublasHandle_t, gpublasFillMode_t,
                                                    gpublasOperation_t, gpublasDiagType_t, int,
                                                    const gpuFloatComplex *, int, gpuFloatComplex *,
                                                    int);
template gpublasStatus_t trsv<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                        gpublasOperation_t, gpublasDiagType_t,
                                                        int64_t, const gpuFloatComplex *, int64_t,
                                                        gpuFloatComplex *, int64_t);
template gpublasStatus_t trsv<gpuDoubleComplex, int>(gpublasHandle_t, gpublasFillMode_t,
                                                     gpublasOperation_t, gpublasDiagType_t, int,
                                                     const gpuDoubleComplex *, int,
                                                     gpuDoubleComplex *, int);
template gpublasStatus_t trsv<gpuDoubleComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                         gpublasOperation_t, gpublasDiagType_t,
                                                         int64_t, const gpuDoubleComplex *, int64_t,
                                                         gpuDoubleComplex *, int64_t);

// Function: tbmv
template gpublasStatus_t tbmv<float, int>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t,
                                          gpublasDiagType_t, int, int, const float *, int, float *,
                                          int);
template gpublasStatus_t tbmv<float, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                              gpublasOperation_t, gpublasDiagType_t, int64_t,
                                              int64_t, const float *, int64_t, float *, int64_t);
template gpublasStatus_t tbmv<double, int>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t,
                                           gpublasDiagType_t, int, int, const double *, int,
                                           double *, int);
template gpublasStatus_t tbmv<double, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                               gpublasOperation_t, gpublasDiagType_t, int64_t,
                                               int64_t, const double *, int64_t, double *, int64_t);
template gpublasStatus_t tbmv<gpuFloatComplex, int>(gpublasHandle_t, gpublasFillMode_t,
                                                    gpublasOperation_t, gpublasDiagType_t, int, int,
                                                    const gpuFloatComplex *, int, gpuFloatComplex *,
                                                    int);
template gpublasStatus_t tbmv<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                        gpublasOperation_t, gpublasDiagType_t,
                                                        int64_t, int64_t, const gpuFloatComplex *,
                                                        int64_t, gpuFloatComplex *, int64_t);
template gpublasStatus_t tbmv<gpuDoubleComplex, int>(gpublasHandle_t, gpublasFillMode_t,
                                                     gpublasOperation_t, gpublasDiagType_t, int,
                                                     int, const gpuDoubleComplex *, int,
                                                     gpuDoubleComplex *, int);
template gpublasStatus_t tbmv<gpuDoubleComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                         gpublasOperation_t, gpublasDiagType_t,
                                                         int64_t, int64_t, const gpuDoubleComplex *,
                                                         int64_t, gpuDoubleComplex *, int64_t);

// Function: tbsv
template gpublasStatus_t tbsv<float, int>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t,
                                          gpublasDiagType_t, int, int, const float *, int, float *,
                                          int);
template gpublasStatus_t tbsv<float, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                              gpublasOperation_t, gpublasDiagType_t, int64_t,
                                              int64_t, const float *, int64_t, float *, int64_t);
template gpublasStatus_t tbsv<double, int>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t,
                                           gpublasDiagType_t, int, int, const double *, int,
                                           double *, int);
template gpublasStatus_t tbsv<double, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                               gpublasOperation_t, gpublasDiagType_t, int64_t,
                                               int64_t, const double *, int64_t, double *, int64_t);
template gpublasStatus_t tbsv<gpuFloatComplex, int>(gpublasHandle_t, gpublasFillMode_t,
                                                    gpublasOperation_t, gpublasDiagType_t, int, int,
                                                    const gpuFloatComplex *, int, gpuFloatComplex *,
                                                    int);
template gpublasStatus_t tbsv<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                        gpublasOperation_t, gpublasDiagType_t,
                                                        int64_t, int64_t, const gpuFloatComplex *,
                                                        int64_t, gpuFloatComplex *, int64_t);
template gpublasStatus_t tbsv<gpuDoubleComplex, int>(gpublasHandle_t, gpublasFillMode_t,
                                                     gpublasOperation_t, gpublasDiagType_t, int,
                                                     int, const gpuDoubleComplex *, int,
                                                     gpuDoubleComplex *, int);
template gpublasStatus_t tbsv<gpuDoubleComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                         gpublasOperation_t, gpublasDiagType_t,
                                                         int64_t, int64_t, const gpuDoubleComplex *,
                                                         int64_t, gpuDoubleComplex *, int64_t);

// Function: tpmv
template gpublasStatus_t tpmv<float, int>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t,
                                          gpublasDiagType_t, int, const float *, float *, int);
template gpublasStatus_t tpmv<float, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                              gpublasOperation_t, gpublasDiagType_t, int64_t,
                                              const float *, float *, int64_t);
template gpublasStatus_t tpmv<double, int>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t,
                                           gpublasDiagType_t, int, const double *, double *, int);
template gpublasStatus_t tpmv<double, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                               gpublasOperation_t, gpublasDiagType_t, int64_t,
                                               const double *, double *, int64_t);
template gpublasStatus_t tpmv<gpuFloatComplex, int>(gpublasHandle_t, gpublasFillMode_t,
                                                    gpublasOperation_t, gpublasDiagType_t, int,
                                                    const gpuFloatComplex *, gpuFloatComplex *,
                                                    int);
template gpublasStatus_t tpmv<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                        gpublasOperation_t, gpublasDiagType_t,
                                                        int64_t, const gpuFloatComplex *,
                                                        gpuFloatComplex *, int64_t);
template gpublasStatus_t tpmv<gpuDoubleComplex, int>(gpublasHandle_t, gpublasFillMode_t,
                                                     gpublasOperation_t, gpublasDiagType_t, int,
                                                     const gpuDoubleComplex *, gpuDoubleComplex *,
                                                     int);
template gpublasStatus_t tpmv<gpuDoubleComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                         gpublasOperation_t, gpublasDiagType_t,
                                                         int64_t, const gpuDoubleComplex *,
                                                         gpuDoubleComplex *, int64_t);

// Function: tpsv
template gpublasStatus_t tpsv<float, int>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t,
                                          gpublasDiagType_t, int, const float *, float *, int);
template gpublasStatus_t tpsv<float, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                              gpublasOperation_t, gpublasDiagType_t, int64_t,
                                              const float *, float *, int64_t);
template gpublasStatus_t tpsv<double, int>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t,
                                           gpublasDiagType_t, int, const double *, double *, int);
template gpublasStatus_t tpsv<double, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                               gpublasOperation_t, gpublasDiagType_t, int64_t,
                                               const double *, double *, int64_t);
template gpublasStatus_t tpsv<gpuFloatComplex, int>(gpublasHandle_t, gpublasFillMode_t,
                                                    gpublasOperation_t, gpublasDiagType_t, int,
                                                    const gpuFloatComplex *, gpuFloatComplex *,
                                                    int);
template gpublasStatus_t tpsv<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                        gpublasOperation_t, gpublasDiagType_t,
                                                        int64_t, const gpuFloatComplex *,
                                                        gpuFloatComplex *, int64_t);
template gpublasStatus_t tpsv<gpuDoubleComplex, int>(gpublasHandle_t, gpublasFillMode_t,
                                                     gpublasOperation_t, gpublasDiagType_t, int,
                                                     const gpuDoubleComplex *, gpuDoubleComplex *,
                                                     int);
template gpublasStatus_t tpsv<gpuDoubleComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                         gpublasOperation_t, gpublasDiagType_t,
                                                         int64_t, const gpuDoubleComplex *,
                                                         gpuDoubleComplex *, int64_t);

// Function: hemv
template gpublasStatus_t
hemv<gpuFloatComplex, int>(gpublasHandle_t, gpublasFillMode_t, int, const gpuFloatComplex *,
                           const gpuFloatComplex *, int, const gpuFloatComplex *, int,
                           const gpuFloatComplex *, gpuFloatComplex *, int);
template gpublasStatus_t
hemv<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t, int64_t, const gpuFloatComplex *,
                               const gpuFloatComplex *, int64_t, const gpuFloatComplex *, int64_t,
                               const gpuFloatComplex *, gpuFloatComplex *, int64_t);
template gpublasStatus_t
hemv<gpuDoubleComplex, int>(gpublasHandle_t, gpublasFillMode_t, int, const gpuDoubleComplex *,
                            const gpuDoubleComplex *, int, const gpuDoubleComplex *, int,
                            const gpuDoubleComplex *, gpuDoubleComplex *, int);
template gpublasStatus_t hemv<gpuDoubleComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                         int64_t, const gpuDoubleComplex *,
                                                         const gpuDoubleComplex *, int64_t,
                                                         const gpuDoubleComplex *, int64_t,
                                                         const gpuDoubleComplex *,
                                                         gpuDoubleComplex *, int64_t);

// Function: hbmv
template gpublasStatus_t
hbmv<gpuFloatComplex, int>(gpublasHandle_t, gpublasFillMode_t, int, int, const gpuFloatComplex *,
                           const gpuFloatComplex *, int, const gpuFloatComplex *, int,
                           const gpuFloatComplex *, gpuFloatComplex *, int);
template gpublasStatus_t hbmv<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t, int64_t,
                                                        int64_t, const gpuFloatComplex *,
                                                        const gpuFloatComplex *, int64_t,
                                                        const gpuFloatComplex *, int64_t,
                                                        const gpuFloatComplex *, gpuFloatComplex *,
                                                        int64_t);
template gpublasStatus_t
hbmv<gpuDoubleComplex, int>(gpublasHandle_t, gpublasFillMode_t, int, int, const gpuDoubleComplex *,
                            const gpuDoubleComplex *, int, const gpuDoubleComplex *, int,
                            const gpuDoubleComplex *, gpuDoubleComplex *, int);
template gpublasStatus_t hbmv<gpuDoubleComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                         int64_t, int64_t, const gpuDoubleComplex *,
                                                         const gpuDoubleComplex *, int64_t,
                                                         const gpuDoubleComplex *, int64_t,
                                                         const gpuDoubleComplex *,
                                                         gpuDoubleComplex *, int64_t);

// Function: hpmv
template gpublasStatus_t
hpmv<gpuFloatComplex, int>(gpublasHandle_t, gpublasFillMode_t, int, const gpuFloatComplex *,
                           const gpuFloatComplex *, const gpuFloatComplex *, int,
                           const gpuFloatComplex *, gpuFloatComplex *, int);
template gpublasStatus_t
hpmv<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t, int64_t, const gpuFloatComplex *,
                               const gpuFloatComplex *, const gpuFloatComplex *, int64_t,
                               const gpuFloatComplex *, gpuFloatComplex *, int64_t);
template gpublasStatus_t
hpmv<gpuDoubleComplex, int>(gpublasHandle_t, gpublasFillMode_t, int, const gpuDoubleComplex *,
                            const gpuDoubleComplex *, const gpuDoubleComplex *, int,
                            const gpuDoubleComplex *, gpuDoubleComplex *, int);
template gpublasStatus_t hpmv<gpuDoubleComplex, int64_t>(
    gpublasHandle_t, gpublasFillMode_t, int64_t, const gpuDoubleComplex *, const gpuDoubleComplex *,
    const gpuDoubleComplex *, int64_t, const gpuDoubleComplex *, gpuDoubleComplex *, int64_t);

// Function: her
template gpublasStatus_t her<gpuFloatComplex, int>(gpublasHandle_t, gpublasFillMode_t, int,
                                                   const ComplexToRealType<gpuFloatComplex> *,
                                                   const gpuFloatComplex *, int, gpuFloatComplex *,
                                                   int);
template gpublasStatus_t her<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t, int64_t,
                                                       const ComplexToRealType<gpuFloatComplex> *,
                                                       const gpuFloatComplex *, int64_t,
                                                       gpuFloatComplex *, int64_t);
template gpublasStatus_t her<gpuDoubleComplex, int>(gpublasHandle_t, gpublasFillMode_t, int,
                                                    const ComplexToRealType<gpuDoubleComplex> *,
                                                    const gpuDoubleComplex *, int,
                                                    gpuDoubleComplex *, int);
template gpublasStatus_t her<gpuDoubleComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t, int64_t,
                                                        const ComplexToRealType<gpuDoubleComplex> *,
                                                        const gpuDoubleComplex *, int64_t,
                                                        gpuDoubleComplex *, int64_t);

// Function: her2
template gpublasStatus_t her2<gpuFloatComplex, int>(gpublasHandle_t, gpublasFillMode_t, int,
                                                    const gpuFloatComplex *,
                                                    const gpuFloatComplex *, int,
                                                    const gpuFloatComplex *, int, gpuFloatComplex *,
                                                    int);
template gpublasStatus_t her2<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t, int64_t,
                                                        const gpuFloatComplex *,
                                                        const gpuFloatComplex *, int64_t,
                                                        const gpuFloatComplex *, int64_t,
                                                        gpuFloatComplex *, int64_t);
template gpublasStatus_t her2<gpuDoubleComplex, int>(gpublasHandle_t, gpublasFillMode_t, int,
                                                     const gpuDoubleComplex *,
                                                     const gpuDoubleComplex *, int,
                                                     const gpuDoubleComplex *, int,
                                                     gpuDoubleComplex *, int);
template gpublasStatus_t her2<gpuDoubleComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                         int64_t, const gpuDoubleComplex *,
                                                         const gpuDoubleComplex *, int64_t,
                                                         const gpuDoubleComplex *, int64_t,
                                                         gpuDoubleComplex *, int64_t);

// Function: hpr
template gpublasStatus_t hpr<gpuFloatComplex, int>(gpublasHandle_t, gpublasFillMode_t, int,
                                                   const ComplexToRealType<gpuFloatComplex> *,
                                                   const gpuFloatComplex *, int, gpuFloatComplex *);
template gpublasStatus_t hpr<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t, int64_t,
                                                       const ComplexToRealType<gpuFloatComplex> *,
                                                       const gpuFloatComplex *, int64_t,
                                                       gpuFloatComplex *);
template gpublasStatus_t hpr<gpuDoubleComplex, int>(gpublasHandle_t, gpublasFillMode_t, int,
                                                    const ComplexToRealType<gpuDoubleComplex> *,
                                                    const gpuDoubleComplex *, int,
                                                    gpuDoubleComplex *);
template gpublasStatus_t hpr<gpuDoubleComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t, int64_t,
                                                        const ComplexToRealType<gpuDoubleComplex> *,
                                                        const gpuDoubleComplex *, int64_t,
                                                        gpuDoubleComplex *);

// Function: hpr2
template gpublasStatus_t hpr2<gpuFloatComplex, int>(gpublasHandle_t, gpublasFillMode_t, int,
                                                    const gpuFloatComplex *,
                                                    const gpuFloatComplex *, int,
                                                    const gpuFloatComplex *, int,
                                                    gpuFloatComplex *);
template gpublasStatus_t hpr2<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t, int64_t,
                                                        const gpuFloatComplex *,
                                                        const gpuFloatComplex *, int64_t,
                                                        const gpuFloatComplex *, int64_t,
                                                        gpuFloatComplex *);
template gpublasStatus_t hpr2<gpuDoubleComplex, int>(gpublasHandle_t, gpublasFillMode_t, int,
                                                     const gpuDoubleComplex *,
                                                     const gpuDoubleComplex *, int,
                                                     const gpuDoubleComplex *, int,
                                                     gpuDoubleComplex *);
template gpublasStatus_t hpr2<gpuDoubleComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                         int64_t, const gpuDoubleComplex *,
                                                         const gpuDoubleComplex *, int64_t,
                                                         const gpuDoubleComplex *, int64_t,
                                                         gpuDoubleComplex *);

// Function: gemvBatched
template gpublasStatus_t gemvBatched<float, int>(gpublasHandle_t, gpublasOperation_t, int, int,
                                                 const float *, const float *const[], int,
                                                 const float *const[], int, const float *,
                                                 float *const[], int, int);
template gpublasStatus_t gemvBatched<float, int64_t>(gpublasHandle_t, gpublasOperation_t, int64_t,
                                                     int64_t, const float *, const float *const[],
                                                     int64_t, const float *const[], int64_t,
                                                     const float *, float *const[], int64_t,
                                                     int64_t);
template gpublasStatus_t gemvBatched<double, int>(gpublasHandle_t, gpublasOperation_t, int, int,
                                                  const double *, const double *const[], int,
                                                  const double *const[], int, const double *,
                                                  double *const[], int, int);
template gpublasStatus_t
gemvBatched<double, int64_t>(gpublasHandle_t, gpublasOperation_t, int64_t, int64_t, const double *,
                             const double *const[], int64_t, const double *const[], int64_t,
                             const double *, double *const[], int64_t, int64_t);
template gpublasStatus_t gemvBatched<gpuFloatComplex, int>(gpublasHandle_t, gpublasOperation_t, int,
                                                           int, const gpuFloatComplex *,
                                                           const gpuFloatComplex *const[], int,
                                                           const gpuFloatComplex *const[], int,
                                                           const gpuFloatComplex *,
                                                           gpuFloatComplex *const[], int, int);
template gpublasStatus_t gemvBatched<gpuFloatComplex, int64_t>(
    gpublasHandle_t, gpublasOperation_t, int64_t, int64_t, const gpuFloatComplex *,
    const gpuFloatComplex *const[], int64_t, const gpuFloatComplex *const[], int64_t,
    const gpuFloatComplex *, gpuFloatComplex *const[], int64_t, int64_t);
template gpublasStatus_t gemvBatched<gpuDoubleComplex, int>(gpublasHandle_t, gpublasOperation_t,
                                                            int, int, const gpuDoubleComplex *,
                                                            const gpuDoubleComplex *const[], int,
                                                            const gpuDoubleComplex *const[], int,
                                                            const gpuDoubleComplex *,
                                                            gpuDoubleComplex *const[], int, int);
template gpublasStatus_t gemvBatched<gpuDoubleComplex, int64_t>(
    gpublasHandle_t, gpublasOperation_t, int64_t, int64_t, const gpuDoubleComplex *,
    const gpuDoubleComplex *const[], int64_t, const gpuDoubleComplex *const[], int64_t,
    const gpuDoubleComplex *, gpuDoubleComplex *const[], int64_t, int64_t);

// Function: gemvStridedBatched
template gpublasStatus_t gemvStridedBatched<float, int>(gpublasHandle_t, gpublasOperation_t, int,
                                                        int, const float *, const float *, int,
                                                        long long int, const float *, int,
                                                        long long int, const float *, float *, int,
                                                        long long int, int);
template gpublasStatus_t gemvStridedBatched<float, int64_t>(gpublasHandle_t, gpublasOperation_t,
                                                            int64_t, int64_t, const float *,
                                                            const float *, int64_t, long long int,
                                                            const float *, int64_t, long long int,
                                                            const float *, float *, int64_t,
                                                            long long int, int64_t);
template gpublasStatus_t gemvStridedBatched<double, int>(gpublasHandle_t, gpublasOperation_t, int,
                                                         int, const double *, const double *, int,
                                                         long long int, const double *, int,
                                                         long long int, const double *, double *,
                                                         int, long long int, int);
template gpublasStatus_t gemvStridedBatched<double, int64_t>(gpublasHandle_t, gpublasOperation_t,
                                                             int64_t, int64_t, const double *,
                                                             const double *, int64_t, long long int,
                                                             const double *, int64_t, long long int,
                                                             const double *, double *, int64_t,
                                                             long long int, int64_t);
template gpublasStatus_t gemvStridedBatched<gpuFloatComplex, int>(
    gpublasHandle_t, gpublasOperation_t, int, int, const gpuFloatComplex *, const gpuFloatComplex *,
    int, long long int, const gpuFloatComplex *, int, long long int, const gpuFloatComplex *,
    gpuFloatComplex *, int, long long int, int);
template gpublasStatus_t gemvStridedBatched<gpuFloatComplex, int64_t>(
    gpublasHandle_t, gpublasOperation_t, int64_t, int64_t, const gpuFloatComplex *,
    const gpuFloatComplex *, int64_t, long long int, const gpuFloatComplex *, int64_t,
    long long int, const gpuFloatComplex *, gpuFloatComplex *, int64_t, long long int, int64_t);
template gpublasStatus_t gemvStridedBatched<gpuDoubleComplex, int>(
    gpublasHandle_t, gpublasOperation_t, int, int, const gpuDoubleComplex *,
    const gpuDoubleComplex *, int, long long int, const gpuDoubleComplex *, int, long long int,
    const gpuDoubleComplex *, gpuDoubleComplex *, int, long long int, int);
template gpublasStatus_t gemvStridedBatched<gpuDoubleComplex, int64_t>(
    gpublasHandle_t, gpublasOperation_t, int64_t, int64_t, const gpuDoubleComplex *,
    const gpuDoubleComplex *, int64_t, long long int, const gpuDoubleComplex *, int64_t,
    long long int, const gpuDoubleComplex *, gpuDoubleComplex *, int64_t, long long int, int64_t);

// Function: gemm
template gpublasStatus_t gemm<float, int>(gpublasHandle_t, gpublasOperation_t, gpublasOperation_t,
                                          int, int, int, const float *, const float *, int,
                                          const float *, int, const float *, float *, int);
template gpublasStatus_t gemm<float, int64_t>(gpublasHandle_t, gpublasOperation_t,
                                              gpublasOperation_t, int64_t, int64_t, int64_t,
                                              const float *, const float *, int64_t, const float *,
                                              int64_t, const float *, float *, int64_t);
template gpublasStatus_t gemm<double, int>(gpublasHandle_t, gpublasOperation_t, gpublasOperation_t,
                                           int, int, int, const double *, const double *, int,
                                           const double *, int, const double *, double *, int);
template gpublasStatus_t gemm<double, int64_t>(gpublasHandle_t, gpublasOperation_t,
                                               gpublasOperation_t, int64_t, int64_t, int64_t,
                                               const double *, const double *, int64_t,
                                               const double *, int64_t, const double *, double *,
                                               int64_t);
template gpublasStatus_t
gemm<gpuFloatComplex, int>(gpublasHandle_t, gpublasOperation_t, gpublasOperation_t, int, int, int,
                           const gpuFloatComplex *, const gpuFloatComplex *, int,
                           const gpuFloatComplex *, int, const gpuFloatComplex *, gpuFloatComplex *,
                           int);
template gpublasStatus_t
gemm<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasOperation_t, gpublasOperation_t, int64_t,
                               int64_t, int64_t, const gpuFloatComplex *, const gpuFloatComplex *,
                               int64_t, const gpuFloatComplex *, int64_t, const gpuFloatComplex *,
                               gpuFloatComplex *, int64_t);
template gpublasStatus_t
gemm<gpuDoubleComplex, int>(gpublasHandle_t, gpublasOperation_t, gpublasOperation_t, int, int, int,
                            const gpuDoubleComplex *, const gpuDoubleComplex *, int,
                            const gpuDoubleComplex *, int, const gpuDoubleComplex *,
                            gpuDoubleComplex *, int);
template gpublasStatus_t
gemm<gpuDoubleComplex, int64_t>(gpublasHandle_t, gpublasOperation_t, gpublasOperation_t, int64_t,
                                int64_t, int64_t, const gpuDoubleComplex *,
                                const gpuDoubleComplex *, int64_t, const gpuDoubleComplex *,
                                int64_t, const gpuDoubleComplex *, gpuDoubleComplex *, int64_t);

// Function: gemmBatched
template gpublasStatus_t gemmBatched<float, int>(gpublasHandle_t, gpublasOperation_t,
                                                 gpublasOperation_t, int, int, int, const float *,
                                                 const float *const[], int, const float *const[],
                                                 int, const float *, float *const[], int, int);
template gpublasStatus_t gemmBatched<float, int64_t>(gpublasHandle_t, gpublasOperation_t,
                                                     gpublasOperation_t, int64_t, int64_t, int64_t,
                                                     const float *, const float *const[], int64_t,
                                                     const float *const[], int64_t, const float *,
                                                     float *const[], int64_t, int64_t);
template gpublasStatus_t gemmBatched<double, int>(gpublasHandle_t, gpublasOperation_t,
                                                  gpublasOperation_t, int, int, int, const double *,
                                                  const double *const[], int, const double *const[],
                                                  int, const double *, double *const[], int, int);
template gpublasStatus_t gemmBatched<double, int64_t>(gpublasHandle_t, gpublasOperation_t,
                                                      gpublasOperation_t, int64_t, int64_t, int64_t,
                                                      const double *, const double *const[],
                                                      int64_t, const double *const[], int64_t,
                                                      const double *, double *const[], int64_t,
                                                      int64_t);
template gpublasStatus_t
gemmBatched<gpuFloatComplex, int>(gpublasHandle_t, gpublasOperation_t, gpublasOperation_t, int, int,
                                  int, const gpuFloatComplex *, const gpuFloatComplex *const[], int,
                                  const gpuFloatComplex *const[], int, const gpuFloatComplex *,
                                  gpuFloatComplex *const[], int, int);
template gpublasStatus_t gemmBatched<gpuFloatComplex, int64_t>(
    gpublasHandle_t, gpublasOperation_t, gpublasOperation_t, int64_t, int64_t, int64_t,
    const gpuFloatComplex *, const gpuFloatComplex *const[], int64_t,
    const gpuFloatComplex *const[], int64_t, const gpuFloatComplex *, gpuFloatComplex *const[],
    int64_t, int64_t);
template gpublasStatus_t gemmBatched<gpuDoubleComplex, int>(
    gpublasHandle_t, gpublasOperation_t, gpublasOperation_t, int, int, int,
    const gpuDoubleComplex *, const gpuDoubleComplex *const[], int, const gpuDoubleComplex *const[],
    int, const gpuDoubleComplex *, gpuDoubleComplex *const[], int, int);
template gpublasStatus_t gemmBatched<gpuDoubleComplex, int64_t>(
    gpublasHandle_t, gpublasOperation_t, gpublasOperation_t, int64_t, int64_t, int64_t,
    const gpuDoubleComplex *, const gpuDoubleComplex *const[], int64_t,
    const gpuDoubleComplex *const[], int64_t, const gpuDoubleComplex *, gpuDoubleComplex *const[],
    int64_t, int64_t);

// Function: gemmStridedBatched
template gpublasStatus_t
gemmStridedBatched<float, int>(gpublasHandle_t, gpublasOperation_t, gpublasOperation_t, int, int,
                               int, const float *, const float *, int, long long int, const float *,
                               int, long long int, const float *, float *, int, long long int, int);
template gpublasStatus_t
gemmStridedBatched<float, int64_t>(gpublasHandle_t, gpublasOperation_t, gpublasOperation_t, int64_t,
                                   int64_t, int64_t, const float *, const float *, int64_t,
                                   long long int, const float *, int64_t, long long int,
                                   const float *, float *, int64_t, long long int, int64_t);
template gpublasStatus_t gemmStridedBatched<double, int>(gpublasHandle_t, gpublasOperation_t,
                                                         gpublasOperation_t, int, int, int,
                                                         const double *, const double *, int,
                                                         long long int, const double *, int,
                                                         long long int, const double *, double *,
                                                         int, long long int, int);
template gpublasStatus_t
gemmStridedBatched<double, int64_t>(gpublasHandle_t, gpublasOperation_t, gpublasOperation_t,
                                    int64_t, int64_t, int64_t, const double *, const double *,
                                    int64_t, long long int, const double *, int64_t, long long int,
                                    const double *, double *, int64_t, long long int, int64_t);
template gpublasStatus_t gemmStridedBatched<gpuFloatComplex, int>(
    gpublasHandle_t, gpublasOperation_t, gpublasOperation_t, int, int, int, const gpuFloatComplex *,
    const gpuFloatComplex *, int, long long int, const gpuFloatComplex *, int, long long int,
    const gpuFloatComplex *, gpuFloatComplex *, int, long long int, int);
template gpublasStatus_t gemmStridedBatched<gpuFloatComplex, int64_t>(
    gpublasHandle_t, gpublasOperation_t, gpublasOperation_t, int64_t, int64_t, int64_t,
    const gpuFloatComplex *, const gpuFloatComplex *, int64_t, long long int,
    const gpuFloatComplex *, int64_t, long long int, const gpuFloatComplex *, gpuFloatComplex *,
    int64_t, long long int, int64_t);
template gpublasStatus_t gemmStridedBatched<gpuDoubleComplex, int>(
    gpublasHandle_t, gpublasOperation_t, gpublasOperation_t, int, int, int,
    const gpuDoubleComplex *, const gpuDoubleComplex *, int, long long int,
    const gpuDoubleComplex *, int, long long int, const gpuDoubleComplex *, gpuDoubleComplex *, int,
    long long int, int);
template gpublasStatus_t gemmStridedBatched<gpuDoubleComplex, int64_t>(
    gpublasHandle_t, gpublasOperation_t, gpublasOperation_t, int64_t, int64_t, int64_t,
    const gpuDoubleComplex *, const gpuDoubleComplex *, int64_t, long long int,
    const gpuDoubleComplex *, int64_t, long long int, const gpuDoubleComplex *, gpuDoubleComplex *,
    int64_t, long long int, int64_t);

// Function: symm
template gpublasStatus_t symm<float, int>(gpublasHandle_t, gpublasSideMode_t, gpublasFillMode_t,
                                          int, int, const float *, const float *, int,
                                          const float *, int, const float *, float *, int);
template gpublasStatus_t symm<float, int64_t>(gpublasHandle_t, gpublasSideMode_t, gpublasFillMode_t,
                                              int64_t, int64_t, const float *, const float *,
                                              int64_t, const float *, int64_t, const float *,
                                              float *, int64_t);
template gpublasStatus_t symm<double, int>(gpublasHandle_t, gpublasSideMode_t, gpublasFillMode_t,
                                           int, int, const double *, const double *, int,
                                           const double *, int, const double *, double *, int);
template gpublasStatus_t symm<double, int64_t>(gpublasHandle_t, gpublasSideMode_t,
                                               gpublasFillMode_t, int64_t, int64_t, const double *,
                                               const double *, int64_t, const double *, int64_t,
                                               const double *, double *, int64_t);
template gpublasStatus_t
symm<gpuFloatComplex, int>(gpublasHandle_t, gpublasSideMode_t, gpublasFillMode_t, int, int,
                           const gpuFloatComplex *, const gpuFloatComplex *, int,
                           const gpuFloatComplex *, int, const gpuFloatComplex *, gpuFloatComplex *,
                           int);
template gpublasStatus_t
symm<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasSideMode_t, gpublasFillMode_t, int64_t,
                               int64_t, const gpuFloatComplex *, const gpuFloatComplex *, int64_t,
                               const gpuFloatComplex *, int64_t, const gpuFloatComplex *,
                               gpuFloatComplex *, int64_t);
template gpublasStatus_t
symm<gpuDoubleComplex, int>(gpublasHandle_t, gpublasSideMode_t, gpublasFillMode_t, int, int,
                            const gpuDoubleComplex *, const gpuDoubleComplex *, int,
                            const gpuDoubleComplex *, int, const gpuDoubleComplex *,
                            gpuDoubleComplex *, int);
template gpublasStatus_t
symm<gpuDoubleComplex, int64_t>(gpublasHandle_t, gpublasSideMode_t, gpublasFillMode_t, int64_t,
                                int64_t, const gpuDoubleComplex *, const gpuDoubleComplex *,
                                int64_t, const gpuDoubleComplex *, int64_t,
                                const gpuDoubleComplex *, gpuDoubleComplex *, int64_t);

// Function: syrk
template gpublasStatus_t syrk<float, int>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t,
                                          int, int, const float *, const float *, int,
                                          const float *, float *, int);
template gpublasStatus_t syrk<float, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                              gpublasOperation_t, int64_t, int64_t, const float *,
                                              const float *, int64_t, const float *, float *,
                                              int64_t);
template gpublasStatus_t syrk<double, int>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t,
                                           int, int, const double *, const double *, int,
                                           const double *, double *, int);
template gpublasStatus_t syrk<double, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                               gpublasOperation_t, int64_t, int64_t, const double *,
                                               const double *, int64_t, const double *, double *,
                                               int64_t);
template gpublasStatus_t
syrk<gpuFloatComplex, int>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t, int, int,
                           const gpuFloatComplex *, const gpuFloatComplex *, int,
                           const gpuFloatComplex *, gpuFloatComplex *, int);
template gpublasStatus_t
syrk<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t, int64_t,
                               int64_t, const gpuFloatComplex *, const gpuFloatComplex *, int64_t,
                               const gpuFloatComplex *, gpuFloatComplex *, int64_t);
template gpublasStatus_t
syrk<gpuDoubleComplex, int>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t, int, int,
                            const gpuDoubleComplex *, const gpuDoubleComplex *, int,
                            const gpuDoubleComplex *, gpuDoubleComplex *, int);
template gpublasStatus_t
syrk<gpuDoubleComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t, int64_t,
                                int64_t, const gpuDoubleComplex *, const gpuDoubleComplex *,
                                int64_t, const gpuDoubleComplex *, gpuDoubleComplex *, int64_t);

// Function: syr2k
template gpublasStatus_t syr2k<float, int>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t,
                                           int, int, const float *, const float *, int,
                                           const float *, int, const float *, float *, int);
template gpublasStatus_t syr2k<float, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                               gpublasOperation_t, int64_t, int64_t, const float *,
                                               const float *, int64_t, const float *, int64_t,
                                               const float *, float *, int64_t);
template gpublasStatus_t syr2k<double, int>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t,
                                            int, int, const double *, const double *, int,
                                            const double *, int, const double *, double *, int);
template gpublasStatus_t syr2k<double, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                gpublasOperation_t, int64_t, int64_t,
                                                const double *, const double *, int64_t,
                                                const double *, int64_t, const double *, double *,
                                                int64_t);
template gpublasStatus_t
syr2k<gpuFloatComplex, int>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t, int, int,
                            const gpuFloatComplex *, const gpuFloatComplex *, int,
                            const gpuFloatComplex *, int, const gpuFloatComplex *,
                            gpuFloatComplex *, int);
template gpublasStatus_t
syr2k<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t, int64_t,
                                int64_t, const gpuFloatComplex *, const gpuFloatComplex *, int64_t,
                                const gpuFloatComplex *, int64_t, const gpuFloatComplex *,
                                gpuFloatComplex *, int64_t);
template gpublasStatus_t
syr2k<gpuDoubleComplex, int>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t, int, int,
                             const gpuDoubleComplex *, const gpuDoubleComplex *, int,
                             const gpuDoubleComplex *, int, const gpuDoubleComplex *,
                             gpuDoubleComplex *, int);
template gpublasStatus_t
syr2k<gpuDoubleComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t, int64_t,
                                 int64_t, const gpuDoubleComplex *, const gpuDoubleComplex *,
                                 int64_t, const gpuDoubleComplex *, int64_t,
                                 const gpuDoubleComplex *, gpuDoubleComplex *, int64_t);

// Function: syrkx
template gpublasStatus_t syrkx<float, int>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t,
                                           int, int, const float *, const float *, int,
                                           const float *, int, const float *, float *, int);
template gpublasStatus_t syrkx<float, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                               gpublasOperation_t, int64_t, int64_t, const float *,
                                               const float *, int64_t, const float *, int64_t,
                                               const float *, float *, int64_t);
template gpublasStatus_t syrkx<double, int>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t,
                                            int, int, const double *, const double *, int,
                                            const double *, int, const double *, double *, int);
template gpublasStatus_t syrkx<double, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                gpublasOperation_t, int64_t, int64_t,
                                                const double *, const double *, int64_t,
                                                const double *, int64_t, const double *, double *,
                                                int64_t);
template gpublasStatus_t
syrkx<gpuFloatComplex, int>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t, int, int,
                            const gpuFloatComplex *, const gpuFloatComplex *, int,
                            const gpuFloatComplex *, int, const gpuFloatComplex *,
                            gpuFloatComplex *, int);
template gpublasStatus_t
syrkx<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t, int64_t,
                                int64_t, const gpuFloatComplex *, const gpuFloatComplex *, int64_t,
                                const gpuFloatComplex *, int64_t, const gpuFloatComplex *,
                                gpuFloatComplex *, int64_t);
template gpublasStatus_t
syrkx<gpuDoubleComplex, int>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t, int, int,
                             const gpuDoubleComplex *, const gpuDoubleComplex *, int,
                             const gpuDoubleComplex *, int, const gpuDoubleComplex *,
                             gpuDoubleComplex *, int);
template gpublasStatus_t
syrkx<gpuDoubleComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t, int64_t,
                                 int64_t, const gpuDoubleComplex *, const gpuDoubleComplex *,
                                 int64_t, const gpuDoubleComplex *, int64_t,
                                 const gpuDoubleComplex *, gpuDoubleComplex *, int64_t);

// Function: trmm
template gpublasStatus_t trmm<float, int>(gpublasHandle_t, gpublasSideMode_t, gpublasFillMode_t,
                                          gpublasOperation_t, gpublasDiagType_t, int, int,
                                          const float *, const float *, int, const float *, int,
                                          float *, int);
template gpublasStatus_t trmm<float, int64_t>(gpublasHandle_t, gpublasSideMode_t, gpublasFillMode_t,
                                              gpublasOperation_t, gpublasDiagType_t, int64_t,
                                              int64_t, const float *, const float *, int64_t,
                                              const float *, int64_t, float *, int64_t);
template gpublasStatus_t trmm<double, int>(gpublasHandle_t, gpublasSideMode_t, gpublasFillMode_t,
                                           gpublasOperation_t, gpublasDiagType_t, int, int,
                                           const double *, const double *, int, const double *, int,
                                           double *, int);
template gpublasStatus_t trmm<double, int64_t>(gpublasHandle_t, gpublasSideMode_t,
                                               gpublasFillMode_t, gpublasOperation_t,
                                               gpublasDiagType_t, int64_t, int64_t, const double *,
                                               const double *, int64_t, const double *, int64_t,
                                               double *, int64_t);
template gpublasStatus_t
trmm<gpuFloatComplex, int>(gpublasHandle_t, gpublasSideMode_t, gpublasFillMode_t,
                           gpublasOperation_t, gpublasDiagType_t, int, int, const gpuFloatComplex *,
                           const gpuFloatComplex *, int, const gpuFloatComplex *, int,
                           gpuFloatComplex *, int);
template gpublasStatus_t
trmm<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasSideMode_t, gpublasFillMode_t,
                               gpublasOperation_t, gpublasDiagType_t, int64_t, int64_t,
                               const gpuFloatComplex *, const gpuFloatComplex *, int64_t,
                               const gpuFloatComplex *, int64_t, gpuFloatComplex *, int64_t);
template gpublasStatus_t
trmm<gpuDoubleComplex, int>(gpublasHandle_t, gpublasSideMode_t, gpublasFillMode_t,
                            gpublasOperation_t, gpublasDiagType_t, int, int,
                            const gpuDoubleComplex *, const gpuDoubleComplex *, int,
                            const gpuDoubleComplex *, int, gpuDoubleComplex *, int);
template gpublasStatus_t
trmm<gpuDoubleComplex, int64_t>(gpublasHandle_t, gpublasSideMode_t, gpublasFillMode_t,
                                gpublasOperation_t, gpublasDiagType_t, int64_t, int64_t,
                                const gpuDoubleComplex *, const gpuDoubleComplex *, int64_t,
                                const gpuDoubleComplex *, int64_t, gpuDoubleComplex *, int64_t);

// Function: trsm
template gpublasStatus_t trsm<float, int>(gpublasHandle_t, gpublasSideMode_t, gpublasFillMode_t,
                                          gpublasOperation_t, gpublasDiagType_t, int, int,
                                          const float *, const float *, int, float *, int);
template gpublasStatus_t trsm<float, int64_t>(gpublasHandle_t, gpublasSideMode_t, gpublasFillMode_t,
                                              gpublasOperation_t, gpublasDiagType_t, int64_t,
                                              int64_t, const float *, const float *, int64_t,
                                              float *, int64_t);
template gpublasStatus_t trsm<double, int>(gpublasHandle_t, gpublasSideMode_t, gpublasFillMode_t,
                                           gpublasOperation_t, gpublasDiagType_t, int, int,
                                           const double *, const double *, int, double *, int);
template gpublasStatus_t trsm<double, int64_t>(gpublasHandle_t, gpublasSideMode_t,
                                               gpublasFillMode_t, gpublasOperation_t,
                                               gpublasDiagType_t, int64_t, int64_t, const double *,
                                               const double *, int64_t, double *, int64_t);
template gpublasStatus_t
trsm<gpuFloatComplex, int>(gpublasHandle_t, gpublasSideMode_t, gpublasFillMode_t,
                           gpublasOperation_t, gpublasDiagType_t, int, int, const gpuFloatComplex *,
                           const gpuFloatComplex *, int, gpuFloatComplex *, int);
template gpublasStatus_t trsm<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasSideMode_t,
                                                        gpublasFillMode_t, gpublasOperation_t,
                                                        gpublasDiagType_t, int64_t, int64_t,
                                                        const gpuFloatComplex *,
                                                        const gpuFloatComplex *, int64_t,
                                                        gpuFloatComplex *, int64_t);
template gpublasStatus_t trsm<gpuDoubleComplex, int>(
    gpublasHandle_t, gpublasSideMode_t, gpublasFillMode_t, gpublasOperation_t, gpublasDiagType_t,
    int, int, const gpuDoubleComplex *, const gpuDoubleComplex *, int, gpuDoubleComplex *, int);
template gpublasStatus_t trsm<gpuDoubleComplex, int64_t>(gpublasHandle_t, gpublasSideMode_t,
                                                         gpublasFillMode_t, gpublasOperation_t,
                                                         gpublasDiagType_t, int64_t, int64_t,
                                                         const gpuDoubleComplex *,
                                                         const gpuDoubleComplex *, int64_t,
                                                         gpuDoubleComplex *, int64_t);

// Function: trsmBatched
template gpublasStatus_t trsmBatched<float, int>(gpublasHandle_t, gpublasSideMode_t,
                                                 gpublasFillMode_t, gpublasOperation_t,
                                                 gpublasDiagType_t, int, int, const float *,
                                                 const float *const[], int, float *const[], int,
                                                 int);
template gpublasStatus_t trsmBatched<float, int64_t>(gpublasHandle_t, gpublasSideMode_t,
                                                     gpublasFillMode_t, gpublasOperation_t,
                                                     gpublasDiagType_t, int64_t, int64_t,
                                                     const float *, const float *const[], int64_t,
                                                     float *const[], int64_t, int64_t);
template gpublasStatus_t trsmBatched<double, int>(gpublasHandle_t, gpublasSideMode_t,
                                                  gpublasFillMode_t, gpublasOperation_t,
                                                  gpublasDiagType_t, int, int, const double *,
                                                  const double *const[], int, double *const[], int,
                                                  int);
template gpublasStatus_t trsmBatched<double, int64_t>(gpublasHandle_t, gpublasSideMode_t,
                                                      gpublasFillMode_t, gpublasOperation_t,
                                                      gpublasDiagType_t, int64_t, int64_t,
                                                      const double *, const double *const[],
                                                      int64_t, double *const[], int64_t, int64_t);
template gpublasStatus_t trsmBatched<gpuFloatComplex, int>(gpublasHandle_t, gpublasSideMode_t,
                                                           gpublasFillMode_t, gpublasOperation_t,
                                                           gpublasDiagType_t, int, int,
                                                           const gpuFloatComplex *,
                                                           const gpuFloatComplex *const[], int,
                                                           gpuFloatComplex *const[], int, int);
template gpublasStatus_t
trsmBatched<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasSideMode_t, gpublasFillMode_t,
                                      gpublasOperation_t, gpublasDiagType_t, int64_t, int64_t,
                                      const gpuFloatComplex *, const gpuFloatComplex *const[],
                                      int64_t, gpuFloatComplex *const[], int64_t, int64_t);
template gpublasStatus_t trsmBatched<gpuDoubleComplex, int>(gpublasHandle_t, gpublasSideMode_t,
                                                            gpublasFillMode_t, gpublasOperation_t,
                                                            gpublasDiagType_t, int, int,
                                                            const gpuDoubleComplex *,
                                                            const gpuDoubleComplex *const[], int,
                                                            gpuDoubleComplex *const[], int, int);
template gpublasStatus_t
trsmBatched<gpuDoubleComplex, int64_t>(gpublasHandle_t, gpublasSideMode_t, gpublasFillMode_t,
                                       gpublasOperation_t, gpublasDiagType_t, int64_t, int64_t,
                                       const gpuDoubleComplex *, const gpuDoubleComplex *const[],
                                       int64_t, gpuDoubleComplex *const[], int64_t, int64_t);

// Function: hemm
template gpublasStatus_t
hemm<gpuFloatComplex, int>(gpublasHandle_t, gpublasSideMode_t, gpublasFillMode_t, int, int,
                           const gpuFloatComplex *, const gpuFloatComplex *, int,
                           const gpuFloatComplex *, int, const gpuFloatComplex *, gpuFloatComplex *,
                           int);
template gpublasStatus_t
hemm<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasSideMode_t, gpublasFillMode_t, int64_t,
                               int64_t, const gpuFloatComplex *, const gpuFloatComplex *, int64_t,
                               const gpuFloatComplex *, int64_t, const gpuFloatComplex *,
                               gpuFloatComplex *, int64_t);
template gpublasStatus_t
hemm<gpuDoubleComplex, int>(gpublasHandle_t, gpublasSideMode_t, gpublasFillMode_t, int, int,
                            const gpuDoubleComplex *, const gpuDoubleComplex *, int,
                            const gpuDoubleComplex *, int, const gpuDoubleComplex *,
                            gpuDoubleComplex *, int);
template gpublasStatus_t
hemm<gpuDoubleComplex, int64_t>(gpublasHandle_t, gpublasSideMode_t, gpublasFillMode_t, int64_t,
                                int64_t, const gpuDoubleComplex *, const gpuDoubleComplex *,
                                int64_t, const gpuDoubleComplex *, int64_t,
                                const gpuDoubleComplex *, gpuDoubleComplex *, int64_t);

// Function: herk
template gpublasStatus_t
herk<gpuFloatComplex, int>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t, int, int,
                           const ComplexToRealType<gpuFloatComplex> *, const gpuFloatComplex *, int,
                           const ComplexToRealType<gpuFloatComplex> *, gpuFloatComplex *, int);
template gpublasStatus_t herk<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                        gpublasOperation_t, int64_t, int64_t,
                                                        const ComplexToRealType<gpuFloatComplex> *,
                                                        const gpuFloatComplex *, int64_t,
                                                        const ComplexToRealType<gpuFloatComplex> *,
                                                        gpuFloatComplex *, int64_t);
template gpublasStatus_t herk<gpuDoubleComplex, int>(gpublasHandle_t, gpublasFillMode_t,
                                                     gpublasOperation_t, int, int,
                                                     const ComplexToRealType<gpuDoubleComplex> *,
                                                     const gpuDoubleComplex *, int,
                                                     const ComplexToRealType<gpuDoubleComplex> *,
                                                     gpuDoubleComplex *, int);
template gpublasStatus_t herk<gpuDoubleComplex, int64_t>(
    gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t, int64_t, int64_t,
    const ComplexToRealType<gpuDoubleComplex> *, const gpuDoubleComplex *, int64_t,
    const ComplexToRealType<gpuDoubleComplex> *, gpuDoubleComplex *, int64_t);

// Function: her2k
template gpublasStatus_t
her2k<gpuFloatComplex, int>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t, int, int,
                            const gpuFloatComplex *, const gpuFloatComplex *, int,
                            const gpuFloatComplex *, int,
                            const ComplexToRealType<gpuFloatComplex> *, gpuFloatComplex *, int);
template gpublasStatus_t her2k<gpuFloatComplex, int64_t>(
    gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t, int64_t, int64_t,
    const gpuFloatComplex *, const gpuFloatComplex *, int64_t, const gpuFloatComplex *, int64_t,
    const ComplexToRealType<gpuFloatComplex> *, gpuFloatComplex *, int64_t);
template gpublasStatus_t
her2k<gpuDoubleComplex, int>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t, int, int,
                             const gpuDoubleComplex *, const gpuDoubleComplex *, int,
                             const gpuDoubleComplex *, int,
                             const ComplexToRealType<gpuDoubleComplex> *, gpuDoubleComplex *, int);
template gpublasStatus_t her2k<gpuDoubleComplex, int64_t>(
    gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t, int64_t, int64_t,
    const gpuDoubleComplex *, const gpuDoubleComplex *, int64_t, const gpuDoubleComplex *, int64_t,
    const ComplexToRealType<gpuDoubleComplex> *, gpuDoubleComplex *, int64_t);

// Function: herkx
template gpublasStatus_t
herkx<gpuFloatComplex, int>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t, int, int,
                            const gpuFloatComplex *, const gpuFloatComplex *, int,
                            const gpuFloatComplex *, int,
                            const ComplexToRealType<gpuFloatComplex> *, gpuFloatComplex *, int);
template gpublasStatus_t herkx<gpuFloatComplex, int64_t>(
    gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t, int64_t, int64_t,
    const gpuFloatComplex *, const gpuFloatComplex *, int64_t, const gpuFloatComplex *, int64_t,
    const ComplexToRealType<gpuFloatComplex> *, gpuFloatComplex *, int64_t);
template gpublasStatus_t
herkx<gpuDoubleComplex, int>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t, int, int,
                             const gpuDoubleComplex *, const gpuDoubleComplex *, int,
                             const gpuDoubleComplex *, int,
                             const ComplexToRealType<gpuDoubleComplex> *, gpuDoubleComplex *, int);
template gpublasStatus_t herkx<gpuDoubleComplex, int64_t>(
    gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t, int64_t, int64_t,
    const gpuDoubleComplex *, const gpuDoubleComplex *, int64_t, const gpuDoubleComplex *, int64_t,
    const ComplexToRealType<gpuDoubleComplex> *, gpuDoubleComplex *, int64_t);

// Function: geam
template gpublasStatus_t geam<float, int>(gpublasHandle_t, gpublasOperation_t, gpublasOperation_t,
                                          int, int, const float *, const float *, int,
                                          const float *, const float *, int, float *, int);
template gpublasStatus_t geam<float, int64_t>(gpublasHandle_t, gpublasOperation_t,
                                              gpublasOperation_t, int64_t, int64_t, const float *,
                                              const float *, int64_t, const float *, const float *,
                                              int64_t, float *, int64_t);
template gpublasStatus_t geam<double, int>(gpublasHandle_t, gpublasOperation_t, gpublasOperation_t,
                                           int, int, const double *, const double *, int,
                                           const double *, const double *, int, double *, int);
template gpublasStatus_t geam<double, int64_t>(gpublasHandle_t, gpublasOperation_t,
                                               gpublasOperation_t, int64_t, int64_t, const double *,
                                               const double *, int64_t, const double *,
                                               const double *, int64_t, double *, int64_t);
template gpublasStatus_t
geam<gpuFloatComplex, int>(gpublasHandle_t, gpublasOperation_t, gpublasOperation_t, int, int,
                           const gpuFloatComplex *, const gpuFloatComplex *, int,
                           const gpuFloatComplex *, const gpuFloatComplex *, int, gpuFloatComplex *,
                           int);
template gpublasStatus_t
geam<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasOperation_t, gpublasOperation_t, int64_t,
                               int64_t, const gpuFloatComplex *, const gpuFloatComplex *, int64_t,
                               const gpuFloatComplex *, const gpuFloatComplex *, int64_t,
                               gpuFloatComplex *, int64_t);
template gpublasStatus_t
geam<gpuDoubleComplex, int>(gpublasHandle_t, gpublasOperation_t, gpublasOperation_t, int, int,
                            const gpuDoubleComplex *, const gpuDoubleComplex *, int,
                            const gpuDoubleComplex *, const gpuDoubleComplex *, int,
                            gpuDoubleComplex *, int);
template gpublasStatus_t
geam<gpuDoubleComplex, int64_t>(gpublasHandle_t, gpublasOperation_t, gpublasOperation_t, int64_t,
                                int64_t, const gpuDoubleComplex *, const gpuDoubleComplex *,
                                int64_t, const gpuDoubleComplex *, const gpuDoubleComplex *,
                                int64_t, gpuDoubleComplex *, int64_t);

// Function: dgmm
template gpublasStatus_t dgmm<float, int>(gpublasHandle_t, gpublasSideMode_t, int, int,
                                          const float *, int, const float *, int, float *, int);
template gpublasStatus_t dgmm<float, int64_t>(gpublasHandle_t, gpublasSideMode_t, int64_t, int64_t,
                                              const float *, int64_t, const float *, int64_t,
                                              float *, int64_t);
template gpublasStatus_t dgmm<double, int>(gpublasHandle_t, gpublasSideMode_t, int, int,
                                           const double *, int, const double *, int, double *, int);
template gpublasStatus_t dgmm<double, int64_t>(gpublasHandle_t, gpublasSideMode_t, int64_t, int64_t,
                                               const double *, int64_t, const double *, int64_t,
                                               double *, int64_t);
template gpublasStatus_t dgmm<gpuFloatComplex, int>(gpublasHandle_t, gpublasSideMode_t, int, int,
                                                    const gpuFloatComplex *, int,
                                                    const gpuFloatComplex *, int, gpuFloatComplex *,
                                                    int);
template gpublasStatus_t dgmm<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasSideMode_t, int64_t,
                                                        int64_t, const gpuFloatComplex *, int64_t,
                                                        const gpuFloatComplex *, int64_t,
                                                        gpuFloatComplex *, int64_t);
template gpublasStatus_t dgmm<gpuDoubleComplex, int>(gpublasHandle_t, gpublasSideMode_t, int, int,
                                                     const gpuDoubleComplex *, int,
                                                     const gpuDoubleComplex *, int,
                                                     gpuDoubleComplex *, int);
template gpublasStatus_t dgmm<gpuDoubleComplex, int64_t>(gpublasHandle_t, gpublasSideMode_t,
                                                         int64_t, int64_t, const gpuDoubleComplex *,
                                                         int64_t, const gpuDoubleComplex *, int64_t,
                                                         gpuDoubleComplex *, int64_t);

// Function: getrfBatched
template gpublasStatus_t getrfBatched<float>(gpublasHandle_t, int, float *const[], int, int *,
                                             int *, int);
template gpublasStatus_t getrfBatched<double>(gpublasHandle_t, int, double *const[], int, int *,
                                              int *, int);
template gpublasStatus_t getrfBatched<gpuFloatComplex>(gpublasHandle_t, int,
                                                       gpuFloatComplex *const[], int, int *, int *,
                                                       int);
template gpublasStatus_t getrfBatched<gpuDoubleComplex>(gpublasHandle_t, int,
                                                        gpuDoubleComplex *const[], int, int *,
                                                        int *, int);

// Function: getrsBatched
template gpublasStatus_t getrsBatched<float>(gpublasHandle_t, gpublasOperation_t, int, int,
                                             const float *const[], int, const int *, float *const[],
                                             int, int *, int);
template gpublasStatus_t getrsBatched<double>(gpublasHandle_t, gpublasOperation_t, int, int,
                                              const double *const[], int, const int *,
                                              double *const[], int, int *, int);
template gpublasStatus_t getrsBatched<gpuFloatComplex>(gpublasHandle_t, gpublasOperation_t, int,
                                                       int, const gpuFloatComplex *const[], int,
                                                       const int *, gpuFloatComplex *const[], int,
                                                       int *, int);
template gpublasStatus_t getrsBatched<gpuDoubleComplex>(gpublasHandle_t, gpublasOperation_t, int,
                                                        int, const gpuDoubleComplex *const[], int,
                                                        const int *, gpuDoubleComplex *const[], int,
                                                        int *, int);

// Function: getriBatched
template gpublasStatus_t getriBatched<float>(gpublasHandle_t, int, const float *const[], int,
                                             const int *, float *const[], int, int *, int);
template gpublasStatus_t getriBatched<double>(gpublasHandle_t, int, const double *const[], int,
                                              const int *, double *const[], int, int *, int);
template gpublasStatus_t getriBatched<gpuFloatComplex>(gpublasHandle_t, int,
                                                       const gpuFloatComplex *const[], int,
                                                       const int *, gpuFloatComplex *const[], int,
                                                       int *, int);
template gpublasStatus_t getriBatched<gpuDoubleComplex>(gpublasHandle_t, int,
                                                        const gpuDoubleComplex *const[], int,
                                                        const int *, gpuDoubleComplex *const[], int,
                                                        int *, int);

// Function: geqrfBatched
template gpublasStatus_t geqrfBatched<float>(gpublasHandle_t, int, int, float *const[], int,
                                             float *const[], int *, int);
template gpublasStatus_t geqrfBatched<double>(gpublasHandle_t, int, int, double *const[], int,
                                              double *const[], int *, int);
template gpublasStatus_t geqrfBatched<gpuFloatComplex>(gpublasHandle_t, int, int,
                                                       gpuFloatComplex *const[], int,
                                                       gpuFloatComplex *const[], int *, int);
template gpublasStatus_t geqrfBatched<gpuDoubleComplex>(gpublasHandle_t, int, int,
                                                        gpuDoubleComplex *const[], int,
                                                        gpuDoubleComplex *const[], int *, int);

// Function: gelsBatched
template gpublasStatus_t gelsBatched<float>(gpublasHandle_t, gpublasOperation_t, int, int, int,
                                            float *const[], int, float *const[], int, int *, int *,
                                            int);
template gpublasStatus_t gelsBatched<double>(gpublasHandle_t, gpublasOperation_t, int, int, int,
                                             double *const[], int, double *const[], int, int *,
                                             int *, int);
template gpublasStatus_t gelsBatched<gpuFloatComplex>(gpublasHandle_t, gpublasOperation_t, int, int,
                                                      int, gpuFloatComplex *const[], int,
                                                      gpuFloatComplex *const[], int, int *, int *,
                                                      int);
template gpublasStatus_t gelsBatched<gpuDoubleComplex>(gpublasHandle_t, gpublasOperation_t, int,
                                                       int, int, gpuDoubleComplex *const[], int,
                                                       gpuDoubleComplex *const[], int, int *, int *,
                                                       int);

} // namespace wwr
