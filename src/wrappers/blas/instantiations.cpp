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
template wwrblasStatus_t iamax<float, int>(wwrblasHandle_t, int, const float *, int, int *);
template wwrblasStatus_t iamax<float, int64_t>(wwrblasHandle_t, int64_t, const float *, int64_t,
                                               int64_t *);
template wwrblasStatus_t iamax<double, int>(wwrblasHandle_t, int, const double *, int, int *);
template wwrblasStatus_t iamax<double, int64_t>(wwrblasHandle_t, int64_t, const double *, int64_t,
                                                int64_t *);
template wwrblasStatus_t iamax<wwrFloatComplex, int>(wwrblasHandle_t, int, const wwrFloatComplex *,
                                                     int, int *);
template wwrblasStatus_t iamax<wwrFloatComplex, int64_t>(wwrblasHandle_t, int64_t,
                                                         const wwrFloatComplex *, int64_t,
                                                         int64_t *);
template wwrblasStatus_t iamax<wwrDoubleComplex, int>(wwrblasHandle_t, int,
                                                      const wwrDoubleComplex *, int, int *);
template wwrblasStatus_t iamax<wwrDoubleComplex, int64_t>(wwrblasHandle_t, int64_t,
                                                          const wwrDoubleComplex *, int64_t,
                                                          int64_t *);

// Function: iamin
template wwrblasStatus_t iamin<float, int>(wwrblasHandle_t, int, const float *, int, int *);
template wwrblasStatus_t iamin<float, int64_t>(wwrblasHandle_t, int64_t, const float *, int64_t,
                                               int64_t *);
template wwrblasStatus_t iamin<double, int>(wwrblasHandle_t, int, const double *, int, int *);
template wwrblasStatus_t iamin<double, int64_t>(wwrblasHandle_t, int64_t, const double *, int64_t,
                                                int64_t *);
template wwrblasStatus_t iamin<wwrFloatComplex, int>(wwrblasHandle_t, int, const wwrFloatComplex *,
                                                     int, int *);
template wwrblasStatus_t iamin<wwrFloatComplex, int64_t>(wwrblasHandle_t, int64_t,
                                                         const wwrFloatComplex *, int64_t,
                                                         int64_t *);
template wwrblasStatus_t iamin<wwrDoubleComplex, int>(wwrblasHandle_t, int,
                                                      const wwrDoubleComplex *, int, int *);
template wwrblasStatus_t iamin<wwrDoubleComplex, int64_t>(wwrblasHandle_t, int64_t,
                                                          const wwrDoubleComplex *, int64_t,
                                                          int64_t *);

// Function: asum
template wwrblasStatus_t asum<float, int>(wwrblasHandle_t, int, const float *, int,
                                          ComplexToRealType<float> *);
template wwrblasStatus_t asum<float, int64_t>(wwrblasHandle_t, int64_t, const float *, int64_t,
                                              ComplexToRealType<float> *);
template wwrblasStatus_t asum<double, int>(wwrblasHandle_t, int, const double *, int,
                                           ComplexToRealType<double> *);
template wwrblasStatus_t asum<double, int64_t>(wwrblasHandle_t, int64_t, const double *, int64_t,
                                               ComplexToRealType<double> *);
template wwrblasStatus_t asum<wwrFloatComplex, int>(wwrblasHandle_t, int, const wwrFloatComplex *,
                                                    int, ComplexToRealType<wwrFloatComplex> *);
template wwrblasStatus_t asum<wwrFloatComplex, int64_t>(wwrblasHandle_t, int64_t,
                                                        const wwrFloatComplex *, int64_t,
                                                        ComplexToRealType<wwrFloatComplex> *);
template wwrblasStatus_t asum<wwrDoubleComplex, int>(wwrblasHandle_t, int, const wwrDoubleComplex *,
                                                     int, ComplexToRealType<wwrDoubleComplex> *);
template wwrblasStatus_t asum<wwrDoubleComplex, int64_t>(wwrblasHandle_t, int64_t,
                                                         const wwrDoubleComplex *, int64_t,
                                                         ComplexToRealType<wwrDoubleComplex> *);

// Function: axpy
template wwrblasStatus_t axpy<float, int>(wwrblasHandle_t, int, const float *, const float *, int,
                                          float *, int);
template wwrblasStatus_t axpy<float, int64_t>(wwrblasHandle_t, int64_t, const float *,
                                              const float *, int64_t, float *, int64_t);
template wwrblasStatus_t axpy<double, int>(wwrblasHandle_t, int, const double *, const double *,
                                           int, double *, int);
template wwrblasStatus_t axpy<double, int64_t>(wwrblasHandle_t, int64_t, const double *,
                                               const double *, int64_t, double *, int64_t);
template wwrblasStatus_t axpy<wwrFloatComplex, int>(wwrblasHandle_t, int, const wwrFloatComplex *,
                                                    const wwrFloatComplex *, int, wwrFloatComplex *,
                                                    int);
template wwrblasStatus_t axpy<wwrFloatComplex, int64_t>(wwrblasHandle_t, int64_t,
                                                        const wwrFloatComplex *,
                                                        const wwrFloatComplex *, int64_t,
                                                        wwrFloatComplex *, int64_t);
template wwrblasStatus_t axpy<wwrDoubleComplex, int>(wwrblasHandle_t, int, const wwrDoubleComplex *,
                                                     const wwrDoubleComplex *, int,
                                                     wwrDoubleComplex *, int);
template wwrblasStatus_t axpy<wwrDoubleComplex, int64_t>(wwrblasHandle_t, int64_t,
                                                         const wwrDoubleComplex *,
                                                         const wwrDoubleComplex *, int64_t,
                                                         wwrDoubleComplex *, int64_t);

// Function: copy
template wwrblasStatus_t copy<float, int>(wwrblasHandle_t, int, const float *, int, float *, int);
template wwrblasStatus_t copy<float, int64_t>(wwrblasHandle_t, int64_t, const float *, int64_t,
                                              float *, int64_t);
template wwrblasStatus_t copy<double, int>(wwrblasHandle_t, int, const double *, int, double *,
                                           int);
template wwrblasStatus_t copy<double, int64_t>(wwrblasHandle_t, int64_t, const double *, int64_t,
                                               double *, int64_t);
template wwrblasStatus_t copy<wwrFloatComplex, int>(wwrblasHandle_t, int, const wwrFloatComplex *,
                                                    int, wwrFloatComplex *, int);
template wwrblasStatus_t copy<wwrFloatComplex, int64_t>(wwrblasHandle_t, int64_t,
                                                        const wwrFloatComplex *, int64_t,
                                                        wwrFloatComplex *, int64_t);
template wwrblasStatus_t copy<wwrDoubleComplex, int>(wwrblasHandle_t, int, const wwrDoubleComplex *,
                                                     int, wwrDoubleComplex *, int);
template wwrblasStatus_t copy<wwrDoubleComplex, int64_t>(wwrblasHandle_t, int64_t,
                                                         const wwrDoubleComplex *, int64_t,
                                                         wwrDoubleComplex *, int64_t);

// Function: dot
template wwrblasStatus_t dot<float, int>(wwrblasHandle_t, int, const float *, int, const float *,
                                         int, float *);
template wwrblasStatus_t dot<float, int64_t>(wwrblasHandle_t, int64_t, const float *, int64_t,
                                             const float *, int64_t, float *);
template wwrblasStatus_t dot<double, int>(wwrblasHandle_t, int, const double *, int, const double *,
                                          int, double *);
template wwrblasStatus_t dot<double, int64_t>(wwrblasHandle_t, int64_t, const double *, int64_t,
                                              const double *, int64_t, double *);

// Function: dotc
template wwrblasStatus_t dotc<wwrFloatComplex, int>(wwrblasHandle_t, int, const wwrFloatComplex *,
                                                    int, const wwrFloatComplex *, int,
                                                    wwrFloatComplex *);
template wwrblasStatus_t dotc<wwrFloatComplex, int64_t>(wwrblasHandle_t, int64_t,
                                                        const wwrFloatComplex *, int64_t,
                                                        const wwrFloatComplex *, int64_t,
                                                        wwrFloatComplex *);
template wwrblasStatus_t dotc<wwrDoubleComplex, int>(wwrblasHandle_t, int, const wwrDoubleComplex *,
                                                     int, const wwrDoubleComplex *, int,
                                                     wwrDoubleComplex *);
template wwrblasStatus_t dotc<wwrDoubleComplex, int64_t>(wwrblasHandle_t, int64_t,
                                                         const wwrDoubleComplex *, int64_t,
                                                         const wwrDoubleComplex *, int64_t,
                                                         wwrDoubleComplex *);

// Function: dotu
template wwrblasStatus_t dotu<wwrFloatComplex, int>(wwrblasHandle_t, int, const wwrFloatComplex *,
                                                    int, const wwrFloatComplex *, int,
                                                    wwrFloatComplex *);
template wwrblasStatus_t dotu<wwrFloatComplex, int64_t>(wwrblasHandle_t, int64_t,
                                                        const wwrFloatComplex *, int64_t,
                                                        const wwrFloatComplex *, int64_t,
                                                        wwrFloatComplex *);
template wwrblasStatus_t dotu<wwrDoubleComplex, int>(wwrblasHandle_t, int, const wwrDoubleComplex *,
                                                     int, const wwrDoubleComplex *, int,
                                                     wwrDoubleComplex *);
template wwrblasStatus_t dotu<wwrDoubleComplex, int64_t>(wwrblasHandle_t, int64_t,
                                                         const wwrDoubleComplex *, int64_t,
                                                         const wwrDoubleComplex *, int64_t,
                                                         wwrDoubleComplex *);

// Function: nrm2
template wwrblasStatus_t nrm2<float, int>(wwrblasHandle_t, int, const float *, int,
                                          ComplexToRealType<float> *);
template wwrblasStatus_t nrm2<float, int64_t>(wwrblasHandle_t, int64_t, const float *, int64_t,
                                              ComplexToRealType<float> *);
template wwrblasStatus_t nrm2<double, int>(wwrblasHandle_t, int, const double *, int,
                                           ComplexToRealType<double> *);
template wwrblasStatus_t nrm2<double, int64_t>(wwrblasHandle_t, int64_t, const double *, int64_t,
                                               ComplexToRealType<double> *);
template wwrblasStatus_t nrm2<wwrFloatComplex, int>(wwrblasHandle_t, int, const wwrFloatComplex *,
                                                    int, ComplexToRealType<wwrFloatComplex> *);
template wwrblasStatus_t nrm2<wwrFloatComplex, int64_t>(wwrblasHandle_t, int64_t,
                                                        const wwrFloatComplex *, int64_t,
                                                        ComplexToRealType<wwrFloatComplex> *);
template wwrblasStatus_t nrm2<wwrDoubleComplex, int>(wwrblasHandle_t, int, const wwrDoubleComplex *,
                                                     int, ComplexToRealType<wwrDoubleComplex> *);
template wwrblasStatus_t nrm2<wwrDoubleComplex, int64_t>(wwrblasHandle_t, int64_t,
                                                         const wwrDoubleComplex *, int64_t,
                                                         ComplexToRealType<wwrDoubleComplex> *);

// Function: rot
template wwrblasStatus_t rot<float, int>(wwrblasHandle_t, int, float *, int, float *, int,
                                         const float *, const float *);
template wwrblasStatus_t rot<float, int64_t>(wwrblasHandle_t, int64_t, float *, int64_t, float *,
                                             int64_t, const float *, const float *);
template wwrblasStatus_t rot<double, int>(wwrblasHandle_t, int, double *, int, double *, int,
                                          const double *, const double *);
template wwrblasStatus_t rot<double, int64_t>(wwrblasHandle_t, int64_t, double *, int64_t, double *,
                                              int64_t, const double *, const double *);
template wwrblasStatus_t rot<wwrFloatComplex, int>(wwrblasHandle_t, int, wwrFloatComplex *, int,
                                                   wwrFloatComplex *, int,
                                                   const ComplexToRealType<wwrFloatComplex> *,
                                                   const wwrFloatComplex *);
template wwrblasStatus_t rot<wwrFloatComplex, int64_t>(wwrblasHandle_t, int64_t, wwrFloatComplex *,
                                                       int64_t, wwrFloatComplex *, int64_t,
                                                       const ComplexToRealType<wwrFloatComplex> *,
                                                       const wwrFloatComplex *);
template wwrblasStatus_t rot<wwrDoubleComplex, int>(wwrblasHandle_t, int, wwrDoubleComplex *, int,
                                                    wwrDoubleComplex *, int,
                                                    const ComplexToRealType<wwrDoubleComplex> *,
                                                    const wwrDoubleComplex *);
template wwrblasStatus_t rot<wwrDoubleComplex, int64_t>(wwrblasHandle_t, int64_t,
                                                        wwrDoubleComplex *, int64_t,
                                                        wwrDoubleComplex *, int64_t,
                                                        const ComplexToRealType<wwrDoubleComplex> *,
                                                        const wwrDoubleComplex *);
template wwrblasStatus_t rot<wwrFloatComplex, int>(wwrblasHandle_t, int, wwrFloatComplex *, int,
                                                   wwrFloatComplex *, int,
                                                   const ComplexToRealType<wwrFloatComplex> *,
                                                   const ComplexToRealType<wwrFloatComplex> *);
template wwrblasStatus_t rot<wwrFloatComplex, int64_t>(wwrblasHandle_t, int64_t, wwrFloatComplex *,
                                                       int64_t, wwrFloatComplex *, int64_t,
                                                       const ComplexToRealType<wwrFloatComplex> *,
                                                       const ComplexToRealType<wwrFloatComplex> *);
template wwrblasStatus_t rot<wwrDoubleComplex, int>(wwrblasHandle_t, int, wwrDoubleComplex *, int,
                                                    wwrDoubleComplex *, int,
                                                    const ComplexToRealType<wwrDoubleComplex> *,
                                                    const ComplexToRealType<wwrDoubleComplex> *);
template wwrblasStatus_t rot<wwrDoubleComplex, int64_t>(
    wwrblasHandle_t, int64_t, wwrDoubleComplex *, int64_t, wwrDoubleComplex *, int64_t,
    const ComplexToRealType<wwrDoubleComplex> *, const ComplexToRealType<wwrDoubleComplex> *);

// Function: rotg
template wwrblasStatus_t rotg<float, int>(wwrblasHandle_t, float *, float *, float *, float *);
template wwrblasStatus_t rotg<double, int>(wwrblasHandle_t, double *, double *, double *, double *);
template wwrblasStatus_t rotg<wwrFloatComplex, int>(wwrblasHandle_t, wwrFloatComplex *,
                                                    wwrFloatComplex *,
                                                    ComplexToRealType<wwrFloatComplex> *,
                                                    wwrFloatComplex *);
template wwrblasStatus_t rotg<wwrDoubleComplex, int>(wwrblasHandle_t, wwrDoubleComplex *,
                                                     wwrDoubleComplex *,
                                                     ComplexToRealType<wwrDoubleComplex> *,
                                                     wwrDoubleComplex *);

// Function: rotm
template wwrblasStatus_t rotm<float, int>(wwrblasHandle_t, int, float *, int, float *, int,
                                          const float *);
template wwrblasStatus_t rotm<float, int64_t>(wwrblasHandle_t, int64_t, float *, int64_t, float *,
                                              int64_t, const float *);
template wwrblasStatus_t rotm<double, int>(wwrblasHandle_t, int, double *, int, double *, int,
                                           const double *);
template wwrblasStatus_t rotm<double, int64_t>(wwrblasHandle_t, int64_t, double *, int64_t,
                                               double *, int64_t, const double *);

// Function: rotmg
template wwrblasStatus_t rotmg<float, int>(wwrblasHandle_t, float *, float *, float *,
                                           const float *, float *);
template wwrblasStatus_t rotmg<double, int>(wwrblasHandle_t, double *, double *, double *,
                                            const double *, double *);

// Function: scal
template wwrblasStatus_t scal<float, int>(wwrblasHandle_t, int, const float *, float *, int);
template wwrblasStatus_t scal<float, int64_t>(wwrblasHandle_t, int64_t, const float *, float *,
                                              int64_t);
template wwrblasStatus_t scal<double, int>(wwrblasHandle_t, int, const double *, double *, int);
template wwrblasStatus_t scal<double, int64_t>(wwrblasHandle_t, int64_t, const double *, double *,
                                               int64_t);
template wwrblasStatus_t scal<wwrFloatComplex, int>(wwrblasHandle_t, int, const wwrFloatComplex *,
                                                    wwrFloatComplex *, int);
template wwrblasStatus_t scal<wwrFloatComplex, int64_t>(wwrblasHandle_t, int64_t,
                                                        const wwrFloatComplex *, wwrFloatComplex *,
                                                        int64_t);
template wwrblasStatus_t scal<wwrDoubleComplex, int>(wwrblasHandle_t, int, const wwrDoubleComplex *,
                                                     wwrDoubleComplex *, int);
template wwrblasStatus_t scal<wwrDoubleComplex, int64_t>(wwrblasHandle_t, int64_t,
                                                         const wwrDoubleComplex *,
                                                         wwrDoubleComplex *, int64_t);
template wwrblasStatus_t scal<wwrFloatComplex, int>(wwrblasHandle_t, int,
                                                    const ComplexToRealType<wwrFloatComplex> *,
                                                    wwrFloatComplex *, int);
template wwrblasStatus_t scal<wwrFloatComplex, int64_t>(wwrblasHandle_t, int64_t,
                                                        const ComplexToRealType<wwrFloatComplex> *,
                                                        wwrFloatComplex *, int64_t);
template wwrblasStatus_t scal<wwrDoubleComplex, int>(wwrblasHandle_t, int,
                                                     const ComplexToRealType<wwrDoubleComplex> *,
                                                     wwrDoubleComplex *, int);
template wwrblasStatus_t
scal<wwrDoubleComplex, int64_t>(wwrblasHandle_t, int64_t,
                                const ComplexToRealType<wwrDoubleComplex> *, wwrDoubleComplex *,
                                int64_t);

// Function: swap
template wwrblasStatus_t swap<float, int>(wwrblasHandle_t, int, float *, int, float *, int);
template wwrblasStatus_t swap<float, int64_t>(wwrblasHandle_t, int64_t, float *, int64_t, float *,
                                              int64_t);
template wwrblasStatus_t swap<double, int>(wwrblasHandle_t, int, double *, int, double *, int);
template wwrblasStatus_t swap<double, int64_t>(wwrblasHandle_t, int64_t, double *, int64_t,
                                               double *, int64_t);
template wwrblasStatus_t swap<wwrFloatComplex, int>(wwrblasHandle_t, int, wwrFloatComplex *, int,
                                                    wwrFloatComplex *, int);
template wwrblasStatus_t swap<wwrFloatComplex, int64_t>(wwrblasHandle_t, int64_t, wwrFloatComplex *,
                                                        int64_t, wwrFloatComplex *, int64_t);
template wwrblasStatus_t swap<wwrDoubleComplex, int>(wwrblasHandle_t, int, wwrDoubleComplex *, int,
                                                     wwrDoubleComplex *, int);
template wwrblasStatus_t swap<wwrDoubleComplex, int64_t>(wwrblasHandle_t, int64_t,
                                                         wwrDoubleComplex *, int64_t,
                                                         wwrDoubleComplex *, int64_t);

// Function: gemv
template wwrblasStatus_t gemv<float, int>(wwrblasHandle_t, wwrblasOperation_t, int, int,
                                          const float *, const float *, int, const float *, int,
                                          const float *, float *, int);
template wwrblasStatus_t gemv<float, int64_t>(wwrblasHandle_t, wwrblasOperation_t, int64_t, int64_t,
                                              const float *, const float *, int64_t, const float *,
                                              int64_t, const float *, float *, int64_t);
template wwrblasStatus_t gemv<double, int>(wwrblasHandle_t, wwrblasOperation_t, int, int,
                                           const double *, const double *, int, const double *, int,
                                           const double *, double *, int);
template wwrblasStatus_t gemv<double, int64_t>(wwrblasHandle_t, wwrblasOperation_t, int64_t,
                                               int64_t, const double *, const double *, int64_t,
                                               const double *, int64_t, const double *, double *,
                                               int64_t);
template wwrblasStatus_t
gemv<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasOperation_t, int, int, const wwrFloatComplex *,
                           const wwrFloatComplex *, int, const wwrFloatComplex *, int,
                           const wwrFloatComplex *, wwrFloatComplex *, int);
template wwrblasStatus_t gemv<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasOperation_t,
                                                        int64_t, int64_t, const wwrFloatComplex *,
                                                        const wwrFloatComplex *, int64_t,
                                                        const wwrFloatComplex *, int64_t,
                                                        const wwrFloatComplex *, wwrFloatComplex *,
                                                        int64_t);
template wwrblasStatus_t
gemv<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasOperation_t, int, int, const wwrDoubleComplex *,
                            const wwrDoubleComplex *, int, const wwrDoubleComplex *, int,
                            const wwrDoubleComplex *, wwrDoubleComplex *, int);
template wwrblasStatus_t gemv<wwrDoubleComplex, int64_t>(wwrblasHandle_t, wwrblasOperation_t,
                                                         int64_t, int64_t, const wwrDoubleComplex *,
                                                         const wwrDoubleComplex *, int64_t,
                                                         const wwrDoubleComplex *, int64_t,
                                                         const wwrDoubleComplex *,
                                                         wwrDoubleComplex *, int64_t);

// Function: gbmv
template wwrblasStatus_t gbmv<float, int>(wwrblasHandle_t, wwrblasOperation_t, int, int, int, int,
                                          const float *, const float *, int, const float *, int,
                                          const float *, float *, int);
template wwrblasStatus_t gbmv<float, int64_t>(wwrblasHandle_t, wwrblasOperation_t, int64_t, int64_t,
                                              int64_t, int64_t, const float *, const float *,
                                              int64_t, const float *, int64_t, const float *,
                                              float *, int64_t);
template wwrblasStatus_t gbmv<double, int>(wwrblasHandle_t, wwrblasOperation_t, int, int, int, int,
                                           const double *, const double *, int, const double *, int,
                                           const double *, double *, int);
template wwrblasStatus_t gbmv<double, int64_t>(wwrblasHandle_t, wwrblasOperation_t, int64_t,
                                               int64_t, int64_t, int64_t, const double *,
                                               const double *, int64_t, const double *, int64_t,
                                               const double *, double *, int64_t);
template wwrblasStatus_t gbmv<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasOperation_t, int, int,
                                                    int, int, const wwrFloatComplex *,
                                                    const wwrFloatComplex *, int,
                                                    const wwrFloatComplex *, int,
                                                    const wwrFloatComplex *, wwrFloatComplex *,
                                                    int);
template wwrblasStatus_t
gbmv<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasOperation_t, int64_t, int64_t, int64_t,
                               int64_t, const wwrFloatComplex *, const wwrFloatComplex *, int64_t,
                               const wwrFloatComplex *, int64_t, const wwrFloatComplex *,
                               wwrFloatComplex *, int64_t);
template wwrblasStatus_t gbmv<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasOperation_t, int, int,
                                                     int, int, const wwrDoubleComplex *,
                                                     const wwrDoubleComplex *, int,
                                                     const wwrDoubleComplex *, int,
                                                     const wwrDoubleComplex *, wwrDoubleComplex *,
                                                     int);
template wwrblasStatus_t
gbmv<wwrDoubleComplex, int64_t>(wwrblasHandle_t, wwrblasOperation_t, int64_t, int64_t, int64_t,
                                int64_t, const wwrDoubleComplex *, const wwrDoubleComplex *,
                                int64_t, const wwrDoubleComplex *, int64_t,
                                const wwrDoubleComplex *, wwrDoubleComplex *, int64_t);

// Function: ger
template wwrblasStatus_t ger<float, int>(wwrblasHandle_t, int, int, const float *, const float *,
                                         int, const float *, int, float *, int);
template wwrblasStatus_t ger<float, int64_t>(wwrblasHandle_t, int64_t, int64_t, const float *,
                                             const float *, int64_t, const float *, int64_t,
                                             float *, int64_t);
template wwrblasStatus_t ger<double, int>(wwrblasHandle_t, int, int, const double *, const double *,
                                          int, const double *, int, double *, int);
template wwrblasStatus_t ger<double, int64_t>(wwrblasHandle_t, int64_t, int64_t, const double *,
                                              const double *, int64_t, const double *, int64_t,
                                              double *, int64_t);

// Function: geru
template wwrblasStatus_t geru<wwrFloatComplex, int>(wwrblasHandle_t, int, int,
                                                    const wwrFloatComplex *,
                                                    const wwrFloatComplex *, int,
                                                    const wwrFloatComplex *, int, wwrFloatComplex *,
                                                    int);
template wwrblasStatus_t geru<wwrFloatComplex, int64_t>(wwrblasHandle_t, int64_t, int64_t,
                                                        const wwrFloatComplex *,
                                                        const wwrFloatComplex *, int64_t,
                                                        const wwrFloatComplex *, int64_t,
                                                        wwrFloatComplex *, int64_t);
template wwrblasStatus_t geru<wwrDoubleComplex, int>(wwrblasHandle_t, int, int,
                                                     const wwrDoubleComplex *,
                                                     const wwrDoubleComplex *, int,
                                                     const wwrDoubleComplex *, int,
                                                     wwrDoubleComplex *, int);
template wwrblasStatus_t geru<wwrDoubleComplex, int64_t>(wwrblasHandle_t, int64_t, int64_t,
                                                         const wwrDoubleComplex *,
                                                         const wwrDoubleComplex *, int64_t,
                                                         const wwrDoubleComplex *, int64_t,
                                                         wwrDoubleComplex *, int64_t);

// Function: gerc
template wwrblasStatus_t gerc<wwrFloatComplex, int>(wwrblasHandle_t, int, int,
                                                    const wwrFloatComplex *,
                                                    const wwrFloatComplex *, int,
                                                    const wwrFloatComplex *, int, wwrFloatComplex *,
                                                    int);
template wwrblasStatus_t gerc<wwrFloatComplex, int64_t>(wwrblasHandle_t, int64_t, int64_t,
                                                        const wwrFloatComplex *,
                                                        const wwrFloatComplex *, int64_t,
                                                        const wwrFloatComplex *, int64_t,
                                                        wwrFloatComplex *, int64_t);
template wwrblasStatus_t gerc<wwrDoubleComplex, int>(wwrblasHandle_t, int, int,
                                                     const wwrDoubleComplex *,
                                                     const wwrDoubleComplex *, int,
                                                     const wwrDoubleComplex *, int,
                                                     wwrDoubleComplex *, int);
template wwrblasStatus_t gerc<wwrDoubleComplex, int64_t>(wwrblasHandle_t, int64_t, int64_t,
                                                         const wwrDoubleComplex *,
                                                         const wwrDoubleComplex *, int64_t,
                                                         const wwrDoubleComplex *, int64_t,
                                                         wwrDoubleComplex *, int64_t);

// Function: symv
template wwrblasStatus_t symv<float, int>(wwrblasHandle_t, wwrblasFillMode_t, int, const float *,
                                          const float *, int, const float *, int, const float *,
                                          float *, int);
template wwrblasStatus_t symv<float, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, int64_t,
                                              const float *, const float *, int64_t, const float *,
                                              int64_t, const float *, float *, int64_t);
template wwrblasStatus_t symv<double, int>(wwrblasHandle_t, wwrblasFillMode_t, int, const double *,
                                           const double *, int, const double *, int, const double *,
                                           double *, int);
template wwrblasStatus_t symv<double, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, int64_t,
                                               const double *, const double *, int64_t,
                                               const double *, int64_t, const double *, double *,
                                               int64_t);

// Function: syr
template wwrblasStatus_t syr<float, int>(wwrblasHandle_t, wwrblasFillMode_t, int, const float *,
                                         const float *, int, float *, int);
template wwrblasStatus_t syr<float, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, int64_t,
                                             const float *, const float *, int64_t, float *,
                                             int64_t);
template wwrblasStatus_t syr<double, int>(wwrblasHandle_t, wwrblasFillMode_t, int, const double *,
                                          const double *, int, double *, int);
template wwrblasStatus_t syr<double, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, int64_t,
                                              const double *, const double *, int64_t, double *,
                                              int64_t);

// Function: syr2
template wwrblasStatus_t syr2<float, int>(wwrblasHandle_t, wwrblasFillMode_t, int, const float *,
                                          const float *, int, const float *, int, float *, int);
template wwrblasStatus_t syr2<float, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, int64_t,
                                              const float *, const float *, int64_t, const float *,
                                              int64_t, float *, int64_t);
template wwrblasStatus_t syr2<double, int>(wwrblasHandle_t, wwrblasFillMode_t, int, const double *,
                                           const double *, int, const double *, int, double *, int);
template wwrblasStatus_t syr2<double, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, int64_t,
                                               const double *, const double *, int64_t,
                                               const double *, int64_t, double *, int64_t);

// Function: sbmv
template wwrblasStatus_t sbmv<float, int>(wwrblasHandle_t, wwrblasFillMode_t, int, int,
                                          const float *, const float *, int, const float *, int,
                                          const float *, float *, int);
template wwrblasStatus_t sbmv<float, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, int64_t, int64_t,
                                              const float *, const float *, int64_t, const float *,
                                              int64_t, const float *, float *, int64_t);
template wwrblasStatus_t sbmv<double, int>(wwrblasHandle_t, wwrblasFillMode_t, int, int,
                                           const double *, const double *, int, const double *, int,
                                           const double *, double *, int);
template wwrblasStatus_t sbmv<double, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, int64_t, int64_t,
                                               const double *, const double *, int64_t,
                                               const double *, int64_t, const double *, double *,
                                               int64_t);

// Function: spmv
template wwrblasStatus_t spmv<float, int>(wwrblasHandle_t, wwrblasFillMode_t, int, const float *,
                                          const float *, const float *, int, const float *, float *,
                                          int);
template wwrblasStatus_t spmv<float, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, int64_t,
                                              const float *, const float *, const float *, int64_t,
                                              const float *, float *, int64_t);
template wwrblasStatus_t spmv<double, int>(wwrblasHandle_t, wwrblasFillMode_t, int, const double *,
                                           const double *, const double *, int, const double *,
                                           double *, int);
template wwrblasStatus_t spmv<double, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, int64_t,
                                               const double *, const double *, const double *,
                                               int64_t, const double *, double *, int64_t);

// Function: spr
template wwrblasStatus_t spr<float, int>(wwrblasHandle_t, wwrblasFillMode_t, int, const float *,
                                         const float *, int, float *);
template wwrblasStatus_t spr<float, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, int64_t,
                                             const float *, const float *, int64_t, float *);
template wwrblasStatus_t spr<double, int>(wwrblasHandle_t, wwrblasFillMode_t, int, const double *,
                                          const double *, int, double *);
template wwrblasStatus_t spr<double, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, int64_t,
                                              const double *, const double *, int64_t, double *);

// Function: spr2
template wwrblasStatus_t spr2<float, int>(wwrblasHandle_t, wwrblasFillMode_t, int, const float *,
                                          const float *, int, const float *, int, float *);
template wwrblasStatus_t spr2<float, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, int64_t,
                                              const float *, const float *, int64_t, const float *,
                                              int64_t, float *);
template wwrblasStatus_t spr2<double, int>(wwrblasHandle_t, wwrblasFillMode_t, int, const double *,
                                           const double *, int, const double *, int, double *);
template wwrblasStatus_t spr2<double, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, int64_t,
                                               const double *, const double *, int64_t,
                                               const double *, int64_t, double *);

// Function: trmv
template wwrblasStatus_t trmv<float, int>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t,
                                          wwrblasDiagType_t, int, const float *, int, float *, int);
template wwrblasStatus_t trmv<float, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                              wwrblasOperation_t, wwrblasDiagType_t, int64_t,
                                              const float *, int64_t, float *, int64_t);
template wwrblasStatus_t trmv<double, int>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t,
                                           wwrblasDiagType_t, int, const double *, int, double *,
                                           int);
template wwrblasStatus_t trmv<double, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                               wwrblasOperation_t, wwrblasDiagType_t, int64_t,
                                               const double *, int64_t, double *, int64_t);
template wwrblasStatus_t trmv<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasFillMode_t,
                                                    wwrblasOperation_t, wwrblasDiagType_t, int,
                                                    const wwrFloatComplex *, int, wwrFloatComplex *,
                                                    int);
template wwrblasStatus_t trmv<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                        wwrblasOperation_t, wwrblasDiagType_t,
                                                        int64_t, const wwrFloatComplex *, int64_t,
                                                        wwrFloatComplex *, int64_t);
template wwrblasStatus_t trmv<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasFillMode_t,
                                                     wwrblasOperation_t, wwrblasDiagType_t, int,
                                                     const wwrDoubleComplex *, int,
                                                     wwrDoubleComplex *, int);
template wwrblasStatus_t trmv<wwrDoubleComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                         wwrblasOperation_t, wwrblasDiagType_t,
                                                         int64_t, const wwrDoubleComplex *, int64_t,
                                                         wwrDoubleComplex *, int64_t);

// Function: trsv
template wwrblasStatus_t trsv<float, int>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t,
                                          wwrblasDiagType_t, int, const float *, int, float *, int);
template wwrblasStatus_t trsv<float, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                              wwrblasOperation_t, wwrblasDiagType_t, int64_t,
                                              const float *, int64_t, float *, int64_t);
template wwrblasStatus_t trsv<double, int>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t,
                                           wwrblasDiagType_t, int, const double *, int, double *,
                                           int);
template wwrblasStatus_t trsv<double, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                               wwrblasOperation_t, wwrblasDiagType_t, int64_t,
                                               const double *, int64_t, double *, int64_t);
template wwrblasStatus_t trsv<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasFillMode_t,
                                                    wwrblasOperation_t, wwrblasDiagType_t, int,
                                                    const wwrFloatComplex *, int, wwrFloatComplex *,
                                                    int);
template wwrblasStatus_t trsv<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                        wwrblasOperation_t, wwrblasDiagType_t,
                                                        int64_t, const wwrFloatComplex *, int64_t,
                                                        wwrFloatComplex *, int64_t);
template wwrblasStatus_t trsv<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasFillMode_t,
                                                     wwrblasOperation_t, wwrblasDiagType_t, int,
                                                     const wwrDoubleComplex *, int,
                                                     wwrDoubleComplex *, int);
template wwrblasStatus_t trsv<wwrDoubleComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                         wwrblasOperation_t, wwrblasDiagType_t,
                                                         int64_t, const wwrDoubleComplex *, int64_t,
                                                         wwrDoubleComplex *, int64_t);

// Function: tbmv
template wwrblasStatus_t tbmv<float, int>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t,
                                          wwrblasDiagType_t, int, int, const float *, int, float *,
                                          int);
template wwrblasStatus_t tbmv<float, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                              wwrblasOperation_t, wwrblasDiagType_t, int64_t,
                                              int64_t, const float *, int64_t, float *, int64_t);
template wwrblasStatus_t tbmv<double, int>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t,
                                           wwrblasDiagType_t, int, int, const double *, int,
                                           double *, int);
template wwrblasStatus_t tbmv<double, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                               wwrblasOperation_t, wwrblasDiagType_t, int64_t,
                                               int64_t, const double *, int64_t, double *, int64_t);
template wwrblasStatus_t tbmv<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasFillMode_t,
                                                    wwrblasOperation_t, wwrblasDiagType_t, int, int,
                                                    const wwrFloatComplex *, int, wwrFloatComplex *,
                                                    int);
template wwrblasStatus_t tbmv<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                        wwrblasOperation_t, wwrblasDiagType_t,
                                                        int64_t, int64_t, const wwrFloatComplex *,
                                                        int64_t, wwrFloatComplex *, int64_t);
template wwrblasStatus_t tbmv<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasFillMode_t,
                                                     wwrblasOperation_t, wwrblasDiagType_t, int,
                                                     int, const wwrDoubleComplex *, int,
                                                     wwrDoubleComplex *, int);
template wwrblasStatus_t tbmv<wwrDoubleComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                         wwrblasOperation_t, wwrblasDiagType_t,
                                                         int64_t, int64_t, const wwrDoubleComplex *,
                                                         int64_t, wwrDoubleComplex *, int64_t);

// Function: tbsv
template wwrblasStatus_t tbsv<float, int>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t,
                                          wwrblasDiagType_t, int, int, const float *, int, float *,
                                          int);
template wwrblasStatus_t tbsv<float, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                              wwrblasOperation_t, wwrblasDiagType_t, int64_t,
                                              int64_t, const float *, int64_t, float *, int64_t);
template wwrblasStatus_t tbsv<double, int>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t,
                                           wwrblasDiagType_t, int, int, const double *, int,
                                           double *, int);
template wwrblasStatus_t tbsv<double, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                               wwrblasOperation_t, wwrblasDiagType_t, int64_t,
                                               int64_t, const double *, int64_t, double *, int64_t);
template wwrblasStatus_t tbsv<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasFillMode_t,
                                                    wwrblasOperation_t, wwrblasDiagType_t, int, int,
                                                    const wwrFloatComplex *, int, wwrFloatComplex *,
                                                    int);
template wwrblasStatus_t tbsv<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                        wwrblasOperation_t, wwrblasDiagType_t,
                                                        int64_t, int64_t, const wwrFloatComplex *,
                                                        int64_t, wwrFloatComplex *, int64_t);
template wwrblasStatus_t tbsv<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasFillMode_t,
                                                     wwrblasOperation_t, wwrblasDiagType_t, int,
                                                     int, const wwrDoubleComplex *, int,
                                                     wwrDoubleComplex *, int);
template wwrblasStatus_t tbsv<wwrDoubleComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                         wwrblasOperation_t, wwrblasDiagType_t,
                                                         int64_t, int64_t, const wwrDoubleComplex *,
                                                         int64_t, wwrDoubleComplex *, int64_t);

// Function: tpmv
template wwrblasStatus_t tpmv<float, int>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t,
                                          wwrblasDiagType_t, int, const float *, float *, int);
template wwrblasStatus_t tpmv<float, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                              wwrblasOperation_t, wwrblasDiagType_t, int64_t,
                                              const float *, float *, int64_t);
template wwrblasStatus_t tpmv<double, int>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t,
                                           wwrblasDiagType_t, int, const double *, double *, int);
template wwrblasStatus_t tpmv<double, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                               wwrblasOperation_t, wwrblasDiagType_t, int64_t,
                                               const double *, double *, int64_t);
template wwrblasStatus_t tpmv<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasFillMode_t,
                                                    wwrblasOperation_t, wwrblasDiagType_t, int,
                                                    const wwrFloatComplex *, wwrFloatComplex *,
                                                    int);
template wwrblasStatus_t tpmv<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                        wwrblasOperation_t, wwrblasDiagType_t,
                                                        int64_t, const wwrFloatComplex *,
                                                        wwrFloatComplex *, int64_t);
template wwrblasStatus_t tpmv<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasFillMode_t,
                                                     wwrblasOperation_t, wwrblasDiagType_t, int,
                                                     const wwrDoubleComplex *, wwrDoubleComplex *,
                                                     int);
template wwrblasStatus_t tpmv<wwrDoubleComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                         wwrblasOperation_t, wwrblasDiagType_t,
                                                         int64_t, const wwrDoubleComplex *,
                                                         wwrDoubleComplex *, int64_t);

// Function: tpsv
template wwrblasStatus_t tpsv<float, int>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t,
                                          wwrblasDiagType_t, int, const float *, float *, int);
template wwrblasStatus_t tpsv<float, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                              wwrblasOperation_t, wwrblasDiagType_t, int64_t,
                                              const float *, float *, int64_t);
template wwrblasStatus_t tpsv<double, int>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t,
                                           wwrblasDiagType_t, int, const double *, double *, int);
template wwrblasStatus_t tpsv<double, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                               wwrblasOperation_t, wwrblasDiagType_t, int64_t,
                                               const double *, double *, int64_t);
template wwrblasStatus_t tpsv<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasFillMode_t,
                                                    wwrblasOperation_t, wwrblasDiagType_t, int,
                                                    const wwrFloatComplex *, wwrFloatComplex *,
                                                    int);
template wwrblasStatus_t tpsv<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                        wwrblasOperation_t, wwrblasDiagType_t,
                                                        int64_t, const wwrFloatComplex *,
                                                        wwrFloatComplex *, int64_t);
template wwrblasStatus_t tpsv<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasFillMode_t,
                                                     wwrblasOperation_t, wwrblasDiagType_t, int,
                                                     const wwrDoubleComplex *, wwrDoubleComplex *,
                                                     int);
template wwrblasStatus_t tpsv<wwrDoubleComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                         wwrblasOperation_t, wwrblasDiagType_t,
                                                         int64_t, const wwrDoubleComplex *,
                                                         wwrDoubleComplex *, int64_t);

// Function: hemv
template wwrblasStatus_t
hemv<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasFillMode_t, int, const wwrFloatComplex *,
                           const wwrFloatComplex *, int, const wwrFloatComplex *, int,
                           const wwrFloatComplex *, wwrFloatComplex *, int);
template wwrblasStatus_t
hemv<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, int64_t, const wwrFloatComplex *,
                               const wwrFloatComplex *, int64_t, const wwrFloatComplex *, int64_t,
                               const wwrFloatComplex *, wwrFloatComplex *, int64_t);
template wwrblasStatus_t
hemv<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasFillMode_t, int, const wwrDoubleComplex *,
                            const wwrDoubleComplex *, int, const wwrDoubleComplex *, int,
                            const wwrDoubleComplex *, wwrDoubleComplex *, int);
template wwrblasStatus_t hemv<wwrDoubleComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                         int64_t, const wwrDoubleComplex *,
                                                         const wwrDoubleComplex *, int64_t,
                                                         const wwrDoubleComplex *, int64_t,
                                                         const wwrDoubleComplex *,
                                                         wwrDoubleComplex *, int64_t);

// Function: hbmv
template wwrblasStatus_t
hbmv<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasFillMode_t, int, int, const wwrFloatComplex *,
                           const wwrFloatComplex *, int, const wwrFloatComplex *, int,
                           const wwrFloatComplex *, wwrFloatComplex *, int);
template wwrblasStatus_t hbmv<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, int64_t,
                                                        int64_t, const wwrFloatComplex *,
                                                        const wwrFloatComplex *, int64_t,
                                                        const wwrFloatComplex *, int64_t,
                                                        const wwrFloatComplex *, wwrFloatComplex *,
                                                        int64_t);
template wwrblasStatus_t
hbmv<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasFillMode_t, int, int, const wwrDoubleComplex *,
                            const wwrDoubleComplex *, int, const wwrDoubleComplex *, int,
                            const wwrDoubleComplex *, wwrDoubleComplex *, int);
template wwrblasStatus_t hbmv<wwrDoubleComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                         int64_t, int64_t, const wwrDoubleComplex *,
                                                         const wwrDoubleComplex *, int64_t,
                                                         const wwrDoubleComplex *, int64_t,
                                                         const wwrDoubleComplex *,
                                                         wwrDoubleComplex *, int64_t);

// Function: hpmv
template wwrblasStatus_t
hpmv<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasFillMode_t, int, const wwrFloatComplex *,
                           const wwrFloatComplex *, const wwrFloatComplex *, int,
                           const wwrFloatComplex *, wwrFloatComplex *, int);
template wwrblasStatus_t
hpmv<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, int64_t, const wwrFloatComplex *,
                               const wwrFloatComplex *, const wwrFloatComplex *, int64_t,
                               const wwrFloatComplex *, wwrFloatComplex *, int64_t);
template wwrblasStatus_t
hpmv<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasFillMode_t, int, const wwrDoubleComplex *,
                            const wwrDoubleComplex *, const wwrDoubleComplex *, int,
                            const wwrDoubleComplex *, wwrDoubleComplex *, int);
template wwrblasStatus_t hpmv<wwrDoubleComplex, int64_t>(
    wwrblasHandle_t, wwrblasFillMode_t, int64_t, const wwrDoubleComplex *, const wwrDoubleComplex *,
    const wwrDoubleComplex *, int64_t, const wwrDoubleComplex *, wwrDoubleComplex *, int64_t);

// Function: her
template wwrblasStatus_t her<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasFillMode_t, int,
                                                   const ComplexToRealType<wwrFloatComplex> *,
                                                   const wwrFloatComplex *, int, wwrFloatComplex *,
                                                   int);
template wwrblasStatus_t her<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, int64_t,
                                                       const ComplexToRealType<wwrFloatComplex> *,
                                                       const wwrFloatComplex *, int64_t,
                                                       wwrFloatComplex *, int64_t);
template wwrblasStatus_t her<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasFillMode_t, int,
                                                    const ComplexToRealType<wwrDoubleComplex> *,
                                                    const wwrDoubleComplex *, int,
                                                    wwrDoubleComplex *, int);
template wwrblasStatus_t her<wwrDoubleComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, int64_t,
                                                        const ComplexToRealType<wwrDoubleComplex> *,
                                                        const wwrDoubleComplex *, int64_t,
                                                        wwrDoubleComplex *, int64_t);

// Function: her2
template wwrblasStatus_t her2<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasFillMode_t, int,
                                                    const wwrFloatComplex *,
                                                    const wwrFloatComplex *, int,
                                                    const wwrFloatComplex *, int, wwrFloatComplex *,
                                                    int);
template wwrblasStatus_t her2<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, int64_t,
                                                        const wwrFloatComplex *,
                                                        const wwrFloatComplex *, int64_t,
                                                        const wwrFloatComplex *, int64_t,
                                                        wwrFloatComplex *, int64_t);
template wwrblasStatus_t her2<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasFillMode_t, int,
                                                     const wwrDoubleComplex *,
                                                     const wwrDoubleComplex *, int,
                                                     const wwrDoubleComplex *, int,
                                                     wwrDoubleComplex *, int);
template wwrblasStatus_t her2<wwrDoubleComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                         int64_t, const wwrDoubleComplex *,
                                                         const wwrDoubleComplex *, int64_t,
                                                         const wwrDoubleComplex *, int64_t,
                                                         wwrDoubleComplex *, int64_t);

// Function: hpr
template wwrblasStatus_t hpr<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasFillMode_t, int,
                                                   const ComplexToRealType<wwrFloatComplex> *,
                                                   const wwrFloatComplex *, int, wwrFloatComplex *);
template wwrblasStatus_t hpr<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, int64_t,
                                                       const ComplexToRealType<wwrFloatComplex> *,
                                                       const wwrFloatComplex *, int64_t,
                                                       wwrFloatComplex *);
template wwrblasStatus_t hpr<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasFillMode_t, int,
                                                    const ComplexToRealType<wwrDoubleComplex> *,
                                                    const wwrDoubleComplex *, int,
                                                    wwrDoubleComplex *);
template wwrblasStatus_t hpr<wwrDoubleComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, int64_t,
                                                        const ComplexToRealType<wwrDoubleComplex> *,
                                                        const wwrDoubleComplex *, int64_t,
                                                        wwrDoubleComplex *);

// Function: hpr2
template wwrblasStatus_t hpr2<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasFillMode_t, int,
                                                    const wwrFloatComplex *,
                                                    const wwrFloatComplex *, int,
                                                    const wwrFloatComplex *, int,
                                                    wwrFloatComplex *);
template wwrblasStatus_t hpr2<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, int64_t,
                                                        const wwrFloatComplex *,
                                                        const wwrFloatComplex *, int64_t,
                                                        const wwrFloatComplex *, int64_t,
                                                        wwrFloatComplex *);
template wwrblasStatus_t hpr2<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasFillMode_t, int,
                                                     const wwrDoubleComplex *,
                                                     const wwrDoubleComplex *, int,
                                                     const wwrDoubleComplex *, int,
                                                     wwrDoubleComplex *);
template wwrblasStatus_t hpr2<wwrDoubleComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                         int64_t, const wwrDoubleComplex *,
                                                         const wwrDoubleComplex *, int64_t,
                                                         const wwrDoubleComplex *, int64_t,
                                                         wwrDoubleComplex *);

// Function: gemvBatched
template wwrblasStatus_t gemvBatched<float, int>(wwrblasHandle_t, wwrblasOperation_t, int, int,
                                                 const float *, const float *const[], int,
                                                 const float *const[], int, const float *,
                                                 float *const[], int, int);
template wwrblasStatus_t gemvBatched<float, int64_t>(wwrblasHandle_t, wwrblasOperation_t, int64_t,
                                                     int64_t, const float *, const float *const[],
                                                     int64_t, const float *const[], int64_t,
                                                     const float *, float *const[], int64_t,
                                                     int64_t);
template wwrblasStatus_t gemvBatched<double, int>(wwrblasHandle_t, wwrblasOperation_t, int, int,
                                                  const double *, const double *const[], int,
                                                  const double *const[], int, const double *,
                                                  double *const[], int, int);
template wwrblasStatus_t
gemvBatched<double, int64_t>(wwrblasHandle_t, wwrblasOperation_t, int64_t, int64_t, const double *,
                             const double *const[], int64_t, const double *const[], int64_t,
                             const double *, double *const[], int64_t, int64_t);
template wwrblasStatus_t gemvBatched<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasOperation_t, int,
                                                           int, const wwrFloatComplex *,
                                                           const wwrFloatComplex *const[], int,
                                                           const wwrFloatComplex *const[], int,
                                                           const wwrFloatComplex *,
                                                           wwrFloatComplex *const[], int, int);
template wwrblasStatus_t gemvBatched<wwrFloatComplex, int64_t>(
    wwrblasHandle_t, wwrblasOperation_t, int64_t, int64_t, const wwrFloatComplex *,
    const wwrFloatComplex *const[], int64_t, const wwrFloatComplex *const[], int64_t,
    const wwrFloatComplex *, wwrFloatComplex *const[], int64_t, int64_t);
template wwrblasStatus_t gemvBatched<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasOperation_t,
                                                            int, int, const wwrDoubleComplex *,
                                                            const wwrDoubleComplex *const[], int,
                                                            const wwrDoubleComplex *const[], int,
                                                            const wwrDoubleComplex *,
                                                            wwrDoubleComplex *const[], int, int);
template wwrblasStatus_t gemvBatched<wwrDoubleComplex, int64_t>(
    wwrblasHandle_t, wwrblasOperation_t, int64_t, int64_t, const wwrDoubleComplex *,
    const wwrDoubleComplex *const[], int64_t, const wwrDoubleComplex *const[], int64_t,
    const wwrDoubleComplex *, wwrDoubleComplex *const[], int64_t, int64_t);

// Function: gemvStridedBatched
template wwrblasStatus_t gemvStridedBatched<float, int>(wwrblasHandle_t, wwrblasOperation_t, int,
                                                        int, const float *, const float *, int,
                                                        long long int, const float *, int,
                                                        long long int, const float *, float *, int,
                                                        long long int, int);
template wwrblasStatus_t gemvStridedBatched<float, int64_t>(wwrblasHandle_t, wwrblasOperation_t,
                                                            int64_t, int64_t, const float *,
                                                            const float *, int64_t, long long int,
                                                            const float *, int64_t, long long int,
                                                            const float *, float *, int64_t,
                                                            long long int, int64_t);
template wwrblasStatus_t gemvStridedBatched<double, int>(wwrblasHandle_t, wwrblasOperation_t, int,
                                                         int, const double *, const double *, int,
                                                         long long int, const double *, int,
                                                         long long int, const double *, double *,
                                                         int, long long int, int);
template wwrblasStatus_t gemvStridedBatched<double, int64_t>(wwrblasHandle_t, wwrblasOperation_t,
                                                             int64_t, int64_t, const double *,
                                                             const double *, int64_t, long long int,
                                                             const double *, int64_t, long long int,
                                                             const double *, double *, int64_t,
                                                             long long int, int64_t);
template wwrblasStatus_t gemvStridedBatched<wwrFloatComplex, int>(
    wwrblasHandle_t, wwrblasOperation_t, int, int, const wwrFloatComplex *, const wwrFloatComplex *,
    int, long long int, const wwrFloatComplex *, int, long long int, const wwrFloatComplex *,
    wwrFloatComplex *, int, long long int, int);
template wwrblasStatus_t gemvStridedBatched<wwrFloatComplex, int64_t>(
    wwrblasHandle_t, wwrblasOperation_t, int64_t, int64_t, const wwrFloatComplex *,
    const wwrFloatComplex *, int64_t, long long int, const wwrFloatComplex *, int64_t,
    long long int, const wwrFloatComplex *, wwrFloatComplex *, int64_t, long long int, int64_t);
template wwrblasStatus_t gemvStridedBatched<wwrDoubleComplex, int>(
    wwrblasHandle_t, wwrblasOperation_t, int, int, const wwrDoubleComplex *,
    const wwrDoubleComplex *, int, long long int, const wwrDoubleComplex *, int, long long int,
    const wwrDoubleComplex *, wwrDoubleComplex *, int, long long int, int);
template wwrblasStatus_t gemvStridedBatched<wwrDoubleComplex, int64_t>(
    wwrblasHandle_t, wwrblasOperation_t, int64_t, int64_t, const wwrDoubleComplex *,
    const wwrDoubleComplex *, int64_t, long long int, const wwrDoubleComplex *, int64_t,
    long long int, const wwrDoubleComplex *, wwrDoubleComplex *, int64_t, long long int, int64_t);

// Function: gemm
template wwrblasStatus_t gemm<float, int>(wwrblasHandle_t, wwrblasOperation_t, wwrblasOperation_t,
                                          int, int, int, const float *, const float *, int,
                                          const float *, int, const float *, float *, int);
template wwrblasStatus_t gemm<float, int64_t>(wwrblasHandle_t, wwrblasOperation_t,
                                              wwrblasOperation_t, int64_t, int64_t, int64_t,
                                              const float *, const float *, int64_t, const float *,
                                              int64_t, const float *, float *, int64_t);
template wwrblasStatus_t gemm<double, int>(wwrblasHandle_t, wwrblasOperation_t, wwrblasOperation_t,
                                           int, int, int, const double *, const double *, int,
                                           const double *, int, const double *, double *, int);
template wwrblasStatus_t gemm<double, int64_t>(wwrblasHandle_t, wwrblasOperation_t,
                                               wwrblasOperation_t, int64_t, int64_t, int64_t,
                                               const double *, const double *, int64_t,
                                               const double *, int64_t, const double *, double *,
                                               int64_t);
template wwrblasStatus_t
gemm<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasOperation_t, wwrblasOperation_t, int, int, int,
                           const wwrFloatComplex *, const wwrFloatComplex *, int,
                           const wwrFloatComplex *, int, const wwrFloatComplex *, wwrFloatComplex *,
                           int);
template wwrblasStatus_t
gemm<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasOperation_t, wwrblasOperation_t, int64_t,
                               int64_t, int64_t, const wwrFloatComplex *, const wwrFloatComplex *,
                               int64_t, const wwrFloatComplex *, int64_t, const wwrFloatComplex *,
                               wwrFloatComplex *, int64_t);
template wwrblasStatus_t
gemm<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasOperation_t, wwrblasOperation_t, int, int, int,
                            const wwrDoubleComplex *, const wwrDoubleComplex *, int,
                            const wwrDoubleComplex *, int, const wwrDoubleComplex *,
                            wwrDoubleComplex *, int);
template wwrblasStatus_t
gemm<wwrDoubleComplex, int64_t>(wwrblasHandle_t, wwrblasOperation_t, wwrblasOperation_t, int64_t,
                                int64_t, int64_t, const wwrDoubleComplex *,
                                const wwrDoubleComplex *, int64_t, const wwrDoubleComplex *,
                                int64_t, const wwrDoubleComplex *, wwrDoubleComplex *, int64_t);

// Function: gemmBatched
template wwrblasStatus_t gemmBatched<float, int>(wwrblasHandle_t, wwrblasOperation_t,
                                                 wwrblasOperation_t, int, int, int, const float *,
                                                 const float *const[], int, const float *const[],
                                                 int, const float *, float *const[], int, int);
template wwrblasStatus_t gemmBatched<float, int64_t>(wwrblasHandle_t, wwrblasOperation_t,
                                                     wwrblasOperation_t, int64_t, int64_t, int64_t,
                                                     const float *, const float *const[], int64_t,
                                                     const float *const[], int64_t, const float *,
                                                     float *const[], int64_t, int64_t);
template wwrblasStatus_t gemmBatched<double, int>(wwrblasHandle_t, wwrblasOperation_t,
                                                  wwrblasOperation_t, int, int, int, const double *,
                                                  const double *const[], int, const double *const[],
                                                  int, const double *, double *const[], int, int);
template wwrblasStatus_t gemmBatched<double, int64_t>(wwrblasHandle_t, wwrblasOperation_t,
                                                      wwrblasOperation_t, int64_t, int64_t, int64_t,
                                                      const double *, const double *const[],
                                                      int64_t, const double *const[], int64_t,
                                                      const double *, double *const[], int64_t,
                                                      int64_t);
template wwrblasStatus_t
gemmBatched<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasOperation_t, wwrblasOperation_t, int, int,
                                  int, const wwrFloatComplex *, const wwrFloatComplex *const[], int,
                                  const wwrFloatComplex *const[], int, const wwrFloatComplex *,
                                  wwrFloatComplex *const[], int, int);
template wwrblasStatus_t gemmBatched<wwrFloatComplex, int64_t>(
    wwrblasHandle_t, wwrblasOperation_t, wwrblasOperation_t, int64_t, int64_t, int64_t,
    const wwrFloatComplex *, const wwrFloatComplex *const[], int64_t,
    const wwrFloatComplex *const[], int64_t, const wwrFloatComplex *, wwrFloatComplex *const[],
    int64_t, int64_t);
template wwrblasStatus_t gemmBatched<wwrDoubleComplex, int>(
    wwrblasHandle_t, wwrblasOperation_t, wwrblasOperation_t, int, int, int,
    const wwrDoubleComplex *, const wwrDoubleComplex *const[], int, const wwrDoubleComplex *const[],
    int, const wwrDoubleComplex *, wwrDoubleComplex *const[], int, int);
template wwrblasStatus_t gemmBatched<wwrDoubleComplex, int64_t>(
    wwrblasHandle_t, wwrblasOperation_t, wwrblasOperation_t, int64_t, int64_t, int64_t,
    const wwrDoubleComplex *, const wwrDoubleComplex *const[], int64_t,
    const wwrDoubleComplex *const[], int64_t, const wwrDoubleComplex *, wwrDoubleComplex *const[],
    int64_t, int64_t);

// Function: gemmStridedBatched
template wwrblasStatus_t
gemmStridedBatched<float, int>(wwrblasHandle_t, wwrblasOperation_t, wwrblasOperation_t, int, int,
                               int, const float *, const float *, int, long long int, const float *,
                               int, long long int, const float *, float *, int, long long int, int);
template wwrblasStatus_t
gemmStridedBatched<float, int64_t>(wwrblasHandle_t, wwrblasOperation_t, wwrblasOperation_t, int64_t,
                                   int64_t, int64_t, const float *, const float *, int64_t,
                                   long long int, const float *, int64_t, long long int,
                                   const float *, float *, int64_t, long long int, int64_t);
template wwrblasStatus_t gemmStridedBatched<double, int>(wwrblasHandle_t, wwrblasOperation_t,
                                                         wwrblasOperation_t, int, int, int,
                                                         const double *, const double *, int,
                                                         long long int, const double *, int,
                                                         long long int, const double *, double *,
                                                         int, long long int, int);
template wwrblasStatus_t
gemmStridedBatched<double, int64_t>(wwrblasHandle_t, wwrblasOperation_t, wwrblasOperation_t,
                                    int64_t, int64_t, int64_t, const double *, const double *,
                                    int64_t, long long int, const double *, int64_t, long long int,
                                    const double *, double *, int64_t, long long int, int64_t);
template wwrblasStatus_t gemmStridedBatched<wwrFloatComplex, int>(
    wwrblasHandle_t, wwrblasOperation_t, wwrblasOperation_t, int, int, int, const wwrFloatComplex *,
    const wwrFloatComplex *, int, long long int, const wwrFloatComplex *, int, long long int,
    const wwrFloatComplex *, wwrFloatComplex *, int, long long int, int);
template wwrblasStatus_t gemmStridedBatched<wwrFloatComplex, int64_t>(
    wwrblasHandle_t, wwrblasOperation_t, wwrblasOperation_t, int64_t, int64_t, int64_t,
    const wwrFloatComplex *, const wwrFloatComplex *, int64_t, long long int,
    const wwrFloatComplex *, int64_t, long long int, const wwrFloatComplex *, wwrFloatComplex *,
    int64_t, long long int, int64_t);
template wwrblasStatus_t gemmStridedBatched<wwrDoubleComplex, int>(
    wwrblasHandle_t, wwrblasOperation_t, wwrblasOperation_t, int, int, int,
    const wwrDoubleComplex *, const wwrDoubleComplex *, int, long long int,
    const wwrDoubleComplex *, int, long long int, const wwrDoubleComplex *, wwrDoubleComplex *, int,
    long long int, int);
template wwrblasStatus_t gemmStridedBatched<wwrDoubleComplex, int64_t>(
    wwrblasHandle_t, wwrblasOperation_t, wwrblasOperation_t, int64_t, int64_t, int64_t,
    const wwrDoubleComplex *, const wwrDoubleComplex *, int64_t, long long int,
    const wwrDoubleComplex *, int64_t, long long int, const wwrDoubleComplex *, wwrDoubleComplex *,
    int64_t, long long int, int64_t);

// Function: symm
template wwrblasStatus_t symm<float, int>(wwrblasHandle_t, wwrblasSideMode_t, wwrblasFillMode_t,
                                          int, int, const float *, const float *, int,
                                          const float *, int, const float *, float *, int);
template wwrblasStatus_t symm<float, int64_t>(wwrblasHandle_t, wwrblasSideMode_t, wwrblasFillMode_t,
                                              int64_t, int64_t, const float *, const float *,
                                              int64_t, const float *, int64_t, const float *,
                                              float *, int64_t);
template wwrblasStatus_t symm<double, int>(wwrblasHandle_t, wwrblasSideMode_t, wwrblasFillMode_t,
                                           int, int, const double *, const double *, int,
                                           const double *, int, const double *, double *, int);
template wwrblasStatus_t symm<double, int64_t>(wwrblasHandle_t, wwrblasSideMode_t,
                                               wwrblasFillMode_t, int64_t, int64_t, const double *,
                                               const double *, int64_t, const double *, int64_t,
                                               const double *, double *, int64_t);
template wwrblasStatus_t
symm<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasSideMode_t, wwrblasFillMode_t, int, int,
                           const wwrFloatComplex *, const wwrFloatComplex *, int,
                           const wwrFloatComplex *, int, const wwrFloatComplex *, wwrFloatComplex *,
                           int);
template wwrblasStatus_t
symm<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasSideMode_t, wwrblasFillMode_t, int64_t,
                               int64_t, const wwrFloatComplex *, const wwrFloatComplex *, int64_t,
                               const wwrFloatComplex *, int64_t, const wwrFloatComplex *,
                               wwrFloatComplex *, int64_t);
template wwrblasStatus_t
symm<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasSideMode_t, wwrblasFillMode_t, int, int,
                            const wwrDoubleComplex *, const wwrDoubleComplex *, int,
                            const wwrDoubleComplex *, int, const wwrDoubleComplex *,
                            wwrDoubleComplex *, int);
template wwrblasStatus_t
symm<wwrDoubleComplex, int64_t>(wwrblasHandle_t, wwrblasSideMode_t, wwrblasFillMode_t, int64_t,
                                int64_t, const wwrDoubleComplex *, const wwrDoubleComplex *,
                                int64_t, const wwrDoubleComplex *, int64_t,
                                const wwrDoubleComplex *, wwrDoubleComplex *, int64_t);

// Function: syrk
template wwrblasStatus_t syrk<float, int>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t,
                                          int, int, const float *, const float *, int,
                                          const float *, float *, int);
template wwrblasStatus_t syrk<float, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                              wwrblasOperation_t, int64_t, int64_t, const float *,
                                              const float *, int64_t, const float *, float *,
                                              int64_t);
template wwrblasStatus_t syrk<double, int>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t,
                                           int, int, const double *, const double *, int,
                                           const double *, double *, int);
template wwrblasStatus_t syrk<double, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                               wwrblasOperation_t, int64_t, int64_t, const double *,
                                               const double *, int64_t, const double *, double *,
                                               int64_t);
template wwrblasStatus_t
syrk<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t, int, int,
                           const wwrFloatComplex *, const wwrFloatComplex *, int,
                           const wwrFloatComplex *, wwrFloatComplex *, int);
template wwrblasStatus_t
syrk<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t, int64_t,
                               int64_t, const wwrFloatComplex *, const wwrFloatComplex *, int64_t,
                               const wwrFloatComplex *, wwrFloatComplex *, int64_t);
template wwrblasStatus_t
syrk<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t, int, int,
                            const wwrDoubleComplex *, const wwrDoubleComplex *, int,
                            const wwrDoubleComplex *, wwrDoubleComplex *, int);
template wwrblasStatus_t
syrk<wwrDoubleComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t, int64_t,
                                int64_t, const wwrDoubleComplex *, const wwrDoubleComplex *,
                                int64_t, const wwrDoubleComplex *, wwrDoubleComplex *, int64_t);

// Function: syr2k
template wwrblasStatus_t syr2k<float, int>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t,
                                           int, int, const float *, const float *, int,
                                           const float *, int, const float *, float *, int);
template wwrblasStatus_t syr2k<float, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                               wwrblasOperation_t, int64_t, int64_t, const float *,
                                               const float *, int64_t, const float *, int64_t,
                                               const float *, float *, int64_t);
template wwrblasStatus_t syr2k<double, int>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t,
                                            int, int, const double *, const double *, int,
                                            const double *, int, const double *, double *, int);
template wwrblasStatus_t syr2k<double, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                wwrblasOperation_t, int64_t, int64_t,
                                                const double *, const double *, int64_t,
                                                const double *, int64_t, const double *, double *,
                                                int64_t);
template wwrblasStatus_t
syr2k<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t, int, int,
                            const wwrFloatComplex *, const wwrFloatComplex *, int,
                            const wwrFloatComplex *, int, const wwrFloatComplex *,
                            wwrFloatComplex *, int);
template wwrblasStatus_t
syr2k<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t, int64_t,
                                int64_t, const wwrFloatComplex *, const wwrFloatComplex *, int64_t,
                                const wwrFloatComplex *, int64_t, const wwrFloatComplex *,
                                wwrFloatComplex *, int64_t);
template wwrblasStatus_t
syr2k<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t, int, int,
                             const wwrDoubleComplex *, const wwrDoubleComplex *, int,
                             const wwrDoubleComplex *, int, const wwrDoubleComplex *,
                             wwrDoubleComplex *, int);
template wwrblasStatus_t
syr2k<wwrDoubleComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t, int64_t,
                                 int64_t, const wwrDoubleComplex *, const wwrDoubleComplex *,
                                 int64_t, const wwrDoubleComplex *, int64_t,
                                 const wwrDoubleComplex *, wwrDoubleComplex *, int64_t);

// Function: syrkx
template wwrblasStatus_t syrkx<float, int>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t,
                                           int, int, const float *, const float *, int,
                                           const float *, int, const float *, float *, int);
template wwrblasStatus_t syrkx<float, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                               wwrblasOperation_t, int64_t, int64_t, const float *,
                                               const float *, int64_t, const float *, int64_t,
                                               const float *, float *, int64_t);
template wwrblasStatus_t syrkx<double, int>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t,
                                            int, int, const double *, const double *, int,
                                            const double *, int, const double *, double *, int);
template wwrblasStatus_t syrkx<double, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                wwrblasOperation_t, int64_t, int64_t,
                                                const double *, const double *, int64_t,
                                                const double *, int64_t, const double *, double *,
                                                int64_t);
template wwrblasStatus_t
syrkx<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t, int, int,
                            const wwrFloatComplex *, const wwrFloatComplex *, int,
                            const wwrFloatComplex *, int, const wwrFloatComplex *,
                            wwrFloatComplex *, int);
template wwrblasStatus_t
syrkx<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t, int64_t,
                                int64_t, const wwrFloatComplex *, const wwrFloatComplex *, int64_t,
                                const wwrFloatComplex *, int64_t, const wwrFloatComplex *,
                                wwrFloatComplex *, int64_t);
template wwrblasStatus_t
syrkx<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t, int, int,
                             const wwrDoubleComplex *, const wwrDoubleComplex *, int,
                             const wwrDoubleComplex *, int, const wwrDoubleComplex *,
                             wwrDoubleComplex *, int);
template wwrblasStatus_t
syrkx<wwrDoubleComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t, int64_t,
                                 int64_t, const wwrDoubleComplex *, const wwrDoubleComplex *,
                                 int64_t, const wwrDoubleComplex *, int64_t,
                                 const wwrDoubleComplex *, wwrDoubleComplex *, int64_t);

// Function: trmm
template wwrblasStatus_t trmm<float, int>(wwrblasHandle_t, wwrblasSideMode_t, wwrblasFillMode_t,
                                          wwrblasOperation_t, wwrblasDiagType_t, int, int,
                                          const float *, const float *, int, const float *, int,
                                          float *, int);
template wwrblasStatus_t trmm<float, int64_t>(wwrblasHandle_t, wwrblasSideMode_t, wwrblasFillMode_t,
                                              wwrblasOperation_t, wwrblasDiagType_t, int64_t,
                                              int64_t, const float *, const float *, int64_t,
                                              const float *, int64_t, float *, int64_t);
template wwrblasStatus_t trmm<double, int>(wwrblasHandle_t, wwrblasSideMode_t, wwrblasFillMode_t,
                                           wwrblasOperation_t, wwrblasDiagType_t, int, int,
                                           const double *, const double *, int, const double *, int,
                                           double *, int);
template wwrblasStatus_t trmm<double, int64_t>(wwrblasHandle_t, wwrblasSideMode_t,
                                               wwrblasFillMode_t, wwrblasOperation_t,
                                               wwrblasDiagType_t, int64_t, int64_t, const double *,
                                               const double *, int64_t, const double *, int64_t,
                                               double *, int64_t);
template wwrblasStatus_t
trmm<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasSideMode_t, wwrblasFillMode_t,
                           wwrblasOperation_t, wwrblasDiagType_t, int, int, const wwrFloatComplex *,
                           const wwrFloatComplex *, int, const wwrFloatComplex *, int,
                           wwrFloatComplex *, int);
template wwrblasStatus_t
trmm<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasSideMode_t, wwrblasFillMode_t,
                               wwrblasOperation_t, wwrblasDiagType_t, int64_t, int64_t,
                               const wwrFloatComplex *, const wwrFloatComplex *, int64_t,
                               const wwrFloatComplex *, int64_t, wwrFloatComplex *, int64_t);
template wwrblasStatus_t
trmm<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasSideMode_t, wwrblasFillMode_t,
                            wwrblasOperation_t, wwrblasDiagType_t, int, int,
                            const wwrDoubleComplex *, const wwrDoubleComplex *, int,
                            const wwrDoubleComplex *, int, wwrDoubleComplex *, int);
template wwrblasStatus_t
trmm<wwrDoubleComplex, int64_t>(wwrblasHandle_t, wwrblasSideMode_t, wwrblasFillMode_t,
                                wwrblasOperation_t, wwrblasDiagType_t, int64_t, int64_t,
                                const wwrDoubleComplex *, const wwrDoubleComplex *, int64_t,
                                const wwrDoubleComplex *, int64_t, wwrDoubleComplex *, int64_t);

// Function: trsm
template wwrblasStatus_t trsm<float, int>(wwrblasHandle_t, wwrblasSideMode_t, wwrblasFillMode_t,
                                          wwrblasOperation_t, wwrblasDiagType_t, int, int,
                                          const float *, const float *, int, float *, int);
template wwrblasStatus_t trsm<float, int64_t>(wwrblasHandle_t, wwrblasSideMode_t, wwrblasFillMode_t,
                                              wwrblasOperation_t, wwrblasDiagType_t, int64_t,
                                              int64_t, const float *, const float *, int64_t,
                                              float *, int64_t);
template wwrblasStatus_t trsm<double, int>(wwrblasHandle_t, wwrblasSideMode_t, wwrblasFillMode_t,
                                           wwrblasOperation_t, wwrblasDiagType_t, int, int,
                                           const double *, const double *, int, double *, int);
template wwrblasStatus_t trsm<double, int64_t>(wwrblasHandle_t, wwrblasSideMode_t,
                                               wwrblasFillMode_t, wwrblasOperation_t,
                                               wwrblasDiagType_t, int64_t, int64_t, const double *,
                                               const double *, int64_t, double *, int64_t);
template wwrblasStatus_t
trsm<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasSideMode_t, wwrblasFillMode_t,
                           wwrblasOperation_t, wwrblasDiagType_t, int, int, const wwrFloatComplex *,
                           const wwrFloatComplex *, int, wwrFloatComplex *, int);
template wwrblasStatus_t trsm<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasSideMode_t,
                                                        wwrblasFillMode_t, wwrblasOperation_t,
                                                        wwrblasDiagType_t, int64_t, int64_t,
                                                        const wwrFloatComplex *,
                                                        const wwrFloatComplex *, int64_t,
                                                        wwrFloatComplex *, int64_t);
template wwrblasStatus_t trsm<wwrDoubleComplex, int>(
    wwrblasHandle_t, wwrblasSideMode_t, wwrblasFillMode_t, wwrblasOperation_t, wwrblasDiagType_t,
    int, int, const wwrDoubleComplex *, const wwrDoubleComplex *, int, wwrDoubleComplex *, int);
template wwrblasStatus_t trsm<wwrDoubleComplex, int64_t>(wwrblasHandle_t, wwrblasSideMode_t,
                                                         wwrblasFillMode_t, wwrblasOperation_t,
                                                         wwrblasDiagType_t, int64_t, int64_t,
                                                         const wwrDoubleComplex *,
                                                         const wwrDoubleComplex *, int64_t,
                                                         wwrDoubleComplex *, int64_t);

// Function: trsmBatched
template wwrblasStatus_t trsmBatched<float, int>(wwrblasHandle_t, wwrblasSideMode_t,
                                                 wwrblasFillMode_t, wwrblasOperation_t,
                                                 wwrblasDiagType_t, int, int, const float *,
                                                 const float *const[], int, float *const[], int,
                                                 int);
template wwrblasStatus_t trsmBatched<float, int64_t>(wwrblasHandle_t, wwrblasSideMode_t,
                                                     wwrblasFillMode_t, wwrblasOperation_t,
                                                     wwrblasDiagType_t, int64_t, int64_t,
                                                     const float *, const float *const[], int64_t,
                                                     float *const[], int64_t, int64_t);
template wwrblasStatus_t trsmBatched<double, int>(wwrblasHandle_t, wwrblasSideMode_t,
                                                  wwrblasFillMode_t, wwrblasOperation_t,
                                                  wwrblasDiagType_t, int, int, const double *,
                                                  const double *const[], int, double *const[], int,
                                                  int);
template wwrblasStatus_t trsmBatched<double, int64_t>(wwrblasHandle_t, wwrblasSideMode_t,
                                                      wwrblasFillMode_t, wwrblasOperation_t,
                                                      wwrblasDiagType_t, int64_t, int64_t,
                                                      const double *, const double *const[],
                                                      int64_t, double *const[], int64_t, int64_t);
template wwrblasStatus_t trsmBatched<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasSideMode_t,
                                                           wwrblasFillMode_t, wwrblasOperation_t,
                                                           wwrblasDiagType_t, int, int,
                                                           const wwrFloatComplex *,
                                                           const wwrFloatComplex *const[], int,
                                                           wwrFloatComplex *const[], int, int);
template wwrblasStatus_t
trsmBatched<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasSideMode_t, wwrblasFillMode_t,
                                      wwrblasOperation_t, wwrblasDiagType_t, int64_t, int64_t,
                                      const wwrFloatComplex *, const wwrFloatComplex *const[],
                                      int64_t, wwrFloatComplex *const[], int64_t, int64_t);
template wwrblasStatus_t trsmBatched<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasSideMode_t,
                                                            wwrblasFillMode_t, wwrblasOperation_t,
                                                            wwrblasDiagType_t, int, int,
                                                            const wwrDoubleComplex *,
                                                            const wwrDoubleComplex *const[], int,
                                                            wwrDoubleComplex *const[], int, int);
template wwrblasStatus_t
trsmBatched<wwrDoubleComplex, int64_t>(wwrblasHandle_t, wwrblasSideMode_t, wwrblasFillMode_t,
                                       wwrblasOperation_t, wwrblasDiagType_t, int64_t, int64_t,
                                       const wwrDoubleComplex *, const wwrDoubleComplex *const[],
                                       int64_t, wwrDoubleComplex *const[], int64_t, int64_t);

// Function: hemm
template wwrblasStatus_t
hemm<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasSideMode_t, wwrblasFillMode_t, int, int,
                           const wwrFloatComplex *, const wwrFloatComplex *, int,
                           const wwrFloatComplex *, int, const wwrFloatComplex *, wwrFloatComplex *,
                           int);
template wwrblasStatus_t
hemm<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasSideMode_t, wwrblasFillMode_t, int64_t,
                               int64_t, const wwrFloatComplex *, const wwrFloatComplex *, int64_t,
                               const wwrFloatComplex *, int64_t, const wwrFloatComplex *,
                               wwrFloatComplex *, int64_t);
template wwrblasStatus_t
hemm<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasSideMode_t, wwrblasFillMode_t, int, int,
                            const wwrDoubleComplex *, const wwrDoubleComplex *, int,
                            const wwrDoubleComplex *, int, const wwrDoubleComplex *,
                            wwrDoubleComplex *, int);
template wwrblasStatus_t
hemm<wwrDoubleComplex, int64_t>(wwrblasHandle_t, wwrblasSideMode_t, wwrblasFillMode_t, int64_t,
                                int64_t, const wwrDoubleComplex *, const wwrDoubleComplex *,
                                int64_t, const wwrDoubleComplex *, int64_t,
                                const wwrDoubleComplex *, wwrDoubleComplex *, int64_t);

// Function: herk
template wwrblasStatus_t
herk<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t, int, int,
                           const ComplexToRealType<wwrFloatComplex> *, const wwrFloatComplex *, int,
                           const ComplexToRealType<wwrFloatComplex> *, wwrFloatComplex *, int);
template wwrblasStatus_t herk<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                        wwrblasOperation_t, int64_t, int64_t,
                                                        const ComplexToRealType<wwrFloatComplex> *,
                                                        const wwrFloatComplex *, int64_t,
                                                        const ComplexToRealType<wwrFloatComplex> *,
                                                        wwrFloatComplex *, int64_t);
template wwrblasStatus_t herk<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasFillMode_t,
                                                     wwrblasOperation_t, int, int,
                                                     const ComplexToRealType<wwrDoubleComplex> *,
                                                     const wwrDoubleComplex *, int,
                                                     const ComplexToRealType<wwrDoubleComplex> *,
                                                     wwrDoubleComplex *, int);
template wwrblasStatus_t herk<wwrDoubleComplex, int64_t>(
    wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t, int64_t, int64_t,
    const ComplexToRealType<wwrDoubleComplex> *, const wwrDoubleComplex *, int64_t,
    const ComplexToRealType<wwrDoubleComplex> *, wwrDoubleComplex *, int64_t);

// Function: her2k
template wwrblasStatus_t
her2k<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t, int, int,
                            const wwrFloatComplex *, const wwrFloatComplex *, int,
                            const wwrFloatComplex *, int,
                            const ComplexToRealType<wwrFloatComplex> *, wwrFloatComplex *, int);
template wwrblasStatus_t her2k<wwrFloatComplex, int64_t>(
    wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t, int64_t, int64_t,
    const wwrFloatComplex *, const wwrFloatComplex *, int64_t, const wwrFloatComplex *, int64_t,
    const ComplexToRealType<wwrFloatComplex> *, wwrFloatComplex *, int64_t);
template wwrblasStatus_t
her2k<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t, int, int,
                             const wwrDoubleComplex *, const wwrDoubleComplex *, int,
                             const wwrDoubleComplex *, int,
                             const ComplexToRealType<wwrDoubleComplex> *, wwrDoubleComplex *, int);
template wwrblasStatus_t her2k<wwrDoubleComplex, int64_t>(
    wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t, int64_t, int64_t,
    const wwrDoubleComplex *, const wwrDoubleComplex *, int64_t, const wwrDoubleComplex *, int64_t,
    const ComplexToRealType<wwrDoubleComplex> *, wwrDoubleComplex *, int64_t);

// Function: herkx
template wwrblasStatus_t
herkx<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t, int, int,
                            const wwrFloatComplex *, const wwrFloatComplex *, int,
                            const wwrFloatComplex *, int,
                            const ComplexToRealType<wwrFloatComplex> *, wwrFloatComplex *, int);
template wwrblasStatus_t herkx<wwrFloatComplex, int64_t>(
    wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t, int64_t, int64_t,
    const wwrFloatComplex *, const wwrFloatComplex *, int64_t, const wwrFloatComplex *, int64_t,
    const ComplexToRealType<wwrFloatComplex> *, wwrFloatComplex *, int64_t);
template wwrblasStatus_t
herkx<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t, int, int,
                             const wwrDoubleComplex *, const wwrDoubleComplex *, int,
                             const wwrDoubleComplex *, int,
                             const ComplexToRealType<wwrDoubleComplex> *, wwrDoubleComplex *, int);
template wwrblasStatus_t herkx<wwrDoubleComplex, int64_t>(
    wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t, int64_t, int64_t,
    const wwrDoubleComplex *, const wwrDoubleComplex *, int64_t, const wwrDoubleComplex *, int64_t,
    const ComplexToRealType<wwrDoubleComplex> *, wwrDoubleComplex *, int64_t);

// Function: geam
template wwrblasStatus_t geam<float, int>(wwrblasHandle_t, wwrblasOperation_t, wwrblasOperation_t,
                                          int, int, const float *, const float *, int,
                                          const float *, const float *, int, float *, int);
template wwrblasStatus_t geam<float, int64_t>(wwrblasHandle_t, wwrblasOperation_t,
                                              wwrblasOperation_t, int64_t, int64_t, const float *,
                                              const float *, int64_t, const float *, const float *,
                                              int64_t, float *, int64_t);
template wwrblasStatus_t geam<double, int>(wwrblasHandle_t, wwrblasOperation_t, wwrblasOperation_t,
                                           int, int, const double *, const double *, int,
                                           const double *, const double *, int, double *, int);
template wwrblasStatus_t geam<double, int64_t>(wwrblasHandle_t, wwrblasOperation_t,
                                               wwrblasOperation_t, int64_t, int64_t, const double *,
                                               const double *, int64_t, const double *,
                                               const double *, int64_t, double *, int64_t);
template wwrblasStatus_t
geam<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasOperation_t, wwrblasOperation_t, int, int,
                           const wwrFloatComplex *, const wwrFloatComplex *, int,
                           const wwrFloatComplex *, const wwrFloatComplex *, int, wwrFloatComplex *,
                           int);
template wwrblasStatus_t
geam<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasOperation_t, wwrblasOperation_t, int64_t,
                               int64_t, const wwrFloatComplex *, const wwrFloatComplex *, int64_t,
                               const wwrFloatComplex *, const wwrFloatComplex *, int64_t,
                               wwrFloatComplex *, int64_t);
template wwrblasStatus_t
geam<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasOperation_t, wwrblasOperation_t, int, int,
                            const wwrDoubleComplex *, const wwrDoubleComplex *, int,
                            const wwrDoubleComplex *, const wwrDoubleComplex *, int,
                            wwrDoubleComplex *, int);
template wwrblasStatus_t
geam<wwrDoubleComplex, int64_t>(wwrblasHandle_t, wwrblasOperation_t, wwrblasOperation_t, int64_t,
                                int64_t, const wwrDoubleComplex *, const wwrDoubleComplex *,
                                int64_t, const wwrDoubleComplex *, const wwrDoubleComplex *,
                                int64_t, wwrDoubleComplex *, int64_t);

// Function: dgmm
template wwrblasStatus_t dgmm<float, int>(wwrblasHandle_t, wwrblasSideMode_t, int, int,
                                          const float *, int, const float *, int, float *, int);
template wwrblasStatus_t dgmm<float, int64_t>(wwrblasHandle_t, wwrblasSideMode_t, int64_t, int64_t,
                                              const float *, int64_t, const float *, int64_t,
                                              float *, int64_t);
template wwrblasStatus_t dgmm<double, int>(wwrblasHandle_t, wwrblasSideMode_t, int, int,
                                           const double *, int, const double *, int, double *, int);
template wwrblasStatus_t dgmm<double, int64_t>(wwrblasHandle_t, wwrblasSideMode_t, int64_t, int64_t,
                                               const double *, int64_t, const double *, int64_t,
                                               double *, int64_t);
template wwrblasStatus_t dgmm<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasSideMode_t, int, int,
                                                    const wwrFloatComplex *, int,
                                                    const wwrFloatComplex *, int, wwrFloatComplex *,
                                                    int);
template wwrblasStatus_t dgmm<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasSideMode_t, int64_t,
                                                        int64_t, const wwrFloatComplex *, int64_t,
                                                        const wwrFloatComplex *, int64_t,
                                                        wwrFloatComplex *, int64_t);
template wwrblasStatus_t dgmm<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasSideMode_t, int, int,
                                                     const wwrDoubleComplex *, int,
                                                     const wwrDoubleComplex *, int,
                                                     wwrDoubleComplex *, int);
template wwrblasStatus_t dgmm<wwrDoubleComplex, int64_t>(wwrblasHandle_t, wwrblasSideMode_t,
                                                         int64_t, int64_t, const wwrDoubleComplex *,
                                                         int64_t, const wwrDoubleComplex *, int64_t,
                                                         wwrDoubleComplex *, int64_t);

// Function: getrfBatched
template wwrblasStatus_t getrfBatched<float>(wwrblasHandle_t, int, float *const[], int, int *,
                                             int *, int);
template wwrblasStatus_t getrfBatched<double>(wwrblasHandle_t, int, double *const[], int, int *,
                                              int *, int);
template wwrblasStatus_t getrfBatched<wwrFloatComplex>(wwrblasHandle_t, int,
                                                       wwrFloatComplex *const[], int, int *, int *,
                                                       int);
template wwrblasStatus_t getrfBatched<wwrDoubleComplex>(wwrblasHandle_t, int,
                                                        wwrDoubleComplex *const[], int, int *,
                                                        int *, int);

// Function: getrsBatched
template wwrblasStatus_t getrsBatched<float>(wwrblasHandle_t, wwrblasOperation_t, int, int,
                                             const float *const[], int, const int *, float *const[],
                                             int, int *, int);
template wwrblasStatus_t getrsBatched<double>(wwrblasHandle_t, wwrblasOperation_t, int, int,
                                              const double *const[], int, const int *,
                                              double *const[], int, int *, int);
template wwrblasStatus_t getrsBatched<wwrFloatComplex>(wwrblasHandle_t, wwrblasOperation_t, int,
                                                       int, const wwrFloatComplex *const[], int,
                                                       const int *, wwrFloatComplex *const[], int,
                                                       int *, int);
template wwrblasStatus_t getrsBatched<wwrDoubleComplex>(wwrblasHandle_t, wwrblasOperation_t, int,
                                                        int, const wwrDoubleComplex *const[], int,
                                                        const int *, wwrDoubleComplex *const[], int,
                                                        int *, int);

// Function: getriBatched
template wwrblasStatus_t getriBatched<float>(wwrblasHandle_t, int, const float *const[], int,
                                             const int *, float *const[], int, int *, int);
template wwrblasStatus_t getriBatched<double>(wwrblasHandle_t, int, const double *const[], int,
                                              const int *, double *const[], int, int *, int);
template wwrblasStatus_t getriBatched<wwrFloatComplex>(wwrblasHandle_t, int,
                                                       const wwrFloatComplex *const[], int,
                                                       const int *, wwrFloatComplex *const[], int,
                                                       int *, int);
template wwrblasStatus_t getriBatched<wwrDoubleComplex>(wwrblasHandle_t, int,
                                                        const wwrDoubleComplex *const[], int,
                                                        const int *, wwrDoubleComplex *const[], int,
                                                        int *, int);

// Function: geqrfBatched
template wwrblasStatus_t geqrfBatched<float>(wwrblasHandle_t, int, int, float *const[], int,
                                             float *const[], int *, int);
template wwrblasStatus_t geqrfBatched<double>(wwrblasHandle_t, int, int, double *const[], int,
                                              double *const[], int *, int);
template wwrblasStatus_t geqrfBatched<wwrFloatComplex>(wwrblasHandle_t, int, int,
                                                       wwrFloatComplex *const[], int,
                                                       wwrFloatComplex *const[], int *, int);
template wwrblasStatus_t geqrfBatched<wwrDoubleComplex>(wwrblasHandle_t, int, int,
                                                        wwrDoubleComplex *const[], int,
                                                        wwrDoubleComplex *const[], int *, int);

// Function: gelsBatched
template wwrblasStatus_t gelsBatched<float>(wwrblasHandle_t, wwrblasOperation_t, int, int, int,
                                            float *const[], int, float *const[], int, int *, int *,
                                            int);
template wwrblasStatus_t gelsBatched<double>(wwrblasHandle_t, wwrblasOperation_t, int, int, int,
                                             double *const[], int, double *const[], int, int *,
                                             int *, int);
template wwrblasStatus_t gelsBatched<wwrFloatComplex>(wwrblasHandle_t, wwrblasOperation_t, int, int,
                                                      int, wwrFloatComplex *const[], int,
                                                      wwrFloatComplex *const[], int, int *, int *,
                                                      int);
template wwrblasStatus_t gelsBatched<wwrDoubleComplex>(wwrblasHandle_t, wwrblasOperation_t, int,
                                                       int, int, wwrDoubleComplex *const[], int,
                                                       wwrDoubleComplex *const[], int, int *, int *,
                                                       int);

} // namespace wwr
