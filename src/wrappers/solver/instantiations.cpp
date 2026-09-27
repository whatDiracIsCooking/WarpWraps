/**
 * @file instantiations.cpp
 * @brief Explicit template instantiations for the GPU solver wrappers
 *
 * This file contains explicit template instantiations to avoid code bloat
 * from implicit instantiation at every call site. Hand-written -- see
 * math/triple_gemm for the pattern; extern template lives in the .cppm
 * partition, template here.
 */

module wwr.wrappers.solver;

import wwr.solver;
import wwr.blas;
import wwr.complex;
import wwr.wrappers.common;

namespace wwr {

// Function: potrf_bufferSize
template wwrsolverStatus_t potrf_bufferSize<float>(wwrsolverDnHandle_t, wwrblasFillMode_t, int,
                                                   float *, int, int *);
template wwrsolverStatus_t potrf_bufferSize<double>(wwrsolverDnHandle_t, wwrblasFillMode_t, int,
                                                    double *, int, int *);
template wwrsolverStatus_t potrf_bufferSize<wwrFloatComplex>(wwrsolverDnHandle_t, wwrblasFillMode_t,
                                                             int, wwrFloatComplex *, int, int *);
template wwrsolverStatus_t potrf_bufferSize<wwrDoubleComplex>(wwrsolverDnHandle_t,
                                                              wwrblasFillMode_t, int,
                                                              wwrDoubleComplex *, int, int *);

// Function: potrf
template wwrsolverStatus_t potrf<float>(wwrsolverDnHandle_t, wwrblasFillMode_t, int, float *, int,
                                        float *, int, int *);
template wwrsolverStatus_t potrf<double>(wwrsolverDnHandle_t, wwrblasFillMode_t, int, double *, int,
                                         double *, int, int *);
template wwrsolverStatus_t potrf<wwrFloatComplex>(wwrsolverDnHandle_t, wwrblasFillMode_t, int,
                                                  wwrFloatComplex *, int, wwrFloatComplex *, int,
                                                  int *);
template wwrsolverStatus_t potrf<wwrDoubleComplex>(wwrsolverDnHandle_t, wwrblasFillMode_t, int,
                                                   wwrDoubleComplex *, int, wwrDoubleComplex *, int,
                                                   int *);

// Function: potrs
template wwrsolverStatus_t potrs<float>(wwrsolverDnHandle_t, wwrblasFillMode_t, int, int,
                                        const float *, int, float *, int, int *);
template wwrsolverStatus_t potrs<double>(wwrsolverDnHandle_t, wwrblasFillMode_t, int, int,
                                         const double *, int, double *, int, int *);
template wwrsolverStatus_t potrs<wwrFloatComplex>(wwrsolverDnHandle_t, wwrblasFillMode_t, int, int,
                                                  const wwrFloatComplex *, int, wwrFloatComplex *,
                                                  int, int *);
template wwrsolverStatus_t potrs<wwrDoubleComplex>(wwrsolverDnHandle_t, wwrblasFillMode_t, int, int,
                                                   const wwrDoubleComplex *, int,
                                                   wwrDoubleComplex *, int, int *);

// Function: potri_bufferSize
template wwrsolverStatus_t potri_bufferSize<float>(wwrsolverDnHandle_t, wwrblasFillMode_t, int,
                                                   float *, int, int *);
template wwrsolverStatus_t potri_bufferSize<double>(wwrsolverDnHandle_t, wwrblasFillMode_t, int,
                                                    double *, int, int *);
template wwrsolverStatus_t potri_bufferSize<wwrFloatComplex>(wwrsolverDnHandle_t, wwrblasFillMode_t,
                                                             int, wwrFloatComplex *, int, int *);
template wwrsolverStatus_t potri_bufferSize<wwrDoubleComplex>(wwrsolverDnHandle_t,
                                                              wwrblasFillMode_t, int,
                                                              wwrDoubleComplex *, int, int *);

// Function: potri
template wwrsolverStatus_t potri<float>(wwrsolverDnHandle_t, wwrblasFillMode_t, int, float *, int,
                                        float *, int, int *);
template wwrsolverStatus_t potri<double>(wwrsolverDnHandle_t, wwrblasFillMode_t, int, double *, int,
                                         double *, int, int *);
template wwrsolverStatus_t potri<wwrFloatComplex>(wwrsolverDnHandle_t, wwrblasFillMode_t, int,
                                                  wwrFloatComplex *, int, wwrFloatComplex *, int,
                                                  int *);
template wwrsolverStatus_t potri<wwrDoubleComplex>(wwrsolverDnHandle_t, wwrblasFillMode_t, int,
                                                   wwrDoubleComplex *, int, wwrDoubleComplex *, int,
                                                   int *);

// Function: getrf_bufferSize
template wwrsolverStatus_t getrf_bufferSize<float>(wwrsolverDnHandle_t, int, int, float *, int,
                                                   int *);
template wwrsolverStatus_t getrf_bufferSize<double>(wwrsolverDnHandle_t, int, int, double *, int,
                                                    int *);
template wwrsolverStatus_t getrf_bufferSize<wwrFloatComplex>(wwrsolverDnHandle_t, int, int,
                                                             wwrFloatComplex *, int, int *);
template wwrsolverStatus_t getrf_bufferSize<wwrDoubleComplex>(wwrsolverDnHandle_t, int, int,
                                                              wwrDoubleComplex *, int, int *);

// Function: getrf
template wwrsolverStatus_t getrf<float>(wwrsolverDnHandle_t, int, int, float *, int, float *, int *,
                                        int *);
template wwrsolverStatus_t getrf<double>(wwrsolverDnHandle_t, int, int, double *, int, double *,
                                         int *, int *);
template wwrsolverStatus_t getrf<wwrFloatComplex>(wwrsolverDnHandle_t, int, int, wwrFloatComplex *,
                                                  int, wwrFloatComplex *, int *, int *);
template wwrsolverStatus_t getrf<wwrDoubleComplex>(wwrsolverDnHandle_t, int, int,
                                                   wwrDoubleComplex *, int, wwrDoubleComplex *,
                                                   int *, int *);

// Function: getrs
template wwrsolverStatus_t getrs<float>(wwrsolverDnHandle_t, wwrblasOperation_t, int, int,
                                        const float *, int, const int *, float *, int, int *);
template wwrsolverStatus_t getrs<double>(wwrsolverDnHandle_t, wwrblasOperation_t, int, int,
                                         const double *, int, const int *, double *, int, int *);
template wwrsolverStatus_t getrs<wwrFloatComplex>(wwrsolverDnHandle_t, wwrblasOperation_t, int, int,
                                                  const wwrFloatComplex *, int, const int *,
                                                  wwrFloatComplex *, int, int *);
template wwrsolverStatus_t getrs<wwrDoubleComplex>(wwrsolverDnHandle_t, wwrblasOperation_t, int,
                                                   int, const wwrDoubleComplex *, int, const int *,
                                                   wwrDoubleComplex *, int, int *);

// Function: geqrf_bufferSize
template wwrsolverStatus_t geqrf_bufferSize<float>(wwrsolverDnHandle_t, int, int, float *, int,
                                                   int *);
template wwrsolverStatus_t geqrf_bufferSize<double>(wwrsolverDnHandle_t, int, int, double *, int,
                                                    int *);
template wwrsolverStatus_t geqrf_bufferSize<wwrFloatComplex>(wwrsolverDnHandle_t, int, int,
                                                             wwrFloatComplex *, int, int *);
template wwrsolverStatus_t geqrf_bufferSize<wwrDoubleComplex>(wwrsolverDnHandle_t, int, int,
                                                              wwrDoubleComplex *, int, int *);

// Function: geqrf
template wwrsolverStatus_t geqrf<float>(wwrsolverDnHandle_t, int, int, float *, int, float *,
                                        float *, int, int *);
template wwrsolverStatus_t geqrf<double>(wwrsolverDnHandle_t, int, int, double *, int, double *,
                                         double *, int, int *);
template wwrsolverStatus_t geqrf<wwrFloatComplex>(wwrsolverDnHandle_t, int, int, wwrFloatComplex *,
                                                  int, wwrFloatComplex *, wwrFloatComplex *, int,
                                                  int *);
template wwrsolverStatus_t geqrf<wwrDoubleComplex>(wwrsolverDnHandle_t, int, int,
                                                   wwrDoubleComplex *, int, wwrDoubleComplex *,
                                                   wwrDoubleComplex *, int, int *);

// Function: ormqr_bufferSize
template wwrsolverStatus_t ormqr_bufferSize<float>(wwrsolverDnHandle_t, wwrblasSideMode_t,
                                                   wwrblasOperation_t, int, int, int, const float *,
                                                   int, const float *, const float *, int, int *);
template wwrsolverStatus_t ormqr_bufferSize<double>(wwrsolverDnHandle_t, wwrblasSideMode_t,
                                                    wwrblasOperation_t, int, int, int,
                                                    const double *, int, const double *,
                                                    const double *, int, int *);

// Function: ormqr
template wwrsolverStatus_t ormqr<float>(wwrsolverDnHandle_t, wwrblasSideMode_t, wwrblasOperation_t,
                                        int, int, int, const float *, int, const float *, float *,
                                        int, float *, int, int *);
template wwrsolverStatus_t ormqr<double>(wwrsolverDnHandle_t, wwrblasSideMode_t, wwrblasOperation_t,
                                         int, int, int, const double *, int, const double *,
                                         double *, int, double *, int, int *);

// Function: unmqr_bufferSize
template wwrsolverStatus_t unmqr_bufferSize<wwrFloatComplex>(wwrsolverDnHandle_t, wwrblasSideMode_t,
                                                             wwrblasOperation_t, int, int, int,
                                                             const wwrFloatComplex *, int,
                                                             const wwrFloatComplex *,
                                                             const wwrFloatComplex *, int, int *);
template wwrsolverStatus_t
unmqr_bufferSize<wwrDoubleComplex>(wwrsolverDnHandle_t, wwrblasSideMode_t, wwrblasOperation_t, int,
                                   int, int, const wwrDoubleComplex *, int,
                                   const wwrDoubleComplex *, const wwrDoubleComplex *, int, int *);

// Function: unmqr
template wwrsolverStatus_t unmqr<wwrFloatComplex>(wwrsolverDnHandle_t, wwrblasSideMode_t,
                                                  wwrblasOperation_t, int, int, int,
                                                  const wwrFloatComplex *, int,
                                                  const wwrFloatComplex *, wwrFloatComplex *, int,
                                                  wwrFloatComplex *, int, int *);
template wwrsolverStatus_t unmqr<wwrDoubleComplex>(wwrsolverDnHandle_t, wwrblasSideMode_t,
                                                   wwrblasOperation_t, int, int, int,
                                                   const wwrDoubleComplex *, int,
                                                   const wwrDoubleComplex *, wwrDoubleComplex *,
                                                   int, wwrDoubleComplex *, int, int *);

// Function: gels_bufferSize
template wwrsolverStatus_t gels_bufferSize<float>(wwrsolverDnHandle_t, int, int, int, float *, int,
                                                  float *, int, float *, int, void *,
                                                  std::size_t *);
template wwrsolverStatus_t gels_bufferSize<double>(wwrsolverDnHandle_t, int, int, int, double *,
                                                   int, double *, int, double *, int, void *,
                                                   std::size_t *);
template wwrsolverStatus_t gels_bufferSize<wwrFloatComplex>(wwrsolverDnHandle_t, int, int, int,
                                                            wwrFloatComplex *, int,
                                                            wwrFloatComplex *, int,
                                                            wwrFloatComplex *, int, void *,
                                                            std::size_t *);
template wwrsolverStatus_t gels_bufferSize<wwrDoubleComplex>(wwrsolverDnHandle_t, int, int, int,
                                                             wwrDoubleComplex *, int,
                                                             wwrDoubleComplex *, int,
                                                             wwrDoubleComplex *, int, void *,
                                                             std::size_t *);

// Function: gels
template wwrsolverStatus_t gels<float>(wwrsolverDnHandle_t, int, int, int, float *, int, float *,
                                       int, float *, int, void *, std::size_t, int *, int *);
template wwrsolverStatus_t gels<double>(wwrsolverDnHandle_t, int, int, int, double *, int, double *,
                                        int, double *, int, void *, std::size_t, int *, int *);
template wwrsolverStatus_t gels<wwrFloatComplex>(wwrsolverDnHandle_t, int, int, int,
                                                 wwrFloatComplex *, int, wwrFloatComplex *, int,
                                                 wwrFloatComplex *, int, void *, std::size_t, int *,
                                                 int *);
template wwrsolverStatus_t gels<wwrDoubleComplex>(wwrsolverDnHandle_t, int, int, int,
                                                  wwrDoubleComplex *, int, wwrDoubleComplex *, int,
                                                  wwrDoubleComplex *, int, void *, std::size_t,
                                                  int *, int *);

// Function: gesv_bufferSize
template wwrsolverStatus_t gesv_bufferSize<float>(wwrsolverDnHandle_t, int, int, float *, int,
                                                  int *, float *, int, float *, int, void *,
                                                  std::size_t *);
template wwrsolverStatus_t gesv_bufferSize<double>(wwrsolverDnHandle_t, int, int, double *, int,
                                                   int *, double *, int, double *, int, void *,
                                                   std::size_t *);
template wwrsolverStatus_t gesv_bufferSize<wwrFloatComplex>(wwrsolverDnHandle_t, int, int,
                                                            wwrFloatComplex *, int, int *,
                                                            wwrFloatComplex *, int,
                                                            wwrFloatComplex *, int, void *,
                                                            std::size_t *);
template wwrsolverStatus_t gesv_bufferSize<wwrDoubleComplex>(wwrsolverDnHandle_t, int, int,
                                                             wwrDoubleComplex *, int, int *,
                                                             wwrDoubleComplex *, int,
                                                             wwrDoubleComplex *, int, void *,
                                                             std::size_t *);

// Function: gesv
template wwrsolverStatus_t gesv<float>(wwrsolverDnHandle_t, int, int, float *, int, int *, float *,
                                       int, float *, int, void *, std::size_t, int *, int *);
template wwrsolverStatus_t gesv<double>(wwrsolverDnHandle_t, int, int, double *, int, int *,
                                        double *, int, double *, int, void *, std::size_t, int *,
                                        int *);
template wwrsolverStatus_t gesv<wwrFloatComplex>(wwrsolverDnHandle_t, int, int, wwrFloatComplex *,
                                                 int, int *, wwrFloatComplex *, int,
                                                 wwrFloatComplex *, int, void *, std::size_t, int *,
                                                 int *);
template wwrsolverStatus_t gesv<wwrDoubleComplex>(wwrsolverDnHandle_t, int, int, wwrDoubleComplex *,
                                                  int, int *, wwrDoubleComplex *, int,
                                                  wwrDoubleComplex *, int, void *, std::size_t,
                                                  int *, int *);

// Function: potrfBatched
template wwrsolverStatus_t potrfBatched<float>(wwrsolverDnHandle_t, wwrblasFillMode_t, int,
                                               float *[], int, int *, int);
template wwrsolverStatus_t potrfBatched<double>(wwrsolverDnHandle_t, wwrblasFillMode_t, int,
                                                double *[], int, int *, int);
template wwrsolverStatus_t potrfBatched<wwrFloatComplex>(wwrsolverDnHandle_t, wwrblasFillMode_t,
                                                         int, wwrFloatComplex *[], int, int *, int);
template wwrsolverStatus_t potrfBatched<wwrDoubleComplex>(wwrsolverDnHandle_t, wwrblasFillMode_t,
                                                          int, wwrDoubleComplex *[], int, int *,
                                                          int);

// Function: potrsBatched
template wwrsolverStatus_t potrsBatched<float>(wwrsolverDnHandle_t, wwrblasFillMode_t, int, int,
                                               float *[], int, float *[], int, int *, int);
template wwrsolverStatus_t potrsBatched<double>(wwrsolverDnHandle_t, wwrblasFillMode_t, int, int,
                                                double *[], int, double *[], int, int *, int);
template wwrsolverStatus_t potrsBatched<wwrFloatComplex>(wwrsolverDnHandle_t, wwrblasFillMode_t,
                                                         int, int, wwrFloatComplex *[], int,
                                                         wwrFloatComplex *[], int, int *, int);
template wwrsolverStatus_t potrsBatched<wwrDoubleComplex>(wwrsolverDnHandle_t, wwrblasFillMode_t,
                                                          int, int, wwrDoubleComplex *[], int,
                                                          wwrDoubleComplex *[], int, int *, int);

// Function: sytrf_bufferSize
template wwrsolverStatus_t sytrf_bufferSize<float>(wwrsolverDnHandle_t, int, float *, int, int *);
template wwrsolverStatus_t sytrf_bufferSize<double>(wwrsolverDnHandle_t, int, double *, int, int *);
template wwrsolverStatus_t sytrf_bufferSize<wwrFloatComplex>(wwrsolverDnHandle_t, int,
                                                             wwrFloatComplex *, int, int *);
template wwrsolverStatus_t sytrf_bufferSize<wwrDoubleComplex>(wwrsolverDnHandle_t, int,
                                                              wwrDoubleComplex *, int, int *);

// Function: sytrf
template wwrsolverStatus_t sytrf<float>(wwrsolverDnHandle_t, wwrblasFillMode_t, int, float *, int,
                                        int *, float *, int, int *);
template wwrsolverStatus_t sytrf<double>(wwrsolverDnHandle_t, wwrblasFillMode_t, int, double *, int,
                                         int *, double *, int, int *);
template wwrsolverStatus_t sytrf<wwrFloatComplex>(wwrsolverDnHandle_t, wwrblasFillMode_t, int,
                                                  wwrFloatComplex *, int, int *, wwrFloatComplex *,
                                                  int, int *);
template wwrsolverStatus_t sytrf<wwrDoubleComplex>(wwrsolverDnHandle_t, wwrblasFillMode_t, int,
                                                   wwrDoubleComplex *, int, int *,
                                                   wwrDoubleComplex *, int, int *);

// Function: gebrd_bufferSize
template wwrsolverStatus_t gebrd_bufferSize<float>(wwrsolverDnHandle_t, int, int, int *);
template wwrsolverStatus_t gebrd_bufferSize<double>(wwrsolverDnHandle_t, int, int, int *);
template wwrsolverStatus_t gebrd_bufferSize<wwrFloatComplex>(wwrsolverDnHandle_t, int, int, int *);
template wwrsolverStatus_t gebrd_bufferSize<wwrDoubleComplex>(wwrsolverDnHandle_t, int, int, int *);

// Function: gebrd
template wwrsolverStatus_t gebrd<float>(wwrsolverDnHandle_t, int, int, float *, int,
                                        ComplexToRealType<float> *, ComplexToRealType<float> *,
                                        float *, float *, float *, int, int *);
template wwrsolverStatus_t gebrd<double>(wwrsolverDnHandle_t, int, int, double *, int,
                                         ComplexToRealType<double> *, ComplexToRealType<double> *,
                                         double *, double *, double *, int, int *);
template wwrsolverStatus_t gebrd<wwrFloatComplex>(wwrsolverDnHandle_t, int, int, wwrFloatComplex *,
                                                  int, ComplexToRealType<wwrFloatComplex> *,
                                                  ComplexToRealType<wwrFloatComplex> *,
                                                  wwrFloatComplex *, wwrFloatComplex *,
                                                  wwrFloatComplex *, int, int *);
template wwrsolverStatus_t gebrd<wwrDoubleComplex>(wwrsolverDnHandle_t, int, int,
                                                   wwrDoubleComplex *, int,
                                                   ComplexToRealType<wwrDoubleComplex> *,
                                                   ComplexToRealType<wwrDoubleComplex> *,
                                                   wwrDoubleComplex *, wwrDoubleComplex *,
                                                   wwrDoubleComplex *, int, int *);

// Function: orgqr_bufferSize
template wwrsolverStatus_t orgqr_bufferSize<float>(wwrsolverDnHandle_t, int, int, int,
                                                   const float *, int, const float *, int *);
template wwrsolverStatus_t orgqr_bufferSize<double>(wwrsolverDnHandle_t, int, int, int,
                                                    const double *, int, const double *, int *);

// Function: ungqr_bufferSize
template wwrsolverStatus_t ungqr_bufferSize<wwrFloatComplex>(wwrsolverDnHandle_t, int, int, int,
                                                             const wwrFloatComplex *, int,
                                                             const wwrFloatComplex *, int *);
template wwrsolverStatus_t ungqr_bufferSize<wwrDoubleComplex>(wwrsolverDnHandle_t, int, int, int,
                                                              const wwrDoubleComplex *, int,
                                                              const wwrDoubleComplex *, int *);

// Function: orgqr
template wwrsolverStatus_t orgqr<float>(wwrsolverDnHandle_t, int, int, int, float *, int,
                                        const float *, float *, int, int *);
template wwrsolverStatus_t orgqr<double>(wwrsolverDnHandle_t, int, int, int, double *, int,
                                         const double *, double *, int, int *);

// Function: ungqr
template wwrsolverStatus_t ungqr<wwrFloatComplex>(wwrsolverDnHandle_t, int, int, int,
                                                  wwrFloatComplex *, int, const wwrFloatComplex *,
                                                  wwrFloatComplex *, int, int *);
template wwrsolverStatus_t ungqr<wwrDoubleComplex>(wwrsolverDnHandle_t, int, int, int,
                                                   wwrDoubleComplex *, int,
                                                   const wwrDoubleComplex *, wwrDoubleComplex *,
                                                   int, int *);

// Function: orgbr_bufferSize
template wwrsolverStatus_t orgbr_bufferSize<float>(wwrsolverDnHandle_t, wwrblasSideMode_t, int, int,
                                                   int, const float *, int, const float *, int *);
template wwrsolverStatus_t orgbr_bufferSize<double>(wwrsolverDnHandle_t, wwrblasSideMode_t, int,
                                                    int, int, const double *, int, const double *,
                                                    int *);

// Function: ungbr_bufferSize
template wwrsolverStatus_t ungbr_bufferSize<wwrFloatComplex>(wwrsolverDnHandle_t, wwrblasSideMode_t,
                                                             int, int, int, const wwrFloatComplex *,
                                                             int, const wwrFloatComplex *, int *);
template wwrsolverStatus_t ungbr_bufferSize<wwrDoubleComplex>(wwrsolverDnHandle_t,
                                                              wwrblasSideMode_t, int, int, int,
                                                              const wwrDoubleComplex *, int,
                                                              const wwrDoubleComplex *, int *);

// Function: orgbr
template wwrsolverStatus_t orgbr<float>(wwrsolverDnHandle_t, wwrblasSideMode_t, int, int, int,
                                        float *, int, const float *, float *, int, int *);
template wwrsolverStatus_t orgbr<double>(wwrsolverDnHandle_t, wwrblasSideMode_t, int, int, int,
                                         double *, int, const double *, double *, int, int *);

// Function: ungbr
template wwrsolverStatus_t ungbr<wwrFloatComplex>(wwrsolverDnHandle_t, wwrblasSideMode_t, int, int,
                                                  int, wwrFloatComplex *, int,
                                                  const wwrFloatComplex *, wwrFloatComplex *, int,
                                                  int *);
template wwrsolverStatus_t ungbr<wwrDoubleComplex>(wwrsolverDnHandle_t, wwrblasSideMode_t, int, int,
                                                   int, wwrDoubleComplex *, int,
                                                   const wwrDoubleComplex *, wwrDoubleComplex *,
                                                   int, int *);

// Function: gesvd_bufferSize
template wwrsolverStatus_t gesvd_bufferSize<float>(wwrsolverDnHandle_t, int, int, int *);
template wwrsolverStatus_t gesvd_bufferSize<double>(wwrsolverDnHandle_t, int, int, int *);
template wwrsolverStatus_t gesvd_bufferSize<wwrFloatComplex>(wwrsolverDnHandle_t, int, int, int *);
template wwrsolverStatus_t gesvd_bufferSize<wwrDoubleComplex>(wwrsolverDnHandle_t, int, int, int *);

// Function: gesvd
template wwrsolverStatus_t gesvd<float>(wwrsolverDnHandle_t, signed char, signed char, int, int,
                                        float *, int, ComplexToRealType<float> *, float *, int,
                                        float *, int, float *, int, ComplexToRealType<float> *,
                                        int *);
template wwrsolverStatus_t gesvd<double>(wwrsolverDnHandle_t, signed char, signed char, int, int,
                                         double *, int, ComplexToRealType<double> *, double *, int,
                                         double *, int, double *, int, ComplexToRealType<double> *,
                                         int *);
template wwrsolverStatus_t gesvd<wwrFloatComplex>(wwrsolverDnHandle_t, signed char, signed char,
                                                  int, int, wwrFloatComplex *, int,
                                                  ComplexToRealType<wwrFloatComplex> *,
                                                  wwrFloatComplex *, int, wwrFloatComplex *, int,
                                                  wwrFloatComplex *, int,
                                                  ComplexToRealType<wwrFloatComplex> *, int *);
template wwrsolverStatus_t gesvd<wwrDoubleComplex>(wwrsolverDnHandle_t, signed char, signed char,
                                                   int, int, wwrDoubleComplex *, int,
                                                   ComplexToRealType<wwrDoubleComplex> *,
                                                   wwrDoubleComplex *, int, wwrDoubleComplex *, int,
                                                   wwrDoubleComplex *, int,
                                                   ComplexToRealType<wwrDoubleComplex> *, int *);

// Function: syevd_bufferSize
template wwrsolverStatus_t syevd_bufferSize<float>(wwrsolverDnHandle_t, wwrsolverEigMode_t,
                                                   wwrblasFillMode_t, int, const float *, int,
                                                   const float *, int *);
template wwrsolverStatus_t syevd_bufferSize<double>(wwrsolverDnHandle_t, wwrsolverEigMode_t,
                                                    wwrblasFillMode_t, int, const double *, int,
                                                    const double *, int *);

// Function: syevd
template wwrsolverStatus_t syevd<float>(wwrsolverDnHandle_t, wwrsolverEigMode_t, wwrblasFillMode_t,
                                        int, float *, int, float *, float *, int, int *);
template wwrsolverStatus_t syevd<double>(wwrsolverDnHandle_t, wwrsolverEigMode_t, wwrblasFillMode_t,
                                         int, double *, int, double *, double *, int, int *);

// Function: syevdx_bufferSize
template wwrsolverStatus_t syevdx_bufferSize<float>(wwrsolverDnHandle_t, wwrsolverEigMode_t,
                                                    wwrsolverEigRange_t, wwrblasFillMode_t, int,
                                                    const float *, int, float, float, int, int,
                                                    int *, const float *, int *);
template wwrsolverStatus_t syevdx_bufferSize<double>(wwrsolverDnHandle_t, wwrsolverEigMode_t,
                                                     wwrsolverEigRange_t, wwrblasFillMode_t, int,
                                                     const double *, int, double, double, int, int,
                                                     int *, const double *, int *);

// Function: syevdx
template wwrsolverStatus_t syevdx<float>(wwrsolverDnHandle_t, wwrsolverEigMode_t,
                                         wwrsolverEigRange_t, wwrblasFillMode_t, int, float *, int,
                                         float, float, int, int, int *, float *, float *, int,
                                         int *);
template wwrsolverStatus_t syevdx<double>(wwrsolverDnHandle_t, wwrsolverEigMode_t,
                                          wwrsolverEigRange_t, wwrblasFillMode_t, int, double *,
                                          int, double, double, int, int, int *, double *, double *,
                                          int, int *);

// Function: heevd_bufferSize
template wwrsolverStatus_t
heevd_bufferSize<wwrFloatComplex>(wwrsolverDnHandle_t, wwrsolverEigMode_t, wwrblasFillMode_t, int,
                                  const wwrFloatComplex *, int,
                                  const ComplexToRealType<wwrFloatComplex> *, int *);
template wwrsolverStatus_t
heevd_bufferSize<wwrDoubleComplex>(wwrsolverDnHandle_t, wwrsolverEigMode_t, wwrblasFillMode_t, int,
                                   const wwrDoubleComplex *, int,
                                   const ComplexToRealType<wwrDoubleComplex> *, int *);

// Function: heevd
template wwrsolverStatus_t heevd<wwrFloatComplex>(wwrsolverDnHandle_t, wwrsolverEigMode_t,
                                                  wwrblasFillMode_t, int, wwrFloatComplex *, int,
                                                  ComplexToRealType<wwrFloatComplex> *,
                                                  wwrFloatComplex *, int, int *);
template wwrsolverStatus_t heevd<wwrDoubleComplex>(wwrsolverDnHandle_t, wwrsolverEigMode_t,
                                                   wwrblasFillMode_t, int, wwrDoubleComplex *, int,
                                                   ComplexToRealType<wwrDoubleComplex> *,
                                                   wwrDoubleComplex *, int, int *);

// Function: heevdx_bufferSize
template wwrsolverStatus_t
heevdx_bufferSize<wwrFloatComplex>(wwrsolverDnHandle_t, wwrsolverEigMode_t, wwrsolverEigRange_t,
                                   wwrblasFillMode_t, int, const wwrFloatComplex *, int,
                                   ComplexToRealType<wwrFloatComplex>,
                                   ComplexToRealType<wwrFloatComplex>, int, int, int *,
                                   const ComplexToRealType<wwrFloatComplex> *, int *);
template wwrsolverStatus_t
heevdx_bufferSize<wwrDoubleComplex>(wwrsolverDnHandle_t, wwrsolverEigMode_t, wwrsolverEigRange_t,
                                    wwrblasFillMode_t, int, const wwrDoubleComplex *, int,
                                    ComplexToRealType<wwrDoubleComplex>,
                                    ComplexToRealType<wwrDoubleComplex>, int, int, int *,
                                    const ComplexToRealType<wwrDoubleComplex> *, int *);

// Function: heevdx
template wwrsolverStatus_t heevdx<wwrFloatComplex>(
    wwrsolverDnHandle_t, wwrsolverEigMode_t, wwrsolverEigRange_t, wwrblasFillMode_t, int,
    wwrFloatComplex *, int, ComplexToRealType<wwrFloatComplex>, ComplexToRealType<wwrFloatComplex>,
    int, int, int *, ComplexToRealType<wwrFloatComplex> *, wwrFloatComplex *, int, int *);
template wwrsolverStatus_t heevdx<wwrDoubleComplex>(wwrsolverDnHandle_t, wwrsolverEigMode_t,
                                                    wwrsolverEigRange_t, wwrblasFillMode_t, int,
                                                    wwrDoubleComplex *, int,
                                                    ComplexToRealType<wwrDoubleComplex>,
                                                    ComplexToRealType<wwrDoubleComplex>, int, int,
                                                    int *, ComplexToRealType<wwrDoubleComplex> *,
                                                    wwrDoubleComplex *, int, int *);

// Function: syevj_bufferSize
template wwrsolverStatus_t syevj_bufferSize<float>(wwrsolverDnHandle_t, wwrsolverEigMode_t,
                                                   wwrblasFillMode_t, int, const float *, int,
                                                   const float *, int *, wwrsolverSyevjInfo_t);
template wwrsolverStatus_t syevj_bufferSize<double>(wwrsolverDnHandle_t, wwrsolverEigMode_t,
                                                    wwrblasFillMode_t, int, const double *, int,
                                                    const double *, int *, wwrsolverSyevjInfo_t);

// Function: syevj
template wwrsolverStatus_t syevj<float>(wwrsolverDnHandle_t, wwrsolverEigMode_t, wwrblasFillMode_t,
                                        int, float *, int, float *, float *, int, int *,
                                        wwrsolverSyevjInfo_t);
template wwrsolverStatus_t syevj<double>(wwrsolverDnHandle_t, wwrsolverEigMode_t, wwrblasFillMode_t,
                                         int, double *, int, double *, double *, int, int *,
                                         wwrsolverSyevjInfo_t);

// Function: syevjBatched_bufferSize
template wwrsolverStatus_t syevjBatched_bufferSize<float>(wwrsolverDnHandle_t, wwrsolverEigMode_t,
                                                          wwrblasFillMode_t, int, const float *,
                                                          int, const float *, int *,
                                                          wwrsolverSyevjInfo_t, int);
template wwrsolverStatus_t syevjBatched_bufferSize<double>(wwrsolverDnHandle_t, wwrsolverEigMode_t,
                                                           wwrblasFillMode_t, int, const double *,
                                                           int, const double *, int *,
                                                           wwrsolverSyevjInfo_t, int);

// Function: syevjBatched
template wwrsolverStatus_t syevjBatched<float>(wwrsolverDnHandle_t, wwrsolverEigMode_t,
                                               wwrblasFillMode_t, int, float *, int, float *,
                                               float *, int, int *, wwrsolverSyevjInfo_t, int);
template wwrsolverStatus_t syevjBatched<double>(wwrsolverDnHandle_t, wwrsolverEigMode_t,
                                                wwrblasFillMode_t, int, double *, int, double *,
                                                double *, int, int *, wwrsolverSyevjInfo_t, int);

// Function: heevj_bufferSize
template wwrsolverStatus_t heevj_bufferSize<wwrFloatComplex>(
    wwrsolverDnHandle_t, wwrsolverEigMode_t, wwrblasFillMode_t, int, const wwrFloatComplex *, int,
    const ComplexToRealType<wwrFloatComplex> *, int *, wwrsolverSyevjInfo_t);
template wwrsolverStatus_t heevj_bufferSize<wwrDoubleComplex>(
    wwrsolverDnHandle_t, wwrsolverEigMode_t, wwrblasFillMode_t, int, const wwrDoubleComplex *, int,
    const ComplexToRealType<wwrDoubleComplex> *, int *, wwrsolverSyevjInfo_t);

// Function: heevj
template wwrsolverStatus_t heevj<wwrFloatComplex>(wwrsolverDnHandle_t, wwrsolverEigMode_t,
                                                  wwrblasFillMode_t, int, wwrFloatComplex *, int,
                                                  ComplexToRealType<wwrFloatComplex> *,
                                                  wwrFloatComplex *, int, int *,
                                                  wwrsolverSyevjInfo_t);
template wwrsolverStatus_t heevj<wwrDoubleComplex>(wwrsolverDnHandle_t, wwrsolverEigMode_t,
                                                   wwrblasFillMode_t, int, wwrDoubleComplex *, int,
                                                   ComplexToRealType<wwrDoubleComplex> *,
                                                   wwrDoubleComplex *, int, int *,
                                                   wwrsolverSyevjInfo_t);

// Function: heevjBatched_bufferSize
template wwrsolverStatus_t heevjBatched_bufferSize<wwrFloatComplex>(
    wwrsolverDnHandle_t, wwrsolverEigMode_t, wwrblasFillMode_t, int, const wwrFloatComplex *, int,
    const ComplexToRealType<wwrFloatComplex> *, int *, wwrsolverSyevjInfo_t, int);
template wwrsolverStatus_t heevjBatched_bufferSize<wwrDoubleComplex>(
    wwrsolverDnHandle_t, wwrsolverEigMode_t, wwrblasFillMode_t, int, const wwrDoubleComplex *, int,
    const ComplexToRealType<wwrDoubleComplex> *, int *, wwrsolverSyevjInfo_t, int);

// Function: heevjBatched
template wwrsolverStatus_t heevjBatched<wwrFloatComplex>(wwrsolverDnHandle_t, wwrsolverEigMode_t,
                                                         wwrblasFillMode_t, int, wwrFloatComplex *,
                                                         int, ComplexToRealType<wwrFloatComplex> *,
                                                         wwrFloatComplex *, int, int *,
                                                         wwrsolverSyevjInfo_t, int);
template wwrsolverStatus_t
heevjBatched<wwrDoubleComplex>(wwrsolverDnHandle_t, wwrsolverEigMode_t, wwrblasFillMode_t, int,
                               wwrDoubleComplex *, int, ComplexToRealType<wwrDoubleComplex> *,
                               wwrDoubleComplex *, int, int *, wwrsolverSyevjInfo_t, int);

// Function: gesvdj_bufferSize
template wwrsolverStatus_t gesvdj_bufferSize<float>(wwrsolverDnHandle_t, wwrsolverEigMode_t, int,
                                                    int, int, const float *, int,
                                                    const ComplexToRealType<float> *, const float *,
                                                    int, const float *, int, int *,
                                                    wwrsolverGesvdjInfo_t);
template wwrsolverStatus_t gesvdj_bufferSize<double>(wwrsolverDnHandle_t, wwrsolverEigMode_t, int,
                                                     int, int, const double *, int,
                                                     const ComplexToRealType<double> *,
                                                     const double *, int, const double *, int,
                                                     int *, wwrsolverGesvdjInfo_t);
template wwrsolverStatus_t gesvdj_bufferSize<wwrFloatComplex>(
    wwrsolverDnHandle_t, wwrsolverEigMode_t, int, int, int, const wwrFloatComplex *, int,
    const ComplexToRealType<wwrFloatComplex> *, const wwrFloatComplex *, int,
    const wwrFloatComplex *, int, int *, wwrsolverGesvdjInfo_t);
template wwrsolverStatus_t gesvdj_bufferSize<wwrDoubleComplex>(
    wwrsolverDnHandle_t, wwrsolverEigMode_t, int, int, int, const wwrDoubleComplex *, int,
    const ComplexToRealType<wwrDoubleComplex> *, const wwrDoubleComplex *, int,
    const wwrDoubleComplex *, int, int *, wwrsolverGesvdjInfo_t);

// Function: gesvdj
template wwrsolverStatus_t gesvdj<float>(wwrsolverDnHandle_t, wwrsolverEigMode_t, int, int, int,
                                         float *, int, ComplexToRealType<float> *, float *, int,
                                         float *, int, float *, int, int *, wwrsolverGesvdjInfo_t);
template wwrsolverStatus_t gesvdj<double>(wwrsolverDnHandle_t, wwrsolverEigMode_t, int, int, int,
                                          double *, int, ComplexToRealType<double> *, double *, int,
                                          double *, int, double *, int, int *,
                                          wwrsolverGesvdjInfo_t);
template wwrsolverStatus_t gesvdj<wwrFloatComplex>(wwrsolverDnHandle_t, wwrsolverEigMode_t, int,
                                                   int, int, wwrFloatComplex *, int,
                                                   ComplexToRealType<wwrFloatComplex> *,
                                                   wwrFloatComplex *, int, wwrFloatComplex *, int,
                                                   wwrFloatComplex *, int, int *,
                                                   wwrsolverGesvdjInfo_t);
template wwrsolverStatus_t gesvdj<wwrDoubleComplex>(wwrsolverDnHandle_t, wwrsolverEigMode_t, int,
                                                    int, int, wwrDoubleComplex *, int,
                                                    ComplexToRealType<wwrDoubleComplex> *,
                                                    wwrDoubleComplex *, int, wwrDoubleComplex *,
                                                    int, wwrDoubleComplex *, int, int *,
                                                    wwrsolverGesvdjInfo_t);

// Function: gesvdjBatched_bufferSize
template wwrsolverStatus_t gesvdjBatched_bufferSize<float>(wwrsolverDnHandle_t, wwrsolverEigMode_t,
                                                           int, int, const float *, int,
                                                           const ComplexToRealType<float> *,
                                                           const float *, int, const float *, int,
                                                           int *, wwrsolverGesvdjInfo_t, int);
template wwrsolverStatus_t gesvdjBatched_bufferSize<double>(wwrsolverDnHandle_t, wwrsolverEigMode_t,
                                                            int, int, const double *, int,
                                                            const ComplexToRealType<double> *,
                                                            const double *, int, const double *,
                                                            int, int *, wwrsolverGesvdjInfo_t, int);
template wwrsolverStatus_t gesvdjBatched_bufferSize<wwrFloatComplex>(
    wwrsolverDnHandle_t, wwrsolverEigMode_t, int, int, const wwrFloatComplex *, int,
    const ComplexToRealType<wwrFloatComplex> *, const wwrFloatComplex *, int,
    const wwrFloatComplex *, int, int *, wwrsolverGesvdjInfo_t, int);
template wwrsolverStatus_t gesvdjBatched_bufferSize<wwrDoubleComplex>(
    wwrsolverDnHandle_t, wwrsolverEigMode_t, int, int, const wwrDoubleComplex *, int,
    const ComplexToRealType<wwrDoubleComplex> *, const wwrDoubleComplex *, int,
    const wwrDoubleComplex *, int, int *, wwrsolverGesvdjInfo_t, int);

// Function: gesvdjBatched
template wwrsolverStatus_t gesvdjBatched<float>(wwrsolverDnHandle_t, wwrsolverEigMode_t, int, int,
                                                float *, int, ComplexToRealType<float> *, float *,
                                                int, float *, int, float *, int, int *,
                                                wwrsolverGesvdjInfo_t, int);
template wwrsolverStatus_t gesvdjBatched<double>(wwrsolverDnHandle_t, wwrsolverEigMode_t, int, int,
                                                 double *, int, ComplexToRealType<double> *,
                                                 double *, int, double *, int, double *, int, int *,
                                                 wwrsolverGesvdjInfo_t, int);
template wwrsolverStatus_t gesvdjBatched<wwrFloatComplex>(wwrsolverDnHandle_t, wwrsolverEigMode_t,
                                                          int, int, wwrFloatComplex *, int,
                                                          ComplexToRealType<wwrFloatComplex> *,
                                                          wwrFloatComplex *, int, wwrFloatComplex *,
                                                          int, wwrFloatComplex *, int, int *,
                                                          wwrsolverGesvdjInfo_t, int);
template wwrsolverStatus_t
gesvdjBatched<wwrDoubleComplex>(wwrsolverDnHandle_t, wwrsolverEigMode_t, int, int,
                                wwrDoubleComplex *, int, ComplexToRealType<wwrDoubleComplex> *,
                                wwrDoubleComplex *, int, wwrDoubleComplex *, int,
                                wwrDoubleComplex *, int, int *, wwrsolverGesvdjInfo_t, int);

// Function: sygvd_bufferSize
template wwrsolverStatus_t sygvd_bufferSize<float>(wwrsolverDnHandle_t, wwrsolverEigType_t,
                                                   wwrsolverEigMode_t, wwrblasFillMode_t, int,
                                                   const float *, int, const float *, int,
                                                   const float *, int *);
template wwrsolverStatus_t sygvd_bufferSize<double>(wwrsolverDnHandle_t, wwrsolverEigType_t,
                                                    wwrsolverEigMode_t, wwrblasFillMode_t, int,
                                                    const double *, int, const double *, int,
                                                    const double *, int *);

// Function: sygvd
template wwrsolverStatus_t sygvd<float>(wwrsolverDnHandle_t, wwrsolverEigType_t, wwrsolverEigMode_t,
                                        wwrblasFillMode_t, int, float *, int, float *, int, float *,
                                        float *, int, int *);
template wwrsolverStatus_t sygvd<double>(wwrsolverDnHandle_t, wwrsolverEigType_t,
                                         wwrsolverEigMode_t, wwrblasFillMode_t, int, double *, int,
                                         double *, int, double *, double *, int, int *);

// Function: hegvd_bufferSize
template wwrsolverStatus_t
hegvd_bufferSize<wwrFloatComplex>(wwrsolverDnHandle_t, wwrsolverEigType_t, wwrsolverEigMode_t,
                                  wwrblasFillMode_t, int, const wwrFloatComplex *, int,
                                  const wwrFloatComplex *, int,
                                  const ComplexToRealType<wwrFloatComplex> *, int *);
template wwrsolverStatus_t
hegvd_bufferSize<wwrDoubleComplex>(wwrsolverDnHandle_t, wwrsolverEigType_t, wwrsolverEigMode_t,
                                   wwrblasFillMode_t, int, const wwrDoubleComplex *, int,
                                   const wwrDoubleComplex *, int,
                                   const ComplexToRealType<wwrDoubleComplex> *, int *);

// Function: hegvd
template wwrsolverStatus_t hegvd<wwrFloatComplex>(wwrsolverDnHandle_t, wwrsolverEigType_t,
                                                  wwrsolverEigMode_t, wwrblasFillMode_t, int,
                                                  wwrFloatComplex *, int, wwrFloatComplex *, int,
                                                  ComplexToRealType<wwrFloatComplex> *,
                                                  wwrFloatComplex *, int, int *);
template wwrsolverStatus_t hegvd<wwrDoubleComplex>(wwrsolverDnHandle_t, wwrsolverEigType_t,
                                                   wwrsolverEigMode_t, wwrblasFillMode_t, int,
                                                   wwrDoubleComplex *, int, wwrDoubleComplex *, int,
                                                   ComplexToRealType<wwrDoubleComplex> *,
                                                   wwrDoubleComplex *, int, int *);

// Function: sygvj_bufferSize
template wwrsolverStatus_t sygvj_bufferSize<float>(wwrsolverDnHandle_t, wwrsolverEigType_t,
                                                   wwrsolverEigMode_t, wwrblasFillMode_t, int,
                                                   const float *, int, const float *, int,
                                                   const float *, int *, wwrsolverSyevjInfo_t);
template wwrsolverStatus_t sygvj_bufferSize<double>(wwrsolverDnHandle_t, wwrsolverEigType_t,
                                                    wwrsolverEigMode_t, wwrblasFillMode_t, int,
                                                    const double *, int, const double *, int,
                                                    const double *, int *, wwrsolverSyevjInfo_t);

// Function: sygvj
template wwrsolverStatus_t sygvj<float>(wwrsolverDnHandle_t, wwrsolverEigType_t, wwrsolverEigMode_t,
                                        wwrblasFillMode_t, int, float *, int, float *, int, float *,
                                        float *, int, int *, wwrsolverSyevjInfo_t);
template wwrsolverStatus_t sygvj<double>(wwrsolverDnHandle_t, wwrsolverEigType_t,
                                         wwrsolverEigMode_t, wwrblasFillMode_t, int, double *, int,
                                         double *, int, double *, double *, int, int *,
                                         wwrsolverSyevjInfo_t);

// Function: hegvj_bufferSize
template wwrsolverStatus_t hegvj_bufferSize<wwrFloatComplex>(
    wwrsolverDnHandle_t, wwrsolverEigType_t, wwrsolverEigMode_t, wwrblasFillMode_t, int,
    const wwrFloatComplex *, int, const wwrFloatComplex *, int,
    const ComplexToRealType<wwrFloatComplex> *, int *, wwrsolverSyevjInfo_t);
template wwrsolverStatus_t hegvj_bufferSize<wwrDoubleComplex>(
    wwrsolverDnHandle_t, wwrsolverEigType_t, wwrsolverEigMode_t, wwrblasFillMode_t, int,
    const wwrDoubleComplex *, int, const wwrDoubleComplex *, int,
    const ComplexToRealType<wwrDoubleComplex> *, int *, wwrsolverSyevjInfo_t);

// Function: hegvj
template wwrsolverStatus_t hegvj<wwrFloatComplex>(wwrsolverDnHandle_t, wwrsolverEigType_t,
                                                  wwrsolverEigMode_t, wwrblasFillMode_t, int,
                                                  wwrFloatComplex *, int, wwrFloatComplex *, int,
                                                  ComplexToRealType<wwrFloatComplex> *,
                                                  wwrFloatComplex *, int, int *,
                                                  wwrsolverSyevjInfo_t);
template wwrsolverStatus_t hegvj<wwrDoubleComplex>(wwrsolverDnHandle_t, wwrsolverEigType_t,
                                                   wwrsolverEigMode_t, wwrblasFillMode_t, int,
                                                   wwrDoubleComplex *, int, wwrDoubleComplex *, int,
                                                   ComplexToRealType<wwrDoubleComplex> *,
                                                   wwrDoubleComplex *, int, int *,
                                                   wwrsolverSyevjInfo_t);

// Function: sygvdx_bufferSize
template wwrsolverStatus_t sygvdx_bufferSize<float>(wwrsolverDnHandle_t, wwrsolverEigType_t,
                                                    wwrsolverEigMode_t, wwrsolverEigRange_t,
                                                    wwrblasFillMode_t, int, const float *, int,
                                                    const float *, int, float, float, int, int,
                                                    int *, const float *, int *);
template wwrsolverStatus_t sygvdx_bufferSize<double>(wwrsolverDnHandle_t, wwrsolverEigType_t,
                                                     wwrsolverEigMode_t, wwrsolverEigRange_t,
                                                     wwrblasFillMode_t, int, const double *, int,
                                                     const double *, int, double, double, int, int,
                                                     int *, const double *, int *);

// Function: sygvdx
template wwrsolverStatus_t sygvdx<float>(wwrsolverDnHandle_t, wwrsolverEigType_t,
                                         wwrsolverEigMode_t, wwrsolverEigRange_t, wwrblasFillMode_t,
                                         int, float *, int, float *, int, float, float, int, int,
                                         int *, float *, float *, int, int *);
template wwrsolverStatus_t sygvdx<double>(wwrsolverDnHandle_t, wwrsolverEigType_t,
                                          wwrsolverEigMode_t, wwrsolverEigRange_t,
                                          wwrblasFillMode_t, int, double *, int, double *, int,
                                          double, double, int, int, int *, double *, double *, int,
                                          int *);

// Function: hegvdx_bufferSize
template wwrsolverStatus_t hegvdx_bufferSize<wwrFloatComplex>(
    wwrsolverDnHandle_t, wwrsolverEigType_t, wwrsolverEigMode_t, wwrsolverEigRange_t,
    wwrblasFillMode_t, int, const wwrFloatComplex *, int, const wwrFloatComplex *, int,
    ComplexToRealType<wwrFloatComplex>, ComplexToRealType<wwrFloatComplex>, int, int, int *,
    const ComplexToRealType<wwrFloatComplex> *, int *);
template wwrsolverStatus_t hegvdx_bufferSize<wwrDoubleComplex>(
    wwrsolverDnHandle_t, wwrsolverEigType_t, wwrsolverEigMode_t, wwrsolverEigRange_t,
    wwrblasFillMode_t, int, const wwrDoubleComplex *, int, const wwrDoubleComplex *, int,
    ComplexToRealType<wwrDoubleComplex>, ComplexToRealType<wwrDoubleComplex>, int, int, int *,
    const ComplexToRealType<wwrDoubleComplex> *, int *);

// Function: hegvdx
template wwrsolverStatus_t
hegvdx<wwrFloatComplex>(wwrsolverDnHandle_t, wwrsolverEigType_t, wwrsolverEigMode_t,
                        wwrsolverEigRange_t, wwrblasFillMode_t, int, wwrFloatComplex *, int,
                        wwrFloatComplex *, int, ComplexToRealType<wwrFloatComplex>,
                        ComplexToRealType<wwrFloatComplex>, int, int, int *,
                        ComplexToRealType<wwrFloatComplex> *, wwrFloatComplex *, int, int *);
template wwrsolverStatus_t
hegvdx<wwrDoubleComplex>(wwrsolverDnHandle_t, wwrsolverEigType_t, wwrsolverEigMode_t,
                         wwrsolverEigRange_t, wwrblasFillMode_t, int, wwrDoubleComplex *, int,
                         wwrDoubleComplex *, int, ComplexToRealType<wwrDoubleComplex>,
                         ComplexToRealType<wwrDoubleComplex>, int, int, int *,
                         ComplexToRealType<wwrDoubleComplex> *, wwrDoubleComplex *, int, int *);

// Function: gesvdaStridedBatched_bufferSize
template wwrsolverStatus_t gesvdaStridedBatched_bufferSize<float>(
    wwrsolverDnHandle_t, wwrsolverEigMode_t, int, int, int, const float *, int, long long int,
    const ComplexToRealType<float> *, long long int, const float *, int, long long int,
    const float *, int, long long int, int *, int);
template wwrsolverStatus_t gesvdaStridedBatched_bufferSize<double>(
    wwrsolverDnHandle_t, wwrsolverEigMode_t, int, int, int, const double *, int, long long int,
    const ComplexToRealType<double> *, long long int, const double *, int, long long int,
    const double *, int, long long int, int *, int);
template wwrsolverStatus_t gesvdaStridedBatched_bufferSize<wwrFloatComplex>(
    wwrsolverDnHandle_t, wwrsolverEigMode_t, int, int, int, const wwrFloatComplex *, int,
    long long int, const ComplexToRealType<wwrFloatComplex> *, long long int,
    const wwrFloatComplex *, int, long long int, const wwrFloatComplex *, int, long long int, int *,
    int);
template wwrsolverStatus_t gesvdaStridedBatched_bufferSize<wwrDoubleComplex>(
    wwrsolverDnHandle_t, wwrsolverEigMode_t, int, int, int, const wwrDoubleComplex *, int,
    long long int, const ComplexToRealType<wwrDoubleComplex> *, long long int,
    const wwrDoubleComplex *, int, long long int, const wwrDoubleComplex *, int, long long int,
    int *, int);

// Function: gesvdaStridedBatched
template wwrsolverStatus_t gesvdaStridedBatched<float>(wwrsolverDnHandle_t, wwrsolverEigMode_t, int,
                                                       int, int, const float *, int, long long int,
                                                       ComplexToRealType<float> *, long long int,
                                                       float *, int, long long int, float *, int,
                                                       long long int, float *, int, int *, double *,
                                                       int);
template wwrsolverStatus_t gesvdaStridedBatched<double>(wwrsolverDnHandle_t, wwrsolverEigMode_t,
                                                        int, int, int, const double *, int,
                                                        long long int, ComplexToRealType<double> *,
                                                        long long int, double *, int, long long int,
                                                        double *, int, long long int, double *, int,
                                                        int *, double *, int);
template wwrsolverStatus_t
gesvdaStridedBatched<wwrFloatComplex>(wwrsolverDnHandle_t, wwrsolverEigMode_t, int, int, int,
                                      const wwrFloatComplex *, int, long long int,
                                      ComplexToRealType<wwrFloatComplex> *, long long int,
                                      wwrFloatComplex *, int, long long int, wwrFloatComplex *, int,
                                      long long int, wwrFloatComplex *, int, int *, double *, int);
template wwrsolverStatus_t gesvdaStridedBatched<wwrDoubleComplex>(
    wwrsolverDnHandle_t, wwrsolverEigMode_t, int, int, int, const wwrDoubleComplex *, int,
    long long int, ComplexToRealType<wwrDoubleComplex> *, long long int, wwrDoubleComplex *, int,
    long long int, wwrDoubleComplex *, int, long long int, wwrDoubleComplex *, int, int *, double *,
    int);

// Function: sytrd_bufferSize
template wwrsolverStatus_t sytrd_bufferSize<float>(wwrsolverDnHandle_t, wwrblasFillMode_t, int,
                                                   const float *, int, const float *, const float *,
                                                   const float *, int *);
template wwrsolverStatus_t sytrd_bufferSize<double>(wwrsolverDnHandle_t, wwrblasFillMode_t, int,
                                                    const double *, int, const double *,
                                                    const double *, const double *, int *);

// Function: sytrd
template wwrsolverStatus_t sytrd<float>(wwrsolverDnHandle_t, wwrblasFillMode_t, int, float *, int,
                                        float *, float *, float *, float *, int, int *);
template wwrsolverStatus_t sytrd<double>(wwrsolverDnHandle_t, wwrblasFillMode_t, int, double *, int,
                                         double *, double *, double *, double *, int, int *);

// Function: hetrd_bufferSize
template wwrsolverStatus_t hetrd_bufferSize<wwrFloatComplex>(
    wwrsolverDnHandle_t, wwrblasFillMode_t, int, const wwrFloatComplex *, int,
    const ComplexToRealType<wwrFloatComplex> *, const ComplexToRealType<wwrFloatComplex> *,
    const wwrFloatComplex *, int *);
template wwrsolverStatus_t hetrd_bufferSize<wwrDoubleComplex>(
    wwrsolverDnHandle_t, wwrblasFillMode_t, int, const wwrDoubleComplex *, int,
    const ComplexToRealType<wwrDoubleComplex> *, const ComplexToRealType<wwrDoubleComplex> *,
    const wwrDoubleComplex *, int *);

// Function: hetrd
template wwrsolverStatus_t hetrd<wwrFloatComplex>(wwrsolverDnHandle_t, wwrblasFillMode_t, int,
                                                  wwrFloatComplex *, int,
                                                  ComplexToRealType<wwrFloatComplex> *,
                                                  ComplexToRealType<wwrFloatComplex> *,
                                                  wwrFloatComplex *, wwrFloatComplex *, int, int *);
template wwrsolverStatus_t hetrd<wwrDoubleComplex>(wwrsolverDnHandle_t, wwrblasFillMode_t, int,
                                                   wwrDoubleComplex *, int,
                                                   ComplexToRealType<wwrDoubleComplex> *,
                                                   ComplexToRealType<wwrDoubleComplex> *,
                                                   wwrDoubleComplex *, wwrDoubleComplex *, int,
                                                   int *);

// Function: orgtr_bufferSize
template wwrsolverStatus_t orgtr_bufferSize<float>(wwrsolverDnHandle_t, wwrblasFillMode_t, int,
                                                   const float *, int, const float *, int *);
template wwrsolverStatus_t orgtr_bufferSize<double>(wwrsolverDnHandle_t, wwrblasFillMode_t, int,
                                                    const double *, int, const double *, int *);

// Function: orgtr
template wwrsolverStatus_t orgtr<float>(wwrsolverDnHandle_t, wwrblasFillMode_t, int, float *, int,
                                        const float *, float *, int, int *);
template wwrsolverStatus_t orgtr<double>(wwrsolverDnHandle_t, wwrblasFillMode_t, int, double *, int,
                                         const double *, double *, int, int *);

// Function: ungtr_bufferSize
template wwrsolverStatus_t ungtr_bufferSize<wwrFloatComplex>(wwrsolverDnHandle_t, wwrblasFillMode_t,
                                                             int, const wwrFloatComplex *, int,
                                                             const wwrFloatComplex *, int *);
template wwrsolverStatus_t ungtr_bufferSize<wwrDoubleComplex>(wwrsolverDnHandle_t,
                                                              wwrblasFillMode_t, int,
                                                              const wwrDoubleComplex *, int,
                                                              const wwrDoubleComplex *, int *);

// Function: ungtr
template wwrsolverStatus_t ungtr<wwrFloatComplex>(wwrsolverDnHandle_t, wwrblasFillMode_t, int,
                                                  wwrFloatComplex *, int, const wwrFloatComplex *,
                                                  wwrFloatComplex *, int, int *);
template wwrsolverStatus_t ungtr<wwrDoubleComplex>(wwrsolverDnHandle_t, wwrblasFillMode_t, int,
                                                   wwrDoubleComplex *, int,
                                                   const wwrDoubleComplex *, wwrDoubleComplex *,
                                                   int, int *);

// Function: ormtr_bufferSize
template wwrsolverStatus_t ormtr_bufferSize<float>(wwrsolverDnHandle_t, wwrblasSideMode_t,
                                                   wwrblasFillMode_t, wwrblasOperation_t, int, int,
                                                   const float *, int, const float *, const float *,
                                                   int, int *);
template wwrsolverStatus_t ormtr_bufferSize<double>(wwrsolverDnHandle_t, wwrblasSideMode_t,
                                                    wwrblasFillMode_t, wwrblasOperation_t, int, int,
                                                    const double *, int, const double *,
                                                    const double *, int, int *);

// Function: ormtr
template wwrsolverStatus_t ormtr<float>(wwrsolverDnHandle_t, wwrblasSideMode_t, wwrblasFillMode_t,
                                        wwrblasOperation_t, int, int, float *, int, float *,
                                        float *, int, float *, int, int *);
template wwrsolverStatus_t ormtr<double>(wwrsolverDnHandle_t, wwrblasSideMode_t, wwrblasFillMode_t,
                                         wwrblasOperation_t, int, int, double *, int, double *,
                                         double *, int, double *, int, int *);

// Function: unmtr_bufferSize
template wwrsolverStatus_t unmtr_bufferSize<wwrFloatComplex>(wwrsolverDnHandle_t, wwrblasSideMode_t,
                                                             wwrblasFillMode_t, wwrblasOperation_t,
                                                             int, int, const wwrFloatComplex *, int,
                                                             const wwrFloatComplex *,
                                                             const wwrFloatComplex *, int, int *);
template wwrsolverStatus_t
unmtr_bufferSize<wwrDoubleComplex>(wwrsolverDnHandle_t, wwrblasSideMode_t, wwrblasFillMode_t,
                                   wwrblasOperation_t, int, int, const wwrDoubleComplex *, int,
                                   const wwrDoubleComplex *, const wwrDoubleComplex *, int, int *);

// Function: unmtr
template wwrsolverStatus_t unmtr<wwrFloatComplex>(wwrsolverDnHandle_t, wwrblasSideMode_t,
                                                  wwrblasFillMode_t, wwrblasOperation_t, int, int,
                                                  wwrFloatComplex *, int, wwrFloatComplex *,
                                                  wwrFloatComplex *, int, wwrFloatComplex *, int,
                                                  int *);
template wwrsolverStatus_t unmtr<wwrDoubleComplex>(wwrsolverDnHandle_t, wwrblasSideMode_t,
                                                   wwrblasFillMode_t, wwrblasOperation_t, int, int,
                                                   wwrDoubleComplex *, int, wwrDoubleComplex *,
                                                   wwrDoubleComplex *, int, wwrDoubleComplex *, int,
                                                   int *);

} // namespace wwr
