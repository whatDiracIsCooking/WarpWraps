/**
 * @file instantiations.cpp
 * @brief Explicit template instantiations for the GPU solver wrappers
 *
 * This file contains explicit template instantiations to avoid code bloat
 * from implicit instantiation at every call site. Hand-written -- see
 * math/triple_gemm for the pattern; extern template lives in the .cppm
 * partition, template here.
 */

module gpumod.wrappers.solver;

import gpumod.solver;
import gpumod.blas;
import gpumod.complex;
import gpumod.wrappers.common;

namespace gpumod {

// Function: potrf_bufferSize
template gpusolverStatus_t potrf_bufferSize<float>(gpusolverDnHandle_t, gpublasFillMode_t, int,
                                                   float *, int, int *);
template gpusolverStatus_t potrf_bufferSize<double>(gpusolverDnHandle_t, gpublasFillMode_t, int,
                                                    double *, int, int *);
template gpusolverStatus_t potrf_bufferSize<gpuFloatComplex>(gpusolverDnHandle_t, gpublasFillMode_t,
                                                             int, gpuFloatComplex *, int, int *);
template gpusolverStatus_t potrf_bufferSize<gpuDoubleComplex>(gpusolverDnHandle_t,
                                                              gpublasFillMode_t, int,
                                                              gpuDoubleComplex *, int, int *);

// Function: potrf
template gpusolverStatus_t potrf<float>(gpusolverDnHandle_t, gpublasFillMode_t, int, float *, int,
                                        float *, int, int *);
template gpusolverStatus_t potrf<double>(gpusolverDnHandle_t, gpublasFillMode_t, int, double *, int,
                                         double *, int, int *);
template gpusolverStatus_t potrf<gpuFloatComplex>(gpusolverDnHandle_t, gpublasFillMode_t, int,
                                                  gpuFloatComplex *, int, gpuFloatComplex *, int,
                                                  int *);
template gpusolverStatus_t potrf<gpuDoubleComplex>(gpusolverDnHandle_t, gpublasFillMode_t, int,
                                                   gpuDoubleComplex *, int, gpuDoubleComplex *, int,
                                                   int *);

// Function: potrs
template gpusolverStatus_t potrs<float>(gpusolverDnHandle_t, gpublasFillMode_t, int, int,
                                        const float *, int, float *, int, int *);
template gpusolverStatus_t potrs<double>(gpusolverDnHandle_t, gpublasFillMode_t, int, int,
                                         const double *, int, double *, int, int *);
template gpusolverStatus_t potrs<gpuFloatComplex>(gpusolverDnHandle_t, gpublasFillMode_t, int, int,
                                                  const gpuFloatComplex *, int, gpuFloatComplex *,
                                                  int, int *);
template gpusolverStatus_t potrs<gpuDoubleComplex>(gpusolverDnHandle_t, gpublasFillMode_t, int, int,
                                                   const gpuDoubleComplex *, int,
                                                   gpuDoubleComplex *, int, int *);

// Function: potri_bufferSize
template gpusolverStatus_t potri_bufferSize<float>(gpusolverDnHandle_t, gpublasFillMode_t, int,
                                                   float *, int, int *);
template gpusolverStatus_t potri_bufferSize<double>(gpusolverDnHandle_t, gpublasFillMode_t, int,
                                                    double *, int, int *);
template gpusolverStatus_t potri_bufferSize<gpuFloatComplex>(gpusolverDnHandle_t, gpublasFillMode_t,
                                                             int, gpuFloatComplex *, int, int *);
template gpusolverStatus_t potri_bufferSize<gpuDoubleComplex>(gpusolverDnHandle_t,
                                                              gpublasFillMode_t, int,
                                                              gpuDoubleComplex *, int, int *);

// Function: potri
template gpusolverStatus_t potri<float>(gpusolverDnHandle_t, gpublasFillMode_t, int, float *, int,
                                        float *, int, int *);
template gpusolverStatus_t potri<double>(gpusolverDnHandle_t, gpublasFillMode_t, int, double *, int,
                                         double *, int, int *);
template gpusolverStatus_t potri<gpuFloatComplex>(gpusolverDnHandle_t, gpublasFillMode_t, int,
                                                  gpuFloatComplex *, int, gpuFloatComplex *, int,
                                                  int *);
template gpusolverStatus_t potri<gpuDoubleComplex>(gpusolverDnHandle_t, gpublasFillMode_t, int,
                                                   gpuDoubleComplex *, int, gpuDoubleComplex *, int,
                                                   int *);

// Function: getrf_bufferSize
template gpusolverStatus_t getrf_bufferSize<float>(gpusolverDnHandle_t, int, int, float *, int,
                                                   int *);
template gpusolverStatus_t getrf_bufferSize<double>(gpusolverDnHandle_t, int, int, double *, int,
                                                    int *);
template gpusolverStatus_t getrf_bufferSize<gpuFloatComplex>(gpusolverDnHandle_t, int, int,
                                                             gpuFloatComplex *, int, int *);
template gpusolverStatus_t getrf_bufferSize<gpuDoubleComplex>(gpusolverDnHandle_t, int, int,
                                                              gpuDoubleComplex *, int, int *);

// Function: getrf
template gpusolverStatus_t getrf<float>(gpusolverDnHandle_t, int, int, float *, int, float *, int *,
                                        int *);
template gpusolverStatus_t getrf<double>(gpusolverDnHandle_t, int, int, double *, int, double *,
                                         int *, int *);
template gpusolverStatus_t getrf<gpuFloatComplex>(gpusolverDnHandle_t, int, int, gpuFloatComplex *,
                                                  int, gpuFloatComplex *, int *, int *);
template gpusolverStatus_t getrf<gpuDoubleComplex>(gpusolverDnHandle_t, int, int,
                                                   gpuDoubleComplex *, int, gpuDoubleComplex *,
                                                   int *, int *);

// Function: getrs
template gpusolverStatus_t getrs<float>(gpusolverDnHandle_t, gpublasOperation_t, int, int,
                                        const float *, int, const int *, float *, int, int *);
template gpusolverStatus_t getrs<double>(gpusolverDnHandle_t, gpublasOperation_t, int, int,
                                         const double *, int, const int *, double *, int, int *);
template gpusolverStatus_t getrs<gpuFloatComplex>(gpusolverDnHandle_t, gpublasOperation_t, int, int,
                                                  const gpuFloatComplex *, int, const int *,
                                                  gpuFloatComplex *, int, int *);
template gpusolverStatus_t getrs<gpuDoubleComplex>(gpusolverDnHandle_t, gpublasOperation_t, int,
                                                   int, const gpuDoubleComplex *, int, const int *,
                                                   gpuDoubleComplex *, int, int *);

// Function: geqrf_bufferSize
template gpusolverStatus_t geqrf_bufferSize<float>(gpusolverDnHandle_t, int, int, float *, int,
                                                   int *);
template gpusolverStatus_t geqrf_bufferSize<double>(gpusolverDnHandle_t, int, int, double *, int,
                                                    int *);
template gpusolverStatus_t geqrf_bufferSize<gpuFloatComplex>(gpusolverDnHandle_t, int, int,
                                                             gpuFloatComplex *, int, int *);
template gpusolverStatus_t geqrf_bufferSize<gpuDoubleComplex>(gpusolverDnHandle_t, int, int,
                                                              gpuDoubleComplex *, int, int *);

// Function: geqrf
template gpusolverStatus_t geqrf<float>(gpusolverDnHandle_t, int, int, float *, int, float *,
                                        float *, int, int *);
template gpusolverStatus_t geqrf<double>(gpusolverDnHandle_t, int, int, double *, int, double *,
                                         double *, int, int *);
template gpusolverStatus_t geqrf<gpuFloatComplex>(gpusolverDnHandle_t, int, int, gpuFloatComplex *,
                                                  int, gpuFloatComplex *, gpuFloatComplex *, int,
                                                  int *);
template gpusolverStatus_t geqrf<gpuDoubleComplex>(gpusolverDnHandle_t, int, int,
                                                   gpuDoubleComplex *, int, gpuDoubleComplex *,
                                                   gpuDoubleComplex *, int, int *);

// Function: ormqr_bufferSize
template gpusolverStatus_t ormqr_bufferSize<float>(gpusolverDnHandle_t, gpublasSideMode_t,
                                                   gpublasOperation_t, int, int, int, const float *,
                                                   int, const float *, const float *, int, int *);
template gpusolverStatus_t ormqr_bufferSize<double>(gpusolverDnHandle_t, gpublasSideMode_t,
                                                    gpublasOperation_t, int, int, int,
                                                    const double *, int, const double *,
                                                    const double *, int, int *);

// Function: ormqr
template gpusolverStatus_t ormqr<float>(gpusolverDnHandle_t, gpublasSideMode_t, gpublasOperation_t,
                                        int, int, int, const float *, int, const float *, float *,
                                        int, float *, int, int *);
template gpusolverStatus_t ormqr<double>(gpusolverDnHandle_t, gpublasSideMode_t, gpublasOperation_t,
                                         int, int, int, const double *, int, const double *,
                                         double *, int, double *, int, int *);

// Function: unmqr_bufferSize
template gpusolverStatus_t unmqr_bufferSize<gpuFloatComplex>(gpusolverDnHandle_t, gpublasSideMode_t,
                                                             gpublasOperation_t, int, int, int,
                                                             const gpuFloatComplex *, int,
                                                             const gpuFloatComplex *,
                                                             const gpuFloatComplex *, int, int *);
template gpusolverStatus_t
unmqr_bufferSize<gpuDoubleComplex>(gpusolverDnHandle_t, gpublasSideMode_t, gpublasOperation_t, int,
                                   int, int, const gpuDoubleComplex *, int,
                                   const gpuDoubleComplex *, const gpuDoubleComplex *, int, int *);

// Function: unmqr
template gpusolverStatus_t unmqr<gpuFloatComplex>(gpusolverDnHandle_t, gpublasSideMode_t,
                                                  gpublasOperation_t, int, int, int,
                                                  const gpuFloatComplex *, int,
                                                  const gpuFloatComplex *, gpuFloatComplex *, int,
                                                  gpuFloatComplex *, int, int *);
template gpusolverStatus_t unmqr<gpuDoubleComplex>(gpusolverDnHandle_t, gpublasSideMode_t,
                                                   gpublasOperation_t, int, int, int,
                                                   const gpuDoubleComplex *, int,
                                                   const gpuDoubleComplex *, gpuDoubleComplex *,
                                                   int, gpuDoubleComplex *, int, int *);

// Function: gels_bufferSize
template gpusolverStatus_t gels_bufferSize<float>(gpusolverDnHandle_t, int, int, int, float *, int,
                                                  float *, int, float *, int, void *,
                                                  std::size_t *);
template gpusolverStatus_t gels_bufferSize<double>(gpusolverDnHandle_t, int, int, int, double *,
                                                   int, double *, int, double *, int, void *,
                                                   std::size_t *);
template gpusolverStatus_t gels_bufferSize<gpuFloatComplex>(gpusolverDnHandle_t, int, int, int,
                                                            gpuFloatComplex *, int,
                                                            gpuFloatComplex *, int,
                                                            gpuFloatComplex *, int, void *,
                                                            std::size_t *);
template gpusolverStatus_t gels_bufferSize<gpuDoubleComplex>(gpusolverDnHandle_t, int, int, int,
                                                             gpuDoubleComplex *, int,
                                                             gpuDoubleComplex *, int,
                                                             gpuDoubleComplex *, int, void *,
                                                             std::size_t *);

// Function: gels
template gpusolverStatus_t gels<float>(gpusolverDnHandle_t, int, int, int, float *, int, float *,
                                       int, float *, int, void *, std::size_t, int *, int *);
template gpusolverStatus_t gels<double>(gpusolverDnHandle_t, int, int, int, double *, int, double *,
                                        int, double *, int, void *, std::size_t, int *, int *);
template gpusolverStatus_t gels<gpuFloatComplex>(gpusolverDnHandle_t, int, int, int,
                                                 gpuFloatComplex *, int, gpuFloatComplex *, int,
                                                 gpuFloatComplex *, int, void *, std::size_t, int *,
                                                 int *);
template gpusolverStatus_t gels<gpuDoubleComplex>(gpusolverDnHandle_t, int, int, int,
                                                  gpuDoubleComplex *, int, gpuDoubleComplex *, int,
                                                  gpuDoubleComplex *, int, void *, std::size_t,
                                                  int *, int *);

// Function: gesv_bufferSize
template gpusolverStatus_t gesv_bufferSize<float>(gpusolverDnHandle_t, int, int, float *, int,
                                                  int *, float *, int, float *, int, void *,
                                                  std::size_t *);
template gpusolverStatus_t gesv_bufferSize<double>(gpusolverDnHandle_t, int, int, double *, int,
                                                   int *, double *, int, double *, int, void *,
                                                   std::size_t *);
template gpusolverStatus_t gesv_bufferSize<gpuFloatComplex>(gpusolverDnHandle_t, int, int,
                                                            gpuFloatComplex *, int, int *,
                                                            gpuFloatComplex *, int,
                                                            gpuFloatComplex *, int, void *,
                                                            std::size_t *);
template gpusolverStatus_t gesv_bufferSize<gpuDoubleComplex>(gpusolverDnHandle_t, int, int,
                                                             gpuDoubleComplex *, int, int *,
                                                             gpuDoubleComplex *, int,
                                                             gpuDoubleComplex *, int, void *,
                                                             std::size_t *);

// Function: gesv
template gpusolverStatus_t gesv<float>(gpusolverDnHandle_t, int, int, float *, int, int *, float *,
                                       int, float *, int, void *, std::size_t, int *, int *);
template gpusolverStatus_t gesv<double>(gpusolverDnHandle_t, int, int, double *, int, int *,
                                        double *, int, double *, int, void *, std::size_t, int *,
                                        int *);
template gpusolverStatus_t gesv<gpuFloatComplex>(gpusolverDnHandle_t, int, int, gpuFloatComplex *,
                                                 int, int *, gpuFloatComplex *, int,
                                                 gpuFloatComplex *, int, void *, std::size_t, int *,
                                                 int *);
template gpusolverStatus_t gesv<gpuDoubleComplex>(gpusolverDnHandle_t, int, int, gpuDoubleComplex *,
                                                  int, int *, gpuDoubleComplex *, int,
                                                  gpuDoubleComplex *, int, void *, std::size_t,
                                                  int *, int *);

// Function: potrfBatched
template gpusolverStatus_t potrfBatched<float>(gpusolverDnHandle_t, gpublasFillMode_t, int,
                                               float *[], int, int *, int);
template gpusolverStatus_t potrfBatched<double>(gpusolverDnHandle_t, gpublasFillMode_t, int,
                                                double *[], int, int *, int);
template gpusolverStatus_t potrfBatched<gpuFloatComplex>(gpusolverDnHandle_t, gpublasFillMode_t,
                                                         int, gpuFloatComplex *[], int, int *, int);
template gpusolverStatus_t potrfBatched<gpuDoubleComplex>(gpusolverDnHandle_t, gpublasFillMode_t,
                                                          int, gpuDoubleComplex *[], int, int *,
                                                          int);

// Function: potrsBatched
template gpusolverStatus_t potrsBatched<float>(gpusolverDnHandle_t, gpublasFillMode_t, int, int,
                                               float *[], int, float *[], int, int *, int);
template gpusolverStatus_t potrsBatched<double>(gpusolverDnHandle_t, gpublasFillMode_t, int, int,
                                                double *[], int, double *[], int, int *, int);
template gpusolverStatus_t potrsBatched<gpuFloatComplex>(gpusolverDnHandle_t, gpublasFillMode_t,
                                                         int, int, gpuFloatComplex *[], int,
                                                         gpuFloatComplex *[], int, int *, int);
template gpusolverStatus_t potrsBatched<gpuDoubleComplex>(gpusolverDnHandle_t, gpublasFillMode_t,
                                                          int, int, gpuDoubleComplex *[], int,
                                                          gpuDoubleComplex *[], int, int *, int);

// Function: sytrf_bufferSize
template gpusolverStatus_t sytrf_bufferSize<float>(gpusolverDnHandle_t, int, float *, int, int *);
template gpusolverStatus_t sytrf_bufferSize<double>(gpusolverDnHandle_t, int, double *, int, int *);
template gpusolverStatus_t sytrf_bufferSize<gpuFloatComplex>(gpusolverDnHandle_t, int,
                                                             gpuFloatComplex *, int, int *);
template gpusolverStatus_t sytrf_bufferSize<gpuDoubleComplex>(gpusolverDnHandle_t, int,
                                                              gpuDoubleComplex *, int, int *);

// Function: sytrf
template gpusolverStatus_t sytrf<float>(gpusolverDnHandle_t, gpublasFillMode_t, int, float *, int,
                                        int *, float *, int, int *);
template gpusolverStatus_t sytrf<double>(gpusolverDnHandle_t, gpublasFillMode_t, int, double *, int,
                                         int *, double *, int, int *);
template gpusolverStatus_t sytrf<gpuFloatComplex>(gpusolverDnHandle_t, gpublasFillMode_t, int,
                                                  gpuFloatComplex *, int, int *, gpuFloatComplex *,
                                                  int, int *);
template gpusolverStatus_t sytrf<gpuDoubleComplex>(gpusolverDnHandle_t, gpublasFillMode_t, int,
                                                   gpuDoubleComplex *, int, int *,
                                                   gpuDoubleComplex *, int, int *);

// Function: gebrd_bufferSize
template gpusolverStatus_t gebrd_bufferSize<float>(gpusolverDnHandle_t, int, int, int *);
template gpusolverStatus_t gebrd_bufferSize<double>(gpusolverDnHandle_t, int, int, int *);
template gpusolverStatus_t gebrd_bufferSize<gpuFloatComplex>(gpusolverDnHandle_t, int, int, int *);
template gpusolverStatus_t gebrd_bufferSize<gpuDoubleComplex>(gpusolverDnHandle_t, int, int, int *);

// Function: gebrd
template gpusolverStatus_t gebrd<float>(gpusolverDnHandle_t, int, int, float *, int,
                                        ComplexToRealType<float> *, ComplexToRealType<float> *,
                                        float *, float *, float *, int, int *);
template gpusolverStatus_t gebrd<double>(gpusolverDnHandle_t, int, int, double *, int,
                                         ComplexToRealType<double> *, ComplexToRealType<double> *,
                                         double *, double *, double *, int, int *);
template gpusolverStatus_t gebrd<gpuFloatComplex>(gpusolverDnHandle_t, int, int, gpuFloatComplex *,
                                                  int, ComplexToRealType<gpuFloatComplex> *,
                                                  ComplexToRealType<gpuFloatComplex> *,
                                                  gpuFloatComplex *, gpuFloatComplex *,
                                                  gpuFloatComplex *, int, int *);
template gpusolverStatus_t gebrd<gpuDoubleComplex>(gpusolverDnHandle_t, int, int,
                                                   gpuDoubleComplex *, int,
                                                   ComplexToRealType<gpuDoubleComplex> *,
                                                   ComplexToRealType<gpuDoubleComplex> *,
                                                   gpuDoubleComplex *, gpuDoubleComplex *,
                                                   gpuDoubleComplex *, int, int *);

// Function: orgqr_bufferSize
template gpusolverStatus_t orgqr_bufferSize<float>(gpusolverDnHandle_t, int, int, int,
                                                   const float *, int, const float *, int *);
template gpusolverStatus_t orgqr_bufferSize<double>(gpusolverDnHandle_t, int, int, int,
                                                    const double *, int, const double *, int *);

// Function: ungqr_bufferSize
template gpusolverStatus_t ungqr_bufferSize<gpuFloatComplex>(gpusolverDnHandle_t, int, int, int,
                                                             const gpuFloatComplex *, int,
                                                             const gpuFloatComplex *, int *);
template gpusolverStatus_t ungqr_bufferSize<gpuDoubleComplex>(gpusolverDnHandle_t, int, int, int,
                                                              const gpuDoubleComplex *, int,
                                                              const gpuDoubleComplex *, int *);

// Function: orgqr
template gpusolverStatus_t orgqr<float>(gpusolverDnHandle_t, int, int, int, float *, int,
                                        const float *, float *, int, int *);
template gpusolverStatus_t orgqr<double>(gpusolverDnHandle_t, int, int, int, double *, int,
                                         const double *, double *, int, int *);

// Function: ungqr
template gpusolverStatus_t ungqr<gpuFloatComplex>(gpusolverDnHandle_t, int, int, int,
                                                  gpuFloatComplex *, int, const gpuFloatComplex *,
                                                  gpuFloatComplex *, int, int *);
template gpusolverStatus_t ungqr<gpuDoubleComplex>(gpusolverDnHandle_t, int, int, int,
                                                   gpuDoubleComplex *, int,
                                                   const gpuDoubleComplex *, gpuDoubleComplex *,
                                                   int, int *);

// Function: orgbr_bufferSize
template gpusolverStatus_t orgbr_bufferSize<float>(gpusolverDnHandle_t, gpublasSideMode_t, int, int,
                                                   int, const float *, int, const float *, int *);
template gpusolverStatus_t orgbr_bufferSize<double>(gpusolverDnHandle_t, gpublasSideMode_t, int,
                                                    int, int, const double *, int, const double *,
                                                    int *);

// Function: ungbr_bufferSize
template gpusolverStatus_t ungbr_bufferSize<gpuFloatComplex>(gpusolverDnHandle_t, gpublasSideMode_t,
                                                             int, int, int, const gpuFloatComplex *,
                                                             int, const gpuFloatComplex *, int *);
template gpusolverStatus_t ungbr_bufferSize<gpuDoubleComplex>(gpusolverDnHandle_t,
                                                              gpublasSideMode_t, int, int, int,
                                                              const gpuDoubleComplex *, int,
                                                              const gpuDoubleComplex *, int *);

// Function: orgbr
template gpusolverStatus_t orgbr<float>(gpusolverDnHandle_t, gpublasSideMode_t, int, int, int,
                                        float *, int, const float *, float *, int, int *);
template gpusolverStatus_t orgbr<double>(gpusolverDnHandle_t, gpublasSideMode_t, int, int, int,
                                         double *, int, const double *, double *, int, int *);

// Function: ungbr
template gpusolverStatus_t ungbr<gpuFloatComplex>(gpusolverDnHandle_t, gpublasSideMode_t, int, int,
                                                  int, gpuFloatComplex *, int,
                                                  const gpuFloatComplex *, gpuFloatComplex *, int,
                                                  int *);
template gpusolverStatus_t ungbr<gpuDoubleComplex>(gpusolverDnHandle_t, gpublasSideMode_t, int, int,
                                                   int, gpuDoubleComplex *, int,
                                                   const gpuDoubleComplex *, gpuDoubleComplex *,
                                                   int, int *);

// Function: gesvd_bufferSize
template gpusolverStatus_t gesvd_bufferSize<float>(gpusolverDnHandle_t, int, int, int *);
template gpusolverStatus_t gesvd_bufferSize<double>(gpusolverDnHandle_t, int, int, int *);
template gpusolverStatus_t gesvd_bufferSize<gpuFloatComplex>(gpusolverDnHandle_t, int, int, int *);
template gpusolverStatus_t gesvd_bufferSize<gpuDoubleComplex>(gpusolverDnHandle_t, int, int, int *);

// Function: gesvd
template gpusolverStatus_t gesvd<float>(gpusolverDnHandle_t, signed char, signed char, int, int,
                                        float *, int, ComplexToRealType<float> *, float *, int,
                                        float *, int, float *, int, ComplexToRealType<float> *,
                                        int *);
template gpusolverStatus_t gesvd<double>(gpusolverDnHandle_t, signed char, signed char, int, int,
                                         double *, int, ComplexToRealType<double> *, double *, int,
                                         double *, int, double *, int, ComplexToRealType<double> *,
                                         int *);
template gpusolverStatus_t gesvd<gpuFloatComplex>(gpusolverDnHandle_t, signed char, signed char,
                                                  int, int, gpuFloatComplex *, int,
                                                  ComplexToRealType<gpuFloatComplex> *,
                                                  gpuFloatComplex *, int, gpuFloatComplex *, int,
                                                  gpuFloatComplex *, int,
                                                  ComplexToRealType<gpuFloatComplex> *, int *);
template gpusolverStatus_t gesvd<gpuDoubleComplex>(gpusolverDnHandle_t, signed char, signed char,
                                                   int, int, gpuDoubleComplex *, int,
                                                   ComplexToRealType<gpuDoubleComplex> *,
                                                   gpuDoubleComplex *, int, gpuDoubleComplex *, int,
                                                   gpuDoubleComplex *, int,
                                                   ComplexToRealType<gpuDoubleComplex> *, int *);

// Function: syevd_bufferSize
template gpusolverStatus_t syevd_bufferSize<float>(gpusolverDnHandle_t, gpusolverEigMode_t,
                                                   gpublasFillMode_t, int, const float *, int,
                                                   const float *, int *);
template gpusolverStatus_t syevd_bufferSize<double>(gpusolverDnHandle_t, gpusolverEigMode_t,
                                                    gpublasFillMode_t, int, const double *, int,
                                                    const double *, int *);

// Function: syevd
template gpusolverStatus_t syevd<float>(gpusolverDnHandle_t, gpusolverEigMode_t, gpublasFillMode_t,
                                        int, float *, int, float *, float *, int, int *);
template gpusolverStatus_t syevd<double>(gpusolverDnHandle_t, gpusolverEigMode_t, gpublasFillMode_t,
                                         int, double *, int, double *, double *, int, int *);

// Function: syevdx_bufferSize
template gpusolverStatus_t syevdx_bufferSize<float>(gpusolverDnHandle_t, gpusolverEigMode_t,
                                                    gpusolverEigRange_t, gpublasFillMode_t, int,
                                                    const float *, int, float, float, int, int,
                                                    int *, const float *, int *);
template gpusolverStatus_t syevdx_bufferSize<double>(gpusolverDnHandle_t, gpusolverEigMode_t,
                                                     gpusolverEigRange_t, gpublasFillMode_t, int,
                                                     const double *, int, double, double, int, int,
                                                     int *, const double *, int *);

// Function: syevdx
template gpusolverStatus_t syevdx<float>(gpusolverDnHandle_t, gpusolverEigMode_t,
                                         gpusolverEigRange_t, gpublasFillMode_t, int, float *, int,
                                         float, float, int, int, int *, float *, float *, int,
                                         int *);
template gpusolverStatus_t syevdx<double>(gpusolverDnHandle_t, gpusolverEigMode_t,
                                          gpusolverEigRange_t, gpublasFillMode_t, int, double *,
                                          int, double, double, int, int, int *, double *, double *,
                                          int, int *);

// Function: heevd_bufferSize
template gpusolverStatus_t
heevd_bufferSize<gpuFloatComplex>(gpusolverDnHandle_t, gpusolverEigMode_t, gpublasFillMode_t, int,
                                  const gpuFloatComplex *, int,
                                  const ComplexToRealType<gpuFloatComplex> *, int *);
template gpusolverStatus_t
heevd_bufferSize<gpuDoubleComplex>(gpusolverDnHandle_t, gpusolverEigMode_t, gpublasFillMode_t, int,
                                   const gpuDoubleComplex *, int,
                                   const ComplexToRealType<gpuDoubleComplex> *, int *);

// Function: heevd
template gpusolverStatus_t heevd<gpuFloatComplex>(gpusolverDnHandle_t, gpusolverEigMode_t,
                                                  gpublasFillMode_t, int, gpuFloatComplex *, int,
                                                  ComplexToRealType<gpuFloatComplex> *,
                                                  gpuFloatComplex *, int, int *);
template gpusolverStatus_t heevd<gpuDoubleComplex>(gpusolverDnHandle_t, gpusolverEigMode_t,
                                                   gpublasFillMode_t, int, gpuDoubleComplex *, int,
                                                   ComplexToRealType<gpuDoubleComplex> *,
                                                   gpuDoubleComplex *, int, int *);

// Function: heevdx_bufferSize
template gpusolverStatus_t
heevdx_bufferSize<gpuFloatComplex>(gpusolverDnHandle_t, gpusolverEigMode_t, gpusolverEigRange_t,
                                   gpublasFillMode_t, int, const gpuFloatComplex *, int,
                                   ComplexToRealType<gpuFloatComplex>,
                                   ComplexToRealType<gpuFloatComplex>, int, int, int *,
                                   const ComplexToRealType<gpuFloatComplex> *, int *);
template gpusolverStatus_t
heevdx_bufferSize<gpuDoubleComplex>(gpusolverDnHandle_t, gpusolverEigMode_t, gpusolverEigRange_t,
                                    gpublasFillMode_t, int, const gpuDoubleComplex *, int,
                                    ComplexToRealType<gpuDoubleComplex>,
                                    ComplexToRealType<gpuDoubleComplex>, int, int, int *,
                                    const ComplexToRealType<gpuDoubleComplex> *, int *);

// Function: heevdx
template gpusolverStatus_t heevdx<gpuFloatComplex>(
    gpusolverDnHandle_t, gpusolverEigMode_t, gpusolverEigRange_t, gpublasFillMode_t, int,
    gpuFloatComplex *, int, ComplexToRealType<gpuFloatComplex>, ComplexToRealType<gpuFloatComplex>,
    int, int, int *, ComplexToRealType<gpuFloatComplex> *, gpuFloatComplex *, int, int *);
template gpusolverStatus_t heevdx<gpuDoubleComplex>(gpusolverDnHandle_t, gpusolverEigMode_t,
                                                    gpusolverEigRange_t, gpublasFillMode_t, int,
                                                    gpuDoubleComplex *, int,
                                                    ComplexToRealType<gpuDoubleComplex>,
                                                    ComplexToRealType<gpuDoubleComplex>, int, int,
                                                    int *, ComplexToRealType<gpuDoubleComplex> *,
                                                    gpuDoubleComplex *, int, int *);

// Function: syevj_bufferSize
template gpusolverStatus_t syevj_bufferSize<float>(gpusolverDnHandle_t, gpusolverEigMode_t,
                                                   gpublasFillMode_t, int, const float *, int,
                                                   const float *, int *, gpusolverSyevjInfo_t);
template gpusolverStatus_t syevj_bufferSize<double>(gpusolverDnHandle_t, gpusolverEigMode_t,
                                                    gpublasFillMode_t, int, const double *, int,
                                                    const double *, int *, gpusolverSyevjInfo_t);

// Function: syevj
template gpusolverStatus_t syevj<float>(gpusolverDnHandle_t, gpusolverEigMode_t, gpublasFillMode_t,
                                        int, float *, int, float *, float *, int, int *,
                                        gpusolverSyevjInfo_t);
template gpusolverStatus_t syevj<double>(gpusolverDnHandle_t, gpusolverEigMode_t, gpublasFillMode_t,
                                         int, double *, int, double *, double *, int, int *,
                                         gpusolverSyevjInfo_t);

// Function: syevjBatched_bufferSize
template gpusolverStatus_t syevjBatched_bufferSize<float>(gpusolverDnHandle_t, gpusolverEigMode_t,
                                                          gpublasFillMode_t, int, const float *,
                                                          int, const float *, int *,
                                                          gpusolverSyevjInfo_t, int);
template gpusolverStatus_t syevjBatched_bufferSize<double>(gpusolverDnHandle_t, gpusolverEigMode_t,
                                                           gpublasFillMode_t, int, const double *,
                                                           int, const double *, int *,
                                                           gpusolverSyevjInfo_t, int);

// Function: syevjBatched
template gpusolverStatus_t syevjBatched<float>(gpusolverDnHandle_t, gpusolverEigMode_t,
                                               gpublasFillMode_t, int, float *, int, float *,
                                               float *, int, int *, gpusolverSyevjInfo_t, int);
template gpusolverStatus_t syevjBatched<double>(gpusolverDnHandle_t, gpusolverEigMode_t,
                                                gpublasFillMode_t, int, double *, int, double *,
                                                double *, int, int *, gpusolverSyevjInfo_t, int);

// Function: heevj_bufferSize
template gpusolverStatus_t heevj_bufferSize<gpuFloatComplex>(
    gpusolverDnHandle_t, gpusolverEigMode_t, gpublasFillMode_t, int, const gpuFloatComplex *, int,
    const ComplexToRealType<gpuFloatComplex> *, int *, gpusolverSyevjInfo_t);
template gpusolverStatus_t heevj_bufferSize<gpuDoubleComplex>(
    gpusolverDnHandle_t, gpusolverEigMode_t, gpublasFillMode_t, int, const gpuDoubleComplex *, int,
    const ComplexToRealType<gpuDoubleComplex> *, int *, gpusolverSyevjInfo_t);

// Function: heevj
template gpusolverStatus_t heevj<gpuFloatComplex>(gpusolverDnHandle_t, gpusolverEigMode_t,
                                                  gpublasFillMode_t, int, gpuFloatComplex *, int,
                                                  ComplexToRealType<gpuFloatComplex> *,
                                                  gpuFloatComplex *, int, int *,
                                                  gpusolverSyevjInfo_t);
template gpusolverStatus_t heevj<gpuDoubleComplex>(gpusolverDnHandle_t, gpusolverEigMode_t,
                                                   gpublasFillMode_t, int, gpuDoubleComplex *, int,
                                                   ComplexToRealType<gpuDoubleComplex> *,
                                                   gpuDoubleComplex *, int, int *,
                                                   gpusolverSyevjInfo_t);

// Function: heevjBatched_bufferSize
template gpusolverStatus_t heevjBatched_bufferSize<gpuFloatComplex>(
    gpusolverDnHandle_t, gpusolverEigMode_t, gpublasFillMode_t, int, const gpuFloatComplex *, int,
    const ComplexToRealType<gpuFloatComplex> *, int *, gpusolverSyevjInfo_t, int);
template gpusolverStatus_t heevjBatched_bufferSize<gpuDoubleComplex>(
    gpusolverDnHandle_t, gpusolverEigMode_t, gpublasFillMode_t, int, const gpuDoubleComplex *, int,
    const ComplexToRealType<gpuDoubleComplex> *, int *, gpusolverSyevjInfo_t, int);

// Function: heevjBatched
template gpusolverStatus_t heevjBatched<gpuFloatComplex>(gpusolverDnHandle_t, gpusolverEigMode_t,
                                                         gpublasFillMode_t, int, gpuFloatComplex *,
                                                         int, ComplexToRealType<gpuFloatComplex> *,
                                                         gpuFloatComplex *, int, int *,
                                                         gpusolverSyevjInfo_t, int);
template gpusolverStatus_t
heevjBatched<gpuDoubleComplex>(gpusolverDnHandle_t, gpusolverEigMode_t, gpublasFillMode_t, int,
                               gpuDoubleComplex *, int, ComplexToRealType<gpuDoubleComplex> *,
                               gpuDoubleComplex *, int, int *, gpusolverSyevjInfo_t, int);

// Function: gesvdj_bufferSize
template gpusolverStatus_t gesvdj_bufferSize<float>(gpusolverDnHandle_t, gpusolverEigMode_t, int,
                                                    int, int, const float *, int,
                                                    const ComplexToRealType<float> *, const float *,
                                                    int, const float *, int, int *,
                                                    gpusolverGesvdjInfo_t);
template gpusolverStatus_t gesvdj_bufferSize<double>(gpusolverDnHandle_t, gpusolverEigMode_t, int,
                                                     int, int, const double *, int,
                                                     const ComplexToRealType<double> *,
                                                     const double *, int, const double *, int,
                                                     int *, gpusolverGesvdjInfo_t);
template gpusolverStatus_t gesvdj_bufferSize<gpuFloatComplex>(
    gpusolverDnHandle_t, gpusolverEigMode_t, int, int, int, const gpuFloatComplex *, int,
    const ComplexToRealType<gpuFloatComplex> *, const gpuFloatComplex *, int,
    const gpuFloatComplex *, int, int *, gpusolverGesvdjInfo_t);
template gpusolverStatus_t gesvdj_bufferSize<gpuDoubleComplex>(
    gpusolverDnHandle_t, gpusolverEigMode_t, int, int, int, const gpuDoubleComplex *, int,
    const ComplexToRealType<gpuDoubleComplex> *, const gpuDoubleComplex *, int,
    const gpuDoubleComplex *, int, int *, gpusolverGesvdjInfo_t);

// Function: gesvdj
template gpusolverStatus_t gesvdj<float>(gpusolverDnHandle_t, gpusolverEigMode_t, int, int, int,
                                         float *, int, ComplexToRealType<float> *, float *, int,
                                         float *, int, float *, int, int *, gpusolverGesvdjInfo_t);
template gpusolverStatus_t gesvdj<double>(gpusolverDnHandle_t, gpusolverEigMode_t, int, int, int,
                                          double *, int, ComplexToRealType<double> *, double *, int,
                                          double *, int, double *, int, int *,
                                          gpusolverGesvdjInfo_t);
template gpusolverStatus_t gesvdj<gpuFloatComplex>(gpusolverDnHandle_t, gpusolverEigMode_t, int,
                                                   int, int, gpuFloatComplex *, int,
                                                   ComplexToRealType<gpuFloatComplex> *,
                                                   gpuFloatComplex *, int, gpuFloatComplex *, int,
                                                   gpuFloatComplex *, int, int *,
                                                   gpusolverGesvdjInfo_t);
template gpusolverStatus_t gesvdj<gpuDoubleComplex>(gpusolverDnHandle_t, gpusolverEigMode_t, int,
                                                    int, int, gpuDoubleComplex *, int,
                                                    ComplexToRealType<gpuDoubleComplex> *,
                                                    gpuDoubleComplex *, int, gpuDoubleComplex *,
                                                    int, gpuDoubleComplex *, int, int *,
                                                    gpusolverGesvdjInfo_t);

// Function: gesvdjBatched_bufferSize
template gpusolverStatus_t gesvdjBatched_bufferSize<float>(gpusolverDnHandle_t, gpusolverEigMode_t,
                                                           int, int, const float *, int,
                                                           const ComplexToRealType<float> *,
                                                           const float *, int, const float *, int,
                                                           int *, gpusolverGesvdjInfo_t, int);
template gpusolverStatus_t gesvdjBatched_bufferSize<double>(gpusolverDnHandle_t, gpusolverEigMode_t,
                                                            int, int, const double *, int,
                                                            const ComplexToRealType<double> *,
                                                            const double *, int, const double *,
                                                            int, int *, gpusolverGesvdjInfo_t, int);
template gpusolverStatus_t gesvdjBatched_bufferSize<gpuFloatComplex>(
    gpusolverDnHandle_t, gpusolverEigMode_t, int, int, const gpuFloatComplex *, int,
    const ComplexToRealType<gpuFloatComplex> *, const gpuFloatComplex *, int,
    const gpuFloatComplex *, int, int *, gpusolverGesvdjInfo_t, int);
template gpusolverStatus_t gesvdjBatched_bufferSize<gpuDoubleComplex>(
    gpusolverDnHandle_t, gpusolverEigMode_t, int, int, const gpuDoubleComplex *, int,
    const ComplexToRealType<gpuDoubleComplex> *, const gpuDoubleComplex *, int,
    const gpuDoubleComplex *, int, int *, gpusolverGesvdjInfo_t, int);

// Function: gesvdjBatched
template gpusolverStatus_t gesvdjBatched<float>(gpusolverDnHandle_t, gpusolverEigMode_t, int, int,
                                                float *, int, ComplexToRealType<float> *, float *,
                                                int, float *, int, float *, int, int *,
                                                gpusolverGesvdjInfo_t, int);
template gpusolverStatus_t gesvdjBatched<double>(gpusolverDnHandle_t, gpusolverEigMode_t, int, int,
                                                 double *, int, ComplexToRealType<double> *,
                                                 double *, int, double *, int, double *, int, int *,
                                                 gpusolverGesvdjInfo_t, int);
template gpusolverStatus_t gesvdjBatched<gpuFloatComplex>(gpusolverDnHandle_t, gpusolverEigMode_t,
                                                          int, int, gpuFloatComplex *, int,
                                                          ComplexToRealType<gpuFloatComplex> *,
                                                          gpuFloatComplex *, int, gpuFloatComplex *,
                                                          int, gpuFloatComplex *, int, int *,
                                                          gpusolverGesvdjInfo_t, int);
template gpusolverStatus_t
gesvdjBatched<gpuDoubleComplex>(gpusolverDnHandle_t, gpusolverEigMode_t, int, int,
                                gpuDoubleComplex *, int, ComplexToRealType<gpuDoubleComplex> *,
                                gpuDoubleComplex *, int, gpuDoubleComplex *, int,
                                gpuDoubleComplex *, int, int *, gpusolverGesvdjInfo_t, int);

// Function: sygvd_bufferSize
template gpusolverStatus_t sygvd_bufferSize<float>(gpusolverDnHandle_t, gpusolverEigType_t,
                                                   gpusolverEigMode_t, gpublasFillMode_t, int,
                                                   const float *, int, const float *, int,
                                                   const float *, int *);
template gpusolverStatus_t sygvd_bufferSize<double>(gpusolverDnHandle_t, gpusolverEigType_t,
                                                    gpusolverEigMode_t, gpublasFillMode_t, int,
                                                    const double *, int, const double *, int,
                                                    const double *, int *);

// Function: sygvd
template gpusolverStatus_t sygvd<float>(gpusolverDnHandle_t, gpusolverEigType_t, gpusolverEigMode_t,
                                        gpublasFillMode_t, int, float *, int, float *, int, float *,
                                        float *, int, int *);
template gpusolverStatus_t sygvd<double>(gpusolverDnHandle_t, gpusolverEigType_t,
                                         gpusolverEigMode_t, gpublasFillMode_t, int, double *, int,
                                         double *, int, double *, double *, int, int *);

// Function: hegvd_bufferSize
template gpusolverStatus_t
hegvd_bufferSize<gpuFloatComplex>(gpusolverDnHandle_t, gpusolverEigType_t, gpusolverEigMode_t,
                                  gpublasFillMode_t, int, const gpuFloatComplex *, int,
                                  const gpuFloatComplex *, int,
                                  const ComplexToRealType<gpuFloatComplex> *, int *);
template gpusolverStatus_t
hegvd_bufferSize<gpuDoubleComplex>(gpusolverDnHandle_t, gpusolverEigType_t, gpusolverEigMode_t,
                                   gpublasFillMode_t, int, const gpuDoubleComplex *, int,
                                   const gpuDoubleComplex *, int,
                                   const ComplexToRealType<gpuDoubleComplex> *, int *);

// Function: hegvd
template gpusolverStatus_t hegvd<gpuFloatComplex>(gpusolverDnHandle_t, gpusolverEigType_t,
                                                  gpusolverEigMode_t, gpublasFillMode_t, int,
                                                  gpuFloatComplex *, int, gpuFloatComplex *, int,
                                                  ComplexToRealType<gpuFloatComplex> *,
                                                  gpuFloatComplex *, int, int *);
template gpusolverStatus_t hegvd<gpuDoubleComplex>(gpusolverDnHandle_t, gpusolverEigType_t,
                                                   gpusolverEigMode_t, gpublasFillMode_t, int,
                                                   gpuDoubleComplex *, int, gpuDoubleComplex *, int,
                                                   ComplexToRealType<gpuDoubleComplex> *,
                                                   gpuDoubleComplex *, int, int *);

// Function: sygvj_bufferSize
template gpusolverStatus_t sygvj_bufferSize<float>(gpusolverDnHandle_t, gpusolverEigType_t,
                                                   gpusolverEigMode_t, gpublasFillMode_t, int,
                                                   const float *, int, const float *, int,
                                                   const float *, int *, gpusolverSyevjInfo_t);
template gpusolverStatus_t sygvj_bufferSize<double>(gpusolverDnHandle_t, gpusolverEigType_t,
                                                    gpusolverEigMode_t, gpublasFillMode_t, int,
                                                    const double *, int, const double *, int,
                                                    const double *, int *, gpusolverSyevjInfo_t);

// Function: sygvj
template gpusolverStatus_t sygvj<float>(gpusolverDnHandle_t, gpusolverEigType_t, gpusolverEigMode_t,
                                        gpublasFillMode_t, int, float *, int, float *, int, float *,
                                        float *, int, int *, gpusolverSyevjInfo_t);
template gpusolverStatus_t sygvj<double>(gpusolverDnHandle_t, gpusolverEigType_t,
                                         gpusolverEigMode_t, gpublasFillMode_t, int, double *, int,
                                         double *, int, double *, double *, int, int *,
                                         gpusolverSyevjInfo_t);

// Function: hegvj_bufferSize
template gpusolverStatus_t hegvj_bufferSize<gpuFloatComplex>(
    gpusolverDnHandle_t, gpusolverEigType_t, gpusolverEigMode_t, gpublasFillMode_t, int,
    const gpuFloatComplex *, int, const gpuFloatComplex *, int,
    const ComplexToRealType<gpuFloatComplex> *, int *, gpusolverSyevjInfo_t);
template gpusolverStatus_t hegvj_bufferSize<gpuDoubleComplex>(
    gpusolverDnHandle_t, gpusolverEigType_t, gpusolverEigMode_t, gpublasFillMode_t, int,
    const gpuDoubleComplex *, int, const gpuDoubleComplex *, int,
    const ComplexToRealType<gpuDoubleComplex> *, int *, gpusolverSyevjInfo_t);

// Function: hegvj
template gpusolverStatus_t hegvj<gpuFloatComplex>(gpusolverDnHandle_t, gpusolverEigType_t,
                                                  gpusolverEigMode_t, gpublasFillMode_t, int,
                                                  gpuFloatComplex *, int, gpuFloatComplex *, int,
                                                  ComplexToRealType<gpuFloatComplex> *,
                                                  gpuFloatComplex *, int, int *,
                                                  gpusolverSyevjInfo_t);
template gpusolverStatus_t hegvj<gpuDoubleComplex>(gpusolverDnHandle_t, gpusolverEigType_t,
                                                   gpusolverEigMode_t, gpublasFillMode_t, int,
                                                   gpuDoubleComplex *, int, gpuDoubleComplex *, int,
                                                   ComplexToRealType<gpuDoubleComplex> *,
                                                   gpuDoubleComplex *, int, int *,
                                                   gpusolverSyevjInfo_t);

// Function: sygvdx_bufferSize
template gpusolverStatus_t sygvdx_bufferSize<float>(gpusolverDnHandle_t, gpusolverEigType_t,
                                                    gpusolverEigMode_t, gpusolverEigRange_t,
                                                    gpublasFillMode_t, int, const float *, int,
                                                    const float *, int, float, float, int, int,
                                                    int *, const float *, int *);
template gpusolverStatus_t sygvdx_bufferSize<double>(gpusolverDnHandle_t, gpusolverEigType_t,
                                                     gpusolverEigMode_t, gpusolverEigRange_t,
                                                     gpublasFillMode_t, int, const double *, int,
                                                     const double *, int, double, double, int, int,
                                                     int *, const double *, int *);

// Function: sygvdx
template gpusolverStatus_t sygvdx<float>(gpusolverDnHandle_t, gpusolverEigType_t,
                                         gpusolverEigMode_t, gpusolverEigRange_t, gpublasFillMode_t,
                                         int, float *, int, float *, int, float, float, int, int,
                                         int *, float *, float *, int, int *);
template gpusolverStatus_t sygvdx<double>(gpusolverDnHandle_t, gpusolverEigType_t,
                                          gpusolverEigMode_t, gpusolverEigRange_t,
                                          gpublasFillMode_t, int, double *, int, double *, int,
                                          double, double, int, int, int *, double *, double *, int,
                                          int *);

// Function: hegvdx_bufferSize
template gpusolverStatus_t hegvdx_bufferSize<gpuFloatComplex>(
    gpusolverDnHandle_t, gpusolverEigType_t, gpusolverEigMode_t, gpusolverEigRange_t,
    gpublasFillMode_t, int, const gpuFloatComplex *, int, const gpuFloatComplex *, int,
    ComplexToRealType<gpuFloatComplex>, ComplexToRealType<gpuFloatComplex>, int, int, int *,
    const ComplexToRealType<gpuFloatComplex> *, int *);
template gpusolverStatus_t hegvdx_bufferSize<gpuDoubleComplex>(
    gpusolverDnHandle_t, gpusolverEigType_t, gpusolverEigMode_t, gpusolverEigRange_t,
    gpublasFillMode_t, int, const gpuDoubleComplex *, int, const gpuDoubleComplex *, int,
    ComplexToRealType<gpuDoubleComplex>, ComplexToRealType<gpuDoubleComplex>, int, int, int *,
    const ComplexToRealType<gpuDoubleComplex> *, int *);

// Function: hegvdx
template gpusolverStatus_t
hegvdx<gpuFloatComplex>(gpusolverDnHandle_t, gpusolverEigType_t, gpusolverEigMode_t,
                        gpusolverEigRange_t, gpublasFillMode_t, int, gpuFloatComplex *, int,
                        gpuFloatComplex *, int, ComplexToRealType<gpuFloatComplex>,
                        ComplexToRealType<gpuFloatComplex>, int, int, int *,
                        ComplexToRealType<gpuFloatComplex> *, gpuFloatComplex *, int, int *);
template gpusolverStatus_t
hegvdx<gpuDoubleComplex>(gpusolverDnHandle_t, gpusolverEigType_t, gpusolverEigMode_t,
                         gpusolverEigRange_t, gpublasFillMode_t, int, gpuDoubleComplex *, int,
                         gpuDoubleComplex *, int, ComplexToRealType<gpuDoubleComplex>,
                         ComplexToRealType<gpuDoubleComplex>, int, int, int *,
                         ComplexToRealType<gpuDoubleComplex> *, gpuDoubleComplex *, int, int *);

// Function: gesvdaStridedBatched_bufferSize
template gpusolverStatus_t gesvdaStridedBatched_bufferSize<float>(
    gpusolverDnHandle_t, gpusolverEigMode_t, int, int, int, const float *, int, long long int,
    const ComplexToRealType<float> *, long long int, const float *, int, long long int,
    const float *, int, long long int, int *, int);
template gpusolverStatus_t gesvdaStridedBatched_bufferSize<double>(
    gpusolverDnHandle_t, gpusolverEigMode_t, int, int, int, const double *, int, long long int,
    const ComplexToRealType<double> *, long long int, const double *, int, long long int,
    const double *, int, long long int, int *, int);
template gpusolverStatus_t gesvdaStridedBatched_bufferSize<gpuFloatComplex>(
    gpusolverDnHandle_t, gpusolverEigMode_t, int, int, int, const gpuFloatComplex *, int,
    long long int, const ComplexToRealType<gpuFloatComplex> *, long long int,
    const gpuFloatComplex *, int, long long int, const gpuFloatComplex *, int, long long int, int *,
    int);
template gpusolverStatus_t gesvdaStridedBatched_bufferSize<gpuDoubleComplex>(
    gpusolverDnHandle_t, gpusolverEigMode_t, int, int, int, const gpuDoubleComplex *, int,
    long long int, const ComplexToRealType<gpuDoubleComplex> *, long long int,
    const gpuDoubleComplex *, int, long long int, const gpuDoubleComplex *, int, long long int,
    int *, int);

// Function: gesvdaStridedBatched
template gpusolverStatus_t gesvdaStridedBatched<float>(gpusolverDnHandle_t, gpusolverEigMode_t, int,
                                                       int, int, const float *, int, long long int,
                                                       ComplexToRealType<float> *, long long int,
                                                       float *, int, long long int, float *, int,
                                                       long long int, float *, int, int *, double *,
                                                       int);
template gpusolverStatus_t gesvdaStridedBatched<double>(gpusolverDnHandle_t, gpusolverEigMode_t,
                                                        int, int, int, const double *, int,
                                                        long long int, ComplexToRealType<double> *,
                                                        long long int, double *, int, long long int,
                                                        double *, int, long long int, double *, int,
                                                        int *, double *, int);
template gpusolverStatus_t
gesvdaStridedBatched<gpuFloatComplex>(gpusolverDnHandle_t, gpusolverEigMode_t, int, int, int,
                                      const gpuFloatComplex *, int, long long int,
                                      ComplexToRealType<gpuFloatComplex> *, long long int,
                                      gpuFloatComplex *, int, long long int, gpuFloatComplex *, int,
                                      long long int, gpuFloatComplex *, int, int *, double *, int);
template gpusolverStatus_t gesvdaStridedBatched<gpuDoubleComplex>(
    gpusolverDnHandle_t, gpusolverEigMode_t, int, int, int, const gpuDoubleComplex *, int,
    long long int, ComplexToRealType<gpuDoubleComplex> *, long long int, gpuDoubleComplex *, int,
    long long int, gpuDoubleComplex *, int, long long int, gpuDoubleComplex *, int, int *, double *,
    int);

// Function: sytrd_bufferSize
template gpusolverStatus_t sytrd_bufferSize<float>(gpusolverDnHandle_t, gpublasFillMode_t, int,
                                                   const float *, int, const float *, const float *,
                                                   const float *, int *);
template gpusolverStatus_t sytrd_bufferSize<double>(gpusolverDnHandle_t, gpublasFillMode_t, int,
                                                    const double *, int, const double *,
                                                    const double *, const double *, int *);

// Function: sytrd
template gpusolverStatus_t sytrd<float>(gpusolverDnHandle_t, gpublasFillMode_t, int, float *, int,
                                        float *, float *, float *, float *, int, int *);
template gpusolverStatus_t sytrd<double>(gpusolverDnHandle_t, gpublasFillMode_t, int, double *, int,
                                         double *, double *, double *, double *, int, int *);

// Function: hetrd_bufferSize
template gpusolverStatus_t hetrd_bufferSize<gpuFloatComplex>(
    gpusolverDnHandle_t, gpublasFillMode_t, int, const gpuFloatComplex *, int,
    const ComplexToRealType<gpuFloatComplex> *, const ComplexToRealType<gpuFloatComplex> *,
    const gpuFloatComplex *, int *);
template gpusolverStatus_t hetrd_bufferSize<gpuDoubleComplex>(
    gpusolverDnHandle_t, gpublasFillMode_t, int, const gpuDoubleComplex *, int,
    const ComplexToRealType<gpuDoubleComplex> *, const ComplexToRealType<gpuDoubleComplex> *,
    const gpuDoubleComplex *, int *);

// Function: hetrd
template gpusolverStatus_t hetrd<gpuFloatComplex>(gpusolverDnHandle_t, gpublasFillMode_t, int,
                                                  gpuFloatComplex *, int,
                                                  ComplexToRealType<gpuFloatComplex> *,
                                                  ComplexToRealType<gpuFloatComplex> *,
                                                  gpuFloatComplex *, gpuFloatComplex *, int, int *);
template gpusolverStatus_t hetrd<gpuDoubleComplex>(gpusolverDnHandle_t, gpublasFillMode_t, int,
                                                   gpuDoubleComplex *, int,
                                                   ComplexToRealType<gpuDoubleComplex> *,
                                                   ComplexToRealType<gpuDoubleComplex> *,
                                                   gpuDoubleComplex *, gpuDoubleComplex *, int,
                                                   int *);

// Function: orgtr_bufferSize
template gpusolverStatus_t orgtr_bufferSize<float>(gpusolverDnHandle_t, gpublasFillMode_t, int,
                                                   const float *, int, const float *, int *);
template gpusolverStatus_t orgtr_bufferSize<double>(gpusolverDnHandle_t, gpublasFillMode_t, int,
                                                    const double *, int, const double *, int *);

// Function: orgtr
template gpusolverStatus_t orgtr<float>(gpusolverDnHandle_t, gpublasFillMode_t, int, float *, int,
                                        const float *, float *, int, int *);
template gpusolverStatus_t orgtr<double>(gpusolverDnHandle_t, gpublasFillMode_t, int, double *, int,
                                         const double *, double *, int, int *);

// Function: ungtr_bufferSize
template gpusolverStatus_t ungtr_bufferSize<gpuFloatComplex>(gpusolverDnHandle_t, gpublasFillMode_t,
                                                             int, const gpuFloatComplex *, int,
                                                             const gpuFloatComplex *, int *);
template gpusolverStatus_t ungtr_bufferSize<gpuDoubleComplex>(gpusolverDnHandle_t,
                                                              gpublasFillMode_t, int,
                                                              const gpuDoubleComplex *, int,
                                                              const gpuDoubleComplex *, int *);

// Function: ungtr
template gpusolverStatus_t ungtr<gpuFloatComplex>(gpusolverDnHandle_t, gpublasFillMode_t, int,
                                                  gpuFloatComplex *, int, const gpuFloatComplex *,
                                                  gpuFloatComplex *, int, int *);
template gpusolverStatus_t ungtr<gpuDoubleComplex>(gpusolverDnHandle_t, gpublasFillMode_t, int,
                                                   gpuDoubleComplex *, int,
                                                   const gpuDoubleComplex *, gpuDoubleComplex *,
                                                   int, int *);

// Function: ormtr_bufferSize
template gpusolverStatus_t ormtr_bufferSize<float>(gpusolverDnHandle_t, gpublasSideMode_t,
                                                   gpublasFillMode_t, gpublasOperation_t, int, int,
                                                   const float *, int, const float *, const float *,
                                                   int, int *);
template gpusolverStatus_t ormtr_bufferSize<double>(gpusolverDnHandle_t, gpublasSideMode_t,
                                                    gpublasFillMode_t, gpublasOperation_t, int, int,
                                                    const double *, int, const double *,
                                                    const double *, int, int *);

// Function: ormtr
template gpusolverStatus_t ormtr<float>(gpusolverDnHandle_t, gpublasSideMode_t, gpublasFillMode_t,
                                        gpublasOperation_t, int, int, float *, int, float *,
                                        float *, int, float *, int, int *);
template gpusolverStatus_t ormtr<double>(gpusolverDnHandle_t, gpublasSideMode_t, gpublasFillMode_t,
                                         gpublasOperation_t, int, int, double *, int, double *,
                                         double *, int, double *, int, int *);

// Function: unmtr_bufferSize
template gpusolverStatus_t unmtr_bufferSize<gpuFloatComplex>(gpusolverDnHandle_t, gpublasSideMode_t,
                                                             gpublasFillMode_t, gpublasOperation_t,
                                                             int, int, const gpuFloatComplex *, int,
                                                             const gpuFloatComplex *,
                                                             const gpuFloatComplex *, int, int *);
template gpusolverStatus_t
unmtr_bufferSize<gpuDoubleComplex>(gpusolverDnHandle_t, gpublasSideMode_t, gpublasFillMode_t,
                                   gpublasOperation_t, int, int, const gpuDoubleComplex *, int,
                                   const gpuDoubleComplex *, const gpuDoubleComplex *, int, int *);

// Function: unmtr
template gpusolverStatus_t unmtr<gpuFloatComplex>(gpusolverDnHandle_t, gpublasSideMode_t,
                                                  gpublasFillMode_t, gpublasOperation_t, int, int,
                                                  gpuFloatComplex *, int, gpuFloatComplex *,
                                                  gpuFloatComplex *, int, gpuFloatComplex *, int,
                                                  int *);
template gpusolverStatus_t unmtr<gpuDoubleComplex>(gpusolverDnHandle_t, gpublasSideMode_t,
                                                   gpublasFillMode_t, gpublasOperation_t, int, int,
                                                   gpuDoubleComplex *, int, gpuDoubleComplex *,
                                                   gpuDoubleComplex *, int, gpuDoubleComplex *, int,
                                                   int *);

} // namespace gpumod
