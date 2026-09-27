/**
 * @file instantiations.cpp
 * @brief Explicit template instantiations for the GPU sparse wrappers
 *
 * This file contains all explicit template instantiations to avoid
 * code bloat from implicit instantiation at every call site.
 */

module gpumod.wrappers.sparse;

import gpumod.sparse;
import gpumod.complex;
import gpumod.wrappers.common;

namespace wwr {

// Function: bsrmv
template gpusparseStatus_t bsrmv<float>(gpusparseHandle_t, gpusparseDirection_t,
                                        gpusparseOperation_t, int, int, int, const float *,
                                        const gpusparseMatDescr_t, const float *, const int *,
                                        const int *, int, const float *, const float *, float *);
template gpusparseStatus_t bsrmv<double>(gpusparseHandle_t, gpusparseDirection_t,
                                         gpusparseOperation_t, int, int, int, const double *,
                                         const gpusparseMatDescr_t, const double *, const int *,
                                         const int *, int, const double *, const double *,
                                         double *);
template gpusparseStatus_t
bsrmv<gpuFloatComplex>(gpusparseHandle_t, gpusparseDirection_t, gpusparseOperation_t, int, int, int,
                       const gpuFloatComplex *, const gpusparseMatDescr_t, const gpuFloatComplex *,
                       const int *, const int *, int, const gpuFloatComplex *,
                       const gpuFloatComplex *, gpuFloatComplex *);
template gpusparseStatus_t
bsrmv<gpuDoubleComplex>(gpusparseHandle_t, gpusparseDirection_t, gpusparseOperation_t, int, int,
                        int, const gpuDoubleComplex *, const gpusparseMatDescr_t,
                        const gpuDoubleComplex *, const int *, const int *, int,
                        const gpuDoubleComplex *, const gpuDoubleComplex *, gpuDoubleComplex *);

// Function: gtsv2_bufferSizeExt
template gpusparseStatus_t gtsv2_bufferSizeExt<float>(gpusparseHandle_t, int, int, const float *,
                                                      const float *, const float *, const float *,
                                                      int, size_t *);
template gpusparseStatus_t gtsv2_bufferSizeExt<double>(gpusparseHandle_t, int, int, const double *,
                                                       const double *, const double *,
                                                       const double *, int, size_t *);
template gpusparseStatus_t
gtsv2_bufferSizeExt<gpuFloatComplex>(gpusparseHandle_t, int, int, const gpuFloatComplex *,
                                     const gpuFloatComplex *, const gpuFloatComplex *,
                                     const gpuFloatComplex *, int, size_t *);
template gpusparseStatus_t
gtsv2_bufferSizeExt<gpuDoubleComplex>(gpusparseHandle_t, int, int, const gpuDoubleComplex *,
                                      const gpuDoubleComplex *, const gpuDoubleComplex *,
                                      const gpuDoubleComplex *, int, size_t *);

// Function: gtsv2
template gpusparseStatus_t gtsv2<float>(gpusparseHandle_t, int, int, const float *, const float *,
                                        const float *, float *, int, void *);
template gpusparseStatus_t gtsv2<double>(gpusparseHandle_t, int, int, const double *,
                                         const double *, const double *, double *, int, void *);
template gpusparseStatus_t gtsv2<gpuFloatComplex>(gpusparseHandle_t, int, int,
                                                  const gpuFloatComplex *, const gpuFloatComplex *,
                                                  const gpuFloatComplex *, gpuFloatComplex *, int,
                                                  void *);
template gpusparseStatus_t gtsv2<gpuDoubleComplex>(gpusparseHandle_t, int, int,
                                                   const gpuDoubleComplex *,
                                                   const gpuDoubleComplex *,
                                                   const gpuDoubleComplex *, gpuDoubleComplex *,
                                                   int, void *);

// Function: gtsv2_nopivot_bufferSizeExt
template gpusparseStatus_t gtsv2_nopivot_bufferSizeExt<float>(gpusparseHandle_t, int, int,
                                                              const float *, const float *,
                                                              const float *, const float *, int,
                                                              size_t *);
template gpusparseStatus_t gtsv2_nopivot_bufferSizeExt<double>(gpusparseHandle_t, int, int,
                                                               const double *, const double *,
                                                               const double *, const double *, int,
                                                               size_t *);
template gpusparseStatus_t
gtsv2_nopivot_bufferSizeExt<gpuFloatComplex>(gpusparseHandle_t, int, int, const gpuFloatComplex *,
                                             const gpuFloatComplex *, const gpuFloatComplex *,
                                             const gpuFloatComplex *, int, size_t *);
template gpusparseStatus_t
gtsv2_nopivot_bufferSizeExt<gpuDoubleComplex>(gpusparseHandle_t, int, int, const gpuDoubleComplex *,
                                              const gpuDoubleComplex *, const gpuDoubleComplex *,
                                              const gpuDoubleComplex *, int, size_t *);

// Function: gtsv2_nopivot
template gpusparseStatus_t gtsv2_nopivot<float>(gpusparseHandle_t, int, int, const float *,
                                                const float *, const float *, float *, int, void *);
template gpusparseStatus_t gtsv2_nopivot<double>(gpusparseHandle_t, int, int, const double *,
                                                 const double *, const double *, double *, int,
                                                 void *);
template gpusparseStatus_t gtsv2_nopivot<gpuFloatComplex>(gpusparseHandle_t, int, int,
                                                          const gpuFloatComplex *,
                                                          const gpuFloatComplex *,
                                                          const gpuFloatComplex *,
                                                          gpuFloatComplex *, int, void *);
template gpusparseStatus_t gtsv2_nopivot<gpuDoubleComplex>(gpusparseHandle_t, int, int,
                                                           const gpuDoubleComplex *,
                                                           const gpuDoubleComplex *,
                                                           const gpuDoubleComplex *,
                                                           gpuDoubleComplex *, int, void *);

// Function: gtsv2StridedBatch_bufferSizeExt
template gpusparseStatus_t gtsv2StridedBatch_bufferSizeExt<float>(gpusparseHandle_t, int,
                                                                  const float *, const float *,
                                                                  const float *, const float *, int,
                                                                  int, size_t *);
template gpusparseStatus_t gtsv2StridedBatch_bufferSizeExt<double>(gpusparseHandle_t, int,
                                                                   const double *, const double *,
                                                                   const double *, const double *,
                                                                   int, int, size_t *);
template gpusparseStatus_t
gtsv2StridedBatch_bufferSizeExt<gpuFloatComplex>(gpusparseHandle_t, int, const gpuFloatComplex *,
                                                 const gpuFloatComplex *, const gpuFloatComplex *,
                                                 const gpuFloatComplex *, int, int, size_t *);
template gpusparseStatus_t gtsv2StridedBatch_bufferSizeExt<gpuDoubleComplex>(
    gpusparseHandle_t, int, const gpuDoubleComplex *, const gpuDoubleComplex *,
    const gpuDoubleComplex *, const gpuDoubleComplex *, int, int, size_t *);

// Function: gtsv2StridedBatch
template gpusparseStatus_t gtsv2StridedBatch<float>(gpusparseHandle_t, int, const float *,
                                                    const float *, const float *, float *, int, int,
                                                    void *);
template gpusparseStatus_t gtsv2StridedBatch<double>(gpusparseHandle_t, int, const double *,
                                                     const double *, const double *, double *, int,
                                                     int, void *);
template gpusparseStatus_t gtsv2StridedBatch<gpuFloatComplex>(gpusparseHandle_t, int,
                                                              const gpuFloatComplex *,
                                                              const gpuFloatComplex *,
                                                              const gpuFloatComplex *,
                                                              gpuFloatComplex *, int, int, void *);
template gpusparseStatus_t
gtsv2StridedBatch<gpuDoubleComplex>(gpusparseHandle_t, int, const gpuDoubleComplex *,
                                    const gpuDoubleComplex *, const gpuDoubleComplex *,
                                    gpuDoubleComplex *, int, int, void *);

// Function: gtsvInterleavedBatch_bufferSizeExt
template gpusparseStatus_t gtsvInterleavedBatch_bufferSizeExt<float>(gpusparseHandle_t, int, int,
                                                                     const float *, const float *,
                                                                     const float *, const float *,
                                                                     int, size_t *);
template gpusparseStatus_t
gtsvInterleavedBatch_bufferSizeExt<double>(gpusparseHandle_t, int, int, const double *,
                                           const double *, const double *, const double *, int,
                                           size_t *);
template gpusparseStatus_t gtsvInterleavedBatch_bufferSizeExt<gpuFloatComplex>(
    gpusparseHandle_t, int, int, const gpuFloatComplex *, const gpuFloatComplex *,
    const gpuFloatComplex *, const gpuFloatComplex *, int, size_t *);
template gpusparseStatus_t gtsvInterleavedBatch_bufferSizeExt<gpuDoubleComplex>(
    gpusparseHandle_t, int, int, const gpuDoubleComplex *, const gpuDoubleComplex *,
    const gpuDoubleComplex *, const gpuDoubleComplex *, int, size_t *);

// Function: gtsvInterleavedBatch
template gpusparseStatus_t gtsvInterleavedBatch<float>(gpusparseHandle_t, int, int, float *,
                                                       float *, float *, float *, int, void *);
template gpusparseStatus_t gtsvInterleavedBatch<double>(gpusparseHandle_t, int, int, double *,
                                                        double *, double *, double *, int, void *);
template gpusparseStatus_t gtsvInterleavedBatch<gpuFloatComplex>(gpusparseHandle_t, int, int,
                                                                 gpuFloatComplex *,
                                                                 gpuFloatComplex *,
                                                                 gpuFloatComplex *,
                                                                 gpuFloatComplex *, int, void *);
template gpusparseStatus_t gtsvInterleavedBatch<gpuDoubleComplex>(gpusparseHandle_t, int, int,
                                                                  gpuDoubleComplex *,
                                                                  gpuDoubleComplex *,
                                                                  gpuDoubleComplex *,
                                                                  gpuDoubleComplex *, int, void *);

// Function: gpsvInterleavedBatch_bufferSizeExt
template gpusparseStatus_t gpsvInterleavedBatch_bufferSizeExt<float>(gpusparseHandle_t, int, int,
                                                                     const float *, const float *,
                                                                     const float *, const float *,
                                                                     const float *, const float *,
                                                                     int, size_t *);
template gpusparseStatus_t
gpsvInterleavedBatch_bufferSizeExt<double>(gpusparseHandle_t, int, int, const double *,
                                           const double *, const double *, const double *,
                                           const double *, const double *, int, size_t *);
template gpusparseStatus_t gpsvInterleavedBatch_bufferSizeExt<gpuFloatComplex>(
    gpusparseHandle_t, int, int, const gpuFloatComplex *, const gpuFloatComplex *,
    const gpuFloatComplex *, const gpuFloatComplex *, const gpuFloatComplex *,
    const gpuFloatComplex *, int, size_t *);
template gpusparseStatus_t gpsvInterleavedBatch_bufferSizeExt<gpuDoubleComplex>(
    gpusparseHandle_t, int, int, const gpuDoubleComplex *, const gpuDoubleComplex *,
    const gpuDoubleComplex *, const gpuDoubleComplex *, const gpuDoubleComplex *,
    const gpuDoubleComplex *, int, size_t *);

// Function: gpsvInterleavedBatch
template gpusparseStatus_t gpsvInterleavedBatch<float>(gpusparseHandle_t, int, int, float *,
                                                       float *, float *, float *, float *, float *,
                                                       int, void *);
template gpusparseStatus_t gpsvInterleavedBatch<double>(gpusparseHandle_t, int, int, double *,
                                                        double *, double *, double *, double *,
                                                        double *, int, void *);
template gpusparseStatus_t
gpsvInterleavedBatch<gpuFloatComplex>(gpusparseHandle_t, int, int, gpuFloatComplex *,
                                      gpuFloatComplex *, gpuFloatComplex *, gpuFloatComplex *,
                                      gpuFloatComplex *, gpuFloatComplex *, int, void *);
template gpusparseStatus_t
gpsvInterleavedBatch<gpuDoubleComplex>(gpusparseHandle_t, int, int, gpuDoubleComplex *,
                                       gpuDoubleComplex *, gpuDoubleComplex *, gpuDoubleComplex *,
                                       gpuDoubleComplex *, gpuDoubleComplex *, int, void *);

// Function: csrgeam2_bufferSizeExt
template gpusparseStatus_t
csrgeam2_bufferSizeExt<float>(gpusparseHandle_t, int, int, const float *, const gpusparseMatDescr_t,
                              int, const float *, const int *, const int *, const float *,
                              const gpusparseMatDescr_t, int, const float *, const int *,
                              const int *, const gpusparseMatDescr_t, const float *, const int *,
                              const int *, size_t *);
template gpusparseStatus_t
csrgeam2_bufferSizeExt<double>(gpusparseHandle_t, int, int, const double *,
                               const gpusparseMatDescr_t, int, const double *, const int *,
                               const int *, const double *, const gpusparseMatDescr_t, int,
                               const double *, const int *, const int *, const gpusparseMatDescr_t,
                               const double *, const int *, const int *, size_t *);
template gpusparseStatus_t csrgeam2_bufferSizeExt<gpuFloatComplex>(
    gpusparseHandle_t, int, int, const gpuFloatComplex *, const gpusparseMatDescr_t, int,
    const gpuFloatComplex *, const int *, const int *, const gpuFloatComplex *,
    const gpusparseMatDescr_t, int, const gpuFloatComplex *, const int *, const int *,
    const gpusparseMatDescr_t, const gpuFloatComplex *, const int *, const int *, size_t *);
template gpusparseStatus_t csrgeam2_bufferSizeExt<gpuDoubleComplex>(
    gpusparseHandle_t, int, int, const gpuDoubleComplex *, const gpusparseMatDescr_t, int,
    const gpuDoubleComplex *, const int *, const int *, const gpuDoubleComplex *,
    const gpusparseMatDescr_t, int, const gpuDoubleComplex *, const int *, const int *,
    const gpusparseMatDescr_t, const gpuDoubleComplex *, const int *, const int *, size_t *);

// Function: csrgeam2
template gpusparseStatus_t csrgeam2<float>(gpusparseHandle_t, int, int, const float *,
                                           const gpusparseMatDescr_t, int, const float *,
                                           const int *, const int *, const float *,
                                           const gpusparseMatDescr_t, int, const float *,
                                           const int *, const int *, const gpusparseMatDescr_t,
                                           float *, int *, int *, void *);
template gpusparseStatus_t csrgeam2<double>(gpusparseHandle_t, int, int, const double *,
                                            const gpusparseMatDescr_t, int, const double *,
                                            const int *, const int *, const double *,
                                            const gpusparseMatDescr_t, int, const double *,
                                            const int *, const int *, const gpusparseMatDescr_t,
                                            double *, int *, int *, void *);
template gpusparseStatus_t
csrgeam2<gpuFloatComplex>(gpusparseHandle_t, int, int, const gpuFloatComplex *,
                          const gpusparseMatDescr_t, int, const gpuFloatComplex *, const int *,
                          const int *, const gpuFloatComplex *, const gpusparseMatDescr_t, int,
                          const gpuFloatComplex *, const int *, const int *,
                          const gpusparseMatDescr_t, gpuFloatComplex *, int *, int *, void *);
template gpusparseStatus_t
csrgeam2<gpuDoubleComplex>(gpusparseHandle_t, int, int, const gpuDoubleComplex *,
                           const gpusparseMatDescr_t, int, const gpuDoubleComplex *, const int *,
                           const int *, const gpuDoubleComplex *, const gpusparseMatDescr_t, int,
                           const gpuDoubleComplex *, const int *, const int *,
                           const gpusparseMatDescr_t, gpuDoubleComplex *, int *, int *, void *);

// Function: nnz
template gpusparseStatus_t nnz<float>(gpusparseHandle_t, gpusparseDirection_t, int, int,
                                      const gpusparseMatDescr_t, const float *, int, int *, int *);
template gpusparseStatus_t nnz<double>(gpusparseHandle_t, gpusparseDirection_t, int, int,
                                       const gpusparseMatDescr_t, const double *, int, int *,
                                       int *);
template gpusparseStatus_t nnz<gpuFloatComplex>(gpusparseHandle_t, gpusparseDirection_t, int, int,
                                                const gpusparseMatDescr_t, const gpuFloatComplex *,
                                                int, int *, int *);
template gpusparseStatus_t nnz<gpuDoubleComplex>(gpusparseHandle_t, gpusparseDirection_t, int, int,
                                                 const gpusparseMatDescr_t,
                                                 const gpuDoubleComplex *, int, int *, int *);

// Function: gebsr2gebsc_bufferSize
template gpusparseStatus_t gebsr2gebsc_bufferSize<float>(gpusparseHandle_t, int, int, int,
                                                         const float *, const int *, const int *,
                                                         int, int, size_t *);
template gpusparseStatus_t gebsr2gebsc_bufferSize<double>(gpusparseHandle_t, int, int, int,
                                                          const double *, const int *, const int *,
                                                          int, int, size_t *);
template gpusparseStatus_t gebsr2gebsc_bufferSize<gpuFloatComplex>(gpusparseHandle_t, int, int, int,
                                                                   const gpuFloatComplex *,
                                                                   const int *, const int *, int,
                                                                   int, size_t *);
template gpusparseStatus_t gebsr2gebsc_bufferSize<gpuDoubleComplex>(gpusparseHandle_t, int, int,
                                                                    int, const gpuDoubleComplex *,
                                                                    const int *, const int *, int,
                                                                    int, size_t *);

// Function: gebsr2gebsc
template gpusparseStatus_t gebsr2gebsc<float>(gpusparseHandle_t, int, int, int, const float *,
                                              const int *, const int *, int, int, float *, int *,
                                              int *, gpusparseAction_t, gpusparseIndexBase_t,
                                              void *);
template gpusparseStatus_t gebsr2gebsc<double>(gpusparseHandle_t, int, int, int, const double *,
                                               const int *, const int *, int, int, double *, int *,
                                               int *, gpusparseAction_t, gpusparseIndexBase_t,
                                               void *);
template gpusparseStatus_t gebsr2gebsc<gpuFloatComplex>(gpusparseHandle_t, int, int, int,
                                                        const gpuFloatComplex *, const int *,
                                                        const int *, int, int, gpuFloatComplex *,
                                                        int *, int *, gpusparseAction_t,
                                                        gpusparseIndexBase_t, void *);
template gpusparseStatus_t gebsr2gebsc<gpuDoubleComplex>(gpusparseHandle_t, int, int, int,
                                                         const gpuDoubleComplex *, const int *,
                                                         const int *, int, int, gpuDoubleComplex *,
                                                         int *, int *, gpusparseAction_t,
                                                         gpusparseIndexBase_t, void *);

// Function: csr2gebsr_bufferSize
template gpusparseStatus_t csr2gebsr_bufferSize<float>(gpusparseHandle_t, gpusparseDirection_t, int,
                                                       int, const gpusparseMatDescr_t,
                                                       const float *, const int *, const int *, int,
                                                       int, size_t *);
template gpusparseStatus_t csr2gebsr_bufferSize<double>(gpusparseHandle_t, gpusparseDirection_t,
                                                        int, int, const gpusparseMatDescr_t,
                                                        const double *, const int *, const int *,
                                                        int, int, size_t *);
template gpusparseStatus_t
csr2gebsr_bufferSize<gpuFloatComplex>(gpusparseHandle_t, gpusparseDirection_t, int, int,
                                      const gpusparseMatDescr_t, const gpuFloatComplex *,
                                      const int *, const int *, int, int, size_t *);
template gpusparseStatus_t
csr2gebsr_bufferSize<gpuDoubleComplex>(gpusparseHandle_t, gpusparseDirection_t, int, int,
                                       const gpusparseMatDescr_t, const gpuDoubleComplex *,
                                       const int *, const int *, int, int, size_t *);

// Function: csr2gebsr
template gpusparseStatus_t csr2gebsr<float>(gpusparseHandle_t, gpusparseDirection_t, int, int,
                                            const gpusparseMatDescr_t, const float *, const int *,
                                            const int *, const gpusparseMatDescr_t, float *, int *,
                                            int *, int, int, void *);
template gpusparseStatus_t csr2gebsr<double>(gpusparseHandle_t, gpusparseDirection_t, int, int,
                                             const gpusparseMatDescr_t, const double *, const int *,
                                             const int *, const gpusparseMatDescr_t, double *,
                                             int *, int *, int, int, void *);
template gpusparseStatus_t csr2gebsr<gpuFloatComplex>(gpusparseHandle_t, gpusparseDirection_t, int,
                                                      int, const gpusparseMatDescr_t,
                                                      const gpuFloatComplex *, const int *,
                                                      const int *, const gpusparseMatDescr_t,
                                                      gpuFloatComplex *, int *, int *, int, int,
                                                      void *);
template gpusparseStatus_t csr2gebsr<gpuDoubleComplex>(gpusparseHandle_t, gpusparseDirection_t, int,
                                                       int, const gpusparseMatDescr_t,
                                                       const gpuDoubleComplex *, const int *,
                                                       const int *, const gpusparseMatDescr_t,
                                                       gpuDoubleComplex *, int *, int *, int, int,
                                                       void *);

} // namespace wwr
