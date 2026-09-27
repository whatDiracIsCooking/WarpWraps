/**
 * @file instantiations.cpp
 * @brief Explicit template instantiations for the GPU sparse wrappers
 *
 * This file contains all explicit template instantiations to avoid
 * code bloat from implicit instantiation at every call site.
 */

module wwr.wrappers.sparse;

import wwr.sparse;
import wwr.complex;
import wwr.wrappers.common;

namespace wwr {

// Function: bsrmv
template wwrsparseStatus_t bsrmv<float>(wwrsparseHandle_t, wwrsparseDirection_t,
                                        wwrsparseOperation_t, int, int, int, const float *,
                                        const wwrsparseMatDescr_t, const float *, const int *,
                                        const int *, int, const float *, const float *, float *);
template wwrsparseStatus_t bsrmv<double>(wwrsparseHandle_t, wwrsparseDirection_t,
                                         wwrsparseOperation_t, int, int, int, const double *,
                                         const wwrsparseMatDescr_t, const double *, const int *,
                                         const int *, int, const double *, const double *,
                                         double *);
template wwrsparseStatus_t
bsrmv<wwrFloatComplex>(wwrsparseHandle_t, wwrsparseDirection_t, wwrsparseOperation_t, int, int, int,
                       const wwrFloatComplex *, const wwrsparseMatDescr_t, const wwrFloatComplex *,
                       const int *, const int *, int, const wwrFloatComplex *,
                       const wwrFloatComplex *, wwrFloatComplex *);
template wwrsparseStatus_t
bsrmv<wwrDoubleComplex>(wwrsparseHandle_t, wwrsparseDirection_t, wwrsparseOperation_t, int, int,
                        int, const wwrDoubleComplex *, const wwrsparseMatDescr_t,
                        const wwrDoubleComplex *, const int *, const int *, int,
                        const wwrDoubleComplex *, const wwrDoubleComplex *, wwrDoubleComplex *);

// Function: gtsv2_bufferSizeExt
template wwrsparseStatus_t gtsv2_bufferSizeExt<float>(wwrsparseHandle_t, int, int, const float *,
                                                      const float *, const float *, const float *,
                                                      int, size_t *);
template wwrsparseStatus_t gtsv2_bufferSizeExt<double>(wwrsparseHandle_t, int, int, const double *,
                                                       const double *, const double *,
                                                       const double *, int, size_t *);
template wwrsparseStatus_t
gtsv2_bufferSizeExt<wwrFloatComplex>(wwrsparseHandle_t, int, int, const wwrFloatComplex *,
                                     const wwrFloatComplex *, const wwrFloatComplex *,
                                     const wwrFloatComplex *, int, size_t *);
template wwrsparseStatus_t
gtsv2_bufferSizeExt<wwrDoubleComplex>(wwrsparseHandle_t, int, int, const wwrDoubleComplex *,
                                      const wwrDoubleComplex *, const wwrDoubleComplex *,
                                      const wwrDoubleComplex *, int, size_t *);

// Function: gtsv2
template wwrsparseStatus_t gtsv2<float>(wwrsparseHandle_t, int, int, const float *, const float *,
                                        const float *, float *, int, void *);
template wwrsparseStatus_t gtsv2<double>(wwrsparseHandle_t, int, int, const double *,
                                         const double *, const double *, double *, int, void *);
template wwrsparseStatus_t gtsv2<wwrFloatComplex>(wwrsparseHandle_t, int, int,
                                                  const wwrFloatComplex *, const wwrFloatComplex *,
                                                  const wwrFloatComplex *, wwrFloatComplex *, int,
                                                  void *);
template wwrsparseStatus_t gtsv2<wwrDoubleComplex>(wwrsparseHandle_t, int, int,
                                                   const wwrDoubleComplex *,
                                                   const wwrDoubleComplex *,
                                                   const wwrDoubleComplex *, wwrDoubleComplex *,
                                                   int, void *);

// Function: gtsv2_nopivot_bufferSizeExt
template wwrsparseStatus_t gtsv2_nopivot_bufferSizeExt<float>(wwrsparseHandle_t, int, int,
                                                              const float *, const float *,
                                                              const float *, const float *, int,
                                                              size_t *);
template wwrsparseStatus_t gtsv2_nopivot_bufferSizeExt<double>(wwrsparseHandle_t, int, int,
                                                               const double *, const double *,
                                                               const double *, const double *, int,
                                                               size_t *);
template wwrsparseStatus_t
gtsv2_nopivot_bufferSizeExt<wwrFloatComplex>(wwrsparseHandle_t, int, int, const wwrFloatComplex *,
                                             const wwrFloatComplex *, const wwrFloatComplex *,
                                             const wwrFloatComplex *, int, size_t *);
template wwrsparseStatus_t
gtsv2_nopivot_bufferSizeExt<wwrDoubleComplex>(wwrsparseHandle_t, int, int, const wwrDoubleComplex *,
                                              const wwrDoubleComplex *, const wwrDoubleComplex *,
                                              const wwrDoubleComplex *, int, size_t *);

// Function: gtsv2_nopivot
template wwrsparseStatus_t gtsv2_nopivot<float>(wwrsparseHandle_t, int, int, const float *,
                                                const float *, const float *, float *, int, void *);
template wwrsparseStatus_t gtsv2_nopivot<double>(wwrsparseHandle_t, int, int, const double *,
                                                 const double *, const double *, double *, int,
                                                 void *);
template wwrsparseStatus_t gtsv2_nopivot<wwrFloatComplex>(wwrsparseHandle_t, int, int,
                                                          const wwrFloatComplex *,
                                                          const wwrFloatComplex *,
                                                          const wwrFloatComplex *,
                                                          wwrFloatComplex *, int, void *);
template wwrsparseStatus_t gtsv2_nopivot<wwrDoubleComplex>(wwrsparseHandle_t, int, int,
                                                           const wwrDoubleComplex *,
                                                           const wwrDoubleComplex *,
                                                           const wwrDoubleComplex *,
                                                           wwrDoubleComplex *, int, void *);

// Function: gtsv2StridedBatch_bufferSizeExt
template wwrsparseStatus_t gtsv2StridedBatch_bufferSizeExt<float>(wwrsparseHandle_t, int,
                                                                  const float *, const float *,
                                                                  const float *, const float *, int,
                                                                  int, size_t *);
template wwrsparseStatus_t gtsv2StridedBatch_bufferSizeExt<double>(wwrsparseHandle_t, int,
                                                                   const double *, const double *,
                                                                   const double *, const double *,
                                                                   int, int, size_t *);
template wwrsparseStatus_t
gtsv2StridedBatch_bufferSizeExt<wwrFloatComplex>(wwrsparseHandle_t, int, const wwrFloatComplex *,
                                                 const wwrFloatComplex *, const wwrFloatComplex *,
                                                 const wwrFloatComplex *, int, int, size_t *);
template wwrsparseStatus_t gtsv2StridedBatch_bufferSizeExt<wwrDoubleComplex>(
    wwrsparseHandle_t, int, const wwrDoubleComplex *, const wwrDoubleComplex *,
    const wwrDoubleComplex *, const wwrDoubleComplex *, int, int, size_t *);

// Function: gtsv2StridedBatch
template wwrsparseStatus_t gtsv2StridedBatch<float>(wwrsparseHandle_t, int, const float *,
                                                    const float *, const float *, float *, int, int,
                                                    void *);
template wwrsparseStatus_t gtsv2StridedBatch<double>(wwrsparseHandle_t, int, const double *,
                                                     const double *, const double *, double *, int,
                                                     int, void *);
template wwrsparseStatus_t gtsv2StridedBatch<wwrFloatComplex>(wwrsparseHandle_t, int,
                                                              const wwrFloatComplex *,
                                                              const wwrFloatComplex *,
                                                              const wwrFloatComplex *,
                                                              wwrFloatComplex *, int, int, void *);
template wwrsparseStatus_t
gtsv2StridedBatch<wwrDoubleComplex>(wwrsparseHandle_t, int, const wwrDoubleComplex *,
                                    const wwrDoubleComplex *, const wwrDoubleComplex *,
                                    wwrDoubleComplex *, int, int, void *);

// Function: gtsvInterleavedBatch_bufferSizeExt
template wwrsparseStatus_t gtsvInterleavedBatch_bufferSizeExt<float>(wwrsparseHandle_t, int, int,
                                                                     const float *, const float *,
                                                                     const float *, const float *,
                                                                     int, size_t *);
template wwrsparseStatus_t
gtsvInterleavedBatch_bufferSizeExt<double>(wwrsparseHandle_t, int, int, const double *,
                                           const double *, const double *, const double *, int,
                                           size_t *);
template wwrsparseStatus_t gtsvInterleavedBatch_bufferSizeExt<wwrFloatComplex>(
    wwrsparseHandle_t, int, int, const wwrFloatComplex *, const wwrFloatComplex *,
    const wwrFloatComplex *, const wwrFloatComplex *, int, size_t *);
template wwrsparseStatus_t gtsvInterleavedBatch_bufferSizeExt<wwrDoubleComplex>(
    wwrsparseHandle_t, int, int, const wwrDoubleComplex *, const wwrDoubleComplex *,
    const wwrDoubleComplex *, const wwrDoubleComplex *, int, size_t *);

// Function: gtsvInterleavedBatch
template wwrsparseStatus_t gtsvInterleavedBatch<float>(wwrsparseHandle_t, int, int, float *,
                                                       float *, float *, float *, int, void *);
template wwrsparseStatus_t gtsvInterleavedBatch<double>(wwrsparseHandle_t, int, int, double *,
                                                        double *, double *, double *, int, void *);
template wwrsparseStatus_t gtsvInterleavedBatch<wwrFloatComplex>(wwrsparseHandle_t, int, int,
                                                                 wwrFloatComplex *,
                                                                 wwrFloatComplex *,
                                                                 wwrFloatComplex *,
                                                                 wwrFloatComplex *, int, void *);
template wwrsparseStatus_t gtsvInterleavedBatch<wwrDoubleComplex>(wwrsparseHandle_t, int, int,
                                                                  wwrDoubleComplex *,
                                                                  wwrDoubleComplex *,
                                                                  wwrDoubleComplex *,
                                                                  wwrDoubleComplex *, int, void *);

// Function: gpsvInterleavedBatch_bufferSizeExt
template wwrsparseStatus_t gpsvInterleavedBatch_bufferSizeExt<float>(wwrsparseHandle_t, int, int,
                                                                     const float *, const float *,
                                                                     const float *, const float *,
                                                                     const float *, const float *,
                                                                     int, size_t *);
template wwrsparseStatus_t
gpsvInterleavedBatch_bufferSizeExt<double>(wwrsparseHandle_t, int, int, const double *,
                                           const double *, const double *, const double *,
                                           const double *, const double *, int, size_t *);
template wwrsparseStatus_t gpsvInterleavedBatch_bufferSizeExt<wwrFloatComplex>(
    wwrsparseHandle_t, int, int, const wwrFloatComplex *, const wwrFloatComplex *,
    const wwrFloatComplex *, const wwrFloatComplex *, const wwrFloatComplex *,
    const wwrFloatComplex *, int, size_t *);
template wwrsparseStatus_t gpsvInterleavedBatch_bufferSizeExt<wwrDoubleComplex>(
    wwrsparseHandle_t, int, int, const wwrDoubleComplex *, const wwrDoubleComplex *,
    const wwrDoubleComplex *, const wwrDoubleComplex *, const wwrDoubleComplex *,
    const wwrDoubleComplex *, int, size_t *);

// Function: gpsvInterleavedBatch
template wwrsparseStatus_t gpsvInterleavedBatch<float>(wwrsparseHandle_t, int, int, float *,
                                                       float *, float *, float *, float *, float *,
                                                       int, void *);
template wwrsparseStatus_t gpsvInterleavedBatch<double>(wwrsparseHandle_t, int, int, double *,
                                                        double *, double *, double *, double *,
                                                        double *, int, void *);
template wwrsparseStatus_t
gpsvInterleavedBatch<wwrFloatComplex>(wwrsparseHandle_t, int, int, wwrFloatComplex *,
                                      wwrFloatComplex *, wwrFloatComplex *, wwrFloatComplex *,
                                      wwrFloatComplex *, wwrFloatComplex *, int, void *);
template wwrsparseStatus_t
gpsvInterleavedBatch<wwrDoubleComplex>(wwrsparseHandle_t, int, int, wwrDoubleComplex *,
                                       wwrDoubleComplex *, wwrDoubleComplex *, wwrDoubleComplex *,
                                       wwrDoubleComplex *, wwrDoubleComplex *, int, void *);

// Function: csrgeam2_bufferSizeExt
template wwrsparseStatus_t
csrgeam2_bufferSizeExt<float>(wwrsparseHandle_t, int, int, const float *, const wwrsparseMatDescr_t,
                              int, const float *, const int *, const int *, const float *,
                              const wwrsparseMatDescr_t, int, const float *, const int *,
                              const int *, const wwrsparseMatDescr_t, const float *, const int *,
                              const int *, size_t *);
template wwrsparseStatus_t
csrgeam2_bufferSizeExt<double>(wwrsparseHandle_t, int, int, const double *,
                               const wwrsparseMatDescr_t, int, const double *, const int *,
                               const int *, const double *, const wwrsparseMatDescr_t, int,
                               const double *, const int *, const int *, const wwrsparseMatDescr_t,
                               const double *, const int *, const int *, size_t *);
template wwrsparseStatus_t csrgeam2_bufferSizeExt<wwrFloatComplex>(
    wwrsparseHandle_t, int, int, const wwrFloatComplex *, const wwrsparseMatDescr_t, int,
    const wwrFloatComplex *, const int *, const int *, const wwrFloatComplex *,
    const wwrsparseMatDescr_t, int, const wwrFloatComplex *, const int *, const int *,
    const wwrsparseMatDescr_t, const wwrFloatComplex *, const int *, const int *, size_t *);
template wwrsparseStatus_t csrgeam2_bufferSizeExt<wwrDoubleComplex>(
    wwrsparseHandle_t, int, int, const wwrDoubleComplex *, const wwrsparseMatDescr_t, int,
    const wwrDoubleComplex *, const int *, const int *, const wwrDoubleComplex *,
    const wwrsparseMatDescr_t, int, const wwrDoubleComplex *, const int *, const int *,
    const wwrsparseMatDescr_t, const wwrDoubleComplex *, const int *, const int *, size_t *);

// Function: csrgeam2
template wwrsparseStatus_t csrgeam2<float>(wwrsparseHandle_t, int, int, const float *,
                                           const wwrsparseMatDescr_t, int, const float *,
                                           const int *, const int *, const float *,
                                           const wwrsparseMatDescr_t, int, const float *,
                                           const int *, const int *, const wwrsparseMatDescr_t,
                                           float *, int *, int *, void *);
template wwrsparseStatus_t csrgeam2<double>(wwrsparseHandle_t, int, int, const double *,
                                            const wwrsparseMatDescr_t, int, const double *,
                                            const int *, const int *, const double *,
                                            const wwrsparseMatDescr_t, int, const double *,
                                            const int *, const int *, const wwrsparseMatDescr_t,
                                            double *, int *, int *, void *);
template wwrsparseStatus_t
csrgeam2<wwrFloatComplex>(wwrsparseHandle_t, int, int, const wwrFloatComplex *,
                          const wwrsparseMatDescr_t, int, const wwrFloatComplex *, const int *,
                          const int *, const wwrFloatComplex *, const wwrsparseMatDescr_t, int,
                          const wwrFloatComplex *, const int *, const int *,
                          const wwrsparseMatDescr_t, wwrFloatComplex *, int *, int *, void *);
template wwrsparseStatus_t
csrgeam2<wwrDoubleComplex>(wwrsparseHandle_t, int, int, const wwrDoubleComplex *,
                           const wwrsparseMatDescr_t, int, const wwrDoubleComplex *, const int *,
                           const int *, const wwrDoubleComplex *, const wwrsparseMatDescr_t, int,
                           const wwrDoubleComplex *, const int *, const int *,
                           const wwrsparseMatDescr_t, wwrDoubleComplex *, int *, int *, void *);

// Function: nnz
template wwrsparseStatus_t nnz<float>(wwrsparseHandle_t, wwrsparseDirection_t, int, int,
                                      const wwrsparseMatDescr_t, const float *, int, int *, int *);
template wwrsparseStatus_t nnz<double>(wwrsparseHandle_t, wwrsparseDirection_t, int, int,
                                       const wwrsparseMatDescr_t, const double *, int, int *,
                                       int *);
template wwrsparseStatus_t nnz<wwrFloatComplex>(wwrsparseHandle_t, wwrsparseDirection_t, int, int,
                                                const wwrsparseMatDescr_t, const wwrFloatComplex *,
                                                int, int *, int *);
template wwrsparseStatus_t nnz<wwrDoubleComplex>(wwrsparseHandle_t, wwrsparseDirection_t, int, int,
                                                 const wwrsparseMatDescr_t,
                                                 const wwrDoubleComplex *, int, int *, int *);

// Function: gebsr2gebsc_bufferSize
template wwrsparseStatus_t gebsr2gebsc_bufferSize<float>(wwrsparseHandle_t, int, int, int,
                                                         const float *, const int *, const int *,
                                                         int, int, size_t *);
template wwrsparseStatus_t gebsr2gebsc_bufferSize<double>(wwrsparseHandle_t, int, int, int,
                                                          const double *, const int *, const int *,
                                                          int, int, size_t *);
template wwrsparseStatus_t gebsr2gebsc_bufferSize<wwrFloatComplex>(wwrsparseHandle_t, int, int, int,
                                                                   const wwrFloatComplex *,
                                                                   const int *, const int *, int,
                                                                   int, size_t *);
template wwrsparseStatus_t gebsr2gebsc_bufferSize<wwrDoubleComplex>(wwrsparseHandle_t, int, int,
                                                                    int, const wwrDoubleComplex *,
                                                                    const int *, const int *, int,
                                                                    int, size_t *);

// Function: gebsr2gebsc
template wwrsparseStatus_t gebsr2gebsc<float>(wwrsparseHandle_t, int, int, int, const float *,
                                              const int *, const int *, int, int, float *, int *,
                                              int *, wwrsparseAction_t, wwrsparseIndexBase_t,
                                              void *);
template wwrsparseStatus_t gebsr2gebsc<double>(wwrsparseHandle_t, int, int, int, const double *,
                                               const int *, const int *, int, int, double *, int *,
                                               int *, wwrsparseAction_t, wwrsparseIndexBase_t,
                                               void *);
template wwrsparseStatus_t gebsr2gebsc<wwrFloatComplex>(wwrsparseHandle_t, int, int, int,
                                                        const wwrFloatComplex *, const int *,
                                                        const int *, int, int, wwrFloatComplex *,
                                                        int *, int *, wwrsparseAction_t,
                                                        wwrsparseIndexBase_t, void *);
template wwrsparseStatus_t gebsr2gebsc<wwrDoubleComplex>(wwrsparseHandle_t, int, int, int,
                                                         const wwrDoubleComplex *, const int *,
                                                         const int *, int, int, wwrDoubleComplex *,
                                                         int *, int *, wwrsparseAction_t,
                                                         wwrsparseIndexBase_t, void *);

// Function: csr2gebsr_bufferSize
template wwrsparseStatus_t csr2gebsr_bufferSize<float>(wwrsparseHandle_t, wwrsparseDirection_t, int,
                                                       int, const wwrsparseMatDescr_t,
                                                       const float *, const int *, const int *, int,
                                                       int, size_t *);
template wwrsparseStatus_t csr2gebsr_bufferSize<double>(wwrsparseHandle_t, wwrsparseDirection_t,
                                                        int, int, const wwrsparseMatDescr_t,
                                                        const double *, const int *, const int *,
                                                        int, int, size_t *);
template wwrsparseStatus_t
csr2gebsr_bufferSize<wwrFloatComplex>(wwrsparseHandle_t, wwrsparseDirection_t, int, int,
                                      const wwrsparseMatDescr_t, const wwrFloatComplex *,
                                      const int *, const int *, int, int, size_t *);
template wwrsparseStatus_t
csr2gebsr_bufferSize<wwrDoubleComplex>(wwrsparseHandle_t, wwrsparseDirection_t, int, int,
                                       const wwrsparseMatDescr_t, const wwrDoubleComplex *,
                                       const int *, const int *, int, int, size_t *);

// Function: csr2gebsr
template wwrsparseStatus_t csr2gebsr<float>(wwrsparseHandle_t, wwrsparseDirection_t, int, int,
                                            const wwrsparseMatDescr_t, const float *, const int *,
                                            const int *, const wwrsparseMatDescr_t, float *, int *,
                                            int *, int, int, void *);
template wwrsparseStatus_t csr2gebsr<double>(wwrsparseHandle_t, wwrsparseDirection_t, int, int,
                                             const wwrsparseMatDescr_t, const double *, const int *,
                                             const int *, const wwrsparseMatDescr_t, double *,
                                             int *, int *, int, int, void *);
template wwrsparseStatus_t csr2gebsr<wwrFloatComplex>(wwrsparseHandle_t, wwrsparseDirection_t, int,
                                                      int, const wwrsparseMatDescr_t,
                                                      const wwrFloatComplex *, const int *,
                                                      const int *, const wwrsparseMatDescr_t,
                                                      wwrFloatComplex *, int *, int *, int, int,
                                                      void *);
template wwrsparseStatus_t csr2gebsr<wwrDoubleComplex>(wwrsparseHandle_t, wwrsparseDirection_t, int,
                                                       int, const wwrsparseMatDescr_t,
                                                       const wwrDoubleComplex *, const int *,
                                                       const int *, const wwrsparseMatDescr_t,
                                                       wwrDoubleComplex *, int *, int *, int, int,
                                                       void *);

} // namespace wwr
