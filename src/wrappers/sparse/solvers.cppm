/**
 * @file solvers.cppm
 * @brief GPU sparse tridiagonal / pentadiagonal batch solvers
 *
 * Type-safe wrappers for the gtsv2 (tridiagonal) and gpsvInterleavedBatch
 * (pentadiagonal) direct solvers, each with its companion buffer-size query.
 *
 * Usage:
 *   import wwr.wrappers.sparse;
 */

module;

#include "dispatch_macros.h"

export module wwr.wrappers.sparse:solvers;

import wwr.sparse;
import wwr.complex;
import wwr.wrappers.common;
import std;

export namespace wwr {

// ========================================================================
// gtsv2: tridiagonal solve A*X = B (dense RHS)
// ========================================================================

template<usual_fp T>
gpusparseStatus_t gtsv2_bufferSizeExt(gpusparseHandle_t handle, int m, int n, const T *dl,
                                      const T *d, const T *du, const T *B, int ldb,
                                      size_t *bufferSizeInBytes) {
  WWR_USUAL_DISPATCH(T, gtsv2_bufferSizeExt, handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes);
}

template<usual_fp T>
gpusparseStatus_t gtsv2(gpusparseHandle_t handle, int m, int n, const T *dl, const T *d,
                        const T *du, T *B, int ldb, void *pBuffer) {
  WWR_USUAL_DISPATCH(T, gtsv2, handle, m, n, dl, d, du, B, ldb, pBuffer);
}

// ========================================================================
// gtsv2_nopivot: tridiagonal solve without pivoting
// ========================================================================

template<usual_fp T>
gpusparseStatus_t gtsv2_nopivot_bufferSizeExt(gpusparseHandle_t handle, int m, int n, const T *dl,
                                              const T *d, const T *du, const T *B, int ldb,
                                              size_t *bufferSizeInBytes) {
  WWR_USUAL_DISPATCH(T, gtsv2_nopivot_bufferSizeExt, handle, m, n, dl, d, du, B, ldb,
                        bufferSizeInBytes);
}

template<usual_fp T>
gpusparseStatus_t gtsv2_nopivot(gpusparseHandle_t handle, int m, int n, const T *dl, const T *d,
                                const T *du, T *B, int ldb, void *pBuffer) {
  WWR_USUAL_DISPATCH(T, gtsv2_nopivot, handle, m, n, dl, d, du, B, ldb, pBuffer);
}

// ========================================================================
// gtsv2StridedBatch: batch of tridiagonal solves, strided storage
// ========================================================================

template<usual_fp T>
gpusparseStatus_t gtsv2StridedBatch_bufferSizeExt(gpusparseHandle_t handle, int m, const T *dl,
                                                  const T *d, const T *du, const T *x,
                                                  int batchCount, int batchStride,
                                                  size_t *bufferSizeInBytes) {
  WWR_USUAL_DISPATCH(T, gtsv2StridedBatch_bufferSizeExt, handle, m, dl, d, du, x, batchCount,
                        batchStride, bufferSizeInBytes);
}

template<usual_fp T>
gpusparseStatus_t gtsv2StridedBatch(gpusparseHandle_t handle, int m, const T *dl, const T *d,
                                    const T *du, T *x, int batchCount, int batchStride,
                                    void *pBuffer) {
  WWR_USUAL_DISPATCH(T, gtsv2StridedBatch, handle, m, dl, d, du, x, batchCount, batchStride,
                        pBuffer);
}

// ========================================================================
// gtsvInterleavedBatch: batch of tridiagonal solves, interleaved storage
// NOTE: the bands (dl, d, du) and x are overwritten in place, hence non-const.
// ========================================================================

template<usual_fp T>
gpusparseStatus_t gtsvInterleavedBatch_bufferSizeExt(gpusparseHandle_t handle, int algo, int m,
                                                     const T *dl, const T *d, const T *du,
                                                     const T *x, int batchCount,
                                                     size_t *pBufferSizeInBytes) {
  WWR_USUAL_DISPATCH(T, gtsvInterleavedBatch_bufferSizeExt, handle, algo, m, dl, d, du, x,
                        batchCount, pBufferSizeInBytes);
}

template<usual_fp T>
gpusparseStatus_t gtsvInterleavedBatch(gpusparseHandle_t handle, int algo, int m, T *dl, T *d,
                                       T *du, T *x, int batchCount, void *pBuffer) {
  WWR_USUAL_DISPATCH(T, gtsvInterleavedBatch, handle, algo, m, dl, d, du, x, batchCount,
                        pBuffer);
}

// ========================================================================
// gpsvInterleavedBatch: batch of pentadiagonal solves, interleaved storage
// NOTE: the bands (ds, dl, d, du, dw) and x are overwritten in place.
// ========================================================================

template<usual_fp T>
gpusparseStatus_t gpsvInterleavedBatch_bufferSizeExt(gpusparseHandle_t handle, int algo, int m,
                                                     const T *ds, const T *dl, const T *d,
                                                     const T *du, const T *dw, const T *x,
                                                     int batchCount, size_t *pBufferSizeInBytes) {
  WWR_USUAL_DISPATCH(T, gpsvInterleavedBatch_bufferSizeExt, handle, algo, m, ds, dl, d, du, dw,
                        x, batchCount, pBufferSizeInBytes);
}

template<usual_fp T>
gpusparseStatus_t gpsvInterleavedBatch(gpusparseHandle_t handle, int algo, int m, T *ds, T *dl,
                                       T *d, T *du, T *dw, T *x, int batchCount, void *pBuffer) {
  WWR_USUAL_DISPATCH(T, gpsvInterleavedBatch, handle, algo, m, ds, dl, d, du, dw, x, batchCount,
                        pBuffer);
}

// ==================== Explicit Template Instantiations ====================
// Matching `template` instantiations live in instantiations.cpp.

// Function: gtsv2_bufferSizeExt
extern template gpusparseStatus_t gtsv2_bufferSizeExt<float>(gpusparseHandle_t, int, int,
                                                             const float *, const float *,
                                                             const float *, const float *, int,
                                                             size_t *);
extern template gpusparseStatus_t gtsv2_bufferSizeExt<double>(gpusparseHandle_t, int, int,
                                                              const double *, const double *,
                                                              const double *, const double *, int,
                                                              size_t *);
extern template gpusparseStatus_t
gtsv2_bufferSizeExt<gpuFloatComplex>(gpusparseHandle_t, int, int, const gpuFloatComplex *,
                                     const gpuFloatComplex *, const gpuFloatComplex *,
                                     const gpuFloatComplex *, int, size_t *);
extern template gpusparseStatus_t
gtsv2_bufferSizeExt<gpuDoubleComplex>(gpusparseHandle_t, int, int, const gpuDoubleComplex *,
                                      const gpuDoubleComplex *, const gpuDoubleComplex *,
                                      const gpuDoubleComplex *, int, size_t *);

// Function: gtsv2
extern template gpusparseStatus_t gtsv2<float>(gpusparseHandle_t, int, int, const float *,
                                               const float *, const float *, float *, int, void *);
extern template gpusparseStatus_t gtsv2<double>(gpusparseHandle_t, int, int, const double *,
                                                const double *, const double *, double *, int,
                                                void *);
extern template gpusparseStatus_t gtsv2<gpuFloatComplex>(gpusparseHandle_t, int, int,
                                                         const gpuFloatComplex *,
                                                         const gpuFloatComplex *,
                                                         const gpuFloatComplex *, gpuFloatComplex *,
                                                         int, void *);
extern template gpusparseStatus_t gtsv2<gpuDoubleComplex>(gpusparseHandle_t, int, int,
                                                          const gpuDoubleComplex *,
                                                          const gpuDoubleComplex *,
                                                          const gpuDoubleComplex *,
                                                          gpuDoubleComplex *, int, void *);

// Function: gtsv2_nopivot_bufferSizeExt
extern template gpusparseStatus_t gtsv2_nopivot_bufferSizeExt<float>(gpusparseHandle_t, int, int,
                                                                     const float *, const float *,
                                                                     const float *, const float *,
                                                                     int, size_t *);
extern template gpusparseStatus_t
gtsv2_nopivot_bufferSizeExt<double>(gpusparseHandle_t, int, int, const double *, const double *,
                                    const double *, const double *, int, size_t *);
extern template gpusparseStatus_t
gtsv2_nopivot_bufferSizeExt<gpuFloatComplex>(gpusparseHandle_t, int, int, const gpuFloatComplex *,
                                             const gpuFloatComplex *, const gpuFloatComplex *,
                                             const gpuFloatComplex *, int, size_t *);
extern template gpusparseStatus_t
gtsv2_nopivot_bufferSizeExt<gpuDoubleComplex>(gpusparseHandle_t, int, int, const gpuDoubleComplex *,
                                              const gpuDoubleComplex *, const gpuDoubleComplex *,
                                              const gpuDoubleComplex *, int, size_t *);

// Function: gtsv2_nopivot
extern template gpusparseStatus_t gtsv2_nopivot<float>(gpusparseHandle_t, int, int, const float *,
                                                       const float *, const float *, float *, int,
                                                       void *);
extern template gpusparseStatus_t gtsv2_nopivot<double>(gpusparseHandle_t, int, int, const double *,
                                                        const double *, const double *, double *,
                                                        int, void *);
extern template gpusparseStatus_t gtsv2_nopivot<gpuFloatComplex>(gpusparseHandle_t, int, int,
                                                                 const gpuFloatComplex *,
                                                                 const gpuFloatComplex *,
                                                                 const gpuFloatComplex *,
                                                                 gpuFloatComplex *, int, void *);
extern template gpusparseStatus_t gtsv2_nopivot<gpuDoubleComplex>(gpusparseHandle_t, int, int,
                                                                  const gpuDoubleComplex *,
                                                                  const gpuDoubleComplex *,
                                                                  const gpuDoubleComplex *,
                                                                  gpuDoubleComplex *, int, void *);

// Function: gtsv2StridedBatch_bufferSizeExt
extern template gpusparseStatus_t
gtsv2StridedBatch_bufferSizeExt<float>(gpusparseHandle_t, int, const float *, const float *,
                                       const float *, const float *, int, int, size_t *);
extern template gpusparseStatus_t
gtsv2StridedBatch_bufferSizeExt<double>(gpusparseHandle_t, int, const double *, const double *,
                                        const double *, const double *, int, int, size_t *);
extern template gpusparseStatus_t
gtsv2StridedBatch_bufferSizeExt<gpuFloatComplex>(gpusparseHandle_t, int, const gpuFloatComplex *,
                                                 const gpuFloatComplex *, const gpuFloatComplex *,
                                                 const gpuFloatComplex *, int, int, size_t *);
extern template gpusparseStatus_t gtsv2StridedBatch_bufferSizeExt<gpuDoubleComplex>(
    gpusparseHandle_t, int, const gpuDoubleComplex *, const gpuDoubleComplex *,
    const gpuDoubleComplex *, const gpuDoubleComplex *, int, int, size_t *);

// Function: gtsv2StridedBatch
extern template gpusparseStatus_t gtsv2StridedBatch<float>(gpusparseHandle_t, int, const float *,
                                                           const float *, const float *, float *,
                                                           int, int, void *);
extern template gpusparseStatus_t gtsv2StridedBatch<double>(gpusparseHandle_t, int, const double *,
                                                            const double *, const double *,
                                                            double *, int, int, void *);
extern template gpusparseStatus_t
gtsv2StridedBatch<gpuFloatComplex>(gpusparseHandle_t, int, const gpuFloatComplex *,
                                   const gpuFloatComplex *, const gpuFloatComplex *,
                                   gpuFloatComplex *, int, int, void *);
extern template gpusparseStatus_t
gtsv2StridedBatch<gpuDoubleComplex>(gpusparseHandle_t, int, const gpuDoubleComplex *,
                                    const gpuDoubleComplex *, const gpuDoubleComplex *,
                                    gpuDoubleComplex *, int, int, void *);

// Function: gtsvInterleavedBatch_bufferSizeExt
extern template gpusparseStatus_t
gtsvInterleavedBatch_bufferSizeExt<float>(gpusparseHandle_t, int, int, const float *, const float *,
                                          const float *, const float *, int, size_t *);
extern template gpusparseStatus_t
gtsvInterleavedBatch_bufferSizeExt<double>(gpusparseHandle_t, int, int, const double *,
                                           const double *, const double *, const double *, int,
                                           size_t *);
extern template gpusparseStatus_t gtsvInterleavedBatch_bufferSizeExt<gpuFloatComplex>(
    gpusparseHandle_t, int, int, const gpuFloatComplex *, const gpuFloatComplex *,
    const gpuFloatComplex *, const gpuFloatComplex *, int, size_t *);
extern template gpusparseStatus_t gtsvInterleavedBatch_bufferSizeExt<gpuDoubleComplex>(
    gpusparseHandle_t, int, int, const gpuDoubleComplex *, const gpuDoubleComplex *,
    const gpuDoubleComplex *, const gpuDoubleComplex *, int, size_t *);

// Function: gtsvInterleavedBatch
extern template gpusparseStatus_t gtsvInterleavedBatch<float>(gpusparseHandle_t, int, int, float *,
                                                              float *, float *, float *, int,
                                                              void *);
extern template gpusparseStatus_t gtsvInterleavedBatch<double>(gpusparseHandle_t, int, int,
                                                               double *, double *, double *,
                                                               double *, int, void *);
extern template gpusparseStatus_t
gtsvInterleavedBatch<gpuFloatComplex>(gpusparseHandle_t, int, int, gpuFloatComplex *,
                                      gpuFloatComplex *, gpuFloatComplex *, gpuFloatComplex *, int,
                                      void *);
extern template gpusparseStatus_t
gtsvInterleavedBatch<gpuDoubleComplex>(gpusparseHandle_t, int, int, gpuDoubleComplex *,
                                       gpuDoubleComplex *, gpuDoubleComplex *, gpuDoubleComplex *,
                                       int, void *);

// Function: gpsvInterleavedBatch_bufferSizeExt
extern template gpusparseStatus_t
gpsvInterleavedBatch_bufferSizeExt<float>(gpusparseHandle_t, int, int, const float *, const float *,
                                          const float *, const float *, const float *,
                                          const float *, int, size_t *);
extern template gpusparseStatus_t
gpsvInterleavedBatch_bufferSizeExt<double>(gpusparseHandle_t, int, int, const double *,
                                           const double *, const double *, const double *,
                                           const double *, const double *, int, size_t *);
extern template gpusparseStatus_t gpsvInterleavedBatch_bufferSizeExt<gpuFloatComplex>(
    gpusparseHandle_t, int, int, const gpuFloatComplex *, const gpuFloatComplex *,
    const gpuFloatComplex *, const gpuFloatComplex *, const gpuFloatComplex *,
    const gpuFloatComplex *, int, size_t *);
extern template gpusparseStatus_t gpsvInterleavedBatch_bufferSizeExt<gpuDoubleComplex>(
    gpusparseHandle_t, int, int, const gpuDoubleComplex *, const gpuDoubleComplex *,
    const gpuDoubleComplex *, const gpuDoubleComplex *, const gpuDoubleComplex *,
    const gpuDoubleComplex *, int, size_t *);

// Function: gpsvInterleavedBatch
extern template gpusparseStatus_t gpsvInterleavedBatch<float>(gpusparseHandle_t, int, int, float *,
                                                              float *, float *, float *, float *,
                                                              float *, int, void *);
extern template gpusparseStatus_t gpsvInterleavedBatch<double>(gpusparseHandle_t, int, int,
                                                               double *, double *, double *,
                                                               double *, double *, double *, int,
                                                               void *);
extern template gpusparseStatus_t
gpsvInterleavedBatch<gpuFloatComplex>(gpusparseHandle_t, int, int, gpuFloatComplex *,
                                      gpuFloatComplex *, gpuFloatComplex *, gpuFloatComplex *,
                                      gpuFloatComplex *, gpuFloatComplex *, int, void *);
extern template gpusparseStatus_t
gpsvInterleavedBatch<gpuDoubleComplex>(gpusparseHandle_t, int, int, gpuDoubleComplex *,
                                       gpuDoubleComplex *, gpuDoubleComplex *, gpuDoubleComplex *,
                                       gpuDoubleComplex *, gpuDoubleComplex *, int, void *);

} // namespace wwr
