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
wwrsparseStatus_t gtsv2_bufferSizeExt(wwrsparseHandle_t handle, int m, int n, const T *dl,
                                      const T *d, const T *du, const T *B, int ldb,
                                      size_t *bufferSizeInBytes) {
  WWR_USUAL_DISPATCH(T, gtsv2_bufferSizeExt, handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes);
}

template<usual_fp T>
wwrsparseStatus_t gtsv2(wwrsparseHandle_t handle, int m, int n, const T *dl, const T *d,
                        const T *du, T *B, int ldb, void *pBuffer) {
  WWR_USUAL_DISPATCH(T, gtsv2, handle, m, n, dl, d, du, B, ldb, pBuffer);
}

// ========================================================================
// gtsv2_nopivot: tridiagonal solve without pivoting
// ========================================================================

template<usual_fp T>
wwrsparseStatus_t gtsv2_nopivot_bufferSizeExt(wwrsparseHandle_t handle, int m, int n, const T *dl,
                                              const T *d, const T *du, const T *B, int ldb,
                                              size_t *bufferSizeInBytes) {
  WWR_USUAL_DISPATCH(T, gtsv2_nopivot_bufferSizeExt, handle, m, n, dl, d, du, B, ldb,
                        bufferSizeInBytes);
}

template<usual_fp T>
wwrsparseStatus_t gtsv2_nopivot(wwrsparseHandle_t handle, int m, int n, const T *dl, const T *d,
                                const T *du, T *B, int ldb, void *pBuffer) {
  WWR_USUAL_DISPATCH(T, gtsv2_nopivot, handle, m, n, dl, d, du, B, ldb, pBuffer);
}

// ========================================================================
// gtsv2StridedBatch: batch of tridiagonal solves, strided storage
// ========================================================================

template<usual_fp T>
wwrsparseStatus_t gtsv2StridedBatch_bufferSizeExt(wwrsparseHandle_t handle, int m, const T *dl,
                                                  const T *d, const T *du, const T *x,
                                                  int batchCount, int batchStride,
                                                  size_t *bufferSizeInBytes) {
  WWR_USUAL_DISPATCH(T, gtsv2StridedBatch_bufferSizeExt, handle, m, dl, d, du, x, batchCount,
                        batchStride, bufferSizeInBytes);
}

template<usual_fp T>
wwrsparseStatus_t gtsv2StridedBatch(wwrsparseHandle_t handle, int m, const T *dl, const T *d,
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
wwrsparseStatus_t gtsvInterleavedBatch_bufferSizeExt(wwrsparseHandle_t handle, int algo, int m,
                                                     const T *dl, const T *d, const T *du,
                                                     const T *x, int batchCount,
                                                     size_t *pBufferSizeInBytes) {
  WWR_USUAL_DISPATCH(T, gtsvInterleavedBatch_bufferSizeExt, handle, algo, m, dl, d, du, x,
                        batchCount, pBufferSizeInBytes);
}

template<usual_fp T>
wwrsparseStatus_t gtsvInterleavedBatch(wwrsparseHandle_t handle, int algo, int m, T *dl, T *d,
                                       T *du, T *x, int batchCount, void *pBuffer) {
  WWR_USUAL_DISPATCH(T, gtsvInterleavedBatch, handle, algo, m, dl, d, du, x, batchCount,
                        pBuffer);
}

// ========================================================================
// gpsvInterleavedBatch: batch of pentadiagonal solves, interleaved storage
// NOTE: the bands (ds, dl, d, du, dw) and x are overwritten in place.
// ========================================================================

template<usual_fp T>
wwrsparseStatus_t gpsvInterleavedBatch_bufferSizeExt(wwrsparseHandle_t handle, int algo, int m,
                                                     const T *ds, const T *dl, const T *d,
                                                     const T *du, const T *dw, const T *x,
                                                     int batchCount, size_t *pBufferSizeInBytes) {
  WWR_USUAL_DISPATCH(T, gpsvInterleavedBatch_bufferSizeExt, handle, algo, m, ds, dl, d, du, dw,
                        x, batchCount, pBufferSizeInBytes);
}

template<usual_fp T>
wwrsparseStatus_t gpsvInterleavedBatch(wwrsparseHandle_t handle, int algo, int m, T *ds, T *dl,
                                       T *d, T *du, T *dw, T *x, int batchCount, void *pBuffer) {
  WWR_USUAL_DISPATCH(T, gpsvInterleavedBatch, handle, algo, m, ds, dl, d, du, dw, x, batchCount,
                        pBuffer);
}

// ==================== Explicit Template Instantiations ====================
// Matching `template` instantiations live in instantiations.cpp.

// Function: gtsv2_bufferSizeExt
extern template wwrsparseStatus_t gtsv2_bufferSizeExt<float>(wwrsparseHandle_t, int, int,
                                                             const float *, const float *,
                                                             const float *, const float *, int,
                                                             size_t *);
extern template wwrsparseStatus_t gtsv2_bufferSizeExt<double>(wwrsparseHandle_t, int, int,
                                                              const double *, const double *,
                                                              const double *, const double *, int,
                                                              size_t *);
extern template wwrsparseStatus_t
gtsv2_bufferSizeExt<wwrFloatComplex>(wwrsparseHandle_t, int, int, const wwrFloatComplex *,
                                     const wwrFloatComplex *, const wwrFloatComplex *,
                                     const wwrFloatComplex *, int, size_t *);
extern template wwrsparseStatus_t
gtsv2_bufferSizeExt<wwrDoubleComplex>(wwrsparseHandle_t, int, int, const wwrDoubleComplex *,
                                      const wwrDoubleComplex *, const wwrDoubleComplex *,
                                      const wwrDoubleComplex *, int, size_t *);

// Function: gtsv2
extern template wwrsparseStatus_t gtsv2<float>(wwrsparseHandle_t, int, int, const float *,
                                               const float *, const float *, float *, int, void *);
extern template wwrsparseStatus_t gtsv2<double>(wwrsparseHandle_t, int, int, const double *,
                                                const double *, const double *, double *, int,
                                                void *);
extern template wwrsparseStatus_t gtsv2<wwrFloatComplex>(wwrsparseHandle_t, int, int,
                                                         const wwrFloatComplex *,
                                                         const wwrFloatComplex *,
                                                         const wwrFloatComplex *, wwrFloatComplex *,
                                                         int, void *);
extern template wwrsparseStatus_t gtsv2<wwrDoubleComplex>(wwrsparseHandle_t, int, int,
                                                          const wwrDoubleComplex *,
                                                          const wwrDoubleComplex *,
                                                          const wwrDoubleComplex *,
                                                          wwrDoubleComplex *, int, void *);

// Function: gtsv2_nopivot_bufferSizeExt
extern template wwrsparseStatus_t gtsv2_nopivot_bufferSizeExt<float>(wwrsparseHandle_t, int, int,
                                                                     const float *, const float *,
                                                                     const float *, const float *,
                                                                     int, size_t *);
extern template wwrsparseStatus_t
gtsv2_nopivot_bufferSizeExt<double>(wwrsparseHandle_t, int, int, const double *, const double *,
                                    const double *, const double *, int, size_t *);
extern template wwrsparseStatus_t
gtsv2_nopivot_bufferSizeExt<wwrFloatComplex>(wwrsparseHandle_t, int, int, const wwrFloatComplex *,
                                             const wwrFloatComplex *, const wwrFloatComplex *,
                                             const wwrFloatComplex *, int, size_t *);
extern template wwrsparseStatus_t
gtsv2_nopivot_bufferSizeExt<wwrDoubleComplex>(wwrsparseHandle_t, int, int, const wwrDoubleComplex *,
                                              const wwrDoubleComplex *, const wwrDoubleComplex *,
                                              const wwrDoubleComplex *, int, size_t *);

// Function: gtsv2_nopivot
extern template wwrsparseStatus_t gtsv2_nopivot<float>(wwrsparseHandle_t, int, int, const float *,
                                                       const float *, const float *, float *, int,
                                                       void *);
extern template wwrsparseStatus_t gtsv2_nopivot<double>(wwrsparseHandle_t, int, int, const double *,
                                                        const double *, const double *, double *,
                                                        int, void *);
extern template wwrsparseStatus_t gtsv2_nopivot<wwrFloatComplex>(wwrsparseHandle_t, int, int,
                                                                 const wwrFloatComplex *,
                                                                 const wwrFloatComplex *,
                                                                 const wwrFloatComplex *,
                                                                 wwrFloatComplex *, int, void *);
extern template wwrsparseStatus_t gtsv2_nopivot<wwrDoubleComplex>(wwrsparseHandle_t, int, int,
                                                                  const wwrDoubleComplex *,
                                                                  const wwrDoubleComplex *,
                                                                  const wwrDoubleComplex *,
                                                                  wwrDoubleComplex *, int, void *);

// Function: gtsv2StridedBatch_bufferSizeExt
extern template wwrsparseStatus_t
gtsv2StridedBatch_bufferSizeExt<float>(wwrsparseHandle_t, int, const float *, const float *,
                                       const float *, const float *, int, int, size_t *);
extern template wwrsparseStatus_t
gtsv2StridedBatch_bufferSizeExt<double>(wwrsparseHandle_t, int, const double *, const double *,
                                        const double *, const double *, int, int, size_t *);
extern template wwrsparseStatus_t
gtsv2StridedBatch_bufferSizeExt<wwrFloatComplex>(wwrsparseHandle_t, int, const wwrFloatComplex *,
                                                 const wwrFloatComplex *, const wwrFloatComplex *,
                                                 const wwrFloatComplex *, int, int, size_t *);
extern template wwrsparseStatus_t gtsv2StridedBatch_bufferSizeExt<wwrDoubleComplex>(
    wwrsparseHandle_t, int, const wwrDoubleComplex *, const wwrDoubleComplex *,
    const wwrDoubleComplex *, const wwrDoubleComplex *, int, int, size_t *);

// Function: gtsv2StridedBatch
extern template wwrsparseStatus_t gtsv2StridedBatch<float>(wwrsparseHandle_t, int, const float *,
                                                           const float *, const float *, float *,
                                                           int, int, void *);
extern template wwrsparseStatus_t gtsv2StridedBatch<double>(wwrsparseHandle_t, int, const double *,
                                                            const double *, const double *,
                                                            double *, int, int, void *);
extern template wwrsparseStatus_t
gtsv2StridedBatch<wwrFloatComplex>(wwrsparseHandle_t, int, const wwrFloatComplex *,
                                   const wwrFloatComplex *, const wwrFloatComplex *,
                                   wwrFloatComplex *, int, int, void *);
extern template wwrsparseStatus_t
gtsv2StridedBatch<wwrDoubleComplex>(wwrsparseHandle_t, int, const wwrDoubleComplex *,
                                    const wwrDoubleComplex *, const wwrDoubleComplex *,
                                    wwrDoubleComplex *, int, int, void *);

// Function: gtsvInterleavedBatch_bufferSizeExt
extern template wwrsparseStatus_t
gtsvInterleavedBatch_bufferSizeExt<float>(wwrsparseHandle_t, int, int, const float *, const float *,
                                          const float *, const float *, int, size_t *);
extern template wwrsparseStatus_t
gtsvInterleavedBatch_bufferSizeExt<double>(wwrsparseHandle_t, int, int, const double *,
                                           const double *, const double *, const double *, int,
                                           size_t *);
extern template wwrsparseStatus_t gtsvInterleavedBatch_bufferSizeExt<wwrFloatComplex>(
    wwrsparseHandle_t, int, int, const wwrFloatComplex *, const wwrFloatComplex *,
    const wwrFloatComplex *, const wwrFloatComplex *, int, size_t *);
extern template wwrsparseStatus_t gtsvInterleavedBatch_bufferSizeExt<wwrDoubleComplex>(
    wwrsparseHandle_t, int, int, const wwrDoubleComplex *, const wwrDoubleComplex *,
    const wwrDoubleComplex *, const wwrDoubleComplex *, int, size_t *);

// Function: gtsvInterleavedBatch
extern template wwrsparseStatus_t gtsvInterleavedBatch<float>(wwrsparseHandle_t, int, int, float *,
                                                              float *, float *, float *, int,
                                                              void *);
extern template wwrsparseStatus_t gtsvInterleavedBatch<double>(wwrsparseHandle_t, int, int,
                                                               double *, double *, double *,
                                                               double *, int, void *);
extern template wwrsparseStatus_t
gtsvInterleavedBatch<wwrFloatComplex>(wwrsparseHandle_t, int, int, wwrFloatComplex *,
                                      wwrFloatComplex *, wwrFloatComplex *, wwrFloatComplex *, int,
                                      void *);
extern template wwrsparseStatus_t
gtsvInterleavedBatch<wwrDoubleComplex>(wwrsparseHandle_t, int, int, wwrDoubleComplex *,
                                       wwrDoubleComplex *, wwrDoubleComplex *, wwrDoubleComplex *,
                                       int, void *);

// Function: gpsvInterleavedBatch_bufferSizeExt
extern template wwrsparseStatus_t
gpsvInterleavedBatch_bufferSizeExt<float>(wwrsparseHandle_t, int, int, const float *, const float *,
                                          const float *, const float *, const float *,
                                          const float *, int, size_t *);
extern template wwrsparseStatus_t
gpsvInterleavedBatch_bufferSizeExt<double>(wwrsparseHandle_t, int, int, const double *,
                                           const double *, const double *, const double *,
                                           const double *, const double *, int, size_t *);
extern template wwrsparseStatus_t gpsvInterleavedBatch_bufferSizeExt<wwrFloatComplex>(
    wwrsparseHandle_t, int, int, const wwrFloatComplex *, const wwrFloatComplex *,
    const wwrFloatComplex *, const wwrFloatComplex *, const wwrFloatComplex *,
    const wwrFloatComplex *, int, size_t *);
extern template wwrsparseStatus_t gpsvInterleavedBatch_bufferSizeExt<wwrDoubleComplex>(
    wwrsparseHandle_t, int, int, const wwrDoubleComplex *, const wwrDoubleComplex *,
    const wwrDoubleComplex *, const wwrDoubleComplex *, const wwrDoubleComplex *,
    const wwrDoubleComplex *, int, size_t *);

// Function: gpsvInterleavedBatch
extern template wwrsparseStatus_t gpsvInterleavedBatch<float>(wwrsparseHandle_t, int, int, float *,
                                                              float *, float *, float *, float *,
                                                              float *, int, void *);
extern template wwrsparseStatus_t gpsvInterleavedBatch<double>(wwrsparseHandle_t, int, int,
                                                               double *, double *, double *,
                                                               double *, double *, double *, int,
                                                               void *);
extern template wwrsparseStatus_t
gpsvInterleavedBatch<wwrFloatComplex>(wwrsparseHandle_t, int, int, wwrFloatComplex *,
                                      wwrFloatComplex *, wwrFloatComplex *, wwrFloatComplex *,
                                      wwrFloatComplex *, wwrFloatComplex *, int, void *);
extern template wwrsparseStatus_t
gpsvInterleavedBatch<wwrDoubleComplex>(wwrsparseHandle_t, int, int, wwrDoubleComplex *,
                                       wwrDoubleComplex *, wwrDoubleComplex *, wwrDoubleComplex *,
                                       wwrDoubleComplex *, wwrDoubleComplex *, int, void *);

} // namespace wwr
