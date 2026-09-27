/**
 * @file linear_solver.cppm
 * @brief Type-safe templated wrappers for the modern (X-prefixed) GPU solver linear API
 *
 * This module provides C++ templated wrappers for the 8 modern (X-prefixed,
 * gpusolverDnParams_t-based, int64_t-dimensioned) linear-solver functions
 * that both cuSOLVER Dense and hipSOLVER Dense have: potrf/potrf_bufferSize,
 * potrs, getrf/getrf_bufferSize, getrs, geqrf/geqrf_bufferSize -- verified
 * signature by signature against hipsolver-dense.h. cuSOLVER's modern API
 * has a wider surface (larft, sytrs, trtri, plus the entire modern
 * eigenvalue/SVD API); those have no hipSOLVER counterpart, so they are not
 * wrapped -- call them through gpumod.cuda.cusolverDn directly.
 *
 * Usage:
 *   import gpumod.wrappers.solver;
 *   using namespace wwr;
 *
 *   potrf_bufferSize<float>(handle, params, uplo, n, A, lda, ...);
 */

export module gpumod.wrappers.solver:linear_solver;

import gpumod.solver;
import gpumod.blas;
import :type_traits;
import std;

export namespace wwr {

// ========================================================================
// Cholesky Factorization - potrf
// ========================================================================

template<usual_fp T>
gpusolverStatus_t potrf_bufferSize(gpusolverDnHandle_t handle, gpusolverDnParams_t params,
                                   gpublasFillMode_t uplo, int64_t n, const T *A, int64_t lda,
                                   size_t *workspaceInBytesOnDevice,
                                   size_t *workspaceInBytesOnHost) {
  return gpusolverDnXpotrf_bufferSize(
      handle, params, uplo, n, get_gpusolver_type<T>(), reinterpret_cast<const void *>(A), lda,
      get_gpusolver_type<T>(), workspaceInBytesOnDevice, workspaceInBytesOnHost);
}

template<usual_fp T>
gpusolverStatus_t potrf(gpusolverDnHandle_t handle, gpusolverDnParams_t params,
                        gpublasFillMode_t uplo, int64_t n, T *A, int64_t lda, void *bufferOnDevice,
                        size_t workspaceInBytesOnDevice, void *bufferOnHost,
                        size_t workspaceInBytesOnHost, int *info) {
  return gpusolverDnXpotrf(handle, params, uplo, n, get_gpusolver_type<T>(),
                           reinterpret_cast<void *>(A), lda, get_gpusolver_type<T>(),
                           bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost,
                           workspaceInBytesOnHost, info);
}

// ========================================================================
// Cholesky Solver - potrs
// ========================================================================

template<usual_fp T>
gpusolverStatus_t potrs(gpusolverDnHandle_t handle, gpusolverDnParams_t params,
                        gpublasFillMode_t uplo, int64_t n, int64_t nrhs, const T *A, int64_t lda,
                        T *B, int64_t ldb, int *info) {
  return gpusolverDnXpotrs(handle, params, uplo, n, nrhs, get_gpusolver_type<T>(),
                           reinterpret_cast<const void *>(A), lda, get_gpusolver_type<T>(),
                           reinterpret_cast<void *>(B), ldb, info);
}

// ========================================================================
// LU Factorization - getrf
// ========================================================================

template<usual_fp T>
gpusolverStatus_t getrf_bufferSize(gpusolverDnHandle_t handle, gpusolverDnParams_t params,
                                   int64_t m, int64_t n, const T *A, int64_t lda,
                                   size_t *workspaceInBytesOnDevice,
                                   size_t *workspaceInBytesOnHost) {
  return gpusolverDnXgetrf_bufferSize(
      handle, params, m, n, get_gpusolver_type<T>(), reinterpret_cast<const void *>(A), lda,
      get_gpusolver_type<T>(), workspaceInBytesOnDevice, workspaceInBytesOnHost);
}

template<usual_fp T>
gpusolverStatus_t getrf(gpusolverDnHandle_t handle, gpusolverDnParams_t params, int64_t m,
                        int64_t n, T *A, int64_t lda, int64_t *ipiv, void *bufferOnDevice,
                        size_t workspaceInBytesOnDevice, void *bufferOnHost,
                        size_t workspaceInBytesOnHost, int *info) {
  return gpusolverDnXgetrf(handle, params, m, n, get_gpusolver_type<T>(),
                           reinterpret_cast<void *>(A), lda, ipiv, get_gpusolver_type<T>(),
                           bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost,
                           workspaceInBytesOnHost, info);
}

// ========================================================================
// LU Solver - getrs
// ========================================================================

template<usual_fp T>
gpusolverStatus_t getrs(gpusolverDnHandle_t handle, gpusolverDnParams_t params,
                        gpublasOperation_t trans, int64_t n, int64_t nrhs, const T *A, int64_t lda,
                        const int64_t *ipiv, T *B, int64_t ldb, int *info) {
  return gpusolverDnXgetrs(handle, params, trans, n, nrhs, get_gpusolver_type<T>(),
                           reinterpret_cast<const void *>(A), lda, ipiv, get_gpusolver_type<T>(),
                           reinterpret_cast<void *>(B), ldb, info);
}

// ========================================================================
// QR Factorization - geqrf
// ========================================================================

template<usual_fp T>
gpusolverStatus_t geqrf_bufferSize(gpusolverDnHandle_t handle, gpusolverDnParams_t params,
                                   int64_t m, int64_t n, const T *A, int64_t lda, const T *tau,
                                   size_t *workspaceInBytesOnDevice,
                                   size_t *workspaceInBytesOnHost) {
  return gpusolverDnXgeqrf_bufferSize(
      handle, params, m, n, get_gpusolver_type<T>(), reinterpret_cast<const void *>(A), lda,
      get_gpusolver_type<T>(), reinterpret_cast<const void *>(tau), get_gpusolver_type<T>(),
      workspaceInBytesOnDevice, workspaceInBytesOnHost);
}

template<usual_fp T>
gpusolverStatus_t geqrf(gpusolverDnHandle_t handle, gpusolverDnParams_t params, int64_t m,
                        int64_t n, T *A, int64_t lda, T *tau, void *bufferOnDevice,
                        size_t workspaceInBytesOnDevice, void *bufferOnHost,
                        size_t workspaceInBytesOnHost, int *info) {
  return gpusolverDnXgeqrf(handle, params, m, n, get_gpusolver_type<T>(),
                           reinterpret_cast<void *>(A), lda, get_gpusolver_type<T>(),
                           reinterpret_cast<void *>(tau), get_gpusolver_type<T>(), bufferOnDevice,
                           workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info);
}

} // namespace wwr
