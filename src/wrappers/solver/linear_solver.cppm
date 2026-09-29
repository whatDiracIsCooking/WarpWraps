/**
 * @file linear_solver.cppm
 * @brief Type-safe templated wrappers for the modern (X-prefixed) GPU solver linear API
 *
 * This module provides C++ templated wrappers for the 8 modern (X-prefixed,
 * wwrsolverDnParams_t-based, int64_t-dimensioned) linear-solver functions
 * that both cuSOLVER Dense and hipSOLVER Dense have: potrf/potrf_bufferSize,
 * potrs, getrf/getrf_bufferSize, getrs, geqrf/geqrf_bufferSize -- verified
 * signature by signature against hipsolver-dense.h. cuSOLVER's modern API
 * has a wider surface (larft, sytrs, trtri, plus the entire modern
 * eigenvalue/SVD API); those have no hipSOLVER counterpart, so they are not
 * wrapped -- call them through wwr.cuda.cusolverDn directly.
 *
 * Usage:
 *   import wwr.wrappers.solver;
 *   using namespace wwr;
 *
 *   potrf_bufferSize<float>(handle, params, uplo, n, A, lda, ...);
 */

export module wwr.wrappers.solver:linear_solver;

import wwr.solver;
import wwr.blas;
import :type_traits;
import std;

export namespace wwr {

// ========================================================================
// Cholesky Factorization - potrf
// ========================================================================

template<usual_fp T>
wwrsolverStatus_t potrf_bufferSize(wwrsolverDnHandle_t handle, wwrsolverDnParams_t params,
                                   wwrblasFillMode_t uplo, int64_t n, const T *A, int64_t lda,
                                   size_t *workspaceInBytesOnDevice,
                                   size_t *workspaceInBytesOnHost) {
  return wwrsolverDnXpotrf_bufferSize(
      handle, params, uplo, n, get_wwrsolver_type<T>(), reinterpret_cast<const void *>(A), lda,
      get_wwrsolver_type<T>(), workspaceInBytesOnDevice, workspaceInBytesOnHost);
}

template<usual_fp T>
wwrsolverStatus_t potrf(wwrsolverDnHandle_t handle, wwrsolverDnParams_t params,
                        wwrblasFillMode_t uplo, int64_t n, T *A, int64_t lda, void *bufferOnDevice,
                        size_t workspaceInBytesOnDevice, void *bufferOnHost,
                        size_t workspaceInBytesOnHost, int *info) {
  return wwrsolverDnXpotrf(handle, params, uplo, n, get_wwrsolver_type<T>(),
                           reinterpret_cast<void *>(A), lda, get_wwrsolver_type<T>(),
                           bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost,
                           workspaceInBytesOnHost, info);
}

// ========================================================================
// Cholesky Solver - potrs
// ========================================================================

template<usual_fp T>
wwrsolverStatus_t potrs(wwrsolverDnHandle_t handle, wwrsolverDnParams_t params,
                        wwrblasFillMode_t uplo, int64_t n, int64_t nrhs, const T *A, int64_t lda,
                        T *B, int64_t ldb, int *info) {
  return wwrsolverDnXpotrs(handle, params, uplo, n, nrhs, get_wwrsolver_type<T>(),
                           reinterpret_cast<const void *>(A), lda, get_wwrsolver_type<T>(),
                           reinterpret_cast<void *>(B), ldb, info);
}

// ========================================================================
// LU Factorization - getrf
// ========================================================================

template<usual_fp T>
wwrsolverStatus_t getrf_bufferSize(wwrsolverDnHandle_t handle, wwrsolverDnParams_t params,
                                   int64_t m, int64_t n, const T *A, int64_t lda,
                                   size_t *workspaceInBytesOnDevice,
                                   size_t *workspaceInBytesOnHost) {
  return wwrsolverDnXgetrf_bufferSize(
      handle, params, m, n, get_wwrsolver_type<T>(), reinterpret_cast<const void *>(A), lda,
      get_wwrsolver_type<T>(), workspaceInBytesOnDevice, workspaceInBytesOnHost);
}

template<usual_fp T>
wwrsolverStatus_t getrf(wwrsolverDnHandle_t handle, wwrsolverDnParams_t params, int64_t m,
                        int64_t n, T *A, int64_t lda, int64_t *ipiv, void *bufferOnDevice,
                        size_t workspaceInBytesOnDevice, void *bufferOnHost,
                        size_t workspaceInBytesOnHost, int *info) {
  return wwrsolverDnXgetrf(handle, params, m, n, get_wwrsolver_type<T>(),
                           reinterpret_cast<void *>(A), lda, ipiv, get_wwrsolver_type<T>(),
                           bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost,
                           workspaceInBytesOnHost, info);
}

// ========================================================================
// LU Solver - getrs
// ========================================================================

template<usual_fp T>
wwrsolverStatus_t getrs(wwrsolverDnHandle_t handle, wwrsolverDnParams_t params,
                        wwrblasOperation_t trans, int64_t n, int64_t nrhs, const T *A, int64_t lda,
                        const int64_t *ipiv, T *B, int64_t ldb, int *info) {
  return wwrsolverDnXgetrs(handle, params, trans, n, nrhs, get_wwrsolver_type<T>(),
                           reinterpret_cast<const void *>(A), lda, ipiv, get_wwrsolver_type<T>(),
                           reinterpret_cast<void *>(B), ldb, info);
}

// ========================================================================
// QR Factorization - geqrf
// ========================================================================

template<usual_fp T>
wwrsolverStatus_t geqrf_bufferSize(wwrsolverDnHandle_t handle, wwrsolverDnParams_t params,
                                   int64_t m, int64_t n, const T *A, int64_t lda, const T *tau,
                                   size_t *workspaceInBytesOnDevice,
                                   size_t *workspaceInBytesOnHost) {
  return wwrsolverDnXgeqrf_bufferSize(
      handle, params, m, n, get_wwrsolver_type<T>(), reinterpret_cast<const void *>(A), lda,
      get_wwrsolver_type<T>(), reinterpret_cast<const void *>(tau), get_wwrsolver_type<T>(),
      workspaceInBytesOnDevice, workspaceInBytesOnHost);
}

template<usual_fp T>
wwrsolverStatus_t geqrf(wwrsolverDnHandle_t handle, wwrsolverDnParams_t params, int64_t m,
                        int64_t n, T *A, int64_t lda, T *tau, void *bufferOnDevice,
                        size_t workspaceInBytesOnDevice, void *bufferOnHost,
                        size_t workspaceInBytesOnHost, int *info) {
  return wwrsolverDnXgeqrf(handle, params, m, n, get_wwrsolver_type<T>(),
                           reinterpret_cast<void *>(A), lda, get_wwrsolver_type<T>(),
                           reinterpret_cast<void *>(tau), get_wwrsolver_type<T>(), bufferOnDevice,
                           workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info);
}

} // namespace wwr
