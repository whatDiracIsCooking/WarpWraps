/**
 * @file linear_solver_legacy.cppm
 * @brief Legacy GPU solver linear-solver API wrappers (int-based, pre-params API)
 *
 * This module provides type-safe C++ wrappers for the legacy cuSOLVER Dense /
 * hipSOLVER Dense API -- 32 functions, all shared between both backends
 * (verified signature by signature against hipsolver-dense.h). For new code,
 * prefer the modern API in the :linear_solver partition, which uses
 * wwrsolverDnParams_t and int64_t dimensions -- though only 8 of its
 * functions have a hipSOLVER counterpart at all.
 *
 * Legacy API characteristics:
 * - Uses int for all dimensions (not int64_t)
 * - No wwrsolverDnParams_t parameter
 * - Single workspace buffer (not separate device/host)
 * - Pivot arrays use int* (not int64_t*)
 *
 * Usage:
 *   import wwr.wrappers.solver;
 *   using namespace wwr;
 *
 *   // Query workspace size
 *   int Lwork;
 *   potrf_bufferSize<float>(handle, uplo, n, A, lda, &Lwork);
 *
 *   // Allocate and compute
 *   float* workspace;
 *   wwrMalloc(&workspace, Lwork * sizeof(float));
 *   potrf<float>(handle, uplo, n, A, lda, workspace, Lwork, devInfo);
 */

module;

#include "dispatch_macros.h"

export module wwr.wrappers.solver:linear_solver_legacy;

import wwr.solver;
import wwr.blas;
import wwr.complex;
import wwr.wrappers.common;
import std;

export namespace wwr {

// ========================================================================
// Cholesky Factorization (potrf, potrs)
// ========================================================================

/**
 * @brief Query workspace size for Cholesky factorization
 *
 * This function computes the size of the workspace buffer required for the
 * Cholesky factorization operation.
 *
 * @tparam T Data type (float, double, wwrFloatComplex, wwrDoubleComplex)
 * @param handle cuSOLVER handle
 * @param uplo Fill mode: CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER
 * @param n Order of matrix A
 * @param A Pointer to matrix A (device memory)
 * @param lda Leading dimension of A
 * @param Lwork Output: required workspace size in elements of type T
 * @return wwrsolverStatus_t status code
 */
template<usual_fp T>
wwrsolverStatus_t potrf_bufferSize(wwrsolverDnHandle_t handle, wwrblasFillMode_t uplo, int n, T *A,
                                   int lda, int *Lwork) {
  WWR_REAL_DISPATCH(T, wwrsolverDn, S, D, potrf_bufferSize, handle, uplo, n, A, lda, Lwork);
  WWR_COMPLEX_DISPATCH(T, wwrsolverDn, C, Z, potrf_bufferSize, handle, uplo, n, A, lda, Lwork);
}

/**
 * @brief Compute Cholesky factorization A = L*L^T or U^T*U
 *
 * Computes the Cholesky factorization of a Hermitian positive-definite matrix A.
 * - If uplo = CUBLAS_FILL_MODE_LOWER: A = L*L^T (lower triangular)
 * - If uplo = CUBLAS_FILL_MODE_UPPER: A = U^T*U (upper triangular)
 *
 * @tparam T Data type (float, double, wwrFloatComplex, wwrDoubleComplex)
 * @param handle cuSOLVER handle
 * @param uplo Fill mode: CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER
 * @param n Order of matrix A
 * @param A Pointer to matrix A (device memory); overwritten with factorization
 * @param lda Leading dimension of A (lda >= max(1, n))
 * @param Workspace Device workspace buffer (size from potrf_bufferSize)
 * @param Lwork Workspace size in elements of type T
 * @param devInfo Device pointer: 0 on success, i if A(i,i) is not positive definite
 * @return wwrsolverStatus_t status code
 */
template<usual_fp T>
wwrsolverStatus_t potrf(wwrsolverDnHandle_t handle, wwrblasFillMode_t uplo, int n, T *A, int lda,
                        T *Workspace, int Lwork, int *devInfo) {
  WWR_REAL_DISPATCH(T, wwrsolverDn, S, D, potrf, handle, uplo, n, A, lda, Workspace, Lwork,
                       devInfo);
  WWR_COMPLEX_DISPATCH(T, wwrsolverDn, C, Z, potrf, handle, uplo, n, A, lda, Workspace, Lwork,
                          devInfo);
}

/**
 * @brief Solve system using Cholesky factorization
 *
 * Solves a system of linear equations A*X = B using the Cholesky factorization
 * computed by potrf(). Matrix A must have been factorized before calling this function.
 *
 * @tparam T Data type (float, double, wwrFloatComplex, wwrDoubleComplex)
 * @param handle cuSOLVER handle
 * @param uplo Fill mode used in potrf: CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER
 * @param n Order of matrix A
 * @param nrhs Number of right-hand sides (columns of B)
 * @param A Pointer to factorized matrix from potrf (device memory, not modified)
 * @param lda Leading dimension of A (lda >= max(1, n))
 * @param B Pointer to matrix B (device memory); overwritten with solution X
 * @param ldb Leading dimension of B (ldb >= max(1, n))
 * @param devInfo Device pointer: 0 on success
 * @return wwrsolverStatus_t status code
 */
template<usual_fp T>
wwrsolverStatus_t potrs(wwrsolverDnHandle_t handle, wwrblasFillMode_t uplo, int n, int nrhs,
                        const T *A, int lda, T *B, int ldb, int *devInfo) {
  WWR_REAL_DISPATCH(T, wwrsolverDn, S, D, potrs, handle, uplo, n, nrhs, A, lda, B, ldb, devInfo);
  WWR_COMPLEX_DISPATCH(T, wwrsolverDn, C, Z, potrs, handle, uplo, n, nrhs, A, lda, B, ldb,
                          devInfo);
}

/**
 * @brief Query workspace size for matrix inversion using Cholesky factorization
 *
 * This function computes the size of the workspace buffer required for the
 * matrix inversion operation using Cholesky factorization.
 *
 * @tparam T Data type (float, double, wwrFloatComplex, wwrDoubleComplex)
 * @param handle cuSOLVER handle
 * @param uplo Fill mode: CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER
 * @param n Order of matrix A
 * @param A Pointer to factorized matrix from potrf (device memory)
 * @param lda Leading dimension of A
 * @param Lwork Output: required workspace size in elements of type T
 * @return wwrsolverStatus_t status code
 */
template<usual_fp T>
wwrsolverStatus_t potri_bufferSize(wwrsolverDnHandle_t handle, wwrblasFillMode_t uplo, int n, T *A,
                                   int lda, int *Lwork) {
  WWR_REAL_DISPATCH(T, wwrsolverDn, S, D, potri_bufferSize, handle, uplo, n, A, lda, Lwork);
  WWR_COMPLEX_DISPATCH(T, wwrsolverDn, C, Z, potri_bufferSize, handle, uplo, n, A, lda, Lwork);
}

/**
 * @brief Compute matrix inverse using Cholesky factorization
 *
 * Computes the inverse of a symmetric positive-definite matrix A using the
 * Cholesky factorization computed by potrf():
 * - If uplo = CUBLAS_FILL_MODE_LOWER: A^-1 is computed from L*L^T factorization
 * - If uplo = CUBLAS_FILL_MODE_UPPER: A^-1 is computed from U^T*U factorization
 *
 * On entry, A contains the triangular factor L or U from the Cholesky factorization.
 * On exit, A is overwritten with the inverse matrix A^-1.
 *
 * @tparam T Data type (float, double, wwrFloatComplex, wwrDoubleComplex)
 * @param handle cuSOLVER handle
 * @param uplo Fill mode used in potrf: CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER
 * @param n Order of matrix A (n >= 0)
 * @param A Pointer to factorized matrix from potrf (device memory); overwritten with A^-1
 * @param lda Leading dimension of A (lda >= max(1, n))
 * @param Workspace Device workspace buffer (size from potri_bufferSize)
 * @param Lwork Workspace size in elements of type T
 * @param devInfo Device pointer: 0 on success, i if A(i,i) is zero (singular matrix)
 * @return wwrsolverStatus_t status code
 */
template<usual_fp T>
wwrsolverStatus_t potri(wwrsolverDnHandle_t handle, wwrblasFillMode_t uplo, int n, T *A, int lda,
                        T *Workspace, int Lwork, int *devInfo) {
  WWR_REAL_DISPATCH(T, wwrsolverDn, S, D, potri, handle, uplo, n, A, lda, Workspace, Lwork,
                       devInfo);
  WWR_COMPLEX_DISPATCH(T, wwrsolverDn, C, Z, potri, handle, uplo, n, A, lda, Workspace, Lwork,
                          devInfo);
}

// ========================================================================
// LU Factorization (getrf, getrs)
// ========================================================================

/**
 * @brief Query workspace size for LU factorization
 *
 * This function computes the size of the workspace buffer required for the
 * LU factorization operation.
 *
 * @tparam T Data type (float, double, wwrFloatComplex, wwrDoubleComplex)
 * @param handle cuSOLVER handle
 * @param m Number of rows of matrix A
 * @param n Number of columns of matrix A
 * @param A Pointer to matrix A (device memory)
 * @param lda Leading dimension of A
 * @param Lwork Output: required workspace size in elements of type T
 * @return wwrsolverStatus_t status code
 */
template<usual_fp T>
wwrsolverStatus_t getrf_bufferSize(wwrsolverDnHandle_t handle, int m, int n, T *A, int lda,
                                   int *Lwork) {
  WWR_REAL_DISPATCH(T, wwrsolverDn, S, D, getrf_bufferSize, handle, m, n, A, lda, Lwork);
  WWR_COMPLEX_DISPATCH(T, wwrsolverDn, C, Z, getrf_bufferSize, handle, m, n, A, lda, Lwork);
}

/**
 * @brief Compute LU factorization with partial pivoting
 *
 * Computes the LU factorization of a general m×n matrix A:
 * P*A = L*U
 * where P is a permutation matrix, L is lower triangular with unit diagonal,
 * and U is upper triangular.
 *
 * @tparam T Data type (float, double, wwrFloatComplex, wwrDoubleComplex)
 * @param handle cuSOLVER handle
 * @param m Number of rows of matrix A
 * @param n Number of columns of matrix A
 * @param A Pointer to matrix A (device memory); overwritten with L and U
 * @param lda Leading dimension of A (lda >= max(1, m))
 * @param Workspace Device workspace buffer (size from getrf_bufferSize)
 * @param devIpiv Device array of pivot indices (min(m,n) elements)
 * @param devInfo Device pointer: 0 on success, i if U(i,i)=0 (singular matrix)
 * @return wwrsolverStatus_t status code
 */
template<usual_fp T>
wwrsolverStatus_t getrf(wwrsolverDnHandle_t handle, int m, int n, T *A, int lda, T *Workspace,
                        int *devIpiv, int *devInfo) {
  WWR_REAL_DISPATCH(T, wwrsolverDn, S, D, getrf, handle, m, n, A, lda, Workspace, devIpiv,
                       devInfo);
  WWR_COMPLEX_DISPATCH(T, wwrsolverDn, C, Z, getrf, handle, m, n, A, lda, Workspace, devIpiv,
                          devInfo);
}

/**
 * @brief Solve system using LU factorization
 *
 * Solves a system of linear equations using the LU factorization computed by getrf():
 * - If trans = CUBLAS_OP_N: A*X = B
 * - If trans = CUBLAS_OP_T: A^T*X = B
 * - If trans = CUBLAS_OP_C: A^H*X = B (Hermitian transpose)
 *
 * @tparam T Data type (float, double, wwrFloatComplex, wwrDoubleComplex)
 * @param handle cuSOLVER handle
 * @param trans Operation: CUBLAS_OP_N, CUBLAS_OP_T, or CUBLAS_OP_C
 * @param n Order of matrix A (n >= 0)
 * @param nrhs Number of right-hand sides (columns of B)
 * @param A Pointer to factorized matrix from getrf (device memory, not modified)
 * @param lda Leading dimension of A (lda >= max(1, n))
 * @param devIpiv Device array of pivot indices from getrf
 * @param B Pointer to matrix B (device memory); overwritten with solution X
 * @param ldb Leading dimension of B (ldb >= max(1, n))
 * @param devInfo Device pointer: 0 on success
 * @return wwrsolverStatus_t status code
 */
template<usual_fp T>
wwrsolverStatus_t getrs(wwrsolverDnHandle_t handle, wwrblasOperation_t trans, int n, int nrhs,
                        const T *A, int lda, const int *devIpiv, T *B, int ldb, int *devInfo) {
  WWR_REAL_DISPATCH(T, wwrsolverDn, S, D, getrs, handle, trans, n, nrhs, A, lda, devIpiv, B, ldb,
                       devInfo);
  WWR_COMPLEX_DISPATCH(T, wwrsolverDn, C, Z, getrs, handle, trans, n, nrhs, A, lda, devIpiv, B,
                          ldb, devInfo);
}

// ========================================================================
// QR Factorization (geqrf)
// ========================================================================

/**
 * @brief Query workspace size for QR factorization
 *
 * This function computes the size of the workspace buffer required for the
 * QR factorization operation.
 *
 * @tparam T Data type (float, double, wwrFloatComplex, wwrDoubleComplex)
 * @param handle cuSOLVER handle
 * @param m Number of rows of matrix A
 * @param n Number of columns of matrix A
 * @param A Pointer to matrix A (device memory)
 * @param lda Leading dimension of A
 * @param Lwork Output: required workspace size in elements of type T
 * @return wwrsolverStatus_t status code
 */
template<usual_fp T>
wwrsolverStatus_t geqrf_bufferSize(wwrsolverDnHandle_t handle, int m, int n, T *A, int lda,
                                   int *Lwork) {
  WWR_REAL_DISPATCH(T, wwrsolverDn, S, D, geqrf_bufferSize, handle, m, n, A, lda, Lwork);
  WWR_COMPLEX_DISPATCH(T, wwrsolverDn, C, Z, geqrf_bufferSize, handle, m, n, A, lda, Lwork);
}

/**
 * @brief Compute QR factorization
 *
 * Computes the QR factorization of a general m×n matrix A:
 * A = Q*R
 * where Q is an orthogonal/unitary matrix and R is upper triangular.
 *
 * The matrix Q is represented as a product of elementary reflectors:
 * Q = H(1) * H(2) * ... * H(k), where k = min(m,n)
 *
 * @tparam T Data type (float, double, wwrFloatComplex, wwrDoubleComplex)
 * @param handle cuSOLVER handle
 * @param m Number of rows of matrix A (m >= 0)
 * @param n Number of columns of matrix A (n >= 0)
 * @param A Pointer to matrix A (device memory); overwritten with R and reflectors
 * @param lda Leading dimension of A (lda >= max(1, m))
 * @param TAU Device array of scalar factors of reflectors (min(m,n) elements)
 * @param Workspace Device workspace buffer (size from geqrf_bufferSize)
 * @param Lwork Workspace size in elements of type T
 * @param devInfo Device pointer: 0 on success
 * @return wwrsolverStatus_t status code
 */
template<usual_fp T>
wwrsolverStatus_t geqrf(wwrsolverDnHandle_t handle, int m, int n, T *A, int lda, T *TAU,
                        T *Workspace, int Lwork, int *devInfo) {
  WWR_REAL_DISPATCH(T, wwrsolverDn, S, D, geqrf, handle, m, n, A, lda, TAU, Workspace, Lwork,
                       devInfo);
  WWR_COMPLEX_DISPATCH(T, wwrsolverDn, C, Z, geqrf, handle, m, n, A, lda, TAU, Workspace, Lwork,
                          devInfo);
}

// ========================================================================
// QR Multiplication - Real (ormqr) and Complex (unmqr)
// ========================================================================

/**
 * @brief Query workspace size for applying orthogonal matrix Q from geqrf (real types)
 *
 * This function computes the size of the workspace buffer required for the
 * ormqr operation (real float and double types only).
 *
 * @tparam T Data type (float or double only)
 * @param handle cuSOLVER handle
 * @param side Left (CUBLAS_SIDE_LEFT) or right (CUBLAS_SIDE_RIGHT) multiplication
 * @param trans Operation: CUBLAS_OP_N (Q) or CUBLAS_OP_T (Q^T)
 * @param m Number of rows of matrix C
 * @param n Number of columns of matrix C
 * @param k Number of elementary reflectors (k <= m if side=LEFT, k <= n if side=RIGHT)
 * @param A Pointer to reflectors from geqrf (device memory)
 * @param lda Leading dimension of A
 * @param tau Device array of scalar factors from geqrf (k elements)
 * @param C Pointer to matrix C (device memory)
 * @param ldc Leading dimension of C
 * @param Lwork Output: required workspace size in elements of type T
 * @return wwrsolverStatus_t status code
 */
template<real_fp T>
wwrsolverStatus_t ormqr_bufferSize(wwrsolverDnHandle_t handle, wwrblasSideMode_t side,
                                   wwrblasOperation_t trans, int m, int n, int k, const T *A,
                                   int lda, const T *tau, const T *C, int ldc, int *Lwork) {
  WWR_REAL_DISPATCH(T, wwrsolverDn, S, D, ormqr_bufferSize, handle, side, trans, m, n, k, A, lda,
                       tau, C, ldc, Lwork);
}

/**
 * @brief Apply orthogonal matrix Q from geqrf to a matrix (real types)
 *
 * Multiplies a real matrix C by the orthogonal matrix Q from geqrf:
 * - If side = CUBLAS_SIDE_LEFT and trans = CUBLAS_OP_N: C := Q*C
 * - If side = CUBLAS_SIDE_LEFT and trans = CUBLAS_OP_T: C := Q^T*C
 * - If side = CUBLAS_SIDE_RIGHT and trans = CUBLAS_OP_N: C := C*Q
 * - If side = CUBLAS_SIDE_RIGHT and trans = CUBLAS_OP_T: C := C*Q^T
 *
 * @tparam T Data type (float or double only)
 * @param handle cuSOLVER handle
 * @param side Left or right multiplication
 * @param trans Operation: CUBLAS_OP_N (Q) or CUBLAS_OP_T (Q^T)
 * @param m Number of rows of matrix C
 * @param n Number of columns of matrix C
 * @param k Number of elementary reflectors
 * @param A Pointer to reflectors from geqrf (device memory, not modified)
 * @param lda Leading dimension of A
 * @param tau Device array of scalar factors from geqrf
 * @param C Pointer to matrix C (device memory); overwritten with result
 * @param ldc Leading dimension of C
 * @param Workspace Device workspace buffer (size from ormqr_bufferSize)
 * @param Lwork Workspace size in elements of type T
 * @param devInfo Device pointer: 0 on success
 * @return wwrsolverStatus_t status code
 */
template<real_fp T>
wwrsolverStatus_t ormqr(wwrsolverDnHandle_t handle, wwrblasSideMode_t side,
                        wwrblasOperation_t trans, int m, int n, int k, const T *A, int lda,
                        const T *tau, T *C, int ldc, T *Workspace, int Lwork, int *devInfo) {
  WWR_REAL_DISPATCH(T, wwrsolverDn, S, D, ormqr, handle, side, trans, m, n, k, A, lda, tau, C,
                       ldc, Workspace, Lwork, devInfo);
}

/**
 * @brief Query workspace size for applying unitary matrix Q from geqrf (complex types)
 *
 * This function computes the size of the workspace buffer required for the
 * unmqr operation (complex types only).
 *
 * @tparam T Data type (wwrFloatComplex or wwrDoubleComplex only)
 * @param handle cuSOLVER handle
 * @param side Left (CUBLAS_SIDE_LEFT) or right (CUBLAS_SIDE_RIGHT) multiplication
 * @param trans Operation: CUBLAS_OP_N (Q) or CUBLAS_OP_C (Q^H, Hermitian transpose)
 * @param m Number of rows of matrix C
 * @param n Number of columns of matrix C
 * @param k Number of elementary reflectors
 * @param A Pointer to reflectors from geqrf (device memory)
 * @param lda Leading dimension of A
 * @param tau Device array of scalar factors from geqrf (k elements)
 * @param C Pointer to matrix C (device memory)
 * @param ldc Leading dimension of C
 * @param Lwork Output: required workspace size in elements of type T
 * @return wwrsolverStatus_t status code
 */
template<complex_fp T>
wwrsolverStatus_t unmqr_bufferSize(wwrsolverDnHandle_t handle, wwrblasSideMode_t side,
                                   wwrblasOperation_t trans, int m, int n, int k, const T *A,
                                   int lda, const T *tau, const T *C, int ldc, int *Lwork) {
  WWR_COMPLEX_DISPATCH(T, wwrsolverDn, C, Z, unmqr_bufferSize, handle, side, trans, m, n, k, A,
                          lda, tau, C, ldc, Lwork);
}

/**
 * @brief Apply unitary matrix Q from geqrf to a matrix (complex types)
 *
 * Multiplies a complex matrix C by the unitary matrix Q from geqrf:
 * - If side = CUBLAS_SIDE_LEFT and trans = CUBLAS_OP_N: C := Q*C
 * - If side = CUBLAS_SIDE_LEFT and trans = CUBLAS_OP_C: C := Q^H*C
 * - If side = CUBLAS_SIDE_RIGHT and trans = CUBLAS_OP_N: C := C*Q
 * - If side = CUBLAS_SIDE_RIGHT and trans = CUBLAS_OP_C: C := C*Q^H
 *
 * @tparam T Data type (wwrFloatComplex or wwrDoubleComplex only)
 * @param handle cuSOLVER handle
 * @param side Left or right multiplication
 * @param trans Operation: CUBLAS_OP_N (Q) or CUBLAS_OP_C (Q^H, Hermitian transpose)
 * @param m Number of rows of matrix C
 * @param n Number of columns of matrix C
 * @param k Number of elementary reflectors
 * @param A Pointer to reflectors from geqrf (device memory, not modified)
 * @param lda Leading dimension of A
 * @param tau Device array of scalar factors from geqrf
 * @param C Pointer to matrix C (device memory); overwritten with result
 * @param ldc Leading dimension of C
 * @param Workspace Device workspace buffer (size from unmqr_bufferSize)
 * @param Lwork Workspace size in elements of type T
 * @param devInfo Device pointer: 0 on success
 * @return wwrsolverStatus_t status code
 */
template<complex_fp T>
wwrsolverStatus_t unmqr(wwrsolverDnHandle_t handle, wwrblasSideMode_t side,
                        wwrblasOperation_t trans, int m, int n, int k, const T *A, int lda,
                        const T *tau, T *C, int ldc, T *Workspace, int Lwork, int *devInfo) {
  WWR_COMPLEX_DISPATCH(T, wwrsolverDn, C, Z, unmqr, handle, side, trans, m, n, k, A, lda, tau, C,
                          ldc, Workspace, Lwork, devInfo);
}

// ========================================================================
// Least Squares Solver (gels) - Iterative Refinement
// ========================================================================

/**
 * @brief Query workspace size for least squares solver (gels)
 *
 * This function computes the size of the workspace buffer required for the
 * least squares solver operation using iterative refinement.
 *
 * Solves overdetermined or underdetermined linear systems:
 * - If m >= n: minimize ||b - A*x|| (least squares)
 * - If m < n: find minimum norm solution
 *
 * @tparam T Data type (float, double, wwrFloatComplex, wwrDoubleComplex)
 * @param handle cuSOLVER handle
 * @param m Number of rows of matrix A
 * @param n Number of columns of matrix A
 * @param nrhs Number of right-hand sides (columns of B and X)
 * @param dA Pointer to matrix A (device memory, m×n)
 * @param ldda Leading dimension of A (ldda >= max(1, m))
 * @param dB Pointer to matrix B (device memory, max(m,n)×nrhs)
 * @param lddb Leading dimension of B (lddb >= max(1, m, n))
 * @param dX Pointer to solution matrix X (device memory, n×nrhs)
 * @param lddx Leading dimension of X (lddx >= max(1, n))
 * @param dWorkspace Pointer to workspace buffer (device memory)
 * @param lwork_bytes Output: required workspace size in bytes
 * @return wwrsolverStatus_t status code
 */
template<usual_fp T>
wwrsolverStatus_t gels_bufferSize(wwrsolverDnHandle_t handle, int m, int n, int nrhs, T *dA,
                                  int ldda, T *dB, int lddb, T *dX, int lddx, void *dWorkspace,
                                  std::size_t *lwork_bytes) {
  WWR_REAL_DISPATCH(T, wwrsolverDn, SS, DD, gels_bufferSize, handle, m, n, nrhs, dA, ldda, dB,
                       lddb, dX, lddx, dWorkspace, lwork_bytes);
  WWR_COMPLEX_DISPATCH(T, wwrsolverDn, CC, ZZ, gels_bufferSize, handle, m, n, nrhs, dA, ldda, dB,
                          lddb, dX, lddx, dWorkspace, lwork_bytes);
}

/**
 * @brief Solve least squares problem using iterative refinement
 *
 * Solves overdetermined or underdetermined linear systems using QR or LQ factorization
 * with iterative refinement:
 * - If m >= n: minimize ||b - A*x|| (overdetermined, least squares)
 * - If m < n: find minimum norm solution (underdetermined)
 *
 * The function uses iterative refinement to improve the solution accuracy.
 *
 * @tparam T Data type (float, double, wwrFloatComplex, wwrDoubleComplex)
 * @param handle cuSOLVER handle
 * @param m Number of rows of matrix A
 * @param n Number of columns of matrix A
 * @param nrhs Number of right-hand sides (columns of B and X)
 * @param dA Pointer to matrix A (device memory, m×n); overwritten during computation
 * @param ldda Leading dimension of A (ldda >= max(1, m))
 * @param dB Pointer to matrix B (device memory, max(m,n)×nrhs); not modified
 * @param lddb Leading dimension of B (lddb >= max(1, m, n))
 * @param dX Pointer to solution matrix X (device memory, n×nrhs); overwritten with solution
 * @param lddx Leading dimension of X (lddx >= max(1, n))
 * @param dWorkspace Device workspace buffer (size from gels_bufferSize)
 * @param lwork_bytes Workspace size in bytes
 * @param iter Device pointer: number of iterations performed (output)
 * @param d_info Device pointer: 0 on success, >0 if singular
 * @return wwrsolverStatus_t status code
 */
template<usual_fp T>
wwrsolverStatus_t gels(wwrsolverDnHandle_t handle, int m, int n, int nrhs, T *dA, int ldda, T *dB,
                       int lddb, T *dX, int lddx, void *dWorkspace, std::size_t lwork_bytes,
                       int *iter, int *d_info) {
  WWR_REAL_DISPATCH(T, wwrsolverDn, SS, DD, gels, handle, m, n, nrhs, dA, ldda, dB, lddb, dX,
                       lddx, dWorkspace, lwork_bytes, iter, d_info);
  WWR_COMPLEX_DISPATCH(T, wwrsolverDn, CC, ZZ, gels, handle, m, n, nrhs, dA, ldda, dB, lddb, dX,
                          lddx, dWorkspace, lwork_bytes, iter, d_info);
}

// ========================================================================
// General Linear System Solver (gesv) - Iterative Refinement
// ========================================================================

/**
 * @brief Query workspace size for general linear system solver (gesv)
 *
 * This function computes the size of the workspace buffer required for the
 * general linear system solver operation using iterative refinement.
 *
 * Solves the linear system A*X = B where A is a square n×n matrix.
 *
 * @tparam T Data type (float, double, wwrFloatComplex, wwrDoubleComplex)
 * @param handle cuSOLVER handle
 * @param n Order of matrix A (n×n square matrix)
 * @param nrhs Number of right-hand sides (columns of B and X)
 * @param dA Pointer to matrix A (device memory, n×n)
 * @param ldda Leading dimension of A (ldda >= max(1, n))
 * @param dipiv Device array for pivot indices (n elements)
 * @param dB Pointer to matrix B (device memory, n×nrhs)
 * @param lddb Leading dimension of B (lddb >= max(1, n))
 * @param dX Pointer to solution matrix X (device memory, n×nrhs)
 * @param lddx Leading dimension of X (lddx >= max(1, n))
 * @param dWorkspace Pointer to workspace buffer (device memory)
 * @param lwork_bytes Output: required workspace size in bytes
 * @return wwrsolverStatus_t status code
 */
template<usual_fp T>
wwrsolverStatus_t gesv_bufferSize(wwrsolverDnHandle_t handle, int n, int nrhs, T *dA, int ldda,
                                  int *dipiv, T *dB, int lddb, T *dX, int lddx, void *dWorkspace,
                                  std::size_t *lwork_bytes) {
  WWR_REAL_DISPATCH(T, wwrsolverDn, SS, DD, gesv_bufferSize, handle, n, nrhs, dA, ldda, dipiv,
                       dB, lddb, dX, lddx, dWorkspace, lwork_bytes);
  WWR_COMPLEX_DISPATCH(T, wwrsolverDn, CC, ZZ, gesv_bufferSize, handle, n, nrhs, dA, ldda, dipiv,
                          dB, lddb, dX, lddx, dWorkspace, lwork_bytes);
}

/**
 * @brief Solve general linear system using iterative refinement
 *
 * Solves a general linear system A*X = B using LU factorization with
 * iterative refinement for improved accuracy. The matrix A is a square n×n matrix.
 *
 * The function uses iterative refinement to improve the solution accuracy.
 *
 * @tparam T Data type (float, double, wwrFloatComplex, wwrDoubleComplex)
 * @param handle cuSOLVER handle
 * @param n Order of matrix A (n×n square matrix)
 * @param nrhs Number of right-hand sides (columns of B and X)
 * @param dA Pointer to matrix A (device memory, n×n); overwritten during computation
 * @param ldda Leading dimension of A (ldda >= max(1, n))
 * @param dipiv Device array for pivot indices (n elements); output
 * @param dB Pointer to matrix B (device memory, n×nrhs); not modified
 * @param lddb Leading dimension of B (lddb >= max(1, n))
 * @param dX Pointer to solution matrix X (device memory, n×nrhs); overwritten with solution
 * @param lddx Leading dimension of X (lddx >= max(1, n))
 * @param dWorkspace Device workspace buffer (size from gesv_bufferSize)
 * @param lwork_bytes Workspace size in bytes
 * @param iter Device pointer: number of iterations performed (output)
 * @param d_info Device pointer: 0 on success, >0 if singular
 * @return wwrsolverStatus_t status code
 */
template<usual_fp T>
wwrsolverStatus_t gesv(wwrsolverDnHandle_t handle, int n, int nrhs, T *dA, int ldda, int *dipiv,
                       T *dB, int lddb, T *dX, int lddx, void *dWorkspace, std::size_t lwork_bytes,
                       int *iter, int *d_info) {
  WWR_REAL_DISPATCH(T, wwrsolverDn, SS, DD, gesv, handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX,
                       lddx, dWorkspace, lwork_bytes, iter, d_info);
  WWR_COMPLEX_DISPATCH(T, wwrsolverDn, CC, ZZ, gesv, handle, n, nrhs, dA, ldda, dipiv, dB, lddb,
                          dX, lddx, dWorkspace, lwork_bytes, iter, d_info);
}

// ========================================================================
// Batched Cholesky Factorization
// ========================================================================

/**
 * @brief Compute batched Cholesky factorization A = L*L^T or U^T*U
 *
 * @tparam T Data type (float, double, wwrFloatComplex, wwrDoubleComplex)
 * @param handle cuSOLVER handle
 * @param uplo Fill mode
 * @param n Order of each matrix
 * @param Aarray Array of pointers to matrices (device memory)
 * @param lda Leading dimension
 * @param infoArray Array of status codes (device memory)
 * @param batchSize Number of matrices
 * @return wwrsolverStatus_t status code
 */
template<usual_fp T>
wwrsolverStatus_t potrfBatched(wwrsolverDnHandle_t handle, wwrblasFillMode_t uplo, int n,
                               T *Aarray[], int lda, int *infoArray, int batchSize) {
  WWR_REAL_DISPATCH(T, wwrsolverDn, S, D, potrfBatched, handle, uplo, n, Aarray, lda, infoArray,
                       batchSize);
  WWR_COMPLEX_DISPATCH(T, wwrsolverDn, C, Z, potrfBatched, handle, uplo, n, Aarray, lda,
                          infoArray, batchSize);
}

/**
 * @brief Solve batched systems using Cholesky factorization
 *
 * @tparam T Data type (float, double, wwrFloatComplex, wwrDoubleComplex)
 * @param handle cuSOLVER handle
 * @param uplo Fill mode
 * @param n Order of each matrix
 * @param nrhs Number of right-hand sides
 * @param A Array of pointers to factorized matrices (device memory)
 * @param lda Leading dimension of A
 * @param B Array of pointers to RHS matrices (device memory)
 * @param ldb Leading dimension of B
 * @param info Array of status codes (device memory)
 * @param batchSize Number of matrices
 * @return wwrsolverStatus_t status code
 */
template<usual_fp T>
wwrsolverStatus_t potrsBatched(wwrsolverDnHandle_t handle, wwrblasFillMode_t uplo, int n, int nrhs,
                               T *A[], int lda, T *B[], int ldb, int *info, int batchSize) {
  WWR_REAL_DISPATCH(T, wwrsolverDn, S, D, potrsBatched, handle, uplo, n, nrhs, A, lda, B, ldb,
                       info, batchSize);
  WWR_COMPLEX_DISPATCH(T, wwrsolverDn, C, Z, potrsBatched, handle, uplo, n, nrhs, A, lda, B, ldb,
                          info, batchSize);
}

// ========================================================================
// Symmetric/Hermitian Factorization (Bunch-Kaufman LDLT)
// ========================================================================

/**
 * @brief Query workspace size for symmetric factorization (Bunch-Kaufman)
 *
 * @tparam T Data type (float, double, wwrFloatComplex, wwrDoubleComplex)
 * @param handle cuSOLVER handle
 * @param uplo Fill mode
 * @param n Order of matrix
 * @param A Pointer to matrix (device memory)
 * @param lda Leading dimension
 * @param ipiv Pivot indices (device memory)
 * @param lwork Output: workspace size
 * @return wwrsolverStatus_t status code
 */
template<usual_fp T>
wwrsolverStatus_t sytrf_bufferSize(wwrsolverDnHandle_t handle, int n, T *A, int lda, int *lwork) {
  WWR_REAL_DISPATCH(T, wwrsolverDn, S, D, sytrf_bufferSize, handle, n, A, lda, lwork);
  WWR_COMPLEX_DISPATCH(T, wwrsolverDn, C, Z, sytrf_bufferSize, handle, n, A, lda, lwork);
}

/**
 * @brief Compute symmetric factorization using Bunch-Kaufman (A = U*D*U^T or L*D*L^T)
 *
 * @tparam T Data type (float, double, wwrFloatComplex, wwrDoubleComplex)
 * @param handle cuSOLVER handle
 * @param uplo Fill mode
 * @param n Order of matrix
 * @param A Pointer to matrix (device memory); overwritten with factorization
 * @param lda Leading dimension
 * @param ipiv Pivot indices (device memory, n elements)
 * @param work Workspace buffer
 * @param lwork Workspace size
 * @param devInfo Device pointer: 0 on success
 * @return wwrsolverStatus_t status code
 */
template<usual_fp T>
wwrsolverStatus_t sytrf(wwrsolverDnHandle_t handle, wwrblasFillMode_t uplo, int n, T *A, int lda,
                        int *ipiv, T *work, int lwork, int *devInfo) {
  WWR_REAL_DISPATCH(T, wwrsolverDn, S, D, sytrf, handle, uplo, n, A, lda, ipiv, work, lwork,
                       devInfo);
  WWR_COMPLEX_DISPATCH(T, wwrsolverDn, C, Z, sytrf, handle, uplo, n, A, lda, ipiv, work, lwork,
                          devInfo);
}

// ========================================================================
// Bidiagonal Reduction
// ========================================================================

/**
 * @brief Query workspace size for bidiagonal reduction
 *
 * @tparam T Data type (float, double, wwrFloatComplex, wwrDoubleComplex)
 * @param handle cuSOLVER handle
 * @param m Number of rows
 * @param n Number of columns
 * @param lwork Output: workspace size
 * @return wwrsolverStatus_t status code
 */
template<usual_fp T>
wwrsolverStatus_t gebrd_bufferSize(wwrsolverDnHandle_t handle, int m, int n, int *lwork) {
  WWR_REAL_DISPATCH(T, wwrsolverDn, S, D, gebrd_bufferSize, handle, m, n, lwork);
  WWR_COMPLEX_DISPATCH(T, wwrsolverDn, C, Z, gebrd_bufferSize, handle, m, n, lwork);
}

/**
 * @brief Reduce general matrix to bidiagonal form
 *
 * @tparam T Data type (float, double, wwrFloatComplex, wwrDoubleComplex)
 * @param handle cuSOLVER handle
 * @param m Number of rows
 * @param n Number of columns
 * @param A Pointer to matrix (device memory); overwritten with bidiagonal form
 * @param lda Leading dimension
 * @param D Diagonal elements (device memory, min(m,n) elements)
 * @param E Off-diagonal elements (device memory, min(m,n)-1 elements)
 * @param tauq Elementary reflectors for Q (device memory)
 * @param taup Elementary reflectors for P (device memory)
 * @param work Workspace buffer
 * @param lwork Workspace size
 * @param devInfo Device pointer: 0 on success
 * @return wwrsolverStatus_t status code
 */
template<usual_fp T>
wwrsolverStatus_t gebrd(wwrsolverDnHandle_t handle, int m, int n, T *A, int lda,
                        ComplexToRealType<T> *D, ComplexToRealType<T> *E, T *tauq, T *taup, T *work,
                        int lwork, int *devInfo) {
  WWR_REAL_DISPATCH(T, wwrsolverDn, S, D, gebrd, handle, m, n, A, lda, D, E, tauq, taup, work,
                       lwork, devInfo);
  WWR_COMPLEX_DISPATCH(T, wwrsolverDn, C, Z, gebrd, handle, m, n, A, lda, D, E, tauq, taup, work,
                          lwork, devInfo);
}

// ========================================================================
// Generate Orthogonal/Unitary Matrix from QR (orgqr/ungqr)
// ========================================================================

/**
 * @brief Query workspace size for generating orthogonal/unitary matrix from QR
 *
 * @tparam T Data type (float, double for orgqr; wwrFloatComplex, wwrDoubleComplex for ungqr)
 * @param handle cuSOLVER handle
 * @param m Number of rows
 * @param n Number of columns
 * @param k Number of elementary reflectors
 * @param A Pointer to matrix (device memory)
 * @param lda Leading dimension
 * @param tau Elementary reflectors (device memory)
 * @param lwork Output: workspace size
 * @return wwrsolverStatus_t status code
 */
template<real_fp T>
wwrsolverStatus_t orgqr_bufferSize(wwrsolverDnHandle_t handle, int m, int n, int k, const T *A,
                                   int lda, const T *tau, int *lwork) {
  WWR_REAL_DISPATCH(T, wwrsolverDn, S, D, orgqr_bufferSize, handle, m, n, k, A, lda, tau, lwork);
}

template<complex_fp T>
wwrsolverStatus_t ungqr_bufferSize(wwrsolverDnHandle_t handle, int m, int n, int k, const T *A,
                                   int lda, const T *tau, int *lwork) {
  WWR_COMPLEX_DISPATCH(T, wwrsolverDn, C, Z, ungqr_bufferSize, handle, m, n, k, A, lda, tau,
                          lwork);
}

/**
 * @brief Generate orthogonal/unitary matrix from QR factorization
 *
 * @tparam T Data type
 * @param handle cuSOLVER handle
 * @param m Number of rows
 * @param n Number of columns
 * @param k Number of elementary reflectors
 * @param A Pointer to matrix (device memory); overwritten with Q
 * @param lda Leading dimension
 * @param tau Elementary reflectors (device memory)
 * @param work Workspace buffer
 * @param lwork Workspace size
 * @param devInfo Device pointer: 0 on success
 * @return wwrsolverStatus_t status code
 */
template<real_fp T>
wwrsolverStatus_t orgqr(wwrsolverDnHandle_t handle, int m, int n, int k, T *A, int lda,
                        const T *tau, T *work, int lwork, int *devInfo) {
  WWR_REAL_DISPATCH(T, wwrsolverDn, S, D, orgqr, handle, m, n, k, A, lda, tau, work, lwork,
                       devInfo);
}

template<complex_fp T>
wwrsolverStatus_t ungqr(wwrsolverDnHandle_t handle, int m, int n, int k, T *A, int lda,
                        const T *tau, T *work, int lwork, int *devInfo) {
  WWR_COMPLEX_DISPATCH(T, wwrsolverDn, C, Z, ungqr, handle, m, n, k, A, lda, tau, work, lwork,
                          devInfo);
}

// ========================================================================
// Generate Orthogonal/Unitary Matrix from Bidiagonal (orgbr/ungbr)
// ========================================================================

/**
 * @brief Query workspace size for generating orthogonal/unitary matrix from bidiagonal
 *
 * @tparam T Data type
 * @param handle cuSOLVER handle
 * @param vect 'Q' or 'P' to specify which matrix to generate
 * @param m Number of rows
 * @param n Number of columns
 * @param k Number of elementary reflectors
 * @param A Pointer to matrix (device memory)
 * @param lda Leading dimension
 * @param tau Elementary reflectors (device memory)
 * @param lwork Output: workspace size
 * @return wwrsolverStatus_t status code
 */
template<real_fp T>
wwrsolverStatus_t orgbr_bufferSize(wwrsolverDnHandle_t handle, wwrblasSideMode_t vect, int m, int n,
                                   int k, const T *A, int lda, const T *tau, int *lwork) {
  WWR_REAL_DISPATCH(T, wwrsolverDn, S, D, orgbr_bufferSize, handle, vect, m, n, k, A, lda, tau,
                       lwork);
}

template<complex_fp T>
wwrsolverStatus_t ungbr_bufferSize(wwrsolverDnHandle_t handle, wwrblasSideMode_t vect, int m, int n,
                                   int k, const T *A, int lda, const T *tau, int *lwork) {
  WWR_COMPLEX_DISPATCH(T, wwrsolverDn, C, Z, ungbr_bufferSize, handle, vect, m, n, k, A, lda,
                          tau, lwork);
}

/**
 * @brief Generate orthogonal/unitary matrix from bidiagonal reduction
 *
 * @tparam T Data type
 * @param handle cuSOLVER handle
 * @param vect 'Q' or 'P' to specify which matrix to generate
 * @param m Number of rows
 * @param n Number of columns
 * @param k Number of elementary reflectors
 * @param A Pointer to matrix (device memory); overwritten with Q or P^T
 * @param lda Leading dimension
 * @param tau Elementary reflectors (device memory)
 * @param work Workspace buffer
 * @param lwork Workspace size
 * @param devInfo Device pointer: 0 on success
 * @return wwrsolverStatus_t status code
 */
template<real_fp T>
wwrsolverStatus_t orgbr(wwrsolverDnHandle_t handle, wwrblasSideMode_t vect, int m, int n, int k,
                        T *A, int lda, const T *tau, T *work, int lwork, int *devInfo) {
  WWR_REAL_DISPATCH(T, wwrsolverDn, S, D, orgbr, handle, vect, m, n, k, A, lda, tau, work, lwork,
                       devInfo);
}

template<complex_fp T>
wwrsolverStatus_t ungbr(wwrsolverDnHandle_t handle, wwrblasSideMode_t vect, int m, int n, int k,
                        T *A, int lda, const T *tau, T *work, int lwork, int *devInfo) {
  WWR_COMPLEX_DISPATCH(T, wwrsolverDn, C, Z, ungbr, handle, vect, m, n, k, A, lda, tau, work,
                          lwork, devInfo);
}

// ==================== Explicit Template Instantiation Declarations ====================
// Generated code will be inserted here by cmake/instantiation/generate_instantiations.py
// To regenerate: cmake --build build --target wwrsolverDn_generate_instantiations

// Function: potrf_bufferSize
extern template wwrsolverStatus_t potrf_bufferSize<float>(wwrsolverDnHandle_t, wwrblasFillMode_t,
                                                          int, float *, int, int *);
extern template wwrsolverStatus_t potrf_bufferSize<double>(wwrsolverDnHandle_t, wwrblasFillMode_t,
                                                           int, double *, int, int *);
extern template wwrsolverStatus_t potrf_bufferSize<wwrFloatComplex>(wwrsolverDnHandle_t,
                                                                    wwrblasFillMode_t, int,
                                                                    wwrFloatComplex *, int, int *);
extern template wwrsolverStatus_t potrf_bufferSize<wwrDoubleComplex>(wwrsolverDnHandle_t,
                                                                     wwrblasFillMode_t, int,
                                                                     wwrDoubleComplex *, int,
                                                                     int *);

// Function: potrf
extern template wwrsolverStatus_t potrf<float>(wwrsolverDnHandle_t, wwrblasFillMode_t, int, float *,
                                               int, float *, int, int *);
extern template wwrsolverStatus_t potrf<double>(wwrsolverDnHandle_t, wwrblasFillMode_t, int,
                                                double *, int, double *, int, int *);
extern template wwrsolverStatus_t potrf<wwrFloatComplex>(wwrsolverDnHandle_t, wwrblasFillMode_t,
                                                         int, wwrFloatComplex *, int,
                                                         wwrFloatComplex *, int, int *);
extern template wwrsolverStatus_t potrf<wwrDoubleComplex>(wwrsolverDnHandle_t, wwrblasFillMode_t,
                                                          int, wwrDoubleComplex *, int,
                                                          wwrDoubleComplex *, int, int *);

// Function: potrs
extern template wwrsolverStatus_t potrs<float>(wwrsolverDnHandle_t, wwrblasFillMode_t, int, int,
                                               const float *, int, float *, int, int *);
extern template wwrsolverStatus_t potrs<double>(wwrsolverDnHandle_t, wwrblasFillMode_t, int, int,
                                                const double *, int, double *, int, int *);
extern template wwrsolverStatus_t potrs<wwrFloatComplex>(wwrsolverDnHandle_t, wwrblasFillMode_t,
                                                         int, int, const wwrFloatComplex *, int,
                                                         wwrFloatComplex *, int, int *);
extern template wwrsolverStatus_t potrs<wwrDoubleComplex>(wwrsolverDnHandle_t, wwrblasFillMode_t,
                                                          int, int, const wwrDoubleComplex *, int,
                                                          wwrDoubleComplex *, int, int *);

// Function: potri_bufferSize
extern template wwrsolverStatus_t potri_bufferSize<float>(wwrsolverDnHandle_t, wwrblasFillMode_t,
                                                          int, float *, int, int *);
extern template wwrsolverStatus_t potri_bufferSize<double>(wwrsolverDnHandle_t, wwrblasFillMode_t,
                                                           int, double *, int, int *);
extern template wwrsolverStatus_t potri_bufferSize<wwrFloatComplex>(wwrsolverDnHandle_t,
                                                                    wwrblasFillMode_t, int,
                                                                    wwrFloatComplex *, int, int *);
extern template wwrsolverStatus_t potri_bufferSize<wwrDoubleComplex>(wwrsolverDnHandle_t,
                                                                     wwrblasFillMode_t, int,
                                                                     wwrDoubleComplex *, int,
                                                                     int *);

// Function: potri
extern template wwrsolverStatus_t potri<float>(wwrsolverDnHandle_t, wwrblasFillMode_t, int, float *,
                                               int, float *, int, int *);
extern template wwrsolverStatus_t potri<double>(wwrsolverDnHandle_t, wwrblasFillMode_t, int,
                                                double *, int, double *, int, int *);
extern template wwrsolverStatus_t potri<wwrFloatComplex>(wwrsolverDnHandle_t, wwrblasFillMode_t,
                                                         int, wwrFloatComplex *, int,
                                                         wwrFloatComplex *, int, int *);
extern template wwrsolverStatus_t potri<wwrDoubleComplex>(wwrsolverDnHandle_t, wwrblasFillMode_t,
                                                          int, wwrDoubleComplex *, int,
                                                          wwrDoubleComplex *, int, int *);

// Function: getrf_bufferSize
extern template wwrsolverStatus_t getrf_bufferSize<float>(wwrsolverDnHandle_t, int, int, float *,
                                                          int, int *);
extern template wwrsolverStatus_t getrf_bufferSize<double>(wwrsolverDnHandle_t, int, int, double *,
                                                           int, int *);
extern template wwrsolverStatus_t getrf_bufferSize<wwrFloatComplex>(wwrsolverDnHandle_t, int, int,
                                                                    wwrFloatComplex *, int, int *);
extern template wwrsolverStatus_t
getrf_bufferSize<wwrDoubleComplex>(wwrsolverDnHandle_t, int, int, wwrDoubleComplex *, int, int *);

// Function: getrf
extern template wwrsolverStatus_t getrf<float>(wwrsolverDnHandle_t, int, int, float *, int, float *,
                                               int *, int *);
extern template wwrsolverStatus_t getrf<double>(wwrsolverDnHandle_t, int, int, double *, int,
                                                double *, int *, int *);
extern template wwrsolverStatus_t getrf<wwrFloatComplex>(wwrsolverDnHandle_t, int, int,
                                                         wwrFloatComplex *, int, wwrFloatComplex *,
                                                         int *, int *);
extern template wwrsolverStatus_t getrf<wwrDoubleComplex>(wwrsolverDnHandle_t, int, int,
                                                          wwrDoubleComplex *, int,
                                                          wwrDoubleComplex *, int *, int *);

// Function: getrs
extern template wwrsolverStatus_t getrs<float>(wwrsolverDnHandle_t, wwrblasOperation_t, int, int,
                                               const float *, int, const int *, float *, int,
                                               int *);
extern template wwrsolverStatus_t getrs<double>(wwrsolverDnHandle_t, wwrblasOperation_t, int, int,
                                                const double *, int, const int *, double *, int,
                                                int *);
extern template wwrsolverStatus_t getrs<wwrFloatComplex>(wwrsolverDnHandle_t, wwrblasOperation_t,
                                                         int, int, const wwrFloatComplex *, int,
                                                         const int *, wwrFloatComplex *, int,
                                                         int *);
extern template wwrsolverStatus_t getrs<wwrDoubleComplex>(wwrsolverDnHandle_t, wwrblasOperation_t,
                                                          int, int, const wwrDoubleComplex *, int,
                                                          const int *, wwrDoubleComplex *, int,
                                                          int *);

// Function: geqrf_bufferSize
extern template wwrsolverStatus_t geqrf_bufferSize<float>(wwrsolverDnHandle_t, int, int, float *,
                                                          int, int *);
extern template wwrsolverStatus_t geqrf_bufferSize<double>(wwrsolverDnHandle_t, int, int, double *,
                                                           int, int *);
extern template wwrsolverStatus_t geqrf_bufferSize<wwrFloatComplex>(wwrsolverDnHandle_t, int, int,
                                                                    wwrFloatComplex *, int, int *);
extern template wwrsolverStatus_t
geqrf_bufferSize<wwrDoubleComplex>(wwrsolverDnHandle_t, int, int, wwrDoubleComplex *, int, int *);

// Function: geqrf
extern template wwrsolverStatus_t geqrf<float>(wwrsolverDnHandle_t, int, int, float *, int, float *,
                                               float *, int, int *);
extern template wwrsolverStatus_t geqrf<double>(wwrsolverDnHandle_t, int, int, double *, int,
                                                double *, double *, int, int *);
extern template wwrsolverStatus_t geqrf<wwrFloatComplex>(wwrsolverDnHandle_t, int, int,
                                                         wwrFloatComplex *, int, wwrFloatComplex *,
                                                         wwrFloatComplex *, int, int *);
extern template wwrsolverStatus_t geqrf<wwrDoubleComplex>(wwrsolverDnHandle_t, int, int,
                                                          wwrDoubleComplex *, int,
                                                          wwrDoubleComplex *, wwrDoubleComplex *,
                                                          int, int *);

// Function: ormqr_bufferSize
extern template wwrsolverStatus_t ormqr_bufferSize<float>(wwrsolverDnHandle_t, wwrblasSideMode_t,
                                                          wwrblasOperation_t, int, int, int,
                                                          const float *, int, const float *,
                                                          const float *, int, int *);
extern template wwrsolverStatus_t ormqr_bufferSize<double>(wwrsolverDnHandle_t, wwrblasSideMode_t,
                                                           wwrblasOperation_t, int, int, int,
                                                           const double *, int, const double *,
                                                           const double *, int, int *);

// Function: ormqr
extern template wwrsolverStatus_t ormqr<float>(wwrsolverDnHandle_t, wwrblasSideMode_t,
                                               wwrblasOperation_t, int, int, int, const float *,
                                               int, const float *, float *, int, float *, int,
                                               int *);
extern template wwrsolverStatus_t ormqr<double>(wwrsolverDnHandle_t, wwrblasSideMode_t,
                                                wwrblasOperation_t, int, int, int, const double *,
                                                int, const double *, double *, int, double *, int,
                                                int *);

// Function: unmqr_bufferSize
extern template wwrsolverStatus_t
unmqr_bufferSize<wwrFloatComplex>(wwrsolverDnHandle_t, wwrblasSideMode_t, wwrblasOperation_t, int,
                                  int, int, const wwrFloatComplex *, int, const wwrFloatComplex *,
                                  const wwrFloatComplex *, int, int *);
extern template wwrsolverStatus_t
unmqr_bufferSize<wwrDoubleComplex>(wwrsolverDnHandle_t, wwrblasSideMode_t, wwrblasOperation_t, int,
                                   int, int, const wwrDoubleComplex *, int,
                                   const wwrDoubleComplex *, const wwrDoubleComplex *, int, int *);

// Function: unmqr
extern template wwrsolverStatus_t unmqr<wwrFloatComplex>(wwrsolverDnHandle_t, wwrblasSideMode_t,
                                                         wwrblasOperation_t, int, int, int,
                                                         const wwrFloatComplex *, int,
                                                         const wwrFloatComplex *, wwrFloatComplex *,
                                                         int, wwrFloatComplex *, int, int *);
extern template wwrsolverStatus_t
unmqr<wwrDoubleComplex>(wwrsolverDnHandle_t, wwrblasSideMode_t, wwrblasOperation_t, int, int, int,
                        const wwrDoubleComplex *, int, const wwrDoubleComplex *, wwrDoubleComplex *,
                        int, wwrDoubleComplex *, int, int *);

// Function: gels_bufferSize
extern template wwrsolverStatus_t gels_bufferSize<float>(wwrsolverDnHandle_t, int, int, int,
                                                         float *, int, float *, int, float *, int,
                                                         void *, std::size_t *);
extern template wwrsolverStatus_t gels_bufferSize<double>(wwrsolverDnHandle_t, int, int, int,
                                                          double *, int, double *, int, double *,
                                                          int, void *, std::size_t *);
extern template wwrsolverStatus_t gels_bufferSize<wwrFloatComplex>(wwrsolverDnHandle_t, int, int,
                                                                   int, wwrFloatComplex *, int,
                                                                   wwrFloatComplex *, int,
                                                                   wwrFloatComplex *, int, void *,
                                                                   std::size_t *);
extern template wwrsolverStatus_t gels_bufferSize<wwrDoubleComplex>(wwrsolverDnHandle_t, int, int,
                                                                    int, wwrDoubleComplex *, int,
                                                                    wwrDoubleComplex *, int,
                                                                    wwrDoubleComplex *, int, void *,
                                                                    std::size_t *);

// Function: gels
extern template wwrsolverStatus_t gels<float>(wwrsolverDnHandle_t, int, int, int, float *, int,
                                              float *, int, float *, int, void *, std::size_t,
                                              int *, int *);
extern template wwrsolverStatus_t gels<double>(wwrsolverDnHandle_t, int, int, int, double *, int,
                                               double *, int, double *, int, void *, std::size_t,
                                               int *, int *);
extern template wwrsolverStatus_t gels<wwrFloatComplex>(wwrsolverDnHandle_t, int, int, int,
                                                        wwrFloatComplex *, int, wwrFloatComplex *,
                                                        int, wwrFloatComplex *, int, void *,
                                                        std::size_t, int *, int *);
extern template wwrsolverStatus_t gels<wwrDoubleComplex>(wwrsolverDnHandle_t, int, int, int,
                                                         wwrDoubleComplex *, int,
                                                         wwrDoubleComplex *, int,
                                                         wwrDoubleComplex *, int, void *,
                                                         std::size_t, int *, int *);

// Function: gesv_bufferSize
extern template wwrsolverStatus_t gesv_bufferSize<float>(wwrsolverDnHandle_t, int, int, float *,
                                                         int, int *, float *, int, float *, int,
                                                         void *, std::size_t *);
extern template wwrsolverStatus_t gesv_bufferSize<double>(wwrsolverDnHandle_t, int, int, double *,
                                                          int, int *, double *, int, double *, int,
                                                          void *, std::size_t *);
extern template wwrsolverStatus_t gesv_bufferSize<wwrFloatComplex>(wwrsolverDnHandle_t, int, int,
                                                                   wwrFloatComplex *, int, int *,
                                                                   wwrFloatComplex *, int,
                                                                   wwrFloatComplex *, int, void *,
                                                                   std::size_t *);
extern template wwrsolverStatus_t gesv_bufferSize<wwrDoubleComplex>(wwrsolverDnHandle_t, int, int,
                                                                    wwrDoubleComplex *, int, int *,
                                                                    wwrDoubleComplex *, int,
                                                                    wwrDoubleComplex *, int, void *,
                                                                    std::size_t *);

// Function: gesv
extern template wwrsolverStatus_t gesv<float>(wwrsolverDnHandle_t, int, int, float *, int, int *,
                                              float *, int, float *, int, void *, std::size_t,
                                              int *, int *);
extern template wwrsolverStatus_t gesv<double>(wwrsolverDnHandle_t, int, int, double *, int, int *,
                                               double *, int, double *, int, void *, std::size_t,
                                               int *, int *);
extern template wwrsolverStatus_t gesv<wwrFloatComplex>(wwrsolverDnHandle_t, int, int,
                                                        wwrFloatComplex *, int, int *,
                                                        wwrFloatComplex *, int, wwrFloatComplex *,
                                                        int, void *, std::size_t, int *, int *);
extern template wwrsolverStatus_t gesv<wwrDoubleComplex>(wwrsolverDnHandle_t, int, int,
                                                         wwrDoubleComplex *, int, int *,
                                                         wwrDoubleComplex *, int,
                                                         wwrDoubleComplex *, int, void *,
                                                         std::size_t, int *, int *);

// Function: potrfBatched
extern template wwrsolverStatus_t potrfBatched<float>(wwrsolverDnHandle_t, wwrblasFillMode_t, int,
                                                      float *[], int, int *, int);
extern template wwrsolverStatus_t potrfBatched<double>(wwrsolverDnHandle_t, wwrblasFillMode_t, int,
                                                       double *[], int, int *, int);
extern template wwrsolverStatus_t potrfBatched<wwrFloatComplex>(wwrsolverDnHandle_t,
                                                                wwrblasFillMode_t, int,
                                                                wwrFloatComplex *[], int, int *,
                                                                int);
extern template wwrsolverStatus_t potrfBatched<wwrDoubleComplex>(wwrsolverDnHandle_t,
                                                                 wwrblasFillMode_t, int,
                                                                 wwrDoubleComplex *[], int, int *,
                                                                 int);

// Function: potrsBatched
extern template wwrsolverStatus_t potrsBatched<float>(wwrsolverDnHandle_t, wwrblasFillMode_t, int,
                                                      int, float *[], int, float *[], int, int *,
                                                      int);
extern template wwrsolverStatus_t potrsBatched<double>(wwrsolverDnHandle_t, wwrblasFillMode_t, int,
                                                       int, double *[], int, double *[], int, int *,
                                                       int);
extern template wwrsolverStatus_t
potrsBatched<wwrFloatComplex>(wwrsolverDnHandle_t, wwrblasFillMode_t, int, int, wwrFloatComplex *[],
                              int, wwrFloatComplex *[], int, int *, int);
extern template wwrsolverStatus_t
potrsBatched<wwrDoubleComplex>(wwrsolverDnHandle_t, wwrblasFillMode_t, int, int,
                               wwrDoubleComplex *[], int, wwrDoubleComplex *[], int, int *, int);

// Function: sytrf_bufferSize
extern template wwrsolverStatus_t sytrf_bufferSize<float>(wwrsolverDnHandle_t, int, float *, int,
                                                          int *);
extern template wwrsolverStatus_t sytrf_bufferSize<double>(wwrsolverDnHandle_t, int, double *, int,
                                                           int *);
extern template wwrsolverStatus_t sytrf_bufferSize<wwrFloatComplex>(wwrsolverDnHandle_t, int,
                                                                    wwrFloatComplex *, int, int *);
extern template wwrsolverStatus_t
sytrf_bufferSize<wwrDoubleComplex>(wwrsolverDnHandle_t, int, wwrDoubleComplex *, int, int *);

// Function: sytrf
extern template wwrsolverStatus_t sytrf<float>(wwrsolverDnHandle_t, wwrblasFillMode_t, int, float *,
                                               int, int *, float *, int, int *);
extern template wwrsolverStatus_t sytrf<double>(wwrsolverDnHandle_t, wwrblasFillMode_t, int,
                                                double *, int, int *, double *, int, int *);
extern template wwrsolverStatus_t sytrf<wwrFloatComplex>(wwrsolverDnHandle_t, wwrblasFillMode_t,
                                                         int, wwrFloatComplex *, int, int *,
                                                         wwrFloatComplex *, int, int *);
extern template wwrsolverStatus_t sytrf<wwrDoubleComplex>(wwrsolverDnHandle_t, wwrblasFillMode_t,
                                                          int, wwrDoubleComplex *, int, int *,
                                                          wwrDoubleComplex *, int, int *);

// Function: gebrd_bufferSize
extern template wwrsolverStatus_t gebrd_bufferSize<float>(wwrsolverDnHandle_t, int, int, int *);
extern template wwrsolverStatus_t gebrd_bufferSize<double>(wwrsolverDnHandle_t, int, int, int *);
extern template wwrsolverStatus_t gebrd_bufferSize<wwrFloatComplex>(wwrsolverDnHandle_t, int, int,
                                                                    int *);
extern template wwrsolverStatus_t gebrd_bufferSize<wwrDoubleComplex>(wwrsolverDnHandle_t, int, int,
                                                                     int *);

// Function: gebrd
extern template wwrsolverStatus_t gebrd<float>(wwrsolverDnHandle_t, int, int, float *, int,
                                               ComplexToRealType<float> *,
                                               ComplexToRealType<float> *, float *, float *,
                                               float *, int, int *);
extern template wwrsolverStatus_t gebrd<double>(wwrsolverDnHandle_t, int, int, double *, int,
                                                ComplexToRealType<double> *,
                                                ComplexToRealType<double> *, double *, double *,
                                                double *, int, int *);
extern template wwrsolverStatus_t
gebrd<wwrFloatComplex>(wwrsolverDnHandle_t, int, int, wwrFloatComplex *, int,
                       ComplexToRealType<wwrFloatComplex> *, ComplexToRealType<wwrFloatComplex> *,
                       wwrFloatComplex *, wwrFloatComplex *, wwrFloatComplex *, int, int *);
extern template wwrsolverStatus_t gebrd<wwrDoubleComplex>(wwrsolverDnHandle_t, int, int,
                                                          wwrDoubleComplex *, int,
                                                          ComplexToRealType<wwrDoubleComplex> *,
                                                          ComplexToRealType<wwrDoubleComplex> *,
                                                          wwrDoubleComplex *, wwrDoubleComplex *,
                                                          wwrDoubleComplex *, int, int *);

// Function: orgqr_bufferSize
extern template wwrsolverStatus_t orgqr_bufferSize<float>(wwrsolverDnHandle_t, int, int, int,
                                                          const float *, int, const float *, int *);
extern template wwrsolverStatus_t orgqr_bufferSize<double>(wwrsolverDnHandle_t, int, int, int,
                                                           const double *, int, const double *,
                                                           int *);

// Function: ungqr_bufferSize
extern template wwrsolverStatus_t ungqr_bufferSize<wwrFloatComplex>(wwrsolverDnHandle_t, int, int,
                                                                    int, const wwrFloatComplex *,
                                                                    int, const wwrFloatComplex *,
                                                                    int *);
extern template wwrsolverStatus_t ungqr_bufferSize<wwrDoubleComplex>(wwrsolverDnHandle_t, int, int,
                                                                     int, const wwrDoubleComplex *,
                                                                     int, const wwrDoubleComplex *,
                                                                     int *);

// Function: orgqr
extern template wwrsolverStatus_t orgqr<float>(wwrsolverDnHandle_t, int, int, int, float *, int,
                                               const float *, float *, int, int *);
extern template wwrsolverStatus_t orgqr<double>(wwrsolverDnHandle_t, int, int, int, double *, int,
                                                const double *, double *, int, int *);

// Function: ungqr
extern template wwrsolverStatus_t ungqr<wwrFloatComplex>(wwrsolverDnHandle_t, int, int, int,
                                                         wwrFloatComplex *, int,
                                                         const wwrFloatComplex *, wwrFloatComplex *,
                                                         int, int *);
extern template wwrsolverStatus_t ungqr<wwrDoubleComplex>(wwrsolverDnHandle_t, int, int, int,
                                                          wwrDoubleComplex *, int,
                                                          const wwrDoubleComplex *,
                                                          wwrDoubleComplex *, int, int *);

// Function: orgbr_bufferSize
extern template wwrsolverStatus_t orgbr_bufferSize<float>(wwrsolverDnHandle_t, wwrblasSideMode_t,
                                                          int, int, int, const float *, int,
                                                          const float *, int *);
extern template wwrsolverStatus_t orgbr_bufferSize<double>(wwrsolverDnHandle_t, wwrblasSideMode_t,
                                                           int, int, int, const double *, int,
                                                           const double *, int *);

// Function: ungbr_bufferSize
extern template wwrsolverStatus_t
ungbr_bufferSize<wwrFloatComplex>(wwrsolverDnHandle_t, wwrblasSideMode_t, int, int, int,
                                  const wwrFloatComplex *, int, const wwrFloatComplex *, int *);
extern template wwrsolverStatus_t
ungbr_bufferSize<wwrDoubleComplex>(wwrsolverDnHandle_t, wwrblasSideMode_t, int, int, int,
                                   const wwrDoubleComplex *, int, const wwrDoubleComplex *, int *);

// Function: orgbr
extern template wwrsolverStatus_t orgbr<float>(wwrsolverDnHandle_t, wwrblasSideMode_t, int, int,
                                               int, float *, int, const float *, float *, int,
                                               int *);
extern template wwrsolverStatus_t orgbr<double>(wwrsolverDnHandle_t, wwrblasSideMode_t, int, int,
                                                int, double *, int, const double *, double *, int,
                                                int *);

// Function: ungbr
extern template wwrsolverStatus_t ungbr<wwrFloatComplex>(wwrsolverDnHandle_t, wwrblasSideMode_t,
                                                         int, int, int, wwrFloatComplex *, int,
                                                         const wwrFloatComplex *, wwrFloatComplex *,
                                                         int, int *);
extern template wwrsolverStatus_t ungbr<wwrDoubleComplex>(wwrsolverDnHandle_t, wwrblasSideMode_t,
                                                          int, int, int, wwrDoubleComplex *, int,
                                                          const wwrDoubleComplex *,
                                                          wwrDoubleComplex *, int, int *);

} // namespace wwr
