/**
 * @file linear_solver_legacy.cppm
 * @brief Legacy GPU solver linear-solver API wrappers (int-based, pre-params API)
 *
 * This module provides type-safe C++ wrappers for the legacy cuSOLVER Dense /
 * hipSOLVER Dense API -- 32 functions, all shared between both backends
 * (verified signature by signature against hipsolver-dense.h). For new code,
 * prefer the modern API in the :linear_solver partition, which uses
 * gpusolverDnParams_t and int64_t dimensions -- though only 8 of its
 * functions have a hipSOLVER counterpart at all.
 *
 * Legacy API characteristics:
 * - Uses int for all dimensions (not int64_t)
 * - No gpusolverDnParams_t parameter
 * - Single workspace buffer (not separate device/host)
 * - Pivot arrays use int* (not int64_t*)
 *
 * Usage:
 *   import gpumod.wrappers.solver;
 *   using namespace wwr;
 *
 *   // Query workspace size
 *   int Lwork;
 *   potrf_bufferSize<float>(handle, uplo, n, A, lda, &Lwork);
 *
 *   // Allocate and compute
 *   float* workspace;
 *   gpuMalloc(&workspace, Lwork * sizeof(float));
 *   potrf<float>(handle, uplo, n, A, lda, workspace, Lwork, devInfo);
 */

module;

#include "dispatch_macros.h"

export module gpumod.wrappers.solver:linear_solver_legacy;

import gpumod.solver;
import gpumod.blas;
import gpumod.complex;
import gpumod.wrappers.common;
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
 * @tparam T Data type (float, double, gpuFloatComplex, gpuDoubleComplex)
 * @param handle cuSOLVER handle
 * @param uplo Fill mode: CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER
 * @param n Order of matrix A
 * @param A Pointer to matrix A (device memory)
 * @param lda Leading dimension of A
 * @param Lwork Output: required workspace size in elements of type T
 * @return gpusolverStatus_t status code
 */
template<usual_fp T>
gpusolverStatus_t potrf_bufferSize(gpusolverDnHandle_t handle, gpublasFillMode_t uplo, int n, T *A,
                                   int lda, int *Lwork) {
  WWR_REAL_DISPATCH(T, gpusolverDn, S, D, potrf_bufferSize, handle, uplo, n, A, lda, Lwork);
  WWR_COMPLEX_DISPATCH(T, gpusolverDn, C, Z, potrf_bufferSize, handle, uplo, n, A, lda, Lwork);
}

/**
 * @brief Compute Cholesky factorization A = L*L^T or U^T*U
 *
 * Computes the Cholesky factorization of a Hermitian positive-definite matrix A.
 * - If uplo = CUBLAS_FILL_MODE_LOWER: A = L*L^T (lower triangular)
 * - If uplo = CUBLAS_FILL_MODE_UPPER: A = U^T*U (upper triangular)
 *
 * @tparam T Data type (float, double, gpuFloatComplex, gpuDoubleComplex)
 * @param handle cuSOLVER handle
 * @param uplo Fill mode: CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER
 * @param n Order of matrix A
 * @param A Pointer to matrix A (device memory); overwritten with factorization
 * @param lda Leading dimension of A (lda >= max(1, n))
 * @param Workspace Device workspace buffer (size from potrf_bufferSize)
 * @param Lwork Workspace size in elements of type T
 * @param devInfo Device pointer: 0 on success, i if A(i,i) is not positive definite
 * @return gpusolverStatus_t status code
 */
template<usual_fp T>
gpusolverStatus_t potrf(gpusolverDnHandle_t handle, gpublasFillMode_t uplo, int n, T *A, int lda,
                        T *Workspace, int Lwork, int *devInfo) {
  WWR_REAL_DISPATCH(T, gpusolverDn, S, D, potrf, handle, uplo, n, A, lda, Workspace, Lwork,
                       devInfo);
  WWR_COMPLEX_DISPATCH(T, gpusolverDn, C, Z, potrf, handle, uplo, n, A, lda, Workspace, Lwork,
                          devInfo);
}

/**
 * @brief Solve system using Cholesky factorization
 *
 * Solves a system of linear equations A*X = B using the Cholesky factorization
 * computed by potrf(). Matrix A must have been factorized before calling this function.
 *
 * @tparam T Data type (float, double, gpuFloatComplex, gpuDoubleComplex)
 * @param handle cuSOLVER handle
 * @param uplo Fill mode used in potrf: CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER
 * @param n Order of matrix A
 * @param nrhs Number of right-hand sides (columns of B)
 * @param A Pointer to factorized matrix from potrf (device memory, not modified)
 * @param lda Leading dimension of A (lda >= max(1, n))
 * @param B Pointer to matrix B (device memory); overwritten with solution X
 * @param ldb Leading dimension of B (ldb >= max(1, n))
 * @param devInfo Device pointer: 0 on success
 * @return gpusolverStatus_t status code
 */
template<usual_fp T>
gpusolverStatus_t potrs(gpusolverDnHandle_t handle, gpublasFillMode_t uplo, int n, int nrhs,
                        const T *A, int lda, T *B, int ldb, int *devInfo) {
  WWR_REAL_DISPATCH(T, gpusolverDn, S, D, potrs, handle, uplo, n, nrhs, A, lda, B, ldb, devInfo);
  WWR_COMPLEX_DISPATCH(T, gpusolverDn, C, Z, potrs, handle, uplo, n, nrhs, A, lda, B, ldb,
                          devInfo);
}

/**
 * @brief Query workspace size for matrix inversion using Cholesky factorization
 *
 * This function computes the size of the workspace buffer required for the
 * matrix inversion operation using Cholesky factorization.
 *
 * @tparam T Data type (float, double, gpuFloatComplex, gpuDoubleComplex)
 * @param handle cuSOLVER handle
 * @param uplo Fill mode: CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER
 * @param n Order of matrix A
 * @param A Pointer to factorized matrix from potrf (device memory)
 * @param lda Leading dimension of A
 * @param Lwork Output: required workspace size in elements of type T
 * @return gpusolverStatus_t status code
 */
template<usual_fp T>
gpusolverStatus_t potri_bufferSize(gpusolverDnHandle_t handle, gpublasFillMode_t uplo, int n, T *A,
                                   int lda, int *Lwork) {
  WWR_REAL_DISPATCH(T, gpusolverDn, S, D, potri_bufferSize, handle, uplo, n, A, lda, Lwork);
  WWR_COMPLEX_DISPATCH(T, gpusolverDn, C, Z, potri_bufferSize, handle, uplo, n, A, lda, Lwork);
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
 * @tparam T Data type (float, double, gpuFloatComplex, gpuDoubleComplex)
 * @param handle cuSOLVER handle
 * @param uplo Fill mode used in potrf: CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER
 * @param n Order of matrix A (n >= 0)
 * @param A Pointer to factorized matrix from potrf (device memory); overwritten with A^-1
 * @param lda Leading dimension of A (lda >= max(1, n))
 * @param Workspace Device workspace buffer (size from potri_bufferSize)
 * @param Lwork Workspace size in elements of type T
 * @param devInfo Device pointer: 0 on success, i if A(i,i) is zero (singular matrix)
 * @return gpusolverStatus_t status code
 */
template<usual_fp T>
gpusolverStatus_t potri(gpusolverDnHandle_t handle, gpublasFillMode_t uplo, int n, T *A, int lda,
                        T *Workspace, int Lwork, int *devInfo) {
  WWR_REAL_DISPATCH(T, gpusolverDn, S, D, potri, handle, uplo, n, A, lda, Workspace, Lwork,
                       devInfo);
  WWR_COMPLEX_DISPATCH(T, gpusolverDn, C, Z, potri, handle, uplo, n, A, lda, Workspace, Lwork,
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
 * @tparam T Data type (float, double, gpuFloatComplex, gpuDoubleComplex)
 * @param handle cuSOLVER handle
 * @param m Number of rows of matrix A
 * @param n Number of columns of matrix A
 * @param A Pointer to matrix A (device memory)
 * @param lda Leading dimension of A
 * @param Lwork Output: required workspace size in elements of type T
 * @return gpusolverStatus_t status code
 */
template<usual_fp T>
gpusolverStatus_t getrf_bufferSize(gpusolverDnHandle_t handle, int m, int n, T *A, int lda,
                                   int *Lwork) {
  WWR_REAL_DISPATCH(T, gpusolverDn, S, D, getrf_bufferSize, handle, m, n, A, lda, Lwork);
  WWR_COMPLEX_DISPATCH(T, gpusolverDn, C, Z, getrf_bufferSize, handle, m, n, A, lda, Lwork);
}

/**
 * @brief Compute LU factorization with partial pivoting
 *
 * Computes the LU factorization of a general m×n matrix A:
 * P*A = L*U
 * where P is a permutation matrix, L is lower triangular with unit diagonal,
 * and U is upper triangular.
 *
 * @tparam T Data type (float, double, gpuFloatComplex, gpuDoubleComplex)
 * @param handle cuSOLVER handle
 * @param m Number of rows of matrix A
 * @param n Number of columns of matrix A
 * @param A Pointer to matrix A (device memory); overwritten with L and U
 * @param lda Leading dimension of A (lda >= max(1, m))
 * @param Workspace Device workspace buffer (size from getrf_bufferSize)
 * @param devIpiv Device array of pivot indices (min(m,n) elements)
 * @param devInfo Device pointer: 0 on success, i if U(i,i)=0 (singular matrix)
 * @return gpusolverStatus_t status code
 */
template<usual_fp T>
gpusolverStatus_t getrf(gpusolverDnHandle_t handle, int m, int n, T *A, int lda, T *Workspace,
                        int *devIpiv, int *devInfo) {
  WWR_REAL_DISPATCH(T, gpusolverDn, S, D, getrf, handle, m, n, A, lda, Workspace, devIpiv,
                       devInfo);
  WWR_COMPLEX_DISPATCH(T, gpusolverDn, C, Z, getrf, handle, m, n, A, lda, Workspace, devIpiv,
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
 * @tparam T Data type (float, double, gpuFloatComplex, gpuDoubleComplex)
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
 * @return gpusolverStatus_t status code
 */
template<usual_fp T>
gpusolverStatus_t getrs(gpusolverDnHandle_t handle, gpublasOperation_t trans, int n, int nrhs,
                        const T *A, int lda, const int *devIpiv, T *B, int ldb, int *devInfo) {
  WWR_REAL_DISPATCH(T, gpusolverDn, S, D, getrs, handle, trans, n, nrhs, A, lda, devIpiv, B, ldb,
                       devInfo);
  WWR_COMPLEX_DISPATCH(T, gpusolverDn, C, Z, getrs, handle, trans, n, nrhs, A, lda, devIpiv, B,
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
 * @tparam T Data type (float, double, gpuFloatComplex, gpuDoubleComplex)
 * @param handle cuSOLVER handle
 * @param m Number of rows of matrix A
 * @param n Number of columns of matrix A
 * @param A Pointer to matrix A (device memory)
 * @param lda Leading dimension of A
 * @param Lwork Output: required workspace size in elements of type T
 * @return gpusolverStatus_t status code
 */
template<usual_fp T>
gpusolverStatus_t geqrf_bufferSize(gpusolverDnHandle_t handle, int m, int n, T *A, int lda,
                                   int *Lwork) {
  WWR_REAL_DISPATCH(T, gpusolverDn, S, D, geqrf_bufferSize, handle, m, n, A, lda, Lwork);
  WWR_COMPLEX_DISPATCH(T, gpusolverDn, C, Z, geqrf_bufferSize, handle, m, n, A, lda, Lwork);
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
 * @tparam T Data type (float, double, gpuFloatComplex, gpuDoubleComplex)
 * @param handle cuSOLVER handle
 * @param m Number of rows of matrix A (m >= 0)
 * @param n Number of columns of matrix A (n >= 0)
 * @param A Pointer to matrix A (device memory); overwritten with R and reflectors
 * @param lda Leading dimension of A (lda >= max(1, m))
 * @param TAU Device array of scalar factors of reflectors (min(m,n) elements)
 * @param Workspace Device workspace buffer (size from geqrf_bufferSize)
 * @param Lwork Workspace size in elements of type T
 * @param devInfo Device pointer: 0 on success
 * @return gpusolverStatus_t status code
 */
template<usual_fp T>
gpusolverStatus_t geqrf(gpusolverDnHandle_t handle, int m, int n, T *A, int lda, T *TAU,
                        T *Workspace, int Lwork, int *devInfo) {
  WWR_REAL_DISPATCH(T, gpusolverDn, S, D, geqrf, handle, m, n, A, lda, TAU, Workspace, Lwork,
                       devInfo);
  WWR_COMPLEX_DISPATCH(T, gpusolverDn, C, Z, geqrf, handle, m, n, A, lda, TAU, Workspace, Lwork,
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
 * @return gpusolverStatus_t status code
 */
template<real_fp T>
gpusolverStatus_t ormqr_bufferSize(gpusolverDnHandle_t handle, gpublasSideMode_t side,
                                   gpublasOperation_t trans, int m, int n, int k, const T *A,
                                   int lda, const T *tau, const T *C, int ldc, int *Lwork) {
  WWR_REAL_DISPATCH(T, gpusolverDn, S, D, ormqr_bufferSize, handle, side, trans, m, n, k, A, lda,
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
 * @return gpusolverStatus_t status code
 */
template<real_fp T>
gpusolverStatus_t ormqr(gpusolverDnHandle_t handle, gpublasSideMode_t side,
                        gpublasOperation_t trans, int m, int n, int k, const T *A, int lda,
                        const T *tau, T *C, int ldc, T *Workspace, int Lwork, int *devInfo) {
  WWR_REAL_DISPATCH(T, gpusolverDn, S, D, ormqr, handle, side, trans, m, n, k, A, lda, tau, C,
                       ldc, Workspace, Lwork, devInfo);
}

/**
 * @brief Query workspace size for applying unitary matrix Q from geqrf (complex types)
 *
 * This function computes the size of the workspace buffer required for the
 * unmqr operation (complex types only).
 *
 * @tparam T Data type (gpuFloatComplex or gpuDoubleComplex only)
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
 * @return gpusolverStatus_t status code
 */
template<complex_fp T>
gpusolverStatus_t unmqr_bufferSize(gpusolverDnHandle_t handle, gpublasSideMode_t side,
                                   gpublasOperation_t trans, int m, int n, int k, const T *A,
                                   int lda, const T *tau, const T *C, int ldc, int *Lwork) {
  WWR_COMPLEX_DISPATCH(T, gpusolverDn, C, Z, unmqr_bufferSize, handle, side, trans, m, n, k, A,
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
 * @tparam T Data type (gpuFloatComplex or gpuDoubleComplex only)
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
 * @return gpusolverStatus_t status code
 */
template<complex_fp T>
gpusolverStatus_t unmqr(gpusolverDnHandle_t handle, gpublasSideMode_t side,
                        gpublasOperation_t trans, int m, int n, int k, const T *A, int lda,
                        const T *tau, T *C, int ldc, T *Workspace, int Lwork, int *devInfo) {
  WWR_COMPLEX_DISPATCH(T, gpusolverDn, C, Z, unmqr, handle, side, trans, m, n, k, A, lda, tau, C,
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
 * @tparam T Data type (float, double, gpuFloatComplex, gpuDoubleComplex)
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
 * @return gpusolverStatus_t status code
 */
template<usual_fp T>
gpusolverStatus_t gels_bufferSize(gpusolverDnHandle_t handle, int m, int n, int nrhs, T *dA,
                                  int ldda, T *dB, int lddb, T *dX, int lddx, void *dWorkspace,
                                  std::size_t *lwork_bytes) {
  WWR_REAL_DISPATCH(T, gpusolverDn, SS, DD, gels_bufferSize, handle, m, n, nrhs, dA, ldda, dB,
                       lddb, dX, lddx, dWorkspace, lwork_bytes);
  WWR_COMPLEX_DISPATCH(T, gpusolverDn, CC, ZZ, gels_bufferSize, handle, m, n, nrhs, dA, ldda, dB,
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
 * @tparam T Data type (float, double, gpuFloatComplex, gpuDoubleComplex)
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
 * @return gpusolverStatus_t status code
 */
template<usual_fp T>
gpusolverStatus_t gels(gpusolverDnHandle_t handle, int m, int n, int nrhs, T *dA, int ldda, T *dB,
                       int lddb, T *dX, int lddx, void *dWorkspace, std::size_t lwork_bytes,
                       int *iter, int *d_info) {
  WWR_REAL_DISPATCH(T, gpusolverDn, SS, DD, gels, handle, m, n, nrhs, dA, ldda, dB, lddb, dX,
                       lddx, dWorkspace, lwork_bytes, iter, d_info);
  WWR_COMPLEX_DISPATCH(T, gpusolverDn, CC, ZZ, gels, handle, m, n, nrhs, dA, ldda, dB, lddb, dX,
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
 * @tparam T Data type (float, double, gpuFloatComplex, gpuDoubleComplex)
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
 * @return gpusolverStatus_t status code
 */
template<usual_fp T>
gpusolverStatus_t gesv_bufferSize(gpusolverDnHandle_t handle, int n, int nrhs, T *dA, int ldda,
                                  int *dipiv, T *dB, int lddb, T *dX, int lddx, void *dWorkspace,
                                  std::size_t *lwork_bytes) {
  WWR_REAL_DISPATCH(T, gpusolverDn, SS, DD, gesv_bufferSize, handle, n, nrhs, dA, ldda, dipiv,
                       dB, lddb, dX, lddx, dWorkspace, lwork_bytes);
  WWR_COMPLEX_DISPATCH(T, gpusolverDn, CC, ZZ, gesv_bufferSize, handle, n, nrhs, dA, ldda, dipiv,
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
 * @tparam T Data type (float, double, gpuFloatComplex, gpuDoubleComplex)
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
 * @return gpusolverStatus_t status code
 */
template<usual_fp T>
gpusolverStatus_t gesv(gpusolverDnHandle_t handle, int n, int nrhs, T *dA, int ldda, int *dipiv,
                       T *dB, int lddb, T *dX, int lddx, void *dWorkspace, std::size_t lwork_bytes,
                       int *iter, int *d_info) {
  WWR_REAL_DISPATCH(T, gpusolverDn, SS, DD, gesv, handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX,
                       lddx, dWorkspace, lwork_bytes, iter, d_info);
  WWR_COMPLEX_DISPATCH(T, gpusolverDn, CC, ZZ, gesv, handle, n, nrhs, dA, ldda, dipiv, dB, lddb,
                          dX, lddx, dWorkspace, lwork_bytes, iter, d_info);
}

// ========================================================================
// Batched Cholesky Factorization
// ========================================================================

/**
 * @brief Compute batched Cholesky factorization A = L*L^T or U^T*U
 *
 * @tparam T Data type (float, double, gpuFloatComplex, gpuDoubleComplex)
 * @param handle cuSOLVER handle
 * @param uplo Fill mode
 * @param n Order of each matrix
 * @param Aarray Array of pointers to matrices (device memory)
 * @param lda Leading dimension
 * @param infoArray Array of status codes (device memory)
 * @param batchSize Number of matrices
 * @return gpusolverStatus_t status code
 */
template<usual_fp T>
gpusolverStatus_t potrfBatched(gpusolverDnHandle_t handle, gpublasFillMode_t uplo, int n,
                               T *Aarray[], int lda, int *infoArray, int batchSize) {
  WWR_REAL_DISPATCH(T, gpusolverDn, S, D, potrfBatched, handle, uplo, n, Aarray, lda, infoArray,
                       batchSize);
  WWR_COMPLEX_DISPATCH(T, gpusolverDn, C, Z, potrfBatched, handle, uplo, n, Aarray, lda,
                          infoArray, batchSize);
}

/**
 * @brief Solve batched systems using Cholesky factorization
 *
 * @tparam T Data type (float, double, gpuFloatComplex, gpuDoubleComplex)
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
 * @return gpusolverStatus_t status code
 */
template<usual_fp T>
gpusolverStatus_t potrsBatched(gpusolverDnHandle_t handle, gpublasFillMode_t uplo, int n, int nrhs,
                               T *A[], int lda, T *B[], int ldb, int *info, int batchSize) {
  WWR_REAL_DISPATCH(T, gpusolverDn, S, D, potrsBatched, handle, uplo, n, nrhs, A, lda, B, ldb,
                       info, batchSize);
  WWR_COMPLEX_DISPATCH(T, gpusolverDn, C, Z, potrsBatched, handle, uplo, n, nrhs, A, lda, B, ldb,
                          info, batchSize);
}

// ========================================================================
// Symmetric/Hermitian Factorization (Bunch-Kaufman LDLT)
// ========================================================================

/**
 * @brief Query workspace size for symmetric factorization (Bunch-Kaufman)
 *
 * @tparam T Data type (float, double, gpuFloatComplex, gpuDoubleComplex)
 * @param handle cuSOLVER handle
 * @param uplo Fill mode
 * @param n Order of matrix
 * @param A Pointer to matrix (device memory)
 * @param lda Leading dimension
 * @param ipiv Pivot indices (device memory)
 * @param lwork Output: workspace size
 * @return gpusolverStatus_t status code
 */
template<usual_fp T>
gpusolverStatus_t sytrf_bufferSize(gpusolverDnHandle_t handle, int n, T *A, int lda, int *lwork) {
  WWR_REAL_DISPATCH(T, gpusolverDn, S, D, sytrf_bufferSize, handle, n, A, lda, lwork);
  WWR_COMPLEX_DISPATCH(T, gpusolverDn, C, Z, sytrf_bufferSize, handle, n, A, lda, lwork);
}

/**
 * @brief Compute symmetric factorization using Bunch-Kaufman (A = U*D*U^T or L*D*L^T)
 *
 * @tparam T Data type (float, double, gpuFloatComplex, gpuDoubleComplex)
 * @param handle cuSOLVER handle
 * @param uplo Fill mode
 * @param n Order of matrix
 * @param A Pointer to matrix (device memory); overwritten with factorization
 * @param lda Leading dimension
 * @param ipiv Pivot indices (device memory, n elements)
 * @param work Workspace buffer
 * @param lwork Workspace size
 * @param devInfo Device pointer: 0 on success
 * @return gpusolverStatus_t status code
 */
template<usual_fp T>
gpusolverStatus_t sytrf(gpusolverDnHandle_t handle, gpublasFillMode_t uplo, int n, T *A, int lda,
                        int *ipiv, T *work, int lwork, int *devInfo) {
  WWR_REAL_DISPATCH(T, gpusolverDn, S, D, sytrf, handle, uplo, n, A, lda, ipiv, work, lwork,
                       devInfo);
  WWR_COMPLEX_DISPATCH(T, gpusolverDn, C, Z, sytrf, handle, uplo, n, A, lda, ipiv, work, lwork,
                          devInfo);
}

// ========================================================================
// Bidiagonal Reduction
// ========================================================================

/**
 * @brief Query workspace size for bidiagonal reduction
 *
 * @tparam T Data type (float, double, gpuFloatComplex, gpuDoubleComplex)
 * @param handle cuSOLVER handle
 * @param m Number of rows
 * @param n Number of columns
 * @param lwork Output: workspace size
 * @return gpusolverStatus_t status code
 */
template<usual_fp T>
gpusolverStatus_t gebrd_bufferSize(gpusolverDnHandle_t handle, int m, int n, int *lwork) {
  WWR_REAL_DISPATCH(T, gpusolverDn, S, D, gebrd_bufferSize, handle, m, n, lwork);
  WWR_COMPLEX_DISPATCH(T, gpusolverDn, C, Z, gebrd_bufferSize, handle, m, n, lwork);
}

/**
 * @brief Reduce general matrix to bidiagonal form
 *
 * @tparam T Data type (float, double, gpuFloatComplex, gpuDoubleComplex)
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
 * @return gpusolverStatus_t status code
 */
template<usual_fp T>
gpusolverStatus_t gebrd(gpusolverDnHandle_t handle, int m, int n, T *A, int lda,
                        ComplexToRealType<T> *D, ComplexToRealType<T> *E, T *tauq, T *taup, T *work,
                        int lwork, int *devInfo) {
  WWR_REAL_DISPATCH(T, gpusolverDn, S, D, gebrd, handle, m, n, A, lda, D, E, tauq, taup, work,
                       lwork, devInfo);
  WWR_COMPLEX_DISPATCH(T, gpusolverDn, C, Z, gebrd, handle, m, n, A, lda, D, E, tauq, taup, work,
                          lwork, devInfo);
}

// ========================================================================
// Generate Orthogonal/Unitary Matrix from QR (orgqr/ungqr)
// ========================================================================

/**
 * @brief Query workspace size for generating orthogonal/unitary matrix from QR
 *
 * @tparam T Data type (float, double for orgqr; gpuFloatComplex, gpuDoubleComplex for ungqr)
 * @param handle cuSOLVER handle
 * @param m Number of rows
 * @param n Number of columns
 * @param k Number of elementary reflectors
 * @param A Pointer to matrix (device memory)
 * @param lda Leading dimension
 * @param tau Elementary reflectors (device memory)
 * @param lwork Output: workspace size
 * @return gpusolverStatus_t status code
 */
template<real_fp T>
gpusolverStatus_t orgqr_bufferSize(gpusolverDnHandle_t handle, int m, int n, int k, const T *A,
                                   int lda, const T *tau, int *lwork) {
  WWR_REAL_DISPATCH(T, gpusolverDn, S, D, orgqr_bufferSize, handle, m, n, k, A, lda, tau, lwork);
}

template<complex_fp T>
gpusolverStatus_t ungqr_bufferSize(gpusolverDnHandle_t handle, int m, int n, int k, const T *A,
                                   int lda, const T *tau, int *lwork) {
  WWR_COMPLEX_DISPATCH(T, gpusolverDn, C, Z, ungqr_bufferSize, handle, m, n, k, A, lda, tau,
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
 * @return gpusolverStatus_t status code
 */
template<real_fp T>
gpusolverStatus_t orgqr(gpusolverDnHandle_t handle, int m, int n, int k, T *A, int lda,
                        const T *tau, T *work, int lwork, int *devInfo) {
  WWR_REAL_DISPATCH(T, gpusolverDn, S, D, orgqr, handle, m, n, k, A, lda, tau, work, lwork,
                       devInfo);
}

template<complex_fp T>
gpusolverStatus_t ungqr(gpusolverDnHandle_t handle, int m, int n, int k, T *A, int lda,
                        const T *tau, T *work, int lwork, int *devInfo) {
  WWR_COMPLEX_DISPATCH(T, gpusolverDn, C, Z, ungqr, handle, m, n, k, A, lda, tau, work, lwork,
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
 * @return gpusolverStatus_t status code
 */
template<real_fp T>
gpusolverStatus_t orgbr_bufferSize(gpusolverDnHandle_t handle, gpublasSideMode_t vect, int m, int n,
                                   int k, const T *A, int lda, const T *tau, int *lwork) {
  WWR_REAL_DISPATCH(T, gpusolverDn, S, D, orgbr_bufferSize, handle, vect, m, n, k, A, lda, tau,
                       lwork);
}

template<complex_fp T>
gpusolverStatus_t ungbr_bufferSize(gpusolverDnHandle_t handle, gpublasSideMode_t vect, int m, int n,
                                   int k, const T *A, int lda, const T *tau, int *lwork) {
  WWR_COMPLEX_DISPATCH(T, gpusolverDn, C, Z, ungbr_bufferSize, handle, vect, m, n, k, A, lda,
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
 * @return gpusolverStatus_t status code
 */
template<real_fp T>
gpusolverStatus_t orgbr(gpusolverDnHandle_t handle, gpublasSideMode_t vect, int m, int n, int k,
                        T *A, int lda, const T *tau, T *work, int lwork, int *devInfo) {
  WWR_REAL_DISPATCH(T, gpusolverDn, S, D, orgbr, handle, vect, m, n, k, A, lda, tau, work, lwork,
                       devInfo);
}

template<complex_fp T>
gpusolverStatus_t ungbr(gpusolverDnHandle_t handle, gpublasSideMode_t vect, int m, int n, int k,
                        T *A, int lda, const T *tau, T *work, int lwork, int *devInfo) {
  WWR_COMPLEX_DISPATCH(T, gpusolverDn, C, Z, ungbr, handle, vect, m, n, k, A, lda, tau, work,
                          lwork, devInfo);
}

// ==================== Explicit Template Instantiation Declarations ====================
// Generated code will be inserted here by cmake/instantiation/generate_instantiations.py
// To regenerate: cmake --build build --target gpusolverDn_generate_instantiations

// Function: potrf_bufferSize
extern template gpusolverStatus_t potrf_bufferSize<float>(gpusolverDnHandle_t, gpublasFillMode_t,
                                                          int, float *, int, int *);
extern template gpusolverStatus_t potrf_bufferSize<double>(gpusolverDnHandle_t, gpublasFillMode_t,
                                                           int, double *, int, int *);
extern template gpusolverStatus_t potrf_bufferSize<gpuFloatComplex>(gpusolverDnHandle_t,
                                                                    gpublasFillMode_t, int,
                                                                    gpuFloatComplex *, int, int *);
extern template gpusolverStatus_t potrf_bufferSize<gpuDoubleComplex>(gpusolverDnHandle_t,
                                                                     gpublasFillMode_t, int,
                                                                     gpuDoubleComplex *, int,
                                                                     int *);

// Function: potrf
extern template gpusolverStatus_t potrf<float>(gpusolverDnHandle_t, gpublasFillMode_t, int, float *,
                                               int, float *, int, int *);
extern template gpusolverStatus_t potrf<double>(gpusolverDnHandle_t, gpublasFillMode_t, int,
                                                double *, int, double *, int, int *);
extern template gpusolverStatus_t potrf<gpuFloatComplex>(gpusolverDnHandle_t, gpublasFillMode_t,
                                                         int, gpuFloatComplex *, int,
                                                         gpuFloatComplex *, int, int *);
extern template gpusolverStatus_t potrf<gpuDoubleComplex>(gpusolverDnHandle_t, gpublasFillMode_t,
                                                          int, gpuDoubleComplex *, int,
                                                          gpuDoubleComplex *, int, int *);

// Function: potrs
extern template gpusolverStatus_t potrs<float>(gpusolverDnHandle_t, gpublasFillMode_t, int, int,
                                               const float *, int, float *, int, int *);
extern template gpusolverStatus_t potrs<double>(gpusolverDnHandle_t, gpublasFillMode_t, int, int,
                                                const double *, int, double *, int, int *);
extern template gpusolverStatus_t potrs<gpuFloatComplex>(gpusolverDnHandle_t, gpublasFillMode_t,
                                                         int, int, const gpuFloatComplex *, int,
                                                         gpuFloatComplex *, int, int *);
extern template gpusolverStatus_t potrs<gpuDoubleComplex>(gpusolverDnHandle_t, gpublasFillMode_t,
                                                          int, int, const gpuDoubleComplex *, int,
                                                          gpuDoubleComplex *, int, int *);

// Function: potri_bufferSize
extern template gpusolverStatus_t potri_bufferSize<float>(gpusolverDnHandle_t, gpublasFillMode_t,
                                                          int, float *, int, int *);
extern template gpusolverStatus_t potri_bufferSize<double>(gpusolverDnHandle_t, gpublasFillMode_t,
                                                           int, double *, int, int *);
extern template gpusolverStatus_t potri_bufferSize<gpuFloatComplex>(gpusolverDnHandle_t,
                                                                    gpublasFillMode_t, int,
                                                                    gpuFloatComplex *, int, int *);
extern template gpusolverStatus_t potri_bufferSize<gpuDoubleComplex>(gpusolverDnHandle_t,
                                                                     gpublasFillMode_t, int,
                                                                     gpuDoubleComplex *, int,
                                                                     int *);

// Function: potri
extern template gpusolverStatus_t potri<float>(gpusolverDnHandle_t, gpublasFillMode_t, int, float *,
                                               int, float *, int, int *);
extern template gpusolverStatus_t potri<double>(gpusolverDnHandle_t, gpublasFillMode_t, int,
                                                double *, int, double *, int, int *);
extern template gpusolverStatus_t potri<gpuFloatComplex>(gpusolverDnHandle_t, gpublasFillMode_t,
                                                         int, gpuFloatComplex *, int,
                                                         gpuFloatComplex *, int, int *);
extern template gpusolverStatus_t potri<gpuDoubleComplex>(gpusolverDnHandle_t, gpublasFillMode_t,
                                                          int, gpuDoubleComplex *, int,
                                                          gpuDoubleComplex *, int, int *);

// Function: getrf_bufferSize
extern template gpusolverStatus_t getrf_bufferSize<float>(gpusolverDnHandle_t, int, int, float *,
                                                          int, int *);
extern template gpusolverStatus_t getrf_bufferSize<double>(gpusolverDnHandle_t, int, int, double *,
                                                           int, int *);
extern template gpusolverStatus_t getrf_bufferSize<gpuFloatComplex>(gpusolverDnHandle_t, int, int,
                                                                    gpuFloatComplex *, int, int *);
extern template gpusolverStatus_t
getrf_bufferSize<gpuDoubleComplex>(gpusolverDnHandle_t, int, int, gpuDoubleComplex *, int, int *);

// Function: getrf
extern template gpusolverStatus_t getrf<float>(gpusolverDnHandle_t, int, int, float *, int, float *,
                                               int *, int *);
extern template gpusolverStatus_t getrf<double>(gpusolverDnHandle_t, int, int, double *, int,
                                                double *, int *, int *);
extern template gpusolverStatus_t getrf<gpuFloatComplex>(gpusolverDnHandle_t, int, int,
                                                         gpuFloatComplex *, int, gpuFloatComplex *,
                                                         int *, int *);
extern template gpusolverStatus_t getrf<gpuDoubleComplex>(gpusolverDnHandle_t, int, int,
                                                          gpuDoubleComplex *, int,
                                                          gpuDoubleComplex *, int *, int *);

// Function: getrs
extern template gpusolverStatus_t getrs<float>(gpusolverDnHandle_t, gpublasOperation_t, int, int,
                                               const float *, int, const int *, float *, int,
                                               int *);
extern template gpusolverStatus_t getrs<double>(gpusolverDnHandle_t, gpublasOperation_t, int, int,
                                                const double *, int, const int *, double *, int,
                                                int *);
extern template gpusolverStatus_t getrs<gpuFloatComplex>(gpusolverDnHandle_t, gpublasOperation_t,
                                                         int, int, const gpuFloatComplex *, int,
                                                         const int *, gpuFloatComplex *, int,
                                                         int *);
extern template gpusolverStatus_t getrs<gpuDoubleComplex>(gpusolverDnHandle_t, gpublasOperation_t,
                                                          int, int, const gpuDoubleComplex *, int,
                                                          const int *, gpuDoubleComplex *, int,
                                                          int *);

// Function: geqrf_bufferSize
extern template gpusolverStatus_t geqrf_bufferSize<float>(gpusolverDnHandle_t, int, int, float *,
                                                          int, int *);
extern template gpusolverStatus_t geqrf_bufferSize<double>(gpusolverDnHandle_t, int, int, double *,
                                                           int, int *);
extern template gpusolverStatus_t geqrf_bufferSize<gpuFloatComplex>(gpusolverDnHandle_t, int, int,
                                                                    gpuFloatComplex *, int, int *);
extern template gpusolverStatus_t
geqrf_bufferSize<gpuDoubleComplex>(gpusolverDnHandle_t, int, int, gpuDoubleComplex *, int, int *);

// Function: geqrf
extern template gpusolverStatus_t geqrf<float>(gpusolverDnHandle_t, int, int, float *, int, float *,
                                               float *, int, int *);
extern template gpusolverStatus_t geqrf<double>(gpusolverDnHandle_t, int, int, double *, int,
                                                double *, double *, int, int *);
extern template gpusolverStatus_t geqrf<gpuFloatComplex>(gpusolverDnHandle_t, int, int,
                                                         gpuFloatComplex *, int, gpuFloatComplex *,
                                                         gpuFloatComplex *, int, int *);
extern template gpusolverStatus_t geqrf<gpuDoubleComplex>(gpusolverDnHandle_t, int, int,
                                                          gpuDoubleComplex *, int,
                                                          gpuDoubleComplex *, gpuDoubleComplex *,
                                                          int, int *);

// Function: ormqr_bufferSize
extern template gpusolverStatus_t ormqr_bufferSize<float>(gpusolverDnHandle_t, gpublasSideMode_t,
                                                          gpublasOperation_t, int, int, int,
                                                          const float *, int, const float *,
                                                          const float *, int, int *);
extern template gpusolverStatus_t ormqr_bufferSize<double>(gpusolverDnHandle_t, gpublasSideMode_t,
                                                           gpublasOperation_t, int, int, int,
                                                           const double *, int, const double *,
                                                           const double *, int, int *);

// Function: ormqr
extern template gpusolverStatus_t ormqr<float>(gpusolverDnHandle_t, gpublasSideMode_t,
                                               gpublasOperation_t, int, int, int, const float *,
                                               int, const float *, float *, int, float *, int,
                                               int *);
extern template gpusolverStatus_t ormqr<double>(gpusolverDnHandle_t, gpublasSideMode_t,
                                                gpublasOperation_t, int, int, int, const double *,
                                                int, const double *, double *, int, double *, int,
                                                int *);

// Function: unmqr_bufferSize
extern template gpusolverStatus_t
unmqr_bufferSize<gpuFloatComplex>(gpusolverDnHandle_t, gpublasSideMode_t, gpublasOperation_t, int,
                                  int, int, const gpuFloatComplex *, int, const gpuFloatComplex *,
                                  const gpuFloatComplex *, int, int *);
extern template gpusolverStatus_t
unmqr_bufferSize<gpuDoubleComplex>(gpusolverDnHandle_t, gpublasSideMode_t, gpublasOperation_t, int,
                                   int, int, const gpuDoubleComplex *, int,
                                   const gpuDoubleComplex *, const gpuDoubleComplex *, int, int *);

// Function: unmqr
extern template gpusolverStatus_t unmqr<gpuFloatComplex>(gpusolverDnHandle_t, gpublasSideMode_t,
                                                         gpublasOperation_t, int, int, int,
                                                         const gpuFloatComplex *, int,
                                                         const gpuFloatComplex *, gpuFloatComplex *,
                                                         int, gpuFloatComplex *, int, int *);
extern template gpusolverStatus_t
unmqr<gpuDoubleComplex>(gpusolverDnHandle_t, gpublasSideMode_t, gpublasOperation_t, int, int, int,
                        const gpuDoubleComplex *, int, const gpuDoubleComplex *, gpuDoubleComplex *,
                        int, gpuDoubleComplex *, int, int *);

// Function: gels_bufferSize
extern template gpusolverStatus_t gels_bufferSize<float>(gpusolverDnHandle_t, int, int, int,
                                                         float *, int, float *, int, float *, int,
                                                         void *, std::size_t *);
extern template gpusolverStatus_t gels_bufferSize<double>(gpusolverDnHandle_t, int, int, int,
                                                          double *, int, double *, int, double *,
                                                          int, void *, std::size_t *);
extern template gpusolverStatus_t gels_bufferSize<gpuFloatComplex>(gpusolverDnHandle_t, int, int,
                                                                   int, gpuFloatComplex *, int,
                                                                   gpuFloatComplex *, int,
                                                                   gpuFloatComplex *, int, void *,
                                                                   std::size_t *);
extern template gpusolverStatus_t gels_bufferSize<gpuDoubleComplex>(gpusolverDnHandle_t, int, int,
                                                                    int, gpuDoubleComplex *, int,
                                                                    gpuDoubleComplex *, int,
                                                                    gpuDoubleComplex *, int, void *,
                                                                    std::size_t *);

// Function: gels
extern template gpusolverStatus_t gels<float>(gpusolverDnHandle_t, int, int, int, float *, int,
                                              float *, int, float *, int, void *, std::size_t,
                                              int *, int *);
extern template gpusolverStatus_t gels<double>(gpusolverDnHandle_t, int, int, int, double *, int,
                                               double *, int, double *, int, void *, std::size_t,
                                               int *, int *);
extern template gpusolverStatus_t gels<gpuFloatComplex>(gpusolverDnHandle_t, int, int, int,
                                                        gpuFloatComplex *, int, gpuFloatComplex *,
                                                        int, gpuFloatComplex *, int, void *,
                                                        std::size_t, int *, int *);
extern template gpusolverStatus_t gels<gpuDoubleComplex>(gpusolverDnHandle_t, int, int, int,
                                                         gpuDoubleComplex *, int,
                                                         gpuDoubleComplex *, int,
                                                         gpuDoubleComplex *, int, void *,
                                                         std::size_t, int *, int *);

// Function: gesv_bufferSize
extern template gpusolverStatus_t gesv_bufferSize<float>(gpusolverDnHandle_t, int, int, float *,
                                                         int, int *, float *, int, float *, int,
                                                         void *, std::size_t *);
extern template gpusolverStatus_t gesv_bufferSize<double>(gpusolverDnHandle_t, int, int, double *,
                                                          int, int *, double *, int, double *, int,
                                                          void *, std::size_t *);
extern template gpusolverStatus_t gesv_bufferSize<gpuFloatComplex>(gpusolverDnHandle_t, int, int,
                                                                   gpuFloatComplex *, int, int *,
                                                                   gpuFloatComplex *, int,
                                                                   gpuFloatComplex *, int, void *,
                                                                   std::size_t *);
extern template gpusolverStatus_t gesv_bufferSize<gpuDoubleComplex>(gpusolverDnHandle_t, int, int,
                                                                    gpuDoubleComplex *, int, int *,
                                                                    gpuDoubleComplex *, int,
                                                                    gpuDoubleComplex *, int, void *,
                                                                    std::size_t *);

// Function: gesv
extern template gpusolverStatus_t gesv<float>(gpusolverDnHandle_t, int, int, float *, int, int *,
                                              float *, int, float *, int, void *, std::size_t,
                                              int *, int *);
extern template gpusolverStatus_t gesv<double>(gpusolverDnHandle_t, int, int, double *, int, int *,
                                               double *, int, double *, int, void *, std::size_t,
                                               int *, int *);
extern template gpusolverStatus_t gesv<gpuFloatComplex>(gpusolverDnHandle_t, int, int,
                                                        gpuFloatComplex *, int, int *,
                                                        gpuFloatComplex *, int, gpuFloatComplex *,
                                                        int, void *, std::size_t, int *, int *);
extern template gpusolverStatus_t gesv<gpuDoubleComplex>(gpusolverDnHandle_t, int, int,
                                                         gpuDoubleComplex *, int, int *,
                                                         gpuDoubleComplex *, int,
                                                         gpuDoubleComplex *, int, void *,
                                                         std::size_t, int *, int *);

// Function: potrfBatched
extern template gpusolverStatus_t potrfBatched<float>(gpusolverDnHandle_t, gpublasFillMode_t, int,
                                                      float *[], int, int *, int);
extern template gpusolverStatus_t potrfBatched<double>(gpusolverDnHandle_t, gpublasFillMode_t, int,
                                                       double *[], int, int *, int);
extern template gpusolverStatus_t potrfBatched<gpuFloatComplex>(gpusolverDnHandle_t,
                                                                gpublasFillMode_t, int,
                                                                gpuFloatComplex *[], int, int *,
                                                                int);
extern template gpusolverStatus_t potrfBatched<gpuDoubleComplex>(gpusolverDnHandle_t,
                                                                 gpublasFillMode_t, int,
                                                                 gpuDoubleComplex *[], int, int *,
                                                                 int);

// Function: potrsBatched
extern template gpusolverStatus_t potrsBatched<float>(gpusolverDnHandle_t, gpublasFillMode_t, int,
                                                      int, float *[], int, float *[], int, int *,
                                                      int);
extern template gpusolverStatus_t potrsBatched<double>(gpusolverDnHandle_t, gpublasFillMode_t, int,
                                                       int, double *[], int, double *[], int, int *,
                                                       int);
extern template gpusolverStatus_t
potrsBatched<gpuFloatComplex>(gpusolverDnHandle_t, gpublasFillMode_t, int, int, gpuFloatComplex *[],
                              int, gpuFloatComplex *[], int, int *, int);
extern template gpusolverStatus_t
potrsBatched<gpuDoubleComplex>(gpusolverDnHandle_t, gpublasFillMode_t, int, int,
                               gpuDoubleComplex *[], int, gpuDoubleComplex *[], int, int *, int);

// Function: sytrf_bufferSize
extern template gpusolverStatus_t sytrf_bufferSize<float>(gpusolverDnHandle_t, int, float *, int,
                                                          int *);
extern template gpusolverStatus_t sytrf_bufferSize<double>(gpusolverDnHandle_t, int, double *, int,
                                                           int *);
extern template gpusolverStatus_t sytrf_bufferSize<gpuFloatComplex>(gpusolverDnHandle_t, int,
                                                                    gpuFloatComplex *, int, int *);
extern template gpusolverStatus_t
sytrf_bufferSize<gpuDoubleComplex>(gpusolverDnHandle_t, int, gpuDoubleComplex *, int, int *);

// Function: sytrf
extern template gpusolverStatus_t sytrf<float>(gpusolverDnHandle_t, gpublasFillMode_t, int, float *,
                                               int, int *, float *, int, int *);
extern template gpusolverStatus_t sytrf<double>(gpusolverDnHandle_t, gpublasFillMode_t, int,
                                                double *, int, int *, double *, int, int *);
extern template gpusolverStatus_t sytrf<gpuFloatComplex>(gpusolverDnHandle_t, gpublasFillMode_t,
                                                         int, gpuFloatComplex *, int, int *,
                                                         gpuFloatComplex *, int, int *);
extern template gpusolverStatus_t sytrf<gpuDoubleComplex>(gpusolverDnHandle_t, gpublasFillMode_t,
                                                          int, gpuDoubleComplex *, int, int *,
                                                          gpuDoubleComplex *, int, int *);

// Function: gebrd_bufferSize
extern template gpusolverStatus_t gebrd_bufferSize<float>(gpusolverDnHandle_t, int, int, int *);
extern template gpusolverStatus_t gebrd_bufferSize<double>(gpusolverDnHandle_t, int, int, int *);
extern template gpusolverStatus_t gebrd_bufferSize<gpuFloatComplex>(gpusolverDnHandle_t, int, int,
                                                                    int *);
extern template gpusolverStatus_t gebrd_bufferSize<gpuDoubleComplex>(gpusolverDnHandle_t, int, int,
                                                                     int *);

// Function: gebrd
extern template gpusolverStatus_t gebrd<float>(gpusolverDnHandle_t, int, int, float *, int,
                                               ComplexToRealType<float> *,
                                               ComplexToRealType<float> *, float *, float *,
                                               float *, int, int *);
extern template gpusolverStatus_t gebrd<double>(gpusolverDnHandle_t, int, int, double *, int,
                                                ComplexToRealType<double> *,
                                                ComplexToRealType<double> *, double *, double *,
                                                double *, int, int *);
extern template gpusolverStatus_t
gebrd<gpuFloatComplex>(gpusolverDnHandle_t, int, int, gpuFloatComplex *, int,
                       ComplexToRealType<gpuFloatComplex> *, ComplexToRealType<gpuFloatComplex> *,
                       gpuFloatComplex *, gpuFloatComplex *, gpuFloatComplex *, int, int *);
extern template gpusolverStatus_t gebrd<gpuDoubleComplex>(gpusolverDnHandle_t, int, int,
                                                          gpuDoubleComplex *, int,
                                                          ComplexToRealType<gpuDoubleComplex> *,
                                                          ComplexToRealType<gpuDoubleComplex> *,
                                                          gpuDoubleComplex *, gpuDoubleComplex *,
                                                          gpuDoubleComplex *, int, int *);

// Function: orgqr_bufferSize
extern template gpusolverStatus_t orgqr_bufferSize<float>(gpusolverDnHandle_t, int, int, int,
                                                          const float *, int, const float *, int *);
extern template gpusolverStatus_t orgqr_bufferSize<double>(gpusolverDnHandle_t, int, int, int,
                                                           const double *, int, const double *,
                                                           int *);

// Function: ungqr_bufferSize
extern template gpusolverStatus_t ungqr_bufferSize<gpuFloatComplex>(gpusolverDnHandle_t, int, int,
                                                                    int, const gpuFloatComplex *,
                                                                    int, const gpuFloatComplex *,
                                                                    int *);
extern template gpusolverStatus_t ungqr_bufferSize<gpuDoubleComplex>(gpusolverDnHandle_t, int, int,
                                                                     int, const gpuDoubleComplex *,
                                                                     int, const gpuDoubleComplex *,
                                                                     int *);

// Function: orgqr
extern template gpusolverStatus_t orgqr<float>(gpusolverDnHandle_t, int, int, int, float *, int,
                                               const float *, float *, int, int *);
extern template gpusolverStatus_t orgqr<double>(gpusolverDnHandle_t, int, int, int, double *, int,
                                                const double *, double *, int, int *);

// Function: ungqr
extern template gpusolverStatus_t ungqr<gpuFloatComplex>(gpusolverDnHandle_t, int, int, int,
                                                         gpuFloatComplex *, int,
                                                         const gpuFloatComplex *, gpuFloatComplex *,
                                                         int, int *);
extern template gpusolverStatus_t ungqr<gpuDoubleComplex>(gpusolverDnHandle_t, int, int, int,
                                                          gpuDoubleComplex *, int,
                                                          const gpuDoubleComplex *,
                                                          gpuDoubleComplex *, int, int *);

// Function: orgbr_bufferSize
extern template gpusolverStatus_t orgbr_bufferSize<float>(gpusolverDnHandle_t, gpublasSideMode_t,
                                                          int, int, int, const float *, int,
                                                          const float *, int *);
extern template gpusolverStatus_t orgbr_bufferSize<double>(gpusolverDnHandle_t, gpublasSideMode_t,
                                                           int, int, int, const double *, int,
                                                           const double *, int *);

// Function: ungbr_bufferSize
extern template gpusolverStatus_t
ungbr_bufferSize<gpuFloatComplex>(gpusolverDnHandle_t, gpublasSideMode_t, int, int, int,
                                  const gpuFloatComplex *, int, const gpuFloatComplex *, int *);
extern template gpusolverStatus_t
ungbr_bufferSize<gpuDoubleComplex>(gpusolverDnHandle_t, gpublasSideMode_t, int, int, int,
                                   const gpuDoubleComplex *, int, const gpuDoubleComplex *, int *);

// Function: orgbr
extern template gpusolverStatus_t orgbr<float>(gpusolverDnHandle_t, gpublasSideMode_t, int, int,
                                               int, float *, int, const float *, float *, int,
                                               int *);
extern template gpusolverStatus_t orgbr<double>(gpusolverDnHandle_t, gpublasSideMode_t, int, int,
                                                int, double *, int, const double *, double *, int,
                                                int *);

// Function: ungbr
extern template gpusolverStatus_t ungbr<gpuFloatComplex>(gpusolverDnHandle_t, gpublasSideMode_t,
                                                         int, int, int, gpuFloatComplex *, int,
                                                         const gpuFloatComplex *, gpuFloatComplex *,
                                                         int, int *);
extern template gpusolverStatus_t ungbr<gpuDoubleComplex>(gpusolverDnHandle_t, gpublasSideMode_t,
                                                          int, int, int, gpuDoubleComplex *, int,
                                                          const gpuDoubleComplex *,
                                                          gpuDoubleComplex *, int, int *);

} // namespace wwr
