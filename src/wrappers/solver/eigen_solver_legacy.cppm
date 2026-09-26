/**
 * @file eigen_solver_legacy.cppm
 * @brief Legacy GPU solver eigenvalue/SVD solver API wrappers (int-based, pre-params API)
 *
 * Type-safe C++ wrappers for the legacy cuSOLVER Dense / hipSOLVER Dense
 * eigenvalue/SVD API -- 48 functions, all shared between both backends.
 *
 * There is no modern (:eigen_solver) counterpart partition, because hipsolverDn
 * has no generic X-prefixed eigenvalue/SVD API at all; this legacy-typed API is
 * the only eigenvalue/SVD surface here. The modern-only cuSOLVER functions
 * (Xgeev, Xgesvd, Xgesvdp, Xgesvdr, Xsyevd, Xsyevdx, XsyevBatched) are not
 * wrapped -- call them through gpumod.cuda.cusolverDn.
 *
 * Legacy API characteristics:
 * - int for all dimensions, not int64_t
 * - no gpusolverDnParams_t parameter
 * - a single workspace buffer, not separate device/host
 * - for the complex eigenvalue solvers (heevd), eigenvalues are real-valued
 *
 * The two Jacobi info types are opaque handles created and destroyed through
 * gpusolverDnCreateSyevjInfo / gpusolverDnCreateGesvdjInfo and passed through
 * unmodified; they alias each backend's own name (see src/solver.cppm).
 *
 * Usage:
 *   import gpumod.wrappers.solver;
 *   using namespace gpumod;
 *
 *   // Query workspace size
 *   int lwork;
 *   syevd_bufferSize<float>(handle, jobz, uplo, n, A, lda, W, &lwork);
 *
 *   // Allocate and compute
 *   float* workspace;
 *   gpuMalloc(&workspace, lwork * sizeof(float));
 *   syevd<float>(handle, jobz, uplo, n, A, lda, W, workspace, lwork, devInfo);
 */

module;

#include "dispatch_macros.h"

export module gpumod.wrappers.solver:eigen_solver_legacy;

import gpumod.solver;
import gpumod.blas;
import gpumod.complex;
import gpumod.wrappers.common;
import std;

export namespace gpumod {

// ========================================================================
// Singular Value Decomposition (gesvd)
// ========================================================================

/**
 * @brief Query workspace size for singular value decomposition
 *
 * This function computes the size of the workspace buffer required for the
 * SVD operation.
 *
 * @tparam T Data type (float, double, gpuFloatComplex, gpuDoubleComplex)
 * @param handle cuSOLVER handle
 * @param m Number of rows of matrix A
 * @param n Number of columns of matrix A
 * @param lwork Output: required workspace size in elements of type T
 * @return gpusolverStatus_t status code
 */
template<usual_fp T>
gpusolverStatus_t gesvd_bufferSize(gpusolverDnHandle_t handle, int m, int n, int *lwork) {
  GPUMOD_REAL_DISPATCH(T, gpusolverDn, S, D, gesvd_bufferSize, handle, m, n, lwork);
  GPUMOD_COMPLEX_DISPATCH(T, gpusolverDn, C, Z, gesvd_bufferSize, handle, m, n, lwork);
}

/**
 * @brief Compute singular value decomposition A = U * Sigma * V^T
 *
 * Computes the singular value decomposition of a general m×n matrix A.
 * The diagonal matrix Sigma contains the singular values in descending order.
 *
 * @tparam T Data type (float, double, gpuFloatComplex, gpuDoubleComplex)
 * @param handle cuSOLVER handle
 * @param jobu Specifies options for computing U:
 *             'A': all m columns of U are returned in array U
 *             'S': first min(m,n) columns of U are returned in array U
 *             'O': first min(m,n) columns of U overwrite array A
 *             'N': no columns of U are computed
 * @param jobvt Specifies options for computing V^T:
 *              'A': all n rows of V^T are returned in array VT
 *              'S': first min(m,n) rows of V^T are returned in array VT
 *              'O': first min(m,n) rows of V^T overwrite array A
 *              'N': no rows of V^T are computed
 * @param m Number of rows of matrix A (m >= 0)
 * @param n Number of columns of matrix A (n >= 0)
 * @param A Pointer to matrix A (device memory); overwritten on exit
 * @param lda Leading dimension of A (lda >= max(1, m))
 * @param S Output: singular values in descending order (real-valued) (min(m,n) elements, real-valued)
 * @param U Output: left singular vectors (device memory)
 * @param ldu Leading dimension of U
 * @param VT Output: right singular vectors transposed (device memory)
 * @param ldvt Leading dimension of VT
 * @param work Device workspace buffer (size from gesvd_bufferSize)
 * @param lwork Workspace size in elements of type T
 * @param rwork Additional real workspace for complex types (can be NULL for real types)
 * @param devInfo Device pointer: 0 on success, >0 if convergence failed
 * @return gpusolverStatus_t status code
 */
template<usual_fp T>
gpusolverStatus_t gesvd(gpusolverDnHandle_t handle, signed char jobu, signed char jobvt, int m,
                        int n, T *A, int lda, ComplexToRealType<T> *S, T *U, int ldu, T *VT,
                        int ldvt, T *work, int lwork, ComplexToRealType<T> *rwork, int *devInfo) {
  GPUMOD_REAL_DISPATCH(T, gpusolverDn, S, D, gesvd, handle, jobu, jobvt, m, n, A, lda, S, U, ldu,
                       VT, ldvt, work, lwork, rwork, devInfo);
  GPUMOD_COMPLEX_DISPATCH(T, gpusolverDn, C, Z, gesvd, handle, jobu, jobvt, m, n, A, lda, S, U, ldu,
                          VT, ldvt, work, lwork, rwork, devInfo);
}

// ========================================================================
// Symmetric Eigenvalue Decomposition (syevd) - Real types only
// ========================================================================

/**
 * @brief Query workspace size for symmetric eigenvalue decomposition (real types)
 *
 * This function computes the size of the workspace buffer required for the
 * symmetric eigenvalue decomposition operation.
 *
 * @tparam T Data type (float or double only)
 * @param handle cuSOLVER handle
 * @param jobz CUSOLVER_EIG_MODE_VECTOR: compute eigenvalues and eigenvectors
 *             CUSOLVER_EIG_MODE_NOVECTOR: compute eigenvalues only
 * @param uplo CUBLAS_FILL_MODE_UPPER: upper triangle of A is stored
 *             CUBLAS_FILL_MODE_LOWER: lower triangle of A is stored
 * @param n Order of matrix A (n >= 0)
 * @param A Pointer to matrix A (device memory)
 * @param lda Leading dimension of A (lda >= max(1, n))
 * @param W Pointer to eigenvalues array (device memory, n elements)
 * @param lwork Output: required workspace size in elements of type T
 * @return gpusolverStatus_t status code
 */
template<real_fp T>
gpusolverStatus_t syevd_bufferSize(gpusolverDnHandle_t handle, gpusolverEigMode_t jobz,
                                   gpublasFillMode_t uplo, int n, const T *A, int lda, const T *W,
                                   int *lwork) {
  GPUMOD_REAL_DISPATCH(T, gpusolverDn, S, D, syevd_bufferSize, handle, jobz, uplo, n, A, lda, W,
                       lwork);
}

/**
 * @brief Compute symmetric eigenvalue decomposition (real types)
 *
 * Computes all eigenvalues and, optionally, eigenvectors of a real symmetric matrix A.
 * The eigenvalue decomposition is: A = Q * Lambda * Q^T
 * where Lambda is a diagonal matrix of eigenvalues and Q is orthogonal.
 *
 * @tparam T Data type (float or double only)
 * @param handle cuSOLVER handle
 * @param jobz CUSOLVER_EIG_MODE_VECTOR: compute eigenvalues and eigenvectors
 *             CUSOLVER_EIG_MODE_NOVECTOR: compute eigenvalues only
 * @param uplo CUBLAS_FILL_MODE_UPPER: upper triangle of A is stored
 *             CUBLAS_FILL_MODE_LOWER: lower triangle of A is stored
 * @param n Order of matrix A (n >= 0)
 * @param A Pointer to matrix A (device memory); on exit, contains eigenvectors if jobz=VECTOR
 * @param lda Leading dimension of A (lda >= max(1, n))
 * @param W Output: eigenvalues in ascending order (device memory, n elements)
 * @param work Device workspace buffer (size from syevd_bufferSize)
 * @param lwork Workspace size in elements of type T
 * @param devInfo Device pointer: 0 on success, i if i-th parameter is invalid, >n if convergence failed
 * @return gpusolverStatus_t status code
 */
template<real_fp T>
gpusolverStatus_t syevd(gpusolverDnHandle_t handle, gpusolverEigMode_t jobz, gpublasFillMode_t uplo,
                        int n, T *A, int lda, T *W, T *work, int lwork, int *devInfo) {
  GPUMOD_REAL_DISPATCH(T, gpusolverDn, S, D, syevd, handle, jobz, uplo, n, A, lda, W, work, lwork,
                       devInfo);
}

// ========================================================================
// Symmetric Eigenvalue Decomposition with Subset Selection (syevdx) - Real types only
// ========================================================================

/**
 * @brief Query workspace size for symmetric eigenvalue decomposition with subset selection (real types)
 *
 * This function computes the size of the workspace buffer required for the
 * syevdx operation, which can compute a subset of eigenvalues/eigenvectors.
 *
 * @tparam T Data type (float or double only)
 * @param handle cuSOLVER handle
 * @param jobz CUSOLVER_EIG_MODE_VECTOR: compute eigenvalues and eigenvectors
 *             CUSOLVER_EIG_MODE_NOVECTOR: compute eigenvalues only
 * @param range CUSOLVER_EIG_RANGE_ALL: all eigenvalues
 *              CUSOLVER_EIG_RANGE_V: eigenvalues in (vl, vu]
 *              CUSOLVER_EIG_RANGE_I: il-th through iu-th eigenvalues
 * @param uplo CUBLAS_FILL_MODE_UPPER: upper triangle of A is stored
 *             CUBLAS_FILL_MODE_LOWER: lower triangle of A is stored
 * @param n Order of matrix A (n >= 0)
 * @param A Pointer to matrix A (device memory)
 * @param lda Leading dimension of A (lda >= max(1, n))
 * @param vl Lower bound of eigenvalue range (if range = CUSOLVER_EIG_RANGE_V)
 * @param vu Upper bound of eigenvalue range (if range = CUSOLVER_EIG_RANGE_V)
 * @param il Lower index of eigenvalue range (if range = CUSOLVER_EIG_RANGE_I, 1-indexed)
 * @param iu Upper index of eigenvalue range (if range = CUSOLVER_EIG_RANGE_I, 1-indexed)
 * @param meig Output: number of eigenvalues found (host or device pointer)
 * @param W Pointer to eigenvalues array (device memory)
 * @param lwork Output: required workspace size in elements of type T
 * @return gpusolverStatus_t status code
 */
template<real_fp T>
gpusolverStatus_t syevdx_bufferSize(gpusolverDnHandle_t handle, gpusolverEigMode_t jobz,
                                    gpusolverEigRange_t range, gpublasFillMode_t uplo, int n,
                                    const T *A, int lda, T vl, T vu, int il, int iu, int *meig,
                                    const T *W, int *lwork) {
  GPUMOD_REAL_DISPATCH(T, gpusolverDn, S, D, syevdx_bufferSize, handle, jobz, range, uplo, n, A,
                       lda, vl, vu, il, iu, meig, W, lwork);
}

/**
 * @brief Compute symmetric eigenvalue decomposition with subset selection (real types)
 *
 * Computes selected eigenvalues and, optionally, eigenvectors of a real symmetric matrix A.
 * Can compute all eigenvalues, eigenvalues in a range, or the il-th through iu-th eigenvalues.
 *
 * @tparam T Data type (float or double only)
 * @param handle cuSOLVER handle
 * @param jobz CUSOLVER_EIG_MODE_VECTOR: compute eigenvalues and eigenvectors
 *             CUSOLVER_EIG_MODE_NOVECTOR: compute eigenvalues only
 * @param range CUSOLVER_EIG_RANGE_ALL: all eigenvalues
 *              CUSOLVER_EIG_RANGE_V: eigenvalues in (vl, vu]
 *              CUSOLVER_EIG_RANGE_I: il-th through iu-th eigenvalues
 * @param uplo CUBLAS_FILL_MODE_UPPER: upper triangle of A is stored
 *             CUBLAS_FILL_MODE_LOWER: lower triangle of A is stored
 * @param n Order of matrix A (n >= 0)
 * @param A Pointer to matrix A (device memory); on exit, may contain eigenvectors if jobz=VECTOR
 * @param lda Leading dimension of A (lda >= max(1, n))
 * @param vl Lower bound of eigenvalue range (if range = CUSOLVER_EIG_RANGE_V)
 * @param vu Upper bound of eigenvalue range (if range = CUSOLVER_EIG_RANGE_V, vu > vl)
 * @param il Lower index of eigenvalue range (if range = CUSOLVER_EIG_RANGE_I, 1-indexed, 1 <= il <= iu <= n)
 * @param iu Upper index of eigenvalue range (if range = CUSOLVER_EIG_RANGE_I, 1-indexed, 1 <= il <= iu <= n)
 * @param meig Output: number of eigenvalues found (host or device pointer)
 * @param W Output: selected eigenvalues in ascending order (device memory, meig elements)
 * @param work Device workspace buffer (size from syevdx_bufferSize)
 * @param lwork Workspace size in elements of type T
 * @param info Device pointer: 0 on success, i if i-th parameter is invalid
 * @return gpusolverStatus_t status code
 */
template<real_fp T>
gpusolverStatus_t syevdx(gpusolverDnHandle_t handle, gpusolverEigMode_t jobz,
                         gpusolverEigRange_t range, gpublasFillMode_t uplo, int n, T *A, int lda,
                         T vl, T vu, int il, int iu, int *meig, T *W, T *work, int lwork,
                         int *info) {
  GPUMOD_REAL_DISPATCH(T, gpusolverDn, S, D, syevdx, handle, jobz, range, uplo, n, A, lda, vl, vu,
                       il, iu, meig, W, work, lwork, info);
}

// ========================================================================
// Hermitian Eigenvalue Decomposition (heevd) - Complex types only
// ========================================================================

/**
 * @brief Query workspace size for Hermitian eigenvalue decomposition (complex types)
 *
 * This function computes the size of the workspace buffer required for the
 * Hermitian eigenvalue decomposition operation.
 *
 * @tparam T Data type (gpuFloatComplex or gpuDoubleComplex only)
 * @param handle cuSOLVER handle
 * @param jobz CUSOLVER_EIG_MODE_VECTOR: compute eigenvalues and eigenvectors
 *             CUSOLVER_EIG_MODE_NOVECTOR: compute eigenvalues only
 * @param uplo CUBLAS_FILL_MODE_UPPER: upper triangle of A is stored
 *             CUBLAS_FILL_MODE_LOWER: lower triangle of A is stored
 * @param n Order of matrix A (n >= 0)
 * @param A Pointer to matrix A (device memory)
 * @param lda Leading dimension of A (lda >= max(1, n))
 * @param W Pointer to eigenvalues array (device memory, n elements, real-valued)
 * @param lwork Output: required workspace size in elements of type T
 * @return gpusolverStatus_t status code
 *
 * @note For complex matrices, the eigenvalues W are real (float for gpuFloatComplex, double for gpuDoubleComplex)
 */
template<complex_fp T>
gpusolverStatus_t heevd_bufferSize(gpusolverDnHandle_t handle, gpusolverEigMode_t jobz,
                                   gpublasFillMode_t uplo, int n, const T *A, int lda,
                                   const ComplexToRealType<T> *W, int *lwork) {
  GPUMOD_COMPLEX_DISPATCH(T, gpusolverDn, C, Z, heevd_bufferSize, handle, jobz, uplo, n, A, lda, W,
                          lwork);
}

/**
 * @brief Compute Hermitian eigenvalue decomposition (complex types)
 *
 * Computes all eigenvalues and, optionally, eigenvectors of a complex Hermitian matrix A.
 * The eigenvalue decomposition is: A = Q * Lambda * Q^H
 * where Lambda is a diagonal matrix of real eigenvalues and Q is unitary.
 *
 * @tparam T Data type (gpuFloatComplex or gpuDoubleComplex only)
 * @param handle cuSOLVER handle
 * @param jobz CUSOLVER_EIG_MODE_VECTOR: compute eigenvalues and eigenvectors
 *             CUSOLVER_EIG_MODE_NOVECTOR: compute eigenvalues only
 * @param uplo CUBLAS_FILL_MODE_UPPER: upper triangle of A is stored
 *             CUBLAS_FILL_MODE_LOWER: lower triangle of A is stored
 * @param n Order of matrix A (n >= 0)
 * @param A Pointer to matrix A (device memory); on exit, contains eigenvectors if jobz=VECTOR
 * @param lda Leading dimension of A (lda >= max(1, n))
 * @param W Output: eigenvalues in ascending order (device memory, n elements, real-valued)
 * @param work Device workspace buffer (size from heevd_bufferSize)
 * @param lwork Workspace size in elements of type T
 * @param devInfo Device pointer: 0 on success, i if i-th parameter is invalid, >n if convergence failed
 * @return gpusolverStatus_t status code
 *
 * @note For complex matrices, the eigenvalues W are real (float for gpuFloatComplex, double for gpuDoubleComplex)
 */
template<complex_fp T>
gpusolverStatus_t heevd(gpusolverDnHandle_t handle, gpusolverEigMode_t jobz, gpublasFillMode_t uplo,
                        int n, T *A, int lda, ComplexToRealType<T> *W, T *work, int lwork,
                        int *devInfo) {
  GPUMOD_COMPLEX_DISPATCH(T, gpusolverDn, C, Z, heevd, handle, jobz, uplo, n, A, lda, W, work,
                          lwork, devInfo);
}

// ========================================================================
// Hermitian Eigenvalue Decomposition with Subset Selection (heevdx) - Complex types only
// ========================================================================

/**
 * @brief Query workspace size for Hermitian eigenvalue decomposition with subset selection (complex types)
 *
 * This function computes the size of the workspace buffer required for the
 * heevdx operation, which can compute a subset of eigenvalues/eigenvectors.
 *
 * @tparam T Data type (gpuFloatComplex or gpuDoubleComplex only)
 * @param handle cuSOLVER handle
 * @param jobz CUSOLVER_EIG_MODE_VECTOR: compute eigenvalues and eigenvectors
 *             CUSOLVER_EIG_MODE_NOVECTOR: compute eigenvalues only
 * @param range CUSOLVER_EIG_RANGE_ALL: all eigenvalues
 *              CUSOLVER_EIG_RANGE_V: eigenvalues in (vl, vu]
 *              CUSOLVER_EIG_RANGE_I: il-th through iu-th eigenvalues
 * @param uplo CUBLAS_FILL_MODE_UPPER: upper triangle of A is stored
 *             CUBLAS_FILL_MODE_LOWER: lower triangle of A is stored
 * @param n Order of matrix A (n >= 0)
 * @param A Pointer to matrix A (device memory)
 * @param lda Leading dimension of A (lda >= max(1, n))
 * @param vl Lower bound of eigenvalue range (if range = CUSOLVER_EIG_RANGE_V, real-valued)
 * @param vu Upper bound of eigenvalue range (if range = CUSOLVER_EIG_RANGE_V, real-valued)
 * @param il Lower index of eigenvalue range (if range = CUSOLVER_EIG_RANGE_I, 1-indexed)
 * @param iu Upper index of eigenvalue range (if range = CUSOLVER_EIG_RANGE_I, 1-indexed)
 * @param meig Output: number of eigenvalues found (host or device pointer)
 * @param W Pointer to eigenvalues array (device memory, real-valued)
 * @param lwork Output: required workspace size in elements of type T
 * @return gpusolverStatus_t status code
 *
 * @note For complex matrices, the eigenvalues W are real (float for gpuFloatComplex, double for gpuDoubleComplex)
 */
template<complex_fp T>
gpusolverStatus_t heevdx_bufferSize(gpusolverDnHandle_t handle, gpusolverEigMode_t jobz,
                                    gpusolverEigRange_t range, gpublasFillMode_t uplo, int n,
                                    const T *A, int lda, ComplexToRealType<T> vl,
                                    ComplexToRealType<T> vu, int il, int iu, int *meig,
                                    const ComplexToRealType<T> *W, int *lwork) {
  GPUMOD_COMPLEX_DISPATCH(T, gpusolverDn, C, Z, heevdx_bufferSize, handle, jobz, range, uplo, n, A,
                          lda, vl, vu, il, iu, meig, W, lwork);
}

/**
 * @brief Compute Hermitian eigenvalue decomposition with subset selection (complex types)
 *
 * Computes selected eigenvalues and, optionally, eigenvectors of a complex Hermitian matrix A.
 * Can compute all eigenvalues, eigenvalues in a range, or the il-th through iu-th eigenvalues.
 *
 * @tparam T Data type (gpuFloatComplex or gpuDoubleComplex only)
 * @param handle cuSOLVER handle
 * @param jobz CUSOLVER_EIG_MODE_VECTOR: compute eigenvalues and eigenvectors
 *             CUSOLVER_EIG_MODE_NOVECTOR: compute eigenvalues only
 * @param range CUSOLVER_EIG_RANGE_ALL: all eigenvalues
 *              CUSOLVER_EIG_RANGE_V: eigenvalues in (vl, vu]
 *              CUSOLVER_EIG_RANGE_I: il-th through iu-th eigenvalues
 * @param uplo CUBLAS_FILL_MODE_UPPER: upper triangle of A is stored
 *             CUBLAS_FILL_MODE_LOWER: lower triangle of A is stored
 * @param n Order of matrix A (n >= 0)
 * @param A Pointer to matrix A (device memory); on exit, may contain eigenvectors if jobz=VECTOR
 * @param lda Leading dimension of A (lda >= max(1, n))
 * @param vl Lower bound of eigenvalue range (if range = CUSOLVER_EIG_RANGE_V, real-valued)
 * @param vu Upper bound of eigenvalue range (if range = CUSOLVER_EIG_RANGE_V, real-valued, vu > vl)
 * @param il Lower index of eigenvalue range (if range = CUSOLVER_EIG_RANGE_I, 1-indexed, 1 <= il <= iu <= n)
 * @param iu Upper index of eigenvalue range (if range = CUSOLVER_EIG_RANGE_I, 1-indexed, 1 <= il <= iu <= n)
 * @param meig Output: number of eigenvalues found (host or device pointer)
 * @param W Output: selected eigenvalues in ascending order (device memory, meig elements, real-valued)
 * @param work Device workspace buffer (size from heevdx_bufferSize)
 * @param lwork Workspace size in elements of type T
 * @param info Device pointer: 0 on success, i if i-th parameter is invalid
 * @return gpusolverStatus_t status code
 *
 * @note For complex matrices, the eigenvalues W are real (float for gpuFloatComplex, double for gpuDoubleComplex)
 */
template<complex_fp T>
gpusolverStatus_t heevdx(gpusolverDnHandle_t handle, gpusolverEigMode_t jobz,
                         gpusolverEigRange_t range, gpublasFillMode_t uplo, int n, T *A, int lda,
                         ComplexToRealType<T> vl, ComplexToRealType<T> vu, int il, int iu,
                         int *meig, ComplexToRealType<T> *W, T *work, int lwork, int *info) {
  GPUMOD_COMPLEX_DISPATCH(T, gpusolverDn, C, Z, heevdx, handle, jobz, range, uplo, n, A, lda, vl,
                          vu, il, iu, meig, W, work, lwork, info);
}

// ========================================================================
// Tridiagonal Reduction (sytrd/hetrd)
// ========================================================================

/**
 * @brief Query workspace size for symmetric tridiagonal reduction (real types)
 *
 * This function computes the size of the workspace buffer required for the
 * symmetric tridiagonal reduction operation.
 *
 * @tparam T Data type (float or double only)
 * @param handle cuSOLVER handle
 * @param uplo CUBLAS_FILL_MODE_UPPER: upper triangle of A is stored
 *             CUBLAS_FILL_MODE_LOWER: lower triangle of A is stored
 * @param n Order of matrix A (n >= 0)
 * @param A Pointer to matrix A (device memory)
 * @param lda Leading dimension of A (lda >= max(1, n))
 * @param D Diagonal elements (device memory, n elements)
 * @param E Off-diagonal elements (device memory, n-1 elements)
 * @param tau Elementary reflectors (device memory, n-1 elements)
 * @param lwork Output: required workspace size in elements of type T
 * @return gpusolverStatus_t status code
 */
template<real_fp T>
gpusolverStatus_t sytrd_bufferSize(gpusolverDnHandle_t handle, gpublasFillMode_t uplo, int n,
                                   const T *A, int lda, const T *D, const T *E, const T *tau,
                                   int *lwork) {
  GPUMOD_REAL_DISPATCH(T, gpusolverDn, S, D, sytrd_bufferSize, handle, uplo, n, A, lda, D, E, tau,
                       lwork);
}

/**
 * @brief Reduce symmetric matrix to tridiagonal form (real types)
 *
 * Reduces a real symmetric matrix A to real symmetric tridiagonal form T
 * by an orthogonal similarity transformation: Q^T * A * Q = T
 *
 * @tparam T Data type (float or double only)
 * @param handle cuSOLVER handle
 * @param uplo CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER
 * @param n Order of matrix A (n >= 0)
 * @param A Pointer to matrix A (device memory); overwritten on exit
 * @param lda Leading dimension of A (lda >= max(1, n))
 * @param D Output: diagonal elements of T (device memory, n elements)
 * @param E Output: off-diagonal elements of T (device memory, n-1 elements)
 * @param tau Output: scalar factors of elementary reflectors (device memory, n-1 elements)
 * @param work Device workspace buffer (size from sytrd_bufferSize)
 * @param lwork Workspace size in elements of type T
 * @param devInfo Device pointer: 0 on success
 * @return gpusolverStatus_t status code
 */
template<real_fp T>
gpusolverStatus_t sytrd(gpusolverDnHandle_t handle, gpublasFillMode_t uplo, int n, T *A, int lda,
                        T *D, T *E, T *tau, T *work, int lwork, int *devInfo) {
  GPUMOD_REAL_DISPATCH(T, gpusolverDn, S, D, sytrd, handle, uplo, n, A, lda, D, E, tau, work, lwork,
                       devInfo);
}

/**
 * @brief Query workspace size for Hermitian tridiagonal reduction (complex types)
 *
 * This function computes the size of the workspace buffer required for the
 * Hermitian tridiagonal reduction operation.
 *
 * @tparam T Data type (gpuFloatComplex or gpuDoubleComplex only)
 * @param handle cuSOLVER handle
 * @param uplo CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER
 * @param n Order of matrix A (n >= 0)
 * @param A Pointer to matrix A (device memory)
 * @param lda Leading dimension of A (lda >= max(1, n))
 * @param D Diagonal elements (device memory, n elements, real-valued)
 * @param E Off-diagonal elements (device memory, n-1 elements, real-valued)
 * @param tau Elementary reflectors (device memory, n-1 elements)
 * @param lwork Output: required workspace size in elements of type T
 * @return gpusolverStatus_t status code
 */
template<complex_fp T>
gpusolverStatus_t hetrd_bufferSize(gpusolverDnHandle_t handle, gpublasFillMode_t uplo, int n,
                                   const T *A, int lda, const ComplexToRealType<T> *D,
                                   const ComplexToRealType<T> *E, const T *tau, int *lwork) {
  GPUMOD_COMPLEX_DISPATCH(T, gpusolverDn, C, Z, hetrd_bufferSize, handle, uplo, n, A, lda, D, E,
                          tau, lwork);
}

/**
 * @brief Reduce Hermitian matrix to tridiagonal form (complex types)
 *
 * Reduces a complex Hermitian matrix A to real symmetric tridiagonal form T
 * by a unitary similarity transformation: Q^H * A * Q = T
 *
 * @tparam T Data type (gpuFloatComplex or gpuDoubleComplex only)
 * @param handle cuSOLVER handle
 * @param uplo CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER
 * @param n Order of matrix A (n >= 0)
 * @param A Pointer to matrix A (device memory); overwritten on exit
 * @param lda Leading dimension of A (lda >= max(1, n))
 * @param D Output: diagonal elements of T (device memory, n elements, real-valued)
 * @param E Output: off-diagonal elements of T (device memory, n-1 elements, real-valued)
 * @param tau Output: scalar factors of elementary reflectors (device memory, n-1 elements)
 * @param work Device workspace buffer (size from hetrd_bufferSize)
 * @param lwork Workspace size in elements of type T
 * @param devInfo Device pointer: 0 on success
 * @return gpusolverStatus_t status code
 */
template<complex_fp T>
gpusolverStatus_t hetrd(gpusolverDnHandle_t handle, gpublasFillMode_t uplo, int n, T *A, int lda,
                        ComplexToRealType<T> *D, ComplexToRealType<T> *E, T *tau, T *work,
                        int lwork, int *devInfo) {
  GPUMOD_COMPLEX_DISPATCH(T, gpusolverDn, C, Z, hetrd, handle, uplo, n, A, lda, D, E, tau, work,
                          lwork, devInfo);
}

// ========================================================================
// Generate Orthogonal/Unitary Matrix from Tridiagonal (orgtr/ungtr)
// ========================================================================

/**
 * @brief Query workspace size for generating orthogonal matrix from tridiagonal (real types)
 *
 * @tparam T Data type (float or double only)
 * @param handle cuSOLVER handle
 * @param uplo CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER
 * @param n Order of matrix (n >= 0)
 * @param A Pointer to matrix from sytrd (device memory)
 * @param lda Leading dimension of A
 * @param tau Elementary reflectors from sytrd (device memory)
 * @param lwork Output: required workspace size
 * @return gpusolverStatus_t status code
 */
template<real_fp T>
gpusolverStatus_t orgtr_bufferSize(gpusolverDnHandle_t handle, gpublasFillMode_t uplo, int n,
                                   const T *A, int lda, const T *tau, int *lwork) {
  GPUMOD_REAL_DISPATCH(T, gpusolverDn, S, D, orgtr_bufferSize, handle, uplo, n, A, lda, tau, lwork);
}

/**
 * @brief Generate orthogonal matrix from tridiagonal reduction (real types)
 *
 * Generates the real orthogonal matrix Q from sytrd that reduced
 * a symmetric matrix to tridiagonal form.
 *
 * @tparam T Data type (float or double only)
 * @param handle cuSOLVER handle
 * @param uplo CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER
 * @param n Order of matrix (n >= 0)
 * @param A Pointer to matrix from sytrd (device memory); overwritten with Q
 * @param lda Leading dimension of A
 * @param tau Elementary reflectors from sytrd (device memory)
 * @param work Device workspace buffer
 * @param lwork Workspace size
 * @param devInfo Device pointer: 0 on success
 * @return gpusolverStatus_t status code
 */
template<real_fp T>
gpusolverStatus_t orgtr(gpusolverDnHandle_t handle, gpublasFillMode_t uplo, int n, T *A, int lda,
                        const T *tau, T *work, int lwork, int *devInfo) {
  GPUMOD_REAL_DISPATCH(T, gpusolverDn, S, D, orgtr, handle, uplo, n, A, lda, tau, work, lwork,
                       devInfo);
}

/**
 * @brief Query workspace size for generating unitary matrix from tridiagonal (complex types)
 *
 * @tparam T Data type (gpuFloatComplex or gpuDoubleComplex only)
 * @param handle cuSOLVER handle
 * @param uplo CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER
 * @param n Order of matrix (n >= 0)
 * @param A Pointer to matrix from hetrd (device memory)
 * @param lda Leading dimension of A
 * @param tau Elementary reflectors from hetrd (device memory)
 * @param lwork Output: required workspace size
 * @return gpusolverStatus_t status code
 */
template<complex_fp T>
gpusolverStatus_t ungtr_bufferSize(gpusolverDnHandle_t handle, gpublasFillMode_t uplo, int n,
                                   const T *A, int lda, const T *tau, int *lwork) {
  GPUMOD_COMPLEX_DISPATCH(T, gpusolverDn, C, Z, ungtr_bufferSize, handle, uplo, n, A, lda, tau,
                          lwork);
}

/**
 * @brief Generate unitary matrix from tridiagonal reduction (complex types)
 *
 * Generates the complex unitary matrix Q from hetrd that reduced
 * a Hermitian matrix to tridiagonal form.
 *
 * @tparam T Data type (gpuFloatComplex or gpuDoubleComplex only)
 * @param handle cuSOLVER handle
 * @param uplo CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER
 * @param n Order of matrix (n >= 0)
 * @param A Pointer to matrix from hetrd (device memory); overwritten with Q
 * @param lda Leading dimension of A
 * @param tau Elementary reflectors from hetrd (device memory)
 * @param work Device workspace buffer
 * @param lwork Workspace size
 * @param devInfo Device pointer: 0 on success
 * @return gpusolverStatus_t status code
 */
template<complex_fp T>
gpusolverStatus_t ungtr(gpusolverDnHandle_t handle, gpublasFillMode_t uplo, int n, T *A, int lda,
                        const T *tau, T *work, int lwork, int *devInfo) {
  GPUMOD_COMPLEX_DISPATCH(T, gpusolverDn, C, Z, ungtr, handle, uplo, n, A, lda, tau, work, lwork,
                          devInfo);
}

// ========================================================================
// Apply Orthogonal/Unitary Matrix from Tridiagonal (ormtr/unmtr)
// ========================================================================

/**
 * @brief Query workspace size for applying orthogonal matrix from tridiagonal (real types)
 *
 * @tparam T Data type (float or double only)
 * @param handle cuSOLVER handle
 * @param side CUBLAS_SIDE_LEFT or CUBLAS_SIDE_RIGHT
 * @param uplo CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER
 * @param trans CUBLAS_OP_N or CUBLAS_OP_T
 * @param m Number of rows of matrix C
 * @param n Number of columns of matrix C
 * @param A Pointer to matrix from sytrd (device memory)
 * @param lda Leading dimension of A
 * @param tau Elementary reflectors from sytrd (device memory)
 * @param C Pointer to matrix C (device memory)
 * @param ldc Leading dimension of C
 * @param lwork Output: required workspace size
 * @return gpusolverStatus_t status code
 */
template<real_fp T>
gpusolverStatus_t ormtr_bufferSize(gpusolverDnHandle_t handle, gpublasSideMode_t side,
                                   gpublasFillMode_t uplo, gpublasOperation_t trans, int m, int n,
                                   const T *A, int lda, const T *tau, const T *C, int ldc,
                                   int *lwork) {
  GPUMOD_REAL_DISPATCH(T, gpusolverDn, S, D, ormtr_bufferSize, handle, side, uplo, trans, m, n, A,
                       lda, tau, C, ldc, lwork);
}

/**
 * @brief Apply orthogonal matrix from tridiagonal reduction (real types)
 *
 * Multiplies a real matrix C by the orthogonal matrix Q from sytrd.
 *
 * @tparam T Data type (float or double only)
 * @param handle cuSOLVER handle
 * @param side CUBLAS_SIDE_LEFT (Q*C) or CUBLAS_SIDE_RIGHT (C*Q)
 * @param uplo CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER
 * @param trans CUBLAS_OP_N (Q) or CUBLAS_OP_T (Q^T)
 * @param m Number of rows of matrix C
 * @param n Number of columns of matrix C
 * @param A Pointer to matrix from sytrd (device memory, not modified)
 * @param lda Leading dimension of A
 * @param tau Elementary reflectors from sytrd (device memory)
 * @param C Pointer to matrix C (device memory); overwritten with result
 * @param ldc Leading dimension of C
 * @param work Device workspace buffer
 * @param lwork Workspace size
 * @param devInfo Device pointer: 0 on success
 * @return gpusolverStatus_t status code
 */
template<real_fp T>
gpusolverStatus_t ormtr(gpusolverDnHandle_t handle, gpublasSideMode_t side, gpublasFillMode_t uplo,
                        gpublasOperation_t trans, int m, int n, T *A, int lda, T *tau, T *C,
                        int ldc, T *work, int lwork, int *devInfo) {
  GPUMOD_REAL_DISPATCH(T, gpusolverDn, S, D, ormtr, handle, side, uplo, trans, m, n, A, lda, tau, C,
                       ldc, work, lwork, devInfo);
}

/**
 * @brief Query workspace size for applying unitary matrix from tridiagonal (complex types)
 *
 * @tparam T Data type (gpuFloatComplex or gpuDoubleComplex only)
 * @param handle cuSOLVER handle
 * @param side CUBLAS_SIDE_LEFT or CUBLAS_SIDE_RIGHT
 * @param uplo CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER
 * @param trans CUBLAS_OP_N or CUBLAS_OP_C (Hermitian transpose)
 * @param m Number of rows of matrix C
 * @param n Number of columns of matrix C
 * @param A Pointer to matrix from hetrd (device memory)
 * @param lda Leading dimension of A
 * @param tau Elementary reflectors from hetrd (device memory)
 * @param C Pointer to matrix C (device memory)
 * @param ldc Leading dimension of C
 * @param lwork Output: required workspace size
 * @return gpusolverStatus_t status code
 */
template<complex_fp T>
gpusolverStatus_t unmtr_bufferSize(gpusolverDnHandle_t handle, gpublasSideMode_t side,
                                   gpublasFillMode_t uplo, gpublasOperation_t trans, int m, int n,
                                   const T *A, int lda, const T *tau, const T *C, int ldc,
                                   int *lwork) {
  GPUMOD_COMPLEX_DISPATCH(T, gpusolverDn, C, Z, unmtr_bufferSize, handle, side, uplo, trans, m, n,
                          A, lda, tau, C, ldc, lwork);
}

/**
 * @brief Apply unitary matrix from tridiagonal reduction (complex types)
 *
 * Multiplies a complex matrix C by the unitary matrix Q from hetrd.
 *
 * @tparam T Data type (gpuFloatComplex or gpuDoubleComplex only)
 * @param handle cuSOLVER handle
 * @param side CUBLAS_SIDE_LEFT (Q*C) or CUBLAS_SIDE_RIGHT (C*Q)
 * @param uplo CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER
 * @param trans CUBLAS_OP_N (Q) or CUBLAS_OP_C (Q^H)
 * @param m Number of rows of matrix C
 * @param n Number of columns of matrix C
 * @param A Pointer to matrix from hetrd (device memory, not modified)
 * @param lda Leading dimension of A
 * @param tau Elementary reflectors from hetrd (device memory)
 * @param C Pointer to matrix C (device memory); overwritten with result
 * @param ldc Leading dimension of C
 * @param work Device workspace buffer
 * @param lwork Workspace size
 * @param devInfo Device pointer: 0 on success
 * @return gpusolverStatus_t status code
 */
template<complex_fp T>
gpusolverStatus_t unmtr(gpusolverDnHandle_t handle, gpublasSideMode_t side, gpublasFillMode_t uplo,
                        gpublasOperation_t trans, int m, int n, T *A, int lda, T *tau, T *C,
                        int ldc, T *work, int lwork, int *devInfo) {
  GPUMOD_COMPLEX_DISPATCH(T, gpusolverDn, C, Z, unmtr, handle, side, uplo, trans, m, n, A, lda, tau,
                          C, ldc, work, lwork, devInfo);
}

// ========================================================================
// Jacobi-based Symmetric Eigenvalue Decomposition (syevj) - Real types only
// ========================================================================

/**
 * @brief Query workspace size for Jacobi symmetric eigenvalue decomposition (real types)
 *
 * @tparam T Data type (float or double only)
 * @param handle cuSOLVER handle
 * @param jobz CUSOLVER_EIG_MODE_VECTOR: compute eigenvalues and eigenvectors
 *             CUSOLVER_EIG_MODE_NOVECTOR: compute eigenvalues only
 * @param uplo CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER
 * @param n Order of matrix A (n >= 0)
 * @param A Pointer to matrix A (device memory)
 * @param lda Leading dimension of A (lda >= max(1, n))
 * @param W Pointer to eigenvalues array (device memory, n elements)
 * @param lwork Output: required workspace size in elements of type T
 * @param params Jacobi parameters (can be NULL for default)
 * @return gpusolverStatus_t status code
 */
template<real_fp T>
gpusolverStatus_t syevj_bufferSize(gpusolverDnHandle_t handle, gpusolverEigMode_t jobz,
                                   gpublasFillMode_t uplo, int n, const T *A, int lda, const T *W,
                                   int *lwork, gpusolverSyevjInfo_t params) {
  GPUMOD_REAL_DISPATCH(T, gpusolverDn, S, D, syevj_bufferSize, handle, jobz, uplo, n, A, lda, W,
                       lwork, params);
}

/**
 * @brief Compute symmetric eigenvalue decomposition using Jacobi method (real types)
 *
 * @tparam T Data type (float or double only)
 * @param handle cuSOLVER handle
 * @param jobz CUSOLVER_EIG_MODE_VECTOR: compute eigenvalues and eigenvectors
 * @param uplo CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER
 * @param n Order of matrix A (n >= 0)
 * @param A Pointer to matrix A (device memory); on exit, contains eigenvectors if jobz=VECTOR
 * @param lda Leading dimension of A (lda >= max(1, n))
 * @param W Output: eigenvalues in ascending order (device memory, n elements)
 * @param work Device workspace buffer (size from syevj_bufferSize)
 * @param lwork Workspace size in elements of type T
 * @param info Device pointer: 0 on success
 * @param params Jacobi parameters (can be NULL for default)
 * @return gpusolverStatus_t status code
 */
template<real_fp T>
gpusolverStatus_t syevj(gpusolverDnHandle_t handle, gpusolverEigMode_t jobz, gpublasFillMode_t uplo,
                        int n, T *A, int lda, T *W, T *work, int lwork, int *info,
                        gpusolverSyevjInfo_t params) {
  GPUMOD_REAL_DISPATCH(T, gpusolverDn, S, D, syevj, handle, jobz, uplo, n, A, lda, W, work, lwork,
                       info, params);
}

/**
 * @brief Query workspace size for batched Jacobi symmetric eigenvalue decomposition
 *
 * @tparam T Data type (float or double only)
 * @param handle cuSOLVER handle
 * @param jobz CUSOLVER_EIG_MODE_VECTOR or CUSOLVER_EIG_MODE_NOVECTOR
 * @param uplo CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER
 * @param n Order of each matrix (n >= 0)
 * @param A Pointer to array of matrices (device memory)
 * @param lda Leading dimension of each matrix
 * @param W Pointer to eigenvalues array (device memory)
 * @param lwork Output: required workspace size
 * @param params Jacobi parameters
 * @param batchSize Number of matrices in batch
 * @return gpusolverStatus_t status code
 */
template<real_fp T>
gpusolverStatus_t syevjBatched_bufferSize(gpusolverDnHandle_t handle, gpusolverEigMode_t jobz,
                                          gpublasFillMode_t uplo, int n, const T *A, int lda,
                                          const T *W, int *lwork, gpusolverSyevjInfo_t params,
                                          int batchSize) {
  GPUMOD_REAL_DISPATCH(T, gpusolverDn, S, D, syevjBatched_bufferSize, handle, jobz, uplo, n, A, lda,
                       W, lwork, params, batchSize);
}

/**
 * @brief Compute batched symmetric eigenvalue decomposition using Jacobi method
 *
 * @tparam T Data type (float or double only)
 * @param handle cuSOLVER handle
 * @param jobz CUSOLVER_EIG_MODE_VECTOR or CUSOLVER_EIG_MODE_NOVECTOR
 * @param uplo CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER
 * @param n Order of each matrix
 * @param A Pointer to array of matrices (device memory)
 * @param lda Leading dimension of each matrix
 * @param W Output: eigenvalues (device memory)
 * @param work Device workspace buffer
 * @param lwork Workspace size
 * @param info Device pointer: array of status codes
 * @param params Jacobi parameters
 * @param batchSize Number of matrices in batch
 * @return gpusolverStatus_t status code
 */
template<real_fp T>
gpusolverStatus_t syevjBatched(gpusolverDnHandle_t handle, gpusolverEigMode_t jobz,
                               gpublasFillMode_t uplo, int n, T *A, int lda, T *W, T *work,
                               int lwork, int *info, gpusolverSyevjInfo_t params, int batchSize) {
  GPUMOD_REAL_DISPATCH(T, gpusolverDn, S, D, syevjBatched, handle, jobz, uplo, n, A, lda, W, work,
                       lwork, info, params, batchSize);
}

// ========================================================================
// Jacobi-based Hermitian Eigenvalue Decomposition (heevj) - Complex types only
// ========================================================================

/**
 * @brief Query workspace size for Jacobi Hermitian eigenvalue decomposition
 *
 * @tparam T Data type (gpuFloatComplex or gpuDoubleComplex only)
 * @param handle cuSOLVER handle
 * @param jobz CUSOLVER_EIG_MODE_VECTOR or CUSOLVER_EIG_MODE_NOVECTOR
 * @param uplo CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER
 * @param n Order of matrix A
 * @param A Pointer to matrix A (device memory)
 * @param lda Leading dimension of A
 * @param W Pointer to eigenvalues array (real-valued)
 * @param lwork Output: required workspace size
 * @param params Jacobi parameters
 * @return gpusolverStatus_t status code
 */
template<complex_fp T>
gpusolverStatus_t heevj_bufferSize(gpusolverDnHandle_t handle, gpusolverEigMode_t jobz,
                                   gpublasFillMode_t uplo, int n, const T *A, int lda,
                                   const ComplexToRealType<T> *W, int *lwork,
                                   gpusolverSyevjInfo_t params) {
  GPUMOD_COMPLEX_DISPATCH(T, gpusolverDn, C, Z, heevj_bufferSize, handle, jobz, uplo, n, A, lda, W,
                          lwork, params);
}

/**
 * @brief Compute Hermitian eigenvalue decomposition using Jacobi method
 *
 * @tparam T Data type (gpuFloatComplex or gpuDoubleComplex only)
 * @param handle cuSOLVER handle
 * @param jobz CUSOLVER_EIG_MODE_VECTOR or CUSOLVER_EIG_MODE_NOVECTOR
 * @param uplo CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER
 * @param n Order of matrix A
 * @param A Pointer to matrix A (device memory); on exit, contains eigenvectors if jobz=VECTOR
 * @param lda Leading dimension of A
 * @param W Output: eigenvalues in ascending order (real-valued)
 * @param work Device workspace buffer
 * @param lwork Workspace size
 * @param info Device pointer: 0 on success
 * @param params Jacobi parameters
 * @return gpusolverStatus_t status code
 */
template<complex_fp T>
gpusolverStatus_t heevj(gpusolverDnHandle_t handle, gpusolverEigMode_t jobz, gpublasFillMode_t uplo,
                        int n, T *A, int lda, ComplexToRealType<T> *W, T *work, int lwork,
                        int *info, gpusolverSyevjInfo_t params) {
  GPUMOD_COMPLEX_DISPATCH(T, gpusolverDn, C, Z, heevj, handle, jobz, uplo, n, A, lda, W, work,
                          lwork, info, params);
}

/**
 * @brief Query workspace size for batched Jacobi Hermitian eigenvalue decomposition
 *
 * @tparam T Data type (gpuFloatComplex or gpuDoubleComplex only)
 * @param handle cuSOLVER handle
 * @param jobz CUSOLVER_EIG_MODE_VECTOR or CUSOLVER_EIG_MODE_NOVECTOR
 * @param uplo CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER
 * @param n Order of each matrix
 * @param A Pointer to array of matrices (device memory)
 * @param lda Leading dimension of each matrix
 * @param W Pointer to eigenvalues array (real-valued)
 * @param lwork Output: required workspace size
 * @param params Jacobi parameters
 * @param batchSize Number of matrices in batch
 * @return gpusolverStatus_t status code
 */
template<complex_fp T>
gpusolverStatus_t heevjBatched_bufferSize(gpusolverDnHandle_t handle, gpusolverEigMode_t jobz,
                                          gpublasFillMode_t uplo, int n, const T *A, int lda,
                                          const ComplexToRealType<T> *W, int *lwork,
                                          gpusolverSyevjInfo_t params, int batchSize) {
  GPUMOD_COMPLEX_DISPATCH(T, gpusolverDn, C, Z, heevjBatched_bufferSize, handle, jobz, uplo, n, A,
                          lda, W, lwork, params, batchSize);
}

/**
 * @brief Compute batched Hermitian eigenvalue decomposition using Jacobi method
 *
 * @tparam T Data type (gpuFloatComplex or gpuDoubleComplex only)
 * @param handle cuSOLVER handle
 * @param jobz CUSOLVER_EIG_MODE_VECTOR or CUSOLVER_EIG_MODE_NOVECTOR
 * @param uplo CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER
 * @param n Order of each matrix
 * @param A Pointer to array of matrices (device memory)
 * @param lda Leading dimension of each matrix
 * @param W Output: eigenvalues (real-valued)
 * @param work Device workspace buffer
 * @param lwork Workspace size
 * @param info Device pointer: array of status codes
 * @param params Jacobi parameters
 * @param batchSize Number of matrices in batch
 * @return gpusolverStatus_t status code
 */
template<complex_fp T>
gpusolverStatus_t heevjBatched(gpusolverDnHandle_t handle, gpusolverEigMode_t jobz,
                               gpublasFillMode_t uplo, int n, T *A, int lda,
                               ComplexToRealType<T> *W, T *work, int lwork, int *info,
                               gpusolverSyevjInfo_t params, int batchSize) {
  GPUMOD_COMPLEX_DISPATCH(T, gpusolverDn, C, Z, heevjBatched, handle, jobz, uplo, n, A, lda, W,
                          work, lwork, info, params, batchSize);
}

// ========================================================================
// Jacobi-based SVD (gesvdj)
// ========================================================================

/**
 * @brief Query workspace size for Jacobi SVD
 *
 * @tparam T Data type (float, double, gpuFloatComplex, gpuDoubleComplex)
 * @param handle cuSOLVER handle
 * @param jobz CUSOLVER_EIG_MODE_VECTOR: compute singular vectors
 * @param econ 0 for full SVD, 1 for economy-size SVD
 * @param m Number of rows of matrix A
 * @param n Number of columns of matrix A
 * @param A Pointer to matrix A (device memory)
 * @param lda Leading dimension of A
 * @param S Pointer to singular values array
 * @param U Pointer to left singular vectors (device memory)
 * @param ldu Leading dimension of U
 * @param V Pointer to right singular vectors (device memory)
 * @param ldv Leading dimension of V
 * @param lwork Output: required workspace size
 * @param params Jacobi parameters
 * @return gpusolverStatus_t status code
 */
template<usual_fp T>
gpusolverStatus_t gesvdj_bufferSize(gpusolverDnHandle_t handle, gpusolverEigMode_t jobz, int econ,
                                    int m, int n, const T *A, int lda,
                                    const ComplexToRealType<T> *S, const T *U, int ldu, const T *V,
                                    int ldv, int *lwork, gpusolverGesvdjInfo_t params) {
  GPUMOD_REAL_DISPATCH(T, gpusolverDn, S, D, gesvdj_bufferSize, handle, jobz, econ, m, n, A, lda, S,
                       U, ldu, V, ldv, lwork, params);
  GPUMOD_COMPLEX_DISPATCH(T, gpusolverDn, C, Z, gesvdj_bufferSize, handle, jobz, econ, m, n, A, lda,
                          S, U, ldu, V, ldv, lwork, params);
}

/**
 * @brief Compute SVD using Jacobi method
 *
 * @tparam T Data type (float, double, gpuFloatComplex, gpuDoubleComplex)
 * @param handle cuSOLVER handle
 * @param jobz CUSOLVER_EIG_MODE_VECTOR: compute singular vectors
 * @param econ 0 for full SVD, 1 for economy-size SVD
 * @param m Number of rows of matrix A
 * @param n Number of columns of matrix A
 * @param A Pointer to matrix A (device memory)
 * @param lda Leading dimension of A
 * @param S Output: singular values in descending order (real-valued)
 * @param U Output: left singular vectors (device memory)
 * @param ldu Leading dimension of U
 * @param V Output: right singular vectors (device memory)
 * @param ldv Leading dimension of V
 * @param work Device workspace buffer
 * @param lwork Workspace size
 * @param info Device pointer: 0 on success
 * @param params Jacobi parameters
 * @return gpusolverStatus_t status code
 */
template<usual_fp T>
gpusolverStatus_t gesvdj(gpusolverDnHandle_t handle, gpusolverEigMode_t jobz, int econ, int m,
                         int n, T *A, int lda, ComplexToRealType<T> *S, T *U, int ldu, T *V,
                         int ldv, T *work, int lwork, int *info, gpusolverGesvdjInfo_t params) {
  GPUMOD_REAL_DISPATCH(T, gpusolverDn, S, D, gesvdj, handle, jobz, econ, m, n, A, lda, S, U, ldu, V,
                       ldv, work, lwork, info, params);
  GPUMOD_COMPLEX_DISPATCH(T, gpusolverDn, C, Z, gesvdj, handle, jobz, econ, m, n, A, lda, S, U, ldu,
                          V, ldv, work, lwork, info, params);
}

/**
 * @brief Query workspace size for batched Jacobi SVD
 *
 * @tparam T Data type (float, double, gpuFloatComplex, gpuDoubleComplex)
 * @param handle cuSOLVER handle
 * @param jobz CUSOLVER_EIG_MODE_VECTOR: compute singular vectors
 * @param m Number of rows of each matrix
 * @param n Number of columns of each matrix
 * @param A Pointer to array of matrices (device memory)
 * @param lda Leading dimension of each matrix
 * @param S Pointer to singular values array
 * @param U Pointer to left singular vectors (device memory)
 * @param ldu Leading dimension of U
 * @param V Pointer to right singular vectors (device memory)
 * @param ldv Leading dimension of V
 * @param lwork Output: required workspace size
 * @param params Jacobi parameters
 * @param batchSize Number of matrices in batch
 * @return gpusolverStatus_t status code
 */
template<usual_fp T>
gpusolverStatus_t gesvdjBatched_bufferSize(gpusolverDnHandle_t handle, gpusolverEigMode_t jobz,
                                           int m, int n, const T *A, int lda,
                                           const ComplexToRealType<T> *S, const T *U, int ldu,
                                           const T *V, int ldv, int *lwork,
                                           gpusolverGesvdjInfo_t params, int batchSize) {
  GPUMOD_REAL_DISPATCH(T, gpusolverDn, S, D, gesvdjBatched_bufferSize, handle, jobz, m, n, A, lda,
                       S, U, ldu, V, ldv, lwork, params, batchSize);
  GPUMOD_COMPLEX_DISPATCH(T, gpusolverDn, C, Z, gesvdjBatched_bufferSize, handle, jobz, m, n, A,
                          lda, S, U, ldu, V, ldv, lwork, params, batchSize);
}

/**
 * @brief Compute batched SVD using Jacobi method
 *
 * @tparam T Data type (float, double, gpuFloatComplex, gpuDoubleComplex)
 * @param handle cuSOLVER handle
 * @param jobz CUSOLVER_EIG_MODE_VECTOR: compute singular vectors
 * @param m Number of rows of each matrix
 * @param n Number of columns of each matrix
 * @param A Pointer to array of matrices (device memory)
 * @param lda Leading dimension of each matrix
 * @param S Output: singular values
 * @param U Output: left singular vectors (device memory)
 * @param ldu Leading dimension of U
 * @param V Output: right singular vectors (device memory)
 * @param ldv Leading dimension of V
 * @param work Device workspace buffer
 * @param lwork Workspace size
 * @param info Device pointer: array of status codes
 * @param params Jacobi parameters
 * @param batchSize Number of matrices in batch
 * @return gpusolverStatus_t status code
 */
template<usual_fp T>
gpusolverStatus_t gesvdjBatched(gpusolverDnHandle_t handle, gpusolverEigMode_t jobz, int m, int n,
                                T *A, int lda, ComplexToRealType<T> *S, T *U, int ldu, T *V,
                                int ldv, T *work, int lwork, int *info,
                                gpusolverGesvdjInfo_t params, int batchSize) {
  GPUMOD_REAL_DISPATCH(T, gpusolverDn, S, D, gesvdjBatched, handle, jobz, m, n, A, lda, S, U, ldu,
                       V, ldv, work, lwork, info, params, batchSize);
  GPUMOD_COMPLEX_DISPATCH(T, gpusolverDn, C, Z, gesvdjBatched, handle, jobz, m, n, A, lda, S, U,
                          ldu, V, ldv, work, lwork, info, params, batchSize);
}

// ========================================================================
// Generalized Symmetric Eigenvalue Decomposition (sygvd) - Real types only
// ========================================================================

/**
 * @brief Query workspace size for generalized symmetric eigenvalue problem
 *
 * @tparam T Data type (float or double only)
 * @param handle cuSOLVER handle
 * @param itype Problem type: 1 (A*x = lambda*B*x), 2 (A*B*x = lambda*x), 3 (B*A*x = lambda*x)
 * @param jobz CUSOLVER_EIG_MODE_VECTOR or CUSOLVER_EIG_MODE_NOVECTOR
 * @param uplo CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER
 * @param n Order of matrices A and B
 * @param A Pointer to matrix A (device memory)
 * @param lda Leading dimension of A
 * @param B Pointer to matrix B (device memory)
 * @param ldb Leading dimension of B
 * @param W Pointer to eigenvalues array
 * @param lwork Output: required workspace size
 * @return gpusolverStatus_t status code
 */
template<real_fp T>
gpusolverStatus_t sygvd_bufferSize(gpusolverDnHandle_t handle, gpusolverEigType_t itype,
                                   gpusolverEigMode_t jobz, gpublasFillMode_t uplo, int n,
                                   const T *A, int lda, const T *B, int ldb, const T *W,
                                   int *lwork) {
  GPUMOD_REAL_DISPATCH(T, gpusolverDn, S, D, sygvd_bufferSize, handle, itype, jobz, uplo, n, A, lda,
                       B, ldb, W, lwork);
}

/**
 * @brief Compute generalized symmetric eigenvalue problem
 *
 * @tparam T Data type (float or double only)
 * @param handle cuSOLVER handle
 * @param itype Problem type
 * @param jobz CUSOLVER_EIG_MODE_VECTOR or CUSOLVER_EIG_MODE_NOVECTOR
 * @param uplo CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER
 * @param n Order of matrices A and B
 * @param A Pointer to matrix A (device memory)
 * @param lda Leading dimension of A
 * @param B Pointer to matrix B (device memory)
 * @param ldb Leading dimension of B
 * @param W Output: eigenvalues
 * @param work Device workspace buffer
 * @param lwork Workspace size
 * @param info Device pointer: 0 on success
 * @return gpusolverStatus_t status code
 */
template<real_fp T>
gpusolverStatus_t sygvd(gpusolverDnHandle_t handle, gpusolverEigType_t itype,
                        gpusolverEigMode_t jobz, gpublasFillMode_t uplo, int n, T *A, int lda, T *B,
                        int ldb, T *W, T *work, int lwork, int *info) {
  GPUMOD_REAL_DISPATCH(T, gpusolverDn, S, D, sygvd, handle, itype, jobz, uplo, n, A, lda, B, ldb, W,
                       work, lwork, info);
}

// ========================================================================
// Generalized Symmetric Eigenvalue with Subset Selection (sygvdx) - Real types only
// ========================================================================

/**
 * @brief Query workspace size for generalized symmetric eigenvalue problem with subset selection
 *
 * @tparam T Data type (float or double only)
 * @param handle cuSOLVER handle
 * @param itype Problem type: 1 (A*x = lambda*B*x), 2 (A*B*x = lambda*x), 3 (B*A*x = lambda*x)
 * @param jobz CUSOLVER_EIG_MODE_VECTOR or CUSOLVER_EIG_MODE_NOVECTOR
 * @param range CUSOLVER_EIG_RANGE_ALL: all eigenvalues
 *              CUSOLVER_EIG_RANGE_V: eigenvalues in (vl, vu]
 *              CUSOLVER_EIG_RANGE_I: il-th through iu-th eigenvalues
 * @param uplo CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER
 * @param n Order of matrices A and B
 * @param A Pointer to matrix A (device memory)
 * @param lda Leading dimension of A
 * @param B Pointer to matrix B (device memory)
 * @param ldb Leading dimension of B
 * @param vl Lower bound of eigenvalue range (if range = CUSOLVER_EIG_RANGE_V)
 * @param vu Upper bound of eigenvalue range (if range = CUSOLVER_EIG_RANGE_V)
 * @param il Lower index of eigenvalue range (if range = CUSOLVER_EIG_RANGE_I, 1-indexed)
 * @param iu Upper index of eigenvalue range (if range = CUSOLVER_EIG_RANGE_I, 1-indexed)
 * @param meig Output: number of eigenvalues found
 * @param W Pointer to eigenvalues array
 * @param lwork Output: required workspace size
 * @return gpusolverStatus_t status code
 */
template<real_fp T>
gpusolverStatus_t sygvdx_bufferSize(gpusolverDnHandle_t handle, gpusolverEigType_t itype,
                                    gpusolverEigMode_t jobz, gpusolverEigRange_t range,
                                    gpublasFillMode_t uplo, int n, const T *A, int lda, const T *B,
                                    int ldb, T vl, T vu, int il, int iu, int *meig, const T *W,
                                    int *lwork) {
  GPUMOD_REAL_DISPATCH(T, gpusolverDn, S, D, sygvdx_bufferSize, handle, itype, jobz, range, uplo, n,
                       A, lda, B, ldb, vl, vu, il, iu, meig, W, lwork);
}

/**
 * @brief Compute generalized symmetric eigenvalue problem with subset selection
 *
 * @tparam T Data type (float or double only)
 * @param handle cuSOLVER handle
 * @param itype Problem type
 * @param jobz CUSOLVER_EIG_MODE_VECTOR or CUSOLVER_EIG_MODE_NOVECTOR
 * @param range CUSOLVER_EIG_RANGE_ALL: all eigenvalues
 *              CUSOLVER_EIG_RANGE_V: eigenvalues in (vl, vu]
 *              CUSOLVER_EIG_RANGE_I: il-th through iu-th eigenvalues
 * @param uplo CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER
 * @param n Order of matrices A and B
 * @param A Pointer to matrix A (device memory)
 * @param lda Leading dimension of A
 * @param B Pointer to matrix B (device memory)
 * @param ldb Leading dimension of B
 * @param vl Lower bound of eigenvalue range (if range = CUSOLVER_EIG_RANGE_V)
 * @param vu Upper bound of eigenvalue range (if range = CUSOLVER_EIG_RANGE_V, vu > vl)
 * @param il Lower index of eigenvalue range (if range = CUSOLVER_EIG_RANGE_I, 1-indexed)
 * @param iu Upper index of eigenvalue range (if range = CUSOLVER_EIG_RANGE_I, 1-indexed)
 * @param meig Output: number of eigenvalues found
 * @param W Output: selected eigenvalues
 * @param work Device workspace buffer
 * @param lwork Workspace size
 * @param info Device pointer: 0 on success
 * @return gpusolverStatus_t status code
 */
template<real_fp T>
gpusolverStatus_t sygvdx(gpusolverDnHandle_t handle, gpusolverEigType_t itype,
                         gpusolverEigMode_t jobz, gpusolverEigRange_t range, gpublasFillMode_t uplo,
                         int n, T *A, int lda, T *B, int ldb, T vl, T vu, int il, int iu, int *meig,
                         T *W, T *work, int lwork, int *info) {
  GPUMOD_REAL_DISPATCH(T, gpusolverDn, S, D, sygvdx, handle, itype, jobz, range, uplo, n, A, lda, B,
                       ldb, vl, vu, il, iu, meig, W, work, lwork, info);
}

// ========================================================================
// Generalized Hermitian Eigenvalue Decomposition (hegvd) - Complex types only
// ========================================================================

/**
 * @brief Query workspace size for generalized Hermitian eigenvalue problem
 *
 * @tparam T Data type (gpuFloatComplex or gpuDoubleComplex only)
 * @param handle cuSOLVER handle
 * @param itype Problem type
 * @param jobz CUSOLVER_EIG_MODE_VECTOR or CUSOLVER_EIG_MODE_NOVECTOR
 * @param uplo CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER
 * @param n Order of matrices A and B
 * @param A Pointer to matrix A (device memory)
 * @param lda Leading dimension of A
 * @param B Pointer to matrix B (device memory)
 * @param ldb Leading dimension of B
 * @param W Pointer to eigenvalues array (real-valued)
 * @param lwork Output: required workspace size
 * @return gpusolverStatus_t status code
 */
template<complex_fp T>
gpusolverStatus_t hegvd_bufferSize(gpusolverDnHandle_t handle, gpusolverEigType_t itype,
                                   gpusolverEigMode_t jobz, gpublasFillMode_t uplo, int n,
                                   const T *A, int lda, const T *B, int ldb,
                                   const ComplexToRealType<T> *W, int *lwork) {
  GPUMOD_COMPLEX_DISPATCH(T, gpusolverDn, C, Z, hegvd_bufferSize, handle, itype, jobz, uplo, n, A,
                          lda, B, ldb, W, lwork);
}

/**
 * @brief Compute generalized Hermitian eigenvalue problem
 *
 * @tparam T Data type (gpuFloatComplex or gpuDoubleComplex only)
 * @param handle cuSOLVER handle
 * @param itype Problem type
 * @param jobz CUSOLVER_EIG_MODE_VECTOR or CUSOLVER_EIG_MODE_NOVECTOR
 * @param uplo CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER
 * @param n Order of matrices A and B
 * @param A Pointer to matrix A (device memory)
 * @param lda Leading dimension of A
 * @param B Pointer to matrix B (device memory)
 * @param ldb Leading dimension of B
 * @param W Output: eigenvalues (real-valued)
 * @param work Device workspace buffer
 * @param lwork Workspace size
 * @param info Device pointer: 0 on success
 * @return gpusolverStatus_t status code
 */
template<complex_fp T>
gpusolverStatus_t hegvd(gpusolverDnHandle_t handle, gpusolverEigType_t itype,
                        gpusolverEigMode_t jobz, gpublasFillMode_t uplo, int n, T *A, int lda, T *B,
                        int ldb, ComplexToRealType<T> *W, T *work, int lwork, int *info) {
  GPUMOD_COMPLEX_DISPATCH(T, gpusolverDn, C, Z, hegvd, handle, itype, jobz, uplo, n, A, lda, B, ldb,
                          W, work, lwork, info);
}

// ========================================================================
// Generalized Hermitian Eigenvalue with Subset Selection (hegvdx) - Complex types only
// ========================================================================

/**
 * @brief Query workspace size for generalized Hermitian eigenvalue problem with subset selection
 *
 * @tparam T Data type (gpuFloatComplex or gpuDoubleComplex only)
 * @param handle cuSOLVER handle
 * @param itype Problem type
 * @param jobz CUSOLVER_EIG_MODE_VECTOR or CUSOLVER_EIG_MODE_NOVECTOR
 * @param range CUSOLVER_EIG_RANGE_ALL: all eigenvalues
 *              CUSOLVER_EIG_RANGE_V: eigenvalues in (vl, vu]
 *              CUSOLVER_EIG_RANGE_I: il-th through iu-th eigenvalues
 * @param uplo CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER
 * @param n Order of matrices A and B
 * @param A Pointer to matrix A (device memory)
 * @param lda Leading dimension of A
 * @param B Pointer to matrix B (device memory)
 * @param ldb Leading dimension of B
 * @param vl Lower bound of eigenvalue range (if range = CUSOLVER_EIG_RANGE_V, real-valued)
 * @param vu Upper bound of eigenvalue range (if range = CUSOLVER_EIG_RANGE_V, real-valued)
 * @param il Lower index of eigenvalue range (if range = CUSOLVER_EIG_RANGE_I, 1-indexed)
 * @param iu Upper index of eigenvalue range (if range = CUSOLVER_EIG_RANGE_I, 1-indexed)
 * @param meig Output: number of eigenvalues found
 * @param W Pointer to eigenvalues array (real-valued)
 * @param lwork Output: required workspace size
 * @return gpusolverStatus_t status code
 */
template<complex_fp T>
gpusolverStatus_t
hegvdx_bufferSize(gpusolverDnHandle_t handle, gpusolverEigType_t itype, gpusolverEigMode_t jobz,
                  gpusolverEigRange_t range, gpublasFillMode_t uplo, int n, const T *A, int lda,
                  const T *B, int ldb, ComplexToRealType<T> vl, ComplexToRealType<T> vu, int il,
                  int iu, int *meig, const ComplexToRealType<T> *W, int *lwork) {
  GPUMOD_COMPLEX_DISPATCH(T, gpusolverDn, C, Z, hegvdx_bufferSize, handle, itype, jobz, range, uplo,
                          n, A, lda, B, ldb, vl, vu, il, iu, meig, W, lwork);
}

/**
 * @brief Compute generalized Hermitian eigenvalue problem with subset selection
 *
 * @tparam T Data type (gpuFloatComplex or gpuDoubleComplex only)
 * @param handle cuSOLVER handle
 * @param itype Problem type
 * @param jobz CUSOLVER_EIG_MODE_VECTOR or CUSOLVER_EIG_MODE_NOVECTOR
 * @param range CUSOLVER_EIG_RANGE_ALL: all eigenvalues
 *              CUSOLVER_EIG_RANGE_V: eigenvalues in (vl, vu]
 *              CUSOLVER_EIG_RANGE_I: il-th through iu-th eigenvalues
 * @param uplo CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER
 * @param n Order of matrices A and B
 * @param A Pointer to matrix A (device memory)
 * @param lda Leading dimension of A
 * @param B Pointer to matrix B (device memory)
 * @param ldb Leading dimension of B
 * @param vl Lower bound of eigenvalue range (if range = CUSOLVER_EIG_RANGE_V, real-valued)
 * @param vu Upper bound of eigenvalue range (if range = CUSOLVER_EIG_RANGE_V, real-valued, vu > vl)
 * @param il Lower index of eigenvalue range (if range = CUSOLVER_EIG_RANGE_I, 1-indexed)
 * @param iu Upper index of eigenvalue range (if range = CUSOLVER_EIG_RANGE_I, 1-indexed)
 * @param meig Output: number of eigenvalues found
 * @param W Output: selected eigenvalues (real-valued)
 * @param work Device workspace buffer
 * @param lwork Workspace size
 * @param info Device pointer: 0 on success
 * @return gpusolverStatus_t status code
 */
template<complex_fp T>
gpusolverStatus_t hegvdx(gpusolverDnHandle_t handle, gpusolverEigType_t itype,
                         gpusolverEigMode_t jobz, gpusolverEigRange_t range, gpublasFillMode_t uplo,
                         int n, T *A, int lda, T *B, int ldb, ComplexToRealType<T> vl,
                         ComplexToRealType<T> vu, int il, int iu, int *meig,
                         ComplexToRealType<T> *W, T *work, int lwork, int *info) {
  GPUMOD_COMPLEX_DISPATCH(T, gpusolverDn, C, Z, hegvdx, handle, itype, jobz, range, uplo, n, A, lda,
                          B, ldb, vl, vu, il, iu, meig, W, work, lwork, info);
}

// ========================================================================
// Generalized Symmetric Eigenvalue using Jacobi (sygvj) - Real types only
// ========================================================================

/**
 * @brief Query workspace size for generalized symmetric eigenvalue using Jacobi
 *
 * @tparam T Data type (float or double only)
 * @param handle cuSOLVER handle
 * @param itype Problem type
 * @param jobz CUSOLVER_EIG_MODE_VECTOR or CUSOLVER_EIG_MODE_NOVECTOR
 * @param uplo CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER
 * @param n Order of matrices A and B
 * @param A Pointer to matrix A (device memory)
 * @param lda Leading dimension of A
 * @param B Pointer to matrix B (device memory)
 * @param ldb Leading dimension of B
 * @param W Pointer to eigenvalues array
 * @param lwork Output: required workspace size
 * @param params Jacobi parameters
 * @return gpusolverStatus_t status code
 */
template<real_fp T>
gpusolverStatus_t sygvj_bufferSize(gpusolverDnHandle_t handle, gpusolverEigType_t itype,
                                   gpusolverEigMode_t jobz, gpublasFillMode_t uplo, int n,
                                   const T *A, int lda, const T *B, int ldb, const T *W, int *lwork,
                                   gpusolverSyevjInfo_t params) {
  GPUMOD_REAL_DISPATCH(T, gpusolverDn, S, D, sygvj_bufferSize, handle, itype, jobz, uplo, n, A, lda,
                       B, ldb, W, lwork, params);
}

/**
 * @brief Compute generalized symmetric eigenvalue using Jacobi method
 *
 * @tparam T Data type (float or double only)
 * @param handle cuSOLVER handle
 * @param itype Problem type
 * @param jobz CUSOLVER_EIG_MODE_VECTOR or CUSOLVER_EIG_MODE_NOVECTOR
 * @param uplo CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER
 * @param n Order of matrices A and B
 * @param A Pointer to matrix A (device memory)
 * @param lda Leading dimension of A
 * @param B Pointer to matrix B (device memory)
 * @param ldb Leading dimension of B
 * @param W Output: eigenvalues
 * @param work Device workspace buffer
 * @param lwork Workspace size
 * @param info Device pointer: 0 on success
 * @param params Jacobi parameters
 * @return gpusolverStatus_t status code
 */
template<real_fp T>
gpusolverStatus_t sygvj(gpusolverDnHandle_t handle, gpusolverEigType_t itype,
                        gpusolverEigMode_t jobz, gpublasFillMode_t uplo, int n, T *A, int lda, T *B,
                        int ldb, T *W, T *work, int lwork, int *info, gpusolverSyevjInfo_t params) {
  GPUMOD_REAL_DISPATCH(T, gpusolverDn, S, D, sygvj, handle, itype, jobz, uplo, n, A, lda, B, ldb, W,
                       work, lwork, info, params);
}

// ========================================================================
// Generalized Hermitian Eigenvalue using Jacobi (hegvj) - Complex types only
// ========================================================================

/**
 * @brief Query workspace size for generalized Hermitian eigenvalue using Jacobi
 *
 * @tparam T Data type (gpuFloatComplex or gpuDoubleComplex only)
 * @param handle cuSOLVER handle
 * @param itype Problem type
 * @param jobz CUSOLVER_EIG_MODE_VECTOR or CUSOLVER_EIG_MODE_NOVECTOR
 * @param uplo CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER
 * @param n Order of matrices A and B
 * @param A Pointer to matrix A (device memory)
 * @param lda Leading dimension of A
 * @param B Pointer to matrix B (device memory)
 * @param ldb Leading dimension of B
 * @param W Pointer to eigenvalues array (real-valued)
 * @param lwork Output: required workspace size
 * @param params Jacobi parameters
 * @return gpusolverStatus_t status code
 */
template<complex_fp T>
gpusolverStatus_t
hegvj_bufferSize(gpusolverDnHandle_t handle, gpusolverEigType_t itype, gpusolverEigMode_t jobz,
                 gpublasFillMode_t uplo, int n, const T *A, int lda, const T *B, int ldb,
                 const ComplexToRealType<T> *W, int *lwork, gpusolverSyevjInfo_t params) {
  GPUMOD_COMPLEX_DISPATCH(T, gpusolverDn, C, Z, hegvj_bufferSize, handle, itype, jobz, uplo, n, A,
                          lda, B, ldb, W, lwork, params);
}

/**
 * @brief Compute generalized Hermitian eigenvalue using Jacobi method
 *
 * @tparam T Data type (gpuFloatComplex or gpuDoubleComplex only)
 * @param handle cuSOLVER handle
 * @param itype Problem type
 * @param jobz CUSOLVER_EIG_MODE_VECTOR or CUSOLVER_EIG_MODE_NOVECTOR
 * @param uplo CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER
 * @param n Order of matrices A and B
 * @param A Pointer to matrix A (device memory)
 * @param lda Leading dimension of A
 * @param B Pointer to matrix B (device memory)
 * @param ldb Leading dimension of B
 * @param W Output: eigenvalues (real-valued)
 * @param work Device workspace buffer
 * @param lwork Workspace size
 * @param info Device pointer: 0 on success
 * @param params Jacobi parameters
 * @return gpusolverStatus_t status code
 */
template<complex_fp T>
gpusolverStatus_t hegvj(gpusolverDnHandle_t handle, gpusolverEigType_t itype,
                        gpusolverEigMode_t jobz, gpublasFillMode_t uplo, int n, T *A, int lda, T *B,
                        int ldb, ComplexToRealType<T> *W, T *work, int lwork, int *info,
                        gpusolverSyevjInfo_t params) {
  GPUMOD_COMPLEX_DISPATCH(T, gpusolverDn, C, Z, hegvj, handle, itype, jobz, uplo, n, A, lda, B, ldb,
                          W, work, lwork, info, params);
}

// ========================================================================
// Strided Batched Singular Value Decomposition (gesvdaStridedBatched)
// ========================================================================

/**
 * @brief Query workspace size for strided batched approximate SVD
 *
 * This function computes the workspace size required for gesvdaStridedBatched,
 * which performs approximate SVD on a batch of matrices with strided layout.
 *
 * @tparam T Data type (float, double, gpuFloatComplex, gpuDoubleComplex)
 * @param handle cuSOLVER handle
 * @param jobz CUSOLVER_EIG_MODE_VECTOR: compute singular vectors
 *             CUSOLVER_EIG_MODE_NOVECTOR: compute singular values only
 * @param rank Numerical rank of the matrices (approximation parameter)
 * @param m Number of rows of each matrix A
 * @param n Number of columns of each matrix A
 * @param d_A Pointer to first matrix in batch (device memory)
 * @param lda Leading dimension of each matrix A
 * @param strideA Stride between matrices A (in elements)
 * @param d_S Pointer to first singular values array (device memory, real-valued)
 * @param strideS Stride between singular value arrays (in elements)
 * @param d_U Pointer to first left singular vectors matrix (device memory)
 * @param ldu Leading dimension of each U matrix
 * @param strideU Stride between U matrices (in elements)
 * @param d_V Pointer to first right singular vectors matrix (device memory)
 * @param ldv Leading dimension of each V matrix
 * @param strideV Stride between V matrices (in elements)
 * @param lwork Output: required workspace size in elements of type T
 * @param batchSize Number of matrices in the batch
 * @return gpusolverStatus_t status code
 */
template<usual_fp T>
gpusolverStatus_t
gesvdaStridedBatched_bufferSize(gpusolverDnHandle_t handle, gpusolverEigMode_t jobz, int rank,
                                int m, int n, const T *d_A, int lda, long long int strideA,
                                const ComplexToRealType<T> *d_S, long long int strideS,
                                const T *d_U, int ldu, long long int strideU, const T *d_V, int ldv,
                                long long int strideV, int *lwork, int batchSize) {
  GPUMOD_REAL_DISPATCH(T, gpusolverDn, S, D, gesvdaStridedBatched_bufferSize, handle, jobz, rank, m,
                       n, d_A, lda, strideA, d_S, strideS, d_U, ldu, strideU, d_V, ldv, strideV,
                       lwork, batchSize);
  GPUMOD_COMPLEX_DISPATCH(T, gpusolverDn, C, Z, gesvdaStridedBatched_bufferSize, handle, jobz, rank,
                          m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu, strideU, d_V, ldv,
                          strideV, lwork, batchSize);
}

/**
 * @brief Compute approximate SVD for a batch of matrices with strided layout
 *
 * Computes an approximate singular value decomposition of a batch of m×n matrices using
 * randomized algorithms. Matrices are stored in strided layout (fixed stride between matrices).
 * This is faster than exact SVD but provides approximate results.
 *
 * @tparam T Data type (float, double, gpuFloatComplex, gpuDoubleComplex)
 * @param handle cuSOLVER handle
 * @param jobz CUSOLVER_EIG_MODE_VECTOR: compute singular vectors
 *             CUSOLVER_EIG_MODE_NOVECTOR: compute singular values only
 * @param rank Numerical rank (number of singular values/vectors to compute)
 * @param m Number of rows of each matrix A (m >= 0)
 * @param n Number of columns of each matrix A (n >= 0)
 * @param d_A Pointer to first matrix in batch (device memory); overwritten on exit
 * @param lda Leading dimension of each matrix A (lda >= max(1, m))
 * @param strideA Stride between consecutive matrices A (in elements)
 * @param d_S Output: singular values in descending order (real-valued, rank elements per matrix)
 * @param strideS Stride between consecutive singular value arrays (in elements)
 * @param d_U Output: left singular vectors (device memory, rank columns)
 * @param ldu Leading dimension of each U matrix (ldu >= max(1, m))
 * @param strideU Stride between consecutive U matrices (in elements)
 * @param d_V Output: right singular vectors (device memory, rank columns)
 * @param ldv Leading dimension of each V matrix (ldv >= max(1, n))
 * @param strideV Stride between consecutive V matrices (in elements)
 * @param d_work Device workspace buffer (size from gesvdaStridedBatched_bufferSize)
 * @param lwork Workspace size in elements of type T
 * @param d_info Device pointer: array of status codes (batchSize elements), 0 on success for each matrix
 * @param h_R_nrmF Host pointer: array of residual Frobenius norms (double, batchSize elements)
 * @param batchSize Number of matrices in the batch
 * @return gpusolverStatus_t status code
 */
template<usual_fp T>
gpusolverStatus_t
gesvdaStridedBatched(gpusolverDnHandle_t handle, gpusolverEigMode_t jobz, int rank, int m, int n,
                     const T *d_A, int lda, long long int strideA, ComplexToRealType<T> *d_S,
                     long long int strideS, T *d_U, int ldu, long long int strideU, T *d_V, int ldv,
                     long long int strideV, T *d_work, int lwork, int *d_info, double *h_R_nrmF,
                     int batchSize) {
  GPUMOD_REAL_DISPATCH(T, gpusolverDn, S, D, gesvdaStridedBatched, handle, jobz, rank, m, n, d_A,
                       lda, strideA, d_S, strideS, d_U, ldu, strideU, d_V, ldv, strideV, d_work,
                       lwork, d_info, h_R_nrmF, batchSize);
  GPUMOD_COMPLEX_DISPATCH(T, gpusolverDn, C, Z, gesvdaStridedBatched, handle, jobz, rank, m, n, d_A,
                          lda, strideA, d_S, strideS, d_U, ldu, strideU, d_V, ldv, strideV, d_work,
                          lwork, d_info, h_R_nrmF, batchSize);
}

// ==================== Explicit Template Instantiation Declarations ====================
// Generated code will be inserted here by cmake/instantiation/generate_instantiations.py
// To regenerate: cmake --build build --target gpusolverDn_generate_instantiations

// Function: gesvd_bufferSize
extern template gpusolverStatus_t gesvd_bufferSize<float>(gpusolverDnHandle_t, int, int, int *);
extern template gpusolverStatus_t gesvd_bufferSize<double>(gpusolverDnHandle_t, int, int, int *);
extern template gpusolverStatus_t gesvd_bufferSize<gpuFloatComplex>(gpusolverDnHandle_t, int, int,
                                                                    int *);
extern template gpusolverStatus_t gesvd_bufferSize<gpuDoubleComplex>(gpusolverDnHandle_t, int, int,
                                                                     int *);

// Function: gesvd
extern template gpusolverStatus_t gesvd<float>(gpusolverDnHandle_t, signed char, signed char, int,
                                               int, float *, int, ComplexToRealType<float> *,
                                               float *, int, float *, int, float *, int,
                                               ComplexToRealType<float> *, int *);
extern template gpusolverStatus_t gesvd<double>(gpusolverDnHandle_t, signed char, signed char, int,
                                                int, double *, int, ComplexToRealType<double> *,
                                                double *, int, double *, int, double *, int,
                                                ComplexToRealType<double> *, int *);
extern template gpusolverStatus_t
gesvd<gpuFloatComplex>(gpusolverDnHandle_t, signed char, signed char, int, int, gpuFloatComplex *,
                       int, ComplexToRealType<gpuFloatComplex> *, gpuFloatComplex *, int,
                       gpuFloatComplex *, int, gpuFloatComplex *, int,
                       ComplexToRealType<gpuFloatComplex> *, int *);
extern template gpusolverStatus_t
gesvd<gpuDoubleComplex>(gpusolverDnHandle_t, signed char, signed char, int, int, gpuDoubleComplex *,
                        int, ComplexToRealType<gpuDoubleComplex> *, gpuDoubleComplex *, int,
                        gpuDoubleComplex *, int, gpuDoubleComplex *, int,
                        ComplexToRealType<gpuDoubleComplex> *, int *);

// Function: syevd_bufferSize
extern template gpusolverStatus_t syevd_bufferSize<float>(gpusolverDnHandle_t, gpusolverEigMode_t,
                                                          gpublasFillMode_t, int, const float *,
                                                          int, const float *, int *);
extern template gpusolverStatus_t syevd_bufferSize<double>(gpusolverDnHandle_t, gpusolverEigMode_t,
                                                           gpublasFillMode_t, int, const double *,
                                                           int, const double *, int *);

// Function: syevd
extern template gpusolverStatus_t syevd<float>(gpusolverDnHandle_t, gpusolverEigMode_t,
                                               gpublasFillMode_t, int, float *, int, float *,
                                               float *, int, int *);
extern template gpusolverStatus_t syevd<double>(gpusolverDnHandle_t, gpusolverEigMode_t,
                                                gpublasFillMode_t, int, double *, int, double *,
                                                double *, int, int *);

// Function: syevdx_bufferSize
extern template gpusolverStatus_t syevdx_bufferSize<float>(gpusolverDnHandle_t, gpusolverEigMode_t,
                                                           gpusolverEigRange_t, gpublasFillMode_t,
                                                           int, const float *, int, float, float,
                                                           int, int, int *, const float *, int *);
extern template gpusolverStatus_t syevdx_bufferSize<double>(gpusolverDnHandle_t, gpusolverEigMode_t,
                                                            gpusolverEigRange_t, gpublasFillMode_t,
                                                            int, const double *, int, double,
                                                            double, int, int, int *, const double *,
                                                            int *);

// Function: syevdx
extern template gpusolverStatus_t syevdx<float>(gpusolverDnHandle_t, gpusolverEigMode_t,
                                                gpusolverEigRange_t, gpublasFillMode_t, int,
                                                float *, int, float, float, int, int, int *,
                                                float *, float *, int, int *);
extern template gpusolverStatus_t syevdx<double>(gpusolverDnHandle_t, gpusolverEigMode_t,
                                                 gpusolverEigRange_t, gpublasFillMode_t, int,
                                                 double *, int, double, double, int, int, int *,
                                                 double *, double *, int, int *);

} // namespace gpumod
