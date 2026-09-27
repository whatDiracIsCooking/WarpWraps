/**
 * @file level_3.cppm
 * @brief GPU BLAS Level 3 (matrix-matrix) operations
 *
 * This module provides type-safe wrappers for GPU BLAS Level 3 BLAS operations.
 *
 * Usage:
 *   import gpumod.wrappers.blas;
 */

module;

#include "dispatch_macros.h"

export module gpumod.wrappers.blas:level_3;

import gpumod.blas;
import gpumod.complex;
import :type_traits;
import std;

export namespace wwr {

// ========================================================================
// General Matrix-Matrix Multiplication (GEMM)
// ========================================================================

/**
 * @brief General matrix-matrix multiplication: C = alpha*op(A)*op(B) + beta*C
 *
 * @param handle GPU BLAS handle
 * @param transa Operation on matrix A (GPUBLAS_OP_N, GPUBLAS_OP_T, GPUBLAS_OP_C)
 * @param transb Operation on matrix B
 * @param m Number of rows of matrix op(A) and C
 * @param n Number of columns of matrix op(B) and C
 * @param k Number of columns of op(A) and rows of op(B)
 * @param alpha Scalar multiplier for A*B
 * @param A Matrix A
 * @param lda Leading dimension of A
 * @param B Matrix B
 * @param ldb Leading dimension of B
 * @param beta Scalar multiplier for C
 * @param C Matrix C (input/output)
 * @param ldc Leading dimension of C
 */
template<usual_fp T, int_type IntT>
gpublasStatus_t gemm(gpublasHandle_t handle, gpublasOperation_t transa, gpublasOperation_t transb,
                     IntT m, IntT n, IntT k, const T *alpha, const T *A, IntT lda, const T *B,
                     IntT ldb, const T *beta, T *C, IntT ldc) {
  WWR_USUAL_DISPATCH_64(T, IntT, gemm, handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                           beta, C, ldc);
}

// ========================================================================
// Batched GEMM
// ========================================================================

/**
 * @brief Batched matrix-matrix multiplication
 *
 * Performs multiple independent GEMM operations in a single call.
 *
 * @param Aarray Array of pointers to matrices A
 * @param Barray Array of pointers to matrices B
 * @param Carray Array of pointers to matrices C
 * @param batchCount Number of GEMM operations to perform
 */
template<usual_fp T, int_type IntT>
gpublasStatus_t gemmBatched(gpublasHandle_t handle, gpublasOperation_t transa,
                            gpublasOperation_t transb, IntT m, IntT n, IntT k, const T *alpha,
                            const T *const Aarray[], IntT lda, const T *const Barray[], IntT ldb,
                            const T *beta, T *const Carray[], IntT ldc, IntT batchCount) {
  WWR_USUAL_DISPATCH_64(T, IntT, gemmBatched, handle, transa, transb, m, n, k, alpha, Aarray,
                           lda, Barray, ldb, beta, Carray, ldc, batchCount);
}

// ========================================================================
// Strided Batched GEMM
// ========================================================================

/**
 * @brief Strided batched matrix-matrix multiplication
 *
 * Performs multiple GEMM operations where matrices are stored with fixed strides.
 * More efficient than gemmBatched when matrices are uniformly strided in memory.
 *
 * @param strideA Stride between consecutive A matrices
 * @param strideB Stride between consecutive B matrices
 * @param strideC Stride between consecutive C matrices
 * @param batchCount Number of GEMM operations to perform
 */
template<usual_fp T, int_type IntT>
gpublasStatus_t gemmStridedBatched(gpublasHandle_t handle, gpublasOperation_t transa,
                                   gpublasOperation_t transb, IntT m, IntT n, IntT k,
                                   const T *alpha, const T *A, IntT lda, long long int strideA,
                                   const T *B, IntT ldb, long long int strideB, const T *beta, T *C,
                                   IntT ldc, long long int strideC, IntT batchCount) {
  WWR_USUAL_DISPATCH_64(T, IntT, gemmStridedBatched, handle, transa, transb, m, n, k, alpha, A,
                           lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
}

// ========================================================================
// Symmetric Matrix-Matrix Multiplication
// ========================================================================

/**
 * @brief Symmetric matrix-matrix multiplication: C = alpha*A*B + beta*C or C = alpha*B*A + beta*C
 *
 * @param side Specifies whether symmetric matrix A appears on left or right
 * @param uplo Specifies whether upper or lower triangular part of A is referenced
 */
template<usual_fp T, int_type IntT>
gpublasStatus_t symm(gpublasHandle_t handle, gpublasSideMode_t side, gpublasFillMode_t uplo, IntT m,
                     IntT n, const T *alpha, const T *A, IntT lda, const T *B, IntT ldb,
                     const T *beta, T *C, IntT ldc) {
  WWR_USUAL_DISPATCH_64(T, IntT, symm, handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C,
                           ldc);
}

// ========================================================================
// Symmetric Rank-K Update
// ========================================================================

/**
 * @brief Symmetric rank-k update: C = alpha*op(A)*op(A)^T + beta*C
 *
 * @param uplo Specifies whether upper or lower triangular part of C is updated
 * @param trans Operation on matrix A (GPUBLAS_OP_N or GPUBLAS_OP_T)
 */
template<usual_fp T, int_type IntT>
gpublasStatus_t syrk(gpublasHandle_t handle, gpublasFillMode_t uplo, gpublasOperation_t trans,
                     IntT n, IntT k, const T *alpha, const T *A, IntT lda, const T *beta, T *C,
                     IntT ldc) {
  WWR_USUAL_DISPATCH_64(T, IntT, syrk, handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}

// ========================================================================
// Symmetric Rank-2K Update
// ========================================================================

/**
 * @brief Symmetric rank-2k update: C = alpha*(op(A)*op(B)^T + op(B)*op(A)^T) + beta*C
 */
template<usual_fp T, int_type IntT>
gpublasStatus_t syr2k(gpublasHandle_t handle, gpublasFillMode_t uplo, gpublasOperation_t trans,
                      IntT n, IntT k, const T *alpha, const T *A, IntT lda, const T *B, IntT ldb,
                      const T *beta, T *C, IntT ldc) {
  WWR_USUAL_DISPATCH_64(T, IntT, syr2k, handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta,
                           C, ldc);
}

// ========================================================================
// Symmetric Rank-K Update Variant
// ========================================================================

/**
 * @brief Variant of symmetric rank-k update: C = alpha*op(A)*op(B)^T + beta*C
 */
template<usual_fp T, int_type IntT>
gpublasStatus_t syrkx(gpublasHandle_t handle, gpublasFillMode_t uplo, gpublasOperation_t trans,
                      IntT n, IntT k, const T *alpha, const T *A, IntT lda, const T *B, IntT ldb,
                      const T *beta, T *C, IntT ldc) {
  WWR_USUAL_DISPATCH_64(T, IntT, syrkx, handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta,
                           C, ldc);
}

// ========================================================================
// Triangular Matrix-Matrix Multiplication
// ========================================================================

/**
 * @brief Triangular matrix-matrix multiplication: B = alpha*op(A)*B or B = alpha*B*op(A)
 *
 * @param side Specifies whether triangular matrix A appears on left or right
 * @param uplo Specifies whether A is upper or lower triangular
 * @param trans Operation on matrix A
 * @param diag Specifies whether A is unit triangular
 */
template<usual_fp T, int_type IntT>
gpublasStatus_t trmm(gpublasHandle_t handle, gpublasSideMode_t side, gpublasFillMode_t uplo,
                     gpublasOperation_t trans, gpublasDiagType_t diag, IntT m, IntT n,
                     const T *alpha, const T *A, IntT lda, const T *B, IntT ldb, T *C, IntT ldc) {
  WWR_USUAL_DISPATCH_64(T, IntT, trmm, handle, side, uplo, trans, diag, m, n, alpha, A, lda, B,
                           ldb, C, ldc);
}

// ========================================================================
// Triangular Solve with Multiple Right-Hand Sides
// ========================================================================

/**
 * @brief Solves triangular system: op(A)*X = alpha*B or X*op(A) = alpha*B
 *
 * @param side Specifies whether triangular matrix A appears on left or right
 * @param uplo Specifies whether A is upper or lower triangular
 * @param trans Operation on matrix A
 * @param diag Specifies whether A is unit triangular
 */
template<usual_fp T, int_type IntT>
gpublasStatus_t trsm(gpublasHandle_t handle, gpublasSideMode_t side, gpublasFillMode_t uplo,
                     gpublasOperation_t trans, gpublasDiagType_t diag, IntT m, IntT n,
                     const T *alpha, const T *A, IntT lda, T *B, IntT ldb) {
  WWR_USUAL_DISPATCH_64(T, IntT, trsm, handle, side, uplo, trans, diag, m, n, alpha, A, lda, B,
                           ldb);
}

// ========================================================================
// Batched Triangular Solve
// ========================================================================

/**
 * @brief Batched triangular solver
 *
 * Solves multiple independent triangular systems in a single call.
 */
template<usual_fp T, int_type IntT>
gpublasStatus_t trsmBatched(gpublasHandle_t handle, gpublasSideMode_t side, gpublasFillMode_t uplo,
                            gpublasOperation_t trans, gpublasDiagType_t diag, IntT m, IntT n,
                            const T *alpha, const T *const A[], IntT lda, T *const B[], IntT ldb,
                            IntT batchCount) {
  WWR_USUAL_DISPATCH_64(T, IntT, trsmBatched, handle, side, uplo, trans, diag, m, n, alpha, A,
                           lda, B, ldb, batchCount);
}

// ========================================================================
// Hermitian Matrix-Matrix Multiplication (Complex Only)
// ========================================================================

/**
 * @brief Hermitian matrix-matrix multiplication: C = alpha*A*B + beta*C or C = alpha*B*A + beta*C
 *
 * Only available for complex types.
 *
 * @param side Specifies whether Hermitian matrix A appears on left or right
 * @param uplo Specifies whether upper or lower triangular part of A is referenced
 */
template<complex_fp T, int_type IntT>
gpublasStatus_t hemm(gpublasHandle_t handle, gpublasSideMode_t side, gpublasFillMode_t uplo, IntT m,
                     IntT n, const T *alpha, const T *A, IntT lda, const T *B, IntT ldb,
                     const T *beta, T *C, IntT ldc) {
  WWR_COMPLEX_DISPATCH_64(T, IntT, gpublas, C, Z, hemm, handle, side, uplo, m, n, alpha, A, lda,
                             B, ldb, beta, C, ldc);
}

// ========================================================================
// Hermitian Rank-K Update (Complex Only)
// ========================================================================

/**
 * @brief Hermitian rank-k update: C = alpha*op(A)*op(A)^H + beta*C
 *
 * Only available for complex types. Note that alpha and beta are real scalars.
 *
 * @param uplo Specifies whether upper or lower triangular part of C is updated
 * @param trans Operation on matrix A (GPUBLAS_OP_N or GPUBLAS_OP_C)
 */
template<complex_fp T, int_type IntT>
gpublasStatus_t herk(gpublasHandle_t handle, gpublasFillMode_t uplo, gpublasOperation_t trans,
                     IntT n, IntT k, const ComplexToRealType<T> *alpha, const T *A, IntT lda,
                     const ComplexToRealType<T> *beta, T *C, IntT ldc) {
  WWR_COMPLEX_DISPATCH_64(T, IntT, gpublas, C, Z, herk, handle, uplo, trans, n, k, alpha, A, lda,
                             beta, C, ldc);
}

// ========================================================================
// Hermitian Rank-2K Update (Complex Only)
// ========================================================================

/**
 * @brief Hermitian rank-2k update: C = alpha*op(A)*op(B)^H + conj(alpha)*op(B)*op(A)^H + beta*C
 *
 * Only available for complex types. Note that alpha is complex but beta is real.
 */
template<complex_fp T, int_type IntT>
gpublasStatus_t her2k(gpublasHandle_t handle, gpublasFillMode_t uplo, gpublasOperation_t trans,
                      IntT n, IntT k, const T *alpha, const T *A, IntT lda, const T *B, IntT ldb,
                      const ComplexToRealType<T> *beta, T *C, IntT ldc) {
  WWR_COMPLEX_DISPATCH_64(T, IntT, gpublas, C, Z, her2k, handle, uplo, trans, n, k, alpha, A,
                             lda, B, ldb, beta, C, ldc);
}

// ========================================================================
// Hermitian Rank-K Update Variant (Complex Only)
// ========================================================================

/**
 * @brief Variant of Hermitian rank-k update: C = alpha*op(A)*op(B)^H + beta*C
 *
 * Only available for complex types. Note that alpha is complex but beta is real.
 */
template<complex_fp T, int_type IntT>
gpublasStatus_t herkx(gpublasHandle_t handle, gpublasFillMode_t uplo, gpublasOperation_t trans,
                      IntT n, IntT k, const T *alpha, const T *A, IntT lda, const T *B, IntT ldb,
                      const ComplexToRealType<T> *beta, T *C, IntT ldc) {
  WWR_COMPLEX_DISPATCH_64(T, IntT, gpublas, C, Z, herkx, handle, uplo, trans, n, k, alpha, A,
                             lda, B, ldb, beta, C, ldc);
}

// ==================== Explicit Template Instantiations ====================
// Matching `template` instantiations live in instantiations.cpp.

// Function: gemm
extern template gpublasStatus_t gemm<float, int>(gpublasHandle_t, gpublasOperation_t,
                                                 gpublasOperation_t, int, int, int, const float *,
                                                 const float *, int, const float *, int,
                                                 const float *, float *, int);
extern template gpublasStatus_t gemm<float, int64_t>(gpublasHandle_t, gpublasOperation_t,
                                                     gpublasOperation_t, int64_t, int64_t, int64_t,
                                                     const float *, const float *, int64_t,
                                                     const float *, int64_t, const float *, float *,
                                                     int64_t);
extern template gpublasStatus_t gemm<double, int>(gpublasHandle_t, gpublasOperation_t,
                                                  gpublasOperation_t, int, int, int, const double *,
                                                  const double *, int, const double *, int,
                                                  const double *, double *, int);
extern template gpublasStatus_t gemm<double, int64_t>(gpublasHandle_t, gpublasOperation_t,
                                                      gpublasOperation_t, int64_t, int64_t, int64_t,
                                                      const double *, const double *, int64_t,
                                                      const double *, int64_t, const double *,
                                                      double *, int64_t);
extern template gpublasStatus_t
gemm<gpuFloatComplex, int>(gpublasHandle_t, gpublasOperation_t, gpublasOperation_t, int, int, int,
                           const gpuFloatComplex *, const gpuFloatComplex *, int,
                           const gpuFloatComplex *, int, const gpuFloatComplex *, gpuFloatComplex *,
                           int);
extern template gpublasStatus_t
gemm<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasOperation_t, gpublasOperation_t, int64_t,
                               int64_t, int64_t, const gpuFloatComplex *, const gpuFloatComplex *,
                               int64_t, const gpuFloatComplex *, int64_t, const gpuFloatComplex *,
                               gpuFloatComplex *, int64_t);
extern template gpublasStatus_t
gemm<gpuDoubleComplex, int>(gpublasHandle_t, gpublasOperation_t, gpublasOperation_t, int, int, int,
                            const gpuDoubleComplex *, const gpuDoubleComplex *, int,
                            const gpuDoubleComplex *, int, const gpuDoubleComplex *,
                            gpuDoubleComplex *, int);
extern template gpublasStatus_t
gemm<gpuDoubleComplex, int64_t>(gpublasHandle_t, gpublasOperation_t, gpublasOperation_t, int64_t,
                                int64_t, int64_t, const gpuDoubleComplex *,
                                const gpuDoubleComplex *, int64_t, const gpuDoubleComplex *,
                                int64_t, const gpuDoubleComplex *, gpuDoubleComplex *, int64_t);

// Function: gemmBatched
extern template gpublasStatus_t gemmBatched<float, int>(gpublasHandle_t, gpublasOperation_t,
                                                        gpublasOperation_t, int, int, int,
                                                        const float *, const float *const[], int,
                                                        const float *const[], int, const float *,
                                                        float *const[], int, int);
extern template gpublasStatus_t
gemmBatched<float, int64_t>(gpublasHandle_t, gpublasOperation_t, gpublasOperation_t, int64_t,
                            int64_t, int64_t, const float *, const float *const[], int64_t,
                            const float *const[], int64_t, const float *, float *const[], int64_t,
                            int64_t);
extern template gpublasStatus_t gemmBatched<double, int>(gpublasHandle_t, gpublasOperation_t,
                                                         gpublasOperation_t, int, int, int,
                                                         const double *, const double *const[], int,
                                                         const double *const[], int, const double *,
                                                         double *const[], int, int);
extern template gpublasStatus_t
gemmBatched<double, int64_t>(gpublasHandle_t, gpublasOperation_t, gpublasOperation_t, int64_t,
                             int64_t, int64_t, const double *, const double *const[], int64_t,
                             const double *const[], int64_t, const double *, double *const[],
                             int64_t, int64_t);
extern template gpublasStatus_t
gemmBatched<gpuFloatComplex, int>(gpublasHandle_t, gpublasOperation_t, gpublasOperation_t, int, int,
                                  int, const gpuFloatComplex *, const gpuFloatComplex *const[], int,
                                  const gpuFloatComplex *const[], int, const gpuFloatComplex *,
                                  gpuFloatComplex *const[], int, int);
extern template gpublasStatus_t gemmBatched<gpuFloatComplex, int64_t>(
    gpublasHandle_t, gpublasOperation_t, gpublasOperation_t, int64_t, int64_t, int64_t,
    const gpuFloatComplex *, const gpuFloatComplex *const[], int64_t,
    const gpuFloatComplex *const[], int64_t, const gpuFloatComplex *, gpuFloatComplex *const[],
    int64_t, int64_t);
extern template gpublasStatus_t gemmBatched<gpuDoubleComplex, int>(
    gpublasHandle_t, gpublasOperation_t, gpublasOperation_t, int, int, int,
    const gpuDoubleComplex *, const gpuDoubleComplex *const[], int, const gpuDoubleComplex *const[],
    int, const gpuDoubleComplex *, gpuDoubleComplex *const[], int, int);
extern template gpublasStatus_t gemmBatched<gpuDoubleComplex, int64_t>(
    gpublasHandle_t, gpublasOperation_t, gpublasOperation_t, int64_t, int64_t, int64_t,
    const gpuDoubleComplex *, const gpuDoubleComplex *const[], int64_t,
    const gpuDoubleComplex *const[], int64_t, const gpuDoubleComplex *, gpuDoubleComplex *const[],
    int64_t, int64_t);

// Function: gemmStridedBatched
extern template gpublasStatus_t
gemmStridedBatched<float, int>(gpublasHandle_t, gpublasOperation_t, gpublasOperation_t, int, int,
                               int, const float *, const float *, int, long long int, const float *,
                               int, long long int, const float *, float *, int, long long int, int);
extern template gpublasStatus_t
gemmStridedBatched<float, int64_t>(gpublasHandle_t, gpublasOperation_t, gpublasOperation_t, int64_t,
                                   int64_t, int64_t, const float *, const float *, int64_t,
                                   long long int, const float *, int64_t, long long int,
                                   const float *, float *, int64_t, long long int, int64_t);
extern template gpublasStatus_t gemmStridedBatched<double, int>(gpublasHandle_t, gpublasOperation_t,
                                                                gpublasOperation_t, int, int, int,
                                                                const double *, const double *, int,
                                                                long long int, const double *, int,
                                                                long long int, const double *,
                                                                double *, int, long long int, int);
extern template gpublasStatus_t
gemmStridedBatched<double, int64_t>(gpublasHandle_t, gpublasOperation_t, gpublasOperation_t,
                                    int64_t, int64_t, int64_t, const double *, const double *,
                                    int64_t, long long int, const double *, int64_t, long long int,
                                    const double *, double *, int64_t, long long int, int64_t);
extern template gpublasStatus_t gemmStridedBatched<gpuFloatComplex, int>(
    gpublasHandle_t, gpublasOperation_t, gpublasOperation_t, int, int, int, const gpuFloatComplex *,
    const gpuFloatComplex *, int, long long int, const gpuFloatComplex *, int, long long int,
    const gpuFloatComplex *, gpuFloatComplex *, int, long long int, int);
extern template gpublasStatus_t gemmStridedBatched<gpuFloatComplex, int64_t>(
    gpublasHandle_t, gpublasOperation_t, gpublasOperation_t, int64_t, int64_t, int64_t,
    const gpuFloatComplex *, const gpuFloatComplex *, int64_t, long long int,
    const gpuFloatComplex *, int64_t, long long int, const gpuFloatComplex *, gpuFloatComplex *,
    int64_t, long long int, int64_t);
extern template gpublasStatus_t gemmStridedBatched<gpuDoubleComplex, int>(
    gpublasHandle_t, gpublasOperation_t, gpublasOperation_t, int, int, int,
    const gpuDoubleComplex *, const gpuDoubleComplex *, int, long long int,
    const gpuDoubleComplex *, int, long long int, const gpuDoubleComplex *, gpuDoubleComplex *, int,
    long long int, int);
extern template gpublasStatus_t gemmStridedBatched<gpuDoubleComplex, int64_t>(
    gpublasHandle_t, gpublasOperation_t, gpublasOperation_t, int64_t, int64_t, int64_t,
    const gpuDoubleComplex *, const gpuDoubleComplex *, int64_t, long long int,
    const gpuDoubleComplex *, int64_t, long long int, const gpuDoubleComplex *, gpuDoubleComplex *,
    int64_t, long long int, int64_t);

// Function: symm
extern template gpublasStatus_t symm<float, int>(gpublasHandle_t, gpublasSideMode_t,
                                                 gpublasFillMode_t, int, int, const float *,
                                                 const float *, int, const float *, int,
                                                 const float *, float *, int);
extern template gpublasStatus_t symm<float, int64_t>(gpublasHandle_t, gpublasSideMode_t,
                                                     gpublasFillMode_t, int64_t, int64_t,
                                                     const float *, const float *, int64_t,
                                                     const float *, int64_t, const float *, float *,
                                                     int64_t);
extern template gpublasStatus_t symm<double, int>(gpublasHandle_t, gpublasSideMode_t,
                                                  gpublasFillMode_t, int, int, const double *,
                                                  const double *, int, const double *, int,
                                                  const double *, double *, int);
extern template gpublasStatus_t symm<double, int64_t>(gpublasHandle_t, gpublasSideMode_t,
                                                      gpublasFillMode_t, int64_t, int64_t,
                                                      const double *, const double *, int64_t,
                                                      const double *, int64_t, const double *,
                                                      double *, int64_t);
extern template gpublasStatus_t
symm<gpuFloatComplex, int>(gpublasHandle_t, gpublasSideMode_t, gpublasFillMode_t, int, int,
                           const gpuFloatComplex *, const gpuFloatComplex *, int,
                           const gpuFloatComplex *, int, const gpuFloatComplex *, gpuFloatComplex *,
                           int);
extern template gpublasStatus_t
symm<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasSideMode_t, gpublasFillMode_t, int64_t,
                               int64_t, const gpuFloatComplex *, const gpuFloatComplex *, int64_t,
                               const gpuFloatComplex *, int64_t, const gpuFloatComplex *,
                               gpuFloatComplex *, int64_t);
extern template gpublasStatus_t
symm<gpuDoubleComplex, int>(gpublasHandle_t, gpublasSideMode_t, gpublasFillMode_t, int, int,
                            const gpuDoubleComplex *, const gpuDoubleComplex *, int,
                            const gpuDoubleComplex *, int, const gpuDoubleComplex *,
                            gpuDoubleComplex *, int);
extern template gpublasStatus_t
symm<gpuDoubleComplex, int64_t>(gpublasHandle_t, gpublasSideMode_t, gpublasFillMode_t, int64_t,
                                int64_t, const gpuDoubleComplex *, const gpuDoubleComplex *,
                                int64_t, const gpuDoubleComplex *, int64_t,
                                const gpuDoubleComplex *, gpuDoubleComplex *, int64_t);

// Function: syrk
extern template gpublasStatus_t syrk<float, int>(gpublasHandle_t, gpublasFillMode_t,
                                                 gpublasOperation_t, int, int, const float *,
                                                 const float *, int, const float *, float *, int);
extern template gpublasStatus_t syrk<float, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                     gpublasOperation_t, int64_t, int64_t,
                                                     const float *, const float *, int64_t,
                                                     const float *, float *, int64_t);
extern template gpublasStatus_t syrk<double, int>(gpublasHandle_t, gpublasFillMode_t,
                                                  gpublasOperation_t, int, int, const double *,
                                                  const double *, int, const double *, double *,
                                                  int);
extern template gpublasStatus_t syrk<double, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                      gpublasOperation_t, int64_t, int64_t,
                                                      const double *, const double *, int64_t,
                                                      const double *, double *, int64_t);
extern template gpublasStatus_t
syrk<gpuFloatComplex, int>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t, int, int,
                           const gpuFloatComplex *, const gpuFloatComplex *, int,
                           const gpuFloatComplex *, gpuFloatComplex *, int);
extern template gpublasStatus_t
syrk<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t, int64_t,
                               int64_t, const gpuFloatComplex *, const gpuFloatComplex *, int64_t,
                               const gpuFloatComplex *, gpuFloatComplex *, int64_t);
extern template gpublasStatus_t
syrk<gpuDoubleComplex, int>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t, int, int,
                            const gpuDoubleComplex *, const gpuDoubleComplex *, int,
                            const gpuDoubleComplex *, gpuDoubleComplex *, int);
extern template gpublasStatus_t
syrk<gpuDoubleComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t, int64_t,
                                int64_t, const gpuDoubleComplex *, const gpuDoubleComplex *,
                                int64_t, const gpuDoubleComplex *, gpuDoubleComplex *, int64_t);

// Function: syr2k
extern template gpublasStatus_t syr2k<float, int>(gpublasHandle_t, gpublasFillMode_t,
                                                  gpublasOperation_t, int, int, const float *,
                                                  const float *, int, const float *, int,
                                                  const float *, float *, int);
extern template gpublasStatus_t syr2k<float, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                      gpublasOperation_t, int64_t, int64_t,
                                                      const float *, const float *, int64_t,
                                                      const float *, int64_t, const float *,
                                                      float *, int64_t);
extern template gpublasStatus_t syr2k<double, int>(gpublasHandle_t, gpublasFillMode_t,
                                                   gpublasOperation_t, int, int, const double *,
                                                   const double *, int, const double *, int,
                                                   const double *, double *, int);
extern template gpublasStatus_t syr2k<double, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                       gpublasOperation_t, int64_t, int64_t,
                                                       const double *, const double *, int64_t,
                                                       const double *, int64_t, const double *,
                                                       double *, int64_t);
extern template gpublasStatus_t
syr2k<gpuFloatComplex, int>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t, int, int,
                            const gpuFloatComplex *, const gpuFloatComplex *, int,
                            const gpuFloatComplex *, int, const gpuFloatComplex *,
                            gpuFloatComplex *, int);
extern template gpublasStatus_t
syr2k<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t, int64_t,
                                int64_t, const gpuFloatComplex *, const gpuFloatComplex *, int64_t,
                                const gpuFloatComplex *, int64_t, const gpuFloatComplex *,
                                gpuFloatComplex *, int64_t);
extern template gpublasStatus_t
syr2k<gpuDoubleComplex, int>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t, int, int,
                             const gpuDoubleComplex *, const gpuDoubleComplex *, int,
                             const gpuDoubleComplex *, int, const gpuDoubleComplex *,
                             gpuDoubleComplex *, int);
extern template gpublasStatus_t
syr2k<gpuDoubleComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t, int64_t,
                                 int64_t, const gpuDoubleComplex *, const gpuDoubleComplex *,
                                 int64_t, const gpuDoubleComplex *, int64_t,
                                 const gpuDoubleComplex *, gpuDoubleComplex *, int64_t);

// Function: syrkx
extern template gpublasStatus_t syrkx<float, int>(gpublasHandle_t, gpublasFillMode_t,
                                                  gpublasOperation_t, int, int, const float *,
                                                  const float *, int, const float *, int,
                                                  const float *, float *, int);
extern template gpublasStatus_t syrkx<float, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                      gpublasOperation_t, int64_t, int64_t,
                                                      const float *, const float *, int64_t,
                                                      const float *, int64_t, const float *,
                                                      float *, int64_t);
extern template gpublasStatus_t syrkx<double, int>(gpublasHandle_t, gpublasFillMode_t,
                                                   gpublasOperation_t, int, int, const double *,
                                                   const double *, int, const double *, int,
                                                   const double *, double *, int);
extern template gpublasStatus_t syrkx<double, int64_t>(gpublasHandle_t, gpublasFillMode_t,
                                                       gpublasOperation_t, int64_t, int64_t,
                                                       const double *, const double *, int64_t,
                                                       const double *, int64_t, const double *,
                                                       double *, int64_t);
extern template gpublasStatus_t
syrkx<gpuFloatComplex, int>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t, int, int,
                            const gpuFloatComplex *, const gpuFloatComplex *, int,
                            const gpuFloatComplex *, int, const gpuFloatComplex *,
                            gpuFloatComplex *, int);
extern template gpublasStatus_t
syrkx<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t, int64_t,
                                int64_t, const gpuFloatComplex *, const gpuFloatComplex *, int64_t,
                                const gpuFloatComplex *, int64_t, const gpuFloatComplex *,
                                gpuFloatComplex *, int64_t);
extern template gpublasStatus_t
syrkx<gpuDoubleComplex, int>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t, int, int,
                             const gpuDoubleComplex *, const gpuDoubleComplex *, int,
                             const gpuDoubleComplex *, int, const gpuDoubleComplex *,
                             gpuDoubleComplex *, int);
extern template gpublasStatus_t
syrkx<gpuDoubleComplex, int64_t>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t, int64_t,
                                 int64_t, const gpuDoubleComplex *, const gpuDoubleComplex *,
                                 int64_t, const gpuDoubleComplex *, int64_t,
                                 const gpuDoubleComplex *, gpuDoubleComplex *, int64_t);

// Function: trmm
extern template gpublasStatus_t trmm<float, int>(gpublasHandle_t, gpublasSideMode_t,
                                                 gpublasFillMode_t, gpublasOperation_t,
                                                 gpublasDiagType_t, int, int, const float *,
                                                 const float *, int, const float *, int, float *,
                                                 int);
extern template gpublasStatus_t trmm<float, int64_t>(gpublasHandle_t, gpublasSideMode_t,
                                                     gpublasFillMode_t, gpublasOperation_t,
                                                     gpublasDiagType_t, int64_t, int64_t,
                                                     const float *, const float *, int64_t,
                                                     const float *, int64_t, float *, int64_t);
extern template gpublasStatus_t trmm<double, int>(gpublasHandle_t, gpublasSideMode_t,
                                                  gpublasFillMode_t, gpublasOperation_t,
                                                  gpublasDiagType_t, int, int, const double *,
                                                  const double *, int, const double *, int,
                                                  double *, int);
extern template gpublasStatus_t trmm<double, int64_t>(gpublasHandle_t, gpublasSideMode_t,
                                                      gpublasFillMode_t, gpublasOperation_t,
                                                      gpublasDiagType_t, int64_t, int64_t,
                                                      const double *, const double *, int64_t,
                                                      const double *, int64_t, double *, int64_t);
extern template gpublasStatus_t
trmm<gpuFloatComplex, int>(gpublasHandle_t, gpublasSideMode_t, gpublasFillMode_t,
                           gpublasOperation_t, gpublasDiagType_t, int, int, const gpuFloatComplex *,
                           const gpuFloatComplex *, int, const gpuFloatComplex *, int,
                           gpuFloatComplex *, int);
extern template gpublasStatus_t
trmm<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasSideMode_t, gpublasFillMode_t,
                               gpublasOperation_t, gpublasDiagType_t, int64_t, int64_t,
                               const gpuFloatComplex *, const gpuFloatComplex *, int64_t,
                               const gpuFloatComplex *, int64_t, gpuFloatComplex *, int64_t);
extern template gpublasStatus_t
trmm<gpuDoubleComplex, int>(gpublasHandle_t, gpublasSideMode_t, gpublasFillMode_t,
                            gpublasOperation_t, gpublasDiagType_t, int, int,
                            const gpuDoubleComplex *, const gpuDoubleComplex *, int,
                            const gpuDoubleComplex *, int, gpuDoubleComplex *, int);
extern template gpublasStatus_t
trmm<gpuDoubleComplex, int64_t>(gpublasHandle_t, gpublasSideMode_t, gpublasFillMode_t,
                                gpublasOperation_t, gpublasDiagType_t, int64_t, int64_t,
                                const gpuDoubleComplex *, const gpuDoubleComplex *, int64_t,
                                const gpuDoubleComplex *, int64_t, gpuDoubleComplex *, int64_t);

// Function: trsm
extern template gpublasStatus_t trsm<float, int>(gpublasHandle_t, gpublasSideMode_t,
                                                 gpublasFillMode_t, gpublasOperation_t,
                                                 gpublasDiagType_t, int, int, const float *,
                                                 const float *, int, float *, int);
extern template gpublasStatus_t trsm<float, int64_t>(gpublasHandle_t, gpublasSideMode_t,
                                                     gpublasFillMode_t, gpublasOperation_t,
                                                     gpublasDiagType_t, int64_t, int64_t,
                                                     const float *, const float *, int64_t, float *,
                                                     int64_t);
extern template gpublasStatus_t trsm<double, int>(gpublasHandle_t, gpublasSideMode_t,
                                                  gpublasFillMode_t, gpublasOperation_t,
                                                  gpublasDiagType_t, int, int, const double *,
                                                  const double *, int, double *, int);
extern template gpublasStatus_t trsm<double, int64_t>(gpublasHandle_t, gpublasSideMode_t,
                                                      gpublasFillMode_t, gpublasOperation_t,
                                                      gpublasDiagType_t, int64_t, int64_t,
                                                      const double *, const double *, int64_t,
                                                      double *, int64_t);
extern template gpublasStatus_t
trsm<gpuFloatComplex, int>(gpublasHandle_t, gpublasSideMode_t, gpublasFillMode_t,
                           gpublasOperation_t, gpublasDiagType_t, int, int, const gpuFloatComplex *,
                           const gpuFloatComplex *, int, gpuFloatComplex *, int);
extern template gpublasStatus_t
trsm<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasSideMode_t, gpublasFillMode_t,
                               gpublasOperation_t, gpublasDiagType_t, int64_t, int64_t,
                               const gpuFloatComplex *, const gpuFloatComplex *, int64_t,
                               gpuFloatComplex *, int64_t);
extern template gpublasStatus_t trsm<gpuDoubleComplex, int>(
    gpublasHandle_t, gpublasSideMode_t, gpublasFillMode_t, gpublasOperation_t, gpublasDiagType_t,
    int, int, const gpuDoubleComplex *, const gpuDoubleComplex *, int, gpuDoubleComplex *, int);
extern template gpublasStatus_t
trsm<gpuDoubleComplex, int64_t>(gpublasHandle_t, gpublasSideMode_t, gpublasFillMode_t,
                                gpublasOperation_t, gpublasDiagType_t, int64_t, int64_t,
                                const gpuDoubleComplex *, const gpuDoubleComplex *, int64_t,
                                gpuDoubleComplex *, int64_t);

// Function: trsmBatched
extern template gpublasStatus_t trsmBatched<float, int>(gpublasHandle_t, gpublasSideMode_t,
                                                        gpublasFillMode_t, gpublasOperation_t,
                                                        gpublasDiagType_t, int, int, const float *,
                                                        const float *const[], int, float *const[],
                                                        int, int);
extern template gpublasStatus_t
trsmBatched<float, int64_t>(gpublasHandle_t, gpublasSideMode_t, gpublasFillMode_t,
                            gpublasOperation_t, gpublasDiagType_t, int64_t, int64_t, const float *,
                            const float *const[], int64_t, float *const[], int64_t, int64_t);
extern template gpublasStatus_t trsmBatched<double, int>(gpublasHandle_t, gpublasSideMode_t,
                                                         gpublasFillMode_t, gpublasOperation_t,
                                                         gpublasDiagType_t, int, int,
                                                         const double *, const double *const[], int,
                                                         double *const[], int, int);
extern template gpublasStatus_t trsmBatched<double, int64_t>(gpublasHandle_t, gpublasSideMode_t,
                                                             gpublasFillMode_t, gpublasOperation_t,
                                                             gpublasDiagType_t, int64_t, int64_t,
                                                             const double *, const double *const[],
                                                             int64_t, double *const[], int64_t,
                                                             int64_t);
extern template gpublasStatus_t
trsmBatched<gpuFloatComplex, int>(gpublasHandle_t, gpublasSideMode_t, gpublasFillMode_t,
                                  gpublasOperation_t, gpublasDiagType_t, int, int,
                                  const gpuFloatComplex *, const gpuFloatComplex *const[], int,
                                  gpuFloatComplex *const[], int, int);
extern template gpublasStatus_t
trsmBatched<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasSideMode_t, gpublasFillMode_t,
                                      gpublasOperation_t, gpublasDiagType_t, int64_t, int64_t,
                                      const gpuFloatComplex *, const gpuFloatComplex *const[],
                                      int64_t, gpuFloatComplex *const[], int64_t, int64_t);
extern template gpublasStatus_t
trsmBatched<gpuDoubleComplex, int>(gpublasHandle_t, gpublasSideMode_t, gpublasFillMode_t,
                                   gpublasOperation_t, gpublasDiagType_t, int, int,
                                   const gpuDoubleComplex *, const gpuDoubleComplex *const[], int,
                                   gpuDoubleComplex *const[], int, int);
extern template gpublasStatus_t
trsmBatched<gpuDoubleComplex, int64_t>(gpublasHandle_t, gpublasSideMode_t, gpublasFillMode_t,
                                       gpublasOperation_t, gpublasDiagType_t, int64_t, int64_t,
                                       const gpuDoubleComplex *, const gpuDoubleComplex *const[],
                                       int64_t, gpuDoubleComplex *const[], int64_t, int64_t);

// Function: hemm
extern template gpublasStatus_t
hemm<gpuFloatComplex, int>(gpublasHandle_t, gpublasSideMode_t, gpublasFillMode_t, int, int,
                           const gpuFloatComplex *, const gpuFloatComplex *, int,
                           const gpuFloatComplex *, int, const gpuFloatComplex *, gpuFloatComplex *,
                           int);
extern template gpublasStatus_t
hemm<gpuFloatComplex, int64_t>(gpublasHandle_t, gpublasSideMode_t, gpublasFillMode_t, int64_t,
                               int64_t, const gpuFloatComplex *, const gpuFloatComplex *, int64_t,
                               const gpuFloatComplex *, int64_t, const gpuFloatComplex *,
                               gpuFloatComplex *, int64_t);
extern template gpublasStatus_t
hemm<gpuDoubleComplex, int>(gpublasHandle_t, gpublasSideMode_t, gpublasFillMode_t, int, int,
                            const gpuDoubleComplex *, const gpuDoubleComplex *, int,
                            const gpuDoubleComplex *, int, const gpuDoubleComplex *,
                            gpuDoubleComplex *, int);
extern template gpublasStatus_t
hemm<gpuDoubleComplex, int64_t>(gpublasHandle_t, gpublasSideMode_t, gpublasFillMode_t, int64_t,
                                int64_t, const gpuDoubleComplex *, const gpuDoubleComplex *,
                                int64_t, const gpuDoubleComplex *, int64_t,
                                const gpuDoubleComplex *, gpuDoubleComplex *, int64_t);

// Function: herk
extern template gpublasStatus_t
herk<gpuFloatComplex, int>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t, int, int,
                           const ComplexToRealType<gpuFloatComplex> *, const gpuFloatComplex *, int,
                           const ComplexToRealType<gpuFloatComplex> *, gpuFloatComplex *, int);
extern template gpublasStatus_t herk<gpuFloatComplex, int64_t>(
    gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t, int64_t, int64_t,
    const ComplexToRealType<gpuFloatComplex> *, const gpuFloatComplex *, int64_t,
    const ComplexToRealType<gpuFloatComplex> *, gpuFloatComplex *, int64_t);
extern template gpublasStatus_t
herk<gpuDoubleComplex, int>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t, int, int,
                            const ComplexToRealType<gpuDoubleComplex> *, const gpuDoubleComplex *,
                            int, const ComplexToRealType<gpuDoubleComplex> *, gpuDoubleComplex *,
                            int);
extern template gpublasStatus_t herk<gpuDoubleComplex, int64_t>(
    gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t, int64_t, int64_t,
    const ComplexToRealType<gpuDoubleComplex> *, const gpuDoubleComplex *, int64_t,
    const ComplexToRealType<gpuDoubleComplex> *, gpuDoubleComplex *, int64_t);

// Function: her2k
extern template gpublasStatus_t
her2k<gpuFloatComplex, int>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t, int, int,
                            const gpuFloatComplex *, const gpuFloatComplex *, int,
                            const gpuFloatComplex *, int,
                            const ComplexToRealType<gpuFloatComplex> *, gpuFloatComplex *, int);
extern template gpublasStatus_t her2k<gpuFloatComplex, int64_t>(
    gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t, int64_t, int64_t,
    const gpuFloatComplex *, const gpuFloatComplex *, int64_t, const gpuFloatComplex *, int64_t,
    const ComplexToRealType<gpuFloatComplex> *, gpuFloatComplex *, int64_t);
extern template gpublasStatus_t
her2k<gpuDoubleComplex, int>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t, int, int,
                             const gpuDoubleComplex *, const gpuDoubleComplex *, int,
                             const gpuDoubleComplex *, int,
                             const ComplexToRealType<gpuDoubleComplex> *, gpuDoubleComplex *, int);
extern template gpublasStatus_t her2k<gpuDoubleComplex, int64_t>(
    gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t, int64_t, int64_t,
    const gpuDoubleComplex *, const gpuDoubleComplex *, int64_t, const gpuDoubleComplex *, int64_t,
    const ComplexToRealType<gpuDoubleComplex> *, gpuDoubleComplex *, int64_t);

// Function: herkx
extern template gpublasStatus_t
herkx<gpuFloatComplex, int>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t, int, int,
                            const gpuFloatComplex *, const gpuFloatComplex *, int,
                            const gpuFloatComplex *, int,
                            const ComplexToRealType<gpuFloatComplex> *, gpuFloatComplex *, int);
extern template gpublasStatus_t herkx<gpuFloatComplex, int64_t>(
    gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t, int64_t, int64_t,
    const gpuFloatComplex *, const gpuFloatComplex *, int64_t, const gpuFloatComplex *, int64_t,
    const ComplexToRealType<gpuFloatComplex> *, gpuFloatComplex *, int64_t);
extern template gpublasStatus_t
herkx<gpuDoubleComplex, int>(gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t, int, int,
                             const gpuDoubleComplex *, const gpuDoubleComplex *, int,
                             const gpuDoubleComplex *, int,
                             const ComplexToRealType<gpuDoubleComplex> *, gpuDoubleComplex *, int);
extern template gpublasStatus_t herkx<gpuDoubleComplex, int64_t>(
    gpublasHandle_t, gpublasFillMode_t, gpublasOperation_t, int64_t, int64_t,
    const gpuDoubleComplex *, const gpuDoubleComplex *, int64_t, const gpuDoubleComplex *, int64_t,
    const ComplexToRealType<gpuDoubleComplex> *, gpuDoubleComplex *, int64_t);

} // namespace wwr
