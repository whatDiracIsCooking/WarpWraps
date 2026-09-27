/**
 * @file level_3.cppm
 * @brief GPU BLAS Level 3 (matrix-matrix) operations
 *
 * This module provides type-safe wrappers for GPU BLAS Level 3 BLAS operations.
 *
 * Usage:
 *   import wwr.wrappers.blas;
 */

module;

#include "dispatch_macros.h"

export module wwr.wrappers.blas:level_3;

import wwr.blas;
import wwr.complex;
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
 * @param transa Operation on matrix A (WWRBLAS_OP_N, WWRBLAS_OP_T, WWRBLAS_OP_C)
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
wwrblasStatus_t gemm(wwrblasHandle_t handle, wwrblasOperation_t transa, wwrblasOperation_t transb,
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
wwrblasStatus_t gemmBatched(wwrblasHandle_t handle, wwrblasOperation_t transa,
                            wwrblasOperation_t transb, IntT m, IntT n, IntT k, const T *alpha,
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
wwrblasStatus_t gemmStridedBatched(wwrblasHandle_t handle, wwrblasOperation_t transa,
                                   wwrblasOperation_t transb, IntT m, IntT n, IntT k,
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
wwrblasStatus_t symm(wwrblasHandle_t handle, wwrblasSideMode_t side, wwrblasFillMode_t uplo, IntT m,
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
 * @param trans Operation on matrix A (WWRBLAS_OP_N or WWRBLAS_OP_T)
 */
template<usual_fp T, int_type IntT>
wwrblasStatus_t syrk(wwrblasHandle_t handle, wwrblasFillMode_t uplo, wwrblasOperation_t trans,
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
wwrblasStatus_t syr2k(wwrblasHandle_t handle, wwrblasFillMode_t uplo, wwrblasOperation_t trans,
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
wwrblasStatus_t syrkx(wwrblasHandle_t handle, wwrblasFillMode_t uplo, wwrblasOperation_t trans,
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
wwrblasStatus_t trmm(wwrblasHandle_t handle, wwrblasSideMode_t side, wwrblasFillMode_t uplo,
                     wwrblasOperation_t trans, wwrblasDiagType_t diag, IntT m, IntT n,
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
wwrblasStatus_t trsm(wwrblasHandle_t handle, wwrblasSideMode_t side, wwrblasFillMode_t uplo,
                     wwrblasOperation_t trans, wwrblasDiagType_t diag, IntT m, IntT n,
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
wwrblasStatus_t trsmBatched(wwrblasHandle_t handle, wwrblasSideMode_t side, wwrblasFillMode_t uplo,
                            wwrblasOperation_t trans, wwrblasDiagType_t diag, IntT m, IntT n,
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
wwrblasStatus_t hemm(wwrblasHandle_t handle, wwrblasSideMode_t side, wwrblasFillMode_t uplo, IntT m,
                     IntT n, const T *alpha, const T *A, IntT lda, const T *B, IntT ldb,
                     const T *beta, T *C, IntT ldc) {
  WWR_COMPLEX_DISPATCH_64(T, IntT, wwrblas, C, Z, hemm, handle, side, uplo, m, n, alpha, A, lda,
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
 * @param trans Operation on matrix A (WWRBLAS_OP_N or WWRBLAS_OP_C)
 */
template<complex_fp T, int_type IntT>
wwrblasStatus_t herk(wwrblasHandle_t handle, wwrblasFillMode_t uplo, wwrblasOperation_t trans,
                     IntT n, IntT k, const ComplexToRealType<T> *alpha, const T *A, IntT lda,
                     const ComplexToRealType<T> *beta, T *C, IntT ldc) {
  WWR_COMPLEX_DISPATCH_64(T, IntT, wwrblas, C, Z, herk, handle, uplo, trans, n, k, alpha, A, lda,
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
wwrblasStatus_t her2k(wwrblasHandle_t handle, wwrblasFillMode_t uplo, wwrblasOperation_t trans,
                      IntT n, IntT k, const T *alpha, const T *A, IntT lda, const T *B, IntT ldb,
                      const ComplexToRealType<T> *beta, T *C, IntT ldc) {
  WWR_COMPLEX_DISPATCH_64(T, IntT, wwrblas, C, Z, her2k, handle, uplo, trans, n, k, alpha, A,
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
wwrblasStatus_t herkx(wwrblasHandle_t handle, wwrblasFillMode_t uplo, wwrblasOperation_t trans,
                      IntT n, IntT k, const T *alpha, const T *A, IntT lda, const T *B, IntT ldb,
                      const ComplexToRealType<T> *beta, T *C, IntT ldc) {
  WWR_COMPLEX_DISPATCH_64(T, IntT, wwrblas, C, Z, herkx, handle, uplo, trans, n, k, alpha, A,
                             lda, B, ldb, beta, C, ldc);
}

// ==================== Explicit Template Instantiations ====================
// Matching `template` instantiations live in instantiations.cpp.

// Function: gemm
extern template wwrblasStatus_t gemm<float, int>(wwrblasHandle_t, wwrblasOperation_t,
                                                 wwrblasOperation_t, int, int, int, const float *,
                                                 const float *, int, const float *, int,
                                                 const float *, float *, int);
extern template wwrblasStatus_t gemm<float, int64_t>(wwrblasHandle_t, wwrblasOperation_t,
                                                     wwrblasOperation_t, int64_t, int64_t, int64_t,
                                                     const float *, const float *, int64_t,
                                                     const float *, int64_t, const float *, float *,
                                                     int64_t);
extern template wwrblasStatus_t gemm<double, int>(wwrblasHandle_t, wwrblasOperation_t,
                                                  wwrblasOperation_t, int, int, int, const double *,
                                                  const double *, int, const double *, int,
                                                  const double *, double *, int);
extern template wwrblasStatus_t gemm<double, int64_t>(wwrblasHandle_t, wwrblasOperation_t,
                                                      wwrblasOperation_t, int64_t, int64_t, int64_t,
                                                      const double *, const double *, int64_t,
                                                      const double *, int64_t, const double *,
                                                      double *, int64_t);
extern template wwrblasStatus_t
gemm<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasOperation_t, wwrblasOperation_t, int, int, int,
                           const wwrFloatComplex *, const wwrFloatComplex *, int,
                           const wwrFloatComplex *, int, const wwrFloatComplex *, wwrFloatComplex *,
                           int);
extern template wwrblasStatus_t
gemm<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasOperation_t, wwrblasOperation_t, int64_t,
                               int64_t, int64_t, const wwrFloatComplex *, const wwrFloatComplex *,
                               int64_t, const wwrFloatComplex *, int64_t, const wwrFloatComplex *,
                               wwrFloatComplex *, int64_t);
extern template wwrblasStatus_t
gemm<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasOperation_t, wwrblasOperation_t, int, int, int,
                            const wwrDoubleComplex *, const wwrDoubleComplex *, int,
                            const wwrDoubleComplex *, int, const wwrDoubleComplex *,
                            wwrDoubleComplex *, int);
extern template wwrblasStatus_t
gemm<wwrDoubleComplex, int64_t>(wwrblasHandle_t, wwrblasOperation_t, wwrblasOperation_t, int64_t,
                                int64_t, int64_t, const wwrDoubleComplex *,
                                const wwrDoubleComplex *, int64_t, const wwrDoubleComplex *,
                                int64_t, const wwrDoubleComplex *, wwrDoubleComplex *, int64_t);

// Function: gemmBatched
extern template wwrblasStatus_t gemmBatched<float, int>(wwrblasHandle_t, wwrblasOperation_t,
                                                        wwrblasOperation_t, int, int, int,
                                                        const float *, const float *const[], int,
                                                        const float *const[], int, const float *,
                                                        float *const[], int, int);
extern template wwrblasStatus_t
gemmBatched<float, int64_t>(wwrblasHandle_t, wwrblasOperation_t, wwrblasOperation_t, int64_t,
                            int64_t, int64_t, const float *, const float *const[], int64_t,
                            const float *const[], int64_t, const float *, float *const[], int64_t,
                            int64_t);
extern template wwrblasStatus_t gemmBatched<double, int>(wwrblasHandle_t, wwrblasOperation_t,
                                                         wwrblasOperation_t, int, int, int,
                                                         const double *, const double *const[], int,
                                                         const double *const[], int, const double *,
                                                         double *const[], int, int);
extern template wwrblasStatus_t
gemmBatched<double, int64_t>(wwrblasHandle_t, wwrblasOperation_t, wwrblasOperation_t, int64_t,
                             int64_t, int64_t, const double *, const double *const[], int64_t,
                             const double *const[], int64_t, const double *, double *const[],
                             int64_t, int64_t);
extern template wwrblasStatus_t
gemmBatched<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasOperation_t, wwrblasOperation_t, int, int,
                                  int, const wwrFloatComplex *, const wwrFloatComplex *const[], int,
                                  const wwrFloatComplex *const[], int, const wwrFloatComplex *,
                                  wwrFloatComplex *const[], int, int);
extern template wwrblasStatus_t gemmBatched<wwrFloatComplex, int64_t>(
    wwrblasHandle_t, wwrblasOperation_t, wwrblasOperation_t, int64_t, int64_t, int64_t,
    const wwrFloatComplex *, const wwrFloatComplex *const[], int64_t,
    const wwrFloatComplex *const[], int64_t, const wwrFloatComplex *, wwrFloatComplex *const[],
    int64_t, int64_t);
extern template wwrblasStatus_t gemmBatched<wwrDoubleComplex, int>(
    wwrblasHandle_t, wwrblasOperation_t, wwrblasOperation_t, int, int, int,
    const wwrDoubleComplex *, const wwrDoubleComplex *const[], int, const wwrDoubleComplex *const[],
    int, const wwrDoubleComplex *, wwrDoubleComplex *const[], int, int);
extern template wwrblasStatus_t gemmBatched<wwrDoubleComplex, int64_t>(
    wwrblasHandle_t, wwrblasOperation_t, wwrblasOperation_t, int64_t, int64_t, int64_t,
    const wwrDoubleComplex *, const wwrDoubleComplex *const[], int64_t,
    const wwrDoubleComplex *const[], int64_t, const wwrDoubleComplex *, wwrDoubleComplex *const[],
    int64_t, int64_t);

// Function: gemmStridedBatched
extern template wwrblasStatus_t
gemmStridedBatched<float, int>(wwrblasHandle_t, wwrblasOperation_t, wwrblasOperation_t, int, int,
                               int, const float *, const float *, int, long long int, const float *,
                               int, long long int, const float *, float *, int, long long int, int);
extern template wwrblasStatus_t
gemmStridedBatched<float, int64_t>(wwrblasHandle_t, wwrblasOperation_t, wwrblasOperation_t, int64_t,
                                   int64_t, int64_t, const float *, const float *, int64_t,
                                   long long int, const float *, int64_t, long long int,
                                   const float *, float *, int64_t, long long int, int64_t);
extern template wwrblasStatus_t gemmStridedBatched<double, int>(wwrblasHandle_t, wwrblasOperation_t,
                                                                wwrblasOperation_t, int, int, int,
                                                                const double *, const double *, int,
                                                                long long int, const double *, int,
                                                                long long int, const double *,
                                                                double *, int, long long int, int);
extern template wwrblasStatus_t
gemmStridedBatched<double, int64_t>(wwrblasHandle_t, wwrblasOperation_t, wwrblasOperation_t,
                                    int64_t, int64_t, int64_t, const double *, const double *,
                                    int64_t, long long int, const double *, int64_t, long long int,
                                    const double *, double *, int64_t, long long int, int64_t);
extern template wwrblasStatus_t gemmStridedBatched<wwrFloatComplex, int>(
    wwrblasHandle_t, wwrblasOperation_t, wwrblasOperation_t, int, int, int, const wwrFloatComplex *,
    const wwrFloatComplex *, int, long long int, const wwrFloatComplex *, int, long long int,
    const wwrFloatComplex *, wwrFloatComplex *, int, long long int, int);
extern template wwrblasStatus_t gemmStridedBatched<wwrFloatComplex, int64_t>(
    wwrblasHandle_t, wwrblasOperation_t, wwrblasOperation_t, int64_t, int64_t, int64_t,
    const wwrFloatComplex *, const wwrFloatComplex *, int64_t, long long int,
    const wwrFloatComplex *, int64_t, long long int, const wwrFloatComplex *, wwrFloatComplex *,
    int64_t, long long int, int64_t);
extern template wwrblasStatus_t gemmStridedBatched<wwrDoubleComplex, int>(
    wwrblasHandle_t, wwrblasOperation_t, wwrblasOperation_t, int, int, int,
    const wwrDoubleComplex *, const wwrDoubleComplex *, int, long long int,
    const wwrDoubleComplex *, int, long long int, const wwrDoubleComplex *, wwrDoubleComplex *, int,
    long long int, int);
extern template wwrblasStatus_t gemmStridedBatched<wwrDoubleComplex, int64_t>(
    wwrblasHandle_t, wwrblasOperation_t, wwrblasOperation_t, int64_t, int64_t, int64_t,
    const wwrDoubleComplex *, const wwrDoubleComplex *, int64_t, long long int,
    const wwrDoubleComplex *, int64_t, long long int, const wwrDoubleComplex *, wwrDoubleComplex *,
    int64_t, long long int, int64_t);

// Function: symm
extern template wwrblasStatus_t symm<float, int>(wwrblasHandle_t, wwrblasSideMode_t,
                                                 wwrblasFillMode_t, int, int, const float *,
                                                 const float *, int, const float *, int,
                                                 const float *, float *, int);
extern template wwrblasStatus_t symm<float, int64_t>(wwrblasHandle_t, wwrblasSideMode_t,
                                                     wwrblasFillMode_t, int64_t, int64_t,
                                                     const float *, const float *, int64_t,
                                                     const float *, int64_t, const float *, float *,
                                                     int64_t);
extern template wwrblasStatus_t symm<double, int>(wwrblasHandle_t, wwrblasSideMode_t,
                                                  wwrblasFillMode_t, int, int, const double *,
                                                  const double *, int, const double *, int,
                                                  const double *, double *, int);
extern template wwrblasStatus_t symm<double, int64_t>(wwrblasHandle_t, wwrblasSideMode_t,
                                                      wwrblasFillMode_t, int64_t, int64_t,
                                                      const double *, const double *, int64_t,
                                                      const double *, int64_t, const double *,
                                                      double *, int64_t);
extern template wwrblasStatus_t
symm<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasSideMode_t, wwrblasFillMode_t, int, int,
                           const wwrFloatComplex *, const wwrFloatComplex *, int,
                           const wwrFloatComplex *, int, const wwrFloatComplex *, wwrFloatComplex *,
                           int);
extern template wwrblasStatus_t
symm<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasSideMode_t, wwrblasFillMode_t, int64_t,
                               int64_t, const wwrFloatComplex *, const wwrFloatComplex *, int64_t,
                               const wwrFloatComplex *, int64_t, const wwrFloatComplex *,
                               wwrFloatComplex *, int64_t);
extern template wwrblasStatus_t
symm<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasSideMode_t, wwrblasFillMode_t, int, int,
                            const wwrDoubleComplex *, const wwrDoubleComplex *, int,
                            const wwrDoubleComplex *, int, const wwrDoubleComplex *,
                            wwrDoubleComplex *, int);
extern template wwrblasStatus_t
symm<wwrDoubleComplex, int64_t>(wwrblasHandle_t, wwrblasSideMode_t, wwrblasFillMode_t, int64_t,
                                int64_t, const wwrDoubleComplex *, const wwrDoubleComplex *,
                                int64_t, const wwrDoubleComplex *, int64_t,
                                const wwrDoubleComplex *, wwrDoubleComplex *, int64_t);

// Function: syrk
extern template wwrblasStatus_t syrk<float, int>(wwrblasHandle_t, wwrblasFillMode_t,
                                                 wwrblasOperation_t, int, int, const float *,
                                                 const float *, int, const float *, float *, int);
extern template wwrblasStatus_t syrk<float, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                     wwrblasOperation_t, int64_t, int64_t,
                                                     const float *, const float *, int64_t,
                                                     const float *, float *, int64_t);
extern template wwrblasStatus_t syrk<double, int>(wwrblasHandle_t, wwrblasFillMode_t,
                                                  wwrblasOperation_t, int, int, const double *,
                                                  const double *, int, const double *, double *,
                                                  int);
extern template wwrblasStatus_t syrk<double, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                      wwrblasOperation_t, int64_t, int64_t,
                                                      const double *, const double *, int64_t,
                                                      const double *, double *, int64_t);
extern template wwrblasStatus_t
syrk<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t, int, int,
                           const wwrFloatComplex *, const wwrFloatComplex *, int,
                           const wwrFloatComplex *, wwrFloatComplex *, int);
extern template wwrblasStatus_t
syrk<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t, int64_t,
                               int64_t, const wwrFloatComplex *, const wwrFloatComplex *, int64_t,
                               const wwrFloatComplex *, wwrFloatComplex *, int64_t);
extern template wwrblasStatus_t
syrk<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t, int, int,
                            const wwrDoubleComplex *, const wwrDoubleComplex *, int,
                            const wwrDoubleComplex *, wwrDoubleComplex *, int);
extern template wwrblasStatus_t
syrk<wwrDoubleComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t, int64_t,
                                int64_t, const wwrDoubleComplex *, const wwrDoubleComplex *,
                                int64_t, const wwrDoubleComplex *, wwrDoubleComplex *, int64_t);

// Function: syr2k
extern template wwrblasStatus_t syr2k<float, int>(wwrblasHandle_t, wwrblasFillMode_t,
                                                  wwrblasOperation_t, int, int, const float *,
                                                  const float *, int, const float *, int,
                                                  const float *, float *, int);
extern template wwrblasStatus_t syr2k<float, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                      wwrblasOperation_t, int64_t, int64_t,
                                                      const float *, const float *, int64_t,
                                                      const float *, int64_t, const float *,
                                                      float *, int64_t);
extern template wwrblasStatus_t syr2k<double, int>(wwrblasHandle_t, wwrblasFillMode_t,
                                                   wwrblasOperation_t, int, int, const double *,
                                                   const double *, int, const double *, int,
                                                   const double *, double *, int);
extern template wwrblasStatus_t syr2k<double, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                       wwrblasOperation_t, int64_t, int64_t,
                                                       const double *, const double *, int64_t,
                                                       const double *, int64_t, const double *,
                                                       double *, int64_t);
extern template wwrblasStatus_t
syr2k<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t, int, int,
                            const wwrFloatComplex *, const wwrFloatComplex *, int,
                            const wwrFloatComplex *, int, const wwrFloatComplex *,
                            wwrFloatComplex *, int);
extern template wwrblasStatus_t
syr2k<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t, int64_t,
                                int64_t, const wwrFloatComplex *, const wwrFloatComplex *, int64_t,
                                const wwrFloatComplex *, int64_t, const wwrFloatComplex *,
                                wwrFloatComplex *, int64_t);
extern template wwrblasStatus_t
syr2k<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t, int, int,
                             const wwrDoubleComplex *, const wwrDoubleComplex *, int,
                             const wwrDoubleComplex *, int, const wwrDoubleComplex *,
                             wwrDoubleComplex *, int);
extern template wwrblasStatus_t
syr2k<wwrDoubleComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t, int64_t,
                                 int64_t, const wwrDoubleComplex *, const wwrDoubleComplex *,
                                 int64_t, const wwrDoubleComplex *, int64_t,
                                 const wwrDoubleComplex *, wwrDoubleComplex *, int64_t);

// Function: syrkx
extern template wwrblasStatus_t syrkx<float, int>(wwrblasHandle_t, wwrblasFillMode_t,
                                                  wwrblasOperation_t, int, int, const float *,
                                                  const float *, int, const float *, int,
                                                  const float *, float *, int);
extern template wwrblasStatus_t syrkx<float, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                      wwrblasOperation_t, int64_t, int64_t,
                                                      const float *, const float *, int64_t,
                                                      const float *, int64_t, const float *,
                                                      float *, int64_t);
extern template wwrblasStatus_t syrkx<double, int>(wwrblasHandle_t, wwrblasFillMode_t,
                                                   wwrblasOperation_t, int, int, const double *,
                                                   const double *, int, const double *, int,
                                                   const double *, double *, int);
extern template wwrblasStatus_t syrkx<double, int64_t>(wwrblasHandle_t, wwrblasFillMode_t,
                                                       wwrblasOperation_t, int64_t, int64_t,
                                                       const double *, const double *, int64_t,
                                                       const double *, int64_t, const double *,
                                                       double *, int64_t);
extern template wwrblasStatus_t
syrkx<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t, int, int,
                            const wwrFloatComplex *, const wwrFloatComplex *, int,
                            const wwrFloatComplex *, int, const wwrFloatComplex *,
                            wwrFloatComplex *, int);
extern template wwrblasStatus_t
syrkx<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t, int64_t,
                                int64_t, const wwrFloatComplex *, const wwrFloatComplex *, int64_t,
                                const wwrFloatComplex *, int64_t, const wwrFloatComplex *,
                                wwrFloatComplex *, int64_t);
extern template wwrblasStatus_t
syrkx<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t, int, int,
                             const wwrDoubleComplex *, const wwrDoubleComplex *, int,
                             const wwrDoubleComplex *, int, const wwrDoubleComplex *,
                             wwrDoubleComplex *, int);
extern template wwrblasStatus_t
syrkx<wwrDoubleComplex, int64_t>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t, int64_t,
                                 int64_t, const wwrDoubleComplex *, const wwrDoubleComplex *,
                                 int64_t, const wwrDoubleComplex *, int64_t,
                                 const wwrDoubleComplex *, wwrDoubleComplex *, int64_t);

// Function: trmm
extern template wwrblasStatus_t trmm<float, int>(wwrblasHandle_t, wwrblasSideMode_t,
                                                 wwrblasFillMode_t, wwrblasOperation_t,
                                                 wwrblasDiagType_t, int, int, const float *,
                                                 const float *, int, const float *, int, float *,
                                                 int);
extern template wwrblasStatus_t trmm<float, int64_t>(wwrblasHandle_t, wwrblasSideMode_t,
                                                     wwrblasFillMode_t, wwrblasOperation_t,
                                                     wwrblasDiagType_t, int64_t, int64_t,
                                                     const float *, const float *, int64_t,
                                                     const float *, int64_t, float *, int64_t);
extern template wwrblasStatus_t trmm<double, int>(wwrblasHandle_t, wwrblasSideMode_t,
                                                  wwrblasFillMode_t, wwrblasOperation_t,
                                                  wwrblasDiagType_t, int, int, const double *,
                                                  const double *, int, const double *, int,
                                                  double *, int);
extern template wwrblasStatus_t trmm<double, int64_t>(wwrblasHandle_t, wwrblasSideMode_t,
                                                      wwrblasFillMode_t, wwrblasOperation_t,
                                                      wwrblasDiagType_t, int64_t, int64_t,
                                                      const double *, const double *, int64_t,
                                                      const double *, int64_t, double *, int64_t);
extern template wwrblasStatus_t
trmm<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasSideMode_t, wwrblasFillMode_t,
                           wwrblasOperation_t, wwrblasDiagType_t, int, int, const wwrFloatComplex *,
                           const wwrFloatComplex *, int, const wwrFloatComplex *, int,
                           wwrFloatComplex *, int);
extern template wwrblasStatus_t
trmm<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasSideMode_t, wwrblasFillMode_t,
                               wwrblasOperation_t, wwrblasDiagType_t, int64_t, int64_t,
                               const wwrFloatComplex *, const wwrFloatComplex *, int64_t,
                               const wwrFloatComplex *, int64_t, wwrFloatComplex *, int64_t);
extern template wwrblasStatus_t
trmm<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasSideMode_t, wwrblasFillMode_t,
                            wwrblasOperation_t, wwrblasDiagType_t, int, int,
                            const wwrDoubleComplex *, const wwrDoubleComplex *, int,
                            const wwrDoubleComplex *, int, wwrDoubleComplex *, int);
extern template wwrblasStatus_t
trmm<wwrDoubleComplex, int64_t>(wwrblasHandle_t, wwrblasSideMode_t, wwrblasFillMode_t,
                                wwrblasOperation_t, wwrblasDiagType_t, int64_t, int64_t,
                                const wwrDoubleComplex *, const wwrDoubleComplex *, int64_t,
                                const wwrDoubleComplex *, int64_t, wwrDoubleComplex *, int64_t);

// Function: trsm
extern template wwrblasStatus_t trsm<float, int>(wwrblasHandle_t, wwrblasSideMode_t,
                                                 wwrblasFillMode_t, wwrblasOperation_t,
                                                 wwrblasDiagType_t, int, int, const float *,
                                                 const float *, int, float *, int);
extern template wwrblasStatus_t trsm<float, int64_t>(wwrblasHandle_t, wwrblasSideMode_t,
                                                     wwrblasFillMode_t, wwrblasOperation_t,
                                                     wwrblasDiagType_t, int64_t, int64_t,
                                                     const float *, const float *, int64_t, float *,
                                                     int64_t);
extern template wwrblasStatus_t trsm<double, int>(wwrblasHandle_t, wwrblasSideMode_t,
                                                  wwrblasFillMode_t, wwrblasOperation_t,
                                                  wwrblasDiagType_t, int, int, const double *,
                                                  const double *, int, double *, int);
extern template wwrblasStatus_t trsm<double, int64_t>(wwrblasHandle_t, wwrblasSideMode_t,
                                                      wwrblasFillMode_t, wwrblasOperation_t,
                                                      wwrblasDiagType_t, int64_t, int64_t,
                                                      const double *, const double *, int64_t,
                                                      double *, int64_t);
extern template wwrblasStatus_t
trsm<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasSideMode_t, wwrblasFillMode_t,
                           wwrblasOperation_t, wwrblasDiagType_t, int, int, const wwrFloatComplex *,
                           const wwrFloatComplex *, int, wwrFloatComplex *, int);
extern template wwrblasStatus_t
trsm<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasSideMode_t, wwrblasFillMode_t,
                               wwrblasOperation_t, wwrblasDiagType_t, int64_t, int64_t,
                               const wwrFloatComplex *, const wwrFloatComplex *, int64_t,
                               wwrFloatComplex *, int64_t);
extern template wwrblasStatus_t trsm<wwrDoubleComplex, int>(
    wwrblasHandle_t, wwrblasSideMode_t, wwrblasFillMode_t, wwrblasOperation_t, wwrblasDiagType_t,
    int, int, const wwrDoubleComplex *, const wwrDoubleComplex *, int, wwrDoubleComplex *, int);
extern template wwrblasStatus_t
trsm<wwrDoubleComplex, int64_t>(wwrblasHandle_t, wwrblasSideMode_t, wwrblasFillMode_t,
                                wwrblasOperation_t, wwrblasDiagType_t, int64_t, int64_t,
                                const wwrDoubleComplex *, const wwrDoubleComplex *, int64_t,
                                wwrDoubleComplex *, int64_t);

// Function: trsmBatched
extern template wwrblasStatus_t trsmBatched<float, int>(wwrblasHandle_t, wwrblasSideMode_t,
                                                        wwrblasFillMode_t, wwrblasOperation_t,
                                                        wwrblasDiagType_t, int, int, const float *,
                                                        const float *const[], int, float *const[],
                                                        int, int);
extern template wwrblasStatus_t
trsmBatched<float, int64_t>(wwrblasHandle_t, wwrblasSideMode_t, wwrblasFillMode_t,
                            wwrblasOperation_t, wwrblasDiagType_t, int64_t, int64_t, const float *,
                            const float *const[], int64_t, float *const[], int64_t, int64_t);
extern template wwrblasStatus_t trsmBatched<double, int>(wwrblasHandle_t, wwrblasSideMode_t,
                                                         wwrblasFillMode_t, wwrblasOperation_t,
                                                         wwrblasDiagType_t, int, int,
                                                         const double *, const double *const[], int,
                                                         double *const[], int, int);
extern template wwrblasStatus_t trsmBatched<double, int64_t>(wwrblasHandle_t, wwrblasSideMode_t,
                                                             wwrblasFillMode_t, wwrblasOperation_t,
                                                             wwrblasDiagType_t, int64_t, int64_t,
                                                             const double *, const double *const[],
                                                             int64_t, double *const[], int64_t,
                                                             int64_t);
extern template wwrblasStatus_t
trsmBatched<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasSideMode_t, wwrblasFillMode_t,
                                  wwrblasOperation_t, wwrblasDiagType_t, int, int,
                                  const wwrFloatComplex *, const wwrFloatComplex *const[], int,
                                  wwrFloatComplex *const[], int, int);
extern template wwrblasStatus_t
trsmBatched<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasSideMode_t, wwrblasFillMode_t,
                                      wwrblasOperation_t, wwrblasDiagType_t, int64_t, int64_t,
                                      const wwrFloatComplex *, const wwrFloatComplex *const[],
                                      int64_t, wwrFloatComplex *const[], int64_t, int64_t);
extern template wwrblasStatus_t
trsmBatched<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasSideMode_t, wwrblasFillMode_t,
                                   wwrblasOperation_t, wwrblasDiagType_t, int, int,
                                   const wwrDoubleComplex *, const wwrDoubleComplex *const[], int,
                                   wwrDoubleComplex *const[], int, int);
extern template wwrblasStatus_t
trsmBatched<wwrDoubleComplex, int64_t>(wwrblasHandle_t, wwrblasSideMode_t, wwrblasFillMode_t,
                                       wwrblasOperation_t, wwrblasDiagType_t, int64_t, int64_t,
                                       const wwrDoubleComplex *, const wwrDoubleComplex *const[],
                                       int64_t, wwrDoubleComplex *const[], int64_t, int64_t);

// Function: hemm
extern template wwrblasStatus_t
hemm<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasSideMode_t, wwrblasFillMode_t, int, int,
                           const wwrFloatComplex *, const wwrFloatComplex *, int,
                           const wwrFloatComplex *, int, const wwrFloatComplex *, wwrFloatComplex *,
                           int);
extern template wwrblasStatus_t
hemm<wwrFloatComplex, int64_t>(wwrblasHandle_t, wwrblasSideMode_t, wwrblasFillMode_t, int64_t,
                               int64_t, const wwrFloatComplex *, const wwrFloatComplex *, int64_t,
                               const wwrFloatComplex *, int64_t, const wwrFloatComplex *,
                               wwrFloatComplex *, int64_t);
extern template wwrblasStatus_t
hemm<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasSideMode_t, wwrblasFillMode_t, int, int,
                            const wwrDoubleComplex *, const wwrDoubleComplex *, int,
                            const wwrDoubleComplex *, int, const wwrDoubleComplex *,
                            wwrDoubleComplex *, int);
extern template wwrblasStatus_t
hemm<wwrDoubleComplex, int64_t>(wwrblasHandle_t, wwrblasSideMode_t, wwrblasFillMode_t, int64_t,
                                int64_t, const wwrDoubleComplex *, const wwrDoubleComplex *,
                                int64_t, const wwrDoubleComplex *, int64_t,
                                const wwrDoubleComplex *, wwrDoubleComplex *, int64_t);

// Function: herk
extern template wwrblasStatus_t
herk<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t, int, int,
                           const ComplexToRealType<wwrFloatComplex> *, const wwrFloatComplex *, int,
                           const ComplexToRealType<wwrFloatComplex> *, wwrFloatComplex *, int);
extern template wwrblasStatus_t herk<wwrFloatComplex, int64_t>(
    wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t, int64_t, int64_t,
    const ComplexToRealType<wwrFloatComplex> *, const wwrFloatComplex *, int64_t,
    const ComplexToRealType<wwrFloatComplex> *, wwrFloatComplex *, int64_t);
extern template wwrblasStatus_t
herk<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t, int, int,
                            const ComplexToRealType<wwrDoubleComplex> *, const wwrDoubleComplex *,
                            int, const ComplexToRealType<wwrDoubleComplex> *, wwrDoubleComplex *,
                            int);
extern template wwrblasStatus_t herk<wwrDoubleComplex, int64_t>(
    wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t, int64_t, int64_t,
    const ComplexToRealType<wwrDoubleComplex> *, const wwrDoubleComplex *, int64_t,
    const ComplexToRealType<wwrDoubleComplex> *, wwrDoubleComplex *, int64_t);

// Function: her2k
extern template wwrblasStatus_t
her2k<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t, int, int,
                            const wwrFloatComplex *, const wwrFloatComplex *, int,
                            const wwrFloatComplex *, int,
                            const ComplexToRealType<wwrFloatComplex> *, wwrFloatComplex *, int);
extern template wwrblasStatus_t her2k<wwrFloatComplex, int64_t>(
    wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t, int64_t, int64_t,
    const wwrFloatComplex *, const wwrFloatComplex *, int64_t, const wwrFloatComplex *, int64_t,
    const ComplexToRealType<wwrFloatComplex> *, wwrFloatComplex *, int64_t);
extern template wwrblasStatus_t
her2k<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t, int, int,
                             const wwrDoubleComplex *, const wwrDoubleComplex *, int,
                             const wwrDoubleComplex *, int,
                             const ComplexToRealType<wwrDoubleComplex> *, wwrDoubleComplex *, int);
extern template wwrblasStatus_t her2k<wwrDoubleComplex, int64_t>(
    wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t, int64_t, int64_t,
    const wwrDoubleComplex *, const wwrDoubleComplex *, int64_t, const wwrDoubleComplex *, int64_t,
    const ComplexToRealType<wwrDoubleComplex> *, wwrDoubleComplex *, int64_t);

// Function: herkx
extern template wwrblasStatus_t
herkx<wwrFloatComplex, int>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t, int, int,
                            const wwrFloatComplex *, const wwrFloatComplex *, int,
                            const wwrFloatComplex *, int,
                            const ComplexToRealType<wwrFloatComplex> *, wwrFloatComplex *, int);
extern template wwrblasStatus_t herkx<wwrFloatComplex, int64_t>(
    wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t, int64_t, int64_t,
    const wwrFloatComplex *, const wwrFloatComplex *, int64_t, const wwrFloatComplex *, int64_t,
    const ComplexToRealType<wwrFloatComplex> *, wwrFloatComplex *, int64_t);
extern template wwrblasStatus_t
herkx<wwrDoubleComplex, int>(wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t, int, int,
                             const wwrDoubleComplex *, const wwrDoubleComplex *, int,
                             const wwrDoubleComplex *, int,
                             const ComplexToRealType<wwrDoubleComplex> *, wwrDoubleComplex *, int);
extern template wwrblasStatus_t herkx<wwrDoubleComplex, int64_t>(
    wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t, int64_t, int64_t,
    const wwrDoubleComplex *, const wwrDoubleComplex *, int64_t, const wwrDoubleComplex *, int64_t,
    const ComplexToRealType<wwrDoubleComplex> *, wwrDoubleComplex *, int64_t);

} // namespace wwr
