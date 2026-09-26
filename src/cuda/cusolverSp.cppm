/**
 * @file cusolverSp.cppm
 * @brief Primary interface for gpumod.cuda.cusolverSp
 *
 * This module wraps the native cuSOLVER Sparse API and exports types, constants,
 * and functions for sparse linear algebra operations.
 *
 * Usage:
 *   import gpumod.cuda.cusolverSp;
 */

module;

#include <cusolverSp.h>

export module gpumod.cuda.cusolverSp;

import std;

// ========================================================================
// Export all cuSOLVER Sparse types and functions in gpumod namespace
// ========================================================================

export namespace gpumod::cuda {

// ========================================================================
// Core Types
// ========================================================================
using ::cusolverSpContext;
using ::cusolverSpHandle_t;
using ::cusolverStatus_t;

// Batched QR info structure
using ::csrqrInfo;
using ::csrqrInfo_t;

// cuSPARSE matrix descriptor (required by cusolverSp API)
using ::cusparseMatDescr_t;

// ========================================================================
// Status Codes (from cusolver_common.h)
// ========================================================================
using ::CUSOLVER_STATUS_ALLOC_FAILED;
using ::CUSOLVER_STATUS_ARCH_MISMATCH;
using ::CUSOLVER_STATUS_EXECUTION_FAILED;
using ::CUSOLVER_STATUS_INTERNAL_ERROR;
using ::CUSOLVER_STATUS_INVALID_LICENSE;
using ::CUSOLVER_STATUS_INVALID_VALUE;
using ::CUSOLVER_STATUS_INVALID_WORKSPACE;
using ::CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED;
using ::CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED;
using ::CUSOLVER_STATUS_IRS_INTERNAL_ERROR;
using ::CUSOLVER_STATUS_IRS_MATRIX_SINGULAR;
using ::CUSOLVER_STATUS_IRS_NOT_SUPPORTED;
using ::CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES;
using ::CUSOLVER_STATUS_IRS_OUT_OF_RANGE;
using ::CUSOLVER_STATUS_IRS_PARAMS_INVALID;
using ::CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER;
using ::CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC;
using ::CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE;
using ::CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED;
using ::CUSOLVER_STATUS_MAPPING_ERROR;
using ::CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
using ::CUSOLVER_STATUS_NOT_INITIALIZED;
using ::CUSOLVER_STATUS_NOT_SUPPORTED;
using ::CUSOLVER_STATUS_SUCCESS;
using ::CUSOLVER_STATUS_ZERO_PIVOT;

// ========================================================================
// Handle Management Functions
// ========================================================================
using ::cusolverSpCreate;
using ::cusolverSpDestroy;
using ::cusolverSpGetStream;
using ::cusolverSpSetStream;

// ========================================================================
// Symmetry Check (Host)
// ========================================================================
using ::cusolverSpXcsrissymHost;

// ========================================================================
// GPU Linear Solvers by LU Factorization (Host)
// ========================================================================
using ::cusolverSpCcsrlsvluHost;
using ::cusolverSpDcsrlsvluHost;
using ::cusolverSpScsrlsvluHost;
using ::cusolverSpZcsrlsvluHost;

// ========================================================================
// GPU Linear Solvers by QR Factorization (Device)
// ========================================================================
using ::cusolverSpCcsrlsvqr;
using ::cusolverSpDcsrlsvqr;
using ::cusolverSpScsrlsvqr;
using ::cusolverSpZcsrlsvqr;

// ========================================================================
// CPU Linear Solvers by QR Factorization (Host)
// ========================================================================
using ::cusolverSpCcsrlsvqrHost;
using ::cusolverSpDcsrlsvqrHost;
using ::cusolverSpScsrlsvqrHost;
using ::cusolverSpZcsrlsvqrHost;

// ========================================================================
// CPU Linear Solvers by Cholesky Factorization (Host)
// ========================================================================
using ::cusolverSpCcsrlsvcholHost;
using ::cusolverSpDcsrlsvcholHost;
using ::cusolverSpScsrlsvcholHost;
using ::cusolverSpZcsrlsvcholHost;

// ========================================================================
// GPU Linear Solvers by Cholesky Factorization (Device)
// ========================================================================
using ::cusolverSpCcsrlsvchol;
using ::cusolverSpDcsrlsvchol;
using ::cusolverSpScsrlsvchol;
using ::cusolverSpZcsrlsvchol;

// ========================================================================
// CPU Least-Squares Solvers by QR Factorization (Host)
// ========================================================================
using ::cusolverSpCcsrlsqvqrHost;
using ::cusolverSpDcsrlsqvqrHost;
using ::cusolverSpScsrlsqvqrHost;
using ::cusolverSpZcsrlsqvqrHost;

// ========================================================================
// CPU Eigenvalue Solvers by Shift-Inverse (Host)
// ========================================================================
using ::cusolverSpCcsreigvsiHost;
using ::cusolverSpDcsreigvsiHost;
using ::cusolverSpScsreigvsiHost;
using ::cusolverSpZcsreigvsiHost;

// ========================================================================
// GPU Eigenvalue Solvers by Shift-Inverse (Device)
// ========================================================================
using ::cusolverSpCcsreigvsi;
using ::cusolverSpDcsreigvsi;
using ::cusolverSpScsreigvsi;
using ::cusolverSpZcsreigvsi;

// ========================================================================
// CPU Enclosed Eigenvalue Count (Host)
// ========================================================================
using ::cusolverSpCcsreigsHost;
using ::cusolverSpDcsreigsHost;
using ::cusolverSpScsreigsHost;
using ::cusolverSpZcsreigsHost;

// ========================================================================
// CPU Reordering: Symmetric Reverse Cuthill-McKee (symrcm)
// ========================================================================
using ::cusolverSpXcsrsymrcmHost;

// ========================================================================
// CPU Reordering: Symmetric Minimum Degree by Quotient Graph (symmdq)
// ========================================================================
using ::cusolverSpXcsrsymmdqHost;

// ========================================================================
// CPU Reordering: Approximate Minimum Degree by Quotient Graph (symamd)
// ========================================================================
using ::cusolverSpXcsrsymamdHost;

// ========================================================================
// CPU Reordering: Nested Dissection via METIS (metisnd)
// ========================================================================
using ::cusolverSpXcsrmetisndHost;

// ========================================================================
// CPU Zero-Free Diagonal Reordering (zfd)
// ========================================================================
using ::cusolverSpCcsrzfdHost;
using ::cusolverSpDcsrzfdHost;
using ::cusolverSpScsrzfdHost;
using ::cusolverSpZcsrzfdHost;

// ========================================================================
// CPU Permutation: P*A*Q^T (csrperm)
// ========================================================================
using ::cusolverSpXcsrperm_bufferSizeHost;
using ::cusolverSpXcsrpermHost;

// ========================================================================
// Low-Level Batched QR: csrqrInfo Management
// ========================================================================
using ::cusolverSpCreateCsrqrInfo;
using ::cusolverSpDestroyCsrqrInfo;

// ========================================================================
// Low-Level Batched QR: Analysis and Buffer
// ========================================================================
using ::cusolverSpCcsrqrBufferInfoBatched;
using ::cusolverSpDcsrqrBufferInfoBatched;
using ::cusolverSpScsrqrBufferInfoBatched;
using ::cusolverSpXcsrqrAnalysisBatched;
using ::cusolverSpZcsrqrBufferInfoBatched;

// ========================================================================
// Low-Level Batched QR: Solve
// ========================================================================
using ::cusolverSpCcsrqrsvBatched;
using ::cusolverSpDcsrqrsvBatched;
using ::cusolverSpScsrqrsvBatched;
using ::cusolverSpZcsrqrsvBatched;

} // namespace gpumod::cuda
