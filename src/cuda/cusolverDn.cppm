/**
 * @file cusolverDn.cppm
 * @brief Primary interface for gpumod.cuda.cusolverDn
 *
 * This module wraps the native cuSOLVER Dense API and exports types, constants,
 * and functions for cuSOLVER library management and linear algebra operations.
 *
 * Usage:
 *   import gpumod.cuda.cusolverDn;
 */

module;

#include <cusolverDn.h>

export module gpumod.cuda.cusolverDn;

import std;

// ========================================================================
// Export all cuSOLVER Dense types and functions in gpumod namespace
// ========================================================================

export namespace gpumod::cuda {

// ========================================================================
// Core Types
// ========================================================================
using ::cusolverDnContext;
using ::cusolverDnHandle_t;
using ::cusolverStatus_t;

// Opaque parameter structures
using ::cusolverDnIRSInfos;
using ::cusolverDnIRSInfos_t;
using ::cusolverDnIRSParams;
using ::cusolverDnIRSParams_t;
using ::cusolverDnParams;
using ::cusolverDnParams_t;
using ::gesvdjInfo;
using ::gesvdjInfo_t;
using ::syevjInfo;
using ::syevjInfo_t;

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
// Enumerations
// ========================================================================

// Function type enumeration
using ::CUSOLVERDN_GETRF;
using ::CUSOLVERDN_POTRF;
using ::CUSOLVERDN_SYEVBATCHED;
using ::cusolverDnFunction_t;

// Eigenvalue problem types
using ::CUSOLVER_EIG_TYPE_1;
using ::CUSOLVER_EIG_TYPE_2;
using ::CUSOLVER_EIG_TYPE_3;
using ::cusolverEigType_t;

// Eigenvalue computation mode
using ::CUSOLVER_EIG_MODE_NOVECTOR;
using ::CUSOLVER_EIG_MODE_VECTOR;
using ::cusolverEigMode_t;

// Eigenvalue range specification
using ::CUSOLVER_EIG_RANGE_ALL;
using ::CUSOLVER_EIG_RANGE_I;
using ::CUSOLVER_EIG_RANGE_V;
using ::cusolverEigRange_t;

// Matrix norm types
using ::CUSOLVER_FRO_NORM;
using ::CUSOLVER_INF_NORM;
using ::CUSOLVER_MAX_NORM;
using ::CUSOLVER_ONE_NORM;
using ::cusolverNorm_t;

// IRS (Iterative Refinement Solver) refinement methods
using ::CUSOLVER_IRS_REFINE_CLASSICAL;
using ::CUSOLVER_IRS_REFINE_CLASSICAL_GMRES;
using ::CUSOLVER_IRS_REFINE_GMRES;
using ::CUSOLVER_IRS_REFINE_GMRES_GMRES;
using ::CUSOLVER_IRS_REFINE_GMRES_NOPCOND;
using ::CUSOLVER_IRS_REFINE_NONE;
using ::CUSOLVER_IRS_REFINE_NOT_SET;
using ::cusolverIRSRefinement_t;

// Precision modes for mixed-precision solvers
using ::CUSOLVER_PREC_DD;
using ::CUSOLVER_PREC_SHT;
using ::CUSOLVER_PREC_SS;
using ::cusolverPrecType_t;

// Data types for generic operations
using ::cudaDataType;

// Algorithm mode
using ::CUSOLVER_ALG_0;
using ::CUSOLVER_ALG_1;
using ::CUSOLVER_ALG_2;
using ::cusolverAlgMode_t;

// Storage mode
using ::CUBLAS_STOREV_COLUMNWISE;
using ::CUBLAS_STOREV_ROWWISE;
using ::cusolverStorevMode_t;

// Direction mode
using ::CUBLAS_DIRECT_BACKWARD;
using ::CUBLAS_DIRECT_FORWARD;
using ::cusolverDirectMode_t;

// Deterministic mode
using ::CUSOLVER_ALLOW_NON_DETERMINISTIC_RESULTS;
using ::CUSOLVER_DETERMINISTIC_RESULTS;
using ::cusolverDeterministicMode_t;

// Math mode
using ::CUSOLVER_DEFAULT_MATH;
using ::CUSOLVER_FP32_EMULATED_BF16X9_MATH;
using ::cusolverMathMode_t;

// ========================================================================
// Library Management Functions
// ========================================================================
using ::cusolverGetProperty;
using ::cusolverGetVersion;

// ========================================================================
// Context Management Functions
// ========================================================================
using ::cusolverDnCreate;
using ::cusolverDnDestroy;
using ::cusolverDnGetStream;
using ::cusolverDnSetStream;

// ========================================================================
// Mode and Strategy Functions
// ========================================================================
using ::cusolverDnGetDeterministicMode;
using ::cusolverDnGetEmulationStrategy;
using ::cusolverDnGetMathMode;
using ::cusolverDnSetAdvOptions;
using ::cusolverDnSetDeterministicMode;
using ::cusolverDnSetEmulationStrategy;
using ::cusolverDnSetMathMode;

// ========================================================================
// Logging Functions
// ========================================================================
using ::cusolverDnLoggerForceDisable;
using ::cusolverDnLoggerOpenFile;
using ::cusolverDnLoggerSetFile;
using ::cusolverDnLoggerSetLevel;
using ::cusolverDnLoggerSetMask;

// ========================================================================
// Info Structure Management
// ========================================================================
using ::cusolverDnCreateGesvdjInfo;
using ::cusolverDnCreateSyevjInfo;
using ::cusolverDnDestroyGesvdjInfo;
using ::cusolverDnDestroySyevjInfo;

// ========================================================================
// Params Management
// ========================================================================
using ::cusolverDnCreateParams;
using ::cusolverDnDestroyParams;

// ========================================================================
// IRS (Iterative Refinement Solver) Parameter Management
// ========================================================================
using ::cusolverDnIRSParamsGetMaxIters;
using ::cusolverDnIRSParamsSetMaxIters;
using ::cusolverDnIRSParamsSetMaxItersInner;
using ::cusolverDnIRSParamsSetRefinementSolver;
using ::cusolverDnIRSParamsSetSolverLowestPrecision;
using ::cusolverDnIRSParamsSetSolverMainPrecision;
using ::cusolverDnIRSParamsSetSolverPrecisions;

// ========================================================================
// IRS Info Management
// ========================================================================
using ::cusolverDnIRSInfosGetMaxIters;
using ::cusolverDnIRSInfosGetNiters;
using ::cusolverDnIRSInfosGetOuterNiters;
using ::cusolverDnIRSInfosGetResidualHistory;

// ========================================================================
// Generic (X-prefix) Solver Functions - New API
// ========================================================================
using ::cusolverDnXgeev;
using ::cusolverDnXgeev_bufferSize;
using ::cusolverDnXgeqrf;
using ::cusolverDnXgeqrf_bufferSize;
using ::cusolverDnXgesvd;
using ::cusolverDnXgesvd_bufferSize;
using ::cusolverDnXgesvdp;
using ::cusolverDnXgesvdp_bufferSize;
using ::cusolverDnXgesvdr;
using ::cusolverDnXgesvdr_bufferSize;
using ::cusolverDnXgetrf;
using ::cusolverDnXgetrf_bufferSize;
using ::cusolverDnXgetrs;
using ::cusolverDnXlarft;
using ::cusolverDnXlarft_bufferSize;
using ::cusolverDnXpotrf;
using ::cusolverDnXpotrf_bufferSize;
using ::cusolverDnXpotrs;
using ::cusolverDnXsyevBatched;
using ::cusolverDnXsyevBatched_bufferSize;
using ::cusolverDnXsyevd;
using ::cusolverDnXsyevd_bufferSize;
using ::cusolverDnXsyevdx;
using ::cusolverDnXsyevdx_bufferSize;
using ::cusolverDnXsytrs;
using ::cusolverDnXsytrs_bufferSize;
using ::cusolverDnXtrtri;
using ::cusolverDnXtrtri_bufferSize;

// Generic API - Info retrieval
using ::cusolverDnXgesvdjGetResidual;
using ::cusolverDnXgesvdjGetSweeps;
using ::cusolverDnXsyevjGetResidual;
using ::cusolverDnXsyevjGetSweeps;

// ========================================================================
// IRS Generic Solvers
// ========================================================================
using ::cusolverDnIRSXgels;
using ::cusolverDnIRSXgels_bufferSize;
using ::cusolverDnIRSXgesv;
using ::cusolverDnIRSXgesv_bufferSize;

// ========================================================================
// Mixed-Precision GESV Solvers (General Linear Systems)
// ========================================================================

// Single precision variants
using ::cusolverDnSBgesv;
using ::cusolverDnSBgesv_bufferSize;
using ::cusolverDnSHgesv;
using ::cusolverDnSHgesv_bufferSize;
using ::cusolverDnSSgesv;
using ::cusolverDnSSgesv_bufferSize;
using ::cusolverDnSXgesv;
using ::cusolverDnSXgesv_bufferSize;

// Double precision variants
using ::cusolverDnDBgesv;
using ::cusolverDnDBgesv_bufferSize;
using ::cusolverDnDDgesv;
using ::cusolverDnDDgesv_bufferSize;
using ::cusolverDnDHgesv;
using ::cusolverDnDHgesv_bufferSize;
using ::cusolverDnDSgesv;
using ::cusolverDnDSgesv_bufferSize;
using ::cusolverDnDXgesv;
using ::cusolverDnDXgesv_bufferSize;

// Complex single precision variants
using ::cusolverDnCCgesv;
using ::cusolverDnCCgesv_bufferSize;
using ::cusolverDnCEgesv;
using ::cusolverDnCEgesv_bufferSize;
using ::cusolverDnCKgesv;
using ::cusolverDnCKgesv_bufferSize;
using ::cusolverDnCYgesv;
using ::cusolverDnCYgesv_bufferSize;

// Complex double precision variants
using ::cusolverDnZCgesv;
using ::cusolverDnZCgesv_bufferSize;
using ::cusolverDnZEgesv;
using ::cusolverDnZEgesv_bufferSize;
using ::cusolverDnZKgesv;
using ::cusolverDnZKgesv_bufferSize;
using ::cusolverDnZYgesv;
using ::cusolverDnZYgesv_bufferSize;
using ::cusolverDnZZgesv;
using ::cusolverDnZZgesv_bufferSize;

// ========================================================================
// Mixed-Precision GELS Solvers (Least Squares)
// ========================================================================

// Single precision variants
using ::cusolverDnSBgels;
using ::cusolverDnSBgels_bufferSize;
using ::cusolverDnSHgels;
using ::cusolverDnSHgels_bufferSize;
using ::cusolverDnSSgels;
using ::cusolverDnSSgels_bufferSize;
using ::cusolverDnSXgels;
using ::cusolverDnSXgels_bufferSize;

// Double precision variants
using ::cusolverDnDBgels;
using ::cusolverDnDBgels_bufferSize;
using ::cusolverDnDDgels;
using ::cusolverDnDDgels_bufferSize;
using ::cusolverDnDHgels;
using ::cusolverDnDHgels_bufferSize;
using ::cusolverDnDSgels;
using ::cusolverDnDSgels_bufferSize;
using ::cusolverDnDXgels;
using ::cusolverDnDXgels_bufferSize;

// Complex single precision variants
using ::cusolverDnCCgels;
using ::cusolverDnCCgels_bufferSize;
using ::cusolverDnCEgels;
using ::cusolverDnCEgels_bufferSize;
using ::cusolverDnCKgels;
using ::cusolverDnCKgels_bufferSize;
using ::cusolverDnCYgels;
using ::cusolverDnCYgels_bufferSize;

// Complex double precision variants
using ::cusolverDnZCgels;
using ::cusolverDnZCgels_bufferSize;
using ::cusolverDnZEgels;
using ::cusolverDnZEgels_bufferSize;
using ::cusolverDnZKgels;
using ::cusolverDnZKgels_bufferSize;
using ::cusolverDnZYgels;
using ::cusolverDnZYgels_bufferSize;
using ::cusolverDnZZgels;
using ::cusolverDnZZgels_bufferSize;

// ========================================================================
// LU Factorization (GETRF)
// ========================================================================
using ::cusolverDnCgetrf;
using ::cusolverDnCgetrf_bufferSize;
using ::cusolverDnDgetrf;
using ::cusolverDnDgetrf_bufferSize;
using ::cusolverDnSgetrf;
using ::cusolverDnSgetrf_bufferSize;
using ::cusolverDnZgetrf;
using ::cusolverDnZgetrf_bufferSize;

// ========================================================================
// LU Solve (GETRS)
// ========================================================================
using ::cusolverDnCgetrs;
using ::cusolverDnDgetrs;
using ::cusolverDnSgetrs;
using ::cusolverDnZgetrs;

// ========================================================================
// Cholesky Factorization (POTRF)
// ========================================================================
using ::cusolverDnCpotrf;
using ::cusolverDnCpotrf_bufferSize;
using ::cusolverDnDpotrf;
using ::cusolverDnDpotrf_bufferSize;
using ::cusolverDnSpotrf;
using ::cusolverDnSpotrf_bufferSize;
using ::cusolverDnZpotrf;
using ::cusolverDnZpotrf_bufferSize;

// Batched Cholesky
using ::cusolverDnCpotrfBatched;
using ::cusolverDnDpotrfBatched;
using ::cusolverDnSpotrfBatched;
using ::cusolverDnZpotrfBatched;

// ========================================================================
// Cholesky Solve (POTRS)
// ========================================================================
using ::cusolverDnCpotrs;
using ::cusolverDnDpotrs;
using ::cusolverDnSpotrs;
using ::cusolverDnZpotrs;

// Batched Cholesky Solve
using ::cusolverDnCpotrsBatched;
using ::cusolverDnDpotrsBatched;
using ::cusolverDnSpotrsBatched;
using ::cusolverDnZpotrsBatched;

// ========================================================================
// Cholesky Inverse (POTRI)
// ========================================================================
using ::cusolverDnCpotri;
using ::cusolverDnCpotri_bufferSize;
using ::cusolverDnDpotri;
using ::cusolverDnDpotri_bufferSize;
using ::cusolverDnSpotri;
using ::cusolverDnSpotri_bufferSize;
using ::cusolverDnZpotri;
using ::cusolverDnZpotri_bufferSize;

// ========================================================================
// Triangular Multiplication (LAUUM)
// ========================================================================
using ::cusolverDnClauum;
using ::cusolverDnClauum_bufferSize;
using ::cusolverDnDlauum;
using ::cusolverDnDlauum_bufferSize;
using ::cusolverDnSlauum;
using ::cusolverDnSlauum_bufferSize;
using ::cusolverDnZlauum;
using ::cusolverDnZlauum_bufferSize;

// ========================================================================
// Symmetric/Hermitian Factorization (SYTRF)
// ========================================================================
using ::cusolverDnCsytrf;
using ::cusolverDnCsytrf_bufferSize;
using ::cusolverDnDsytrf;
using ::cusolverDnDsytrf_bufferSize;
using ::cusolverDnSsytrf;
using ::cusolverDnSsytrf_bufferSize;
using ::cusolverDnZsytrf;
using ::cusolverDnZsytrf_bufferSize;

// ========================================================================
// Symmetric/Hermitian Inverse (SYTRI)
// ========================================================================
using ::cusolverDnCsytri;
using ::cusolverDnCsytri_bufferSize;
using ::cusolverDnDsytri;
using ::cusolverDnDsytri_bufferSize;
using ::cusolverDnSsytri;
using ::cusolverDnSsytri_bufferSize;
using ::cusolverDnZsytri;
using ::cusolverDnZsytri_bufferSize;

// ========================================================================
// QR Factorization (GEQRF)
// ========================================================================
using ::cusolverDnCgeqrf;
using ::cusolverDnCgeqrf_bufferSize;
using ::cusolverDnDgeqrf;
using ::cusolverDnDgeqrf_bufferSize;
using ::cusolverDnSgeqrf;
using ::cusolverDnSgeqrf_bufferSize;
using ::cusolverDnZgeqrf;
using ::cusolverDnZgeqrf_bufferSize;

// ========================================================================
// Generate Orthogonal/Unitary Matrix Q (ORGQR/UNGQR)
// ========================================================================
using ::cusolverDnCungqr;
using ::cusolverDnCungqr_bufferSize;
using ::cusolverDnDorgqr;
using ::cusolverDnDorgqr_bufferSize;
using ::cusolverDnSorgqr;
using ::cusolverDnSorgqr_bufferSize;
using ::cusolverDnZungqr;
using ::cusolverDnZungqr_bufferSize;

// ========================================================================
// Multiply by Q (ORMQR/UNMQR)
// ========================================================================
using ::cusolverDnCunmqr;
using ::cusolverDnCunmqr_bufferSize;
using ::cusolverDnDormqr;
using ::cusolverDnDormqr_bufferSize;
using ::cusolverDnSormqr;
using ::cusolverDnSormqr_bufferSize;
using ::cusolverDnZunmqr;
using ::cusolverDnZunmqr_bufferSize;

// ========================================================================
// Bidiagonal Reduction (GEBRD)
// ========================================================================
using ::cusolverDnCgebrd;
using ::cusolverDnCgebrd_bufferSize;
using ::cusolverDnDgebrd;
using ::cusolverDnDgebrd_bufferSize;
using ::cusolverDnSgebrd;
using ::cusolverDnSgebrd_bufferSize;
using ::cusolverDnZgebrd;
using ::cusolverDnZgebrd_bufferSize;

// ========================================================================
// Generate Orthogonal/Unitary Matrices from GEBRD (ORGBR/UNGBR)
// ========================================================================
using ::cusolverDnCungbr;
using ::cusolverDnCungbr_bufferSize;
using ::cusolverDnDorgbr;
using ::cusolverDnDorgbr_bufferSize;
using ::cusolverDnSorgbr;
using ::cusolverDnSorgbr_bufferSize;
using ::cusolverDnZungbr;
using ::cusolverDnZungbr_bufferSize;

// ========================================================================
// Tridiagonal Reduction (SYTRD/HETRD)
// ========================================================================
using ::cusolverDnChetrd;
using ::cusolverDnChetrd_bufferSize;
using ::cusolverDnDsytrd;
using ::cusolverDnDsytrd_bufferSize;
using ::cusolverDnSsytrd;
using ::cusolverDnSsytrd_bufferSize;
using ::cusolverDnZhetrd;
using ::cusolverDnZhetrd_bufferSize;

// ========================================================================
// Generate Orthogonal/Unitary Matrix from SYTRD/HETRD (ORGTR/UNGTR)
// ========================================================================
using ::cusolverDnCungtr;
using ::cusolverDnCungtr_bufferSize;
using ::cusolverDnDorgtr;
using ::cusolverDnDorgtr_bufferSize;
using ::cusolverDnSorgtr;
using ::cusolverDnSorgtr_bufferSize;
using ::cusolverDnZungtr;
using ::cusolverDnZungtr_bufferSize;

// ========================================================================
// Multiply by Orthogonal/Unitary Matrix from SYTRD/HETRD (ORMTR/UNMTR)
// ========================================================================
using ::cusolverDnCunmtr;
using ::cusolverDnCunmtr_bufferSize;
using ::cusolverDnDormtr;
using ::cusolverDnDormtr_bufferSize;
using ::cusolverDnSormtr;
using ::cusolverDnSormtr_bufferSize;
using ::cusolverDnZunmtr;
using ::cusolverDnZunmtr_bufferSize;

// ========================================================================
// SVD - Singular Value Decomposition (GESVD)
// ========================================================================
using ::cusolverDnCgesvd;
using ::cusolverDnCgesvd_bufferSize;
using ::cusolverDnDgesvd;
using ::cusolverDnDgesvd_bufferSize;
using ::cusolverDnSgesvd;
using ::cusolverDnSgesvd_bufferSize;
using ::cusolverDnZgesvd;
using ::cusolverDnZgesvd_bufferSize;

// ========================================================================
// SVD - Jacobi Algorithm (GESVDJ)
// ========================================================================
using ::cusolverDnCgesvdj;
using ::cusolverDnCgesvdj_bufferSize;
using ::cusolverDnDgesvdj;
using ::cusolverDnDgesvdj_bufferSize;
using ::cusolverDnSgesvdj;
using ::cusolverDnSgesvdj_bufferSize;
using ::cusolverDnZgesvdj;
using ::cusolverDnZgesvdj_bufferSize;

// Batched Jacobi SVD
using ::cusolverDnCgesvdjBatched;
using ::cusolverDnCgesvdjBatched_bufferSize;
using ::cusolverDnDgesvdjBatched;
using ::cusolverDnDgesvdjBatched_bufferSize;
using ::cusolverDnSgesvdjBatched;
using ::cusolverDnSgesvdjBatched_bufferSize;
using ::cusolverDnZgesvdjBatched;
using ::cusolverDnZgesvdjBatched_bufferSize;

// ========================================================================
// SVD - Approximate Algorithm (GESVDA - Strided Batched)
// ========================================================================
using ::cusolverDnCgesvdaStridedBatched;
using ::cusolverDnCgesvdaStridedBatched_bufferSize;
using ::cusolverDnDgesvdaStridedBatched;
using ::cusolverDnDgesvdaStridedBatched_bufferSize;
using ::cusolverDnSgesvdaStridedBatched;
using ::cusolverDnSgesvdaStridedBatched_bufferSize;
using ::cusolverDnZgesvdaStridedBatched;
using ::cusolverDnZgesvdaStridedBatched_bufferSize;

// ========================================================================
// Symmetric Eigenvalue Problem (SYEVD/HEEVD)
// ========================================================================
using ::cusolverDnCheevd;
using ::cusolverDnCheevd_bufferSize;
using ::cusolverDnDsyevd;
using ::cusolverDnDsyevd_bufferSize;
using ::cusolverDnSsyevd;
using ::cusolverDnSsyevd_bufferSize;
using ::cusolverDnZheevd;
using ::cusolverDnZheevd_bufferSize;

// ========================================================================
// Symmetric Eigenvalue Problem with Range Selection (SYEVDX/HEEVDX)
// ========================================================================
using ::cusolverDnCheevdx;
using ::cusolverDnCheevdx_bufferSize;
using ::cusolverDnDsyevdx;
using ::cusolverDnDsyevdx_bufferSize;
using ::cusolverDnSsyevdx;
using ::cusolverDnSsyevdx_bufferSize;
using ::cusolverDnZheevdx;
using ::cusolverDnZheevdx_bufferSize;

// ========================================================================
// Symmetric Eigenvalue Problem - Jacobi Algorithm (SYEVJ/HEEVJ)
// ========================================================================
using ::cusolverDnCheevj;
using ::cusolverDnCheevj_bufferSize;
using ::cusolverDnDsyevj;
using ::cusolverDnDsyevj_bufferSize;
using ::cusolverDnSsyevj;
using ::cusolverDnSsyevj_bufferSize;
using ::cusolverDnZheevj;
using ::cusolverDnZheevj_bufferSize;

// Batched Jacobi Eigenvalue
using ::cusolverDnCheevjBatched;
using ::cusolverDnCheevjBatched_bufferSize;
using ::cusolverDnDsyevjBatched;
using ::cusolverDnDsyevjBatched_bufferSize;
using ::cusolverDnSsyevjBatched;
using ::cusolverDnSsyevjBatched_bufferSize;
using ::cusolverDnZheevjBatched;
using ::cusolverDnZheevjBatched_bufferSize;

// ========================================================================
// Generalized Symmetric Eigenvalue Problem (SYGVD/HEGVD)
// ========================================================================
using ::cusolverDnChegvd;
using ::cusolverDnChegvd_bufferSize;
using ::cusolverDnDsygvd;
using ::cusolverDnDsygvd_bufferSize;
using ::cusolverDnSsygvd;
using ::cusolverDnSsygvd_bufferSize;
using ::cusolverDnZhegvd;
using ::cusolverDnZhegvd_bufferSize;

// ========================================================================
// Generalized Symmetric Eigenvalue Problem with Range (SYGVDX/HEGVDX)
// ========================================================================
using ::cusolverDnChegvdx;
using ::cusolverDnChegvdx_bufferSize;
using ::cusolverDnDsygvdx;
using ::cusolverDnDsygvdx_bufferSize;
using ::cusolverDnSsygvdx;
using ::cusolverDnSsygvdx_bufferSize;
using ::cusolverDnZhegvdx;
using ::cusolverDnZhegvdx_bufferSize;

// ========================================================================
// Generalized Symmetric Eigenvalue Problem - Jacobi (SYGVJ/HEGVJ)
// ========================================================================
using ::cusolverDnChegvj;
using ::cusolverDnChegvj_bufferSize;
using ::cusolverDnDsygvj;
using ::cusolverDnDsygvj_bufferSize;
using ::cusolverDnSsygvj;
using ::cusolverDnSsygvj_bufferSize;
using ::cusolverDnZhegvj;
using ::cusolverDnZhegvj_bufferSize;

// ========================================================================
// Pivot Application (LASWP)
// ========================================================================
using ::cusolverDnClaswp;
using ::cusolverDnDlaswp;
using ::cusolverDnSlaswp;
using ::cusolverDnZlaswp;

} // namespace gpumod::cuda
