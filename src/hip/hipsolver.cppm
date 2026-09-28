/**
 * @file hipsolver.cppm
 * @brief hipSOLVER API module wrapper for wwr project
 *
 * Wraps hipsolver/hipsolver.h. CUDA counterparts: wwr.cuda.cusolverDn and
 * wwr.cuda.cusolverSp together -- hipSOLVER's single umbrella header pulls
 * in both the Dn and Sp surfaces, so one module covers what CUDA splits in
 * two. See src/hip/README.md for the collapse-and-narrow writeup.
 *
 * Not wrapped, though hipsolver.h declares them: the "native" hipsolver* API
 * (internal/hipsolver-functions.h), a rocSOLVER-optimized alternative with no
 * cuSOLVER counterpart to mirror against, and hipsolverRf*
 * (internal/hipsolver-refactor.h), which has a cuSOLVER counterpart this
 * project has never wrapped either.
 *
 * Both surfaces are also narrower than CUDA's. hipsolverSp implements 11
 * functions total; hipsolverDn lacks the IRS solvers, mixed-precision
 * GESV/GELS, Laswp, Lauum, Sytri, the logger and math-mode setters, and most
 * cusolverDnParams_t-based selection. It adds six of its own: the
 * Xgesvdj/Xsyevj Jacobi parameter setters, exported here. Missing functions
 * are not stubbed, simply not exported.
 *
 * hipblas-owned types used in these signatures are not re-exported, mirroring
 * cusolverDn.cppm; hipsolver's own aliases of them, and hipDataType, are.
 *
 * Usage:
 *   import wwr.hip.hipsolver;
 */

module;

// Pre-include <array> before the HIP header -- see src/hip/hip_complex.cppm's
// file header for why (amd_hip_vector_types.h, pulled in transitively via
// hip/hip_complex.h, #includes host_defines.h immediately before <array>,
// poisoning __has_attribute(__noinline__) for any later first-inclusion of
// <array> in the TU). Confirmed necessary here by direct experiment too.
// Load-bearing, and must stay before the HIP header: host_defines.h poisons
// __noinline__ for libc++'s __config. docs/architecture.md, section 9.
#include <array>
#include <hipsolver/hipsolver.h>

export module wwr.hip.hipsolver;

import std;

export namespace wwr::hip {

// ========================================================================
// Core Types
// ========================================================================
using ::hipsolverDnHandle_t;
using ::hipsolverHandle_t;
using ::hipsolverStatus_t;

// Opaque parameter/info structures
using ::hipsolverDnParams_t;
using ::hipsolverGesvdjInfo_t;
using ::hipsolverSyevjInfo_t;

// Sparse-side handle and matrix descriptor (hipsolverSp owns its own copy of
// hipsparseMatDescr_t as an opaque void* -- hipsolver.h does not include
// hipsparse.h)
using ::hipsolverSpHandle_t;
using ::hipsparseMatDescr_t;

// ========================================================================
// Status Codes
// ========================================================================
using ::HIPSOLVER_STATUS_ALLOC_FAILED;
using ::HIPSOLVER_STATUS_ARCH_MISMATCH;
using ::HIPSOLVER_STATUS_EXECUTION_FAILED;
using ::HIPSOLVER_STATUS_HANDLE_IS_NULLPTR;
using ::HIPSOLVER_STATUS_INTERNAL_ERROR;
using ::HIPSOLVER_STATUS_INVALID_ENUM;
using ::HIPSOLVER_STATUS_INVALID_VALUE;
using ::HIPSOLVER_STATUS_MAPPING_ERROR;
using ::HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
using ::HIPSOLVER_STATUS_NOT_INITIALIZED;
using ::HIPSOLVER_STATUS_NOT_SUPPORTED;
using ::HIPSOLVER_STATUS_SUCCESS;
using ::HIPSOLVER_STATUS_UNKNOWN;
using ::HIPSOLVER_STATUS_ZERO_PIVOT;

// ========================================================================
// Enumerations
// ========================================================================

// Eigenvalue computation mode
using ::HIPSOLVER_EIG_MODE_NOVECTOR;
using ::HIPSOLVER_EIG_MODE_VECTOR;
using ::hipsolverEigMode_t;

// Eigenvalue problem types
using ::HIPSOLVER_EIG_TYPE_1;
using ::HIPSOLVER_EIG_TYPE_2;
using ::HIPSOLVER_EIG_TYPE_3;
using ::hipsolverEigType_t;

// Eigenvalue range specification
using ::HIPSOLVER_EIG_RANGE_ALL;
using ::HIPSOLVER_EIG_RANGE_I;
using ::HIPSOLVER_EIG_RANGE_V;
using ::hipsolverEigRange_t;

// Deterministic mode
using ::HIPSOLVER_ALLOW_NON_DETERMINISTIC_RESULTS;
using ::HIPSOLVER_DETERMINISTIC_RESULTS;
using ::hipsolverDeterministicMode_t;

// Algorithm mode (hipsolverDnSetAdvOptions)
using ::HIPSOLVER_ALG_0;
using ::HIPSOLVER_ALG_1;
using ::hipsolverAlgMode_t;

// Function selector for hipsolverDnSetAdvOptions (narrower than cuSOLVER's
// cusolverDnFunction_t -- only GETRF is selectable here)
using ::HIPSOLVERDN_GETRF;
using ::hipsolverDnFunction_t;

// hipsolver's own names for the hipblas types used by the X-prefixed 64-bit
// generic API (typedef aliases of hipblasOperation_t/FillMode_t/SideMode_t;
// the underlying types and their enumerators are wwr.hip.hipblas's
// to export, not re-exported redundantly here)
using ::hipsolverFillMode_t;
using ::hipsolverOperation_t;
using ::hipsolverSideMode_t;

// Data type tag (from hip/library_types.h; no home module in src/hip, see
// file header comment)
using ::hipDataType;

// ========================================================================
// Context Management Functions
// ========================================================================
using ::hipsolverDnCreate;
using ::hipsolverDnDestroy;
using ::hipsolverDnGetStream;
using ::hipsolverDnSetStream;

// ========================================================================
// Mode and Strategy Functions
// ========================================================================
using ::hipsolverDnGetDeterministicMode;
using ::hipsolverDnSetAdvOptions;
using ::hipsolverDnSetDeterministicMode;

// ========================================================================
// Info Structure Management
// ========================================================================
using ::hipsolverDnCreateGesvdjInfo;
using ::hipsolverDnDestroyGesvdjInfo;
using ::hipsolverDnXgesvdjGetResidual;
using ::hipsolverDnXgesvdjGetSweeps;
using ::hipsolverDnXgesvdjSetMaxSweeps;
using ::hipsolverDnXgesvdjSetSortEig;
using ::hipsolverDnXgesvdjSetTolerance;

using ::hipsolverDnCreateSyevjInfo;
using ::hipsolverDnDestroySyevjInfo;
using ::hipsolverDnXsyevjGetResidual;
using ::hipsolverDnXsyevjGetSweeps;
using ::hipsolverDnXsyevjSetMaxSweeps;
using ::hipsolverDnXsyevjSetSortEig;
using ::hipsolverDnXsyevjSetTolerance;

// ========================================================================
// Params Management
// ========================================================================
using ::hipsolverDnCreateParams;
using ::hipsolverDnDestroyParams;

// ========================================================================
// Generic (X-prefix) Solver Functions -- 64-bit-index API
// ========================================================================
using ::hipsolverDnXgeqrf;
using ::hipsolverDnXgeqrf_bufferSize;
using ::hipsolverDnXgetrf;
using ::hipsolverDnXgetrf_bufferSize;
using ::hipsolverDnXgetrs;
using ::hipsolverDnXpotrf;
using ::hipsolverDnXpotrf_bufferSize;
using ::hipsolverDnXpotrs;

// ========================================================================
// Mixed-Precision GESV Solvers (General Linear Systems) -- same-precision
// variants only (SS/DD/CC/ZZ); cuSOLVER's SH/SB/SX/DH/DB/DX/etc. mixed-
// precision variants have no hipsolverDn counterpart.
// ========================================================================
using ::hipsolverDnCCgesv;
using ::hipsolverDnCCgesv_bufferSize;
using ::hipsolverDnDDgesv;
using ::hipsolverDnDDgesv_bufferSize;
using ::hipsolverDnSSgesv;
using ::hipsolverDnSSgesv_bufferSize;
using ::hipsolverDnZZgesv;
using ::hipsolverDnZZgesv_bufferSize;

// ========================================================================
// Mixed-Precision GELS Solvers (Least Squares) -- same-precision variants
// only (SS/DD/CC/ZZ), for the same reason as GESV above.
// ========================================================================
using ::hipsolverDnCCgels;
using ::hipsolverDnCCgels_bufferSize;
using ::hipsolverDnDDgels;
using ::hipsolverDnDDgels_bufferSize;
using ::hipsolverDnSSgels;
using ::hipsolverDnSSgels_bufferSize;
using ::hipsolverDnZZgels;
using ::hipsolverDnZZgels_bufferSize;

// ========================================================================
// LU Factorization (GETRF)
// ========================================================================
using ::hipsolverDnCgetrf;
using ::hipsolverDnCgetrf_bufferSize;
using ::hipsolverDnDgetrf;
using ::hipsolverDnDgetrf_bufferSize;
using ::hipsolverDnSgetrf;
using ::hipsolverDnSgetrf_bufferSize;
using ::hipsolverDnZgetrf;
using ::hipsolverDnZgetrf_bufferSize;

// ========================================================================
// LU Solve (GETRS)
// ========================================================================
using ::hipsolverDnCgetrs;
using ::hipsolverDnDgetrs;
using ::hipsolverDnSgetrs;
using ::hipsolverDnZgetrs;

// ========================================================================
// Cholesky Factorization (POTRF)
// ========================================================================
using ::hipsolverDnCpotrf;
using ::hipsolverDnCpotrf_bufferSize;
using ::hipsolverDnDpotrf;
using ::hipsolverDnDpotrf_bufferSize;
using ::hipsolverDnSpotrf;
using ::hipsolverDnSpotrf_bufferSize;
using ::hipsolverDnZpotrf;
using ::hipsolverDnZpotrf_bufferSize;

// Batched Cholesky
using ::hipsolverDnCpotrfBatched;
using ::hipsolverDnDpotrfBatched;
using ::hipsolverDnSpotrfBatched;
using ::hipsolverDnZpotrfBatched;

// ========================================================================
// Cholesky Solve (POTRS)
// ========================================================================
using ::hipsolverDnCpotrs;
using ::hipsolverDnDpotrs;
using ::hipsolverDnSpotrs;
using ::hipsolverDnZpotrs;

// Batched Cholesky Solve
using ::hipsolverDnCpotrsBatched;
using ::hipsolverDnDpotrsBatched;
using ::hipsolverDnSpotrsBatched;
using ::hipsolverDnZpotrsBatched;

// ========================================================================
// Cholesky Inverse (POTRI)
// ========================================================================
using ::hipsolverDnCpotri;
using ::hipsolverDnCpotri_bufferSize;
using ::hipsolverDnDpotri;
using ::hipsolverDnDpotri_bufferSize;
using ::hipsolverDnSpotri;
using ::hipsolverDnSpotri_bufferSize;
using ::hipsolverDnZpotri;
using ::hipsolverDnZpotri_bufferSize;

// ========================================================================
// Symmetric/Hermitian Factorization (SYTRF)
// ========================================================================
using ::hipsolverDnCsytrf;
using ::hipsolverDnCsytrf_bufferSize;
using ::hipsolverDnDsytrf;
using ::hipsolverDnDsytrf_bufferSize;
using ::hipsolverDnSsytrf;
using ::hipsolverDnSsytrf_bufferSize;
using ::hipsolverDnZsytrf;
using ::hipsolverDnZsytrf_bufferSize;

// ========================================================================
// QR Factorization (GEQRF)
// ========================================================================
using ::hipsolverDnCgeqrf;
using ::hipsolverDnCgeqrf_bufferSize;
using ::hipsolverDnDgeqrf;
using ::hipsolverDnDgeqrf_bufferSize;
using ::hipsolverDnSgeqrf;
using ::hipsolverDnSgeqrf_bufferSize;
using ::hipsolverDnZgeqrf;
using ::hipsolverDnZgeqrf_bufferSize;

// ========================================================================
// Generate Orthogonal/Unitary Matrix Q (ORGQR/UNGQR)
// ========================================================================
using ::hipsolverDnCungqr;
using ::hipsolverDnCungqr_bufferSize;
using ::hipsolverDnDorgqr;
using ::hipsolverDnDorgqr_bufferSize;
using ::hipsolverDnSorgqr;
using ::hipsolverDnSorgqr_bufferSize;
using ::hipsolverDnZungqr;
using ::hipsolverDnZungqr_bufferSize;

// ========================================================================
// Multiply by Q (ORMQR/UNMQR)
// ========================================================================
using ::hipsolverDnCunmqr;
using ::hipsolverDnCunmqr_bufferSize;
using ::hipsolverDnDormqr;
using ::hipsolverDnDormqr_bufferSize;
using ::hipsolverDnSormqr;
using ::hipsolverDnSormqr_bufferSize;
using ::hipsolverDnZunmqr;
using ::hipsolverDnZunmqr_bufferSize;

// ========================================================================
// Bidiagonal Reduction (GEBRD)
// ========================================================================
using ::hipsolverDnCgebrd;
using ::hipsolverDnCgebrd_bufferSize;
using ::hipsolverDnDgebrd;
using ::hipsolverDnDgebrd_bufferSize;
using ::hipsolverDnSgebrd;
using ::hipsolverDnSgebrd_bufferSize;
using ::hipsolverDnZgebrd;
using ::hipsolverDnZgebrd_bufferSize;

// ========================================================================
// Generate Orthogonal/Unitary Matrices from GEBRD (ORGBR/UNGBR)
// ========================================================================
using ::hipsolverDnCungbr;
using ::hipsolverDnCungbr_bufferSize;
using ::hipsolverDnDorgbr;
using ::hipsolverDnDorgbr_bufferSize;
using ::hipsolverDnSorgbr;
using ::hipsolverDnSorgbr_bufferSize;
using ::hipsolverDnZungbr;
using ::hipsolverDnZungbr_bufferSize;

// ========================================================================
// Tridiagonal Reduction (SYTRD/HETRD)
// ========================================================================
using ::hipsolverDnChetrd;
using ::hipsolverDnChetrd_bufferSize;
using ::hipsolverDnDsytrd;
using ::hipsolverDnDsytrd_bufferSize;
using ::hipsolverDnSsytrd;
using ::hipsolverDnSsytrd_bufferSize;
using ::hipsolverDnZhetrd;
using ::hipsolverDnZhetrd_bufferSize;

// ========================================================================
// Generate Orthogonal/Unitary Matrix from SYTRD/HETRD (ORGTR/UNGTR)
// ========================================================================
using ::hipsolverDnCungtr;
using ::hipsolverDnCungtr_bufferSize;
using ::hipsolverDnDorgtr;
using ::hipsolverDnDorgtr_bufferSize;
using ::hipsolverDnSorgtr;
using ::hipsolverDnSorgtr_bufferSize;
using ::hipsolverDnZungtr;
using ::hipsolverDnZungtr_bufferSize;

// ========================================================================
// Multiply by Orthogonal/Unitary Matrix from SYTRD/HETRD (ORMTR/UNMTR)
// ========================================================================
using ::hipsolverDnCunmtr;
using ::hipsolverDnCunmtr_bufferSize;
using ::hipsolverDnDormtr;
using ::hipsolverDnDormtr_bufferSize;
using ::hipsolverDnSormtr;
using ::hipsolverDnSormtr_bufferSize;
using ::hipsolverDnZunmtr;
using ::hipsolverDnZunmtr_bufferSize;

// ========================================================================
// SVD - Singular Value Decomposition (GESVD)
// ========================================================================
using ::hipsolverDnCgesvd;
using ::hipsolverDnCgesvd_bufferSize;
using ::hipsolverDnDgesvd;
using ::hipsolverDnDgesvd_bufferSize;
using ::hipsolverDnSgesvd;
using ::hipsolverDnSgesvd_bufferSize;
using ::hipsolverDnZgesvd;
using ::hipsolverDnZgesvd_bufferSize;

// ========================================================================
// SVD - Jacobi Algorithm (GESVDJ)
// ========================================================================
using ::hipsolverDnCgesvdj;
using ::hipsolverDnCgesvdj_bufferSize;
using ::hipsolverDnDgesvdj;
using ::hipsolverDnDgesvdj_bufferSize;
using ::hipsolverDnSgesvdj;
using ::hipsolverDnSgesvdj_bufferSize;
using ::hipsolverDnZgesvdj;
using ::hipsolverDnZgesvdj_bufferSize;

// Batched Jacobi SVD
using ::hipsolverDnCgesvdjBatched;
using ::hipsolverDnCgesvdjBatched_bufferSize;
using ::hipsolverDnDgesvdjBatched;
using ::hipsolverDnDgesvdjBatched_bufferSize;
using ::hipsolverDnSgesvdjBatched;
using ::hipsolverDnSgesvdjBatched_bufferSize;
using ::hipsolverDnZgesvdjBatched;
using ::hipsolverDnZgesvdjBatched_bufferSize;

// ========================================================================
// SVD - Approximate Algorithm (GESVDA - Strided Batched)
// ========================================================================
using ::hipsolverDnCgesvdaStridedBatched;
using ::hipsolverDnCgesvdaStridedBatched_bufferSize;
using ::hipsolverDnDgesvdaStridedBatched;
using ::hipsolverDnDgesvdaStridedBatched_bufferSize;
using ::hipsolverDnSgesvdaStridedBatched;
using ::hipsolverDnSgesvdaStridedBatched_bufferSize;
using ::hipsolverDnZgesvdaStridedBatched;
using ::hipsolverDnZgesvdaStridedBatched_bufferSize;

// ========================================================================
// Symmetric Eigenvalue Problem (SYEVD/HEEVD)
// ========================================================================
using ::hipsolverDnCheevd;
using ::hipsolverDnCheevd_bufferSize;
using ::hipsolverDnDsyevd;
using ::hipsolverDnDsyevd_bufferSize;
using ::hipsolverDnSsyevd;
using ::hipsolverDnSsyevd_bufferSize;
using ::hipsolverDnZheevd;
using ::hipsolverDnZheevd_bufferSize;

// ========================================================================
// Symmetric Eigenvalue Problem with Range Selection (SYEVDX/HEEVDX)
// ========================================================================
using ::hipsolverDnCheevdx;
using ::hipsolverDnCheevdx_bufferSize;
using ::hipsolverDnDsyevdx;
using ::hipsolverDnDsyevdx_bufferSize;
using ::hipsolverDnSsyevdx;
using ::hipsolverDnSsyevdx_bufferSize;
using ::hipsolverDnZheevdx;
using ::hipsolverDnZheevdx_bufferSize;

// ========================================================================
// Symmetric Eigenvalue Problem - Jacobi Algorithm (SYEVJ/HEEVJ)
// ========================================================================
using ::hipsolverDnCheevj;
using ::hipsolverDnCheevj_bufferSize;
using ::hipsolverDnDsyevj;
using ::hipsolverDnDsyevj_bufferSize;
using ::hipsolverDnSsyevj;
using ::hipsolverDnSsyevj_bufferSize;
using ::hipsolverDnZheevj;
using ::hipsolverDnZheevj_bufferSize;

// Batched Jacobi Eigenvalue
using ::hipsolverDnCheevjBatched;
using ::hipsolverDnCheevjBatched_bufferSize;
using ::hipsolverDnDsyevjBatched;
using ::hipsolverDnDsyevjBatched_bufferSize;
using ::hipsolverDnSsyevjBatched;
using ::hipsolverDnSsyevjBatched_bufferSize;
using ::hipsolverDnZheevjBatched;
using ::hipsolverDnZheevjBatched_bufferSize;

// ========================================================================
// Generalized Symmetric Eigenvalue Problem (SYGVD/HEGVD)
// ========================================================================
using ::hipsolverDnChegvd;
using ::hipsolverDnChegvd_bufferSize;
using ::hipsolverDnDsygvd;
using ::hipsolverDnDsygvd_bufferSize;
using ::hipsolverDnSsygvd;
using ::hipsolverDnSsygvd_bufferSize;
using ::hipsolverDnZhegvd;
using ::hipsolverDnZhegvd_bufferSize;

// ========================================================================
// Generalized Symmetric Eigenvalue Problem with Range (SYGVDX/HEGVDX)
// ========================================================================
using ::hipsolverDnChegvdx;
using ::hipsolverDnChegvdx_bufferSize;
using ::hipsolverDnDsygvdx;
using ::hipsolverDnDsygvdx_bufferSize;
using ::hipsolverDnSsygvdx;
using ::hipsolverDnSsygvdx_bufferSize;
using ::hipsolverDnZhegvdx;
using ::hipsolverDnZhegvdx_bufferSize;

// ========================================================================
// Generalized Symmetric Eigenvalue Problem - Jacobi (SYGVJ/HEGVJ)
// ========================================================================
using ::hipsolverDnChegvj;
using ::hipsolverDnChegvj_bufferSize;
using ::hipsolverDnDsygvj;
using ::hipsolverDnDsygvj_bufferSize;
using ::hipsolverDnSsygvj;
using ::hipsolverDnSsygvj_bufferSize;
using ::hipsolverDnZhegvj;
using ::hipsolverDnZhegvj_bufferSize;

// ========================================================================
// Sparse: Context Management (hipsolverSp)
// ========================================================================
using ::hipsolverSpCreate;
using ::hipsolverSpDestroy;
using ::hipsolverSpSetStream;

// ========================================================================
// Sparse: Cholesky Linear Solver (device and host variants; single/double
// precision only -- no complex Cholesky counterpart exists)
// ========================================================================
using ::hipsolverSpDcsrlsvchol;
using ::hipsolverSpDcsrlsvcholHost;
using ::hipsolverSpScsrlsvchol;
using ::hipsolverSpScsrlsvcholHost;

// ========================================================================
// Sparse: QR Linear Solver (device-only; all four precisions)
// ========================================================================
using ::hipsolverSpCcsrlsvqr;
using ::hipsolverSpDcsrlsvqr;
using ::hipsolverSpScsrlsvqr;
using ::hipsolverSpZcsrlsvqr;

} // namespace wwr::hip
