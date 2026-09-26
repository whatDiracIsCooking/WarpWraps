// hipsolver.cppm - Compile-time tests for gpumod.hip.hipsolver

module;

#include "test/shared/link_check.h"

export module gpumod.test.hip.hipsolver;

import std;
import gpumod.hip.hipsolver;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time tests for gpumod.hip.hipsolver
//
// Covers both the dense (hipsolverDn*) and narrow sparse (hipsolverSp*)
// surface this one module wraps -- see src/hip/hipsolver.cppm and
// src/hip/README.md for the collapse-plus-narrowing rationale (issue #177).
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace gpumod::hip::test {

using namespace gpumod::hip;

// ────────────────────────────────────────────────────────────────────────
// Enum type checks
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_enum_v<hipsolverStatus_t>);
static_assert(std::is_enum_v<hipsolverEigMode_t>);
static_assert(std::is_enum_v<hipsolverEigType_t>);
static_assert(std::is_enum_v<hipsolverEigRange_t>);
static_assert(std::is_enum_v<hipsolverDeterministicMode_t>);
static_assert(std::is_enum_v<hipsolverAlgMode_t>);
static_assert(std::is_enum_v<hipsolverDnFunction_t>);

// ────────────────────────────────────────────────────────────────────────
// Enum values: hipsolverStatus_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(HIPSOLVER_STATUS_SUCCESS) == 0);
static_assert(static_cast<int>(HIPSOLVER_STATUS_NOT_INITIALIZED) == 1);
static_assert(static_cast<int>(HIPSOLVER_STATUS_ALLOC_FAILED) == 2);
static_assert(static_cast<int>(HIPSOLVER_STATUS_INVALID_VALUE) == 3);
static_assert(static_cast<int>(HIPSOLVER_STATUS_MAPPING_ERROR) == 4);
static_assert(static_cast<int>(HIPSOLVER_STATUS_EXECUTION_FAILED) == 5);
static_assert(static_cast<int>(HIPSOLVER_STATUS_INTERNAL_ERROR) == 6);
static_assert(static_cast<int>(HIPSOLVER_STATUS_NOT_SUPPORTED) == 7);
static_assert(static_cast<int>(HIPSOLVER_STATUS_ARCH_MISMATCH) == 8);
static_assert(static_cast<int>(HIPSOLVER_STATUS_HANDLE_IS_NULLPTR) == 9);
static_assert(static_cast<int>(HIPSOLVER_STATUS_INVALID_ENUM) == 10);
static_assert(static_cast<int>(HIPSOLVER_STATUS_UNKNOWN) == 11);
static_assert(static_cast<int>(HIPSOLVER_STATUS_ZERO_PIVOT) == 12);
static_assert(static_cast<int>(HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED) == 13);

// ────────────────────────────────────────────────────────────────────────
// Enum values: hipsolverEigMode_t / hipsolverEigType_t / hipsolverEigRange_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(HIPSOLVER_EIG_MODE_NOVECTOR) == 201);
static_assert(static_cast<int>(HIPSOLVER_EIG_MODE_VECTOR) == 202);

static_assert(static_cast<int>(HIPSOLVER_EIG_TYPE_1) == 211);
static_assert(static_cast<int>(HIPSOLVER_EIG_TYPE_2) == 212);
static_assert(static_cast<int>(HIPSOLVER_EIG_TYPE_3) == 213);

static_assert(static_cast<int>(HIPSOLVER_EIG_RANGE_ALL) == 221);
static_assert(static_cast<int>(HIPSOLVER_EIG_RANGE_V) == 222);
static_assert(static_cast<int>(HIPSOLVER_EIG_RANGE_I) == 223);

// ────────────────────────────────────────────────────────────────────────
// Enum values: hipsolverDeterministicMode_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(HIPSOLVER_DETERMINISTIC_RESULTS) == 241);
static_assert(static_cast<int>(HIPSOLVER_ALLOW_NON_DETERMINISTIC_RESULTS) == 242);

// ────────────────────────────────────────────────────────────────────────
// Enum values: hipsolverAlgMode_t / hipsolverDnFunction_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(HIPSOLVER_ALG_0) == 231);
static_assert(static_cast<int>(HIPSOLVER_ALG_1) == 232);

static_assert(static_cast<int>(HIPSOLVERDN_GETRF) == 0);

// ────────────────────────────────────────────────────────────────────────
// Handle type traits (opaque pointer types)
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_pointer_v<hipsolverHandle_t>);
static_assert(std::is_same_v<hipsolverDnHandle_t, hipsolverHandle_t>);
static_assert(std::is_pointer_v<hipsolverGesvdjInfo_t>);
static_assert(std::is_pointer_v<hipsolverSyevjInfo_t>);
static_assert(std::is_pointer_v<hipsolverDnParams_t>);
static_assert(std::is_pointer_v<hipsolverSpHandle_t>);
static_assert(std::is_pointer_v<hipsparseMatDescr_t>);

// hipsolver's own names for the hipblas operation/fill-mode/side-mode types
// used by the X-prefixed 64-bit generic API
static_assert(std::is_enum_v<hipsolverOperation_t>);
static_assert(std::is_enum_v<hipsolverFillMode_t>);
static_assert(std::is_enum_v<hipsolverSideMode_t>);

// ────────────────────────────────────────────────────────────────────────
// GPUMOD_LINK_CHECK: dense (hipsolverDn) + sparse (hipsolverSp) function surface
// ────────────────────────────────────────────────────────────────────────

GPUMOD_LINK_CHECK(hipsolverDnCreate)
GPUMOD_LINK_CHECK(hipsolverDnDestroy)
GPUMOD_LINK_CHECK(hipsolverDnSetStream)
GPUMOD_LINK_CHECK(hipsolverDnGetStream)
GPUMOD_LINK_CHECK(hipsolverDnSetDeterministicMode)
GPUMOD_LINK_CHECK(hipsolverDnGetDeterministicMode)
GPUMOD_LINK_CHECK(hipsolverDnSetAdvOptions)
GPUMOD_LINK_CHECK(hipsolverDnCreateGesvdjInfo)
GPUMOD_LINK_CHECK(hipsolverDnDestroyGesvdjInfo)
GPUMOD_LINK_CHECK(hipsolverDnXgesvdjSetMaxSweeps)
GPUMOD_LINK_CHECK(hipsolverDnXgesvdjSetSortEig)
GPUMOD_LINK_CHECK(hipsolverDnXgesvdjSetTolerance)
GPUMOD_LINK_CHECK(hipsolverDnXgesvdjGetResidual)
GPUMOD_LINK_CHECK(hipsolverDnXgesvdjGetSweeps)
GPUMOD_LINK_CHECK(hipsolverDnCreateSyevjInfo)
GPUMOD_LINK_CHECK(hipsolverDnDestroySyevjInfo)
GPUMOD_LINK_CHECK(hipsolverDnXsyevjSetMaxSweeps)
GPUMOD_LINK_CHECK(hipsolverDnXsyevjSetSortEig)
GPUMOD_LINK_CHECK(hipsolverDnXsyevjSetTolerance)
GPUMOD_LINK_CHECK(hipsolverDnXsyevjGetResidual)
GPUMOD_LINK_CHECK(hipsolverDnXsyevjGetSweeps)
GPUMOD_LINK_CHECK(hipsolverDnCreateParams)
GPUMOD_LINK_CHECK(hipsolverDnDestroyParams)
GPUMOD_LINK_CHECK(hipsolverDnXgeqrf)
GPUMOD_LINK_CHECK(hipsolverDnXgeqrf_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnXgetrf)
GPUMOD_LINK_CHECK(hipsolverDnXgetrf_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnXgetrs)
GPUMOD_LINK_CHECK(hipsolverDnXpotrf)
GPUMOD_LINK_CHECK(hipsolverDnXpotrf_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnXpotrs)
GPUMOD_LINK_CHECK(hipsolverDnSSgesv)
GPUMOD_LINK_CHECK(hipsolverDnSSgesv_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnDDgesv)
GPUMOD_LINK_CHECK(hipsolverDnDDgesv_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnCCgesv)
GPUMOD_LINK_CHECK(hipsolverDnCCgesv_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnZZgesv)
GPUMOD_LINK_CHECK(hipsolverDnZZgesv_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnSSgels)
GPUMOD_LINK_CHECK(hipsolverDnSSgels_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnDDgels)
GPUMOD_LINK_CHECK(hipsolverDnDDgels_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnCCgels)
GPUMOD_LINK_CHECK(hipsolverDnCCgels_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnZZgels)
GPUMOD_LINK_CHECK(hipsolverDnZZgels_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnSgetrf)
GPUMOD_LINK_CHECK(hipsolverDnSgetrf_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnDgetrf)
GPUMOD_LINK_CHECK(hipsolverDnDgetrf_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnCgetrf)
GPUMOD_LINK_CHECK(hipsolverDnCgetrf_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnZgetrf)
GPUMOD_LINK_CHECK(hipsolverDnZgetrf_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnSgetrs)
GPUMOD_LINK_CHECK(hipsolverDnDgetrs)
GPUMOD_LINK_CHECK(hipsolverDnCgetrs)
GPUMOD_LINK_CHECK(hipsolverDnZgetrs)
GPUMOD_LINK_CHECK(hipsolverDnSpotrf)
GPUMOD_LINK_CHECK(hipsolverDnSpotrf_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnDpotrf)
GPUMOD_LINK_CHECK(hipsolverDnDpotrf_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnCpotrf)
GPUMOD_LINK_CHECK(hipsolverDnCpotrf_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnZpotrf)
GPUMOD_LINK_CHECK(hipsolverDnZpotrf_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnSpotrfBatched)
GPUMOD_LINK_CHECK(hipsolverDnDpotrfBatched)
GPUMOD_LINK_CHECK(hipsolverDnCpotrfBatched)
GPUMOD_LINK_CHECK(hipsolverDnZpotrfBatched)
GPUMOD_LINK_CHECK(hipsolverDnSpotrs)
GPUMOD_LINK_CHECK(hipsolverDnDpotrs)
GPUMOD_LINK_CHECK(hipsolverDnCpotrs)
GPUMOD_LINK_CHECK(hipsolverDnZpotrs)
GPUMOD_LINK_CHECK(hipsolverDnSpotrsBatched)
GPUMOD_LINK_CHECK(hipsolverDnDpotrsBatched)
GPUMOD_LINK_CHECK(hipsolverDnCpotrsBatched)
GPUMOD_LINK_CHECK(hipsolverDnZpotrsBatched)
GPUMOD_LINK_CHECK(hipsolverDnSpotri)
GPUMOD_LINK_CHECK(hipsolverDnSpotri_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnDpotri)
GPUMOD_LINK_CHECK(hipsolverDnDpotri_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnCpotri)
GPUMOD_LINK_CHECK(hipsolverDnCpotri_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnZpotri)
GPUMOD_LINK_CHECK(hipsolverDnZpotri_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnSsytrf)
GPUMOD_LINK_CHECK(hipsolverDnSsytrf_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnDsytrf)
GPUMOD_LINK_CHECK(hipsolverDnDsytrf_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnCsytrf)
GPUMOD_LINK_CHECK(hipsolverDnCsytrf_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnZsytrf)
GPUMOD_LINK_CHECK(hipsolverDnZsytrf_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnSgeqrf)
GPUMOD_LINK_CHECK(hipsolverDnSgeqrf_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnDgeqrf)
GPUMOD_LINK_CHECK(hipsolverDnDgeqrf_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnCgeqrf)
GPUMOD_LINK_CHECK(hipsolverDnCgeqrf_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnZgeqrf)
GPUMOD_LINK_CHECK(hipsolverDnZgeqrf_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnSorgqr)
GPUMOD_LINK_CHECK(hipsolverDnSorgqr_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnDorgqr)
GPUMOD_LINK_CHECK(hipsolverDnDorgqr_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnCungqr)
GPUMOD_LINK_CHECK(hipsolverDnCungqr_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnZungqr)
GPUMOD_LINK_CHECK(hipsolverDnZungqr_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnSormqr)
GPUMOD_LINK_CHECK(hipsolverDnSormqr_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnDormqr)
GPUMOD_LINK_CHECK(hipsolverDnDormqr_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnCunmqr)
GPUMOD_LINK_CHECK(hipsolverDnCunmqr_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnZunmqr)
GPUMOD_LINK_CHECK(hipsolverDnZunmqr_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnSgebrd)
GPUMOD_LINK_CHECK(hipsolverDnSgebrd_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnDgebrd)
GPUMOD_LINK_CHECK(hipsolverDnDgebrd_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnCgebrd)
GPUMOD_LINK_CHECK(hipsolverDnCgebrd_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnZgebrd)
GPUMOD_LINK_CHECK(hipsolverDnZgebrd_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnSorgbr)
GPUMOD_LINK_CHECK(hipsolverDnSorgbr_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnDorgbr)
GPUMOD_LINK_CHECK(hipsolverDnDorgbr_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnCungbr)
GPUMOD_LINK_CHECK(hipsolverDnCungbr_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnZungbr)
GPUMOD_LINK_CHECK(hipsolverDnZungbr_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnSsytrd)
GPUMOD_LINK_CHECK(hipsolverDnSsytrd_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnDsytrd)
GPUMOD_LINK_CHECK(hipsolverDnDsytrd_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnChetrd)
GPUMOD_LINK_CHECK(hipsolverDnChetrd_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnZhetrd)
GPUMOD_LINK_CHECK(hipsolverDnZhetrd_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnSorgtr)
GPUMOD_LINK_CHECK(hipsolverDnSorgtr_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnDorgtr)
GPUMOD_LINK_CHECK(hipsolverDnDorgtr_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnCungtr)
GPUMOD_LINK_CHECK(hipsolverDnCungtr_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnZungtr)
GPUMOD_LINK_CHECK(hipsolverDnZungtr_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnSormtr)
GPUMOD_LINK_CHECK(hipsolverDnSormtr_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnDormtr)
GPUMOD_LINK_CHECK(hipsolverDnDormtr_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnCunmtr)
GPUMOD_LINK_CHECK(hipsolverDnCunmtr_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnZunmtr)
GPUMOD_LINK_CHECK(hipsolverDnZunmtr_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnSgesvd)
GPUMOD_LINK_CHECK(hipsolverDnSgesvd_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnDgesvd)
GPUMOD_LINK_CHECK(hipsolverDnDgesvd_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnCgesvd)
GPUMOD_LINK_CHECK(hipsolverDnCgesvd_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnZgesvd)
GPUMOD_LINK_CHECK(hipsolverDnZgesvd_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnSgesvdj)
GPUMOD_LINK_CHECK(hipsolverDnSgesvdj_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnDgesvdj)
GPUMOD_LINK_CHECK(hipsolverDnDgesvdj_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnCgesvdj)
GPUMOD_LINK_CHECK(hipsolverDnCgesvdj_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnZgesvdj)
GPUMOD_LINK_CHECK(hipsolverDnZgesvdj_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnSgesvdjBatched)
GPUMOD_LINK_CHECK(hipsolverDnSgesvdjBatched_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnDgesvdjBatched)
GPUMOD_LINK_CHECK(hipsolverDnDgesvdjBatched_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnCgesvdjBatched)
GPUMOD_LINK_CHECK(hipsolverDnCgesvdjBatched_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnZgesvdjBatched)
GPUMOD_LINK_CHECK(hipsolverDnZgesvdjBatched_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnSgesvdaStridedBatched)
GPUMOD_LINK_CHECK(hipsolverDnSgesvdaStridedBatched_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnDgesvdaStridedBatched)
GPUMOD_LINK_CHECK(hipsolverDnDgesvdaStridedBatched_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnCgesvdaStridedBatched)
GPUMOD_LINK_CHECK(hipsolverDnCgesvdaStridedBatched_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnZgesvdaStridedBatched)
GPUMOD_LINK_CHECK(hipsolverDnZgesvdaStridedBatched_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnSsyevd)
GPUMOD_LINK_CHECK(hipsolverDnSsyevd_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnDsyevd)
GPUMOD_LINK_CHECK(hipsolverDnDsyevd_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnCheevd)
GPUMOD_LINK_CHECK(hipsolverDnCheevd_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnZheevd)
GPUMOD_LINK_CHECK(hipsolverDnZheevd_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnSsyevdx)
GPUMOD_LINK_CHECK(hipsolverDnSsyevdx_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnDsyevdx)
GPUMOD_LINK_CHECK(hipsolverDnDsyevdx_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnCheevdx)
GPUMOD_LINK_CHECK(hipsolverDnCheevdx_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnZheevdx)
GPUMOD_LINK_CHECK(hipsolverDnZheevdx_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnSsyevj)
GPUMOD_LINK_CHECK(hipsolverDnSsyevj_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnDsyevj)
GPUMOD_LINK_CHECK(hipsolverDnDsyevj_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnCheevj)
GPUMOD_LINK_CHECK(hipsolverDnCheevj_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnZheevj)
GPUMOD_LINK_CHECK(hipsolverDnZheevj_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnSsyevjBatched)
GPUMOD_LINK_CHECK(hipsolverDnSsyevjBatched_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnDsyevjBatched)
GPUMOD_LINK_CHECK(hipsolverDnDsyevjBatched_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnCheevjBatched)
GPUMOD_LINK_CHECK(hipsolverDnCheevjBatched_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnZheevjBatched)
GPUMOD_LINK_CHECK(hipsolverDnZheevjBatched_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnSsygvd)
GPUMOD_LINK_CHECK(hipsolverDnSsygvd_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnDsygvd)
GPUMOD_LINK_CHECK(hipsolverDnDsygvd_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnChegvd)
GPUMOD_LINK_CHECK(hipsolverDnChegvd_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnZhegvd)
GPUMOD_LINK_CHECK(hipsolverDnZhegvd_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnSsygvdx)
GPUMOD_LINK_CHECK(hipsolverDnSsygvdx_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnDsygvdx)
GPUMOD_LINK_CHECK(hipsolverDnDsygvdx_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnChegvdx)
GPUMOD_LINK_CHECK(hipsolverDnChegvdx_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnZhegvdx)
GPUMOD_LINK_CHECK(hipsolverDnZhegvdx_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnSsygvj)
GPUMOD_LINK_CHECK(hipsolverDnSsygvj_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnDsygvj)
GPUMOD_LINK_CHECK(hipsolverDnDsygvj_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnChegvj)
GPUMOD_LINK_CHECK(hipsolverDnChegvj_bufferSize)
GPUMOD_LINK_CHECK(hipsolverDnZhegvj)
GPUMOD_LINK_CHECK(hipsolverDnZhegvj_bufferSize)
GPUMOD_LINK_CHECK(hipsolverSpCreate)
GPUMOD_LINK_CHECK(hipsolverSpDestroy)
GPUMOD_LINK_CHECK(hipsolverSpSetStream)
GPUMOD_LINK_CHECK(hipsolverSpScsrlsvchol)
GPUMOD_LINK_CHECK(hipsolverSpDcsrlsvchol)
GPUMOD_LINK_CHECK(hipsolverSpScsrlsvcholHost)
GPUMOD_LINK_CHECK(hipsolverSpDcsrlsvcholHost)
GPUMOD_LINK_CHECK(hipsolverSpScsrlsvqr)
GPUMOD_LINK_CHECK(hipsolverSpDcsrlsvqr)
GPUMOD_LINK_CHECK(hipsolverSpCcsrlsvqr)
GPUMOD_LINK_CHECK(hipsolverSpZcsrlsvqr)
} // namespace gpumod::hip::test
