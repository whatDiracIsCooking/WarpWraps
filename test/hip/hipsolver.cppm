// hipsolver.cppm - Compile-time tests for wwr.hip.hipsolver

module;

#include "test/shared/link_check.h"

export module wwr.test.hip.hipsolver;

import std;
import wwr.hip.hipsolver;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time tests for wwr.hip.hipsolver
//
// Covers both the dense (hipsolverDn*) and narrow sparse (hipsolverSp*)
// surface this one module wraps -- see src/hip/hipsolver.cppm and
// src/hip/README.md for the collapse-plus-narrowing rationale (issue #177).
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace wwr::hip::test {

using namespace wwr::hip;

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
// WWR_LINK_CHECK: dense (hipsolverDn) + sparse (hipsolverSp) function surface
// ────────────────────────────────────────────────────────────────────────

WWR_LINK_CHECK(hipsolverDnCreate)
WWR_LINK_CHECK(hipsolverDnDestroy)
WWR_LINK_CHECK(hipsolverDnSetStream)
WWR_LINK_CHECK(hipsolverDnGetStream)
WWR_LINK_CHECK(hipsolverDnSetDeterministicMode)
WWR_LINK_CHECK(hipsolverDnGetDeterministicMode)
WWR_LINK_CHECK(hipsolverDnSetAdvOptions)
WWR_LINK_CHECK(hipsolverDnCreateGesvdjInfo)
WWR_LINK_CHECK(hipsolverDnDestroyGesvdjInfo)
WWR_LINK_CHECK(hipsolverDnXgesvdjSetMaxSweeps)
WWR_LINK_CHECK(hipsolverDnXgesvdjSetSortEig)
WWR_LINK_CHECK(hipsolverDnXgesvdjSetTolerance)
WWR_LINK_CHECK(hipsolverDnXgesvdjGetResidual)
WWR_LINK_CHECK(hipsolverDnXgesvdjGetSweeps)
WWR_LINK_CHECK(hipsolverDnCreateSyevjInfo)
WWR_LINK_CHECK(hipsolverDnDestroySyevjInfo)
WWR_LINK_CHECK(hipsolverDnXsyevjSetMaxSweeps)
WWR_LINK_CHECK(hipsolverDnXsyevjSetSortEig)
WWR_LINK_CHECK(hipsolverDnXsyevjSetTolerance)
WWR_LINK_CHECK(hipsolverDnXsyevjGetResidual)
WWR_LINK_CHECK(hipsolverDnXsyevjGetSweeps)
WWR_LINK_CHECK(hipsolverDnCreateParams)
WWR_LINK_CHECK(hipsolverDnDestroyParams)
WWR_LINK_CHECK(hipsolverDnXgeqrf)
WWR_LINK_CHECK(hipsolverDnXgeqrf_bufferSize)
WWR_LINK_CHECK(hipsolverDnXgetrf)
WWR_LINK_CHECK(hipsolverDnXgetrf_bufferSize)
WWR_LINK_CHECK(hipsolverDnXgetrs)
WWR_LINK_CHECK(hipsolverDnXpotrf)
WWR_LINK_CHECK(hipsolverDnXpotrf_bufferSize)
WWR_LINK_CHECK(hipsolverDnXpotrs)
WWR_LINK_CHECK(hipsolverDnSSgesv)
WWR_LINK_CHECK(hipsolverDnSSgesv_bufferSize)
WWR_LINK_CHECK(hipsolverDnDDgesv)
WWR_LINK_CHECK(hipsolverDnDDgesv_bufferSize)
WWR_LINK_CHECK(hipsolverDnCCgesv)
WWR_LINK_CHECK(hipsolverDnCCgesv_bufferSize)
WWR_LINK_CHECK(hipsolverDnZZgesv)
WWR_LINK_CHECK(hipsolverDnZZgesv_bufferSize)
WWR_LINK_CHECK(hipsolverDnSSgels)
WWR_LINK_CHECK(hipsolverDnSSgels_bufferSize)
WWR_LINK_CHECK(hipsolverDnDDgels)
WWR_LINK_CHECK(hipsolverDnDDgels_bufferSize)
WWR_LINK_CHECK(hipsolverDnCCgels)
WWR_LINK_CHECK(hipsolverDnCCgels_bufferSize)
WWR_LINK_CHECK(hipsolverDnZZgels)
WWR_LINK_CHECK(hipsolverDnZZgels_bufferSize)
WWR_LINK_CHECK(hipsolverDnSgetrf)
WWR_LINK_CHECK(hipsolverDnSgetrf_bufferSize)
WWR_LINK_CHECK(hipsolverDnDgetrf)
WWR_LINK_CHECK(hipsolverDnDgetrf_bufferSize)
WWR_LINK_CHECK(hipsolverDnCgetrf)
WWR_LINK_CHECK(hipsolverDnCgetrf_bufferSize)
WWR_LINK_CHECK(hipsolverDnZgetrf)
WWR_LINK_CHECK(hipsolverDnZgetrf_bufferSize)
WWR_LINK_CHECK(hipsolverDnSgetrs)
WWR_LINK_CHECK(hipsolverDnDgetrs)
WWR_LINK_CHECK(hipsolverDnCgetrs)
WWR_LINK_CHECK(hipsolverDnZgetrs)
WWR_LINK_CHECK(hipsolverDnSpotrf)
WWR_LINK_CHECK(hipsolverDnSpotrf_bufferSize)
WWR_LINK_CHECK(hipsolverDnDpotrf)
WWR_LINK_CHECK(hipsolverDnDpotrf_bufferSize)
WWR_LINK_CHECK(hipsolverDnCpotrf)
WWR_LINK_CHECK(hipsolverDnCpotrf_bufferSize)
WWR_LINK_CHECK(hipsolverDnZpotrf)
WWR_LINK_CHECK(hipsolverDnZpotrf_bufferSize)
WWR_LINK_CHECK(hipsolverDnSpotrfBatched)
WWR_LINK_CHECK(hipsolverDnDpotrfBatched)
WWR_LINK_CHECK(hipsolverDnCpotrfBatched)
WWR_LINK_CHECK(hipsolverDnZpotrfBatched)
WWR_LINK_CHECK(hipsolverDnSpotrs)
WWR_LINK_CHECK(hipsolverDnDpotrs)
WWR_LINK_CHECK(hipsolverDnCpotrs)
WWR_LINK_CHECK(hipsolverDnZpotrs)
WWR_LINK_CHECK(hipsolverDnSpotrsBatched)
WWR_LINK_CHECK(hipsolverDnDpotrsBatched)
WWR_LINK_CHECK(hipsolverDnCpotrsBatched)
WWR_LINK_CHECK(hipsolverDnZpotrsBatched)
WWR_LINK_CHECK(hipsolverDnSpotri)
WWR_LINK_CHECK(hipsolverDnSpotri_bufferSize)
WWR_LINK_CHECK(hipsolverDnDpotri)
WWR_LINK_CHECK(hipsolverDnDpotri_bufferSize)
WWR_LINK_CHECK(hipsolverDnCpotri)
WWR_LINK_CHECK(hipsolverDnCpotri_bufferSize)
WWR_LINK_CHECK(hipsolverDnZpotri)
WWR_LINK_CHECK(hipsolverDnZpotri_bufferSize)
WWR_LINK_CHECK(hipsolverDnSsytrf)
WWR_LINK_CHECK(hipsolverDnSsytrf_bufferSize)
WWR_LINK_CHECK(hipsolverDnDsytrf)
WWR_LINK_CHECK(hipsolverDnDsytrf_bufferSize)
WWR_LINK_CHECK(hipsolverDnCsytrf)
WWR_LINK_CHECK(hipsolverDnCsytrf_bufferSize)
WWR_LINK_CHECK(hipsolverDnZsytrf)
WWR_LINK_CHECK(hipsolverDnZsytrf_bufferSize)
WWR_LINK_CHECK(hipsolverDnSgeqrf)
WWR_LINK_CHECK(hipsolverDnSgeqrf_bufferSize)
WWR_LINK_CHECK(hipsolverDnDgeqrf)
WWR_LINK_CHECK(hipsolverDnDgeqrf_bufferSize)
WWR_LINK_CHECK(hipsolverDnCgeqrf)
WWR_LINK_CHECK(hipsolverDnCgeqrf_bufferSize)
WWR_LINK_CHECK(hipsolverDnZgeqrf)
WWR_LINK_CHECK(hipsolverDnZgeqrf_bufferSize)
WWR_LINK_CHECK(hipsolverDnSorgqr)
WWR_LINK_CHECK(hipsolverDnSorgqr_bufferSize)
WWR_LINK_CHECK(hipsolverDnDorgqr)
WWR_LINK_CHECK(hipsolverDnDorgqr_bufferSize)
WWR_LINK_CHECK(hipsolverDnCungqr)
WWR_LINK_CHECK(hipsolverDnCungqr_bufferSize)
WWR_LINK_CHECK(hipsolverDnZungqr)
WWR_LINK_CHECK(hipsolverDnZungqr_bufferSize)
WWR_LINK_CHECK(hipsolverDnSormqr)
WWR_LINK_CHECK(hipsolverDnSormqr_bufferSize)
WWR_LINK_CHECK(hipsolverDnDormqr)
WWR_LINK_CHECK(hipsolverDnDormqr_bufferSize)
WWR_LINK_CHECK(hipsolverDnCunmqr)
WWR_LINK_CHECK(hipsolverDnCunmqr_bufferSize)
WWR_LINK_CHECK(hipsolverDnZunmqr)
WWR_LINK_CHECK(hipsolverDnZunmqr_bufferSize)
WWR_LINK_CHECK(hipsolverDnSgebrd)
WWR_LINK_CHECK(hipsolverDnSgebrd_bufferSize)
WWR_LINK_CHECK(hipsolverDnDgebrd)
WWR_LINK_CHECK(hipsolverDnDgebrd_bufferSize)
WWR_LINK_CHECK(hipsolverDnCgebrd)
WWR_LINK_CHECK(hipsolverDnCgebrd_bufferSize)
WWR_LINK_CHECK(hipsolverDnZgebrd)
WWR_LINK_CHECK(hipsolverDnZgebrd_bufferSize)
WWR_LINK_CHECK(hipsolverDnSorgbr)
WWR_LINK_CHECK(hipsolverDnSorgbr_bufferSize)
WWR_LINK_CHECK(hipsolverDnDorgbr)
WWR_LINK_CHECK(hipsolverDnDorgbr_bufferSize)
WWR_LINK_CHECK(hipsolverDnCungbr)
WWR_LINK_CHECK(hipsolverDnCungbr_bufferSize)
WWR_LINK_CHECK(hipsolverDnZungbr)
WWR_LINK_CHECK(hipsolverDnZungbr_bufferSize)
WWR_LINK_CHECK(hipsolverDnSsytrd)
WWR_LINK_CHECK(hipsolverDnSsytrd_bufferSize)
WWR_LINK_CHECK(hipsolverDnDsytrd)
WWR_LINK_CHECK(hipsolverDnDsytrd_bufferSize)
WWR_LINK_CHECK(hipsolverDnChetrd)
WWR_LINK_CHECK(hipsolverDnChetrd_bufferSize)
WWR_LINK_CHECK(hipsolverDnZhetrd)
WWR_LINK_CHECK(hipsolverDnZhetrd_bufferSize)
WWR_LINK_CHECK(hipsolverDnSorgtr)
WWR_LINK_CHECK(hipsolverDnSorgtr_bufferSize)
WWR_LINK_CHECK(hipsolverDnDorgtr)
WWR_LINK_CHECK(hipsolverDnDorgtr_bufferSize)
WWR_LINK_CHECK(hipsolverDnCungtr)
WWR_LINK_CHECK(hipsolverDnCungtr_bufferSize)
WWR_LINK_CHECK(hipsolverDnZungtr)
WWR_LINK_CHECK(hipsolverDnZungtr_bufferSize)
WWR_LINK_CHECK(hipsolverDnSormtr)
WWR_LINK_CHECK(hipsolverDnSormtr_bufferSize)
WWR_LINK_CHECK(hipsolverDnDormtr)
WWR_LINK_CHECK(hipsolverDnDormtr_bufferSize)
WWR_LINK_CHECK(hipsolverDnCunmtr)
WWR_LINK_CHECK(hipsolverDnCunmtr_bufferSize)
WWR_LINK_CHECK(hipsolverDnZunmtr)
WWR_LINK_CHECK(hipsolverDnZunmtr_bufferSize)
WWR_LINK_CHECK(hipsolverDnSgesvd)
WWR_LINK_CHECK(hipsolverDnSgesvd_bufferSize)
WWR_LINK_CHECK(hipsolverDnDgesvd)
WWR_LINK_CHECK(hipsolverDnDgesvd_bufferSize)
WWR_LINK_CHECK(hipsolverDnCgesvd)
WWR_LINK_CHECK(hipsolverDnCgesvd_bufferSize)
WWR_LINK_CHECK(hipsolverDnZgesvd)
WWR_LINK_CHECK(hipsolverDnZgesvd_bufferSize)
WWR_LINK_CHECK(hipsolverDnSgesvdj)
WWR_LINK_CHECK(hipsolverDnSgesvdj_bufferSize)
WWR_LINK_CHECK(hipsolverDnDgesvdj)
WWR_LINK_CHECK(hipsolverDnDgesvdj_bufferSize)
WWR_LINK_CHECK(hipsolverDnCgesvdj)
WWR_LINK_CHECK(hipsolverDnCgesvdj_bufferSize)
WWR_LINK_CHECK(hipsolverDnZgesvdj)
WWR_LINK_CHECK(hipsolverDnZgesvdj_bufferSize)
WWR_LINK_CHECK(hipsolverDnSgesvdjBatched)
WWR_LINK_CHECK(hipsolverDnSgesvdjBatched_bufferSize)
WWR_LINK_CHECK(hipsolverDnDgesvdjBatched)
WWR_LINK_CHECK(hipsolverDnDgesvdjBatched_bufferSize)
WWR_LINK_CHECK(hipsolverDnCgesvdjBatched)
WWR_LINK_CHECK(hipsolverDnCgesvdjBatched_bufferSize)
WWR_LINK_CHECK(hipsolverDnZgesvdjBatched)
WWR_LINK_CHECK(hipsolverDnZgesvdjBatched_bufferSize)
WWR_LINK_CHECK(hipsolverDnSgesvdaStridedBatched)
WWR_LINK_CHECK(hipsolverDnSgesvdaStridedBatched_bufferSize)
WWR_LINK_CHECK(hipsolverDnDgesvdaStridedBatched)
WWR_LINK_CHECK(hipsolverDnDgesvdaStridedBatched_bufferSize)
WWR_LINK_CHECK(hipsolverDnCgesvdaStridedBatched)
WWR_LINK_CHECK(hipsolverDnCgesvdaStridedBatched_bufferSize)
WWR_LINK_CHECK(hipsolverDnZgesvdaStridedBatched)
WWR_LINK_CHECK(hipsolverDnZgesvdaStridedBatched_bufferSize)
WWR_LINK_CHECK(hipsolverDnSsyevd)
WWR_LINK_CHECK(hipsolverDnSsyevd_bufferSize)
WWR_LINK_CHECK(hipsolverDnDsyevd)
WWR_LINK_CHECK(hipsolverDnDsyevd_bufferSize)
WWR_LINK_CHECK(hipsolverDnCheevd)
WWR_LINK_CHECK(hipsolverDnCheevd_bufferSize)
WWR_LINK_CHECK(hipsolverDnZheevd)
WWR_LINK_CHECK(hipsolverDnZheevd_bufferSize)
WWR_LINK_CHECK(hipsolverDnSsyevdx)
WWR_LINK_CHECK(hipsolverDnSsyevdx_bufferSize)
WWR_LINK_CHECK(hipsolverDnDsyevdx)
WWR_LINK_CHECK(hipsolverDnDsyevdx_bufferSize)
WWR_LINK_CHECK(hipsolverDnCheevdx)
WWR_LINK_CHECK(hipsolverDnCheevdx_bufferSize)
WWR_LINK_CHECK(hipsolverDnZheevdx)
WWR_LINK_CHECK(hipsolverDnZheevdx_bufferSize)
WWR_LINK_CHECK(hipsolverDnSsyevj)
WWR_LINK_CHECK(hipsolverDnSsyevj_bufferSize)
WWR_LINK_CHECK(hipsolverDnDsyevj)
WWR_LINK_CHECK(hipsolverDnDsyevj_bufferSize)
WWR_LINK_CHECK(hipsolverDnCheevj)
WWR_LINK_CHECK(hipsolverDnCheevj_bufferSize)
WWR_LINK_CHECK(hipsolverDnZheevj)
WWR_LINK_CHECK(hipsolverDnZheevj_bufferSize)
WWR_LINK_CHECK(hipsolverDnSsyevjBatched)
WWR_LINK_CHECK(hipsolverDnSsyevjBatched_bufferSize)
WWR_LINK_CHECK(hipsolverDnDsyevjBatched)
WWR_LINK_CHECK(hipsolverDnDsyevjBatched_bufferSize)
WWR_LINK_CHECK(hipsolverDnCheevjBatched)
WWR_LINK_CHECK(hipsolverDnCheevjBatched_bufferSize)
WWR_LINK_CHECK(hipsolverDnZheevjBatched)
WWR_LINK_CHECK(hipsolverDnZheevjBatched_bufferSize)
WWR_LINK_CHECK(hipsolverDnSsygvd)
WWR_LINK_CHECK(hipsolverDnSsygvd_bufferSize)
WWR_LINK_CHECK(hipsolverDnDsygvd)
WWR_LINK_CHECK(hipsolverDnDsygvd_bufferSize)
WWR_LINK_CHECK(hipsolverDnChegvd)
WWR_LINK_CHECK(hipsolverDnChegvd_bufferSize)
WWR_LINK_CHECK(hipsolverDnZhegvd)
WWR_LINK_CHECK(hipsolverDnZhegvd_bufferSize)
WWR_LINK_CHECK(hipsolverDnSsygvdx)
WWR_LINK_CHECK(hipsolverDnSsygvdx_bufferSize)
WWR_LINK_CHECK(hipsolverDnDsygvdx)
WWR_LINK_CHECK(hipsolverDnDsygvdx_bufferSize)
WWR_LINK_CHECK(hipsolverDnChegvdx)
WWR_LINK_CHECK(hipsolverDnChegvdx_bufferSize)
WWR_LINK_CHECK(hipsolverDnZhegvdx)
WWR_LINK_CHECK(hipsolverDnZhegvdx_bufferSize)
WWR_LINK_CHECK(hipsolverDnSsygvj)
WWR_LINK_CHECK(hipsolverDnSsygvj_bufferSize)
WWR_LINK_CHECK(hipsolverDnDsygvj)
WWR_LINK_CHECK(hipsolverDnDsygvj_bufferSize)
WWR_LINK_CHECK(hipsolverDnChegvj)
WWR_LINK_CHECK(hipsolverDnChegvj_bufferSize)
WWR_LINK_CHECK(hipsolverDnZhegvj)
WWR_LINK_CHECK(hipsolverDnZhegvj_bufferSize)
WWR_LINK_CHECK(hipsolverSpCreate)
WWR_LINK_CHECK(hipsolverSpDestroy)
WWR_LINK_CHECK(hipsolverSpSetStream)
WWR_LINK_CHECK(hipsolverSpScsrlsvchol)
WWR_LINK_CHECK(hipsolverSpDcsrlsvchol)
WWR_LINK_CHECK(hipsolverSpScsrlsvcholHost)
WWR_LINK_CHECK(hipsolverSpDcsrlsvcholHost)
WWR_LINK_CHECK(hipsolverSpScsrlsvqr)
WWR_LINK_CHECK(hipsolverSpDcsrlsvqr)
WWR_LINK_CHECK(hipsolverSpCcsrlsvqr)
WWR_LINK_CHECK(hipsolverSpZcsrlsvqr)
} // namespace wwr::hip::test
