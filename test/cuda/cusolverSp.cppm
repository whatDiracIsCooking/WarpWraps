// cusolverSp.cppm - Compile-time tests for wwr.cuda.cusolverSp

module;

#include "test/shared/link_check.h"

export module wwr.test.cuda.cusolverSp;

import std;
import wwr.cuda.cusolverSp;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time tests for wwr.cuda.cusolverSp
//
// The module is a pure re-export (using declarations).
// We verify at compile-time that:
//   1. Key enum types are actually enum types (std::is_enum_v<>)
//   2. Key enumerator values with cuSOLVER-specified numeric values are correct
//   3. Opaque handle types are pointer types (std::is_pointer_v<>)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace wwr::cuda::test {

using namespace wwr::cuda;

// ────────────────────────────────────────────────────────────────────────
// Enum type checks
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_enum_v<cusolverStatus_t>);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cusolverStatus_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUSOLVER_STATUS_SUCCESS) == 0);
static_assert(static_cast<int>(CUSOLVER_STATUS_NOT_INITIALIZED) == 1);
static_assert(static_cast<int>(CUSOLVER_STATUS_ALLOC_FAILED) == 2);
static_assert(static_cast<int>(CUSOLVER_STATUS_INVALID_VALUE) == 3);
static_assert(static_cast<int>(CUSOLVER_STATUS_ARCH_MISMATCH) == 4);
static_assert(static_cast<int>(CUSOLVER_STATUS_MAPPING_ERROR) == 5);
static_assert(static_cast<int>(CUSOLVER_STATUS_EXECUTION_FAILED) == 6);
static_assert(static_cast<int>(CUSOLVER_STATUS_INTERNAL_ERROR) == 7);
static_assert(static_cast<int>(CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED) == 8);
static_assert(static_cast<int>(CUSOLVER_STATUS_NOT_SUPPORTED) == 9);
static_assert(static_cast<int>(CUSOLVER_STATUS_ZERO_PIVOT) == 10);
static_assert(static_cast<int>(CUSOLVER_STATUS_INVALID_LICENSE) == 11);
static_assert(static_cast<int>(CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED) == 12);
static_assert(static_cast<int>(CUSOLVER_STATUS_IRS_PARAMS_INVALID) == 13);
static_assert(static_cast<int>(CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC) == 14);
static_assert(static_cast<int>(CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE) == 15);
static_assert(static_cast<int>(CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER) == 16);
static_assert(static_cast<int>(CUSOLVER_STATUS_IRS_INTERNAL_ERROR) == 20);
static_assert(static_cast<int>(CUSOLVER_STATUS_IRS_NOT_SUPPORTED) == 21);
static_assert(static_cast<int>(CUSOLVER_STATUS_IRS_OUT_OF_RANGE) == 22);
static_assert(static_cast<int>(CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES) == 23);
static_assert(static_cast<int>(CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED) == 25);
static_assert(static_cast<int>(CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED) == 26);
static_assert(static_cast<int>(CUSOLVER_STATUS_IRS_MATRIX_SINGULAR) == 30);
static_assert(static_cast<int>(CUSOLVER_STATUS_INVALID_WORKSPACE) == 31);

// ────────────────────────────────────────────────────────────────────────
// Opaque handle type checks — all must be pointer types
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_pointer_v<cusolverSpHandle_t>);
static_assert(std::is_pointer_v<csrqrInfo_t>);
static_assert(std::is_pointer_v<cusparseMatDescr_t>);

// ────────────────────────────────────────────────────────────────────────
// Link checks — verify all exported functions resolve at link time
// ────────────────────────────────────────────────────────────────────────

// Handle management
WWR_LINK_CHECK(cusolverSpCreate)
WWR_LINK_CHECK(cusolverSpDestroy)
WWR_LINK_CHECK(cusolverSpSetStream)
WWR_LINK_CHECK(cusolverSpGetStream)

// Symmetry check
WWR_LINK_CHECK(cusolverSpXcsrissymHost)

// GPU LU linear solvers (Host)
WWR_LINK_CHECK(cusolverSpScsrlsvluHost)
WWR_LINK_CHECK(cusolverSpDcsrlsvluHost)
WWR_LINK_CHECK(cusolverSpCcsrlsvluHost)
WWR_LINK_CHECK(cusolverSpZcsrlsvluHost)

// GPU QR linear solvers (Device)
WWR_LINK_CHECK(cusolverSpScsrlsvqr)
WWR_LINK_CHECK(cusolverSpDcsrlsvqr)
WWR_LINK_CHECK(cusolverSpCcsrlsvqr)
WWR_LINK_CHECK(cusolverSpZcsrlsvqr)

// CPU QR linear solvers (Host)
WWR_LINK_CHECK(cusolverSpScsrlsvqrHost)
WWR_LINK_CHECK(cusolverSpDcsrlsvqrHost)
WWR_LINK_CHECK(cusolverSpCcsrlsvqrHost)
WWR_LINK_CHECK(cusolverSpZcsrlsvqrHost)

// CPU Cholesky linear solvers (Host)
WWR_LINK_CHECK(cusolverSpScsrlsvcholHost)
WWR_LINK_CHECK(cusolverSpDcsrlsvcholHost)
WWR_LINK_CHECK(cusolverSpCcsrlsvcholHost)
WWR_LINK_CHECK(cusolverSpZcsrlsvcholHost)

// GPU Cholesky linear solvers (Device)
WWR_LINK_CHECK(cusolverSpScsrlsvchol)
WWR_LINK_CHECK(cusolverSpDcsrlsvchol)
WWR_LINK_CHECK(cusolverSpCcsrlsvchol)
WWR_LINK_CHECK(cusolverSpZcsrlsvchol)

// CPU least-squares QR solvers (Host)
WWR_LINK_CHECK(cusolverSpScsrlsqvqrHost)
WWR_LINK_CHECK(cusolverSpDcsrlsqvqrHost)
WWR_LINK_CHECK(cusolverSpCcsrlsqvqrHost)
WWR_LINK_CHECK(cusolverSpZcsrlsqvqrHost)

// CPU shift-inverse eigenvalue solvers (Host)
WWR_LINK_CHECK(cusolverSpScsreigvsiHost)
WWR_LINK_CHECK(cusolverSpDcsreigvsiHost)
WWR_LINK_CHECK(cusolverSpCcsreigvsiHost)
WWR_LINK_CHECK(cusolverSpZcsreigvsiHost)

// GPU shift-inverse eigenvalue solvers (Device)
WWR_LINK_CHECK(cusolverSpScsreigvsi)
WWR_LINK_CHECK(cusolverSpDcsreigvsi)
WWR_LINK_CHECK(cusolverSpCcsreigvsi)
WWR_LINK_CHECK(cusolverSpZcsreigvsi)

// CPU enclosed eigenvalue count (Host)
WWR_LINK_CHECK(cusolverSpScsreigsHost)
WWR_LINK_CHECK(cusolverSpDcsreigsHost)
WWR_LINK_CHECK(cusolverSpCcsreigsHost)
WWR_LINK_CHECK(cusolverSpZcsreigsHost)

// CPU reordering
WWR_LINK_CHECK(cusolverSpXcsrsymrcmHost)
WWR_LINK_CHECK(cusolverSpXcsrsymmdqHost)
WWR_LINK_CHECK(cusolverSpXcsrsymamdHost)
WWR_LINK_CHECK(cusolverSpXcsrmetisndHost)

// CPU zero-free diagonal reordering
WWR_LINK_CHECK(cusolverSpScsrzfdHost)
WWR_LINK_CHECK(cusolverSpDcsrzfdHost)
WWR_LINK_CHECK(cusolverSpCcsrzfdHost)
WWR_LINK_CHECK(cusolverSpZcsrzfdHost)

// CPU permutation
WWR_LINK_CHECK(cusolverSpXcsrperm_bufferSizeHost)
WWR_LINK_CHECK(cusolverSpXcsrpermHost)

// Batched QR info management
WWR_LINK_CHECK(cusolverSpCreateCsrqrInfo)
WWR_LINK_CHECK(cusolverSpDestroyCsrqrInfo)

// Batched QR analysis and buffer
WWR_LINK_CHECK(cusolverSpXcsrqrAnalysisBatched)
WWR_LINK_CHECK(cusolverSpScsrqrBufferInfoBatched)
WWR_LINK_CHECK(cusolverSpDcsrqrBufferInfoBatched)
WWR_LINK_CHECK(cusolverSpCcsrqrBufferInfoBatched)
WWR_LINK_CHECK(cusolverSpZcsrqrBufferInfoBatched)

// Batched QR solve
WWR_LINK_CHECK(cusolverSpScsrqrsvBatched)
WWR_LINK_CHECK(cusolverSpDcsrqrsvBatched)
WWR_LINK_CHECK(cusolverSpCcsrqrsvBatched)
WWR_LINK_CHECK(cusolverSpZcsrqrsvBatched)

} // namespace wwr::cuda::test
