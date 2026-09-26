// cusolverSp.cppm - Compile-time tests for gpumod.cuda.cusolverSp

module;

#include "test/shared/link_check.h"

export module gpumod.test.cuda.cusolverSp;

import std;
import gpumod.cuda.cusolverSp;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time tests for gpumod.cuda.cusolverSp
//
// The module is a pure re-export (using declarations).
// We verify at compile-time that:
//   1. Key enum types are actually enum types (std::is_enum_v<>)
//   2. Key enumerator values with cuSOLVER-specified numeric values are correct
//   3. Opaque handle types are pointer types (std::is_pointer_v<>)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace gpumod::cuda::test {

using namespace gpumod::cuda;

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
GPUMOD_LINK_CHECK(cusolverSpCreate)
GPUMOD_LINK_CHECK(cusolverSpDestroy)
GPUMOD_LINK_CHECK(cusolverSpSetStream)
GPUMOD_LINK_CHECK(cusolverSpGetStream)

// Symmetry check
GPUMOD_LINK_CHECK(cusolverSpXcsrissymHost)

// GPU LU linear solvers (Host)
GPUMOD_LINK_CHECK(cusolverSpScsrlsvluHost)
GPUMOD_LINK_CHECK(cusolverSpDcsrlsvluHost)
GPUMOD_LINK_CHECK(cusolverSpCcsrlsvluHost)
GPUMOD_LINK_CHECK(cusolverSpZcsrlsvluHost)

// GPU QR linear solvers (Device)
GPUMOD_LINK_CHECK(cusolverSpScsrlsvqr)
GPUMOD_LINK_CHECK(cusolverSpDcsrlsvqr)
GPUMOD_LINK_CHECK(cusolverSpCcsrlsvqr)
GPUMOD_LINK_CHECK(cusolverSpZcsrlsvqr)

// CPU QR linear solvers (Host)
GPUMOD_LINK_CHECK(cusolverSpScsrlsvqrHost)
GPUMOD_LINK_CHECK(cusolverSpDcsrlsvqrHost)
GPUMOD_LINK_CHECK(cusolverSpCcsrlsvqrHost)
GPUMOD_LINK_CHECK(cusolverSpZcsrlsvqrHost)

// CPU Cholesky linear solvers (Host)
GPUMOD_LINK_CHECK(cusolverSpScsrlsvcholHost)
GPUMOD_LINK_CHECK(cusolverSpDcsrlsvcholHost)
GPUMOD_LINK_CHECK(cusolverSpCcsrlsvcholHost)
GPUMOD_LINK_CHECK(cusolverSpZcsrlsvcholHost)

// GPU Cholesky linear solvers (Device)
GPUMOD_LINK_CHECK(cusolverSpScsrlsvchol)
GPUMOD_LINK_CHECK(cusolverSpDcsrlsvchol)
GPUMOD_LINK_CHECK(cusolverSpCcsrlsvchol)
GPUMOD_LINK_CHECK(cusolverSpZcsrlsvchol)

// CPU least-squares QR solvers (Host)
GPUMOD_LINK_CHECK(cusolverSpScsrlsqvqrHost)
GPUMOD_LINK_CHECK(cusolverSpDcsrlsqvqrHost)
GPUMOD_LINK_CHECK(cusolverSpCcsrlsqvqrHost)
GPUMOD_LINK_CHECK(cusolverSpZcsrlsqvqrHost)

// CPU shift-inverse eigenvalue solvers (Host)
GPUMOD_LINK_CHECK(cusolverSpScsreigvsiHost)
GPUMOD_LINK_CHECK(cusolverSpDcsreigvsiHost)
GPUMOD_LINK_CHECK(cusolverSpCcsreigvsiHost)
GPUMOD_LINK_CHECK(cusolverSpZcsreigvsiHost)

// GPU shift-inverse eigenvalue solvers (Device)
GPUMOD_LINK_CHECK(cusolverSpScsreigvsi)
GPUMOD_LINK_CHECK(cusolverSpDcsreigvsi)
GPUMOD_LINK_CHECK(cusolverSpCcsreigvsi)
GPUMOD_LINK_CHECK(cusolverSpZcsreigvsi)

// CPU enclosed eigenvalue count (Host)
GPUMOD_LINK_CHECK(cusolverSpScsreigsHost)
GPUMOD_LINK_CHECK(cusolverSpDcsreigsHost)
GPUMOD_LINK_CHECK(cusolverSpCcsreigsHost)
GPUMOD_LINK_CHECK(cusolverSpZcsreigsHost)

// CPU reordering
GPUMOD_LINK_CHECK(cusolverSpXcsrsymrcmHost)
GPUMOD_LINK_CHECK(cusolverSpXcsrsymmdqHost)
GPUMOD_LINK_CHECK(cusolverSpXcsrsymamdHost)
GPUMOD_LINK_CHECK(cusolverSpXcsrmetisndHost)

// CPU zero-free diagonal reordering
GPUMOD_LINK_CHECK(cusolverSpScsrzfdHost)
GPUMOD_LINK_CHECK(cusolverSpDcsrzfdHost)
GPUMOD_LINK_CHECK(cusolverSpCcsrzfdHost)
GPUMOD_LINK_CHECK(cusolverSpZcsrzfdHost)

// CPU permutation
GPUMOD_LINK_CHECK(cusolverSpXcsrperm_bufferSizeHost)
GPUMOD_LINK_CHECK(cusolverSpXcsrpermHost)

// Batched QR info management
GPUMOD_LINK_CHECK(cusolverSpCreateCsrqrInfo)
GPUMOD_LINK_CHECK(cusolverSpDestroyCsrqrInfo)

// Batched QR analysis and buffer
GPUMOD_LINK_CHECK(cusolverSpXcsrqrAnalysisBatched)
GPUMOD_LINK_CHECK(cusolverSpScsrqrBufferInfoBatched)
GPUMOD_LINK_CHECK(cusolverSpDcsrqrBufferInfoBatched)
GPUMOD_LINK_CHECK(cusolverSpCcsrqrBufferInfoBatched)
GPUMOD_LINK_CHECK(cusolverSpZcsrqrBufferInfoBatched)

// Batched QR solve
GPUMOD_LINK_CHECK(cusolverSpScsrqrsvBatched)
GPUMOD_LINK_CHECK(cusolverSpDcsrqrsvBatched)
GPUMOD_LINK_CHECK(cusolverSpCcsrqrsvBatched)
GPUMOD_LINK_CHECK(cusolverSpZcsrqrsvBatched)

} // namespace gpumod::cuda::test
