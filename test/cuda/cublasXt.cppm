// cublasXt.cppm - Compile-time tests for gpumod.cuda.cublasXt

module;

#include "test/shared/link_check.h"

export module gpumod.test.cuda.cublasXt;

import std;
import gpumod.cuda.cublasXt;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time tests for gpumod.cuda.cublasXt
//
// We verify at compile-time that:
//   1. Key enum types exist (std::is_enum_v)
//   2. Key enum enumerator values with stable ABI values are correct
//   3. The opaque handle type is a pointer
//   4. Complex scalar types satisfy trivial copyability and standard layout
//   5. GPUMOD_LINK_CHECK for all exported non-inline functions
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace gpumod::cuda::test {

using namespace gpumod::cuda;

// ────────────────────────────────────────────────────────────────────────
// Enum type checks
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_enum_v<cublasStatus_t>);
static_assert(std::is_enum_v<cublasOperation_t>);
static_assert(std::is_enum_v<cublasFillMode_t>);
static_assert(std::is_enum_v<cublasDiagType_t>);
static_assert(std::is_enum_v<cublasSideMode_t>);
static_assert(std::is_enum_v<cublasXtPinnedMemMode_t>);
static_assert(std::is_enum_v<cublasXtOpType_t>);
static_assert(std::is_enum_v<cublasXtBlasOp_t>);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cublasStatus_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUBLAS_STATUS_SUCCESS) == 0);
static_assert(static_cast<int>(CUBLAS_STATUS_NOT_INITIALIZED) == 1);
static_assert(static_cast<int>(CUBLAS_STATUS_ALLOC_FAILED) == 3);
static_assert(static_cast<int>(CUBLAS_STATUS_INVALID_VALUE) == 7);
static_assert(static_cast<int>(CUBLAS_STATUS_ARCH_MISMATCH) == 8);
static_assert(static_cast<int>(CUBLAS_STATUS_MAPPING_ERROR) == 11);
static_assert(static_cast<int>(CUBLAS_STATUS_EXECUTION_FAILED) == 13);
static_assert(static_cast<int>(CUBLAS_STATUS_INTERNAL_ERROR) == 14);
static_assert(static_cast<int>(CUBLAS_STATUS_NOT_SUPPORTED) == 15);
static_assert(static_cast<int>(CUBLAS_STATUS_LICENSE_ERROR) == 16);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cublasOperation_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUBLAS_OP_N) == 0);
static_assert(static_cast<int>(CUBLAS_OP_T) == 1);
static_assert(static_cast<int>(CUBLAS_OP_C) == 2);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cublasFillMode_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUBLAS_FILL_MODE_LOWER) == 0);
static_assert(static_cast<int>(CUBLAS_FILL_MODE_UPPER) == 1);
static_assert(static_cast<int>(CUBLAS_FILL_MODE_FULL) == 2);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cublasDiagType_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUBLAS_DIAG_NON_UNIT) == 0);
static_assert(static_cast<int>(CUBLAS_DIAG_UNIT) == 1);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cublasSideMode_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUBLAS_SIDE_LEFT) == 0);
static_assert(static_cast<int>(CUBLAS_SIDE_RIGHT) == 1);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cublasXtPinnedMemMode_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUBLASXT_PINNING_DISABLED) == 0);
static_assert(static_cast<int>(CUBLASXT_PINNING_ENABLED) == 1);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cublasXtOpType_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUBLASXT_FLOAT) == 0);
static_assert(static_cast<int>(CUBLASXT_DOUBLE) == 1);
static_assert(static_cast<int>(CUBLASXT_COMPLEX) == 2);
static_assert(static_cast<int>(CUBLASXT_DOUBLECOMPLEX) == 3);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cublasXtBlasOp_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUBLASXT_GEMM) == 0);
static_assert(static_cast<int>(CUBLASXT_SYRK) == 1);
static_assert(static_cast<int>(CUBLASXT_HERK) == 2);
static_assert(static_cast<int>(CUBLASXT_SYMM) == 3);
static_assert(static_cast<int>(CUBLASXT_HEMM) == 4);
static_assert(static_cast<int>(CUBLASXT_TRSM) == 5);
static_assert(static_cast<int>(CUBLASXT_SYR2K) == 6);
static_assert(static_cast<int>(CUBLASXT_HER2K) == 7);
static_assert(static_cast<int>(CUBLASXT_SPMM) == 8);
static_assert(static_cast<int>(CUBLASXT_SYRKX) == 9);
static_assert(static_cast<int>(CUBLASXT_HERKX) == 10);
static_assert(static_cast<int>(CUBLASXT_TRMM) == 11);
static_assert(static_cast<int>(CUBLASXT_ROUTINE_MAX) == 12);

// ────────────────────────────────────────────────────────────────────────
// Handle type traits
// cublasXtHandle_t is a pointer to an opaque struct
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_pointer_v<cublasXtHandle_t>);

// ────────────────────────────────────────────────────────────────────────
// Complex scalar type traits
// ────────────────────────────────────────────────────────────────────────

static_assert(sizeof(cuComplex) == 2 * sizeof(float));
static_assert(std::is_trivially_copyable_v<cuComplex>);
static_assert(std::is_standard_layout_v<cuComplex>);

static_assert(sizeof(cuDoubleComplex) == 2 * sizeof(double));
static_assert(std::is_trivially_copyable_v<cuDoubleComplex>);
static_assert(std::is_standard_layout_v<cuDoubleComplex>);

// ────────────────────────────────────────────────────────────────────────
// GPUMOD_LINK_CHECK: context management
// ────────────────────────────────────────────────────────────────────────

GPUMOD_LINK_CHECK(cublasXtCreate)
GPUMOD_LINK_CHECK(cublasXtDestroy)

// ────────────────────────────────────────────────────────────────────────
// GPUMOD_LINK_CHECK: device selection and configuration
// ────────────────────────────────────────────────────────────────────────

GPUMOD_LINK_CHECK(cublasXtGetNumBoards)
GPUMOD_LINK_CHECK(cublasXtMaxBoards)
GPUMOD_LINK_CHECK(cublasXtDeviceSelect)
GPUMOD_LINK_CHECK(cublasXtSetBlockDim)
GPUMOD_LINK_CHECK(cublasXtGetBlockDim)

// ────────────────────────────────────────────────────────────────────────
// GPUMOD_LINK_CHECK: pinned memory mode
// ────────────────────────────────────────────────────────────────────────

GPUMOD_LINK_CHECK(cublasXtGetPinningMemMode)
GPUMOD_LINK_CHECK(cublasXtSetPinningMemMode)

// ────────────────────────────────────────────────────────────────────────
// GPUMOD_LINK_CHECK: CPU BLAS offload
// ────────────────────────────────────────────────────────────────────────

GPUMOD_LINK_CHECK(cublasXtSetCpuRoutine)
GPUMOD_LINK_CHECK(cublasXtSetCpuRatio)

// ────────────────────────────────────────────────────────────────────────
// GPUMOD_LINK_CHECK: GEMM
// ────────────────────────────────────────────────────────────────────────

GPUMOD_LINK_CHECK(cublasXtSgemm)
GPUMOD_LINK_CHECK(cublasXtDgemm)
GPUMOD_LINK_CHECK(cublasXtCgemm)
GPUMOD_LINK_CHECK(cublasXtZgemm)

// ────────────────────────────────────────────────────────────────────────
// GPUMOD_LINK_CHECK: SYRK
// ────────────────────────────────────────────────────────────────────────

GPUMOD_LINK_CHECK(cublasXtSsyrk)
GPUMOD_LINK_CHECK(cublasXtDsyrk)
GPUMOD_LINK_CHECK(cublasXtCsyrk)
GPUMOD_LINK_CHECK(cublasXtZsyrk)

// ────────────────────────────────────────────────────────────────────────
// GPUMOD_LINK_CHECK: HERK
// ────────────────────────────────────────────────────────────────────────

GPUMOD_LINK_CHECK(cublasXtCherk)
GPUMOD_LINK_CHECK(cublasXtZherk)

// ────────────────────────────────────────────────────────────────────────
// GPUMOD_LINK_CHECK: SYR2K
// ────────────────────────────────────────────────────────────────────────

GPUMOD_LINK_CHECK(cublasXtSsyr2k)
GPUMOD_LINK_CHECK(cublasXtDsyr2k)
GPUMOD_LINK_CHECK(cublasXtCsyr2k)
GPUMOD_LINK_CHECK(cublasXtZsyr2k)

// ────────────────────────────────────────────────────────────────────────
// GPUMOD_LINK_CHECK: HERKX
// ────────────────────────────────────────────────────────────────────────

GPUMOD_LINK_CHECK(cublasXtCherkx)
GPUMOD_LINK_CHECK(cublasXtZherkx)

// ────────────────────────────────────────────────────────────────────────
// GPUMOD_LINK_CHECK: TRSM
// ────────────────────────────────────────────────────────────────────────

GPUMOD_LINK_CHECK(cublasXtStrsm)
GPUMOD_LINK_CHECK(cublasXtDtrsm)
GPUMOD_LINK_CHECK(cublasXtCtrsm)
GPUMOD_LINK_CHECK(cublasXtZtrsm)

// ────────────────────────────────────────────────────────────────────────
// GPUMOD_LINK_CHECK: SYMM
// ────────────────────────────────────────────────────────────────────────

GPUMOD_LINK_CHECK(cublasXtSsymm)
GPUMOD_LINK_CHECK(cublasXtDsymm)
GPUMOD_LINK_CHECK(cublasXtCsymm)
GPUMOD_LINK_CHECK(cublasXtZsymm)

// ────────────────────────────────────────────────────────────────────────
// GPUMOD_LINK_CHECK: HEMM
// ────────────────────────────────────────────────────────────────────────

GPUMOD_LINK_CHECK(cublasXtChemm)
GPUMOD_LINK_CHECK(cublasXtZhemm)

// ────────────────────────────────────────────────────────────────────────
// GPUMOD_LINK_CHECK: SYRKX
// ────────────────────────────────────────────────────────────────────────

GPUMOD_LINK_CHECK(cublasXtSsyrkx)
GPUMOD_LINK_CHECK(cublasXtDsyrkx)
GPUMOD_LINK_CHECK(cublasXtCsyrkx)
GPUMOD_LINK_CHECK(cublasXtZsyrkx)

// ────────────────────────────────────────────────────────────────────────
// GPUMOD_LINK_CHECK: HER2K
// ────────────────────────────────────────────────────────────────────────

GPUMOD_LINK_CHECK(cublasXtCher2k)
GPUMOD_LINK_CHECK(cublasXtZher2k)

// ────────────────────────────────────────────────────────────────────────
// GPUMOD_LINK_CHECK: SPMM
// ────────────────────────────────────────────────────────────────────────

GPUMOD_LINK_CHECK(cublasXtSspmm)
GPUMOD_LINK_CHECK(cublasXtDspmm)
GPUMOD_LINK_CHECK(cublasXtCspmm)
GPUMOD_LINK_CHECK(cublasXtZspmm)

// ────────────────────────────────────────────────────────────────────────
// GPUMOD_LINK_CHECK: TRMM
// ────────────────────────────────────────────────────────────────────────

GPUMOD_LINK_CHECK(cublasXtStrmm)
GPUMOD_LINK_CHECK(cublasXtDtrmm)
GPUMOD_LINK_CHECK(cublasXtCtrmm)
GPUMOD_LINK_CHECK(cublasXtZtrmm)

} // namespace gpumod::cuda::test
