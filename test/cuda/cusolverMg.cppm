// cusolverMg.cppm - Compile-time tests for wwr.cuda.cusolverMg

module;

#include "test/shared/link_check.h"

export module wwr.test.cuda.cusolverMg;

import std;
import wwr.cuda.cusolverMg;

// ========================================================================
// Compile-time tests for wwr.cuda.cusolverMg
//
// We verify at compile-time that:
//   1. Key enum types exist (std::is_enum_v)
//   2. Key enum enumerator values with stable ABI values are correct
//   3. Opaque handle types are pointers
//   4. WWR_LINK_CHECK for all exported functions
// ========================================================================

namespace wwr::cuda::test {

using namespace wwr::cuda;

// ────────────────────────────────────────────────────────────────────────
// Enum type checks
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_enum_v<cusolverMgGridMapping_t>);
static_assert(std::is_enum_v<cusolverEigMode_t>);
static_assert(std::is_enum_v<cusolverStatus_t>);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cusolverMgGridMapping_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUDALIBMG_GRID_MAPPING_COL_MAJOR) == 0);
static_assert(static_cast<int>(CUDALIBMG_GRID_MAPPING_ROW_MAJOR) == 1);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cusolverEigMode_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUSOLVER_EIG_MODE_NOVECTOR) == 0);
static_assert(static_cast<int>(CUSOLVER_EIG_MODE_VECTOR) == 1);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cusolverStatus_t (stable ABI values)
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

// ────────────────────────────────────────────────────────────────────────
// Handle type traits (opaque pointer types)
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_pointer_v<cusolverMgHandle_t>);
static_assert(std::is_pointer_v<cudaLibMgGrid_t>);
static_assert(std::is_pointer_v<cudaLibMgMatrixDesc_t>);

// CUDA 13 deprecated the whole cusolverMg API (every function below carries
// CUSOLVERMG_DEPRECATED). These checks only prove the symbols still link, not
// that anyone should call them, so silence the deprecation here -- the module's
// re-exports keep the [[deprecated]] attribute, so real consumers are still
// warned.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

// ────────────────────────────────────────────────────────────────────────
// WWR_LINK_CHECK: handle management
// ────────────────────────────────────────────────────────────────────────

WWR_LINK_CHECK(cusolverMgCreate)
WWR_LINK_CHECK(cusolverMgDestroy)

// ────────────────────────────────────────────────────────────────────────
// WWR_LINK_CHECK: device selection
// ────────────────────────────────────────────────────────────────────────

WWR_LINK_CHECK(cusolverMgDeviceSelect)

// ────────────────────────────────────────────────────────────────────────
// WWR_LINK_CHECK: grid management
// ────────────────────────────────────────────────────────────────────────

WWR_LINK_CHECK(cusolverMgCreateDeviceGrid)
WWR_LINK_CHECK(cusolverMgDestroyGrid)

// ────────────────────────────────────────────────────────────────────────
// WWR_LINK_CHECK: matrix descriptor management
// ────────────────────────────────────────────────────────────────────────

WWR_LINK_CHECK(cusolverMgCreateMatrixDesc)
WWR_LINK_CHECK(cusolverMgDestroyMatrixDesc)

// ────────────────────────────────────────────────────────────────────────
// WWR_LINK_CHECK: symmetric eigenvalue (SYEVD)
// ────────────────────────────────────────────────────────────────────────

WWR_LINK_CHECK(cusolverMgSyevd_bufferSize)
WWR_LINK_CHECK(cusolverMgSyevd)

// ────────────────────────────────────────────────────────────────────────
// WWR_LINK_CHECK: LU factorization (GETRF)
// ────────────────────────────────────────────────────────────────────────

WWR_LINK_CHECK(cusolverMgGetrf_bufferSize)
WWR_LINK_CHECK(cusolverMgGetrf)

// ────────────────────────────────────────────────────────────────────────
// WWR_LINK_CHECK: LU solve (GETRS)
// ────────────────────────────────────────────────────────────────────────

WWR_LINK_CHECK(cusolverMgGetrs_bufferSize)
WWR_LINK_CHECK(cusolverMgGetrs)

// ────────────────────────────────────────────────────────────────────────
// WWR_LINK_CHECK: Cholesky factorization (POTRF)
// ────────────────────────────────────────────────────────────────────────

WWR_LINK_CHECK(cusolverMgPotrf_bufferSize)
WWR_LINK_CHECK(cusolverMgPotrf)

// ────────────────────────────────────────────────────────────────────────
// WWR_LINK_CHECK: Cholesky solve (POTRS)
// ────────────────────────────────────────────────────────────────────────

WWR_LINK_CHECK(cusolverMgPotrs_bufferSize)
WWR_LINK_CHECK(cusolverMgPotrs)

// ────────────────────────────────────────────────────────────────────────
// WWR_LINK_CHECK: Cholesky inverse (POTRI)
// ────────────────────────────────────────────────────────────────────────

WWR_LINK_CHECK(cusolverMgPotri_bufferSize)
WWR_LINK_CHECK(cusolverMgPotri)

#pragma clang diagnostic pop

} // namespace wwr::cuda::test
