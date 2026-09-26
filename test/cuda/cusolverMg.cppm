// cusolverMg.cppm - Compile-time tests for gpumod.cuda.cusolverMg

module;

#include "test/shared/link_check.h"

export module gpumod.test.cuda.cusolverMg;

import std;
import gpumod.cuda.cusolverMg;

// ========================================================================
// Compile-time tests for gpumod.cuda.cusolverMg
//
// We verify at compile-time that:
//   1. Key enum types exist (std::is_enum_v)
//   2. Key enum enumerator values with stable ABI values are correct
//   3. Opaque handle types are pointers
//   4. GPUMOD_LINK_CHECK for all exported functions
// ========================================================================

namespace gpumod::cuda::test {

using namespace gpumod::cuda;

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
// GPUMOD_LINK_CHECK: handle management
// ────────────────────────────────────────────────────────────────────────

GPUMOD_LINK_CHECK(cusolverMgCreate)
GPUMOD_LINK_CHECK(cusolverMgDestroy)

// ────────────────────────────────────────────────────────────────────────
// GPUMOD_LINK_CHECK: device selection
// ────────────────────────────────────────────────────────────────────────

GPUMOD_LINK_CHECK(cusolverMgDeviceSelect)

// ────────────────────────────────────────────────────────────────────────
// GPUMOD_LINK_CHECK: grid management
// ────────────────────────────────────────────────────────────────────────

GPUMOD_LINK_CHECK(cusolverMgCreateDeviceGrid)
GPUMOD_LINK_CHECK(cusolverMgDestroyGrid)

// ────────────────────────────────────────────────────────────────────────
// GPUMOD_LINK_CHECK: matrix descriptor management
// ────────────────────────────────────────────────────────────────────────

GPUMOD_LINK_CHECK(cusolverMgCreateMatrixDesc)
GPUMOD_LINK_CHECK(cusolverMgDestroyMatrixDesc)

// ────────────────────────────────────────────────────────────────────────
// GPUMOD_LINK_CHECK: symmetric eigenvalue (SYEVD)
// ────────────────────────────────────────────────────────────────────────

GPUMOD_LINK_CHECK(cusolverMgSyevd_bufferSize)
GPUMOD_LINK_CHECK(cusolverMgSyevd)

// ────────────────────────────────────────────────────────────────────────
// GPUMOD_LINK_CHECK: LU factorization (GETRF)
// ────────────────────────────────────────────────────────────────────────

GPUMOD_LINK_CHECK(cusolverMgGetrf_bufferSize)
GPUMOD_LINK_CHECK(cusolverMgGetrf)

// ────────────────────────────────────────────────────────────────────────
// GPUMOD_LINK_CHECK: LU solve (GETRS)
// ────────────────────────────────────────────────────────────────────────

GPUMOD_LINK_CHECK(cusolverMgGetrs_bufferSize)
GPUMOD_LINK_CHECK(cusolverMgGetrs)

// ────────────────────────────────────────────────────────────────────────
// GPUMOD_LINK_CHECK: Cholesky factorization (POTRF)
// ────────────────────────────────────────────────────────────────────────

GPUMOD_LINK_CHECK(cusolverMgPotrf_bufferSize)
GPUMOD_LINK_CHECK(cusolverMgPotrf)

// ────────────────────────────────────────────────────────────────────────
// GPUMOD_LINK_CHECK: Cholesky solve (POTRS)
// ────────────────────────────────────────────────────────────────────────

GPUMOD_LINK_CHECK(cusolverMgPotrs_bufferSize)
GPUMOD_LINK_CHECK(cusolverMgPotrs)

// ────────────────────────────────────────────────────────────────────────
// GPUMOD_LINK_CHECK: Cholesky inverse (POTRI)
// ────────────────────────────────────────────────────────────────────────

GPUMOD_LINK_CHECK(cusolverMgPotri_bufferSize)
GPUMOD_LINK_CHECK(cusolverMgPotri)

#pragma clang diagnostic pop

} // namespace gpumod::cuda::test
