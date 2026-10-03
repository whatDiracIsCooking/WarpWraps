// solver.cppm - Compile-time tests for wwr.solver
//
// Types, constants, and the handle/params/Jacobi-info/stream functions are
// each checked against the backend's own entity (see gpu_check_macros.h).
//
// The 200+ typed legacy entry points (wwrsolverDnSpotrf, ...) are not
// repeated here, for the same reason test/gpu/blas.cppm skips its 374: they
// are written out in full in src/detail/solver_names.h, not derived by a
// prefix-pasting macro, so a line here would restate that line. A misspelled
// backend name does not compile, and a wrong-but-existing one is a signature
// mismatch at the instantiation in src/wrappers/solver/instantiations.cpp,
// or a byte mismatch in test/wrappers/solver. A representative sample is
// checked below: one legacy linear function, one legacy eigen function, and
// the 8 shared modern (X-prefixed) functions in full, since that surface is
// short enough to list completely.
//
// wwrsolverDataType_t additionally gets a cross-backend value check: the
// hipsolver README states hipDataType's enumerators match cudaDataType's
// one-for-one, so this checks the actual numeric literal from both headers
// rather than assuming it.

module;

#include "gpu_check_macros.h"

#if defined(WWR_GPU_BACKEND_CUDA)
#include <library_types.h>
#else
#include <hip/library_types.h>
#endif

export module wwr.test.gpu.solver;

import std;
import wwr.solver;
#if defined(WWR_GPU_BACKEND_CUDA)
import wwr.cuda.cusolverDn;
#else
import wwr.hip.hipsolver;
#endif

namespace wwr::test {

using namespace wwr;

#if defined(WWR_GPU_BACKEND_CUDA)

using namespace wwr::cuda;

// ────────────────────────────────────────────────────────────────────────
// CUDA backend
// ────────────────────────────────────────────────────────────────────────

WWR_SAME_TYPE(wwrsolverDnHandle_t, cusolverDnHandle_t)
WWR_SAME_TYPE(wwrsolverDnParams_t, cusolverDnParams_t)
WWR_SAME_TYPE(wwrsolverStatus_t, cusolverStatus_t)
WWR_SAME_TYPE(wwrsolverEigMode_t, cusolverEigMode_t)
WWR_SAME_TYPE(wwrsolverEigType_t, cusolverEigType_t)
WWR_SAME_TYPE(wwrsolverEigRange_t, cusolverEigRange_t)
WWR_SAME_TYPE(wwrsolverSyevjInfo_t, syevjInfo_t)
WWR_SAME_TYPE(wwrsolverGesvdjInfo_t, gesvdjInfo_t)
WWR_SAME_TYPE(wwrsolverDataType_t, cudaDataType)

WWR_SAME_VALUE(WWRSOLVER_STATUS_SUCCESS, CUSOLVER_STATUS_SUCCESS)
WWR_SAME_VALUE(WWRSOLVER_EIG_MODE_NOVECTOR, CUSOLVER_EIG_MODE_NOVECTOR)
WWR_SAME_VALUE(WWRSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_MODE_VECTOR)
WWR_SAME_VALUE(WWRSOLVER_EIG_RANGE_ALL, CUSOLVER_EIG_RANGE_ALL)
WWR_SAME_VALUE(WWRSOLVER_EIG_RANGE_V, CUSOLVER_EIG_RANGE_V)
WWR_SAME_VALUE(WWRSOLVER_EIG_RANGE_I, CUSOLVER_EIG_RANGE_I)
WWR_SAME_VALUE(WWRSOLVER_EIG_TYPE_1, CUSOLVER_EIG_TYPE_1)
WWR_SAME_VALUE(WWRSOLVER_EIG_TYPE_2, CUSOLVER_EIG_TYPE_2)
WWR_SAME_VALUE(WWRSOLVER_EIG_TYPE_3, CUSOLVER_EIG_TYPE_3)

static_assert(WWRSOLVER_R_32F == CUDA_R_32F, "WWRSOLVER_R_32F != CUDA_R_32F");
static_assert(WWRSOLVER_R_64F == CUDA_R_64F, "WWRSOLVER_R_64F != CUDA_R_64F");
static_assert(WWRSOLVER_C_32F == CUDA_C_32F, "WWRSOLVER_C_32F != CUDA_C_32F");
static_assert(WWRSOLVER_C_64F == CUDA_C_64F, "WWRSOLVER_C_64F != CUDA_C_64F");

// hipsolver-dense.h's own numeric literals (0/1/4/5, from
// /opt/rocm/include/hip/library_types.h) -- checked directly rather than
// assumed, per hipsolver/README.md's claim that the two enums match
// one-for-one.
static_assert(
    CUDA_R_32F == 0,
    "cudaDataType's numeric value moved -- re-check against hip/library_types.h's HIP_R_32F");
static_assert(
    CUDA_R_64F == 1,
    "cudaDataType's numeric value moved -- re-check against hip/library_types.h's HIP_R_64F");
static_assert(
    CUDA_C_32F == 4,
    "cudaDataType's numeric value moved -- re-check against hip/library_types.h's HIP_C_32F");
static_assert(
    CUDA_C_64F == 5,
    "cudaDataType's numeric value moved -- re-check against hip/library_types.h's HIP_C_64F");

WWR_SAME_FUNCTION(wwrsolverDnCreate, cusolverDnCreate)
WWR_SAME_FUNCTION(wwrsolverDnDestroy, cusolverDnDestroy)
WWR_SAME_FUNCTION(wwrsolverDnSetStream, cusolverDnSetStream)
WWR_SAME_FUNCTION(wwrsolverDnGetStream, cusolverDnGetStream)
WWR_SAME_FUNCTION(wwrsolverDnCreateParams, cusolverDnCreateParams)
WWR_SAME_FUNCTION(wwrsolverDnDestroyParams, cusolverDnDestroyParams)
WWR_SAME_FUNCTION(wwrsolverDnCreateSyevjInfo, cusolverDnCreateSyevjInfo)
WWR_SAME_FUNCTION(wwrsolverDnDestroySyevjInfo, cusolverDnDestroySyevjInfo)
WWR_SAME_FUNCTION(wwrsolverDnCreateGesvdjInfo, cusolverDnCreateGesvdjInfo)
WWR_SAME_FUNCTION(wwrsolverDnDestroyGesvdjInfo, cusolverDnDestroyGesvdjInfo)

// One legacy linear, one legacy eigen entry point.
WWR_SAME_FUNCTION(wwrsolverDnSgetrf, cusolverDnSgetrf)
WWR_SAME_FUNCTION(wwrsolverDnSsyevd, cusolverDnSsyevd)

// The 8 modern (X-prefixed) functions shared with hipsolverDn, in full.
WWR_SAME_FUNCTION(wwrsolverDnXpotrf_bufferSize, cusolverDnXpotrf_bufferSize)
WWR_SAME_FUNCTION(wwrsolverDnXpotrf, cusolverDnXpotrf)
WWR_SAME_FUNCTION(wwrsolverDnXpotrs, cusolverDnXpotrs)
WWR_SAME_FUNCTION(wwrsolverDnXgetrf_bufferSize, cusolverDnXgetrf_bufferSize)
WWR_SAME_FUNCTION(wwrsolverDnXgetrf, cusolverDnXgetrf)
WWR_SAME_FUNCTION(wwrsolverDnXgetrs, cusolverDnXgetrs)
WWR_SAME_FUNCTION(wwrsolverDnXgeqrf_bufferSize, cusolverDnXgeqrf_bufferSize)
WWR_SAME_FUNCTION(wwrsolverDnXgeqrf, cusolverDnXgeqrf)

#else

// ────────────────────────────────────────────────────────────────────────
// HIP backend
// ────────────────────────────────────────────────────────────────────────

using namespace wwr::hip;

WWR_SAME_TYPE(wwrsolverDnHandle_t, hipsolverDnHandle_t)
WWR_SAME_TYPE(wwrsolverDnParams_t, hipsolverDnParams_t)
WWR_SAME_TYPE(wwrsolverStatus_t, hipsolverStatus_t)
WWR_SAME_TYPE(wwrsolverEigMode_t, hipsolverEigMode_t)
WWR_SAME_TYPE(wwrsolverEigType_t, hipsolverEigType_t)
WWR_SAME_TYPE(wwrsolverEigRange_t, hipsolverEigRange_t)
WWR_SAME_TYPE(wwrsolverSyevjInfo_t, hipsolverSyevjInfo_t)
WWR_SAME_TYPE(wwrsolverGesvdjInfo_t, hipsolverGesvdjInfo_t)
WWR_SAME_TYPE(wwrsolverDataType_t, hipDataType)

WWR_SAME_VALUE(WWRSOLVER_STATUS_SUCCESS, HIPSOLVER_STATUS_SUCCESS)
WWR_SAME_VALUE(WWRSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_EIG_MODE_NOVECTOR)
WWR_SAME_VALUE(WWRSOLVER_EIG_MODE_VECTOR, HIPSOLVER_EIG_MODE_VECTOR)
WWR_SAME_VALUE(WWRSOLVER_EIG_RANGE_ALL, HIPSOLVER_EIG_RANGE_ALL)
WWR_SAME_VALUE(WWRSOLVER_EIG_RANGE_V, HIPSOLVER_EIG_RANGE_V)
WWR_SAME_VALUE(WWRSOLVER_EIG_RANGE_I, HIPSOLVER_EIG_RANGE_I)
WWR_SAME_VALUE(WWRSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_TYPE_1)
WWR_SAME_VALUE(WWRSOLVER_EIG_TYPE_2, HIPSOLVER_EIG_TYPE_2)
WWR_SAME_VALUE(WWRSOLVER_EIG_TYPE_3, HIPSOLVER_EIG_TYPE_3)

static_assert(WWRSOLVER_R_32F == HIP_R_32F, "WWRSOLVER_R_32F != HIP_R_32F");
static_assert(WWRSOLVER_R_64F == HIP_R_64F, "WWRSOLVER_R_64F != HIP_R_64F");
static_assert(WWRSOLVER_C_32F == HIP_C_32F, "WWRSOLVER_C_32F != HIP_C_32F");
static_assert(WWRSOLVER_C_64F == HIP_C_64F, "WWRSOLVER_C_64F != HIP_C_64F");

// hip/library_types.h's own numeric literals (0/1/4/5) -- checked directly
// rather than assumed, per hipsolver/README.md's claim that the two enums
// match cudaDataType's one-for-one (see /usr/local/cuda/include/library_types.h).
static_assert(
    HIP_R_32F == 0,
    "hipDataType's numeric value moved -- re-check against cuda's library_types.h's CUDA_R_32F");
static_assert(
    HIP_R_64F == 1,
    "hipDataType's numeric value moved -- re-check against cuda's library_types.h's CUDA_R_64F");
static_assert(
    HIP_C_32F == 4,
    "hipDataType's numeric value moved -- re-check against cuda's library_types.h's CUDA_C_32F");
static_assert(
    HIP_C_64F == 5,
    "hipDataType's numeric value moved -- re-check against cuda's library_types.h's CUDA_C_64F");

WWR_SAME_FUNCTION(wwrsolverDnCreate, hipsolverDnCreate)
WWR_SAME_FUNCTION(wwrsolverDnDestroy, hipsolverDnDestroy)
WWR_SAME_FUNCTION(wwrsolverDnSetStream, hipsolverDnSetStream)
WWR_SAME_FUNCTION(wwrsolverDnGetStream, hipsolverDnGetStream)
WWR_SAME_FUNCTION(wwrsolverDnCreateParams, hipsolverDnCreateParams)
WWR_SAME_FUNCTION(wwrsolverDnDestroyParams, hipsolverDnDestroyParams)
WWR_SAME_FUNCTION(wwrsolverDnCreateSyevjInfo, hipsolverDnCreateSyevjInfo)
WWR_SAME_FUNCTION(wwrsolverDnDestroySyevjInfo, hipsolverDnDestroySyevjInfo)
WWR_SAME_FUNCTION(wwrsolverDnCreateGesvdjInfo, hipsolverDnCreateGesvdjInfo)
WWR_SAME_FUNCTION(wwrsolverDnDestroyGesvdjInfo, hipsolverDnDestroyGesvdjInfo)

// One legacy linear, one legacy eigen entry point.
WWR_SAME_FUNCTION(wwrsolverDnSgetrf, hipsolverDnSgetrf)
WWR_SAME_FUNCTION(wwrsolverDnSsyevd, hipsolverDnSsyevd)

// The 8 modern (X-prefixed) functions shared with cusolverDn, in full.
WWR_SAME_FUNCTION(wwrsolverDnXpotrf_bufferSize, hipsolverDnXpotrf_bufferSize)
WWR_SAME_FUNCTION(wwrsolverDnXpotrf, hipsolverDnXpotrf)
WWR_SAME_FUNCTION(wwrsolverDnXpotrs, hipsolverDnXpotrs)
WWR_SAME_FUNCTION(wwrsolverDnXgetrf_bufferSize, hipsolverDnXgetrf_bufferSize)
WWR_SAME_FUNCTION(wwrsolverDnXgetrf, hipsolverDnXgetrf)
WWR_SAME_FUNCTION(wwrsolverDnXgetrs, hipsolverDnXgetrs)
WWR_SAME_FUNCTION(wwrsolverDnXgeqrf_bufferSize, hipsolverDnXgeqrf_bufferSize)
WWR_SAME_FUNCTION(wwrsolverDnXgeqrf, hipsolverDnXgeqrf)

#endif

} // namespace wwr::test
