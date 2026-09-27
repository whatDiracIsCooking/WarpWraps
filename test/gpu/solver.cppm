// solver.cppm - Compile-time tests for wwr.solver
//
// Types, constants, and the handle/params/Jacobi-info/stream functions are
// each checked against the backend's own entity (see gpu_check_macros.h).
//
// The 200+ typed legacy entry points (gpusolverDnSpotrf, ...) are not
// repeated here, for the same reason test/gpu/blas.cppm skips its 374: they
// are written out in full in src/solver.cppm, not derived by a
// prefix-pasting macro, so a line here would restate that line. A misspelled
// backend name does not compile, and a wrong-but-existing one is a signature
// mismatch at the instantiation in src/wrappers/solver/instantiations.cpp,
// or a byte mismatch in test/wrappers/solver. A representative sample is
// checked below: one legacy linear function, one legacy eigen function, and
// the 8 shared modern (X-prefixed) functions in full, since that surface is
// short enough to list completely.
//
// gpusolverDataType_t additionally gets a cross-backend value check: the
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
import wwr.blas;
import wwr.complex;
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

WWR_SAME_TYPE(gpusolverDnHandle_t, cusolverDnHandle_t)
WWR_SAME_TYPE(gpusolverDnParams_t, cusolverDnParams_t)
WWR_SAME_TYPE(gpusolverStatus_t, cusolverStatus_t)
WWR_SAME_TYPE(gpusolverEigMode_t, cusolverEigMode_t)
WWR_SAME_TYPE(gpusolverEigType_t, cusolverEigType_t)
WWR_SAME_TYPE(gpusolverEigRange_t, cusolverEigRange_t)
WWR_SAME_TYPE(gpusolverSyevjInfo_t, syevjInfo_t)
WWR_SAME_TYPE(gpusolverGesvdjInfo_t, gesvdjInfo_t)
WWR_SAME_TYPE(gpusolverDataType_t, cudaDataType)

WWR_SAME_VALUE(GPUSOLVER_STATUS_SUCCESS, CUSOLVER_STATUS_SUCCESS)
WWR_SAME_VALUE(GPUSOLVER_EIG_MODE_NOVECTOR, CUSOLVER_EIG_MODE_NOVECTOR)
WWR_SAME_VALUE(GPUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_MODE_VECTOR)
WWR_SAME_VALUE(GPUSOLVER_EIG_RANGE_ALL, CUSOLVER_EIG_RANGE_ALL)
WWR_SAME_VALUE(GPUSOLVER_EIG_RANGE_V, CUSOLVER_EIG_RANGE_V)
WWR_SAME_VALUE(GPUSOLVER_EIG_RANGE_I, CUSOLVER_EIG_RANGE_I)
WWR_SAME_VALUE(GPUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_TYPE_1)
WWR_SAME_VALUE(GPUSOLVER_EIG_TYPE_2, CUSOLVER_EIG_TYPE_2)
WWR_SAME_VALUE(GPUSOLVER_EIG_TYPE_3, CUSOLVER_EIG_TYPE_3)

static_assert(GPUSOLVER_R_32F == CUDA_R_32F, "GPUSOLVER_R_32F != CUDA_R_32F");
static_assert(GPUSOLVER_R_64F == CUDA_R_64F, "GPUSOLVER_R_64F != CUDA_R_64F");
static_assert(GPUSOLVER_C_32F == CUDA_C_32F, "GPUSOLVER_C_32F != CUDA_C_32F");
static_assert(GPUSOLVER_C_64F == CUDA_C_64F, "GPUSOLVER_C_64F != CUDA_C_64F");

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

WWR_SAME_FUNCTION(gpusolverDnCreate, cusolverDnCreate)
WWR_SAME_FUNCTION(gpusolverDnDestroy, cusolverDnDestroy)
WWR_SAME_FUNCTION(gpusolverDnSetStream, cusolverDnSetStream)
WWR_SAME_FUNCTION(gpusolverDnGetStream, cusolverDnGetStream)
WWR_SAME_FUNCTION(gpusolverDnCreateParams, cusolverDnCreateParams)
WWR_SAME_FUNCTION(gpusolverDnDestroyParams, cusolverDnDestroyParams)
WWR_SAME_FUNCTION(gpusolverDnCreateSyevjInfo, cusolverDnCreateSyevjInfo)
WWR_SAME_FUNCTION(gpusolverDnDestroySyevjInfo, cusolverDnDestroySyevjInfo)
WWR_SAME_FUNCTION(gpusolverDnCreateGesvdjInfo, cusolverDnCreateGesvdjInfo)
WWR_SAME_FUNCTION(gpusolverDnDestroyGesvdjInfo, cusolverDnDestroyGesvdjInfo)

// One legacy linear, one legacy eigen entry point.
WWR_SAME_FUNCTION(gpusolverDnSgetrf, cusolverDnSgetrf)
WWR_SAME_FUNCTION(gpusolverDnSsyevd, cusolverDnSsyevd)

// The 8 modern (X-prefixed) functions shared with hipsolverDn, in full.
WWR_SAME_FUNCTION(gpusolverDnXpotrf_bufferSize, cusolverDnXpotrf_bufferSize)
WWR_SAME_FUNCTION(gpusolverDnXpotrf, cusolverDnXpotrf)
WWR_SAME_FUNCTION(gpusolverDnXpotrs, cusolverDnXpotrs)
WWR_SAME_FUNCTION(gpusolverDnXgetrf_bufferSize, cusolverDnXgetrf_bufferSize)
WWR_SAME_FUNCTION(gpusolverDnXgetrf, cusolverDnXgetrf)
WWR_SAME_FUNCTION(gpusolverDnXgetrs, cusolverDnXgetrs)
WWR_SAME_FUNCTION(gpusolverDnXgeqrf_bufferSize, cusolverDnXgeqrf_bufferSize)
WWR_SAME_FUNCTION(gpusolverDnXgeqrf, cusolverDnXgeqrf)

#else

// ────────────────────────────────────────────────────────────────────────
// HIP backend
// ────────────────────────────────────────────────────────────────────────

using namespace wwr::hip;

WWR_SAME_TYPE(gpusolverDnHandle_t, hipsolverDnHandle_t)
WWR_SAME_TYPE(gpusolverDnParams_t, hipsolverDnParams_t)
WWR_SAME_TYPE(gpusolverStatus_t, hipsolverStatus_t)
WWR_SAME_TYPE(gpusolverEigMode_t, hipsolverEigMode_t)
WWR_SAME_TYPE(gpusolverEigType_t, hipsolverEigType_t)
WWR_SAME_TYPE(gpusolverEigRange_t, hipsolverEigRange_t)
WWR_SAME_TYPE(gpusolverSyevjInfo_t, hipsolverSyevjInfo_t)
WWR_SAME_TYPE(gpusolverGesvdjInfo_t, hipsolverGesvdjInfo_t)
WWR_SAME_TYPE(gpusolverDataType_t, hipDataType)

WWR_SAME_VALUE(GPUSOLVER_STATUS_SUCCESS, HIPSOLVER_STATUS_SUCCESS)
WWR_SAME_VALUE(GPUSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_EIG_MODE_NOVECTOR)
WWR_SAME_VALUE(GPUSOLVER_EIG_MODE_VECTOR, HIPSOLVER_EIG_MODE_VECTOR)
WWR_SAME_VALUE(GPUSOLVER_EIG_RANGE_ALL, HIPSOLVER_EIG_RANGE_ALL)
WWR_SAME_VALUE(GPUSOLVER_EIG_RANGE_V, HIPSOLVER_EIG_RANGE_V)
WWR_SAME_VALUE(GPUSOLVER_EIG_RANGE_I, HIPSOLVER_EIG_RANGE_I)
WWR_SAME_VALUE(GPUSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_TYPE_1)
WWR_SAME_VALUE(GPUSOLVER_EIG_TYPE_2, HIPSOLVER_EIG_TYPE_2)
WWR_SAME_VALUE(GPUSOLVER_EIG_TYPE_3, HIPSOLVER_EIG_TYPE_3)

static_assert(GPUSOLVER_R_32F == HIP_R_32F, "GPUSOLVER_R_32F != HIP_R_32F");
static_assert(GPUSOLVER_R_64F == HIP_R_64F, "GPUSOLVER_R_64F != HIP_R_64F");
static_assert(GPUSOLVER_C_32F == HIP_C_32F, "GPUSOLVER_C_32F != HIP_C_32F");
static_assert(GPUSOLVER_C_64F == HIP_C_64F, "GPUSOLVER_C_64F != HIP_C_64F");

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

WWR_SAME_FUNCTION(gpusolverDnCreate, hipsolverDnCreate)
WWR_SAME_FUNCTION(gpusolverDnDestroy, hipsolverDnDestroy)
WWR_SAME_FUNCTION(gpusolverDnSetStream, hipsolverDnSetStream)
WWR_SAME_FUNCTION(gpusolverDnGetStream, hipsolverDnGetStream)
WWR_SAME_FUNCTION(gpusolverDnCreateParams, hipsolverDnCreateParams)
WWR_SAME_FUNCTION(gpusolverDnDestroyParams, hipsolverDnDestroyParams)
WWR_SAME_FUNCTION(gpusolverDnCreateSyevjInfo, hipsolverDnCreateSyevjInfo)
WWR_SAME_FUNCTION(gpusolverDnDestroySyevjInfo, hipsolverDnDestroySyevjInfo)
WWR_SAME_FUNCTION(gpusolverDnCreateGesvdjInfo, hipsolverDnCreateGesvdjInfo)
WWR_SAME_FUNCTION(gpusolverDnDestroyGesvdjInfo, hipsolverDnDestroyGesvdjInfo)

// One legacy linear, one legacy eigen entry point.
WWR_SAME_FUNCTION(gpusolverDnSgetrf, hipsolverDnSgetrf)
WWR_SAME_FUNCTION(gpusolverDnSsyevd, hipsolverDnSsyevd)

// The 8 modern (X-prefixed) functions shared with cusolverDn, in full.
WWR_SAME_FUNCTION(gpusolverDnXpotrf_bufferSize, hipsolverDnXpotrf_bufferSize)
WWR_SAME_FUNCTION(gpusolverDnXpotrf, hipsolverDnXpotrf)
WWR_SAME_FUNCTION(gpusolverDnXpotrs, hipsolverDnXpotrs)
WWR_SAME_FUNCTION(gpusolverDnXgetrf_bufferSize, hipsolverDnXgetrf_bufferSize)
WWR_SAME_FUNCTION(gpusolverDnXgetrf, hipsolverDnXgetrf)
WWR_SAME_FUNCTION(gpusolverDnXgetrs, hipsolverDnXgetrs)
WWR_SAME_FUNCTION(gpusolverDnXgeqrf_bufferSize, hipsolverDnXgeqrf_bufferSize)
WWR_SAME_FUNCTION(gpusolverDnXgeqrf, hipsolverDnXgeqrf)

#endif

} // namespace wwr::test
