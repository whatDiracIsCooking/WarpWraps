// blas.cppm - Compile-time tests for gpumod.blas
//
// Types, constants, and the handle/stream/status functions are each checked
// against the backend's own entity (see gpu_check_macros.h). The two places
// where src/blas.cppm does more than rename -- the status strings on HIP
// and the const-correct getrsBatched/getriBatched -- are checked too.
//
// The typed BLAS entry points (gpublasSgemm, ...) are not repeated here:
// blas.cppm writes each one out in full (GPUMOD_FUNCTION(gpublasSgemm,
// cublasSgemm_v2, hipblasSgemm)), not derived by a prefix-pasting macro, so a
// line here would restate that line. A misspelled backend name does not
// compile (gpumod.cuda.cublas_v2 exports only the _v2 spellings), and a
// wrong-but-existing one is a signature mismatch at the instantiation in
// src/wrappers/blas/instantiations.cpp or a dispatch mismatch in the blas
// dispatch check (test/wrappers/blas/blas_dispatch.toml). That every alias is
// covered by one of those, or by a GPUMOD_SAME_FUNCTION above, is enforced by
// test/shared/alias_coverage.py.

module;

#include "gpu_check_macros.h"

export module gpumod.test.gpu.blas;

import std;
import gpumod.blas;
import gpumod.complex;
#if defined(GPUMOD_GPU_BACKEND_CUDA)
import gpumod.cuda.cublas_v2;
#else
import gpumod.hip.hipblas;
#endif

namespace gpumod::test {

using namespace gpumod;

#if defined(GPUMOD_GPU_BACKEND_CUDA)

using namespace gpumod::cuda;

// ────────────────────────────────────────────────────────────────────────
// CUDA backend
// ────────────────────────────────────────────────────────────────────────

GPUMOD_SAME_TYPE(gpublasHandle_t, cublasHandle_t)
GPUMOD_SAME_TYPE(gpublasStatus_t, cublasStatus_t)
GPUMOD_SAME_TYPE(gpublasOperation_t, cublasOperation_t)
GPUMOD_SAME_TYPE(gpublasFillMode_t, cublasFillMode_t)
GPUMOD_SAME_TYPE(gpublasDiagType_t, cublasDiagType_t)
GPUMOD_SAME_TYPE(gpublasSideMode_t, cublasSideMode_t)
GPUMOD_SAME_TYPE(gpublasPointerMode_t, cublasPointerMode_t)

GPUMOD_SAME_VALUE(GPUBLAS_STATUS_SUCCESS, CUBLAS_STATUS_SUCCESS)
GPUMOD_SAME_VALUE(GPUBLAS_STATUS_NOT_INITIALIZED, CUBLAS_STATUS_NOT_INITIALIZED)
GPUMOD_SAME_VALUE(GPUBLAS_OP_N, CUBLAS_OP_N)
GPUMOD_SAME_VALUE(GPUBLAS_OP_T, CUBLAS_OP_T)
GPUMOD_SAME_VALUE(GPUBLAS_OP_C, CUBLAS_OP_C)
GPUMOD_SAME_VALUE(GPUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_LOWER)
GPUMOD_SAME_VALUE(GPUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_UPPER)
GPUMOD_SAME_VALUE(GPUBLAS_DIAG_NON_UNIT, CUBLAS_DIAG_NON_UNIT)
GPUMOD_SAME_VALUE(GPUBLAS_DIAG_UNIT, CUBLAS_DIAG_UNIT)
GPUMOD_SAME_VALUE(GPUBLAS_SIDE_LEFT, CUBLAS_SIDE_LEFT)
GPUMOD_SAME_VALUE(GPUBLAS_SIDE_RIGHT, CUBLAS_SIDE_RIGHT)
GPUMOD_SAME_VALUE(GPUBLAS_POINTER_MODE_HOST, CUBLAS_POINTER_MODE_HOST)
GPUMOD_SAME_VALUE(GPUBLAS_POINTER_MODE_DEVICE, CUBLAS_POINTER_MODE_DEVICE)

GPUMOD_SAME_FUNCTION(gpublasCreate, cublasCreate_v2)
GPUMOD_SAME_FUNCTION(gpublasDestroy, cublasDestroy_v2)
GPUMOD_SAME_FUNCTION(gpublasSetStream, cublasSetStream_v2)
GPUMOD_SAME_FUNCTION(gpublasGetStream, cublasGetStream_v2)
GPUMOD_SAME_FUNCTION(gpublasSetPointerMode, cublasSetPointerMode_v2)
GPUMOD_SAME_FUNCTION(gpublasGetStatusName, cublasGetStatusName)
GPUMOD_SAME_FUNCTION(gpublasGetStatusString, cublasGetStatusString)

GPUMOD_SAME_FUNCTION(gpublasSgetrsBatched, cublasSgetrsBatched)
GPUMOD_SAME_FUNCTION(gpublasDgetrsBatched, cublasDgetrsBatched)
GPUMOD_SAME_FUNCTION(gpublasCgetrsBatched, cublasCgetrsBatched)
GPUMOD_SAME_FUNCTION(gpublasZgetrsBatched, cublasZgetrsBatched)
GPUMOD_SAME_FUNCTION(gpublasSgetriBatched, cublasSgetriBatched)
GPUMOD_SAME_FUNCTION(gpublasDgetriBatched, cublasDgetriBatched)
GPUMOD_SAME_FUNCTION(gpublasCgetriBatched, cublasCgetriBatched)
GPUMOD_SAME_FUNCTION(gpublasZgetriBatched, cublasZgetriBatched)

#else

// ────────────────────────────────────────────────────────────────────────
// HIP backend
// ────────────────────────────────────────────────────────────────────────

using namespace gpumod::hip;

GPUMOD_SAME_TYPE(gpublasHandle_t, hipblasHandle_t)
GPUMOD_SAME_TYPE(gpublasStatus_t, hipblasStatus_t)
GPUMOD_SAME_TYPE(gpublasOperation_t, hipblasOperation_t)
GPUMOD_SAME_TYPE(gpublasFillMode_t, hipblasFillMode_t)
GPUMOD_SAME_TYPE(gpublasDiagType_t, hipblasDiagType_t)
GPUMOD_SAME_TYPE(gpublasSideMode_t, hipblasSideMode_t)
GPUMOD_SAME_TYPE(gpublasPointerMode_t, hipblasPointerMode_t)

GPUMOD_SAME_VALUE(GPUBLAS_STATUS_SUCCESS, HIPBLAS_STATUS_SUCCESS)
GPUMOD_SAME_VALUE(GPUBLAS_STATUS_NOT_INITIALIZED, HIPBLAS_STATUS_NOT_INITIALIZED)
GPUMOD_SAME_VALUE(GPUBLAS_OP_N, HIPBLAS_OP_N)
GPUMOD_SAME_VALUE(GPUBLAS_OP_T, HIPBLAS_OP_T)
GPUMOD_SAME_VALUE(GPUBLAS_OP_C, HIPBLAS_OP_C)
GPUMOD_SAME_VALUE(GPUBLAS_FILL_MODE_LOWER, HIPBLAS_FILL_MODE_LOWER)
GPUMOD_SAME_VALUE(GPUBLAS_FILL_MODE_UPPER, HIPBLAS_FILL_MODE_UPPER)
GPUMOD_SAME_VALUE(GPUBLAS_DIAG_NON_UNIT, HIPBLAS_DIAG_NON_UNIT)
GPUMOD_SAME_VALUE(GPUBLAS_DIAG_UNIT, HIPBLAS_DIAG_UNIT)
GPUMOD_SAME_VALUE(GPUBLAS_SIDE_LEFT, HIPBLAS_SIDE_LEFT)
GPUMOD_SAME_VALUE(GPUBLAS_SIDE_RIGHT, HIPBLAS_SIDE_RIGHT)
GPUMOD_SAME_VALUE(GPUBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_HOST)
GPUMOD_SAME_VALUE(GPUBLAS_POINTER_MODE_DEVICE, HIPBLAS_POINTER_MODE_DEVICE)

GPUMOD_SAME_FUNCTION(gpublasCreate, hipblasCreate)
GPUMOD_SAME_FUNCTION(gpublasDestroy, hipblasDestroy)
GPUMOD_SAME_FUNCTION(gpublasSetStream, hipblasSetStream)
GPUMOD_SAME_FUNCTION(gpublasGetStream, hipblasGetStream)
GPUMOD_SAME_FUNCTION(gpublasSetPointerMode, hipblasSetPointerMode)
// hipBLAS's one status-to-string function backs both names.
GPUMOD_SAME_FUNCTION(gpublasGetStatusName, hipblasStatusToString)
GPUMOD_SAME_FUNCTION(gpublasGetStatusString, hipblasStatusToString)

// getrsBatched/getriBatched are forwarding functions on HIP (hipBLAS declares
// the input arrays non-const); what matters is the signature, checked below
// for both backends. That they forward correctly is test/wrappers/blas's job.

#endif

// ────────────────────────────────────────────────────────────────────────
// Both backends: getrsBatched/getriBatched have cuBLAS's const-correct signature
// ────────────────────────────────────────────────────────────────────────

template<typename T>
using GetrsBatchedFn = gpublasStatus_t(gpublasHandle_t, gpublasOperation_t, int, int,
                                       const T *const *, int, const int *, T *const *, int, int *,
                                       int);
template<typename T>
using GetriBatchedFn = gpublasStatus_t(gpublasHandle_t, int, const T *const *, int, const int *,
                                       T *const *, int, int *, int);

GPUMOD_SAME_TYPE(std::remove_cvref_t<decltype(gpublasSgetrsBatched)>, GetrsBatchedFn<float>)
GPUMOD_SAME_TYPE(std::remove_cvref_t<decltype(gpublasDgetrsBatched)>, GetrsBatchedFn<double>)
GPUMOD_SAME_TYPE(std::remove_cvref_t<decltype(gpublasCgetrsBatched)>,
                 GetrsBatchedFn<gpuFloatComplex>)
GPUMOD_SAME_TYPE(std::remove_cvref_t<decltype(gpublasZgetrsBatched)>,
                 GetrsBatchedFn<gpuDoubleComplex>)
GPUMOD_SAME_TYPE(std::remove_cvref_t<decltype(gpublasSgetriBatched)>, GetriBatchedFn<float>)
GPUMOD_SAME_TYPE(std::remove_cvref_t<decltype(gpublasDgetriBatched)>, GetriBatchedFn<double>)
GPUMOD_SAME_TYPE(std::remove_cvref_t<decltype(gpublasCgetriBatched)>,
                 GetriBatchedFn<gpuFloatComplex>)
GPUMOD_SAME_TYPE(std::remove_cvref_t<decltype(gpublasZgetriBatched)>,
                 GetriBatchedFn<gpuDoubleComplex>)

} // namespace gpumod::test
