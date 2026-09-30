// blas.cppm - Compile-time tests for wwr.blas
//
// Types, constants, and the handle/stream/status functions are each checked
// against the backend's own entity (see gpu_check_macros.h). The two places
// where src/blas.cppm does more than rename -- the status strings on HIP
// and the const-correct getrsBatched/getriBatched -- are checked too.
//
// The typed BLAS entry points (wwrblasSgemm, ...) are not repeated here:
// blas.cppm writes each one out in full (WWR_FUNCTION(wwrblasSgemm,
// cublasSgemm_v2, hipblasSgemm)), not derived by a prefix-pasting macro, so a
// line here would restate that line. A misspelled backend name does not
// compile (wwr.cuda.cublas_v2 exports only the _v2 spellings), and a
// wrong-but-existing one is a signature mismatch at the instantiation in
// src/wrappers/blas/instantiations.cpp or a dispatch mismatch in the blas
// dispatch check (test/wrappers/blas/blas_dispatch.toml). That every alias is
// covered by one of those, or by a WWR_SAME_FUNCTION above, is enforced by
// test/shared/alias_coverage.py.

module;

#include "gpu_check_macros.h"

export module wwr.test.gpu.blas;

import std;
import wwr.blas;
import wwr.complex;
#if defined(WWR_GPU_BACKEND_CUDA)
import wwr.cuda.cublas_v2;
#else
import wwr.hip.hipblas;
#endif

namespace wwr::test {

using namespace wwr;

#if defined(WWR_GPU_BACKEND_CUDA)

using namespace wwr::cuda;

// ────────────────────────────────────────────────────────────────────────
// CUDA backend
// ────────────────────────────────────────────────────────────────────────

WWR_SAME_TYPE(wwrblasHandle_t, cublasHandle_t)
WWR_SAME_TYPE(wwrblasStatus_t, cublasStatus_t)
WWR_SAME_TYPE(wwrblasOperation_t, cublasOperation_t)
WWR_SAME_TYPE(wwrblasFillMode_t, cublasFillMode_t)
WWR_SAME_TYPE(wwrblasDiagType_t, cublasDiagType_t)
WWR_SAME_TYPE(wwrblasSideMode_t, cublasSideMode_t)
WWR_SAME_TYPE(wwrblasPointerMode_t, cublasPointerMode_t)

WWR_SAME_VALUE(WWRBLAS_STATUS_SUCCESS, CUBLAS_STATUS_SUCCESS)
WWR_SAME_VALUE(WWRBLAS_STATUS_NOT_INITIALIZED, CUBLAS_STATUS_NOT_INITIALIZED)
WWR_SAME_VALUE(WWRBLAS_STATUS_ALLOC_FAILED, CUBLAS_STATUS_ALLOC_FAILED)
WWR_SAME_VALUE(WWRBLAS_STATUS_ARCH_MISMATCH, CUBLAS_STATUS_ARCH_MISMATCH)
WWR_SAME_VALUE(WWRBLAS_STATUS_EXECUTION_FAILED, CUBLAS_STATUS_EXECUTION_FAILED)
WWR_SAME_VALUE(WWRBLAS_STATUS_INTERNAL_ERROR, CUBLAS_STATUS_INTERNAL_ERROR)
WWR_SAME_VALUE(WWRBLAS_STATUS_INVALID_VALUE, CUBLAS_STATUS_INVALID_VALUE)
WWR_SAME_VALUE(WWRBLAS_STATUS_MAPPING_ERROR, CUBLAS_STATUS_MAPPING_ERROR)
WWR_SAME_VALUE(WWRBLAS_STATUS_NOT_SUPPORTED, CUBLAS_STATUS_NOT_SUPPORTED)
WWR_SAME_VALUE(WWRBLAS_OP_N, CUBLAS_OP_N)
WWR_SAME_VALUE(WWRBLAS_OP_T, CUBLAS_OP_T)
WWR_SAME_VALUE(WWRBLAS_OP_C, CUBLAS_OP_C)
WWR_SAME_VALUE(WWRBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_LOWER)
WWR_SAME_VALUE(WWRBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_UPPER)
WWR_SAME_VALUE(WWRBLAS_DIAG_NON_UNIT, CUBLAS_DIAG_NON_UNIT)
WWR_SAME_VALUE(WWRBLAS_DIAG_UNIT, CUBLAS_DIAG_UNIT)
WWR_SAME_VALUE(WWRBLAS_SIDE_LEFT, CUBLAS_SIDE_LEFT)
WWR_SAME_VALUE(WWRBLAS_SIDE_RIGHT, CUBLAS_SIDE_RIGHT)
WWR_SAME_VALUE(WWRBLAS_POINTER_MODE_HOST, CUBLAS_POINTER_MODE_HOST)
WWR_SAME_VALUE(WWRBLAS_POINTER_MODE_DEVICE, CUBLAS_POINTER_MODE_DEVICE)

WWR_SAME_FUNCTION(wwrblasCreate, cublasCreate_v2)
WWR_SAME_FUNCTION(wwrblasDestroy, cublasDestroy_v2)
WWR_SAME_FUNCTION(wwrblasSetStream, cublasSetStream_v2)
WWR_SAME_FUNCTION(wwrblasGetStream, cublasGetStream_v2)
WWR_SAME_FUNCTION(wwrblasSetPointerMode, cublasSetPointerMode_v2)
WWR_SAME_FUNCTION(wwrblasGetPointerMode, cublasGetPointerMode_v2)
WWR_SAME_FUNCTION(wwrblasGetStatusName, cublasGetStatusName)
WWR_SAME_FUNCTION(wwrblasGetStatusString, cublasGetStatusString)

WWR_SAME_FUNCTION(wwrblasSgetrsBatched, cublasSgetrsBatched)
WWR_SAME_FUNCTION(wwrblasDgetrsBatched, cublasDgetrsBatched)
WWR_SAME_FUNCTION(wwrblasCgetrsBatched, cublasCgetrsBatched)
WWR_SAME_FUNCTION(wwrblasZgetrsBatched, cublasZgetrsBatched)
WWR_SAME_FUNCTION(wwrblasSgetriBatched, cublasSgetriBatched)
WWR_SAME_FUNCTION(wwrblasDgetriBatched, cublasDgetriBatched)
WWR_SAME_FUNCTION(wwrblasCgetriBatched, cublasCgetriBatched)
WWR_SAME_FUNCTION(wwrblasZgetriBatched, cublasZgetriBatched)

#else

// ────────────────────────────────────────────────────────────────────────
// HIP backend
// ────────────────────────────────────────────────────────────────────────

using namespace wwr::hip;

WWR_SAME_TYPE(wwrblasHandle_t, hipblasHandle_t)
WWR_SAME_TYPE(wwrblasStatus_t, hipblasStatus_t)
WWR_SAME_TYPE(wwrblasOperation_t, hipblasOperation_t)
WWR_SAME_TYPE(wwrblasFillMode_t, hipblasFillMode_t)
WWR_SAME_TYPE(wwrblasDiagType_t, hipblasDiagType_t)
WWR_SAME_TYPE(wwrblasSideMode_t, hipblasSideMode_t)
WWR_SAME_TYPE(wwrblasPointerMode_t, hipblasPointerMode_t)

WWR_SAME_VALUE(WWRBLAS_STATUS_SUCCESS, HIPBLAS_STATUS_SUCCESS)
WWR_SAME_VALUE(WWRBLAS_STATUS_NOT_INITIALIZED, HIPBLAS_STATUS_NOT_INITIALIZED)
WWR_SAME_VALUE(WWRBLAS_STATUS_ALLOC_FAILED, HIPBLAS_STATUS_ALLOC_FAILED)
WWR_SAME_VALUE(WWRBLAS_STATUS_ARCH_MISMATCH, HIPBLAS_STATUS_ARCH_MISMATCH)
WWR_SAME_VALUE(WWRBLAS_STATUS_EXECUTION_FAILED, HIPBLAS_STATUS_EXECUTION_FAILED)
WWR_SAME_VALUE(WWRBLAS_STATUS_INTERNAL_ERROR, HIPBLAS_STATUS_INTERNAL_ERROR)
WWR_SAME_VALUE(WWRBLAS_STATUS_INVALID_VALUE, HIPBLAS_STATUS_INVALID_VALUE)
WWR_SAME_VALUE(WWRBLAS_STATUS_MAPPING_ERROR, HIPBLAS_STATUS_MAPPING_ERROR)
WWR_SAME_VALUE(WWRBLAS_STATUS_NOT_SUPPORTED, HIPBLAS_STATUS_NOT_SUPPORTED)
WWR_SAME_VALUE(WWRBLAS_OP_N, HIPBLAS_OP_N)
WWR_SAME_VALUE(WWRBLAS_OP_T, HIPBLAS_OP_T)
WWR_SAME_VALUE(WWRBLAS_OP_C, HIPBLAS_OP_C)
WWR_SAME_VALUE(WWRBLAS_FILL_MODE_LOWER, HIPBLAS_FILL_MODE_LOWER)
WWR_SAME_VALUE(WWRBLAS_FILL_MODE_UPPER, HIPBLAS_FILL_MODE_UPPER)
WWR_SAME_VALUE(WWRBLAS_DIAG_NON_UNIT, HIPBLAS_DIAG_NON_UNIT)
WWR_SAME_VALUE(WWRBLAS_DIAG_UNIT, HIPBLAS_DIAG_UNIT)
WWR_SAME_VALUE(WWRBLAS_SIDE_LEFT, HIPBLAS_SIDE_LEFT)
WWR_SAME_VALUE(WWRBLAS_SIDE_RIGHT, HIPBLAS_SIDE_RIGHT)
WWR_SAME_VALUE(WWRBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_HOST)
WWR_SAME_VALUE(WWRBLAS_POINTER_MODE_DEVICE, HIPBLAS_POINTER_MODE_DEVICE)

WWR_SAME_FUNCTION(wwrblasCreate, hipblasCreate)
WWR_SAME_FUNCTION(wwrblasDestroy, hipblasDestroy)
WWR_SAME_FUNCTION(wwrblasSetStream, hipblasSetStream)
WWR_SAME_FUNCTION(wwrblasGetStream, hipblasGetStream)
WWR_SAME_FUNCTION(wwrblasSetPointerMode, hipblasSetPointerMode)
// hipBLAS's one status-to-string function backs both names.
WWR_SAME_FUNCTION(wwrblasGetStatusName, hipblasStatusToString)
WWR_SAME_FUNCTION(wwrblasGetStatusString, hipblasStatusToString)

// getrsBatched/getriBatched are forwarding functions on HIP (hipBLAS declares
// the input arrays non-const); what matters is the signature, checked below
// for both backends. That they forward correctly is test/wrappers/blas's job.

#endif

// ────────────────────────────────────────────────────────────────────────
// Both backends: getrsBatched/getriBatched have cuBLAS's const-correct signature
// ────────────────────────────────────────────────────────────────────────

template<typename T>
using GetrsBatchedFn = wwrblasStatus_t(wwrblasHandle_t, wwrblasOperation_t, int, int,
                                       const T *const *, int, const int *, T *const *, int, int *,
                                       int);
template<typename T>
using GetriBatchedFn = wwrblasStatus_t(wwrblasHandle_t, int, const T *const *, int, const int *,
                                       T *const *, int, int *, int);

WWR_SAME_TYPE(std::remove_cvref_t<decltype(wwrblasSgetrsBatched)>, GetrsBatchedFn<float>)
WWR_SAME_TYPE(std::remove_cvref_t<decltype(wwrblasDgetrsBatched)>, GetrsBatchedFn<double>)
WWR_SAME_TYPE(std::remove_cvref_t<decltype(wwrblasCgetrsBatched)>,
                 GetrsBatchedFn<wwrFloatComplex>)
WWR_SAME_TYPE(std::remove_cvref_t<decltype(wwrblasZgetrsBatched)>,
                 GetrsBatchedFn<wwrDoubleComplex>)
WWR_SAME_TYPE(std::remove_cvref_t<decltype(wwrblasSgetriBatched)>, GetriBatchedFn<float>)
WWR_SAME_TYPE(std::remove_cvref_t<decltype(wwrblasDgetriBatched)>, GetriBatchedFn<double>)
WWR_SAME_TYPE(std::remove_cvref_t<decltype(wwrblasCgetriBatched)>,
                 GetriBatchedFn<wwrFloatComplex>)
WWR_SAME_TYPE(std::remove_cvref_t<decltype(wwrblasZgetriBatched)>,
                 GetriBatchedFn<wwrDoubleComplex>)

} // namespace wwr::test
