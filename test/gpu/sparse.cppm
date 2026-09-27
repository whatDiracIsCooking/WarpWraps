// sparse.cppm - Compile-time tests for gpumod.sparse
//
// Types, constants, and the handle/stream/pointer-mode/error-string and
// matrix-descriptor functions are each checked against the backend's own entity
// (see gpu_check_macros.h). The one place src/sparse.cppm does more than
// rename -- the std::size_t* buffer-size signature of gebsr2gebsc_bufferSize /
// csr2gebsr_bufferSize, forwarded through an int on CUDA -- is checked too.
//
// The typed S/D/C/Z entry points (gpusparseSbsrmv, ...) are not repeated here:
// sparse.cppm writes each one out in full (WWR_FUNCTION(gpusparseSbsrmv,
// cusparseSbsrmv, hipsparseSbsrmv)), so a line here would restate that line. A
// misspelled backend name does not compile (the raw module exports only the
// real spellings), and a wrong-but-existing one is a signature mismatch at the
// instantiation in src/wrappers/sparse/instantiations.cpp or a dispatch
// mismatch in test/wrappers/sparse.

module;

#include "gpu_check_macros.h"

export module gpumod.test.gpu.sparse;

import std;
import gpumod.sparse;
import gpumod.complex;
#if defined(WWR_GPU_BACKEND_CUDA)
import gpumod.cuda.cusparse;
#else
import gpumod.hip.hipsparse;
#endif

namespace wwr::test {

using namespace wwr;

#if defined(WWR_GPU_BACKEND_CUDA)

using namespace wwr::cuda;

// ────────────────────────────────────────────────────────────────────────
// CUDA backend
// ────────────────────────────────────────────────────────────────────────

WWR_SAME_TYPE(gpusparseHandle_t, cusparseHandle_t)
WWR_SAME_TYPE(gpusparseStatus_t, cusparseStatus_t)
WWR_SAME_TYPE(gpusparseMatDescr_t, cusparseMatDescr_t)
WWR_SAME_TYPE(gpusparseOperation_t, cusparseOperation_t)
WWR_SAME_TYPE(gpusparseDirection_t, cusparseDirection_t)
WWR_SAME_TYPE(gpusparseAction_t, cusparseAction_t)
WWR_SAME_TYPE(gpusparseIndexBase_t, cusparseIndexBase_t)
WWR_SAME_TYPE(gpusparseMatrixType_t, cusparseMatrixType_t)
WWR_SAME_TYPE(gpusparseFillMode_t, cusparseFillMode_t)
WWR_SAME_TYPE(gpusparseDiagType_t, cusparseDiagType_t)
WWR_SAME_TYPE(gpusparsePointerMode_t, cusparsePointerMode_t)

WWR_SAME_VALUE(GPUSPARSE_STATUS_SUCCESS, CUSPARSE_STATUS_SUCCESS)
WWR_SAME_VALUE(GPUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE)
WWR_SAME_VALUE(GPUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE)
WWR_SAME_VALUE(GPUSPARSE_OPERATION_CONJUGATE_TRANSPOSE, CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE)
WWR_SAME_VALUE(GPUSPARSE_DIRECTION_ROW, CUSPARSE_DIRECTION_ROW)
WWR_SAME_VALUE(GPUSPARSE_DIRECTION_COLUMN, CUSPARSE_DIRECTION_COLUMN)
WWR_SAME_VALUE(GPUSPARSE_ACTION_SYMBOLIC, CUSPARSE_ACTION_SYMBOLIC)
WWR_SAME_VALUE(GPUSPARSE_ACTION_NUMERIC, CUSPARSE_ACTION_NUMERIC)
WWR_SAME_VALUE(GPUSPARSE_INDEX_BASE_ZERO, CUSPARSE_INDEX_BASE_ZERO)
WWR_SAME_VALUE(GPUSPARSE_INDEX_BASE_ONE, CUSPARSE_INDEX_BASE_ONE)
WWR_SAME_VALUE(GPUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_MATRIX_TYPE_GENERAL)
WWR_SAME_VALUE(GPUSPARSE_MATRIX_TYPE_SYMMETRIC, CUSPARSE_MATRIX_TYPE_SYMMETRIC)
WWR_SAME_VALUE(GPUSPARSE_MATRIX_TYPE_HERMITIAN, CUSPARSE_MATRIX_TYPE_HERMITIAN)
WWR_SAME_VALUE(GPUSPARSE_MATRIX_TYPE_TRIANGULAR, CUSPARSE_MATRIX_TYPE_TRIANGULAR)
WWR_SAME_VALUE(GPUSPARSE_FILL_MODE_LOWER, CUSPARSE_FILL_MODE_LOWER)
WWR_SAME_VALUE(GPUSPARSE_FILL_MODE_UPPER, CUSPARSE_FILL_MODE_UPPER)
WWR_SAME_VALUE(GPUSPARSE_DIAG_TYPE_NON_UNIT, CUSPARSE_DIAG_TYPE_NON_UNIT)
WWR_SAME_VALUE(GPUSPARSE_DIAG_TYPE_UNIT, CUSPARSE_DIAG_TYPE_UNIT)
WWR_SAME_VALUE(GPUSPARSE_POINTER_MODE_HOST, CUSPARSE_POINTER_MODE_HOST)
WWR_SAME_VALUE(GPUSPARSE_POINTER_MODE_DEVICE, CUSPARSE_POINTER_MODE_DEVICE)

WWR_SAME_FUNCTION(gpusparseCreate, cusparseCreate)
WWR_SAME_FUNCTION(gpusparseDestroy, cusparseDestroy)
WWR_SAME_FUNCTION(gpusparseSetStream, cusparseSetStream)
WWR_SAME_FUNCTION(gpusparseGetStream, cusparseGetStream)
WWR_SAME_FUNCTION(gpusparseSetPointerMode, cusparseSetPointerMode)
WWR_SAME_FUNCTION(gpusparseGetPointerMode, cusparseGetPointerMode)
WWR_SAME_FUNCTION(gpusparseGetErrorName, cusparseGetErrorName)
WWR_SAME_FUNCTION(gpusparseGetErrorString, cusparseGetErrorString)
WWR_SAME_FUNCTION(gpusparseCreateMatDescr, cusparseCreateMatDescr)
WWR_SAME_FUNCTION(gpusparseDestroyMatDescr, cusparseDestroyMatDescr)
WWR_SAME_FUNCTION(gpusparseSetMatType, cusparseSetMatType)
WWR_SAME_FUNCTION(gpusparseGetMatType, cusparseGetMatType)
WWR_SAME_FUNCTION(gpusparseSetMatFillMode, cusparseSetMatFillMode)
WWR_SAME_FUNCTION(gpusparseGetMatFillMode, cusparseGetMatFillMode)
WWR_SAME_FUNCTION(gpusparseSetMatDiagType, cusparseSetMatDiagType)
WWR_SAME_FUNCTION(gpusparseGetMatDiagType, cusparseGetMatDiagType)
WWR_SAME_FUNCTION(gpusparseSetMatIndexBase, cusparseSetMatIndexBase)
WWR_SAME_FUNCTION(gpusparseGetMatIndexBase, cusparseGetMatIndexBase)

#else

// ────────────────────────────────────────────────────────────────────────
// HIP backend
// ────────────────────────────────────────────────────────────────────────

using namespace wwr::hip;

WWR_SAME_TYPE(gpusparseHandle_t, hipsparseHandle_t)
WWR_SAME_TYPE(gpusparseStatus_t, hipsparseStatus_t)
WWR_SAME_TYPE(gpusparseMatDescr_t, hipsparseMatDescr_t)
WWR_SAME_TYPE(gpusparseOperation_t, hipsparseOperation_t)
WWR_SAME_TYPE(gpusparseDirection_t, hipsparseDirection_t)
WWR_SAME_TYPE(gpusparseAction_t, hipsparseAction_t)
WWR_SAME_TYPE(gpusparseIndexBase_t, hipsparseIndexBase_t)
WWR_SAME_TYPE(gpusparseMatrixType_t, hipsparseMatrixType_t)
WWR_SAME_TYPE(gpusparseFillMode_t, hipsparseFillMode_t)
WWR_SAME_TYPE(gpusparseDiagType_t, hipsparseDiagType_t)
WWR_SAME_TYPE(gpusparsePointerMode_t, hipsparsePointerMode_t)

WWR_SAME_VALUE(GPUSPARSE_STATUS_SUCCESS, HIPSPARSE_STATUS_SUCCESS)
WWR_SAME_VALUE(GPUSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE)
WWR_SAME_VALUE(GPUSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_TRANSPOSE)
WWR_SAME_VALUE(GPUSPARSE_OPERATION_CONJUGATE_TRANSPOSE, HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE)
WWR_SAME_VALUE(GPUSPARSE_DIRECTION_ROW, HIPSPARSE_DIRECTION_ROW)
WWR_SAME_VALUE(GPUSPARSE_DIRECTION_COLUMN, HIPSPARSE_DIRECTION_COLUMN)
WWR_SAME_VALUE(GPUSPARSE_ACTION_SYMBOLIC, HIPSPARSE_ACTION_SYMBOLIC)
WWR_SAME_VALUE(GPUSPARSE_ACTION_NUMERIC, HIPSPARSE_ACTION_NUMERIC)
WWR_SAME_VALUE(GPUSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ZERO)
WWR_SAME_VALUE(GPUSPARSE_INDEX_BASE_ONE, HIPSPARSE_INDEX_BASE_ONE)
WWR_SAME_VALUE(GPUSPARSE_MATRIX_TYPE_GENERAL, HIPSPARSE_MATRIX_TYPE_GENERAL)
WWR_SAME_VALUE(GPUSPARSE_MATRIX_TYPE_SYMMETRIC, HIPSPARSE_MATRIX_TYPE_SYMMETRIC)
WWR_SAME_VALUE(GPUSPARSE_MATRIX_TYPE_HERMITIAN, HIPSPARSE_MATRIX_TYPE_HERMITIAN)
WWR_SAME_VALUE(GPUSPARSE_MATRIX_TYPE_TRIANGULAR, HIPSPARSE_MATRIX_TYPE_TRIANGULAR)
WWR_SAME_VALUE(GPUSPARSE_FILL_MODE_LOWER, HIPSPARSE_FILL_MODE_LOWER)
WWR_SAME_VALUE(GPUSPARSE_FILL_MODE_UPPER, HIPSPARSE_FILL_MODE_UPPER)
WWR_SAME_VALUE(GPUSPARSE_DIAG_TYPE_NON_UNIT, HIPSPARSE_DIAG_TYPE_NON_UNIT)
WWR_SAME_VALUE(GPUSPARSE_DIAG_TYPE_UNIT, HIPSPARSE_DIAG_TYPE_UNIT)
WWR_SAME_VALUE(GPUSPARSE_POINTER_MODE_HOST, HIPSPARSE_POINTER_MODE_HOST)
WWR_SAME_VALUE(GPUSPARSE_POINTER_MODE_DEVICE, HIPSPARSE_POINTER_MODE_DEVICE)

WWR_SAME_FUNCTION(gpusparseCreate, hipsparseCreate)
WWR_SAME_FUNCTION(gpusparseDestroy, hipsparseDestroy)
WWR_SAME_FUNCTION(gpusparseSetStream, hipsparseSetStream)
WWR_SAME_FUNCTION(gpusparseGetStream, hipsparseGetStream)
WWR_SAME_FUNCTION(gpusparseSetPointerMode, hipsparseSetPointerMode)
WWR_SAME_FUNCTION(gpusparseGetPointerMode, hipsparseGetPointerMode)
WWR_SAME_FUNCTION(gpusparseGetErrorName, hipsparseGetErrorName)
WWR_SAME_FUNCTION(gpusparseGetErrorString, hipsparseGetErrorString)
WWR_SAME_FUNCTION(gpusparseCreateMatDescr, hipsparseCreateMatDescr)
WWR_SAME_FUNCTION(gpusparseDestroyMatDescr, hipsparseDestroyMatDescr)
WWR_SAME_FUNCTION(gpusparseSetMatType, hipsparseSetMatType)
WWR_SAME_FUNCTION(gpusparseGetMatType, hipsparseGetMatType)
WWR_SAME_FUNCTION(gpusparseSetMatFillMode, hipsparseSetMatFillMode)
WWR_SAME_FUNCTION(gpusparseGetMatFillMode, hipsparseGetMatFillMode)
WWR_SAME_FUNCTION(gpusparseSetMatDiagType, hipsparseSetMatDiagType)
WWR_SAME_FUNCTION(gpusparseGetMatDiagType, hipsparseGetMatDiagType)
WWR_SAME_FUNCTION(gpusparseSetMatIndexBase, hipsparseSetMatIndexBase)
WWR_SAME_FUNCTION(gpusparseGetMatIndexBase, hipsparseGetMatIndexBase)

// gebsr2gebsc_bufferSize / csr2gebsr_bufferSize are forwarding functions on
// CUDA (cuSPARSE writes the byte count as int*); what matters is the unified
// std::size_t* signature, checked below for both backends.

#endif

// ────────────────────────────────────────────────────────────────────────
// Both backends: the two *_bufferSize families take std::size_t* on the neutral API
// ────────────────────────────────────────────────────────────────────────

template<typename T>
using Gebsr2gebscBufferSizeFn = gpusparseStatus_t(gpusparseHandle_t, int, int, int, const T *,
                                                  const int *, const int *, int, int,
                                                  std::size_t *);
template<typename T>
using Csr2gebsrBufferSizeFn = gpusparseStatus_t(gpusparseHandle_t, gpusparseDirection_t, int, int,
                                                const gpusparseMatDescr_t, const T *, const int *,
                                                const int *, int, int, std::size_t *);

WWR_SAME_TYPE(std::remove_cvref_t<decltype(gpusparseSgebsr2gebsc_bufferSize)>,
                 Gebsr2gebscBufferSizeFn<float>)
WWR_SAME_TYPE(std::remove_cvref_t<decltype(gpusparseDgebsr2gebsc_bufferSize)>,
                 Gebsr2gebscBufferSizeFn<double>)
WWR_SAME_TYPE(std::remove_cvref_t<decltype(gpusparseCgebsr2gebsc_bufferSize)>,
                 Gebsr2gebscBufferSizeFn<gpuFloatComplex>)
WWR_SAME_TYPE(std::remove_cvref_t<decltype(gpusparseZgebsr2gebsc_bufferSize)>,
                 Gebsr2gebscBufferSizeFn<gpuDoubleComplex>)
WWR_SAME_TYPE(std::remove_cvref_t<decltype(gpusparseScsr2gebsr_bufferSize)>,
                 Csr2gebsrBufferSizeFn<float>)
WWR_SAME_TYPE(std::remove_cvref_t<decltype(gpusparseDcsr2gebsr_bufferSize)>,
                 Csr2gebsrBufferSizeFn<double>)
WWR_SAME_TYPE(std::remove_cvref_t<decltype(gpusparseCcsr2gebsr_bufferSize)>,
                 Csr2gebsrBufferSizeFn<gpuFloatComplex>)
WWR_SAME_TYPE(std::remove_cvref_t<decltype(gpusparseZcsr2gebsr_bufferSize)>,
                 Csr2gebsrBufferSizeFn<gpuDoubleComplex>)

} // namespace wwr::test
