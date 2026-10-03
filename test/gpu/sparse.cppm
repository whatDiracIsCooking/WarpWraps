// sparse.cppm - Compile-time tests for wwr.sparse
//
// Types, constants, and the handle/stream/pointer-mode/error-string and
// matrix-descriptor functions are each checked against the backend's own entity
// (see gpu_check_macros.h). The one place the sparse surface does more than
// rename -- the std::size_t* buffer-size signature of gebsr2gebsc_bufferSize /
// csr2gebsr_bufferSize, forwarded through an int on CUDA -- is checked too.
//
// The typed S/D/C/Z entry points (wwrsparseSbsrmv, ...) are not repeated here:
// src/detail/sparse_names.h writes each one out in full
// (WWR_FUNCTION_RAW(wwrsparseSbsrmv, cusparseSbsrmv, hipsparseSbsrmv)), so a
// line here would restate that line. A
// misspelled backend name does not compile (the raw module exports only the
// real spellings), and a wrong-but-existing one is a signature mismatch at the
// instantiation in src/wrappers/sparse/instantiations.cpp or a dispatch
// mismatch in test/wrappers/sparse.

module;

#include "gpu_check_macros.h"

export module wwr.test.gpu.sparse;

import std;
import wwr.sparse;
import wwr.complex;
#if defined(WWR_GPU_BACKEND_CUDA)
import wwr.cuda.cusparse;
#else
import wwr.hip.hipsparse;
#endif

namespace wwr::test {

using namespace wwr;

#if defined(WWR_GPU_BACKEND_CUDA)

using namespace wwr::cuda;

// ────────────────────────────────────────────────────────────────────────
// CUDA backend
// ────────────────────────────────────────────────────────────────────────

WWR_SAME_TYPE(wwrsparseHandle_t, cusparseHandle_t)
WWR_SAME_TYPE(wwrsparseStatus_t, cusparseStatus_t)
WWR_SAME_TYPE(wwrsparseMatDescr_t, cusparseMatDescr_t)
WWR_SAME_TYPE(wwrsparseOperation_t, cusparseOperation_t)
WWR_SAME_TYPE(wwrsparseDirection_t, cusparseDirection_t)
WWR_SAME_TYPE(wwrsparseAction_t, cusparseAction_t)
WWR_SAME_TYPE(wwrsparseIndexBase_t, cusparseIndexBase_t)
WWR_SAME_TYPE(wwrsparseMatrixType_t, cusparseMatrixType_t)
WWR_SAME_TYPE(wwrsparseFillMode_t, cusparseFillMode_t)
WWR_SAME_TYPE(wwrsparseDiagType_t, cusparseDiagType_t)
WWR_SAME_TYPE(wwrsparsePointerMode_t, cusparsePointerMode_t)

WWR_SAME_VALUE(WWRSPARSE_STATUS_SUCCESS, CUSPARSE_STATUS_SUCCESS)
WWR_SAME_VALUE(WWRSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE)
WWR_SAME_VALUE(WWRSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE)
WWR_SAME_VALUE(WWRSPARSE_OPERATION_CONJUGATE_TRANSPOSE, CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE)
WWR_SAME_VALUE(WWRSPARSE_DIRECTION_ROW, CUSPARSE_DIRECTION_ROW)
WWR_SAME_VALUE(WWRSPARSE_DIRECTION_COLUMN, CUSPARSE_DIRECTION_COLUMN)
WWR_SAME_VALUE(WWRSPARSE_ACTION_SYMBOLIC, CUSPARSE_ACTION_SYMBOLIC)
WWR_SAME_VALUE(WWRSPARSE_ACTION_NUMERIC, CUSPARSE_ACTION_NUMERIC)
WWR_SAME_VALUE(WWRSPARSE_INDEX_BASE_ZERO, CUSPARSE_INDEX_BASE_ZERO)
WWR_SAME_VALUE(WWRSPARSE_INDEX_BASE_ONE, CUSPARSE_INDEX_BASE_ONE)
WWR_SAME_VALUE(WWRSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_MATRIX_TYPE_GENERAL)
WWR_SAME_VALUE(WWRSPARSE_MATRIX_TYPE_SYMMETRIC, CUSPARSE_MATRIX_TYPE_SYMMETRIC)
WWR_SAME_VALUE(WWRSPARSE_MATRIX_TYPE_HERMITIAN, CUSPARSE_MATRIX_TYPE_HERMITIAN)
WWR_SAME_VALUE(WWRSPARSE_MATRIX_TYPE_TRIANGULAR, CUSPARSE_MATRIX_TYPE_TRIANGULAR)
WWR_SAME_VALUE(WWRSPARSE_FILL_MODE_LOWER, CUSPARSE_FILL_MODE_LOWER)
WWR_SAME_VALUE(WWRSPARSE_FILL_MODE_UPPER, CUSPARSE_FILL_MODE_UPPER)
WWR_SAME_VALUE(WWRSPARSE_DIAG_TYPE_NON_UNIT, CUSPARSE_DIAG_TYPE_NON_UNIT)
WWR_SAME_VALUE(WWRSPARSE_DIAG_TYPE_UNIT, CUSPARSE_DIAG_TYPE_UNIT)
WWR_SAME_VALUE(WWRSPARSE_POINTER_MODE_HOST, CUSPARSE_POINTER_MODE_HOST)
WWR_SAME_VALUE(WWRSPARSE_POINTER_MODE_DEVICE, CUSPARSE_POINTER_MODE_DEVICE)

WWR_SAME_FUNCTION(wwrsparseCreate, cusparseCreate)
WWR_SAME_FUNCTION(wwrsparseDestroy, cusparseDestroy)
WWR_SAME_FUNCTION(wwrsparseSetStream, cusparseSetStream)
WWR_SAME_FUNCTION(wwrsparseGetStream, cusparseGetStream)
WWR_SAME_FUNCTION(wwrsparseSetPointerMode, cusparseSetPointerMode)
WWR_SAME_FUNCTION(wwrsparseGetPointerMode, cusparseGetPointerMode)
WWR_SAME_FUNCTION(wwrsparseGetErrorName, cusparseGetErrorName)
WWR_SAME_FUNCTION(wwrsparseGetErrorString, cusparseGetErrorString)
WWR_SAME_FUNCTION(wwrsparseCreateMatDescr, cusparseCreateMatDescr)
WWR_SAME_FUNCTION(wwrsparseDestroyMatDescr, cusparseDestroyMatDescr)
WWR_SAME_FUNCTION(wwrsparseSetMatType, cusparseSetMatType)
WWR_SAME_FUNCTION(wwrsparseGetMatType, cusparseGetMatType)
WWR_SAME_FUNCTION(wwrsparseSetMatFillMode, cusparseSetMatFillMode)
WWR_SAME_FUNCTION(wwrsparseGetMatFillMode, cusparseGetMatFillMode)
WWR_SAME_FUNCTION(wwrsparseSetMatDiagType, cusparseSetMatDiagType)
WWR_SAME_FUNCTION(wwrsparseGetMatDiagType, cusparseGetMatDiagType)
WWR_SAME_FUNCTION(wwrsparseSetMatIndexBase, cusparseSetMatIndexBase)
WWR_SAME_FUNCTION(wwrsparseGetMatIndexBase, cusparseGetMatIndexBase)

#else

// ────────────────────────────────────────────────────────────────────────
// HIP backend
// ────────────────────────────────────────────────────────────────────────

using namespace wwr::hip;

WWR_SAME_TYPE(wwrsparseHandle_t, hipsparseHandle_t)
WWR_SAME_TYPE(wwrsparseStatus_t, hipsparseStatus_t)
WWR_SAME_TYPE(wwrsparseMatDescr_t, hipsparseMatDescr_t)
WWR_SAME_TYPE(wwrsparseOperation_t, hipsparseOperation_t)
WWR_SAME_TYPE(wwrsparseDirection_t, hipsparseDirection_t)
WWR_SAME_TYPE(wwrsparseAction_t, hipsparseAction_t)
WWR_SAME_TYPE(wwrsparseIndexBase_t, hipsparseIndexBase_t)
WWR_SAME_TYPE(wwrsparseMatrixType_t, hipsparseMatrixType_t)
WWR_SAME_TYPE(wwrsparseFillMode_t, hipsparseFillMode_t)
WWR_SAME_TYPE(wwrsparseDiagType_t, hipsparseDiagType_t)
WWR_SAME_TYPE(wwrsparsePointerMode_t, hipsparsePointerMode_t)

WWR_SAME_VALUE(WWRSPARSE_STATUS_SUCCESS, HIPSPARSE_STATUS_SUCCESS)
WWR_SAME_VALUE(WWRSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE)
WWR_SAME_VALUE(WWRSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_TRANSPOSE)
WWR_SAME_VALUE(WWRSPARSE_OPERATION_CONJUGATE_TRANSPOSE, HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE)
WWR_SAME_VALUE(WWRSPARSE_DIRECTION_ROW, HIPSPARSE_DIRECTION_ROW)
WWR_SAME_VALUE(WWRSPARSE_DIRECTION_COLUMN, HIPSPARSE_DIRECTION_COLUMN)
WWR_SAME_VALUE(WWRSPARSE_ACTION_SYMBOLIC, HIPSPARSE_ACTION_SYMBOLIC)
WWR_SAME_VALUE(WWRSPARSE_ACTION_NUMERIC, HIPSPARSE_ACTION_NUMERIC)
WWR_SAME_VALUE(WWRSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ZERO)
WWR_SAME_VALUE(WWRSPARSE_INDEX_BASE_ONE, HIPSPARSE_INDEX_BASE_ONE)
WWR_SAME_VALUE(WWRSPARSE_MATRIX_TYPE_GENERAL, HIPSPARSE_MATRIX_TYPE_GENERAL)
WWR_SAME_VALUE(WWRSPARSE_MATRIX_TYPE_SYMMETRIC, HIPSPARSE_MATRIX_TYPE_SYMMETRIC)
WWR_SAME_VALUE(WWRSPARSE_MATRIX_TYPE_HERMITIAN, HIPSPARSE_MATRIX_TYPE_HERMITIAN)
WWR_SAME_VALUE(WWRSPARSE_MATRIX_TYPE_TRIANGULAR, HIPSPARSE_MATRIX_TYPE_TRIANGULAR)
WWR_SAME_VALUE(WWRSPARSE_FILL_MODE_LOWER, HIPSPARSE_FILL_MODE_LOWER)
WWR_SAME_VALUE(WWRSPARSE_FILL_MODE_UPPER, HIPSPARSE_FILL_MODE_UPPER)
WWR_SAME_VALUE(WWRSPARSE_DIAG_TYPE_NON_UNIT, HIPSPARSE_DIAG_TYPE_NON_UNIT)
WWR_SAME_VALUE(WWRSPARSE_DIAG_TYPE_UNIT, HIPSPARSE_DIAG_TYPE_UNIT)
WWR_SAME_VALUE(WWRSPARSE_POINTER_MODE_HOST, HIPSPARSE_POINTER_MODE_HOST)
WWR_SAME_VALUE(WWRSPARSE_POINTER_MODE_DEVICE, HIPSPARSE_POINTER_MODE_DEVICE)

WWR_SAME_FUNCTION(wwrsparseCreate, hipsparseCreate)
WWR_SAME_FUNCTION(wwrsparseDestroy, hipsparseDestroy)
WWR_SAME_FUNCTION(wwrsparseSetStream, hipsparseSetStream)
WWR_SAME_FUNCTION(wwrsparseGetStream, hipsparseGetStream)
WWR_SAME_FUNCTION(wwrsparseSetPointerMode, hipsparseSetPointerMode)
WWR_SAME_FUNCTION(wwrsparseGetPointerMode, hipsparseGetPointerMode)
WWR_SAME_FUNCTION(wwrsparseGetErrorName, hipsparseGetErrorName)
WWR_SAME_FUNCTION(wwrsparseGetErrorString, hipsparseGetErrorString)
WWR_SAME_FUNCTION(wwrsparseCreateMatDescr, hipsparseCreateMatDescr)
WWR_SAME_FUNCTION(wwrsparseDestroyMatDescr, hipsparseDestroyMatDescr)
WWR_SAME_FUNCTION(wwrsparseSetMatType, hipsparseSetMatType)
WWR_SAME_FUNCTION(wwrsparseGetMatType, hipsparseGetMatType)
WWR_SAME_FUNCTION(wwrsparseSetMatFillMode, hipsparseSetMatFillMode)
WWR_SAME_FUNCTION(wwrsparseGetMatFillMode, hipsparseGetMatFillMode)
WWR_SAME_FUNCTION(wwrsparseSetMatDiagType, hipsparseSetMatDiagType)
WWR_SAME_FUNCTION(wwrsparseGetMatDiagType, hipsparseGetMatDiagType)
WWR_SAME_FUNCTION(wwrsparseSetMatIndexBase, hipsparseSetMatIndexBase)
WWR_SAME_FUNCTION(wwrsparseGetMatIndexBase, hipsparseGetMatIndexBase)

// gebsr2gebsc_bufferSize / csr2gebsr_bufferSize are forwarding functions on
// CUDA (cuSPARSE writes the byte count as int*); what matters is the unified
// std::size_t* signature, checked below for both backends.

#endif

// ────────────────────────────────────────────────────────────────────────
// Both backends: the two *_bufferSize families take std::size_t* on the neutral API
// ────────────────────────────────────────────────────────────────────────

template<typename T>
using Gebsr2gebscBufferSizeFn = wwrsparseStatus_t(wwrsparseHandle_t, int, int, int, const T *,
                                                  const int *, const int *, int, int,
                                                  std::size_t *);
template<typename T>
using Csr2gebsrBufferSizeFn = wwrsparseStatus_t(wwrsparseHandle_t, wwrsparseDirection_t, int, int,
                                                const wwrsparseMatDescr_t, const T *, const int *,
                                                const int *, int, int, std::size_t *);

WWR_SAME_TYPE(std::remove_cvref_t<decltype(wwrsparseSgebsr2gebsc_bufferSize)>,
                 Gebsr2gebscBufferSizeFn<float>)
WWR_SAME_TYPE(std::remove_cvref_t<decltype(wwrsparseDgebsr2gebsc_bufferSize)>,
                 Gebsr2gebscBufferSizeFn<double>)
WWR_SAME_TYPE(std::remove_cvref_t<decltype(wwrsparseCgebsr2gebsc_bufferSize)>,
                 Gebsr2gebscBufferSizeFn<wwrFloatComplex>)
WWR_SAME_TYPE(std::remove_cvref_t<decltype(wwrsparseZgebsr2gebsc_bufferSize)>,
                 Gebsr2gebscBufferSizeFn<wwrDoubleComplex>)
WWR_SAME_TYPE(std::remove_cvref_t<decltype(wwrsparseScsr2gebsr_bufferSize)>,
                 Csr2gebsrBufferSizeFn<float>)
WWR_SAME_TYPE(std::remove_cvref_t<decltype(wwrsparseDcsr2gebsr_bufferSize)>,
                 Csr2gebsrBufferSizeFn<double>)
WWR_SAME_TYPE(std::remove_cvref_t<decltype(wwrsparseCcsr2gebsr_bufferSize)>,
                 Csr2gebsrBufferSizeFn<wwrFloatComplex>)
WWR_SAME_TYPE(std::remove_cvref_t<decltype(wwrsparseZcsr2gebsr_bufferSize)>,
                 Csr2gebsrBufferSizeFn<wwrDoubleComplex>)

} // namespace wwr::test
