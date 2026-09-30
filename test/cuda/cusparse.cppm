// cusparse.cppm - Compile-time tests for wwr.cuda.cusparse

module;

#include "test/shared/link_check.h"

export module wwr.test.cuda.cusparse;

import std;
import wwr.cuda.cusparse;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time tests for wwr.cuda.cusparse
//
// The module is a pure re-export (using declarations).
// We verify at compile-time that:
//   1. Key enum types are actually enum types (std::is_enum_v<>)
//   2. Key enumerator values with cuSPARSE-specified values are correct
//   3. Opaque handle types are pointer types (std::is_pointer_v<>)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace wwr::cuda::test {

using namespace wwr::cuda;

// ────────────────────────────────────────────────────────────────────────
// Enum type checks
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_enum_v<cusparseStatus_t>);
static_assert(std::is_enum_v<cusparsePointerMode_t>);
static_assert(std::is_enum_v<cusparseAction_t>);
static_assert(std::is_enum_v<cusparseMatrixType_t>);
static_assert(std::is_enum_v<cusparseFillMode_t>);
static_assert(std::is_enum_v<cusparseDiagType_t>);
static_assert(std::is_enum_v<cusparseIndexBase_t>);
static_assert(std::is_enum_v<cusparseOperation_t>);
static_assert(std::is_enum_v<cusparseDirection_t>);
static_assert(std::is_enum_v<cusparseSolvePolicy_t>);
static_assert(std::is_enum_v<cusparseColorAlg_t>);
static_assert(std::is_enum_v<cusparseCsr2CscAlg_t>);
static_assert(std::is_enum_v<cusparseFormat_t>);
static_assert(std::is_enum_v<cusparseOrder_t>);
static_assert(std::is_enum_v<cusparseIndexType_t>);
static_assert(std::is_enum_v<cusparseSpMatAttribute_t>);
static_assert(std::is_enum_v<cusparseSpMVAlg_t>);
static_assert(std::is_enum_v<cusparseSpSVAlg_t>);
static_assert(std::is_enum_v<cusparseSpSVUpdate_t>);
static_assert(std::is_enum_v<cusparseSpSMAlg_t>);
static_assert(std::is_enum_v<cusparseSpSMUpdate_t>);
static_assert(std::is_enum_v<cusparseSpMMAlg_t>);
static_assert(std::is_enum_v<cusparseSpGEMMAlg_t>);
static_assert(std::is_enum_v<cusparseSDDMMAlg_t>);
static_assert(std::is_enum_v<cusparseSparseToDenseAlg_t>);
static_assert(std::is_enum_v<cusparseDenseToSparseAlg_t>);
static_assert(std::is_enum_v<cusparseSpMMOpAlg_t>);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cusparseStatus_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUSPARSE_STATUS_SUCCESS) == 0);
static_assert(static_cast<int>(CUSPARSE_STATUS_NOT_INITIALIZED) == 1);
static_assert(static_cast<int>(CUSPARSE_STATUS_ALLOC_FAILED) == 2);
static_assert(static_cast<int>(CUSPARSE_STATUS_INVALID_VALUE) == 3);
static_assert(static_cast<int>(CUSPARSE_STATUS_ARCH_MISMATCH) == 4);
static_assert(static_cast<int>(CUSPARSE_STATUS_MAPPING_ERROR) == 5);
static_assert(static_cast<int>(CUSPARSE_STATUS_EXECUTION_FAILED) == 6);
static_assert(static_cast<int>(CUSPARSE_STATUS_INTERNAL_ERROR) == 7);
static_assert(static_cast<int>(CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED) == 8);
static_assert(static_cast<int>(CUSPARSE_STATUS_ZERO_PIVOT) == 9);
static_assert(static_cast<int>(CUSPARSE_STATUS_NOT_SUPPORTED) == 10);
static_assert(static_cast<int>(CUSPARSE_STATUS_INSUFFICIENT_RESOURCES) == 11);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cusparsePointerMode_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUSPARSE_POINTER_MODE_HOST) == 0);
static_assert(static_cast<int>(CUSPARSE_POINTER_MODE_DEVICE) == 1);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cusparseAction_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUSPARSE_ACTION_SYMBOLIC) == 0);
static_assert(static_cast<int>(CUSPARSE_ACTION_NUMERIC) == 1);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cusparseMatrixType_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUSPARSE_MATRIX_TYPE_GENERAL) == 0);
static_assert(static_cast<int>(CUSPARSE_MATRIX_TYPE_SYMMETRIC) == 1);
static_assert(static_cast<int>(CUSPARSE_MATRIX_TYPE_HERMITIAN) == 2);
static_assert(static_cast<int>(CUSPARSE_MATRIX_TYPE_TRIANGULAR) == 3);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cusparseFillMode_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUSPARSE_FILL_MODE_LOWER) == 0);
static_assert(static_cast<int>(CUSPARSE_FILL_MODE_UPPER) == 1);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cusparseDiagType_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUSPARSE_DIAG_TYPE_NON_UNIT) == 0);
static_assert(static_cast<int>(CUSPARSE_DIAG_TYPE_UNIT) == 1);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cusparseIndexBase_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUSPARSE_INDEX_BASE_ZERO) == 0);
static_assert(static_cast<int>(CUSPARSE_INDEX_BASE_ONE) == 1);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cusparseOperation_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUSPARSE_OPERATION_NON_TRANSPOSE) == 0);
static_assert(static_cast<int>(CUSPARSE_OPERATION_TRANSPOSE) == 1);
static_assert(static_cast<int>(CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE) == 2);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cusparseDirection_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUSPARSE_DIRECTION_ROW) == 0);
static_assert(static_cast<int>(CUSPARSE_DIRECTION_COLUMN) == 1);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cusparseCsr2CscAlg_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUSPARSE_CSR2CSC_ALG_DEFAULT) == 1);
static_assert(static_cast<int>(CUSPARSE_CSR2CSC_ALG1) == 1);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cusparseFormat_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUSPARSE_FORMAT_CSR) == 1);
static_assert(static_cast<int>(CUSPARSE_FORMAT_CSC) == 2);
static_assert(static_cast<int>(CUSPARSE_FORMAT_COO) == 3);
static_assert(static_cast<int>(CUSPARSE_FORMAT_BLOCKED_ELL) == 5);
static_assert(static_cast<int>(CUSPARSE_FORMAT_BSR) == 6);
static_assert(static_cast<int>(CUSPARSE_FORMAT_SLICED_ELLPACK) == 7);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cusparseOrder_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUSPARSE_ORDER_COL) == 1);
static_assert(static_cast<int>(CUSPARSE_ORDER_ROW) == 2);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cusparseIndexType_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUSPARSE_INDEX_16U) == 1);
static_assert(static_cast<int>(CUSPARSE_INDEX_32I) == 2);
static_assert(static_cast<int>(CUSPARSE_INDEX_64I) == 3);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cusparseSpMVAlg_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUSPARSE_SPMV_ALG_DEFAULT) == 0);
static_assert(static_cast<int>(CUSPARSE_SPMV_CSR_ALG1) == 2);
static_assert(static_cast<int>(CUSPARSE_SPMV_CSR_ALG2) == 3);
static_assert(static_cast<int>(CUSPARSE_SPMV_COO_ALG1) == 1);
static_assert(static_cast<int>(CUSPARSE_SPMV_COO_ALG2) == 4);
static_assert(static_cast<int>(CUSPARSE_SPMV_SELL_ALG1) == 5);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cusparseSpMMAlg_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUSPARSE_SPMM_ALG_DEFAULT) == 0);
static_assert(static_cast<int>(CUSPARSE_SPMM_COO_ALG1) == 1);
static_assert(static_cast<int>(CUSPARSE_SPMM_COO_ALG2) == 2);
static_assert(static_cast<int>(CUSPARSE_SPMM_COO_ALG3) == 3);
static_assert(static_cast<int>(CUSPARSE_SPMM_COO_ALG4) == 5);
static_assert(static_cast<int>(CUSPARSE_SPMM_CSR_ALG1) == 4);
static_assert(static_cast<int>(CUSPARSE_SPMM_CSR_ALG2) == 6);
static_assert(static_cast<int>(CUSPARSE_SPMM_CSR_ALG3) == 12);
static_assert(static_cast<int>(CUSPARSE_SPMM_BLOCKED_ELL_ALG1) == 13);
static_assert(static_cast<int>(CUSPARSE_SPMM_BSR_ALG1) == 14);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cusparseSpGEMMAlg_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUSPARSE_SPGEMM_DEFAULT) == 0);
static_assert(static_cast<int>(CUSPARSE_SPGEMM_CSR_ALG_DETERMINITIC) == 1);
static_assert(static_cast<int>(CUSPARSE_SPGEMM_CSR_ALG_NONDETERMINITIC) == 2);
static_assert(static_cast<int>(CUSPARSE_SPGEMM_ALG1) == 3);
static_assert(static_cast<int>(CUSPARSE_SPGEMM_ALG2) == 4);
static_assert(static_cast<int>(CUSPARSE_SPGEMM_ALG3) == 5);

// ────────────────────────────────────────────────────────────────────────
// Opaque handle type checks — all must be pointer types
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_pointer_v<cusparseHandle_t>);
static_assert(std::is_pointer_v<cusparseMatDescr_t>);
static_assert(std::is_pointer_v<cusparseSpVecDescr_t>);
static_assert(std::is_pointer_v<cusparseDnVecDescr_t>);
static_assert(std::is_pointer_v<cusparseSpMatDescr_t>);
static_assert(std::is_pointer_v<cusparseDnMatDescr_t>);
static_assert(std::is_pointer_v<cusparseConstSpVecDescr_t>);
static_assert(std::is_pointer_v<cusparseConstDnVecDescr_t>);
static_assert(std::is_pointer_v<cusparseConstSpMatDescr_t>);
static_assert(std::is_pointer_v<cusparseConstDnMatDescr_t>);
static_assert(std::is_pointer_v<cusparseSpSVDescr_t>);
static_assert(std::is_pointer_v<cusparseSpSMDescr_t>);
static_assert(std::is_pointer_v<cusparseSpGEMMDescr_t>);
static_assert(std::is_pointer_v<cusparseSpMMOpPlan_t>);

// ────────────────────────────────────────────────────────────────────────
// Link checks — verify all exported functions resolve at link time
// ────────────────────────────────────────────────────────────────────────

// Context management
WWR_LINK_CHECK(cusparseCreate)
WWR_LINK_CHECK(cusparseDestroy)
WWR_LINK_CHECK(cusparseGetVersion)
WWR_LINK_CHECK(cusparseGetProperty)
WWR_LINK_CHECK(cusparseGetErrorName)
WWR_LINK_CHECK(cusparseGetErrorString)
WWR_LINK_CHECK(cusparseSetStream)
WWR_LINK_CHECK(cusparseGetStream)
WWR_LINK_CHECK(cusparseGetPointerMode)
WWR_LINK_CHECK(cusparseSetPointerMode)

// Logging
WWR_LINK_CHECK(cusparseLoggerSetCallback)
WWR_LINK_CHECK(cusparseLoggerSetFile)
WWR_LINK_CHECK(cusparseLoggerOpenFile)
WWR_LINK_CHECK(cusparseLoggerSetLevel)
WWR_LINK_CHECK(cusparseLoggerSetMask)
WWR_LINK_CHECK(cusparseLoggerForceDisable)

// Matrix descriptor helpers
WWR_LINK_CHECK(cusparseCreateMatDescr)
WWR_LINK_CHECK(cusparseDestroyMatDescr)
WWR_LINK_CHECK(cusparseSetMatType)
WWR_LINK_CHECK(cusparseGetMatType)
WWR_LINK_CHECK(cusparseSetMatFillMode)
WWR_LINK_CHECK(cusparseGetMatFillMode)
WWR_LINK_CHECK(cusparseSetMatDiagType)
WWR_LINK_CHECK(cusparseGetMatDiagType)
WWR_LINK_CHECK(cusparseSetMatIndexBase)
WWR_LINK_CHECK(cusparseGetMatIndexBase)

// BSR SpMV
WWR_LINK_CHECK(cusparseSbsrmv)
WWR_LINK_CHECK(cusparseDbsrmv)
WWR_LINK_CHECK(cusparseCbsrmv)
WWR_LINK_CHECK(cusparseZbsrmv)

// Tridiagonal solvers
WWR_LINK_CHECK(cusparseSgtsv2_bufferSizeExt)
WWR_LINK_CHECK(cusparseDgtsv2_bufferSizeExt)
WWR_LINK_CHECK(cusparseCgtsv2_bufferSizeExt)
WWR_LINK_CHECK(cusparseZgtsv2_bufferSizeExt)
WWR_LINK_CHECK(cusparseSgtsv2)
WWR_LINK_CHECK(cusparseDgtsv2)
WWR_LINK_CHECK(cusparseCgtsv2)
WWR_LINK_CHECK(cusparseZgtsv2)
WWR_LINK_CHECK(cusparseSgtsv2_nopivot_bufferSizeExt)
WWR_LINK_CHECK(cusparseDgtsv2_nopivot_bufferSizeExt)
WWR_LINK_CHECK(cusparseCgtsv2_nopivot_bufferSizeExt)
WWR_LINK_CHECK(cusparseZgtsv2_nopivot_bufferSizeExt)
WWR_LINK_CHECK(cusparseSgtsv2_nopivot)
WWR_LINK_CHECK(cusparseDgtsv2_nopivot)
WWR_LINK_CHECK(cusparseCgtsv2_nopivot)
WWR_LINK_CHECK(cusparseZgtsv2_nopivot)
WWR_LINK_CHECK(cusparseSgtsv2StridedBatch_bufferSizeExt)
WWR_LINK_CHECK(cusparseDgtsv2StridedBatch_bufferSizeExt)
WWR_LINK_CHECK(cusparseCgtsv2StridedBatch_bufferSizeExt)
WWR_LINK_CHECK(cusparseZgtsv2StridedBatch_bufferSizeExt)
WWR_LINK_CHECK(cusparseSgtsv2StridedBatch)
WWR_LINK_CHECK(cusparseDgtsv2StridedBatch)
WWR_LINK_CHECK(cusparseCgtsv2StridedBatch)
WWR_LINK_CHECK(cusparseZgtsv2StridedBatch)
WWR_LINK_CHECK(cusparseSgtsvInterleavedBatch_bufferSizeExt)
WWR_LINK_CHECK(cusparseDgtsvInterleavedBatch_bufferSizeExt)
WWR_LINK_CHECK(cusparseCgtsvInterleavedBatch_bufferSizeExt)
WWR_LINK_CHECK(cusparseZgtsvInterleavedBatch_bufferSizeExt)
WWR_LINK_CHECK(cusparseSgtsvInterleavedBatch)
WWR_LINK_CHECK(cusparseDgtsvInterleavedBatch)
WWR_LINK_CHECK(cusparseCgtsvInterleavedBatch)
WWR_LINK_CHECK(cusparseZgtsvInterleavedBatch)
WWR_LINK_CHECK(cusparseSgpsvInterleavedBatch_bufferSizeExt)
WWR_LINK_CHECK(cusparseDgpsvInterleavedBatch_bufferSizeExt)
WWR_LINK_CHECK(cusparseCgpsvInterleavedBatch_bufferSizeExt)
WWR_LINK_CHECK(cusparseZgpsvInterleavedBatch_bufferSizeExt)
WWR_LINK_CHECK(cusparseSgpsvInterleavedBatch)
WWR_LINK_CHECK(cusparseDgpsvInterleavedBatch)
WWR_LINK_CHECK(cusparseCgpsvInterleavedBatch)
WWR_LINK_CHECK(cusparseZgpsvInterleavedBatch)

// CSR matrix addition
WWR_LINK_CHECK(cusparseScsrgeam2_bufferSizeExt)
WWR_LINK_CHECK(cusparseDcsrgeam2_bufferSizeExt)
WWR_LINK_CHECK(cusparseCcsrgeam2_bufferSizeExt)
WWR_LINK_CHECK(cusparseZcsrgeam2_bufferSizeExt)
WWR_LINK_CHECK(cusparseXcsrgeam2Nnz)
WWR_LINK_CHECK(cusparseScsrgeam2)
WWR_LINK_CHECK(cusparseDcsrgeam2)
WWR_LINK_CHECK(cusparseCcsrgeam2)
WWR_LINK_CHECK(cusparseZcsrgeam2)

// Sparse format conversion
WWR_LINK_CHECK(cusparseSnnz)
WWR_LINK_CHECK(cusparseDnnz)
WWR_LINK_CHECK(cusparseCnnz)
WWR_LINK_CHECK(cusparseZnnz)
WWR_LINK_CHECK(cusparseXcoo2csr)
WWR_LINK_CHECK(cusparseXcsr2coo)
// The *_bufferSizeExt variants of gebsr2gebsc and csr2gebsr are declared in
// cusparse.h but not exported by libcusparse.so (12.6.2, CUDA 13.0): it defines
// the plain *_bufferSize of both, and the *_bufferSizeExt of 43 other routines.
// Calling one fails to link, so only the declaration is checked. Recorded as
// `sparse` omissions in devtools/coverage_decisions.json (and in
// cusparse.json's declared_not_linkable).
WWR_DECLARED_CHECK(cusparseSgebsr2gebsc_bufferSizeExt)
WWR_DECLARED_CHECK(cusparseDgebsr2gebsc_bufferSizeExt)
WWR_DECLARED_CHECK(cusparseCgebsr2gebsc_bufferSizeExt)
WWR_DECLARED_CHECK(cusparseZgebsr2gebsc_bufferSizeExt)
WWR_LINK_CHECK(cusparseSgebsr2gebsc)
WWR_LINK_CHECK(cusparseDgebsr2gebsc)
WWR_LINK_CHECK(cusparseCgebsr2gebsc)
WWR_LINK_CHECK(cusparseZgebsr2gebsc)
WWR_DECLARED_CHECK(cusparseScsr2gebsr_bufferSizeExt)
WWR_DECLARED_CHECK(cusparseDcsr2gebsr_bufferSizeExt)
WWR_DECLARED_CHECK(cusparseCcsr2gebsr_bufferSizeExt)
WWR_DECLARED_CHECK(cusparseZcsr2gebsr_bufferSizeExt)
WWR_LINK_CHECK(cusparseXcsr2gebsrNnz)
WWR_LINK_CHECK(cusparseScsr2gebsr)
WWR_LINK_CHECK(cusparseDcsr2gebsr)
WWR_LINK_CHECK(cusparseCcsr2gebsr)
WWR_LINK_CHECK(cusparseZcsr2gebsr)

// Sorting
WWR_LINK_CHECK(cusparseXcoosort_bufferSizeExt)
WWR_LINK_CHECK(cusparseXcoosortByRow)
WWR_LINK_CHECK(cusparseXcoosortByColumn)
WWR_LINK_CHECK(cusparseXcsrsort_bufferSizeExt)
WWR_LINK_CHECK(cusparseXcsrsort)
WWR_LINK_CHECK(cusparseXcscsort_bufferSizeExt)
WWR_LINK_CHECK(cusparseXcscsort)

// CSR to CSC
WWR_LINK_CHECK(cusparseCsr2cscEx2)
WWR_LINK_CHECK(cusparseCsr2cscEx2_bufferSize)

// Sparse vector descriptor
WWR_LINK_CHECK(cusparseCreateSpVec)
WWR_LINK_CHECK(cusparseCreateConstSpVec)
WWR_LINK_CHECK(cusparseDestroySpVec)
WWR_LINK_CHECK(cusparseSpVecGet)
WWR_LINK_CHECK(cusparseConstSpVecGet)
WWR_LINK_CHECK(cusparseSpVecGetIndexBase)
WWR_LINK_CHECK(cusparseSpVecGetValues)
WWR_LINK_CHECK(cusparseConstSpVecGetValues)
WWR_LINK_CHECK(cusparseSpVecSetValues)

// Dense vector descriptor
WWR_LINK_CHECK(cusparseCreateDnVec)
WWR_LINK_CHECK(cusparseCreateConstDnVec)
WWR_LINK_CHECK(cusparseDestroyDnVec)
WWR_LINK_CHECK(cusparseDnVecGet)
WWR_LINK_CHECK(cusparseConstDnVecGet)
WWR_LINK_CHECK(cusparseDnVecGetValues)
WWR_LINK_CHECK(cusparseConstDnVecGetValues)
WWR_LINK_CHECK(cusparseDnVecSetValues)

// Sparse matrix descriptor
WWR_LINK_CHECK(cusparseDestroySpMat)
WWR_LINK_CHECK(cusparseSpMatGetFormat)
WWR_LINK_CHECK(cusparseSpMatGetIndexBase)
WWR_LINK_CHECK(cusparseSpMatGetValues)
WWR_LINK_CHECK(cusparseConstSpMatGetValues)
WWR_LINK_CHECK(cusparseSpMatSetValues)
WWR_LINK_CHECK(cusparseSpMatGetSize)
WWR_LINK_CHECK(cusparseSpMatGetStridedBatch)
WWR_LINK_CHECK(cusparseCooSetStridedBatch)
WWR_LINK_CHECK(cusparseCsrSetStridedBatch)
WWR_LINK_CHECK(cusparseBsrSetStridedBatch)
WWR_LINK_CHECK(cusparseSpMatGetAttribute)
WWR_LINK_CHECK(cusparseSpMatSetAttribute)
WWR_LINK_CHECK(cusparseCreateCsr)
WWR_LINK_CHECK(cusparseCreateConstCsr)
WWR_LINK_CHECK(cusparseCsrGet)
WWR_LINK_CHECK(cusparseConstCsrGet)
WWR_LINK_CHECK(cusparseCsrSetPointers)
WWR_LINK_CHECK(cusparseCreateCsc)
WWR_LINK_CHECK(cusparseCreateConstCsc)
WWR_LINK_CHECK(cusparseCscGet)
WWR_LINK_CHECK(cusparseConstCscGet)
WWR_LINK_CHECK(cusparseCscSetPointers)
WWR_LINK_CHECK(cusparseCreateBsr)
WWR_LINK_CHECK(cusparseCreateConstBsr)
WWR_LINK_CHECK(cusparseCreateCoo)
WWR_LINK_CHECK(cusparseCreateConstCoo)
WWR_LINK_CHECK(cusparseCooGet)
WWR_LINK_CHECK(cusparseConstCooGet)
WWR_LINK_CHECK(cusparseCooSetPointers)
WWR_LINK_CHECK(cusparseCreateBlockedEll)
WWR_LINK_CHECK(cusparseCreateConstBlockedEll)
WWR_LINK_CHECK(cusparseBlockedEllGet)
WWR_LINK_CHECK(cusparseConstBlockedEllGet)
WWR_LINK_CHECK(cusparseCreateSlicedEll)
WWR_LINK_CHECK(cusparseCreateConstSlicedEll)

// Dense matrix descriptor
WWR_LINK_CHECK(cusparseCreateDnMat)
WWR_LINK_CHECK(cusparseCreateConstDnMat)
WWR_LINK_CHECK(cusparseDestroyDnMat)
WWR_LINK_CHECK(cusparseDnMatGet)
WWR_LINK_CHECK(cusparseConstDnMatGet)
WWR_LINK_CHECK(cusparseDnMatGetValues)
WWR_LINK_CHECK(cusparseConstDnMatGetValues)
WWR_LINK_CHECK(cusparseDnMatSetValues)
WWR_LINK_CHECK(cusparseDnMatSetStridedBatch)
WWR_LINK_CHECK(cusparseDnMatGetStridedBatch)

// Vector-vector operations
WWR_LINK_CHECK(cusparseGather)
WWR_LINK_CHECK(cusparseScatter)

// Sparse to dense / dense to sparse
WWR_LINK_CHECK(cusparseSparseToDense_bufferSize)
WWR_LINK_CHECK(cusparseSparseToDense)
WWR_LINK_CHECK(cusparseDenseToSparse_bufferSize)
WWR_LINK_CHECK(cusparseDenseToSparse_analysis)
WWR_LINK_CHECK(cusparseDenseToSparse_convert)

// SpMV
WWR_LINK_CHECK(cusparseSpMV)
WWR_LINK_CHECK(cusparseSpMV_bufferSize)
WWR_LINK_CHECK(cusparseSpMV_preprocess)

// SpSV
WWR_LINK_CHECK(cusparseSpSV_createDescr)
WWR_LINK_CHECK(cusparseSpSV_destroyDescr)
WWR_LINK_CHECK(cusparseSpSV_bufferSize)
WWR_LINK_CHECK(cusparseSpSV_analysis)
WWR_LINK_CHECK(cusparseSpSV_solve)
WWR_LINK_CHECK(cusparseSpSV_updateMatrix)

// SpSM
WWR_LINK_CHECK(cusparseSpSM_createDescr)
WWR_LINK_CHECK(cusparseSpSM_destroyDescr)
WWR_LINK_CHECK(cusparseSpSM_bufferSize)
WWR_LINK_CHECK(cusparseSpSM_analysis)
WWR_LINK_CHECK(cusparseSpSM_solve)
WWR_LINK_CHECK(cusparseSpSM_updateMatrix)

// SpMM
WWR_LINK_CHECK(cusparseSpMM_bufferSize)
WWR_LINK_CHECK(cusparseSpMM_preprocess)
WWR_LINK_CHECK(cusparseSpMM)

// SpGEMM
WWR_LINK_CHECK(cusparseSpGEMM_createDescr)
WWR_LINK_CHECK(cusparseSpGEMM_destroyDescr)
WWR_LINK_CHECK(cusparseSpGEMM_workEstimation)
WWR_LINK_CHECK(cusparseSpGEMM_getNumProducts)
WWR_LINK_CHECK(cusparseSpGEMM_estimateMemory)
WWR_LINK_CHECK(cusparseSpGEMM_compute)
WWR_LINK_CHECK(cusparseSpGEMM_copy)
WWR_LINK_CHECK(cusparseSpGEMMreuse_workEstimation)
WWR_LINK_CHECK(cusparseSpGEMMreuse_nnz)
WWR_LINK_CHECK(cusparseSpGEMMreuse_copy)
WWR_LINK_CHECK(cusparseSpGEMMreuse_compute)

// SDDMM
WWR_LINK_CHECK(cusparseSDDMM_bufferSize)
WWR_LINK_CHECK(cusparseSDDMM_preprocess)
WWR_LINK_CHECK(cusparseSDDMM)

// SpMM with custom operators
WWR_LINK_CHECK(cusparseSpMMOp_createPlan)
WWR_LINK_CHECK(cusparseSpMMOp)
WWR_LINK_CHECK(cusparseSpMMOp_destroyPlan)


// ────────────────────────────────────────────────────────────────────────
// Additional link-time coverage (#117 completeness): re-exported functions
// not previously link-checked.
// ────────────────────────────────────────────────────────────────────────

WWR_LINK_CHECK(cusparseCreateBsric02Info)
WWR_LINK_CHECK(cusparseCreateBsrilu02Info)
WWR_LINK_CHECK(cusparseCreateBsrsm2Info)
WWR_LINK_CHECK(cusparseCreateBsrsv2Info)
WWR_LINK_CHECK(cusparseCreateColorInfo)
WWR_LINK_CHECK(cusparseCreateCsric02Info)
WWR_LINK_CHECK(cusparseCreateCsrilu02Info)
WWR_LINK_CHECK(cusparseCreateCsru2csrInfo)
WWR_LINK_CHECK(cusparseCreatePruneInfo)
WWR_LINK_CHECK(cusparseDestroyBsric02Info)
WWR_LINK_CHECK(cusparseDestroyBsrilu02Info)
WWR_LINK_CHECK(cusparseDestroyBsrsm2Info)
WWR_LINK_CHECK(cusparseDestroyBsrsv2Info)
WWR_LINK_CHECK(cusparseDestroyColorInfo)
WWR_LINK_CHECK(cusparseDestroyCsric02Info)
WWR_LINK_CHECK(cusparseDestroyCsrilu02Info)
WWR_LINK_CHECK(cusparseDestroyCsru2csrInfo)
WWR_LINK_CHECK(cusparseDestroyPruneInfo)
WWR_LINK_CHECK(cusparseCcsr2gebsr_bufferSize)
WWR_LINK_CHECK(cusparseCgebsr2gebsc_bufferSize)
WWR_LINK_CHECK(cusparseDcsr2gebsr_bufferSize)
WWR_LINK_CHECK(cusparseDgebsr2gebsc_bufferSize)
WWR_LINK_CHECK(cusparseScsr2gebsr_bufferSize)
WWR_LINK_CHECK(cusparseSgebsr2gebsc_bufferSize)
WWR_LINK_CHECK(cusparseZcsr2gebsr_bufferSize)
WWR_LINK_CHECK(cusparseZgebsr2gebsc_bufferSize)

} // namespace wwr::cuda::test
