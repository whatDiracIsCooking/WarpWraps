// cusparse.cppm - Compile-time tests for gpumod.cuda.cusparse

module;

#include "test/shared/link_check.h"

export module gpumod.test.cuda.cusparse;

import std;
import gpumod.cuda.cusparse;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time tests for gpumod.cuda.cusparse
//
// The module is a pure re-export (using declarations).
// We verify at compile-time that:
//   1. Key enum types are actually enum types (std::is_enum_v<>)
//   2. Key enumerator values with cuSPARSE-specified values are correct
//   3. Opaque handle types are pointer types (std::is_pointer_v<>)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace gpumod::cuda::test {

using namespace gpumod::cuda;

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
GPUMOD_LINK_CHECK(cusparseCreate)
GPUMOD_LINK_CHECK(cusparseDestroy)
GPUMOD_LINK_CHECK(cusparseGetVersion)
GPUMOD_LINK_CHECK(cusparseGetProperty)
GPUMOD_LINK_CHECK(cusparseGetErrorName)
GPUMOD_LINK_CHECK(cusparseGetErrorString)
GPUMOD_LINK_CHECK(cusparseSetStream)
GPUMOD_LINK_CHECK(cusparseGetStream)
GPUMOD_LINK_CHECK(cusparseGetPointerMode)
GPUMOD_LINK_CHECK(cusparseSetPointerMode)

// Logging
GPUMOD_LINK_CHECK(cusparseLoggerSetCallback)
GPUMOD_LINK_CHECK(cusparseLoggerSetFile)
GPUMOD_LINK_CHECK(cusparseLoggerOpenFile)
GPUMOD_LINK_CHECK(cusparseLoggerSetLevel)
GPUMOD_LINK_CHECK(cusparseLoggerSetMask)
GPUMOD_LINK_CHECK(cusparseLoggerForceDisable)

// Matrix descriptor helpers
GPUMOD_LINK_CHECK(cusparseCreateMatDescr)
GPUMOD_LINK_CHECK(cusparseDestroyMatDescr)
GPUMOD_LINK_CHECK(cusparseSetMatType)
GPUMOD_LINK_CHECK(cusparseGetMatType)
GPUMOD_LINK_CHECK(cusparseSetMatFillMode)
GPUMOD_LINK_CHECK(cusparseGetMatFillMode)
GPUMOD_LINK_CHECK(cusparseSetMatDiagType)
GPUMOD_LINK_CHECK(cusparseGetMatDiagType)
GPUMOD_LINK_CHECK(cusparseSetMatIndexBase)
GPUMOD_LINK_CHECK(cusparseGetMatIndexBase)

// BSR SpMV
GPUMOD_LINK_CHECK(cusparseSbsrmv)
GPUMOD_LINK_CHECK(cusparseDbsrmv)
GPUMOD_LINK_CHECK(cusparseCbsrmv)
GPUMOD_LINK_CHECK(cusparseZbsrmv)

// Tridiagonal solvers
GPUMOD_LINK_CHECK(cusparseSgtsv2_bufferSizeExt)
GPUMOD_LINK_CHECK(cusparseDgtsv2_bufferSizeExt)
GPUMOD_LINK_CHECK(cusparseCgtsv2_bufferSizeExt)
GPUMOD_LINK_CHECK(cusparseZgtsv2_bufferSizeExt)
GPUMOD_LINK_CHECK(cusparseSgtsv2)
GPUMOD_LINK_CHECK(cusparseDgtsv2)
GPUMOD_LINK_CHECK(cusparseCgtsv2)
GPUMOD_LINK_CHECK(cusparseZgtsv2)
GPUMOD_LINK_CHECK(cusparseSgtsv2_nopivot_bufferSizeExt)
GPUMOD_LINK_CHECK(cusparseDgtsv2_nopivot_bufferSizeExt)
GPUMOD_LINK_CHECK(cusparseCgtsv2_nopivot_bufferSizeExt)
GPUMOD_LINK_CHECK(cusparseZgtsv2_nopivot_bufferSizeExt)
GPUMOD_LINK_CHECK(cusparseSgtsv2_nopivot)
GPUMOD_LINK_CHECK(cusparseDgtsv2_nopivot)
GPUMOD_LINK_CHECK(cusparseCgtsv2_nopivot)
GPUMOD_LINK_CHECK(cusparseZgtsv2_nopivot)
GPUMOD_LINK_CHECK(cusparseSgtsv2StridedBatch_bufferSizeExt)
GPUMOD_LINK_CHECK(cusparseDgtsv2StridedBatch_bufferSizeExt)
GPUMOD_LINK_CHECK(cusparseCgtsv2StridedBatch_bufferSizeExt)
GPUMOD_LINK_CHECK(cusparseZgtsv2StridedBatch_bufferSizeExt)
GPUMOD_LINK_CHECK(cusparseSgtsv2StridedBatch)
GPUMOD_LINK_CHECK(cusparseDgtsv2StridedBatch)
GPUMOD_LINK_CHECK(cusparseCgtsv2StridedBatch)
GPUMOD_LINK_CHECK(cusparseZgtsv2StridedBatch)
GPUMOD_LINK_CHECK(cusparseSgtsvInterleavedBatch_bufferSizeExt)
GPUMOD_LINK_CHECK(cusparseDgtsvInterleavedBatch_bufferSizeExt)
GPUMOD_LINK_CHECK(cusparseCgtsvInterleavedBatch_bufferSizeExt)
GPUMOD_LINK_CHECK(cusparseZgtsvInterleavedBatch_bufferSizeExt)
GPUMOD_LINK_CHECK(cusparseSgtsvInterleavedBatch)
GPUMOD_LINK_CHECK(cusparseDgtsvInterleavedBatch)
GPUMOD_LINK_CHECK(cusparseCgtsvInterleavedBatch)
GPUMOD_LINK_CHECK(cusparseZgtsvInterleavedBatch)
GPUMOD_LINK_CHECK(cusparseSgpsvInterleavedBatch_bufferSizeExt)
GPUMOD_LINK_CHECK(cusparseDgpsvInterleavedBatch_bufferSizeExt)
GPUMOD_LINK_CHECK(cusparseCgpsvInterleavedBatch_bufferSizeExt)
GPUMOD_LINK_CHECK(cusparseZgpsvInterleavedBatch_bufferSizeExt)
GPUMOD_LINK_CHECK(cusparseSgpsvInterleavedBatch)
GPUMOD_LINK_CHECK(cusparseDgpsvInterleavedBatch)
GPUMOD_LINK_CHECK(cusparseCgpsvInterleavedBatch)
GPUMOD_LINK_CHECK(cusparseZgpsvInterleavedBatch)

// CSR matrix addition
GPUMOD_LINK_CHECK(cusparseScsrgeam2_bufferSizeExt)
GPUMOD_LINK_CHECK(cusparseDcsrgeam2_bufferSizeExt)
GPUMOD_LINK_CHECK(cusparseCcsrgeam2_bufferSizeExt)
GPUMOD_LINK_CHECK(cusparseZcsrgeam2_bufferSizeExt)
GPUMOD_LINK_CHECK(cusparseXcsrgeam2Nnz)
GPUMOD_LINK_CHECK(cusparseScsrgeam2)
GPUMOD_LINK_CHECK(cusparseDcsrgeam2)
GPUMOD_LINK_CHECK(cusparseCcsrgeam2)
GPUMOD_LINK_CHECK(cusparseZcsrgeam2)

// Sparse format conversion
GPUMOD_LINK_CHECK(cusparseSnnz)
GPUMOD_LINK_CHECK(cusparseDnnz)
GPUMOD_LINK_CHECK(cusparseCnnz)
GPUMOD_LINK_CHECK(cusparseZnnz)
GPUMOD_LINK_CHECK(cusparseXcoo2csr)
GPUMOD_LINK_CHECK(cusparseXcsr2coo)
// The *_bufferSizeExt variants of gebsr2gebsc and csr2gebsr are declared in
// cusparse.h but not exported by libcusparse.so (12.6.2, CUDA 13.0): it defines
// the plain *_bufferSize of both, and the *_bufferSizeExt of 43 other routines.
// Calling one fails to link, so only the declaration is checked.
GPUMOD_DECLARED_CHECK(cusparseSgebsr2gebsc_bufferSizeExt)
GPUMOD_DECLARED_CHECK(cusparseDgebsr2gebsc_bufferSizeExt)
GPUMOD_DECLARED_CHECK(cusparseCgebsr2gebsc_bufferSizeExt)
GPUMOD_DECLARED_CHECK(cusparseZgebsr2gebsc_bufferSizeExt)
GPUMOD_LINK_CHECK(cusparseSgebsr2gebsc)
GPUMOD_LINK_CHECK(cusparseDgebsr2gebsc)
GPUMOD_LINK_CHECK(cusparseCgebsr2gebsc)
GPUMOD_LINK_CHECK(cusparseZgebsr2gebsc)
GPUMOD_DECLARED_CHECK(cusparseScsr2gebsr_bufferSizeExt)
GPUMOD_DECLARED_CHECK(cusparseDcsr2gebsr_bufferSizeExt)
GPUMOD_DECLARED_CHECK(cusparseCcsr2gebsr_bufferSizeExt)
GPUMOD_DECLARED_CHECK(cusparseZcsr2gebsr_bufferSizeExt)
GPUMOD_LINK_CHECK(cusparseXcsr2gebsrNnz)
GPUMOD_LINK_CHECK(cusparseScsr2gebsr)
GPUMOD_LINK_CHECK(cusparseDcsr2gebsr)
GPUMOD_LINK_CHECK(cusparseCcsr2gebsr)
GPUMOD_LINK_CHECK(cusparseZcsr2gebsr)

// Sorting
GPUMOD_LINK_CHECK(cusparseXcoosort_bufferSizeExt)
GPUMOD_LINK_CHECK(cusparseXcoosortByRow)
GPUMOD_LINK_CHECK(cusparseXcoosortByColumn)
GPUMOD_LINK_CHECK(cusparseXcsrsort_bufferSizeExt)
GPUMOD_LINK_CHECK(cusparseXcsrsort)
GPUMOD_LINK_CHECK(cusparseXcscsort_bufferSizeExt)
GPUMOD_LINK_CHECK(cusparseXcscsort)

// CSR to CSC
GPUMOD_LINK_CHECK(cusparseCsr2cscEx2)
GPUMOD_LINK_CHECK(cusparseCsr2cscEx2_bufferSize)

// Sparse vector descriptor
GPUMOD_LINK_CHECK(cusparseCreateSpVec)
GPUMOD_LINK_CHECK(cusparseCreateConstSpVec)
GPUMOD_LINK_CHECK(cusparseDestroySpVec)
GPUMOD_LINK_CHECK(cusparseSpVecGet)
GPUMOD_LINK_CHECK(cusparseConstSpVecGet)
GPUMOD_LINK_CHECK(cusparseSpVecGetIndexBase)
GPUMOD_LINK_CHECK(cusparseSpVecGetValues)
GPUMOD_LINK_CHECK(cusparseConstSpVecGetValues)
GPUMOD_LINK_CHECK(cusparseSpVecSetValues)

// Dense vector descriptor
GPUMOD_LINK_CHECK(cusparseCreateDnVec)
GPUMOD_LINK_CHECK(cusparseCreateConstDnVec)
GPUMOD_LINK_CHECK(cusparseDestroyDnVec)
GPUMOD_LINK_CHECK(cusparseDnVecGet)
GPUMOD_LINK_CHECK(cusparseConstDnVecGet)
GPUMOD_LINK_CHECK(cusparseDnVecGetValues)
GPUMOD_LINK_CHECK(cusparseConstDnVecGetValues)
GPUMOD_LINK_CHECK(cusparseDnVecSetValues)

// Sparse matrix descriptor
GPUMOD_LINK_CHECK(cusparseDestroySpMat)
GPUMOD_LINK_CHECK(cusparseSpMatGetFormat)
GPUMOD_LINK_CHECK(cusparseSpMatGetIndexBase)
GPUMOD_LINK_CHECK(cusparseSpMatGetValues)
GPUMOD_LINK_CHECK(cusparseConstSpMatGetValues)
GPUMOD_LINK_CHECK(cusparseSpMatSetValues)
GPUMOD_LINK_CHECK(cusparseSpMatGetSize)
GPUMOD_LINK_CHECK(cusparseSpMatGetStridedBatch)
GPUMOD_LINK_CHECK(cusparseCooSetStridedBatch)
GPUMOD_LINK_CHECK(cusparseCsrSetStridedBatch)
GPUMOD_LINK_CHECK(cusparseBsrSetStridedBatch)
GPUMOD_LINK_CHECK(cusparseSpMatGetAttribute)
GPUMOD_LINK_CHECK(cusparseSpMatSetAttribute)
GPUMOD_LINK_CHECK(cusparseCreateCsr)
GPUMOD_LINK_CHECK(cusparseCreateConstCsr)
GPUMOD_LINK_CHECK(cusparseCsrGet)
GPUMOD_LINK_CHECK(cusparseConstCsrGet)
GPUMOD_LINK_CHECK(cusparseCsrSetPointers)
GPUMOD_LINK_CHECK(cusparseCreateCsc)
GPUMOD_LINK_CHECK(cusparseCreateConstCsc)
GPUMOD_LINK_CHECK(cusparseCscGet)
GPUMOD_LINK_CHECK(cusparseConstCscGet)
GPUMOD_LINK_CHECK(cusparseCscSetPointers)
GPUMOD_LINK_CHECK(cusparseCreateBsr)
GPUMOD_LINK_CHECK(cusparseCreateConstBsr)
GPUMOD_LINK_CHECK(cusparseCreateCoo)
GPUMOD_LINK_CHECK(cusparseCreateConstCoo)
GPUMOD_LINK_CHECK(cusparseCooGet)
GPUMOD_LINK_CHECK(cusparseConstCooGet)
GPUMOD_LINK_CHECK(cusparseCooSetPointers)
GPUMOD_LINK_CHECK(cusparseCreateBlockedEll)
GPUMOD_LINK_CHECK(cusparseCreateConstBlockedEll)
GPUMOD_LINK_CHECK(cusparseBlockedEllGet)
GPUMOD_LINK_CHECK(cusparseConstBlockedEllGet)
GPUMOD_LINK_CHECK(cusparseCreateSlicedEll)
GPUMOD_LINK_CHECK(cusparseCreateConstSlicedEll)

// Dense matrix descriptor
GPUMOD_LINK_CHECK(cusparseCreateDnMat)
GPUMOD_LINK_CHECK(cusparseCreateConstDnMat)
GPUMOD_LINK_CHECK(cusparseDestroyDnMat)
GPUMOD_LINK_CHECK(cusparseDnMatGet)
GPUMOD_LINK_CHECK(cusparseConstDnMatGet)
GPUMOD_LINK_CHECK(cusparseDnMatGetValues)
GPUMOD_LINK_CHECK(cusparseConstDnMatGetValues)
GPUMOD_LINK_CHECK(cusparseDnMatSetValues)
GPUMOD_LINK_CHECK(cusparseDnMatSetStridedBatch)
GPUMOD_LINK_CHECK(cusparseDnMatGetStridedBatch)

// Vector-vector operations
GPUMOD_LINK_CHECK(cusparseGather)
GPUMOD_LINK_CHECK(cusparseScatter)

// Sparse to dense / dense to sparse
GPUMOD_LINK_CHECK(cusparseSparseToDense_bufferSize)
GPUMOD_LINK_CHECK(cusparseSparseToDense)
GPUMOD_LINK_CHECK(cusparseDenseToSparse_bufferSize)
GPUMOD_LINK_CHECK(cusparseDenseToSparse_analysis)
GPUMOD_LINK_CHECK(cusparseDenseToSparse_convert)

// SpMV
GPUMOD_LINK_CHECK(cusparseSpMV)
GPUMOD_LINK_CHECK(cusparseSpMV_bufferSize)
GPUMOD_LINK_CHECK(cusparseSpMV_preprocess)

// SpSV
GPUMOD_LINK_CHECK(cusparseSpSV_createDescr)
GPUMOD_LINK_CHECK(cusparseSpSV_destroyDescr)
GPUMOD_LINK_CHECK(cusparseSpSV_bufferSize)
GPUMOD_LINK_CHECK(cusparseSpSV_analysis)
GPUMOD_LINK_CHECK(cusparseSpSV_solve)
GPUMOD_LINK_CHECK(cusparseSpSV_updateMatrix)

// SpSM
GPUMOD_LINK_CHECK(cusparseSpSM_createDescr)
GPUMOD_LINK_CHECK(cusparseSpSM_destroyDescr)
GPUMOD_LINK_CHECK(cusparseSpSM_bufferSize)
GPUMOD_LINK_CHECK(cusparseSpSM_analysis)
GPUMOD_LINK_CHECK(cusparseSpSM_solve)
GPUMOD_LINK_CHECK(cusparseSpSM_updateMatrix)

// SpMM
GPUMOD_LINK_CHECK(cusparseSpMM_bufferSize)
GPUMOD_LINK_CHECK(cusparseSpMM_preprocess)
GPUMOD_LINK_CHECK(cusparseSpMM)

// SpGEMM
GPUMOD_LINK_CHECK(cusparseSpGEMM_createDescr)
GPUMOD_LINK_CHECK(cusparseSpGEMM_destroyDescr)
GPUMOD_LINK_CHECK(cusparseSpGEMM_workEstimation)
GPUMOD_LINK_CHECK(cusparseSpGEMM_getNumProducts)
GPUMOD_LINK_CHECK(cusparseSpGEMM_estimateMemory)
GPUMOD_LINK_CHECK(cusparseSpGEMM_compute)
GPUMOD_LINK_CHECK(cusparseSpGEMM_copy)
GPUMOD_LINK_CHECK(cusparseSpGEMMreuse_workEstimation)
GPUMOD_LINK_CHECK(cusparseSpGEMMreuse_nnz)
GPUMOD_LINK_CHECK(cusparseSpGEMMreuse_copy)
GPUMOD_LINK_CHECK(cusparseSpGEMMreuse_compute)

// SDDMM
GPUMOD_LINK_CHECK(cusparseSDDMM_bufferSize)
GPUMOD_LINK_CHECK(cusparseSDDMM_preprocess)
GPUMOD_LINK_CHECK(cusparseSDDMM)

// SpMM with custom operators
GPUMOD_LINK_CHECK(cusparseSpMMOp_createPlan)
GPUMOD_LINK_CHECK(cusparseSpMMOp)
GPUMOD_LINK_CHECK(cusparseSpMMOp_destroyPlan)

} // namespace gpumod::cuda::test
