// hipsparse.cppm - Compile-time tests for wwr.hip.hipsparse

module;

#include "test/shared/link_check.h"

export module wwr.test.hip.hipsparse;

import std;
import wwr.hip.hipsparse;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time tests for wwr.hip.hipsparse
//
// WWR_LINK_CHECK covers all 546 HIPSPARSE_EXPORT functions declared across
// hipsparse.h and its internal/{level1,level2,level3,extra,precond,
// conversion,reorder,generic}/ headers -- extracted directly, not
// hand-copied.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace wwr::hip::test {

using namespace wwr::hip;

// ────────────────────────────────────────────────────────────────────────
// Enum type checks
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_enum_v<hipsparseStatus_t>);
static_assert(std::is_enum_v<hipsparsePointerMode_t>);
static_assert(std::is_enum_v<hipsparseAction_t>);
static_assert(std::is_enum_v<hipsparseMatrixType_t>);
static_assert(std::is_enum_v<hipsparseFillMode_t>);
static_assert(std::is_enum_v<hipsparseDiagType_t>);
static_assert(std::is_enum_v<hipsparseIndexBase_t>);
static_assert(std::is_enum_v<hipsparseOperation_t>);
static_assert(std::is_enum_v<hipsparseHybPartition_t>);
static_assert(std::is_enum_v<hipsparseSolvePolicy_t>);
static_assert(std::is_enum_v<hipsparseSideMode_t>);
static_assert(std::is_enum_v<hipsparseDirection_t>);
static_assert(std::is_enum_v<hipsparseCsr2CscAlg_t>);
static_assert(std::is_enum_v<hipsparseFormat_t>);
static_assert(std::is_enum_v<hipsparseOrder_t>);
static_assert(std::is_enum_v<hipsparseIndexType_t>);
static_assert(std::is_enum_v<hipsparseSpMVAlg_t>);
static_assert(std::is_enum_v<hipsparseSpMMAlg_t>);
static_assert(std::is_enum_v<hipsparseSparseToDenseAlg_t>);
static_assert(std::is_enum_v<hipsparseDenseToSparseAlg_t>);
static_assert(std::is_enum_v<hipsparseSDDMMAlg_t>);
static_assert(std::is_enum_v<hipsparseSpSVAlg_t>);
static_assert(std::is_enum_v<hipsparseSpSMAlg_t>);
static_assert(std::is_enum_v<hipsparseSpMatAttribute_t>);
static_assert(std::is_enum_v<hipsparseSpGEMMAlg_t>);

// ────────────────────────────────────────────────────────────────────────
// Enum values: hipsparseStatus_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(HIPSPARSE_STATUS_SUCCESS) == 0);
static_assert(static_cast<int>(HIPSPARSE_STATUS_NOT_INITIALIZED) == 1);
static_assert(static_cast<int>(HIPSPARSE_STATUS_ALLOC_FAILED) == 2);
static_assert(static_cast<int>(HIPSPARSE_STATUS_INVALID_VALUE) == 3);
static_assert(static_cast<int>(HIPSPARSE_STATUS_ARCH_MISMATCH) == 4);
static_assert(static_cast<int>(HIPSPARSE_STATUS_MAPPING_ERROR) == 5);
static_assert(static_cast<int>(HIPSPARSE_STATUS_EXECUTION_FAILED) == 6);
static_assert(static_cast<int>(HIPSPARSE_STATUS_INTERNAL_ERROR) == 7);
static_assert(static_cast<int>(HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED) == 8);
static_assert(static_cast<int>(HIPSPARSE_STATUS_ZERO_PIVOT) == 9);
static_assert(static_cast<int>(HIPSPARSE_STATUS_NOT_SUPPORTED) == 10);
static_assert(static_cast<int>(HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES) == 11);

// ────────────────────────────────────────────────────────────────────────
// Enum values: hipsparsePointerMode_t / hipsparseAction_t / hipsparseMatrixType_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(HIPSPARSE_POINTER_MODE_HOST) == 0);
static_assert(static_cast<int>(HIPSPARSE_POINTER_MODE_DEVICE) == 1);

static_assert(static_cast<int>(HIPSPARSE_ACTION_SYMBOLIC) == 0);
static_assert(static_cast<int>(HIPSPARSE_ACTION_NUMERIC) == 1);

static_assert(static_cast<int>(HIPSPARSE_MATRIX_TYPE_GENERAL) == 0);
static_assert(static_cast<int>(HIPSPARSE_MATRIX_TYPE_SYMMETRIC) == 1);
static_assert(static_cast<int>(HIPSPARSE_MATRIX_TYPE_HERMITIAN) == 2);
static_assert(static_cast<int>(HIPSPARSE_MATRIX_TYPE_TRIANGULAR) == 3);

// ────────────────────────────────────────────────────────────────────────
// Enum values: hipsparseFillMode_t / hipsparseDiagType_t / hipsparseIndexBase_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(HIPSPARSE_FILL_MODE_LOWER) == 0);
static_assert(static_cast<int>(HIPSPARSE_FILL_MODE_UPPER) == 1);

static_assert(static_cast<int>(HIPSPARSE_DIAG_TYPE_NON_UNIT) == 0);
static_assert(static_cast<int>(HIPSPARSE_DIAG_TYPE_UNIT) == 1);

static_assert(static_cast<int>(HIPSPARSE_INDEX_BASE_ZERO) == 0);
static_assert(static_cast<int>(HIPSPARSE_INDEX_BASE_ONE) == 1);

// ────────────────────────────────────────────────────────────────────────
// Enum values: hipsparseOperation_t / hipsparseHybPartition_t /
// hipsparseSolvePolicy_t / hipsparseSideMode_t / hipsparseDirection_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(HIPSPARSE_OPERATION_NON_TRANSPOSE) == 0);
static_assert(static_cast<int>(HIPSPARSE_OPERATION_TRANSPOSE) == 1);
static_assert(static_cast<int>(HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE) == 2);

static_assert(static_cast<int>(HIPSPARSE_HYB_PARTITION_AUTO) == 0);
static_assert(static_cast<int>(HIPSPARSE_HYB_PARTITION_USER) == 1);
static_assert(static_cast<int>(HIPSPARSE_HYB_PARTITION_MAX) == 2);

static_assert(static_cast<int>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL) == 0);
static_assert(static_cast<int>(HIPSPARSE_SOLVE_POLICY_USE_LEVEL) == 1);

static_assert(static_cast<int>(HIPSPARSE_SIDE_LEFT) == 0);
static_assert(static_cast<int>(HIPSPARSE_SIDE_RIGHT) == 1);

static_assert(static_cast<int>(HIPSPARSE_DIRECTION_ROW) == 0);
static_assert(static_cast<int>(HIPSPARSE_DIRECTION_COLUMN) == 1);

// ────────────────────────────────────────────────────────────────────────
// Enum values: hipsparseCsr2CscAlg_t / hipsparseFormat_t / hipsparseOrder_t /
// hipsparseIndexType_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(HIPSPARSE_CSR2CSC_ALG_DEFAULT) == 0);
static_assert(static_cast<int>(HIPSPARSE_CSR2CSC_ALG1) == 1);
static_assert(static_cast<int>(HIPSPARSE_CSR2CSC_ALG2) == 2);

static_assert(static_cast<int>(HIPSPARSE_FORMAT_CSR) == 1);
static_assert(static_cast<int>(HIPSPARSE_FORMAT_CSC) == 2);
static_assert(static_cast<int>(HIPSPARSE_FORMAT_COO) == 3);
static_assert(static_cast<int>(HIPSPARSE_FORMAT_COO_AOS) == 4);
static_assert(static_cast<int>(HIPSPARSE_FORMAT_BLOCKED_ELL) == 5);

// HIPSPARSE_ORDER_COLUMN is deprecated in favor of HIPSPARSE_ORDER_COL (same value);
// re-exported with the attribute intact, so this value check silences the warning locally.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
static_assert(static_cast<int>(HIPSPARSE_ORDER_COLUMN) == 1);
#pragma clang diagnostic pop
static_assert(static_cast<int>(HIPSPARSE_ORDER_COL) == 1);
static_assert(static_cast<int>(HIPSPARSE_ORDER_ROW) == 2);

static_assert(static_cast<int>(HIPSPARSE_INDEX_16U) == 1);
static_assert(static_cast<int>(HIPSPARSE_INDEX_32I) == 2);
static_assert(static_cast<int>(HIPSPARSE_INDEX_64I) == 3);

// ────────────────────────────────────────────────────────────────────────
// Enum values: hipsparseSpMVAlg_t / hipsparseSpMMAlg_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(HIPSPARSE_MV_ALG_DEFAULT) == 0);
static_assert(static_cast<int>(HIPSPARSE_COOMV_ALG) == 1);
static_assert(static_cast<int>(HIPSPARSE_CSRMV_ALG1) == 2);
static_assert(static_cast<int>(HIPSPARSE_CSRMV_ALG2) == 3);
static_assert(static_cast<int>(HIPSPARSE_SPMV_ALG_DEFAULT) == 0);
static_assert(static_cast<int>(HIPSPARSE_SPMV_COO_ALG1) == 1);
static_assert(static_cast<int>(HIPSPARSE_SPMV_CSR_ALG1) == 2);
static_assert(static_cast<int>(HIPSPARSE_SPMV_CSR_ALG2) == 3);
static_assert(static_cast<int>(HIPSPARSE_SPMV_COO_ALG2) == 4);

static_assert(static_cast<int>(HIPSPARSE_MM_ALG_DEFAULT) == 0);
static_assert(static_cast<int>(HIPSPARSE_COOMM_ALG1) == 1);
static_assert(static_cast<int>(HIPSPARSE_COOMM_ALG2) == 2);
static_assert(static_cast<int>(HIPSPARSE_COOMM_ALG3) == 3);
static_assert(static_cast<int>(HIPSPARSE_CSRMM_ALG1) == 4);
static_assert(static_cast<int>(HIPSPARSE_SPMM_ALG_DEFAULT) == 0);
static_assert(static_cast<int>(HIPSPARSE_SPMM_COO_ALG1) == 1);
static_assert(static_cast<int>(HIPSPARSE_SPMM_COO_ALG2) == 2);
static_assert(static_cast<int>(HIPSPARSE_SPMM_COO_ALG3) == 3);
static_assert(static_cast<int>(HIPSPARSE_SPMM_COO_ALG4) == 5);
static_assert(static_cast<int>(HIPSPARSE_SPMM_CSR_ALG1) == 4);
static_assert(static_cast<int>(HIPSPARSE_SPMM_CSR_ALG2) == 6);
static_assert(static_cast<int>(HIPSPARSE_SPMM_CSR_ALG3) == 12);
static_assert(static_cast<int>(HIPSPARSE_SPMM_BLOCKED_ELL_ALG1) == 13);

// ────────────────────────────────────────────────────────────────────────
// Enum values: SparseToDense / DenseToSparse / SDDMM / SpSV / SpSM /
// SpMatAttribute / SpGEMM
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(HIPSPARSE_SPARSETODENSE_ALG_DEFAULT) == 0);
static_assert(static_cast<int>(HIPSPARSE_DENSETOSPARSE_ALG_DEFAULT) == 0);
static_assert(static_cast<int>(HIPSPARSE_SDDMM_ALG_DEFAULT) == 0);
static_assert(static_cast<int>(HIPSPARSE_SPSV_ALG_DEFAULT) == 0);
static_assert(static_cast<int>(HIPSPARSE_SPSM_ALG_DEFAULT) == 0);

static_assert(static_cast<int>(HIPSPARSE_SPMAT_FILL_MODE) == 0);
static_assert(static_cast<int>(HIPSPARSE_SPMAT_DIAG_TYPE) == 1);

static_assert(static_cast<int>(HIPSPARSE_SPGEMM_DEFAULT) == 0);
static_assert(static_cast<int>(HIPSPARSE_SPGEMM_CSR_ALG_DETERMINISTIC) == 1);
static_assert(static_cast<int>(HIPSPARSE_SPGEMM_CSR_ALG_NONDETERMINISTIC) == 2);
static_assert(static_cast<int>(HIPSPARSE_SPGEMM_ALG1) == 3);
static_assert(static_cast<int>(HIPSPARSE_SPGEMM_ALG2) == 4);
static_assert(static_cast<int>(HIPSPARSE_SPGEMM_ALG3) == 5);

// ────────────────────────────────────────────────────────────────────────
// Handle type traits (opaque pointer types)
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_pointer_v<hipsparseHandle_t>);
static_assert(std::is_pointer_v<hipsparseMatDescr_t>);
static_assert(std::is_pointer_v<hipsparseHybMat_t>);
static_assert(std::is_pointer_v<hipsparseColorInfo_t>);
static_assert(std::is_pointer_v<bsrsv2Info_t>);
static_assert(std::is_pointer_v<csrgemm2Info_t>);
static_assert(std::is_pointer_v<pruneInfo_t>);
static_assert(std::is_pointer_v<hipsparseSpVecDescr_t>);
static_assert(std::is_pointer_v<hipsparseDnVecDescr_t>);
static_assert(std::is_pointer_v<hipsparseSpMatDescr_t>);
static_assert(std::is_pointer_v<hipsparseDnMatDescr_t>);
static_assert(std::is_pointer_v<hipsparseSpGEMMDescr_t>);
static_assert(std::is_pointer_v<hipsparseSpSVDescr_t>);
static_assert(std::is_pointer_v<hipsparseSpSMDescr_t>);

// ────────────────────────────────────────────────────────────────────────
// WWR_LINK_CHECK: full function surface (all 546 HIPSPARSE_EXPORT declarations)
// ────────────────────────────────────────────────────────────────────────

WWR_LINK_CHECK(hipsparseAxpby)
WWR_LINK_CHECK(hipsparseBlockedEllGet)
WWR_LINK_CHECK(hipsparseCaxpyi)
WWR_LINK_CHECK(hipsparseCbsr2csr)
WWR_LINK_CHECK(hipsparseCbsric02)
WWR_LINK_CHECK(hipsparseCbsric02_analysis)
WWR_LINK_CHECK(hipsparseCbsric02_bufferSize)
WWR_LINK_CHECK(hipsparseCbsrilu02)
WWR_LINK_CHECK(hipsparseCbsrilu02_analysis)
WWR_LINK_CHECK(hipsparseCbsrilu02_bufferSize)
WWR_LINK_CHECK(hipsparseCbsrilu02_numericBoost)
WWR_LINK_CHECK(hipsparseCbsrmm)
WWR_LINK_CHECK(hipsparseCbsrmv)
WWR_LINK_CHECK(hipsparseCbsrsm2_analysis)
WWR_LINK_CHECK(hipsparseCbsrsm2_bufferSize)
WWR_LINK_CHECK(hipsparseCbsrsm2_solve)
WWR_LINK_CHECK(hipsparseCbsrsv2_analysis)
WWR_LINK_CHECK(hipsparseCbsrsv2_bufferSize)
WWR_LINK_CHECK(hipsparseCbsrsv2_bufferSizeExt)
WWR_LINK_CHECK(hipsparseCbsrsv2_solve)
WWR_LINK_CHECK(hipsparseCbsrxmv)
WWR_LINK_CHECK(hipsparseCcsc2dense)
WWR_LINK_CHECK(hipsparseCcsr2bsr)
WWR_LINK_CHECK(hipsparseCcsr2csc)
WWR_LINK_CHECK(hipsparseCcsr2csr_compress)
WWR_LINK_CHECK(hipsparseCcsr2csru)
WWR_LINK_CHECK(hipsparseCcsr2dense)
WWR_LINK_CHECK(hipsparseCcsr2gebsr)
WWR_LINK_CHECK(hipsparseCcsr2gebsr_bufferSize)
WWR_LINK_CHECK(hipsparseCcsr2hyb)
WWR_LINK_CHECK(hipsparseCcsrcolor)
WWR_LINK_CHECK(hipsparseCcsrgeam)
WWR_LINK_CHECK(hipsparseCcsrgeam2)
WWR_LINK_CHECK(hipsparseCcsrgeam2_bufferSizeExt)
WWR_LINK_CHECK(hipsparseCcsrgemm)
WWR_LINK_CHECK(hipsparseCcsrgemm2)
WWR_LINK_CHECK(hipsparseCcsrgemm2_bufferSizeExt)
WWR_LINK_CHECK(hipsparseCcsric02)
WWR_LINK_CHECK(hipsparseCcsric02_analysis)
WWR_LINK_CHECK(hipsparseCcsric02_bufferSize)
WWR_LINK_CHECK(hipsparseCcsric02_bufferSizeExt)
WWR_LINK_CHECK(hipsparseCcsrilu02)
WWR_LINK_CHECK(hipsparseCcsrilu02_analysis)
WWR_LINK_CHECK(hipsparseCcsrilu02_bufferSize)
WWR_LINK_CHECK(hipsparseCcsrilu02_bufferSizeExt)
WWR_LINK_CHECK(hipsparseCcsrilu02_numericBoost)
WWR_LINK_CHECK(hipsparseCcsrmm)
WWR_LINK_CHECK(hipsparseCcsrmm2)
WWR_LINK_CHECK(hipsparseCcsrmv)
WWR_LINK_CHECK(hipsparseCcsrsm2_analysis)
WWR_LINK_CHECK(hipsparseCcsrsm2_bufferSizeExt)
WWR_LINK_CHECK(hipsparseCcsrsm2_solve)
WWR_LINK_CHECK(hipsparseCcsrsv2_analysis)
WWR_LINK_CHECK(hipsparseCcsrsv2_bufferSize)
WWR_LINK_CHECK(hipsparseCcsrsv2_bufferSizeExt)
WWR_LINK_CHECK(hipsparseCcsrsv2_solve)
WWR_LINK_CHECK(hipsparseCcsru2csr)
WWR_LINK_CHECK(hipsparseCcsru2csr_bufferSizeExt)
WWR_LINK_CHECK(hipsparseCdense2csc)
WWR_LINK_CHECK(hipsparseCdense2csr)
WWR_LINK_CHECK(hipsparseCdotci)
WWR_LINK_CHECK(hipsparseCdoti)
WWR_LINK_CHECK(hipsparseCgebsr2csr)
WWR_LINK_CHECK(hipsparseCgebsr2gebsc)
WWR_LINK_CHECK(hipsparseCgebsr2gebsc_bufferSize)
WWR_LINK_CHECK(hipsparseCgebsr2gebsr)
WWR_LINK_CHECK(hipsparseCgebsr2gebsr_bufferSize)
WWR_LINK_CHECK(hipsparseCgemmi)
WWR_LINK_CHECK(hipsparseCgemvi)
WWR_LINK_CHECK(hipsparseCgemvi_bufferSize)
WWR_LINK_CHECK(hipsparseCgpsvInterleavedBatch)
WWR_LINK_CHECK(hipsparseCgpsvInterleavedBatch_bufferSizeExt)
WWR_LINK_CHECK(hipsparseCgthr)
WWR_LINK_CHECK(hipsparseCgthrz)
WWR_LINK_CHECK(hipsparseCgtsv2)
WWR_LINK_CHECK(hipsparseCgtsv2StridedBatch)
WWR_LINK_CHECK(hipsparseCgtsv2StridedBatch_bufferSizeExt)
WWR_LINK_CHECK(hipsparseCgtsv2_bufferSizeExt)
WWR_LINK_CHECK(hipsparseCgtsv2_nopivot)
WWR_LINK_CHECK(hipsparseCgtsv2_nopivot_bufferSizeExt)
WWR_LINK_CHECK(hipsparseCgtsvInterleavedBatch)
WWR_LINK_CHECK(hipsparseCgtsvInterleavedBatch_bufferSizeExt)
WWR_LINK_CHECK(hipsparseChyb2csr)
WWR_LINK_CHECK(hipsparseChybmv)
WWR_LINK_CHECK(hipsparseCnnz)
WWR_LINK_CHECK(hipsparseCnnz_compress)
WWR_LINK_CHECK(hipsparseConstBlockedEllGet)
WWR_LINK_CHECK(hipsparseConstCooGet)
WWR_LINK_CHECK(hipsparseConstCscGet)
WWR_LINK_CHECK(hipsparseConstCsrGet)
WWR_LINK_CHECK(hipsparseConstDnMatGet)
WWR_LINK_CHECK(hipsparseConstDnMatGetValues)
WWR_LINK_CHECK(hipsparseConstDnVecGet)
WWR_LINK_CHECK(hipsparseConstDnVecGetValues)
WWR_LINK_CHECK(hipsparseConstSpMatGetValues)
WWR_LINK_CHECK(hipsparseConstSpVecGet)
WWR_LINK_CHECK(hipsparseConstSpVecGetValues)
WWR_LINK_CHECK(hipsparseCooAoSGet)
WWR_LINK_CHECK(hipsparseCooGet)
WWR_LINK_CHECK(hipsparseCooSetPointers)
WWR_LINK_CHECK(hipsparseCooSetStridedBatch)
WWR_LINK_CHECK(hipsparseCopyMatDescr)
WWR_LINK_CHECK(hipsparseCreate)
WWR_LINK_CHECK(hipsparseCreateBlockedEll)
WWR_LINK_CHECK(hipsparseCreateBsric02Info)
WWR_LINK_CHECK(hipsparseCreateBsrilu02Info)
WWR_LINK_CHECK(hipsparseCreateBsrsm2Info)
WWR_LINK_CHECK(hipsparseCreateBsrsv2Info)
WWR_LINK_CHECK(hipsparseCreateColorInfo)
WWR_LINK_CHECK(hipsparseCreateConstBlockedEll)
WWR_LINK_CHECK(hipsparseCreateConstCoo)
WWR_LINK_CHECK(hipsparseCreateConstCsc)
WWR_LINK_CHECK(hipsparseCreateConstCsr)
WWR_LINK_CHECK(hipsparseCreateConstDnMat)
WWR_LINK_CHECK(hipsparseCreateConstDnVec)
WWR_LINK_CHECK(hipsparseCreateConstSpVec)
WWR_LINK_CHECK(hipsparseCreateCoo)
WWR_LINK_CHECK(hipsparseCreateCooAoS)
WWR_LINK_CHECK(hipsparseCreateCsc)
WWR_LINK_CHECK(hipsparseCreateCsr)
WWR_LINK_CHECK(hipsparseCreateCsrgemm2Info)
WWR_LINK_CHECK(hipsparseCreateCsric02Info)
WWR_LINK_CHECK(hipsparseCreateCsrilu02Info)
WWR_LINK_CHECK(hipsparseCreateCsrsm2Info)
WWR_LINK_CHECK(hipsparseCreateCsrsv2Info)
WWR_LINK_CHECK(hipsparseCreateCsru2csrInfo)
WWR_LINK_CHECK(hipsparseCreateDnMat)
WWR_LINK_CHECK(hipsparseCreateDnVec)
WWR_LINK_CHECK(hipsparseCreateHybMat)
WWR_LINK_CHECK(hipsparseCreateIdentityPermutation)
WWR_LINK_CHECK(hipsparseCreateMatDescr)
WWR_LINK_CHECK(hipsparseCreatePruneInfo)
WWR_LINK_CHECK(hipsparseCreateSpVec)
WWR_LINK_CHECK(hipsparseCscGet)
WWR_LINK_CHECK(hipsparseCscSetPointers)
WWR_LINK_CHECK(hipsparseCsctr)
WWR_LINK_CHECK(hipsparseCsr2cscEx2)
WWR_LINK_CHECK(hipsparseCsr2cscEx2_bufferSize)
WWR_LINK_CHECK(hipsparseCsrGet)
WWR_LINK_CHECK(hipsparseCsrSetPointers)
WWR_LINK_CHECK(hipsparseCsrSetStridedBatch)
WWR_LINK_CHECK(hipsparseDaxpyi)
WWR_LINK_CHECK(hipsparseDbsr2csr)
WWR_LINK_CHECK(hipsparseDbsric02)
WWR_LINK_CHECK(hipsparseDbsric02_analysis)
WWR_LINK_CHECK(hipsparseDbsric02_bufferSize)
WWR_LINK_CHECK(hipsparseDbsrilu02)
WWR_LINK_CHECK(hipsparseDbsrilu02_analysis)
WWR_LINK_CHECK(hipsparseDbsrilu02_bufferSize)
WWR_LINK_CHECK(hipsparseDbsrilu02_numericBoost)
WWR_LINK_CHECK(hipsparseDbsrmm)
WWR_LINK_CHECK(hipsparseDbsrmv)
WWR_LINK_CHECK(hipsparseDbsrsm2_analysis)
WWR_LINK_CHECK(hipsparseDbsrsm2_bufferSize)
WWR_LINK_CHECK(hipsparseDbsrsm2_solve)
WWR_LINK_CHECK(hipsparseDbsrsv2_analysis)
WWR_LINK_CHECK(hipsparseDbsrsv2_bufferSize)
WWR_LINK_CHECK(hipsparseDbsrsv2_bufferSizeExt)
WWR_LINK_CHECK(hipsparseDbsrsv2_solve)
WWR_LINK_CHECK(hipsparseDbsrxmv)
WWR_LINK_CHECK(hipsparseDcsc2dense)
WWR_LINK_CHECK(hipsparseDcsr2bsr)
WWR_LINK_CHECK(hipsparseDcsr2csc)
WWR_LINK_CHECK(hipsparseDcsr2csr_compress)
WWR_LINK_CHECK(hipsparseDcsr2csru)
WWR_LINK_CHECK(hipsparseDcsr2dense)
WWR_LINK_CHECK(hipsparseDcsr2gebsr)
WWR_LINK_CHECK(hipsparseDcsr2gebsr_bufferSize)
WWR_LINK_CHECK(hipsparseDcsr2hyb)
WWR_LINK_CHECK(hipsparseDcsrcolor)
WWR_LINK_CHECK(hipsparseDcsrgeam)
WWR_LINK_CHECK(hipsparseDcsrgeam2)
WWR_LINK_CHECK(hipsparseDcsrgeam2_bufferSizeExt)
WWR_LINK_CHECK(hipsparseDcsrgemm)
WWR_LINK_CHECK(hipsparseDcsrgemm2)
WWR_LINK_CHECK(hipsparseDcsrgemm2_bufferSizeExt)
WWR_LINK_CHECK(hipsparseDcsric02)
WWR_LINK_CHECK(hipsparseDcsric02_analysis)
WWR_LINK_CHECK(hipsparseDcsric02_bufferSize)
WWR_LINK_CHECK(hipsparseDcsric02_bufferSizeExt)
WWR_LINK_CHECK(hipsparseDcsrilu02)
WWR_LINK_CHECK(hipsparseDcsrilu02_analysis)
WWR_LINK_CHECK(hipsparseDcsrilu02_bufferSize)
WWR_LINK_CHECK(hipsparseDcsrilu02_bufferSizeExt)
WWR_LINK_CHECK(hipsparseDcsrilu02_numericBoost)
WWR_LINK_CHECK(hipsparseDcsrmm)
WWR_LINK_CHECK(hipsparseDcsrmm2)
WWR_LINK_CHECK(hipsparseDcsrmv)
WWR_LINK_CHECK(hipsparseDcsrsm2_analysis)
WWR_LINK_CHECK(hipsparseDcsrsm2_bufferSizeExt)
WWR_LINK_CHECK(hipsparseDcsrsm2_solve)
WWR_LINK_CHECK(hipsparseDcsrsv2_analysis)
WWR_LINK_CHECK(hipsparseDcsrsv2_bufferSize)
WWR_LINK_CHECK(hipsparseDcsrsv2_bufferSizeExt)
WWR_LINK_CHECK(hipsparseDcsrsv2_solve)
WWR_LINK_CHECK(hipsparseDcsru2csr)
WWR_LINK_CHECK(hipsparseDcsru2csr_bufferSizeExt)
WWR_LINK_CHECK(hipsparseDdense2csc)
WWR_LINK_CHECK(hipsparseDdense2csr)
WWR_LINK_CHECK(hipsparseDdoti)
WWR_LINK_CHECK(hipsparseDenseToSparse_analysis)
WWR_LINK_CHECK(hipsparseDenseToSparse_bufferSize)
WWR_LINK_CHECK(hipsparseDenseToSparse_convert)
WWR_LINK_CHECK(hipsparseDestroy)
WWR_LINK_CHECK(hipsparseDestroyBsric02Info)
WWR_LINK_CHECK(hipsparseDestroyBsrilu02Info)
WWR_LINK_CHECK(hipsparseDestroyBsrsm2Info)
WWR_LINK_CHECK(hipsparseDestroyBsrsv2Info)
WWR_LINK_CHECK(hipsparseDestroyColorInfo)
WWR_LINK_CHECK(hipsparseDestroyCsrgemm2Info)
WWR_LINK_CHECK(hipsparseDestroyCsric02Info)
WWR_LINK_CHECK(hipsparseDestroyCsrilu02Info)
WWR_LINK_CHECK(hipsparseDestroyCsrsm2Info)
WWR_LINK_CHECK(hipsparseDestroyCsrsv2Info)
WWR_LINK_CHECK(hipsparseDestroyCsru2csrInfo)
WWR_LINK_CHECK(hipsparseDestroyDnMat)
WWR_LINK_CHECK(hipsparseDestroyDnVec)
WWR_LINK_CHECK(hipsparseDestroyHybMat)
WWR_LINK_CHECK(hipsparseDestroyMatDescr)
WWR_LINK_CHECK(hipsparseDestroyPruneInfo)
WWR_LINK_CHECK(hipsparseDestroySpMat)
WWR_LINK_CHECK(hipsparseDestroySpVec)
WWR_LINK_CHECK(hipsparseDgebsr2csr)
WWR_LINK_CHECK(hipsparseDgebsr2gebsc)
WWR_LINK_CHECK(hipsparseDgebsr2gebsc_bufferSize)
WWR_LINK_CHECK(hipsparseDgebsr2gebsr)
WWR_LINK_CHECK(hipsparseDgebsr2gebsr_bufferSize)
WWR_LINK_CHECK(hipsparseDgemmi)
WWR_LINK_CHECK(hipsparseDgemvi)
WWR_LINK_CHECK(hipsparseDgemvi_bufferSize)
WWR_LINK_CHECK(hipsparseDgpsvInterleavedBatch)
WWR_LINK_CHECK(hipsparseDgpsvInterleavedBatch_bufferSizeExt)
WWR_LINK_CHECK(hipsparseDgthr)
WWR_LINK_CHECK(hipsparseDgthrz)
WWR_LINK_CHECK(hipsparseDgtsv2)
WWR_LINK_CHECK(hipsparseDgtsv2StridedBatch)
WWR_LINK_CHECK(hipsparseDgtsv2StridedBatch_bufferSizeExt)
WWR_LINK_CHECK(hipsparseDgtsv2_bufferSizeExt)
WWR_LINK_CHECK(hipsparseDgtsv2_nopivot)
WWR_LINK_CHECK(hipsparseDgtsv2_nopivot_bufferSizeExt)
WWR_LINK_CHECK(hipsparseDgtsvInterleavedBatch)
WWR_LINK_CHECK(hipsparseDgtsvInterleavedBatch_bufferSizeExt)
WWR_LINK_CHECK(hipsparseDhyb2csr)
WWR_LINK_CHECK(hipsparseDhybmv)
WWR_LINK_CHECK(hipsparseDnMatGet)
WWR_LINK_CHECK(hipsparseDnMatGetStridedBatch)
WWR_LINK_CHECK(hipsparseDnMatGetValues)
WWR_LINK_CHECK(hipsparseDnMatSetStridedBatch)
WWR_LINK_CHECK(hipsparseDnMatSetValues)
WWR_LINK_CHECK(hipsparseDnVecGet)
WWR_LINK_CHECK(hipsparseDnVecGetValues)
WWR_LINK_CHECK(hipsparseDnVecSetValues)
WWR_LINK_CHECK(hipsparseDnnz)
WWR_LINK_CHECK(hipsparseDnnz_compress)
WWR_LINK_CHECK(hipsparseDpruneCsr2csr)
WWR_LINK_CHECK(hipsparseDpruneCsr2csrByPercentage)
WWR_LINK_CHECK(hipsparseDpruneCsr2csrByPercentage_bufferSize)
WWR_LINK_CHECK(hipsparseDpruneCsr2csrByPercentage_bufferSizeExt)
WWR_LINK_CHECK(hipsparseDpruneCsr2csrNnz)
WWR_LINK_CHECK(hipsparseDpruneCsr2csrNnzByPercentage)
WWR_LINK_CHECK(hipsparseDpruneCsr2csr_bufferSize)
WWR_LINK_CHECK(hipsparseDpruneCsr2csr_bufferSizeExt)
WWR_LINK_CHECK(hipsparseDpruneDense2csr)
WWR_LINK_CHECK(hipsparseDpruneDense2csrByPercentage)
WWR_LINK_CHECK(hipsparseDpruneDense2csrByPercentage_bufferSize)
WWR_LINK_CHECK(hipsparseDpruneDense2csrByPercentage_bufferSizeExt)
WWR_LINK_CHECK(hipsparseDpruneDense2csrNnz)
WWR_LINK_CHECK(hipsparseDpruneDense2csrNnzByPercentage)
WWR_LINK_CHECK(hipsparseDpruneDense2csr_bufferSize)
WWR_LINK_CHECK(hipsparseDpruneDense2csr_bufferSizeExt)
WWR_LINK_CHECK(hipsparseDroti)
WWR_LINK_CHECK(hipsparseDsctr)
WWR_LINK_CHECK(hipsparseGather)
WWR_LINK_CHECK(hipsparseGetErrorName)
WWR_LINK_CHECK(hipsparseGetErrorString)
WWR_LINK_CHECK(hipsparseGetGitRevision)
WWR_LINK_CHECK(hipsparseGetMatDiagType)
WWR_LINK_CHECK(hipsparseGetMatFillMode)
WWR_LINK_CHECK(hipsparseGetMatIndexBase)
WWR_LINK_CHECK(hipsparseGetMatType)
WWR_LINK_CHECK(hipsparseGetPointerMode)
WWR_LINK_CHECK(hipsparseGetStream)
WWR_LINK_CHECK(hipsparseGetVersion)
WWR_LINK_CHECK(hipsparseRot)
WWR_LINK_CHECK(hipsparseSDDMM)
WWR_LINK_CHECK(hipsparseSDDMM_bufferSize)
WWR_LINK_CHECK(hipsparseSDDMM_preprocess)
WWR_LINK_CHECK(hipsparseSaxpyi)
WWR_LINK_CHECK(hipsparseSbsr2csr)
WWR_LINK_CHECK(hipsparseSbsric02)
WWR_LINK_CHECK(hipsparseSbsric02_analysis)
WWR_LINK_CHECK(hipsparseSbsric02_bufferSize)
WWR_LINK_CHECK(hipsparseSbsrilu02)
WWR_LINK_CHECK(hipsparseSbsrilu02_analysis)
WWR_LINK_CHECK(hipsparseSbsrilu02_bufferSize)
WWR_LINK_CHECK(hipsparseSbsrilu02_numericBoost)
WWR_LINK_CHECK(hipsparseSbsrmm)
WWR_LINK_CHECK(hipsparseSbsrmv)
WWR_LINK_CHECK(hipsparseSbsrsm2_analysis)
WWR_LINK_CHECK(hipsparseSbsrsm2_bufferSize)
WWR_LINK_CHECK(hipsparseSbsrsm2_solve)
WWR_LINK_CHECK(hipsparseSbsrsv2_analysis)
WWR_LINK_CHECK(hipsparseSbsrsv2_bufferSize)
WWR_LINK_CHECK(hipsparseSbsrsv2_bufferSizeExt)
WWR_LINK_CHECK(hipsparseSbsrsv2_solve)
WWR_LINK_CHECK(hipsparseSbsrxmv)
WWR_LINK_CHECK(hipsparseScatter)
WWR_LINK_CHECK(hipsparseScsc2dense)
WWR_LINK_CHECK(hipsparseScsr2bsr)
WWR_LINK_CHECK(hipsparseScsr2csc)
WWR_LINK_CHECK(hipsparseScsr2csr_compress)
WWR_LINK_CHECK(hipsparseScsr2csru)
WWR_LINK_CHECK(hipsparseScsr2dense)
WWR_LINK_CHECK(hipsparseScsr2gebsr)
WWR_LINK_CHECK(hipsparseScsr2gebsr_bufferSize)
WWR_LINK_CHECK(hipsparseScsr2hyb)
WWR_LINK_CHECK(hipsparseScsrcolor)
WWR_LINK_CHECK(hipsparseScsrgeam)
WWR_LINK_CHECK(hipsparseScsrgeam2)
WWR_LINK_CHECK(hipsparseScsrgeam2_bufferSizeExt)
WWR_LINK_CHECK(hipsparseScsrgemm)
WWR_LINK_CHECK(hipsparseScsrgemm2)
WWR_LINK_CHECK(hipsparseScsrgemm2_bufferSizeExt)
WWR_LINK_CHECK(hipsparseScsric02)
WWR_LINK_CHECK(hipsparseScsric02_analysis)
WWR_LINK_CHECK(hipsparseScsric02_bufferSize)
WWR_LINK_CHECK(hipsparseScsric02_bufferSizeExt)
WWR_LINK_CHECK(hipsparseScsrilu02)
WWR_LINK_CHECK(hipsparseScsrilu02_analysis)
WWR_LINK_CHECK(hipsparseScsrilu02_bufferSize)
WWR_LINK_CHECK(hipsparseScsrilu02_bufferSizeExt)
WWR_LINK_CHECK(hipsparseScsrilu02_numericBoost)
WWR_LINK_CHECK(hipsparseScsrmm)
WWR_LINK_CHECK(hipsparseScsrmm2)
WWR_LINK_CHECK(hipsparseScsrmv)
WWR_LINK_CHECK(hipsparseScsrsm2_analysis)
WWR_LINK_CHECK(hipsparseScsrsm2_bufferSizeExt)
WWR_LINK_CHECK(hipsparseScsrsm2_solve)
WWR_LINK_CHECK(hipsparseScsrsv2_analysis)
WWR_LINK_CHECK(hipsparseScsrsv2_bufferSize)
WWR_LINK_CHECK(hipsparseScsrsv2_bufferSizeExt)
WWR_LINK_CHECK(hipsparseScsrsv2_solve)
WWR_LINK_CHECK(hipsparseScsru2csr)
WWR_LINK_CHECK(hipsparseScsru2csr_bufferSizeExt)
WWR_LINK_CHECK(hipsparseSdense2csc)
WWR_LINK_CHECK(hipsparseSdense2csr)
WWR_LINK_CHECK(hipsparseSdoti)
WWR_LINK_CHECK(hipsparseSetMatDiagType)
WWR_LINK_CHECK(hipsparseSetMatFillMode)
WWR_LINK_CHECK(hipsparseSetMatIndexBase)
WWR_LINK_CHECK(hipsparseSetMatType)
WWR_LINK_CHECK(hipsparseSetPointerMode)
WWR_LINK_CHECK(hipsparseSetStream)
WWR_LINK_CHECK(hipsparseSgebsr2csr)
WWR_LINK_CHECK(hipsparseSgebsr2gebsc)
WWR_LINK_CHECK(hipsparseSgebsr2gebsc_bufferSize)
WWR_LINK_CHECK(hipsparseSgebsr2gebsr)
WWR_LINK_CHECK(hipsparseSgebsr2gebsr_bufferSize)
WWR_LINK_CHECK(hipsparseSgemmi)
WWR_LINK_CHECK(hipsparseSgemvi)
WWR_LINK_CHECK(hipsparseSgemvi_bufferSize)
WWR_LINK_CHECK(hipsparseSgpsvInterleavedBatch)
WWR_LINK_CHECK(hipsparseSgpsvInterleavedBatch_bufferSizeExt)
WWR_LINK_CHECK(hipsparseSgthr)
WWR_LINK_CHECK(hipsparseSgthrz)
WWR_LINK_CHECK(hipsparseSgtsv2)
WWR_LINK_CHECK(hipsparseSgtsv2StridedBatch)
WWR_LINK_CHECK(hipsparseSgtsv2StridedBatch_bufferSizeExt)
WWR_LINK_CHECK(hipsparseSgtsv2_bufferSizeExt)
WWR_LINK_CHECK(hipsparseSgtsv2_nopivot)
WWR_LINK_CHECK(hipsparseSgtsv2_nopivot_bufferSizeExt)
WWR_LINK_CHECK(hipsparseSgtsvInterleavedBatch)
WWR_LINK_CHECK(hipsparseSgtsvInterleavedBatch_bufferSizeExt)
WWR_LINK_CHECK(hipsparseShyb2csr)
WWR_LINK_CHECK(hipsparseShybmv)
WWR_LINK_CHECK(hipsparseSnnz)
WWR_LINK_CHECK(hipsparseSnnz_compress)
WWR_LINK_CHECK(hipsparseSpGEMM_compute)
WWR_LINK_CHECK(hipsparseSpGEMM_copy)
WWR_LINK_CHECK(hipsparseSpGEMM_createDescr)
WWR_LINK_CHECK(hipsparseSpGEMM_destroyDescr)
WWR_LINK_CHECK(hipsparseSpGEMM_workEstimation)
WWR_LINK_CHECK(hipsparseSpGEMMreuse_compute)
WWR_LINK_CHECK(hipsparseSpGEMMreuse_copy)
WWR_LINK_CHECK(hipsparseSpGEMMreuse_nnz)
WWR_LINK_CHECK(hipsparseSpGEMMreuse_workEstimation)
WWR_LINK_CHECK(hipsparseSpMM)
WWR_LINK_CHECK(hipsparseSpMM_bufferSize)
WWR_LINK_CHECK(hipsparseSpMM_preprocess)
WWR_LINK_CHECK(hipsparseSpMV)
WWR_LINK_CHECK(hipsparseSpMV_bufferSize)
WWR_LINK_CHECK(hipsparseSpMV_preprocess)
WWR_LINK_CHECK(hipsparseSpMatGetAttribute)
WWR_LINK_CHECK(hipsparseSpMatGetFormat)
WWR_LINK_CHECK(hipsparseSpMatGetIndexBase)
WWR_LINK_CHECK(hipsparseSpMatGetSize)
WWR_LINK_CHECK(hipsparseSpMatGetStridedBatch)
WWR_LINK_CHECK(hipsparseSpMatGetValues)
WWR_LINK_CHECK(hipsparseSpMatSetAttribute)
WWR_LINK_CHECK(hipsparseSpMatSetStridedBatch)
WWR_LINK_CHECK(hipsparseSpMatSetValues)
WWR_LINK_CHECK(hipsparseSpSM_analysis)
WWR_LINK_CHECK(hipsparseSpSM_bufferSize)
WWR_LINK_CHECK(hipsparseSpSM_createDescr)
WWR_LINK_CHECK(hipsparseSpSM_destroyDescr)
WWR_LINK_CHECK(hipsparseSpSM_solve)
WWR_LINK_CHECK(hipsparseSpSV_analysis)
WWR_LINK_CHECK(hipsparseSpSV_bufferSize)
WWR_LINK_CHECK(hipsparseSpSV_createDescr)
WWR_LINK_CHECK(hipsparseSpSV_destroyDescr)
WWR_LINK_CHECK(hipsparseSpSV_solve)
WWR_LINK_CHECK(hipsparseSpVV)
WWR_LINK_CHECK(hipsparseSpVV_bufferSize)
WWR_LINK_CHECK(hipsparseSpVecGet)
WWR_LINK_CHECK(hipsparseSpVecGetIndexBase)
WWR_LINK_CHECK(hipsparseSpVecGetValues)
WWR_LINK_CHECK(hipsparseSpVecSetValues)
WWR_LINK_CHECK(hipsparseSparseToDense)
WWR_LINK_CHECK(hipsparseSparseToDense_bufferSize)
WWR_LINK_CHECK(hipsparseSpruneCsr2csr)
WWR_LINK_CHECK(hipsparseSpruneCsr2csrByPercentage)
WWR_LINK_CHECK(hipsparseSpruneCsr2csrByPercentage_bufferSize)
WWR_LINK_CHECK(hipsparseSpruneCsr2csrByPercentage_bufferSizeExt)
WWR_LINK_CHECK(hipsparseSpruneCsr2csrNnz)
WWR_LINK_CHECK(hipsparseSpruneCsr2csrNnzByPercentage)
WWR_LINK_CHECK(hipsparseSpruneCsr2csr_bufferSize)
WWR_LINK_CHECK(hipsparseSpruneCsr2csr_bufferSizeExt)
WWR_LINK_CHECK(hipsparseSpruneDense2csr)
WWR_LINK_CHECK(hipsparseSpruneDense2csrByPercentage)
WWR_LINK_CHECK(hipsparseSpruneDense2csrByPercentage_bufferSize)
WWR_LINK_CHECK(hipsparseSpruneDense2csrByPercentage_bufferSizeExt)
WWR_LINK_CHECK(hipsparseSpruneDense2csrNnz)
WWR_LINK_CHECK(hipsparseSpruneDense2csrNnzByPercentage)
WWR_LINK_CHECK(hipsparseSpruneDense2csr_bufferSize)
WWR_LINK_CHECK(hipsparseSpruneDense2csr_bufferSizeExt)
WWR_LINK_CHECK(hipsparseSroti)
WWR_LINK_CHECK(hipsparseSsctr)
WWR_LINK_CHECK(hipsparseXbsric02_zeroPivot)
WWR_LINK_CHECK(hipsparseXbsrilu02_zeroPivot)
WWR_LINK_CHECK(hipsparseXbsrsm2_zeroPivot)
WWR_LINK_CHECK(hipsparseXbsrsv2_zeroPivot)
WWR_LINK_CHECK(hipsparseXcoo2csr)
WWR_LINK_CHECK(hipsparseXcoosortByColumn)
WWR_LINK_CHECK(hipsparseXcoosortByRow)
WWR_LINK_CHECK(hipsparseXcoosort_bufferSizeExt)
WWR_LINK_CHECK(hipsparseXcscsort)
WWR_LINK_CHECK(hipsparseXcscsort_bufferSizeExt)
WWR_LINK_CHECK(hipsparseXcsr2bsrNnz)
WWR_LINK_CHECK(hipsparseXcsr2coo)
WWR_LINK_CHECK(hipsparseXcsr2gebsrNnz)
WWR_LINK_CHECK(hipsparseXcsrgeam2Nnz)
WWR_LINK_CHECK(hipsparseXcsrgeamNnz)
WWR_LINK_CHECK(hipsparseXcsrgemm2Nnz)
WWR_LINK_CHECK(hipsparseXcsrgemmNnz)
WWR_LINK_CHECK(hipsparseXcsric02_zeroPivot)
WWR_LINK_CHECK(hipsparseXcsrilu02_zeroPivot)
WWR_LINK_CHECK(hipsparseXcsrsm2_zeroPivot)
WWR_LINK_CHECK(hipsparseXcsrsort)
WWR_LINK_CHECK(hipsparseXcsrsort_bufferSizeExt)
WWR_LINK_CHECK(hipsparseXcsrsv2_zeroPivot)
WWR_LINK_CHECK(hipsparseXgebsr2gebsrNnz)
WWR_LINK_CHECK(hipsparseZaxpyi)
WWR_LINK_CHECK(hipsparseZbsr2csr)
WWR_LINK_CHECK(hipsparseZbsric02)
WWR_LINK_CHECK(hipsparseZbsric02_analysis)
WWR_LINK_CHECK(hipsparseZbsric02_bufferSize)
WWR_LINK_CHECK(hipsparseZbsrilu02)
WWR_LINK_CHECK(hipsparseZbsrilu02_analysis)
WWR_LINK_CHECK(hipsparseZbsrilu02_bufferSize)
WWR_LINK_CHECK(hipsparseZbsrilu02_numericBoost)
WWR_LINK_CHECK(hipsparseZbsrmm)
WWR_LINK_CHECK(hipsparseZbsrmv)
WWR_LINK_CHECK(hipsparseZbsrsm2_analysis)
WWR_LINK_CHECK(hipsparseZbsrsm2_bufferSize)
WWR_LINK_CHECK(hipsparseZbsrsm2_solve)
WWR_LINK_CHECK(hipsparseZbsrsv2_analysis)
WWR_LINK_CHECK(hipsparseZbsrsv2_bufferSize)
WWR_LINK_CHECK(hipsparseZbsrsv2_bufferSizeExt)
WWR_LINK_CHECK(hipsparseZbsrsv2_solve)
WWR_LINK_CHECK(hipsparseZbsrxmv)
WWR_LINK_CHECK(hipsparseZcsc2dense)
WWR_LINK_CHECK(hipsparseZcsr2bsr)
WWR_LINK_CHECK(hipsparseZcsr2csc)
WWR_LINK_CHECK(hipsparseZcsr2csr_compress)
WWR_LINK_CHECK(hipsparseZcsr2csru)
WWR_LINK_CHECK(hipsparseZcsr2dense)
WWR_LINK_CHECK(hipsparseZcsr2gebsr)
WWR_LINK_CHECK(hipsparseZcsr2gebsr_bufferSize)
WWR_LINK_CHECK(hipsparseZcsr2hyb)
WWR_LINK_CHECK(hipsparseZcsrcolor)
WWR_LINK_CHECK(hipsparseZcsrgeam)
WWR_LINK_CHECK(hipsparseZcsrgeam2)
WWR_LINK_CHECK(hipsparseZcsrgeam2_bufferSizeExt)
WWR_LINK_CHECK(hipsparseZcsrgemm)
WWR_LINK_CHECK(hipsparseZcsrgemm2)
WWR_LINK_CHECK(hipsparseZcsrgemm2_bufferSizeExt)
WWR_LINK_CHECK(hipsparseZcsric02)
WWR_LINK_CHECK(hipsparseZcsric02_analysis)
WWR_LINK_CHECK(hipsparseZcsric02_bufferSize)
WWR_LINK_CHECK(hipsparseZcsric02_bufferSizeExt)
WWR_LINK_CHECK(hipsparseZcsrilu02)
WWR_LINK_CHECK(hipsparseZcsrilu02_analysis)
WWR_LINK_CHECK(hipsparseZcsrilu02_bufferSize)
WWR_LINK_CHECK(hipsparseZcsrilu02_bufferSizeExt)
WWR_LINK_CHECK(hipsparseZcsrilu02_numericBoost)
WWR_LINK_CHECK(hipsparseZcsrmm)
WWR_LINK_CHECK(hipsparseZcsrmm2)
WWR_LINK_CHECK(hipsparseZcsrmv)
WWR_LINK_CHECK(hipsparseZcsrsm2_analysis)
WWR_LINK_CHECK(hipsparseZcsrsm2_bufferSizeExt)
WWR_LINK_CHECK(hipsparseZcsrsm2_solve)
WWR_LINK_CHECK(hipsparseZcsrsv2_analysis)
WWR_LINK_CHECK(hipsparseZcsrsv2_bufferSize)
WWR_LINK_CHECK(hipsparseZcsrsv2_bufferSizeExt)
WWR_LINK_CHECK(hipsparseZcsrsv2_solve)
WWR_LINK_CHECK(hipsparseZcsru2csr)
WWR_LINK_CHECK(hipsparseZcsru2csr_bufferSizeExt)
WWR_LINK_CHECK(hipsparseZdense2csc)
WWR_LINK_CHECK(hipsparseZdense2csr)
WWR_LINK_CHECK(hipsparseZdotci)
WWR_LINK_CHECK(hipsparseZdoti)
WWR_LINK_CHECK(hipsparseZgebsr2csr)
WWR_LINK_CHECK(hipsparseZgebsr2gebsc)
WWR_LINK_CHECK(hipsparseZgebsr2gebsc_bufferSize)
WWR_LINK_CHECK(hipsparseZgebsr2gebsr)
WWR_LINK_CHECK(hipsparseZgebsr2gebsr_bufferSize)
WWR_LINK_CHECK(hipsparseZgemmi)
WWR_LINK_CHECK(hipsparseZgemvi)
WWR_LINK_CHECK(hipsparseZgemvi_bufferSize)
WWR_LINK_CHECK(hipsparseZgpsvInterleavedBatch)
WWR_LINK_CHECK(hipsparseZgpsvInterleavedBatch_bufferSizeExt)
WWR_LINK_CHECK(hipsparseZgthr)
WWR_LINK_CHECK(hipsparseZgthrz)
WWR_LINK_CHECK(hipsparseZgtsv2)
WWR_LINK_CHECK(hipsparseZgtsv2StridedBatch)
WWR_LINK_CHECK(hipsparseZgtsv2StridedBatch_bufferSizeExt)
WWR_LINK_CHECK(hipsparseZgtsv2_bufferSizeExt)
WWR_LINK_CHECK(hipsparseZgtsv2_nopivot)
WWR_LINK_CHECK(hipsparseZgtsv2_nopivot_bufferSizeExt)
WWR_LINK_CHECK(hipsparseZgtsvInterleavedBatch)
WWR_LINK_CHECK(hipsparseZgtsvInterleavedBatch_bufferSizeExt)
WWR_LINK_CHECK(hipsparseZhyb2csr)
WWR_LINK_CHECK(hipsparseZhybmv)
WWR_LINK_CHECK(hipsparseZnnz)
WWR_LINK_CHECK(hipsparseZnnz_compress)
WWR_LINK_CHECK(hipsparseZsctr)
} // namespace wwr::hip::test
