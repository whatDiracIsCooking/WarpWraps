// hipsparse.cppm - Compile-time tests for gpumod.hip.hipsparse

module;

#include "test/shared/link_check.h"

export module gpumod.test.hip.hipsparse;

import std;
import gpumod.hip.hipsparse;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time tests for gpumod.hip.hipsparse
//
// GPUMOD_LINK_CHECK covers all 546 HIPSPARSE_EXPORT functions declared across
// hipsparse.h and its internal/{level1,level2,level3,extra,precond,
// conversion,reorder,generic}/ headers -- extracted directly, not
// hand-copied.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace gpumod::hip::test {

using namespace gpumod::hip;

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
// GPUMOD_LINK_CHECK: full function surface (all 546 HIPSPARSE_EXPORT declarations)
// ────────────────────────────────────────────────────────────────────────

GPUMOD_LINK_CHECK(hipsparseAxpby)
GPUMOD_LINK_CHECK(hipsparseBlockedEllGet)
GPUMOD_LINK_CHECK(hipsparseCaxpyi)
GPUMOD_LINK_CHECK(hipsparseCbsr2csr)
GPUMOD_LINK_CHECK(hipsparseCbsric02)
GPUMOD_LINK_CHECK(hipsparseCbsric02_analysis)
GPUMOD_LINK_CHECK(hipsparseCbsric02_bufferSize)
GPUMOD_LINK_CHECK(hipsparseCbsrilu02)
GPUMOD_LINK_CHECK(hipsparseCbsrilu02_analysis)
GPUMOD_LINK_CHECK(hipsparseCbsrilu02_bufferSize)
GPUMOD_LINK_CHECK(hipsparseCbsrilu02_numericBoost)
GPUMOD_LINK_CHECK(hipsparseCbsrmm)
GPUMOD_LINK_CHECK(hipsparseCbsrmv)
GPUMOD_LINK_CHECK(hipsparseCbsrsm2_analysis)
GPUMOD_LINK_CHECK(hipsparseCbsrsm2_bufferSize)
GPUMOD_LINK_CHECK(hipsparseCbsrsm2_solve)
GPUMOD_LINK_CHECK(hipsparseCbsrsv2_analysis)
GPUMOD_LINK_CHECK(hipsparseCbsrsv2_bufferSize)
GPUMOD_LINK_CHECK(hipsparseCbsrsv2_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseCbsrsv2_solve)
GPUMOD_LINK_CHECK(hipsparseCbsrxmv)
GPUMOD_LINK_CHECK(hipsparseCcsc2dense)
GPUMOD_LINK_CHECK(hipsparseCcsr2bsr)
GPUMOD_LINK_CHECK(hipsparseCcsr2csc)
GPUMOD_LINK_CHECK(hipsparseCcsr2csr_compress)
GPUMOD_LINK_CHECK(hipsparseCcsr2csru)
GPUMOD_LINK_CHECK(hipsparseCcsr2dense)
GPUMOD_LINK_CHECK(hipsparseCcsr2gebsr)
GPUMOD_LINK_CHECK(hipsparseCcsr2gebsr_bufferSize)
GPUMOD_LINK_CHECK(hipsparseCcsr2hyb)
GPUMOD_LINK_CHECK(hipsparseCcsrcolor)
GPUMOD_LINK_CHECK(hipsparseCcsrgeam)
GPUMOD_LINK_CHECK(hipsparseCcsrgeam2)
GPUMOD_LINK_CHECK(hipsparseCcsrgeam2_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseCcsrgemm)
GPUMOD_LINK_CHECK(hipsparseCcsrgemm2)
GPUMOD_LINK_CHECK(hipsparseCcsrgemm2_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseCcsric02)
GPUMOD_LINK_CHECK(hipsparseCcsric02_analysis)
GPUMOD_LINK_CHECK(hipsparseCcsric02_bufferSize)
GPUMOD_LINK_CHECK(hipsparseCcsric02_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseCcsrilu02)
GPUMOD_LINK_CHECK(hipsparseCcsrilu02_analysis)
GPUMOD_LINK_CHECK(hipsparseCcsrilu02_bufferSize)
GPUMOD_LINK_CHECK(hipsparseCcsrilu02_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseCcsrilu02_numericBoost)
GPUMOD_LINK_CHECK(hipsparseCcsrmm)
GPUMOD_LINK_CHECK(hipsparseCcsrmm2)
GPUMOD_LINK_CHECK(hipsparseCcsrmv)
GPUMOD_LINK_CHECK(hipsparseCcsrsm2_analysis)
GPUMOD_LINK_CHECK(hipsparseCcsrsm2_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseCcsrsm2_solve)
GPUMOD_LINK_CHECK(hipsparseCcsrsv2_analysis)
GPUMOD_LINK_CHECK(hipsparseCcsrsv2_bufferSize)
GPUMOD_LINK_CHECK(hipsparseCcsrsv2_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseCcsrsv2_solve)
GPUMOD_LINK_CHECK(hipsparseCcsru2csr)
GPUMOD_LINK_CHECK(hipsparseCcsru2csr_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseCdense2csc)
GPUMOD_LINK_CHECK(hipsparseCdense2csr)
GPUMOD_LINK_CHECK(hipsparseCdotci)
GPUMOD_LINK_CHECK(hipsparseCdoti)
GPUMOD_LINK_CHECK(hipsparseCgebsr2csr)
GPUMOD_LINK_CHECK(hipsparseCgebsr2gebsc)
GPUMOD_LINK_CHECK(hipsparseCgebsr2gebsc_bufferSize)
GPUMOD_LINK_CHECK(hipsparseCgebsr2gebsr)
GPUMOD_LINK_CHECK(hipsparseCgebsr2gebsr_bufferSize)
GPUMOD_LINK_CHECK(hipsparseCgemmi)
GPUMOD_LINK_CHECK(hipsparseCgemvi)
GPUMOD_LINK_CHECK(hipsparseCgemvi_bufferSize)
GPUMOD_LINK_CHECK(hipsparseCgpsvInterleavedBatch)
GPUMOD_LINK_CHECK(hipsparseCgpsvInterleavedBatch_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseCgthr)
GPUMOD_LINK_CHECK(hipsparseCgthrz)
GPUMOD_LINK_CHECK(hipsparseCgtsv2)
GPUMOD_LINK_CHECK(hipsparseCgtsv2StridedBatch)
GPUMOD_LINK_CHECK(hipsparseCgtsv2StridedBatch_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseCgtsv2_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseCgtsv2_nopivot)
GPUMOD_LINK_CHECK(hipsparseCgtsv2_nopivot_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseCgtsvInterleavedBatch)
GPUMOD_LINK_CHECK(hipsparseCgtsvInterleavedBatch_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseChyb2csr)
GPUMOD_LINK_CHECK(hipsparseChybmv)
GPUMOD_LINK_CHECK(hipsparseCnnz)
GPUMOD_LINK_CHECK(hipsparseCnnz_compress)
GPUMOD_LINK_CHECK(hipsparseConstBlockedEllGet)
GPUMOD_LINK_CHECK(hipsparseConstCooGet)
GPUMOD_LINK_CHECK(hipsparseConstCscGet)
GPUMOD_LINK_CHECK(hipsparseConstCsrGet)
GPUMOD_LINK_CHECK(hipsparseConstDnMatGet)
GPUMOD_LINK_CHECK(hipsparseConstDnMatGetValues)
GPUMOD_LINK_CHECK(hipsparseConstDnVecGet)
GPUMOD_LINK_CHECK(hipsparseConstDnVecGetValues)
GPUMOD_LINK_CHECK(hipsparseConstSpMatGetValues)
GPUMOD_LINK_CHECK(hipsparseConstSpVecGet)
GPUMOD_LINK_CHECK(hipsparseConstSpVecGetValues)
GPUMOD_LINK_CHECK(hipsparseCooAoSGet)
GPUMOD_LINK_CHECK(hipsparseCooGet)
GPUMOD_LINK_CHECK(hipsparseCooSetPointers)
GPUMOD_LINK_CHECK(hipsparseCooSetStridedBatch)
GPUMOD_LINK_CHECK(hipsparseCopyMatDescr)
GPUMOD_LINK_CHECK(hipsparseCreate)
GPUMOD_LINK_CHECK(hipsparseCreateBlockedEll)
GPUMOD_LINK_CHECK(hipsparseCreateBsric02Info)
GPUMOD_LINK_CHECK(hipsparseCreateBsrilu02Info)
GPUMOD_LINK_CHECK(hipsparseCreateBsrsm2Info)
GPUMOD_LINK_CHECK(hipsparseCreateBsrsv2Info)
GPUMOD_LINK_CHECK(hipsparseCreateColorInfo)
GPUMOD_LINK_CHECK(hipsparseCreateConstBlockedEll)
GPUMOD_LINK_CHECK(hipsparseCreateConstCoo)
GPUMOD_LINK_CHECK(hipsparseCreateConstCsc)
GPUMOD_LINK_CHECK(hipsparseCreateConstCsr)
GPUMOD_LINK_CHECK(hipsparseCreateConstDnMat)
GPUMOD_LINK_CHECK(hipsparseCreateConstDnVec)
GPUMOD_LINK_CHECK(hipsparseCreateConstSpVec)
GPUMOD_LINK_CHECK(hipsparseCreateCoo)
GPUMOD_LINK_CHECK(hipsparseCreateCooAoS)
GPUMOD_LINK_CHECK(hipsparseCreateCsc)
GPUMOD_LINK_CHECK(hipsparseCreateCsr)
GPUMOD_LINK_CHECK(hipsparseCreateCsrgemm2Info)
GPUMOD_LINK_CHECK(hipsparseCreateCsric02Info)
GPUMOD_LINK_CHECK(hipsparseCreateCsrilu02Info)
GPUMOD_LINK_CHECK(hipsparseCreateCsrsm2Info)
GPUMOD_LINK_CHECK(hipsparseCreateCsrsv2Info)
GPUMOD_LINK_CHECK(hipsparseCreateCsru2csrInfo)
GPUMOD_LINK_CHECK(hipsparseCreateDnMat)
GPUMOD_LINK_CHECK(hipsparseCreateDnVec)
GPUMOD_LINK_CHECK(hipsparseCreateHybMat)
GPUMOD_LINK_CHECK(hipsparseCreateIdentityPermutation)
GPUMOD_LINK_CHECK(hipsparseCreateMatDescr)
GPUMOD_LINK_CHECK(hipsparseCreatePruneInfo)
GPUMOD_LINK_CHECK(hipsparseCreateSpVec)
GPUMOD_LINK_CHECK(hipsparseCscGet)
GPUMOD_LINK_CHECK(hipsparseCscSetPointers)
GPUMOD_LINK_CHECK(hipsparseCsctr)
GPUMOD_LINK_CHECK(hipsparseCsr2cscEx2)
GPUMOD_LINK_CHECK(hipsparseCsr2cscEx2_bufferSize)
GPUMOD_LINK_CHECK(hipsparseCsrGet)
GPUMOD_LINK_CHECK(hipsparseCsrSetPointers)
GPUMOD_LINK_CHECK(hipsparseCsrSetStridedBatch)
GPUMOD_LINK_CHECK(hipsparseDaxpyi)
GPUMOD_LINK_CHECK(hipsparseDbsr2csr)
GPUMOD_LINK_CHECK(hipsparseDbsric02)
GPUMOD_LINK_CHECK(hipsparseDbsric02_analysis)
GPUMOD_LINK_CHECK(hipsparseDbsric02_bufferSize)
GPUMOD_LINK_CHECK(hipsparseDbsrilu02)
GPUMOD_LINK_CHECK(hipsparseDbsrilu02_analysis)
GPUMOD_LINK_CHECK(hipsparseDbsrilu02_bufferSize)
GPUMOD_LINK_CHECK(hipsparseDbsrilu02_numericBoost)
GPUMOD_LINK_CHECK(hipsparseDbsrmm)
GPUMOD_LINK_CHECK(hipsparseDbsrmv)
GPUMOD_LINK_CHECK(hipsparseDbsrsm2_analysis)
GPUMOD_LINK_CHECK(hipsparseDbsrsm2_bufferSize)
GPUMOD_LINK_CHECK(hipsparseDbsrsm2_solve)
GPUMOD_LINK_CHECK(hipsparseDbsrsv2_analysis)
GPUMOD_LINK_CHECK(hipsparseDbsrsv2_bufferSize)
GPUMOD_LINK_CHECK(hipsparseDbsrsv2_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseDbsrsv2_solve)
GPUMOD_LINK_CHECK(hipsparseDbsrxmv)
GPUMOD_LINK_CHECK(hipsparseDcsc2dense)
GPUMOD_LINK_CHECK(hipsparseDcsr2bsr)
GPUMOD_LINK_CHECK(hipsparseDcsr2csc)
GPUMOD_LINK_CHECK(hipsparseDcsr2csr_compress)
GPUMOD_LINK_CHECK(hipsparseDcsr2csru)
GPUMOD_LINK_CHECK(hipsparseDcsr2dense)
GPUMOD_LINK_CHECK(hipsparseDcsr2gebsr)
GPUMOD_LINK_CHECK(hipsparseDcsr2gebsr_bufferSize)
GPUMOD_LINK_CHECK(hipsparseDcsr2hyb)
GPUMOD_LINK_CHECK(hipsparseDcsrcolor)
GPUMOD_LINK_CHECK(hipsparseDcsrgeam)
GPUMOD_LINK_CHECK(hipsparseDcsrgeam2)
GPUMOD_LINK_CHECK(hipsparseDcsrgeam2_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseDcsrgemm)
GPUMOD_LINK_CHECK(hipsparseDcsrgemm2)
GPUMOD_LINK_CHECK(hipsparseDcsrgemm2_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseDcsric02)
GPUMOD_LINK_CHECK(hipsparseDcsric02_analysis)
GPUMOD_LINK_CHECK(hipsparseDcsric02_bufferSize)
GPUMOD_LINK_CHECK(hipsparseDcsric02_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseDcsrilu02)
GPUMOD_LINK_CHECK(hipsparseDcsrilu02_analysis)
GPUMOD_LINK_CHECK(hipsparseDcsrilu02_bufferSize)
GPUMOD_LINK_CHECK(hipsparseDcsrilu02_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseDcsrilu02_numericBoost)
GPUMOD_LINK_CHECK(hipsparseDcsrmm)
GPUMOD_LINK_CHECK(hipsparseDcsrmm2)
GPUMOD_LINK_CHECK(hipsparseDcsrmv)
GPUMOD_LINK_CHECK(hipsparseDcsrsm2_analysis)
GPUMOD_LINK_CHECK(hipsparseDcsrsm2_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseDcsrsm2_solve)
GPUMOD_LINK_CHECK(hipsparseDcsrsv2_analysis)
GPUMOD_LINK_CHECK(hipsparseDcsrsv2_bufferSize)
GPUMOD_LINK_CHECK(hipsparseDcsrsv2_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseDcsrsv2_solve)
GPUMOD_LINK_CHECK(hipsparseDcsru2csr)
GPUMOD_LINK_CHECK(hipsparseDcsru2csr_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseDdense2csc)
GPUMOD_LINK_CHECK(hipsparseDdense2csr)
GPUMOD_LINK_CHECK(hipsparseDdoti)
GPUMOD_LINK_CHECK(hipsparseDenseToSparse_analysis)
GPUMOD_LINK_CHECK(hipsparseDenseToSparse_bufferSize)
GPUMOD_LINK_CHECK(hipsparseDenseToSparse_convert)
GPUMOD_LINK_CHECK(hipsparseDestroy)
GPUMOD_LINK_CHECK(hipsparseDestroyBsric02Info)
GPUMOD_LINK_CHECK(hipsparseDestroyBsrilu02Info)
GPUMOD_LINK_CHECK(hipsparseDestroyBsrsm2Info)
GPUMOD_LINK_CHECK(hipsparseDestroyBsrsv2Info)
GPUMOD_LINK_CHECK(hipsparseDestroyColorInfo)
GPUMOD_LINK_CHECK(hipsparseDestroyCsrgemm2Info)
GPUMOD_LINK_CHECK(hipsparseDestroyCsric02Info)
GPUMOD_LINK_CHECK(hipsparseDestroyCsrilu02Info)
GPUMOD_LINK_CHECK(hipsparseDestroyCsrsm2Info)
GPUMOD_LINK_CHECK(hipsparseDestroyCsrsv2Info)
GPUMOD_LINK_CHECK(hipsparseDestroyCsru2csrInfo)
GPUMOD_LINK_CHECK(hipsparseDestroyDnMat)
GPUMOD_LINK_CHECK(hipsparseDestroyDnVec)
GPUMOD_LINK_CHECK(hipsparseDestroyHybMat)
GPUMOD_LINK_CHECK(hipsparseDestroyMatDescr)
GPUMOD_LINK_CHECK(hipsparseDestroyPruneInfo)
GPUMOD_LINK_CHECK(hipsparseDestroySpMat)
GPUMOD_LINK_CHECK(hipsparseDestroySpVec)
GPUMOD_LINK_CHECK(hipsparseDgebsr2csr)
GPUMOD_LINK_CHECK(hipsparseDgebsr2gebsc)
GPUMOD_LINK_CHECK(hipsparseDgebsr2gebsc_bufferSize)
GPUMOD_LINK_CHECK(hipsparseDgebsr2gebsr)
GPUMOD_LINK_CHECK(hipsparseDgebsr2gebsr_bufferSize)
GPUMOD_LINK_CHECK(hipsparseDgemmi)
GPUMOD_LINK_CHECK(hipsparseDgemvi)
GPUMOD_LINK_CHECK(hipsparseDgemvi_bufferSize)
GPUMOD_LINK_CHECK(hipsparseDgpsvInterleavedBatch)
GPUMOD_LINK_CHECK(hipsparseDgpsvInterleavedBatch_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseDgthr)
GPUMOD_LINK_CHECK(hipsparseDgthrz)
GPUMOD_LINK_CHECK(hipsparseDgtsv2)
GPUMOD_LINK_CHECK(hipsparseDgtsv2StridedBatch)
GPUMOD_LINK_CHECK(hipsparseDgtsv2StridedBatch_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseDgtsv2_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseDgtsv2_nopivot)
GPUMOD_LINK_CHECK(hipsparseDgtsv2_nopivot_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseDgtsvInterleavedBatch)
GPUMOD_LINK_CHECK(hipsparseDgtsvInterleavedBatch_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseDhyb2csr)
GPUMOD_LINK_CHECK(hipsparseDhybmv)
GPUMOD_LINK_CHECK(hipsparseDnMatGet)
GPUMOD_LINK_CHECK(hipsparseDnMatGetStridedBatch)
GPUMOD_LINK_CHECK(hipsparseDnMatGetValues)
GPUMOD_LINK_CHECK(hipsparseDnMatSetStridedBatch)
GPUMOD_LINK_CHECK(hipsparseDnMatSetValues)
GPUMOD_LINK_CHECK(hipsparseDnVecGet)
GPUMOD_LINK_CHECK(hipsparseDnVecGetValues)
GPUMOD_LINK_CHECK(hipsparseDnVecSetValues)
GPUMOD_LINK_CHECK(hipsparseDnnz)
GPUMOD_LINK_CHECK(hipsparseDnnz_compress)
GPUMOD_LINK_CHECK(hipsparseDpruneCsr2csr)
GPUMOD_LINK_CHECK(hipsparseDpruneCsr2csrByPercentage)
GPUMOD_LINK_CHECK(hipsparseDpruneCsr2csrByPercentage_bufferSize)
GPUMOD_LINK_CHECK(hipsparseDpruneCsr2csrByPercentage_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseDpruneCsr2csrNnz)
GPUMOD_LINK_CHECK(hipsparseDpruneCsr2csrNnzByPercentage)
GPUMOD_LINK_CHECK(hipsparseDpruneCsr2csr_bufferSize)
GPUMOD_LINK_CHECK(hipsparseDpruneCsr2csr_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseDpruneDense2csr)
GPUMOD_LINK_CHECK(hipsparseDpruneDense2csrByPercentage)
GPUMOD_LINK_CHECK(hipsparseDpruneDense2csrByPercentage_bufferSize)
GPUMOD_LINK_CHECK(hipsparseDpruneDense2csrByPercentage_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseDpruneDense2csrNnz)
GPUMOD_LINK_CHECK(hipsparseDpruneDense2csrNnzByPercentage)
GPUMOD_LINK_CHECK(hipsparseDpruneDense2csr_bufferSize)
GPUMOD_LINK_CHECK(hipsparseDpruneDense2csr_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseDroti)
GPUMOD_LINK_CHECK(hipsparseDsctr)
GPUMOD_LINK_CHECK(hipsparseGather)
GPUMOD_LINK_CHECK(hipsparseGetErrorName)
GPUMOD_LINK_CHECK(hipsparseGetErrorString)
GPUMOD_LINK_CHECK(hipsparseGetGitRevision)
GPUMOD_LINK_CHECK(hipsparseGetMatDiagType)
GPUMOD_LINK_CHECK(hipsparseGetMatFillMode)
GPUMOD_LINK_CHECK(hipsparseGetMatIndexBase)
GPUMOD_LINK_CHECK(hipsparseGetMatType)
GPUMOD_LINK_CHECK(hipsparseGetPointerMode)
GPUMOD_LINK_CHECK(hipsparseGetStream)
GPUMOD_LINK_CHECK(hipsparseGetVersion)
GPUMOD_LINK_CHECK(hipsparseRot)
GPUMOD_LINK_CHECK(hipsparseSDDMM)
GPUMOD_LINK_CHECK(hipsparseSDDMM_bufferSize)
GPUMOD_LINK_CHECK(hipsparseSDDMM_preprocess)
GPUMOD_LINK_CHECK(hipsparseSaxpyi)
GPUMOD_LINK_CHECK(hipsparseSbsr2csr)
GPUMOD_LINK_CHECK(hipsparseSbsric02)
GPUMOD_LINK_CHECK(hipsparseSbsric02_analysis)
GPUMOD_LINK_CHECK(hipsparseSbsric02_bufferSize)
GPUMOD_LINK_CHECK(hipsparseSbsrilu02)
GPUMOD_LINK_CHECK(hipsparseSbsrilu02_analysis)
GPUMOD_LINK_CHECK(hipsparseSbsrilu02_bufferSize)
GPUMOD_LINK_CHECK(hipsparseSbsrilu02_numericBoost)
GPUMOD_LINK_CHECK(hipsparseSbsrmm)
GPUMOD_LINK_CHECK(hipsparseSbsrmv)
GPUMOD_LINK_CHECK(hipsparseSbsrsm2_analysis)
GPUMOD_LINK_CHECK(hipsparseSbsrsm2_bufferSize)
GPUMOD_LINK_CHECK(hipsparseSbsrsm2_solve)
GPUMOD_LINK_CHECK(hipsparseSbsrsv2_analysis)
GPUMOD_LINK_CHECK(hipsparseSbsrsv2_bufferSize)
GPUMOD_LINK_CHECK(hipsparseSbsrsv2_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseSbsrsv2_solve)
GPUMOD_LINK_CHECK(hipsparseSbsrxmv)
GPUMOD_LINK_CHECK(hipsparseScatter)
GPUMOD_LINK_CHECK(hipsparseScsc2dense)
GPUMOD_LINK_CHECK(hipsparseScsr2bsr)
GPUMOD_LINK_CHECK(hipsparseScsr2csc)
GPUMOD_LINK_CHECK(hipsparseScsr2csr_compress)
GPUMOD_LINK_CHECK(hipsparseScsr2csru)
GPUMOD_LINK_CHECK(hipsparseScsr2dense)
GPUMOD_LINK_CHECK(hipsparseScsr2gebsr)
GPUMOD_LINK_CHECK(hipsparseScsr2gebsr_bufferSize)
GPUMOD_LINK_CHECK(hipsparseScsr2hyb)
GPUMOD_LINK_CHECK(hipsparseScsrcolor)
GPUMOD_LINK_CHECK(hipsparseScsrgeam)
GPUMOD_LINK_CHECK(hipsparseScsrgeam2)
GPUMOD_LINK_CHECK(hipsparseScsrgeam2_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseScsrgemm)
GPUMOD_LINK_CHECK(hipsparseScsrgemm2)
GPUMOD_LINK_CHECK(hipsparseScsrgemm2_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseScsric02)
GPUMOD_LINK_CHECK(hipsparseScsric02_analysis)
GPUMOD_LINK_CHECK(hipsparseScsric02_bufferSize)
GPUMOD_LINK_CHECK(hipsparseScsric02_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseScsrilu02)
GPUMOD_LINK_CHECK(hipsparseScsrilu02_analysis)
GPUMOD_LINK_CHECK(hipsparseScsrilu02_bufferSize)
GPUMOD_LINK_CHECK(hipsparseScsrilu02_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseScsrilu02_numericBoost)
GPUMOD_LINK_CHECK(hipsparseScsrmm)
GPUMOD_LINK_CHECK(hipsparseScsrmm2)
GPUMOD_LINK_CHECK(hipsparseScsrmv)
GPUMOD_LINK_CHECK(hipsparseScsrsm2_analysis)
GPUMOD_LINK_CHECK(hipsparseScsrsm2_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseScsrsm2_solve)
GPUMOD_LINK_CHECK(hipsparseScsrsv2_analysis)
GPUMOD_LINK_CHECK(hipsparseScsrsv2_bufferSize)
GPUMOD_LINK_CHECK(hipsparseScsrsv2_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseScsrsv2_solve)
GPUMOD_LINK_CHECK(hipsparseScsru2csr)
GPUMOD_LINK_CHECK(hipsparseScsru2csr_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseSdense2csc)
GPUMOD_LINK_CHECK(hipsparseSdense2csr)
GPUMOD_LINK_CHECK(hipsparseSdoti)
GPUMOD_LINK_CHECK(hipsparseSetMatDiagType)
GPUMOD_LINK_CHECK(hipsparseSetMatFillMode)
GPUMOD_LINK_CHECK(hipsparseSetMatIndexBase)
GPUMOD_LINK_CHECK(hipsparseSetMatType)
GPUMOD_LINK_CHECK(hipsparseSetPointerMode)
GPUMOD_LINK_CHECK(hipsparseSetStream)
GPUMOD_LINK_CHECK(hipsparseSgebsr2csr)
GPUMOD_LINK_CHECK(hipsparseSgebsr2gebsc)
GPUMOD_LINK_CHECK(hipsparseSgebsr2gebsc_bufferSize)
GPUMOD_LINK_CHECK(hipsparseSgebsr2gebsr)
GPUMOD_LINK_CHECK(hipsparseSgebsr2gebsr_bufferSize)
GPUMOD_LINK_CHECK(hipsparseSgemmi)
GPUMOD_LINK_CHECK(hipsparseSgemvi)
GPUMOD_LINK_CHECK(hipsparseSgemvi_bufferSize)
GPUMOD_LINK_CHECK(hipsparseSgpsvInterleavedBatch)
GPUMOD_LINK_CHECK(hipsparseSgpsvInterleavedBatch_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseSgthr)
GPUMOD_LINK_CHECK(hipsparseSgthrz)
GPUMOD_LINK_CHECK(hipsparseSgtsv2)
GPUMOD_LINK_CHECK(hipsparseSgtsv2StridedBatch)
GPUMOD_LINK_CHECK(hipsparseSgtsv2StridedBatch_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseSgtsv2_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseSgtsv2_nopivot)
GPUMOD_LINK_CHECK(hipsparseSgtsv2_nopivot_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseSgtsvInterleavedBatch)
GPUMOD_LINK_CHECK(hipsparseSgtsvInterleavedBatch_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseShyb2csr)
GPUMOD_LINK_CHECK(hipsparseShybmv)
GPUMOD_LINK_CHECK(hipsparseSnnz)
GPUMOD_LINK_CHECK(hipsparseSnnz_compress)
GPUMOD_LINK_CHECK(hipsparseSpGEMM_compute)
GPUMOD_LINK_CHECK(hipsparseSpGEMM_copy)
GPUMOD_LINK_CHECK(hipsparseSpGEMM_createDescr)
GPUMOD_LINK_CHECK(hipsparseSpGEMM_destroyDescr)
GPUMOD_LINK_CHECK(hipsparseSpGEMM_workEstimation)
GPUMOD_LINK_CHECK(hipsparseSpGEMMreuse_compute)
GPUMOD_LINK_CHECK(hipsparseSpGEMMreuse_copy)
GPUMOD_LINK_CHECK(hipsparseSpGEMMreuse_nnz)
GPUMOD_LINK_CHECK(hipsparseSpGEMMreuse_workEstimation)
GPUMOD_LINK_CHECK(hipsparseSpMM)
GPUMOD_LINK_CHECK(hipsparseSpMM_bufferSize)
GPUMOD_LINK_CHECK(hipsparseSpMM_preprocess)
GPUMOD_LINK_CHECK(hipsparseSpMV)
GPUMOD_LINK_CHECK(hipsparseSpMV_bufferSize)
GPUMOD_LINK_CHECK(hipsparseSpMV_preprocess)
GPUMOD_LINK_CHECK(hipsparseSpMatGetAttribute)
GPUMOD_LINK_CHECK(hipsparseSpMatGetFormat)
GPUMOD_LINK_CHECK(hipsparseSpMatGetIndexBase)
GPUMOD_LINK_CHECK(hipsparseSpMatGetSize)
GPUMOD_LINK_CHECK(hipsparseSpMatGetStridedBatch)
GPUMOD_LINK_CHECK(hipsparseSpMatGetValues)
GPUMOD_LINK_CHECK(hipsparseSpMatSetAttribute)
GPUMOD_LINK_CHECK(hipsparseSpMatSetStridedBatch)
GPUMOD_LINK_CHECK(hipsparseSpMatSetValues)
GPUMOD_LINK_CHECK(hipsparseSpSM_analysis)
GPUMOD_LINK_CHECK(hipsparseSpSM_bufferSize)
GPUMOD_LINK_CHECK(hipsparseSpSM_createDescr)
GPUMOD_LINK_CHECK(hipsparseSpSM_destroyDescr)
GPUMOD_LINK_CHECK(hipsparseSpSM_solve)
GPUMOD_LINK_CHECK(hipsparseSpSV_analysis)
GPUMOD_LINK_CHECK(hipsparseSpSV_bufferSize)
GPUMOD_LINK_CHECK(hipsparseSpSV_createDescr)
GPUMOD_LINK_CHECK(hipsparseSpSV_destroyDescr)
GPUMOD_LINK_CHECK(hipsparseSpSV_solve)
GPUMOD_LINK_CHECK(hipsparseSpVV)
GPUMOD_LINK_CHECK(hipsparseSpVV_bufferSize)
GPUMOD_LINK_CHECK(hipsparseSpVecGet)
GPUMOD_LINK_CHECK(hipsparseSpVecGetIndexBase)
GPUMOD_LINK_CHECK(hipsparseSpVecGetValues)
GPUMOD_LINK_CHECK(hipsparseSpVecSetValues)
GPUMOD_LINK_CHECK(hipsparseSparseToDense)
GPUMOD_LINK_CHECK(hipsparseSparseToDense_bufferSize)
GPUMOD_LINK_CHECK(hipsparseSpruneCsr2csr)
GPUMOD_LINK_CHECK(hipsparseSpruneCsr2csrByPercentage)
GPUMOD_LINK_CHECK(hipsparseSpruneCsr2csrByPercentage_bufferSize)
GPUMOD_LINK_CHECK(hipsparseSpruneCsr2csrByPercentage_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseSpruneCsr2csrNnz)
GPUMOD_LINK_CHECK(hipsparseSpruneCsr2csrNnzByPercentage)
GPUMOD_LINK_CHECK(hipsparseSpruneCsr2csr_bufferSize)
GPUMOD_LINK_CHECK(hipsparseSpruneCsr2csr_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseSpruneDense2csr)
GPUMOD_LINK_CHECK(hipsparseSpruneDense2csrByPercentage)
GPUMOD_LINK_CHECK(hipsparseSpruneDense2csrByPercentage_bufferSize)
GPUMOD_LINK_CHECK(hipsparseSpruneDense2csrByPercentage_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseSpruneDense2csrNnz)
GPUMOD_LINK_CHECK(hipsparseSpruneDense2csrNnzByPercentage)
GPUMOD_LINK_CHECK(hipsparseSpruneDense2csr_bufferSize)
GPUMOD_LINK_CHECK(hipsparseSpruneDense2csr_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseSroti)
GPUMOD_LINK_CHECK(hipsparseSsctr)
GPUMOD_LINK_CHECK(hipsparseXbsric02_zeroPivot)
GPUMOD_LINK_CHECK(hipsparseXbsrilu02_zeroPivot)
GPUMOD_LINK_CHECK(hipsparseXbsrsm2_zeroPivot)
GPUMOD_LINK_CHECK(hipsparseXbsrsv2_zeroPivot)
GPUMOD_LINK_CHECK(hipsparseXcoo2csr)
GPUMOD_LINK_CHECK(hipsparseXcoosortByColumn)
GPUMOD_LINK_CHECK(hipsparseXcoosortByRow)
GPUMOD_LINK_CHECK(hipsparseXcoosort_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseXcscsort)
GPUMOD_LINK_CHECK(hipsparseXcscsort_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseXcsr2bsrNnz)
GPUMOD_LINK_CHECK(hipsparseXcsr2coo)
GPUMOD_LINK_CHECK(hipsparseXcsr2gebsrNnz)
GPUMOD_LINK_CHECK(hipsparseXcsrgeam2Nnz)
GPUMOD_LINK_CHECK(hipsparseXcsrgeamNnz)
GPUMOD_LINK_CHECK(hipsparseXcsrgemm2Nnz)
GPUMOD_LINK_CHECK(hipsparseXcsrgemmNnz)
GPUMOD_LINK_CHECK(hipsparseXcsric02_zeroPivot)
GPUMOD_LINK_CHECK(hipsparseXcsrilu02_zeroPivot)
GPUMOD_LINK_CHECK(hipsparseXcsrsm2_zeroPivot)
GPUMOD_LINK_CHECK(hipsparseXcsrsort)
GPUMOD_LINK_CHECK(hipsparseXcsrsort_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseXcsrsv2_zeroPivot)
GPUMOD_LINK_CHECK(hipsparseXgebsr2gebsrNnz)
GPUMOD_LINK_CHECK(hipsparseZaxpyi)
GPUMOD_LINK_CHECK(hipsparseZbsr2csr)
GPUMOD_LINK_CHECK(hipsparseZbsric02)
GPUMOD_LINK_CHECK(hipsparseZbsric02_analysis)
GPUMOD_LINK_CHECK(hipsparseZbsric02_bufferSize)
GPUMOD_LINK_CHECK(hipsparseZbsrilu02)
GPUMOD_LINK_CHECK(hipsparseZbsrilu02_analysis)
GPUMOD_LINK_CHECK(hipsparseZbsrilu02_bufferSize)
GPUMOD_LINK_CHECK(hipsparseZbsrilu02_numericBoost)
GPUMOD_LINK_CHECK(hipsparseZbsrmm)
GPUMOD_LINK_CHECK(hipsparseZbsrmv)
GPUMOD_LINK_CHECK(hipsparseZbsrsm2_analysis)
GPUMOD_LINK_CHECK(hipsparseZbsrsm2_bufferSize)
GPUMOD_LINK_CHECK(hipsparseZbsrsm2_solve)
GPUMOD_LINK_CHECK(hipsparseZbsrsv2_analysis)
GPUMOD_LINK_CHECK(hipsparseZbsrsv2_bufferSize)
GPUMOD_LINK_CHECK(hipsparseZbsrsv2_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseZbsrsv2_solve)
GPUMOD_LINK_CHECK(hipsparseZbsrxmv)
GPUMOD_LINK_CHECK(hipsparseZcsc2dense)
GPUMOD_LINK_CHECK(hipsparseZcsr2bsr)
GPUMOD_LINK_CHECK(hipsparseZcsr2csc)
GPUMOD_LINK_CHECK(hipsparseZcsr2csr_compress)
GPUMOD_LINK_CHECK(hipsparseZcsr2csru)
GPUMOD_LINK_CHECK(hipsparseZcsr2dense)
GPUMOD_LINK_CHECK(hipsparseZcsr2gebsr)
GPUMOD_LINK_CHECK(hipsparseZcsr2gebsr_bufferSize)
GPUMOD_LINK_CHECK(hipsparseZcsr2hyb)
GPUMOD_LINK_CHECK(hipsparseZcsrcolor)
GPUMOD_LINK_CHECK(hipsparseZcsrgeam)
GPUMOD_LINK_CHECK(hipsparseZcsrgeam2)
GPUMOD_LINK_CHECK(hipsparseZcsrgeam2_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseZcsrgemm)
GPUMOD_LINK_CHECK(hipsparseZcsrgemm2)
GPUMOD_LINK_CHECK(hipsparseZcsrgemm2_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseZcsric02)
GPUMOD_LINK_CHECK(hipsparseZcsric02_analysis)
GPUMOD_LINK_CHECK(hipsparseZcsric02_bufferSize)
GPUMOD_LINK_CHECK(hipsparseZcsric02_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseZcsrilu02)
GPUMOD_LINK_CHECK(hipsparseZcsrilu02_analysis)
GPUMOD_LINK_CHECK(hipsparseZcsrilu02_bufferSize)
GPUMOD_LINK_CHECK(hipsparseZcsrilu02_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseZcsrilu02_numericBoost)
GPUMOD_LINK_CHECK(hipsparseZcsrmm)
GPUMOD_LINK_CHECK(hipsparseZcsrmm2)
GPUMOD_LINK_CHECK(hipsparseZcsrmv)
GPUMOD_LINK_CHECK(hipsparseZcsrsm2_analysis)
GPUMOD_LINK_CHECK(hipsparseZcsrsm2_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseZcsrsm2_solve)
GPUMOD_LINK_CHECK(hipsparseZcsrsv2_analysis)
GPUMOD_LINK_CHECK(hipsparseZcsrsv2_bufferSize)
GPUMOD_LINK_CHECK(hipsparseZcsrsv2_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseZcsrsv2_solve)
GPUMOD_LINK_CHECK(hipsparseZcsru2csr)
GPUMOD_LINK_CHECK(hipsparseZcsru2csr_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseZdense2csc)
GPUMOD_LINK_CHECK(hipsparseZdense2csr)
GPUMOD_LINK_CHECK(hipsparseZdotci)
GPUMOD_LINK_CHECK(hipsparseZdoti)
GPUMOD_LINK_CHECK(hipsparseZgebsr2csr)
GPUMOD_LINK_CHECK(hipsparseZgebsr2gebsc)
GPUMOD_LINK_CHECK(hipsparseZgebsr2gebsc_bufferSize)
GPUMOD_LINK_CHECK(hipsparseZgebsr2gebsr)
GPUMOD_LINK_CHECK(hipsparseZgebsr2gebsr_bufferSize)
GPUMOD_LINK_CHECK(hipsparseZgemmi)
GPUMOD_LINK_CHECK(hipsparseZgemvi)
GPUMOD_LINK_CHECK(hipsparseZgemvi_bufferSize)
GPUMOD_LINK_CHECK(hipsparseZgpsvInterleavedBatch)
GPUMOD_LINK_CHECK(hipsparseZgpsvInterleavedBatch_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseZgthr)
GPUMOD_LINK_CHECK(hipsparseZgthrz)
GPUMOD_LINK_CHECK(hipsparseZgtsv2)
GPUMOD_LINK_CHECK(hipsparseZgtsv2StridedBatch)
GPUMOD_LINK_CHECK(hipsparseZgtsv2StridedBatch_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseZgtsv2_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseZgtsv2_nopivot)
GPUMOD_LINK_CHECK(hipsparseZgtsv2_nopivot_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseZgtsvInterleavedBatch)
GPUMOD_LINK_CHECK(hipsparseZgtsvInterleavedBatch_bufferSizeExt)
GPUMOD_LINK_CHECK(hipsparseZhyb2csr)
GPUMOD_LINK_CHECK(hipsparseZhybmv)
GPUMOD_LINK_CHECK(hipsparseZnnz)
GPUMOD_LINK_CHECK(hipsparseZnnz_compress)
GPUMOD_LINK_CHECK(hipsparseZsctr)
} // namespace gpumod::hip::test
