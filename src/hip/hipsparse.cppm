/**
 * @file hipsparse.cppm
 * @brief hipSPARSE API module wrapper for wwr project
 *
 * Wraps hipsparse/hipsparse.h. CUDA counterpart: wwr.cuda.cusparse.
 *
 * hipsparse.h is an umbrella of #includes -- types, auxiliary, generic, and
 * one header per routine family under internal/{level1,level2,level3,extra,
 * precond,conversion,reorder,generic}/. The categorisation below follows that
 * directory layout, the granularity cusparse.cppm uses for its own.
 *
 * This TU never defines CUDART_VERSION, so every version-gated branch in
 * hipsparse-types.h resolves to hipSPARSE's own modern definitions rather than
 * a cuSPARSE-compatibility fallback. Every enumerator below is the current ABI.
 *
 * Absent: hipsparseBfloat16, referenced by no signature in the header and
 * skipped exactly as cublas_v2.cppm skips cublasBfloat16; and any counterpart
 * to cuSPARSE's Preview SpMM-with-custom-operators API, which hipSPARSE does
 * not have. Not stubbed, simply not exported -- recorded as documented
 * omissions in devtools/coverage_decisions.json.
 *
 * Usage:
 *   import wwr.hip.hipsparse;
 */

module;

// Pre-include <array> before the HIP header -- see src/hip/hip_complex.cppm's
// file header for why (amd_hip_vector_types.h, pulled in transitively via
// hip/hip_complex.h, #includes host_defines.h immediately before <array>,
// poisoning __has_attribute(__noinline__) for any later first-inclusion of
// <array> in the TU). Confirmed necessary here by direct experiment too.
// Load-bearing, and must stay before the HIP header: host_defines.h poisons
// __noinline__ for libc++'s __config. docs/architecture.md, section 9.
#include <array>
#include <hipsparse/hipsparse.h>

export module wwr.hip.hipsparse;

import std;

export namespace wwr::hip {

// ========================================================================
// Opaque Handle Types
// ========================================================================
using ::hipsparseColorInfo_t;
using ::hipsparseHandle_t;
using ::hipsparseHybMat_t;
using ::hipsparseMatDescr_t;

// Legacy opaque info types (deprecated but still present in the ABI)
using ::bsric02Info_t;
using ::bsrilu02Info_t;
using ::bsrsm2Info_t;
using ::bsrsv2Info_t;
using ::csrgemm2Info_t;
using ::csric02Info_t;
using ::csrilu02Info_t;
using ::csrsm2Info_t;
using ::csrsv2Info_t;
using ::csru2csrInfo_t;
using ::pruneInfo_t;

// Generic API descriptor types
using ::hipsparseConstDnMatDescr_t;
using ::hipsparseConstDnVecDescr_t;
using ::hipsparseConstSpMatDescr_t;
using ::hipsparseConstSpVecDescr_t;
using ::hipsparseDnMatDescr_t;
using ::hipsparseDnVecDescr_t;
using ::hipsparseSpMatDescr_t;
using ::hipsparseSpVecDescr_t;

// Algorithm descriptor types
using ::hipsparseSpGEMMDescr_t;
using ::hipsparseSpSMDescr_t;
using ::hipsparseSpSVDescr_t;

// ========================================================================
// Enumerations
// ========================================================================

using ::HIPSPARSE_STATUS_ALLOC_FAILED;
using ::HIPSPARSE_STATUS_ARCH_MISMATCH;
using ::HIPSPARSE_STATUS_EXECUTION_FAILED;
using ::HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES;
using ::HIPSPARSE_STATUS_INTERNAL_ERROR;
using ::HIPSPARSE_STATUS_INVALID_VALUE;
using ::HIPSPARSE_STATUS_MAPPING_ERROR;
using ::HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
using ::HIPSPARSE_STATUS_NOT_INITIALIZED;
using ::HIPSPARSE_STATUS_NOT_SUPPORTED;
using ::HIPSPARSE_STATUS_SUCCESS;
using ::HIPSPARSE_STATUS_ZERO_PIVOT;
using ::hipsparseStatus_t;

using ::HIPSPARSE_POINTER_MODE_DEVICE;
using ::HIPSPARSE_POINTER_MODE_HOST;
using ::hipsparsePointerMode_t;

using ::HIPSPARSE_ACTION_NUMERIC;
using ::HIPSPARSE_ACTION_SYMBOLIC;
using ::hipsparseAction_t;

using ::HIPSPARSE_MATRIX_TYPE_GENERAL;
using ::HIPSPARSE_MATRIX_TYPE_HERMITIAN;
using ::HIPSPARSE_MATRIX_TYPE_SYMMETRIC;
using ::HIPSPARSE_MATRIX_TYPE_TRIANGULAR;
using ::hipsparseMatrixType_t;

using ::HIPSPARSE_FILL_MODE_LOWER;
using ::HIPSPARSE_FILL_MODE_UPPER;
using ::hipsparseFillMode_t;

using ::HIPSPARSE_DIAG_TYPE_NON_UNIT;
using ::HIPSPARSE_DIAG_TYPE_UNIT;
using ::hipsparseDiagType_t;

using ::HIPSPARSE_INDEX_BASE_ONE;
using ::HIPSPARSE_INDEX_BASE_ZERO;
using ::hipsparseIndexBase_t;

using ::HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
using ::HIPSPARSE_OPERATION_NON_TRANSPOSE;
using ::HIPSPARSE_OPERATION_TRANSPOSE;
using ::hipsparseOperation_t;

using ::HIPSPARSE_HYB_PARTITION_AUTO;
using ::HIPSPARSE_HYB_PARTITION_MAX;
using ::HIPSPARSE_HYB_PARTITION_USER;
using ::hipsparseHybPartition_t;

using ::HIPSPARSE_SOLVE_POLICY_NO_LEVEL;
using ::HIPSPARSE_SOLVE_POLICY_USE_LEVEL;
using ::hipsparseSolvePolicy_t;

using ::HIPSPARSE_SIDE_LEFT;
using ::HIPSPARSE_SIDE_RIGHT;
using ::hipsparseSideMode_t;

using ::HIPSPARSE_DIRECTION_COLUMN;
using ::HIPSPARSE_DIRECTION_ROW;
using ::hipsparseDirection_t;

using ::HIPSPARSE_CSR2CSC_ALG1;
using ::HIPSPARSE_CSR2CSC_ALG2;
using ::HIPSPARSE_CSR2CSC_ALG_DEFAULT;
using ::hipsparseCsr2CscAlg_t;

using ::HIPSPARSE_FORMAT_BLOCKED_ELL;
using ::HIPSPARSE_FORMAT_COO;
using ::HIPSPARSE_FORMAT_COO_AOS;
using ::HIPSPARSE_FORMAT_CSC;
using ::HIPSPARSE_FORMAT_CSR;
using ::hipsparseFormat_t;

using ::HIPSPARSE_ORDER_COL;
using ::HIPSPARSE_ORDER_COLUMN;
using ::HIPSPARSE_ORDER_ROW;
using ::hipsparseOrder_t;

using ::HIPSPARSE_INDEX_16U;
using ::HIPSPARSE_INDEX_32I;
using ::HIPSPARSE_INDEX_64I;
using ::hipsparseIndexType_t;

using ::HIPSPARSE_COOMV_ALG;
using ::HIPSPARSE_CSRMV_ALG1;
using ::HIPSPARSE_CSRMV_ALG2;
using ::HIPSPARSE_MV_ALG_DEFAULT;
using ::HIPSPARSE_SPMV_ALG_DEFAULT;
using ::HIPSPARSE_SPMV_COO_ALG1;
using ::HIPSPARSE_SPMV_COO_ALG2;
using ::HIPSPARSE_SPMV_CSR_ALG1;
using ::HIPSPARSE_SPMV_CSR_ALG2;
using ::hipsparseSpMVAlg_t;

using ::HIPSPARSE_COOMM_ALG1;
using ::HIPSPARSE_COOMM_ALG2;
using ::HIPSPARSE_COOMM_ALG3;
using ::HIPSPARSE_CSRMM_ALG1;
using ::HIPSPARSE_MM_ALG_DEFAULT;
using ::HIPSPARSE_SPMM_ALG_DEFAULT;
using ::HIPSPARSE_SPMM_BLOCKED_ELL_ALG1;
using ::HIPSPARSE_SPMM_COO_ALG1;
using ::HIPSPARSE_SPMM_COO_ALG2;
using ::HIPSPARSE_SPMM_COO_ALG3;
using ::HIPSPARSE_SPMM_COO_ALG4;
using ::HIPSPARSE_SPMM_CSR_ALG1;
using ::HIPSPARSE_SPMM_CSR_ALG2;
using ::HIPSPARSE_SPMM_CSR_ALG3;
using ::hipsparseSpMMAlg_t;

using ::HIPSPARSE_SPARSETODENSE_ALG_DEFAULT;
using ::hipsparseSparseToDenseAlg_t;

using ::HIPSPARSE_DENSETOSPARSE_ALG_DEFAULT;
using ::hipsparseDenseToSparseAlg_t;

using ::HIPSPARSE_SDDMM_ALG_DEFAULT;
using ::hipsparseSDDMMAlg_t;

using ::HIPSPARSE_SPSV_ALG_DEFAULT;
using ::hipsparseSpSVAlg_t;

using ::HIPSPARSE_SPSM_ALG_DEFAULT;
using ::hipsparseSpSMAlg_t;

using ::HIPSPARSE_SPMAT_DIAG_TYPE;
using ::HIPSPARSE_SPMAT_FILL_MODE;
using ::hipsparseSpMatAttribute_t;

using ::HIPSPARSE_SPGEMM_ALG1;
using ::HIPSPARSE_SPGEMM_ALG2;
using ::HIPSPARSE_SPGEMM_ALG3;
using ::HIPSPARSE_SPGEMM_CSR_ALG_DETERMINISTIC;
using ::HIPSPARSE_SPGEMM_CSR_ALG_NONDETERMINISTIC;
using ::HIPSPARSE_SPGEMM_DEFAULT;
using ::hipsparseSpGEMMAlg_t;

// ========================================================================
// Library, Context, and Descriptor Management (handle, MatDescr, generic-API descriptor get/set, info object create/destroy)
// ========================================================================
using ::hipsparseBlockedEllGet;
using ::hipsparseConstBlockedEllGet;
using ::hipsparseConstCooGet;
using ::hipsparseConstCscGet;
using ::hipsparseConstCsrGet;
using ::hipsparseConstDnMatGet;
using ::hipsparseConstDnMatGetValues;
using ::hipsparseConstDnVecGet;
using ::hipsparseConstDnVecGetValues;
using ::hipsparseConstSpMatGetValues;
using ::hipsparseConstSpVecGet;
using ::hipsparseConstSpVecGetValues;
using ::hipsparseCooAoSGet;
using ::hipsparseCooGet;
using ::hipsparseCooSetPointers;
using ::hipsparseCooSetStridedBatch;
using ::hipsparseCopyMatDescr;
using ::hipsparseCreate;
using ::hipsparseCreateBlockedEll;
using ::hipsparseCreateBsric02Info;
using ::hipsparseCreateBsrilu02Info;
using ::hipsparseCreateBsrsm2Info;
using ::hipsparseCreateBsrsv2Info;
using ::hipsparseCreateColorInfo;
using ::hipsparseCreateConstBlockedEll;
using ::hipsparseCreateConstCoo;
using ::hipsparseCreateConstCsc;
using ::hipsparseCreateConstCsr;
using ::hipsparseCreateConstDnMat;
using ::hipsparseCreateConstDnVec;
using ::hipsparseCreateConstSpVec;
using ::hipsparseCreateCoo;
using ::hipsparseCreateCooAoS;
using ::hipsparseCreateCsc;
using ::hipsparseCreateCsr;
using ::hipsparseCreateCsrgemm2Info;
using ::hipsparseCreateCsric02Info;
using ::hipsparseCreateCsrilu02Info;
using ::hipsparseCreateCsrsm2Info;
using ::hipsparseCreateCsrsv2Info;
using ::hipsparseCreateCsru2csrInfo;
using ::hipsparseCreateDnMat;
using ::hipsparseCreateDnVec;
using ::hipsparseCreateHybMat;
using ::hipsparseCreateMatDescr;
using ::hipsparseCreatePruneInfo;
using ::hipsparseCreateSpVec;
using ::hipsparseCscGet;
using ::hipsparseCscSetPointers;
using ::hipsparseCsrGet;
using ::hipsparseCsrSetPointers;
using ::hipsparseCsrSetStridedBatch;
using ::hipsparseDestroy;
using ::hipsparseDestroyBsric02Info;
using ::hipsparseDestroyBsrilu02Info;
using ::hipsparseDestroyBsrsm2Info;
using ::hipsparseDestroyBsrsv2Info;
using ::hipsparseDestroyColorInfo;
using ::hipsparseDestroyCsrgemm2Info;
using ::hipsparseDestroyCsric02Info;
using ::hipsparseDestroyCsrilu02Info;
using ::hipsparseDestroyCsrsm2Info;
using ::hipsparseDestroyCsrsv2Info;
using ::hipsparseDestroyCsru2csrInfo;
using ::hipsparseDestroyDnMat;
using ::hipsparseDestroyDnVec;
using ::hipsparseDestroyHybMat;
using ::hipsparseDestroyMatDescr;
using ::hipsparseDestroyPruneInfo;
using ::hipsparseDestroySpMat;
using ::hipsparseDestroySpVec;
using ::hipsparseDnMatGet;
using ::hipsparseDnMatGetStridedBatch;
using ::hipsparseDnMatGetValues;
using ::hipsparseDnMatSetStridedBatch;
using ::hipsparseDnMatSetValues;
using ::hipsparseDnVecGet;
using ::hipsparseDnVecGetValues;
using ::hipsparseDnVecSetValues;
using ::hipsparseGetErrorName;
using ::hipsparseGetErrorString;
using ::hipsparseGetGitRevision;
using ::hipsparseGetMatDiagType;
using ::hipsparseGetMatFillMode;
using ::hipsparseGetMatIndexBase;
using ::hipsparseGetMatType;
using ::hipsparseGetPointerMode;
using ::hipsparseGetStream;
using ::hipsparseGetVersion;
using ::hipsparseSetMatDiagType;
using ::hipsparseSetMatFillMode;
using ::hipsparseSetMatIndexBase;
using ::hipsparseSetMatType;
using ::hipsparseSetPointerMode;
using ::hipsparseSetStream;
using ::hipsparseSpMatGetAttribute;
using ::hipsparseSpMatGetFormat;
using ::hipsparseSpMatGetIndexBase;
using ::hipsparseSpMatGetSize;
using ::hipsparseSpMatGetStridedBatch;
using ::hipsparseSpMatGetValues;
using ::hipsparseSpMatSetAttribute;
using ::hipsparseSpMatSetStridedBatch;
using ::hipsparseSpMatSetValues;
using ::hipsparseSpVecGet;
using ::hipsparseSpVecGetIndexBase;
using ::hipsparseSpVecGetValues;
using ::hipsparseSpVecSetValues;

// ========================================================================
// Sparse Level 1
// ========================================================================
using ::hipsparseCaxpyi;
using ::hipsparseCdotci;
using ::hipsparseCdoti;
using ::hipsparseCgthr;
using ::hipsparseCgthrz;
using ::hipsparseCsctr;
using ::hipsparseDaxpyi;
using ::hipsparseDdoti;
using ::hipsparseDgthr;
using ::hipsparseDgthrz;
using ::hipsparseDroti;
using ::hipsparseDsctr;
using ::hipsparseSaxpyi;
using ::hipsparseSdoti;
using ::hipsparseSgthr;
using ::hipsparseSgthrz;
using ::hipsparseSroti;
using ::hipsparseSsctr;
using ::hipsparseZaxpyi;
using ::hipsparseZdotci;
using ::hipsparseZdoti;
using ::hipsparseZgthr;
using ::hipsparseZgthrz;
using ::hipsparseZsctr;

// ========================================================================
// Sparse Level 2
// ========================================================================
using ::hipsparseCbsrmv;
using ::hipsparseCbsrsv2_analysis;
using ::hipsparseCbsrsv2_bufferSize;
using ::hipsparseCbsrsv2_bufferSizeExt;
using ::hipsparseCbsrsv2_solve;
using ::hipsparseCbsrxmv;
using ::hipsparseCcsrmv;
using ::hipsparseCcsrsv2_analysis;
using ::hipsparseCcsrsv2_bufferSize;
using ::hipsparseCcsrsv2_bufferSizeExt;
using ::hipsparseCcsrsv2_solve;
using ::hipsparseCgemvi;
using ::hipsparseCgemvi_bufferSize;
using ::hipsparseChybmv;
using ::hipsparseDbsrmv;
using ::hipsparseDbsrsv2_analysis;
using ::hipsparseDbsrsv2_bufferSize;
using ::hipsparseDbsrsv2_bufferSizeExt;
using ::hipsparseDbsrsv2_solve;
using ::hipsparseDbsrxmv;
using ::hipsparseDcsrmv;
using ::hipsparseDcsrsv2_analysis;
using ::hipsparseDcsrsv2_bufferSize;
using ::hipsparseDcsrsv2_bufferSizeExt;
using ::hipsparseDcsrsv2_solve;
using ::hipsparseDgemvi;
using ::hipsparseDgemvi_bufferSize;
using ::hipsparseDhybmv;
using ::hipsparseSbsrmv;
using ::hipsparseSbsrsv2_analysis;
using ::hipsparseSbsrsv2_bufferSize;
using ::hipsparseSbsrsv2_bufferSizeExt;
using ::hipsparseSbsrsv2_solve;
using ::hipsparseSbsrxmv;
using ::hipsparseScsrmv;
using ::hipsparseScsrsv2_analysis;
using ::hipsparseScsrsv2_bufferSize;
using ::hipsparseScsrsv2_bufferSizeExt;
using ::hipsparseScsrsv2_solve;
using ::hipsparseSgemvi;
using ::hipsparseSgemvi_bufferSize;
using ::hipsparseShybmv;
using ::hipsparseXbsrsv2_zeroPivot;
using ::hipsparseXcsrsv2_zeroPivot;
using ::hipsparseZbsrmv;
using ::hipsparseZbsrsv2_analysis;
using ::hipsparseZbsrsv2_bufferSize;
using ::hipsparseZbsrsv2_bufferSizeExt;
using ::hipsparseZbsrsv2_solve;
using ::hipsparseZbsrxmv;
using ::hipsparseZcsrmv;
using ::hipsparseZcsrsv2_analysis;
using ::hipsparseZcsrsv2_bufferSize;
using ::hipsparseZcsrsv2_bufferSizeExt;
using ::hipsparseZcsrsv2_solve;
using ::hipsparseZgemvi;
using ::hipsparseZgemvi_bufferSize;
using ::hipsparseZhybmv;

// ========================================================================
// Sparse Level 3
// ========================================================================
using ::hipsparseCbsrmm;
using ::hipsparseCbsrsm2_analysis;
using ::hipsparseCbsrsm2_bufferSize;
using ::hipsparseCbsrsm2_solve;
using ::hipsparseCcsrmm;
using ::hipsparseCcsrmm2;
using ::hipsparseCcsrsm2_analysis;
using ::hipsparseCcsrsm2_bufferSizeExt;
using ::hipsparseCcsrsm2_solve;
using ::hipsparseCgemmi;
using ::hipsparseDbsrmm;
using ::hipsparseDbsrsm2_analysis;
using ::hipsparseDbsrsm2_bufferSize;
using ::hipsparseDbsrsm2_solve;
using ::hipsparseDcsrmm;
using ::hipsparseDcsrmm2;
using ::hipsparseDcsrsm2_analysis;
using ::hipsparseDcsrsm2_bufferSizeExt;
using ::hipsparseDcsrsm2_solve;
using ::hipsparseDgemmi;
using ::hipsparseSbsrmm;
using ::hipsparseSbsrsm2_analysis;
using ::hipsparseSbsrsm2_bufferSize;
using ::hipsparseSbsrsm2_solve;
using ::hipsparseScsrmm;
using ::hipsparseScsrmm2;
using ::hipsparseScsrsm2_analysis;
using ::hipsparseScsrsm2_bufferSizeExt;
using ::hipsparseScsrsm2_solve;
using ::hipsparseSgemmi;
using ::hipsparseXbsrsm2_zeroPivot;
using ::hipsparseXcsrsm2_zeroPivot;
using ::hipsparseZbsrmm;
using ::hipsparseZbsrsm2_analysis;
using ::hipsparseZbsrsm2_bufferSize;
using ::hipsparseZbsrsm2_solve;
using ::hipsparseZcsrmm;
using ::hipsparseZcsrmm2;
using ::hipsparseZcsrsm2_analysis;
using ::hipsparseZcsrsm2_bufferSizeExt;
using ::hipsparseZcsrsm2_solve;
using ::hipsparseZgemmi;

// ========================================================================
// Extra Routines (CSR Matrix Addition / Multiplication)
// ========================================================================
using ::hipsparseCcsrgeam;
using ::hipsparseCcsrgeam2;
using ::hipsparseCcsrgeam2_bufferSizeExt;
using ::hipsparseCcsrgemm;
using ::hipsparseCcsrgemm2;
using ::hipsparseCcsrgemm2_bufferSizeExt;
using ::hipsparseDcsrgeam;
using ::hipsparseDcsrgeam2;
using ::hipsparseDcsrgeam2_bufferSizeExt;
using ::hipsparseDcsrgemm;
using ::hipsparseDcsrgemm2;
using ::hipsparseDcsrgemm2_bufferSizeExt;
using ::hipsparseScsrgeam;
using ::hipsparseScsrgeam2;
using ::hipsparseScsrgeam2_bufferSizeExt;
using ::hipsparseScsrgemm;
using ::hipsparseScsrgemm2;
using ::hipsparseScsrgemm2_bufferSizeExt;
using ::hipsparseXcsrgeam2Nnz;
using ::hipsparseXcsrgeamNnz;
using ::hipsparseXcsrgemm2Nnz;
using ::hipsparseXcsrgemmNnz;
using ::hipsparseZcsrgeam;
using ::hipsparseZcsrgeam2;
using ::hipsparseZcsrgeam2_bufferSizeExt;
using ::hipsparseZcsrgemm;
using ::hipsparseZcsrgemm2;
using ::hipsparseZcsrgemm2_bufferSizeExt;

// ========================================================================
// Preconditioners (ILU0/IC0, tridiagonal/pentadiagonal solvers)
// ========================================================================
using ::hipsparseCbsric02;
using ::hipsparseCbsric02_analysis;
using ::hipsparseCbsric02_bufferSize;
using ::hipsparseCbsrilu02;
using ::hipsparseCbsrilu02_analysis;
using ::hipsparseCbsrilu02_bufferSize;
using ::hipsparseCbsrilu02_numericBoost;
using ::hipsparseCcsric02;
using ::hipsparseCcsric02_analysis;
using ::hipsparseCcsric02_bufferSize;
using ::hipsparseCcsric02_bufferSizeExt;
using ::hipsparseCcsrilu02;
using ::hipsparseCcsrilu02_analysis;
using ::hipsparseCcsrilu02_bufferSize;
using ::hipsparseCcsrilu02_bufferSizeExt;
using ::hipsparseCcsrilu02_numericBoost;
using ::hipsparseCgpsvInterleavedBatch;
using ::hipsparseCgpsvInterleavedBatch_bufferSizeExt;
using ::hipsparseCgtsv2;
using ::hipsparseCgtsv2_bufferSizeExt;
using ::hipsparseCgtsv2_nopivot;
using ::hipsparseCgtsv2_nopivot_bufferSizeExt;
using ::hipsparseCgtsv2StridedBatch;
using ::hipsparseCgtsv2StridedBatch_bufferSizeExt;
using ::hipsparseCgtsvInterleavedBatch;
using ::hipsparseCgtsvInterleavedBatch_bufferSizeExt;
using ::hipsparseDbsric02;
using ::hipsparseDbsric02_analysis;
using ::hipsparseDbsric02_bufferSize;
using ::hipsparseDbsrilu02;
using ::hipsparseDbsrilu02_analysis;
using ::hipsparseDbsrilu02_bufferSize;
using ::hipsparseDbsrilu02_numericBoost;
using ::hipsparseDcsric02;
using ::hipsparseDcsric02_analysis;
using ::hipsparseDcsric02_bufferSize;
using ::hipsparseDcsric02_bufferSizeExt;
using ::hipsparseDcsrilu02;
using ::hipsparseDcsrilu02_analysis;
using ::hipsparseDcsrilu02_bufferSize;
using ::hipsparseDcsrilu02_bufferSizeExt;
using ::hipsparseDcsrilu02_numericBoost;
using ::hipsparseDgpsvInterleavedBatch;
using ::hipsparseDgpsvInterleavedBatch_bufferSizeExt;
using ::hipsparseDgtsv2;
using ::hipsparseDgtsv2_bufferSizeExt;
using ::hipsparseDgtsv2_nopivot;
using ::hipsparseDgtsv2_nopivot_bufferSizeExt;
using ::hipsparseDgtsv2StridedBatch;
using ::hipsparseDgtsv2StridedBatch_bufferSizeExt;
using ::hipsparseDgtsvInterleavedBatch;
using ::hipsparseDgtsvInterleavedBatch_bufferSizeExt;
using ::hipsparseSbsric02;
using ::hipsparseSbsric02_analysis;
using ::hipsparseSbsric02_bufferSize;
using ::hipsparseSbsrilu02;
using ::hipsparseSbsrilu02_analysis;
using ::hipsparseSbsrilu02_bufferSize;
using ::hipsparseSbsrilu02_numericBoost;
using ::hipsparseScsric02;
using ::hipsparseScsric02_analysis;
using ::hipsparseScsric02_bufferSize;
using ::hipsparseScsric02_bufferSizeExt;
using ::hipsparseScsrilu02;
using ::hipsparseScsrilu02_analysis;
using ::hipsparseScsrilu02_bufferSize;
using ::hipsparseScsrilu02_bufferSizeExt;
using ::hipsparseScsrilu02_numericBoost;
using ::hipsparseSgpsvInterleavedBatch;
using ::hipsparseSgpsvInterleavedBatch_bufferSizeExt;
using ::hipsparseSgtsv2;
using ::hipsparseSgtsv2_bufferSizeExt;
using ::hipsparseSgtsv2_nopivot;
using ::hipsparseSgtsv2_nopivot_bufferSizeExt;
using ::hipsparseSgtsv2StridedBatch;
using ::hipsparseSgtsv2StridedBatch_bufferSizeExt;
using ::hipsparseSgtsvInterleavedBatch;
using ::hipsparseSgtsvInterleavedBatch_bufferSizeExt;
using ::hipsparseXbsric02_zeroPivot;
using ::hipsparseXbsrilu02_zeroPivot;
using ::hipsparseXcsric02_zeroPivot;
using ::hipsparseXcsrilu02_zeroPivot;
using ::hipsparseZbsric02;
using ::hipsparseZbsric02_analysis;
using ::hipsparseZbsric02_bufferSize;
using ::hipsparseZbsrilu02;
using ::hipsparseZbsrilu02_analysis;
using ::hipsparseZbsrilu02_bufferSize;
using ::hipsparseZbsrilu02_numericBoost;
using ::hipsparseZcsric02;
using ::hipsparseZcsric02_analysis;
using ::hipsparseZcsric02_bufferSize;
using ::hipsparseZcsric02_bufferSizeExt;
using ::hipsparseZcsrilu02;
using ::hipsparseZcsrilu02_analysis;
using ::hipsparseZcsrilu02_bufferSize;
using ::hipsparseZcsrilu02_bufferSizeExt;
using ::hipsparseZcsrilu02_numericBoost;
using ::hipsparseZgpsvInterleavedBatch;
using ::hipsparseZgpsvInterleavedBatch_bufferSizeExt;
using ::hipsparseZgtsv2;
using ::hipsparseZgtsv2_bufferSizeExt;
using ::hipsparseZgtsv2_nopivot;
using ::hipsparseZgtsv2_nopivot_bufferSizeExt;
using ::hipsparseZgtsv2StridedBatch;
using ::hipsparseZgtsv2StridedBatch_bufferSizeExt;
using ::hipsparseZgtsvInterleavedBatch;
using ::hipsparseZgtsvInterleavedBatch_bufferSizeExt;

// ========================================================================
// Sparse Format Conversion
// ========================================================================
using ::hipsparseCbsr2csr;
using ::hipsparseCcsc2dense;
using ::hipsparseCcsr2bsr;
using ::hipsparseCcsr2csc;
using ::hipsparseCcsr2csr_compress;
using ::hipsparseCcsr2csru;
using ::hipsparseCcsr2dense;
using ::hipsparseCcsr2gebsr;
using ::hipsparseCcsr2gebsr_bufferSize;
using ::hipsparseCcsr2hyb;
using ::hipsparseCcsru2csr;
using ::hipsparseCcsru2csr_bufferSizeExt;
using ::hipsparseCdense2csc;
using ::hipsparseCdense2csr;
using ::hipsparseCgebsr2csr;
using ::hipsparseCgebsr2gebsc;
using ::hipsparseCgebsr2gebsc_bufferSize;
using ::hipsparseCgebsr2gebsr;
using ::hipsparseCgebsr2gebsr_bufferSize;
using ::hipsparseChyb2csr;
using ::hipsparseCnnz;
using ::hipsparseCnnz_compress;
using ::hipsparseCreateIdentityPermutation;
using ::hipsparseCsr2cscEx2;
using ::hipsparseCsr2cscEx2_bufferSize;
using ::hipsparseDbsr2csr;
using ::hipsparseDcsc2dense;
using ::hipsparseDcsr2bsr;
using ::hipsparseDcsr2csc;
using ::hipsparseDcsr2csr_compress;
using ::hipsparseDcsr2csru;
using ::hipsparseDcsr2dense;
using ::hipsparseDcsr2gebsr;
using ::hipsparseDcsr2gebsr_bufferSize;
using ::hipsparseDcsr2hyb;
using ::hipsparseDcsru2csr;
using ::hipsparseDcsru2csr_bufferSizeExt;
using ::hipsparseDdense2csc;
using ::hipsparseDdense2csr;
using ::hipsparseDgebsr2csr;
using ::hipsparseDgebsr2gebsc;
using ::hipsparseDgebsr2gebsc_bufferSize;
using ::hipsparseDgebsr2gebsr;
using ::hipsparseDgebsr2gebsr_bufferSize;
using ::hipsparseDhyb2csr;
using ::hipsparseDnnz;
using ::hipsparseDnnz_compress;
using ::hipsparseDpruneCsr2csr;
using ::hipsparseDpruneCsr2csr_bufferSize;
using ::hipsparseDpruneCsr2csr_bufferSizeExt;
using ::hipsparseDpruneCsr2csrByPercentage;
using ::hipsparseDpruneCsr2csrByPercentage_bufferSize;
using ::hipsparseDpruneCsr2csrByPercentage_bufferSizeExt;
using ::hipsparseDpruneCsr2csrNnz;
using ::hipsparseDpruneCsr2csrNnzByPercentage;
using ::hipsparseDpruneDense2csr;
using ::hipsparseDpruneDense2csr_bufferSize;
using ::hipsparseDpruneDense2csr_bufferSizeExt;
using ::hipsparseDpruneDense2csrByPercentage;
using ::hipsparseDpruneDense2csrByPercentage_bufferSize;
using ::hipsparseDpruneDense2csrByPercentage_bufferSizeExt;
using ::hipsparseDpruneDense2csrNnz;
using ::hipsparseDpruneDense2csrNnzByPercentage;
using ::hipsparseSbsr2csr;
using ::hipsparseScsc2dense;
using ::hipsparseScsr2bsr;
using ::hipsparseScsr2csc;
using ::hipsparseScsr2csr_compress;
using ::hipsparseScsr2csru;
using ::hipsparseScsr2dense;
using ::hipsparseScsr2gebsr;
using ::hipsparseScsr2gebsr_bufferSize;
using ::hipsparseScsr2hyb;
using ::hipsparseScsru2csr;
using ::hipsparseScsru2csr_bufferSizeExt;
using ::hipsparseSdense2csc;
using ::hipsparseSdense2csr;
using ::hipsparseSgebsr2csr;
using ::hipsparseSgebsr2gebsc;
using ::hipsparseSgebsr2gebsc_bufferSize;
using ::hipsparseSgebsr2gebsr;
using ::hipsparseSgebsr2gebsr_bufferSize;
using ::hipsparseShyb2csr;
using ::hipsparseSnnz;
using ::hipsparseSnnz_compress;
using ::hipsparseSpruneCsr2csr;
using ::hipsparseSpruneCsr2csr_bufferSize;
using ::hipsparseSpruneCsr2csr_bufferSizeExt;
using ::hipsparseSpruneCsr2csrByPercentage;
using ::hipsparseSpruneCsr2csrByPercentage_bufferSize;
using ::hipsparseSpruneCsr2csrByPercentage_bufferSizeExt;
using ::hipsparseSpruneCsr2csrNnz;
using ::hipsparseSpruneCsr2csrNnzByPercentage;
using ::hipsparseSpruneDense2csr;
using ::hipsparseSpruneDense2csr_bufferSize;
using ::hipsparseSpruneDense2csr_bufferSizeExt;
using ::hipsparseSpruneDense2csrByPercentage;
using ::hipsparseSpruneDense2csrByPercentage_bufferSize;
using ::hipsparseSpruneDense2csrByPercentage_bufferSizeExt;
using ::hipsparseSpruneDense2csrNnz;
using ::hipsparseSpruneDense2csrNnzByPercentage;
using ::hipsparseXcoo2csr;
using ::hipsparseXcoosort_bufferSizeExt;
using ::hipsparseXcoosortByColumn;
using ::hipsparseXcoosortByRow;
using ::hipsparseXcscsort;
using ::hipsparseXcscsort_bufferSizeExt;
using ::hipsparseXcsr2bsrNnz;
using ::hipsparseXcsr2coo;
using ::hipsparseXcsr2gebsrNnz;
using ::hipsparseXcsrsort;
using ::hipsparseXcsrsort_bufferSizeExt;
using ::hipsparseXgebsr2gebsrNnz;
using ::hipsparseZbsr2csr;
using ::hipsparseZcsc2dense;
using ::hipsparseZcsr2bsr;
using ::hipsparseZcsr2csc;
using ::hipsparseZcsr2csr_compress;
using ::hipsparseZcsr2csru;
using ::hipsparseZcsr2dense;
using ::hipsparseZcsr2gebsr;
using ::hipsparseZcsr2gebsr_bufferSize;
using ::hipsparseZcsr2hyb;
using ::hipsparseZcsru2csr;
using ::hipsparseZcsru2csr_bufferSizeExt;
using ::hipsparseZdense2csc;
using ::hipsparseZdense2csr;
using ::hipsparseZgebsr2csr;
using ::hipsparseZgebsr2gebsc;
using ::hipsparseZgebsr2gebsc_bufferSize;
using ::hipsparseZgebsr2gebsr;
using ::hipsparseZgebsr2gebsr_bufferSize;
using ::hipsparseZhyb2csr;
using ::hipsparseZnnz;
using ::hipsparseZnnz_compress;

// ========================================================================
// Reordering
// ========================================================================
using ::hipsparseCcsrcolor;
using ::hipsparseDcsrcolor;
using ::hipsparseScsrcolor;
using ::hipsparseZcsrcolor;

// ========================================================================
// Generic API (SpVec/DnVec/SpMat/DnMat operations: SpMV, SpSV, SpSM, SpMM, SpGEMM, SDDMM, sparse<->dense, axpby, gather/scatter, rot)
// ========================================================================
using ::hipsparseAxpby;
using ::hipsparseDenseToSparse_analysis;
using ::hipsparseDenseToSparse_bufferSize;
using ::hipsparseDenseToSparse_convert;
using ::hipsparseGather;
using ::hipsparseRot;
using ::hipsparseScatter;
using ::hipsparseSDDMM;
using ::hipsparseSDDMM_bufferSize;
using ::hipsparseSDDMM_preprocess;
using ::hipsparseSparseToDense;
using ::hipsparseSparseToDense_bufferSize;
using ::hipsparseSpGEMM_compute;
using ::hipsparseSpGEMM_copy;
using ::hipsparseSpGEMM_createDescr;
using ::hipsparseSpGEMM_destroyDescr;
using ::hipsparseSpGEMM_workEstimation;
using ::hipsparseSpGEMMreuse_compute;
using ::hipsparseSpGEMMreuse_copy;
using ::hipsparseSpGEMMreuse_nnz;
using ::hipsparseSpGEMMreuse_workEstimation;
using ::hipsparseSpMM;
using ::hipsparseSpMM_bufferSize;
using ::hipsparseSpMM_preprocess;
using ::hipsparseSpMV;
using ::hipsparseSpMV_bufferSize;
using ::hipsparseSpMV_preprocess;
using ::hipsparseSpSM_analysis;
using ::hipsparseSpSM_bufferSize;
using ::hipsparseSpSM_createDescr;
using ::hipsparseSpSM_destroyDescr;
using ::hipsparseSpSM_solve;
using ::hipsparseSpSV_analysis;
using ::hipsparseSpSV_bufferSize;
using ::hipsparseSpSV_createDescr;
using ::hipsparseSpSV_destroyDescr;
using ::hipsparseSpSV_solve;
using ::hipsparseSpVV;
using ::hipsparseSpVV_bufferSize;
} // namespace wwr::hip
