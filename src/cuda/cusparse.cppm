/**
 * @file cusparse.cppm
 * @brief Primary interface for wwr.cuda.cusparse
 *
 * This module wraps the native cuSPARSE API and exports types, constants,
 * and functions for cuSPARSE library management and operations.
 *
 * Usage:
 *   import wwr.cuda.cusparse;
 */

module;

#include <cusparse.h>

export module wwr.cuda.cusparse;

export namespace wwr::cuda {

// ========================================================================
// Opaque Handle Types
// ========================================================================
using ::cusparseHandle_t;
using ::cusparseMatDescr_t;

// Legacy opaque info types (deprecated but still present in the ABI)
using ::bsric02Info_t;
using ::bsrilu02Info_t;
using ::bsrsm2Info_t;
using ::bsrsv2Info_t;
using ::csric02Info_t;
using ::csrilu02Info_t;
using ::csru2csrInfo_t;
using ::cusparseColorInfo_t;
using ::pruneInfo_t;

// Generic API descriptor types
using ::cusparseConstDnMatDescr_t;
using ::cusparseConstDnVecDescr_t;
using ::cusparseConstSpMatDescr_t;
using ::cusparseConstSpVecDescr_t;
using ::cusparseDnMatDescr_t;
using ::cusparseDnVecDescr_t;
using ::cusparseSpMatDescr_t;
using ::cusparseSpVecDescr_t;

// Algorithm descriptor types
using ::cusparseSpGEMMDescr_t;
using ::cusparseSpMMOpPlan_t;
using ::cusparseSpSMDescr_t;
using ::cusparseSpSVDescr_t;

// ========================================================================
// Enumerations
// ========================================================================

using ::cusparseStatus_t;
// cusparseStatus_t enumerators
using ::CUSPARSE_STATUS_ALLOC_FAILED;
using ::CUSPARSE_STATUS_ARCH_MISMATCH;
using ::CUSPARSE_STATUS_EXECUTION_FAILED;
using ::CUSPARSE_STATUS_INSUFFICIENT_RESOURCES;
using ::CUSPARSE_STATUS_INTERNAL_ERROR;
using ::CUSPARSE_STATUS_INVALID_VALUE;
using ::CUSPARSE_STATUS_MAPPING_ERROR;
using ::CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
using ::CUSPARSE_STATUS_NOT_INITIALIZED;
using ::CUSPARSE_STATUS_NOT_SUPPORTED;
using ::CUSPARSE_STATUS_SUCCESS;
using ::CUSPARSE_STATUS_ZERO_PIVOT;

using ::cusparsePointerMode_t;
// cusparsePointerMode_t enumerators
using ::CUSPARSE_POINTER_MODE_DEVICE;
using ::CUSPARSE_POINTER_MODE_HOST;

using ::cusparseAction_t;
// cusparseAction_t enumerators
using ::CUSPARSE_ACTION_NUMERIC;
using ::CUSPARSE_ACTION_SYMBOLIC;

using ::cusparseMatrixType_t;
// cusparseMatrixType_t enumerators
using ::CUSPARSE_MATRIX_TYPE_GENERAL;
using ::CUSPARSE_MATRIX_TYPE_HERMITIAN;
using ::CUSPARSE_MATRIX_TYPE_SYMMETRIC;
using ::CUSPARSE_MATRIX_TYPE_TRIANGULAR;

using ::cusparseFillMode_t;
// cusparseFillMode_t enumerators
using ::CUSPARSE_FILL_MODE_LOWER;
using ::CUSPARSE_FILL_MODE_UPPER;

using ::cusparseDiagType_t;
// cusparseDiagType_t enumerators
using ::CUSPARSE_DIAG_TYPE_NON_UNIT;
using ::CUSPARSE_DIAG_TYPE_UNIT;

using ::cusparseIndexBase_t;
// cusparseIndexBase_t enumerators
using ::CUSPARSE_INDEX_BASE_ONE;
using ::CUSPARSE_INDEX_BASE_ZERO;

using ::cusparseOperation_t;
// cusparseOperation_t enumerators
using ::CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
using ::CUSPARSE_OPERATION_NON_TRANSPOSE;
using ::CUSPARSE_OPERATION_TRANSPOSE;

using ::cusparseDirection_t;
// cusparseDirection_t enumerators
using ::CUSPARSE_DIRECTION_COLUMN;
using ::CUSPARSE_DIRECTION_ROW;

using ::cusparseSolvePolicy_t;
// cusparseSolvePolicy_t enumerators (deprecated type)
using ::CUSPARSE_SOLVE_POLICY_NO_LEVEL;
using ::CUSPARSE_SOLVE_POLICY_USE_LEVEL;

using ::cusparseColorAlg_t;
// cusparseColorAlg_t enumerators (deprecated type)
using ::CUSPARSE_COLOR_ALG0;
using ::CUSPARSE_COLOR_ALG1;

using ::cusparseCsr2CscAlg_t;
// cusparseCsr2CscAlg_t enumerators
using ::CUSPARSE_CSR2CSC_ALG1;
using ::CUSPARSE_CSR2CSC_ALG_DEFAULT;

using ::cusparseFormat_t;
// cusparseFormat_t enumerators
using ::CUSPARSE_FORMAT_BLOCKED_ELL;
using ::CUSPARSE_FORMAT_BSR;
using ::CUSPARSE_FORMAT_COO;
using ::CUSPARSE_FORMAT_CSC;
using ::CUSPARSE_FORMAT_CSR;
using ::CUSPARSE_FORMAT_SLICED_ELLPACK;

using ::cusparseOrder_t;
// cusparseOrder_t enumerators
using ::CUSPARSE_ORDER_COL;
using ::CUSPARSE_ORDER_ROW;

using ::cusparseIndexType_t;
// cusparseIndexType_t enumerators
using ::CUSPARSE_INDEX_16U;
using ::CUSPARSE_INDEX_32I;
using ::CUSPARSE_INDEX_64I;

using ::cusparseSpMatAttribute_t;
// cusparseSpMatAttribute_t enumerators
using ::CUSPARSE_SPMAT_DIAG_TYPE;
using ::CUSPARSE_SPMAT_FILL_MODE;

using ::cusparseSpMVAlg_t;
// cusparseSpMVAlg_t enumerators
using ::CUSPARSE_SPMV_ALG_DEFAULT;
using ::CUSPARSE_SPMV_COO_ALG1;
using ::CUSPARSE_SPMV_COO_ALG2;
using ::CUSPARSE_SPMV_CSR_ALG1;
using ::CUSPARSE_SPMV_CSR_ALG2;
using ::CUSPARSE_SPMV_SELL_ALG1;

using ::cusparseSpSVAlg_t;
// cusparseSpSVAlg_t enumerators
using ::CUSPARSE_SPSV_ALG_DEFAULT;

using ::cusparseSpSVUpdate_t;
// cusparseSpSVUpdate_t enumerators
using ::CUSPARSE_SPSV_UPDATE_DIAGONAL;
using ::CUSPARSE_SPSV_UPDATE_GENERAL;

using ::cusparseSpSMAlg_t;
// cusparseSpSMAlg_t enumerators
using ::CUSPARSE_SPSM_ALG_DEFAULT;

using ::cusparseSpSMUpdate_t;
// cusparseSpSMUpdate_t enumerators
using ::CUSPARSE_SPSM_UPDATE_DIAGONAL;
using ::CUSPARSE_SPSM_UPDATE_GENERAL;

using ::cusparseSpMMAlg_t;
// cusparseSpMMAlg_t enumerators
using ::CUSPARSE_SPMM_ALG_DEFAULT;
using ::CUSPARSE_SPMM_BLOCKED_ELL_ALG1;
using ::CUSPARSE_SPMM_BSR_ALG1;
using ::CUSPARSE_SPMM_COO_ALG1;
using ::CUSPARSE_SPMM_COO_ALG2;
using ::CUSPARSE_SPMM_COO_ALG3;
using ::CUSPARSE_SPMM_COO_ALG4;
using ::CUSPARSE_SPMM_CSR_ALG1;
using ::CUSPARSE_SPMM_CSR_ALG2;
using ::CUSPARSE_SPMM_CSR_ALG3;

using ::cusparseSpGEMMAlg_t;
// cusparseSpGEMMAlg_t enumerators
using ::CUSPARSE_SPGEMM_ALG1;
using ::CUSPARSE_SPGEMM_ALG2;
using ::CUSPARSE_SPGEMM_ALG3;
using ::CUSPARSE_SPGEMM_CSR_ALG_DETERMINITIC;
using ::CUSPARSE_SPGEMM_CSR_ALG_NONDETERMINITIC;
using ::CUSPARSE_SPGEMM_DEFAULT;

using ::cusparseSDDMMAlg_t;
// cusparseSDDMMAlg_t enumerators
using ::CUSPARSE_SDDMM_ALG_DEFAULT;

using ::cusparseSparseToDenseAlg_t;
// cusparseSparseToDenseAlg_t enumerators
using ::CUSPARSE_SPARSETODENSE_ALG_DEFAULT;

using ::cusparseDenseToSparseAlg_t;
// cusparseDenseToSparseAlg_t enumerators
using ::CUSPARSE_DENSETOSPARSE_ALG_DEFAULT;

using ::cusparseSpMMOpAlg_t;
// cusparseSpMMOpAlg_t enumerators
using ::CUSPARSE_SPMM_OP_ALG_DEFAULT;

// ========================================================================
// Logging Callback Type
// ========================================================================
using ::cusparseLoggerCallback_t;

// ========================================================================
// Context Management
// ========================================================================
using ::cusparseCreate;
using ::cusparseDestroy;
using ::cusparseGetErrorName;
using ::cusparseGetErrorString;
using ::cusparseGetPointerMode;
using ::cusparseGetProperty;
using ::cusparseGetStream;
using ::cusparseGetVersion;
using ::cusparseSetPointerMode;
using ::cusparseSetStream;

// ========================================================================
// Logging
// ========================================================================
using ::cusparseLoggerForceDisable;
using ::cusparseLoggerOpenFile;
using ::cusparseLoggerSetCallback;
using ::cusparseLoggerSetFile;
using ::cusparseLoggerSetLevel;
using ::cusparseLoggerSetMask;

// ========================================================================
// Matrix Descriptor (Legacy Helper Routines)
// ========================================================================
using ::cusparseCreateMatDescr;
using ::cusparseDestroyMatDescr;
using ::cusparseGetMatDiagType;
using ::cusparseGetMatFillMode;
using ::cusparseGetMatIndexBase;
using ::cusparseGetMatType;
using ::cusparseSetMatDiagType;
using ::cusparseSetMatFillMode;
using ::cusparseSetMatIndexBase;
using ::cusparseSetMatType;

// Legacy info object helpers (deprecated types/functions — kept for ABI completeness)
using ::cusparseCreateBsric02Info;
using ::cusparseCreateBsrilu02Info;
using ::cusparseCreateBsrsm2Info;
using ::cusparseCreateBsrsv2Info;
using ::cusparseCreateColorInfo;
using ::cusparseCreateCsric02Info;
using ::cusparseCreateCsrilu02Info;
using ::cusparseCreateCsru2csrInfo;
using ::cusparseCreatePruneInfo;
using ::cusparseDestroyBsric02Info;
using ::cusparseDestroyBsrilu02Info;
using ::cusparseDestroyBsrsm2Info;
using ::cusparseDestroyBsrsv2Info;
using ::cusparseDestroyColorInfo;
using ::cusparseDestroyCsric02Info;
using ::cusparseDestroyCsrilu02Info;
using ::cusparseDestroyCsru2csrInfo;
using ::cusparseDestroyPruneInfo;

// ========================================================================
// Sparse Level 2 — BSR Matrix-Vector Multiplication (non-deprecated)
// ========================================================================
using ::cusparseCbsrmv;
using ::cusparseDbsrmv;
using ::cusparseSbsrmv;
using ::cusparseZbsrmv;

// ========================================================================
// Sparse Level 3 — Tridiagonal/Pentadiagonal Solvers (non-deprecated)
// ========================================================================
using ::cusparseCgpsvInterleavedBatch;
using ::cusparseCgpsvInterleavedBatch_bufferSizeExt;
using ::cusparseCgtsv2;
using ::cusparseCgtsv2_bufferSizeExt;
using ::cusparseCgtsv2_nopivot;
using ::cusparseCgtsv2_nopivot_bufferSizeExt;
using ::cusparseCgtsv2StridedBatch;
using ::cusparseCgtsv2StridedBatch_bufferSizeExt;
using ::cusparseCgtsvInterleavedBatch;
using ::cusparseCgtsvInterleavedBatch_bufferSizeExt;
using ::cusparseDgpsvInterleavedBatch;
using ::cusparseDgpsvInterleavedBatch_bufferSizeExt;
using ::cusparseDgtsv2;
using ::cusparseDgtsv2_bufferSizeExt;
using ::cusparseDgtsv2_nopivot;
using ::cusparseDgtsv2_nopivot_bufferSizeExt;
using ::cusparseDgtsv2StridedBatch;
using ::cusparseDgtsv2StridedBatch_bufferSizeExt;
using ::cusparseDgtsvInterleavedBatch;
using ::cusparseDgtsvInterleavedBatch_bufferSizeExt;
using ::cusparseSgpsvInterleavedBatch;
using ::cusparseSgpsvInterleavedBatch_bufferSizeExt;
using ::cusparseSgtsv2;
using ::cusparseSgtsv2_bufferSizeExt;
using ::cusparseSgtsv2_nopivot;
using ::cusparseSgtsv2_nopivot_bufferSizeExt;
using ::cusparseSgtsv2StridedBatch;
using ::cusparseSgtsv2StridedBatch_bufferSizeExt;
using ::cusparseSgtsvInterleavedBatch;
using ::cusparseSgtsvInterleavedBatch_bufferSizeExt;
using ::cusparseZgpsvInterleavedBatch;
using ::cusparseZgpsvInterleavedBatch_bufferSizeExt;
using ::cusparseZgtsv2;
using ::cusparseZgtsv2_bufferSizeExt;
using ::cusparseZgtsv2_nopivot;
using ::cusparseZgtsv2_nopivot_bufferSizeExt;
using ::cusparseZgtsv2StridedBatch;
using ::cusparseZgtsv2StridedBatch_bufferSizeExt;
using ::cusparseZgtsvInterleavedBatch;
using ::cusparseZgtsvInterleavedBatch_bufferSizeExt;

// ========================================================================
// Extra Routines — CSR Matrix Addition (non-deprecated)
// ========================================================================
using ::cusparseCcsrgeam2;
using ::cusparseCcsrgeam2_bufferSizeExt;
using ::cusparseDcsrgeam2;
using ::cusparseDcsrgeam2_bufferSizeExt;
using ::cusparseScsrgeam2;
using ::cusparseScsrgeam2_bufferSizeExt;
using ::cusparseXcsrgeam2Nnz;
using ::cusparseZcsrgeam2;
using ::cusparseZcsrgeam2_bufferSizeExt;

// ========================================================================
// Sparse Format Conversion (non-deprecated)
// ========================================================================
using ::cusparseCcsr2gebsr;
using ::cusparseCcsr2gebsr_bufferSize;
using ::cusparseCcsr2gebsr_bufferSizeExt;
using ::cusparseCgebsr2gebsc;
using ::cusparseCgebsr2gebsc_bufferSize;
using ::cusparseCgebsr2gebsc_bufferSizeExt;
using ::cusparseCnnz;
using ::cusparseDcsr2gebsr;
using ::cusparseDcsr2gebsr_bufferSize;
using ::cusparseDcsr2gebsr_bufferSizeExt;
using ::cusparseDgebsr2gebsc;
using ::cusparseDgebsr2gebsc_bufferSize;
using ::cusparseDgebsr2gebsc_bufferSizeExt;
using ::cusparseDnnz;
using ::cusparseScsr2gebsr;
using ::cusparseScsr2gebsr_bufferSize;
using ::cusparseScsr2gebsr_bufferSizeExt;
using ::cusparseSgebsr2gebsc;
using ::cusparseSgebsr2gebsc_bufferSize;
using ::cusparseSgebsr2gebsc_bufferSizeExt;
using ::cusparseSnnz;
using ::cusparseXcoo2csr;
using ::cusparseXcsr2coo;
using ::cusparseXcsr2gebsrNnz;
using ::cusparseZcsr2gebsr;
using ::cusparseZcsr2gebsr_bufferSize;
using ::cusparseZcsr2gebsr_bufferSizeExt;
using ::cusparseZgebsr2gebsc;
using ::cusparseZgebsr2gebsc_bufferSize;
using ::cusparseZgebsr2gebsc_bufferSizeExt;
using ::cusparseZnnz;

// ========================================================================
// Sparse Matrix Sorting (non-deprecated)
// ========================================================================
using ::cusparseXcoosort_bufferSizeExt;
using ::cusparseXcoosortByColumn;
using ::cusparseXcoosortByRow;
using ::cusparseXcscsort;
using ::cusparseXcscsort_bufferSizeExt;
using ::cusparseXcsrsort;
using ::cusparseXcsrsort_bufferSizeExt;

// ========================================================================
// CSR to CSC Conversion
// ========================================================================
using ::cusparseCsr2cscEx2;
using ::cusparseCsr2cscEx2_bufferSize;

// ========================================================================
// Generic API — Sparse Vector Descriptor
// ========================================================================
using ::cusparseConstSpVecGet;
using ::cusparseConstSpVecGetValues;
using ::cusparseCreateConstSpVec;
using ::cusparseCreateSpVec;
using ::cusparseDestroySpVec;
using ::cusparseSpVecGet;
using ::cusparseSpVecGetIndexBase;
using ::cusparseSpVecGetValues;
using ::cusparseSpVecSetValues;

// ========================================================================
// Generic API — Dense Vector Descriptor
// ========================================================================
using ::cusparseConstDnVecGet;
using ::cusparseConstDnVecGetValues;
using ::cusparseCreateConstDnVec;
using ::cusparseCreateDnVec;
using ::cusparseDestroyDnVec;
using ::cusparseDnVecGet;
using ::cusparseDnVecGetValues;
using ::cusparseDnVecSetValues;

// ========================================================================
// Generic API — Sparse Matrix Descriptor
// ========================================================================
using ::cusparseBsrSetStridedBatch;
using ::cusparseConstSpMatGetValues;
using ::cusparseCooSetStridedBatch;
using ::cusparseCsrSetStridedBatch;
using ::cusparseDestroySpMat;
using ::cusparseSpMatGetAttribute;
using ::cusparseSpMatGetFormat;
using ::cusparseSpMatGetIndexBase;
using ::cusparseSpMatGetSize;
using ::cusparseSpMatGetStridedBatch;
using ::cusparseSpMatGetValues;
using ::cusparseSpMatSetAttribute;
using ::cusparseSpMatSetValues;

// CSR
using ::cusparseConstCsrGet;
using ::cusparseCreateConstCsr;
using ::cusparseCreateCsr;
using ::cusparseCsrGet;
using ::cusparseCsrSetPointers;

// CSC
using ::cusparseConstCscGet;
using ::cusparseCreateConstCsc;
using ::cusparseCreateCsc;
using ::cusparseCscGet;
using ::cusparseCscSetPointers;

// BSR
using ::cusparseCreateBsr;
using ::cusparseCreateConstBsr;

// COO
using ::cusparseConstCooGet;
using ::cusparseCooGet;
using ::cusparseCooSetPointers;
using ::cusparseCreateConstCoo;
using ::cusparseCreateCoo;

// Blocked ELL
using ::cusparseBlockedEllGet;
using ::cusparseConstBlockedEllGet;
using ::cusparseCreateBlockedEll;
using ::cusparseCreateConstBlockedEll;

// Sliced ELL
using ::cusparseCreateConstSlicedEll;
using ::cusparseCreateSlicedEll;

// ========================================================================
// Generic API — Dense Matrix Descriptor
// ========================================================================
using ::cusparseConstDnMatGet;
using ::cusparseConstDnMatGetValues;
using ::cusparseCreateConstDnMat;
using ::cusparseCreateDnMat;
using ::cusparseDestroyDnMat;
using ::cusparseDnMatGet;
using ::cusparseDnMatGetStridedBatch;
using ::cusparseDnMatGetValues;
using ::cusparseDnMatSetStridedBatch;
using ::cusparseDnMatSetValues;

// ========================================================================
// Generic API — Vector-Vector Operations
// ========================================================================
using ::cusparseGather;
using ::cusparseScatter;

// ========================================================================
// Generic API — Sparse to Dense / Dense to Sparse
// ========================================================================
using ::cusparseDenseToSparse_analysis;
using ::cusparseDenseToSparse_bufferSize;
using ::cusparseDenseToSparse_convert;
using ::cusparseSparseToDense;
using ::cusparseSparseToDense_bufferSize;

// ========================================================================
// Generic API — Sparse Matrix-Vector Multiplication (SpMV)
// ========================================================================
using ::cusparseSpMV;
using ::cusparseSpMV_bufferSize;
using ::cusparseSpMV_preprocess;

// ========================================================================
// Generic API — Sparse Triangular Vector Solve (SpSV)
// ========================================================================
using ::cusparseSpSV_analysis;
using ::cusparseSpSV_bufferSize;
using ::cusparseSpSV_createDescr;
using ::cusparseSpSV_destroyDescr;
using ::cusparseSpSV_solve;
using ::cusparseSpSV_updateMatrix;

// ========================================================================
// Generic API — Sparse Triangular Matrix Solve (SpSM)
// ========================================================================
using ::cusparseSpSM_analysis;
using ::cusparseSpSM_bufferSize;
using ::cusparseSpSM_createDescr;
using ::cusparseSpSM_destroyDescr;
using ::cusparseSpSM_solve;
using ::cusparseSpSM_updateMatrix;

// ========================================================================
// Generic API — Sparse Matrix-Matrix Multiplication (SpMM)
// ========================================================================
using ::cusparseSpMM;
using ::cusparseSpMM_bufferSize;
using ::cusparseSpMM_preprocess;

// ========================================================================
// Generic API — Sparse Matrix-Sparse Matrix Multiplication (SpGEMM)
// ========================================================================
using ::cusparseSpGEMM_compute;
using ::cusparseSpGEMM_copy;
using ::cusparseSpGEMM_createDescr;
using ::cusparseSpGEMM_destroyDescr;
using ::cusparseSpGEMM_estimateMemory;
using ::cusparseSpGEMM_getNumProducts;
using ::cusparseSpGEMM_workEstimation;
using ::cusparseSpGEMMreuse_compute;
using ::cusparseSpGEMMreuse_copy;
using ::cusparseSpGEMMreuse_nnz;
using ::cusparseSpGEMMreuse_workEstimation;

// ========================================================================
// Generic API — Sampled Dense-Dense Matrix Multiplication (SDDMM)
// ========================================================================
using ::cusparseSDDMM;
using ::cusparseSDDMM_bufferSize;
using ::cusparseSDDMM_preprocess;

// ========================================================================
// Generic API — SpMM with Custom Operators (Preview)
// ========================================================================
using ::cusparseSpMMOp;
using ::cusparseSpMMOp_createPlan;
using ::cusparseSpMMOp_destroyPlan;

} // namespace wwr::cuda
