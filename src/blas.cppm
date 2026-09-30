/**
 * @file blas.cppm
 * @brief Backend-neutral BLAS: wwrblas* names for cuBLAS / hipBLAS
 *
 * wwrblas<X> stands for cublas<X>_v2 where cuBLAS has a _v2 name pair and
 * cublas<X> otherwise, and for hipblas<X>, which has no _v2 names; 64-bit
 * variants are wwrblas<X>_64. Each is written out in full, one line per name.
 * See backend.h.
 *
 * Only the names src/wrappers/blas uses are listed, plus the handle, stream
 * and pointer-mode calls a caller needs. cuBLAS-only functions (gemm3m,
 * gemmGroupedBatched, matinvBatched, tpttr, trttp) are absent; reach those
 * through wwr.cuda.cublas_v2.
 *
 * Two backend differences are resolved here, not above: hipBLAS's one
 * status-to-string function backs both wwrblasGetStatusName and
 * wwrblasGetStatusString, and getrsBatched / getriBatched keep cuBLAS's
 * const-correct signature, forwarding with a const_cast on HIP.
 * See docs/architecture.md, section 5.
 *
 * Usage:
 *   import wwr.blas;
 *
 *   wwrblasHandle_t handle;
 *   wwrblasCreate(&handle);
 */

module;

#include "backend.h"

export module wwr.blas;

import wwr.complex;
#if defined(WWR_GPU_BACKEND_CUDA)
import wwr.cuda.cublas_v2;
#else
import wwr.hip.hipblas;
#endif

export namespace wwr {

// ========================================================================
// Types
// ========================================================================

WWR_TYPE(wwrblasHandle_t, cublasHandle_t, hipblasHandle_t)
WWR_TYPE(wwrblasStatus_t, cublasStatus_t, hipblasStatus_t)
WWR_TYPE(wwrblasOperation_t, cublasOperation_t, hipblasOperation_t)
WWR_TYPE(wwrblasFillMode_t, cublasFillMode_t, hipblasFillMode_t)
WWR_TYPE(wwrblasDiagType_t, cublasDiagType_t, hipblasDiagType_t)
WWR_TYPE(wwrblasSideMode_t, cublasSideMode_t, hipblasSideMode_t)
WWR_TYPE(wwrblasPointerMode_t, cublasPointerMode_t, hipblasPointerMode_t)

// ========================================================================
// Constants
// ========================================================================

// The status codes whose names cuBLAS and hipBLAS share -- their numeric
// values differ between the backends, which is exactly what these neutral
// names paper over. LICENSE_ERROR (cuBLAS-only) and HANDLE_IS_NULLPTR /
// INVALID_ENUM / UNKNOWN (hipBLAS-only) have no counterpart, so they cannot
// be neutral and are deliberately absent.
WWR_VALUE(WWRBLAS_STATUS_SUCCESS, CUBLAS_STATUS_SUCCESS, HIPBLAS_STATUS_SUCCESS)
WWR_VALUE(WWRBLAS_STATUS_NOT_INITIALIZED, CUBLAS_STATUS_NOT_INITIALIZED,
             HIPBLAS_STATUS_NOT_INITIALIZED)
WWR_VALUE(WWRBLAS_STATUS_ALLOC_FAILED, CUBLAS_STATUS_ALLOC_FAILED, HIPBLAS_STATUS_ALLOC_FAILED)
WWR_VALUE(WWRBLAS_STATUS_ARCH_MISMATCH, CUBLAS_STATUS_ARCH_MISMATCH, HIPBLAS_STATUS_ARCH_MISMATCH)
WWR_VALUE(WWRBLAS_STATUS_EXECUTION_FAILED, CUBLAS_STATUS_EXECUTION_FAILED,
             HIPBLAS_STATUS_EXECUTION_FAILED)
WWR_VALUE(WWRBLAS_STATUS_INTERNAL_ERROR, CUBLAS_STATUS_INTERNAL_ERROR,
             HIPBLAS_STATUS_INTERNAL_ERROR)
WWR_VALUE(WWRBLAS_STATUS_INVALID_VALUE, CUBLAS_STATUS_INVALID_VALUE, HIPBLAS_STATUS_INVALID_VALUE)
WWR_VALUE(WWRBLAS_STATUS_MAPPING_ERROR, CUBLAS_STATUS_MAPPING_ERROR, HIPBLAS_STATUS_MAPPING_ERROR)
WWR_VALUE(WWRBLAS_STATUS_NOT_SUPPORTED, CUBLAS_STATUS_NOT_SUPPORTED, HIPBLAS_STATUS_NOT_SUPPORTED)

WWR_VALUE(WWRBLAS_OP_N, CUBLAS_OP_N, HIPBLAS_OP_N)
WWR_VALUE(WWRBLAS_OP_T, CUBLAS_OP_T, HIPBLAS_OP_T)
WWR_VALUE(WWRBLAS_OP_C, CUBLAS_OP_C, HIPBLAS_OP_C)

WWR_VALUE(WWRBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_LOWER, HIPBLAS_FILL_MODE_LOWER)
WWR_VALUE(WWRBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_UPPER, HIPBLAS_FILL_MODE_UPPER)

WWR_VALUE(WWRBLAS_DIAG_NON_UNIT, CUBLAS_DIAG_NON_UNIT, HIPBLAS_DIAG_NON_UNIT)
WWR_VALUE(WWRBLAS_DIAG_UNIT, CUBLAS_DIAG_UNIT, HIPBLAS_DIAG_UNIT)

WWR_VALUE(WWRBLAS_SIDE_LEFT, CUBLAS_SIDE_LEFT, HIPBLAS_SIDE_LEFT)
WWR_VALUE(WWRBLAS_SIDE_RIGHT, CUBLAS_SIDE_RIGHT, HIPBLAS_SIDE_RIGHT)

WWR_VALUE(WWRBLAS_POINTER_MODE_HOST, CUBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_HOST)
WWR_VALUE(WWRBLAS_POINTER_MODE_DEVICE, CUBLAS_POINTER_MODE_DEVICE, HIPBLAS_POINTER_MODE_DEVICE)

// ========================================================================
// Handle, stream, pointer mode, status strings
// ========================================================================

WWR_FUNCTION(wwrblasCreate, cublasCreate_v2, hipblasCreate)
WWR_FUNCTION(wwrblasDestroy, cublasDestroy_v2, hipblasDestroy)
WWR_FUNCTION(wwrblasSetStream, cublasSetStream_v2, hipblasSetStream)
WWR_FUNCTION(wwrblasGetStream, cublasGetStream_v2, hipblasGetStream)
WWR_FUNCTION(wwrblasSetPointerMode, cublasSetPointerMode_v2, hipblasSetPointerMode)
WWR_FUNCTION(wwrblasGetPointerMode, cublasGetPointerMode_v2, hipblasGetPointerMode)

// hipBLAS has a single status-to-string function; both names map to it.
WWR_FUNCTION(wwrblasGetStatusName, cublasGetStatusName, hipblasStatusToString)
WWR_FUNCTION(wwrblasGetStatusString, cublasGetStatusString, hipblasStatusToString)

// ────────────────────────────────────────────────────────────────────────
// Level 1 (vector-vector)
// ────────────────────────────────────────────────────────────────────────

WWR_FUNCTION(wwrblasSasum, cublasSasum_v2, hipblasSasum)
WWR_FUNCTION(wwrblasDasum, cublasDasum_v2, hipblasDasum)
WWR_FUNCTION(wwrblasSasum_64, cublasSasum_v2_64, hipblasSasum_64)
WWR_FUNCTION(wwrblasDasum_64, cublasDasum_v2_64, hipblasDasum_64)

WWR_FUNCTION(wwrblasSaxpy, cublasSaxpy_v2, hipblasSaxpy)
WWR_FUNCTION(wwrblasDaxpy, cublasDaxpy_v2, hipblasDaxpy)
WWR_FUNCTION(wwrblasCaxpy, cublasCaxpy_v2, hipblasCaxpy)
WWR_FUNCTION(wwrblasZaxpy, cublasZaxpy_v2, hipblasZaxpy)
WWR_FUNCTION(wwrblasSaxpy_64, cublasSaxpy_v2_64, hipblasSaxpy_64)
WWR_FUNCTION(wwrblasDaxpy_64, cublasDaxpy_v2_64, hipblasDaxpy_64)
WWR_FUNCTION(wwrblasCaxpy_64, cublasCaxpy_v2_64, hipblasCaxpy_64)
WWR_FUNCTION(wwrblasZaxpy_64, cublasZaxpy_v2_64, hipblasZaxpy_64)

WWR_FUNCTION(wwrblasScasum, cublasScasum_v2, hipblasScasum)
WWR_FUNCTION(wwrblasScasum_64, cublasScasum_v2_64, hipblasScasum_64)

WWR_FUNCTION(wwrblasScnrm2, cublasScnrm2_v2, hipblasScnrm2)
WWR_FUNCTION(wwrblasScnrm2_64, cublasScnrm2_v2_64, hipblasScnrm2_64)

WWR_FUNCTION(wwrblasScopy, cublasScopy_v2, hipblasScopy)
WWR_FUNCTION(wwrblasDcopy, cublasDcopy_v2, hipblasDcopy)
WWR_FUNCTION(wwrblasCcopy, cublasCcopy_v2, hipblasCcopy)
WWR_FUNCTION(wwrblasZcopy, cublasZcopy_v2, hipblasZcopy)
WWR_FUNCTION(wwrblasScopy_64, cublasScopy_v2_64, hipblasScopy_64)
WWR_FUNCTION(wwrblasDcopy_64, cublasDcopy_v2_64, hipblasDcopy_64)
WWR_FUNCTION(wwrblasCcopy_64, cublasCcopy_v2_64, hipblasCcopy_64)
WWR_FUNCTION(wwrblasZcopy_64, cublasZcopy_v2_64, hipblasZcopy_64)

WWR_FUNCTION(wwrblasSdot, cublasSdot_v2, hipblasSdot)
WWR_FUNCTION(wwrblasDdot, cublasDdot_v2, hipblasDdot)
WWR_FUNCTION(wwrblasSdot_64, cublasSdot_v2_64, hipblasSdot_64)
WWR_FUNCTION(wwrblasDdot_64, cublasDdot_v2_64, hipblasDdot_64)

WWR_FUNCTION(wwrblasCdotc, cublasCdotc_v2, hipblasCdotc)
WWR_FUNCTION(wwrblasZdotc, cublasZdotc_v2, hipblasZdotc)
WWR_FUNCTION(wwrblasCdotc_64, cublasCdotc_v2_64, hipblasCdotc_64)
WWR_FUNCTION(wwrblasZdotc_64, cublasZdotc_v2_64, hipblasZdotc_64)

WWR_FUNCTION(wwrblasCdotu, cublasCdotu_v2, hipblasCdotu)
WWR_FUNCTION(wwrblasZdotu, cublasZdotu_v2, hipblasZdotu)
WWR_FUNCTION(wwrblasCdotu_64, cublasCdotu_v2_64, hipblasCdotu_64)
WWR_FUNCTION(wwrblasZdotu_64, cublasZdotu_v2_64, hipblasZdotu_64)

WWR_FUNCTION(wwrblasZdrot, cublasZdrot_v2, hipblasZdrot)
WWR_FUNCTION(wwrblasZdrot_64, cublasZdrot_v2_64, hipblasZdrot_64)

WWR_FUNCTION(wwrblasZdscal, cublasZdscal_v2, hipblasZdscal)
WWR_FUNCTION(wwrblasZdscal_64, cublasZdscal_v2_64, hipblasZdscal_64)

WWR_FUNCTION(wwrblasIsamax, cublasIsamax_v2, hipblasIsamax)
WWR_FUNCTION(wwrblasIdamax, cublasIdamax_v2, hipblasIdamax)
WWR_FUNCTION(wwrblasIcamax, cublasIcamax_v2, hipblasIcamax)
WWR_FUNCTION(wwrblasIzamax, cublasIzamax_v2, hipblasIzamax)
WWR_FUNCTION(wwrblasIsamax_64, cublasIsamax_v2_64, hipblasIsamax_64)
WWR_FUNCTION(wwrblasIdamax_64, cublasIdamax_v2_64, hipblasIdamax_64)
WWR_FUNCTION(wwrblasIcamax_64, cublasIcamax_v2_64, hipblasIcamax_64)
WWR_FUNCTION(wwrblasIzamax_64, cublasIzamax_v2_64, hipblasIzamax_64)

WWR_FUNCTION(wwrblasIsamin, cublasIsamin_v2, hipblasIsamin)
WWR_FUNCTION(wwrblasIdamin, cublasIdamin_v2, hipblasIdamin)
WWR_FUNCTION(wwrblasIcamin, cublasIcamin_v2, hipblasIcamin)
WWR_FUNCTION(wwrblasIzamin, cublasIzamin_v2, hipblasIzamin)
WWR_FUNCTION(wwrblasIsamin_64, cublasIsamin_v2_64, hipblasIsamin_64)
WWR_FUNCTION(wwrblasIdamin_64, cublasIdamin_v2_64, hipblasIdamin_64)
WWR_FUNCTION(wwrblasIcamin_64, cublasIcamin_v2_64, hipblasIcamin_64)
WWR_FUNCTION(wwrblasIzamin_64, cublasIzamin_v2_64, hipblasIzamin_64)

WWR_FUNCTION(wwrblasSnrm2, cublasSnrm2_v2, hipblasSnrm2)
WWR_FUNCTION(wwrblasDnrm2, cublasDnrm2_v2, hipblasDnrm2)
WWR_FUNCTION(wwrblasSnrm2_64, cublasSnrm2_v2_64, hipblasSnrm2_64)
WWR_FUNCTION(wwrblasDnrm2_64, cublasDnrm2_v2_64, hipblasDnrm2_64)

WWR_FUNCTION(wwrblasSrot, cublasSrot_v2, hipblasSrot)
WWR_FUNCTION(wwrblasDrot, cublasDrot_v2, hipblasDrot)
WWR_FUNCTION(wwrblasCrot, cublasCrot_v2, hipblasCrot)
WWR_FUNCTION(wwrblasZrot, cublasZrot_v2, hipblasZrot)
WWR_FUNCTION(wwrblasSrot_64, cublasSrot_v2_64, hipblasSrot_64)
WWR_FUNCTION(wwrblasDrot_64, cublasDrot_v2_64, hipblasDrot_64)
WWR_FUNCTION(wwrblasCrot_64, cublasCrot_v2_64, hipblasCrot_64)
WWR_FUNCTION(wwrblasZrot_64, cublasZrot_v2_64, hipblasZrot_64)

WWR_FUNCTION(wwrblasSrotg, cublasSrotg_v2, hipblasSrotg)
WWR_FUNCTION(wwrblasDrotg, cublasDrotg_v2, hipblasDrotg)
WWR_FUNCTION(wwrblasCrotg, cublasCrotg_v2, hipblasCrotg)
WWR_FUNCTION(wwrblasZrotg, cublasZrotg_v2, hipblasZrotg)

WWR_FUNCTION(wwrblasSrotm, cublasSrotm_v2, hipblasSrotm)
WWR_FUNCTION(wwrblasDrotm, cublasDrotm_v2, hipblasDrotm)
WWR_FUNCTION(wwrblasSrotm_64, cublasSrotm_v2_64, hipblasSrotm_64)
WWR_FUNCTION(wwrblasDrotm_64, cublasDrotm_v2_64, hipblasDrotm_64)

WWR_FUNCTION(wwrblasSrotmg, cublasSrotmg_v2, hipblasSrotmg)
WWR_FUNCTION(wwrblasDrotmg, cublasDrotmg_v2, hipblasDrotmg)

WWR_FUNCTION(wwrblasSscal, cublasSscal_v2, hipblasSscal)
WWR_FUNCTION(wwrblasDscal, cublasDscal_v2, hipblasDscal)
WWR_FUNCTION(wwrblasCscal, cublasCscal_v2, hipblasCscal)
WWR_FUNCTION(wwrblasZscal, cublasZscal_v2, hipblasZscal)
WWR_FUNCTION(wwrblasSscal_64, cublasSscal_v2_64, hipblasSscal_64)
WWR_FUNCTION(wwrblasDscal_64, cublasDscal_v2_64, hipblasDscal_64)
WWR_FUNCTION(wwrblasCscal_64, cublasCscal_v2_64, hipblasCscal_64)
WWR_FUNCTION(wwrblasZscal_64, cublasZscal_v2_64, hipblasZscal_64)

WWR_FUNCTION(wwrblasCsrot, cublasCsrot_v2, hipblasCsrot)
WWR_FUNCTION(wwrblasCsrot_64, cublasCsrot_v2_64, hipblasCsrot_64)

WWR_FUNCTION(wwrblasCsscal, cublasCsscal_v2, hipblasCsscal)
WWR_FUNCTION(wwrblasCsscal_64, cublasCsscal_v2_64, hipblasCsscal_64)

WWR_FUNCTION(wwrblasSswap, cublasSswap_v2, hipblasSswap)
WWR_FUNCTION(wwrblasDswap, cublasDswap_v2, hipblasDswap)
WWR_FUNCTION(wwrblasCswap, cublasCswap_v2, hipblasCswap)
WWR_FUNCTION(wwrblasZswap, cublasZswap_v2, hipblasZswap)
WWR_FUNCTION(wwrblasSswap_64, cublasSswap_v2_64, hipblasSswap_64)
WWR_FUNCTION(wwrblasDswap_64, cublasDswap_v2_64, hipblasDswap_64)
WWR_FUNCTION(wwrblasCswap_64, cublasCswap_v2_64, hipblasCswap_64)
WWR_FUNCTION(wwrblasZswap_64, cublasZswap_v2_64, hipblasZswap_64)

WWR_FUNCTION(wwrblasDzasum, cublasDzasum_v2, hipblasDzasum)
WWR_FUNCTION(wwrblasDzasum_64, cublasDzasum_v2_64, hipblasDzasum_64)

WWR_FUNCTION(wwrblasDznrm2, cublasDznrm2_v2, hipblasDznrm2)
WWR_FUNCTION(wwrblasDznrm2_64, cublasDznrm2_v2_64, hipblasDznrm2_64)

// ────────────────────────────────────────────────────────────────────────
// Level 2 (matrix-vector)
// ────────────────────────────────────────────────────────────────────────

WWR_FUNCTION(wwrblasSgbmv, cublasSgbmv_v2, hipblasSgbmv)
WWR_FUNCTION(wwrblasDgbmv, cublasDgbmv_v2, hipblasDgbmv)
WWR_FUNCTION(wwrblasCgbmv, cublasCgbmv_v2, hipblasCgbmv)
WWR_FUNCTION(wwrblasZgbmv, cublasZgbmv_v2, hipblasZgbmv)
WWR_FUNCTION(wwrblasSgbmv_64, cublasSgbmv_v2_64, hipblasSgbmv_64)
WWR_FUNCTION(wwrblasDgbmv_64, cublasDgbmv_v2_64, hipblasDgbmv_64)
WWR_FUNCTION(wwrblasCgbmv_64, cublasCgbmv_v2_64, hipblasCgbmv_64)
WWR_FUNCTION(wwrblasZgbmv_64, cublasZgbmv_v2_64, hipblasZgbmv_64)

WWR_FUNCTION(wwrblasSgemv, cublasSgemv_v2, hipblasSgemv)
WWR_FUNCTION(wwrblasDgemv, cublasDgemv_v2, hipblasDgemv)
WWR_FUNCTION(wwrblasCgemv, cublasCgemv_v2, hipblasCgemv)
WWR_FUNCTION(wwrblasZgemv, cublasZgemv_v2, hipblasZgemv)
WWR_FUNCTION(wwrblasSgemv_64, cublasSgemv_v2_64, hipblasSgemv_64)
WWR_FUNCTION(wwrblasDgemv_64, cublasDgemv_v2_64, hipblasDgemv_64)
WWR_FUNCTION(wwrblasCgemv_64, cublasCgemv_v2_64, hipblasCgemv_64)
WWR_FUNCTION(wwrblasZgemv_64, cublasZgemv_v2_64, hipblasZgemv_64)

WWR_FUNCTION(wwrblasSgemvBatched, cublasSgemvBatched, hipblasSgemvBatched)
WWR_FUNCTION(wwrblasDgemvBatched, cublasDgemvBatched, hipblasDgemvBatched)
WWR_FUNCTION(wwrblasCgemvBatched, cublasCgemvBatched, hipblasCgemvBatched)
WWR_FUNCTION(wwrblasZgemvBatched, cublasZgemvBatched, hipblasZgemvBatched)
WWR_FUNCTION(wwrblasSgemvBatched_64, cublasSgemvBatched_64, hipblasSgemvBatched_64)
WWR_FUNCTION(wwrblasDgemvBatched_64, cublasDgemvBatched_64, hipblasDgemvBatched_64)
WWR_FUNCTION(wwrblasCgemvBatched_64, cublasCgemvBatched_64, hipblasCgemvBatched_64)
WWR_FUNCTION(wwrblasZgemvBatched_64, cublasZgemvBatched_64, hipblasZgemvBatched_64)

WWR_FUNCTION(wwrblasSgemvStridedBatched, cublasSgemvStridedBatched, hipblasSgemvStridedBatched)
WWR_FUNCTION(wwrblasDgemvStridedBatched, cublasDgemvStridedBatched, hipblasDgemvStridedBatched)
WWR_FUNCTION(wwrblasCgemvStridedBatched, cublasCgemvStridedBatched, hipblasCgemvStridedBatched)
WWR_FUNCTION(wwrblasZgemvStridedBatched, cublasZgemvStridedBatched, hipblasZgemvStridedBatched)
WWR_FUNCTION(wwrblasSgemvStridedBatched_64, cublasSgemvStridedBatched_64,
                hipblasSgemvStridedBatched_64)
WWR_FUNCTION(wwrblasDgemvStridedBatched_64, cublasDgemvStridedBatched_64,
                hipblasDgemvStridedBatched_64)
WWR_FUNCTION(wwrblasCgemvStridedBatched_64, cublasCgemvStridedBatched_64,
                hipblasCgemvStridedBatched_64)
WWR_FUNCTION(wwrblasZgemvStridedBatched_64, cublasZgemvStridedBatched_64,
                hipblasZgemvStridedBatched_64)

WWR_FUNCTION(wwrblasSger, cublasSger_v2, hipblasSger)
WWR_FUNCTION(wwrblasDger, cublasDger_v2, hipblasDger)
WWR_FUNCTION(wwrblasSger_64, cublasSger_v2_64, hipblasSger_64)
WWR_FUNCTION(wwrblasDger_64, cublasDger_v2_64, hipblasDger_64)

WWR_FUNCTION(wwrblasCgerc, cublasCgerc_v2, hipblasCgerc)
WWR_FUNCTION(wwrblasZgerc, cublasZgerc_v2, hipblasZgerc)
WWR_FUNCTION(wwrblasCgerc_64, cublasCgerc_v2_64, hipblasCgerc_64)
WWR_FUNCTION(wwrblasZgerc_64, cublasZgerc_v2_64, hipblasZgerc_64)

WWR_FUNCTION(wwrblasCgeru, cublasCgeru_v2, hipblasCgeru)
WWR_FUNCTION(wwrblasZgeru, cublasZgeru_v2, hipblasZgeru)
WWR_FUNCTION(wwrblasCgeru_64, cublasCgeru_v2_64, hipblasCgeru_64)
WWR_FUNCTION(wwrblasZgeru_64, cublasZgeru_v2_64, hipblasZgeru_64)

WWR_FUNCTION(wwrblasChbmv, cublasChbmv_v2, hipblasChbmv)
WWR_FUNCTION(wwrblasZhbmv, cublasZhbmv_v2, hipblasZhbmv)
WWR_FUNCTION(wwrblasChbmv_64, cublasChbmv_v2_64, hipblasChbmv_64)
WWR_FUNCTION(wwrblasZhbmv_64, cublasZhbmv_v2_64, hipblasZhbmv_64)

WWR_FUNCTION(wwrblasChemv, cublasChemv_v2, hipblasChemv)
WWR_FUNCTION(wwrblasZhemv, cublasZhemv_v2, hipblasZhemv)
WWR_FUNCTION(wwrblasChemv_64, cublasChemv_v2_64, hipblasChemv_64)
WWR_FUNCTION(wwrblasZhemv_64, cublasZhemv_v2_64, hipblasZhemv_64)

WWR_FUNCTION(wwrblasCher, cublasCher_v2, hipblasCher)
WWR_FUNCTION(wwrblasZher, cublasZher_v2, hipblasZher)
WWR_FUNCTION(wwrblasCher_64, cublasCher_v2_64, hipblasCher_64)
WWR_FUNCTION(wwrblasZher_64, cublasZher_v2_64, hipblasZher_64)

WWR_FUNCTION(wwrblasCher2, cublasCher2_v2, hipblasCher2)
WWR_FUNCTION(wwrblasZher2, cublasZher2_v2, hipblasZher2)
WWR_FUNCTION(wwrblasCher2_64, cublasCher2_v2_64, hipblasCher2_64)
WWR_FUNCTION(wwrblasZher2_64, cublasZher2_v2_64, hipblasZher2_64)

WWR_FUNCTION(wwrblasChpmv, cublasChpmv_v2, hipblasChpmv)
WWR_FUNCTION(wwrblasZhpmv, cublasZhpmv_v2, hipblasZhpmv)
WWR_FUNCTION(wwrblasChpmv_64, cublasChpmv_v2_64, hipblasChpmv_64)
WWR_FUNCTION(wwrblasZhpmv_64, cublasZhpmv_v2_64, hipblasZhpmv_64)

WWR_FUNCTION(wwrblasChpr, cublasChpr_v2, hipblasChpr)
WWR_FUNCTION(wwrblasZhpr, cublasZhpr_v2, hipblasZhpr)
WWR_FUNCTION(wwrblasChpr_64, cublasChpr_v2_64, hipblasChpr_64)
WWR_FUNCTION(wwrblasZhpr_64, cublasZhpr_v2_64, hipblasZhpr_64)

WWR_FUNCTION(wwrblasChpr2, cublasChpr2_v2, hipblasChpr2)
WWR_FUNCTION(wwrblasZhpr2, cublasZhpr2_v2, hipblasZhpr2)
WWR_FUNCTION(wwrblasChpr2_64, cublasChpr2_v2_64, hipblasChpr2_64)
WWR_FUNCTION(wwrblasZhpr2_64, cublasZhpr2_v2_64, hipblasZhpr2_64)

WWR_FUNCTION(wwrblasSsbmv, cublasSsbmv_v2, hipblasSsbmv)
WWR_FUNCTION(wwrblasDsbmv, cublasDsbmv_v2, hipblasDsbmv)
WWR_FUNCTION(wwrblasSsbmv_64, cublasSsbmv_v2_64, hipblasSsbmv_64)
WWR_FUNCTION(wwrblasDsbmv_64, cublasDsbmv_v2_64, hipblasDsbmv_64)

WWR_FUNCTION(wwrblasSspmv, cublasSspmv_v2, hipblasSspmv)
WWR_FUNCTION(wwrblasDspmv, cublasDspmv_v2, hipblasDspmv)
WWR_FUNCTION(wwrblasSspmv_64, cublasSspmv_v2_64, hipblasSspmv_64)
WWR_FUNCTION(wwrblasDspmv_64, cublasDspmv_v2_64, hipblasDspmv_64)

WWR_FUNCTION(wwrblasSspr, cublasSspr_v2, hipblasSspr)
WWR_FUNCTION(wwrblasDspr, cublasDspr_v2, hipblasDspr)
WWR_FUNCTION(wwrblasSspr_64, cublasSspr_v2_64, hipblasSspr_64)
WWR_FUNCTION(wwrblasDspr_64, cublasDspr_v2_64, hipblasDspr_64)

WWR_FUNCTION(wwrblasSspr2, cublasSspr2_v2, hipblasSspr2)
WWR_FUNCTION(wwrblasDspr2, cublasDspr2_v2, hipblasDspr2)
WWR_FUNCTION(wwrblasSspr2_64, cublasSspr2_v2_64, hipblasSspr2_64)
WWR_FUNCTION(wwrblasDspr2_64, cublasDspr2_v2_64, hipblasDspr2_64)

WWR_FUNCTION(wwrblasSsymv, cublasSsymv_v2, hipblasSsymv)
WWR_FUNCTION(wwrblasDsymv, cublasDsymv_v2, hipblasDsymv)
WWR_FUNCTION(wwrblasSsymv_64, cublasSsymv_v2_64, hipblasSsymv_64)
WWR_FUNCTION(wwrblasDsymv_64, cublasDsymv_v2_64, hipblasDsymv_64)

WWR_FUNCTION(wwrblasSsyr, cublasSsyr_v2, hipblasSsyr)
WWR_FUNCTION(wwrblasDsyr, cublasDsyr_v2, hipblasDsyr)
WWR_FUNCTION(wwrblasSsyr_64, cublasSsyr_v2_64, hipblasSsyr_64)
WWR_FUNCTION(wwrblasDsyr_64, cublasDsyr_v2_64, hipblasDsyr_64)

WWR_FUNCTION(wwrblasSsyr2, cublasSsyr2_v2, hipblasSsyr2)
WWR_FUNCTION(wwrblasDsyr2, cublasDsyr2_v2, hipblasDsyr2)
WWR_FUNCTION(wwrblasSsyr2_64, cublasSsyr2_v2_64, hipblasSsyr2_64)
WWR_FUNCTION(wwrblasDsyr2_64, cublasDsyr2_v2_64, hipblasDsyr2_64)

WWR_FUNCTION(wwrblasStbmv, cublasStbmv_v2, hipblasStbmv)
WWR_FUNCTION(wwrblasDtbmv, cublasDtbmv_v2, hipblasDtbmv)
WWR_FUNCTION(wwrblasCtbmv, cublasCtbmv_v2, hipblasCtbmv)
WWR_FUNCTION(wwrblasZtbmv, cublasZtbmv_v2, hipblasZtbmv)
WWR_FUNCTION(wwrblasStbmv_64, cublasStbmv_v2_64, hipblasStbmv_64)
WWR_FUNCTION(wwrblasDtbmv_64, cublasDtbmv_v2_64, hipblasDtbmv_64)
WWR_FUNCTION(wwrblasCtbmv_64, cublasCtbmv_v2_64, hipblasCtbmv_64)
WWR_FUNCTION(wwrblasZtbmv_64, cublasZtbmv_v2_64, hipblasZtbmv_64)

WWR_FUNCTION(wwrblasStbsv, cublasStbsv_v2, hipblasStbsv)
WWR_FUNCTION(wwrblasDtbsv, cublasDtbsv_v2, hipblasDtbsv)
WWR_FUNCTION(wwrblasCtbsv, cublasCtbsv_v2, hipblasCtbsv)
WWR_FUNCTION(wwrblasZtbsv, cublasZtbsv_v2, hipblasZtbsv)
WWR_FUNCTION(wwrblasStbsv_64, cublasStbsv_v2_64, hipblasStbsv_64)
WWR_FUNCTION(wwrblasDtbsv_64, cublasDtbsv_v2_64, hipblasDtbsv_64)
WWR_FUNCTION(wwrblasCtbsv_64, cublasCtbsv_v2_64, hipblasCtbsv_64)
WWR_FUNCTION(wwrblasZtbsv_64, cublasZtbsv_v2_64, hipblasZtbsv_64)

WWR_FUNCTION(wwrblasStpmv, cublasStpmv_v2, hipblasStpmv)
WWR_FUNCTION(wwrblasDtpmv, cublasDtpmv_v2, hipblasDtpmv)
WWR_FUNCTION(wwrblasCtpmv, cublasCtpmv_v2, hipblasCtpmv)
WWR_FUNCTION(wwrblasZtpmv, cublasZtpmv_v2, hipblasZtpmv)
WWR_FUNCTION(wwrblasStpmv_64, cublasStpmv_v2_64, hipblasStpmv_64)
WWR_FUNCTION(wwrblasDtpmv_64, cublasDtpmv_v2_64, hipblasDtpmv_64)
WWR_FUNCTION(wwrblasCtpmv_64, cublasCtpmv_v2_64, hipblasCtpmv_64)
WWR_FUNCTION(wwrblasZtpmv_64, cublasZtpmv_v2_64, hipblasZtpmv_64)

WWR_FUNCTION(wwrblasStpsv, cublasStpsv_v2, hipblasStpsv)
WWR_FUNCTION(wwrblasDtpsv, cublasDtpsv_v2, hipblasDtpsv)
WWR_FUNCTION(wwrblasCtpsv, cublasCtpsv_v2, hipblasCtpsv)
WWR_FUNCTION(wwrblasZtpsv, cublasZtpsv_v2, hipblasZtpsv)
WWR_FUNCTION(wwrblasStpsv_64, cublasStpsv_v2_64, hipblasStpsv_64)
WWR_FUNCTION(wwrblasDtpsv_64, cublasDtpsv_v2_64, hipblasDtpsv_64)
WWR_FUNCTION(wwrblasCtpsv_64, cublasCtpsv_v2_64, hipblasCtpsv_64)
WWR_FUNCTION(wwrblasZtpsv_64, cublasZtpsv_v2_64, hipblasZtpsv_64)

WWR_FUNCTION(wwrblasStrmv, cublasStrmv_v2, hipblasStrmv)
WWR_FUNCTION(wwrblasDtrmv, cublasDtrmv_v2, hipblasDtrmv)
WWR_FUNCTION(wwrblasCtrmv, cublasCtrmv_v2, hipblasCtrmv)
WWR_FUNCTION(wwrblasZtrmv, cublasZtrmv_v2, hipblasZtrmv)
WWR_FUNCTION(wwrblasStrmv_64, cublasStrmv_v2_64, hipblasStrmv_64)
WWR_FUNCTION(wwrblasDtrmv_64, cublasDtrmv_v2_64, hipblasDtrmv_64)
WWR_FUNCTION(wwrblasCtrmv_64, cublasCtrmv_v2_64, hipblasCtrmv_64)
WWR_FUNCTION(wwrblasZtrmv_64, cublasZtrmv_v2_64, hipblasZtrmv_64)

WWR_FUNCTION(wwrblasStrsv, cublasStrsv_v2, hipblasStrsv)
WWR_FUNCTION(wwrblasDtrsv, cublasDtrsv_v2, hipblasDtrsv)
WWR_FUNCTION(wwrblasCtrsv, cublasCtrsv_v2, hipblasCtrsv)
WWR_FUNCTION(wwrblasZtrsv, cublasZtrsv_v2, hipblasZtrsv)
WWR_FUNCTION(wwrblasStrsv_64, cublasStrsv_v2_64, hipblasStrsv_64)
WWR_FUNCTION(wwrblasDtrsv_64, cublasDtrsv_v2_64, hipblasDtrsv_64)
WWR_FUNCTION(wwrblasCtrsv_64, cublasCtrsv_v2_64, hipblasCtrsv_64)
WWR_FUNCTION(wwrblasZtrsv_64, cublasZtrsv_v2_64, hipblasZtrsv_64)

// ────────────────────────────────────────────────────────────────────────
// Level 3 (matrix-matrix)
// ────────────────────────────────────────────────────────────────────────

WWR_FUNCTION(wwrblasSgemm, cublasSgemm_v2, hipblasSgemm)
WWR_FUNCTION(wwrblasDgemm, cublasDgemm_v2, hipblasDgemm)
WWR_FUNCTION(wwrblasCgemm, cublasCgemm_v2, hipblasCgemm)
WWR_FUNCTION(wwrblasZgemm, cublasZgemm_v2, hipblasZgemm)
WWR_FUNCTION(wwrblasSgemm_64, cublasSgemm_v2_64, hipblasSgemm_64)
WWR_FUNCTION(wwrblasDgemm_64, cublasDgemm_v2_64, hipblasDgemm_64)
WWR_FUNCTION(wwrblasCgemm_64, cublasCgemm_v2_64, hipblasCgemm_64)
WWR_FUNCTION(wwrblasZgemm_64, cublasZgemm_v2_64, hipblasZgemm_64)

WWR_FUNCTION(wwrblasSgemmBatched, cublasSgemmBatched, hipblasSgemmBatched)
WWR_FUNCTION(wwrblasDgemmBatched, cublasDgemmBatched, hipblasDgemmBatched)
WWR_FUNCTION(wwrblasCgemmBatched, cublasCgemmBatched, hipblasCgemmBatched)
WWR_FUNCTION(wwrblasZgemmBatched, cublasZgemmBatched, hipblasZgemmBatched)
WWR_FUNCTION(wwrblasSgemmBatched_64, cublasSgemmBatched_64, hipblasSgemmBatched_64)
WWR_FUNCTION(wwrblasDgemmBatched_64, cublasDgemmBatched_64, hipblasDgemmBatched_64)
WWR_FUNCTION(wwrblasCgemmBatched_64, cublasCgemmBatched_64, hipblasCgemmBatched_64)
WWR_FUNCTION(wwrblasZgemmBatched_64, cublasZgemmBatched_64, hipblasZgemmBatched_64)

WWR_FUNCTION(wwrblasSgemmStridedBatched, cublasSgemmStridedBatched, hipblasSgemmStridedBatched)
WWR_FUNCTION(wwrblasDgemmStridedBatched, cublasDgemmStridedBatched, hipblasDgemmStridedBatched)
WWR_FUNCTION(wwrblasCgemmStridedBatched, cublasCgemmStridedBatched, hipblasCgemmStridedBatched)
WWR_FUNCTION(wwrblasZgemmStridedBatched, cublasZgemmStridedBatched, hipblasZgemmStridedBatched)
WWR_FUNCTION(wwrblasSgemmStridedBatched_64, cublasSgemmStridedBatched_64,
                hipblasSgemmStridedBatched_64)
WWR_FUNCTION(wwrblasDgemmStridedBatched_64, cublasDgemmStridedBatched_64,
                hipblasDgemmStridedBatched_64)
WWR_FUNCTION(wwrblasCgemmStridedBatched_64, cublasCgemmStridedBatched_64,
                hipblasCgemmStridedBatched_64)
WWR_FUNCTION(wwrblasZgemmStridedBatched_64, cublasZgemmStridedBatched_64,
                hipblasZgemmStridedBatched_64)

WWR_FUNCTION(wwrblasChemm, cublasChemm_v2, hipblasChemm)
WWR_FUNCTION(wwrblasZhemm, cublasZhemm_v2, hipblasZhemm)
WWR_FUNCTION(wwrblasChemm_64, cublasChemm_v2_64, hipblasChemm_64)
WWR_FUNCTION(wwrblasZhemm_64, cublasZhemm_v2_64, hipblasZhemm_64)

WWR_FUNCTION(wwrblasCher2k, cublasCher2k_v2, hipblasCher2k)
WWR_FUNCTION(wwrblasZher2k, cublasZher2k_v2, hipblasZher2k)
WWR_FUNCTION(wwrblasCher2k_64, cublasCher2k_v2_64, hipblasCher2k_64)
WWR_FUNCTION(wwrblasZher2k_64, cublasZher2k_v2_64, hipblasZher2k_64)

WWR_FUNCTION(wwrblasCherk, cublasCherk_v2, hipblasCherk)
WWR_FUNCTION(wwrblasZherk, cublasZherk_v2, hipblasZherk)
WWR_FUNCTION(wwrblasCherk_64, cublasCherk_v2_64, hipblasCherk_64)
WWR_FUNCTION(wwrblasZherk_64, cublasZherk_v2_64, hipblasZherk_64)

WWR_FUNCTION(wwrblasCherkx, cublasCherkx, hipblasCherkx)
WWR_FUNCTION(wwrblasZherkx, cublasZherkx, hipblasZherkx)
WWR_FUNCTION(wwrblasCherkx_64, cublasCherkx_64, hipblasCherkx_64)
WWR_FUNCTION(wwrblasZherkx_64, cublasZherkx_64, hipblasZherkx_64)

WWR_FUNCTION(wwrblasSsymm, cublasSsymm_v2, hipblasSsymm)
WWR_FUNCTION(wwrblasDsymm, cublasDsymm_v2, hipblasDsymm)
WWR_FUNCTION(wwrblasCsymm, cublasCsymm_v2, hipblasCsymm)
WWR_FUNCTION(wwrblasZsymm, cublasZsymm_v2, hipblasZsymm)
WWR_FUNCTION(wwrblasSsymm_64, cublasSsymm_v2_64, hipblasSsymm_64)
WWR_FUNCTION(wwrblasDsymm_64, cublasDsymm_v2_64, hipblasDsymm_64)
WWR_FUNCTION(wwrblasCsymm_64, cublasCsymm_v2_64, hipblasCsymm_64)
WWR_FUNCTION(wwrblasZsymm_64, cublasZsymm_v2_64, hipblasZsymm_64)

WWR_FUNCTION(wwrblasSsyr2k, cublasSsyr2k_v2, hipblasSsyr2k)
WWR_FUNCTION(wwrblasDsyr2k, cublasDsyr2k_v2, hipblasDsyr2k)
WWR_FUNCTION(wwrblasCsyr2k, cublasCsyr2k_v2, hipblasCsyr2k)
WWR_FUNCTION(wwrblasZsyr2k, cublasZsyr2k_v2, hipblasZsyr2k)
WWR_FUNCTION(wwrblasSsyr2k_64, cublasSsyr2k_v2_64, hipblasSsyr2k_64)
WWR_FUNCTION(wwrblasDsyr2k_64, cublasDsyr2k_v2_64, hipblasDsyr2k_64)
WWR_FUNCTION(wwrblasCsyr2k_64, cublasCsyr2k_v2_64, hipblasCsyr2k_64)
WWR_FUNCTION(wwrblasZsyr2k_64, cublasZsyr2k_v2_64, hipblasZsyr2k_64)

WWR_FUNCTION(wwrblasSsyrk, cublasSsyrk_v2, hipblasSsyrk)
WWR_FUNCTION(wwrblasDsyrk, cublasDsyrk_v2, hipblasDsyrk)
WWR_FUNCTION(wwrblasCsyrk, cublasCsyrk_v2, hipblasCsyrk)
WWR_FUNCTION(wwrblasZsyrk, cublasZsyrk_v2, hipblasZsyrk)
WWR_FUNCTION(wwrblasSsyrk_64, cublasSsyrk_v2_64, hipblasSsyrk_64)
WWR_FUNCTION(wwrblasDsyrk_64, cublasDsyrk_v2_64, hipblasDsyrk_64)
WWR_FUNCTION(wwrblasCsyrk_64, cublasCsyrk_v2_64, hipblasCsyrk_64)
WWR_FUNCTION(wwrblasZsyrk_64, cublasZsyrk_v2_64, hipblasZsyrk_64)

WWR_FUNCTION(wwrblasSsyrkx, cublasSsyrkx, hipblasSsyrkx)
WWR_FUNCTION(wwrblasDsyrkx, cublasDsyrkx, hipblasDsyrkx)
WWR_FUNCTION(wwrblasCsyrkx, cublasCsyrkx, hipblasCsyrkx)
WWR_FUNCTION(wwrblasZsyrkx, cublasZsyrkx, hipblasZsyrkx)
WWR_FUNCTION(wwrblasSsyrkx_64, cublasSsyrkx_64, hipblasSsyrkx_64)
WWR_FUNCTION(wwrblasDsyrkx_64, cublasDsyrkx_64, hipblasDsyrkx_64)
WWR_FUNCTION(wwrblasCsyrkx_64, cublasCsyrkx_64, hipblasCsyrkx_64)
WWR_FUNCTION(wwrblasZsyrkx_64, cublasZsyrkx_64, hipblasZsyrkx_64)

WWR_FUNCTION(wwrblasStrmm, cublasStrmm_v2, hipblasStrmm)
WWR_FUNCTION(wwrblasDtrmm, cublasDtrmm_v2, hipblasDtrmm)
WWR_FUNCTION(wwrblasCtrmm, cublasCtrmm_v2, hipblasCtrmm)
WWR_FUNCTION(wwrblasZtrmm, cublasZtrmm_v2, hipblasZtrmm)
WWR_FUNCTION(wwrblasStrmm_64, cublasStrmm_v2_64, hipblasStrmm_64)
WWR_FUNCTION(wwrblasDtrmm_64, cublasDtrmm_v2_64, hipblasDtrmm_64)
WWR_FUNCTION(wwrblasCtrmm_64, cublasCtrmm_v2_64, hipblasCtrmm_64)
WWR_FUNCTION(wwrblasZtrmm_64, cublasZtrmm_v2_64, hipblasZtrmm_64)

WWR_FUNCTION(wwrblasStrsm, cublasStrsm_v2, hipblasStrsm)
WWR_FUNCTION(wwrblasDtrsm, cublasDtrsm_v2, hipblasDtrsm)
WWR_FUNCTION(wwrblasCtrsm, cublasCtrsm_v2, hipblasCtrsm)
WWR_FUNCTION(wwrblasZtrsm, cublasZtrsm_v2, hipblasZtrsm)
WWR_FUNCTION(wwrblasStrsm_64, cublasStrsm_v2_64, hipblasStrsm_64)
WWR_FUNCTION(wwrblasDtrsm_64, cublasDtrsm_v2_64, hipblasDtrsm_64)
WWR_FUNCTION(wwrblasCtrsm_64, cublasCtrsm_v2_64, hipblasCtrsm_64)
WWR_FUNCTION(wwrblasZtrsm_64, cublasZtrsm_v2_64, hipblasZtrsm_64)

WWR_FUNCTION(wwrblasStrsmBatched, cublasStrsmBatched, hipblasStrsmBatched)
WWR_FUNCTION(wwrblasDtrsmBatched, cublasDtrsmBatched, hipblasDtrsmBatched)
WWR_FUNCTION(wwrblasCtrsmBatched, cublasCtrsmBatched, hipblasCtrsmBatched)
WWR_FUNCTION(wwrblasZtrsmBatched, cublasZtrsmBatched, hipblasZtrsmBatched)
WWR_FUNCTION(wwrblasStrsmBatched_64, cublasStrsmBatched_64, hipblasStrsmBatched_64)
WWR_FUNCTION(wwrblasDtrsmBatched_64, cublasDtrsmBatched_64, hipblasDtrsmBatched_64)
WWR_FUNCTION(wwrblasCtrsmBatched_64, cublasCtrsmBatched_64, hipblasCtrsmBatched_64)
WWR_FUNCTION(wwrblasZtrsmBatched_64, cublasZtrsmBatched_64, hipblasZtrsmBatched_64)

// ────────────────────────────────────────────────────────────────────────
// BLAS-like extensions
// ────────────────────────────────────────────────────────────────────────

WWR_FUNCTION(wwrblasSdgmm, cublasSdgmm, hipblasSdgmm)
WWR_FUNCTION(wwrblasDdgmm, cublasDdgmm, hipblasDdgmm)
WWR_FUNCTION(wwrblasCdgmm, cublasCdgmm, hipblasCdgmm)
WWR_FUNCTION(wwrblasZdgmm, cublasZdgmm, hipblasZdgmm)
WWR_FUNCTION(wwrblasSdgmm_64, cublasSdgmm_64, hipblasSdgmm_64)
WWR_FUNCTION(wwrblasDdgmm_64, cublasDdgmm_64, hipblasDdgmm_64)
WWR_FUNCTION(wwrblasCdgmm_64, cublasCdgmm_64, hipblasCdgmm_64)
WWR_FUNCTION(wwrblasZdgmm_64, cublasZdgmm_64, hipblasZdgmm_64)

WWR_FUNCTION(wwrblasSgeam, cublasSgeam, hipblasSgeam)
WWR_FUNCTION(wwrblasDgeam, cublasDgeam, hipblasDgeam)
WWR_FUNCTION(wwrblasCgeam, cublasCgeam, hipblasCgeam)
WWR_FUNCTION(wwrblasZgeam, cublasZgeam, hipblasZgeam)
WWR_FUNCTION(wwrblasSgeam_64, cublasSgeam_64, hipblasSgeam_64)
WWR_FUNCTION(wwrblasDgeam_64, cublasDgeam_64, hipblasDgeam_64)
WWR_FUNCTION(wwrblasCgeam_64, cublasCgeam_64, hipblasCgeam_64)
WWR_FUNCTION(wwrblasZgeam_64, cublasZgeam_64, hipblasZgeam_64)

WWR_FUNCTION(wwrblasSgelsBatched, cublasSgelsBatched, hipblasSgelsBatched)
WWR_FUNCTION(wwrblasDgelsBatched, cublasDgelsBatched, hipblasDgelsBatched)
WWR_FUNCTION(wwrblasCgelsBatched, cublasCgelsBatched, hipblasCgelsBatched)
WWR_FUNCTION(wwrblasZgelsBatched, cublasZgelsBatched, hipblasZgelsBatched)

WWR_FUNCTION(wwrblasSgeqrfBatched, cublasSgeqrfBatched, hipblasSgeqrfBatched)
WWR_FUNCTION(wwrblasDgeqrfBatched, cublasDgeqrfBatched, hipblasDgeqrfBatched)
WWR_FUNCTION(wwrblasCgeqrfBatched, cublasCgeqrfBatched, hipblasCgeqrfBatched)
WWR_FUNCTION(wwrblasZgeqrfBatched, cublasZgeqrfBatched, hipblasZgeqrfBatched)

WWR_FUNCTION(wwrblasSgetrfBatched, cublasSgetrfBatched, hipblasSgetrfBatched)
WWR_FUNCTION(wwrblasDgetrfBatched, cublasDgetrfBatched, hipblasDgetrfBatched)
WWR_FUNCTION(wwrblasCgetrfBatched, cublasCgetrfBatched, hipblasCgetrfBatched)
WWR_FUNCTION(wwrblasZgetrfBatched, cublasZgetrfBatched, hipblasZgetrfBatched)

// ────────────────────────────────────────────────────────────────────────
// getrsBatched / getriBatched: const-correct on both backends
// ────────────────────────────────────────────────────────────────────────

#if defined(WWR_GPU_BACKEND_CUDA)

WWR_FUNCTION(wwrblasSgetrsBatched, cublasSgetrsBatched, hipblasSgetrsBatched)
WWR_FUNCTION(wwrblasDgetrsBatched, cublasDgetrsBatched, hipblasDgetrsBatched)
WWR_FUNCTION(wwrblasCgetrsBatched, cublasCgetrsBatched, hipblasCgetrsBatched)
WWR_FUNCTION(wwrblasZgetrsBatched, cublasZgetrsBatched, hipblasZgetrsBatched)

WWR_FUNCTION(wwrblasSgetriBatched, cublasSgetriBatched, hipblasSgetriBatched)
WWR_FUNCTION(wwrblasDgetriBatched, cublasDgetriBatched, hipblasDgetriBatched)
WWR_FUNCTION(wwrblasCgetriBatched, cublasCgetriBatched, hipblasCgetriBatched)
WWR_FUNCTION(wwrblasZgetriBatched, cublasZgetriBatched, hipblasZgetriBatched)

#else

// hipBLAS declares Aarray as T* const[] (and getriBatched's P as int*) where
// cuBLAS declares const T* const[] (and const int*). hipBLAS only reads them;
// these keep cuBLAS's signature so src/wrappers has one const-correct API.

inline wwrblasStatus_t wwrblasSgetrsBatched(wwrblasHandle_t handle, wwrblasOperation_t trans, int n,
                                            int nrhs, const float *const Aarray[], int lda,
                                            const int *devIpiv, float *const Barray[], int ldb,
                                            int *info, int batchSize) {
  return hip::hipblasSgetrsBatched(handle, trans, n, nrhs, const_cast<float *const *>(Aarray), lda,
                                   devIpiv, Barray, ldb, info, batchSize);
}
inline wwrblasStatus_t wwrblasDgetrsBatched(wwrblasHandle_t handle, wwrblasOperation_t trans, int n,
                                            int nrhs, const double *const Aarray[], int lda,
                                            const int *devIpiv, double *const Barray[], int ldb,
                                            int *info, int batchSize) {
  return hip::hipblasDgetrsBatched(handle, trans, n, nrhs, const_cast<double *const *>(Aarray), lda,
                                   devIpiv, Barray, ldb, info, batchSize);
}
inline wwrblasStatus_t wwrblasCgetrsBatched(wwrblasHandle_t handle, wwrblasOperation_t trans, int n,
                                            int nrhs, const wwrFloatComplex *const Aarray[],
                                            int lda, const int *devIpiv,
                                            wwrFloatComplex *const Barray[], int ldb, int *info,
                                            int batchSize) {
  return hip::hipblasCgetrsBatched(handle, trans, n, nrhs,
                                   const_cast<wwrFloatComplex *const *>(Aarray), lda, devIpiv,
                                   Barray, ldb, info, batchSize);
}
inline wwrblasStatus_t wwrblasZgetrsBatched(wwrblasHandle_t handle, wwrblasOperation_t trans, int n,
                                            int nrhs, const wwrDoubleComplex *const Aarray[],
                                            int lda, const int *devIpiv,
                                            wwrDoubleComplex *const Barray[], int ldb, int *info,
                                            int batchSize) {
  return hip::hipblasZgetrsBatched(handle, trans, n, nrhs,
                                   const_cast<wwrDoubleComplex *const *>(Aarray), lda, devIpiv,
                                   Barray, ldb, info, batchSize);
}

inline wwrblasStatus_t wwrblasSgetriBatched(wwrblasHandle_t handle, int n, const float *const A[],
                                            int lda, const int *P, float *const C[], int ldc,
                                            int *info, int batchSize) {
  return hip::hipblasSgetriBatched(handle, n, const_cast<float *const *>(A), lda,
                                   const_cast<int *>(P), C, ldc, info, batchSize);
}
inline wwrblasStatus_t wwrblasDgetriBatched(wwrblasHandle_t handle, int n, const double *const A[],
                                            int lda, const int *P, double *const C[], int ldc,
                                            int *info, int batchSize) {
  return hip::hipblasDgetriBatched(handle, n, const_cast<double *const *>(A), lda,
                                   const_cast<int *>(P), C, ldc, info, batchSize);
}
inline wwrblasStatus_t wwrblasCgetriBatched(wwrblasHandle_t handle, int n,
                                            const wwrFloatComplex *const A[], int lda, const int *P,
                                            wwrFloatComplex *const C[], int ldc, int *info,
                                            int batchSize) {
  return hip::hipblasCgetriBatched(handle, n, const_cast<wwrFloatComplex *const *>(A), lda,
                                   const_cast<int *>(P), C, ldc, info, batchSize);
}
inline wwrblasStatus_t wwrblasZgetriBatched(wwrblasHandle_t handle, int n,
                                            const wwrDoubleComplex *const A[], int lda,
                                            const int *P, wwrDoubleComplex *const C[], int ldc,
                                            int *info, int batchSize) {
  return hip::hipblasZgetriBatched(handle, n, const_cast<wwrDoubleComplex *const *>(A), lda,
                                   const_cast<int *>(P), C, ldc, info, batchSize);
}

#endif

} // namespace wwr
