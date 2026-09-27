/**
 * @file blas.cppm
 * @brief Backend-neutral BLAS: gpublas* names for cuBLAS / hipBLAS
 *
 * gpublas<X> stands for cublas<X>_v2 where cuBLAS has a _v2 name pair and
 * cublas<X> otherwise, and for hipblas<X>, which has no _v2 names; 64-bit
 * variants are gpublas<X>_64. Each is written out in full, one line per name.
 * See gpu_backend.h.
 *
 * Only the names src/wrappers/blas uses are listed, plus the handle, stream
 * and pointer-mode calls a caller needs. cuBLAS-only functions (gemm3m,
 * gemmGroupedBatched, matinvBatched, tpttr, trttp) are absent; reach those
 * through wwr.cuda.cublas_v2.
 *
 * Two backend differences are resolved here, not above: hipBLAS's one
 * status-to-string function backs both gpublasGetStatusName and
 * gpublasGetStatusString, and getrsBatched / getriBatched keep cuBLAS's
 * const-correct signature, forwarding with a const_cast on HIP.
 * See docs/architecture.md, section 5.
 *
 * Usage:
 *   import wwr.blas;
 *
 *   gpublasHandle_t handle;
 *   gpublasCreate(&handle);
 */

module;

#include "gpu_backend.h"

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

WWR_TYPE(gpublasHandle_t, cublasHandle_t, hipblasHandle_t)
WWR_TYPE(gpublasStatus_t, cublasStatus_t, hipblasStatus_t)
WWR_TYPE(gpublasOperation_t, cublasOperation_t, hipblasOperation_t)
WWR_TYPE(gpublasFillMode_t, cublasFillMode_t, hipblasFillMode_t)
WWR_TYPE(gpublasDiagType_t, cublasDiagType_t, hipblasDiagType_t)
WWR_TYPE(gpublasSideMode_t, cublasSideMode_t, hipblasSideMode_t)
WWR_TYPE(gpublasPointerMode_t, cublasPointerMode_t, hipblasPointerMode_t)

// ========================================================================
// Constants
// ========================================================================

WWR_VALUE(GPUBLAS_STATUS_SUCCESS, CUBLAS_STATUS_SUCCESS, HIPBLAS_STATUS_SUCCESS)
WWR_VALUE(GPUBLAS_STATUS_NOT_INITIALIZED, CUBLAS_STATUS_NOT_INITIALIZED,
             HIPBLAS_STATUS_NOT_INITIALIZED)

WWR_VALUE(GPUBLAS_OP_N, CUBLAS_OP_N, HIPBLAS_OP_N)
WWR_VALUE(GPUBLAS_OP_T, CUBLAS_OP_T, HIPBLAS_OP_T)
WWR_VALUE(GPUBLAS_OP_C, CUBLAS_OP_C, HIPBLAS_OP_C)

WWR_VALUE(GPUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_LOWER, HIPBLAS_FILL_MODE_LOWER)
WWR_VALUE(GPUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_UPPER, HIPBLAS_FILL_MODE_UPPER)

WWR_VALUE(GPUBLAS_DIAG_NON_UNIT, CUBLAS_DIAG_NON_UNIT, HIPBLAS_DIAG_NON_UNIT)
WWR_VALUE(GPUBLAS_DIAG_UNIT, CUBLAS_DIAG_UNIT, HIPBLAS_DIAG_UNIT)

WWR_VALUE(GPUBLAS_SIDE_LEFT, CUBLAS_SIDE_LEFT, HIPBLAS_SIDE_LEFT)
WWR_VALUE(GPUBLAS_SIDE_RIGHT, CUBLAS_SIDE_RIGHT, HIPBLAS_SIDE_RIGHT)

WWR_VALUE(GPUBLAS_POINTER_MODE_HOST, CUBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_HOST)
WWR_VALUE(GPUBLAS_POINTER_MODE_DEVICE, CUBLAS_POINTER_MODE_DEVICE, HIPBLAS_POINTER_MODE_DEVICE)

// ========================================================================
// Handle, stream, pointer mode, status strings
// ========================================================================

WWR_FUNCTION(gpublasCreate, cublasCreate_v2, hipblasCreate)
WWR_FUNCTION(gpublasDestroy, cublasDestroy_v2, hipblasDestroy)
WWR_FUNCTION(gpublasSetStream, cublasSetStream_v2, hipblasSetStream)
WWR_FUNCTION(gpublasGetStream, cublasGetStream_v2, hipblasGetStream)
WWR_FUNCTION(gpublasSetPointerMode, cublasSetPointerMode_v2, hipblasSetPointerMode)

// hipBLAS has a single status-to-string function; both names map to it.
WWR_FUNCTION(gpublasGetStatusName, cublasGetStatusName, hipblasStatusToString)
WWR_FUNCTION(gpublasGetStatusString, cublasGetStatusString, hipblasStatusToString)

// ────────────────────────────────────────────────────────────────────────
// Level 1 (vector-vector)
// ────────────────────────────────────────────────────────────────────────

WWR_FUNCTION(gpublasSasum, cublasSasum_v2, hipblasSasum)
WWR_FUNCTION(gpublasDasum, cublasDasum_v2, hipblasDasum)
WWR_FUNCTION(gpublasSasum_64, cublasSasum_v2_64, hipblasSasum_64)
WWR_FUNCTION(gpublasDasum_64, cublasDasum_v2_64, hipblasDasum_64)

WWR_FUNCTION(gpublasSaxpy, cublasSaxpy_v2, hipblasSaxpy)
WWR_FUNCTION(gpublasDaxpy, cublasDaxpy_v2, hipblasDaxpy)
WWR_FUNCTION(gpublasCaxpy, cublasCaxpy_v2, hipblasCaxpy)
WWR_FUNCTION(gpublasZaxpy, cublasZaxpy_v2, hipblasZaxpy)
WWR_FUNCTION(gpublasSaxpy_64, cublasSaxpy_v2_64, hipblasSaxpy_64)
WWR_FUNCTION(gpublasDaxpy_64, cublasDaxpy_v2_64, hipblasDaxpy_64)
WWR_FUNCTION(gpublasCaxpy_64, cublasCaxpy_v2_64, hipblasCaxpy_64)
WWR_FUNCTION(gpublasZaxpy_64, cublasZaxpy_v2_64, hipblasZaxpy_64)

WWR_FUNCTION(gpublasScasum, cublasScasum_v2, hipblasScasum)
WWR_FUNCTION(gpublasScasum_64, cublasScasum_v2_64, hipblasScasum_64)

WWR_FUNCTION(gpublasScnrm2, cublasScnrm2_v2, hipblasScnrm2)
WWR_FUNCTION(gpublasScnrm2_64, cublasScnrm2_v2_64, hipblasScnrm2_64)

WWR_FUNCTION(gpublasScopy, cublasScopy_v2, hipblasScopy)
WWR_FUNCTION(gpublasDcopy, cublasDcopy_v2, hipblasDcopy)
WWR_FUNCTION(gpublasCcopy, cublasCcopy_v2, hipblasCcopy)
WWR_FUNCTION(gpublasZcopy, cublasZcopy_v2, hipblasZcopy)
WWR_FUNCTION(gpublasScopy_64, cublasScopy_v2_64, hipblasScopy_64)
WWR_FUNCTION(gpublasDcopy_64, cublasDcopy_v2_64, hipblasDcopy_64)
WWR_FUNCTION(gpublasCcopy_64, cublasCcopy_v2_64, hipblasCcopy_64)
WWR_FUNCTION(gpublasZcopy_64, cublasZcopy_v2_64, hipblasZcopy_64)

WWR_FUNCTION(gpublasSdot, cublasSdot_v2, hipblasSdot)
WWR_FUNCTION(gpublasDdot, cublasDdot_v2, hipblasDdot)
WWR_FUNCTION(gpublasSdot_64, cublasSdot_v2_64, hipblasSdot_64)
WWR_FUNCTION(gpublasDdot_64, cublasDdot_v2_64, hipblasDdot_64)

WWR_FUNCTION(gpublasCdotc, cublasCdotc_v2, hipblasCdotc)
WWR_FUNCTION(gpublasZdotc, cublasZdotc_v2, hipblasZdotc)
WWR_FUNCTION(gpublasCdotc_64, cublasCdotc_v2_64, hipblasCdotc_64)
WWR_FUNCTION(gpublasZdotc_64, cublasZdotc_v2_64, hipblasZdotc_64)

WWR_FUNCTION(gpublasCdotu, cublasCdotu_v2, hipblasCdotu)
WWR_FUNCTION(gpublasZdotu, cublasZdotu_v2, hipblasZdotu)
WWR_FUNCTION(gpublasCdotu_64, cublasCdotu_v2_64, hipblasCdotu_64)
WWR_FUNCTION(gpublasZdotu_64, cublasZdotu_v2_64, hipblasZdotu_64)

WWR_FUNCTION(gpublasZdrot, cublasZdrot_v2, hipblasZdrot)
WWR_FUNCTION(gpublasZdrot_64, cublasZdrot_v2_64, hipblasZdrot_64)

WWR_FUNCTION(gpublasZdscal, cublasZdscal_v2, hipblasZdscal)
WWR_FUNCTION(gpublasZdscal_64, cublasZdscal_v2_64, hipblasZdscal_64)

WWR_FUNCTION(gpublasIsamax, cublasIsamax_v2, hipblasIsamax)
WWR_FUNCTION(gpublasIdamax, cublasIdamax_v2, hipblasIdamax)
WWR_FUNCTION(gpublasIcamax, cublasIcamax_v2, hipblasIcamax)
WWR_FUNCTION(gpublasIzamax, cublasIzamax_v2, hipblasIzamax)
WWR_FUNCTION(gpublasIsamax_64, cublasIsamax_v2_64, hipblasIsamax_64)
WWR_FUNCTION(gpublasIdamax_64, cublasIdamax_v2_64, hipblasIdamax_64)
WWR_FUNCTION(gpublasIcamax_64, cublasIcamax_v2_64, hipblasIcamax_64)
WWR_FUNCTION(gpublasIzamax_64, cublasIzamax_v2_64, hipblasIzamax_64)

WWR_FUNCTION(gpublasIsamin, cublasIsamin_v2, hipblasIsamin)
WWR_FUNCTION(gpublasIdamin, cublasIdamin_v2, hipblasIdamin)
WWR_FUNCTION(gpublasIcamin, cublasIcamin_v2, hipblasIcamin)
WWR_FUNCTION(gpublasIzamin, cublasIzamin_v2, hipblasIzamin)
WWR_FUNCTION(gpublasIsamin_64, cublasIsamin_v2_64, hipblasIsamin_64)
WWR_FUNCTION(gpublasIdamin_64, cublasIdamin_v2_64, hipblasIdamin_64)
WWR_FUNCTION(gpublasIcamin_64, cublasIcamin_v2_64, hipblasIcamin_64)
WWR_FUNCTION(gpublasIzamin_64, cublasIzamin_v2_64, hipblasIzamin_64)

WWR_FUNCTION(gpublasSnrm2, cublasSnrm2_v2, hipblasSnrm2)
WWR_FUNCTION(gpublasDnrm2, cublasDnrm2_v2, hipblasDnrm2)
WWR_FUNCTION(gpublasSnrm2_64, cublasSnrm2_v2_64, hipblasSnrm2_64)
WWR_FUNCTION(gpublasDnrm2_64, cublasDnrm2_v2_64, hipblasDnrm2_64)

WWR_FUNCTION(gpublasSrot, cublasSrot_v2, hipblasSrot)
WWR_FUNCTION(gpublasDrot, cublasDrot_v2, hipblasDrot)
WWR_FUNCTION(gpublasCrot, cublasCrot_v2, hipblasCrot)
WWR_FUNCTION(gpublasZrot, cublasZrot_v2, hipblasZrot)
WWR_FUNCTION(gpublasSrot_64, cublasSrot_v2_64, hipblasSrot_64)
WWR_FUNCTION(gpublasDrot_64, cublasDrot_v2_64, hipblasDrot_64)
WWR_FUNCTION(gpublasCrot_64, cublasCrot_v2_64, hipblasCrot_64)
WWR_FUNCTION(gpublasZrot_64, cublasZrot_v2_64, hipblasZrot_64)

WWR_FUNCTION(gpublasSrotg, cublasSrotg_v2, hipblasSrotg)
WWR_FUNCTION(gpublasDrotg, cublasDrotg_v2, hipblasDrotg)
WWR_FUNCTION(gpublasCrotg, cublasCrotg_v2, hipblasCrotg)
WWR_FUNCTION(gpublasZrotg, cublasZrotg_v2, hipblasZrotg)

WWR_FUNCTION(gpublasSrotm, cublasSrotm_v2, hipblasSrotm)
WWR_FUNCTION(gpublasDrotm, cublasDrotm_v2, hipblasDrotm)
WWR_FUNCTION(gpublasSrotm_64, cublasSrotm_v2_64, hipblasSrotm_64)
WWR_FUNCTION(gpublasDrotm_64, cublasDrotm_v2_64, hipblasDrotm_64)

WWR_FUNCTION(gpublasSrotmg, cublasSrotmg_v2, hipblasSrotmg)
WWR_FUNCTION(gpublasDrotmg, cublasDrotmg_v2, hipblasDrotmg)

WWR_FUNCTION(gpublasSscal, cublasSscal_v2, hipblasSscal)
WWR_FUNCTION(gpublasDscal, cublasDscal_v2, hipblasDscal)
WWR_FUNCTION(gpublasCscal, cublasCscal_v2, hipblasCscal)
WWR_FUNCTION(gpublasZscal, cublasZscal_v2, hipblasZscal)
WWR_FUNCTION(gpublasSscal_64, cublasSscal_v2_64, hipblasSscal_64)
WWR_FUNCTION(gpublasDscal_64, cublasDscal_v2_64, hipblasDscal_64)
WWR_FUNCTION(gpublasCscal_64, cublasCscal_v2_64, hipblasCscal_64)
WWR_FUNCTION(gpublasZscal_64, cublasZscal_v2_64, hipblasZscal_64)

WWR_FUNCTION(gpublasCsrot, cublasCsrot_v2, hipblasCsrot)
WWR_FUNCTION(gpublasCsrot_64, cublasCsrot_v2_64, hipblasCsrot_64)

WWR_FUNCTION(gpublasCsscal, cublasCsscal_v2, hipblasCsscal)
WWR_FUNCTION(gpublasCsscal_64, cublasCsscal_v2_64, hipblasCsscal_64)

WWR_FUNCTION(gpublasSswap, cublasSswap_v2, hipblasSswap)
WWR_FUNCTION(gpublasDswap, cublasDswap_v2, hipblasDswap)
WWR_FUNCTION(gpublasCswap, cublasCswap_v2, hipblasCswap)
WWR_FUNCTION(gpublasZswap, cublasZswap_v2, hipblasZswap)
WWR_FUNCTION(gpublasSswap_64, cublasSswap_v2_64, hipblasSswap_64)
WWR_FUNCTION(gpublasDswap_64, cublasDswap_v2_64, hipblasDswap_64)
WWR_FUNCTION(gpublasCswap_64, cublasCswap_v2_64, hipblasCswap_64)
WWR_FUNCTION(gpublasZswap_64, cublasZswap_v2_64, hipblasZswap_64)

WWR_FUNCTION(gpublasDzasum, cublasDzasum_v2, hipblasDzasum)
WWR_FUNCTION(gpublasDzasum_64, cublasDzasum_v2_64, hipblasDzasum_64)

WWR_FUNCTION(gpublasDznrm2, cublasDznrm2_v2, hipblasDznrm2)
WWR_FUNCTION(gpublasDznrm2_64, cublasDznrm2_v2_64, hipblasDznrm2_64)

// ────────────────────────────────────────────────────────────────────────
// Level 2 (matrix-vector)
// ────────────────────────────────────────────────────────────────────────

WWR_FUNCTION(gpublasSgbmv, cublasSgbmv_v2, hipblasSgbmv)
WWR_FUNCTION(gpublasDgbmv, cublasDgbmv_v2, hipblasDgbmv)
WWR_FUNCTION(gpublasCgbmv, cublasCgbmv_v2, hipblasCgbmv)
WWR_FUNCTION(gpublasZgbmv, cublasZgbmv_v2, hipblasZgbmv)
WWR_FUNCTION(gpublasSgbmv_64, cublasSgbmv_v2_64, hipblasSgbmv_64)
WWR_FUNCTION(gpublasDgbmv_64, cublasDgbmv_v2_64, hipblasDgbmv_64)
WWR_FUNCTION(gpublasCgbmv_64, cublasCgbmv_v2_64, hipblasCgbmv_64)
WWR_FUNCTION(gpublasZgbmv_64, cublasZgbmv_v2_64, hipblasZgbmv_64)

WWR_FUNCTION(gpublasSgemv, cublasSgemv_v2, hipblasSgemv)
WWR_FUNCTION(gpublasDgemv, cublasDgemv_v2, hipblasDgemv)
WWR_FUNCTION(gpublasCgemv, cublasCgemv_v2, hipblasCgemv)
WWR_FUNCTION(gpublasZgemv, cublasZgemv_v2, hipblasZgemv)
WWR_FUNCTION(gpublasSgemv_64, cublasSgemv_v2_64, hipblasSgemv_64)
WWR_FUNCTION(gpublasDgemv_64, cublasDgemv_v2_64, hipblasDgemv_64)
WWR_FUNCTION(gpublasCgemv_64, cublasCgemv_v2_64, hipblasCgemv_64)
WWR_FUNCTION(gpublasZgemv_64, cublasZgemv_v2_64, hipblasZgemv_64)

WWR_FUNCTION(gpublasSgemvBatched, cublasSgemvBatched, hipblasSgemvBatched)
WWR_FUNCTION(gpublasDgemvBatched, cublasDgemvBatched, hipblasDgemvBatched)
WWR_FUNCTION(gpublasCgemvBatched, cublasCgemvBatched, hipblasCgemvBatched)
WWR_FUNCTION(gpublasZgemvBatched, cublasZgemvBatched, hipblasZgemvBatched)
WWR_FUNCTION(gpublasSgemvBatched_64, cublasSgemvBatched_64, hipblasSgemvBatched_64)
WWR_FUNCTION(gpublasDgemvBatched_64, cublasDgemvBatched_64, hipblasDgemvBatched_64)
WWR_FUNCTION(gpublasCgemvBatched_64, cublasCgemvBatched_64, hipblasCgemvBatched_64)
WWR_FUNCTION(gpublasZgemvBatched_64, cublasZgemvBatched_64, hipblasZgemvBatched_64)

WWR_FUNCTION(gpublasSgemvStridedBatched, cublasSgemvStridedBatched, hipblasSgemvStridedBatched)
WWR_FUNCTION(gpublasDgemvStridedBatched, cublasDgemvStridedBatched, hipblasDgemvStridedBatched)
WWR_FUNCTION(gpublasCgemvStridedBatched, cublasCgemvStridedBatched, hipblasCgemvStridedBatched)
WWR_FUNCTION(gpublasZgemvStridedBatched, cublasZgemvStridedBatched, hipblasZgemvStridedBatched)
WWR_FUNCTION(gpublasSgemvStridedBatched_64, cublasSgemvStridedBatched_64,
                hipblasSgemvStridedBatched_64)
WWR_FUNCTION(gpublasDgemvStridedBatched_64, cublasDgemvStridedBatched_64,
                hipblasDgemvStridedBatched_64)
WWR_FUNCTION(gpublasCgemvStridedBatched_64, cublasCgemvStridedBatched_64,
                hipblasCgemvStridedBatched_64)
WWR_FUNCTION(gpublasZgemvStridedBatched_64, cublasZgemvStridedBatched_64,
                hipblasZgemvStridedBatched_64)

WWR_FUNCTION(gpublasSger, cublasSger_v2, hipblasSger)
WWR_FUNCTION(gpublasDger, cublasDger_v2, hipblasDger)
WWR_FUNCTION(gpublasSger_64, cublasSger_v2_64, hipblasSger_64)
WWR_FUNCTION(gpublasDger_64, cublasDger_v2_64, hipblasDger_64)

WWR_FUNCTION(gpublasCgerc, cublasCgerc_v2, hipblasCgerc)
WWR_FUNCTION(gpublasZgerc, cublasZgerc_v2, hipblasZgerc)
WWR_FUNCTION(gpublasCgerc_64, cublasCgerc_v2_64, hipblasCgerc_64)
WWR_FUNCTION(gpublasZgerc_64, cublasZgerc_v2_64, hipblasZgerc_64)

WWR_FUNCTION(gpublasCgeru, cublasCgeru_v2, hipblasCgeru)
WWR_FUNCTION(gpublasZgeru, cublasZgeru_v2, hipblasZgeru)
WWR_FUNCTION(gpublasCgeru_64, cublasCgeru_v2_64, hipblasCgeru_64)
WWR_FUNCTION(gpublasZgeru_64, cublasZgeru_v2_64, hipblasZgeru_64)

WWR_FUNCTION(gpublasChbmv, cublasChbmv_v2, hipblasChbmv)
WWR_FUNCTION(gpublasZhbmv, cublasZhbmv_v2, hipblasZhbmv)
WWR_FUNCTION(gpublasChbmv_64, cublasChbmv_v2_64, hipblasChbmv_64)
WWR_FUNCTION(gpublasZhbmv_64, cublasZhbmv_v2_64, hipblasZhbmv_64)

WWR_FUNCTION(gpublasChemv, cublasChemv_v2, hipblasChemv)
WWR_FUNCTION(gpublasZhemv, cublasZhemv_v2, hipblasZhemv)
WWR_FUNCTION(gpublasChemv_64, cublasChemv_v2_64, hipblasChemv_64)
WWR_FUNCTION(gpublasZhemv_64, cublasZhemv_v2_64, hipblasZhemv_64)

WWR_FUNCTION(gpublasCher, cublasCher_v2, hipblasCher)
WWR_FUNCTION(gpublasZher, cublasZher_v2, hipblasZher)
WWR_FUNCTION(gpublasCher_64, cublasCher_v2_64, hipblasCher_64)
WWR_FUNCTION(gpublasZher_64, cublasZher_v2_64, hipblasZher_64)

WWR_FUNCTION(gpublasCher2, cublasCher2_v2, hipblasCher2)
WWR_FUNCTION(gpublasZher2, cublasZher2_v2, hipblasZher2)
WWR_FUNCTION(gpublasCher2_64, cublasCher2_v2_64, hipblasCher2_64)
WWR_FUNCTION(gpublasZher2_64, cublasZher2_v2_64, hipblasZher2_64)

WWR_FUNCTION(gpublasChpmv, cublasChpmv_v2, hipblasChpmv)
WWR_FUNCTION(gpublasZhpmv, cublasZhpmv_v2, hipblasZhpmv)
WWR_FUNCTION(gpublasChpmv_64, cublasChpmv_v2_64, hipblasChpmv_64)
WWR_FUNCTION(gpublasZhpmv_64, cublasZhpmv_v2_64, hipblasZhpmv_64)

WWR_FUNCTION(gpublasChpr, cublasChpr_v2, hipblasChpr)
WWR_FUNCTION(gpublasZhpr, cublasZhpr_v2, hipblasZhpr)
WWR_FUNCTION(gpublasChpr_64, cublasChpr_v2_64, hipblasChpr_64)
WWR_FUNCTION(gpublasZhpr_64, cublasZhpr_v2_64, hipblasZhpr_64)

WWR_FUNCTION(gpublasChpr2, cublasChpr2_v2, hipblasChpr2)
WWR_FUNCTION(gpublasZhpr2, cublasZhpr2_v2, hipblasZhpr2)
WWR_FUNCTION(gpublasChpr2_64, cublasChpr2_v2_64, hipblasChpr2_64)
WWR_FUNCTION(gpublasZhpr2_64, cublasZhpr2_v2_64, hipblasZhpr2_64)

WWR_FUNCTION(gpublasSsbmv, cublasSsbmv_v2, hipblasSsbmv)
WWR_FUNCTION(gpublasDsbmv, cublasDsbmv_v2, hipblasDsbmv)
WWR_FUNCTION(gpublasSsbmv_64, cublasSsbmv_v2_64, hipblasSsbmv_64)
WWR_FUNCTION(gpublasDsbmv_64, cublasDsbmv_v2_64, hipblasDsbmv_64)

WWR_FUNCTION(gpublasSspmv, cublasSspmv_v2, hipblasSspmv)
WWR_FUNCTION(gpublasDspmv, cublasDspmv_v2, hipblasDspmv)
WWR_FUNCTION(gpublasSspmv_64, cublasSspmv_v2_64, hipblasSspmv_64)
WWR_FUNCTION(gpublasDspmv_64, cublasDspmv_v2_64, hipblasDspmv_64)

WWR_FUNCTION(gpublasSspr, cublasSspr_v2, hipblasSspr)
WWR_FUNCTION(gpublasDspr, cublasDspr_v2, hipblasDspr)
WWR_FUNCTION(gpublasSspr_64, cublasSspr_v2_64, hipblasSspr_64)
WWR_FUNCTION(gpublasDspr_64, cublasDspr_v2_64, hipblasDspr_64)

WWR_FUNCTION(gpublasSspr2, cublasSspr2_v2, hipblasSspr2)
WWR_FUNCTION(gpublasDspr2, cublasDspr2_v2, hipblasDspr2)
WWR_FUNCTION(gpublasSspr2_64, cublasSspr2_v2_64, hipblasSspr2_64)
WWR_FUNCTION(gpublasDspr2_64, cublasDspr2_v2_64, hipblasDspr2_64)

WWR_FUNCTION(gpublasSsymv, cublasSsymv_v2, hipblasSsymv)
WWR_FUNCTION(gpublasDsymv, cublasDsymv_v2, hipblasDsymv)
WWR_FUNCTION(gpublasSsymv_64, cublasSsymv_v2_64, hipblasSsymv_64)
WWR_FUNCTION(gpublasDsymv_64, cublasDsymv_v2_64, hipblasDsymv_64)

WWR_FUNCTION(gpublasSsyr, cublasSsyr_v2, hipblasSsyr)
WWR_FUNCTION(gpublasDsyr, cublasDsyr_v2, hipblasDsyr)
WWR_FUNCTION(gpublasSsyr_64, cublasSsyr_v2_64, hipblasSsyr_64)
WWR_FUNCTION(gpublasDsyr_64, cublasDsyr_v2_64, hipblasDsyr_64)

WWR_FUNCTION(gpublasSsyr2, cublasSsyr2_v2, hipblasSsyr2)
WWR_FUNCTION(gpublasDsyr2, cublasDsyr2_v2, hipblasDsyr2)
WWR_FUNCTION(gpublasSsyr2_64, cublasSsyr2_v2_64, hipblasSsyr2_64)
WWR_FUNCTION(gpublasDsyr2_64, cublasDsyr2_v2_64, hipblasDsyr2_64)

WWR_FUNCTION(gpublasStbmv, cublasStbmv_v2, hipblasStbmv)
WWR_FUNCTION(gpublasDtbmv, cublasDtbmv_v2, hipblasDtbmv)
WWR_FUNCTION(gpublasCtbmv, cublasCtbmv_v2, hipblasCtbmv)
WWR_FUNCTION(gpublasZtbmv, cublasZtbmv_v2, hipblasZtbmv)
WWR_FUNCTION(gpublasStbmv_64, cublasStbmv_v2_64, hipblasStbmv_64)
WWR_FUNCTION(gpublasDtbmv_64, cublasDtbmv_v2_64, hipblasDtbmv_64)
WWR_FUNCTION(gpublasCtbmv_64, cublasCtbmv_v2_64, hipblasCtbmv_64)
WWR_FUNCTION(gpublasZtbmv_64, cublasZtbmv_v2_64, hipblasZtbmv_64)

WWR_FUNCTION(gpublasStbsv, cublasStbsv_v2, hipblasStbsv)
WWR_FUNCTION(gpublasDtbsv, cublasDtbsv_v2, hipblasDtbsv)
WWR_FUNCTION(gpublasCtbsv, cublasCtbsv_v2, hipblasCtbsv)
WWR_FUNCTION(gpublasZtbsv, cublasZtbsv_v2, hipblasZtbsv)
WWR_FUNCTION(gpublasStbsv_64, cublasStbsv_v2_64, hipblasStbsv_64)
WWR_FUNCTION(gpublasDtbsv_64, cublasDtbsv_v2_64, hipblasDtbsv_64)
WWR_FUNCTION(gpublasCtbsv_64, cublasCtbsv_v2_64, hipblasCtbsv_64)
WWR_FUNCTION(gpublasZtbsv_64, cublasZtbsv_v2_64, hipblasZtbsv_64)

WWR_FUNCTION(gpublasStpmv, cublasStpmv_v2, hipblasStpmv)
WWR_FUNCTION(gpublasDtpmv, cublasDtpmv_v2, hipblasDtpmv)
WWR_FUNCTION(gpublasCtpmv, cublasCtpmv_v2, hipblasCtpmv)
WWR_FUNCTION(gpublasZtpmv, cublasZtpmv_v2, hipblasZtpmv)
WWR_FUNCTION(gpublasStpmv_64, cublasStpmv_v2_64, hipblasStpmv_64)
WWR_FUNCTION(gpublasDtpmv_64, cublasDtpmv_v2_64, hipblasDtpmv_64)
WWR_FUNCTION(gpublasCtpmv_64, cublasCtpmv_v2_64, hipblasCtpmv_64)
WWR_FUNCTION(gpublasZtpmv_64, cublasZtpmv_v2_64, hipblasZtpmv_64)

WWR_FUNCTION(gpublasStpsv, cublasStpsv_v2, hipblasStpsv)
WWR_FUNCTION(gpublasDtpsv, cublasDtpsv_v2, hipblasDtpsv)
WWR_FUNCTION(gpublasCtpsv, cublasCtpsv_v2, hipblasCtpsv)
WWR_FUNCTION(gpublasZtpsv, cublasZtpsv_v2, hipblasZtpsv)
WWR_FUNCTION(gpublasStpsv_64, cublasStpsv_v2_64, hipblasStpsv_64)
WWR_FUNCTION(gpublasDtpsv_64, cublasDtpsv_v2_64, hipblasDtpsv_64)
WWR_FUNCTION(gpublasCtpsv_64, cublasCtpsv_v2_64, hipblasCtpsv_64)
WWR_FUNCTION(gpublasZtpsv_64, cublasZtpsv_v2_64, hipblasZtpsv_64)

WWR_FUNCTION(gpublasStrmv, cublasStrmv_v2, hipblasStrmv)
WWR_FUNCTION(gpublasDtrmv, cublasDtrmv_v2, hipblasDtrmv)
WWR_FUNCTION(gpublasCtrmv, cublasCtrmv_v2, hipblasCtrmv)
WWR_FUNCTION(gpublasZtrmv, cublasZtrmv_v2, hipblasZtrmv)
WWR_FUNCTION(gpublasStrmv_64, cublasStrmv_v2_64, hipblasStrmv_64)
WWR_FUNCTION(gpublasDtrmv_64, cublasDtrmv_v2_64, hipblasDtrmv_64)
WWR_FUNCTION(gpublasCtrmv_64, cublasCtrmv_v2_64, hipblasCtrmv_64)
WWR_FUNCTION(gpublasZtrmv_64, cublasZtrmv_v2_64, hipblasZtrmv_64)

WWR_FUNCTION(gpublasStrsv, cublasStrsv_v2, hipblasStrsv)
WWR_FUNCTION(gpublasDtrsv, cublasDtrsv_v2, hipblasDtrsv)
WWR_FUNCTION(gpublasCtrsv, cublasCtrsv_v2, hipblasCtrsv)
WWR_FUNCTION(gpublasZtrsv, cublasZtrsv_v2, hipblasZtrsv)
WWR_FUNCTION(gpublasStrsv_64, cublasStrsv_v2_64, hipblasStrsv_64)
WWR_FUNCTION(gpublasDtrsv_64, cublasDtrsv_v2_64, hipblasDtrsv_64)
WWR_FUNCTION(gpublasCtrsv_64, cublasCtrsv_v2_64, hipblasCtrsv_64)
WWR_FUNCTION(gpublasZtrsv_64, cublasZtrsv_v2_64, hipblasZtrsv_64)

// ────────────────────────────────────────────────────────────────────────
// Level 3 (matrix-matrix)
// ────────────────────────────────────────────────────────────────────────

WWR_FUNCTION(gpublasSgemm, cublasSgemm_v2, hipblasSgemm)
WWR_FUNCTION(gpublasDgemm, cublasDgemm_v2, hipblasDgemm)
WWR_FUNCTION(gpublasCgemm, cublasCgemm_v2, hipblasCgemm)
WWR_FUNCTION(gpublasZgemm, cublasZgemm_v2, hipblasZgemm)
WWR_FUNCTION(gpublasSgemm_64, cublasSgemm_v2_64, hipblasSgemm_64)
WWR_FUNCTION(gpublasDgemm_64, cublasDgemm_v2_64, hipblasDgemm_64)
WWR_FUNCTION(gpublasCgemm_64, cublasCgemm_v2_64, hipblasCgemm_64)
WWR_FUNCTION(gpublasZgemm_64, cublasZgemm_v2_64, hipblasZgemm_64)

WWR_FUNCTION(gpublasSgemmBatched, cublasSgemmBatched, hipblasSgemmBatched)
WWR_FUNCTION(gpublasDgemmBatched, cublasDgemmBatched, hipblasDgemmBatched)
WWR_FUNCTION(gpublasCgemmBatched, cublasCgemmBatched, hipblasCgemmBatched)
WWR_FUNCTION(gpublasZgemmBatched, cublasZgemmBatched, hipblasZgemmBatched)
WWR_FUNCTION(gpublasSgemmBatched_64, cublasSgemmBatched_64, hipblasSgemmBatched_64)
WWR_FUNCTION(gpublasDgemmBatched_64, cublasDgemmBatched_64, hipblasDgemmBatched_64)
WWR_FUNCTION(gpublasCgemmBatched_64, cublasCgemmBatched_64, hipblasCgemmBatched_64)
WWR_FUNCTION(gpublasZgemmBatched_64, cublasZgemmBatched_64, hipblasZgemmBatched_64)

WWR_FUNCTION(gpublasSgemmStridedBatched, cublasSgemmStridedBatched, hipblasSgemmStridedBatched)
WWR_FUNCTION(gpublasDgemmStridedBatched, cublasDgemmStridedBatched, hipblasDgemmStridedBatched)
WWR_FUNCTION(gpublasCgemmStridedBatched, cublasCgemmStridedBatched, hipblasCgemmStridedBatched)
WWR_FUNCTION(gpublasZgemmStridedBatched, cublasZgemmStridedBatched, hipblasZgemmStridedBatched)
WWR_FUNCTION(gpublasSgemmStridedBatched_64, cublasSgemmStridedBatched_64,
                hipblasSgemmStridedBatched_64)
WWR_FUNCTION(gpublasDgemmStridedBatched_64, cublasDgemmStridedBatched_64,
                hipblasDgemmStridedBatched_64)
WWR_FUNCTION(gpublasCgemmStridedBatched_64, cublasCgemmStridedBatched_64,
                hipblasCgemmStridedBatched_64)
WWR_FUNCTION(gpublasZgemmStridedBatched_64, cublasZgemmStridedBatched_64,
                hipblasZgemmStridedBatched_64)

WWR_FUNCTION(gpublasChemm, cublasChemm_v2, hipblasChemm)
WWR_FUNCTION(gpublasZhemm, cublasZhemm_v2, hipblasZhemm)
WWR_FUNCTION(gpublasChemm_64, cublasChemm_v2_64, hipblasChemm_64)
WWR_FUNCTION(gpublasZhemm_64, cublasZhemm_v2_64, hipblasZhemm_64)

WWR_FUNCTION(gpublasCher2k, cublasCher2k_v2, hipblasCher2k)
WWR_FUNCTION(gpublasZher2k, cublasZher2k_v2, hipblasZher2k)
WWR_FUNCTION(gpublasCher2k_64, cublasCher2k_v2_64, hipblasCher2k_64)
WWR_FUNCTION(gpublasZher2k_64, cublasZher2k_v2_64, hipblasZher2k_64)

WWR_FUNCTION(gpublasCherk, cublasCherk_v2, hipblasCherk)
WWR_FUNCTION(gpublasZherk, cublasZherk_v2, hipblasZherk)
WWR_FUNCTION(gpublasCherk_64, cublasCherk_v2_64, hipblasCherk_64)
WWR_FUNCTION(gpublasZherk_64, cublasZherk_v2_64, hipblasZherk_64)

WWR_FUNCTION(gpublasCherkx, cublasCherkx, hipblasCherkx)
WWR_FUNCTION(gpublasZherkx, cublasZherkx, hipblasZherkx)
WWR_FUNCTION(gpublasCherkx_64, cublasCherkx_64, hipblasCherkx_64)
WWR_FUNCTION(gpublasZherkx_64, cublasZherkx_64, hipblasZherkx_64)

WWR_FUNCTION(gpublasSsymm, cublasSsymm_v2, hipblasSsymm)
WWR_FUNCTION(gpublasDsymm, cublasDsymm_v2, hipblasDsymm)
WWR_FUNCTION(gpublasCsymm, cublasCsymm_v2, hipblasCsymm)
WWR_FUNCTION(gpublasZsymm, cublasZsymm_v2, hipblasZsymm)
WWR_FUNCTION(gpublasSsymm_64, cublasSsymm_v2_64, hipblasSsymm_64)
WWR_FUNCTION(gpublasDsymm_64, cublasDsymm_v2_64, hipblasDsymm_64)
WWR_FUNCTION(gpublasCsymm_64, cublasCsymm_v2_64, hipblasCsymm_64)
WWR_FUNCTION(gpublasZsymm_64, cublasZsymm_v2_64, hipblasZsymm_64)

WWR_FUNCTION(gpublasSsyr2k, cublasSsyr2k_v2, hipblasSsyr2k)
WWR_FUNCTION(gpublasDsyr2k, cublasDsyr2k_v2, hipblasDsyr2k)
WWR_FUNCTION(gpublasCsyr2k, cublasCsyr2k_v2, hipblasCsyr2k)
WWR_FUNCTION(gpublasZsyr2k, cublasZsyr2k_v2, hipblasZsyr2k)
WWR_FUNCTION(gpublasSsyr2k_64, cublasSsyr2k_v2_64, hipblasSsyr2k_64)
WWR_FUNCTION(gpublasDsyr2k_64, cublasDsyr2k_v2_64, hipblasDsyr2k_64)
WWR_FUNCTION(gpublasCsyr2k_64, cublasCsyr2k_v2_64, hipblasCsyr2k_64)
WWR_FUNCTION(gpublasZsyr2k_64, cublasZsyr2k_v2_64, hipblasZsyr2k_64)

WWR_FUNCTION(gpublasSsyrk, cublasSsyrk_v2, hipblasSsyrk)
WWR_FUNCTION(gpublasDsyrk, cublasDsyrk_v2, hipblasDsyrk)
WWR_FUNCTION(gpublasCsyrk, cublasCsyrk_v2, hipblasCsyrk)
WWR_FUNCTION(gpublasZsyrk, cublasZsyrk_v2, hipblasZsyrk)
WWR_FUNCTION(gpublasSsyrk_64, cublasSsyrk_v2_64, hipblasSsyrk_64)
WWR_FUNCTION(gpublasDsyrk_64, cublasDsyrk_v2_64, hipblasDsyrk_64)
WWR_FUNCTION(gpublasCsyrk_64, cublasCsyrk_v2_64, hipblasCsyrk_64)
WWR_FUNCTION(gpublasZsyrk_64, cublasZsyrk_v2_64, hipblasZsyrk_64)

WWR_FUNCTION(gpublasSsyrkx, cublasSsyrkx, hipblasSsyrkx)
WWR_FUNCTION(gpublasDsyrkx, cublasDsyrkx, hipblasDsyrkx)
WWR_FUNCTION(gpublasCsyrkx, cublasCsyrkx, hipblasCsyrkx)
WWR_FUNCTION(gpublasZsyrkx, cublasZsyrkx, hipblasZsyrkx)
WWR_FUNCTION(gpublasSsyrkx_64, cublasSsyrkx_64, hipblasSsyrkx_64)
WWR_FUNCTION(gpublasDsyrkx_64, cublasDsyrkx_64, hipblasDsyrkx_64)
WWR_FUNCTION(gpublasCsyrkx_64, cublasCsyrkx_64, hipblasCsyrkx_64)
WWR_FUNCTION(gpublasZsyrkx_64, cublasZsyrkx_64, hipblasZsyrkx_64)

WWR_FUNCTION(gpublasStrmm, cublasStrmm_v2, hipblasStrmm)
WWR_FUNCTION(gpublasDtrmm, cublasDtrmm_v2, hipblasDtrmm)
WWR_FUNCTION(gpublasCtrmm, cublasCtrmm_v2, hipblasCtrmm)
WWR_FUNCTION(gpublasZtrmm, cublasZtrmm_v2, hipblasZtrmm)
WWR_FUNCTION(gpublasStrmm_64, cublasStrmm_v2_64, hipblasStrmm_64)
WWR_FUNCTION(gpublasDtrmm_64, cublasDtrmm_v2_64, hipblasDtrmm_64)
WWR_FUNCTION(gpublasCtrmm_64, cublasCtrmm_v2_64, hipblasCtrmm_64)
WWR_FUNCTION(gpublasZtrmm_64, cublasZtrmm_v2_64, hipblasZtrmm_64)

WWR_FUNCTION(gpublasStrsm, cublasStrsm_v2, hipblasStrsm)
WWR_FUNCTION(gpublasDtrsm, cublasDtrsm_v2, hipblasDtrsm)
WWR_FUNCTION(gpublasCtrsm, cublasCtrsm_v2, hipblasCtrsm)
WWR_FUNCTION(gpublasZtrsm, cublasZtrsm_v2, hipblasZtrsm)
WWR_FUNCTION(gpublasStrsm_64, cublasStrsm_v2_64, hipblasStrsm_64)
WWR_FUNCTION(gpublasDtrsm_64, cublasDtrsm_v2_64, hipblasDtrsm_64)
WWR_FUNCTION(gpublasCtrsm_64, cublasCtrsm_v2_64, hipblasCtrsm_64)
WWR_FUNCTION(gpublasZtrsm_64, cublasZtrsm_v2_64, hipblasZtrsm_64)

WWR_FUNCTION(gpublasStrsmBatched, cublasStrsmBatched, hipblasStrsmBatched)
WWR_FUNCTION(gpublasDtrsmBatched, cublasDtrsmBatched, hipblasDtrsmBatched)
WWR_FUNCTION(gpublasCtrsmBatched, cublasCtrsmBatched, hipblasCtrsmBatched)
WWR_FUNCTION(gpublasZtrsmBatched, cublasZtrsmBatched, hipblasZtrsmBatched)
WWR_FUNCTION(gpublasStrsmBatched_64, cublasStrsmBatched_64, hipblasStrsmBatched_64)
WWR_FUNCTION(gpublasDtrsmBatched_64, cublasDtrsmBatched_64, hipblasDtrsmBatched_64)
WWR_FUNCTION(gpublasCtrsmBatched_64, cublasCtrsmBatched_64, hipblasCtrsmBatched_64)
WWR_FUNCTION(gpublasZtrsmBatched_64, cublasZtrsmBatched_64, hipblasZtrsmBatched_64)

// ────────────────────────────────────────────────────────────────────────
// BLAS-like extensions
// ────────────────────────────────────────────────────────────────────────

WWR_FUNCTION(gpublasSdgmm, cublasSdgmm, hipblasSdgmm)
WWR_FUNCTION(gpublasDdgmm, cublasDdgmm, hipblasDdgmm)
WWR_FUNCTION(gpublasCdgmm, cublasCdgmm, hipblasCdgmm)
WWR_FUNCTION(gpublasZdgmm, cublasZdgmm, hipblasZdgmm)
WWR_FUNCTION(gpublasSdgmm_64, cublasSdgmm_64, hipblasSdgmm_64)
WWR_FUNCTION(gpublasDdgmm_64, cublasDdgmm_64, hipblasDdgmm_64)
WWR_FUNCTION(gpublasCdgmm_64, cublasCdgmm_64, hipblasCdgmm_64)
WWR_FUNCTION(gpublasZdgmm_64, cublasZdgmm_64, hipblasZdgmm_64)

WWR_FUNCTION(gpublasSgeam, cublasSgeam, hipblasSgeam)
WWR_FUNCTION(gpublasDgeam, cublasDgeam, hipblasDgeam)
WWR_FUNCTION(gpublasCgeam, cublasCgeam, hipblasCgeam)
WWR_FUNCTION(gpublasZgeam, cublasZgeam, hipblasZgeam)
WWR_FUNCTION(gpublasSgeam_64, cublasSgeam_64, hipblasSgeam_64)
WWR_FUNCTION(gpublasDgeam_64, cublasDgeam_64, hipblasDgeam_64)
WWR_FUNCTION(gpublasCgeam_64, cublasCgeam_64, hipblasCgeam_64)
WWR_FUNCTION(gpublasZgeam_64, cublasZgeam_64, hipblasZgeam_64)

WWR_FUNCTION(gpublasSgelsBatched, cublasSgelsBatched, hipblasSgelsBatched)
WWR_FUNCTION(gpublasDgelsBatched, cublasDgelsBatched, hipblasDgelsBatched)
WWR_FUNCTION(gpublasCgelsBatched, cublasCgelsBatched, hipblasCgelsBatched)
WWR_FUNCTION(gpublasZgelsBatched, cublasZgelsBatched, hipblasZgelsBatched)

WWR_FUNCTION(gpublasSgeqrfBatched, cublasSgeqrfBatched, hipblasSgeqrfBatched)
WWR_FUNCTION(gpublasDgeqrfBatched, cublasDgeqrfBatched, hipblasDgeqrfBatched)
WWR_FUNCTION(gpublasCgeqrfBatched, cublasCgeqrfBatched, hipblasCgeqrfBatched)
WWR_FUNCTION(gpublasZgeqrfBatched, cublasZgeqrfBatched, hipblasZgeqrfBatched)

WWR_FUNCTION(gpublasSgetrfBatched, cublasSgetrfBatched, hipblasSgetrfBatched)
WWR_FUNCTION(gpublasDgetrfBatched, cublasDgetrfBatched, hipblasDgetrfBatched)
WWR_FUNCTION(gpublasCgetrfBatched, cublasCgetrfBatched, hipblasCgetrfBatched)
WWR_FUNCTION(gpublasZgetrfBatched, cublasZgetrfBatched, hipblasZgetrfBatched)

// ────────────────────────────────────────────────────────────────────────
// getrsBatched / getriBatched: const-correct on both backends
// ────────────────────────────────────────────────────────────────────────

#if defined(WWR_GPU_BACKEND_CUDA)

WWR_FUNCTION(gpublasSgetrsBatched, cublasSgetrsBatched, hipblasSgetrsBatched)
WWR_FUNCTION(gpublasDgetrsBatched, cublasDgetrsBatched, hipblasDgetrsBatched)
WWR_FUNCTION(gpublasCgetrsBatched, cublasCgetrsBatched, hipblasCgetrsBatched)
WWR_FUNCTION(gpublasZgetrsBatched, cublasZgetrsBatched, hipblasZgetrsBatched)

WWR_FUNCTION(gpublasSgetriBatched, cublasSgetriBatched, hipblasSgetriBatched)
WWR_FUNCTION(gpublasDgetriBatched, cublasDgetriBatched, hipblasDgetriBatched)
WWR_FUNCTION(gpublasCgetriBatched, cublasCgetriBatched, hipblasCgetriBatched)
WWR_FUNCTION(gpublasZgetriBatched, cublasZgetriBatched, hipblasZgetriBatched)

#else

// hipBLAS declares Aarray as T* const[] (and getriBatched's P as int*) where
// cuBLAS declares const T* const[] (and const int*). hipBLAS only reads them;
// these keep cuBLAS's signature so src/wrappers has one const-correct API.

inline gpublasStatus_t gpublasSgetrsBatched(gpublasHandle_t handle, gpublasOperation_t trans, int n,
                                            int nrhs, const float *const Aarray[], int lda,
                                            const int *devIpiv, float *const Barray[], int ldb,
                                            int *info, int batchSize) {
  return hip::hipblasSgetrsBatched(handle, trans, n, nrhs, const_cast<float *const *>(Aarray), lda,
                                   devIpiv, Barray, ldb, info, batchSize);
}
inline gpublasStatus_t gpublasDgetrsBatched(gpublasHandle_t handle, gpublasOperation_t trans, int n,
                                            int nrhs, const double *const Aarray[], int lda,
                                            const int *devIpiv, double *const Barray[], int ldb,
                                            int *info, int batchSize) {
  return hip::hipblasDgetrsBatched(handle, trans, n, nrhs, const_cast<double *const *>(Aarray), lda,
                                   devIpiv, Barray, ldb, info, batchSize);
}
inline gpublasStatus_t gpublasCgetrsBatched(gpublasHandle_t handle, gpublasOperation_t trans, int n,
                                            int nrhs, const gpuFloatComplex *const Aarray[],
                                            int lda, const int *devIpiv,
                                            gpuFloatComplex *const Barray[], int ldb, int *info,
                                            int batchSize) {
  return hip::hipblasCgetrsBatched(handle, trans, n, nrhs,
                                   const_cast<gpuFloatComplex *const *>(Aarray), lda, devIpiv,
                                   Barray, ldb, info, batchSize);
}
inline gpublasStatus_t gpublasZgetrsBatched(gpublasHandle_t handle, gpublasOperation_t trans, int n,
                                            int nrhs, const gpuDoubleComplex *const Aarray[],
                                            int lda, const int *devIpiv,
                                            gpuDoubleComplex *const Barray[], int ldb, int *info,
                                            int batchSize) {
  return hip::hipblasZgetrsBatched(handle, trans, n, nrhs,
                                   const_cast<gpuDoubleComplex *const *>(Aarray), lda, devIpiv,
                                   Barray, ldb, info, batchSize);
}

inline gpublasStatus_t gpublasSgetriBatched(gpublasHandle_t handle, int n, const float *const A[],
                                            int lda, const int *P, float *const C[], int ldc,
                                            int *info, int batchSize) {
  return hip::hipblasSgetriBatched(handle, n, const_cast<float *const *>(A), lda,
                                   const_cast<int *>(P), C, ldc, info, batchSize);
}
inline gpublasStatus_t gpublasDgetriBatched(gpublasHandle_t handle, int n, const double *const A[],
                                            int lda, const int *P, double *const C[], int ldc,
                                            int *info, int batchSize) {
  return hip::hipblasDgetriBatched(handle, n, const_cast<double *const *>(A), lda,
                                   const_cast<int *>(P), C, ldc, info, batchSize);
}
inline gpublasStatus_t gpublasCgetriBatched(gpublasHandle_t handle, int n,
                                            const gpuFloatComplex *const A[], int lda, const int *P,
                                            gpuFloatComplex *const C[], int ldc, int *info,
                                            int batchSize) {
  return hip::hipblasCgetriBatched(handle, n, const_cast<gpuFloatComplex *const *>(A), lda,
                                   const_cast<int *>(P), C, ldc, info, batchSize);
}
inline gpublasStatus_t gpublasZgetriBatched(gpublasHandle_t handle, int n,
                                            const gpuDoubleComplex *const A[], int lda,
                                            const int *P, gpuDoubleComplex *const C[], int ldc,
                                            int *info, int batchSize) {
  return hip::hipblasZgetriBatched(handle, n, const_cast<gpuDoubleComplex *const *>(A), lda,
                                   const_cast<int *>(P), C, ldc, info, batchSize);
}

#endif

} // namespace wwr
