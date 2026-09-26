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
 * through gpumod.cuda.cublas_v2.
 *
 * Two backend differences are resolved here, not above: hipBLAS's one
 * status-to-string function backs both gpublasGetStatusName and
 * gpublasGetStatusString, and getrsBatched / getriBatched keep cuBLAS's
 * const-correct signature, forwarding with a const_cast on HIP.
 * See docs/architecture.md, section 5.
 *
 * Usage:
 *   import gpumod.blas;
 *
 *   gpublasHandle_t handle;
 *   gpublasCreate(&handle);
 */

module;

#include "gpu_backend.h"

export module gpumod.blas;

import gpumod.complex;
#if defined(GPUMOD_GPU_BACKEND_CUDA)
import gpumod.cuda.cublas_v2;
#else
import gpumod.hip.hipblas;
#endif

export namespace gpumod {

// ========================================================================
// Types
// ========================================================================

GPUMOD_TYPE(gpublasHandle_t, cublasHandle_t, hipblasHandle_t)
GPUMOD_TYPE(gpublasStatus_t, cublasStatus_t, hipblasStatus_t)
GPUMOD_TYPE(gpublasOperation_t, cublasOperation_t, hipblasOperation_t)
GPUMOD_TYPE(gpublasFillMode_t, cublasFillMode_t, hipblasFillMode_t)
GPUMOD_TYPE(gpublasDiagType_t, cublasDiagType_t, hipblasDiagType_t)
GPUMOD_TYPE(gpublasSideMode_t, cublasSideMode_t, hipblasSideMode_t)
GPUMOD_TYPE(gpublasPointerMode_t, cublasPointerMode_t, hipblasPointerMode_t)

// ========================================================================
// Constants
// ========================================================================

GPUMOD_VALUE(GPUBLAS_STATUS_SUCCESS, CUBLAS_STATUS_SUCCESS, HIPBLAS_STATUS_SUCCESS)
GPUMOD_VALUE(GPUBLAS_STATUS_NOT_INITIALIZED, CUBLAS_STATUS_NOT_INITIALIZED,
             HIPBLAS_STATUS_NOT_INITIALIZED)

GPUMOD_VALUE(GPUBLAS_OP_N, CUBLAS_OP_N, HIPBLAS_OP_N)
GPUMOD_VALUE(GPUBLAS_OP_T, CUBLAS_OP_T, HIPBLAS_OP_T)
GPUMOD_VALUE(GPUBLAS_OP_C, CUBLAS_OP_C, HIPBLAS_OP_C)

GPUMOD_VALUE(GPUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_LOWER, HIPBLAS_FILL_MODE_LOWER)
GPUMOD_VALUE(GPUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_UPPER, HIPBLAS_FILL_MODE_UPPER)

GPUMOD_VALUE(GPUBLAS_DIAG_NON_UNIT, CUBLAS_DIAG_NON_UNIT, HIPBLAS_DIAG_NON_UNIT)
GPUMOD_VALUE(GPUBLAS_DIAG_UNIT, CUBLAS_DIAG_UNIT, HIPBLAS_DIAG_UNIT)

GPUMOD_VALUE(GPUBLAS_SIDE_LEFT, CUBLAS_SIDE_LEFT, HIPBLAS_SIDE_LEFT)
GPUMOD_VALUE(GPUBLAS_SIDE_RIGHT, CUBLAS_SIDE_RIGHT, HIPBLAS_SIDE_RIGHT)

GPUMOD_VALUE(GPUBLAS_POINTER_MODE_HOST, CUBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_HOST)
GPUMOD_VALUE(GPUBLAS_POINTER_MODE_DEVICE, CUBLAS_POINTER_MODE_DEVICE, HIPBLAS_POINTER_MODE_DEVICE)

// ========================================================================
// Handle, stream, pointer mode, status strings
// ========================================================================

GPUMOD_FUNCTION(gpublasCreate, cublasCreate_v2, hipblasCreate)
GPUMOD_FUNCTION(gpublasDestroy, cublasDestroy_v2, hipblasDestroy)
GPUMOD_FUNCTION(gpublasSetStream, cublasSetStream_v2, hipblasSetStream)
GPUMOD_FUNCTION(gpublasGetStream, cublasGetStream_v2, hipblasGetStream)
GPUMOD_FUNCTION(gpublasSetPointerMode, cublasSetPointerMode_v2, hipblasSetPointerMode)

// hipBLAS has a single status-to-string function; both names map to it.
GPUMOD_FUNCTION(gpublasGetStatusName, cublasGetStatusName, hipblasStatusToString)
GPUMOD_FUNCTION(gpublasGetStatusString, cublasGetStatusString, hipblasStatusToString)

// ────────────────────────────────────────────────────────────────────────
// Level 1 (vector-vector)
// ────────────────────────────────────────────────────────────────────────

GPUMOD_FUNCTION(gpublasSasum, cublasSasum_v2, hipblasSasum)
GPUMOD_FUNCTION(gpublasDasum, cublasDasum_v2, hipblasDasum)
GPUMOD_FUNCTION(gpublasSasum_64, cublasSasum_v2_64, hipblasSasum_64)
GPUMOD_FUNCTION(gpublasDasum_64, cublasDasum_v2_64, hipblasDasum_64)

GPUMOD_FUNCTION(gpublasSaxpy, cublasSaxpy_v2, hipblasSaxpy)
GPUMOD_FUNCTION(gpublasDaxpy, cublasDaxpy_v2, hipblasDaxpy)
GPUMOD_FUNCTION(gpublasCaxpy, cublasCaxpy_v2, hipblasCaxpy)
GPUMOD_FUNCTION(gpublasZaxpy, cublasZaxpy_v2, hipblasZaxpy)
GPUMOD_FUNCTION(gpublasSaxpy_64, cublasSaxpy_v2_64, hipblasSaxpy_64)
GPUMOD_FUNCTION(gpublasDaxpy_64, cublasDaxpy_v2_64, hipblasDaxpy_64)
GPUMOD_FUNCTION(gpublasCaxpy_64, cublasCaxpy_v2_64, hipblasCaxpy_64)
GPUMOD_FUNCTION(gpublasZaxpy_64, cublasZaxpy_v2_64, hipblasZaxpy_64)

GPUMOD_FUNCTION(gpublasScasum, cublasScasum_v2, hipblasScasum)
GPUMOD_FUNCTION(gpublasScasum_64, cublasScasum_v2_64, hipblasScasum_64)

GPUMOD_FUNCTION(gpublasScnrm2, cublasScnrm2_v2, hipblasScnrm2)
GPUMOD_FUNCTION(gpublasScnrm2_64, cublasScnrm2_v2_64, hipblasScnrm2_64)

GPUMOD_FUNCTION(gpublasScopy, cublasScopy_v2, hipblasScopy)
GPUMOD_FUNCTION(gpublasDcopy, cublasDcopy_v2, hipblasDcopy)
GPUMOD_FUNCTION(gpublasCcopy, cublasCcopy_v2, hipblasCcopy)
GPUMOD_FUNCTION(gpublasZcopy, cublasZcopy_v2, hipblasZcopy)
GPUMOD_FUNCTION(gpublasScopy_64, cublasScopy_v2_64, hipblasScopy_64)
GPUMOD_FUNCTION(gpublasDcopy_64, cublasDcopy_v2_64, hipblasDcopy_64)
GPUMOD_FUNCTION(gpublasCcopy_64, cublasCcopy_v2_64, hipblasCcopy_64)
GPUMOD_FUNCTION(gpublasZcopy_64, cublasZcopy_v2_64, hipblasZcopy_64)

GPUMOD_FUNCTION(gpublasSdot, cublasSdot_v2, hipblasSdot)
GPUMOD_FUNCTION(gpublasDdot, cublasDdot_v2, hipblasDdot)
GPUMOD_FUNCTION(gpublasSdot_64, cublasSdot_v2_64, hipblasSdot_64)
GPUMOD_FUNCTION(gpublasDdot_64, cublasDdot_v2_64, hipblasDdot_64)

GPUMOD_FUNCTION(gpublasCdotc, cublasCdotc_v2, hipblasCdotc)
GPUMOD_FUNCTION(gpublasZdotc, cublasZdotc_v2, hipblasZdotc)
GPUMOD_FUNCTION(gpublasCdotc_64, cublasCdotc_v2_64, hipblasCdotc_64)
GPUMOD_FUNCTION(gpublasZdotc_64, cublasZdotc_v2_64, hipblasZdotc_64)

GPUMOD_FUNCTION(gpublasCdotu, cublasCdotu_v2, hipblasCdotu)
GPUMOD_FUNCTION(gpublasZdotu, cublasZdotu_v2, hipblasZdotu)
GPUMOD_FUNCTION(gpublasCdotu_64, cublasCdotu_v2_64, hipblasCdotu_64)
GPUMOD_FUNCTION(gpublasZdotu_64, cublasZdotu_v2_64, hipblasZdotu_64)

GPUMOD_FUNCTION(gpublasZdrot, cublasZdrot_v2, hipblasZdrot)
GPUMOD_FUNCTION(gpublasZdrot_64, cublasZdrot_v2_64, hipblasZdrot_64)

GPUMOD_FUNCTION(gpublasZdscal, cublasZdscal_v2, hipblasZdscal)
GPUMOD_FUNCTION(gpublasZdscal_64, cublasZdscal_v2_64, hipblasZdscal_64)

GPUMOD_FUNCTION(gpublasIsamax, cublasIsamax_v2, hipblasIsamax)
GPUMOD_FUNCTION(gpublasIdamax, cublasIdamax_v2, hipblasIdamax)
GPUMOD_FUNCTION(gpublasIcamax, cublasIcamax_v2, hipblasIcamax)
GPUMOD_FUNCTION(gpublasIzamax, cublasIzamax_v2, hipblasIzamax)
GPUMOD_FUNCTION(gpublasIsamax_64, cublasIsamax_v2_64, hipblasIsamax_64)
GPUMOD_FUNCTION(gpublasIdamax_64, cublasIdamax_v2_64, hipblasIdamax_64)
GPUMOD_FUNCTION(gpublasIcamax_64, cublasIcamax_v2_64, hipblasIcamax_64)
GPUMOD_FUNCTION(gpublasIzamax_64, cublasIzamax_v2_64, hipblasIzamax_64)

GPUMOD_FUNCTION(gpublasIsamin, cublasIsamin_v2, hipblasIsamin)
GPUMOD_FUNCTION(gpublasIdamin, cublasIdamin_v2, hipblasIdamin)
GPUMOD_FUNCTION(gpublasIcamin, cublasIcamin_v2, hipblasIcamin)
GPUMOD_FUNCTION(gpublasIzamin, cublasIzamin_v2, hipblasIzamin)
GPUMOD_FUNCTION(gpublasIsamin_64, cublasIsamin_v2_64, hipblasIsamin_64)
GPUMOD_FUNCTION(gpublasIdamin_64, cublasIdamin_v2_64, hipblasIdamin_64)
GPUMOD_FUNCTION(gpublasIcamin_64, cublasIcamin_v2_64, hipblasIcamin_64)
GPUMOD_FUNCTION(gpublasIzamin_64, cublasIzamin_v2_64, hipblasIzamin_64)

GPUMOD_FUNCTION(gpublasSnrm2, cublasSnrm2_v2, hipblasSnrm2)
GPUMOD_FUNCTION(gpublasDnrm2, cublasDnrm2_v2, hipblasDnrm2)
GPUMOD_FUNCTION(gpublasSnrm2_64, cublasSnrm2_v2_64, hipblasSnrm2_64)
GPUMOD_FUNCTION(gpublasDnrm2_64, cublasDnrm2_v2_64, hipblasDnrm2_64)

GPUMOD_FUNCTION(gpublasSrot, cublasSrot_v2, hipblasSrot)
GPUMOD_FUNCTION(gpublasDrot, cublasDrot_v2, hipblasDrot)
GPUMOD_FUNCTION(gpublasCrot, cublasCrot_v2, hipblasCrot)
GPUMOD_FUNCTION(gpublasZrot, cublasZrot_v2, hipblasZrot)
GPUMOD_FUNCTION(gpublasSrot_64, cublasSrot_v2_64, hipblasSrot_64)
GPUMOD_FUNCTION(gpublasDrot_64, cublasDrot_v2_64, hipblasDrot_64)
GPUMOD_FUNCTION(gpublasCrot_64, cublasCrot_v2_64, hipblasCrot_64)
GPUMOD_FUNCTION(gpublasZrot_64, cublasZrot_v2_64, hipblasZrot_64)

GPUMOD_FUNCTION(gpublasSrotg, cublasSrotg_v2, hipblasSrotg)
GPUMOD_FUNCTION(gpublasDrotg, cublasDrotg_v2, hipblasDrotg)
GPUMOD_FUNCTION(gpublasCrotg, cublasCrotg_v2, hipblasCrotg)
GPUMOD_FUNCTION(gpublasZrotg, cublasZrotg_v2, hipblasZrotg)

GPUMOD_FUNCTION(gpublasSrotm, cublasSrotm_v2, hipblasSrotm)
GPUMOD_FUNCTION(gpublasDrotm, cublasDrotm_v2, hipblasDrotm)
GPUMOD_FUNCTION(gpublasSrotm_64, cublasSrotm_v2_64, hipblasSrotm_64)
GPUMOD_FUNCTION(gpublasDrotm_64, cublasDrotm_v2_64, hipblasDrotm_64)

GPUMOD_FUNCTION(gpublasSrotmg, cublasSrotmg_v2, hipblasSrotmg)
GPUMOD_FUNCTION(gpublasDrotmg, cublasDrotmg_v2, hipblasDrotmg)

GPUMOD_FUNCTION(gpublasSscal, cublasSscal_v2, hipblasSscal)
GPUMOD_FUNCTION(gpublasDscal, cublasDscal_v2, hipblasDscal)
GPUMOD_FUNCTION(gpublasCscal, cublasCscal_v2, hipblasCscal)
GPUMOD_FUNCTION(gpublasZscal, cublasZscal_v2, hipblasZscal)
GPUMOD_FUNCTION(gpublasSscal_64, cublasSscal_v2_64, hipblasSscal_64)
GPUMOD_FUNCTION(gpublasDscal_64, cublasDscal_v2_64, hipblasDscal_64)
GPUMOD_FUNCTION(gpublasCscal_64, cublasCscal_v2_64, hipblasCscal_64)
GPUMOD_FUNCTION(gpublasZscal_64, cublasZscal_v2_64, hipblasZscal_64)

GPUMOD_FUNCTION(gpublasCsrot, cublasCsrot_v2, hipblasCsrot)
GPUMOD_FUNCTION(gpublasCsrot_64, cublasCsrot_v2_64, hipblasCsrot_64)

GPUMOD_FUNCTION(gpublasCsscal, cublasCsscal_v2, hipblasCsscal)
GPUMOD_FUNCTION(gpublasCsscal_64, cublasCsscal_v2_64, hipblasCsscal_64)

GPUMOD_FUNCTION(gpublasSswap, cublasSswap_v2, hipblasSswap)
GPUMOD_FUNCTION(gpublasDswap, cublasDswap_v2, hipblasDswap)
GPUMOD_FUNCTION(gpublasCswap, cublasCswap_v2, hipblasCswap)
GPUMOD_FUNCTION(gpublasZswap, cublasZswap_v2, hipblasZswap)
GPUMOD_FUNCTION(gpublasSswap_64, cublasSswap_v2_64, hipblasSswap_64)
GPUMOD_FUNCTION(gpublasDswap_64, cublasDswap_v2_64, hipblasDswap_64)
GPUMOD_FUNCTION(gpublasCswap_64, cublasCswap_v2_64, hipblasCswap_64)
GPUMOD_FUNCTION(gpublasZswap_64, cublasZswap_v2_64, hipblasZswap_64)

GPUMOD_FUNCTION(gpublasDzasum, cublasDzasum_v2, hipblasDzasum)
GPUMOD_FUNCTION(gpublasDzasum_64, cublasDzasum_v2_64, hipblasDzasum_64)

GPUMOD_FUNCTION(gpublasDznrm2, cublasDznrm2_v2, hipblasDznrm2)
GPUMOD_FUNCTION(gpublasDznrm2_64, cublasDznrm2_v2_64, hipblasDznrm2_64)

// ────────────────────────────────────────────────────────────────────────
// Level 2 (matrix-vector)
// ────────────────────────────────────────────────────────────────────────

GPUMOD_FUNCTION(gpublasSgbmv, cublasSgbmv_v2, hipblasSgbmv)
GPUMOD_FUNCTION(gpublasDgbmv, cublasDgbmv_v2, hipblasDgbmv)
GPUMOD_FUNCTION(gpublasCgbmv, cublasCgbmv_v2, hipblasCgbmv)
GPUMOD_FUNCTION(gpublasZgbmv, cublasZgbmv_v2, hipblasZgbmv)
GPUMOD_FUNCTION(gpublasSgbmv_64, cublasSgbmv_v2_64, hipblasSgbmv_64)
GPUMOD_FUNCTION(gpublasDgbmv_64, cublasDgbmv_v2_64, hipblasDgbmv_64)
GPUMOD_FUNCTION(gpublasCgbmv_64, cublasCgbmv_v2_64, hipblasCgbmv_64)
GPUMOD_FUNCTION(gpublasZgbmv_64, cublasZgbmv_v2_64, hipblasZgbmv_64)

GPUMOD_FUNCTION(gpublasSgemv, cublasSgemv_v2, hipblasSgemv)
GPUMOD_FUNCTION(gpublasDgemv, cublasDgemv_v2, hipblasDgemv)
GPUMOD_FUNCTION(gpublasCgemv, cublasCgemv_v2, hipblasCgemv)
GPUMOD_FUNCTION(gpublasZgemv, cublasZgemv_v2, hipblasZgemv)
GPUMOD_FUNCTION(gpublasSgemv_64, cublasSgemv_v2_64, hipblasSgemv_64)
GPUMOD_FUNCTION(gpublasDgemv_64, cublasDgemv_v2_64, hipblasDgemv_64)
GPUMOD_FUNCTION(gpublasCgemv_64, cublasCgemv_v2_64, hipblasCgemv_64)
GPUMOD_FUNCTION(gpublasZgemv_64, cublasZgemv_v2_64, hipblasZgemv_64)

GPUMOD_FUNCTION(gpublasSgemvBatched, cublasSgemvBatched, hipblasSgemvBatched)
GPUMOD_FUNCTION(gpublasDgemvBatched, cublasDgemvBatched, hipblasDgemvBatched)
GPUMOD_FUNCTION(gpublasCgemvBatched, cublasCgemvBatched, hipblasCgemvBatched)
GPUMOD_FUNCTION(gpublasZgemvBatched, cublasZgemvBatched, hipblasZgemvBatched)
GPUMOD_FUNCTION(gpublasSgemvBatched_64, cublasSgemvBatched_64, hipblasSgemvBatched_64)
GPUMOD_FUNCTION(gpublasDgemvBatched_64, cublasDgemvBatched_64, hipblasDgemvBatched_64)
GPUMOD_FUNCTION(gpublasCgemvBatched_64, cublasCgemvBatched_64, hipblasCgemvBatched_64)
GPUMOD_FUNCTION(gpublasZgemvBatched_64, cublasZgemvBatched_64, hipblasZgemvBatched_64)

GPUMOD_FUNCTION(gpublasSgemvStridedBatched, cublasSgemvStridedBatched, hipblasSgemvStridedBatched)
GPUMOD_FUNCTION(gpublasDgemvStridedBatched, cublasDgemvStridedBatched, hipblasDgemvStridedBatched)
GPUMOD_FUNCTION(gpublasCgemvStridedBatched, cublasCgemvStridedBatched, hipblasCgemvStridedBatched)
GPUMOD_FUNCTION(gpublasZgemvStridedBatched, cublasZgemvStridedBatched, hipblasZgemvStridedBatched)
GPUMOD_FUNCTION(gpublasSgemvStridedBatched_64, cublasSgemvStridedBatched_64,
                hipblasSgemvStridedBatched_64)
GPUMOD_FUNCTION(gpublasDgemvStridedBatched_64, cublasDgemvStridedBatched_64,
                hipblasDgemvStridedBatched_64)
GPUMOD_FUNCTION(gpublasCgemvStridedBatched_64, cublasCgemvStridedBatched_64,
                hipblasCgemvStridedBatched_64)
GPUMOD_FUNCTION(gpublasZgemvStridedBatched_64, cublasZgemvStridedBatched_64,
                hipblasZgemvStridedBatched_64)

GPUMOD_FUNCTION(gpublasSger, cublasSger_v2, hipblasSger)
GPUMOD_FUNCTION(gpublasDger, cublasDger_v2, hipblasDger)
GPUMOD_FUNCTION(gpublasSger_64, cublasSger_v2_64, hipblasSger_64)
GPUMOD_FUNCTION(gpublasDger_64, cublasDger_v2_64, hipblasDger_64)

GPUMOD_FUNCTION(gpublasCgerc, cublasCgerc_v2, hipblasCgerc)
GPUMOD_FUNCTION(gpublasZgerc, cublasZgerc_v2, hipblasZgerc)
GPUMOD_FUNCTION(gpublasCgerc_64, cublasCgerc_v2_64, hipblasCgerc_64)
GPUMOD_FUNCTION(gpublasZgerc_64, cublasZgerc_v2_64, hipblasZgerc_64)

GPUMOD_FUNCTION(gpublasCgeru, cublasCgeru_v2, hipblasCgeru)
GPUMOD_FUNCTION(gpublasZgeru, cublasZgeru_v2, hipblasZgeru)
GPUMOD_FUNCTION(gpublasCgeru_64, cublasCgeru_v2_64, hipblasCgeru_64)
GPUMOD_FUNCTION(gpublasZgeru_64, cublasZgeru_v2_64, hipblasZgeru_64)

GPUMOD_FUNCTION(gpublasChbmv, cublasChbmv_v2, hipblasChbmv)
GPUMOD_FUNCTION(gpublasZhbmv, cublasZhbmv_v2, hipblasZhbmv)
GPUMOD_FUNCTION(gpublasChbmv_64, cublasChbmv_v2_64, hipblasChbmv_64)
GPUMOD_FUNCTION(gpublasZhbmv_64, cublasZhbmv_v2_64, hipblasZhbmv_64)

GPUMOD_FUNCTION(gpublasChemv, cublasChemv_v2, hipblasChemv)
GPUMOD_FUNCTION(gpublasZhemv, cublasZhemv_v2, hipblasZhemv)
GPUMOD_FUNCTION(gpublasChemv_64, cublasChemv_v2_64, hipblasChemv_64)
GPUMOD_FUNCTION(gpublasZhemv_64, cublasZhemv_v2_64, hipblasZhemv_64)

GPUMOD_FUNCTION(gpublasCher, cublasCher_v2, hipblasCher)
GPUMOD_FUNCTION(gpublasZher, cublasZher_v2, hipblasZher)
GPUMOD_FUNCTION(gpublasCher_64, cublasCher_v2_64, hipblasCher_64)
GPUMOD_FUNCTION(gpublasZher_64, cublasZher_v2_64, hipblasZher_64)

GPUMOD_FUNCTION(gpublasCher2, cublasCher2_v2, hipblasCher2)
GPUMOD_FUNCTION(gpublasZher2, cublasZher2_v2, hipblasZher2)
GPUMOD_FUNCTION(gpublasCher2_64, cublasCher2_v2_64, hipblasCher2_64)
GPUMOD_FUNCTION(gpublasZher2_64, cublasZher2_v2_64, hipblasZher2_64)

GPUMOD_FUNCTION(gpublasChpmv, cublasChpmv_v2, hipblasChpmv)
GPUMOD_FUNCTION(gpublasZhpmv, cublasZhpmv_v2, hipblasZhpmv)
GPUMOD_FUNCTION(gpublasChpmv_64, cublasChpmv_v2_64, hipblasChpmv_64)
GPUMOD_FUNCTION(gpublasZhpmv_64, cublasZhpmv_v2_64, hipblasZhpmv_64)

GPUMOD_FUNCTION(gpublasChpr, cublasChpr_v2, hipblasChpr)
GPUMOD_FUNCTION(gpublasZhpr, cublasZhpr_v2, hipblasZhpr)
GPUMOD_FUNCTION(gpublasChpr_64, cublasChpr_v2_64, hipblasChpr_64)
GPUMOD_FUNCTION(gpublasZhpr_64, cublasZhpr_v2_64, hipblasZhpr_64)

GPUMOD_FUNCTION(gpublasChpr2, cublasChpr2_v2, hipblasChpr2)
GPUMOD_FUNCTION(gpublasZhpr2, cublasZhpr2_v2, hipblasZhpr2)
GPUMOD_FUNCTION(gpublasChpr2_64, cublasChpr2_v2_64, hipblasChpr2_64)
GPUMOD_FUNCTION(gpublasZhpr2_64, cublasZhpr2_v2_64, hipblasZhpr2_64)

GPUMOD_FUNCTION(gpublasSsbmv, cublasSsbmv_v2, hipblasSsbmv)
GPUMOD_FUNCTION(gpublasDsbmv, cublasDsbmv_v2, hipblasDsbmv)
GPUMOD_FUNCTION(gpublasSsbmv_64, cublasSsbmv_v2_64, hipblasSsbmv_64)
GPUMOD_FUNCTION(gpublasDsbmv_64, cublasDsbmv_v2_64, hipblasDsbmv_64)

GPUMOD_FUNCTION(gpublasSspmv, cublasSspmv_v2, hipblasSspmv)
GPUMOD_FUNCTION(gpublasDspmv, cublasDspmv_v2, hipblasDspmv)
GPUMOD_FUNCTION(gpublasSspmv_64, cublasSspmv_v2_64, hipblasSspmv_64)
GPUMOD_FUNCTION(gpublasDspmv_64, cublasDspmv_v2_64, hipblasDspmv_64)

GPUMOD_FUNCTION(gpublasSspr, cublasSspr_v2, hipblasSspr)
GPUMOD_FUNCTION(gpublasDspr, cublasDspr_v2, hipblasDspr)
GPUMOD_FUNCTION(gpublasSspr_64, cublasSspr_v2_64, hipblasSspr_64)
GPUMOD_FUNCTION(gpublasDspr_64, cublasDspr_v2_64, hipblasDspr_64)

GPUMOD_FUNCTION(gpublasSspr2, cublasSspr2_v2, hipblasSspr2)
GPUMOD_FUNCTION(gpublasDspr2, cublasDspr2_v2, hipblasDspr2)
GPUMOD_FUNCTION(gpublasSspr2_64, cublasSspr2_v2_64, hipblasSspr2_64)
GPUMOD_FUNCTION(gpublasDspr2_64, cublasDspr2_v2_64, hipblasDspr2_64)

GPUMOD_FUNCTION(gpublasSsymv, cublasSsymv_v2, hipblasSsymv)
GPUMOD_FUNCTION(gpublasDsymv, cublasDsymv_v2, hipblasDsymv)
GPUMOD_FUNCTION(gpublasSsymv_64, cublasSsymv_v2_64, hipblasSsymv_64)
GPUMOD_FUNCTION(gpublasDsymv_64, cublasDsymv_v2_64, hipblasDsymv_64)

GPUMOD_FUNCTION(gpublasSsyr, cublasSsyr_v2, hipblasSsyr)
GPUMOD_FUNCTION(gpublasDsyr, cublasDsyr_v2, hipblasDsyr)
GPUMOD_FUNCTION(gpublasSsyr_64, cublasSsyr_v2_64, hipblasSsyr_64)
GPUMOD_FUNCTION(gpublasDsyr_64, cublasDsyr_v2_64, hipblasDsyr_64)

GPUMOD_FUNCTION(gpublasSsyr2, cublasSsyr2_v2, hipblasSsyr2)
GPUMOD_FUNCTION(gpublasDsyr2, cublasDsyr2_v2, hipblasDsyr2)
GPUMOD_FUNCTION(gpublasSsyr2_64, cublasSsyr2_v2_64, hipblasSsyr2_64)
GPUMOD_FUNCTION(gpublasDsyr2_64, cublasDsyr2_v2_64, hipblasDsyr2_64)

GPUMOD_FUNCTION(gpublasStbmv, cublasStbmv_v2, hipblasStbmv)
GPUMOD_FUNCTION(gpublasDtbmv, cublasDtbmv_v2, hipblasDtbmv)
GPUMOD_FUNCTION(gpublasCtbmv, cublasCtbmv_v2, hipblasCtbmv)
GPUMOD_FUNCTION(gpublasZtbmv, cublasZtbmv_v2, hipblasZtbmv)
GPUMOD_FUNCTION(gpublasStbmv_64, cublasStbmv_v2_64, hipblasStbmv_64)
GPUMOD_FUNCTION(gpublasDtbmv_64, cublasDtbmv_v2_64, hipblasDtbmv_64)
GPUMOD_FUNCTION(gpublasCtbmv_64, cublasCtbmv_v2_64, hipblasCtbmv_64)
GPUMOD_FUNCTION(gpublasZtbmv_64, cublasZtbmv_v2_64, hipblasZtbmv_64)

GPUMOD_FUNCTION(gpublasStbsv, cublasStbsv_v2, hipblasStbsv)
GPUMOD_FUNCTION(gpublasDtbsv, cublasDtbsv_v2, hipblasDtbsv)
GPUMOD_FUNCTION(gpublasCtbsv, cublasCtbsv_v2, hipblasCtbsv)
GPUMOD_FUNCTION(gpublasZtbsv, cublasZtbsv_v2, hipblasZtbsv)
GPUMOD_FUNCTION(gpublasStbsv_64, cublasStbsv_v2_64, hipblasStbsv_64)
GPUMOD_FUNCTION(gpublasDtbsv_64, cublasDtbsv_v2_64, hipblasDtbsv_64)
GPUMOD_FUNCTION(gpublasCtbsv_64, cublasCtbsv_v2_64, hipblasCtbsv_64)
GPUMOD_FUNCTION(gpublasZtbsv_64, cublasZtbsv_v2_64, hipblasZtbsv_64)

GPUMOD_FUNCTION(gpublasStpmv, cublasStpmv_v2, hipblasStpmv)
GPUMOD_FUNCTION(gpublasDtpmv, cublasDtpmv_v2, hipblasDtpmv)
GPUMOD_FUNCTION(gpublasCtpmv, cublasCtpmv_v2, hipblasCtpmv)
GPUMOD_FUNCTION(gpublasZtpmv, cublasZtpmv_v2, hipblasZtpmv)
GPUMOD_FUNCTION(gpublasStpmv_64, cublasStpmv_v2_64, hipblasStpmv_64)
GPUMOD_FUNCTION(gpublasDtpmv_64, cublasDtpmv_v2_64, hipblasDtpmv_64)
GPUMOD_FUNCTION(gpublasCtpmv_64, cublasCtpmv_v2_64, hipblasCtpmv_64)
GPUMOD_FUNCTION(gpublasZtpmv_64, cublasZtpmv_v2_64, hipblasZtpmv_64)

GPUMOD_FUNCTION(gpublasStpsv, cublasStpsv_v2, hipblasStpsv)
GPUMOD_FUNCTION(gpublasDtpsv, cublasDtpsv_v2, hipblasDtpsv)
GPUMOD_FUNCTION(gpublasCtpsv, cublasCtpsv_v2, hipblasCtpsv)
GPUMOD_FUNCTION(gpublasZtpsv, cublasZtpsv_v2, hipblasZtpsv)
GPUMOD_FUNCTION(gpublasStpsv_64, cublasStpsv_v2_64, hipblasStpsv_64)
GPUMOD_FUNCTION(gpublasDtpsv_64, cublasDtpsv_v2_64, hipblasDtpsv_64)
GPUMOD_FUNCTION(gpublasCtpsv_64, cublasCtpsv_v2_64, hipblasCtpsv_64)
GPUMOD_FUNCTION(gpublasZtpsv_64, cublasZtpsv_v2_64, hipblasZtpsv_64)

GPUMOD_FUNCTION(gpublasStrmv, cublasStrmv_v2, hipblasStrmv)
GPUMOD_FUNCTION(gpublasDtrmv, cublasDtrmv_v2, hipblasDtrmv)
GPUMOD_FUNCTION(gpublasCtrmv, cublasCtrmv_v2, hipblasCtrmv)
GPUMOD_FUNCTION(gpublasZtrmv, cublasZtrmv_v2, hipblasZtrmv)
GPUMOD_FUNCTION(gpublasStrmv_64, cublasStrmv_v2_64, hipblasStrmv_64)
GPUMOD_FUNCTION(gpublasDtrmv_64, cublasDtrmv_v2_64, hipblasDtrmv_64)
GPUMOD_FUNCTION(gpublasCtrmv_64, cublasCtrmv_v2_64, hipblasCtrmv_64)
GPUMOD_FUNCTION(gpublasZtrmv_64, cublasZtrmv_v2_64, hipblasZtrmv_64)

GPUMOD_FUNCTION(gpublasStrsv, cublasStrsv_v2, hipblasStrsv)
GPUMOD_FUNCTION(gpublasDtrsv, cublasDtrsv_v2, hipblasDtrsv)
GPUMOD_FUNCTION(gpublasCtrsv, cublasCtrsv_v2, hipblasCtrsv)
GPUMOD_FUNCTION(gpublasZtrsv, cublasZtrsv_v2, hipblasZtrsv)
GPUMOD_FUNCTION(gpublasStrsv_64, cublasStrsv_v2_64, hipblasStrsv_64)
GPUMOD_FUNCTION(gpublasDtrsv_64, cublasDtrsv_v2_64, hipblasDtrsv_64)
GPUMOD_FUNCTION(gpublasCtrsv_64, cublasCtrsv_v2_64, hipblasCtrsv_64)
GPUMOD_FUNCTION(gpublasZtrsv_64, cublasZtrsv_v2_64, hipblasZtrsv_64)

// ────────────────────────────────────────────────────────────────────────
// Level 3 (matrix-matrix)
// ────────────────────────────────────────────────────────────────────────

GPUMOD_FUNCTION(gpublasSgemm, cublasSgemm_v2, hipblasSgemm)
GPUMOD_FUNCTION(gpublasDgemm, cublasDgemm_v2, hipblasDgemm)
GPUMOD_FUNCTION(gpublasCgemm, cublasCgemm_v2, hipblasCgemm)
GPUMOD_FUNCTION(gpublasZgemm, cublasZgemm_v2, hipblasZgemm)
GPUMOD_FUNCTION(gpublasSgemm_64, cublasSgemm_v2_64, hipblasSgemm_64)
GPUMOD_FUNCTION(gpublasDgemm_64, cublasDgemm_v2_64, hipblasDgemm_64)
GPUMOD_FUNCTION(gpublasCgemm_64, cublasCgemm_v2_64, hipblasCgemm_64)
GPUMOD_FUNCTION(gpublasZgemm_64, cublasZgemm_v2_64, hipblasZgemm_64)

GPUMOD_FUNCTION(gpublasSgemmBatched, cublasSgemmBatched, hipblasSgemmBatched)
GPUMOD_FUNCTION(gpublasDgemmBatched, cublasDgemmBatched, hipblasDgemmBatched)
GPUMOD_FUNCTION(gpublasCgemmBatched, cublasCgemmBatched, hipblasCgemmBatched)
GPUMOD_FUNCTION(gpublasZgemmBatched, cublasZgemmBatched, hipblasZgemmBatched)
GPUMOD_FUNCTION(gpublasSgemmBatched_64, cublasSgemmBatched_64, hipblasSgemmBatched_64)
GPUMOD_FUNCTION(gpublasDgemmBatched_64, cublasDgemmBatched_64, hipblasDgemmBatched_64)
GPUMOD_FUNCTION(gpublasCgemmBatched_64, cublasCgemmBatched_64, hipblasCgemmBatched_64)
GPUMOD_FUNCTION(gpublasZgemmBatched_64, cublasZgemmBatched_64, hipblasZgemmBatched_64)

GPUMOD_FUNCTION(gpublasSgemmStridedBatched, cublasSgemmStridedBatched, hipblasSgemmStridedBatched)
GPUMOD_FUNCTION(gpublasDgemmStridedBatched, cublasDgemmStridedBatched, hipblasDgemmStridedBatched)
GPUMOD_FUNCTION(gpublasCgemmStridedBatched, cublasCgemmStridedBatched, hipblasCgemmStridedBatched)
GPUMOD_FUNCTION(gpublasZgemmStridedBatched, cublasZgemmStridedBatched, hipblasZgemmStridedBatched)
GPUMOD_FUNCTION(gpublasSgemmStridedBatched_64, cublasSgemmStridedBatched_64,
                hipblasSgemmStridedBatched_64)
GPUMOD_FUNCTION(gpublasDgemmStridedBatched_64, cublasDgemmStridedBatched_64,
                hipblasDgemmStridedBatched_64)
GPUMOD_FUNCTION(gpublasCgemmStridedBatched_64, cublasCgemmStridedBatched_64,
                hipblasCgemmStridedBatched_64)
GPUMOD_FUNCTION(gpublasZgemmStridedBatched_64, cublasZgemmStridedBatched_64,
                hipblasZgemmStridedBatched_64)

GPUMOD_FUNCTION(gpublasChemm, cublasChemm_v2, hipblasChemm)
GPUMOD_FUNCTION(gpublasZhemm, cublasZhemm_v2, hipblasZhemm)
GPUMOD_FUNCTION(gpublasChemm_64, cublasChemm_v2_64, hipblasChemm_64)
GPUMOD_FUNCTION(gpublasZhemm_64, cublasZhemm_v2_64, hipblasZhemm_64)

GPUMOD_FUNCTION(gpublasCher2k, cublasCher2k_v2, hipblasCher2k)
GPUMOD_FUNCTION(gpublasZher2k, cublasZher2k_v2, hipblasZher2k)
GPUMOD_FUNCTION(gpublasCher2k_64, cublasCher2k_v2_64, hipblasCher2k_64)
GPUMOD_FUNCTION(gpublasZher2k_64, cublasZher2k_v2_64, hipblasZher2k_64)

GPUMOD_FUNCTION(gpublasCherk, cublasCherk_v2, hipblasCherk)
GPUMOD_FUNCTION(gpublasZherk, cublasZherk_v2, hipblasZherk)
GPUMOD_FUNCTION(gpublasCherk_64, cublasCherk_v2_64, hipblasCherk_64)
GPUMOD_FUNCTION(gpublasZherk_64, cublasZherk_v2_64, hipblasZherk_64)

GPUMOD_FUNCTION(gpublasCherkx, cublasCherkx, hipblasCherkx)
GPUMOD_FUNCTION(gpublasZherkx, cublasZherkx, hipblasZherkx)
GPUMOD_FUNCTION(gpublasCherkx_64, cublasCherkx_64, hipblasCherkx_64)
GPUMOD_FUNCTION(gpublasZherkx_64, cublasZherkx_64, hipblasZherkx_64)

GPUMOD_FUNCTION(gpublasSsymm, cublasSsymm_v2, hipblasSsymm)
GPUMOD_FUNCTION(gpublasDsymm, cublasDsymm_v2, hipblasDsymm)
GPUMOD_FUNCTION(gpublasCsymm, cublasCsymm_v2, hipblasCsymm)
GPUMOD_FUNCTION(gpublasZsymm, cublasZsymm_v2, hipblasZsymm)
GPUMOD_FUNCTION(gpublasSsymm_64, cublasSsymm_v2_64, hipblasSsymm_64)
GPUMOD_FUNCTION(gpublasDsymm_64, cublasDsymm_v2_64, hipblasDsymm_64)
GPUMOD_FUNCTION(gpublasCsymm_64, cublasCsymm_v2_64, hipblasCsymm_64)
GPUMOD_FUNCTION(gpublasZsymm_64, cublasZsymm_v2_64, hipblasZsymm_64)

GPUMOD_FUNCTION(gpublasSsyr2k, cublasSsyr2k_v2, hipblasSsyr2k)
GPUMOD_FUNCTION(gpublasDsyr2k, cublasDsyr2k_v2, hipblasDsyr2k)
GPUMOD_FUNCTION(gpublasCsyr2k, cublasCsyr2k_v2, hipblasCsyr2k)
GPUMOD_FUNCTION(gpublasZsyr2k, cublasZsyr2k_v2, hipblasZsyr2k)
GPUMOD_FUNCTION(gpublasSsyr2k_64, cublasSsyr2k_v2_64, hipblasSsyr2k_64)
GPUMOD_FUNCTION(gpublasDsyr2k_64, cublasDsyr2k_v2_64, hipblasDsyr2k_64)
GPUMOD_FUNCTION(gpublasCsyr2k_64, cublasCsyr2k_v2_64, hipblasCsyr2k_64)
GPUMOD_FUNCTION(gpublasZsyr2k_64, cublasZsyr2k_v2_64, hipblasZsyr2k_64)

GPUMOD_FUNCTION(gpublasSsyrk, cublasSsyrk_v2, hipblasSsyrk)
GPUMOD_FUNCTION(gpublasDsyrk, cublasDsyrk_v2, hipblasDsyrk)
GPUMOD_FUNCTION(gpublasCsyrk, cublasCsyrk_v2, hipblasCsyrk)
GPUMOD_FUNCTION(gpublasZsyrk, cublasZsyrk_v2, hipblasZsyrk)
GPUMOD_FUNCTION(gpublasSsyrk_64, cublasSsyrk_v2_64, hipblasSsyrk_64)
GPUMOD_FUNCTION(gpublasDsyrk_64, cublasDsyrk_v2_64, hipblasDsyrk_64)
GPUMOD_FUNCTION(gpublasCsyrk_64, cublasCsyrk_v2_64, hipblasCsyrk_64)
GPUMOD_FUNCTION(gpublasZsyrk_64, cublasZsyrk_v2_64, hipblasZsyrk_64)

GPUMOD_FUNCTION(gpublasSsyrkx, cublasSsyrkx, hipblasSsyrkx)
GPUMOD_FUNCTION(gpublasDsyrkx, cublasDsyrkx, hipblasDsyrkx)
GPUMOD_FUNCTION(gpublasCsyrkx, cublasCsyrkx, hipblasCsyrkx)
GPUMOD_FUNCTION(gpublasZsyrkx, cublasZsyrkx, hipblasZsyrkx)
GPUMOD_FUNCTION(gpublasSsyrkx_64, cublasSsyrkx_64, hipblasSsyrkx_64)
GPUMOD_FUNCTION(gpublasDsyrkx_64, cublasDsyrkx_64, hipblasDsyrkx_64)
GPUMOD_FUNCTION(gpublasCsyrkx_64, cublasCsyrkx_64, hipblasCsyrkx_64)
GPUMOD_FUNCTION(gpublasZsyrkx_64, cublasZsyrkx_64, hipblasZsyrkx_64)

GPUMOD_FUNCTION(gpublasStrmm, cublasStrmm_v2, hipblasStrmm)
GPUMOD_FUNCTION(gpublasDtrmm, cublasDtrmm_v2, hipblasDtrmm)
GPUMOD_FUNCTION(gpublasCtrmm, cublasCtrmm_v2, hipblasCtrmm)
GPUMOD_FUNCTION(gpublasZtrmm, cublasZtrmm_v2, hipblasZtrmm)
GPUMOD_FUNCTION(gpublasStrmm_64, cublasStrmm_v2_64, hipblasStrmm_64)
GPUMOD_FUNCTION(gpublasDtrmm_64, cublasDtrmm_v2_64, hipblasDtrmm_64)
GPUMOD_FUNCTION(gpublasCtrmm_64, cublasCtrmm_v2_64, hipblasCtrmm_64)
GPUMOD_FUNCTION(gpublasZtrmm_64, cublasZtrmm_v2_64, hipblasZtrmm_64)

GPUMOD_FUNCTION(gpublasStrsm, cublasStrsm_v2, hipblasStrsm)
GPUMOD_FUNCTION(gpublasDtrsm, cublasDtrsm_v2, hipblasDtrsm)
GPUMOD_FUNCTION(gpublasCtrsm, cublasCtrsm_v2, hipblasCtrsm)
GPUMOD_FUNCTION(gpublasZtrsm, cublasZtrsm_v2, hipblasZtrsm)
GPUMOD_FUNCTION(gpublasStrsm_64, cublasStrsm_v2_64, hipblasStrsm_64)
GPUMOD_FUNCTION(gpublasDtrsm_64, cublasDtrsm_v2_64, hipblasDtrsm_64)
GPUMOD_FUNCTION(gpublasCtrsm_64, cublasCtrsm_v2_64, hipblasCtrsm_64)
GPUMOD_FUNCTION(gpublasZtrsm_64, cublasZtrsm_v2_64, hipblasZtrsm_64)

GPUMOD_FUNCTION(gpublasStrsmBatched, cublasStrsmBatched, hipblasStrsmBatched)
GPUMOD_FUNCTION(gpublasDtrsmBatched, cublasDtrsmBatched, hipblasDtrsmBatched)
GPUMOD_FUNCTION(gpublasCtrsmBatched, cublasCtrsmBatched, hipblasCtrsmBatched)
GPUMOD_FUNCTION(gpublasZtrsmBatched, cublasZtrsmBatched, hipblasZtrsmBatched)
GPUMOD_FUNCTION(gpublasStrsmBatched_64, cublasStrsmBatched_64, hipblasStrsmBatched_64)
GPUMOD_FUNCTION(gpublasDtrsmBatched_64, cublasDtrsmBatched_64, hipblasDtrsmBatched_64)
GPUMOD_FUNCTION(gpublasCtrsmBatched_64, cublasCtrsmBatched_64, hipblasCtrsmBatched_64)
GPUMOD_FUNCTION(gpublasZtrsmBatched_64, cublasZtrsmBatched_64, hipblasZtrsmBatched_64)

// ────────────────────────────────────────────────────────────────────────
// BLAS-like extensions
// ────────────────────────────────────────────────────────────────────────

GPUMOD_FUNCTION(gpublasSdgmm, cublasSdgmm, hipblasSdgmm)
GPUMOD_FUNCTION(gpublasDdgmm, cublasDdgmm, hipblasDdgmm)
GPUMOD_FUNCTION(gpublasCdgmm, cublasCdgmm, hipblasCdgmm)
GPUMOD_FUNCTION(gpublasZdgmm, cublasZdgmm, hipblasZdgmm)
GPUMOD_FUNCTION(gpublasSdgmm_64, cublasSdgmm_64, hipblasSdgmm_64)
GPUMOD_FUNCTION(gpublasDdgmm_64, cublasDdgmm_64, hipblasDdgmm_64)
GPUMOD_FUNCTION(gpublasCdgmm_64, cublasCdgmm_64, hipblasCdgmm_64)
GPUMOD_FUNCTION(gpublasZdgmm_64, cublasZdgmm_64, hipblasZdgmm_64)

GPUMOD_FUNCTION(gpublasSgeam, cublasSgeam, hipblasSgeam)
GPUMOD_FUNCTION(gpublasDgeam, cublasDgeam, hipblasDgeam)
GPUMOD_FUNCTION(gpublasCgeam, cublasCgeam, hipblasCgeam)
GPUMOD_FUNCTION(gpublasZgeam, cublasZgeam, hipblasZgeam)
GPUMOD_FUNCTION(gpublasSgeam_64, cublasSgeam_64, hipblasSgeam_64)
GPUMOD_FUNCTION(gpublasDgeam_64, cublasDgeam_64, hipblasDgeam_64)
GPUMOD_FUNCTION(gpublasCgeam_64, cublasCgeam_64, hipblasCgeam_64)
GPUMOD_FUNCTION(gpublasZgeam_64, cublasZgeam_64, hipblasZgeam_64)

GPUMOD_FUNCTION(gpublasSgelsBatched, cublasSgelsBatched, hipblasSgelsBatched)
GPUMOD_FUNCTION(gpublasDgelsBatched, cublasDgelsBatched, hipblasDgelsBatched)
GPUMOD_FUNCTION(gpublasCgelsBatched, cublasCgelsBatched, hipblasCgelsBatched)
GPUMOD_FUNCTION(gpublasZgelsBatched, cublasZgelsBatched, hipblasZgelsBatched)

GPUMOD_FUNCTION(gpublasSgeqrfBatched, cublasSgeqrfBatched, hipblasSgeqrfBatched)
GPUMOD_FUNCTION(gpublasDgeqrfBatched, cublasDgeqrfBatched, hipblasDgeqrfBatched)
GPUMOD_FUNCTION(gpublasCgeqrfBatched, cublasCgeqrfBatched, hipblasCgeqrfBatched)
GPUMOD_FUNCTION(gpublasZgeqrfBatched, cublasZgeqrfBatched, hipblasZgeqrfBatched)

GPUMOD_FUNCTION(gpublasSgetrfBatched, cublasSgetrfBatched, hipblasSgetrfBatched)
GPUMOD_FUNCTION(gpublasDgetrfBatched, cublasDgetrfBatched, hipblasDgetrfBatched)
GPUMOD_FUNCTION(gpublasCgetrfBatched, cublasCgetrfBatched, hipblasCgetrfBatched)
GPUMOD_FUNCTION(gpublasZgetrfBatched, cublasZgetrfBatched, hipblasZgetrfBatched)

// ────────────────────────────────────────────────────────────────────────
// getrsBatched / getriBatched: const-correct on both backends
// ────────────────────────────────────────────────────────────────────────

#if defined(GPUMOD_GPU_BACKEND_CUDA)

GPUMOD_FUNCTION(gpublasSgetrsBatched, cublasSgetrsBatched, hipblasSgetrsBatched)
GPUMOD_FUNCTION(gpublasDgetrsBatched, cublasDgetrsBatched, hipblasDgetrsBatched)
GPUMOD_FUNCTION(gpublasCgetrsBatched, cublasCgetrsBatched, hipblasCgetrsBatched)
GPUMOD_FUNCTION(gpublasZgetrsBatched, cublasZgetrsBatched, hipblasZgetrsBatched)

GPUMOD_FUNCTION(gpublasSgetriBatched, cublasSgetriBatched, hipblasSgetriBatched)
GPUMOD_FUNCTION(gpublasDgetriBatched, cublasDgetriBatched, hipblasDgetriBatched)
GPUMOD_FUNCTION(gpublasCgetriBatched, cublasCgetriBatched, hipblasCgetriBatched)
GPUMOD_FUNCTION(gpublasZgetriBatched, cublasZgetriBatched, hipblasZgetriBatched)

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

} // namespace gpumod
