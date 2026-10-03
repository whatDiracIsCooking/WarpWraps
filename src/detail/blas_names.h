/**
 * @file detail/blas_names.h
 * @brief The backend-neutral cuBLAS / hipBLAS surface, as a macro-driven include
 *        fragment shared by the module and the non-module #include path
 *
 * NOT a standalone header: it is the list of wwrblas* / WWRBLAS_* names (types,
 * constants, functions and the const-correct getrs/getri batched forwarders) with
 * NO namespace of its own and NO vendor #include. The includer supplies all of
 * that and pastes this inside its own `namespace wwr` -- so one list binds both
 * ways the surface is consumed: wwr.blas (the module, `export namespace wwr`) and
 * wwr/blas.h (the non-module #include path). Add a name here, once, and both paths
 * gain it.
 *
 * HOST only -- cuBLAS/hipBLAS is a host API. The surface binds straight to the
 * vendor's external-linkage `::cublas*_v2` / `::hipblas*` declarations via the
 * _RAW macros (the "rand.h shape"): no raw vendor module is imported, so the same
 * `::`-prefixed names resolve in the module (vendor header in its GMF) and the
 * #include path alike. The cuBLAS-only extras (gemm3m, matinvBatched, ...) are
 * deliberately absent; reach those through wwr.cuda.cublas_v2.
 *
 * Before including, the includer must have, in order:
 *   - the vendor header in scope (cublas_v2.h / hipblas.h), which blas.h pulls in;
 *   - wwrFloatComplex / wwrDoubleComplex in `namespace wwr` (the getri/getrs HIP
 *     forwarders name them) -- from import wwr.complex in the module, or
 *     #include "complex.h" in the #include path;
 *   - WWR_SELECT_RAW(cuda, hip) plus WWR_TYPE_RAW / WWR_VALUE_RAW /
 *     WWR_FUNCTION_RAW on top of it -- keyed on WWR_GPU_BACKEND_* in the module
 *     (backend.h) or WWR_SELECTED_* in the #include path (wwr/blas.h);
 *   - WWR_SELECTED_CUDA / WWR_SELECTED_HIP (selected_backend.h, via blas.h) for
 *     the getrs/getri forwarders' one backend conditional.
 *
 * See src/blas.cppm, src/blas.h, src/wwr/blas.h and docs/architecture.md
 * section 5.
 */

#pragma once

#ifndef WWR_FUNCTION_RAW
#error                                                                                             \
    "detail/blas_names.h is an include fragment, not a standalone header: define WWR_TYPE_RAW/VALUE_RAW/FUNCTION_RAW and WWR_SELECT_RAW, ensure the vendor header (via blas.h), wwrFloatComplex and WWR_SELECTED_* are in scope, and #include it inside namespace wwr. See src/blas.h, src/blas.cppm and src/wwr/blas.h."
#endif

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables): each wwrblas*
// function below is a deliberate constexpr reference to the selected backend's
// entry point (via WWR_FUNCTION_RAW). A reference to a vendor function has no
// const form, so the check cannot be satisfied without abandoning the alias
// pattern -- see backend.h and detail/runtime_api_names.h.

// ========================================================================
// Types
// ========================================================================

WWR_TYPE_RAW(wwrblasHandle_t, cublasHandle_t, hipblasHandle_t)
WWR_TYPE_RAW(wwrblasStatus_t, cublasStatus_t, hipblasStatus_t)
WWR_TYPE_RAW(wwrblasOperation_t, cublasOperation_t, hipblasOperation_t)
WWR_TYPE_RAW(wwrblasFillMode_t, cublasFillMode_t, hipblasFillMode_t)
WWR_TYPE_RAW(wwrblasDiagType_t, cublasDiagType_t, hipblasDiagType_t)
WWR_TYPE_RAW(wwrblasSideMode_t, cublasSideMode_t, hipblasSideMode_t)
WWR_TYPE_RAW(wwrblasPointerMode_t, cublasPointerMode_t, hipblasPointerMode_t)

// ========================================================================
// Constants
// ========================================================================

// The status codes whose names cuBLAS and hipBLAS share -- their numeric
// values differ between the backends, which is exactly what these neutral
// names paper over. LICENSE_ERROR (cuBLAS-only) and HANDLE_IS_NULLPTR /
// INVALID_ENUM / UNKNOWN (hipBLAS-only) have no counterpart, so they cannot
// be neutral and are deliberately absent.
WWR_VALUE_RAW(WWRBLAS_STATUS_SUCCESS, CUBLAS_STATUS_SUCCESS, HIPBLAS_STATUS_SUCCESS)
WWR_VALUE_RAW(WWRBLAS_STATUS_NOT_INITIALIZED, CUBLAS_STATUS_NOT_INITIALIZED,
             HIPBLAS_STATUS_NOT_INITIALIZED)
WWR_VALUE_RAW(WWRBLAS_STATUS_ALLOC_FAILED, CUBLAS_STATUS_ALLOC_FAILED, HIPBLAS_STATUS_ALLOC_FAILED)
WWR_VALUE_RAW(WWRBLAS_STATUS_ARCH_MISMATCH, CUBLAS_STATUS_ARCH_MISMATCH, HIPBLAS_STATUS_ARCH_MISMATCH)
WWR_VALUE_RAW(WWRBLAS_STATUS_EXECUTION_FAILED, CUBLAS_STATUS_EXECUTION_FAILED,
             HIPBLAS_STATUS_EXECUTION_FAILED)
WWR_VALUE_RAW(WWRBLAS_STATUS_INTERNAL_ERROR, CUBLAS_STATUS_INTERNAL_ERROR,
             HIPBLAS_STATUS_INTERNAL_ERROR)
WWR_VALUE_RAW(WWRBLAS_STATUS_INVALID_VALUE, CUBLAS_STATUS_INVALID_VALUE, HIPBLAS_STATUS_INVALID_VALUE)
WWR_VALUE_RAW(WWRBLAS_STATUS_MAPPING_ERROR, CUBLAS_STATUS_MAPPING_ERROR, HIPBLAS_STATUS_MAPPING_ERROR)
WWR_VALUE_RAW(WWRBLAS_STATUS_NOT_SUPPORTED, CUBLAS_STATUS_NOT_SUPPORTED, HIPBLAS_STATUS_NOT_SUPPORTED)

WWR_VALUE_RAW(WWRBLAS_OP_N, CUBLAS_OP_N, HIPBLAS_OP_N)
WWR_VALUE_RAW(WWRBLAS_OP_T, CUBLAS_OP_T, HIPBLAS_OP_T)
WWR_VALUE_RAW(WWRBLAS_OP_C, CUBLAS_OP_C, HIPBLAS_OP_C)

WWR_VALUE_RAW(WWRBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_LOWER, HIPBLAS_FILL_MODE_LOWER)
WWR_VALUE_RAW(WWRBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_UPPER, HIPBLAS_FILL_MODE_UPPER)

WWR_VALUE_RAW(WWRBLAS_DIAG_NON_UNIT, CUBLAS_DIAG_NON_UNIT, HIPBLAS_DIAG_NON_UNIT)
WWR_VALUE_RAW(WWRBLAS_DIAG_UNIT, CUBLAS_DIAG_UNIT, HIPBLAS_DIAG_UNIT)

WWR_VALUE_RAW(WWRBLAS_SIDE_LEFT, CUBLAS_SIDE_LEFT, HIPBLAS_SIDE_LEFT)
WWR_VALUE_RAW(WWRBLAS_SIDE_RIGHT, CUBLAS_SIDE_RIGHT, HIPBLAS_SIDE_RIGHT)

WWR_VALUE_RAW(WWRBLAS_POINTER_MODE_HOST, CUBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_HOST)
WWR_VALUE_RAW(WWRBLAS_POINTER_MODE_DEVICE, CUBLAS_POINTER_MODE_DEVICE, HIPBLAS_POINTER_MODE_DEVICE)

// ========================================================================
// Handle, stream, pointer mode, status strings
// ========================================================================

WWR_FUNCTION_RAW(wwrblasCreate, cublasCreate_v2, hipblasCreate)
WWR_FUNCTION_RAW(wwrblasDestroy, cublasDestroy_v2, hipblasDestroy)
WWR_FUNCTION_RAW(wwrblasSetStream, cublasSetStream_v2, hipblasSetStream)
WWR_FUNCTION_RAW(wwrblasGetStream, cublasGetStream_v2, hipblasGetStream)
WWR_FUNCTION_RAW(wwrblasSetPointerMode, cublasSetPointerMode_v2, hipblasSetPointerMode)
WWR_FUNCTION_RAW(wwrblasGetPointerMode, cublasGetPointerMode_v2, hipblasGetPointerMode)

// hipBLAS has a single status-to-string function; both names map to it.
WWR_FUNCTION_RAW(wwrblasGetStatusName, cublasGetStatusName, hipblasStatusToString)
WWR_FUNCTION_RAW(wwrblasGetStatusString, cublasGetStatusString, hipblasStatusToString)

// ────────────────────────────────────────────────────────────────────────
// Level 1 (vector-vector)
// ────────────────────────────────────────────────────────────────────────

WWR_FUNCTION_RAW(wwrblasSasum, cublasSasum_v2, hipblasSasum)
WWR_FUNCTION_RAW(wwrblasDasum, cublasDasum_v2, hipblasDasum)
WWR_FUNCTION_RAW(wwrblasSasum_64, cublasSasum_v2_64, hipblasSasum_64)
WWR_FUNCTION_RAW(wwrblasDasum_64, cublasDasum_v2_64, hipblasDasum_64)

WWR_FUNCTION_RAW(wwrblasSaxpy, cublasSaxpy_v2, hipblasSaxpy)
WWR_FUNCTION_RAW(wwrblasDaxpy, cublasDaxpy_v2, hipblasDaxpy)
WWR_FUNCTION_RAW(wwrblasCaxpy, cublasCaxpy_v2, hipblasCaxpy)
WWR_FUNCTION_RAW(wwrblasZaxpy, cublasZaxpy_v2, hipblasZaxpy)
WWR_FUNCTION_RAW(wwrblasSaxpy_64, cublasSaxpy_v2_64, hipblasSaxpy_64)
WWR_FUNCTION_RAW(wwrblasDaxpy_64, cublasDaxpy_v2_64, hipblasDaxpy_64)
WWR_FUNCTION_RAW(wwrblasCaxpy_64, cublasCaxpy_v2_64, hipblasCaxpy_64)
WWR_FUNCTION_RAW(wwrblasZaxpy_64, cublasZaxpy_v2_64, hipblasZaxpy_64)

WWR_FUNCTION_RAW(wwrblasScasum, cublasScasum_v2, hipblasScasum)
WWR_FUNCTION_RAW(wwrblasScasum_64, cublasScasum_v2_64, hipblasScasum_64)

WWR_FUNCTION_RAW(wwrblasScnrm2, cublasScnrm2_v2, hipblasScnrm2)
WWR_FUNCTION_RAW(wwrblasScnrm2_64, cublasScnrm2_v2_64, hipblasScnrm2_64)

WWR_FUNCTION_RAW(wwrblasScopy, cublasScopy_v2, hipblasScopy)
WWR_FUNCTION_RAW(wwrblasDcopy, cublasDcopy_v2, hipblasDcopy)
WWR_FUNCTION_RAW(wwrblasCcopy, cublasCcopy_v2, hipblasCcopy)
WWR_FUNCTION_RAW(wwrblasZcopy, cublasZcopy_v2, hipblasZcopy)
WWR_FUNCTION_RAW(wwrblasScopy_64, cublasScopy_v2_64, hipblasScopy_64)
WWR_FUNCTION_RAW(wwrblasDcopy_64, cublasDcopy_v2_64, hipblasDcopy_64)
WWR_FUNCTION_RAW(wwrblasCcopy_64, cublasCcopy_v2_64, hipblasCcopy_64)
WWR_FUNCTION_RAW(wwrblasZcopy_64, cublasZcopy_v2_64, hipblasZcopy_64)

WWR_FUNCTION_RAW(wwrblasSdot, cublasSdot_v2, hipblasSdot)
WWR_FUNCTION_RAW(wwrblasDdot, cublasDdot_v2, hipblasDdot)
WWR_FUNCTION_RAW(wwrblasSdot_64, cublasSdot_v2_64, hipblasSdot_64)
WWR_FUNCTION_RAW(wwrblasDdot_64, cublasDdot_v2_64, hipblasDdot_64)

WWR_FUNCTION_RAW(wwrblasCdotc, cublasCdotc_v2, hipblasCdotc)
WWR_FUNCTION_RAW(wwrblasZdotc, cublasZdotc_v2, hipblasZdotc)
WWR_FUNCTION_RAW(wwrblasCdotc_64, cublasCdotc_v2_64, hipblasCdotc_64)
WWR_FUNCTION_RAW(wwrblasZdotc_64, cublasZdotc_v2_64, hipblasZdotc_64)

WWR_FUNCTION_RAW(wwrblasCdotu, cublasCdotu_v2, hipblasCdotu)
WWR_FUNCTION_RAW(wwrblasZdotu, cublasZdotu_v2, hipblasZdotu)
WWR_FUNCTION_RAW(wwrblasCdotu_64, cublasCdotu_v2_64, hipblasCdotu_64)
WWR_FUNCTION_RAW(wwrblasZdotu_64, cublasZdotu_v2_64, hipblasZdotu_64)

WWR_FUNCTION_RAW(wwrblasZdrot, cublasZdrot_v2, hipblasZdrot)
WWR_FUNCTION_RAW(wwrblasZdrot_64, cublasZdrot_v2_64, hipblasZdrot_64)

WWR_FUNCTION_RAW(wwrblasZdscal, cublasZdscal_v2, hipblasZdscal)
WWR_FUNCTION_RAW(wwrblasZdscal_64, cublasZdscal_v2_64, hipblasZdscal_64)

WWR_FUNCTION_RAW(wwrblasIsamax, cublasIsamax_v2, hipblasIsamax)
WWR_FUNCTION_RAW(wwrblasIdamax, cublasIdamax_v2, hipblasIdamax)
WWR_FUNCTION_RAW(wwrblasIcamax, cublasIcamax_v2, hipblasIcamax)
WWR_FUNCTION_RAW(wwrblasIzamax, cublasIzamax_v2, hipblasIzamax)
WWR_FUNCTION_RAW(wwrblasIsamax_64, cublasIsamax_v2_64, hipblasIsamax_64)
WWR_FUNCTION_RAW(wwrblasIdamax_64, cublasIdamax_v2_64, hipblasIdamax_64)
WWR_FUNCTION_RAW(wwrblasIcamax_64, cublasIcamax_v2_64, hipblasIcamax_64)
WWR_FUNCTION_RAW(wwrblasIzamax_64, cublasIzamax_v2_64, hipblasIzamax_64)

WWR_FUNCTION_RAW(wwrblasIsamin, cublasIsamin_v2, hipblasIsamin)
WWR_FUNCTION_RAW(wwrblasIdamin, cublasIdamin_v2, hipblasIdamin)
WWR_FUNCTION_RAW(wwrblasIcamin, cublasIcamin_v2, hipblasIcamin)
WWR_FUNCTION_RAW(wwrblasIzamin, cublasIzamin_v2, hipblasIzamin)
WWR_FUNCTION_RAW(wwrblasIsamin_64, cublasIsamin_v2_64, hipblasIsamin_64)
WWR_FUNCTION_RAW(wwrblasIdamin_64, cublasIdamin_v2_64, hipblasIdamin_64)
WWR_FUNCTION_RAW(wwrblasIcamin_64, cublasIcamin_v2_64, hipblasIcamin_64)
WWR_FUNCTION_RAW(wwrblasIzamin_64, cublasIzamin_v2_64, hipblasIzamin_64)

WWR_FUNCTION_RAW(wwrblasSnrm2, cublasSnrm2_v2, hipblasSnrm2)
WWR_FUNCTION_RAW(wwrblasDnrm2, cublasDnrm2_v2, hipblasDnrm2)
WWR_FUNCTION_RAW(wwrblasSnrm2_64, cublasSnrm2_v2_64, hipblasSnrm2_64)
WWR_FUNCTION_RAW(wwrblasDnrm2_64, cublasDnrm2_v2_64, hipblasDnrm2_64)

WWR_FUNCTION_RAW(wwrblasSrot, cublasSrot_v2, hipblasSrot)
WWR_FUNCTION_RAW(wwrblasDrot, cublasDrot_v2, hipblasDrot)
WWR_FUNCTION_RAW(wwrblasCrot, cublasCrot_v2, hipblasCrot)
WWR_FUNCTION_RAW(wwrblasZrot, cublasZrot_v2, hipblasZrot)
WWR_FUNCTION_RAW(wwrblasSrot_64, cublasSrot_v2_64, hipblasSrot_64)
WWR_FUNCTION_RAW(wwrblasDrot_64, cublasDrot_v2_64, hipblasDrot_64)
WWR_FUNCTION_RAW(wwrblasCrot_64, cublasCrot_v2_64, hipblasCrot_64)
WWR_FUNCTION_RAW(wwrblasZrot_64, cublasZrot_v2_64, hipblasZrot_64)

WWR_FUNCTION_RAW(wwrblasSrotg, cublasSrotg_v2, hipblasSrotg)
WWR_FUNCTION_RAW(wwrblasDrotg, cublasDrotg_v2, hipblasDrotg)
WWR_FUNCTION_RAW(wwrblasCrotg, cublasCrotg_v2, hipblasCrotg)
WWR_FUNCTION_RAW(wwrblasZrotg, cublasZrotg_v2, hipblasZrotg)

WWR_FUNCTION_RAW(wwrblasSrotm, cublasSrotm_v2, hipblasSrotm)
WWR_FUNCTION_RAW(wwrblasDrotm, cublasDrotm_v2, hipblasDrotm)
WWR_FUNCTION_RAW(wwrblasSrotm_64, cublasSrotm_v2_64, hipblasSrotm_64)
WWR_FUNCTION_RAW(wwrblasDrotm_64, cublasDrotm_v2_64, hipblasDrotm_64)

WWR_FUNCTION_RAW(wwrblasSrotmg, cublasSrotmg_v2, hipblasSrotmg)
WWR_FUNCTION_RAW(wwrblasDrotmg, cublasDrotmg_v2, hipblasDrotmg)

WWR_FUNCTION_RAW(wwrblasSscal, cublasSscal_v2, hipblasSscal)
WWR_FUNCTION_RAW(wwrblasDscal, cublasDscal_v2, hipblasDscal)
WWR_FUNCTION_RAW(wwrblasCscal, cublasCscal_v2, hipblasCscal)
WWR_FUNCTION_RAW(wwrblasZscal, cublasZscal_v2, hipblasZscal)
WWR_FUNCTION_RAW(wwrblasSscal_64, cublasSscal_v2_64, hipblasSscal_64)
WWR_FUNCTION_RAW(wwrblasDscal_64, cublasDscal_v2_64, hipblasDscal_64)
WWR_FUNCTION_RAW(wwrblasCscal_64, cublasCscal_v2_64, hipblasCscal_64)
WWR_FUNCTION_RAW(wwrblasZscal_64, cublasZscal_v2_64, hipblasZscal_64)

WWR_FUNCTION_RAW(wwrblasCsrot, cublasCsrot_v2, hipblasCsrot)
WWR_FUNCTION_RAW(wwrblasCsrot_64, cublasCsrot_v2_64, hipblasCsrot_64)

WWR_FUNCTION_RAW(wwrblasCsscal, cublasCsscal_v2, hipblasCsscal)
WWR_FUNCTION_RAW(wwrblasCsscal_64, cublasCsscal_v2_64, hipblasCsscal_64)

WWR_FUNCTION_RAW(wwrblasSswap, cublasSswap_v2, hipblasSswap)
WWR_FUNCTION_RAW(wwrblasDswap, cublasDswap_v2, hipblasDswap)
WWR_FUNCTION_RAW(wwrblasCswap, cublasCswap_v2, hipblasCswap)
WWR_FUNCTION_RAW(wwrblasZswap, cublasZswap_v2, hipblasZswap)
WWR_FUNCTION_RAW(wwrblasSswap_64, cublasSswap_v2_64, hipblasSswap_64)
WWR_FUNCTION_RAW(wwrblasDswap_64, cublasDswap_v2_64, hipblasDswap_64)
WWR_FUNCTION_RAW(wwrblasCswap_64, cublasCswap_v2_64, hipblasCswap_64)
WWR_FUNCTION_RAW(wwrblasZswap_64, cublasZswap_v2_64, hipblasZswap_64)

WWR_FUNCTION_RAW(wwrblasDzasum, cublasDzasum_v2, hipblasDzasum)
WWR_FUNCTION_RAW(wwrblasDzasum_64, cublasDzasum_v2_64, hipblasDzasum_64)

WWR_FUNCTION_RAW(wwrblasDznrm2, cublasDznrm2_v2, hipblasDznrm2)
WWR_FUNCTION_RAW(wwrblasDznrm2_64, cublasDznrm2_v2_64, hipblasDznrm2_64)

// ────────────────────────────────────────────────────────────────────────
// Level 2 (matrix-vector)
// ────────────────────────────────────────────────────────────────────────

WWR_FUNCTION_RAW(wwrblasSgbmv, cublasSgbmv_v2, hipblasSgbmv)
WWR_FUNCTION_RAW(wwrblasDgbmv, cublasDgbmv_v2, hipblasDgbmv)
WWR_FUNCTION_RAW(wwrblasCgbmv, cublasCgbmv_v2, hipblasCgbmv)
WWR_FUNCTION_RAW(wwrblasZgbmv, cublasZgbmv_v2, hipblasZgbmv)
WWR_FUNCTION_RAW(wwrblasSgbmv_64, cublasSgbmv_v2_64, hipblasSgbmv_64)
WWR_FUNCTION_RAW(wwrblasDgbmv_64, cublasDgbmv_v2_64, hipblasDgbmv_64)
WWR_FUNCTION_RAW(wwrblasCgbmv_64, cublasCgbmv_v2_64, hipblasCgbmv_64)
WWR_FUNCTION_RAW(wwrblasZgbmv_64, cublasZgbmv_v2_64, hipblasZgbmv_64)

WWR_FUNCTION_RAW(wwrblasSgemv, cublasSgemv_v2, hipblasSgemv)
WWR_FUNCTION_RAW(wwrblasDgemv, cublasDgemv_v2, hipblasDgemv)
WWR_FUNCTION_RAW(wwrblasCgemv, cublasCgemv_v2, hipblasCgemv)
WWR_FUNCTION_RAW(wwrblasZgemv, cublasZgemv_v2, hipblasZgemv)
WWR_FUNCTION_RAW(wwrblasSgemv_64, cublasSgemv_v2_64, hipblasSgemv_64)
WWR_FUNCTION_RAW(wwrblasDgemv_64, cublasDgemv_v2_64, hipblasDgemv_64)
WWR_FUNCTION_RAW(wwrblasCgemv_64, cublasCgemv_v2_64, hipblasCgemv_64)
WWR_FUNCTION_RAW(wwrblasZgemv_64, cublasZgemv_v2_64, hipblasZgemv_64)

WWR_FUNCTION_RAW(wwrblasSgemvBatched, cublasSgemvBatched, hipblasSgemvBatched)
WWR_FUNCTION_RAW(wwrblasDgemvBatched, cublasDgemvBatched, hipblasDgemvBatched)
WWR_FUNCTION_RAW(wwrblasCgemvBatched, cublasCgemvBatched, hipblasCgemvBatched)
WWR_FUNCTION_RAW(wwrblasZgemvBatched, cublasZgemvBatched, hipblasZgemvBatched)
WWR_FUNCTION_RAW(wwrblasSgemvBatched_64, cublasSgemvBatched_64, hipblasSgemvBatched_64)
WWR_FUNCTION_RAW(wwrblasDgemvBatched_64, cublasDgemvBatched_64, hipblasDgemvBatched_64)
WWR_FUNCTION_RAW(wwrblasCgemvBatched_64, cublasCgemvBatched_64, hipblasCgemvBatched_64)
WWR_FUNCTION_RAW(wwrblasZgemvBatched_64, cublasZgemvBatched_64, hipblasZgemvBatched_64)

WWR_FUNCTION_RAW(wwrblasSgemvStridedBatched, cublasSgemvStridedBatched, hipblasSgemvStridedBatched)
WWR_FUNCTION_RAW(wwrblasDgemvStridedBatched, cublasDgemvStridedBatched, hipblasDgemvStridedBatched)
WWR_FUNCTION_RAW(wwrblasCgemvStridedBatched, cublasCgemvStridedBatched, hipblasCgemvStridedBatched)
WWR_FUNCTION_RAW(wwrblasZgemvStridedBatched, cublasZgemvStridedBatched, hipblasZgemvStridedBatched)
WWR_FUNCTION_RAW(wwrblasSgemvStridedBatched_64, cublasSgemvStridedBatched_64,
                hipblasSgemvStridedBatched_64)
WWR_FUNCTION_RAW(wwrblasDgemvStridedBatched_64, cublasDgemvStridedBatched_64,
                hipblasDgemvStridedBatched_64)
WWR_FUNCTION_RAW(wwrblasCgemvStridedBatched_64, cublasCgemvStridedBatched_64,
                hipblasCgemvStridedBatched_64)
WWR_FUNCTION_RAW(wwrblasZgemvStridedBatched_64, cublasZgemvStridedBatched_64,
                hipblasZgemvStridedBatched_64)

WWR_FUNCTION_RAW(wwrblasSger, cublasSger_v2, hipblasSger)
WWR_FUNCTION_RAW(wwrblasDger, cublasDger_v2, hipblasDger)
WWR_FUNCTION_RAW(wwrblasSger_64, cublasSger_v2_64, hipblasSger_64)
WWR_FUNCTION_RAW(wwrblasDger_64, cublasDger_v2_64, hipblasDger_64)

WWR_FUNCTION_RAW(wwrblasCgerc, cublasCgerc_v2, hipblasCgerc)
WWR_FUNCTION_RAW(wwrblasZgerc, cublasZgerc_v2, hipblasZgerc)
WWR_FUNCTION_RAW(wwrblasCgerc_64, cublasCgerc_v2_64, hipblasCgerc_64)
WWR_FUNCTION_RAW(wwrblasZgerc_64, cublasZgerc_v2_64, hipblasZgerc_64)

WWR_FUNCTION_RAW(wwrblasCgeru, cublasCgeru_v2, hipblasCgeru)
WWR_FUNCTION_RAW(wwrblasZgeru, cublasZgeru_v2, hipblasZgeru)
WWR_FUNCTION_RAW(wwrblasCgeru_64, cublasCgeru_v2_64, hipblasCgeru_64)
WWR_FUNCTION_RAW(wwrblasZgeru_64, cublasZgeru_v2_64, hipblasZgeru_64)

WWR_FUNCTION_RAW(wwrblasChbmv, cublasChbmv_v2, hipblasChbmv)
WWR_FUNCTION_RAW(wwrblasZhbmv, cublasZhbmv_v2, hipblasZhbmv)
WWR_FUNCTION_RAW(wwrblasChbmv_64, cublasChbmv_v2_64, hipblasChbmv_64)
WWR_FUNCTION_RAW(wwrblasZhbmv_64, cublasZhbmv_v2_64, hipblasZhbmv_64)

WWR_FUNCTION_RAW(wwrblasChemv, cublasChemv_v2, hipblasChemv)
WWR_FUNCTION_RAW(wwrblasZhemv, cublasZhemv_v2, hipblasZhemv)
WWR_FUNCTION_RAW(wwrblasChemv_64, cublasChemv_v2_64, hipblasChemv_64)
WWR_FUNCTION_RAW(wwrblasZhemv_64, cublasZhemv_v2_64, hipblasZhemv_64)

WWR_FUNCTION_RAW(wwrblasCher, cublasCher_v2, hipblasCher)
WWR_FUNCTION_RAW(wwrblasZher, cublasZher_v2, hipblasZher)
WWR_FUNCTION_RAW(wwrblasCher_64, cublasCher_v2_64, hipblasCher_64)
WWR_FUNCTION_RAW(wwrblasZher_64, cublasZher_v2_64, hipblasZher_64)

WWR_FUNCTION_RAW(wwrblasCher2, cublasCher2_v2, hipblasCher2)
WWR_FUNCTION_RAW(wwrblasZher2, cublasZher2_v2, hipblasZher2)
WWR_FUNCTION_RAW(wwrblasCher2_64, cublasCher2_v2_64, hipblasCher2_64)
WWR_FUNCTION_RAW(wwrblasZher2_64, cublasZher2_v2_64, hipblasZher2_64)

WWR_FUNCTION_RAW(wwrblasChpmv, cublasChpmv_v2, hipblasChpmv)
WWR_FUNCTION_RAW(wwrblasZhpmv, cublasZhpmv_v2, hipblasZhpmv)
WWR_FUNCTION_RAW(wwrblasChpmv_64, cublasChpmv_v2_64, hipblasChpmv_64)
WWR_FUNCTION_RAW(wwrblasZhpmv_64, cublasZhpmv_v2_64, hipblasZhpmv_64)

WWR_FUNCTION_RAW(wwrblasChpr, cublasChpr_v2, hipblasChpr)
WWR_FUNCTION_RAW(wwrblasZhpr, cublasZhpr_v2, hipblasZhpr)
WWR_FUNCTION_RAW(wwrblasChpr_64, cublasChpr_v2_64, hipblasChpr_64)
WWR_FUNCTION_RAW(wwrblasZhpr_64, cublasZhpr_v2_64, hipblasZhpr_64)

WWR_FUNCTION_RAW(wwrblasChpr2, cublasChpr2_v2, hipblasChpr2)
WWR_FUNCTION_RAW(wwrblasZhpr2, cublasZhpr2_v2, hipblasZhpr2)
WWR_FUNCTION_RAW(wwrblasChpr2_64, cublasChpr2_v2_64, hipblasChpr2_64)
WWR_FUNCTION_RAW(wwrblasZhpr2_64, cublasZhpr2_v2_64, hipblasZhpr2_64)

WWR_FUNCTION_RAW(wwrblasSsbmv, cublasSsbmv_v2, hipblasSsbmv)
WWR_FUNCTION_RAW(wwrblasDsbmv, cublasDsbmv_v2, hipblasDsbmv)
WWR_FUNCTION_RAW(wwrblasSsbmv_64, cublasSsbmv_v2_64, hipblasSsbmv_64)
WWR_FUNCTION_RAW(wwrblasDsbmv_64, cublasDsbmv_v2_64, hipblasDsbmv_64)

WWR_FUNCTION_RAW(wwrblasSspmv, cublasSspmv_v2, hipblasSspmv)
WWR_FUNCTION_RAW(wwrblasDspmv, cublasDspmv_v2, hipblasDspmv)
WWR_FUNCTION_RAW(wwrblasSspmv_64, cublasSspmv_v2_64, hipblasSspmv_64)
WWR_FUNCTION_RAW(wwrblasDspmv_64, cublasDspmv_v2_64, hipblasDspmv_64)

WWR_FUNCTION_RAW(wwrblasSspr, cublasSspr_v2, hipblasSspr)
WWR_FUNCTION_RAW(wwrblasDspr, cublasDspr_v2, hipblasDspr)
WWR_FUNCTION_RAW(wwrblasSspr_64, cublasSspr_v2_64, hipblasSspr_64)
WWR_FUNCTION_RAW(wwrblasDspr_64, cublasDspr_v2_64, hipblasDspr_64)

WWR_FUNCTION_RAW(wwrblasSspr2, cublasSspr2_v2, hipblasSspr2)
WWR_FUNCTION_RAW(wwrblasDspr2, cublasDspr2_v2, hipblasDspr2)
WWR_FUNCTION_RAW(wwrblasSspr2_64, cublasSspr2_v2_64, hipblasSspr2_64)
WWR_FUNCTION_RAW(wwrblasDspr2_64, cublasDspr2_v2_64, hipblasDspr2_64)

WWR_FUNCTION_RAW(wwrblasSsymv, cublasSsymv_v2, hipblasSsymv)
WWR_FUNCTION_RAW(wwrblasDsymv, cublasDsymv_v2, hipblasDsymv)
WWR_FUNCTION_RAW(wwrblasSsymv_64, cublasSsymv_v2_64, hipblasSsymv_64)
WWR_FUNCTION_RAW(wwrblasDsymv_64, cublasDsymv_v2_64, hipblasDsymv_64)

WWR_FUNCTION_RAW(wwrblasSsyr, cublasSsyr_v2, hipblasSsyr)
WWR_FUNCTION_RAW(wwrblasDsyr, cublasDsyr_v2, hipblasDsyr)
WWR_FUNCTION_RAW(wwrblasSsyr_64, cublasSsyr_v2_64, hipblasSsyr_64)
WWR_FUNCTION_RAW(wwrblasDsyr_64, cublasDsyr_v2_64, hipblasDsyr_64)

WWR_FUNCTION_RAW(wwrblasSsyr2, cublasSsyr2_v2, hipblasSsyr2)
WWR_FUNCTION_RAW(wwrblasDsyr2, cublasDsyr2_v2, hipblasDsyr2)
WWR_FUNCTION_RAW(wwrblasSsyr2_64, cublasSsyr2_v2_64, hipblasSsyr2_64)
WWR_FUNCTION_RAW(wwrblasDsyr2_64, cublasDsyr2_v2_64, hipblasDsyr2_64)

WWR_FUNCTION_RAW(wwrblasStbmv, cublasStbmv_v2, hipblasStbmv)
WWR_FUNCTION_RAW(wwrblasDtbmv, cublasDtbmv_v2, hipblasDtbmv)
WWR_FUNCTION_RAW(wwrblasCtbmv, cublasCtbmv_v2, hipblasCtbmv)
WWR_FUNCTION_RAW(wwrblasZtbmv, cublasZtbmv_v2, hipblasZtbmv)
WWR_FUNCTION_RAW(wwrblasStbmv_64, cublasStbmv_v2_64, hipblasStbmv_64)
WWR_FUNCTION_RAW(wwrblasDtbmv_64, cublasDtbmv_v2_64, hipblasDtbmv_64)
WWR_FUNCTION_RAW(wwrblasCtbmv_64, cublasCtbmv_v2_64, hipblasCtbmv_64)
WWR_FUNCTION_RAW(wwrblasZtbmv_64, cublasZtbmv_v2_64, hipblasZtbmv_64)

WWR_FUNCTION_RAW(wwrblasStbsv, cublasStbsv_v2, hipblasStbsv)
WWR_FUNCTION_RAW(wwrblasDtbsv, cublasDtbsv_v2, hipblasDtbsv)
WWR_FUNCTION_RAW(wwrblasCtbsv, cublasCtbsv_v2, hipblasCtbsv)
WWR_FUNCTION_RAW(wwrblasZtbsv, cublasZtbsv_v2, hipblasZtbsv)
WWR_FUNCTION_RAW(wwrblasStbsv_64, cublasStbsv_v2_64, hipblasStbsv_64)
WWR_FUNCTION_RAW(wwrblasDtbsv_64, cublasDtbsv_v2_64, hipblasDtbsv_64)
WWR_FUNCTION_RAW(wwrblasCtbsv_64, cublasCtbsv_v2_64, hipblasCtbsv_64)
WWR_FUNCTION_RAW(wwrblasZtbsv_64, cublasZtbsv_v2_64, hipblasZtbsv_64)

WWR_FUNCTION_RAW(wwrblasStpmv, cublasStpmv_v2, hipblasStpmv)
WWR_FUNCTION_RAW(wwrblasDtpmv, cublasDtpmv_v2, hipblasDtpmv)
WWR_FUNCTION_RAW(wwrblasCtpmv, cublasCtpmv_v2, hipblasCtpmv)
WWR_FUNCTION_RAW(wwrblasZtpmv, cublasZtpmv_v2, hipblasZtpmv)
WWR_FUNCTION_RAW(wwrblasStpmv_64, cublasStpmv_v2_64, hipblasStpmv_64)
WWR_FUNCTION_RAW(wwrblasDtpmv_64, cublasDtpmv_v2_64, hipblasDtpmv_64)
WWR_FUNCTION_RAW(wwrblasCtpmv_64, cublasCtpmv_v2_64, hipblasCtpmv_64)
WWR_FUNCTION_RAW(wwrblasZtpmv_64, cublasZtpmv_v2_64, hipblasZtpmv_64)

WWR_FUNCTION_RAW(wwrblasStpsv, cublasStpsv_v2, hipblasStpsv)
WWR_FUNCTION_RAW(wwrblasDtpsv, cublasDtpsv_v2, hipblasDtpsv)
WWR_FUNCTION_RAW(wwrblasCtpsv, cublasCtpsv_v2, hipblasCtpsv)
WWR_FUNCTION_RAW(wwrblasZtpsv, cublasZtpsv_v2, hipblasZtpsv)
WWR_FUNCTION_RAW(wwrblasStpsv_64, cublasStpsv_v2_64, hipblasStpsv_64)
WWR_FUNCTION_RAW(wwrblasDtpsv_64, cublasDtpsv_v2_64, hipblasDtpsv_64)
WWR_FUNCTION_RAW(wwrblasCtpsv_64, cublasCtpsv_v2_64, hipblasCtpsv_64)
WWR_FUNCTION_RAW(wwrblasZtpsv_64, cublasZtpsv_v2_64, hipblasZtpsv_64)

WWR_FUNCTION_RAW(wwrblasStrmv, cublasStrmv_v2, hipblasStrmv)
WWR_FUNCTION_RAW(wwrblasDtrmv, cublasDtrmv_v2, hipblasDtrmv)
WWR_FUNCTION_RAW(wwrblasCtrmv, cublasCtrmv_v2, hipblasCtrmv)
WWR_FUNCTION_RAW(wwrblasZtrmv, cublasZtrmv_v2, hipblasZtrmv)
WWR_FUNCTION_RAW(wwrblasStrmv_64, cublasStrmv_v2_64, hipblasStrmv_64)
WWR_FUNCTION_RAW(wwrblasDtrmv_64, cublasDtrmv_v2_64, hipblasDtrmv_64)
WWR_FUNCTION_RAW(wwrblasCtrmv_64, cublasCtrmv_v2_64, hipblasCtrmv_64)
WWR_FUNCTION_RAW(wwrblasZtrmv_64, cublasZtrmv_v2_64, hipblasZtrmv_64)

WWR_FUNCTION_RAW(wwrblasStrsv, cublasStrsv_v2, hipblasStrsv)
WWR_FUNCTION_RAW(wwrblasDtrsv, cublasDtrsv_v2, hipblasDtrsv)
WWR_FUNCTION_RAW(wwrblasCtrsv, cublasCtrsv_v2, hipblasCtrsv)
WWR_FUNCTION_RAW(wwrblasZtrsv, cublasZtrsv_v2, hipblasZtrsv)
WWR_FUNCTION_RAW(wwrblasStrsv_64, cublasStrsv_v2_64, hipblasStrsv_64)
WWR_FUNCTION_RAW(wwrblasDtrsv_64, cublasDtrsv_v2_64, hipblasDtrsv_64)
WWR_FUNCTION_RAW(wwrblasCtrsv_64, cublasCtrsv_v2_64, hipblasCtrsv_64)
WWR_FUNCTION_RAW(wwrblasZtrsv_64, cublasZtrsv_v2_64, hipblasZtrsv_64)

// ────────────────────────────────────────────────────────────────────────
// Level 3 (matrix-matrix)
// ────────────────────────────────────────────────────────────────────────

WWR_FUNCTION_RAW(wwrblasSgemm, cublasSgemm_v2, hipblasSgemm)
WWR_FUNCTION_RAW(wwrblasDgemm, cublasDgemm_v2, hipblasDgemm)
WWR_FUNCTION_RAW(wwrblasCgemm, cublasCgemm_v2, hipblasCgemm)
WWR_FUNCTION_RAW(wwrblasZgemm, cublasZgemm_v2, hipblasZgemm)
WWR_FUNCTION_RAW(wwrblasSgemm_64, cublasSgemm_v2_64, hipblasSgemm_64)
WWR_FUNCTION_RAW(wwrblasDgemm_64, cublasDgemm_v2_64, hipblasDgemm_64)
WWR_FUNCTION_RAW(wwrblasCgemm_64, cublasCgemm_v2_64, hipblasCgemm_64)
WWR_FUNCTION_RAW(wwrblasZgemm_64, cublasZgemm_v2_64, hipblasZgemm_64)

WWR_FUNCTION_RAW(wwrblasSgemmBatched, cublasSgemmBatched, hipblasSgemmBatched)
WWR_FUNCTION_RAW(wwrblasDgemmBatched, cublasDgemmBatched, hipblasDgemmBatched)
WWR_FUNCTION_RAW(wwrblasCgemmBatched, cublasCgemmBatched, hipblasCgemmBatched)
WWR_FUNCTION_RAW(wwrblasZgemmBatched, cublasZgemmBatched, hipblasZgemmBatched)
WWR_FUNCTION_RAW(wwrblasSgemmBatched_64, cublasSgemmBatched_64, hipblasSgemmBatched_64)
WWR_FUNCTION_RAW(wwrblasDgemmBatched_64, cublasDgemmBatched_64, hipblasDgemmBatched_64)
WWR_FUNCTION_RAW(wwrblasCgemmBatched_64, cublasCgemmBatched_64, hipblasCgemmBatched_64)
WWR_FUNCTION_RAW(wwrblasZgemmBatched_64, cublasZgemmBatched_64, hipblasZgemmBatched_64)

WWR_FUNCTION_RAW(wwrblasSgemmStridedBatched, cublasSgemmStridedBatched, hipblasSgemmStridedBatched)
WWR_FUNCTION_RAW(wwrblasDgemmStridedBatched, cublasDgemmStridedBatched, hipblasDgemmStridedBatched)
WWR_FUNCTION_RAW(wwrblasCgemmStridedBatched, cublasCgemmStridedBatched, hipblasCgemmStridedBatched)
WWR_FUNCTION_RAW(wwrblasZgemmStridedBatched, cublasZgemmStridedBatched, hipblasZgemmStridedBatched)
WWR_FUNCTION_RAW(wwrblasSgemmStridedBatched_64, cublasSgemmStridedBatched_64,
                hipblasSgemmStridedBatched_64)
WWR_FUNCTION_RAW(wwrblasDgemmStridedBatched_64, cublasDgemmStridedBatched_64,
                hipblasDgemmStridedBatched_64)
WWR_FUNCTION_RAW(wwrblasCgemmStridedBatched_64, cublasCgemmStridedBatched_64,
                hipblasCgemmStridedBatched_64)
WWR_FUNCTION_RAW(wwrblasZgemmStridedBatched_64, cublasZgemmStridedBatched_64,
                hipblasZgemmStridedBatched_64)

WWR_FUNCTION_RAW(wwrblasChemm, cublasChemm_v2, hipblasChemm)
WWR_FUNCTION_RAW(wwrblasZhemm, cublasZhemm_v2, hipblasZhemm)
WWR_FUNCTION_RAW(wwrblasChemm_64, cublasChemm_v2_64, hipblasChemm_64)
WWR_FUNCTION_RAW(wwrblasZhemm_64, cublasZhemm_v2_64, hipblasZhemm_64)

WWR_FUNCTION_RAW(wwrblasCher2k, cublasCher2k_v2, hipblasCher2k)
WWR_FUNCTION_RAW(wwrblasZher2k, cublasZher2k_v2, hipblasZher2k)
WWR_FUNCTION_RAW(wwrblasCher2k_64, cublasCher2k_v2_64, hipblasCher2k_64)
WWR_FUNCTION_RAW(wwrblasZher2k_64, cublasZher2k_v2_64, hipblasZher2k_64)

WWR_FUNCTION_RAW(wwrblasCherk, cublasCherk_v2, hipblasCherk)
WWR_FUNCTION_RAW(wwrblasZherk, cublasZherk_v2, hipblasZherk)
WWR_FUNCTION_RAW(wwrblasCherk_64, cublasCherk_v2_64, hipblasCherk_64)
WWR_FUNCTION_RAW(wwrblasZherk_64, cublasZherk_v2_64, hipblasZherk_64)

WWR_FUNCTION_RAW(wwrblasCherkx, cublasCherkx, hipblasCherkx)
WWR_FUNCTION_RAW(wwrblasZherkx, cublasZherkx, hipblasZherkx)
WWR_FUNCTION_RAW(wwrblasCherkx_64, cublasCherkx_64, hipblasCherkx_64)
WWR_FUNCTION_RAW(wwrblasZherkx_64, cublasZherkx_64, hipblasZherkx_64)

WWR_FUNCTION_RAW(wwrblasSsymm, cublasSsymm_v2, hipblasSsymm)
WWR_FUNCTION_RAW(wwrblasDsymm, cublasDsymm_v2, hipblasDsymm)
WWR_FUNCTION_RAW(wwrblasCsymm, cublasCsymm_v2, hipblasCsymm)
WWR_FUNCTION_RAW(wwrblasZsymm, cublasZsymm_v2, hipblasZsymm)
WWR_FUNCTION_RAW(wwrblasSsymm_64, cublasSsymm_v2_64, hipblasSsymm_64)
WWR_FUNCTION_RAW(wwrblasDsymm_64, cublasDsymm_v2_64, hipblasDsymm_64)
WWR_FUNCTION_RAW(wwrblasCsymm_64, cublasCsymm_v2_64, hipblasCsymm_64)
WWR_FUNCTION_RAW(wwrblasZsymm_64, cublasZsymm_v2_64, hipblasZsymm_64)

WWR_FUNCTION_RAW(wwrblasSsyr2k, cublasSsyr2k_v2, hipblasSsyr2k)
WWR_FUNCTION_RAW(wwrblasDsyr2k, cublasDsyr2k_v2, hipblasDsyr2k)
WWR_FUNCTION_RAW(wwrblasCsyr2k, cublasCsyr2k_v2, hipblasCsyr2k)
WWR_FUNCTION_RAW(wwrblasZsyr2k, cublasZsyr2k_v2, hipblasZsyr2k)
WWR_FUNCTION_RAW(wwrblasSsyr2k_64, cublasSsyr2k_v2_64, hipblasSsyr2k_64)
WWR_FUNCTION_RAW(wwrblasDsyr2k_64, cublasDsyr2k_v2_64, hipblasDsyr2k_64)
WWR_FUNCTION_RAW(wwrblasCsyr2k_64, cublasCsyr2k_v2_64, hipblasCsyr2k_64)
WWR_FUNCTION_RAW(wwrblasZsyr2k_64, cublasZsyr2k_v2_64, hipblasZsyr2k_64)

WWR_FUNCTION_RAW(wwrblasSsyrk, cublasSsyrk_v2, hipblasSsyrk)
WWR_FUNCTION_RAW(wwrblasDsyrk, cublasDsyrk_v2, hipblasDsyrk)
WWR_FUNCTION_RAW(wwrblasCsyrk, cublasCsyrk_v2, hipblasCsyrk)
WWR_FUNCTION_RAW(wwrblasZsyrk, cublasZsyrk_v2, hipblasZsyrk)
WWR_FUNCTION_RAW(wwrblasSsyrk_64, cublasSsyrk_v2_64, hipblasSsyrk_64)
WWR_FUNCTION_RAW(wwrblasDsyrk_64, cublasDsyrk_v2_64, hipblasDsyrk_64)
WWR_FUNCTION_RAW(wwrblasCsyrk_64, cublasCsyrk_v2_64, hipblasCsyrk_64)
WWR_FUNCTION_RAW(wwrblasZsyrk_64, cublasZsyrk_v2_64, hipblasZsyrk_64)

WWR_FUNCTION_RAW(wwrblasSsyrkx, cublasSsyrkx, hipblasSsyrkx)
WWR_FUNCTION_RAW(wwrblasDsyrkx, cublasDsyrkx, hipblasDsyrkx)
WWR_FUNCTION_RAW(wwrblasCsyrkx, cublasCsyrkx, hipblasCsyrkx)
WWR_FUNCTION_RAW(wwrblasZsyrkx, cublasZsyrkx, hipblasZsyrkx)
WWR_FUNCTION_RAW(wwrblasSsyrkx_64, cublasSsyrkx_64, hipblasSsyrkx_64)
WWR_FUNCTION_RAW(wwrblasDsyrkx_64, cublasDsyrkx_64, hipblasDsyrkx_64)
WWR_FUNCTION_RAW(wwrblasCsyrkx_64, cublasCsyrkx_64, hipblasCsyrkx_64)
WWR_FUNCTION_RAW(wwrblasZsyrkx_64, cublasZsyrkx_64, hipblasZsyrkx_64)

WWR_FUNCTION_RAW(wwrblasStrmm, cublasStrmm_v2, hipblasStrmm)
WWR_FUNCTION_RAW(wwrblasDtrmm, cublasDtrmm_v2, hipblasDtrmm)
WWR_FUNCTION_RAW(wwrblasCtrmm, cublasCtrmm_v2, hipblasCtrmm)
WWR_FUNCTION_RAW(wwrblasZtrmm, cublasZtrmm_v2, hipblasZtrmm)
WWR_FUNCTION_RAW(wwrblasStrmm_64, cublasStrmm_v2_64, hipblasStrmm_64)
WWR_FUNCTION_RAW(wwrblasDtrmm_64, cublasDtrmm_v2_64, hipblasDtrmm_64)
WWR_FUNCTION_RAW(wwrblasCtrmm_64, cublasCtrmm_v2_64, hipblasCtrmm_64)
WWR_FUNCTION_RAW(wwrblasZtrmm_64, cublasZtrmm_v2_64, hipblasZtrmm_64)

WWR_FUNCTION_RAW(wwrblasStrsm, cublasStrsm_v2, hipblasStrsm)
WWR_FUNCTION_RAW(wwrblasDtrsm, cublasDtrsm_v2, hipblasDtrsm)
WWR_FUNCTION_RAW(wwrblasCtrsm, cublasCtrsm_v2, hipblasCtrsm)
WWR_FUNCTION_RAW(wwrblasZtrsm, cublasZtrsm_v2, hipblasZtrsm)
WWR_FUNCTION_RAW(wwrblasStrsm_64, cublasStrsm_v2_64, hipblasStrsm_64)
WWR_FUNCTION_RAW(wwrblasDtrsm_64, cublasDtrsm_v2_64, hipblasDtrsm_64)
WWR_FUNCTION_RAW(wwrblasCtrsm_64, cublasCtrsm_v2_64, hipblasCtrsm_64)
WWR_FUNCTION_RAW(wwrblasZtrsm_64, cublasZtrsm_v2_64, hipblasZtrsm_64)

WWR_FUNCTION_RAW(wwrblasStrsmBatched, cublasStrsmBatched, hipblasStrsmBatched)
WWR_FUNCTION_RAW(wwrblasDtrsmBatched, cublasDtrsmBatched, hipblasDtrsmBatched)
WWR_FUNCTION_RAW(wwrblasCtrsmBatched, cublasCtrsmBatched, hipblasCtrsmBatched)
WWR_FUNCTION_RAW(wwrblasZtrsmBatched, cublasZtrsmBatched, hipblasZtrsmBatched)
WWR_FUNCTION_RAW(wwrblasStrsmBatched_64, cublasStrsmBatched_64, hipblasStrsmBatched_64)
WWR_FUNCTION_RAW(wwrblasDtrsmBatched_64, cublasDtrsmBatched_64, hipblasDtrsmBatched_64)
WWR_FUNCTION_RAW(wwrblasCtrsmBatched_64, cublasCtrsmBatched_64, hipblasCtrsmBatched_64)
WWR_FUNCTION_RAW(wwrblasZtrsmBatched_64, cublasZtrsmBatched_64, hipblasZtrsmBatched_64)

// ────────────────────────────────────────────────────────────────────────
// BLAS-like extensions
// ────────────────────────────────────────────────────────────────────────

WWR_FUNCTION_RAW(wwrblasSdgmm, cublasSdgmm, hipblasSdgmm)
WWR_FUNCTION_RAW(wwrblasDdgmm, cublasDdgmm, hipblasDdgmm)
WWR_FUNCTION_RAW(wwrblasCdgmm, cublasCdgmm, hipblasCdgmm)
WWR_FUNCTION_RAW(wwrblasZdgmm, cublasZdgmm, hipblasZdgmm)
WWR_FUNCTION_RAW(wwrblasSdgmm_64, cublasSdgmm_64, hipblasSdgmm_64)
WWR_FUNCTION_RAW(wwrblasDdgmm_64, cublasDdgmm_64, hipblasDdgmm_64)
WWR_FUNCTION_RAW(wwrblasCdgmm_64, cublasCdgmm_64, hipblasCdgmm_64)
WWR_FUNCTION_RAW(wwrblasZdgmm_64, cublasZdgmm_64, hipblasZdgmm_64)

WWR_FUNCTION_RAW(wwrblasSgeam, cublasSgeam, hipblasSgeam)
WWR_FUNCTION_RAW(wwrblasDgeam, cublasDgeam, hipblasDgeam)
WWR_FUNCTION_RAW(wwrblasCgeam, cublasCgeam, hipblasCgeam)
WWR_FUNCTION_RAW(wwrblasZgeam, cublasZgeam, hipblasZgeam)
WWR_FUNCTION_RAW(wwrblasSgeam_64, cublasSgeam_64, hipblasSgeam_64)
WWR_FUNCTION_RAW(wwrblasDgeam_64, cublasDgeam_64, hipblasDgeam_64)
WWR_FUNCTION_RAW(wwrblasCgeam_64, cublasCgeam_64, hipblasCgeam_64)
WWR_FUNCTION_RAW(wwrblasZgeam_64, cublasZgeam_64, hipblasZgeam_64)

WWR_FUNCTION_RAW(wwrblasSgelsBatched, cublasSgelsBatched, hipblasSgelsBatched)
WWR_FUNCTION_RAW(wwrblasDgelsBatched, cublasDgelsBatched, hipblasDgelsBatched)
WWR_FUNCTION_RAW(wwrblasCgelsBatched, cublasCgelsBatched, hipblasCgelsBatched)
WWR_FUNCTION_RAW(wwrblasZgelsBatched, cublasZgelsBatched, hipblasZgelsBatched)

WWR_FUNCTION_RAW(wwrblasSgeqrfBatched, cublasSgeqrfBatched, hipblasSgeqrfBatched)
WWR_FUNCTION_RAW(wwrblasDgeqrfBatched, cublasDgeqrfBatched, hipblasDgeqrfBatched)
WWR_FUNCTION_RAW(wwrblasCgeqrfBatched, cublasCgeqrfBatched, hipblasCgeqrfBatched)
WWR_FUNCTION_RAW(wwrblasZgeqrfBatched, cublasZgeqrfBatched, hipblasZgeqrfBatched)

WWR_FUNCTION_RAW(wwrblasSgetrfBatched, cublasSgetrfBatched, hipblasSgetrfBatched)
WWR_FUNCTION_RAW(wwrblasDgetrfBatched, cublasDgetrfBatched, hipblasDgetrfBatched)
WWR_FUNCTION_RAW(wwrblasCgetrfBatched, cublasCgetrfBatched, hipblasCgetrfBatched)
WWR_FUNCTION_RAW(wwrblasZgetrfBatched, cublasZgetrfBatched, hipblasZgetrfBatched)

// ────────────────────────────────────────────────────────────────────────
// getrsBatched / getriBatched: const-correct on both backends
// ────────────────────────────────────────────────────────────────────────

#if defined(WWR_SELECTED_CUDA)

WWR_FUNCTION_RAW(wwrblasSgetrsBatched, cublasSgetrsBatched, hipblasSgetrsBatched)
WWR_FUNCTION_RAW(wwrblasDgetrsBatched, cublasDgetrsBatched, hipblasDgetrsBatched)
WWR_FUNCTION_RAW(wwrblasCgetrsBatched, cublasCgetrsBatched, hipblasCgetrsBatched)
WWR_FUNCTION_RAW(wwrblasZgetrsBatched, cublasZgetrsBatched, hipblasZgetrsBatched)

WWR_FUNCTION_RAW(wwrblasSgetriBatched, cublasSgetriBatched, hipblasSgetriBatched)
WWR_FUNCTION_RAW(wwrblasDgetriBatched, cublasDgetriBatched, hipblasDgetriBatched)
WWR_FUNCTION_RAW(wwrblasCgetriBatched, cublasCgetriBatched, hipblasCgetriBatched)
WWR_FUNCTION_RAW(wwrblasZgetriBatched, cublasZgetriBatched, hipblasZgetriBatched)

#else

// hipBLAS declares Aarray as T* const[] (and getriBatched's P as int*) where
// cuBLAS declares const T* const[] (and const int*). hipBLAS only reads them;
// these keep cuBLAS's signature so src/wrappers has one const-correct API.

inline wwrblasStatus_t wwrblasSgetrsBatched(wwrblasHandle_t handle, wwrblasOperation_t trans, int n,
                                            int nrhs, const float *const Aarray[], int lda,
                                            const int *devIpiv, float *const Barray[], int ldb,
                                            int *info, int batchSize) {
  return ::hipblasSgetrsBatched(handle, trans, n, nrhs, const_cast<float *const *>(Aarray), lda,
                                   devIpiv, Barray, ldb, info, batchSize);
}
inline wwrblasStatus_t wwrblasDgetrsBatched(wwrblasHandle_t handle, wwrblasOperation_t trans, int n,
                                            int nrhs, const double *const Aarray[], int lda,
                                            const int *devIpiv, double *const Barray[], int ldb,
                                            int *info, int batchSize) {
  return ::hipblasDgetrsBatched(handle, trans, n, nrhs, const_cast<double *const *>(Aarray), lda,
                                   devIpiv, Barray, ldb, info, batchSize);
}
inline wwrblasStatus_t wwrblasCgetrsBatched(wwrblasHandle_t handle, wwrblasOperation_t trans, int n,
                                            int nrhs, const wwrFloatComplex *const Aarray[],
                                            int lda, const int *devIpiv,
                                            wwrFloatComplex *const Barray[], int ldb, int *info,
                                            int batchSize) {
  return ::hipblasCgetrsBatched(handle, trans, n, nrhs,
                                   const_cast<wwrFloatComplex *const *>(Aarray), lda, devIpiv,
                                   Barray, ldb, info, batchSize);
}
inline wwrblasStatus_t wwrblasZgetrsBatched(wwrblasHandle_t handle, wwrblasOperation_t trans, int n,
                                            int nrhs, const wwrDoubleComplex *const Aarray[],
                                            int lda, const int *devIpiv,
                                            wwrDoubleComplex *const Barray[], int ldb, int *info,
                                            int batchSize) {
  return ::hipblasZgetrsBatched(handle, trans, n, nrhs,
                                   const_cast<wwrDoubleComplex *const *>(Aarray), lda, devIpiv,
                                   Barray, ldb, info, batchSize);
}

inline wwrblasStatus_t wwrblasSgetriBatched(wwrblasHandle_t handle, int n, const float *const A[],
                                            int lda, const int *P, float *const C[], int ldc,
                                            int *info, int batchSize) {
  return ::hipblasSgetriBatched(handle, n, const_cast<float *const *>(A), lda,
                                   const_cast<int *>(P), C, ldc, info, batchSize);
}
inline wwrblasStatus_t wwrblasDgetriBatched(wwrblasHandle_t handle, int n, const double *const A[],
                                            int lda, const int *P, double *const C[], int ldc,
                                            int *info, int batchSize) {
  return ::hipblasDgetriBatched(handle, n, const_cast<double *const *>(A), lda,
                                   const_cast<int *>(P), C, ldc, info, batchSize);
}
inline wwrblasStatus_t wwrblasCgetriBatched(wwrblasHandle_t handle, int n,
                                            const wwrFloatComplex *const A[], int lda, const int *P,
                                            wwrFloatComplex *const C[], int ldc, int *info,
                                            int batchSize) {
  return ::hipblasCgetriBatched(handle, n, const_cast<wwrFloatComplex *const *>(A), lda,
                                   const_cast<int *>(P), C, ldc, info, batchSize);
}
inline wwrblasStatus_t wwrblasZgetriBatched(wwrblasHandle_t handle, int n,
                                            const wwrDoubleComplex *const A[], int lda,
                                            const int *P, wwrDoubleComplex *const C[], int ldc,
                                            int *info, int batchSize) {
  return ::hipblasZgetriBatched(handle, n, const_cast<wwrDoubleComplex *const *>(A), lda,
                                   const_cast<int *>(P), C, ldc, info, batchSize);
}

#endif

// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)
