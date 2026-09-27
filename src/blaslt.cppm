/**
 * @file blaslt.cppm
 * @brief Backend-neutral cuBLASLt / hipBLASLt: gpublasLt* names
 *
 * gpublasLt<X> stands for cublasLt<X> on a CUDA build and hipblasLt<X> on a HIP
 * build -- the modern "Lt" GEMM surface (epilogue-fused / mixed-precision /
 * narrow-float matmul), the counterpart to wwr.blas's classic API. Each name is
 * written out in full, one line per name, as wwr.blas does. See gpu_backend.h.
 *
 * The surface is exactly the intersection both backends spell the same, as
 * reported by `devtools/header_intersection.py --cuda cublasLt.h --hip
 * hipblaslt.h` (pinned prefixes cublas/hipblas for the functions and types,
 * cublaslt/hipblaslt for the enum constants): the handle, the matmul / matrix
 * layout / preference / matrix-transform descriptors with their attribute
 * get/set, the algo-heuristic search, gpublasLtMatmul and gpublasLtMatrixTransform
 * themselves, and the epilogue / order / pointer-mode / matrix-scale enums.
 *
 * Two shared names inherited from classic cuBLAS/hipBLAS are renamed under this
 * module's prefix, as gpu.solver renames cudaDataType to gpusolverDataType_t:
 * gpublasLtStatus_t is cublasStatus_t / hipblasStatus_t (cuBLASLt has no Lt
 * status type of its own; the STATUS_SUCCESS constants live in wwr.blas), and
 * gpublasLtComputeType_t is cublasComputeType_t / hipblasComputeType_t.
 *
 * Deliberately left out, reach through wwr.cuda.cublasLt / wwr.hip.hipblaslt:
 *
 * - gpublasLtGetVersion: the name is shared but the signatures diverge
 *   irreconcilably -- cublasLtGetVersion() takes no argument and returns size_t,
 *   hipblasLtGetVersion(handle, int*) returns a status.
 * - Everything only one backend exposes: cuBLASLt's algo introspection
 *   (cublasLtMatmulAlgoInit/Check/CapGetAttribute/ConfigSetAttribute), its logger
 *   and heuristics-cache controls, and the large tile / stages / reduction-scheme
 *   enums; hipBLASLt's hipblasLtGetArchName / hipblasLtGetGitRevision and its
 *   *_EXT epilogue and descriptor attributes.
 *
 * Usage:
 *   import wwr.blaslt;
 *
 *   gpublasLtHandle_t handle;
 *   gpublasLtCreate(&handle);
 */

module;

#include "gpu_backend.h"

export module wwr.blaslt;

#if defined(WWR_GPU_BACKEND_CUDA)
import wwr.cuda.cublasLt;
#else
import wwr.hip.hipblaslt;
#endif

export namespace wwr {

// ========================================================================
// Types
// ========================================================================

WWR_TYPE(gpublasLtStatus_t, cublasStatus_t, hipblasStatus_t)
WWR_TYPE(gpublasLtComputeType_t, cublasComputeType_t, hipblasComputeType_t)

WWR_TYPE(gpublasLtHandle_t, cublasLtHandle_t, hipblasLtHandle_t)

WWR_TYPE(gpublasLtMatmulDesc_t, cublasLtMatmulDesc_t, hipblasLtMatmulDesc_t)
WWR_TYPE(gpublasLtMatmulDescOpaque_t, cublasLtMatmulDescOpaque_t, hipblasLtMatmulDescOpaque_t)
WWR_TYPE(gpublasLtMatmulDescAttributes_t, cublasLtMatmulDescAttributes_t,
            hipblasLtMatmulDescAttributes_t)

WWR_TYPE(gpublasLtMatrixLayout_t, cublasLtMatrixLayout_t, hipblasLtMatrixLayout_t)
WWR_TYPE(gpublasLtMatrixLayoutOpaque_t, cublasLtMatrixLayoutOpaque_t,
            hipblasLtMatrixLayoutOpaque_t)
WWR_TYPE(gpublasLtMatrixLayoutAttribute_t, cublasLtMatrixLayoutAttribute_t,
            hipblasLtMatrixLayoutAttribute_t)

WWR_TYPE(gpublasLtMatmulPreference_t, cublasLtMatmulPreference_t, hipblasLtMatmulPreference_t)
WWR_TYPE(gpublasLtMatmulPreferenceOpaque_t, cublasLtMatmulPreferenceOpaque_t,
            hipblasLtMatmulPreferenceOpaque_t)
WWR_TYPE(gpublasLtMatmulPreferenceAttributes_t, cublasLtMatmulPreferenceAttributes_t,
            hipblasLtMatmulPreferenceAttributes_t)

WWR_TYPE(gpublasLtMatmulAlgo_t, cublasLtMatmulAlgo_t, hipblasLtMatmulAlgo_t)
WWR_TYPE(gpublasLtMatmulHeuristicResult_t, cublasLtMatmulHeuristicResult_t,
            hipblasLtMatmulHeuristicResult_t)

WWR_TYPE(gpublasLtMatrixTransformDesc_t, cublasLtMatrixTransformDesc_t,
            hipblasLtMatrixTransformDesc_t)
WWR_TYPE(gpublasLtMatrixTransformDescOpaque_t, cublasLtMatrixTransformDescOpaque_t,
            hipblasLtMatrixTransformDescOpaque_t)
WWR_TYPE(gpublasLtMatrixTransformDescAttributes_t, cublasLtMatrixTransformDescAttributes_t,
            hipblasLtMatrixTransformDescAttributes_t)

WWR_TYPE(gpublasLtEpilogue_t, cublasLtEpilogue_t, hipblasLtEpilogue_t)
WWR_TYPE(gpublasLtMatmulMatrixScale_t, cublasLtMatmulMatrixScale_t, hipblasLtMatmulMatrixScale_t)
WWR_TYPE(gpublasLtOrder_t, cublasLtOrder_t, hipblasLtOrder_t)
WWR_TYPE(gpublasLtPointerMode_t, cublasLtPointerMode_t, hipblasLtPointerMode_t)

// ========================================================================
// Constants: cublasLtEpilogue_t
// ========================================================================

WWR_VALUE(GPUBLASLT_EPILOGUE_DEFAULT, CUBLASLT_EPILOGUE_DEFAULT, HIPBLASLT_EPILOGUE_DEFAULT)
WWR_VALUE(GPUBLASLT_EPILOGUE_RELU, CUBLASLT_EPILOGUE_RELU, HIPBLASLT_EPILOGUE_RELU)
WWR_VALUE(GPUBLASLT_EPILOGUE_RELU_AUX, CUBLASLT_EPILOGUE_RELU_AUX, HIPBLASLT_EPILOGUE_RELU_AUX)
WWR_VALUE(GPUBLASLT_EPILOGUE_BIAS, CUBLASLT_EPILOGUE_BIAS, HIPBLASLT_EPILOGUE_BIAS)
WWR_VALUE(GPUBLASLT_EPILOGUE_RELU_BIAS, CUBLASLT_EPILOGUE_RELU_BIAS, HIPBLASLT_EPILOGUE_RELU_BIAS)
WWR_VALUE(GPUBLASLT_EPILOGUE_RELU_AUX_BIAS, CUBLASLT_EPILOGUE_RELU_AUX_BIAS,
             HIPBLASLT_EPILOGUE_RELU_AUX_BIAS)
WWR_VALUE(GPUBLASLT_EPILOGUE_DGELU, CUBLASLT_EPILOGUE_DGELU, HIPBLASLT_EPILOGUE_DGELU)
WWR_VALUE(GPUBLASLT_EPILOGUE_DGELU_BGRAD, CUBLASLT_EPILOGUE_DGELU_BGRAD,
             HIPBLASLT_EPILOGUE_DGELU_BGRAD)
WWR_VALUE(GPUBLASLT_EPILOGUE_GELU, CUBLASLT_EPILOGUE_GELU, HIPBLASLT_EPILOGUE_GELU)
WWR_VALUE(GPUBLASLT_EPILOGUE_GELU_AUX, CUBLASLT_EPILOGUE_GELU_AUX, HIPBLASLT_EPILOGUE_GELU_AUX)
WWR_VALUE(GPUBLASLT_EPILOGUE_GELU_BIAS, CUBLASLT_EPILOGUE_GELU_BIAS, HIPBLASLT_EPILOGUE_GELU_BIAS)
WWR_VALUE(GPUBLASLT_EPILOGUE_GELU_AUX_BIAS, CUBLASLT_EPILOGUE_GELU_AUX_BIAS,
             HIPBLASLT_EPILOGUE_GELU_AUX_BIAS)
WWR_VALUE(GPUBLASLT_EPILOGUE_BGRADA, CUBLASLT_EPILOGUE_BGRADA, HIPBLASLT_EPILOGUE_BGRADA)
WWR_VALUE(GPUBLASLT_EPILOGUE_BGRADB, CUBLASLT_EPILOGUE_BGRADB, HIPBLASLT_EPILOGUE_BGRADB)

// ========================================================================
// Constants: cublasLtMatmulDescAttributes_t
// ========================================================================

WWR_VALUE(GPUBLASLT_MATMUL_DESC_TRANSA, CUBLASLT_MATMUL_DESC_TRANSA, HIPBLASLT_MATMUL_DESC_TRANSA)
WWR_VALUE(GPUBLASLT_MATMUL_DESC_TRANSB, CUBLASLT_MATMUL_DESC_TRANSB, HIPBLASLT_MATMUL_DESC_TRANSB)
WWR_VALUE(GPUBLASLT_MATMUL_DESC_EPILOGUE, CUBLASLT_MATMUL_DESC_EPILOGUE,
             HIPBLASLT_MATMUL_DESC_EPILOGUE)
WWR_VALUE(GPUBLASLT_MATMUL_DESC_BIAS_POINTER, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
             HIPBLASLT_MATMUL_DESC_BIAS_POINTER)
WWR_VALUE(GPUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
             HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE)
WWR_VALUE(GPUBLASLT_MATMUL_DESC_A_SCALE_POINTER, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
             HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER)
WWR_VALUE(GPUBLASLT_MATMUL_DESC_B_SCALE_POINTER, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
             HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER)
WWR_VALUE(GPUBLASLT_MATMUL_DESC_C_SCALE_POINTER, CUBLASLT_MATMUL_DESC_C_SCALE_POINTER,
             HIPBLASLT_MATMUL_DESC_C_SCALE_POINTER)
WWR_VALUE(GPUBLASLT_MATMUL_DESC_D_SCALE_POINTER, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER,
             HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER)
WWR_VALUE(GPUBLASLT_MATMUL_DESC_A_SCALE_MODE, CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
             HIPBLASLT_MATMUL_DESC_A_SCALE_MODE)
WWR_VALUE(GPUBLASLT_MATMUL_DESC_B_SCALE_MODE, CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
             HIPBLASLT_MATMUL_DESC_B_SCALE_MODE)
WWR_VALUE(GPUBLASLT_MATMUL_DESC_AMAX_D_POINTER, CUBLASLT_MATMUL_DESC_AMAX_D_POINTER,
             HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER)
WWR_VALUE(GPUBLASLT_MATMUL_DESC_POINTER_MODE, CUBLASLT_MATMUL_DESC_POINTER_MODE,
             HIPBLASLT_MATMUL_DESC_POINTER_MODE)
WWR_VALUE(GPUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
             HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER)
WWR_VALUE(GPUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,
             HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD)
WWR_VALUE(GPUBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE,
             CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE,
             HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE)
WWR_VALUE(GPUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE,
             CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE,
             HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE)
WWR_VALUE(GPUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER,
             CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER,
             HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER)

// ========================================================================
// Constants: cublasLtMatmulMatrixScale_t
// ========================================================================

WWR_VALUE(GPUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F, CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F,
             HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F)
WWR_VALUE(GPUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3, CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3,
             HIPBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3)
WWR_VALUE(GPUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0, CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0,
             HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0)
WWR_VALUE(GPUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F, CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F,
             HIPBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F)
WWR_VALUE(GPUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F,
             CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F,
             HIPBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F)
WWR_VALUE(GPUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F, CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F,
             HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F)
WWR_VALUE(GPUBLASLT_MATMUL_MATRIX_SCALE_END, CUBLASLT_MATMUL_MATRIX_SCALE_END,
             HIPBLASLT_MATMUL_MATRIX_SCALE_END)

// ========================================================================
// Constants: cublasLtMatmulPreferenceAttributes_t
// ========================================================================

WWR_VALUE(GPUBLASLT_MATMUL_PREF_SEARCH_MODE, CUBLASLT_MATMUL_PREF_SEARCH_MODE,
             HIPBLASLT_MATMUL_PREF_SEARCH_MODE)
WWR_VALUE(GPUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
             HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES)

// ========================================================================
// Constants: cublasLtMatrixLayoutAttribute_t
// ========================================================================

WWR_VALUE(GPUBLASLT_MATRIX_LAYOUT_TYPE, CUBLASLT_MATRIX_LAYOUT_TYPE, HIPBLASLT_MATRIX_LAYOUT_TYPE)
WWR_VALUE(GPUBLASLT_MATRIX_LAYOUT_ORDER, CUBLASLT_MATRIX_LAYOUT_ORDER,
             HIPBLASLT_MATRIX_LAYOUT_ORDER)
WWR_VALUE(GPUBLASLT_MATRIX_LAYOUT_ROWS, CUBLASLT_MATRIX_LAYOUT_ROWS, HIPBLASLT_MATRIX_LAYOUT_ROWS)
WWR_VALUE(GPUBLASLT_MATRIX_LAYOUT_COLS, CUBLASLT_MATRIX_LAYOUT_COLS, HIPBLASLT_MATRIX_LAYOUT_COLS)
WWR_VALUE(GPUBLASLT_MATRIX_LAYOUT_LD, CUBLASLT_MATRIX_LAYOUT_LD, HIPBLASLT_MATRIX_LAYOUT_LD)
WWR_VALUE(GPUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
             HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT)
WWR_VALUE(GPUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
             CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
             HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET)

// ========================================================================
// Constants: cublasLtMatrixTransformDescAttributes_t
// ========================================================================

WWR_VALUE(GPUBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE, CUBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE,
             HIPBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE)
WWR_VALUE(GPUBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE,
             CUBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE,
             HIPBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE)
WWR_VALUE(GPUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA,
             HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSA)
WWR_VALUE(GPUBLASLT_MATRIX_TRANSFORM_DESC_TRANSB, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSB,
             HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSB)

// ========================================================================
// Constants: cublasLtOrder_t
// ========================================================================

WWR_VALUE(GPUBLASLT_ORDER_COL, CUBLASLT_ORDER_COL, HIPBLASLT_ORDER_COL)
WWR_VALUE(GPUBLASLT_ORDER_ROW, CUBLASLT_ORDER_ROW, HIPBLASLT_ORDER_ROW)

// ========================================================================
// Constants: cublasLtPointerMode_t
// ========================================================================

WWR_VALUE(GPUBLASLT_POINTER_MODE_HOST, CUBLASLT_POINTER_MODE_HOST, HIPBLASLT_POINTER_MODE_HOST)
WWR_VALUE(GPUBLASLT_POINTER_MODE_DEVICE, CUBLASLT_POINTER_MODE_DEVICE,
             HIPBLASLT_POINTER_MODE_DEVICE)
WWR_VALUE(GPUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST,
             CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST,
             HIPBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST)

// ========================================================================
// Handle management
// ========================================================================

WWR_FUNCTION(gpublasLtCreate, cublasLtCreate, hipblasLtCreate)
WWR_FUNCTION(gpublasLtDestroy, cublasLtDestroy, hipblasLtDestroy)

// ========================================================================
// Matmul descriptor
// ========================================================================

WWR_FUNCTION(gpublasLtMatmulDescCreate, cublasLtMatmulDescCreate, hipblasLtMatmulDescCreate)
WWR_FUNCTION(gpublasLtMatmulDescDestroy, cublasLtMatmulDescDestroy, hipblasLtMatmulDescDestroy)
WWR_FUNCTION(gpublasLtMatmulDescSetAttribute, cublasLtMatmulDescSetAttribute,
                hipblasLtMatmulDescSetAttribute)
WWR_FUNCTION(gpublasLtMatmulDescGetAttribute, cublasLtMatmulDescGetAttribute,
                hipblasLtMatmulDescGetAttribute)

// ========================================================================
// Matrix layout descriptor
// ========================================================================

WWR_FUNCTION(gpublasLtMatrixLayoutCreate, cublasLtMatrixLayoutCreate, hipblasLtMatrixLayoutCreate)
WWR_FUNCTION(gpublasLtMatrixLayoutDestroy, cublasLtMatrixLayoutDestroy,
                hipblasLtMatrixLayoutDestroy)
WWR_FUNCTION(gpublasLtMatrixLayoutSetAttribute, cublasLtMatrixLayoutSetAttribute,
                hipblasLtMatrixLayoutSetAttribute)
WWR_FUNCTION(gpublasLtMatrixLayoutGetAttribute, cublasLtMatrixLayoutGetAttribute,
                hipblasLtMatrixLayoutGetAttribute)

// ========================================================================
// Matmul preference descriptor
// ========================================================================

WWR_FUNCTION(gpublasLtMatmulPreferenceCreate, cublasLtMatmulPreferenceCreate,
                hipblasLtMatmulPreferenceCreate)
WWR_FUNCTION(gpublasLtMatmulPreferenceDestroy, cublasLtMatmulPreferenceDestroy,
                hipblasLtMatmulPreferenceDestroy)
WWR_FUNCTION(gpublasLtMatmulPreferenceSetAttribute, cublasLtMatmulPreferenceSetAttribute,
                hipblasLtMatmulPreferenceSetAttribute)
WWR_FUNCTION(gpublasLtMatmulPreferenceGetAttribute, cublasLtMatmulPreferenceGetAttribute,
                hipblasLtMatmulPreferenceGetAttribute)

// ========================================================================
// Heuristic search and matmul execution
// ========================================================================

WWR_FUNCTION(gpublasLtMatmulAlgoGetHeuristic, cublasLtMatmulAlgoGetHeuristic,
                hipblasLtMatmulAlgoGetHeuristic)
WWR_FUNCTION(gpublasLtMatmul, cublasLtMatmul, hipblasLtMatmul)

// ========================================================================
// Matrix transform descriptor and execution
// ========================================================================

WWR_FUNCTION(gpublasLtMatrixTransformDescCreate, cublasLtMatrixTransformDescCreate,
                hipblasLtMatrixTransformDescCreate)
WWR_FUNCTION(gpublasLtMatrixTransformDescDestroy, cublasLtMatrixTransformDescDestroy,
                hipblasLtMatrixTransformDescDestroy)
WWR_FUNCTION(gpublasLtMatrixTransformDescSetAttribute, cublasLtMatrixTransformDescSetAttribute,
                hipblasLtMatrixTransformDescSetAttribute)
WWR_FUNCTION(gpublasLtMatrixTransformDescGetAttribute, cublasLtMatrixTransformDescGetAttribute,
                hipblasLtMatrixTransformDescGetAttribute)
WWR_FUNCTION(gpublasLtMatrixTransform, cublasLtMatrixTransform, hipblasLtMatrixTransform)

} // namespace wwr
