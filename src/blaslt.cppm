/**
 * @file blaslt.cppm
 * @brief Backend-neutral cuBLASLt / hipBLASLt: wwrblasLt* names
 *
 * wwrblasLt<X> stands for cublasLt<X> on a CUDA build and hipblasLt<X> on a HIP
 * build -- the modern "Lt" GEMM surface (epilogue-fused / mixed-precision /
 * narrow-float matmul), the counterpart to wwr.blas's classic API. Each name is
 * written out in full, one line per name, as wwr.blas does. See backend.h.
 *
 * The surface is exactly the intersection both backends spell the same, as
 * reported by `devtools/header_intersection.py --cuda cublasLt.h --hip
 * hipblaslt.h` (pinned prefixes cublas/hipblas for the functions and types,
 * cublaslt/hipblaslt for the enum constants): the handle, the matmul / matrix
 * layout / preference / matrix-transform descriptors with their attribute
 * get/set, the algo-heuristic search, wwrblasLtMatmul and wwrblasLtMatrixTransform
 * themselves, and the epilogue / order / pointer-mode / matrix-scale enums.
 *
 * Two shared names inherited from classic cuBLAS/hipBLAS are renamed under this
 * module's prefix, as gpu.solver renames cudaDataType to wwrsolverDataType_t:
 * wwrblasLtStatus_t is cublasStatus_t / hipblasStatus_t (cuBLASLt has no Lt
 * status type of its own; the STATUS_SUCCESS constants live in wwr.blas), and
 * wwrblasLtComputeType_t is cublasComputeType_t / hipblasComputeType_t.
 *
 * Deliberately left out, reach through wwr.cuda.cublasLt / wwr.hip.hipblaslt:
 *
 * - wwrblasLtGetVersion: the name is shared but the signatures diverge
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
 *   wwrblasLtHandle_t handle;
 *   wwrblasLtCreate(&handle);
 */

module;

#include "backend.h"

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

WWR_TYPE(wwrblasLtStatus_t, cublasStatus_t, hipblasStatus_t)
WWR_TYPE(wwrblasLtComputeType_t, cublasComputeType_t, hipblasComputeType_t)

WWR_TYPE(wwrblasLtHandle_t, cublasLtHandle_t, hipblasLtHandle_t)

WWR_TYPE(wwrblasLtMatmulDesc_t, cublasLtMatmulDesc_t, hipblasLtMatmulDesc_t)
WWR_TYPE(wwrblasLtMatmulDescOpaque_t, cublasLtMatmulDescOpaque_t, hipblasLtMatmulDescOpaque_t)
WWR_TYPE(wwrblasLtMatmulDescAttributes_t, cublasLtMatmulDescAttributes_t,
            hipblasLtMatmulDescAttributes_t)

WWR_TYPE(wwrblasLtMatrixLayout_t, cublasLtMatrixLayout_t, hipblasLtMatrixLayout_t)
WWR_TYPE(wwrblasLtMatrixLayoutOpaque_t, cublasLtMatrixLayoutOpaque_t,
            hipblasLtMatrixLayoutOpaque_t)
WWR_TYPE(wwrblasLtMatrixLayoutAttribute_t, cublasLtMatrixLayoutAttribute_t,
            hipblasLtMatrixLayoutAttribute_t)

WWR_TYPE(wwrblasLtMatmulPreference_t, cublasLtMatmulPreference_t, hipblasLtMatmulPreference_t)
WWR_TYPE(wwrblasLtMatmulPreferenceOpaque_t, cublasLtMatmulPreferenceOpaque_t,
            hipblasLtMatmulPreferenceOpaque_t)
WWR_TYPE(wwrblasLtMatmulPreferenceAttributes_t, cublasLtMatmulPreferenceAttributes_t,
            hipblasLtMatmulPreferenceAttributes_t)

WWR_TYPE(wwrblasLtMatmulAlgo_t, cublasLtMatmulAlgo_t, hipblasLtMatmulAlgo_t)
WWR_TYPE(wwrblasLtMatmulHeuristicResult_t, cublasLtMatmulHeuristicResult_t,
            hipblasLtMatmulHeuristicResult_t)

WWR_TYPE(wwrblasLtMatrixTransformDesc_t, cublasLtMatrixTransformDesc_t,
            hipblasLtMatrixTransformDesc_t)
WWR_TYPE(wwrblasLtMatrixTransformDescOpaque_t, cublasLtMatrixTransformDescOpaque_t,
            hipblasLtMatrixTransformDescOpaque_t)
WWR_TYPE(wwrblasLtMatrixTransformDescAttributes_t, cublasLtMatrixTransformDescAttributes_t,
            hipblasLtMatrixTransformDescAttributes_t)

WWR_TYPE(wwrblasLtEpilogue_t, cublasLtEpilogue_t, hipblasLtEpilogue_t)
WWR_TYPE(wwrblasLtMatmulMatrixScale_t, cublasLtMatmulMatrixScale_t, hipblasLtMatmulMatrixScale_t)
WWR_TYPE(wwrblasLtOrder_t, cublasLtOrder_t, hipblasLtOrder_t)
WWR_TYPE(wwrblasLtPointerMode_t, cublasLtPointerMode_t, hipblasLtPointerMode_t)

// ========================================================================
// Constants: cublasLtEpilogue_t
// ========================================================================

WWR_VALUE(WWRBLASLT_EPILOGUE_DEFAULT, CUBLASLT_EPILOGUE_DEFAULT, HIPBLASLT_EPILOGUE_DEFAULT)
WWR_VALUE(WWRBLASLT_EPILOGUE_RELU, CUBLASLT_EPILOGUE_RELU, HIPBLASLT_EPILOGUE_RELU)
WWR_VALUE(WWRBLASLT_EPILOGUE_RELU_AUX, CUBLASLT_EPILOGUE_RELU_AUX, HIPBLASLT_EPILOGUE_RELU_AUX)
WWR_VALUE(WWRBLASLT_EPILOGUE_BIAS, CUBLASLT_EPILOGUE_BIAS, HIPBLASLT_EPILOGUE_BIAS)
WWR_VALUE(WWRBLASLT_EPILOGUE_RELU_BIAS, CUBLASLT_EPILOGUE_RELU_BIAS, HIPBLASLT_EPILOGUE_RELU_BIAS)
WWR_VALUE(WWRBLASLT_EPILOGUE_RELU_AUX_BIAS, CUBLASLT_EPILOGUE_RELU_AUX_BIAS,
             HIPBLASLT_EPILOGUE_RELU_AUX_BIAS)
WWR_VALUE(WWRBLASLT_EPILOGUE_DGELU, CUBLASLT_EPILOGUE_DGELU, HIPBLASLT_EPILOGUE_DGELU)
WWR_VALUE(WWRBLASLT_EPILOGUE_DGELU_BGRAD, CUBLASLT_EPILOGUE_DGELU_BGRAD,
             HIPBLASLT_EPILOGUE_DGELU_BGRAD)
WWR_VALUE(WWRBLASLT_EPILOGUE_GELU, CUBLASLT_EPILOGUE_GELU, HIPBLASLT_EPILOGUE_GELU)
WWR_VALUE(WWRBLASLT_EPILOGUE_GELU_AUX, CUBLASLT_EPILOGUE_GELU_AUX, HIPBLASLT_EPILOGUE_GELU_AUX)
WWR_VALUE(WWRBLASLT_EPILOGUE_GELU_BIAS, CUBLASLT_EPILOGUE_GELU_BIAS, HIPBLASLT_EPILOGUE_GELU_BIAS)
WWR_VALUE(WWRBLASLT_EPILOGUE_GELU_AUX_BIAS, CUBLASLT_EPILOGUE_GELU_AUX_BIAS,
             HIPBLASLT_EPILOGUE_GELU_AUX_BIAS)
WWR_VALUE(WWRBLASLT_EPILOGUE_BGRADA, CUBLASLT_EPILOGUE_BGRADA, HIPBLASLT_EPILOGUE_BGRADA)
WWR_VALUE(WWRBLASLT_EPILOGUE_BGRADB, CUBLASLT_EPILOGUE_BGRADB, HIPBLASLT_EPILOGUE_BGRADB)

// ========================================================================
// Constants: cublasLtMatmulDescAttributes_t
// ========================================================================

WWR_VALUE(WWRBLASLT_MATMUL_DESC_TRANSA, CUBLASLT_MATMUL_DESC_TRANSA, HIPBLASLT_MATMUL_DESC_TRANSA)
WWR_VALUE(WWRBLASLT_MATMUL_DESC_TRANSB, CUBLASLT_MATMUL_DESC_TRANSB, HIPBLASLT_MATMUL_DESC_TRANSB)
WWR_VALUE(WWRBLASLT_MATMUL_DESC_EPILOGUE, CUBLASLT_MATMUL_DESC_EPILOGUE,
             HIPBLASLT_MATMUL_DESC_EPILOGUE)
WWR_VALUE(WWRBLASLT_MATMUL_DESC_BIAS_POINTER, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
             HIPBLASLT_MATMUL_DESC_BIAS_POINTER)
WWR_VALUE(WWRBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
             HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE)
WWR_VALUE(WWRBLASLT_MATMUL_DESC_A_SCALE_POINTER, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
             HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER)
WWR_VALUE(WWRBLASLT_MATMUL_DESC_B_SCALE_POINTER, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
             HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER)
WWR_VALUE(WWRBLASLT_MATMUL_DESC_C_SCALE_POINTER, CUBLASLT_MATMUL_DESC_C_SCALE_POINTER,
             HIPBLASLT_MATMUL_DESC_C_SCALE_POINTER)
WWR_VALUE(WWRBLASLT_MATMUL_DESC_D_SCALE_POINTER, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER,
             HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER)
WWR_VALUE(WWRBLASLT_MATMUL_DESC_A_SCALE_MODE, CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
             HIPBLASLT_MATMUL_DESC_A_SCALE_MODE)
WWR_VALUE(WWRBLASLT_MATMUL_DESC_B_SCALE_MODE, CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
             HIPBLASLT_MATMUL_DESC_B_SCALE_MODE)
WWR_VALUE(WWRBLASLT_MATMUL_DESC_AMAX_D_POINTER, CUBLASLT_MATMUL_DESC_AMAX_D_POINTER,
             HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER)
WWR_VALUE(WWRBLASLT_MATMUL_DESC_POINTER_MODE, CUBLASLT_MATMUL_DESC_POINTER_MODE,
             HIPBLASLT_MATMUL_DESC_POINTER_MODE)
WWR_VALUE(WWRBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
             HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER)
WWR_VALUE(WWRBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,
             HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD)
WWR_VALUE(WWRBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE,
             CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE,
             HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE)
WWR_VALUE(WWRBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE,
             CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE,
             HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE)
WWR_VALUE(WWRBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER,
             CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER,
             HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER)

// ========================================================================
// Constants: cublasLtMatmulMatrixScale_t
// ========================================================================

WWR_VALUE(WWRBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F, CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F,
             HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F)
WWR_VALUE(WWRBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3, CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3,
             HIPBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3)
WWR_VALUE(WWRBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0, CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0,
             HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0)
WWR_VALUE(WWRBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F, CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F,
             HIPBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F)
WWR_VALUE(WWRBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F,
             CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F,
             HIPBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F)
WWR_VALUE(WWRBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F, CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F,
             HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F)
WWR_VALUE(WWRBLASLT_MATMUL_MATRIX_SCALE_END, CUBLASLT_MATMUL_MATRIX_SCALE_END,
             HIPBLASLT_MATMUL_MATRIX_SCALE_END)

// ========================================================================
// Constants: cublasLtMatmulPreferenceAttributes_t
// ========================================================================

WWR_VALUE(WWRBLASLT_MATMUL_PREF_SEARCH_MODE, CUBLASLT_MATMUL_PREF_SEARCH_MODE,
             HIPBLASLT_MATMUL_PREF_SEARCH_MODE)
WWR_VALUE(WWRBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
             HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES)

// ========================================================================
// Constants: cublasLtMatrixLayoutAttribute_t
// ========================================================================

WWR_VALUE(WWRBLASLT_MATRIX_LAYOUT_TYPE, CUBLASLT_MATRIX_LAYOUT_TYPE, HIPBLASLT_MATRIX_LAYOUT_TYPE)
WWR_VALUE(WWRBLASLT_MATRIX_LAYOUT_ORDER, CUBLASLT_MATRIX_LAYOUT_ORDER,
             HIPBLASLT_MATRIX_LAYOUT_ORDER)
WWR_VALUE(WWRBLASLT_MATRIX_LAYOUT_ROWS, CUBLASLT_MATRIX_LAYOUT_ROWS, HIPBLASLT_MATRIX_LAYOUT_ROWS)
WWR_VALUE(WWRBLASLT_MATRIX_LAYOUT_COLS, CUBLASLT_MATRIX_LAYOUT_COLS, HIPBLASLT_MATRIX_LAYOUT_COLS)
WWR_VALUE(WWRBLASLT_MATRIX_LAYOUT_LD, CUBLASLT_MATRIX_LAYOUT_LD, HIPBLASLT_MATRIX_LAYOUT_LD)
WWR_VALUE(WWRBLASLT_MATRIX_LAYOUT_BATCH_COUNT, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
             HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT)
WWR_VALUE(WWRBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
             CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
             HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET)

// ========================================================================
// Constants: cublasLtMatrixTransformDescAttributes_t
// ========================================================================

WWR_VALUE(WWRBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE, CUBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE,
             HIPBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE)
WWR_VALUE(WWRBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE,
             CUBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE,
             HIPBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE)
WWR_VALUE(WWRBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA,
             HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSA)
WWR_VALUE(WWRBLASLT_MATRIX_TRANSFORM_DESC_TRANSB, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSB,
             HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSB)

// ========================================================================
// Constants: cublasLtOrder_t
// ========================================================================

WWR_VALUE(WWRBLASLT_ORDER_COL, CUBLASLT_ORDER_COL, HIPBLASLT_ORDER_COL)
WWR_VALUE(WWRBLASLT_ORDER_ROW, CUBLASLT_ORDER_ROW, HIPBLASLT_ORDER_ROW)

// ========================================================================
// Constants: cublasLtPointerMode_t
// ========================================================================

WWR_VALUE(WWRBLASLT_POINTER_MODE_HOST, CUBLASLT_POINTER_MODE_HOST, HIPBLASLT_POINTER_MODE_HOST)
WWR_VALUE(WWRBLASLT_POINTER_MODE_DEVICE, CUBLASLT_POINTER_MODE_DEVICE,
             HIPBLASLT_POINTER_MODE_DEVICE)
WWR_VALUE(WWRBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST,
             CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST,
             HIPBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST)

// ========================================================================
// Handle management
// ========================================================================

WWR_FUNCTION(wwrblasLtCreate, cublasLtCreate, hipblasLtCreate)
WWR_FUNCTION(wwrblasLtDestroy, cublasLtDestroy, hipblasLtDestroy)

// ========================================================================
// Matmul descriptor
// ========================================================================

WWR_FUNCTION(wwrblasLtMatmulDescCreate, cublasLtMatmulDescCreate, hipblasLtMatmulDescCreate)
WWR_FUNCTION(wwrblasLtMatmulDescDestroy, cublasLtMatmulDescDestroy, hipblasLtMatmulDescDestroy)
WWR_FUNCTION(wwrblasLtMatmulDescSetAttribute, cublasLtMatmulDescSetAttribute,
                hipblasLtMatmulDescSetAttribute)
WWR_FUNCTION(wwrblasLtMatmulDescGetAttribute, cublasLtMatmulDescGetAttribute,
                hipblasLtMatmulDescGetAttribute)

// ========================================================================
// Matrix layout descriptor
// ========================================================================

WWR_FUNCTION(wwrblasLtMatrixLayoutCreate, cublasLtMatrixLayoutCreate, hipblasLtMatrixLayoutCreate)
WWR_FUNCTION(wwrblasLtMatrixLayoutDestroy, cublasLtMatrixLayoutDestroy,
                hipblasLtMatrixLayoutDestroy)
WWR_FUNCTION(wwrblasLtMatrixLayoutSetAttribute, cublasLtMatrixLayoutSetAttribute,
                hipblasLtMatrixLayoutSetAttribute)
WWR_FUNCTION(wwrblasLtMatrixLayoutGetAttribute, cublasLtMatrixLayoutGetAttribute,
                hipblasLtMatrixLayoutGetAttribute)

// ========================================================================
// Matmul preference descriptor
// ========================================================================

WWR_FUNCTION(wwrblasLtMatmulPreferenceCreate, cublasLtMatmulPreferenceCreate,
                hipblasLtMatmulPreferenceCreate)
WWR_FUNCTION(wwrblasLtMatmulPreferenceDestroy, cublasLtMatmulPreferenceDestroy,
                hipblasLtMatmulPreferenceDestroy)
WWR_FUNCTION(wwrblasLtMatmulPreferenceSetAttribute, cublasLtMatmulPreferenceSetAttribute,
                hipblasLtMatmulPreferenceSetAttribute)
WWR_FUNCTION(wwrblasLtMatmulPreferenceGetAttribute, cublasLtMatmulPreferenceGetAttribute,
                hipblasLtMatmulPreferenceGetAttribute)

// ========================================================================
// Heuristic search and matmul execution
// ========================================================================

WWR_FUNCTION(wwrblasLtMatmulAlgoGetHeuristic, cublasLtMatmulAlgoGetHeuristic,
                hipblasLtMatmulAlgoGetHeuristic)
WWR_FUNCTION(wwrblasLtMatmul, cublasLtMatmul, hipblasLtMatmul)

// ========================================================================
// Matrix transform descriptor and execution
// ========================================================================

WWR_FUNCTION(wwrblasLtMatrixTransformDescCreate, cublasLtMatrixTransformDescCreate,
                hipblasLtMatrixTransformDescCreate)
WWR_FUNCTION(wwrblasLtMatrixTransformDescDestroy, cublasLtMatrixTransformDescDestroy,
                hipblasLtMatrixTransformDescDestroy)
WWR_FUNCTION(wwrblasLtMatrixTransformDescSetAttribute, cublasLtMatrixTransformDescSetAttribute,
                hipblasLtMatrixTransformDescSetAttribute)
WWR_FUNCTION(wwrblasLtMatrixTransformDescGetAttribute, cublasLtMatrixTransformDescGetAttribute,
                hipblasLtMatrixTransformDescGetAttribute)
WWR_FUNCTION(wwrblasLtMatrixTransform, cublasLtMatrixTransform, hipblasLtMatrixTransform)

} // namespace wwr
