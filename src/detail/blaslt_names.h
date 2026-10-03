/**
 * @file detail/blaslt_names.h
 * @brief The backend-neutral cuBLASLt / hipBLASLt surface, as a macro-driven
 *        include fragment shared by the module and the non-module #include path
 *
 * NOT a standalone header: it is the list of wwrblasLt* / WWRBLASLT_* names
 * (types, constants and functions) with NO namespace of its own and NO vendor
 * #include. The includer supplies all of that and pastes this inside its own
 * `namespace wwr` -- so one list binds both ways the surface is consumed:
 * wwr.blaslt (the module, `export namespace wwr`) and wwr/blaslt.h (the
 * non-module #include path). Add a name here, once, and both paths gain it.
 *
 * HOST only -- cuBLASLt / hipBLASLt is a host API. The surface binds straight to
 * the vendor's external-linkage `::cublasLt*` / `::hipblasLt*` declarations via
 * the _RAW macros (the "rand.h shape"): no raw vendor module is imported, so the
 * same `::`-prefixed names resolve in the module (vendor header in its GMF) and
 * the #include path alike. The much larger private surface -- cuBLASLt's algo
 * introspection, logger and tile/stages enums, hipBLASLt's arch-name/git-rev and
 * *_EXT attributes -- is deliberately absent; reach it through
 * wwr.cuda.cublasLt / wwr.hip.hipblaslt.
 *
 * Two shared names inherited from classic cuBLAS/hipBLAS are renamed under this
 * layer's prefix (as detail/solver_names.h renames cudaDataType to
 * wwrsolverDataType_t): wwrblasLtStatus_t is cublasStatus_t / hipblasStatus_t
 * (cuBLASLt has no Lt status type of its own; the STATUS_SUCCESS constants live
 * in wwr.blas), and wwrblasLtComputeType_t is cublasComputeType_t /
 * hipblasComputeType_t -- both resolve against the vendor header blaslt.h pulls
 * in (cublasLt.h brings in cublas_api.h for them).
 *
 * Before including, the includer must have, in order:
 *   - the vendor header in scope (cublasLt.h / hipblaslt/hipblaslt.h), which
 *     blaslt.h pulls in;
 *   - WWR_SELECT_RAW(cuda, hip) plus WWR_TYPE_RAW / WWR_VALUE_RAW /
 *     WWR_FUNCTION_RAW on top of it -- keyed on WWR_GPU_BACKEND_* in the module
 *     (backend.h) or WWR_SELECTED_* in the #include path (wwr/blaslt.h).
 *
 * Unlike detail/blas_names.h this fragment names no complex type and carries no
 * hand-written bodies: every line is a plain type/constant/function alias. See
 * src/blaslt.cppm, src/blaslt.h, src/wwr/blaslt.h and docs/architecture.md
 * section 5.
 */

#pragma once

#ifndef WWR_FUNCTION_RAW
#error                                                                                             \
    "detail/blaslt_names.h is an include fragment, not a standalone header: define WWR_TYPE_RAW/VALUE_RAW/FUNCTION_RAW and WWR_SELECT_RAW, ensure the vendor header (via blaslt.h) is in scope, and #include it inside namespace wwr. See src/blaslt.h, src/blaslt.cppm and src/wwr/blaslt.h."
#endif

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables): each wwrblasLt*
// function below is a deliberate constexpr reference to the selected backend's
// entry point (via WWR_FUNCTION_RAW). A reference to a vendor function has no
// const form, so the check cannot be satisfied without abandoning the alias
// pattern -- see backend.h and detail/blas_names.h.

// ========================================================================
// Types
// ========================================================================

WWR_TYPE_RAW(wwrblasLtStatus_t, cublasStatus_t, hipblasStatus_t)
WWR_TYPE_RAW(wwrblasLtComputeType_t, cublasComputeType_t, hipblasComputeType_t)

WWR_TYPE_RAW(wwrblasLtHandle_t, cublasLtHandle_t, hipblasLtHandle_t)

WWR_TYPE_RAW(wwrblasLtMatmulDesc_t, cublasLtMatmulDesc_t, hipblasLtMatmulDesc_t)
WWR_TYPE_RAW(wwrblasLtMatmulDescOpaque_t, cublasLtMatmulDescOpaque_t, hipblasLtMatmulDescOpaque_t)
WWR_TYPE_RAW(wwrblasLtMatmulDescAttributes_t, cublasLtMatmulDescAttributes_t,
             hipblasLtMatmulDescAttributes_t)

WWR_TYPE_RAW(wwrblasLtMatrixLayout_t, cublasLtMatrixLayout_t, hipblasLtMatrixLayout_t)
WWR_TYPE_RAW(wwrblasLtMatrixLayoutOpaque_t, cublasLtMatrixLayoutOpaque_t,
             hipblasLtMatrixLayoutOpaque_t)
WWR_TYPE_RAW(wwrblasLtMatrixLayoutAttribute_t, cublasLtMatrixLayoutAttribute_t,
             hipblasLtMatrixLayoutAttribute_t)

WWR_TYPE_RAW(wwrblasLtMatmulPreference_t, cublasLtMatmulPreference_t, hipblasLtMatmulPreference_t)
WWR_TYPE_RAW(wwrblasLtMatmulPreferenceOpaque_t, cublasLtMatmulPreferenceOpaque_t,
             hipblasLtMatmulPreferenceOpaque_t)
WWR_TYPE_RAW(wwrblasLtMatmulPreferenceAttributes_t, cublasLtMatmulPreferenceAttributes_t,
             hipblasLtMatmulPreferenceAttributes_t)

WWR_TYPE_RAW(wwrblasLtMatmulAlgo_t, cublasLtMatmulAlgo_t, hipblasLtMatmulAlgo_t)
WWR_TYPE_RAW(wwrblasLtMatmulHeuristicResult_t, cublasLtMatmulHeuristicResult_t,
             hipblasLtMatmulHeuristicResult_t)

WWR_TYPE_RAW(wwrblasLtMatrixTransformDesc_t, cublasLtMatrixTransformDesc_t,
             hipblasLtMatrixTransformDesc_t)
WWR_TYPE_RAW(wwrblasLtMatrixTransformDescOpaque_t, cublasLtMatrixTransformDescOpaque_t,
             hipblasLtMatrixTransformDescOpaque_t)
WWR_TYPE_RAW(wwrblasLtMatrixTransformDescAttributes_t, cublasLtMatrixTransformDescAttributes_t,
             hipblasLtMatrixTransformDescAttributes_t)

WWR_TYPE_RAW(wwrblasLtEpilogue_t, cublasLtEpilogue_t, hipblasLtEpilogue_t)
WWR_TYPE_RAW(wwrblasLtMatmulMatrixScale_t, cublasLtMatmulMatrixScale_t,
             hipblasLtMatmulMatrixScale_t)
WWR_TYPE_RAW(wwrblasLtOrder_t, cublasLtOrder_t, hipblasLtOrder_t)
WWR_TYPE_RAW(wwrblasLtPointerMode_t, cublasLtPointerMode_t, hipblasLtPointerMode_t)

// ========================================================================
// Constants: cublasLtEpilogue_t
// ========================================================================

WWR_VALUE_RAW(WWRBLASLT_EPILOGUE_DEFAULT, CUBLASLT_EPILOGUE_DEFAULT, HIPBLASLT_EPILOGUE_DEFAULT)
WWR_VALUE_RAW(WWRBLASLT_EPILOGUE_RELU, CUBLASLT_EPILOGUE_RELU, HIPBLASLT_EPILOGUE_RELU)
WWR_VALUE_RAW(WWRBLASLT_EPILOGUE_RELU_AUX, CUBLASLT_EPILOGUE_RELU_AUX, HIPBLASLT_EPILOGUE_RELU_AUX)
WWR_VALUE_RAW(WWRBLASLT_EPILOGUE_BIAS, CUBLASLT_EPILOGUE_BIAS, HIPBLASLT_EPILOGUE_BIAS)
WWR_VALUE_RAW(WWRBLASLT_EPILOGUE_RELU_BIAS, CUBLASLT_EPILOGUE_RELU_BIAS,
             HIPBLASLT_EPILOGUE_RELU_BIAS)
WWR_VALUE_RAW(WWRBLASLT_EPILOGUE_RELU_AUX_BIAS, CUBLASLT_EPILOGUE_RELU_AUX_BIAS,
             HIPBLASLT_EPILOGUE_RELU_AUX_BIAS)
WWR_VALUE_RAW(WWRBLASLT_EPILOGUE_DGELU, CUBLASLT_EPILOGUE_DGELU, HIPBLASLT_EPILOGUE_DGELU)
WWR_VALUE_RAW(WWRBLASLT_EPILOGUE_DGELU_BGRAD, CUBLASLT_EPILOGUE_DGELU_BGRAD,
             HIPBLASLT_EPILOGUE_DGELU_BGRAD)
WWR_VALUE_RAW(WWRBLASLT_EPILOGUE_GELU, CUBLASLT_EPILOGUE_GELU, HIPBLASLT_EPILOGUE_GELU)
WWR_VALUE_RAW(WWRBLASLT_EPILOGUE_GELU_AUX, CUBLASLT_EPILOGUE_GELU_AUX, HIPBLASLT_EPILOGUE_GELU_AUX)
WWR_VALUE_RAW(WWRBLASLT_EPILOGUE_GELU_BIAS, CUBLASLT_EPILOGUE_GELU_BIAS,
             HIPBLASLT_EPILOGUE_GELU_BIAS)
WWR_VALUE_RAW(WWRBLASLT_EPILOGUE_GELU_AUX_BIAS, CUBLASLT_EPILOGUE_GELU_AUX_BIAS,
             HIPBLASLT_EPILOGUE_GELU_AUX_BIAS)
WWR_VALUE_RAW(WWRBLASLT_EPILOGUE_BGRADA, CUBLASLT_EPILOGUE_BGRADA, HIPBLASLT_EPILOGUE_BGRADA)
WWR_VALUE_RAW(WWRBLASLT_EPILOGUE_BGRADB, CUBLASLT_EPILOGUE_BGRADB, HIPBLASLT_EPILOGUE_BGRADB)

// ========================================================================
// Constants: cublasLtMatmulDescAttributes_t
// ========================================================================

WWR_VALUE_RAW(WWRBLASLT_MATMUL_DESC_TRANSA, CUBLASLT_MATMUL_DESC_TRANSA,
             HIPBLASLT_MATMUL_DESC_TRANSA)
WWR_VALUE_RAW(WWRBLASLT_MATMUL_DESC_TRANSB, CUBLASLT_MATMUL_DESC_TRANSB,
             HIPBLASLT_MATMUL_DESC_TRANSB)
WWR_VALUE_RAW(WWRBLASLT_MATMUL_DESC_EPILOGUE, CUBLASLT_MATMUL_DESC_EPILOGUE,
             HIPBLASLT_MATMUL_DESC_EPILOGUE)
WWR_VALUE_RAW(WWRBLASLT_MATMUL_DESC_BIAS_POINTER, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
             HIPBLASLT_MATMUL_DESC_BIAS_POINTER)
WWR_VALUE_RAW(WWRBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
             HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE)
WWR_VALUE_RAW(WWRBLASLT_MATMUL_DESC_A_SCALE_POINTER, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
             HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER)
WWR_VALUE_RAW(WWRBLASLT_MATMUL_DESC_B_SCALE_POINTER, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
             HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER)
WWR_VALUE_RAW(WWRBLASLT_MATMUL_DESC_C_SCALE_POINTER, CUBLASLT_MATMUL_DESC_C_SCALE_POINTER,
             HIPBLASLT_MATMUL_DESC_C_SCALE_POINTER)
WWR_VALUE_RAW(WWRBLASLT_MATMUL_DESC_D_SCALE_POINTER, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER,
             HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER)
WWR_VALUE_RAW(WWRBLASLT_MATMUL_DESC_A_SCALE_MODE, CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
             HIPBLASLT_MATMUL_DESC_A_SCALE_MODE)
WWR_VALUE_RAW(WWRBLASLT_MATMUL_DESC_B_SCALE_MODE, CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
             HIPBLASLT_MATMUL_DESC_B_SCALE_MODE)
WWR_VALUE_RAW(WWRBLASLT_MATMUL_DESC_AMAX_D_POINTER, CUBLASLT_MATMUL_DESC_AMAX_D_POINTER,
             HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER)
WWR_VALUE_RAW(WWRBLASLT_MATMUL_DESC_POINTER_MODE, CUBLASLT_MATMUL_DESC_POINTER_MODE,
             HIPBLASLT_MATMUL_DESC_POINTER_MODE)
WWR_VALUE_RAW(WWRBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
             CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
             HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER)
WWR_VALUE_RAW(WWRBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,
             HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD)
WWR_VALUE_RAW(WWRBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE,
             CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE,
             HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE)
WWR_VALUE_RAW(WWRBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE,
             CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE,
             HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE)
WWR_VALUE_RAW(WWRBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER,
             CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER,
             HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER)

// ========================================================================
// Constants: cublasLtMatmulMatrixScale_t
// ========================================================================

WWR_VALUE_RAW(WWRBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F, CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F,
             HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F)
WWR_VALUE_RAW(WWRBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3, CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3,
             HIPBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3)
WWR_VALUE_RAW(WWRBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0, CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0,
             HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0)
WWR_VALUE_RAW(WWRBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F, CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F,
             HIPBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F)
WWR_VALUE_RAW(WWRBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F,
             CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F,
             HIPBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F)
WWR_VALUE_RAW(WWRBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F,
             CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F,
             HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F)
WWR_VALUE_RAW(WWRBLASLT_MATMUL_MATRIX_SCALE_END, CUBLASLT_MATMUL_MATRIX_SCALE_END,
             HIPBLASLT_MATMUL_MATRIX_SCALE_END)

// ========================================================================
// Constants: cublasLtMatmulPreferenceAttributes_t
// ========================================================================

WWR_VALUE_RAW(WWRBLASLT_MATMUL_PREF_SEARCH_MODE, CUBLASLT_MATMUL_PREF_SEARCH_MODE,
             HIPBLASLT_MATMUL_PREF_SEARCH_MODE)
WWR_VALUE_RAW(WWRBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
             HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES)

// ========================================================================
// Constants: cublasLtMatrixLayoutAttribute_t
// ========================================================================

WWR_VALUE_RAW(WWRBLASLT_MATRIX_LAYOUT_TYPE, CUBLASLT_MATRIX_LAYOUT_TYPE,
             HIPBLASLT_MATRIX_LAYOUT_TYPE)
WWR_VALUE_RAW(WWRBLASLT_MATRIX_LAYOUT_ORDER, CUBLASLT_MATRIX_LAYOUT_ORDER,
             HIPBLASLT_MATRIX_LAYOUT_ORDER)
WWR_VALUE_RAW(WWRBLASLT_MATRIX_LAYOUT_ROWS, CUBLASLT_MATRIX_LAYOUT_ROWS,
             HIPBLASLT_MATRIX_LAYOUT_ROWS)
WWR_VALUE_RAW(WWRBLASLT_MATRIX_LAYOUT_COLS, CUBLASLT_MATRIX_LAYOUT_COLS,
             HIPBLASLT_MATRIX_LAYOUT_COLS)
WWR_VALUE_RAW(WWRBLASLT_MATRIX_LAYOUT_LD, CUBLASLT_MATRIX_LAYOUT_LD, HIPBLASLT_MATRIX_LAYOUT_LD)
WWR_VALUE_RAW(WWRBLASLT_MATRIX_LAYOUT_BATCH_COUNT, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
             HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT)
WWR_VALUE_RAW(WWRBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
             CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
             HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET)

// ========================================================================
// Constants: cublasLtMatrixTransformDescAttributes_t
// ========================================================================

WWR_VALUE_RAW(WWRBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE,
             CUBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE,
             HIPBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE)
WWR_VALUE_RAW(WWRBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE,
             CUBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE,
             HIPBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE)
WWR_VALUE_RAW(WWRBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA,
             HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSA)
WWR_VALUE_RAW(WWRBLASLT_MATRIX_TRANSFORM_DESC_TRANSB, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSB,
             HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSB)

// ========================================================================
// Constants: cublasLtOrder_t
// ========================================================================

WWR_VALUE_RAW(WWRBLASLT_ORDER_COL, CUBLASLT_ORDER_COL, HIPBLASLT_ORDER_COL)
WWR_VALUE_RAW(WWRBLASLT_ORDER_ROW, CUBLASLT_ORDER_ROW, HIPBLASLT_ORDER_ROW)

// ========================================================================
// Constants: cublasLtPointerMode_t
// ========================================================================

WWR_VALUE_RAW(WWRBLASLT_POINTER_MODE_HOST, CUBLASLT_POINTER_MODE_HOST, HIPBLASLT_POINTER_MODE_HOST)
WWR_VALUE_RAW(WWRBLASLT_POINTER_MODE_DEVICE, CUBLASLT_POINTER_MODE_DEVICE,
             HIPBLASLT_POINTER_MODE_DEVICE)
WWR_VALUE_RAW(WWRBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST,
             CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST,
             HIPBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST)

// ========================================================================
// Handle management
// ========================================================================

WWR_FUNCTION_RAW(wwrblasLtCreate, cublasLtCreate, hipblasLtCreate)
WWR_FUNCTION_RAW(wwrblasLtDestroy, cublasLtDestroy, hipblasLtDestroy)

// ========================================================================
// Matmul descriptor
// ========================================================================

WWR_FUNCTION_RAW(wwrblasLtMatmulDescCreate, cublasLtMatmulDescCreate, hipblasLtMatmulDescCreate)
WWR_FUNCTION_RAW(wwrblasLtMatmulDescDestroy, cublasLtMatmulDescDestroy, hipblasLtMatmulDescDestroy)
WWR_FUNCTION_RAW(wwrblasLtMatmulDescSetAttribute, cublasLtMatmulDescSetAttribute,
                hipblasLtMatmulDescSetAttribute)
WWR_FUNCTION_RAW(wwrblasLtMatmulDescGetAttribute, cublasLtMatmulDescGetAttribute,
                hipblasLtMatmulDescGetAttribute)

// ========================================================================
// Matrix layout descriptor
// ========================================================================

WWR_FUNCTION_RAW(wwrblasLtMatrixLayoutCreate, cublasLtMatrixLayoutCreate,
                hipblasLtMatrixLayoutCreate)
WWR_FUNCTION_RAW(wwrblasLtMatrixLayoutDestroy, cublasLtMatrixLayoutDestroy,
                hipblasLtMatrixLayoutDestroy)
WWR_FUNCTION_RAW(wwrblasLtMatrixLayoutSetAttribute, cublasLtMatrixLayoutSetAttribute,
                hipblasLtMatrixLayoutSetAttribute)
WWR_FUNCTION_RAW(wwrblasLtMatrixLayoutGetAttribute, cublasLtMatrixLayoutGetAttribute,
                hipblasLtMatrixLayoutGetAttribute)

// ========================================================================
// Matmul preference descriptor
// ========================================================================

WWR_FUNCTION_RAW(wwrblasLtMatmulPreferenceCreate, cublasLtMatmulPreferenceCreate,
                hipblasLtMatmulPreferenceCreate)
WWR_FUNCTION_RAW(wwrblasLtMatmulPreferenceDestroy, cublasLtMatmulPreferenceDestroy,
                hipblasLtMatmulPreferenceDestroy)
WWR_FUNCTION_RAW(wwrblasLtMatmulPreferenceSetAttribute, cublasLtMatmulPreferenceSetAttribute,
                hipblasLtMatmulPreferenceSetAttribute)
WWR_FUNCTION_RAW(wwrblasLtMatmulPreferenceGetAttribute, cublasLtMatmulPreferenceGetAttribute,
                hipblasLtMatmulPreferenceGetAttribute)

// ========================================================================
// Heuristic search and matmul execution
// ========================================================================

WWR_FUNCTION_RAW(wwrblasLtMatmulAlgoGetHeuristic, cublasLtMatmulAlgoGetHeuristic,
                hipblasLtMatmulAlgoGetHeuristic)
WWR_FUNCTION_RAW(wwrblasLtMatmul, cublasLtMatmul, hipblasLtMatmul)

// ========================================================================
// Matrix transform descriptor and execution
// ========================================================================

WWR_FUNCTION_RAW(wwrblasLtMatrixTransformDescCreate, cublasLtMatrixTransformDescCreate,
                hipblasLtMatrixTransformDescCreate)
WWR_FUNCTION_RAW(wwrblasLtMatrixTransformDescDestroy, cublasLtMatrixTransformDescDestroy,
                hipblasLtMatrixTransformDescDestroy)
WWR_FUNCTION_RAW(wwrblasLtMatrixTransformDescSetAttribute, cublasLtMatrixTransformDescSetAttribute,
                hipblasLtMatrixTransformDescSetAttribute)
WWR_FUNCTION_RAW(wwrblasLtMatrixTransformDescGetAttribute, cublasLtMatrixTransformDescGetAttribute,
                hipblasLtMatrixTransformDescGetAttribute)
WWR_FUNCTION_RAW(wwrblasLtMatrixTransform, cublasLtMatrixTransform, hipblasLtMatrixTransform)

// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)
