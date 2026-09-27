/**
 * @file hipblaslt.cppm
 * @brief hipBLASLt API module wrapper for gpumod project
 *
 * Wraps hipblaslt/hipblaslt.h. CUDA counterpart: wwr.cuda.cublasLt.
 *
 * hipBLASLt's public surface is much narrower than cuBLASLt's -- 24 functions:
 * library/handle management, matrix layout / matmul descriptor / preference
 * descriptor CRUD with attribute get/set, heuristic algorithm search,
 * hipblasLtMatmul itself, and the matrix transform descriptor family. There is
 * no logger callback family, no heuristics-cache capacity control, no
 * cublasLtDisableCpuInstructionsSetMask analogue, and no algo introspection
 * beyond the heuristic search: hipblasLtMatmulAlgo_t is an opaque blob
 * obtained only from hipblasLtMatmulAlgoGetHeuristic. hipblasLtGetGitRevision
 * and hipblasLtGetArchName are its own additions. The status and compute-type
 * enums it returns and consumes (hipblasStatus_t, hipblasComputeType_t) belong
 * to hipblas-common and are re-exported here, as cuBLASLt re-exports its own.
 *
 * The *Opaque_t descriptor types are uint64_t data[N] structs, not cuBLASLt's
 * int64_t[N]; hipblasLtMatmulAlgo_t and hipblasLtMatmulHeuristicResult_t are
 * ordinary structs with default member initializers, guarded by
 * __HIP_PLATFORM_AMD__, which hip::host satisfies.
 *
 * Usage:
 *   import wwr.hip.hipblaslt;
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
#include <hipblaslt/hipblaslt.h>

export module wwr.hip.hipblaslt;

import std;

export namespace wwr::hip {

// ========================================================================
// Enum: hipblasLtEpilogue_t
// ========================================================================
using ::HIPBLASLT_EPILOGUE_BGRADA;
using ::HIPBLASLT_EPILOGUE_BGRADB;
using ::HIPBLASLT_EPILOGUE_BIAS;
using ::HIPBLASLT_EPILOGUE_CLAMP_AUX_BIAS_EXT;
using ::HIPBLASLT_EPILOGUE_CLAMP_AUX_EXT;
using ::HIPBLASLT_EPILOGUE_CLAMP_BIAS_EXT;
using ::HIPBLASLT_EPILOGUE_CLAMP_EXT;
using ::HIPBLASLT_EPILOGUE_DEFAULT;
using ::HIPBLASLT_EPILOGUE_DGELU;
using ::HIPBLASLT_EPILOGUE_DGELU_BGRAD;
using ::HIPBLASLT_EPILOGUE_GELU;
using ::HIPBLASLT_EPILOGUE_GELU_AUX;
using ::HIPBLASLT_EPILOGUE_GELU_AUX_BIAS;
using ::HIPBLASLT_EPILOGUE_GELU_BIAS;
using ::HIPBLASLT_EPILOGUE_RELU;
using ::HIPBLASLT_EPILOGUE_RELU_AUX;
using ::HIPBLASLT_EPILOGUE_RELU_AUX_BIAS;
using ::HIPBLASLT_EPILOGUE_RELU_BIAS;
using ::HIPBLASLT_EPILOGUE_SIGMOID_BIAS_EXT;
using ::HIPBLASLT_EPILOGUE_SIGMOID_EXT;
using ::HIPBLASLT_EPILOGUE_SWISH_BIAS_EXT;
using ::HIPBLASLT_EPILOGUE_SWISH_EXT;
using ::hipblasLtEpilogue_t;

// ========================================================================
// Enum: hipblasLtMatrixLayoutAttribute_t
// ========================================================================
using ::HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT;
using ::HIPBLASLT_MATRIX_LAYOUT_COLS;
using ::HIPBLASLT_MATRIX_LAYOUT_LD;
using ::HIPBLASLT_MATRIX_LAYOUT_ORDER;
using ::HIPBLASLT_MATRIX_LAYOUT_ROWS;
using ::HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET;
using ::HIPBLASLT_MATRIX_LAYOUT_TYPE;
using ::hipblasLtMatrixLayoutAttribute_t;

// ========================================================================
// Enum: hipblasLtPointerMode_t
// ========================================================================
using ::HIPBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST;
using ::HIPBLASLT_POINTER_MODE_DEVICE;
using ::HIPBLASLT_POINTER_MODE_HOST;
using ::hipblasLtPointerMode_t;

// ========================================================================
// Enum: hipblasLtMatmulMatrixScale_t
// ========================================================================
using ::HIPBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F;
using ::HIPBLASLT_MATMUL_MATRIX_SCALE_END;
using ::HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
using ::HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
using ::HIPBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F;
using ::HIPBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
using ::HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
using ::hipblasLtMatmulMatrixScale_t;

// ========================================================================
// Enum: hipblasLtMatmulDescAttributes_t
// ========================================================================
using ::HIPBLASLT_MATMUL_DESC_A_SCALE_MODE;
using ::HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER;
using ::HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER;
using ::HIPBLASLT_MATMUL_DESC_B_SCALE_MODE;
using ::HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER;
using ::HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE;
using ::HIPBLASLT_MATMUL_DESC_BIAS_POINTER;
using ::HIPBLASLT_MATMUL_DESC_C_SCALE_POINTER;
using ::HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT;
using ::HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT;
using ::HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER;
using ::HIPBLASLT_MATMUL_DESC_EPILOGUE;
using ::HIPBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG0_EXT;
using ::HIPBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG1_EXT;
using ::HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE;
using ::HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE;
using ::HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD;
using ::HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER;
using ::HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER;
using ::HIPBLASLT_MATMUL_DESC_MAX;
using ::HIPBLASLT_MATMUL_DESC_POINTER_MODE;
using ::HIPBLASLT_MATMUL_DESC_TRANSA;
using ::HIPBLASLT_MATMUL_DESC_TRANSB;
using ::hipblasLtMatmulDescAttributes_t;

// ========================================================================
// Enum: hipblasLtMatmulPreferenceAttributes_t
// ========================================================================
using ::HIPBLASLT_MATMUL_PREF_MAX;
using ::HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES;
using ::HIPBLASLT_MATMUL_PREF_SEARCH_MODE;
using ::hipblasLtMatmulPreferenceAttributes_t;

// ========================================================================
// Enum: hipblasLtOrder_t
// ========================================================================
using ::HIPBLASLT_ORDER_COL;
using ::HIPBLASLT_ORDER_COL16_4R16;
using ::HIPBLASLT_ORDER_COL16_4R2;
using ::HIPBLASLT_ORDER_COL16_4R4;
using ::HIPBLASLT_ORDER_COL16_4R8;
using ::HIPBLASLT_ORDER_ROW;
using ::hipblasLtOrder_t;

// ========================================================================
// Enum: hipblasLtMatrixTransformDescAttributes_t
// ========================================================================
using ::HIPBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE;
using ::HIPBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE;
using ::HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSA;
using ::HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSB;
using ::hipblasLtMatrixTransformDescAttributes_t;

// ========================================================================
// Re-exported hipBLAS common types (from hipblas-common.h)
// ========================================================================
// hipBLASLt has no status/compute type of its own: its functions return
// hipblasStatus_t and its matmul descriptor takes hipblasComputeType_t, both
// from hipblas-common. The cuBLASLt counterpart re-exports cublasStatus_t /
// cublasComputeType_t the same way -- see wwr.cuda.cublasLt.
using ::hipblasComputeType_t;
using ::hipblasStatus_t;

// ========================================================================
// Handle type traits (opaque pointer types)
// ========================================================================
using ::hipblasLtHandle_t;
using ::hipblasLtMatmulDesc_t;
using ::hipblasLtMatmulPreference_t;
using ::hipblasLtMatrixLayout_t;
using ::hipblasLtMatrixTransformDesc_t;

// ========================================================================
// Semi-opaque descriptor struct types
// ========================================================================
using ::hipblasLtMatmulDescOpaque_t;
using ::hipblasLtMatmulPreferenceOpaque_t;
using ::hipblasLtMatrixLayoutOpaque_t;
using ::hipblasLtMatrixTransformDescOpaque_t;

// ========================================================================
// Algorithm and heuristic result structs
// ========================================================================
using ::hipblasLtMatmulAlgo_t;
using ::hipblasLtMatmulHeuristicResult_t;

// ========================================================================
// Library and Handle Management
// ========================================================================
using ::hipblasLtCreate;
using ::hipblasLtDestroy;
using ::hipblasLtGetArchName;
using ::hipblasLtGetGitRevision;
using ::hipblasLtGetVersion;

// ========================================================================
// Matrix Layout Descriptor
// ========================================================================
using ::hipblasLtMatrixLayoutCreate;
using ::hipblasLtMatrixLayoutDestroy;
using ::hipblasLtMatrixLayoutGetAttribute;
using ::hipblasLtMatrixLayoutSetAttribute;

// ========================================================================
// Matmul Descriptor
// ========================================================================
using ::hipblasLtMatmulDescCreate;
using ::hipblasLtMatmulDescDestroy;
using ::hipblasLtMatmulDescGetAttribute;
using ::hipblasLtMatmulDescSetAttribute;

// ========================================================================
// Matmul Preference Descriptor
// ========================================================================
using ::hipblasLtMatmulPreferenceCreate;
using ::hipblasLtMatmulPreferenceDestroy;
using ::hipblasLtMatmulPreferenceGetAttribute;
using ::hipblasLtMatmulPreferenceSetAttribute;

// ========================================================================
// Heuristic Search and Matmul Execution
// ========================================================================
using ::hipblasLtMatmul;
using ::hipblasLtMatmulAlgoGetHeuristic;

// ========================================================================
// Matrix Transform Descriptor and Execution
// ========================================================================
using ::hipblasLtMatrixTransform;
using ::hipblasLtMatrixTransformDescCreate;
using ::hipblasLtMatrixTransformDescDestroy;
using ::hipblasLtMatrixTransformDescGetAttribute;
using ::hipblasLtMatrixTransformDescSetAttribute;

} // namespace wwr::hip
