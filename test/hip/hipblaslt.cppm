// hipblaslt.cppm - Compile-time tests for wwr.hip.hipblaslt

module;

#include "test/shared/link_check.h"

export module wwr.test.hip.hipblaslt;

import std;
import wwr.hip.hipblaslt;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time tests for wwr.hip.hipblaslt
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace wwr::hip::test {

using namespace wwr::hip;

// ────────────────────────────────────────────────────────────────────────
// Enum type checks
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_enum_v<hipblasLtEpilogue_t>);
static_assert(std::is_enum_v<hipblasLtMatrixLayoutAttribute_t>);
static_assert(std::is_enum_v<hipblasLtPointerMode_t>);
static_assert(std::is_enum_v<hipblasLtMatmulMatrixScale_t>);
static_assert(std::is_enum_v<hipblasLtMatmulDescAttributes_t>);
static_assert(std::is_enum_v<hipblasLtMatmulPreferenceAttributes_t>);
static_assert(std::is_enum_v<hipblasLtOrder_t>);
static_assert(std::is_enum_v<hipblasLtMatrixTransformDescAttributes_t>);

// ────────────────────────────────────────────────────────────────────────
// Enum values: hipblasLtEpilogue_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(HIPBLASLT_EPILOGUE_DEFAULT) == 1);
static_assert(static_cast<int>(HIPBLASLT_EPILOGUE_RELU) == 2);
static_assert(static_cast<int>(HIPBLASLT_EPILOGUE_BIAS) == 4);
static_assert(static_cast<int>(HIPBLASLT_EPILOGUE_RELU_BIAS) == 6);
static_assert(static_cast<int>(HIPBLASLT_EPILOGUE_GELU) == 32);
static_assert(static_cast<int>(HIPBLASLT_EPILOGUE_GELU_BIAS) == 36);
static_assert(static_cast<int>(HIPBLASLT_EPILOGUE_RELU_AUX) == 130);
static_assert(static_cast<int>(HIPBLASLT_EPILOGUE_RELU_AUX_BIAS) == 134);
static_assert(static_cast<int>(HIPBLASLT_EPILOGUE_GELU_AUX) == 160);
static_assert(static_cast<int>(HIPBLASLT_EPILOGUE_GELU_AUX_BIAS) == 164);
static_assert(static_cast<int>(HIPBLASLT_EPILOGUE_DGELU) == 192);
static_assert(static_cast<int>(HIPBLASLT_EPILOGUE_DGELU_BGRAD) == 208);
static_assert(static_cast<int>(HIPBLASLT_EPILOGUE_BGRADA) == 256);
static_assert(static_cast<int>(HIPBLASLT_EPILOGUE_BGRADB) == 512);
static_assert(static_cast<int>(HIPBLASLT_EPILOGUE_SWISH_EXT) == 65536);
static_assert(static_cast<int>(HIPBLASLT_EPILOGUE_SWISH_BIAS_EXT) == 65540);
static_assert(static_cast<int>(HIPBLASLT_EPILOGUE_CLAMP_EXT) == 131072);
static_assert(static_cast<int>(HIPBLASLT_EPILOGUE_CLAMP_BIAS_EXT) == 131076);
static_assert(static_cast<int>(HIPBLASLT_EPILOGUE_CLAMP_AUX_EXT) == 131200);
static_assert(static_cast<int>(HIPBLASLT_EPILOGUE_CLAMP_AUX_BIAS_EXT) == 131204);
static_assert(static_cast<int>(HIPBLASLT_EPILOGUE_SIGMOID_EXT) == 262144);
static_assert(static_cast<int>(HIPBLASLT_EPILOGUE_SIGMOID_BIAS_EXT) == 262148);

// ────────────────────────────────────────────────────────────────────────
// Enum values: hipblasLtMatrixLayoutAttribute_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT) == 0);
static_assert(static_cast<int>(HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET) == 1);
static_assert(static_cast<int>(HIPBLASLT_MATRIX_LAYOUT_TYPE) == 2);
static_assert(static_cast<int>(HIPBLASLT_MATRIX_LAYOUT_ORDER) == 3);
static_assert(static_cast<int>(HIPBLASLT_MATRIX_LAYOUT_ROWS) == 4);
static_assert(static_cast<int>(HIPBLASLT_MATRIX_LAYOUT_COLS) == 5);
static_assert(static_cast<int>(HIPBLASLT_MATRIX_LAYOUT_LD) == 6);

// ────────────────────────────────────────────────────────────────────────
// Enum values: hipblasLtPointerMode_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(HIPBLASLT_POINTER_MODE_HOST) == 0);
static_assert(static_cast<int>(HIPBLASLT_POINTER_MODE_DEVICE) == 1);
static_assert(static_cast<int>(HIPBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST) == 4);

// ────────────────────────────────────────────────────────────────────────
// Enum values: hipblasLtMatmulMatrixScale_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F) == 0);
static_assert(static_cast<int>(HIPBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3) == 1);
static_assert(static_cast<int>(HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0) == 2);
static_assert(static_cast<int>(HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F) == 3);
static_assert(static_cast<int>(HIPBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F) == 4);
static_assert(static_cast<int>(HIPBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F) == 5);
static_assert(static_cast<int>(HIPBLASLT_MATMUL_MATRIX_SCALE_END) == 6);

// ────────────────────────────────────────────────────────────────────────
// Enum values: hipblasLtMatmulDescAttributes_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(HIPBLASLT_MATMUL_DESC_TRANSA) == 0);
static_assert(static_cast<int>(HIPBLASLT_MATMUL_DESC_TRANSB) == 1);
static_assert(static_cast<int>(HIPBLASLT_MATMUL_DESC_EPILOGUE) == 2);
static_assert(static_cast<int>(HIPBLASLT_MATMUL_DESC_BIAS_POINTER) == 3);
static_assert(static_cast<int>(HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE) == 4);
static_assert(static_cast<int>(HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER) == 5);
static_assert(static_cast<int>(HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER) == 6);
static_assert(static_cast<int>(HIPBLASLT_MATMUL_DESC_C_SCALE_POINTER) == 7);
static_assert(static_cast<int>(HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER) == 8);
static_assert(static_cast<int>(HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER) == 9);
static_assert(static_cast<int>(HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER) == 10);
static_assert(static_cast<int>(HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD) == 11);
static_assert(static_cast<int>(HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE) == 12);
static_assert(static_cast<int>(HIPBLASLT_MATMUL_DESC_POINTER_MODE) == 13);
static_assert(static_cast<int>(HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER) == 14);
static_assert(static_cast<int>(HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE) == 22);
static_assert(static_cast<int>(HIPBLASLT_MATMUL_DESC_A_SCALE_MODE) == 31);
static_assert(static_cast<int>(HIPBLASLT_MATMUL_DESC_B_SCALE_MODE) == 32);
static_assert(static_cast<int>(HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT) == 100);
static_assert(static_cast<int>(HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT) == 101);
static_assert(static_cast<int>(HIPBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG0_EXT) == 102);
static_assert(static_cast<int>(HIPBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG1_EXT) == 103);
static_assert(static_cast<int>(HIPBLASLT_MATMUL_DESC_MAX) == 104);

// ────────────────────────────────────────────────────────────────────────
// Enum values: hipblasLtMatmulPreferenceAttributes_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(HIPBLASLT_MATMUL_PREF_SEARCH_MODE) == 0);
static_assert(static_cast<int>(HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES) == 1);
static_assert(static_cast<int>(HIPBLASLT_MATMUL_PREF_MAX) == 2);

// ────────────────────────────────────────────────────────────────────────
// Enum values: hipblasLtOrder_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(HIPBLASLT_ORDER_COL) == 0);
static_assert(static_cast<int>(HIPBLASLT_ORDER_ROW) == 1);
static_assert(static_cast<int>(HIPBLASLT_ORDER_COL16_4R16) == 100);
static_assert(static_cast<int>(HIPBLASLT_ORDER_COL16_4R8) == 101);
static_assert(static_cast<int>(HIPBLASLT_ORDER_COL16_4R4) == 102);
static_assert(static_cast<int>(HIPBLASLT_ORDER_COL16_4R2) == 103);

// ────────────────────────────────────────────────────────────────────────
// Enum values: hipblasLtMatrixTransformDescAttributes_t (0-based, implicit)
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(HIPBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE) == 0);
static_assert(static_cast<int>(HIPBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE) == 1);
static_assert(static_cast<int>(HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSA) == 2);
static_assert(static_cast<int>(HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSB) == 3);

// ────────────────────────────────────────────────────────────────────────
// Handle type traits (opaque pointer types)
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_pointer_v<hipblasLtHandle_t>);
static_assert(std::is_pointer_v<hipblasLtMatrixLayout_t>);
static_assert(std::is_pointer_v<hipblasLtMatmulDesc_t>);
static_assert(std::is_pointer_v<hipblasLtMatmulPreference_t>);
static_assert(std::is_pointer_v<hipblasLtMatrixTransformDesc_t>);

// ────────────────────────────────────────────────────────────────────────
// Semi-opaque descriptor struct type traits (uint64_t data[N] -- not
// cuBLASLt's int64_t[N], verified directly)
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_trivially_copyable_v<hipblasLtMatmulDescOpaque_t>);
static_assert(std::is_standard_layout_v<hipblasLtMatmulDescOpaque_t>);
static_assert(sizeof(hipblasLtMatmulDescOpaque_t) == 4 * sizeof(std::uint64_t));

static_assert(std::is_trivially_copyable_v<hipblasLtMatrixLayoutOpaque_t>);
static_assert(std::is_standard_layout_v<hipblasLtMatrixLayoutOpaque_t>);
static_assert(sizeof(hipblasLtMatrixLayoutOpaque_t) == 4 * sizeof(std::uint64_t));

static_assert(std::is_trivially_copyable_v<hipblasLtMatmulPreferenceOpaque_t>);
static_assert(std::is_standard_layout_v<hipblasLtMatmulPreferenceOpaque_t>);
static_assert(sizeof(hipblasLtMatmulPreferenceOpaque_t) == 5 * sizeof(std::uint64_t));

static_assert(std::is_trivially_copyable_v<hipblasLtMatrixTransformDescOpaque_t>);
static_assert(std::is_standard_layout_v<hipblasLtMatrixTransformDescOpaque_t>);
static_assert(sizeof(hipblasLtMatrixTransformDescOpaque_t) == 8 * sizeof(std::uint64_t));

// ────────────────────────────────────────────────────────────────────────
// Algorithm and heuristic result struct traits
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_standard_layout_v<hipblasLtMatmulAlgo_t>);
static_assert(std::is_standard_layout_v<hipblasLtMatmulHeuristicResult_t>);

// ────────────────────────────────────────────────────────────────────────
// WWR_LINK_CHECK: library and handle management
// ────────────────────────────────────────────────────────────────────────

WWR_LINK_CHECK(hipblasLtGetVersion)
WWR_LINK_CHECK(hipblasLtGetGitRevision)
WWR_LINK_CHECK(hipblasLtGetArchName)
WWR_LINK_CHECK(hipblasLtCreate)
WWR_LINK_CHECK(hipblasLtDestroy)

// ────────────────────────────────────────────────────────────────────────
// WWR_LINK_CHECK: matrix layout descriptor
// ────────────────────────────────────────────────────────────────────────

WWR_LINK_CHECK(hipblasLtMatrixLayoutCreate)
WWR_LINK_CHECK(hipblasLtMatrixLayoutDestroy)
WWR_LINK_CHECK(hipblasLtMatrixLayoutSetAttribute)
WWR_LINK_CHECK(hipblasLtMatrixLayoutGetAttribute)

// ────────────────────────────────────────────────────────────────────────
// WWR_LINK_CHECK: matmul descriptor
// ────────────────────────────────────────────────────────────────────────

WWR_LINK_CHECK(hipblasLtMatmulDescCreate)
WWR_LINK_CHECK(hipblasLtMatmulDescDestroy)
WWR_LINK_CHECK(hipblasLtMatmulDescSetAttribute)
WWR_LINK_CHECK(hipblasLtMatmulDescGetAttribute)

// ────────────────────────────────────────────────────────────────────────
// WWR_LINK_CHECK: matmul preference descriptor
// ────────────────────────────────────────────────────────────────────────

WWR_LINK_CHECK(hipblasLtMatmulPreferenceCreate)
WWR_LINK_CHECK(hipblasLtMatmulPreferenceDestroy)
WWR_LINK_CHECK(hipblasLtMatmulPreferenceSetAttribute)
WWR_LINK_CHECK(hipblasLtMatmulPreferenceGetAttribute)

// ────────────────────────────────────────────────────────────────────────
// WWR_LINK_CHECK: heuristic search and matmul execution
// ────────────────────────────────────────────────────────────────────────

WWR_LINK_CHECK(hipblasLtMatmulAlgoGetHeuristic)
WWR_LINK_CHECK(hipblasLtMatmul)

// ────────────────────────────────────────────────────────────────────────
// WWR_LINK_CHECK: matrix transform descriptor and execution
// ────────────────────────────────────────────────────────────────────────

WWR_LINK_CHECK(hipblasLtMatrixTransformDescCreate)
WWR_LINK_CHECK(hipblasLtMatrixTransformDescDestroy)
WWR_LINK_CHECK(hipblasLtMatrixTransformDescSetAttribute)
WWR_LINK_CHECK(hipblasLtMatrixTransformDescGetAttribute)
WWR_LINK_CHECK(hipblasLtMatrixTransform)

} // namespace wwr::hip::test
