// cublasLt.cppm - Compile-time tests for gpumod.cuda.cublasLt

module;

#include "test/shared/link_check.h"

export module gpumod.test.cuda.cublasLt;

import std;
import gpumod.cuda.cublasLt;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time tests for gpumod.cuda.cublasLt
//
// We verify at compile-time that:
//   1. Constexpr numerical impl flag values are correct
//   2. Key enum types exist (std::is_enum_v)
//   3. Key enum enumerator values with stable ABI values are correct
//   4. Opaque handle types are pointers
//   5. Semi-opaque descriptor struct types satisfy type traits
//   6. Result struct satisfies type traits
//   7. WWR_LINK_CHECK for all exported functions
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace wwr::cuda::test {

using namespace wwr::cuda;

// ────────────────────────────────────────────────────────────────────────
// Constexpr numerical implementation flag values
// ────────────────────────────────────────────────────────────────────────

static_assert(CUBLASLT_NUMERICAL_IMPL_FLAGS_FMA == (0x01ull << 0));
static_assert(CUBLASLT_NUMERICAL_IMPL_FLAGS_HMMA == (0x02ull << 0));
static_assert(CUBLASLT_NUMERICAL_IMPL_FLAGS_IMMA == (0x04ull << 0));
static_assert(CUBLASLT_NUMERICAL_IMPL_FLAGS_DMMA == (0x08ull << 0));
static_assert(CUBLASLT_NUMERICAL_IMPL_FLAGS_TENSOR_OP_MASK == (0xfeull << 0));
static_assert(CUBLASLT_NUMERICAL_IMPL_FLAGS_OP_TYPE_MASK == (0xffull << 0));

static_assert(CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_16F == (0x01ull << 8));
static_assert(CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32F == (0x02ull << 8));
static_assert(CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_64F == (0x04ull << 8));
static_assert(CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32I == (0x08ull << 8));
static_assert(CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_TYPE_MASK == (0xffull << 8));

static_assert(CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_16F == (0x01ull << 16));
static_assert(CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_16BF == (0x02ull << 16));
static_assert(CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_TF32 == (0x04ull << 16));
static_assert(CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_32F == (0x08ull << 16));
static_assert(CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_64F == (0x10ull << 16));
static_assert(CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_8I == (0x20ull << 16));
static_assert(CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_8F_E4M3 == (0x40ull << 16));
static_assert(CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_8F_E5M2 == (0x80ull << 16));
static_assert(CUBLASLT_NUMERICAL_IMPL_FLAGS_OP_INPUT_TYPE_MASK == (0xffull << 16));

static_assert(CUBLASLT_NUMERICAL_IMPL_FLAGS_GAUSSIAN == (0x01ull << 32));

// ────────────────────────────────────────────────────────────────────────
// Enum type checks
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_enum_v<cublasLtMatmulTile_t>);
static_assert(std::is_enum_v<cublasLtMatmulStages_t>);
static_assert(std::is_enum_v<cublasLtClusterShape_t>);
static_assert(std::is_enum_v<cublasLtMatmulInnerShape_t>);
static_assert(std::is_enum_v<cublasLtMatmulMatrixScale_t>);
static_assert(std::is_enum_v<cublasLtPointerMode_t>);
static_assert(std::is_enum_v<cublasLtPointerModeMask_t>);
static_assert(std::is_enum_v<cublasLtOrder_t>);
static_assert(std::is_enum_v<cublasLtBatchMode_t>);
static_assert(std::is_enum_v<cublasLtMatrixLayoutAttribute_t>);
static_assert(std::is_enum_v<cublasLtMatmulDescAttributes_t>);
static_assert(std::is_enum_v<cublasLtMatrixTransformDescAttributes_t>);
static_assert(std::is_enum_v<cublasLtReductionScheme_t>);
static_assert(std::is_enum_v<cublasLtEpilogue_t>);
static_assert(std::is_enum_v<cublasLtMatmulSearch_t>);
static_assert(std::is_enum_v<cublasLtMatmulPreferenceAttributes_t>);
static_assert(std::is_enum_v<cublasLtMatmulAlgoCapAttributes_t>);
static_assert(std::is_enum_v<cublasLtMatmulAlgoConfigAttributes_t>);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cublasLtMatmulTile_t (first few stable values)
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUBLASLT_MATMUL_TILE_UNDEFINED) == 0);
static_assert(static_cast<int>(CUBLASLT_MATMUL_TILE_8x8) == 1);
static_assert(static_cast<int>(CUBLASLT_MATMUL_TILE_8x16) == 2);
static_assert(static_cast<int>(CUBLASLT_MATMUL_TILE_16x8) == 3);
static_assert(static_cast<int>(CUBLASLT_MATMUL_TILE_8x32) == 4);
static_assert(static_cast<int>(CUBLASLT_MATMUL_TILE_16x16) == 5);
static_assert(static_cast<int>(CUBLASLT_MATMUL_TILE_32x8) == 6);
static_assert(static_cast<int>(CUBLASLT_MATMUL_TILE_8x64) == 7);
static_assert(static_cast<int>(CUBLASLT_MATMUL_TILE_16x32) == 8);
static_assert(static_cast<int>(CUBLASLT_MATMUL_TILE_32x16) == 9);
static_assert(static_cast<int>(CUBLASLT_MATMUL_TILE_64x8) == 10);
static_assert(static_cast<int>(CUBLASLT_MATMUL_TILE_32x32) == 11);
static_assert(static_cast<int>(CUBLASLT_MATMUL_TILE_32x64) == 12);
static_assert(static_cast<int>(CUBLASLT_MATMUL_TILE_64x32) == 13);
static_assert(static_cast<int>(CUBLASLT_MATMUL_TILE_32x128) == 14);
static_assert(static_cast<int>(CUBLASLT_MATMUL_TILE_64x64) == 15);
static_assert(static_cast<int>(CUBLASLT_MATMUL_TILE_128x32) == 16);
static_assert(static_cast<int>(CUBLASLT_MATMUL_TILE_64x128) == 17);
static_assert(static_cast<int>(CUBLASLT_MATMUL_TILE_128x64) == 18);
static_assert(static_cast<int>(CUBLASLT_MATMUL_TILE_64x256) == 19);
static_assert(static_cast<int>(CUBLASLT_MATMUL_TILE_128x128) == 20);
static_assert(static_cast<int>(CUBLASLT_MATMUL_TILE_256x64) == 21);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cublasLtMatmulStages_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUBLASLT_MATMUL_STAGES_UNDEFINED) == 0);
static_assert(static_cast<int>(CUBLASLT_MATMUL_STAGES_16x1) == 1);
static_assert(static_cast<int>(CUBLASLT_MATMUL_STAGES_16x2) == 2);
static_assert(static_cast<int>(CUBLASLT_MATMUL_STAGES_16x3) == 3);
static_assert(static_cast<int>(CUBLASLT_MATMUL_STAGES_32x1) == 7);
static_assert(static_cast<int>(CUBLASLT_MATMUL_STAGES_64x1) == 13);
static_assert(static_cast<int>(CUBLASLT_MATMUL_STAGES_128x1) == 19);
static_assert(static_cast<int>(CUBLASLT_MATMUL_STAGES_32x10) == 25);
static_assert(static_cast<int>(CUBLASLT_MATMUL_STAGES_8x4) == 26);
static_assert(static_cast<int>(CUBLASLT_MATMUL_STAGES_8x3) == 31);
static_assert(static_cast<int>(CUBLASLT_MATMUL_STAGES_8xAUTO) == 32);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cublasLtClusterShape_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUBLASLT_CLUSTER_SHAPE_AUTO) == 0);
static_assert(static_cast<int>(CUBLASLT_CLUSTER_SHAPE_1x1x1) == 2);
static_assert(static_cast<int>(CUBLASLT_CLUSTER_SHAPE_2x1x1) == 3);
static_assert(static_cast<int>(CUBLASLT_CLUSTER_SHAPE_4x1x1) == 4);
static_assert(static_cast<int>(CUBLASLT_CLUSTER_SHAPE_1x2x1) == 5);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cublasLtMatmulInnerShape_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUBLASLT_MATMUL_INNER_SHAPE_UNDEFINED) == 0);
static_assert(static_cast<int>(CUBLASLT_MATMUL_INNER_SHAPE_MMA884) == 1);
static_assert(static_cast<int>(CUBLASLT_MATMUL_INNER_SHAPE_MMA1684) == 2);
static_assert(static_cast<int>(CUBLASLT_MATMUL_INNER_SHAPE_MMA1688) == 3);
static_assert(static_cast<int>(CUBLASLT_MATMUL_INNER_SHAPE_MMA16816) == 4);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cublasLtMatmulMatrixScale_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F) == 0);
static_assert(static_cast<int>(CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3) == 1);
static_assert(static_cast<int>(CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0) == 2);
static_assert(static_cast<int>(CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F) == 3);
static_assert(static_cast<int>(CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F) == 4);
static_assert(static_cast<int>(CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F) == 5);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cublasLtPointerMode_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUBLASLT_POINTER_MODE_DEVICE_VECTOR) == 2);
static_assert(static_cast<int>(CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO) == 3);
static_assert(static_cast<int>(CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST) == 4);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cublasLtPointerModeMask_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUBLASLT_POINTER_MODE_MASK_HOST) == 1);
static_assert(static_cast<int>(CUBLASLT_POINTER_MODE_MASK_DEVICE) == 2);
static_assert(static_cast<int>(CUBLASLT_POINTER_MODE_MASK_DEVICE_VECTOR) == 4);
static_assert(static_cast<int>(CUBLASLT_POINTER_MODE_MASK_ALPHA_DEVICE_VECTOR_BETA_ZERO) == 8);
static_assert(static_cast<int>(CUBLASLT_POINTER_MODE_MASK_ALPHA_DEVICE_VECTOR_BETA_HOST) == 16);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cublasLtOrder_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUBLASLT_ORDER_COL) == 0);
static_assert(static_cast<int>(CUBLASLT_ORDER_ROW) == 1);
static_assert(static_cast<int>(CUBLASLT_ORDER_COL32) == 2);
static_assert(static_cast<int>(CUBLASLT_ORDER_COL4_4R2_8C) == 3);
static_assert(static_cast<int>(CUBLASLT_ORDER_COL32_2R_4R4) == 4);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cublasLtBatchMode_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUBLASLT_BATCH_MODE_STRIDED) == 0);
static_assert(static_cast<int>(CUBLASLT_BATCH_MODE_POINTER_ARRAY) == 1);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cublasLtMatrixLayoutAttribute_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUBLASLT_MATRIX_LAYOUT_TYPE) == 0);
static_assert(static_cast<int>(CUBLASLT_MATRIX_LAYOUT_ORDER) == 1);
static_assert(static_cast<int>(CUBLASLT_MATRIX_LAYOUT_ROWS) == 2);
static_assert(static_cast<int>(CUBLASLT_MATRIX_LAYOUT_COLS) == 3);
static_assert(static_cast<int>(CUBLASLT_MATRIX_LAYOUT_LD) == 4);
static_assert(static_cast<int>(CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT) == 5);
static_assert(static_cast<int>(CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET) == 6);
static_assert(static_cast<int>(CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET) == 7);
static_assert(static_cast<int>(CUBLASLT_MATRIX_LAYOUT_BATCH_MODE) == 8);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cublasLtMatmulDescAttributes_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUBLASLT_MATMUL_DESC_COMPUTE_TYPE) == 0);
static_assert(static_cast<int>(CUBLASLT_MATMUL_DESC_SCALE_TYPE) == 1);
static_assert(static_cast<int>(CUBLASLT_MATMUL_DESC_POINTER_MODE) == 2);
static_assert(static_cast<int>(CUBLASLT_MATMUL_DESC_TRANSA) == 3);
static_assert(static_cast<int>(CUBLASLT_MATMUL_DESC_TRANSB) == 4);
static_assert(static_cast<int>(CUBLASLT_MATMUL_DESC_TRANSC) == 5);
static_assert(static_cast<int>(CUBLASLT_MATMUL_DESC_FILL_MODE) == 6);
static_assert(static_cast<int>(CUBLASLT_MATMUL_DESC_EPILOGUE) == 7);
static_assert(static_cast<int>(CUBLASLT_MATMUL_DESC_BIAS_POINTER) == 8);
static_assert(static_cast<int>(CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE) == 10);
static_assert(static_cast<int>(CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER) == 11);
static_assert(static_cast<int>(CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD) == 12);
static_assert(static_cast<int>(CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE) == 13);
static_assert(static_cast<int>(CUBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE) == 14);
static_assert(static_cast<int>(CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET) == 15);
static_assert(static_cast<int>(CUBLASLT_MATMUL_DESC_A_SCALE_POINTER) == 17);
static_assert(static_cast<int>(CUBLASLT_MATMUL_DESC_B_SCALE_POINTER) == 18);
static_assert(static_cast<int>(CUBLASLT_MATMUL_DESC_C_SCALE_POINTER) == 19);
static_assert(static_cast<int>(CUBLASLT_MATMUL_DESC_D_SCALE_POINTER) == 20);
static_assert(static_cast<int>(CUBLASLT_MATMUL_DESC_AMAX_D_POINTER) == 21);
static_assert(static_cast<int>(CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE) == 22);
static_assert(static_cast<int>(CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER) == 23);
static_assert(static_cast<int>(CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_AMAX_POINTER) == 24);
static_assert(static_cast<int>(CUBLASLT_MATMUL_DESC_FAST_ACCUM) == 25);
static_assert(static_cast<int>(CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE) == 26);
static_assert(static_cast<int>(CUBLASLT_MATMUL_DESC_A_SCALE_MODE) == 31);
static_assert(static_cast<int>(CUBLASLT_MATMUL_DESC_B_SCALE_MODE) == 32);
static_assert(static_cast<int>(CUBLASLT_MATMUL_DESC_C_SCALE_MODE) == 33);
static_assert(static_cast<int>(CUBLASLT_MATMUL_DESC_D_SCALE_MODE) == 34);
static_assert(static_cast<int>(CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_MODE) == 35);
static_assert(static_cast<int>(CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER) == 36);
static_assert(static_cast<int>(CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE) == 37);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cublasLtMatrixTransformDescAttributes_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE) == 0);
static_assert(static_cast<int>(CUBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE) == 1);
static_assert(static_cast<int>(CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA) == 2);
static_assert(static_cast<int>(CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSB) == 3);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cublasLtReductionScheme_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUBLASLT_REDUCTION_SCHEME_NONE) == 0);
static_assert(static_cast<int>(CUBLASLT_REDUCTION_SCHEME_INPLACE) == 1);
static_assert(static_cast<int>(CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE) == 2);
static_assert(static_cast<int>(CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE) == 4);
static_assert(static_cast<int>(CUBLASLT_REDUCTION_SCHEME_MASK) == 0x7);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cublasLtEpilogue_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUBLASLT_EPILOGUE_DEFAULT) == 1);
static_assert(static_cast<int>(CUBLASLT_EPILOGUE_RELU) == 2);
static_assert(static_cast<int>(CUBLASLT_EPILOGUE_BIAS) == 4);
static_assert(static_cast<int>(CUBLASLT_EPILOGUE_GELU) == 32);
static_assert(static_cast<int>(CUBLASLT_EPILOGUE_BGRADA) == 256);
static_assert(static_cast<int>(CUBLASLT_EPILOGUE_BGRADB) == 512);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cublasLtMatmulSearch_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUBLASLT_SEARCH_BEST_FIT) == 0);
static_assert(static_cast<int>(CUBLASLT_SEARCH_LIMITED_BY_ALGO_ID) == 1);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cublasLtMatmulPreferenceAttributes_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUBLASLT_MATMUL_PREF_SEARCH_MODE) == 0);
static_assert(static_cast<int>(CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES) == 1);
static_assert(static_cast<int>(CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK) == 3);
static_assert(static_cast<int>(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES) == 5);
static_assert(static_cast<int>(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES) == 6);
static_assert(static_cast<int>(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES) == 7);
static_assert(static_cast<int>(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES) == 8);
static_assert(static_cast<int>(CUBLASLT_MATMUL_PREF_MAX_WAVES_COUNT) == 9);
static_assert(static_cast<int>(CUBLASLT_MATMUL_PREF_IMPL_MASK) == 12);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cublasLtMatmulAlgoCapAttributes_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUBLASLT_ALGO_CAP_SPLITK_SUPPORT) == 0);
static_assert(static_cast<int>(CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK) == 1);
static_assert(static_cast<int>(CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT) == 2);
static_assert(static_cast<int>(CUBLASLT_ALGO_CAP_STRIDED_BATCH_SUPPORT) == 3);
static_assert(static_cast<int>(CUBLASLT_ALGO_CAP_OUT_OF_PLACE_RESULT_SUPPORT) == 4);
static_assert(static_cast<int>(CUBLASLT_ALGO_CAP_UPLO_SUPPORT) == 5);
static_assert(static_cast<int>(CUBLASLT_ALGO_CAP_TILE_IDS) == 6);
static_assert(static_cast<int>(CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX) == 7);
static_assert(static_cast<int>(CUBLASLT_ALGO_CAP_CUSTOM_MEMORY_ORDER) == 10);
static_assert(static_cast<int>(CUBLASLT_ALGO_CAP_POINTER_MODE_MASK) == 11);
static_assert(static_cast<int>(CUBLASLT_ALGO_CAP_EPILOGUE_MASK) == 12);
static_assert(static_cast<int>(CUBLASLT_ALGO_CAP_STAGES_IDS) == 13);
static_assert(static_cast<int>(CUBLASLT_ALGO_CAP_LD_NEGATIVE) == 14);
static_assert(static_cast<int>(CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS) == 15);
static_assert(static_cast<int>(CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_A_BYTES) == 16);
static_assert(static_cast<int>(CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_B_BYTES) == 17);
static_assert(static_cast<int>(CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_C_BYTES) == 18);
static_assert(static_cast<int>(CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_D_BYTES) == 19);
static_assert(static_cast<int>(CUBLASLT_ALGO_CAP_POINTER_ARRAY_BATCH_SUPPORT) == 21);
static_assert(static_cast<int>(CUBLASLT_ALGO_CAP_FLOATING_POINT_EMULATION_SUPPORT) == 22);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cublasLtMatmulAlgoConfigAttributes_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CUBLASLT_ALGO_CONFIG_ID) == 0);
static_assert(static_cast<int>(CUBLASLT_ALGO_CONFIG_TILE_ID) == 1);
static_assert(static_cast<int>(CUBLASLT_ALGO_CONFIG_SPLITK_NUM) == 2);
static_assert(static_cast<int>(CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME) == 3);
static_assert(static_cast<int>(CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING) == 4);
static_assert(static_cast<int>(CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION) == 5);
static_assert(static_cast<int>(CUBLASLT_ALGO_CONFIG_STAGES_ID) == 6);
static_assert(static_cast<int>(CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID) == 7);
static_assert(static_cast<int>(CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID) == 8);

// ────────────────────────────────────────────────────────────────────────
// Handle type traits (opaque pointer types)
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_pointer_v<cublasLtHandle_t>);
static_assert(std::is_pointer_v<cublasLtMatrixLayout_t>);
static_assert(std::is_pointer_v<cublasLtMatmulDesc_t>);
static_assert(std::is_pointer_v<cublasLtMatrixTransformDesc_t>);
static_assert(std::is_pointer_v<cublasLtMatmulPreference_t>);

// ────────────────────────────────────────────────────────────────────────
// Semi-opaque descriptor struct type traits
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_trivially_copyable_v<cublasLtMatrixLayoutOpaque_t>);
static_assert(std::is_standard_layout_v<cublasLtMatrixLayoutOpaque_t>);
static_assert(sizeof(cublasLtMatrixLayoutOpaque_t) == 8 * sizeof(std::uint64_t));

static_assert(std::is_trivially_copyable_v<cublasLtMatmulAlgo_t>);
static_assert(std::is_standard_layout_v<cublasLtMatmulAlgo_t>);
static_assert(sizeof(cublasLtMatmulAlgo_t) == 8 * sizeof(std::uint64_t));

static_assert(std::is_trivially_copyable_v<cublasLtMatmulDescOpaque_t>);
static_assert(std::is_standard_layout_v<cublasLtMatmulDescOpaque_t>);
static_assert(sizeof(cublasLtMatmulDescOpaque_t) == 32 * sizeof(std::uint64_t));

static_assert(std::is_trivially_copyable_v<cublasLtMatrixTransformDescOpaque_t>);
static_assert(std::is_standard_layout_v<cublasLtMatrixTransformDescOpaque_t>);
static_assert(sizeof(cublasLtMatrixTransformDescOpaque_t) == 8 * sizeof(std::uint64_t));

static_assert(std::is_trivially_copyable_v<cublasLtMatmulPreferenceOpaque_t>);
static_assert(std::is_standard_layout_v<cublasLtMatmulPreferenceOpaque_t>);
static_assert(sizeof(cublasLtMatmulPreferenceOpaque_t) == 8 * sizeof(std::uint64_t));

// ────────────────────────────────────────────────────────────────────────
// Result struct type traits
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_trivially_copyable_v<cublasLtMatmulHeuristicResult_t>);
static_assert(std::is_standard_layout_v<cublasLtMatmulHeuristicResult_t>);

// ────────────────────────────────────────────────────────────────────────
// WWR_LINK_CHECK: context management
// ────────────────────────────────────────────────────────────────────────

WWR_LINK_CHECK(cublasLtCreate)
WWR_LINK_CHECK(cublasLtDestroy)
WWR_LINK_CHECK(cublasLtGetStatusName)
WWR_LINK_CHECK(cublasLtGetStatusString)
WWR_LINK_CHECK(cublasLtGetVersion)
WWR_LINK_CHECK(cublasLtGetCudartVersion)
WWR_LINK_CHECK(cublasLtGetProperty)
WWR_LINK_CHECK(cublasLtHeuristicsCacheGetCapacity)
WWR_LINK_CHECK(cublasLtHeuristicsCacheSetCapacity)
// Declared in cublasLt.h but not exported by libcublasLt.so (13.0.0.19), nor by
// libcublas.so; it appears only in libcublasLt_static.a. Calling it from a
// program linked against the shared library fails to link, so only the
// declaration is checked.
WWR_DECLARED_CHECK(cublasLtDisableCpuInstructionsSetMask)

// ────────────────────────────────────────────────────────────────────────
// WWR_LINK_CHECK: core computation
// ────────────────────────────────────────────────────────────────────────

WWR_LINK_CHECK(cublasLtMatmul)
WWR_LINK_CHECK(cublasLtMatrixTransform)

// ────────────────────────────────────────────────────────────────────────
// WWR_LINK_CHECK: matrix layout descriptor
// ────────────────────────────────────────────────────────────────────────

WWR_LINK_CHECK(cublasLtMatrixLayoutInit_internal)
WWR_LINK_CHECK(cublasLtMatrixLayoutCreate)
WWR_LINK_CHECK(cublasLtMatrixLayoutDestroy)
WWR_LINK_CHECK(cublasLtMatrixLayoutSetAttribute)
WWR_LINK_CHECK(cublasLtMatrixLayoutGetAttribute)

// ────────────────────────────────────────────────────────────────────────
// WWR_LINK_CHECK: matmul descriptor
// ────────────────────────────────────────────────────────────────────────

WWR_LINK_CHECK(cublasLtMatmulDescInit_internal)
WWR_LINK_CHECK(cublasLtMatmulDescCreate)
WWR_LINK_CHECK(cublasLtMatmulDescDestroy)
WWR_LINK_CHECK(cublasLtMatmulDescSetAttribute)
WWR_LINK_CHECK(cublasLtMatmulDescGetAttribute)

// ────────────────────────────────────────────────────────────────────────
// WWR_LINK_CHECK: matrix transform descriptor
// ────────────────────────────────────────────────────────────────────────

WWR_LINK_CHECK(cublasLtMatrixTransformDescInit_internal)
WWR_LINK_CHECK(cublasLtMatrixTransformDescCreate)
WWR_LINK_CHECK(cublasLtMatrixTransformDescDestroy)
WWR_LINK_CHECK(cublasLtMatrixTransformDescSetAttribute)
WWR_LINK_CHECK(cublasLtMatrixTransformDescGetAttribute)

// ────────────────────────────────────────────────────────────────────────
// WWR_LINK_CHECK: matmul preference descriptor
// ────────────────────────────────────────────────────────────────────────

WWR_LINK_CHECK(cublasLtMatmulPreferenceInit_internal)
WWR_LINK_CHECK(cublasLtMatmulPreferenceCreate)
WWR_LINK_CHECK(cublasLtMatmulPreferenceDestroy)
WWR_LINK_CHECK(cublasLtMatmulPreferenceSetAttribute)
WWR_LINK_CHECK(cublasLtMatmulPreferenceGetAttribute)

// ────────────────────────────────────────────────────────────────────────
// WWR_LINK_CHECK: heuristic and algorithm functions
// ────────────────────────────────────────────────────────────────────────

WWR_LINK_CHECK(cublasLtMatmulAlgoGetHeuristic)
WWR_LINK_CHECK(cublasLtMatmulAlgoGetIds)
WWR_LINK_CHECK(cublasLtMatmulAlgoInit)
WWR_LINK_CHECK(cublasLtMatmulAlgoCheck)
WWR_LINK_CHECK(cublasLtMatmulAlgoCapGetAttribute)
WWR_LINK_CHECK(cublasLtMatmulAlgoConfigSetAttribute)
WWR_LINK_CHECK(cublasLtMatmulAlgoConfigGetAttribute)

// ────────────────────────────────────────────────────────────────────────
// WWR_LINK_CHECK: logger functions
// ────────────────────────────────────────────────────────────────────────

WWR_LINK_CHECK(cublasLtLoggerSetCallback)
WWR_LINK_CHECK(cublasLtLoggerSetFile)
WWR_LINK_CHECK(cublasLtLoggerOpenFile)
WWR_LINK_CHECK(cublasLtLoggerSetLevel)
WWR_LINK_CHECK(cublasLtLoggerSetMask)
WWR_LINK_CHECK(cublasLtLoggerForceDisable)

} // namespace wwr::cuda::test
