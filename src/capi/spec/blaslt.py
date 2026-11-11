"""The `blaslt` (cuBLASLt / hipBLASLt) surface as a FRAGMENT-SOURCED spec.

The modern "Lt" GEMM library (detail/blaslt_names.h: the handle + matmul /
matrix-layout / preference / transform descriptors, their attribute enums, the
epilogue / order / pointer-mode / matrix-scale enums, and 21 functions). Same
SHAPE as blas (spec/blas.py): a status enum returned by value, opaque handles,
by-value value enums -- so it registers through the same captypes machinery and
the generic fragment-sourced library path emits it.

  * HANDLES  the cuBLASLt library handle and the FIVE opaque descriptors
    (matmul / matrix-layout / preference / matrix-transform) -> void*, plus the
    cudaStream_t wwrblasLtMatmul / wwrblasLtMatrixTransform pass. (The
    heuristic-result and algo STRUCTS appear only as out/opaque pointers; the
    generic pointer arm erases a vendor pointee to void*, which is right for an
    opaque result record / pre-selected algo blob a caller only hands back.)
  * VALUE ENUMS  wwrblasLtStatus_t (returned by every function),
    wwrblasLtComputeType_t (MatmulDescCreate -- its raw backend values DIVERGE, so
    it is a stable capi enum, mapped to the WWRBLASLT_COMPUTE_* constants the
    fragment carries), and the by-value ATTRIBUTE enums (desc / layout / pref /
    transform attributes) + epilogue / matrix-scale / order / pointer-mode.

The data-type parameters (cudaDataType / cudaDataType_t -- scaleType, matrix
element type) are NOT a stable enum: cudaDataType and hipDataType share their
enumerator INTEGER values bit-for-bit (library_types.h and hip/library_types.h:
R_32F==0, R_64F==1, R_16F==2, C_32F==4 ... through the narrow-float family), so a
raw `int` crosses the ABI backend-neutrally AND carries every value (including the
exotic FP8/FP6/FP4 types a fixed enum would lose). They are registered as a
SCALAR ALIAS to int. See spec/blas.py and docs/architecture.md section 25.
"""

from __future__ import annotations

from captypes import Handle, ValueEnum
from library import LibrarySpec, snake_capi_name

FRAGMENT = "detail/blaslt_names.h"
PREFIX = "wwrblasLt"
CAPI_PREFIX = "wwr_blaslt_"

CUDA_MANIFEST = "cuda-13.0.x/cublasLt.json"
HIP_MANIFEST = "rocm-10.0.0/hipblaslt.json"
MODULES = ["wwr.blaslt"]

# The library handle, the five opaque descriptors, and the stream. Each -> void*.
HANDLES = [
    Handle(vendor_types=("cublasLtHandle_t", "hipblasLtHandle_t"),
           capi_type="wwr_blaslt_handle"),
    Handle(vendor_types=("cublasLtMatmulDesc_t", "hipblasLtMatmulDesc_t"),
           capi_type="wwr_blaslt_matmul_desc"),
    Handle(vendor_types=("cublasLtMatrixLayout_t", "hipblasLtMatrixLayout_t"),
           capi_type="wwr_blaslt_matrix_layout"),
    Handle(vendor_types=("cublasLtMatmulPreference_t",
                         "hipblasLtMatmulPreference_t"),
           capi_type="wwr_blaslt_matmul_preference"),
    Handle(vendor_types=("cublasLtMatrixTransformDesc_t",
                         "hipblasLtMatrixTransformDesc_t"),
           capi_type="wwr_blaslt_matrix_transform_desc"),
    Handle(vendor_types=("cudaStream_t", "hipStream_t"),
           capi_type="wwr_stream"),
]

POINTEE_PODS: list = []
PODS: list = []

STATUS = ValueEnum(
    vendor_types=("cublasStatus_t", "hipblasStatus_t"),
    capi_type="wwr_blaslt_status",
    wwr_type="wwrblasLtStatus_t",
    fallback="WWR_BLASLT_STATUS_UNKNOWN",
    entries=(
        ("WWR_BLASLT_STATUS_SUCCESS", "WWRBLASLT_STATUS_SUCCESS"),
        ("WWR_BLASLT_STATUS_NOT_INITIALIZED",
         "WWRBLASLT_STATUS_NOT_INITIALIZED"),
        ("WWR_BLASLT_STATUS_ALLOC_FAILED", "WWRBLASLT_STATUS_ALLOC_FAILED"),
        ("WWR_BLASLT_STATUS_ARCH_MISMATCH", "WWRBLASLT_STATUS_ARCH_MISMATCH"),
        ("WWR_BLASLT_STATUS_EXECUTION_FAILED",
         "WWRBLASLT_STATUS_EXECUTION_FAILED"),
        ("WWR_BLASLT_STATUS_INTERNAL_ERROR", "WWRBLASLT_STATUS_INTERNAL_ERROR"),
        ("WWR_BLASLT_STATUS_INVALID_VALUE", "WWRBLASLT_STATUS_INVALID_VALUE"),
        ("WWR_BLASLT_STATUS_MAPPING_ERROR", "WWRBLASLT_STATUS_MAPPING_ERROR"),
        ("WWR_BLASLT_STATUS_NOT_SUPPORTED", "WWRBLASLT_STATUS_NOT_SUPPORTED"),
    ),
)

COMPUTE_TYPE = ValueEnum(
    vendor_types=("cublasComputeType_t", "hipblasComputeType_t"),
    capi_type="wwr_blaslt_compute_type",
    wwr_type="wwrblasLtComputeType_t",
    fallback="WWR_BLASLT_COMPUTE_32F",
    entries=(
        ("WWR_BLASLT_COMPUTE_16F", "WWRBLASLT_COMPUTE_16F"),
        ("WWR_BLASLT_COMPUTE_32F", "WWRBLASLT_COMPUTE_32F"),
        ("WWR_BLASLT_COMPUTE_64F", "WWRBLASLT_COMPUTE_64F"),
    ),
)

EPILOGUE = ValueEnum(
    vendor_types=("cublasLtEpilogue_t", "hipblasLtEpilogue_t"),
    capi_type="wwr_blaslt_epilogue",
    wwr_type="wwrblasLtEpilogue_t",
    fallback="WWR_BLASLT_EPILOGUE_DEFAULT",
    entries=(
        ("WWR_BLASLT_EPILOGUE_DEFAULT", "WWRBLASLT_EPILOGUE_DEFAULT"),
        ("WWR_BLASLT_EPILOGUE_RELU", "WWRBLASLT_EPILOGUE_RELU"),
        ("WWR_BLASLT_EPILOGUE_RELU_AUX", "WWRBLASLT_EPILOGUE_RELU_AUX"),
        ("WWR_BLASLT_EPILOGUE_BIAS", "WWRBLASLT_EPILOGUE_BIAS"),
        ("WWR_BLASLT_EPILOGUE_RELU_BIAS", "WWRBLASLT_EPILOGUE_RELU_BIAS"),
        ("WWR_BLASLT_EPILOGUE_RELU_AUX_BIAS",
         "WWRBLASLT_EPILOGUE_RELU_AUX_BIAS"),
        ("WWR_BLASLT_EPILOGUE_DGELU", "WWRBLASLT_EPILOGUE_DGELU"),
        ("WWR_BLASLT_EPILOGUE_DGELU_BGRAD", "WWRBLASLT_EPILOGUE_DGELU_BGRAD"),
        ("WWR_BLASLT_EPILOGUE_GELU", "WWRBLASLT_EPILOGUE_GELU"),
        ("WWR_BLASLT_EPILOGUE_GELU_AUX", "WWRBLASLT_EPILOGUE_GELU_AUX"),
        ("WWR_BLASLT_EPILOGUE_GELU_BIAS", "WWRBLASLT_EPILOGUE_GELU_BIAS"),
        ("WWR_BLASLT_EPILOGUE_GELU_AUX_BIAS",
         "WWRBLASLT_EPILOGUE_GELU_AUX_BIAS"),
        ("WWR_BLASLT_EPILOGUE_BGRADA", "WWRBLASLT_EPILOGUE_BGRADA"),
        ("WWR_BLASLT_EPILOGUE_BGRADB", "WWRBLASLT_EPILOGUE_BGRADB"),
    ),
)

MATMUL_DESC_ATTRS = ValueEnum(
    vendor_types=("cublasLtMatmulDescAttributes_t",
                  "hipblasLtMatmulDescAttributes_t"),
    capi_type="wwr_blaslt_matmul_desc_attributes",
    wwr_type="wwrblasLtMatmulDescAttributes_t",
    fallback="WWR_BLASLT_MATMUL_DESC_TRANSA",
    entries=(
        ("WWR_BLASLT_MATMUL_DESC_TRANSA", "WWRBLASLT_MATMUL_DESC_TRANSA"),
        ("WWR_BLASLT_MATMUL_DESC_TRANSB", "WWRBLASLT_MATMUL_DESC_TRANSB"),
        ("WWR_BLASLT_MATMUL_DESC_EPILOGUE", "WWRBLASLT_MATMUL_DESC_EPILOGUE"),
        ("WWR_BLASLT_MATMUL_DESC_BIAS_POINTER",
         "WWRBLASLT_MATMUL_DESC_BIAS_POINTER"),
        ("WWR_BLASLT_MATMUL_DESC_BIAS_DATA_TYPE",
         "WWRBLASLT_MATMUL_DESC_BIAS_DATA_TYPE"),
        ("WWR_BLASLT_MATMUL_DESC_A_SCALE_POINTER",
         "WWRBLASLT_MATMUL_DESC_A_SCALE_POINTER"),
        ("WWR_BLASLT_MATMUL_DESC_B_SCALE_POINTER",
         "WWRBLASLT_MATMUL_DESC_B_SCALE_POINTER"),
        ("WWR_BLASLT_MATMUL_DESC_C_SCALE_POINTER",
         "WWRBLASLT_MATMUL_DESC_C_SCALE_POINTER"),
        ("WWR_BLASLT_MATMUL_DESC_D_SCALE_POINTER",
         "WWRBLASLT_MATMUL_DESC_D_SCALE_POINTER"),
        ("WWR_BLASLT_MATMUL_DESC_A_SCALE_MODE",
         "WWRBLASLT_MATMUL_DESC_A_SCALE_MODE"),
        ("WWR_BLASLT_MATMUL_DESC_B_SCALE_MODE",
         "WWRBLASLT_MATMUL_DESC_B_SCALE_MODE"),
        ("WWR_BLASLT_MATMUL_DESC_AMAX_D_POINTER",
         "WWRBLASLT_MATMUL_DESC_AMAX_D_POINTER"),
        ("WWR_BLASLT_MATMUL_DESC_POINTER_MODE",
         "WWRBLASLT_MATMUL_DESC_POINTER_MODE"),
        ("WWR_BLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER",
         "WWRBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER"),
        ("WWR_BLASLT_MATMUL_DESC_EPILOGUE_AUX_LD",
         "WWRBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD"),
        ("WWR_BLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE",
         "WWRBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE"),
        ("WWR_BLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE",
         "WWRBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE"),
        ("WWR_BLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER",
         "WWRBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER"),
    ),
)

MATMUL_MATRIX_SCALE = ValueEnum(
    vendor_types=("cublasLtMatmulMatrixScale_t",
                  "hipblasLtMatmulMatrixScale_t"),
    capi_type="wwr_blaslt_matmul_matrix_scale",
    wwr_type="wwrblasLtMatmulMatrixScale_t",
    fallback="WWR_BLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F",
    entries=(
        ("WWR_BLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F",
         "WWRBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F"),
        ("WWR_BLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3",
         "WWRBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3"),
        ("WWR_BLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0",
         "WWRBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0"),
        ("WWR_BLASLT_MATMUL_MATRIX_SCALE_VEC128_32F",
         "WWRBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F"),
        ("WWR_BLASLT_MATMUL_MATRIX_SCALE_BLK128X128_32F",
         "WWRBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F"),
        ("WWR_BLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F",
         "WWRBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F"),
        ("WWR_BLASLT_MATMUL_MATRIX_SCALE_END",
         "WWRBLASLT_MATMUL_MATRIX_SCALE_END"),
    ),
)

MATMUL_PREF_ATTRS = ValueEnum(
    vendor_types=("cublasLtMatmulPreferenceAttributes_t",
                  "hipblasLtMatmulPreferenceAttributes_t"),
    capi_type="wwr_blaslt_matmul_preference_attributes",
    wwr_type="wwrblasLtMatmulPreferenceAttributes_t",
    fallback="WWR_BLASLT_MATMUL_PREF_SEARCH_MODE",
    entries=(
        ("WWR_BLASLT_MATMUL_PREF_SEARCH_MODE",
         "WWRBLASLT_MATMUL_PREF_SEARCH_MODE"),
        ("WWR_BLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES",
         "WWRBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES"),
    ),
)

MATRIX_LAYOUT_ATTR = ValueEnum(
    vendor_types=("cublasLtMatrixLayoutAttribute_t",
                  "hipblasLtMatrixLayoutAttribute_t"),
    capi_type="wwr_blaslt_matrix_layout_attribute",
    wwr_type="wwrblasLtMatrixLayoutAttribute_t",
    fallback="WWR_BLASLT_MATRIX_LAYOUT_TYPE",
    entries=(
        ("WWR_BLASLT_MATRIX_LAYOUT_TYPE", "WWRBLASLT_MATRIX_LAYOUT_TYPE"),
        ("WWR_BLASLT_MATRIX_LAYOUT_ORDER", "WWRBLASLT_MATRIX_LAYOUT_ORDER"),
        ("WWR_BLASLT_MATRIX_LAYOUT_ROWS", "WWRBLASLT_MATRIX_LAYOUT_ROWS"),
        ("WWR_BLASLT_MATRIX_LAYOUT_COLS", "WWRBLASLT_MATRIX_LAYOUT_COLS"),
        ("WWR_BLASLT_MATRIX_LAYOUT_LD", "WWRBLASLT_MATRIX_LAYOUT_LD"),
        ("WWR_BLASLT_MATRIX_LAYOUT_BATCH_COUNT",
         "WWRBLASLT_MATRIX_LAYOUT_BATCH_COUNT"),
        ("WWR_BLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET",
         "WWRBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET"),
    ),
)

MATRIX_TRANSFORM_DESC_ATTRS = ValueEnum(
    vendor_types=("cublasLtMatrixTransformDescAttributes_t",
                  "hipblasLtMatrixTransformDescAttributes_t"),
    capi_type="wwr_blaslt_matrix_transform_desc_attributes",
    wwr_type="wwrblasLtMatrixTransformDescAttributes_t",
    fallback="WWR_BLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE",
    entries=(
        ("WWR_BLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE",
         "WWRBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE"),
        ("WWR_BLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE",
         "WWRBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE"),
        ("WWR_BLASLT_MATRIX_TRANSFORM_DESC_TRANSA",
         "WWRBLASLT_MATRIX_TRANSFORM_DESC_TRANSA"),
        ("WWR_BLASLT_MATRIX_TRANSFORM_DESC_TRANSB",
         "WWRBLASLT_MATRIX_TRANSFORM_DESC_TRANSB"),
    ),
)

ORDER = ValueEnum(
    vendor_types=("cublasLtOrder_t", "hipblasLtOrder_t"),
    capi_type="wwr_blaslt_order",
    wwr_type="wwrblasLtOrder_t",
    fallback="WWR_BLASLT_ORDER_COL",
    entries=(
        ("WWR_BLASLT_ORDER_COL", "WWRBLASLT_ORDER_COL"),
        ("WWR_BLASLT_ORDER_ROW", "WWRBLASLT_ORDER_ROW"),
    ),
)

POINTER_MODE = ValueEnum(
    vendor_types=("cublasLtPointerMode_t", "hipblasLtPointerMode_t"),
    capi_type="wwr_blaslt_pointer_mode",
    wwr_type="wwrblasLtPointerMode_t",
    fallback="WWR_BLASLT_POINTER_MODE_HOST",
    entries=(
        ("WWR_BLASLT_POINTER_MODE_HOST", "WWRBLASLT_POINTER_MODE_HOST"),
        ("WWR_BLASLT_POINTER_MODE_DEVICE", "WWRBLASLT_POINTER_MODE_DEVICE"),
        ("WWR_BLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST",
         "WWRBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST"),
    ),
)

VALUE_ENUMS = [STATUS, COMPUTE_TYPE, EPILOGUE, MATMUL_DESC_ATTRS,
               MATMUL_MATRIX_SCALE, MATMUL_PREF_ATTRS, MATRIX_LAYOUT_ATTR,
               MATRIX_TRANSFORM_DESC_ATTRS, ORDER, POINTER_MODE]

# Every blaslt value-enum constant is a blaslt fragment entry present in both
# blaslt manifests (the status + compute constants were added to blaslt_names.h,
# resolving against the vendor header blaslt.h pulls in), so none is exempt.
ENUM_CONSTANTS_NOT_IN_MANIFEST: set[str] = set()

# cudaDataType / hipDataType fold to a plain `int`: their enumerator integer
# values are IDENTICAL across backends (library_types.h), so a raw int is
# backend-neutral AND carries every value a fixed enum could not.
SCALAR_ALIASES = {
    "cudaDataType": "int",
    "cudaDataType_t": "int",
}

# No hand-written fragment forwarders: every blaslt function is a plain
# WWR_FUNCTION_RAW macro entry, all 21 wrapped.
EXPLICIT_SKIPS: list[tuple[str, str]] = []


def capi_name(wwr_name: str) -> str:
    return snake_capi_name(wwr_name, PREFIX, CAPI_PREFIX)


SPEC = LibrarySpec(
    name="blaslt",
    fragment=FRAGMENT,
    capi_prefix=CAPI_PREFIX,
    cuda_manifest=CUDA_MANIFEST,
    hip_manifest=HIP_MANIFEST,
    modules=MODULES,
    handles=HANDLES,
    pointee_pods=POINTEE_PODS,
    pods=PODS,
    value_enums=VALUE_ENUMS,
    enum_constants_not_in_manifest=ENUM_CONSTANTS_NOT_IN_MANIFEST,
    scalar_aliases=SCALAR_ALIASES,
    explicit_skips=EXPLICIT_SKIPS,
    capi_name=capi_name,
)
