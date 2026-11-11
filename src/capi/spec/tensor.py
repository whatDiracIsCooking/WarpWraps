"""The `tensor` (cuTENSOR / hipTENSOR) surface as a FRAGMENT-SOURCED spec.

The blas machine (spec/library.py) over detail/tensor_names.h: the handle, the
descriptor lifecycle (tensor / operation / plan / plan-preference descriptors),
the operation builders (contraction / elementwise / permutation / reduction), the
plan + workspace estimation, the execute calls (Contract / Reduce / Permute /
elementwise), the attribute get/set, the logger and plan-cache helpers. This
surface is DESCRIPTOR-HEAVY: every descriptor is an opaque vendor pointer, so it
REGISTERS them all as handles (-> void*), plus the enums its signatures name.

  * HANDLES  the library handle and FOUR opaque descriptors -- TensorDescriptor,
    OperationDescriptor, Plan, PlanPreference -- each a vendor pointer typedef
    passed by value and as an out-param; the stream; and the logger CALLBACK (a
    function-pointer typedef, opaque to the ABI). FILE* has no registration: the
    generic pointer arm erases it to void* already.
  * SCALAR ALIAS  the compute descriptor's NATURE diverges -- cuTENSOR's
    cutensorComputeDescriptor_t is an opaque POINTER (a global compute-descriptor
    object's address), hipTENSOR's is an ENUM. Neither a handle (void*) nor a
    value enum bridges both (static_cast<enum>(void*) is ill-formed on HIP), so it
    is folded to a pointer-width uint64_t the thunk reinterpret_casts to the CUDA
    pointer and static_casts to the HIP enum -- the one sound representation.
  * VALUE ENUMS  Status, Operator, Algo, JitMode, WorksizePreference, the three
    attribute enums (OperationDescriptor / PlanPreference / Plan), and the data
    type (cudaDataType_t -- CUDA aliases cutensorDataType_t to it; the neutral
    WWRTENSOR_R_*/C_* constants are that type, so the one map works both ways).

cudaDataType_t's enumerators are the CUTENSOR_R_*/C_* VALUE entries the fragment
carries (VarDecls present in the cutensor manifest), so none are exempt.
Everything tensor executes runs on the device; the thunks are link-checked on
no-GPU CI.
"""

from __future__ import annotations

from captypes import Handle, ValueEnum
from library import LibrarySpec, snake_capi_name

FRAGMENT = "detail/tensor_names.h"
PREFIX = "wwrtensor"
CAPI_PREFIX = "wwr_tensor_"

CUDA_MANIFEST = "cuda-13.0.x/cutensor.json"
HIP_MANIFEST = "rocm-10.0.0/hiptensor.json"
MODULES = ["wwr.tensor"]

HANDLES = [
    Handle(vendor_types=("cutensorHandle_t", "hiptensorHandle_t"),
           capi_type="wwr_tensor_handle"),
    Handle(vendor_types=("cutensorTensorDescriptor_t",
                         "hiptensorTensorDescriptor_t"),
           capi_type="wwr_tensor_tensor_descriptor"),
    Handle(vendor_types=("cutensorOperationDescriptor_t",
                         "hiptensorOperationDescriptor_t"),
           capi_type="wwr_tensor_operation_descriptor"),
    Handle(vendor_types=("cutensorPlan_t", "hiptensorPlan_t"),
           capi_type="wwr_tensor_plan"),
    Handle(vendor_types=("cutensorPlanPreference_t", "hiptensorPlanPreference_t"),
           capi_type="wwr_tensor_plan_preference"),
    Handle(vendor_types=("cutensorLoggerCallback_t", "hiptensorLoggerCallback_t"),
           capi_type="wwr_tensor_logger_callback"),
    Handle(vendor_types=("cudaStream_t", "hipStream_t"), capi_type="wwr_stream"),
]

# The compute descriptor, pointer on CUDA / enum on HIP, carried as a
# pointer-width uint64_t (see module docstring).
SCALAR_ALIASES = {
    "cutensorComputeDescriptor_t": "uint64_t",
    "hiptensorComputeDescriptor_t": "uint64_t",
}

STATUS = ValueEnum(
    vendor_types=("cutensorStatus_t", "hiptensorStatus_t"),
    capi_type="wwr_tensor_status",
    wwr_type="wwrtensorStatus_t",
    fallback="WWR_TENSOR_STATUS_UNKNOWN",
    entries=(
        ("WWR_TENSOR_STATUS_SUCCESS", "WWRTENSOR_STATUS_SUCCESS"),
        ("WWR_TENSOR_STATUS_NOT_INITIALIZED", "WWRTENSOR_STATUS_NOT_INITIALIZED"),
        ("WWR_TENSOR_STATUS_ALLOC_FAILED", "WWRTENSOR_STATUS_ALLOC_FAILED"),
        ("WWR_TENSOR_STATUS_INVALID_VALUE", "WWRTENSOR_STATUS_INVALID_VALUE"),
        ("WWR_TENSOR_STATUS_ARCH_MISMATCH", "WWRTENSOR_STATUS_ARCH_MISMATCH"),
        ("WWR_TENSOR_STATUS_EXECUTION_FAILED", "WWRTENSOR_STATUS_EXECUTION_FAILED"),
        ("WWR_TENSOR_STATUS_INTERNAL_ERROR", "WWRTENSOR_STATUS_INTERNAL_ERROR"),
        ("WWR_TENSOR_STATUS_NOT_SUPPORTED", "WWRTENSOR_STATUS_NOT_SUPPORTED"),
        ("WWR_TENSOR_STATUS_INSUFFICIENT_WORKSPACE",
         "WWRTENSOR_STATUS_INSUFFICIENT_WORKSPACE"),
        ("WWR_TENSOR_STATUS_INSUFFICIENT_DRIVER",
         "WWRTENSOR_STATUS_INSUFFICIENT_DRIVER"),
        ("WWR_TENSOR_STATUS_IO_ERROR", "WWRTENSOR_STATUS_IO_ERROR"),
    ),
)

OPERATOR = ValueEnum(
    vendor_types=("cutensorOperator_t", "hiptensorOperator_t"),
    capi_type="wwr_tensor_operator",
    wwr_type="wwrtensorOperator_t",
    fallback="WWR_TENSOR_OP_UNKNOWN",
    entries=(
        ("WWR_TENSOR_OP_IDENTITY", "WWRTENSOR_OP_IDENTITY"),
        ("WWR_TENSOR_OP_SQRT", "WWRTENSOR_OP_SQRT"),
        ("WWR_TENSOR_OP_RELU", "WWRTENSOR_OP_RELU"),
        ("WWR_TENSOR_OP_CONJ", "WWRTENSOR_OP_CONJ"),
        ("WWR_TENSOR_OP_RCP", "WWRTENSOR_OP_RCP"),
        ("WWR_TENSOR_OP_SIGMOID", "WWRTENSOR_OP_SIGMOID"),
        ("WWR_TENSOR_OP_TANH", "WWRTENSOR_OP_TANH"),
        ("WWR_TENSOR_OP_EXP", "WWRTENSOR_OP_EXP"),
        ("WWR_TENSOR_OP_LOG", "WWRTENSOR_OP_LOG"),
        ("WWR_TENSOR_OP_ABS", "WWRTENSOR_OP_ABS"),
        ("WWR_TENSOR_OP_NEG", "WWRTENSOR_OP_NEG"),
        ("WWR_TENSOR_OP_SIN", "WWRTENSOR_OP_SIN"),
        ("WWR_TENSOR_OP_COS", "WWRTENSOR_OP_COS"),
        ("WWR_TENSOR_OP_TAN", "WWRTENSOR_OP_TAN"),
        ("WWR_TENSOR_OP_SINH", "WWRTENSOR_OP_SINH"),
        ("WWR_TENSOR_OP_COSH", "WWRTENSOR_OP_COSH"),
        ("WWR_TENSOR_OP_ASIN", "WWRTENSOR_OP_ASIN"),
        ("WWR_TENSOR_OP_ACOS", "WWRTENSOR_OP_ACOS"),
        ("WWR_TENSOR_OP_ATAN", "WWRTENSOR_OP_ATAN"),
        ("WWR_TENSOR_OP_ASINH", "WWRTENSOR_OP_ASINH"),
        ("WWR_TENSOR_OP_ACOSH", "WWRTENSOR_OP_ACOSH"),
        ("WWR_TENSOR_OP_ATANH", "WWRTENSOR_OP_ATANH"),
        ("WWR_TENSOR_OP_CEIL", "WWRTENSOR_OP_CEIL"),
        ("WWR_TENSOR_OP_FLOOR", "WWRTENSOR_OP_FLOOR"),
        ("WWR_TENSOR_OP_ADD", "WWRTENSOR_OP_ADD"),
        ("WWR_TENSOR_OP_MUL", "WWRTENSOR_OP_MUL"),
        ("WWR_TENSOR_OP_MAX", "WWRTENSOR_OP_MAX"),
        ("WWR_TENSOR_OP_MIN", "WWRTENSOR_OP_MIN"),
        ("WWR_TENSOR_OP_UNKNOWN", "WWRTENSOR_OP_UNKNOWN"),
    ),
)

ALGO = ValueEnum(
    vendor_types=("cutensorAlgo_t", "hiptensorAlgo_t"),
    capi_type="wwr_tensor_algo",
    wwr_type="wwrtensorAlgo_t",
    fallback="WWR_TENSOR_ALGO_DEFAULT",
    entries=(
        ("WWR_TENSOR_ALGO_DEFAULT", "WWRTENSOR_ALGO_DEFAULT"),
        ("WWR_TENSOR_ALGO_DEFAULT_PATIENT", "WWRTENSOR_ALGO_DEFAULT_PATIENT"),
    ),
)

JIT_MODE = ValueEnum(
    vendor_types=("cutensorJitMode_t", "hiptensorJitMode_t"),
    capi_type="wwr_tensor_jit_mode",
    wwr_type="wwrtensorJitMode_t",
    fallback="WWR_TENSOR_JIT_MODE_NONE",
    entries=(
        ("WWR_TENSOR_JIT_MODE_NONE", "WWRTENSOR_JIT_MODE_NONE"),
        ("WWR_TENSOR_JIT_MODE_DEFAULT", "WWRTENSOR_JIT_MODE_DEFAULT"),
    ),
)

WORKSIZE_PREFERENCE = ValueEnum(
    vendor_types=("cutensorWorksizePreference_t", "hiptensorWorksizePreference_t"),
    capi_type="wwr_tensor_worksize_preference",
    wwr_type="wwrtensorWorksizePreference_t",
    fallback="WWR_TENSOR_WORKSPACE_DEFAULT",
    entries=(
        ("WWR_TENSOR_WORKSPACE_MIN", "WWRTENSOR_WORKSPACE_MIN"),
        ("WWR_TENSOR_WORKSPACE_DEFAULT", "WWRTENSOR_WORKSPACE_DEFAULT"),
        ("WWR_TENSOR_WORKSPACE_MAX", "WWRTENSOR_WORKSPACE_MAX"),
    ),
)

OPDESC_ATTR = ValueEnum(
    vendor_types=("cutensorOperationDescriptorAttribute_t",
                  "hiptensorOperationDescriptorAttribute_t"),
    capi_type="wwr_tensor_operation_descriptor_attribute",
    wwr_type="wwrtensorOperationDescriptorAttribute_t",
    fallback="WWR_TENSOR_OPERATION_DESCRIPTOR_TAG",
    entries=(
        ("WWR_TENSOR_OPERATION_DESCRIPTOR_TAG",
         "WWRTENSOR_OPERATION_DESCRIPTOR_TAG"),
        ("WWR_TENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE",
         "WWRTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE"),
        ("WWR_TENSOR_OPERATION_DESCRIPTOR_FLOPS",
         "WWRTENSOR_OPERATION_DESCRIPTOR_FLOPS"),
        ("WWR_TENSOR_OPERATION_DESCRIPTOR_MOVED_BYTES",
         "WWRTENSOR_OPERATION_DESCRIPTOR_MOVED_BYTES"),
        ("WWR_TENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT",
         "WWRTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT"),
        ("WWR_TENSOR_OPERATION_DESCRIPTOR_PADDING_RIGHT",
         "WWRTENSOR_OPERATION_DESCRIPTOR_PADDING_RIGHT"),
        ("WWR_TENSOR_OPERATION_DESCRIPTOR_PADDING_VALUE",
         "WWRTENSOR_OPERATION_DESCRIPTOR_PADDING_VALUE"),
    ),
)

PLANPREF_ATTR = ValueEnum(
    vendor_types=("cutensorPlanPreferenceAttribute_t",
                  "hiptensorPlanPreferenceAttribute_t"),
    capi_type="wwr_tensor_plan_preference_attribute",
    wwr_type="wwrtensorPlanPreferenceAttribute_t",
    fallback="WWR_TENSOR_PLAN_PREFERENCE_ALGO",
    entries=(
        ("WWR_TENSOR_PLAN_PREFERENCE_AUTOTUNE_MODE",
         "WWRTENSOR_PLAN_PREFERENCE_AUTOTUNE_MODE"),
        ("WWR_TENSOR_PLAN_PREFERENCE_CACHE_MODE",
         "WWRTENSOR_PLAN_PREFERENCE_CACHE_MODE"),
        ("WWR_TENSOR_PLAN_PREFERENCE_INCREMENTAL_COUNT",
         "WWRTENSOR_PLAN_PREFERENCE_INCREMENTAL_COUNT"),
        ("WWR_TENSOR_PLAN_PREFERENCE_ALGO", "WWRTENSOR_PLAN_PREFERENCE_ALGO"),
        ("WWR_TENSOR_PLAN_PREFERENCE_KERNEL_RANK",
         "WWRTENSOR_PLAN_PREFERENCE_KERNEL_RANK"),
        ("WWR_TENSOR_PLAN_PREFERENCE_JIT", "WWRTENSOR_PLAN_PREFERENCE_JIT"),
    ),
)

PLAN_ATTR = ValueEnum(
    vendor_types=("cutensorPlanAttribute_t", "hiptensorPlanAttribute_t"),
    capi_type="wwr_tensor_plan_attribute",
    wwr_type="wwrtensorPlanAttribute_t",
    fallback="WWR_TENSOR_PLAN_REQUIRED_WORKSPACE",
    entries=(
        ("WWR_TENSOR_PLAN_REQUIRED_WORKSPACE", "WWRTENSOR_PLAN_REQUIRED_WORKSPACE"),
    ),
)

DATA_TYPE = ValueEnum(
    vendor_types=("cudaDataType_t", "cudaDataType", "cutensorDataType_t",
                  "hiptensorDataType_t"),
    capi_type="wwr_tensor_data_type",
    wwr_type="wwrtensorDataType_t",
    fallback="WWR_TENSOR_R_32F",
    entries=(
        ("WWR_TENSOR_R_16F", "WWRTENSOR_R_16F"),
        ("WWR_TENSOR_C_16F", "WWRTENSOR_C_16F"),
        ("WWR_TENSOR_R_16BF", "WWRTENSOR_R_16BF"),
        ("WWR_TENSOR_C_16BF", "WWRTENSOR_C_16BF"),
        ("WWR_TENSOR_R_32F", "WWRTENSOR_R_32F"),
        ("WWR_TENSOR_C_32F", "WWRTENSOR_C_32F"),
        ("WWR_TENSOR_R_64F", "WWRTENSOR_R_64F"),
        ("WWR_TENSOR_C_64F", "WWRTENSOR_C_64F"),
        ("WWR_TENSOR_R_4I", "WWRTENSOR_R_4I"),
        ("WWR_TENSOR_C_4I", "WWRTENSOR_C_4I"),
        ("WWR_TENSOR_R_4U", "WWRTENSOR_R_4U"),
        ("WWR_TENSOR_C_4U", "WWRTENSOR_C_4U"),
        ("WWR_TENSOR_R_8I", "WWRTENSOR_R_8I"),
        ("WWR_TENSOR_C_8I", "WWRTENSOR_C_8I"),
        ("WWR_TENSOR_R_8U", "WWRTENSOR_R_8U"),
        ("WWR_TENSOR_C_8U", "WWRTENSOR_C_8U"),
        ("WWR_TENSOR_R_16I", "WWRTENSOR_R_16I"),
        ("WWR_TENSOR_C_16I", "WWRTENSOR_C_16I"),
        ("WWR_TENSOR_R_16U", "WWRTENSOR_R_16U"),
        ("WWR_TENSOR_C_16U", "WWRTENSOR_C_16U"),
        ("WWR_TENSOR_R_32I", "WWRTENSOR_R_32I"),
        ("WWR_TENSOR_C_32I", "WWRTENSOR_C_32I"),
        ("WWR_TENSOR_R_32U", "WWRTENSOR_R_32U"),
        ("WWR_TENSOR_C_32U", "WWRTENSOR_C_32U"),
        ("WWR_TENSOR_R_64I", "WWRTENSOR_R_64I"),
        ("WWR_TENSOR_C_64I", "WWRTENSOR_C_64I"),
        ("WWR_TENSOR_R_64U", "WWRTENSOR_R_64U"),
        ("WWR_TENSOR_C_64U", "WWRTENSOR_C_64U"),
    ),
)

VALUE_ENUMS = [STATUS, OPERATOR, ALGO, JIT_MODE, WORKSIZE_PREFERENCE,
               OPDESC_ATTR, PLANPREF_ATTR, PLAN_ATTR, DATA_TYPE]


def capi_name(wwr_name: str) -> str:
    return snake_capi_name(wwr_name, PREFIX, CAPI_PREFIX)


SPEC = LibrarySpec(
    name="tensor",
    fragment=FRAGMENT,
    capi_prefix=CAPI_PREFIX,
    cuda_manifest=CUDA_MANIFEST,
    hip_manifest=HIP_MANIFEST,
    modules=MODULES,
    handles=HANDLES,
    value_enums=VALUE_ENUMS,
    scalar_aliases=SCALAR_ALIASES,
    capi_name=capi_name,
)
