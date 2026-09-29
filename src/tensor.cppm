/**
 * @file tensor.cppm
 * @brief Backend-neutral tensor primitives: wwrtensor* names for cuTENSOR / hipTensor
 *
 * Tensor contraction, reduction, permutation and element-wise ops. wwrtensor<X>
 * stands for cutensor<X> on CUDA and hiptensor<X> on HIP, WWRTENSOR_<X> for the
 * corresponding CUTENSOR_<X> / HIPTENSOR_<X> constant. See backend.h.
 *
 * NOT A HIPIFY PAIR. Unlike RCCL/NCCL (src/ccl.cppm, one source-compatible
 * reimplementation, identical spellings) this is TWO independent
 * implementations of the same idea -- cuTENSOR is closed-source over CUDA,
 * hipTensor is built on composable-kernel -- that happen to mirror each other's
 * naming. The intersection was therefore MEASURED, not assumed. Reproduce it:
 *
 *   devtools/header_intersection.py \
 *     --cuda /opt/nvidia/cutensor/include/cutensor.h \
 *            /opt/nvidia/cutensor/include/cutensor/types.h \
 *     --hip  /opt/rocm/include/hiptensor/hiptensor.h \
 *            /opt/rocm/include/hiptensor/hiptensor_types.h
 *
 * cuTENSOR 2.8.1.0 exposes 45 functions; hipTensor 2.2.0 exposes 38; 37 are
 * shared by name, with positionally identical signatures (only the vendor
 * prefix and cudaStream_t/hipStream_t, cutensorDataType_t/hiptensorDataType_t
 * differ -- exactly the substitutions WWR_FUNCTION's reference binding absorbs).
 * Of those 37, 36 are aliased below with WWR_FUNCTION; the one signature
 * divergence is wwrtensorLoggerSetLevel (see below). This layer carries:
 *   - 18 shared types (opaque handles/plans + the enums + the logger callback);
 *   - 93 shared constants whose VALUES also agree per backend -- 28 data types
 *     (hipTensor deliberately numbers HIPTENSOR_R_* to match cudaDataType_t),
 *     29 operators, 11 status codes, 3 workspace prefs, 7 op-descriptor attrs,
 *     6 plan-preference attrs, 1 plan attr, and 2 each autotune/cache/JIT modes;
 *   - the 4 shared compute descriptors, through a per-backend #if (see below).
 * Every value is pinned per backend in test/gpu/tensor.cppm: matching names
 * never guarantee matching values (docs/architecture.md, sections 1 and 6).
 *
 * TWO DIVERGENCES THAT WWR_FUNCTION / WWR_VALUE CANNOT REACH (see architecture
 * §4, §5 -- a backend #if is the sanctioned escape hatch for one-off mismatches):
 *
 *   1. Compute descriptors (WWRTENSOR_COMPUTE_DESC_16F/16BF/32F/64F). On CUDA
 *      these are `extern const` opaque *pointers* (runtime globals, not constant
 *      expressions) -- only a reference alias works. On HIP they are enum
 *      *values* (prvalues) -- only a by-value alias works. No single macro spans
 *      both, so they are declared in a per-backend #if. The neutral TYPE
 *      wwrtensorComputeDescriptor_t (a pointer on CUDA, an enum on HIP) always
 *      matches the descCompute parameter of the backend's own create-op calls,
 *      so a portable caller passes WWRTENSOR_COMPUTE_DESC_* straight through.
 *
 *   2. wwrtensorLoggerSetLevel. cuTENSOR takes `int32_t`, hipTensor takes the
 *      hipTensor-only enum `hiptensorLogLevel_t`. A reference alias would give
 *      the neutral name a different parameter type per backend; a forwarding
 *      function gives it a uniform `int32_t` (HIP static_casts into the enum).
 *
 * ABSENT FROM THIS LAYER (no counterpart -- reachable only through the raw
 * wwr.cuda.cutensor / wwr.hip.hiptensor modules):
 *   - cuTENSOR-only: the block-sparse API (cutensorCreateBlockSparse*,
 *     cutensorBlockSparseContract, cutensor{Create,Destroy}BlockSparseTensor-
 *     Descriptor), the trinary contraction (cutensor{Create,}ContractTrinary),
 *     cutensorPlanPreferenceGetAttribute, cutensorGetCudartVersion, the extra
 *     operators (MISH/SWISH/SOFT_PLUS/SOFT_SIGN), algos (GETT/TGETT/TTGT),
 *     compute descriptors (TF32/3XTF32/9X16BF/8XINT8/4X16F), GPU_ARCH plan
 *     preference, BLOCKSPARSE_REPRODUCIBLE, and the CUDA-specific status codes
 *     (MAPPING_ERROR/LICENSE_ERROR/CUBLAS_ERROR/CUDA_ERROR).
 *   - hipTensor-only: hiptensorGetHiprtVersion, the hiptensorLogLevel_t enum,
 *     HIPTENSOR_ALGO_ACTOR_CRITIC, the extra compute descriptors
 *     (C32F/C64F/NONE/8U/8I/32U/32I), and the CK_ERROR/HIP_ERROR status codes.
 *   The version macros (CUTENSOR_VERSION / HIPTENSOR_VERSION and friends) are
 *   backend-specific; wwrtensorGetVersion is the portable runtime query.
 *
 * Usage:
 *   import wwr.tensor;
 *   wwrtensorHandle_t handle;
 *   wwrtensorCreate(&handle);
 *   ... wwrtensorCreateContraction(handle, &desc, ..., WWRTENSOR_COMPUTE_DESC_32F);
 */

module;

#include "backend.h"

export module wwr.tensor;

import std;

#if defined(WWR_GPU_BACKEND_CUDA)
import wwr.cuda.cutensor;
#else
import wwr.hip.hiptensor;
#endif

export namespace wwr {

// ========================================================================
// Types
// ========================================================================

WWR_TYPE(wwrtensorHandle_t, cutensorHandle_t, hiptensorHandle_t)
WWR_TYPE(wwrtensorTensorDescriptor_t, cutensorTensorDescriptor_t, hiptensorTensorDescriptor_t)
WWR_TYPE(wwrtensorOperationDescriptor_t, cutensorOperationDescriptor_t, hiptensorOperationDescriptor_t)
WWR_TYPE(wwrtensorPlan_t, cutensorPlan_t, hiptensorPlan_t)
WWR_TYPE(wwrtensorPlanPreference_t, cutensorPlanPreference_t, hiptensorPlanPreference_t)
WWR_TYPE(wwrtensorComputeDescriptor_t, cutensorComputeDescriptor_t, hiptensorComputeDescriptor_t)
WWR_TYPE(wwrtensorDataType_t, cutensorDataType_t, hiptensorDataType_t)
WWR_TYPE(wwrtensorStatus_t, cutensorStatus_t, hiptensorStatus_t)
WWR_TYPE(wwrtensorOperator_t, cutensorOperator_t, hiptensorOperator_t)
WWR_TYPE(wwrtensorAlgo_t, cutensorAlgo_t, hiptensorAlgo_t)
WWR_TYPE(wwrtensorWorksizePreference_t, cutensorWorksizePreference_t, hiptensorWorksizePreference_t)
WWR_TYPE(wwrtensorOperationDescriptorAttribute_t, cutensorOperationDescriptorAttribute_t, hiptensorOperationDescriptorAttribute_t)
WWR_TYPE(wwrtensorPlanPreferenceAttribute_t, cutensorPlanPreferenceAttribute_t, hiptensorPlanPreferenceAttribute_t)
WWR_TYPE(wwrtensorPlanAttribute_t, cutensorPlanAttribute_t, hiptensorPlanAttribute_t)
WWR_TYPE(wwrtensorAutotuneMode_t, cutensorAutotuneMode_t, hiptensorAutotuneMode_t)
WWR_TYPE(wwrtensorCacheMode_t, cutensorCacheMode_t, hiptensorCacheMode_t)
WWR_TYPE(wwrtensorJitMode_t, cutensorJitMode_t, hiptensorJitMode_t)
WWR_TYPE(wwrtensorLoggerCallback_t, cutensorLoggerCallback_t, hiptensorLoggerCallback_t)

// ========================================================================
// Constants - data types (hipTensor numbers these to match cudaDataType_t)
// ========================================================================

WWR_VALUE(WWRTENSOR_R_16F, CUTENSOR_R_16F, HIPTENSOR_R_16F)
WWR_VALUE(WWRTENSOR_C_16F, CUTENSOR_C_16F, HIPTENSOR_C_16F)
WWR_VALUE(WWRTENSOR_R_16BF, CUTENSOR_R_16BF, HIPTENSOR_R_16BF)
WWR_VALUE(WWRTENSOR_C_16BF, CUTENSOR_C_16BF, HIPTENSOR_C_16BF)
WWR_VALUE(WWRTENSOR_R_32F, CUTENSOR_R_32F, HIPTENSOR_R_32F)
WWR_VALUE(WWRTENSOR_C_32F, CUTENSOR_C_32F, HIPTENSOR_C_32F)
WWR_VALUE(WWRTENSOR_R_64F, CUTENSOR_R_64F, HIPTENSOR_R_64F)
WWR_VALUE(WWRTENSOR_C_64F, CUTENSOR_C_64F, HIPTENSOR_C_64F)
WWR_VALUE(WWRTENSOR_R_4I, CUTENSOR_R_4I, HIPTENSOR_R_4I)
WWR_VALUE(WWRTENSOR_C_4I, CUTENSOR_C_4I, HIPTENSOR_C_4I)
WWR_VALUE(WWRTENSOR_R_4U, CUTENSOR_R_4U, HIPTENSOR_R_4U)
WWR_VALUE(WWRTENSOR_C_4U, CUTENSOR_C_4U, HIPTENSOR_C_4U)
WWR_VALUE(WWRTENSOR_R_8I, CUTENSOR_R_8I, HIPTENSOR_R_8I)
WWR_VALUE(WWRTENSOR_C_8I, CUTENSOR_C_8I, HIPTENSOR_C_8I)
WWR_VALUE(WWRTENSOR_R_8U, CUTENSOR_R_8U, HIPTENSOR_R_8U)
WWR_VALUE(WWRTENSOR_C_8U, CUTENSOR_C_8U, HIPTENSOR_C_8U)
WWR_VALUE(WWRTENSOR_R_16I, CUTENSOR_R_16I, HIPTENSOR_R_16I)
WWR_VALUE(WWRTENSOR_C_16I, CUTENSOR_C_16I, HIPTENSOR_C_16I)
WWR_VALUE(WWRTENSOR_R_16U, CUTENSOR_R_16U, HIPTENSOR_R_16U)
WWR_VALUE(WWRTENSOR_C_16U, CUTENSOR_C_16U, HIPTENSOR_C_16U)
WWR_VALUE(WWRTENSOR_R_32I, CUTENSOR_R_32I, HIPTENSOR_R_32I)
WWR_VALUE(WWRTENSOR_C_32I, CUTENSOR_C_32I, HIPTENSOR_C_32I)
WWR_VALUE(WWRTENSOR_R_32U, CUTENSOR_R_32U, HIPTENSOR_R_32U)
WWR_VALUE(WWRTENSOR_C_32U, CUTENSOR_C_32U, HIPTENSOR_C_32U)
WWR_VALUE(WWRTENSOR_R_64I, CUTENSOR_R_64I, HIPTENSOR_R_64I)
WWR_VALUE(WWRTENSOR_C_64I, CUTENSOR_C_64I, HIPTENSOR_C_64I)
WWR_VALUE(WWRTENSOR_R_64U, CUTENSOR_R_64U, HIPTENSOR_R_64U)
WWR_VALUE(WWRTENSOR_C_64U, CUTENSOR_C_64U, HIPTENSOR_C_64U)

// ========================================================================
// Constants - element-wise operators
// ========================================================================

WWR_VALUE(WWRTENSOR_OP_IDENTITY, CUTENSOR_OP_IDENTITY, HIPTENSOR_OP_IDENTITY)
WWR_VALUE(WWRTENSOR_OP_SQRT, CUTENSOR_OP_SQRT, HIPTENSOR_OP_SQRT)
WWR_VALUE(WWRTENSOR_OP_RELU, CUTENSOR_OP_RELU, HIPTENSOR_OP_RELU)
WWR_VALUE(WWRTENSOR_OP_CONJ, CUTENSOR_OP_CONJ, HIPTENSOR_OP_CONJ)
WWR_VALUE(WWRTENSOR_OP_RCP, CUTENSOR_OP_RCP, HIPTENSOR_OP_RCP)
WWR_VALUE(WWRTENSOR_OP_SIGMOID, CUTENSOR_OP_SIGMOID, HIPTENSOR_OP_SIGMOID)
WWR_VALUE(WWRTENSOR_OP_TANH, CUTENSOR_OP_TANH, HIPTENSOR_OP_TANH)
WWR_VALUE(WWRTENSOR_OP_EXP, CUTENSOR_OP_EXP, HIPTENSOR_OP_EXP)
WWR_VALUE(WWRTENSOR_OP_LOG, CUTENSOR_OP_LOG, HIPTENSOR_OP_LOG)
WWR_VALUE(WWRTENSOR_OP_ABS, CUTENSOR_OP_ABS, HIPTENSOR_OP_ABS)
WWR_VALUE(WWRTENSOR_OP_NEG, CUTENSOR_OP_NEG, HIPTENSOR_OP_NEG)
WWR_VALUE(WWRTENSOR_OP_SIN, CUTENSOR_OP_SIN, HIPTENSOR_OP_SIN)
WWR_VALUE(WWRTENSOR_OP_COS, CUTENSOR_OP_COS, HIPTENSOR_OP_COS)
WWR_VALUE(WWRTENSOR_OP_TAN, CUTENSOR_OP_TAN, HIPTENSOR_OP_TAN)
WWR_VALUE(WWRTENSOR_OP_SINH, CUTENSOR_OP_SINH, HIPTENSOR_OP_SINH)
WWR_VALUE(WWRTENSOR_OP_COSH, CUTENSOR_OP_COSH, HIPTENSOR_OP_COSH)
WWR_VALUE(WWRTENSOR_OP_ASIN, CUTENSOR_OP_ASIN, HIPTENSOR_OP_ASIN)
WWR_VALUE(WWRTENSOR_OP_ACOS, CUTENSOR_OP_ACOS, HIPTENSOR_OP_ACOS)
WWR_VALUE(WWRTENSOR_OP_ATAN, CUTENSOR_OP_ATAN, HIPTENSOR_OP_ATAN)
WWR_VALUE(WWRTENSOR_OP_ASINH, CUTENSOR_OP_ASINH, HIPTENSOR_OP_ASINH)
WWR_VALUE(WWRTENSOR_OP_ACOSH, CUTENSOR_OP_ACOSH, HIPTENSOR_OP_ACOSH)
WWR_VALUE(WWRTENSOR_OP_ATANH, CUTENSOR_OP_ATANH, HIPTENSOR_OP_ATANH)
WWR_VALUE(WWRTENSOR_OP_CEIL, CUTENSOR_OP_CEIL, HIPTENSOR_OP_CEIL)
WWR_VALUE(WWRTENSOR_OP_FLOOR, CUTENSOR_OP_FLOOR, HIPTENSOR_OP_FLOOR)
WWR_VALUE(WWRTENSOR_OP_ADD, CUTENSOR_OP_ADD, HIPTENSOR_OP_ADD)
WWR_VALUE(WWRTENSOR_OP_MUL, CUTENSOR_OP_MUL, HIPTENSOR_OP_MUL)
WWR_VALUE(WWRTENSOR_OP_MAX, CUTENSOR_OP_MAX, HIPTENSOR_OP_MAX)
WWR_VALUE(WWRTENSOR_OP_MIN, CUTENSOR_OP_MIN, HIPTENSOR_OP_MIN)
WWR_VALUE(WWRTENSOR_OP_UNKNOWN, CUTENSOR_OP_UNKNOWN, HIPTENSOR_OP_UNKNOWN)

// ========================================================================
// Constants - status codes (values agree; the backend-specific codes
// MAPPING/LICENSE/CUBLAS/CUDA_ERROR and CK/HIP_ERROR are excluded, see header)
// ========================================================================

WWR_VALUE(WWRTENSOR_STATUS_SUCCESS, CUTENSOR_STATUS_SUCCESS, HIPTENSOR_STATUS_SUCCESS)
WWR_VALUE(WWRTENSOR_STATUS_NOT_INITIALIZED, CUTENSOR_STATUS_NOT_INITIALIZED, HIPTENSOR_STATUS_NOT_INITIALIZED)
WWR_VALUE(WWRTENSOR_STATUS_ALLOC_FAILED, CUTENSOR_STATUS_ALLOC_FAILED, HIPTENSOR_STATUS_ALLOC_FAILED)
WWR_VALUE(WWRTENSOR_STATUS_INVALID_VALUE, CUTENSOR_STATUS_INVALID_VALUE, HIPTENSOR_STATUS_INVALID_VALUE)
WWR_VALUE(WWRTENSOR_STATUS_ARCH_MISMATCH, CUTENSOR_STATUS_ARCH_MISMATCH, HIPTENSOR_STATUS_ARCH_MISMATCH)
WWR_VALUE(WWRTENSOR_STATUS_EXECUTION_FAILED, CUTENSOR_STATUS_EXECUTION_FAILED, HIPTENSOR_STATUS_EXECUTION_FAILED)
WWR_VALUE(WWRTENSOR_STATUS_INTERNAL_ERROR, CUTENSOR_STATUS_INTERNAL_ERROR, HIPTENSOR_STATUS_INTERNAL_ERROR)
WWR_VALUE(WWRTENSOR_STATUS_NOT_SUPPORTED, CUTENSOR_STATUS_NOT_SUPPORTED, HIPTENSOR_STATUS_NOT_SUPPORTED)
WWR_VALUE(WWRTENSOR_STATUS_INSUFFICIENT_WORKSPACE, CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE, HIPTENSOR_STATUS_INSUFFICIENT_WORKSPACE)
WWR_VALUE(WWRTENSOR_STATUS_INSUFFICIENT_DRIVER, CUTENSOR_STATUS_INSUFFICIENT_DRIVER, HIPTENSOR_STATUS_INSUFFICIENT_DRIVER)
WWR_VALUE(WWRTENSOR_STATUS_IO_ERROR, CUTENSOR_STATUS_IO_ERROR, HIPTENSOR_STATUS_IO_ERROR)

// ========================================================================
// Constants - contraction algorithm (only DEFAULT / DEFAULT_PATIENT are shared)
// ========================================================================

WWR_VALUE(WWRTENSOR_ALGO_DEFAULT, CUTENSOR_ALGO_DEFAULT, HIPTENSOR_ALGO_DEFAULT)
WWR_VALUE(WWRTENSOR_ALGO_DEFAULT_PATIENT, CUTENSOR_ALGO_DEFAULT_PATIENT, HIPTENSOR_ALGO_DEFAULT_PATIENT)

// ========================================================================
// Constants - workspace preference
// ========================================================================

WWR_VALUE(WWRTENSOR_WORKSPACE_MIN, CUTENSOR_WORKSPACE_MIN, HIPTENSOR_WORKSPACE_MIN)
WWR_VALUE(WWRTENSOR_WORKSPACE_DEFAULT, CUTENSOR_WORKSPACE_DEFAULT, HIPTENSOR_WORKSPACE_DEFAULT)
WWR_VALUE(WWRTENSOR_WORKSPACE_MAX, CUTENSOR_WORKSPACE_MAX, HIPTENSOR_WORKSPACE_MAX)

// ========================================================================
// Constants - operation-descriptor attributes (BLOCKSPARSE_REPRODUCIBLE is
// CUDA-only, excluded)
// ========================================================================

WWR_VALUE(WWRTENSOR_OPERATION_DESCRIPTOR_TAG, CUTENSOR_OPERATION_DESCRIPTOR_TAG, HIPTENSOR_OPERATION_DESCRIPTOR_TAG)
WWR_VALUE(WWRTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE, CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE, HIPTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE)
WWR_VALUE(WWRTENSOR_OPERATION_DESCRIPTOR_FLOPS, CUTENSOR_OPERATION_DESCRIPTOR_FLOPS, HIPTENSOR_OPERATION_DESCRIPTOR_FLOPS)
WWR_VALUE(WWRTENSOR_OPERATION_DESCRIPTOR_MOVED_BYTES, CUTENSOR_OPERATION_DESCRIPTOR_MOVED_BYTES, HIPTENSOR_OPERATION_DESCRIPTOR_MOVED_BYTES)
WWR_VALUE(WWRTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT, CUTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT, HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT)
WWR_VALUE(WWRTENSOR_OPERATION_DESCRIPTOR_PADDING_RIGHT, CUTENSOR_OPERATION_DESCRIPTOR_PADDING_RIGHT, HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_RIGHT)
WWR_VALUE(WWRTENSOR_OPERATION_DESCRIPTOR_PADDING_VALUE, CUTENSOR_OPERATION_DESCRIPTOR_PADDING_VALUE, HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_VALUE)

// ========================================================================
// Constants - plan-preference attributes (GPU_ARCH is CUDA-only, excluded)
// ========================================================================

WWR_VALUE(WWRTENSOR_PLAN_PREFERENCE_AUTOTUNE_MODE, CUTENSOR_PLAN_PREFERENCE_AUTOTUNE_MODE, HIPTENSOR_PLAN_PREFERENCE_AUTOTUNE_MODE)
WWR_VALUE(WWRTENSOR_PLAN_PREFERENCE_CACHE_MODE, CUTENSOR_PLAN_PREFERENCE_CACHE_MODE, HIPTENSOR_PLAN_PREFERENCE_CACHE_MODE)
WWR_VALUE(WWRTENSOR_PLAN_PREFERENCE_INCREMENTAL_COUNT, CUTENSOR_PLAN_PREFERENCE_INCREMENTAL_COUNT, HIPTENSOR_PLAN_PREFERENCE_INCREMENTAL_COUNT)
WWR_VALUE(WWRTENSOR_PLAN_PREFERENCE_ALGO, CUTENSOR_PLAN_PREFERENCE_ALGO, HIPTENSOR_PLAN_PREFERENCE_ALGO)
WWR_VALUE(WWRTENSOR_PLAN_PREFERENCE_KERNEL_RANK, CUTENSOR_PLAN_PREFERENCE_KERNEL_RANK, HIPTENSOR_PLAN_PREFERENCE_KERNEL_RANK)
WWR_VALUE(WWRTENSOR_PLAN_PREFERENCE_JIT, CUTENSOR_PLAN_PREFERENCE_JIT, HIPTENSOR_PLAN_PREFERENCE_JIT)

// ========================================================================
// Constants - plan attributes
// ========================================================================

WWR_VALUE(WWRTENSOR_PLAN_REQUIRED_WORKSPACE, CUTENSOR_PLAN_REQUIRED_WORKSPACE, HIPTENSOR_PLAN_REQUIRED_WORKSPACE)

// ========================================================================
// Constants - autotune / cache / JIT modes
// ========================================================================

WWR_VALUE(WWRTENSOR_AUTOTUNE_MODE_NONE, CUTENSOR_AUTOTUNE_MODE_NONE, HIPTENSOR_AUTOTUNE_MODE_NONE)
WWR_VALUE(WWRTENSOR_AUTOTUNE_MODE_INCREMENTAL, CUTENSOR_AUTOTUNE_MODE_INCREMENTAL, HIPTENSOR_AUTOTUNE_MODE_INCREMENTAL)
WWR_VALUE(WWRTENSOR_CACHE_MODE_NONE, CUTENSOR_CACHE_MODE_NONE, HIPTENSOR_CACHE_MODE_NONE)
WWR_VALUE(WWRTENSOR_CACHE_MODE_PEDANTIC, CUTENSOR_CACHE_MODE_PEDANTIC, HIPTENSOR_CACHE_MODE_PEDANTIC)
WWR_VALUE(WWRTENSOR_JIT_MODE_NONE, CUTENSOR_JIT_MODE_NONE, HIPTENSOR_JIT_MODE_NONE)
WWR_VALUE(WWRTENSOR_JIT_MODE_DEFAULT, CUTENSOR_JIT_MODE_DEFAULT, HIPTENSOR_JIT_MODE_DEFAULT)

// ========================================================================
// Constants - compute descriptors (the one-off #if escape hatch; see header).
// CUDA: `extern const` opaque pointer -> reference alias. HIP: enum value ->
// value alias. wwrtensorComputeDescriptor_t matches either on the way into a
// create-op call, which is all a portable caller needs.
// ========================================================================

#if defined(WWR_GPU_BACKEND_CUDA)
inline constexpr auto &WWRTENSOR_COMPUTE_DESC_16F = ::wwr::cuda::CUTENSOR_COMPUTE_DESC_16F;
inline constexpr auto &WWRTENSOR_COMPUTE_DESC_16BF = ::wwr::cuda::CUTENSOR_COMPUTE_DESC_16BF;
inline constexpr auto &WWRTENSOR_COMPUTE_DESC_32F = ::wwr::cuda::CUTENSOR_COMPUTE_DESC_32F;
inline constexpr auto &WWRTENSOR_COMPUTE_DESC_64F = ::wwr::cuda::CUTENSOR_COMPUTE_DESC_64F;
#else
inline constexpr auto WWRTENSOR_COMPUTE_DESC_16F = ::wwr::hip::HIPTENSOR_COMPUTE_DESC_16F;
inline constexpr auto WWRTENSOR_COMPUTE_DESC_16BF = ::wwr::hip::HIPTENSOR_COMPUTE_DESC_16BF;
inline constexpr auto WWRTENSOR_COMPUTE_DESC_32F = ::wwr::hip::HIPTENSOR_COMPUTE_DESC_32F;
inline constexpr auto WWRTENSOR_COMPUTE_DESC_64F = ::wwr::hip::HIPTENSOR_COMPUTE_DESC_64F;
#endif

// ========================================================================
// Library context / version
// ========================================================================

WWR_FUNCTION(wwrtensorCreate, cutensorCreate, hiptensorCreate)
WWR_FUNCTION(wwrtensorDestroy, cutensorDestroy, hiptensorDestroy)
WWR_FUNCTION(wwrtensorGetErrorString, cutensorGetErrorString, hiptensorGetErrorString)
WWR_FUNCTION(wwrtensorGetVersion, cutensorGetVersion, hiptensorGetVersion)

// ========================================================================
// Plan cache / kernel cache
// ========================================================================

WWR_FUNCTION(wwrtensorHandleResizePlanCache, cutensorHandleResizePlanCache, hiptensorHandleResizePlanCache)
WWR_FUNCTION(wwrtensorHandleWritePlanCacheToFile, cutensorHandleWritePlanCacheToFile, hiptensorHandleWritePlanCacheToFile)
WWR_FUNCTION(wwrtensorHandleReadPlanCacheFromFile, cutensorHandleReadPlanCacheFromFile, hiptensorHandleReadPlanCacheFromFile)
WWR_FUNCTION(wwrtensorWriteKernelCacheToFile, cutensorWriteKernelCacheToFile, hiptensorWriteKernelCacheToFile)
WWR_FUNCTION(wwrtensorReadKernelCacheFromFile, cutensorReadKernelCacheFromFile, hiptensorReadKernelCacheFromFile)

// ========================================================================
// Tensor descriptors
// ========================================================================

WWR_FUNCTION(wwrtensorCreateTensorDescriptor, cutensorCreateTensorDescriptor, hiptensorCreateTensorDescriptor)
WWR_FUNCTION(wwrtensorDestroyTensorDescriptor, cutensorDestroyTensorDescriptor, hiptensorDestroyTensorDescriptor)

// ========================================================================
// Operation construction
// ========================================================================

WWR_FUNCTION(wwrtensorCreateElementwiseTrinary, cutensorCreateElementwiseTrinary, hiptensorCreateElementwiseTrinary)
WWR_FUNCTION(wwrtensorCreateElementwiseBinary, cutensorCreateElementwiseBinary, hiptensorCreateElementwiseBinary)
WWR_FUNCTION(wwrtensorCreatePermutation, cutensorCreatePermutation, hiptensorCreatePermutation)
WWR_FUNCTION(wwrtensorCreateContraction, cutensorCreateContraction, hiptensorCreateContraction)
WWR_FUNCTION(wwrtensorCreateReduction, cutensorCreateReduction, hiptensorCreateReduction)
WWR_FUNCTION(wwrtensorDestroyOperationDescriptor, cutensorDestroyOperationDescriptor, hiptensorDestroyOperationDescriptor)
WWR_FUNCTION(wwrtensorOperationDescriptorSetAttribute, cutensorOperationDescriptorSetAttribute, hiptensorOperationDescriptorSetAttribute)
WWR_FUNCTION(wwrtensorOperationDescriptorGetAttribute, cutensorOperationDescriptorGetAttribute, hiptensorOperationDescriptorGetAttribute)

// ========================================================================
// Plan preference and plan lifecycle (cutensorPlanPreferenceGetAttribute has no
// hipTensor counterpart -- absent here, see header)
// ========================================================================

WWR_FUNCTION(wwrtensorCreatePlanPreference, cutensorCreatePlanPreference, hiptensorCreatePlanPreference)
WWR_FUNCTION(wwrtensorDestroyPlanPreference, cutensorDestroyPlanPreference, hiptensorDestroyPlanPreference)
WWR_FUNCTION(wwrtensorPlanPreferenceSetAttribute, cutensorPlanPreferenceSetAttribute, hiptensorPlanPreferenceSetAttribute)
WWR_FUNCTION(wwrtensorEstimateWorkspaceSize, cutensorEstimateWorkspaceSize, hiptensorEstimateWorkspaceSize)
WWR_FUNCTION(wwrtensorCreatePlan, cutensorCreatePlan, hiptensorCreatePlan)
WWR_FUNCTION(wwrtensorDestroyPlan, cutensorDestroyPlan, hiptensorDestroyPlan)
WWR_FUNCTION(wwrtensorPlanGetAttribute, cutensorPlanGetAttribute, hiptensorPlanGetAttribute)

// ========================================================================
// Operation execution
// ========================================================================

WWR_FUNCTION(wwrtensorElementwiseTrinaryExecute, cutensorElementwiseTrinaryExecute, hiptensorElementwiseTrinaryExecute)
WWR_FUNCTION(wwrtensorElementwiseBinaryExecute, cutensorElementwiseBinaryExecute, hiptensorElementwiseBinaryExecute)
WWR_FUNCTION(wwrtensorPermute, cutensorPermute, hiptensorPermute)
WWR_FUNCTION(wwrtensorContract, cutensorContract, hiptensorContract)
WWR_FUNCTION(wwrtensorReduce, cutensorReduce, hiptensorReduce)

// ========================================================================
// Logging (wwrtensorLoggerSetLevel is a forwarding function, not an alias --
// see header, divergence 2)
// ========================================================================

WWR_FUNCTION(wwrtensorLoggerSetCallback, cutensorLoggerSetCallback, hiptensorLoggerSetCallback)
WWR_FUNCTION(wwrtensorLoggerSetFile, cutensorLoggerSetFile, hiptensorLoggerSetFile)
WWR_FUNCTION(wwrtensorLoggerOpenFile, cutensorLoggerOpenFile, hiptensorLoggerOpenFile)
WWR_FUNCTION(wwrtensorLoggerSetMask, cutensorLoggerSetMask, hiptensorLoggerSetMask)
WWR_FUNCTION(wwrtensorLoggerForceDisable, cutensorLoggerForceDisable, hiptensorLoggerForceDisable)

inline wwrtensorStatus_t wwrtensorLoggerSetLevel(std::int32_t level) {
#if defined(WWR_GPU_BACKEND_CUDA)
  return ::wwr::cuda::cutensorLoggerSetLevel(level);
#else
  return ::wwr::hip::hiptensorLoggerSetLevel(static_cast<::wwr::hip::hiptensorLogLevel_t>(level));
#endif
}

} // namespace wwr
