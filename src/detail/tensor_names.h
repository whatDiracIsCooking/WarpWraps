/**
 * @file detail/tensor_names.h
 * @brief The backend-neutral cuTENSOR / hipTensor surface, as a macro-driven
 *        include fragment shared by the module and the non-module #include path
 *
 * NOT a standalone header: it is the list of wwrtensor* / WWRTENSOR_* names
 * (types, constants, the compute-descriptor escape hatch, the functions and the
 * one hand-written logger forwarder) with NO namespace of its own and NO vendor
 * #include. The includer supplies all of that and pastes this inside its own
 * `namespace wwr` -- so one list binds both ways the surface is consumed:
 * wwr.tensor (the module, `export namespace wwr`) and wwr/tensor.h (the
 * non-module #include path). Add a name here, once, and both paths gain it.
 *
 * HOST only -- cuTENSOR / hipTensor is a host API. The surface binds straight to
 * the vendor's external-linkage `::cutensor*` / `::hiptensor*` declarations via
 * the _RAW macros (the "rand.h shape"): no raw vendor module is imported, so the
 * same `::`-prefixed names resolve in the module (vendor header in its GMF) and
 * the #include path alike. The per-backend extras (block-sparse, trinary
 * contraction, the extra compute descriptors) are deliberately absent; reach
 * them through wwr.cuda.cutensor / wwr.hip.hiptensor. See src/tensor.cppm for
 * the measured intersection and what each backend drops.
 *
 * Before including, the includer must have, in order:
 *   - the vendor header in scope (cutensor.h + cutensor/types.h /
 *     hiptensor/hiptensor.h + hiptensor_types.h) and <cstdint> for the
 *     std::int32_t the logger forwarder names, which tensor.h pulls in;
 *   - WWR_SELECT_RAW(cuda, hip) plus WWR_TYPE_RAW / WWR_VALUE_RAW /
 *     WWR_FUNCTION_RAW on top of it -- keyed on WWR_GPU_BACKEND_* in the module
 *     (backend.h) or WWR_SELECTED_* in the #include path (wwr/tensor.h);
 *   - WWR_SELECTED_CUDA / WWR_SELECTED_HIP (selected_backend.h, via tensor.h) for
 *     the compute-descriptor #if and the logger forwarder's backend conditional.
 *
 * TWO DIVERGENCES the plain macros cannot reach (a per-backend #if is the
 * sanctioned escape hatch, see docs/architecture.md §4, §5):
 *   1. WWRTENSOR_COMPUTE_DESC_* -- an `extern const` opaque pointer on CUDA (a
 *      reference alias) but an enum value on HIP (a by-value alias). Bound to
 *      the vendor's own `::CUTENSOR_*` / `::HIPTENSOR_*` globals directly, both
 *      external-linkage / prvalue over the vendor header, so they resolve on
 *      both host paths.
 *   2. wwrtensorLoggerSetLevel -- cuTENSOR takes int32_t, hipTensor its own enum;
 *      a forwarding function gives the neutral name a uniform std::int32_t and
 *      calls the vendor global directly (HIP static_casts into the enum).
 * See src/tensor.cppm, src/tensor.h, src/wwr/tensor.h and docs/architecture.md.
 */

#pragma once

#ifndef WWR_FUNCTION_RAW
#error                                                                                             \
    "detail/tensor_names.h is an include fragment, not a standalone header: define WWR_TYPE_RAW/VALUE_RAW/FUNCTION_RAW and WWR_SELECT_RAW, ensure the vendor header and <cstdint> (via tensor.h) and WWR_SELECTED_* are in scope, and #include it inside namespace wwr. See src/tensor.h, src/tensor.cppm and src/wwr/tensor.h."
#endif

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables): each wwrtensor*
// function below is a deliberate constexpr reference to the selected backend's
// entry point (via WWR_FUNCTION_RAW). A reference to a vendor function has no
// const form, so the check cannot be satisfied without abandoning the alias
// pattern -- see backend.h and detail/blas_names.h.

// ========================================================================
// Types
// ========================================================================

WWR_TYPE_RAW(wwrtensorHandle_t, cutensorHandle_t, hiptensorHandle_t)
WWR_TYPE_RAW(wwrtensorTensorDescriptor_t, cutensorTensorDescriptor_t, hiptensorTensorDescriptor_t)
WWR_TYPE_RAW(wwrtensorOperationDescriptor_t, cutensorOperationDescriptor_t, hiptensorOperationDescriptor_t)
WWR_TYPE_RAW(wwrtensorPlan_t, cutensorPlan_t, hiptensorPlan_t)
WWR_TYPE_RAW(wwrtensorPlanPreference_t, cutensorPlanPreference_t, hiptensorPlanPreference_t)
WWR_TYPE_RAW(wwrtensorComputeDescriptor_t, cutensorComputeDescriptor_t, hiptensorComputeDescriptor_t)
WWR_TYPE_RAW(wwrtensorDataType_t, cutensorDataType_t, hiptensorDataType_t)
WWR_TYPE_RAW(wwrtensorStatus_t, cutensorStatus_t, hiptensorStatus_t)
WWR_TYPE_RAW(wwrtensorOperator_t, cutensorOperator_t, hiptensorOperator_t)
WWR_TYPE_RAW(wwrtensorAlgo_t, cutensorAlgo_t, hiptensorAlgo_t)
WWR_TYPE_RAW(wwrtensorWorksizePreference_t, cutensorWorksizePreference_t, hiptensorWorksizePreference_t)
WWR_TYPE_RAW(wwrtensorOperationDescriptorAttribute_t, cutensorOperationDescriptorAttribute_t, hiptensorOperationDescriptorAttribute_t)
WWR_TYPE_RAW(wwrtensorPlanPreferenceAttribute_t, cutensorPlanPreferenceAttribute_t, hiptensorPlanPreferenceAttribute_t)
WWR_TYPE_RAW(wwrtensorPlanAttribute_t, cutensorPlanAttribute_t, hiptensorPlanAttribute_t)
WWR_TYPE_RAW(wwrtensorAutotuneMode_t, cutensorAutotuneMode_t, hiptensorAutotuneMode_t)
WWR_TYPE_RAW(wwrtensorCacheMode_t, cutensorCacheMode_t, hiptensorCacheMode_t)
WWR_TYPE_RAW(wwrtensorJitMode_t, cutensorJitMode_t, hiptensorJitMode_t)
WWR_TYPE_RAW(wwrtensorLoggerCallback_t, cutensorLoggerCallback_t, hiptensorLoggerCallback_t)

// ========================================================================
// Constants - data types (hipTensor numbers these to match cudaDataType_t)
// ========================================================================

WWR_VALUE_RAW(WWRTENSOR_R_16F, CUTENSOR_R_16F, HIPTENSOR_R_16F)
WWR_VALUE_RAW(WWRTENSOR_C_16F, CUTENSOR_C_16F, HIPTENSOR_C_16F)
WWR_VALUE_RAW(WWRTENSOR_R_16BF, CUTENSOR_R_16BF, HIPTENSOR_R_16BF)
WWR_VALUE_RAW(WWRTENSOR_C_16BF, CUTENSOR_C_16BF, HIPTENSOR_C_16BF)
WWR_VALUE_RAW(WWRTENSOR_R_32F, CUTENSOR_R_32F, HIPTENSOR_R_32F)
WWR_VALUE_RAW(WWRTENSOR_C_32F, CUTENSOR_C_32F, HIPTENSOR_C_32F)
WWR_VALUE_RAW(WWRTENSOR_R_64F, CUTENSOR_R_64F, HIPTENSOR_R_64F)
WWR_VALUE_RAW(WWRTENSOR_C_64F, CUTENSOR_C_64F, HIPTENSOR_C_64F)
WWR_VALUE_RAW(WWRTENSOR_R_4I, CUTENSOR_R_4I, HIPTENSOR_R_4I)
WWR_VALUE_RAW(WWRTENSOR_C_4I, CUTENSOR_C_4I, HIPTENSOR_C_4I)
WWR_VALUE_RAW(WWRTENSOR_R_4U, CUTENSOR_R_4U, HIPTENSOR_R_4U)
WWR_VALUE_RAW(WWRTENSOR_C_4U, CUTENSOR_C_4U, HIPTENSOR_C_4U)
WWR_VALUE_RAW(WWRTENSOR_R_8I, CUTENSOR_R_8I, HIPTENSOR_R_8I)
WWR_VALUE_RAW(WWRTENSOR_C_8I, CUTENSOR_C_8I, HIPTENSOR_C_8I)
WWR_VALUE_RAW(WWRTENSOR_R_8U, CUTENSOR_R_8U, HIPTENSOR_R_8U)
WWR_VALUE_RAW(WWRTENSOR_C_8U, CUTENSOR_C_8U, HIPTENSOR_C_8U)
WWR_VALUE_RAW(WWRTENSOR_R_16I, CUTENSOR_R_16I, HIPTENSOR_R_16I)
WWR_VALUE_RAW(WWRTENSOR_C_16I, CUTENSOR_C_16I, HIPTENSOR_C_16I)
WWR_VALUE_RAW(WWRTENSOR_R_16U, CUTENSOR_R_16U, HIPTENSOR_R_16U)
WWR_VALUE_RAW(WWRTENSOR_C_16U, CUTENSOR_C_16U, HIPTENSOR_C_16U)
WWR_VALUE_RAW(WWRTENSOR_R_32I, CUTENSOR_R_32I, HIPTENSOR_R_32I)
WWR_VALUE_RAW(WWRTENSOR_C_32I, CUTENSOR_C_32I, HIPTENSOR_C_32I)
WWR_VALUE_RAW(WWRTENSOR_R_32U, CUTENSOR_R_32U, HIPTENSOR_R_32U)
WWR_VALUE_RAW(WWRTENSOR_C_32U, CUTENSOR_C_32U, HIPTENSOR_C_32U)
WWR_VALUE_RAW(WWRTENSOR_R_64I, CUTENSOR_R_64I, HIPTENSOR_R_64I)
WWR_VALUE_RAW(WWRTENSOR_C_64I, CUTENSOR_C_64I, HIPTENSOR_C_64I)
WWR_VALUE_RAW(WWRTENSOR_R_64U, CUTENSOR_R_64U, HIPTENSOR_R_64U)
WWR_VALUE_RAW(WWRTENSOR_C_64U, CUTENSOR_C_64U, HIPTENSOR_C_64U)

// ========================================================================
// Constants - element-wise operators
// ========================================================================

WWR_VALUE_RAW(WWRTENSOR_OP_IDENTITY, CUTENSOR_OP_IDENTITY, HIPTENSOR_OP_IDENTITY)
WWR_VALUE_RAW(WWRTENSOR_OP_SQRT, CUTENSOR_OP_SQRT, HIPTENSOR_OP_SQRT)
WWR_VALUE_RAW(WWRTENSOR_OP_RELU, CUTENSOR_OP_RELU, HIPTENSOR_OP_RELU)
WWR_VALUE_RAW(WWRTENSOR_OP_CONJ, CUTENSOR_OP_CONJ, HIPTENSOR_OP_CONJ)
WWR_VALUE_RAW(WWRTENSOR_OP_RCP, CUTENSOR_OP_RCP, HIPTENSOR_OP_RCP)
WWR_VALUE_RAW(WWRTENSOR_OP_SIGMOID, CUTENSOR_OP_SIGMOID, HIPTENSOR_OP_SIGMOID)
WWR_VALUE_RAW(WWRTENSOR_OP_TANH, CUTENSOR_OP_TANH, HIPTENSOR_OP_TANH)
WWR_VALUE_RAW(WWRTENSOR_OP_EXP, CUTENSOR_OP_EXP, HIPTENSOR_OP_EXP)
WWR_VALUE_RAW(WWRTENSOR_OP_LOG, CUTENSOR_OP_LOG, HIPTENSOR_OP_LOG)
WWR_VALUE_RAW(WWRTENSOR_OP_ABS, CUTENSOR_OP_ABS, HIPTENSOR_OP_ABS)
WWR_VALUE_RAW(WWRTENSOR_OP_NEG, CUTENSOR_OP_NEG, HIPTENSOR_OP_NEG)
WWR_VALUE_RAW(WWRTENSOR_OP_SIN, CUTENSOR_OP_SIN, HIPTENSOR_OP_SIN)
WWR_VALUE_RAW(WWRTENSOR_OP_COS, CUTENSOR_OP_COS, HIPTENSOR_OP_COS)
WWR_VALUE_RAW(WWRTENSOR_OP_TAN, CUTENSOR_OP_TAN, HIPTENSOR_OP_TAN)
WWR_VALUE_RAW(WWRTENSOR_OP_SINH, CUTENSOR_OP_SINH, HIPTENSOR_OP_SINH)
WWR_VALUE_RAW(WWRTENSOR_OP_COSH, CUTENSOR_OP_COSH, HIPTENSOR_OP_COSH)
WWR_VALUE_RAW(WWRTENSOR_OP_ASIN, CUTENSOR_OP_ASIN, HIPTENSOR_OP_ASIN)
WWR_VALUE_RAW(WWRTENSOR_OP_ACOS, CUTENSOR_OP_ACOS, HIPTENSOR_OP_ACOS)
WWR_VALUE_RAW(WWRTENSOR_OP_ATAN, CUTENSOR_OP_ATAN, HIPTENSOR_OP_ATAN)
WWR_VALUE_RAW(WWRTENSOR_OP_ASINH, CUTENSOR_OP_ASINH, HIPTENSOR_OP_ASINH)
WWR_VALUE_RAW(WWRTENSOR_OP_ACOSH, CUTENSOR_OP_ACOSH, HIPTENSOR_OP_ACOSH)
WWR_VALUE_RAW(WWRTENSOR_OP_ATANH, CUTENSOR_OP_ATANH, HIPTENSOR_OP_ATANH)
WWR_VALUE_RAW(WWRTENSOR_OP_CEIL, CUTENSOR_OP_CEIL, HIPTENSOR_OP_CEIL)
WWR_VALUE_RAW(WWRTENSOR_OP_FLOOR, CUTENSOR_OP_FLOOR, HIPTENSOR_OP_FLOOR)
WWR_VALUE_RAW(WWRTENSOR_OP_ADD, CUTENSOR_OP_ADD, HIPTENSOR_OP_ADD)
WWR_VALUE_RAW(WWRTENSOR_OP_MUL, CUTENSOR_OP_MUL, HIPTENSOR_OP_MUL)
WWR_VALUE_RAW(WWRTENSOR_OP_MAX, CUTENSOR_OP_MAX, HIPTENSOR_OP_MAX)
WWR_VALUE_RAW(WWRTENSOR_OP_MIN, CUTENSOR_OP_MIN, HIPTENSOR_OP_MIN)
WWR_VALUE_RAW(WWRTENSOR_OP_UNKNOWN, CUTENSOR_OP_UNKNOWN, HIPTENSOR_OP_UNKNOWN)

// ========================================================================
// Constants - status codes (values agree; the backend-specific codes
// MAPPING/LICENSE/CUBLAS/CUDA_ERROR and CK/HIP_ERROR are excluded, see
// src/tensor.cppm)
// ========================================================================

WWR_VALUE_RAW(WWRTENSOR_STATUS_SUCCESS, CUTENSOR_STATUS_SUCCESS, HIPTENSOR_STATUS_SUCCESS)
WWR_VALUE_RAW(WWRTENSOR_STATUS_NOT_INITIALIZED, CUTENSOR_STATUS_NOT_INITIALIZED, HIPTENSOR_STATUS_NOT_INITIALIZED)
WWR_VALUE_RAW(WWRTENSOR_STATUS_ALLOC_FAILED, CUTENSOR_STATUS_ALLOC_FAILED, HIPTENSOR_STATUS_ALLOC_FAILED)
WWR_VALUE_RAW(WWRTENSOR_STATUS_INVALID_VALUE, CUTENSOR_STATUS_INVALID_VALUE, HIPTENSOR_STATUS_INVALID_VALUE)
WWR_VALUE_RAW(WWRTENSOR_STATUS_ARCH_MISMATCH, CUTENSOR_STATUS_ARCH_MISMATCH, HIPTENSOR_STATUS_ARCH_MISMATCH)
WWR_VALUE_RAW(WWRTENSOR_STATUS_EXECUTION_FAILED, CUTENSOR_STATUS_EXECUTION_FAILED, HIPTENSOR_STATUS_EXECUTION_FAILED)
WWR_VALUE_RAW(WWRTENSOR_STATUS_INTERNAL_ERROR, CUTENSOR_STATUS_INTERNAL_ERROR, HIPTENSOR_STATUS_INTERNAL_ERROR)
WWR_VALUE_RAW(WWRTENSOR_STATUS_NOT_SUPPORTED, CUTENSOR_STATUS_NOT_SUPPORTED, HIPTENSOR_STATUS_NOT_SUPPORTED)
WWR_VALUE_RAW(WWRTENSOR_STATUS_INSUFFICIENT_WORKSPACE, CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE, HIPTENSOR_STATUS_INSUFFICIENT_WORKSPACE)
WWR_VALUE_RAW(WWRTENSOR_STATUS_INSUFFICIENT_DRIVER, CUTENSOR_STATUS_INSUFFICIENT_DRIVER, HIPTENSOR_STATUS_INSUFFICIENT_DRIVER)
WWR_VALUE_RAW(WWRTENSOR_STATUS_IO_ERROR, CUTENSOR_STATUS_IO_ERROR, HIPTENSOR_STATUS_IO_ERROR)

// ========================================================================
// Constants - contraction algorithm (only DEFAULT / DEFAULT_PATIENT are shared)
// ========================================================================

WWR_VALUE_RAW(WWRTENSOR_ALGO_DEFAULT, CUTENSOR_ALGO_DEFAULT, HIPTENSOR_ALGO_DEFAULT)
WWR_VALUE_RAW(WWRTENSOR_ALGO_DEFAULT_PATIENT, CUTENSOR_ALGO_DEFAULT_PATIENT, HIPTENSOR_ALGO_DEFAULT_PATIENT)

// ========================================================================
// Constants - workspace preference
// ========================================================================

WWR_VALUE_RAW(WWRTENSOR_WORKSPACE_MIN, CUTENSOR_WORKSPACE_MIN, HIPTENSOR_WORKSPACE_MIN)
WWR_VALUE_RAW(WWRTENSOR_WORKSPACE_DEFAULT, CUTENSOR_WORKSPACE_DEFAULT, HIPTENSOR_WORKSPACE_DEFAULT)
WWR_VALUE_RAW(WWRTENSOR_WORKSPACE_MAX, CUTENSOR_WORKSPACE_MAX, HIPTENSOR_WORKSPACE_MAX)

// ========================================================================
// Constants - operation-descriptor attributes (BLOCKSPARSE_REPRODUCIBLE is
// CUDA-only, excluded)
// ========================================================================

WWR_VALUE_RAW(WWRTENSOR_OPERATION_DESCRIPTOR_TAG, CUTENSOR_OPERATION_DESCRIPTOR_TAG, HIPTENSOR_OPERATION_DESCRIPTOR_TAG)
WWR_VALUE_RAW(WWRTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE, CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE, HIPTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE)
WWR_VALUE_RAW(WWRTENSOR_OPERATION_DESCRIPTOR_FLOPS, CUTENSOR_OPERATION_DESCRIPTOR_FLOPS, HIPTENSOR_OPERATION_DESCRIPTOR_FLOPS)
WWR_VALUE_RAW(WWRTENSOR_OPERATION_DESCRIPTOR_MOVED_BYTES, CUTENSOR_OPERATION_DESCRIPTOR_MOVED_BYTES, HIPTENSOR_OPERATION_DESCRIPTOR_MOVED_BYTES)
WWR_VALUE_RAW(WWRTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT, CUTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT, HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT)
WWR_VALUE_RAW(WWRTENSOR_OPERATION_DESCRIPTOR_PADDING_RIGHT, CUTENSOR_OPERATION_DESCRIPTOR_PADDING_RIGHT, HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_RIGHT)
WWR_VALUE_RAW(WWRTENSOR_OPERATION_DESCRIPTOR_PADDING_VALUE, CUTENSOR_OPERATION_DESCRIPTOR_PADDING_VALUE, HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_VALUE)

// ========================================================================
// Constants - plan-preference attributes (GPU_ARCH is CUDA-only, excluded)
// ========================================================================

WWR_VALUE_RAW(WWRTENSOR_PLAN_PREFERENCE_AUTOTUNE_MODE, CUTENSOR_PLAN_PREFERENCE_AUTOTUNE_MODE, HIPTENSOR_PLAN_PREFERENCE_AUTOTUNE_MODE)
WWR_VALUE_RAW(WWRTENSOR_PLAN_PREFERENCE_CACHE_MODE, CUTENSOR_PLAN_PREFERENCE_CACHE_MODE, HIPTENSOR_PLAN_PREFERENCE_CACHE_MODE)
WWR_VALUE_RAW(WWRTENSOR_PLAN_PREFERENCE_INCREMENTAL_COUNT, CUTENSOR_PLAN_PREFERENCE_INCREMENTAL_COUNT, HIPTENSOR_PLAN_PREFERENCE_INCREMENTAL_COUNT)
WWR_VALUE_RAW(WWRTENSOR_PLAN_PREFERENCE_ALGO, CUTENSOR_PLAN_PREFERENCE_ALGO, HIPTENSOR_PLAN_PREFERENCE_ALGO)
WWR_VALUE_RAW(WWRTENSOR_PLAN_PREFERENCE_KERNEL_RANK, CUTENSOR_PLAN_PREFERENCE_KERNEL_RANK, HIPTENSOR_PLAN_PREFERENCE_KERNEL_RANK)
WWR_VALUE_RAW(WWRTENSOR_PLAN_PREFERENCE_JIT, CUTENSOR_PLAN_PREFERENCE_JIT, HIPTENSOR_PLAN_PREFERENCE_JIT)

// ========================================================================
// Constants - plan attributes
// ========================================================================

WWR_VALUE_RAW(WWRTENSOR_PLAN_REQUIRED_WORKSPACE, CUTENSOR_PLAN_REQUIRED_WORKSPACE, HIPTENSOR_PLAN_REQUIRED_WORKSPACE)

// ========================================================================
// Constants - autotune / cache / JIT modes
// ========================================================================

WWR_VALUE_RAW(WWRTENSOR_AUTOTUNE_MODE_NONE, CUTENSOR_AUTOTUNE_MODE_NONE, HIPTENSOR_AUTOTUNE_MODE_NONE)
WWR_VALUE_RAW(WWRTENSOR_AUTOTUNE_MODE_INCREMENTAL, CUTENSOR_AUTOTUNE_MODE_INCREMENTAL, HIPTENSOR_AUTOTUNE_MODE_INCREMENTAL)
WWR_VALUE_RAW(WWRTENSOR_CACHE_MODE_NONE, CUTENSOR_CACHE_MODE_NONE, HIPTENSOR_CACHE_MODE_NONE)
WWR_VALUE_RAW(WWRTENSOR_CACHE_MODE_PEDANTIC, CUTENSOR_CACHE_MODE_PEDANTIC, HIPTENSOR_CACHE_MODE_PEDANTIC)
WWR_VALUE_RAW(WWRTENSOR_JIT_MODE_NONE, CUTENSOR_JIT_MODE_NONE, HIPTENSOR_JIT_MODE_NONE)
WWR_VALUE_RAW(WWRTENSOR_JIT_MODE_DEFAULT, CUTENSOR_JIT_MODE_DEFAULT, HIPTENSOR_JIT_MODE_DEFAULT)

// ========================================================================
// Constants - compute descriptors (the one-off #if escape hatch; see header).
// Bound straight to the vendor's own globals: CUDA has `extern const` opaque
// pointers (external linkage -> reference alias), HIP has enum values (prvalues
// -> value alias). Both work over the vendor header on either host path, so the
// #if keys on WWR_SELECTED_* rather than importing a raw module.
// wwrtensorComputeDescriptor_t matches either on the way into a create-op call,
// which is all a portable caller needs.
// ========================================================================

#if defined(WWR_SELECTED_CUDA)
inline constexpr auto &WWRTENSOR_COMPUTE_DESC_16F = ::CUTENSOR_COMPUTE_DESC_16F;
inline constexpr auto &WWRTENSOR_COMPUTE_DESC_16BF = ::CUTENSOR_COMPUTE_DESC_16BF;
inline constexpr auto &WWRTENSOR_COMPUTE_DESC_32F = ::CUTENSOR_COMPUTE_DESC_32F;
inline constexpr auto &WWRTENSOR_COMPUTE_DESC_64F = ::CUTENSOR_COMPUTE_DESC_64F;
#else
WWR_VALUE_RAW(WWRTENSOR_COMPUTE_DESC_16F, CUTENSOR_COMPUTE_DESC_16F, HIPTENSOR_COMPUTE_DESC_16F)
WWR_VALUE_RAW(WWRTENSOR_COMPUTE_DESC_16BF, CUTENSOR_COMPUTE_DESC_16BF, HIPTENSOR_COMPUTE_DESC_16BF)
WWR_VALUE_RAW(WWRTENSOR_COMPUTE_DESC_32F, CUTENSOR_COMPUTE_DESC_32F, HIPTENSOR_COMPUTE_DESC_32F)
WWR_VALUE_RAW(WWRTENSOR_COMPUTE_DESC_64F, CUTENSOR_COMPUTE_DESC_64F, HIPTENSOR_COMPUTE_DESC_64F)
#endif

// ========================================================================
// Library context / version
// ========================================================================

WWR_FUNCTION_RAW(wwrtensorCreate, cutensorCreate, hiptensorCreate)
WWR_FUNCTION_RAW(wwrtensorDestroy, cutensorDestroy, hiptensorDestroy)
WWR_FUNCTION_RAW(wwrtensorGetErrorString, cutensorGetErrorString, hiptensorGetErrorString)
WWR_FUNCTION_RAW(wwrtensorGetVersion, cutensorGetVersion, hiptensorGetVersion)

// ========================================================================
// Plan cache / kernel cache
// ========================================================================

WWR_FUNCTION_RAW(wwrtensorHandleResizePlanCache, cutensorHandleResizePlanCache, hiptensorHandleResizePlanCache)
WWR_FUNCTION_RAW(wwrtensorHandleWritePlanCacheToFile, cutensorHandleWritePlanCacheToFile, hiptensorHandleWritePlanCacheToFile)
WWR_FUNCTION_RAW(wwrtensorHandleReadPlanCacheFromFile, cutensorHandleReadPlanCacheFromFile, hiptensorHandleReadPlanCacheFromFile)
WWR_FUNCTION_RAW(wwrtensorWriteKernelCacheToFile, cutensorWriteKernelCacheToFile, hiptensorWriteKernelCacheToFile)
WWR_FUNCTION_RAW(wwrtensorReadKernelCacheFromFile, cutensorReadKernelCacheFromFile, hiptensorReadKernelCacheFromFile)

// ========================================================================
// Tensor descriptors
// ========================================================================

WWR_FUNCTION_RAW(wwrtensorCreateTensorDescriptor, cutensorCreateTensorDescriptor, hiptensorCreateTensorDescriptor)
WWR_FUNCTION_RAW(wwrtensorDestroyTensorDescriptor, cutensorDestroyTensorDescriptor, hiptensorDestroyTensorDescriptor)

// ========================================================================
// Operation construction
// ========================================================================

WWR_FUNCTION_RAW(wwrtensorCreateElementwiseTrinary, cutensorCreateElementwiseTrinary, hiptensorCreateElementwiseTrinary)
WWR_FUNCTION_RAW(wwrtensorCreateElementwiseBinary, cutensorCreateElementwiseBinary, hiptensorCreateElementwiseBinary)
WWR_FUNCTION_RAW(wwrtensorCreatePermutation, cutensorCreatePermutation, hiptensorCreatePermutation)
WWR_FUNCTION_RAW(wwrtensorCreateContraction, cutensorCreateContraction, hiptensorCreateContraction)
WWR_FUNCTION_RAW(wwrtensorCreateReduction, cutensorCreateReduction, hiptensorCreateReduction)
WWR_FUNCTION_RAW(wwrtensorDestroyOperationDescriptor, cutensorDestroyOperationDescriptor, hiptensorDestroyOperationDescriptor)
WWR_FUNCTION_RAW(wwrtensorOperationDescriptorSetAttribute, cutensorOperationDescriptorSetAttribute, hiptensorOperationDescriptorSetAttribute)
WWR_FUNCTION_RAW(wwrtensorOperationDescriptorGetAttribute, cutensorOperationDescriptorGetAttribute, hiptensorOperationDescriptorGetAttribute)

// ========================================================================
// Plan preference and plan lifecycle (cutensorPlanPreferenceGetAttribute has no
// hipTensor counterpart -- absent here, see src/tensor.cppm)
// ========================================================================

WWR_FUNCTION_RAW(wwrtensorCreatePlanPreference, cutensorCreatePlanPreference, hiptensorCreatePlanPreference)
WWR_FUNCTION_RAW(wwrtensorDestroyPlanPreference, cutensorDestroyPlanPreference, hiptensorDestroyPlanPreference)
WWR_FUNCTION_RAW(wwrtensorPlanPreferenceSetAttribute, cutensorPlanPreferenceSetAttribute, hiptensorPlanPreferenceSetAttribute)
WWR_FUNCTION_RAW(wwrtensorEstimateWorkspaceSize, cutensorEstimateWorkspaceSize, hiptensorEstimateWorkspaceSize)
WWR_FUNCTION_RAW(wwrtensorCreatePlan, cutensorCreatePlan, hiptensorCreatePlan)
WWR_FUNCTION_RAW(wwrtensorDestroyPlan, cutensorDestroyPlan, hiptensorDestroyPlan)
WWR_FUNCTION_RAW(wwrtensorPlanGetAttribute, cutensorPlanGetAttribute, hiptensorPlanGetAttribute)

// ========================================================================
// Operation execution
// ========================================================================

WWR_FUNCTION_RAW(wwrtensorElementwiseTrinaryExecute, cutensorElementwiseTrinaryExecute, hiptensorElementwiseTrinaryExecute)
WWR_FUNCTION_RAW(wwrtensorElementwiseBinaryExecute, cutensorElementwiseBinaryExecute, hiptensorElementwiseBinaryExecute)
WWR_FUNCTION_RAW(wwrtensorPermute, cutensorPermute, hiptensorPermute)
WWR_FUNCTION_RAW(wwrtensorContract, cutensorContract, hiptensorContract)
WWR_FUNCTION_RAW(wwrtensorReduce, cutensorReduce, hiptensorReduce)

// ========================================================================
// Logging (wwrtensorLoggerSetLevel is a forwarding function, not an alias --
// see the file header, divergence 2)
// ========================================================================

WWR_FUNCTION_RAW(wwrtensorLoggerSetCallback, cutensorLoggerSetCallback, hiptensorLoggerSetCallback)
WWR_FUNCTION_RAW(wwrtensorLoggerSetFile, cutensorLoggerSetFile, hiptensorLoggerSetFile)
WWR_FUNCTION_RAW(wwrtensorLoggerOpenFile, cutensorLoggerOpenFile, hiptensorLoggerOpenFile)
WWR_FUNCTION_RAW(wwrtensorLoggerSetMask, cutensorLoggerSetMask, hiptensorLoggerSetMask)
WWR_FUNCTION_RAW(wwrtensorLoggerForceDisable, cutensorLoggerForceDisable, hiptensorLoggerForceDisable)

inline wwrtensorStatus_t wwrtensorLoggerSetLevel(std::int32_t level) {
#if defined(WWR_SELECTED_CUDA)
  return ::cutensorLoggerSetLevel(level);
#else
  return ::hiptensorLoggerSetLevel(static_cast<::hiptensorLogLevel_t>(level));
#endif
}

// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)
