/**
 * @file cutensor.cppm
 * @brief cuTENSOR API module wrapper for wwr project
 *
 * Wraps cutensor.h -- NVIDIA's tensor primitives library (contraction,
 * reduction, permutation, element-wise). HIP counterpart: wwr.hip.hiptensor,
 * AMD's independently-implemented library built on composable-kernel. Unlike
 * NCCL/RCCL (§ src/ccl.cppm), hipTensor is NOT a source-compatible
 * reimplementation -- it is its own API that happens to mirror cuTENSOR's
 * naming closely. The two agree on the overwhelming majority of their surface
 * name-for-name, but the intersection was MEASURED, not assumed; see
 * src/tensor.cppm for the backend-neutral wwrtensor* layer and the divergences
 * it bridges or excludes.
 *
 * This module wraps the full public surface of cutensor.h, including the
 * cuTENSOR-only extras absent from the neutral layer (they have no hipTensor
 * counterpart): the block-sparse API, the trinary contraction, the four extra
 * element-wise operators (MISH/SWISH/SOFT_PLUS/SOFT_SIGN), the three extra
 * contraction algos (GETT/TGETT/TTGT), the extra compute descriptors
 * (TF32/3XTF32/9X16BF/8XINT8/4X16F), GPU_ARCH plan preference,
 * cutensorPlanPreferenceGetAttribute, and cutensorGetCudartVersion.
 *
 * The data-type constants (CUTENSOR_R_16F ...) are namespace-scope `const`
 * variables with internal linkage in cutensor/types.h -- a module cannot export
 * an internal-linkage entity -- so they are re-declared here as `inline
 * constexpr`, carrying the same value (cf. the constexpr-for-macro trick in
 * cuda/nccl.cppm). The compute-descriptor globals (CUTENSOR_COMPUTE_DESC_*) are
 * `extern const` opaque pointers with external linkage and re-export directly.
 *
 * Usage:
 *   import wwr.cuda.cutensor;
 */

module;

#include <cutensor.h>

export module wwr.cuda.cutensor;

export namespace wwr::cuda {

// ========================================================================
// Opaque handle / descriptor / plan types
// ========================================================================
using ::cutensorHandle_t;
using ::cutensorTensorDescriptor_t;
using ::cutensorBlockSparseTensorDescriptor_t;
using ::cutensorOperationDescriptor_t;
using ::cutensorPlan_t;
using ::cutensorPlanPreference_t;
using ::cutensorComputeDescriptor_t;

// Dependent type from <cuda_runtime.h> that appears in the execute signatures
using ::cudaStream_t;

// ========================================================================
// Data type (cutensorDataType_t is an alias of cudaDataType_t)
// ========================================================================
using ::cutensorDataType_t;

// The CUTENSOR_R_*/C_* constants are internal-linkage `const` variables in
// cutensor/types.h; re-declare them as external-linkage constexpr so the module
// can export them (a module cannot export an internal-linkage entity).
inline constexpr cutensorDataType_t CUTENSOR_R_16F  = ::CUTENSOR_R_16F;
inline constexpr cutensorDataType_t CUTENSOR_C_16F  = ::CUTENSOR_C_16F;
inline constexpr cutensorDataType_t CUTENSOR_R_16BF = ::CUTENSOR_R_16BF;
inline constexpr cutensorDataType_t CUTENSOR_C_16BF = ::CUTENSOR_C_16BF;
inline constexpr cutensorDataType_t CUTENSOR_R_32F  = ::CUTENSOR_R_32F;
inline constexpr cutensorDataType_t CUTENSOR_C_32F  = ::CUTENSOR_C_32F;
inline constexpr cutensorDataType_t CUTENSOR_R_64F  = ::CUTENSOR_R_64F;
inline constexpr cutensorDataType_t CUTENSOR_C_64F  = ::CUTENSOR_C_64F;
inline constexpr cutensorDataType_t CUTENSOR_R_4I   = ::CUTENSOR_R_4I;
inline constexpr cutensorDataType_t CUTENSOR_C_4I   = ::CUTENSOR_C_4I;
inline constexpr cutensorDataType_t CUTENSOR_R_4U   = ::CUTENSOR_R_4U;
inline constexpr cutensorDataType_t CUTENSOR_C_4U   = ::CUTENSOR_C_4U;
inline constexpr cutensorDataType_t CUTENSOR_R_8I   = ::CUTENSOR_R_8I;
inline constexpr cutensorDataType_t CUTENSOR_C_8I   = ::CUTENSOR_C_8I;
inline constexpr cutensorDataType_t CUTENSOR_R_8U   = ::CUTENSOR_R_8U;
inline constexpr cutensorDataType_t CUTENSOR_C_8U   = ::CUTENSOR_C_8U;
inline constexpr cutensorDataType_t CUTENSOR_R_16I  = ::CUTENSOR_R_16I;
inline constexpr cutensorDataType_t CUTENSOR_C_16I  = ::CUTENSOR_C_16I;
inline constexpr cutensorDataType_t CUTENSOR_R_16U  = ::CUTENSOR_R_16U;
inline constexpr cutensorDataType_t CUTENSOR_C_16U  = ::CUTENSOR_C_16U;
inline constexpr cutensorDataType_t CUTENSOR_R_32I  = ::CUTENSOR_R_32I;
inline constexpr cutensorDataType_t CUTENSOR_C_32I  = ::CUTENSOR_C_32I;
inline constexpr cutensorDataType_t CUTENSOR_R_32U  = ::CUTENSOR_R_32U;
inline constexpr cutensorDataType_t CUTENSOR_C_32U  = ::CUTENSOR_C_32U;
inline constexpr cutensorDataType_t CUTENSOR_R_64I  = ::CUTENSOR_R_64I;
inline constexpr cutensorDataType_t CUTENSOR_C_64I  = ::CUTENSOR_C_64I;
inline constexpr cutensorDataType_t CUTENSOR_R_64U  = ::CUTENSOR_R_64U;
inline constexpr cutensorDataType_t CUTENSOR_C_64U  = ::CUTENSOR_C_64U;

// ========================================================================
// Compute descriptors -- opaque `extern const` pointers (runtime globals, NOT
// compile-time constants; the type is declared with the handles above).
// Contrast hipTensor, where these are enum values.
// ========================================================================
using ::CUTENSOR_COMPUTE_DESC_16F;
using ::CUTENSOR_COMPUTE_DESC_16BF;
using ::CUTENSOR_COMPUTE_DESC_TF32;
using ::CUTENSOR_COMPUTE_DESC_3XTF32;
using ::CUTENSOR_COMPUTE_DESC_32F;
using ::CUTENSOR_COMPUTE_DESC_64F;
using ::CUTENSOR_COMPUTE_DESC_9X16BF;
using ::CUTENSOR_COMPUTE_DESC_8XINT8;
using ::CUTENSOR_COMPUTE_DESC_4X16F;

// ========================================================================
// Element-wise operators (cutensorOperator_t)
// ========================================================================
using ::cutensorOperator_t;
using ::CUTENSOR_OP_IDENTITY;
using ::CUTENSOR_OP_SQRT;
using ::CUTENSOR_OP_RELU;
using ::CUTENSOR_OP_CONJ;
using ::CUTENSOR_OP_RCP;
using ::CUTENSOR_OP_SIGMOID;
using ::CUTENSOR_OP_TANH;
using ::CUTENSOR_OP_EXP;
using ::CUTENSOR_OP_LOG;
using ::CUTENSOR_OP_ABS;
using ::CUTENSOR_OP_NEG;
using ::CUTENSOR_OP_SIN;
using ::CUTENSOR_OP_COS;
using ::CUTENSOR_OP_TAN;
using ::CUTENSOR_OP_SINH;
using ::CUTENSOR_OP_COSH;
using ::CUTENSOR_OP_ASIN;
using ::CUTENSOR_OP_ACOS;
using ::CUTENSOR_OP_ATAN;
using ::CUTENSOR_OP_ASINH;
using ::CUTENSOR_OP_ACOSH;
using ::CUTENSOR_OP_ATANH;
using ::CUTENSOR_OP_CEIL;
using ::CUTENSOR_OP_FLOOR;
using ::CUTENSOR_OP_MISH;
using ::CUTENSOR_OP_SWISH;
using ::CUTENSOR_OP_SOFT_PLUS;
using ::CUTENSOR_OP_SOFT_SIGN;
using ::CUTENSOR_OP_ADD;
using ::CUTENSOR_OP_MUL;
using ::CUTENSOR_OP_MAX;
using ::CUTENSOR_OP_MIN;
using ::CUTENSOR_OP_UNKNOWN;

// ========================================================================
// Status codes (cutensorStatus_t)
// ========================================================================
using ::cutensorStatus_t;
using ::CUTENSOR_STATUS_SUCCESS;
using ::CUTENSOR_STATUS_NOT_INITIALIZED;
using ::CUTENSOR_STATUS_ALLOC_FAILED;
using ::CUTENSOR_STATUS_INVALID_VALUE;
using ::CUTENSOR_STATUS_ARCH_MISMATCH;
using ::CUTENSOR_STATUS_MAPPING_ERROR;
using ::CUTENSOR_STATUS_EXECUTION_FAILED;
using ::CUTENSOR_STATUS_INTERNAL_ERROR;
using ::CUTENSOR_STATUS_NOT_SUPPORTED;
using ::CUTENSOR_STATUS_LICENSE_ERROR;
using ::CUTENSOR_STATUS_CUBLAS_ERROR;
using ::CUTENSOR_STATUS_CUDA_ERROR;
using ::CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE;
using ::CUTENSOR_STATUS_INSUFFICIENT_DRIVER;
using ::CUTENSOR_STATUS_IO_ERROR;

// ========================================================================
// Contraction algorithm (cutensorAlgo_t)
// ========================================================================
using ::cutensorAlgo_t;
using ::CUTENSOR_ALGO_DEFAULT_PATIENT;
using ::CUTENSOR_ALGO_GETT;
using ::CUTENSOR_ALGO_TGETT;
using ::CUTENSOR_ALGO_TTGT;
using ::CUTENSOR_ALGO_DEFAULT;

// ========================================================================
// Workspace preference (cutensorWorksizePreference_t)
// ========================================================================
using ::cutensorWorksizePreference_t;
using ::CUTENSOR_WORKSPACE_MIN;
using ::CUTENSOR_WORKSPACE_DEFAULT;
using ::CUTENSOR_WORKSPACE_MAX;

// ========================================================================
// Operation-descriptor attributes (cutensorOperationDescriptorAttribute_t)
// ========================================================================
using ::cutensorOperationDescriptorAttribute_t;
using ::CUTENSOR_OPERATION_DESCRIPTOR_TAG;
using ::CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE;
using ::CUTENSOR_OPERATION_DESCRIPTOR_FLOPS;
using ::CUTENSOR_OPERATION_DESCRIPTOR_MOVED_BYTES;
using ::CUTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT;
using ::CUTENSOR_OPERATION_DESCRIPTOR_PADDING_RIGHT;
using ::CUTENSOR_OPERATION_DESCRIPTOR_PADDING_VALUE;
using ::CUTENSOR_OPERATION_DESCRIPTOR_BLOCKSPARSE_REPRODUCIBLE;

// ========================================================================
// Plan-preference attributes (cutensorPlanPreferenceAttribute_t)
// ========================================================================
using ::cutensorPlanPreferenceAttribute_t;
using ::CUTENSOR_PLAN_PREFERENCE_AUTOTUNE_MODE;
using ::CUTENSOR_PLAN_PREFERENCE_CACHE_MODE;
using ::CUTENSOR_PLAN_PREFERENCE_INCREMENTAL_COUNT;
using ::CUTENSOR_PLAN_PREFERENCE_ALGO;
using ::CUTENSOR_PLAN_PREFERENCE_KERNEL_RANK;
using ::CUTENSOR_PLAN_PREFERENCE_JIT;
using ::CUTENSOR_PLAN_PREFERENCE_GPU_ARCH;

// ========================================================================
// Plan attributes (cutensorPlanAttribute_t)
// ========================================================================
using ::cutensorPlanAttribute_t;
using ::CUTENSOR_PLAN_REQUIRED_WORKSPACE;

// ========================================================================
// Autotune / cache / JIT modes
// ========================================================================
using ::cutensorAutotuneMode_t;
using ::CUTENSOR_AUTOTUNE_MODE_NONE;
using ::CUTENSOR_AUTOTUNE_MODE_INCREMENTAL;

using ::cutensorCacheMode_t;
using ::CUTENSOR_CACHE_MODE_NONE;
using ::CUTENSOR_CACHE_MODE_PEDANTIC;

using ::cutensorJitMode_t;
using ::CUTENSOR_JIT_MODE_NONE;
using ::CUTENSOR_JIT_MODE_DEFAULT;

// ========================================================================
// Logging callback type
// ========================================================================
using ::cutensorLoggerCallback_t;

// ========================================================================
// Library context / version
// ========================================================================
using ::cutensorCreate;
using ::cutensorDestroy;
using ::cutensorGetErrorString;
using ::cutensorGetVersion;
using ::cutensorGetCudartVersion;

// ========================================================================
// Plan cache / kernel cache (persisted to file)
// ========================================================================
using ::cutensorHandleResizePlanCache;
using ::cutensorHandleWritePlanCacheToFile;
using ::cutensorHandleReadPlanCacheFromFile;
using ::cutensorWriteKernelCacheToFile;
using ::cutensorReadKernelCacheFromFile;

// ========================================================================
// Tensor descriptors
// ========================================================================
using ::cutensorCreateTensorDescriptor;
using ::cutensorDestroyTensorDescriptor;
using ::cutensorCreateBlockSparseTensorDescriptor;
using ::cutensorDestroyBlockSparseTensorDescriptor;

// ========================================================================
// Operation construction (element-wise, permutation, contraction, reduction)
// ========================================================================
using ::cutensorCreateElementwiseTrinary;
using ::cutensorCreateElementwiseBinary;
using ::cutensorCreatePermutation;
using ::cutensorCreateContraction;
using ::cutensorCreateContractionTrinary;
using ::cutensorCreateBlockSparseContraction;
using ::cutensorCreateReduction;
using ::cutensorDestroyOperationDescriptor;
using ::cutensorOperationDescriptorSetAttribute;
using ::cutensorOperationDescriptorGetAttribute;

// ========================================================================
// Plan preference and plan lifecycle
// ========================================================================
using ::cutensorCreatePlanPreference;
using ::cutensorDestroyPlanPreference;
using ::cutensorPlanPreferenceGetAttribute;
using ::cutensorPlanPreferenceSetAttribute;
using ::cutensorEstimateWorkspaceSize;
using ::cutensorCreatePlan;
using ::cutensorDestroyPlan;
using ::cutensorPlanGetAttribute;

// ========================================================================
// Operation execution
// ========================================================================
using ::cutensorElementwiseTrinaryExecute;
using ::cutensorElementwiseBinaryExecute;
using ::cutensorPermute;
using ::cutensorContract;
using ::cutensorContractTrinary;
using ::cutensorBlockSparseContract;
using ::cutensorReduce;

// ========================================================================
// Logging
// ========================================================================
using ::cutensorLoggerSetCallback;
using ::cutensorLoggerSetFile;
using ::cutensorLoggerOpenFile;
using ::cutensorLoggerSetLevel;
using ::cutensorLoggerSetMask;
using ::cutensorLoggerForceDisable;

} // namespace wwr::cuda
