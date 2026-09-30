/**
 * @file hiptensor.cppm
 * @brief hipTensor API module wrapper for wwr project
 *
 * Wraps hiptensor.h -- AMD's tensor primitives library (contraction, reduction,
 * permutation, element-wise), built on composable-kernel. At the ROCm floor
 * (7.1, hipTensor 2.1.0) that header does not exist yet and the same
 * declarations live in hiptensor.hpp; see the include below. CUDA counterpart:
 * wwr.cuda.cutensor. Unlike RCCL/NCCL (§ src/ccl.cppm), hipTensor is NOT a
 * source-compatible reimplementation of cuTENSOR -- it is an independent API
 * that mirrors cuTENSOR's naming closely but is its own library. The neutral
 * wwrtensor* layer bridges only the MEASURED intersection; see src/tensor.cppm.
 *
 * This module wraps the full public surface of hiptensor.h, including the
 * hipTensor-only extras absent from the neutral layer (no cuTENSOR
 * counterpart): the hiptensorLogLevel_t enum and its levels, the
 * HIPTENSOR_ALGO_ACTOR_CRITIC algorithm, the extra compute descriptors
 * (C32F/C64F/NONE/8U/8I/32U/32I), and hiptensorGetHiprtVersion.
 *
 * hipTensor's data-type constants (HIPTENSOR_R_16F ...) and compute descriptors
 * (HIPTENSOR_COMPUTE_DESC_*) are enumerators -- external linkage, re-exported
 * directly with `using`. This is the opposite of cuTENSOR, where the data-type
 * constants are internal-linkage `const` variables (re-declared as constexpr in
 * cuda/cutensor.cppm) and the compute descriptors are `extern const` opaque
 * pointers. See src/tensor.cppm for why the compute descriptors cannot cross
 * the neutral layer through a single alias.
 *
 * Usage:
 *   import wwr.hip.hiptensor;
 */

module;

// Pre-include <array> before the HIP header: hiptensor.h pulls in
// hiptensor_utility.hpp -> hip/hip_complex.h ->
// amd_detail/amd_hip_vector_types.h, which #includes host_defines.h
// immediately before <array>. Outside HIP device-compilation mode
// host_defines.h defines __noinline__ as an *empty* object-like macro -- if
// <array> (and libc++'s __config through it) is first included after that
// point, __has_attribute(__noinline__) inside __config expands to
// __has_attribute() (zero arguments), which clang rejects. Including <array>
// ourselves first sidesteps it via the include guard. Load-bearing, and must
// stay before the HIP header. docs/architecture.md, section 9.
#include <array>
// hipTensor gained a second public header between the floor and the pin: 2.2.0
// (ROCm 7.2) added the C-linkage hiptensor.h, while 2.1.0 (ROCm 7.1, the floor)
// ships only hiptensor.hpp. Both declare all 170 names re-exported below, so
// nothing leaves the surface -- and the linkage difference does not matter,
// because the TU compiles against the header belonging to the libhiptensor.so
// it links. src/hip/CMakeLists.txt's find_path takes the same two NAMES for the
// include dir.
#if __has_include(<hiptensor/hiptensor.h>)
#include <hiptensor/hiptensor.h>
#else
// The one asymmetry: hiptensor.h pulls internal/hiptensor-version.h itself,
// while hiptensor.hpp does NOT pull its .hpp counterpart -- so without this
// second include, hiptensorGetVersion is the single name left undeclared.
#include <hiptensor/hiptensor-version.hpp>
#include <hiptensor/hiptensor.hpp>
#endif

export module wwr.hip.hiptensor;

export namespace wwr::hip {

// ========================================================================
// Opaque handle / descriptor / plan types
// ========================================================================
using ::hiptensorHandle_t;
using ::hiptensorTensorDescriptor_t;
using ::hiptensorOperationDescriptor_t;
using ::hiptensorPlan_t;
using ::hiptensorPlanPreference_t;

// Dependent type from <hip/hip_runtime.h> that appears in the execute signatures
using ::hipStream_t;

// ========================================================================
// Data type (hiptensorDataType_t is its own enum, values chosen to match
// cudaDataType_t -- see test/gpu/tensor.cppm, which pins them)
// ========================================================================
using ::hiptensorDataType_t;
using ::HIPTENSOR_R_32F;
using ::HIPTENSOR_R_64F;
using ::HIPTENSOR_R_16F;
using ::HIPTENSOR_R_8I;
using ::HIPTENSOR_C_32F;
using ::HIPTENSOR_C_64F;
using ::HIPTENSOR_C_16F;
using ::HIPTENSOR_C_8I;
using ::HIPTENSOR_R_8U;
using ::HIPTENSOR_C_8U;
using ::HIPTENSOR_R_32I;
using ::HIPTENSOR_C_32I;
using ::HIPTENSOR_R_32U;
using ::HIPTENSOR_C_32U;
using ::HIPTENSOR_R_16BF;
using ::HIPTENSOR_C_16BF;
using ::HIPTENSOR_R_4I;
using ::HIPTENSOR_C_4I;
using ::HIPTENSOR_R_4U;
using ::HIPTENSOR_C_4U;
using ::HIPTENSOR_R_16I;
using ::HIPTENSOR_C_16I;
using ::HIPTENSOR_R_16U;
using ::HIPTENSOR_C_16U;
using ::HIPTENSOR_R_64I;
using ::HIPTENSOR_C_64I;
using ::HIPTENSOR_R_64U;
using ::HIPTENSOR_C_64U;

// ========================================================================
// Compute descriptors -- an enum (bit-flag values). Contrast cuTENSOR, where
// these are `extern const` opaque pointers.
// ========================================================================
using ::hiptensorComputeDescriptor_t;
using ::HIPTENSOR_COMPUTE_DESC_32F;
using ::HIPTENSOR_COMPUTE_DESC_64F;
using ::HIPTENSOR_COMPUTE_DESC_16F;
using ::HIPTENSOR_COMPUTE_DESC_16BF;
using ::HIPTENSOR_COMPUTE_DESC_C32F;
using ::HIPTENSOR_COMPUTE_DESC_C64F;
using ::HIPTENSOR_COMPUTE_DESC_NONE;
using ::HIPTENSOR_COMPUTE_DESC_8U;
using ::HIPTENSOR_COMPUTE_DESC_8I;
using ::HIPTENSOR_COMPUTE_DESC_32U;
using ::HIPTENSOR_COMPUTE_DESC_32I;

// ========================================================================
// Element-wise operators (hiptensorOperator_t)
// ========================================================================
using ::hiptensorOperator_t;
using ::HIPTENSOR_OP_IDENTITY;
using ::HIPTENSOR_OP_SQRT;
using ::HIPTENSOR_OP_RELU;
using ::HIPTENSOR_OP_CONJ;
using ::HIPTENSOR_OP_RCP;
using ::HIPTENSOR_OP_SIGMOID;
using ::HIPTENSOR_OP_TANH;
using ::HIPTENSOR_OP_EXP;
using ::HIPTENSOR_OP_LOG;
using ::HIPTENSOR_OP_ABS;
using ::HIPTENSOR_OP_NEG;
using ::HIPTENSOR_OP_SIN;
using ::HIPTENSOR_OP_COS;
using ::HIPTENSOR_OP_TAN;
using ::HIPTENSOR_OP_SINH;
using ::HIPTENSOR_OP_COSH;
using ::HIPTENSOR_OP_ASIN;
using ::HIPTENSOR_OP_ACOS;
using ::HIPTENSOR_OP_ATAN;
using ::HIPTENSOR_OP_ASINH;
using ::HIPTENSOR_OP_ACOSH;
using ::HIPTENSOR_OP_ATANH;
using ::HIPTENSOR_OP_CEIL;
using ::HIPTENSOR_OP_FLOOR;
using ::HIPTENSOR_OP_ADD;
using ::HIPTENSOR_OP_MUL;
using ::HIPTENSOR_OP_MAX;
using ::HIPTENSOR_OP_MIN;
using ::HIPTENSOR_OP_UNKNOWN;

// ========================================================================
// Status codes (hiptensorStatus_t)
// ========================================================================
using ::hiptensorStatus_t;
using ::HIPTENSOR_STATUS_SUCCESS;
using ::HIPTENSOR_STATUS_NOT_INITIALIZED;
using ::HIPTENSOR_STATUS_ALLOC_FAILED;
using ::HIPTENSOR_STATUS_INVALID_VALUE;
using ::HIPTENSOR_STATUS_ARCH_MISMATCH;
using ::HIPTENSOR_STATUS_EXECUTION_FAILED;
using ::HIPTENSOR_STATUS_INTERNAL_ERROR;
using ::HIPTENSOR_STATUS_NOT_SUPPORTED;
using ::HIPTENSOR_STATUS_CK_ERROR;
using ::HIPTENSOR_STATUS_HIP_ERROR;
using ::HIPTENSOR_STATUS_INSUFFICIENT_WORKSPACE;
using ::HIPTENSOR_STATUS_INSUFFICIENT_DRIVER;
using ::HIPTENSOR_STATUS_IO_ERROR;

// ========================================================================
// Contraction algorithm (hiptensorAlgo_t)
// ========================================================================
using ::hiptensorAlgo_t;
using ::HIPTENSOR_ALGO_ACTOR_CRITIC;
using ::HIPTENSOR_ALGO_DEFAULT;
using ::HIPTENSOR_ALGO_DEFAULT_PATIENT;

// ========================================================================
// Workspace preference (hiptensorWorksizePreference_t)
// ========================================================================
using ::hiptensorWorksizePreference_t;
using ::HIPTENSOR_WORKSPACE_MIN;
using ::HIPTENSOR_WORKSPACE_DEFAULT;
using ::HIPTENSOR_WORKSPACE_MAX;

// ========================================================================
// Logging levels (hiptensorLogLevel_t) -- no cuTENSOR counterpart type;
// cutensorLoggerSetLevel takes a plain int32_t. See src/tensor.cppm.
// ========================================================================
using ::hiptensorLogLevel_t;
using ::HIPTENSOR_LOG_LEVEL_OFF;
using ::HIPTENSOR_LOG_LEVEL_ERROR;
using ::HIPTENSOR_LOG_LEVEL_PERF_TRACE;
using ::HIPTENSOR_LOG_LEVEL_PERF_HINT;
using ::HIPTENSOR_LOG_LEVEL_HEURISTICS_TRACE;
using ::HIPTENSOR_LOG_LEVEL_API_TRACE;

// ========================================================================
// Operation-descriptor attributes (hiptensorOperationDescriptorAttribute_t)
// ========================================================================
using ::hiptensorOperationDescriptorAttribute_t;
using ::HIPTENSOR_OPERATION_DESCRIPTOR_TAG;
using ::HIPTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE;
using ::HIPTENSOR_OPERATION_DESCRIPTOR_FLOPS;
using ::HIPTENSOR_OPERATION_DESCRIPTOR_MOVED_BYTES;
using ::HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT;
using ::HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_RIGHT;
using ::HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_VALUE;

// ========================================================================
// Plan-preference attributes (hiptensorPlanPreferenceAttribute_t)
// ========================================================================
using ::hiptensorPlanPreferenceAttribute_t;
using ::HIPTENSOR_PLAN_PREFERENCE_AUTOTUNE_MODE;
using ::HIPTENSOR_PLAN_PREFERENCE_CACHE_MODE;
using ::HIPTENSOR_PLAN_PREFERENCE_INCREMENTAL_COUNT;
using ::HIPTENSOR_PLAN_PREFERENCE_ALGO;
using ::HIPTENSOR_PLAN_PREFERENCE_KERNEL_RANK;
using ::HIPTENSOR_PLAN_PREFERENCE_JIT;

// ========================================================================
// Plan attributes (hiptensorPlanAttribute_t)
// ========================================================================
using ::hiptensorPlanAttribute_t;
using ::HIPTENSOR_PLAN_REQUIRED_WORKSPACE;

// ========================================================================
// Autotune / cache / JIT modes
// ========================================================================
using ::hiptensorAutotuneMode_t;
using ::HIPTENSOR_AUTOTUNE_MODE_NONE;
using ::HIPTENSOR_AUTOTUNE_MODE_INCREMENTAL;

using ::hiptensorCacheMode_t;
using ::HIPTENSOR_CACHE_MODE_NONE;
using ::HIPTENSOR_CACHE_MODE_PEDANTIC;

using ::hiptensorJitMode_t;
using ::HIPTENSOR_JIT_MODE_NONE;
using ::HIPTENSOR_JIT_MODE_DEFAULT;

// ========================================================================
// Logging callback type
// ========================================================================
using ::hiptensorLoggerCallback_t;

// ========================================================================
// Library context / version
// ========================================================================
using ::hiptensorCreate;
using ::hiptensorDestroy;
using ::hiptensorGetErrorString;
using ::hiptensorGetVersion;
using ::hiptensorGetHiprtVersion;

// ========================================================================
// Plan cache / kernel cache (persisted to file)
// ========================================================================
using ::hiptensorHandleResizePlanCache;
using ::hiptensorHandleWritePlanCacheToFile;
using ::hiptensorHandleReadPlanCacheFromFile;
using ::hiptensorWriteKernelCacheToFile;
using ::hiptensorReadKernelCacheFromFile;

// ========================================================================
// Tensor descriptors
// ========================================================================
using ::hiptensorCreateTensorDescriptor;
using ::hiptensorDestroyTensorDescriptor;

// ========================================================================
// Operation construction (element-wise, permutation, contraction, reduction)
// ========================================================================
using ::hiptensorCreateElementwiseBinary;
using ::hiptensorCreateElementwiseTrinary;
using ::hiptensorCreatePermutation;
using ::hiptensorCreateContraction;
using ::hiptensorCreateReduction;
using ::hiptensorDestroyOperationDescriptor;
using ::hiptensorOperationDescriptorSetAttribute;
using ::hiptensorOperationDescriptorGetAttribute;

// ========================================================================
// Plan preference and plan lifecycle
// ========================================================================
using ::hiptensorCreatePlanPreference;
using ::hiptensorDestroyPlanPreference;
using ::hiptensorPlanPreferenceSetAttribute;
using ::hiptensorEstimateWorkspaceSize;
using ::hiptensorCreatePlan;
using ::hiptensorDestroyPlan;
using ::hiptensorPlanGetAttribute;

// ========================================================================
// Operation execution
// ========================================================================
using ::hiptensorElementwiseBinaryExecute;
using ::hiptensorElementwiseTrinaryExecute;
using ::hiptensorPermute;
using ::hiptensorContract;
using ::hiptensorReduce;

// ========================================================================
// Logging
// ========================================================================
using ::hiptensorLoggerSetCallback;
using ::hiptensorLoggerSetFile;
using ::hiptensorLoggerOpenFile;
using ::hiptensorLoggerSetLevel;
using ::hiptensorLoggerSetMask;
using ::hiptensorLoggerForceDisable;

} // namespace wwr::hip
