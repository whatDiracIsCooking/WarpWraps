// cupti.cppm - Compile-time tests for gpumod.cuda.cupti

module;

#include "test/shared/link_check.h"

export module gpumod.test.cuda.cupti;

import std;
import gpumod.cuda.cupti;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time tests for gpumod.cuda.cupti
//
// The module is a pure re-export (using declarations). We verify at
// compile-time that:
//   1. CUptiResult is an enum and key enumerators have their documented values
//   2. Key callback domain, resource, sync, and state enumerators are correct
//   3. CUpti_ActivityKind enumerators have their documented integer values
//   4. Opaque/struct types satisfy expected type-trait properties
//   5. Function pointer typedefs are function pointer types
//   6. Every non-inline function symbol resolves at link time (GPUMOD_LINK_CHECK)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace gpumod::cuda::test {

using namespace gpumod::cuda;
using std::uint32_t;

// ────────────────────────────────────────────────────────────────────────
// CUptiResult enum type and selected enumerator values
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_enum_v<CUptiResult>);

static_assert(static_cast<int>(CUPTI_SUCCESS) == 0);
static_assert(static_cast<int>(CUPTI_ERROR_INVALID_PARAMETER) == 1);
static_assert(static_cast<int>(CUPTI_ERROR_INVALID_DEVICE) == 2);
static_assert(static_cast<int>(CUPTI_ERROR_INVALID_CONTEXT) == 3);
static_assert(static_cast<int>(CUPTI_ERROR_INVALID_EVENT_DOMAIN_ID) == 4);
static_assert(static_cast<int>(CUPTI_ERROR_INVALID_EVENT_ID) == 5);
static_assert(static_cast<int>(CUPTI_ERROR_INVALID_EVENT_NAME) == 6);
static_assert(static_cast<int>(CUPTI_ERROR_INVALID_OPERATION) == 7);
static_assert(static_cast<int>(CUPTI_ERROR_OUT_OF_MEMORY) == 8);
static_assert(static_cast<int>(CUPTI_ERROR_HARDWARE) == 9);
static_assert(static_cast<int>(CUPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT) == 10);
static_assert(static_cast<int>(CUPTI_ERROR_API_NOT_IMPLEMENTED) == 11);
static_assert(static_cast<int>(CUPTI_ERROR_MAX_LIMIT_REACHED) == 12);
static_assert(static_cast<int>(CUPTI_ERROR_NOT_READY) == 13);
static_assert(static_cast<int>(CUPTI_ERROR_NOT_COMPATIBLE) == 14);
static_assert(static_cast<int>(CUPTI_ERROR_NOT_INITIALIZED) == 15);
static_assert(static_cast<int>(CUPTI_ERROR_INVALID_METRIC_ID) == 16);
static_assert(static_cast<int>(CUPTI_ERROR_INVALID_METRIC_NAME) == 17);
static_assert(static_cast<int>(CUPTI_ERROR_QUEUE_EMPTY) == 18);
static_assert(static_cast<int>(CUPTI_ERROR_INVALID_HANDLE) == 19);
static_assert(static_cast<int>(CUPTI_ERROR_INVALID_STREAM) == 20);
static_assert(static_cast<int>(CUPTI_ERROR_INVALID_KIND) == 21);
static_assert(static_cast<int>(CUPTI_ERROR_INVALID_EVENT_VALUE) == 22);
static_assert(static_cast<int>(CUPTI_ERROR_DISABLED) == 23);
static_assert(static_cast<int>(CUPTI_ERROR_INVALID_MODULE) == 24);
static_assert(static_cast<int>(CUPTI_ERROR_INVALID_METRIC_VALUE) == 25);
static_assert(static_cast<int>(CUPTI_ERROR_HARDWARE_BUSY) == 26);
static_assert(static_cast<int>(CUPTI_ERROR_NOT_SUPPORTED) == 27);
static_assert(static_cast<int>(CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED) == 28);
static_assert(static_cast<int>(CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_DEVICE) == 29);
static_assert(static_cast<int>(CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_NON_P2P_DEVICES) == 30);
static_assert(static_cast<int>(CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_WITH_MPS) == 31);
static_assert(static_cast<int>(CUPTI_ERROR_CDP_TRACING_NOT_SUPPORTED) == 32);
static_assert(static_cast<int>(CUPTI_ERROR_VIRTUALIZED_DEVICE_NOT_SUPPORTED) == 33);
static_assert(static_cast<int>(CUPTI_ERROR_CUDA_COMPILER_NOT_COMPATIBLE) == 34);
static_assert(static_cast<int>(CUPTI_ERROR_INSUFFICIENT_PRIVILEGES) == 35);
static_assert(static_cast<int>(CUPTI_ERROR_OLD_PROFILER_API_INITIALIZED) == 36);
static_assert(static_cast<int>(CUPTI_ERROR_OPENACC_UNDEFINED_ROUTINE) == 37);
static_assert(static_cast<int>(CUPTI_ERROR_LEGACY_PROFILER_NOT_SUPPORTED) == 38);
static_assert(static_cast<int>(CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED) == 39);
static_assert(static_cast<int>(CUPTI_ERROR_VIRTUALIZED_DEVICE_INSUFFICIENT_PRIVILEGES) == 40);
static_assert(static_cast<int>(CUPTI_ERROR_CONFIDENTIAL_COMPUTING_NOT_SUPPORTED) == 41);
static_assert(static_cast<int>(CUPTI_ERROR_CMP_DEVICE_NOT_SUPPORTED) == 42);
static_assert(static_cast<int>(CUPTI_ERROR_MIG_DEVICE_NOT_SUPPORTED) == 43);
static_assert(static_cast<int>(CUPTI_ERROR_SLI_DEVICE_NOT_SUPPORTED) == 44);
static_assert(static_cast<int>(CUPTI_ERROR_WSL_DEVICE_NOT_SUPPORTED) == 45);
static_assert(static_cast<int>(CUPTI_ERROR_INVALID_CHIP_NAME) == 46);
static_assert(static_cast<int>(CUPTI_ERROR_UNKNOWN) == 999);

// ────────────────────────────────────────────────────────────────────────
// Callback API enum types and selected values
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_enum_v<CUpti_ApiCallbackSite>);
static_assert(static_cast<int>(CUPTI_API_ENTER) == 0);
static_assert(static_cast<int>(CUPTI_API_EXIT) == 1);

static_assert(std::is_enum_v<CUpti_CallbackDomain>);
static_assert(static_cast<int>(CUPTI_CB_DOMAIN_INVALID) == 0);
static_assert(static_cast<int>(CUPTI_CB_DOMAIN_DRIVER_API) == 1);
static_assert(static_cast<int>(CUPTI_CB_DOMAIN_RUNTIME_API) == 2);
static_assert(static_cast<int>(CUPTI_CB_DOMAIN_RESOURCE) == 3);
static_assert(static_cast<int>(CUPTI_CB_DOMAIN_SYNCHRONIZE) == 4);
static_assert(static_cast<int>(CUPTI_CB_DOMAIN_NVTX) == 5);
static_assert(static_cast<int>(CUPTI_CB_DOMAIN_STATE) == 6);

static_assert(std::is_enum_v<CUpti_CallbackIdResource>);
static_assert(static_cast<int>(CUPTI_CBID_RESOURCE_INVALID) == 0);
static_assert(static_cast<int>(CUPTI_CBID_RESOURCE_CONTEXT_CREATED) == 1);
static_assert(static_cast<int>(CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING) == 2);
static_assert(static_cast<int>(CUPTI_CBID_RESOURCE_STREAM_CREATED) == 3);
static_assert(static_cast<int>(CUPTI_CBID_RESOURCE_STREAM_DESTROY_STARTING) == 4);
static_assert(static_cast<int>(CUPTI_CBID_RESOURCE_CU_INIT_FINISHED) == 5);
static_assert(static_cast<int>(CUPTI_CBID_RESOURCE_MODULE_LOADED) == 6);
static_assert(static_cast<int>(CUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING) == 7);
static_assert(static_cast<int>(CUPTI_CBID_RESOURCE_MODULE_PROFILED) == 8);
static_assert(static_cast<int>(CUPTI_CBID_RESOURCE_GRAPH_CREATED) == 9);
static_assert(static_cast<int>(CUPTI_CBID_RESOURCE_GRAPH_DESTROY_STARTING) == 10);

static_assert(std::is_enum_v<CUpti_CallbackIdSync>);
static_assert(static_cast<int>(CUPTI_CBID_SYNCHRONIZE_INVALID) == 0);
static_assert(static_cast<int>(CUPTI_CBID_SYNCHRONIZE_STREAM_SYNCHRONIZED) == 1);
static_assert(static_cast<int>(CUPTI_CBID_SYNCHRONIZE_CONTEXT_SYNCHRONIZED) == 2);

static_assert(std::is_enum_v<CUpti_CallbackIdState>);
static_assert(static_cast<int>(CUPTI_CBID_STATE_INVALID) == 0);
static_assert(static_cast<int>(CUPTI_CBID_STATE_FATAL_ERROR) == 1);
static_assert(static_cast<int>(CUPTI_CBID_STATE_ERROR) == 2);
static_assert(static_cast<int>(CUPTI_CBID_STATE_WARNING) == 3);

// ────────────────────────────────────────────────────────────────────────
// Callback data structures are class/struct types
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_class_v<CUpti_CallbackData>);
static_assert(std::is_class_v<CUpti_ResourceData>);
static_assert(std::is_class_v<CUpti_ModuleResourceData>);
static_assert(std::is_class_v<CUpti_GraphData>);
static_assert(std::is_class_v<CUpti_SynchronizeData>);
static_assert(std::is_class_v<CUpti_NvtxData>);
static_assert(std::is_class_v<CUpti_StreamAttrData>);
static_assert(std::is_class_v<CUpti_StateData>);
static_assert(std::is_class_v<CUpti_SubscriberParams>);

// CUpti_CallbackId is a uint32_t typedef
static_assert(std::is_same_v<CUpti_CallbackId, uint32_t>);

// ────────────────────────────────────────────────────────────────────────
// CUpti_CallbackFunc is a function pointer type
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_pointer_v<CUpti_CallbackFunc>);
static_assert(std::is_function_v<std::remove_pointer_t<CUpti_CallbackFunc>>);

// ────────────────────────────────────────────────────────────────────────
// CUpti_SubscriberHandle is a pointer (opaque handle)
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_pointer_v<CUpti_SubscriberHandle>);

// ────────────────────────────────────────────────────────────────────────
// Driver/runtime cbid enums
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_enum_v<CUpti_driver_api_trace_cbid>);
static_assert(static_cast<int>(CUPTI_DRIVER_TRACE_CBID_INVALID) == 0);
static_assert(static_cast<int>(CUPTI_DRIVER_TRACE_CBID_cuInit) == 1);
static_assert(static_cast<int>(CUPTI_DRIVER_TRACE_CBID_SIZE) == 807);

static_assert(std::is_enum_v<CUpti_runtime_api_trace_cbid>);
static_assert(static_cast<int>(CUPTI_RUNTIME_TRACE_CBID_INVALID) == 0);
static_assert(static_cast<int>(CUPTI_RUNTIME_TRACE_CBID_SIZE) == 523);

static_assert(std::is_enum_v<CUpti_nvtx_api_trace_cbid>);
static_assert(static_cast<int>(CUPTI_CBID_NVTX_INVALID) == 0);
static_assert(static_cast<int>(CUPTI_CBID_NVTX_nvtxMarkA) == 1);

// ────────────────────────────────────────────────────────────────────────
// CUpti_ActivityKind enum and selected enumerator values
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_enum_v<CUpti_ActivityKind>);

static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_INVALID) == 0);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_MEMCPY) == 1);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_MEMSET) == 2);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_KERNEL) == 3);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_DRIVER) == 4);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_RUNTIME) == 5);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_EVENT) == 6);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_METRIC) == 7);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_DEVICE) == 8);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_CONTEXT) == 9);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) == 10);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_NAME) == 11);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_MARKER) == 12);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_MARKER_DATA) == 13);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR) == 14);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_GLOBAL_ACCESS) == 15);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_BRANCH) == 16);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_OVERHEAD) == 17);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_CDP_KERNEL) == 18);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_PREEMPTION) == 19);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_ENVIRONMENT) == 20);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_EVENT_INSTANCE) == 21);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_MEMCPY2) == 22);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_METRIC_INSTANCE) == 23);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_INSTRUCTION_EXECUTION) == 24);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER) == 25);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_FUNCTION) == 26);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_MODULE) == 27);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE) == 28);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_SHARED_ACCESS) == 29);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_PC_SAMPLING) == 30);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_PC_SAMPLING_RECORD_INFO) == 31);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_INSTRUCTION_CORRELATION) == 32);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_OPENACC_DATA) == 33);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_OPENACC_LAUNCH) == 34);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_OPENACC_OTHER) == 35);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_CUDA_EVENT) == 36);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_STREAM) == 37);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_SYNCHRONIZATION) == 38);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION) == 39);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_NVLINK) == 40);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_INSTANTANEOUS_EVENT) == 41);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_INSTANTANEOUS_EVENT_INSTANCE) == 42);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_INSTANTANEOUS_METRIC) == 43);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_INSTANTANEOUS_METRIC_INSTANCE) == 44);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_MEMORY) == 45);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_PCIE) == 46);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_OPENMP) == 47);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_INTERNAL_LAUNCH_API) == 48);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_MEMORY2) == 49);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_MEMORY_POOL) == 50);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_GRAPH_TRACE) == 51);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_JIT) == 52);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_DEVICE_GRAPH_TRACE) == 53);
static_assert(static_cast<int>(CUPTI_ACTIVITY_KIND_MEM_DECOMPRESS) == 54);

// ────────────────────────────────────────────────────────────────────────
// CUpti_ActivityObjectKind enum values
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_enum_v<CUpti_ActivityObjectKind>);
static_assert(static_cast<int>(CUPTI_ACTIVITY_OBJECT_UNKNOWN) == 0);
static_assert(static_cast<int>(CUPTI_ACTIVITY_OBJECT_PROCESS) == 1);
static_assert(static_cast<int>(CUPTI_ACTIVITY_OBJECT_THREAD) == 2);
static_assert(static_cast<int>(CUPTI_ACTIVITY_OBJECT_DEVICE) == 3);
static_assert(static_cast<int>(CUPTI_ACTIVITY_OBJECT_CONTEXT) == 4);
static_assert(static_cast<int>(CUPTI_ACTIVITY_OBJECT_STREAM) == 5);

// ────────────────────────────────────────────────────────────────────────
// CUpti_ActivityOverheadKind selected values
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_enum_v<CUpti_ActivityOverheadKind>);
static_assert(static_cast<int>(CUPTI_ACTIVITY_OVERHEAD_UNKNOWN) == 0);
static_assert(static_cast<int>(CUPTI_ACTIVITY_OVERHEAD_DRIVER_COMPILER) == 1);

// ────────────────────────────────────────────────────────────────────────
// Activity record base struct
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_class_v<CUpti_Activity>);
static_assert(std::is_standard_layout_v<CUpti_Activity>);

// ────────────────────────────────────────────────────────────────────────
// Activity configuration structs
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_class_v<CUpti_ActivityUnifiedMemoryCounterConfig>);
static_assert(std::is_class_v<CUpti_ActivityAutoBoostState>);
static_assert(std::is_class_v<CUpti_ActivityPCSamplingConfig>);

// ────────────────────────────────────────────────────────────────────────
// Activity buffer callback function pointer types
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_pointer_v<CUpti_BuffersCallbackRequestFunc>);
static_assert(std::is_function_v<std::remove_pointer_t<CUpti_BuffersCallbackRequestFunc>>);

static_assert(std::is_pointer_v<CUpti_BuffersCallbackCompleteFunc>);
static_assert(std::is_function_v<std::remove_pointer_t<CUpti_BuffersCallbackCompleteFunc>>);

static_assert(std::is_pointer_v<CUpti_TimestampCallbackFunc>);
static_assert(std::is_function_v<std::remove_pointer_t<CUpti_TimestampCallbackFunc>>);

// ────────────────────────────────────────────────────────────────────────
// Event API types
// ────────────────────────────────────────────────────────────────────────

// CUpti_EventID and CUpti_EventDomainID are uint32_t
static_assert(std::is_same_v<CUpti_EventID, uint32_t>);
static_assert(std::is_same_v<CUpti_EventDomainID, uint32_t>);

static_assert(std::is_enum_v<CUpti_DeviceAttributeDeviceClass>);
static_assert(std::is_enum_v<CUpti_DeviceAttribute>);
static_assert(std::is_enum_v<CUpti_EventDomainAttribute>);
static_assert(std::is_enum_v<CUpti_EventCollectionMethod>);
static_assert(std::is_enum_v<CUpti_EventGroupAttribute>);
static_assert(std::is_enum_v<CUpti_EventProfilingScope>);
static_assert(std::is_enum_v<CUpti_EventAttribute>);
static_assert(std::is_enum_v<CUpti_EventCollectionMode>);
static_assert(std::is_enum_v<CUpti_EventCategory>);
static_assert(std::is_enum_v<CUpti_ReadEventFlags>);

static_assert(std::is_class_v<CUpti_EventGroupSet>);
static_assert(std::is_class_v<CUpti_EventGroupSets>);

// CUpti_KernelReplayUpdateFunc is a function pointer type
static_assert(std::is_pointer_v<CUpti_KernelReplayUpdateFunc>);
static_assert(std::is_function_v<std::remove_pointer_t<CUpti_KernelReplayUpdateFunc>>);

// ────────────────────────────────────────────────────────────────────────
// Metric API types
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_same_v<CUpti_MetricID, uint32_t>);
static_assert(std::is_enum_v<CUpti_MetricCategory>);
static_assert(std::is_enum_v<CUpti_MetricEvaluationMode>);
static_assert(std::is_enum_v<CUpti_MetricValueKind>);
static_assert(std::is_enum_v<CUpti_MetricValueUtilizationLevel>);
static_assert(std::is_enum_v<CUpti_MetricAttribute>);
static_assert(std::is_enum_v<CUpti_MetricPropertyDeviceClass>);
static_assert(std::is_enum_v<CUpti_MetricPropertyID>);

// ────────────────────────────────────────────────────────────────────────
// Link-time symbol resolution
// Forces the linker to resolve every re-exported function symbol.
// ────────────────────────────────────────────────────────────────────────

// Result API
GPUMOD_LINK_CHECK(cuptiGetResultString)
GPUMOD_LINK_CHECK(cuptiGetErrorMessage)

// Version API
GPUMOD_LINK_CHECK(cuptiGetVersion)

// Callback API
GPUMOD_LINK_CHECK(cuptiSupportedDomains)
GPUMOD_LINK_CHECK(cuptiSubscribe)
GPUMOD_LINK_CHECK(cuptiSubscribe_v2)
GPUMOD_LINK_CHECK(cuptiUnsubscribe)
GPUMOD_LINK_CHECK(cuptiGetCallbackState)
GPUMOD_LINK_CHECK(cuptiEnableCallback)
GPUMOD_LINK_CHECK(cuptiEnableDomain)
GPUMOD_LINK_CHECK(cuptiEnableAllDomains)
GPUMOD_LINK_CHECK(cuptiGetCallbackName)

// Activity — timestamp and ID queries
GPUMOD_LINK_CHECK(cuptiGetTimestamp)
GPUMOD_LINK_CHECK(cuptiGetContextId)
GPUMOD_LINK_CHECK(cuptiGetStreamId)
GPUMOD_LINK_CHECK(cuptiGetStreamIdEx)
GPUMOD_LINK_CHECK(cuptiGetDeviceId)
GPUMOD_LINK_CHECK(cuptiGetGraphNodeId)
GPUMOD_LINK_CHECK(cuptiGetGraphId)
GPUMOD_LINK_CHECK(cuptiGetGraphExecId)

// Activity — enable/disable
GPUMOD_LINK_CHECK(cuptiActivityEnable)
GPUMOD_LINK_CHECK(cuptiActivityEnableAndDump)
GPUMOD_LINK_CHECK(cuptiActivityDisable)
GPUMOD_LINK_CHECK(cuptiActivityEnableContext)
GPUMOD_LINK_CHECK(cuptiActivityDisableContext)

// Activity — buffer management
GPUMOD_LINK_CHECK(cuptiActivityGetNumDroppedRecords)
GPUMOD_LINK_CHECK(cuptiActivityGetNextRecord)
GPUMOD_LINK_CHECK(cuptiActivityRegisterCallbacks)
GPUMOD_LINK_CHECK(cuptiActivityFlush)
GPUMOD_LINK_CHECK(cuptiActivityFlushAll)

// Activity — attribute get/set
GPUMOD_LINK_CHECK(cuptiActivityGetAttribute)
GPUMOD_LINK_CHECK(cuptiActivitySetAttribute)

// Activity — configuration
GPUMOD_LINK_CHECK(cuptiActivityConfigureUnifiedMemoryCounter)
GPUMOD_LINK_CHECK(cuptiGetAutoBoostState)
GPUMOD_LINK_CHECK(cuptiActivityConfigurePCSampling)

// Activity — miscellaneous
GPUMOD_LINK_CHECK(cuptiGetLastError)
GPUMOD_LINK_CHECK(cuptiSetThreadIdType)
GPUMOD_LINK_CHECK(cuptiGetThreadIdType)
GPUMOD_LINK_CHECK(cuptiComputeCapabilitySupported)
GPUMOD_LINK_CHECK(cuptiDeviceSupported)
GPUMOD_LINK_CHECK(cuptiDeviceVirtualizationMode)
GPUMOD_LINK_CHECK(cuptiFinalize)

// Activity — external correlation
GPUMOD_LINK_CHECK(cuptiActivityPushExternalCorrelationId)
GPUMOD_LINK_CHECK(cuptiActivityPopExternalCorrelationId)

// Activity — optional feature enables
GPUMOD_LINK_CHECK(cuptiActivityEnableLatencyTimestamps)
GPUMOD_LINK_CHECK(cuptiActivityFlushPeriod)
GPUMOD_LINK_CHECK(cuptiActivityEnableLaunchAttributes)
GPUMOD_LINK_CHECK(cuptiActivityRegisterTimestampCallback)
GPUMOD_LINK_CHECK(cuptiActivityEnableDeviceGraph)
GPUMOD_LINK_CHECK(cuptiActivityEnableDriverApi)
GPUMOD_LINK_CHECK(cuptiActivityEnableRuntimeApi)
GPUMOD_LINK_CHECK(cuptiActivityEnableHWTrace)
GPUMOD_LINK_CHECK(cuptiActivityEnableAllocationSource)
GPUMOD_LINK_CHECK(cuptiActivityEnableAllSyncRecords)
GPUMOD_LINK_CHECK(cuptiActivityEnableCudaEventDeviceTimestamps)

// Event API (deprecated in CUDA 13.0, symbols still present in libcupti)
GPUMOD_LINK_CHECK(cuptiSetEventCollectionMode)
GPUMOD_LINK_CHECK(cuptiDeviceGetAttribute)
GPUMOD_LINK_CHECK(cuptiDeviceGetNumEventDomains)
GPUMOD_LINK_CHECK(cuptiDeviceEnumEventDomains)
GPUMOD_LINK_CHECK(cuptiDeviceGetEventDomainAttribute)
GPUMOD_LINK_CHECK(cuptiGetNumEventDomains)
GPUMOD_LINK_CHECK(cuptiEnumEventDomains)
GPUMOD_LINK_CHECK(cuptiEventDomainGetAttribute)
GPUMOD_LINK_CHECK(cuptiEventDomainGetNumEvents)
GPUMOD_LINK_CHECK(cuptiEventDomainEnumEvents)
GPUMOD_LINK_CHECK(cuptiEventGetAttribute)
GPUMOD_LINK_CHECK(cuptiEventGetIdFromName)
GPUMOD_LINK_CHECK(cuptiEventGroupCreate)
GPUMOD_LINK_CHECK(cuptiEventGroupDestroy)
GPUMOD_LINK_CHECK(cuptiEventGroupGetAttribute)
GPUMOD_LINK_CHECK(cuptiEventGroupSetAttribute)
GPUMOD_LINK_CHECK(cuptiEventGroupAddEvent)
GPUMOD_LINK_CHECK(cuptiEventGroupRemoveEvent)
GPUMOD_LINK_CHECK(cuptiEventGroupRemoveAllEvents)
GPUMOD_LINK_CHECK(cuptiEventGroupResetAllEvents)
GPUMOD_LINK_CHECK(cuptiEventGroupEnable)
GPUMOD_LINK_CHECK(cuptiEventGroupDisable)
GPUMOD_LINK_CHECK(cuptiEventGroupReadEvent)
GPUMOD_LINK_CHECK(cuptiEventGroupReadAllEvents)
GPUMOD_LINK_CHECK(cuptiEventGroupSetsCreate)
GPUMOD_LINK_CHECK(cuptiEventGroupSetsDestroy)
GPUMOD_LINK_CHECK(cuptiEventGroupSetEnable)
GPUMOD_LINK_CHECK(cuptiEventGroupSetDisable)
GPUMOD_LINK_CHECK(cuptiEnableKernelReplayMode)
GPUMOD_LINK_CHECK(cuptiDisableKernelReplayMode)
GPUMOD_LINK_CHECK(cuptiKernelReplaySubscribeUpdate)

// Metric API (deprecated in CUDA 13.0, symbols still present in libcupti)
GPUMOD_LINK_CHECK(cuptiGetNumMetrics)
GPUMOD_LINK_CHECK(cuptiEnumMetrics)
GPUMOD_LINK_CHECK(cuptiDeviceGetNumMetrics)
GPUMOD_LINK_CHECK(cuptiDeviceEnumMetrics)
GPUMOD_LINK_CHECK(cuptiMetricGetAttribute)
GPUMOD_LINK_CHECK(cuptiMetricGetIdFromName)
GPUMOD_LINK_CHECK(cuptiMetricGetNumEvents)
GPUMOD_LINK_CHECK(cuptiMetricEnumEvents)
GPUMOD_LINK_CHECK(cuptiMetricGetNumProperties)
GPUMOD_LINK_CHECK(cuptiMetricEnumProperties)
GPUMOD_LINK_CHECK(cuptiMetricGetRequiredEventGroupSets)
GPUMOD_LINK_CHECK(cuptiMetricCreateEventGroupSets)
GPUMOD_LINK_CHECK(cuptiMetricGetValue)
GPUMOD_LINK_CHECK(cuptiMetricGetValue2)

} // namespace gpumod::cuda::test
