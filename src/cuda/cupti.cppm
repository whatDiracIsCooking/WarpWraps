/**
 * @file cupti.cppm
 * @brief Primary interface for gpumod.cuda.cupti
 *
 * This module wraps the native CUPTI (CUDA Profiling Tools Interface) API
 * and exports all types, enumerations, structures, callback function types,
 * and API functions declared by cupti.h and its transitively included headers:
 *   - cupti_result.h    : CUptiResult error codes
 *   - cupti_version.h   : cuptiGetVersion
 *   - cupti_callbacks.h : callback domain, subscriber, and enable/disable API
 *   - cupti_activity.h  : activity record kinds, structs, and activity API
 *   - cupti_events.h    : event API (deprecated in CUDA 13.0, headers still present)
 *   - cupti_metrics.h   : metric API (deprecated in CUDA 13.0, headers still present)
 *   - cupti_driver_cbid.h  : CUpti_driver_api_trace_cbid enum
 *   - cupti_runtime_cbid.h : CUpti_runtime_api_trace_cbid enum
 *   - cupti_nvtx_cbid.h    : CUpti_nvtx_api_trace_cbid enum
 *
 * Usage:
 *   import gpumod.cuda.cupti;
 */

module;

#include <cupti.h>

export module gpumod.cuda.cupti;

// ========================================================================
// Export all CUPTI types and functions in wwr namespace
// ========================================================================

export namespace wwr::cuda {

// ========================================================================
// Result / Status Type (cupti_result.h)
// ========================================================================

using ::CUptiResult;

// CUptiResult enumerators
using ::CUPTI_ERROR_API_NOT_IMPLEMENTED;
using ::CUPTI_ERROR_CDP_TRACING_NOT_SUPPORTED;
using ::CUPTI_ERROR_CMP_DEVICE_NOT_SUPPORTED;
using ::CUPTI_ERROR_CONFIDENTIAL_COMPUTING_NOT_SUPPORTED;
using ::CUPTI_ERROR_CUDA_COMPILER_NOT_COMPATIBLE;
using ::CUPTI_ERROR_DISABLED;
using ::CUPTI_ERROR_HARDWARE;
using ::CUPTI_ERROR_HARDWARE_BUSY;
using ::CUPTI_ERROR_INSUFFICIENT_PRIVILEGES;
using ::CUPTI_ERROR_INVALID_CHIP_NAME;
using ::CUPTI_ERROR_INVALID_CONTEXT;
using ::CUPTI_ERROR_INVALID_DEVICE;
using ::CUPTI_ERROR_INVALID_EVENT_DOMAIN_ID;
using ::CUPTI_ERROR_INVALID_EVENT_ID;
using ::CUPTI_ERROR_INVALID_EVENT_NAME;
using ::CUPTI_ERROR_INVALID_EVENT_VALUE;
using ::CUPTI_ERROR_INVALID_HANDLE;
using ::CUPTI_ERROR_INVALID_KIND;
using ::CUPTI_ERROR_INVALID_METRIC_ID;
using ::CUPTI_ERROR_INVALID_METRIC_NAME;
using ::CUPTI_ERROR_INVALID_METRIC_VALUE;
using ::CUPTI_ERROR_INVALID_MODULE;
using ::CUPTI_ERROR_INVALID_OPERATION;
using ::CUPTI_ERROR_INVALID_PARAMETER;
using ::CUPTI_ERROR_INVALID_STREAM;
using ::CUPTI_ERROR_LEGACY_PROFILER_NOT_SUPPORTED;
using ::CUPTI_ERROR_MAX_LIMIT_REACHED;
using ::CUPTI_ERROR_MIG_DEVICE_NOT_SUPPORTED;
using ::CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED;
using ::CUPTI_ERROR_NOT_COMPATIBLE;
using ::CUPTI_ERROR_NOT_INITIALIZED;
using ::CUPTI_ERROR_NOT_READY;
using ::CUPTI_ERROR_NOT_SUPPORTED;
using ::CUPTI_ERROR_OLD_PROFILER_API_INITIALIZED;
using ::CUPTI_ERROR_OPENACC_UNDEFINED_ROUTINE;
using ::CUPTI_ERROR_OUT_OF_MEMORY;
using ::CUPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT;
using ::CUPTI_ERROR_QUEUE_EMPTY;
using ::CUPTI_ERROR_SLI_DEVICE_NOT_SUPPORTED;
using ::CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED;
using ::CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_DEVICE;
using ::CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_NON_P2P_DEVICES;
using ::CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_WITH_MPS;
using ::CUPTI_ERROR_UNKNOWN;
using ::CUPTI_ERROR_VIRTUALIZED_DEVICE_INSUFFICIENT_PRIVILEGES;
using ::CUPTI_ERROR_VIRTUALIZED_DEVICE_NOT_SUPPORTED;
using ::CUPTI_ERROR_WSL_DEVICE_NOT_SUPPORTED;
using ::CUPTI_SUCCESS;

// ========================================================================
// Result API Functions (cupti_result.h)
// ========================================================================

using ::cuptiGetErrorMessage;
using ::cuptiGetResultString;

// ========================================================================
// Version API (cupti_version.h)
// ========================================================================

using ::cuptiGetVersion;

// ========================================================================
// Callback API Types (cupti_callbacks.h)
// ========================================================================

// Callback invocation site
using ::CUPTI_API_ENTER;
using ::CUPTI_API_EXIT;
using ::CUpti_ApiCallbackSite;

// Callback domain
using ::CUpti_CallbackDomain;
using ::CUPTI_CB_DOMAIN_DRIVER_API;
using ::CUPTI_CB_DOMAIN_INVALID;
using ::CUPTI_CB_DOMAIN_NVTX;
using ::CUPTI_CB_DOMAIN_RESOURCE;
using ::CUPTI_CB_DOMAIN_RUNTIME_API;
using ::CUPTI_CB_DOMAIN_SIZE;
using ::CUPTI_CB_DOMAIN_STATE;
using ::CUPTI_CB_DOMAIN_SYNCHRONIZE;

// Resource callback IDs
using ::CUpti_CallbackIdResource;
using ::CUPTI_CBID_RESOURCE_CONTEXT_CREATED;
using ::CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING;
using ::CUPTI_CBID_RESOURCE_CU_INIT_FINISHED;
using ::CUPTI_CBID_RESOURCE_GRAPH_CLONED;
using ::CUPTI_CBID_RESOURCE_GRAPH_CREATED;
using ::CUPTI_CBID_RESOURCE_GRAPH_DESTROY_STARTING;
using ::CUPTI_CBID_RESOURCE_GRAPH_NODE_SET_PARAMS;
using ::CUPTI_CBID_RESOURCE_GRAPH_NODE_UPDATED;
using ::CUPTI_CBID_RESOURCE_GRAPHEXEC_CREATE_STARTING;
using ::CUPTI_CBID_RESOURCE_GRAPHEXEC_CREATED;
using ::CUPTI_CBID_RESOURCE_GRAPHEXEC_DESTROY_STARTING;
using ::CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED;
using ::CUPTI_CBID_RESOURCE_GRAPHNODE_CREATE_STARTING;
using ::CUPTI_CBID_RESOURCE_GRAPHNODE_CREATED;
using ::CUPTI_CBID_RESOURCE_GRAPHNODE_DEPENDENCY_CREATED;
using ::CUPTI_CBID_RESOURCE_GRAPHNODE_DEPENDENCY_DESTROY_STARTING;
using ::CUPTI_CBID_RESOURCE_GRAPHNODE_DESTROY_STARTING;
using ::CUPTI_CBID_RESOURCE_INVALID;
using ::CUPTI_CBID_RESOURCE_MODULE_LOADED;
using ::CUPTI_CBID_RESOURCE_MODULE_PROFILED;
using ::CUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING;
using ::CUPTI_CBID_RESOURCE_SIZE;
using ::CUPTI_CBID_RESOURCE_STREAM_ATTRIBUTE_CHANGED;
using ::CUPTI_CBID_RESOURCE_STREAM_CREATED;
using ::CUPTI_CBID_RESOURCE_STREAM_DESTROY_STARTING;

// Synchronization callback IDs
using ::CUpti_CallbackIdSync;
using ::CUPTI_CBID_SYNCHRONIZE_CONTEXT_SYNCHRONIZED;
using ::CUPTI_CBID_SYNCHRONIZE_INVALID;
using ::CUPTI_CBID_SYNCHRONIZE_SIZE;
using ::CUPTI_CBID_SYNCHRONIZE_STREAM_SYNCHRONIZED;

// State callback IDs
using ::CUpti_CallbackIdState;
using ::CUPTI_CBID_STATE_ERROR;
using ::CUPTI_CBID_STATE_FATAL_ERROR;
using ::CUPTI_CBID_STATE_INVALID;
using ::CUPTI_CBID_STATE_SIZE;
using ::CUPTI_CBID_STATE_WARNING;

// Callback data structures
using ::CUpti_CallbackData;
using ::CUpti_GraphData;
using ::CUpti_ModuleResourceData;
using ::CUpti_NvtxData;
using ::CUpti_ResourceData;
using ::CUpti_StateData;
using ::CUpti_StreamAttrData;
using ::CUpti_SynchronizeData;

// Callback ID type (uint32_t alias)
using ::CUpti_CallbackId;

// Callback function pointer type
using ::CUpti_CallbackFunc;

// Subscriber handle and domain table
using ::CUpti_DomainTable;
using ::CUpti_SubscriberHandle;

// Subscribe_v2 parameters struct
using ::CUpti_SubscriberParams;

// ========================================================================
// Callback API Functions (cupti_callbacks.h)
// ========================================================================

using ::cuptiEnableAllDomains;
using ::cuptiEnableCallback;
using ::cuptiEnableDomain;
using ::cuptiGetCallbackName;
using ::cuptiGetCallbackState;
using ::cuptiSubscribe;
using ::cuptiSubscribe_v2;
using ::cuptiSupportedDomains;
using ::cuptiUnsubscribe;

// ========================================================================
// Driver API Callback ID Enum (cupti_driver_cbid.h)
// ========================================================================

using ::CUpti_driver_api_trace_cbid;

// Selected well-known driver cbid enumerators
using ::CUPTI_DRIVER_TRACE_CBID_cuCtxCreate;
using ::CUPTI_DRIVER_TRACE_CBID_cuCtxDestroy;
using ::CUPTI_DRIVER_TRACE_CBID_cuCtxSynchronize;
using ::CUPTI_DRIVER_TRACE_CBID_cuDeviceGet;
using ::CUPTI_DRIVER_TRACE_CBID_cuDeviceGetCount;
using ::CUPTI_DRIVER_TRACE_CBID_cuDeviceGetName;
using ::CUPTI_DRIVER_TRACE_CBID_cuDriverGetVersion;
using ::CUPTI_DRIVER_TRACE_CBID_cuInit;
using ::CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel;
using ::CUPTI_DRIVER_TRACE_CBID_cuMemAlloc;
using ::CUPTI_DRIVER_TRACE_CBID_cuMemFree;
using ::CUPTI_DRIVER_TRACE_CBID_INVALID;
using ::CUPTI_DRIVER_TRACE_CBID_SIZE;

// ========================================================================
// Runtime API Callback ID Enum (cupti_runtime_cbid.h)
// ========================================================================

using ::CUpti_runtime_api_trace_cbid;

// Selected well-known runtime cbid enumerators
using ::CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSynchronize_v3020;
using ::CUPTI_RUNTIME_TRACE_CBID_cudaDriverGetVersion_v3020;
using ::CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3020;
using ::CUPTI_RUNTIME_TRACE_CBID_cudaGetDeviceCount_v3020;
using ::CUPTI_RUNTIME_TRACE_CBID_cudaGetDeviceProperties_v3020;
using ::CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020;
using ::CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020;
using ::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020;
using ::CUPTI_RUNTIME_TRACE_CBID_INVALID;
using ::CUPTI_RUNTIME_TRACE_CBID_SIZE;

// ========================================================================
// NVTX API Callback ID Enum (cupti_nvtx_cbid.h)
// ========================================================================

using ::CUpti_nvtx_api_trace_cbid;

using ::CUPTI_CBID_NVTX_INVALID;
using ::CUPTI_CBID_NVTX_nvtxMarkA;
using ::CUPTI_CBID_NVTX_nvtxMarkW;
using ::CUPTI_CBID_NVTX_nvtxNameCudaDeviceA;
using ::CUPTI_CBID_NVTX_nvtxNameCudaDeviceW;
using ::CUPTI_CBID_NVTX_nvtxNameCudaEventA;
using ::CUPTI_CBID_NVTX_nvtxNameCudaEventW;
using ::CUPTI_CBID_NVTX_nvtxNameCudaStreamA;
using ::CUPTI_CBID_NVTX_nvtxNameCudaStreamW;
using ::CUPTI_CBID_NVTX_nvtxRangeEnd;
using ::CUPTI_CBID_NVTX_nvtxRangePop;
using ::CUPTI_CBID_NVTX_nvtxRangePushA;
using ::CUPTI_CBID_NVTX_nvtxRangePushW;
using ::CUPTI_CBID_NVTX_nvtxRangeStartA;
using ::CUPTI_CBID_NVTX_nvtxRangeStartW;
using ::CUPTI_CBID_NVTX_SIZE;

// ========================================================================
// Activity API Enumerations (cupti_activity.h)
// ========================================================================

// Activity record kinds
using ::CUPTI_ACTIVITY_KIND_BRANCH;
using ::CUPTI_ACTIVITY_KIND_CDP_KERNEL;
using ::CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL;
using ::CUPTI_ACTIVITY_KIND_CONFIDENTIAL_COMPUTE_ROTATION;
using ::CUPTI_ACTIVITY_KIND_CONTEXT;
using ::CUPTI_ACTIVITY_KIND_COUNT;
using ::CUPTI_ACTIVITY_KIND_CUDA_EVENT;
using ::CUPTI_ACTIVITY_KIND_DEVICE;
using ::CUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE;
using ::CUPTI_ACTIVITY_KIND_DEVICE_GRAPH_TRACE;
using ::CUPTI_ACTIVITY_KIND_DRIVER;
using ::CUPTI_ACTIVITY_KIND_ENVIRONMENT;
using ::CUPTI_ACTIVITY_KIND_EVENT;
using ::CUPTI_ACTIVITY_KIND_EVENT_INSTANCE;
using ::CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION;
using ::CUPTI_ACTIVITY_KIND_FUNCTION;
using ::CUPTI_ACTIVITY_KIND_GLOBAL_ACCESS;
using ::CUPTI_ACTIVITY_KIND_GRAPH_TRACE;
using ::CUPTI_ACTIVITY_KIND_INSTANTANEOUS_EVENT;
using ::CUPTI_ACTIVITY_KIND_INSTANTANEOUS_EVENT_INSTANCE;
using ::CUPTI_ACTIVITY_KIND_INSTANTANEOUS_METRIC;
using ::CUPTI_ACTIVITY_KIND_INSTANTANEOUS_METRIC_INSTANCE;
using ::CUPTI_ACTIVITY_KIND_INSTRUCTION_CORRELATION;
using ::CUPTI_ACTIVITY_KIND_INSTRUCTION_EXECUTION;
using ::CUPTI_ACTIVITY_KIND_INTERNAL_LAUNCH_API;
using ::CUPTI_ACTIVITY_KIND_INVALID;
using ::CUPTI_ACTIVITY_KIND_JIT;
using ::CUPTI_ACTIVITY_KIND_KERNEL;
using ::CUPTI_ACTIVITY_KIND_MARKER;
using ::CUPTI_ACTIVITY_KIND_MARKER_DATA;
using ::CUPTI_ACTIVITY_KIND_MEM_DECOMPRESS;
using ::CUPTI_ACTIVITY_KIND_MEMCPY;
using ::CUPTI_ACTIVITY_KIND_MEMCPY2;
using ::CUPTI_ACTIVITY_KIND_MEMORY;
using ::CUPTI_ACTIVITY_KIND_MEMORY2;
using ::CUPTI_ACTIVITY_KIND_MEMORY_POOL;
using ::CUPTI_ACTIVITY_KIND_MEMSET;
using ::CUPTI_ACTIVITY_KIND_METRIC;
using ::CUPTI_ACTIVITY_KIND_METRIC_INSTANCE;
using ::CUPTI_ACTIVITY_KIND_MODULE;
using ::CUPTI_ACTIVITY_KIND_NAME;
using ::CUPTI_ACTIVITY_KIND_NVLINK;
using ::CUPTI_ACTIVITY_KIND_OPENACC_DATA;
using ::CUPTI_ACTIVITY_KIND_OPENACC_LAUNCH;
using ::CUPTI_ACTIVITY_KIND_OPENACC_OTHER;
using ::CUPTI_ACTIVITY_KIND_OPENMP;
using ::CUPTI_ACTIVITY_KIND_OVERHEAD;
using ::CUPTI_ACTIVITY_KIND_PC_SAMPLING;
using ::CUPTI_ACTIVITY_KIND_PC_SAMPLING_RECORD_INFO;
using ::CUPTI_ACTIVITY_KIND_PCIE;
using ::CUPTI_ACTIVITY_KIND_PREEMPTION;
using ::CUPTI_ACTIVITY_KIND_RUNTIME;
using ::CUPTI_ACTIVITY_KIND_SHARED_ACCESS;
using ::CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR;
using ::CUPTI_ACTIVITY_KIND_STREAM;
using ::CUPTI_ACTIVITY_KIND_SYNCHRONIZATION;
using ::CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER;
using ::CUpti_ActivityKind;

// Activity object kinds
using ::CUPTI_ACTIVITY_OBJECT_CONTEXT;
using ::CUPTI_ACTIVITY_OBJECT_DEVICE;
using ::CUPTI_ACTIVITY_OBJECT_PROCESS;
using ::CUPTI_ACTIVITY_OBJECT_STREAM;
using ::CUPTI_ACTIVITY_OBJECT_THREAD;
using ::CUPTI_ACTIVITY_OBJECT_UNKNOWN;
using ::CUpti_ActivityObjectKind;

// Activity object kind ID union
using ::CUpti_ActivityObjectKindId;

// Overhead kinds
using ::CUPTI_ACTIVITY_OVERHEAD_ACTIVITY_BUFFER_REQUEST;
using ::CUPTI_ACTIVITY_OVERHEAD_COMMAND_BUFFER_FULL;
using ::CUPTI_ACTIVITY_OVERHEAD_CUPTI_BUFFER_FLUSH;
using ::CUPTI_ACTIVITY_OVERHEAD_CUPTI_INSTRUMENTATION;
using ::CUPTI_ACTIVITY_OVERHEAD_CUPTI_RESOURCE;
using ::CUPTI_ACTIVITY_OVERHEAD_DRIVER_COMPILER;
using ::CUPTI_ACTIVITY_OVERHEAD_LAZY_FUNCTION_LOADING;
using ::CUPTI_ACTIVITY_OVERHEAD_RUNTIME_TRIGGERED_MODULE_LOADING;
using ::CUPTI_ACTIVITY_OVERHEAD_UNKNOWN;
using ::CUPTI_ACTIVITY_OVERHEAD_UVM_ACTIVITY_INIT;
using ::CUpti_ActivityOverheadKind;

// Overhead command-buffer-full data struct
using ::CUpti_ActivityOverheadCommandBufferFullData;

// Compute API kinds
using ::CUPTI_ACTIVITY_COMPUTE_API_CUDA;
using ::CUPTI_ACTIVITY_COMPUTE_API_CUDA_MPS;
using ::CUPTI_ACTIVITY_COMPUTE_API_UNKNOWN;
using ::CUpti_ActivityComputeApiKind;

// Activity flags
using ::CUpti_ActivityFlag;

// PC sampling stall reasons
using ::CUpti_ActivityPCSamplingStallReason;

// PC sampling period
using ::CUpti_ActivityPCSamplingPeriod;

// Memcpy kinds
using ::CUpti_ActivityMemcpyKind;

// Memory kinds
using ::CUpti_ActivityMemoryKind;

// Preemption kinds
using ::CUpti_ActivityPreemptionKind;

// Environment kinds
using ::CUpti_ActivityEnvironmentKind;

// Clock throttle reasons
using ::CUpti_EnvironmentClocksThrottleReason;

// Unified memory counter scope
using ::CUpti_ActivityUnifiedMemoryCounterScope;

// Unified memory counter kinds
using ::CUpti_ActivityUnifiedMemoryCounterKind;

// Unified memory access types
using ::CUpti_ActivityUnifiedMemoryAccessType;

// Unified memory migration causes
using ::CUpti_ActivityUnifiedMemoryMigrationCause;

// Unified memory remote map causes
using ::CUpti_ActivityUnifiedMemoryRemoteMapCause;

// Instruction classes
using ::CUpti_ActivityInstructionClass;

// Partitioned global cache config
using ::CUpti_ActivityPartitionedGlobalCacheConfig;

// Synchronization types
using ::CUpti_ActivitySynchronizationType;

// Stream flags
using ::CUpti_ActivityStreamFlag;

// NVLink flags
using ::CUpti_LinkFlag;

// Memory operation types
using ::CUpti_ActivityMemoryOperationType;

// Memory pool types
using ::CUpti_ActivityMemoryPoolType;

// Memory pool operation types
using ::CUpti_ActivityMemoryPoolOperationType;

// Channel types
using ::CUpti_ChannelType;

// Context CIG mode
using ::CUpti_ContextCigMode;

// NVTX extended payload types
using ::CUpti_NvtxExtPayloadType;

// NVTX extended payload attributes
using ::CUpti_NvtxExtPayloadAttr;

// Launch type
using ::CUpti_ActivityLaunchType;

// Shared memory limit config
using ::CUpti_FuncShmemLimitConfig;

// JIT entry type
using ::CUpti_ActivityJitEntryType;

// JIT operation type
using ::CUpti_ActivityJitOperationType;

// Device graph launch mode
using ::CUpti_DeviceGraphLaunchMode;

// External correlation kinds
using ::CUpti_ExternalCorrelationKind;

// PCIe device type
using ::CUpti_PcieDeviceType;

// PCIe generation
using ::CUpti_PcieGen;

// Confidential compute rotation event type
using ::CUpti_ConfidentialComputeRotationEventType;

// OpenACC event kinds
using ::CUpti_OpenAccEventKind;

// OpenACC construct kinds
using ::CUpti_OpenAccConstructKind;

// OpenMP event kinds
using ::CUpti_OpenMpEventKind;

// Device type
using ::CUpti_DevType;

// Device virtualization mode
using ::CUpti_DeviceVirtualizationMode;

// Activity thread ID type
using ::CUpti_ActivityThreadIdType;

// Activity attributes
using ::CUpti_ActivityAttribute;

// ========================================================================
// Activity API Configuration Structures (cupti_activity.h)
// ========================================================================

using ::CUpti_ActivityAutoBoostState;
using ::CUpti_ActivityPCSamplingConfig;
using ::CUpti_ActivityUnifiedMemoryCounterConfig;

// ========================================================================
// Activity Record Structures (cupti_activity.h)
// ========================================================================

// Base activity record (discriminated via the kind field)
using ::CUpti_Activity;

// Memcpy records
using ::CUpti_ActivityMemcpy6;
using ::CUpti_ActivityMemcpyPtoP4;

// Memset record
using ::CUpti_ActivityMemset4;

// Memory allocation records
using ::CUpti_ActivityMemory;
using ::CUpti_ActivityMemory4;

// Memory pool record
using ::CUpti_ActivityMemoryPool3;

// Kernel records
using ::CUpti_ActivityCdpKernel;
using ::CUpti_ActivityKernel10;

// Preemption record
using ::CUpti_ActivityPreemption;

// Driver/runtime API record
using ::CUpti_ActivityAPI;

// Event and metric records (deprecated in CUDA 13.0, types still exported)
using ::CUpti_ActivityEvent;
using ::CUpti_ActivityEventInstance;
using ::CUpti_ActivityMetric;
using ::CUpti_ActivityMetricInstance;

// Source-level records
using ::CUpti_ActivityBranch2;
using ::CUpti_ActivityGlobalAccess3;
using ::CUpti_ActivityInstructionCorrelation;
using ::CUpti_ActivityInstructionExecution;
using ::CUpti_ActivitySharedAccess;
using ::CUpti_ActivitySourceLocator;

// Device and context info records
using ::CUpti_ActivityContext3;
using ::CUpti_ActivityDevice5;
using ::CUpti_ActivityDeviceAttribute;

// NVTX annotation records
using ::CUpti_ActivityMarker2;
using ::CUpti_ActivityMarkerData2;
using ::CUpti_ActivityName;

// Overhead record
using ::CUpti_ActivityOverhead3;

// Environment record
using ::CUpti_ActivityEnvironment;

// PC sampling records
using ::CUpti_ActivityPCSampling3;
using ::CUpti_ActivityPCSamplingRecordInfo;

// Unified memory counter record
using ::CUpti_ActivityUnifiedMemoryCounter3;

// Function and module records
using ::CUpti_ActivityFunction;
using ::CUpti_ActivityModule;

// CUDA event and stream records
using ::CUpti_ActivityCudaEvent2;
using ::CUpti_ActivityStream;
using ::CUpti_ActivitySynchronization2;

// OpenACC and OpenMP records
using ::CUpti_ActivityOpenAcc;
using ::CUpti_ActivityOpenAccData;
using ::CUpti_ActivityOpenAccLaunch;
using ::CUpti_ActivityOpenAccOther;
using ::CUpti_ActivityOpenMp;

// External correlation record
using ::CUpti_ActivityExternalCorrelation;

// NVLink, PCIe, and confidential compute records
using ::CUpti_ActivityConfidentialComputeRotation;
using ::CUpti_ActivityNvLink4;
using ::CUpti_ActivityPcie;

// Instantaneous event/metric records
using ::CUpti_ActivityInstantaneousEvent;
using ::CUpti_ActivityInstantaneousEventInstance;
using ::CUpti_ActivityInstantaneousMetric;
using ::CUpti_ActivityInstantaneousMetricInstance;

// JIT record
using ::CUpti_ActivityJit2;

// Graph trace records
using ::CUpti_ActivityDeviceGraphTrace;
using ::CUpti_ActivityGraphTrace2;

// Memory decompression record
using ::CUpti_ActivityMemDecompress;

// ========================================================================
// Activity API Callback Function Types (cupti_activity.h)
// ========================================================================

using ::CUpti_BuffersCallbackCompleteFunc;
using ::CUpti_BuffersCallbackRequestFunc;
using ::CUpti_TimestampCallbackFunc;

// ========================================================================
// Activity API Functions (cupti_activity.h)
// ========================================================================

// Timestamp and ID query functions
using ::cuptiGetContextId;
using ::cuptiGetDeviceId;
using ::cuptiGetGraphExecId;
using ::cuptiGetGraphId;
using ::cuptiGetGraphNodeId;
using ::cuptiGetStreamId;
using ::cuptiGetStreamIdEx;
using ::cuptiGetTimestamp;

// Activity enable/disable
using ::cuptiActivityDisable;
using ::cuptiActivityDisableContext;
using ::cuptiActivityEnable;
using ::cuptiActivityEnableAndDump;
using ::cuptiActivityEnableContext;

// Buffer management
using ::cuptiActivityFlush;
using ::cuptiActivityFlushAll;
using ::cuptiActivityGetNextRecord;
using ::cuptiActivityGetNumDroppedRecords;
using ::cuptiActivityRegisterCallbacks;

// Attribute get/set
using ::cuptiActivityGetAttribute;
using ::cuptiActivitySetAttribute;

// Unified memory counter configuration
using ::cuptiActivityConfigureUnifiedMemoryCounter;

// Auto-boost state
using ::cuptiGetAutoBoostState;

// PC sampling configuration
using ::cuptiActivityConfigurePCSampling;

// Miscellaneous
using ::cuptiComputeCapabilitySupported;
using ::cuptiDeviceSupported;
using ::cuptiDeviceVirtualizationMode;
using ::cuptiFinalize;
using ::cuptiGetLastError;
using ::cuptiGetThreadIdType;
using ::cuptiSetThreadIdType;

// External correlation
using ::cuptiActivityPopExternalCorrelationId;
using ::cuptiActivityPushExternalCorrelationId;

// Latency timestamps
using ::cuptiActivityEnableLatencyTimestamps;

// Flush period
using ::cuptiActivityFlushPeriod;

// Launch attributes
using ::cuptiActivityEnableLaunchAttributes;

// Timestamp callback
using ::cuptiActivityRegisterTimestampCallback;

// Device graph tracing
using ::cuptiActivityEnableDeviceGraph;

// Per-API activity enable
using ::cuptiActivityEnableDriverApi;
using ::cuptiActivityEnableRuntimeApi;

// Hardware trace
using ::cuptiActivityEnableHWTrace;

// Allocation source
using ::cuptiActivityEnableAllocationSource;

// All sync records
using ::cuptiActivityEnableAllSyncRecords;

// CUDA event device timestamps
using ::cuptiActivityEnableCudaEventDeviceTimestamps;

// ========================================================================
// Event API Types (cupti_events.h)
// Note: deprecated in CUDA 12.8, unsupported in CUDA 13.0 — types exported
// so existing code compiles; functions exported for link-check completeness.
// ========================================================================

using ::CUpti_EventDomainID;
using ::CUpti_EventGroup;
using ::CUpti_EventID;

using ::CUPTI_DEVICE_ATTR_DEVICE_CLASS_GEFORCE;
using ::CUPTI_DEVICE_ATTR_DEVICE_CLASS_QUADRO;
using ::CUPTI_DEVICE_ATTR_DEVICE_CLASS_TEGRA;
using ::CUPTI_DEVICE_ATTR_DEVICE_CLASS_TESLA;
using ::CUpti_DeviceAttributeDeviceClass;

using ::CUpti_DeviceAttribute;

using ::CUpti_EventDomainAttribute;

using ::CUpti_EventCollectionMethod;

using ::CUpti_EventGroupAttribute;

using ::CUpti_EventProfilingScope;

using ::CUpti_EventAttribute;

using ::CUpti_EventCollectionMode;

using ::CUpti_EventCategory;

using ::CUpti_ReadEventFlags;

using ::CUpti_EventGroupSet;
using ::CUpti_EventGroupSets;

using ::CUpti_KernelReplayUpdateFunc;

// Event API functions
using ::cuptiDeviceEnumEventDomains;
using ::cuptiDeviceGetAttribute;
using ::cuptiDeviceGetEventDomainAttribute;
using ::cuptiDeviceGetNumEventDomains;
using ::cuptiDisableKernelReplayMode;
using ::cuptiEnableKernelReplayMode;
using ::cuptiEnumEventDomains;
using ::cuptiEventDomainEnumEvents;
using ::cuptiEventDomainGetAttribute;
using ::cuptiEventDomainGetNumEvents;
using ::cuptiEventGetAttribute;
using ::cuptiEventGetIdFromName;
using ::cuptiEventGroupAddEvent;
using ::cuptiEventGroupCreate;
using ::cuptiEventGroupDestroy;
using ::cuptiEventGroupDisable;
using ::cuptiEventGroupEnable;
using ::cuptiEventGroupGetAttribute;
using ::cuptiEventGroupReadAllEvents;
using ::cuptiEventGroupReadEvent;
using ::cuptiEventGroupRemoveAllEvents;
using ::cuptiEventGroupRemoveEvent;
using ::cuptiEventGroupResetAllEvents;
using ::cuptiEventGroupSetAttribute;
using ::cuptiEventGroupSetDisable;
using ::cuptiEventGroupSetEnable;
using ::cuptiEventGroupSetsCreate;
using ::cuptiEventGroupSetsDestroy;
using ::cuptiGetNumEventDomains;
using ::cuptiKernelReplaySubscribeUpdate;
using ::cuptiSetEventCollectionMode;

// ========================================================================
// Metric API Types (cupti_metrics.h)
// Note: deprecated in CUDA 12.8, unsupported in CUDA 13.0 — types exported
// so existing code compiles.
// ========================================================================

using ::CUpti_MetricAttribute;
using ::CUpti_MetricCategory;
using ::CUpti_MetricEvaluationMode;
using ::CUpti_MetricID;
using ::CUpti_MetricPropertyDeviceClass;
using ::CUpti_MetricPropertyID;
using ::CUpti_MetricValue;
using ::CUpti_MetricValueKind;
using ::CUpti_MetricValueUtilizationLevel;

// Metric API functions
using ::cuptiDeviceEnumMetrics;
using ::cuptiDeviceGetNumMetrics;
using ::cuptiEnumMetrics;
using ::cuptiGetNumMetrics;
using ::cuptiMetricCreateEventGroupSets;
using ::cuptiMetricEnumEvents;
using ::cuptiMetricEnumProperties;
using ::cuptiMetricGetAttribute;
using ::cuptiMetricGetIdFromName;
using ::cuptiMetricGetNumEvents;
using ::cuptiMetricGetNumProperties;
using ::cuptiMetricGetRequiredEventGroupSets;
using ::cuptiMetricGetValue;
using ::cuptiMetricGetValue2;

} // namespace wwr::cuda
