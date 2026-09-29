/**
 * @file cuda_runtime_api.cppm
 * @brief CUDA runtime API module wrapper for wwr project
 *
 * This module wraps the CUDA runtime API for use in C++20/C++23 module-based code.
 * It exports types, constants, and functions needed for CUDA memory management,
 * streams, events, and error handling.
 *
 * Usage:
 *   import wwr.cuda.cuda_runtime_api;
 *
 * Note: This uses cuda_runtime_api.h (not cuda_runtime.h) because:
 *   - cuda_runtime_api.h contains only extern function declarations
 *   - cuda_runtime.h adds static inline wrappers which cannot be exported from modules
 */

module;

#include <cuda_runtime_api.h>

// Validate CUDA macro values at compile time before we #undef them
static_assert(cudaHostAllocDefault == 0x00, "cudaHostAllocDefault value mismatch");
static_assert(cudaHostAllocPortable == 0x01, "cudaHostAllocPortable value mismatch");
static_assert(cudaHostAllocMapped == 0x02, "cudaHostAllocMapped value mismatch");
static_assert(cudaHostAllocWriteCombined == 0x04, "cudaHostAllocWriteCombined value mismatch");

static_assert(cudaEventDefault == 0x00, "cudaEventDefault value mismatch");
static_assert(cudaEventBlockingSync == 0x01, "cudaEventBlockingSync value mismatch");
static_assert(cudaEventDisableTiming == 0x02, "cudaEventDisableTiming value mismatch");
static_assert(cudaEventInterprocess == 0x04, "cudaEventInterprocess value mismatch");

static_assert(cudaStreamDefault == 0x00, "cudaStreamDefault value mismatch");
static_assert(cudaStreamNonBlocking == 0x01, "cudaStreamNonBlocking value mismatch");

static_assert(cudaMemAttachGlobal == 0x01, "cudaMemAttachGlobal value mismatch");
static_assert(cudaMemAttachHost == 0x02, "cudaMemAttachHost value mismatch");

static_assert(cudaArrayDefault == 0x00, "cudaArrayDefault value mismatch");
static_assert(cudaArraySurfaceLoadStore == 0x02, "cudaArraySurfaceLoadStore value mismatch");

// Undefine macros so we can create constexpr variables with the same names
#undef cudaHostAllocDefault
#undef cudaHostAllocPortable
#undef cudaHostAllocMapped
#undef cudaHostAllocWriteCombined
#undef cudaEventDefault
#undef cudaEventBlockingSync
#undef cudaEventDisableTiming
#undef cudaEventInterprocess
#undef cudaStreamDefault
#undef cudaStreamNonBlocking
#undef cudaMemAttachGlobal
#undef cudaMemAttachHost
#undef cudaArrayDefault
#undef cudaArraySurfaceLoadStore

export module wwr.cuda.cuda_runtime_api;

// ========================================================================
// Export all CUDA types, functions, and constants in wwr namespace
// ========================================================================

export namespace wwr::cuda {

// ========================================================================
// Constexpr wrappers for CUDA flag macros
// ========================================================================
// Values validated by static_assert above
// Note: 'inline' is implicit for exported module variables

// cudaHostAlloc flags
constexpr unsigned int cudaHostAllocDefault = 0x00;
constexpr unsigned int cudaHostAllocPortable = 0x01;
constexpr unsigned int cudaHostAllocMapped = 0x02;
constexpr unsigned int cudaHostAllocWriteCombined = 0x04;

// cudaEvent flags
constexpr unsigned int cudaEventDefault = 0x00;
constexpr unsigned int cudaEventBlockingSync = 0x01;
constexpr unsigned int cudaEventDisableTiming = 0x02;
constexpr unsigned int cudaEventInterprocess = 0x04;

// cudaStream flags
constexpr unsigned int cudaStreamDefault = 0x00;
constexpr unsigned int cudaStreamNonBlocking = 0x01;

// cudaMallocManaged flags
constexpr unsigned int cudaMemAttachGlobal = 0x01;
constexpr unsigned int cudaMemAttachHost = 0x02;

// cudaMallocArray flags -- cudaArraySurfaceLoadStore must be set to bind a
// surface object to the array (cudaCreateSurfaceObject requires it).
constexpr unsigned int cudaArrayDefault = 0x00;
constexpr unsigned int cudaArraySurfaceLoadStore = 0x02;

// ========================================================================
// Core Types
// ========================================================================
using ::cudaError_t;
// Most commonly used cudaError_t values
using ::cudaErrorDeviceAlreadyInUse;
using ::cudaErrorInitializationError;
using ::cudaErrorInvalidConfiguration;
using ::cudaErrorInvalidDevicePointer;
using ::cudaErrorInvalidMemcpyDirection;
using ::cudaErrorInvalidValue;
using ::cudaErrorMemoryAllocation;
using ::cudaErrorNotReady;
using ::cudaSuccess;

using ::cudaDeviceAttr;
using ::cudaDeviceProp;
using ::cudaEvent_t;
using ::cudaStream_t;
// cudaDeviceAttr enumerators
using ::cudaDevAttrAsyncEngineCount;
using ::cudaDevAttrCanFlushRemoteWrites;
using ::cudaDevAttrCanMapHostMemory;
using ::cudaDevAttrCanUseHostPointerForRegisteredMem;
using ::cudaDevAttrClockRate;
using ::cudaDevAttrClusterLaunch;
using ::cudaDevAttrComputeCapabilityMajor;
using ::cudaDevAttrComputeCapabilityMinor;
using ::cudaDevAttrComputeMode;
using ::cudaDevAttrComputePreemptionSupported;
using ::cudaDevAttrConcurrentKernels;
using ::cudaDevAttrConcurrentManagedAccess;
using ::cudaDevAttrCooperativeLaunch;
using ::cudaDevAttrD3D12CigSupported;
using ::cudaDevAttrDeferredMappingCudaArraySupported;
using ::cudaDevAttrDirectManagedMemAccessFromHost;
using ::cudaDevAttrEccEnabled;
using ::cudaDevAttrGlobalL1CacheSupported;
using ::cudaDevAttrGlobalMemoryBusWidth;
using ::cudaDevAttrGPUDirectRDMAFlushWritesOptions;
using ::cudaDevAttrGPUDirectRDMASupported;
using ::cudaDevAttrGPUDirectRDMAWritesOrdering;
using ::cudaDevAttrGpuOverlap;
using ::cudaDevAttrGpuPciDeviceId;
using ::cudaDevAttrGpuPciSubsystemId;
using ::cudaDevAttrHostMemoryPoolsSupported;
using ::cudaDevAttrHostNativeAtomicSupported;
using ::cudaDevAttrHostNumaId;
using ::cudaDevAttrHostNumaMemoryPoolsSupported;
using ::cudaDevAttrHostNumaMultinodeIpcSupported;
using ::cudaDevAttrHostRegisterReadOnlySupported;
using ::cudaDevAttrHostRegisterSupported;
using ::cudaDevAttrIntegrated;
using ::cudaDevAttrIpcEventSupport;
using ::cudaDevAttrIsMultiGpuBoard;
using ::cudaDevAttrKernelExecTimeout;
using ::cudaDevAttrL2CacheSize;
using ::cudaDevAttrLocalL1CacheSupported;
using ::cudaDevAttrManagedMemory;
using ::cudaDevAttrMaxAccessPolicyWindowSize;
using ::cudaDevAttrMaxBlockDimX;
using ::cudaDevAttrMaxBlockDimY;
using ::cudaDevAttrMaxBlockDimZ;
using ::cudaDevAttrMaxBlocksPerMultiprocessor;
using ::cudaDevAttrMaxGridDimX;
using ::cudaDevAttrMaxGridDimY;
using ::cudaDevAttrMaxGridDimZ;
using ::cudaDevAttrMaxPersistingL2CacheSize;
using ::cudaDevAttrMaxPitch;
using ::cudaDevAttrMaxRegistersPerBlock;
using ::cudaDevAttrMaxRegistersPerMultiprocessor;
using ::cudaDevAttrMaxSharedMemoryPerBlock;
using ::cudaDevAttrMaxSharedMemoryPerBlockOptin;
using ::cudaDevAttrMaxSharedMemoryPerMultiprocessor;
using ::cudaDevAttrMaxSurface1DLayeredLayers;
using ::cudaDevAttrMaxSurface1DLayeredWidth;
using ::cudaDevAttrMaxSurface1DWidth;
using ::cudaDevAttrMaxSurface2DHeight;
using ::cudaDevAttrMaxSurface2DLayeredHeight;
using ::cudaDevAttrMaxSurface2DLayeredLayers;
using ::cudaDevAttrMaxSurface2DLayeredWidth;
using ::cudaDevAttrMaxSurface2DWidth;
using ::cudaDevAttrMaxSurface3DDepth;
using ::cudaDevAttrMaxSurface3DHeight;
using ::cudaDevAttrMaxSurface3DWidth;
using ::cudaDevAttrMaxSurfaceCubemapLayeredLayers;
using ::cudaDevAttrMaxSurfaceCubemapLayeredWidth;
using ::cudaDevAttrMaxSurfaceCubemapWidth;
using ::cudaDevAttrMaxTexture1DLayeredLayers;
using ::cudaDevAttrMaxTexture1DLayeredWidth;
using ::cudaDevAttrMaxTexture1DLinearWidth;
using ::cudaDevAttrMaxTexture1DMipmappedWidth;
using ::cudaDevAttrMaxTexture1DWidth;
using ::cudaDevAttrMaxTexture2DGatherHeight;
using ::cudaDevAttrMaxTexture2DGatherWidth;
using ::cudaDevAttrMaxTexture2DHeight;
using ::cudaDevAttrMaxTexture2DLayeredHeight;
using ::cudaDevAttrMaxTexture2DLayeredLayers;
using ::cudaDevAttrMaxTexture2DLayeredWidth;
using ::cudaDevAttrMaxTexture2DLinearHeight;
using ::cudaDevAttrMaxTexture2DLinearPitch;
using ::cudaDevAttrMaxTexture2DLinearWidth;
using ::cudaDevAttrMaxTexture2DMipmappedHeight;
using ::cudaDevAttrMaxTexture2DMipmappedWidth;
using ::cudaDevAttrMaxTexture2DWidth;
using ::cudaDevAttrMaxTexture3DDepth;
using ::cudaDevAttrMaxTexture3DDepthAlt;
using ::cudaDevAttrMaxTexture3DHeight;
using ::cudaDevAttrMaxTexture3DHeightAlt;
using ::cudaDevAttrMaxTexture3DWidth;
using ::cudaDevAttrMaxTexture3DWidthAlt;
using ::cudaDevAttrMaxTextureCubemapLayeredLayers;
using ::cudaDevAttrMaxTextureCubemapLayeredWidth;
using ::cudaDevAttrMaxTextureCubemapWidth;
using ::cudaDevAttrMaxThreadsPerBlock;
using ::cudaDevAttrMaxThreadsPerMultiProcessor;
using ::cudaDevAttrMemoryClockRate;
using ::cudaDevAttrMemoryPoolsSupported;
using ::cudaDevAttrMemoryPoolSupportedHandleTypes;
using ::cudaDevAttrMemSyncDomainCount;
using ::cudaDevAttrMpsEnabled;
using ::cudaDevAttrMultiGpuBoardGroupID;
using ::cudaDevAttrMultiProcessorCount;
using ::cudaDevAttrNumaConfig;
using ::cudaDevAttrNumaId;
using ::cudaDevAttrOnlyPartialHostNativeAtomicSupported;
using ::cudaDevAttrPageableMemoryAccess;
using ::cudaDevAttrPageableMemoryAccessUsesHostPageTables;
using ::cudaDevAttrPciBusId;
using ::cudaDevAttrPciDeviceId;
using ::cudaDevAttrPciDomainId;
using ::cudaDevAttrReserved122;
using ::cudaDevAttrReserved123;
using ::cudaDevAttrReserved124;
using ::cudaDevAttrReserved127;
using ::cudaDevAttrReserved128;
using ::cudaDevAttrReserved129;
using ::cudaDevAttrReserved132;
using ::cudaDevAttrReserved141;
using ::cudaDevAttrReserved145;
using ::cudaDevAttrReserved92;
using ::cudaDevAttrReserved93;
using ::cudaDevAttrReserved94;
using ::cudaDevAttrReserved96;
using ::cudaDevAttrReservedSharedMemoryPerBlock;
using ::cudaDevAttrSingleToDoublePrecisionPerfRatio;
using ::cudaDevAttrSparseCudaArraySupported;
using ::cudaDevAttrStreamPrioritiesSupported;
using ::cudaDevAttrSurfaceAlignment;
using ::cudaDevAttrTccDriver;
using ::cudaDevAttrTextureAlignment;
using ::cudaDevAttrTexturePitchAlignment;
using ::cudaDevAttrTimelineSemaphoreInteropSupported;
using ::cudaDevAttrTotalConstantMemory;
using ::cudaDevAttrUnifiedAddressing;
using ::cudaDevAttrVulkanCigSupported;
using ::cudaDevAttrWarpSize;

using ::cudaFuncAttributes;
using ::cudaPointerAttributes;

// Array types
using ::cudaArray_const_t;
using ::cudaArray_t;
using ::cudaArrayMemoryRequirements;
using ::cudaArraySparseProperties;
using ::cudaMipmappedArray_const_t;
using ::cudaMipmappedArray_t;

// Memory types
using ::cudaAccessPolicyWindow;
using ::cudaExtent;
using ::cudaMemcpy3DParms;
using ::cudaMemcpy3DPeerParms;
using ::cudaMemcpyNodeParams;
using ::cudaMemsetParams;
using ::cudaPitchedPtr;
using ::cudaPos;

// Host node types
using ::cudaHostFn_t;
using ::cudaHostNodeParams;

// Channel format
using ::cudaChannelFormatDesc;

// Resource types
using ::cudaResourceDesc;
using ::cudaResourceViewDesc;

// IPC types
using ::cudaIpcEventHandle_t;
using ::cudaIpcMemHandle_t;

// External resource types
using ::cudaExternalMemoryBufferDesc;
using ::cudaExternalMemoryHandleDesc;
using ::cudaExternalMemoryMipmappedArrayDesc;
using ::cudaExternalSemaphoreHandleDesc;
using ::cudaExternalSemaphoreSignalParams;
using ::cudaExternalSemaphoreWaitParams;

// Graph types
using ::cudaGraph_t;
using ::cudaGraphExec_t;
using ::cudaGraphNode_t;
using ::cudaGraphNodeType;
// cudaGraphNodeType enumerators
using ::cudaGraphNodeTypeConditional;
using ::cudaGraphNodeTypeEmpty;
using ::cudaGraphNodeTypeEventRecord;
using ::cudaGraphNodeTypeExtSemaphoreSignal;
using ::cudaGraphNodeTypeExtSemaphoreWait;
using ::cudaGraphNodeTypeGraph;
using ::cudaGraphNodeTypeHost;
using ::cudaGraphNodeTypeKernel;
using ::cudaGraphNodeTypeMemAlloc;
using ::cudaGraphNodeTypeMemcpy;
using ::cudaGraphNodeTypeMemFree;
using ::cudaGraphNodeTypeMemset;
using ::cudaGraphNodeTypeWaitEvent;

using ::cudaGraphExecUpdateResult;
// cudaGraphExecUpdateResult enumerators
using ::cudaGraphExecUpdateError;
using ::cudaGraphExecUpdateErrorAttributesChanged;
using ::cudaGraphExecUpdateErrorFunctionChanged;
using ::cudaGraphExecUpdateErrorNodeTypeChanged;
using ::cudaGraphExecUpdateErrorNotSupported;
using ::cudaGraphExecUpdateErrorParametersChanged;
using ::cudaGraphExecUpdateErrorTopologyChanged;
using ::cudaGraphExecUpdateErrorUnsupportedFunctionChange;
using ::cudaGraphExecUpdateSuccess;

using ::cudaKernelNodeParams;
using ::cudaLaunchAttributeID;
// cudaLaunchAttributeID enumerators
using ::cudaLaunchAttributeAccessPolicyWindow;
using ::cudaLaunchAttributeClusterDimension;
using ::cudaLaunchAttributeClusterSchedulingPolicyPreference;
using ::cudaLaunchAttributeCooperative;
using ::cudaLaunchAttributeDeviceUpdatableKernelNode;
using ::cudaLaunchAttributeIgnore;
using ::cudaLaunchAttributeLaunchCompletionEvent;
using ::cudaLaunchAttributeMemSyncDomain;
using ::cudaLaunchAttributeMemSyncDomainMap;
using ::cudaLaunchAttributeNvlinkUtilCentricScheduling;
using ::cudaLaunchAttributePreferredClusterDimension;
using ::cudaLaunchAttributePreferredSharedMemoryCarveout;
using ::cudaLaunchAttributePriority;
using ::cudaLaunchAttributeProgrammaticEvent;
using ::cudaLaunchAttributeProgrammaticStreamSerialization;
using ::cudaLaunchAttributeSynchronizationPolicy;

using ::cudaGraphExecUpdateResultInfo;
using ::cudaGraphInstantiateParams;
using ::cudaLaunchAttributeValue;

// Memory pool types
using ::cudaMemAllocNodeParams;
using ::cudaMemPool_t;
using ::cudaMemPoolProps;
using ::cudaMemPoolPtrExportData;

// Stream callback
using ::cudaStreamCallback_t;

// User object types
using ::cudaUserObject_t;

// Texture and surface types
using ::cudaSurfaceObject_t;
using ::cudaTextureDesc;
using ::cudaTextureObject_t;

// ========================================================================
// Enumerations
// ========================================================================
using ::cudaMemcpyKind;
// cudaMemcpyKind enum values
using ::cudaMemcpyDefault;
using ::cudaMemcpyDeviceToDevice;
using ::cudaMemcpyDeviceToHost;
using ::cudaMemcpyHostToDevice;
using ::cudaMemcpyHostToHost;

using ::cudaMemoryType;
// cudaMemoryType enum values
using ::cudaMemoryTypeDevice;
using ::cudaMemoryTypeHost;
using ::cudaMemoryTypeManaged;
using ::cudaMemoryTypeUnregistered;

using ::cudaChannelFormatKind;
// cudaChannelFormatKind enum values
using ::cudaChannelFormatKindFloat;
using ::cudaChannelFormatKindNone;
using ::cudaChannelFormatKindNV12;
using ::cudaChannelFormatKindSigned;
using ::cudaChannelFormatKindUnsigned;

// Device and function enums
using ::cudaComputeMode;
// cudaComputeMode enum values
using ::cudaComputeModeDefault;
using ::cudaComputeModeExclusive;
using ::cudaComputeModeExclusiveProcess;
using ::cudaComputeModeProhibited;

using ::cudaLimit;
// cudaLimit enum values
using ::cudaLimitDevRuntimePendingLaunchCount;
using ::cudaLimitDevRuntimeSyncDepth;
using ::cudaLimitMallocHeapSize;
using ::cudaLimitMaxL2FetchGranularity;
using ::cudaLimitPersistingL2CacheSize;
using ::cudaLimitPrintfFifoSize;
using ::cudaLimitStackSize;

using ::cudaFuncCache;
// cudaFuncCache enum values
using ::cudaFuncCachePreferEqual;
using ::cudaFuncCachePreferL1;
using ::cudaFuncCachePreferNone;
using ::cudaFuncCachePreferShared;

using ::cudaSharedMemConfig;
// cudaSharedMemConfig enum values
using ::cudaSharedMemBankSizeDefault;
using ::cudaSharedMemBankSizeEightByte;
using ::cudaSharedMemBankSizeFourByte;

using ::cudaFuncAttribute;
// cudaFuncAttribute enum values
using ::cudaFuncAttributeClusterDimMustBeSet;
using ::cudaFuncAttributeClusterSchedulingPolicyPreference;
using ::cudaFuncAttributeMaxDynamicSharedMemorySize;
using ::cudaFuncAttributeNonPortableClusterSizeAllowed;
using ::cudaFuncAttributePreferredSharedMemoryCarveout;
using ::cudaFuncAttributeRequiredClusterDepth;
using ::cudaFuncAttributeRequiredClusterHeight;
using ::cudaFuncAttributeRequiredClusterWidth;

// Memory enums
using ::cudaMemoryAdvise;
// cudaMemoryAdvise enum values
using ::cudaMemAdviseSetAccessedBy;
using ::cudaMemAdviseSetPreferredLocation;
using ::cudaMemAdviseSetReadMostly;
using ::cudaMemAdviseUnsetAccessedBy;
using ::cudaMemAdviseUnsetPreferredLocation;
using ::cudaMemAdviseUnsetReadMostly;

using ::cudaMemRangeAttribute;
// cudaMemRangeAttribute enum values
using ::cudaMemRangeAttributeAccessedBy;
using ::cudaMemRangeAttributeLastPrefetchLocation;
using ::cudaMemRangeAttributeLastPrefetchLocationId;
using ::cudaMemRangeAttributeLastPrefetchLocationType;
using ::cudaMemRangeAttributePreferredLocation;
using ::cudaMemRangeAttributePreferredLocationId;
using ::cudaMemRangeAttributePreferredLocationType;
using ::cudaMemRangeAttributeReadMostly;

using ::cudaMemPoolAttr;
// cudaMemPoolAttr enum values
using ::cudaMemPoolAttrReleaseThreshold;
using ::cudaMemPoolAttrReservedMemCurrent;
using ::cudaMemPoolAttrReservedMemHigh;
using ::cudaMemPoolAttrUsedMemCurrent;
using ::cudaMemPoolAttrUsedMemHigh;
using ::cudaMemPoolReuseAllowInternalDependencies;
using ::cudaMemPoolReuseAllowOpportunistic;
using ::cudaMemPoolReuseFollowEventDependencies;

using ::cudaMemAllocationType;
// cudaMemAllocationType enum values
using ::cudaMemAllocationTypeInvalid;
using ::cudaMemAllocationTypeManaged;
using ::cudaMemAllocationTypePinned;

using ::cudaMemAllocationHandleType;
// cudaMemAllocationHandleType enum values
using ::cudaMemHandleTypeFabric;
using ::cudaMemHandleTypeNone;
using ::cudaMemHandleTypePosixFileDescriptor;
using ::cudaMemHandleTypeWin32;
using ::cudaMemHandleTypeWin32Kmt;

using ::cudaMemLocationType;
// cudaMemLocationType enum values
using ::cudaMemLocationTypeDevice;
using ::cudaMemLocationTypeHost;
using ::cudaMemLocationTypeHostNuma;
using ::cudaMemLocationTypeHostNumaCurrent;
using ::cudaMemLocationTypeInvalid;
using ::cudaMemLocationTypeNone; // alias for Invalid (= 0)

// Stream enums
using ::cudaStreamCaptureStatus;
// cudaStreamCaptureStatus enum values
using ::cudaStreamCaptureStatusActive;
using ::cudaStreamCaptureStatusInvalidated;
using ::cudaStreamCaptureStatusNone;

using ::cudaStreamCaptureMode;
// cudaStreamCaptureMode enum values
using ::cudaStreamCaptureModeGlobal;
using ::cudaStreamCaptureModeRelaxed;
using ::cudaStreamCaptureModeThreadLocal;

using ::cudaStreamUpdateCaptureDependenciesFlags;
// cudaStreamUpdateCaptureDependenciesFlags enum values
using ::cudaStreamAddCaptureDependencies;
using ::cudaStreamSetCaptureDependencies;

// Synchronization
using ::cudaSynchronizationPolicy;
// cudaSynchronizationPolicy enum values
using ::cudaSyncPolicyAuto;
using ::cudaSyncPolicyBlockingSync;
using ::cudaSyncPolicySpin;
using ::cudaSyncPolicyYield;

// Resource types
using ::cudaResourceType;
// cudaResourceType enum values
using ::cudaResourceTypeArray;
using ::cudaResourceTypeLinear;
using ::cudaResourceTypeMipmappedArray;
using ::cudaResourceTypePitch2D;

using ::cudaResourceViewFormat;
// cudaResourceViewFormat enum values
using ::cudaResViewFormatFloat1;
using ::cudaResViewFormatFloat2;
using ::cudaResViewFormatFloat4;
using ::cudaResViewFormatHalf1;
using ::cudaResViewFormatHalf2;
using ::cudaResViewFormatHalf4;
using ::cudaResViewFormatNone;
using ::cudaResViewFormatSignedBlockCompressed4;
using ::cudaResViewFormatSignedBlockCompressed5;
using ::cudaResViewFormatSignedBlockCompressed6H;
using ::cudaResViewFormatSignedChar1;
using ::cudaResViewFormatSignedChar2;
using ::cudaResViewFormatSignedChar4;
using ::cudaResViewFormatSignedInt1;
using ::cudaResViewFormatSignedInt2;
using ::cudaResViewFormatSignedInt4;
using ::cudaResViewFormatSignedShort1;
using ::cudaResViewFormatSignedShort2;
using ::cudaResViewFormatSignedShort4;
using ::cudaResViewFormatUnsignedBlockCompressed1;
using ::cudaResViewFormatUnsignedBlockCompressed2;
using ::cudaResViewFormatUnsignedBlockCompressed3;
using ::cudaResViewFormatUnsignedBlockCompressed4;
using ::cudaResViewFormatUnsignedBlockCompressed5;
using ::cudaResViewFormatUnsignedBlockCompressed6H;
using ::cudaResViewFormatUnsignedBlockCompressed7;
using ::cudaResViewFormatUnsignedChar1;
using ::cudaResViewFormatUnsignedChar2;
using ::cudaResViewFormatUnsignedChar4;
using ::cudaResViewFormatUnsignedInt1;
using ::cudaResViewFormatUnsignedInt2;
using ::cudaResViewFormatUnsignedInt4;
using ::cudaResViewFormatUnsignedShort1;
using ::cudaResViewFormatUnsignedShort2;
using ::cudaResViewFormatUnsignedShort4;

// Graphics interop
using ::cudaGraphicsRegisterFlags;
// cudaGraphicsRegisterFlags enum values
using ::cudaGraphicsRegisterFlagsNone;
using ::cudaGraphicsRegisterFlagsReadOnly;
using ::cudaGraphicsRegisterFlagsSurfaceLoadStore;
using ::cudaGraphicsRegisterFlagsTextureGather;
using ::cudaGraphicsRegisterFlagsWriteDiscard;

using ::cudaGraphicsMapFlags;
// cudaGraphicsMapFlags enum values
using ::cudaGraphicsMapFlagsNone;
using ::cudaGraphicsMapFlagsReadOnly;
using ::cudaGraphicsMapFlagsWriteDiscard;

// Access properties
using ::cudaAccessProperty;
// cudaAccessProperty enum values
using ::cudaAccessPropertyNormal;
using ::cudaAccessPropertyPersisting;
using ::cudaAccessPropertyStreaming;

// User object flags
using ::cudaUserObjectFlags;
// cudaUserObjectFlags enum values
using ::cudaUserObjectNoDestructorSync;

using ::cudaUserObjectRetainFlags;
// cudaUserObjectRetainFlags enum values
using ::cudaGraphUserObjectMove;

// Device P2P attributes
using ::cudaDeviceP2PAttr;
// cudaDeviceP2PAttr enum values
using ::cudaDevP2PAttrAccessSupported;
using ::cudaDevP2PAttrCudaArrayAccessSupported;
using ::cudaDevP2PAttrNativeAtomicSupported;
using ::cudaDevP2PAttrOnlyPartialNativeAtomicSupported;
using ::cudaDevP2PAttrPerformanceRank;

// External memory/semaphore types
using ::cudaExternalMemoryHandleType;
// cudaExternalMemoryHandleType enum values
using ::cudaExternalMemoryHandleTypeD3D11Resource;
using ::cudaExternalMemoryHandleTypeD3D11ResourceKmt;
using ::cudaExternalMemoryHandleTypeD3D12Heap;
using ::cudaExternalMemoryHandleTypeD3D12Resource;
using ::cudaExternalMemoryHandleTypeNvSciBuf;
using ::cudaExternalMemoryHandleTypeOpaqueFd;
using ::cudaExternalMemoryHandleTypeOpaqueWin32;
using ::cudaExternalMemoryHandleTypeOpaqueWin32Kmt;

using ::cudaExternalSemaphoreHandleType;
// cudaExternalSemaphoreHandleType enum values
using ::cudaExternalSemaphoreHandleTypeD3D11Fence;
using ::cudaExternalSemaphoreHandleTypeD3D12Fence;
using ::cudaExternalSemaphoreHandleTypeKeyedMutex;
using ::cudaExternalSemaphoreHandleTypeKeyedMutexKmt;
using ::cudaExternalSemaphoreHandleTypeNvSciSync;
using ::cudaExternalSemaphoreHandleTypeOpaqueFd;
using ::cudaExternalSemaphoreHandleTypeOpaqueWin32;
using ::cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt;
using ::cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
using ::cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;

// ========================================================================
// Error Handling Functions
// ========================================================================
using ::cudaGetErrorName;
using ::cudaGetErrorString;
using ::cudaGetLastError;
using ::cudaPeekAtLastError;

// ========================================================================
// Device Management Functions
// ========================================================================
using ::cudaChooseDevice;
using ::cudaDeviceGetAttribute;
using ::cudaDeviceGetByPCIBusId;
using ::cudaDeviceGetDefaultMemPool;
using ::cudaDeviceGetMemPool;
using ::cudaDeviceGetPCIBusId;
using ::cudaDeviceReset;
using ::cudaDeviceSetMemPool;
using ::cudaDeviceSynchronize;
using ::cudaGetDevice;
using ::cudaGetDeviceCount;
using ::cudaGetDeviceFlags;
using ::cudaGetDeviceProperties;
using ::cudaSetDevice;
using ::cudaSetDeviceFlags;

// Device configuration
using ::cudaDeviceGetCacheConfig;
using ::cudaDeviceGetLimit;
using ::cudaDeviceGetSharedMemConfig;
using ::cudaDeviceGetStreamPriorityRange;
using ::cudaDeviceGetTexture1DLinearMaxWidth;
using ::cudaDeviceSetCacheConfig;
using ::cudaDeviceSetLimit;
using ::cudaDeviceSetSharedMemConfig;

// Peer-to-peer device access
using ::cudaDeviceCanAccessPeer;
using ::cudaDeviceDisablePeerAccess;
using ::cudaDeviceEnablePeerAccess;
using ::cudaDeviceGetP2PAttribute;

// Advanced device features
using ::cudaCtxResetPersistingL2Cache;
using ::cudaDeviceFlushGPUDirectRDMAWrites;
using ::cudaDeviceGetGraphMemAttribute;
using ::cudaDeviceGetHostAtomicCapabilities;
using ::cudaDeviceGetNvSciSyncAttributes;
using ::cudaDeviceGetP2PAtomicCapabilities;
using ::cudaDeviceGraphMemTrim;
using ::cudaDeviceRegisterAsyncNotification;
using ::cudaDeviceSetGraphMemAttribute;
using ::cudaDeviceUnregisterAsyncNotification;

// Driver version
using ::cudaDriverGetVersion;
using ::cudaRuntimeGetVersion;

// ========================================================================
// Memory Management Functions
// ========================================================================
// Device memory allocation
using ::cudaFree;
using ::cudaFreeArray;
using ::cudaFreeAsync;
using ::cudaFreeMipmappedArray;
using ::cudaMalloc;
using ::cudaMalloc3D;
using ::cudaMalloc3DArray;
using ::cudaMallocArray;
using ::cudaMallocAsync;
using ::cudaMallocManaged;
using ::cudaMallocMipmappedArray;
using ::cudaMallocPitch;

// Host memory (pinned)
using ::cudaFreeHost;
using ::cudaHostAlloc;
using ::cudaHostGetDevicePointer;
using ::cudaHostGetFlags;
using ::cudaHostRegister;
using ::cudaHostUnregister;
using ::cudaMallocHost;

// Memory copy operations
using ::cudaMemcpy;
using ::cudaMemcpy2D;
using ::cudaMemcpy2DArrayToArray;
using ::cudaMemcpy2DAsync;
using ::cudaMemcpy2DFromArray;
using ::cudaMemcpy2DFromArrayAsync;
using ::cudaMemcpy2DToArray;
using ::cudaMemcpy2DToArrayAsync;
using ::cudaMemcpy3D;
using ::cudaMemcpy3DAsync;
using ::cudaMemcpy3DPeer;
using ::cudaMemcpy3DPeerAsync;
using ::cudaMemcpyArrayToArray;
using ::cudaMemcpyAsync;
using ::cudaMemcpyFromArray;
using ::cudaMemcpyFromArrayAsync;
using ::cudaMemcpyFromSymbol;
using ::cudaMemcpyFromSymbolAsync;
using ::cudaMemcpyPeer;
using ::cudaMemcpyPeerAsync;
using ::cudaMemcpyToArray;
using ::cudaMemcpyToArrayAsync;
using ::cudaMemcpyToSymbol;
using ::cudaMemcpyToSymbolAsync;

// Memory set operations
using ::cudaMemset;
using ::cudaMemset2D;
using ::cudaMemset2DAsync;
using ::cudaMemset3D;
using ::cudaMemset3DAsync;
using ::cudaMemsetAsync;

// Memory info and advising
using ::cudaMemAdvise;
using ::cudaMemGetInfo;
using ::cudaMemPrefetchAsync;
using ::cudaMemRangeGetAttribute;
using ::cudaMemRangeGetAttributes;

// Pointer queries
using ::cudaPointerGetAttributes;

// Array operations
using ::cudaArrayGetInfo;
using ::cudaArrayGetMemoryRequirements;
using ::cudaArrayGetPlane;
using ::cudaArrayGetSparseProperties;
using ::cudaGetMipmappedArrayLevel;

// Memory pools (CUDA 11.2+)
using ::cudaMallocFromPoolAsync;
using ::cudaMemPoolCreate;
using ::cudaMemPoolDestroy;
using ::cudaMemPoolExportPointer;
using ::cudaMemPoolExportToShareableHandle;
using ::cudaMemPoolGetAccess;
using ::cudaMemPoolGetAttribute;
using ::cudaMemPoolImportFromShareableHandle;
using ::cudaMemPoolImportPointer;
using ::cudaMemPoolSetAccess;
using ::cudaMemPoolSetAttribute;
using ::cudaMemPoolTrimTo;

// External memory
using ::cudaDestroyExternalMemory;
using ::cudaExternalMemoryGetMappedBuffer;
using ::cudaExternalMemoryGetMappedMipmappedArray;
using ::cudaImportExternalMemory;

// IPC (Inter-Process Communication)
using ::cudaIpcCloseMemHandle;
using ::cudaIpcGetEventHandle;
using ::cudaIpcGetMemHandle;
using ::cudaIpcOpenEventHandle;
using ::cudaIpcOpenMemHandle;

// ========================================================================
// Stream Management Functions
// ========================================================================
using ::cudaStreamAddCallback;
using ::cudaStreamAttachMemAsync;
using ::cudaStreamBeginCapture;
using ::cudaStreamCopyAttributes;
using ::cudaStreamCreate;
using ::cudaStreamCreateWithFlags;
using ::cudaStreamCreateWithPriority;
using ::cudaStreamDestroy;
using ::cudaStreamEndCapture;
using ::cudaStreamGetAttribute;
using ::cudaStreamGetCaptureInfo;
using ::cudaStreamGetFlags;
using ::cudaStreamGetId;
using ::cudaStreamGetPriority;
using ::cudaStreamIsCapturing;
using ::cudaStreamQuery;
using ::cudaStreamSetAttribute;
using ::cudaStreamSynchronize;
using ::cudaStreamUpdateCaptureDependencies;
using ::cudaStreamWaitEvent;
using ::cudaThreadExchangeStreamCaptureMode;

// ========================================================================
// Event Management Functions
// ========================================================================
using ::cudaEventCreate;
using ::cudaEventCreateWithFlags;
using ::cudaEventDestroy;
using ::cudaEventElapsedTime;
using ::cudaEventQuery;
using ::cudaEventRecord;
using ::cudaEventRecordWithFlags;
using ::cudaEventSynchronize;

// External semaphores
using ::cudaDestroyExternalSemaphore;
using ::cudaImportExternalSemaphore;
using ::cudaSignalExternalSemaphoresAsync;
using ::cudaWaitExternalSemaphoresAsync;

// ========================================================================
// Execution Control (Kernel Launch)
// ========================================================================
using ::cudaLaunchCooperativeKernel;
using ::cudaLaunchHostFunc;
using ::cudaLaunchKernel;

// ========================================================================
// Occupancy and Function Configuration
// ========================================================================
using ::cudaOccupancyAvailableDynamicSMemPerBlock;
using ::cudaOccupancyMaxActiveBlocksPerMultiprocessor;
using ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags;
using ::cudaOccupancyMaxActiveClusters;
using ::cudaOccupancyMaxPotentialClusterSize;

using ::cudaFuncGetAttributes;
using ::cudaFuncGetName;
using ::cudaFuncGetParamInfo;
using ::cudaFuncSetAttribute;
using ::cudaFuncSetCacheConfig;
using ::cudaFuncSetSharedMemConfig;

// ========================================================================
// Graph Management
// ========================================================================
// Graph creation and destruction
using ::cudaGraphAddDependencies;
using ::cudaGraphClone;
using ::cudaGraphCreate;
using ::cudaGraphDebugDotPrint;
using ::cudaGraphDestroy;
using ::cudaGraphGetEdges;
using ::cudaGraphGetNodes;
using ::cudaGraphGetRootNodes;
using ::cudaGraphNodeFindInClone;
using ::cudaGraphNodeGetDependencies;
using ::cudaGraphNodeGetDependentNodes;
using ::cudaGraphNodeGetEnabled;
using ::cudaGraphNodeGetType;
using ::cudaGraphNodeSetEnabled;
using ::cudaGraphRemoveDependencies;

// Graph nodes
using ::cudaGraphAddChildGraphNode;
using ::cudaGraphAddEmptyNode;
using ::cudaGraphAddEventRecordNode;
using ::cudaGraphAddEventWaitNode;
using ::cudaGraphAddExternalSemaphoresSignalNode;
using ::cudaGraphAddExternalSemaphoresWaitNode;
using ::cudaGraphAddHostNode;
using ::cudaGraphAddKernelNode;
using ::cudaGraphAddMemAllocNode;
using ::cudaGraphAddMemcpyNode;
using ::cudaGraphAddMemcpyNode1D;
using ::cudaGraphAddMemcpyNodeFromSymbol;
using ::cudaGraphAddMemcpyNodeToSymbol;
using ::cudaGraphAddMemFreeNode;
using ::cudaGraphAddMemsetNode;
using ::cudaGraphAddNode;

// Graph node operations
using ::cudaGraphChildGraphNodeGetGraph;
using ::cudaGraphDestroyNode;
using ::cudaGraphEventRecordNodeGetEvent;
using ::cudaGraphEventRecordNodeSetEvent;
using ::cudaGraphEventWaitNodeGetEvent;
using ::cudaGraphEventWaitNodeSetEvent;
using ::cudaGraphExternalSemaphoresSignalNodeGetParams;
using ::cudaGraphExternalSemaphoresSignalNodeSetParams;
using ::cudaGraphExternalSemaphoresWaitNodeGetParams;
using ::cudaGraphExternalSemaphoresWaitNodeSetParams;
using ::cudaGraphHostNodeGetParams;
using ::cudaGraphHostNodeSetParams;
using ::cudaGraphKernelNodeCopyAttributes;
using ::cudaGraphKernelNodeGetAttribute;
using ::cudaGraphKernelNodeGetParams;
using ::cudaGraphKernelNodeSetAttribute;
using ::cudaGraphKernelNodeSetParams;
using ::cudaGraphMemAllocNodeGetParams;
using ::cudaGraphMemcpyNodeGetParams;
using ::cudaGraphMemcpyNodeSetParams;
using ::cudaGraphMemcpyNodeSetParams1D;
using ::cudaGraphMemcpyNodeSetParamsFromSymbol;
using ::cudaGraphMemcpyNodeSetParamsToSymbol;
using ::cudaGraphMemFreeNodeGetParams;
using ::cudaGraphMemsetNodeGetParams;
using ::cudaGraphMemsetNodeSetParams;
using ::cudaGraphNodeSetParams;

// Graph execution
using ::cudaGraphExecDestroy;
using ::cudaGraphExecGetFlags;
using ::cudaGraphExecUpdate;
using ::cudaGraphInstantiate;
using ::cudaGraphInstantiateWithFlags;
using ::cudaGraphInstantiateWithParams;
using ::cudaGraphLaunch;
using ::cudaGraphUpload;

// Graph exec node updates
using ::cudaGraphExecChildGraphNodeSetParams;
using ::cudaGraphExecEventRecordNodeSetEvent;
using ::cudaGraphExecEventWaitNodeSetEvent;
using ::cudaGraphExecExternalSemaphoresSignalNodeSetParams;
using ::cudaGraphExecExternalSemaphoresWaitNodeSetParams;
using ::cudaGraphExecHostNodeSetParams;
using ::cudaGraphExecKernelNodeSetParams;
using ::cudaGraphExecMemcpyNodeSetParams;
using ::cudaGraphExecMemcpyNodeSetParams1D;
using ::cudaGraphExecMemcpyNodeSetParamsFromSymbol;
using ::cudaGraphExecMemcpyNodeSetParamsToSymbol;
using ::cudaGraphExecMemsetNodeSetParams;
using ::cudaGraphExecNodeSetParams;

// Conditional graphs
using ::cudaGraphConditionalHandleCreate;

// ========================================================================
// Texture and Surface Management
// ========================================================================
using ::cudaCreateChannelDesc;
using ::cudaCreateSurfaceObject;
using ::cudaCreateTextureObject;
using ::cudaDestroySurfaceObject;
using ::cudaDestroyTextureObject;
using ::cudaGetChannelDesc;
using ::cudaGetSurfaceObjectResourceDesc;
using ::cudaGetTextureObjectResourceDesc;
using ::cudaGetTextureObjectResourceViewDesc;
using ::cudaGetTextureObjectTextureDesc;

// ========================================================================
// Symbol and Kernel Management
// ========================================================================
using ::cudaGetFuncBySymbol;
using ::cudaGetKernel;
using ::cudaGetSymbolAddress;
using ::cudaGetSymbolSize;

// ========================================================================
// User Object Management
// ========================================================================
using ::cudaGraphReleaseUserObject;
using ::cudaGraphRetainUserObject;
using ::cudaUserObjectCreate;
using ::cudaUserObjectRelease;
using ::cudaUserObjectRetain;

// ========================================================================
// Graphics Interoperability
// ========================================================================
using ::cudaGraphicsMapResources;
using ::cudaGraphicsResourceGetMappedPointer;
using ::cudaGraphicsResourceSetMapFlags;
using ::cudaGraphicsSubResourceGetMappedArray;
using ::cudaGraphicsUnmapResources;
using ::cudaGraphicsUnregisterResource;

// ========================================================================
// Driver Entry Points
// ========================================================================
using ::cudaGetDriverEntryPoint;
using ::cudaGetDriverEntryPointByVersion;
using ::cudaGetExportTable;

} // namespace wwr::cuda
