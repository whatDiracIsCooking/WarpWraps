/**
 * @file hip_runtime_api.cppm
 * @brief HIP runtime API module wrapper for gpumod project
 *
 * Wraps hip/hip_runtime_api.h. HIP does not split the driver API from the
 * runtime API the way CUDA does, so this one module covers what
 * wwr.cuda.cuda_h + wwr.cuda.cuda_runtime_api cover together. See
 * src/hip/README.md "Design decisions".
 *
 * Unlike CUDA, HIP keeps its templated convenience layer (hipMalloc<T>,
 * hipLaunchKernelEx, the hipBindTexture* family, ...) in this same header
 * rather than a separate one. Names existing ONLY as such templates get no
 * `using` declaration; names with both an extern "C" and a template overload
 * get a forwarding function, because a using-declaration cannot select one
 * overload. __HIP_DISABLE_CPP_FUNCTIONS__ (defined below, before the include)
 * is the header's own escape hatch for five of them. See
 * docs/architecture.md, section 12, for the full list and both reasons.
 *
 * The driver/array/texture/surface types come from headers hip_runtime_api.h
 * includes itself, gated on __HIP_PLATFORM_AMD__, which hip::host defines.
 * Nothing extra to include here.
 *
 * Usage:
 *   import wwr.hip.hip_runtime_api;
 */

module;

#define __HIP_DISABLE_CPP_FUNCTIONS__ 1
#include <hip/hip_runtime_api.h>

// Validate HIP flag macro values at compile time before we #undef them.
// These mirror CUDA's cudaHostAlloc*/cudaEvent*/cudaStream* groups in shape,
// plus hipSetDeviceFlags/hipMallocManaged groups CUDA's cuda_runtime_api.cppm
// has no counterpart for. Names and values were verified against
// hip/hip_runtime_api.h -- HIP's values happen to equal CUDA's for the groups
// both have, but nothing here assumes that; each was checked independently.
// hipHostMalloc flags (hipHostAlloc* are equal-valued aliases, not re-exported separately)
static_assert(hipHostMallocDefault == 0x0, "hipHostMallocDefault value mismatch");
static_assert(hipHostMallocPortable == 0x1, "hipHostMallocPortable value mismatch");
static_assert(hipHostMallocMapped == 0x2, "hipHostMallocMapped value mismatch");
static_assert(hipHostMallocWriteCombined == 0x4, "hipHostMallocWriteCombined value mismatch");
static_assert(hipHostMallocNumaUser == 0x20000000, "hipHostMallocNumaUser value mismatch");
static_assert(hipHostMallocCoherent == 0x40000000, "hipHostMallocCoherent value mismatch");
static_assert(hipHostMallocNonCoherent == 0x80000000, "hipHostMallocNonCoherent value mismatch");

// hipEvent flags
static_assert(hipEventDefault == 0x0, "hipEventDefault value mismatch");
static_assert(hipEventBlockingSync == 0x1, "hipEventBlockingSync value mismatch");
static_assert(hipEventDisableTiming == 0x2, "hipEventDisableTiming value mismatch");
static_assert(hipEventInterprocess == 0x4, "hipEventInterprocess value mismatch");

// hipStream flags
static_assert(hipStreamDefault == 0x00, "hipStreamDefault value mismatch");
static_assert(hipStreamNonBlocking == 0x01, "hipStreamNonBlocking value mismatch");

// hipSetDeviceFlags / hipInit flags
static_assert(hipDeviceScheduleAuto == 0x0, "hipDeviceScheduleAuto value mismatch");
static_assert(hipDeviceScheduleSpin == 0x1, "hipDeviceScheduleSpin value mismatch");
static_assert(hipDeviceScheduleYield == 0x2, "hipDeviceScheduleYield value mismatch");
static_assert(hipDeviceScheduleBlockingSync == 0x4, "hipDeviceScheduleBlockingSync value mismatch");
static_assert(hipDeviceScheduleMask == 0x7, "hipDeviceScheduleMask value mismatch");
static_assert(hipDeviceMapHost == 0x8, "hipDeviceMapHost value mismatch");
static_assert(hipDeviceLmemResizeToMax == 0x10, "hipDeviceLmemResizeToMax value mismatch");

// hipMallocManaged / hipHostRegister attach flags
static_assert(hipMemAttachGlobal == 0x01, "hipMemAttachGlobal value mismatch");
static_assert(hipMemAttachHost == 0x02, "hipMemAttachHost value mismatch");
static_assert(hipMemAttachSingle == 0x04, "hipMemAttachSingle value mismatch");

// Undefine macros so we can create constexpr variables with the same names
#undef hipHostMallocDefault
#undef hipHostMallocPortable
#undef hipHostMallocMapped
#undef hipHostMallocWriteCombined
#undef hipHostMallocNumaUser
#undef hipHostMallocCoherent
#undef hipHostMallocNonCoherent
#undef hipEventDefault
#undef hipEventBlockingSync
#undef hipEventDisableTiming
#undef hipEventInterprocess
#undef hipStreamDefault
#undef hipStreamNonBlocking
#undef hipDeviceScheduleAuto
#undef hipDeviceScheduleSpin
#undef hipDeviceScheduleYield
#undef hipDeviceScheduleBlockingSync
#undef hipDeviceScheduleMask
#undef hipDeviceMapHost
#undef hipDeviceLmemResizeToMax
#undef hipMemAttachGlobal
#undef hipMemAttachHost
#undef hipMemAttachSingle

export module wwr.hip.hip_runtime_api;

// ========================================================================
// Export all HIP types, functions, and constants in wwr::hip
// (NOT bare wwr -- see src/hip/README.md "Design decisions": a real
// `half` collision with wwr.cuda.cuda_fp16 if this were bare)
// ========================================================================

export namespace wwr::hip {

// ========================================================================
// Constexpr wrappers for HIP flag macros
// ========================================================================
// Values validated by static_assert above
// Note: 'inline' is implicit for exported module variables

// hipHostMalloc flags (hipHostAlloc* are equal-valued aliases, not re-exported separately)
constexpr unsigned int hipHostMallocDefault = 0x0;
constexpr unsigned int hipHostMallocPortable = 0x1;
constexpr unsigned int hipHostMallocMapped = 0x2;
constexpr unsigned int hipHostMallocWriteCombined = 0x4;
constexpr unsigned int hipHostMallocNumaUser = 0x20000000;
constexpr unsigned int hipHostMallocCoherent = 0x40000000;
constexpr unsigned int hipHostMallocNonCoherent = 0x80000000;

// hipEvent flags
constexpr unsigned int hipEventDefault = 0x0;
constexpr unsigned int hipEventBlockingSync = 0x1;
constexpr unsigned int hipEventDisableTiming = 0x2;
constexpr unsigned int hipEventInterprocess = 0x4;

// hipStream flags
constexpr unsigned int hipStreamDefault = 0x00;
constexpr unsigned int hipStreamNonBlocking = 0x01;

// hipSetDeviceFlags / hipInit flags
constexpr unsigned int hipDeviceScheduleAuto = 0x0;
constexpr unsigned int hipDeviceScheduleSpin = 0x1;
constexpr unsigned int hipDeviceScheduleYield = 0x2;
constexpr unsigned int hipDeviceScheduleBlockingSync = 0x4;
constexpr unsigned int hipDeviceScheduleMask = 0x7;
constexpr unsigned int hipDeviceMapHost = 0x8;
constexpr unsigned int hipDeviceLmemResizeToMax = 0x10;

// hipMallocManaged / hipHostRegister attach flags
constexpr unsigned int hipMemAttachGlobal = 0x01;
constexpr unsigned int hipMemAttachHost = 0x02;
constexpr unsigned int hipMemAttachSingle = 0x04;

// ========================================================================
// Types
// ========================================================================

// ------------------------------------------------------------------------
// Core error type
// ------------------------------------------------------------------------
using ::hipError_t;

// ------------------------------------------------------------------------
// Device properties and identification
// ------------------------------------------------------------------------
using ::hipComputeMode;
using ::hipDeviceArch_t;
using ::hipDeviceAttribute_t;
using ::hipDeviceP2PAttr;
using ::hipDeviceProp_tR0600;
using ::hipFlushGPUDirectRDMAWritesOptions;
using ::hipGPUDirectRDMAWritesOrdering;
using ::hipMemoryType;
using ::hipPointerAttribute_t;
using ::hipUUID;

// ------------------------------------------------------------------------
// Driver-API-shaped handles (module/context/library)
// ------------------------------------------------------------------------
using ::hipCtx_t;
using ::hipDevice_t;
using ::hipDeviceptr_t;
using ::hipDriverEntryPointQueryResult;
using ::hipDriverProcAddressQueryResult;
using ::hipFunction_t;
using ::hipKernel_t;
using ::hipLibrary_t;
using ::hipLinkState_t;
using ::hipModule_t;

// ------------------------------------------------------------------------
// Streams and events
// ------------------------------------------------------------------------
using ::hipBatchMemOpNodeParams;
using ::hipEvent_t;
using ::hipFuncAttributes;
using ::hipIpcEventHandle_t;
using ::hipIpcMemHandle_t;
using ::hipStream_t;
using ::hipStreamBatchMemOpParams;
using ::hipStreamBatchMemOpType;
using ::hipStreamCallback_t;

// ------------------------------------------------------------------------
// Memory advise, pools, and virtual memory
// ------------------------------------------------------------------------
using ::hipArrayMapInfo;
using ::hipArraySparseSubresourceType;
using ::hipMemAccessDesc;
using ::hipMemAccessFlags;
using ::hipMemAllocationGranularity_flags;
using ::hipMemAllocationHandleType;
using ::hipMemAllocationProp;
using ::hipMemAllocationType;
using ::hipMemcpyAttributes;
using ::hipMemcpyFlags;
using ::hipMemcpySrcAccessOrder;
using ::hipMemGenericAllocationHandle_t;
using ::hipMemHandleType;
using ::hipMemLocation;
using ::hipMemLocationType;
using ::hipMemOperationType;
using ::hipMemoryAdvise;
using ::hipMemPool_t;
using ::hipMemPoolAttr;
using ::hipMemPoolProps;
using ::hipMemPoolPtrExportData;
using ::hipMemRangeAttribute;
using ::hipMemRangeCoherencyMode;
using ::hipMemRangeFlags;
using ::hipMemRangeHandleType;

// ------------------------------------------------------------------------
// Function configuration and launch
// ------------------------------------------------------------------------
using ::dim3;
using ::hipFuncAttribute;
using ::hipFuncCache_t;
using ::hipFunction_attribute;
using ::hipFunctionLaunchParams;
using ::hipLaunchParams;
using ::hipPointer_attribute;
using ::hipSharedMemConfig;

// ------------------------------------------------------------------------
// External memory and semaphores
// ------------------------------------------------------------------------
using ::hipExternalMemory_t;
using ::hipExternalMemoryBufferDesc;
using ::hipExternalMemoryHandleDesc;
using ::hipExternalMemoryHandleType;
using ::hipExternalMemoryMipmappedArrayDesc;
using ::hipExternalSemaphore_t;
using ::hipExternalSemaphoreHandleDesc;
using ::hipExternalSemaphoreHandleType;
using ::hipExternalSemaphoreSignalParams;
using ::hipExternalSemaphoreWaitParams;

// ------------------------------------------------------------------------
// Graphics interoperability
// ------------------------------------------------------------------------
using ::hipGraphicsRegisterFlags;
using ::hipGraphicsResource;
using ::hipGraphicsResource_t;

// ------------------------------------------------------------------------
// Graphs
// ------------------------------------------------------------------------
using ::HIP_LAUNCH_CONFIG;
using ::hipAccessPolicyWindow;
using ::hipAccessProperty;
using ::hipChildGraphNodeParams;
using ::hipEventRecordNodeParams;
using ::hipEventWaitNodeParams;
using ::hipExternalSemaphoreSignalNodeParams;
using ::hipExternalSemaphoreWaitNodeParams;
using ::hipGraph_t;
using ::hipGraphDebugDotFlags;
using ::hipGraphDependencyType;
using ::hipGraphEdgeData;
using ::hipGraphExec_t;
using ::hipGraphExecUpdateResult;
using ::hipGraphInstantiateFlags;
using ::hipGraphInstantiateParams;
using ::hipGraphInstantiateResult;
using ::hipGraphMemAttributeType;
using ::hipGraphNode_t;
using ::hipGraphNodeParams;
using ::hipGraphNodeType;
using ::hipHostFn_t;
using ::hipHostNodeParams;
using ::hipKernelNodeParams;
using ::hipLaunchAttribute;
using ::hipLaunchAttributeID;
using ::hipLaunchAttributeValue;
using ::hipLaunchConfig_t;
using ::hipLaunchMemSyncDomain;
using ::hipLaunchMemSyncDomainMap;
using ::hipMemAllocNodeParams;
using ::hipMemcpyNodeParams;
using ::hipMemFreeNodeParams;
using ::hipMemsetParams;
using ::hipStreamCaptureMode;
using ::hipStreamCaptureStatus;
using ::hipStreamUpdateCaptureDependenciesFlags;
using ::hipSynchronizationPolicy;
using ::hipUserObject_t;
using ::hipUserObjectFlags;
using ::hipUserObjectRetainFlags;

// ------------------------------------------------------------------------
// Arrays and mipmapped arrays
// ------------------------------------------------------------------------
using ::HIP_ARRAY3D_DESCRIPTOR;
using ::HIP_ARRAY_DESCRIPTOR;
using ::hip_Memcpy2D;
using ::hipArray_const_t;
using ::hipArray_Format;
using ::hipArray_t;
using ::hipMipmappedArray;
using ::hipMipmappedArray_const_t;
using ::hipMipmappedArray_t;

// ------------------------------------------------------------------------
// Textures, surfaces, and resource descriptors
// ------------------------------------------------------------------------
using ::HIP_RESOURCE_DESC;
using ::HIP_RESOURCE_VIEW_DESC;
using ::HIP_TEXTURE_DESC;
using ::HIPaddress_mode;
using ::hipChannelFormatDesc;
using ::hipChannelFormatKind;
using ::HIPfilter_mode;
using ::hipResourceDesc;
using ::hipResourceType;
using ::HIPresourcetype;
using ::hipResourceViewDesc;
using ::hipResourceViewFormat;
using ::HIPresourceViewFormat;
using ::hipSurfaceBoundaryMode;
using ::hipSurfaceObject_t;
using ::hipTexRef;
using ::hipTextureAddressMode;
using ::hipTextureDesc;
using ::hipTextureFilterMode;
using ::hipTextureObject_t;
using ::hipTextureReadMode;
using ::textureReference;

// ------------------------------------------------------------------------
// Memory copy geometry
// ------------------------------------------------------------------------
using ::HIP_MEMCPY3D;
using ::hipExtent;
using ::hipMemcpy3DBatchOp;
using ::hipMemcpy3DOperand;
using ::hipMemcpy3DOperandType;
using ::hipMemcpy3DParms;
using ::hipMemcpy3DPeerParms;
using ::hipMemcpyKind;
using ::hipOffset3D;
using ::hipPitchedPtr;
using ::hipPos;

// ========================================================================
// Enumerations (types + enumerators)
// ========================================================================

// hipMemoryType
using ::hipMemoryTypeArray;
using ::hipMemoryTypeDevice;
using ::hipMemoryTypeHost;
using ::hipMemoryTypeManaged;
using ::hipMemoryTypeUnified;
using ::hipMemoryTypeUnregistered;

// hipError_t
using ::hipErrorAlreadyAcquired;
using ::hipErrorAlreadyMapped;
using ::hipErrorArrayIsMapped;
using ::hipErrorAssert;
using ::hipErrorCapturedEvent;
using ::hipErrorContextAlreadyCurrent;
using ::hipErrorContextAlreadyInUse;
using ::hipErrorContextIsDestroyed;
using ::hipErrorCooperativeLaunchTooLarge;
using ::hipErrorDeinitialized;
using ::hipErrorECCNotCorrectable;
using ::hipErrorFileNotFound;
using ::hipErrorGraphExecUpdateFailure;
using ::hipErrorHostMemoryAlreadyRegistered;
using ::hipErrorHostMemoryNotRegistered;
using ::hipErrorIllegalAddress;
using ::hipErrorIllegalState;
using ::hipErrorInitializationError;
using ::hipErrorInsufficientDriver;
using ::hipErrorInvalidChannelDescriptor;
using ::hipErrorInvalidConfiguration;
using ::hipErrorInvalidContext;
using ::hipErrorInvalidDevice;
using ::hipErrorInvalidDeviceFunction;
using ::hipErrorInvalidDevicePointer;
using ::hipErrorInvalidGraphicsContext;
using ::hipErrorInvalidHandle;
using ::hipErrorInvalidImage;
using ::hipErrorInvalidKernelFile;
using ::hipErrorInvalidMemcpyDirection;
using ::hipErrorInvalidPitchValue;
using ::hipErrorInvalidResourceHandle;
using ::hipErrorInvalidSource;
using ::hipErrorInvalidSymbol;
using ::hipErrorInvalidTexture;
using ::hipErrorInvalidValue;
using ::hipErrorLaunchFailure;
using ::hipErrorLaunchOutOfResources;
using ::hipErrorLaunchTimeOut;
using ::hipErrorMapBufferObjectFailed;
using ::hipErrorMapFailed;
using ::hipErrorMemoryAllocation;
using ::hipErrorMissingConfiguration;
using ::hipErrorNoBinaryForGpu;
using ::hipErrorNoDevice;
using ::hipErrorNotFound;
using ::hipErrorNotInitialized;
using ::hipErrorNotMapped;
using ::hipErrorNotMappedAsArray;
using ::hipErrorNotMappedAsPointer;
using ::hipErrorNotReady;
using ::hipErrorNotSupported;
using ::hipErrorOperatingSystem;
using ::hipErrorOutOfMemory;
using ::hipErrorPeerAccessAlreadyEnabled;
using ::hipErrorPeerAccessNotEnabled;
using ::hipErrorPeerAccessUnsupported;
using ::hipErrorPriorLaunchFailure;
using ::hipErrorProfilerAlreadyStarted;
using ::hipErrorProfilerAlreadyStopped;
using ::hipErrorProfilerDisabled;
using ::hipErrorProfilerNotInitialized;
using ::hipErrorRuntimeMemory;
using ::hipErrorRuntimeOther;
using ::hipErrorSetOnActiveProcess;
using ::hipErrorSharedObjectInitFailed;
using ::hipErrorSharedObjectSymbolNotFound;
using ::hipErrorStreamCaptureImplicit;
using ::hipErrorStreamCaptureInvalidated;
using ::hipErrorStreamCaptureIsolation;
using ::hipErrorStreamCaptureMerge;
using ::hipErrorStreamCaptureUnjoined;
using ::hipErrorStreamCaptureUnmatched;
using ::hipErrorStreamCaptureUnsupported;
using ::hipErrorStreamCaptureWrongThread;
using ::hipErrorTbd;
using ::hipErrorUnknown;
using ::hipErrorUnmapFailed;
using ::hipErrorUnsupportedLimit;
using ::hipSuccess;

// hipDeviceAttribute_t
using ::hipDeviceAttributeAccessPolicyMaxWindowSize;
using ::hipDeviceAttributeAmdSpecificBegin;
using ::hipDeviceAttributeAmdSpecificEnd;
using ::hipDeviceAttributeAsicRevision;
using ::hipDeviceAttributeAsyncEngineCount;
using ::hipDeviceAttributeCanMapHostMemory;
using ::hipDeviceAttributeCanUseHostPointerForRegisteredMem;
using ::hipDeviceAttributeCanUseStreamWaitValue;
using ::hipDeviceAttributeClockInstructionRate;
using ::hipDeviceAttributeClockRate;
using ::hipDeviceAttributeComputeCapabilityMajor;
using ::hipDeviceAttributeComputeCapabilityMinor;
using ::hipDeviceAttributeComputeMode;
using ::hipDeviceAttributeComputePreemptionSupported;
using ::hipDeviceAttributeConcurrentKernels;
using ::hipDeviceAttributeConcurrentManagedAccess;
using ::hipDeviceAttributeCooperativeLaunch;
using ::hipDeviceAttributeCooperativeMultiDeviceLaunch;
using ::hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim;
using ::hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc;
using ::hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim;
using ::hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem;
using ::hipDeviceAttributeCudaCompatibleBegin;
using ::hipDeviceAttributeCudaCompatibleEnd;
using ::hipDeviceAttributeDeviceOverlap;
using ::hipDeviceAttributeDirectManagedMemAccessFromHost;
using ::hipDeviceAttributeEccEnabled;
using ::hipDeviceAttributeFineGrainSupport;
using ::hipDeviceAttributeGlobalL1CacheSupported;
using ::hipDeviceAttributeHdpMemFlushCntl;
using ::hipDeviceAttributeHdpRegFlushCntl;
using ::hipDeviceAttributeHostNativeAtomicSupported;
using ::hipDeviceAttributeHostNumaId;
using ::hipDeviceAttributeHostRegisterSupported;
using ::hipDeviceAttributeImageSupport;
using ::hipDeviceAttributeIntegrated;
using ::hipDeviceAttributeIsLargeBar;
using ::hipDeviceAttributeIsMultiGpuBoard;
using ::hipDeviceAttributeKernelExecTimeout;
using ::hipDeviceAttributeL2CacheSize;
using ::hipDeviceAttributeLocalL1CacheSupported;
using ::hipDeviceAttributeLuid;
using ::hipDeviceAttributeLuidDeviceNodeMask;
using ::hipDeviceAttributeManagedMemory;
using ::hipDeviceAttributeMaxAvailableVgprsPerThread;
using ::hipDeviceAttributeMaxBlockDimX;
using ::hipDeviceAttributeMaxBlockDimY;
using ::hipDeviceAttributeMaxBlockDimZ;
using ::hipDeviceAttributeMaxBlocksPerMultiProcessor;
using ::hipDeviceAttributeMaxGridDimX;
using ::hipDeviceAttributeMaxGridDimY;
using ::hipDeviceAttributeMaxGridDimZ;
using ::hipDeviceAttributeMaxPitch;
using ::hipDeviceAttributeMaxRegistersPerBlock;
using ::hipDeviceAttributeMaxRegistersPerMultiprocessor;
using ::hipDeviceAttributeMaxSharedMemoryPerBlock;
using ::hipDeviceAttributeMaxSharedMemoryPerMultiprocessor;
using ::hipDeviceAttributeMaxSurface1D;
using ::hipDeviceAttributeMaxSurface1DLayered;
using ::hipDeviceAttributeMaxSurface2D;
using ::hipDeviceAttributeMaxSurface2DLayered;
using ::hipDeviceAttributeMaxSurface3D;
using ::hipDeviceAttributeMaxSurfaceCubemap;
using ::hipDeviceAttributeMaxSurfaceCubemapLayered;
using ::hipDeviceAttributeMaxTexture1DLayered;
using ::hipDeviceAttributeMaxTexture1DLinear;
using ::hipDeviceAttributeMaxTexture1DMipmap;
using ::hipDeviceAttributeMaxTexture1DWidth;
using ::hipDeviceAttributeMaxTexture2DGather;
using ::hipDeviceAttributeMaxTexture2DHeight;
using ::hipDeviceAttributeMaxTexture2DLayered;
using ::hipDeviceAttributeMaxTexture2DLinear;
using ::hipDeviceAttributeMaxTexture2DMipmap;
using ::hipDeviceAttributeMaxTexture2DWidth;
using ::hipDeviceAttributeMaxTexture3DAlt;
using ::hipDeviceAttributeMaxTexture3DDepth;
using ::hipDeviceAttributeMaxTexture3DHeight;
using ::hipDeviceAttributeMaxTexture3DWidth;
using ::hipDeviceAttributeMaxTextureCubemap;
using ::hipDeviceAttributeMaxTextureCubemapLayered;
using ::hipDeviceAttributeMaxThreadsDim;
using ::hipDeviceAttributeMaxThreadsPerBlock;
using ::hipDeviceAttributeMaxThreadsPerMultiProcessor;
using ::hipDeviceAttributeMemoryBusWidth;
using ::hipDeviceAttributeMemoryClockRate;
using ::hipDeviceAttributeMemoryPoolsSupported;
using ::hipDeviceAttributeMemoryPoolSupportedHandleTypes;
using ::hipDeviceAttributeMultiGpuBoardGroupID;
using ::hipDeviceAttributeMultiprocessorCount;
using ::hipDeviceAttributeNumberOfXccs;
using ::hipDeviceAttributePageableMemoryAccess;
using ::hipDeviceAttributePageableMemoryAccessUsesHostPageTables;
using ::hipDeviceAttributePciBusId;
using ::hipDeviceAttributePciChipId;
using ::hipDeviceAttributePciDeviceId;
using ::hipDeviceAttributePciDomainId;
using ::hipDeviceAttributePciDomainID;
using ::hipDeviceAttributePersistingL2CacheMaxSize;
using ::hipDeviceAttributePhysicalMultiProcessorCount;
using ::hipDeviceAttributeReservedSharedMemPerBlock;
using ::hipDeviceAttributeSharedMemPerBlockOptin;
using ::hipDeviceAttributeSharedMemPerMultiprocessor;
using ::hipDeviceAttributeSingleToDoublePrecisionPerfRatio;
using ::hipDeviceAttributeStreamPrioritiesSupported;
using ::hipDeviceAttributeSurfaceAlignment;
using ::hipDeviceAttributeTccDriver;
using ::hipDeviceAttributeTextureAlignment;
using ::hipDeviceAttributeTexturePitchAlignment;
using ::hipDeviceAttributeTotalConstantMemory;
using ::hipDeviceAttributeTotalGlobalMem;
using ::hipDeviceAttributeUnifiedAddressing;
using ::hipDeviceAttributeUnused1;
using ::hipDeviceAttributeUnused2;
using ::hipDeviceAttributeUnused3;
using ::hipDeviceAttributeUnused4;
using ::hipDeviceAttributeUnused5;
using ::hipDeviceAttributeVendorSpecificBegin;
using ::hipDeviceAttributeVirtualMemoryManagementSupported;
using ::hipDeviceAttributeWallClockRate;
using ::hipDeviceAttributeWarpSize;

// hipDriverProcAddressQueryResult
using ::HIP_GET_PROC_ADDRESS_SUCCESS;
using ::HIP_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
using ::HIP_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT;

// hipComputeMode
using ::hipComputeModeDefault;
using ::hipComputeModeExclusive;
using ::hipComputeModeExclusiveProcess;
using ::hipComputeModeProhibited;

// hipFlushGPUDirectRDMAWritesOptions
using ::hipFlushGPUDirectRDMAWritesOptionHost;
using ::hipFlushGPUDirectRDMAWritesOptionMemOps;

// hipGPUDirectRDMAWritesOrdering
using ::hipGPUDirectRDMAWritesOrderingAllDevices;
using ::hipGPUDirectRDMAWritesOrderingNone;
using ::hipGPUDirectRDMAWritesOrderingOwner;

// hipDeviceP2PAttr
using ::hipDevP2PAttrAccessSupported;
using ::hipDevP2PAttrHipArrayAccessSupported;
using ::hipDevP2PAttrNativeAtomicSupported;
using ::hipDevP2PAttrPerformanceRank;

// hipDriverEntryPointQueryResult
using ::hipDriverEntryPointSuccess;
using ::hipDriverEntryPointSymbolNotFound;
using ::hipDriverEntryPointVersionNotSufficent;

// hipLimit_t
using ::hipExtLimitScratchCurrent;
using ::hipExtLimitScratchMax;
using ::hipExtLimitScratchMin;
using ::hipLimit_t;
using ::hipLimitMallocHeapSize;
using ::hipLimitPrintfFifoSize;
using ::hipLimitRange;
using ::hipLimitStackSize;

// hipStreamBatchMemOpType
using ::hipStreamMemOpBarrier;
using ::hipStreamMemOpFlushRemoteWrites;
using ::hipStreamMemOpWaitValue32;
using ::hipStreamMemOpWaitValue64;
using ::hipStreamMemOpWriteValue32;
using ::hipStreamMemOpWriteValue64;

// hipMemoryAdvise
using ::hipMemAdviseSetAccessedBy;
using ::hipMemAdviseSetCoarseGrain;
using ::hipMemAdviseSetPreferredLocation;
using ::hipMemAdviseSetReadMostly;
using ::hipMemAdviseUnsetAccessedBy;
using ::hipMemAdviseUnsetCoarseGrain;
using ::hipMemAdviseUnsetPreferredLocation;
using ::hipMemAdviseUnsetReadMostly;

// hipMemRangeCoherencyMode
using ::hipMemRangeCoherencyModeCoarseGrain;
using ::hipMemRangeCoherencyModeFineGrain;
using ::hipMemRangeCoherencyModeIndeterminate;

// hipMemRangeAttribute
using ::hipMemRangeAttributeAccessedBy;
using ::hipMemRangeAttributeCoherencyMode;
using ::hipMemRangeAttributeLastPrefetchLocation;
using ::hipMemRangeAttributePreferredLocation;
using ::hipMemRangeAttributeReadMostly;

// hipMemPoolAttr
using ::hipMemPoolAttrReleaseThreshold;
using ::hipMemPoolAttrReservedMemCurrent;
using ::hipMemPoolAttrReservedMemHigh;
using ::hipMemPoolAttrUsedMemCurrent;
using ::hipMemPoolAttrUsedMemHigh;
using ::hipMemPoolReuseAllowInternalDependencies;
using ::hipMemPoolReuseAllowOpportunistic;
using ::hipMemPoolReuseFollowEventDependencies;

// hipMemAccessFlags
using ::hipMemAccessFlagsProtNone;
using ::hipMemAccessFlagsProtRead;
using ::hipMemAccessFlagsProtReadWrite;

// hipMemAllocationType
using ::hipMemAllocationTypeInvalid;
using ::hipMemAllocationTypeMax;
using ::hipMemAllocationTypePinned;
using ::hipMemAllocationTypeUncached;

// hipMemAllocationHandleType
using ::hipMemHandleTypeNone;
using ::hipMemHandleTypePosixFileDescriptor;
using ::hipMemHandleTypeWin32;
using ::hipMemHandleTypeWin32Kmt;

// hipFuncAttribute
using ::hipFuncAttributeMax;
using ::hipFuncAttributeMaxDynamicSharedMemorySize;
using ::hipFuncAttributePreferredSharedMemoryCarveout;

// hipFuncCache_t
using ::hipFuncCachePreferEqual;
using ::hipFuncCachePreferL1;
using ::hipFuncCachePreferNone;
using ::hipFuncCachePreferShared;

// hipSharedMemConfig
using ::hipSharedMemBankSizeDefault;
using ::hipSharedMemBankSizeEightByte;
using ::hipSharedMemBankSizeFourByte;

// hipExternalMemoryHandleType
using ::hipExternalMemoryHandleTypeD3D11Resource;
using ::hipExternalMemoryHandleTypeD3D11ResourceKmt;
using ::hipExternalMemoryHandleTypeD3D12Heap;
using ::hipExternalMemoryHandleTypeD3D12Resource;
using ::hipExternalMemoryHandleTypeNvSciBuf;
using ::hipExternalMemoryHandleTypeOpaqueFd;
using ::hipExternalMemoryHandleTypeOpaqueWin32;
using ::hipExternalMemoryHandleTypeOpaqueWin32Kmt;

// hipExternalSemaphoreHandleType
using ::hipExternalSemaphoreHandleTypeD3D11Fence;
using ::hipExternalSemaphoreHandleTypeD3D12Fence;
using ::hipExternalSemaphoreHandleTypeKeyedMutex;
using ::hipExternalSemaphoreHandleTypeKeyedMutexKmt;
using ::hipExternalSemaphoreHandleTypeNvSciSync;
using ::hipExternalSemaphoreHandleTypeOpaqueFd;
using ::hipExternalSemaphoreHandleTypeOpaqueWin32;
using ::hipExternalSemaphoreHandleTypeOpaqueWin32Kmt;
using ::hipExternalSemaphoreHandleTypeTimelineSemaphoreFd;
using ::hipExternalSemaphoreHandleTypeTimelineSemaphoreWin32;

// hipGraphicsRegisterFlags
using ::hipGraphicsRegisterFlagsNone;
using ::hipGraphicsRegisterFlagsReadOnly;
using ::hipGraphicsRegisterFlagsSurfaceLoadStore;
using ::hipGraphicsRegisterFlagsTextureGather;
using ::hipGraphicsRegisterFlagsWriteDiscard;

// hipGraphNodeType
using ::hipGraphNodeTypeBatchMemOp;
using ::hipGraphNodeTypeCount;
using ::hipGraphNodeTypeEmpty;
using ::hipGraphNodeTypeEventRecord;
using ::hipGraphNodeTypeExtSemaphoreSignal;
using ::hipGraphNodeTypeExtSemaphoreWait;
using ::hipGraphNodeTypeGraph;
using ::hipGraphNodeTypeHost;
using ::hipGraphNodeTypeKernel;
using ::hipGraphNodeTypeMemAlloc;
using ::hipGraphNodeTypeMemcpy;
using ::hipGraphNodeTypeMemcpyFromSymbol;
using ::hipGraphNodeTypeMemcpyToSymbol;
using ::hipGraphNodeTypeMemFree;
using ::hipGraphNodeTypeMemset;
using ::hipGraphNodeTypeWaitEvent;

// hipAccessProperty
using ::hipAccessPropertyNormal;
using ::hipAccessPropertyPersisting;
using ::hipAccessPropertyStreaming;

// hipLaunchMemSyncDomain
using ::hipLaunchMemSyncDomainDefault;
using ::hipLaunchMemSyncDomainRemote;

// hipSynchronizationPolicy
using ::hipSyncPolicyAuto;
using ::hipSyncPolicyBlockingSync;
using ::hipSyncPolicySpin;
using ::hipSyncPolicyYield;

// hipLaunchAttributeID
using ::hipLaunchAttributeAccessPolicyWindow;
using ::hipLaunchAttributeCooperative;
using ::hipLaunchAttributeMax;
using ::hipLaunchAttributeMemSyncDomain;
using ::hipLaunchAttributeMemSyncDomainMap;
using ::hipLaunchAttributePriority;
using ::hipLaunchAttributeSynchronizationPolicy;

// hipGraphExecUpdateResult
using ::hipGraphExecUpdateError;
using ::hipGraphExecUpdateErrorFunctionChanged;
using ::hipGraphExecUpdateErrorNodeTypeChanged;
using ::hipGraphExecUpdateErrorNotSupported;
using ::hipGraphExecUpdateErrorParametersChanged;
using ::hipGraphExecUpdateErrorTopologyChanged;
using ::hipGraphExecUpdateErrorUnsupportedFunctionChange;
using ::hipGraphExecUpdateSuccess;

// hipStreamCaptureMode
using ::hipStreamCaptureModeGlobal;
using ::hipStreamCaptureModeRelaxed;
using ::hipStreamCaptureModeThreadLocal;

// hipStreamCaptureStatus
using ::hipStreamCaptureStatusActive;
using ::hipStreamCaptureStatusInvalidated;
using ::hipStreamCaptureStatusNone;

// hipStreamUpdateCaptureDependenciesFlags
using ::hipStreamAddCaptureDependencies;
using ::hipStreamSetCaptureDependencies;

// hipGraphMemAttributeType
using ::hipGraphMemAttrReservedMemCurrent;
using ::hipGraphMemAttrReservedMemHigh;
using ::hipGraphMemAttrUsedMemCurrent;
using ::hipGraphMemAttrUsedMemHigh;

// hipUserObjectFlags
using ::hipUserObjectNoDestructorSync;

// hipUserObjectRetainFlags
using ::hipGraphUserObjectMove;

// hipGraphInstantiateFlags
using ::hipGraphInstantiateFlagAutoFreeOnLaunch;
using ::hipGraphInstantiateFlagDeviceLaunch;
using ::hipGraphInstantiateFlagUpload;
using ::hipGraphInstantiateFlagUseNodePriority;

// hipGraphDebugDotFlags
using ::hipGraphDebugDotFlagsEventNodeParams;
using ::hipGraphDebugDotFlagsExtSemasSignalNodeParams;
using ::hipGraphDebugDotFlagsExtSemasWaitNodeParams;
using ::hipGraphDebugDotFlagsHandles;
using ::hipGraphDebugDotFlagsHostNodeParams;
using ::hipGraphDebugDotFlagsKernelNodeAttributes;
using ::hipGraphDebugDotFlagsKernelNodeParams;
using ::hipGraphDebugDotFlagsMemcpyNodeParams;
using ::hipGraphDebugDotFlagsMemsetNodeParams;
using ::hipGraphDebugDotFlagsVerbose;

// hipGraphInstantiateResult
using ::hipGraphInstantiateError;
using ::hipGraphInstantiateInvalidStructure;
using ::hipGraphInstantiateMultipleDevicesNotSupported;
using ::hipGraphInstantiateNodeOperationNotSupported;
using ::hipGraphInstantiateSuccess;

// hipMemAllocationGranularity_flags
using ::hipMemAllocationGranularityMinimum;
using ::hipMemAllocationGranularityRecommended;

// hipMemHandleType
using ::hipMemHandleTypeGeneric;

// hipMemOperationType
using ::hipMemOperationTypeMap;
using ::hipMemOperationTypeUnmap;

// hipArraySparseSubresourceType
using ::hipArraySparseSubresourceTypeMiptail;
using ::hipArraySparseSubresourceTypeSparseLevel;

// hipGraphDependencyType
using ::hipGraphDependencyTypeDefault;
using ::hipGraphDependencyTypeProgrammatic;

// hipMemRangeHandleType
using ::hipMemRangeHandleTypeDmaBufFd;
using ::hipMemRangeHandleTypeMax;

// hipMemRangeFlags
using ::hipMemRangeFlagDmaBufMappingTypePcie;
using ::hipMemRangeFlagsMax;

// hipChannelFormatKind
using ::hipChannelFormatKindFloat;
using ::hipChannelFormatKindNone;
using ::hipChannelFormatKindSigned;
using ::hipChannelFormatKindUnsigned;

// hipArray_Format
using ::HIP_AD_FORMAT_FLOAT;
using ::HIP_AD_FORMAT_HALF;
using ::HIP_AD_FORMAT_SIGNED_INT16;
using ::HIP_AD_FORMAT_SIGNED_INT32;
using ::HIP_AD_FORMAT_SIGNED_INT8;
using ::HIP_AD_FORMAT_UNSIGNED_INT16;
using ::HIP_AD_FORMAT_UNSIGNED_INT32;
using ::HIP_AD_FORMAT_UNSIGNED_INT8;

// hipResourceType
using ::hipResourceTypeArray;
using ::hipResourceTypeLinear;
using ::hipResourceTypeMipmappedArray;
using ::hipResourceTypePitch2D;

// HIPresourcetype
using ::HIP_RESOURCE_TYPE_ARRAY;
using ::HIP_RESOURCE_TYPE_LINEAR;
using ::HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY;
using ::HIP_RESOURCE_TYPE_PITCH2D;

// HIPaddress_mode
using ::HIP_TR_ADDRESS_MODE_BORDER;
using ::HIP_TR_ADDRESS_MODE_CLAMP;
using ::HIP_TR_ADDRESS_MODE_MIRROR;
using ::HIP_TR_ADDRESS_MODE_WRAP;

// HIPfilter_mode
using ::HIP_TR_FILTER_MODE_LINEAR;
using ::HIP_TR_FILTER_MODE_POINT;

// hipResourceViewFormat
using ::hipResViewFormatFloat1;
using ::hipResViewFormatFloat2;
using ::hipResViewFormatFloat4;
using ::hipResViewFormatHalf1;
using ::hipResViewFormatHalf2;
using ::hipResViewFormatHalf4;
using ::hipResViewFormatNone;
using ::hipResViewFormatSignedBlockCompressed4;
using ::hipResViewFormatSignedBlockCompressed5;
using ::hipResViewFormatSignedBlockCompressed6H;
using ::hipResViewFormatSignedChar1;
using ::hipResViewFormatSignedChar2;
using ::hipResViewFormatSignedChar4;
using ::hipResViewFormatSignedInt1;
using ::hipResViewFormatSignedInt2;
using ::hipResViewFormatSignedInt4;
using ::hipResViewFormatSignedShort1;
using ::hipResViewFormatSignedShort2;
using ::hipResViewFormatSignedShort4;
using ::hipResViewFormatUnsignedBlockCompressed1;
using ::hipResViewFormatUnsignedBlockCompressed2;
using ::hipResViewFormatUnsignedBlockCompressed3;
using ::hipResViewFormatUnsignedBlockCompressed4;
using ::hipResViewFormatUnsignedBlockCompressed5;
using ::hipResViewFormatUnsignedBlockCompressed6H;
using ::hipResViewFormatUnsignedBlockCompressed7;
using ::hipResViewFormatUnsignedChar1;
using ::hipResViewFormatUnsignedChar2;
using ::hipResViewFormatUnsignedChar4;
using ::hipResViewFormatUnsignedInt1;
using ::hipResViewFormatUnsignedInt2;
using ::hipResViewFormatUnsignedInt4;
using ::hipResViewFormatUnsignedShort1;
using ::hipResViewFormatUnsignedShort2;
using ::hipResViewFormatUnsignedShort4;

// HIPresourceViewFormat
using ::HIP_RES_VIEW_FORMAT_FLOAT_1X16;
using ::HIP_RES_VIEW_FORMAT_FLOAT_1X32;
using ::HIP_RES_VIEW_FORMAT_FLOAT_2X16;
using ::HIP_RES_VIEW_FORMAT_FLOAT_2X32;
using ::HIP_RES_VIEW_FORMAT_FLOAT_4X16;
using ::HIP_RES_VIEW_FORMAT_FLOAT_4X32;
using ::HIP_RES_VIEW_FORMAT_NONE;
using ::HIP_RES_VIEW_FORMAT_SIGNED_BC4;
using ::HIP_RES_VIEW_FORMAT_SIGNED_BC5;
using ::HIP_RES_VIEW_FORMAT_SIGNED_BC6H;
using ::HIP_RES_VIEW_FORMAT_SINT_1X16;
using ::HIP_RES_VIEW_FORMAT_SINT_1X32;
using ::HIP_RES_VIEW_FORMAT_SINT_1X8;
using ::HIP_RES_VIEW_FORMAT_SINT_2X16;
using ::HIP_RES_VIEW_FORMAT_SINT_2X32;
using ::HIP_RES_VIEW_FORMAT_SINT_2X8;
using ::HIP_RES_VIEW_FORMAT_SINT_4X16;
using ::HIP_RES_VIEW_FORMAT_SINT_4X32;
using ::HIP_RES_VIEW_FORMAT_SINT_4X8;
using ::HIP_RES_VIEW_FORMAT_UINT_1X16;
using ::HIP_RES_VIEW_FORMAT_UINT_1X32;
using ::HIP_RES_VIEW_FORMAT_UINT_1X8;
using ::HIP_RES_VIEW_FORMAT_UINT_2X16;
using ::HIP_RES_VIEW_FORMAT_UINT_2X32;
using ::HIP_RES_VIEW_FORMAT_UINT_2X8;
using ::HIP_RES_VIEW_FORMAT_UINT_4X16;
using ::HIP_RES_VIEW_FORMAT_UINT_4X32;
using ::HIP_RES_VIEW_FORMAT_UINT_4X8;
using ::HIP_RES_VIEW_FORMAT_UNSIGNED_BC1;
using ::HIP_RES_VIEW_FORMAT_UNSIGNED_BC2;
using ::HIP_RES_VIEW_FORMAT_UNSIGNED_BC3;
using ::HIP_RES_VIEW_FORMAT_UNSIGNED_BC4;
using ::HIP_RES_VIEW_FORMAT_UNSIGNED_BC5;
using ::HIP_RES_VIEW_FORMAT_UNSIGNED_BC6H;
using ::HIP_RES_VIEW_FORMAT_UNSIGNED_BC7;

// hipMemcpyKind
using ::hipMemcpyDefault;
using ::hipMemcpyDeviceToDevice;
using ::hipMemcpyDeviceToDeviceNoCU;
using ::hipMemcpyDeviceToHost;
using ::hipMemcpyHostToDevice;
using ::hipMemcpyHostToHost;

// hipMemLocationType
using ::hipMemLocationTypeDevice;
using ::hipMemLocationTypeHost;
using ::hipMemLocationTypeHostNuma;
using ::hipMemLocationTypeHostNumaCurrent;
using ::hipMemLocationTypeInvalid;
using ::hipMemLocationTypeNone;

// hipMemcpyFlags
using ::hipMemcpyFlagDefault;
using ::hipMemcpyFlagPreferOverlapWithCompute;

// hipMemcpySrcAccessOrder
using ::hipMemcpySrcAccessOrderAny;
using ::hipMemcpySrcAccessOrderDuringApiCall;
using ::hipMemcpySrcAccessOrderInvalid;
using ::hipMemcpySrcAccessOrderMax;
using ::hipMemcpySrcAccessOrderStream;

// hipMemcpy3DOperandType
using ::hipMemcpyOperandTypeArray;
using ::hipMemcpyOperandTypeMax;
using ::hipMemcpyOperandTypePointer;

// hipFunction_attribute
using ::HIP_FUNC_ATTRIBUTE_BINARY_VERSION;
using ::HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA;
using ::HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES;
using ::HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES;
using ::HIP_FUNC_ATTRIBUTE_MAX;
using ::HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES;
using ::HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK;
using ::HIP_FUNC_ATTRIBUTE_NUM_REGS;
using ::HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT;
using ::HIP_FUNC_ATTRIBUTE_PTX_VERSION;
using ::HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES;

// hipPointer_attribute
using ::HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS;
using ::HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES;
using ::HIP_POINTER_ATTRIBUTE_BUFFER_ID;
using ::HIP_POINTER_ATTRIBUTE_CONTEXT;
using ::HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL;
using ::HIP_POINTER_ATTRIBUTE_DEVICE_POINTER;
using ::HIP_POINTER_ATTRIBUTE_HOST_POINTER;
using ::HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE;
using ::HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE;
using ::HIP_POINTER_ATTRIBUTE_IS_MANAGED;
using ::HIP_POINTER_ATTRIBUTE_MAPPED;
using ::HIP_POINTER_ATTRIBUTE_MEMORY_TYPE;
using ::HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE;
using ::HIP_POINTER_ATTRIBUTE_P2P_TOKENS;
using ::HIP_POINTER_ATTRIBUTE_RANGE_SIZE;
using ::HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR;
using ::HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS;

// hipTextureAddressMode
using ::hipAddressModeBorder;
using ::hipAddressModeClamp;
using ::hipAddressModeMirror;
using ::hipAddressModeWrap;

// hipTextureFilterMode
using ::hipFilterModeLinear;
using ::hipFilterModePoint;

// hipTextureReadMode
using ::hipReadModeElementType;
using ::hipReadModeNormalizedFloat;

// hipSurfaceBoundaryMode
using ::hipBoundaryModeClamp;
using ::hipBoundaryModeTrap;
using ::hipBoundaryModeZero;

// ========================================================================
// Functions
// ========================================================================

// ------------------------------------------------------------------------
// Error Handling
// ------------------------------------------------------------------------
using ::hipDrvGetErrorName;
using ::hipDrvGetErrorString;
using ::hipExtGetLastError;
using ::hipGetErrorName;
using ::hipGetErrorString;
using ::hipGetLastError;
using ::hipPeekAtLastError;

// ------------------------------------------------------------------------
// Initialization, Version, and Driver Context Management
// ------------------------------------------------------------------------
using ::hipCtxCreate;
using ::hipCtxDestroy;
using ::hipCtxDisablePeerAccess;
using ::hipCtxEnablePeerAccess;
using ::hipCtxGetApiVersion;
using ::hipCtxGetCacheConfig;
using ::hipCtxGetCurrent;
using ::hipCtxGetDevice;
using ::hipCtxGetFlags;
using ::hipCtxGetSharedMemConfig;
using ::hipCtxPopCurrent;
using ::hipCtxPushCurrent;
using ::hipCtxSetCacheConfig;
using ::hipCtxSetCurrent;
using ::hipCtxSetSharedMemConfig;
using ::hipCtxSynchronize;
using ::hipDevicePrimaryCtxGetState;
using ::hipDevicePrimaryCtxRelease;
using ::hipDevicePrimaryCtxReset;
using ::hipDevicePrimaryCtxRetain;
using ::hipDevicePrimaryCtxSetFlags;
using ::hipDriverGetVersion;
using ::hipInit;
using ::hipRuntimeGetVersion;
using ::hipSetValidDevices;

// ------------------------------------------------------------------------
// Device Management
// ------------------------------------------------------------------------
using ::hipChooseDevice; // preprocessor-renamed by hip_runtime_api.h to hipChooseDeviceR0600
using ::hipDeviceCanAccessPeer;
using ::hipDeviceComputeCapability;
using ::hipDeviceDisablePeerAccess;
using ::hipDeviceEnablePeerAccess;
using ::hipDeviceGet;
using ::hipDeviceGetAttribute;
using ::hipDeviceGetByPCIBusId;
using ::hipDeviceGetCacheConfig;
using ::hipDeviceGetDefaultMemPool;
using ::hipDeviceGetGraphMemAttribute;
using ::hipDeviceGetLimit;
using ::hipDeviceGetMemPool;
using ::hipDeviceGetName;
using ::hipDeviceGetP2PAttribute;
using ::hipDeviceGetPCIBusId;
using ::hipDeviceGetSharedMemConfig;
using ::hipDeviceGetStreamPriorityRange;
using ::hipDeviceGetTexture1DLinearMaxWidth;
using ::hipDeviceGetUuid;
using ::hipDeviceGraphMemTrim;
using ::hipDeviceReset;
using ::hipDeviceSetCacheConfig;
using ::hipDeviceSetGraphMemAttribute;
using ::hipDeviceSetLimit;
using ::hipDeviceSetMemPool;
using ::hipDeviceSetSharedMemConfig;
using ::hipDeviceSynchronize;
using ::hipDeviceTotalMem;
using ::hipExtGetLinkTypeAndHopCount;
using ::hipGetDevice;
using ::hipGetDeviceCount;
using ::hipGetDeviceFlags;
using ::
    hipGetDeviceProperties; // preprocessor-renamed by hip_runtime_api.h to hipGetDevicePropertiesR0600
using ::hipSetDevice;
using ::hipSetDeviceFlags;

// ------------------------------------------------------------------------
// Memory Management — Allocation / Free
// ------------------------------------------------------------------------
using ::hipMalloc;
using ::hipMalloc3D;
using ::hipMalloc3DArray;
using ::hipMallocArray;
using ::hipMallocManaged;
using ::hipMallocMipmappedArray;
using ::hipMallocPitch;
inline hipError_t hipMallocAsync(void **dev_ptr, size_t size, hipStream_t stream) {
  return ::hipMallocAsync(dev_ptr, size, stream);
}
inline hipError_t hipMallocFromPoolAsync(void **dev_ptr, size_t size, hipMemPool_t mem_pool,
                                         hipStream_t stream) {
  return ::hipMallocFromPoolAsync(dev_ptr, size, mem_pool, stream);
}
using ::hipExtMallocWithFlags;
using ::hipFree;
using ::hipFreeArray;
using ::hipFreeAsync;
using ::hipFreeHost;
using ::hipFreeMipmappedArray;
using ::hipHostAlloc;
using ::hipHostFree;
using ::hipHostGetDevicePointer;
using ::hipHostGetFlags;
using ::hipHostMalloc;
using ::hipHostRegister;
using ::hipHostUnregister;
using ::hipMallocHost;
using ::hipMemAllocHost;
using ::hipMemAllocPitch;

// ------------------------------------------------------------------------
// Memory Management — Copy
// ------------------------------------------------------------------------
using ::hipMemcpy;
using ::hipMemcpy2D;
using ::hipMemcpy2DArrayToArray;
using ::hipMemcpy2DAsync;
using ::hipMemcpy2DFromArray;
using ::hipMemcpy2DFromArrayAsync;
using ::hipMemcpy2DToArray;
using ::hipMemcpy2DToArrayAsync;
using ::hipMemcpy3D;
using ::hipMemcpy3DAsync;
using ::hipMemcpy3DBatchAsync;
using ::hipMemcpy3DPeer;
using ::hipMemcpy3DPeerAsync;
using ::hipMemcpyAsync;
using ::hipMemcpyFromArray;
using ::hipMemcpyPeer;
using ::hipMemcpyPeerAsync;
using ::hipMemcpyToArray;
inline hipError_t hipMemcpyToSymbol(const void *symbol, const void *src, size_t sizeBytes,
                                    size_t offset = 0,
                                    hipMemcpyKind kind = ::hipMemcpyHostToDevice) {
  return ::hipMemcpyToSymbol(symbol, src, sizeBytes, offset, kind);
}
inline hipError_t hipMemcpyToSymbolAsync(const void *symbol, const void *src, size_t sizeBytes,
                                         size_t offset, hipMemcpyKind kind,
                                         hipStream_t stream = nullptr) {
  return ::hipMemcpyToSymbolAsync(symbol, src, sizeBytes, offset, kind, stream);
}
inline hipError_t hipMemcpyFromSymbol(void *dst, const void *symbol, size_t sizeBytes,
                                      size_t offset = 0,
                                      hipMemcpyKind kind = ::hipMemcpyDeviceToHost) {
  return ::hipMemcpyFromSymbol(dst, symbol, sizeBytes, offset, kind);
}
inline hipError_t hipMemcpyFromSymbolAsync(void *dst, const void *symbol, size_t sizeBytes,
                                           size_t offset, hipMemcpyKind kind,
                                           hipStream_t stream = nullptr) {
  return ::hipMemcpyFromSymbolAsync(dst, symbol, sizeBytes, offset, kind, stream);
}
using ::hipDrvMemcpy2DUnaligned;
using ::hipDrvMemcpy3D;
using ::hipDrvMemcpy3DAsync;
using ::hipMemcpyAtoA;
using ::hipMemcpyAtoD;
using ::hipMemcpyAtoH;
using ::hipMemcpyAtoHAsync;
using ::hipMemcpyBatchAsync;
using ::hipMemcpyDtoA;
using ::hipMemcpyDtoD;
using ::hipMemcpyDtoDAsync;
using ::hipMemcpyDtoH;
using ::hipMemcpyDtoHAsync;
using ::hipMemcpyHtoA;
using ::hipMemcpyHtoAAsync;
using ::hipMemcpyHtoD;
using ::hipMemcpyHtoDAsync;
using ::hipMemcpyParam2D;
using ::hipMemcpyParam2DAsync;
using ::hipMemcpyWithStream;

// ------------------------------------------------------------------------
// Memory Management — Set
// ------------------------------------------------------------------------
using ::hipMemset;
using ::hipMemset2D;
using ::hipMemset2DAsync;
using ::hipMemset3D;
using ::hipMemset3DAsync;
using ::hipMemsetAsync;
using ::hipMemsetD16;
using ::hipMemsetD16Async;
using ::hipMemsetD2D16;
using ::hipMemsetD2D16Async;
using ::hipMemsetD2D32;
using ::hipMemsetD2D32Async;
using ::hipMemsetD2D8;
using ::hipMemsetD2D8Async;
using ::hipMemsetD32;
using ::hipMemsetD32Async;
using ::hipMemsetD8;
using ::hipMemsetD8Async;

// ------------------------------------------------------------------------
// Memory Management — Info / Advise / Query / Pointer Attributes
// ------------------------------------------------------------------------
using ::hipDrvPointerGetAttributes;
using ::hipMemAdvise;
using ::hipMemAdvise_v2;
using ::hipMemGetAddressRange;
using ::hipMemGetHandleForAddressRange;
using ::hipMemGetInfo;
using ::hipMemPrefetchAsync;
using ::hipMemPrefetchAsync_v2;
using ::hipMemPtrGetInfo;
using ::hipMemRangeGetAttribute;
using ::hipMemRangeGetAttributes;
using ::hipPointerGetAttribute;
using ::hipPointerGetAttributes;
using ::hipPointerSetAttribute;

// ------------------------------------------------------------------------
// Memory Management — Arrays
// ------------------------------------------------------------------------
using ::hipArray3DCreate;
using ::hipArray3DGetDescriptor;
using ::hipArrayCreate;
using ::hipArrayDestroy;
using ::hipArrayGetDescriptor;
using ::hipArrayGetInfo;
using ::hipGetMipmappedArrayLevel;
using ::hipMipmappedArrayCreate;
using ::hipMipmappedArrayDestroy;
using ::hipMipmappedArrayGetLevel;

// ------------------------------------------------------------------------
// Memory Management — Virtual Memory / Pools
// ------------------------------------------------------------------------
using ::hipMemAddressFree;
using ::hipMemAddressReserve;
using ::hipMemCreate;
using ::hipMemExportToShareableHandle;
using ::hipMemGetAccess;
using ::hipMemGetAllocationGranularity;
using ::hipMemGetAllocationPropertiesFromHandle;
using ::hipMemImportFromShareableHandle;
using ::hipMemMap;
using ::hipMemMapArrayAsync;
using ::hipMemPoolCreate;
using ::hipMemPoolDestroy;
using ::hipMemPoolExportPointer;
using ::hipMemPoolExportToShareableHandle;
using ::hipMemPoolGetAccess;
using ::hipMemPoolGetAttribute;
using ::hipMemPoolImportFromShareableHandle;
using ::hipMemPoolImportPointer;
using ::hipMemPoolSetAccess;
using ::hipMemPoolSetAttribute;
using ::hipMemPoolTrimTo;
using ::hipMemRelease;
using ::hipMemRetainAllocationHandle;
using ::hipMemSetAccess;
using ::hipMemUnmap;

// ------------------------------------------------------------------------
// Memory Management — External / IPC
// ------------------------------------------------------------------------
using ::hipDestroyExternalMemory;
using ::hipExternalMemoryGetMappedBuffer;
using ::hipExternalMemoryGetMappedMipmappedArray;
using ::hipImportExternalMemory;
using ::hipIpcCloseMemHandle;
using ::hipIpcGetEventHandle;
using ::hipIpcGetMemHandle;
using ::hipIpcOpenEventHandle;
using ::hipIpcOpenMemHandle;

// ------------------------------------------------------------------------
// Stream Management
// ------------------------------------------------------------------------
using ::hipExtStreamCreateWithCUMask;
using ::hipExtStreamGetCUMask;
using ::hipGetStreamDeviceId;
using ::hipStreamAddCallback;
using ::hipStreamAttachMemAsync;
using ::hipStreamBatchMemOp;
using ::hipStreamBeginCapture;
using ::hipStreamBeginCaptureToGraph;
using ::hipStreamCopyAttributes;
using ::hipStreamCreate;
using ::hipStreamCreateWithFlags;
using ::hipStreamCreateWithPriority;
using ::hipStreamDestroy;
using ::hipStreamEndCapture;
using ::hipStreamGetAttribute;
using ::hipStreamGetCaptureInfo;
using ::hipStreamGetCaptureInfo_v2;
using ::hipStreamGetDevice;
using ::hipStreamGetFlags;
using ::hipStreamGetId;
using ::hipStreamGetPriority;
using ::hipStreamIsCapturing;
using ::hipStreamQuery;
using ::hipStreamSetAttribute;
using ::hipStreamSynchronize;
using ::hipStreamUpdateCaptureDependencies;
using ::hipStreamWaitEvent;
using ::hipStreamWaitValue32;
using ::hipStreamWaitValue64;
using ::hipStreamWriteValue32;
using ::hipStreamWriteValue64;
using ::hipThreadExchangeStreamCaptureMode;

// ------------------------------------------------------------------------
// Event Management
// ------------------------------------------------------------------------
using ::hipEventCreate;
using ::hipEventCreateWithFlags;
using ::hipEventDestroy;
using ::hipEventElapsedTime;
using ::hipEventQuery;
using ::hipEventRecord;
using ::hipEventRecordWithFlags;
using ::hipEventSynchronize;

// ------------------------------------------------------------------------
// External Semaphores
// ------------------------------------------------------------------------
using ::hipDestroyExternalSemaphore;
using ::hipImportExternalSemaphore;
using ::hipSignalExternalSemaphoresAsync;
using ::hipWaitExternalSemaphoresAsync;

// ------------------------------------------------------------------------
// Execution Control (Kernel Launch)
// ------------------------------------------------------------------------
using ::hipLaunchKernel;
using ::hipLaunchKernelExC;
inline hipError_t hipLaunchCooperativeKernel(const void *f, dim3 gridDim, dim3 blockDimX,
                                             void **kernelParams, unsigned int sharedMemBytes,
                                             hipStream_t stream) {
  return ::hipLaunchCooperativeKernel(f, gridDim, blockDimX, kernelParams, sharedMemBytes, stream);
}
inline hipError_t hipLaunchCooperativeKernelMultiDevice(hipLaunchParams *launchParamsList,
                                                        int numDevices, unsigned int flags) {
  return ::hipLaunchCooperativeKernelMultiDevice(launchParamsList, numDevices, flags);
}
using ::hipConfigureCall;
using ::hipExtLaunchKernel;
using ::hipLaunchByPtr;
using ::hipLaunchHostFunc;
using ::hipSetupArgument;
inline hipError_t hipExtLaunchMultiKernelMultiDevice(hipLaunchParams *launchParamsList,
                                                     int numDevices, unsigned int flags) {
  return ::hipExtLaunchMultiKernelMultiDevice(launchParamsList, numDevices, flags);
}
using ::hipDrvLaunchKernelEx;
using ::hipKernelNameRef;
using ::hipKernelNameRefByPtr;
using ::hipModuleLaunchCooperativeKernel;
using ::hipModuleLaunchCooperativeKernelMultiDevice;
using ::hipModuleLaunchKernel;

// ------------------------------------------------------------------------
// Occupancy and Function Configuration
// ------------------------------------------------------------------------
inline hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, const void *f,
                                                               int blockSize,
                                                               size_t dynSharedMemPerBlk) {
  return ::hipOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, f, blockSize,
                                                        dynSharedMemPerBlk);
}
inline hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int *numBlocks, const void *f, int blockSize, size_t dynSharedMemPerBlk,
    unsigned int flags = 0x00 /* hipOccupancyDefault */) {
  return ::hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, f, blockSize,
                                                                 dynSharedMemPerBlk, flags);
}
inline hipError_t hipOccupancyMaxPotentialBlockSize(int *gridSize, int *blockSize, const void *f,
                                                    size_t dynSharedMemPerBlk, int blockSizeLimit) {
  return ::hipOccupancyMaxPotentialBlockSize(gridSize, blockSize, f, dynSharedMemPerBlk,
                                             blockSizeLimit);
}
inline hipError_t hipOccupancyAvailableDynamicSMemPerBlock(size_t *dynamicSmemSize, const void *f,
                                                           int numBlocks, int blockSize) {
  return ::hipOccupancyAvailableDynamicSMemPerBlock(dynamicSmemSize, f, numBlocks, blockSize);
}
using ::hipFuncGetAttribute;
using ::hipFuncGetAttributes;
using ::hipFuncSetAttribute;
using ::hipFuncSetCacheConfig;
using ::hipFuncSetSharedMemConfig;
using ::hipModuleOccupancyMaxActiveBlocksPerMultiprocessor;
using ::hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags;
using ::hipModuleOccupancyMaxPotentialBlockSize;
using ::hipModuleOccupancyMaxPotentialBlockSizeWithFlags;

// ------------------------------------------------------------------------
// Symbol and Kernel Management
// ------------------------------------------------------------------------
inline hipError_t hipGetSymbolAddress(void **devPtr, const void *symbol) {
  return ::hipGetSymbolAddress(devPtr, symbol);
}
inline hipError_t hipGetSymbolSize(size_t *size, const void *symbol) {
  return ::hipGetSymbolSize(size, symbol);
}
using ::hipGetFuncBySymbol;

// ------------------------------------------------------------------------
// Module / Library / Linker Management
// ------------------------------------------------------------------------
using ::hipApiName;
using ::hipKernelGetLibrary;
using ::hipKernelGetName;
using ::hipLibraryEnumerateKernels;
using ::hipLibraryGetKernel;
using ::hipLibraryGetKernelCount;
using ::hipLibraryLoadData;
using ::hipLibraryLoadFromFile;
using ::hipLibraryUnload;
using ::hipLinkAddData;
using ::hipLinkAddFile;
using ::hipLinkComplete;
using ::hipLinkCreate;
using ::hipLinkDestroy;
using ::hipModuleGetFunction;
using ::hipModuleGetFunctionCount;
using ::hipModuleGetGlobal;
using ::hipModuleGetTexRef;
using ::hipModuleLoad;
using ::hipModuleLoadData;
using ::hipModuleLoadDataEx;
using ::hipModuleLoadFatBinary;
using ::hipModuleUnload;

// ------------------------------------------------------------------------
// Graph Management — Lifecycle / Dependencies
// ------------------------------------------------------------------------
using ::hipGraphAddDependencies;
using ::hipGraphClone;
using ::hipGraphCreate;
using ::hipGraphDebugDotPrint;
using ::hipGraphDestroy;
using ::hipGraphDestroyNode;
using ::hipGraphGetEdges;
using ::hipGraphGetNodes;
using ::hipGraphGetRootNodes;
using ::hipGraphNodeFindInClone;
using ::hipGraphNodeGetDependencies;
using ::hipGraphNodeGetDependentNodes;
using ::hipGraphNodeGetEnabled;
using ::hipGraphNodeGetType;
using ::hipGraphNodeSetEnabled;
using ::hipGraphRemoveDependencies;

// ------------------------------------------------------------------------
// Graph Management — Node Addition
// ------------------------------------------------------------------------
using ::hipDrvGraphAddMemcpyNode;
using ::hipDrvGraphAddMemFreeNode;
using ::hipDrvGraphAddMemsetNode;
using ::hipGraphAddBatchMemOpNode;
using ::hipGraphAddChildGraphNode;
using ::hipGraphAddEmptyNode;
using ::hipGraphAddEventRecordNode;
using ::hipGraphAddEventWaitNode;
using ::hipGraphAddExternalSemaphoresSignalNode;
using ::hipGraphAddExternalSemaphoresWaitNode;
using ::hipGraphAddHostNode;
using ::hipGraphAddKernelNode;
using ::hipGraphAddMemAllocNode;
using ::hipGraphAddMemcpyNode;
using ::hipGraphAddMemcpyNode1D;
using ::hipGraphAddMemcpyNodeFromSymbol;
using ::hipGraphAddMemcpyNodeToSymbol;
using ::hipGraphAddMemFreeNode;
using ::hipGraphAddMemsetNode;
using ::hipGraphAddNode;

// ------------------------------------------------------------------------
// Graph Management — Node Params
// ------------------------------------------------------------------------
using ::hipDrvGraphMemcpyNodeGetParams;
using ::hipDrvGraphMemcpyNodeSetParams;
using ::hipGraphBatchMemOpNodeGetParams;
using ::hipGraphBatchMemOpNodeSetParams;
using ::hipGraphChildGraphNodeGetGraph;
using ::hipGraphEventRecordNodeGetEvent;
using ::hipGraphEventRecordNodeSetEvent;
using ::hipGraphEventWaitNodeGetEvent;
using ::hipGraphEventWaitNodeSetEvent;
using ::hipGraphExternalSemaphoresSignalNodeGetParams;
using ::hipGraphExternalSemaphoresSignalNodeSetParams;
using ::hipGraphExternalSemaphoresWaitNodeGetParams;
using ::hipGraphExternalSemaphoresWaitNodeSetParams;
using ::hipGraphHostNodeGetParams;
using ::hipGraphHostNodeSetParams;
using ::hipGraphKernelNodeCopyAttributes;
using ::hipGraphKernelNodeGetAttribute;
using ::hipGraphKernelNodeGetParams;
using ::hipGraphKernelNodeSetAttribute;
using ::hipGraphKernelNodeSetParams;
using ::hipGraphMemAllocNodeGetParams;
using ::hipGraphMemcpyNodeGetParams;
using ::hipGraphMemcpyNodeSetParams;
using ::hipGraphMemcpyNodeSetParams1D;
using ::hipGraphMemcpyNodeSetParamsFromSymbol;
using ::hipGraphMemcpyNodeSetParamsToSymbol;
using ::hipGraphMemFreeNodeGetParams;
using ::hipGraphMemsetNodeGetParams;
using ::hipGraphMemsetNodeSetParams;
using ::hipGraphNodeSetParams;

// ------------------------------------------------------------------------
// Graph Management — Execution
// ------------------------------------------------------------------------
using ::hipDrvGraphExecMemcpyNodeSetParams;
using ::hipDrvGraphExecMemsetNodeSetParams;
using ::hipGraphExecBatchMemOpNodeSetParams;
using ::hipGraphExecChildGraphNodeSetParams;
using ::hipGraphExecDestroy;
using ::hipGraphExecEventRecordNodeSetEvent;
using ::hipGraphExecEventWaitNodeSetEvent;
using ::hipGraphExecExternalSemaphoresSignalNodeSetParams;
using ::hipGraphExecExternalSemaphoresWaitNodeSetParams;
using ::hipGraphExecGetFlags;
using ::hipGraphExecHostNodeSetParams;
using ::hipGraphExecKernelNodeSetParams;
using ::hipGraphExecMemcpyNodeSetParams;
using ::hipGraphExecMemcpyNodeSetParams1D;
using ::hipGraphExecMemcpyNodeSetParamsFromSymbol;
using ::hipGraphExecMemcpyNodeSetParamsToSymbol;
using ::hipGraphExecMemsetNodeSetParams;
using ::hipGraphExecNodeSetParams;
using ::hipGraphExecUpdate;
using ::hipGraphInstantiate;
using ::hipGraphInstantiateWithFlags;
using ::hipGraphInstantiateWithParams;
using ::hipGraphLaunch;
using ::hipGraphUpload;

// ------------------------------------------------------------------------
// Graph Management — User Objects
// ------------------------------------------------------------------------
using ::hipGraphReleaseUserObject;
using ::hipGraphRetainUserObject;
using ::hipUserObjectCreate;
using ::hipUserObjectRelease;
using ::hipUserObjectRetain;

// ------------------------------------------------------------------------
// Texture / Surface Management
// ------------------------------------------------------------------------
using ::hipCreateSurfaceObject;
using ::hipCreateTextureObject;
using ::hipDestroySurfaceObject;
using ::hipDestroyTextureObject;
using ::hipGetChannelDesc;
using ::hipGetTextureObjectResourceDesc;
using ::hipGetTextureObjectResourceViewDesc;
using ::hipGetTextureObjectTextureDesc;
// NOLINTBEGIN
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
inline hipError_t hipBindTexture(size_t *offset, const textureReference *tex, const void *devPtr,
                                 const hipChannelFormatDesc *desc, size_t size = UINT_MAX) {
  return ::hipBindTexture(offset, tex, devPtr, desc, size);
}
inline hipError_t hipBindTexture2D(size_t *offset, const textureReference *tex, const void *devPtr,
                                   const hipChannelFormatDesc *desc, size_t width, size_t height,
                                   size_t pitch) {
  return ::hipBindTexture2D(offset, tex, devPtr, desc, width, height, pitch);
}
inline hipError_t hipBindTextureToArray(const textureReference *tex, hipArray_const_t array,
                                        const hipChannelFormatDesc *desc) {
  return ::hipBindTextureToArray(tex, array, desc);
}
inline hipError_t hipBindTextureToMipmappedArray(const textureReference *tex,
                                                 hipMipmappedArray_const_t mipmappedArray,
                                                 const hipChannelFormatDesc *desc) {
  return ::hipBindTextureToMipmappedArray(tex, mipmappedArray, desc);
}
inline hipError_t hipUnbindTexture(const textureReference *tex) {
  return ::hipUnbindTexture(tex);
}
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
// NOLINTEND
using ::hipGetTextureAlignmentOffset;
using ::hipGetTextureReference;
using ::hipTexObjectCreate;
using ::hipTexObjectDestroy;
using ::hipTexObjectGetResourceDesc;
using ::hipTexObjectGetResourceViewDesc;
using ::hipTexObjectGetTextureDesc;
using ::hipTexRefGetAddress;
using ::hipTexRefGetAddressMode;
using ::hipTexRefGetArray;
using ::hipTexRefGetBorderColor;
using ::hipTexRefGetFilterMode;
using ::hipTexRefGetFlags;
using ::hipTexRefGetFormat;
using ::hipTexRefGetMaxAnisotropy;
using ::hipTexRefGetMipmapFilterMode;
using ::hipTexRefGetMipmapLevelBias;
using ::hipTexRefGetMipmapLevelClamp;
using ::hipTexRefGetMipMappedArray;
using ::hipTexRefSetAddress;
using ::hipTexRefSetAddress2D;
using ::hipTexRefSetAddressMode;
using ::hipTexRefSetArray;
using ::hipTexRefSetBorderColor;
using ::hipTexRefSetFilterMode;
using ::hipTexRefSetFlags;
using ::hipTexRefSetFormat;
using ::hipTexRefSetMaxAnisotropy;
using ::hipTexRefSetMipmapFilterMode;
using ::hipTexRefSetMipmapLevelBias;
using ::hipTexRefSetMipmapLevelClamp;
using ::hipTexRefSetMipmappedArray;

// ------------------------------------------------------------------------
// Graphics Interoperability
// ------------------------------------------------------------------------
using ::hipGraphicsMapResources;
using ::hipGraphicsResourceGetMappedPointer;
using ::hipGraphicsSubResourceGetMappedArray;
using ::hipGraphicsUnmapResources;
using ::hipGraphicsUnregisterResource;

// ------------------------------------------------------------------------
// Driver Entry Points / Proc Address
// ------------------------------------------------------------------------
using ::hipGetDriverEntryPoint;
using ::hipGetProcAddress;

// ------------------------------------------------------------------------
// Profiler Control
// ------------------------------------------------------------------------
using ::hipProfilerStart;
using ::hipProfilerStop;

} // namespace wwr::hip
