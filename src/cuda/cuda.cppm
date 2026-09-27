/**
 * @file cuda.cppm
 * @brief CUDA Driver API module wrapper for gpumod project
 *
 * This module wraps the CUDA Driver API (cuda.h) to make it accessible
 * from C++20/C++23 module-based code.
 *
 * Usage:
 *   import wwr.cuda.cuda_h;
 *
 * Note: The CUDA Driver API uses low-level "cu" prefixed functions,
 * not the "cuda" prefixed Runtime API.
 */

module;

#include <cuda.h>

export module wwr.cuda.cuda_h;

// Export the most commonly used CUDA driver API functions.
// Many CUDA functions use macro-based versioning in the headers
// (e.g., cuMemAlloc expands to cuMemAlloc_v2). We export the versioned
// names directly to avoid macro-expansion issues in module clients.

// Memory Management (versioned)
export using ::cuMemGetInfo_v2;
export using ::cuMemAlloc_v2;
export using ::cuMemAllocPitch_v2;
export using ::cuMemFree_v2;
export using ::cuMemGetAddressRange_v2;
export using ::cuMemAllocHost_v2;
export using ::cuMemFreeHost;
export using ::cuMemHostAlloc;
export using ::cuMemHostGetDevicePointer_v2;
export using ::cuMemHostGetFlags;
export using ::cuMemHostRegister_v2;
export using ::cuMemHostUnregister;

// Memory copy operations
export using ::cuMemcpy;
export using ::cuMemcpyHtoD_v2;
export using ::cuMemcpyDtoH_v2;
export using ::cuMemcpyDtoD_v2;
export using ::cuMemcpyDtoA_v2;
export using ::cuMemcpyAtoD_v2;
export using ::cuMemcpyHtoA_v2;
export using ::cuMemcpyAtoH_v2;
export using ::cuMemcpyAtoA_v2;
export using ::cuMemcpy2D_v2;
export using ::cuMemcpy2DUnaligned_v2;
export using ::cuMemcpy3D_v2;
export using ::cuMemcpyHtoDAsync_v2;
export using ::cuMemcpyDtoHAsync_v2;
export using ::cuMemcpyDtoDAsync_v2;
export using ::cuMemcpyHtoAAsync_v2;
export using ::cuMemcpyAtoHAsync_v2;
export using ::cuMemcpy2DAsync_v2;
export using ::cuMemcpy3DAsync_v2;
export using ::cuMemcpyAsync;
export using ::cuMemcpyPeer;
export using ::cuMemcpyPeerAsync;
export using ::cuMemcpy3DPeer;
export using ::cuMemcpy3DPeerAsync;

// Memory fill operations
export using ::cuMemsetD8_v2;
export using ::cuMemsetD16_v2;
export using ::cuMemsetD32_v2;
export using ::cuMemsetD2D8_v2;
export using ::cuMemsetD2D16_v2;
export using ::cuMemsetD2D32_v2;
export using ::cuMemsetD8Async;
export using ::cuMemsetD16Async;
export using ::cuMemsetD32Async;
export using ::cuMemsetD2D8Async;
export using ::cuMemsetD2D16Async;
export using ::cuMemsetD2D32Async;

// Array operations
export using ::cuArrayCreate_v2;
export using ::cuArrayGetDescriptor_v2;
export using ::cuArrayGetSparseProperties;
export using ::cuArrayGetMemoryRequirements;
export using ::cuArrayGetPlane;
export using ::cuArrayDestroy;
export using ::cuArray3DCreate_v2;
export using ::cuArray3DGetDescriptor_v2;
export using ::cuMipmappedArrayCreate;
export using ::cuMipmappedArrayGetLevel;
export using ::cuMipmappedArrayGetSparseProperties;
export using ::cuMipmappedArrayGetMemoryRequirements;
export using ::cuMipmappedArrayDestroy;

// Pointer operations
export using ::cuPointerGetAttribute;
export using ::cuPointerGetAttributes;
export using ::cuPointerSetAttribute;

// Managed Memory
export using ::cuMemAllocManaged;
export using ::cuMemAdvise_v2;
export using ::cuMemPrefetchAsync_v2;
export using ::cuMemRangeGetAttribute;
export using ::cuMemRangeGetAttributes;

// Memory Pools
export using ::cuMemAllocAsync;
export using ::cuMemFreeAsync;
export using ::cuMemAllocFromPoolAsync;
export using ::cuMemPoolCreate;
export using ::cuMemPoolDestroy;
export using ::cuMemPoolTrimTo;
export using ::cuMemPoolSetAttribute;
export using ::cuMemPoolGetAttribute;
export using ::cuMemPoolSetAccess;
export using ::cuMemPoolGetAccess;
export using ::cuMemPoolExportToShareableHandle;
export using ::cuMemPoolImportFromShareableHandle;
export using ::cuMemPoolExportPointer;
export using ::cuMemPoolImportPointer;

// Virtual Memory Management
export using ::cuMemAddressReserve;
export using ::cuMemAddressFree;
export using ::cuMemCreate;
export using ::cuMemRelease;
export using ::cuMemMap;
export using ::cuMemUnmap;
export using ::cuMemSetAccess;
export using ::cuMemGetAccess;
export using ::cuMemGetAllocationGranularity;
export using ::cuMemGetAllocationPropertiesFromHandle;
export using ::cuMemRetainAllocationHandle;
export using ::cuMemExportToShareableHandle;
export using ::cuMemImportFromShareableHandle;

// Stream Management
export using ::cuStreamCreate;
export using ::cuStreamCreateWithPriority;
export using ::cuStreamDestroy_v2;
export using ::cuStreamIsCapturing;
export using ::cuStreamQuery;
export using ::cuStreamSynchronize;
export using ::cuStreamAddCallback;
export using ::cuStreamBeginCapture_v2;
export using ::cuStreamEndCapture;
export using ::cuStreamGetCaptureInfo_v3;
export using ::cuStreamUpdateCaptureDependencies_v2;
export using ::cuStreamGetFlags;
export using ::cuStreamGetId;
export using ::cuStreamGetPriority;
export using ::cuStreamWaitEvent;
export using ::cuStreamWaitValue32_v2;
export using ::cuStreamWaitValue64_v2;
export using ::cuStreamWriteValue32_v2;
export using ::cuStreamWriteValue64_v2;
export using ::cuStreamBatchMemOp;

// Event Management
export using ::cuEventCreate;
export using ::cuEventRecord;
export using ::cuEventQuery;
export using ::cuEventSynchronize;
export using ::cuEventDestroy;
export using ::cuEventElapsedTime;

// Execution Control
export using ::cuFuncGetAttribute;
export using ::cuFuncSetAttribute;
export using ::cuFuncSetCacheConfig;
export using ::cuFuncSetSharedMemConfig;
export using ::cuLaunchKernel;
export using ::cuLaunchKernelEx;
export using ::cuOccupancyMaxActiveBlocksPerMultiprocessor;
export using ::cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags;
export using ::cuOccupancyMaxPotentialBlockSize;
export using ::cuOccupancyMaxPotentialBlockSizeWithFlags;
export using ::cuOccupancyAvailableDynamicSMemPerBlock;

// Context Management
export using ::cuCtxCreate_v4;
export using ::cuCtxDestroy;
export using ::cuCtxPopCurrent_v2;
export using ::cuCtxPushCurrent_v2;
export using ::cuCtxGetCurrent;
export using ::cuCtxSetCurrent;
export using ::cuCtxGetDevice;
export using ::cuCtxGetFlags;
export using ::cuCtxSynchronize;
export using ::cuCtxGetLimit;
export using ::cuCtxSetLimit;
export using ::cuCtxGetCacheConfig;
export using ::cuCtxSetCacheConfig;
export using ::cuCtxGetSharedMemConfig;
export using ::cuCtxSetSharedMemConfig;
export using ::cuCtxGetApiVersion;
export using ::cuCtxEnablePeerAccess;
export using ::cuCtxDisablePeerAccess;
export using ::cuCtxGetStreamPriorityRange;
export using ::cuCtxResetPersistingL2Cache;
export using ::cuCtxAttach;
export using ::cuCtxDetach;

// Module Loading
export using ::cuModuleLoad;
export using ::cuModuleLoadData;
export using ::cuModuleLoadDataEx;
export using ::cuModuleLoadFatBinary;
export using ::cuModuleUnload;
export using ::cuModuleGetFunction;
export using ::cuModuleGetGlobal;
export using ::cuModuleGetTexRef;
export using ::cuModuleGetSurfRef;
export using ::cuModuleGetLoadingMode;

// Library Loading
export using ::cuLibraryLoadData;
export using ::cuLibraryGetKernel;
export using ::cuLibraryGetModule;
export using ::cuLibraryGetManaged;
export using ::cuLibraryGetUnifiedFunction;
export using ::cuLibraryUnload;

// Device Management
export using ::cuDeviceGet;
export using ::cuDeviceGetCount;
export using ::cuDeviceGetName;
export using ::cuDeviceGetUuid;
export using ::cuDeviceGetLuid;
export using ::cuDeviceGetDefaultMemPool;
export using ::cuDeviceSetMemPool;
export using ::cuDeviceGetMemPool;
export using ::cuDeviceTotalMem;
export using ::cuDeviceGetProperties;
export using ::cuDeviceGetAttribute;
export using ::cuDeviceGetP2PAttribute;
export using ::cuDeviceCanAccessPeer;
export using ::cuDeviceGetByPCIBusId;
export using ::cuDeviceGetPCIBusId;
export using ::cuDeviceGetExecAffinitySupport;
