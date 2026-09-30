// hip_runtime_api.cppm - Compile-time tests for wwr.hip.hip_runtime_api

module;

#include "test/shared/link_check.h"

// The six names ROCm 7.2 added that the 7.1 floor does not declare are guarded
// in src/hip/hip_runtime_api.cppm and so are absent from this module's surface
// at the floor -- the assertions below have to be guarded identically or the
// floor build fails on the TEST rather than the wrapper. This TU imports the
// module and includes no HIP header, so it takes HIP_VERSION from the one
// header that is only macros. Keep the spelling in step with the wrapper's.
#include <hip/hip_version.h>
#define WWR_HIP_SINCE_7_2 (HIP_VERSION >= 70200000)

export module wwr.test.hip.hip_runtime_api;

import std;
import wwr.hip.hip_runtime_api;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time tests for wwr.hip.hip_runtime_api
//
// The module is pure re-export (using declarations + constexpr flag values +
// a handful of one-line forwarding functions -- see hip_runtime_api.cppm's
// header comment for why `using` alone cannot cover hipMalloc/hipMallocAsync/
// hipBindTexture and friends). Runtime tests for the underlying HIP API would
// just test HIP itself. We verify at compile-time that:
//   1. Constexpr flags have the correct values (the #undef/re-export is correct)
//   2. Enum values match hip/hip_runtime_api.h (and driver_types.h /
//      texture_types.h / surface_types.h, included transitively) -- HIP's
//      values were verified independently, not assumed to match CUDA's
//   3. Opaque handle types have the expected type traits -- note
//      hipTextureObject_t/hipSurfaceObject_t are POINTERS in HIP
//      (struct __hip_texture*/__hip_surface*), unlike CUDA where
//      cudaTextureObject_t/cudaSurfaceObject_t are a 64-bit integral handle
//   4. Struct types satisfy trivial copyability and standard layout (C-interop guarantee)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace wwr::hip::test {

using namespace wwr::hip;

// ──────────────────────────────────────────────────────────────────────
// Constexpr flag values
// ──────────────────────────────────────────────────────────────────────

// hipHostMalloc flags
static_assert(hipHostMallocDefault == 0x0u);
static_assert(hipHostMallocPortable == 0x1u);
static_assert(hipHostMallocMapped == 0x2u);
static_assert(hipHostMallocWriteCombined == 0x4u);
static_assert(hipHostMallocNumaUser == 0x20000000u);
static_assert(hipHostMallocCoherent == 0x40000000u);
static_assert(hipHostMallocNonCoherent == 0x80000000u);

// hipEvent flags
static_assert(hipEventDefault == 0x0u);
static_assert(hipEventBlockingSync == 0x1u);
static_assert(hipEventDisableTiming == 0x2u);
static_assert(hipEventInterprocess == 0x4u);

// hipStream flags
static_assert(hipStreamDefault == 0x00u);
static_assert(hipStreamNonBlocking == 0x01u);

// hipSetDeviceFlags / hipInit flags
static_assert(hipDeviceScheduleAuto == 0x0u);
static_assert(hipDeviceScheduleSpin == 0x1u);
static_assert(hipDeviceScheduleYield == 0x2u);
static_assert(hipDeviceScheduleBlockingSync == 0x4u);
static_assert(hipDeviceScheduleMask == 0x7u);
static_assert(hipDeviceMapHost == 0x8u);
static_assert(hipDeviceLmemResizeToMax == 0x10u);

// hipMallocManaged / hipHostRegister attach flags
static_assert(hipMemAttachGlobal == 0x01u);
static_assert(hipMemAttachHost == 0x02u);
static_assert(hipMemAttachSingle == 0x04u);

// ──────────────────────────────────────────────────────────────────────
// Enum type checks
// ──────────────────────────────────────────────────────────────────────

static_assert(std::is_enum_v<hipMemoryType>);
static_assert(std::is_enum_v<hipError_t>);
static_assert(std::is_enum_v<hipDeviceAttribute_t>);
static_assert(std::is_enum_v<hipDriverProcAddressQueryResult>);
static_assert(std::is_enum_v<hipComputeMode>);
static_assert(std::is_enum_v<hipFlushGPUDirectRDMAWritesOptions>);
static_assert(std::is_enum_v<hipGPUDirectRDMAWritesOrdering>);
static_assert(std::is_enum_v<hipDeviceP2PAttr>);
static_assert(std::is_enum_v<hipDriverEntryPointQueryResult>);
static_assert(std::is_enum_v<hipLimit_t>);
static_assert(std::is_enum_v<hipStreamBatchMemOpType>);
static_assert(std::is_enum_v<hipMemoryAdvise>);
static_assert(std::is_enum_v<hipMemRangeCoherencyMode>);
static_assert(std::is_enum_v<hipMemRangeAttribute>);
static_assert(std::is_enum_v<hipMemPoolAttr>);
static_assert(std::is_enum_v<hipMemAccessFlags>);
static_assert(std::is_enum_v<hipMemAllocationType>);
static_assert(std::is_enum_v<hipMemAllocationHandleType>);
static_assert(std::is_enum_v<hipFuncAttribute>);
static_assert(std::is_enum_v<hipFuncCache_t>);
static_assert(std::is_enum_v<hipSharedMemConfig>);
static_assert(std::is_enum_v<hipExternalMemoryHandleType>);
static_assert(std::is_enum_v<hipExternalSemaphoreHandleType>);
static_assert(std::is_enum_v<hipGraphicsRegisterFlags>);
static_assert(std::is_enum_v<hipGraphNodeType>);
static_assert(std::is_enum_v<hipAccessProperty>);
static_assert(std::is_enum_v<hipLaunchMemSyncDomain>);
static_assert(std::is_enum_v<hipSynchronizationPolicy>);
static_assert(std::is_enum_v<hipLaunchAttributeID>);
static_assert(std::is_enum_v<hipGraphExecUpdateResult>);
static_assert(std::is_enum_v<hipStreamCaptureMode>);
static_assert(std::is_enum_v<hipStreamCaptureStatus>);
static_assert(std::is_enum_v<hipStreamUpdateCaptureDependenciesFlags>);
static_assert(std::is_enum_v<hipGraphMemAttributeType>);
static_assert(std::is_enum_v<hipUserObjectFlags>);
static_assert(std::is_enum_v<hipUserObjectRetainFlags>);
static_assert(std::is_enum_v<hipGraphInstantiateFlags>);
static_assert(std::is_enum_v<hipGraphDebugDotFlags>);
static_assert(std::is_enum_v<hipGraphInstantiateResult>);
static_assert(std::is_enum_v<hipMemAllocationGranularity_flags>);
static_assert(std::is_enum_v<hipMemHandleType>);
static_assert(std::is_enum_v<hipMemOperationType>);
static_assert(std::is_enum_v<hipArraySparseSubresourceType>);
static_assert(std::is_enum_v<hipGraphDependencyType>);
static_assert(std::is_enum_v<hipMemRangeHandleType>);
static_assert(std::is_enum_v<hipMemRangeFlags>);
static_assert(std::is_enum_v<hipChannelFormatKind>);
static_assert(std::is_enum_v<hipArray_Format>);
static_assert(std::is_enum_v<hipResourceType>);
static_assert(std::is_enum_v<HIPresourcetype>);
static_assert(std::is_enum_v<HIPaddress_mode>);
static_assert(std::is_enum_v<HIPfilter_mode>);
static_assert(std::is_enum_v<hipResourceViewFormat>);
static_assert(std::is_enum_v<HIPresourceViewFormat>);
static_assert(std::is_enum_v<hipMemcpyKind>);
static_assert(std::is_enum_v<hipMemLocationType>);
static_assert(std::is_enum_v<hipMemcpyFlags>);
static_assert(std::is_enum_v<hipMemcpySrcAccessOrder>);
static_assert(std::is_enum_v<hipMemcpy3DOperandType>);
static_assert(std::is_enum_v<hipFunction_attribute>);
static_assert(std::is_enum_v<hipPointer_attribute>);
static_assert(std::is_enum_v<hipTextureAddressMode>);
static_assert(std::is_enum_v<hipTextureFilterMode>);
static_assert(std::is_enum_v<hipTextureReadMode>);
static_assert(std::is_enum_v<hipSurfaceBoundaryMode>);

// ──────────────────────────────────────────────────────────────────────
// Enum values
// Verified against hip/hip_runtime_api.h, driver_types.h, texture_types.h,
// and surface_types.h -- generated from the actual compiled values (not
// hand-copied), so every one of these is independently checked against HIP,
// not assumed to match CUDA's numbering.
// ──────────────────────────────────────────────────────────────────────

// hipMemoryType
static_assert(static_cast<long long>(hipMemoryTypeUnregistered) == 0);
static_assert(static_cast<long long>(hipMemoryTypeHost) == 1);
static_assert(static_cast<long long>(hipMemoryTypeDevice) == 2);
static_assert(static_cast<long long>(hipMemoryTypeManaged) == 3);
static_assert(static_cast<long long>(hipMemoryTypeArray) == 10);
static_assert(static_cast<long long>(hipMemoryTypeUnified) == 11);

// hipError_t
static_assert(static_cast<long long>(hipSuccess) == 0);
static_assert(static_cast<long long>(hipErrorInvalidValue) == 1);
static_assert(static_cast<long long>(hipErrorOutOfMemory) == 2);
static_assert(static_cast<long long>(hipErrorMemoryAllocation) == 2);
static_assert(static_cast<long long>(hipErrorNotInitialized) == 3);
static_assert(static_cast<long long>(hipErrorInitializationError) == 3);
static_assert(static_cast<long long>(hipErrorDeinitialized) == 4);
static_assert(static_cast<long long>(hipErrorProfilerDisabled) == 5);
static_assert(static_cast<long long>(hipErrorProfilerNotInitialized) == 6);
static_assert(static_cast<long long>(hipErrorProfilerAlreadyStarted) == 7);
static_assert(static_cast<long long>(hipErrorProfilerAlreadyStopped) == 8);
static_assert(static_cast<long long>(hipErrorInvalidConfiguration) == 9);
static_assert(static_cast<long long>(hipErrorInvalidPitchValue) == 12);
static_assert(static_cast<long long>(hipErrorInvalidSymbol) == 13);
static_assert(static_cast<long long>(hipErrorInvalidDevicePointer) == 17);
static_assert(static_cast<long long>(hipErrorInvalidMemcpyDirection) == 21);
static_assert(static_cast<long long>(hipErrorInsufficientDriver) == 35);
static_assert(static_cast<long long>(hipErrorMissingConfiguration) == 52);
static_assert(static_cast<long long>(hipErrorPriorLaunchFailure) == 53);
static_assert(static_cast<long long>(hipErrorInvalidDeviceFunction) == 98);
static_assert(static_cast<long long>(hipErrorNoDevice) == 100);
static_assert(static_cast<long long>(hipErrorInvalidDevice) == 101);
static_assert(static_cast<long long>(hipErrorInvalidImage) == 200);
static_assert(static_cast<long long>(hipErrorInvalidContext) == 201);
static_assert(static_cast<long long>(hipErrorContextAlreadyCurrent) == 202);
static_assert(static_cast<long long>(hipErrorMapFailed) == 205);
static_assert(static_cast<long long>(hipErrorMapBufferObjectFailed) == 205);
static_assert(static_cast<long long>(hipErrorUnmapFailed) == 206);
static_assert(static_cast<long long>(hipErrorArrayIsMapped) == 207);
static_assert(static_cast<long long>(hipErrorAlreadyMapped) == 208);
static_assert(static_cast<long long>(hipErrorNoBinaryForGpu) == 209);
static_assert(static_cast<long long>(hipErrorAlreadyAcquired) == 210);
static_assert(static_cast<long long>(hipErrorNotMapped) == 211);
static_assert(static_cast<long long>(hipErrorNotMappedAsArray) == 212);
static_assert(static_cast<long long>(hipErrorNotMappedAsPointer) == 213);
static_assert(static_cast<long long>(hipErrorECCNotCorrectable) == 214);
static_assert(static_cast<long long>(hipErrorUnsupportedLimit) == 215);
static_assert(static_cast<long long>(hipErrorContextAlreadyInUse) == 216);
static_assert(static_cast<long long>(hipErrorPeerAccessUnsupported) == 217);
static_assert(static_cast<long long>(hipErrorInvalidKernelFile) == 218);
static_assert(static_cast<long long>(hipErrorInvalidGraphicsContext) == 219);
static_assert(static_cast<long long>(hipErrorInvalidSource) == 300);
static_assert(static_cast<long long>(hipErrorFileNotFound) == 301);
static_assert(static_cast<long long>(hipErrorSharedObjectSymbolNotFound) == 302);
static_assert(static_cast<long long>(hipErrorSharedObjectInitFailed) == 303);
static_assert(static_cast<long long>(hipErrorOperatingSystem) == 304);
static_assert(static_cast<long long>(hipErrorInvalidHandle) == 400);
static_assert(static_cast<long long>(hipErrorInvalidResourceHandle) == 400);
static_assert(static_cast<long long>(hipErrorIllegalState) == 401);
static_assert(static_cast<long long>(hipErrorNotFound) == 500);
static_assert(static_cast<long long>(hipErrorNotReady) == 600);
static_assert(static_cast<long long>(hipErrorIllegalAddress) == 700);
static_assert(static_cast<long long>(hipErrorLaunchOutOfResources) == 701);
static_assert(static_cast<long long>(hipErrorLaunchTimeOut) == 702);
static_assert(static_cast<long long>(hipErrorPeerAccessAlreadyEnabled) == 704);
static_assert(static_cast<long long>(hipErrorPeerAccessNotEnabled) == 705);
static_assert(static_cast<long long>(hipErrorSetOnActiveProcess) == 708);
static_assert(static_cast<long long>(hipErrorContextIsDestroyed) == 709);
static_assert(static_cast<long long>(hipErrorAssert) == 710);
static_assert(static_cast<long long>(hipErrorHostMemoryAlreadyRegistered) == 712);
static_assert(static_cast<long long>(hipErrorHostMemoryNotRegistered) == 713);
static_assert(static_cast<long long>(hipErrorLaunchFailure) == 719);
static_assert(static_cast<long long>(hipErrorCooperativeLaunchTooLarge) == 720);
static_assert(static_cast<long long>(hipErrorNotSupported) == 801);
static_assert(static_cast<long long>(hipErrorStreamCaptureUnsupported) == 900);
static_assert(static_cast<long long>(hipErrorStreamCaptureInvalidated) == 901);
static_assert(static_cast<long long>(hipErrorStreamCaptureMerge) == 902);
static_assert(static_cast<long long>(hipErrorStreamCaptureUnmatched) == 903);
static_assert(static_cast<long long>(hipErrorStreamCaptureUnjoined) == 904);
static_assert(static_cast<long long>(hipErrorStreamCaptureIsolation) == 905);
static_assert(static_cast<long long>(hipErrorStreamCaptureImplicit) == 906);
static_assert(static_cast<long long>(hipErrorCapturedEvent) == 907);
static_assert(static_cast<long long>(hipErrorStreamCaptureWrongThread) == 908);
static_assert(static_cast<long long>(hipErrorGraphExecUpdateFailure) == 910);
static_assert(static_cast<long long>(hipErrorInvalidChannelDescriptor) == 911);
static_assert(static_cast<long long>(hipErrorInvalidTexture) == 912);
static_assert(static_cast<long long>(hipErrorUnknown) == 999);
static_assert(static_cast<long long>(hipErrorRuntimeMemory) == 1052);
static_assert(static_cast<long long>(hipErrorRuntimeOther) == 1053);
static_assert(static_cast<long long>(hipErrorTbd) == 1054);

// hipDeviceAttribute_t
static_assert(static_cast<long long>(hipDeviceAttributeCudaCompatibleBegin) == 0);
static_assert(static_cast<long long>(hipDeviceAttributeEccEnabled) == 0);
static_assert(static_cast<long long>(hipDeviceAttributeAccessPolicyMaxWindowSize) == 1);
static_assert(static_cast<long long>(hipDeviceAttributeAsyncEngineCount) == 2);
static_assert(static_cast<long long>(hipDeviceAttributeCanMapHostMemory) == 3);
static_assert(static_cast<long long>(hipDeviceAttributeCanUseHostPointerForRegisteredMem) == 4);
static_assert(static_cast<long long>(hipDeviceAttributeClockRate) == 5);
static_assert(static_cast<long long>(hipDeviceAttributeComputeMode) == 6);
static_assert(static_cast<long long>(hipDeviceAttributeComputePreemptionSupported) == 7);
static_assert(static_cast<long long>(hipDeviceAttributeConcurrentKernels) == 8);
static_assert(static_cast<long long>(hipDeviceAttributeConcurrentManagedAccess) == 9);
static_assert(static_cast<long long>(hipDeviceAttributeCooperativeLaunch) == 10);
static_assert(static_cast<long long>(hipDeviceAttributeCooperativeMultiDeviceLaunch) == 11);
static_assert(static_cast<long long>(hipDeviceAttributeDeviceOverlap) == 12);
static_assert(static_cast<long long>(hipDeviceAttributeDirectManagedMemAccessFromHost) == 13);
static_assert(static_cast<long long>(hipDeviceAttributeGlobalL1CacheSupported) == 14);
static_assert(static_cast<long long>(hipDeviceAttributeHostNativeAtomicSupported) == 15);
static_assert(static_cast<long long>(hipDeviceAttributeIntegrated) == 16);
static_assert(static_cast<long long>(hipDeviceAttributeIsMultiGpuBoard) == 17);
static_assert(static_cast<long long>(hipDeviceAttributeKernelExecTimeout) == 18);
static_assert(static_cast<long long>(hipDeviceAttributeL2CacheSize) == 19);
static_assert(static_cast<long long>(hipDeviceAttributeLocalL1CacheSupported) == 20);
static_assert(static_cast<long long>(hipDeviceAttributeLuid) == 21);
static_assert(static_cast<long long>(hipDeviceAttributeLuidDeviceNodeMask) == 22);
static_assert(static_cast<long long>(hipDeviceAttributeComputeCapabilityMajor) == 23);
static_assert(static_cast<long long>(hipDeviceAttributeManagedMemory) == 24);
static_assert(static_cast<long long>(hipDeviceAttributeMaxBlocksPerMultiProcessor) == 25);
static_assert(static_cast<long long>(hipDeviceAttributeMaxBlockDimX) == 26);
static_assert(static_cast<long long>(hipDeviceAttributeMaxBlockDimY) == 27);
static_assert(static_cast<long long>(hipDeviceAttributeMaxBlockDimZ) == 28);
static_assert(static_cast<long long>(hipDeviceAttributeMaxGridDimX) == 29);
static_assert(static_cast<long long>(hipDeviceAttributeMaxGridDimY) == 30);
static_assert(static_cast<long long>(hipDeviceAttributeMaxGridDimZ) == 31);
static_assert(static_cast<long long>(hipDeviceAttributeMaxSurface1D) == 32);
static_assert(static_cast<long long>(hipDeviceAttributeMaxSurface1DLayered) == 33);
static_assert(static_cast<long long>(hipDeviceAttributeMaxSurface2D) == 34);
static_assert(static_cast<long long>(hipDeviceAttributeMaxSurface2DLayered) == 35);
static_assert(static_cast<long long>(hipDeviceAttributeMaxSurface3D) == 36);
static_assert(static_cast<long long>(hipDeviceAttributeMaxSurfaceCubemap) == 37);
static_assert(static_cast<long long>(hipDeviceAttributeMaxSurfaceCubemapLayered) == 38);
static_assert(static_cast<long long>(hipDeviceAttributeMaxTexture1DWidth) == 39);
static_assert(static_cast<long long>(hipDeviceAttributeMaxTexture1DLayered) == 40);
static_assert(static_cast<long long>(hipDeviceAttributeMaxTexture1DLinear) == 41);
static_assert(static_cast<long long>(hipDeviceAttributeMaxTexture1DMipmap) == 42);
static_assert(static_cast<long long>(hipDeviceAttributeMaxTexture2DWidth) == 43);
static_assert(static_cast<long long>(hipDeviceAttributeMaxTexture2DHeight) == 44);
static_assert(static_cast<long long>(hipDeviceAttributeMaxTexture2DGather) == 45);
static_assert(static_cast<long long>(hipDeviceAttributeMaxTexture2DLayered) == 46);
static_assert(static_cast<long long>(hipDeviceAttributeMaxTexture2DLinear) == 47);
static_assert(static_cast<long long>(hipDeviceAttributeMaxTexture2DMipmap) == 48);
static_assert(static_cast<long long>(hipDeviceAttributeMaxTexture3DWidth) == 49);
static_assert(static_cast<long long>(hipDeviceAttributeMaxTexture3DHeight) == 50);
static_assert(static_cast<long long>(hipDeviceAttributeMaxTexture3DDepth) == 51);
static_assert(static_cast<long long>(hipDeviceAttributeMaxTexture3DAlt) == 52);
static_assert(static_cast<long long>(hipDeviceAttributeMaxTextureCubemap) == 53);
static_assert(static_cast<long long>(hipDeviceAttributeMaxTextureCubemapLayered) == 54);
static_assert(static_cast<long long>(hipDeviceAttributeMaxThreadsDim) == 55);
static_assert(static_cast<long long>(hipDeviceAttributeMaxThreadsPerBlock) == 56);
static_assert(static_cast<long long>(hipDeviceAttributeMaxThreadsPerMultiProcessor) == 57);
static_assert(static_cast<long long>(hipDeviceAttributeMaxPitch) == 58);
static_assert(static_cast<long long>(hipDeviceAttributeMemoryBusWidth) == 59);
static_assert(static_cast<long long>(hipDeviceAttributeMemoryClockRate) == 60);
static_assert(static_cast<long long>(hipDeviceAttributeComputeCapabilityMinor) == 61);
static_assert(static_cast<long long>(hipDeviceAttributeMultiGpuBoardGroupID) == 62);
static_assert(static_cast<long long>(hipDeviceAttributeMultiprocessorCount) == 63);
static_assert(static_cast<long long>(hipDeviceAttributeUnused1) == 64);
static_assert(static_cast<long long>(hipDeviceAttributePageableMemoryAccess) == 65);
static_assert(static_cast<long long>(hipDeviceAttributePageableMemoryAccessUsesHostPageTables) ==
              66);
static_assert(static_cast<long long>(hipDeviceAttributePciBusId) == 67);
static_assert(static_cast<long long>(hipDeviceAttributePciDeviceId) == 68);
static_assert(static_cast<long long>(hipDeviceAttributePciDomainId) == 69);
static_assert(static_cast<long long>(hipDeviceAttributePciDomainID) == 69);
static_assert(static_cast<long long>(hipDeviceAttributePersistingL2CacheMaxSize) == 70);
static_assert(static_cast<long long>(hipDeviceAttributeMaxRegistersPerBlock) == 71);
static_assert(static_cast<long long>(hipDeviceAttributeMaxRegistersPerMultiprocessor) == 72);
static_assert(static_cast<long long>(hipDeviceAttributeReservedSharedMemPerBlock) == 73);
static_assert(static_cast<long long>(hipDeviceAttributeMaxSharedMemoryPerBlock) == 74);
static_assert(static_cast<long long>(hipDeviceAttributeSharedMemPerBlockOptin) == 75);
static_assert(static_cast<long long>(hipDeviceAttributeSharedMemPerMultiprocessor) == 76);
static_assert(static_cast<long long>(hipDeviceAttributeSingleToDoublePrecisionPerfRatio) == 77);
static_assert(static_cast<long long>(hipDeviceAttributeStreamPrioritiesSupported) == 78);
static_assert(static_cast<long long>(hipDeviceAttributeSurfaceAlignment) == 79);
static_assert(static_cast<long long>(hipDeviceAttributeTccDriver) == 80);
static_assert(static_cast<long long>(hipDeviceAttributeTextureAlignment) == 81);
static_assert(static_cast<long long>(hipDeviceAttributeTexturePitchAlignment) == 82);
static_assert(static_cast<long long>(hipDeviceAttributeTotalConstantMemory) == 83);
static_assert(static_cast<long long>(hipDeviceAttributeTotalGlobalMem) == 84);
static_assert(static_cast<long long>(hipDeviceAttributeUnifiedAddressing) == 85);
static_assert(static_cast<long long>(hipDeviceAttributeUnused2) == 86);
static_assert(static_cast<long long>(hipDeviceAttributeWarpSize) == 87);
static_assert(static_cast<long long>(hipDeviceAttributeMemoryPoolsSupported) == 88);
static_assert(static_cast<long long>(hipDeviceAttributeVirtualMemoryManagementSupported) == 89);
static_assert(static_cast<long long>(hipDeviceAttributeHostRegisterSupported) == 90);
static_assert(static_cast<long long>(hipDeviceAttributeMemoryPoolSupportedHandleTypes) == 91);
#if WWR_HIP_SINCE_7_2
static_assert(static_cast<long long>(hipDeviceAttributeHostNumaId) == 92);
#endif
static_assert(static_cast<long long>(hipDeviceAttributeCudaCompatibleEnd) == 9999);
static_assert(static_cast<long long>(hipDeviceAttributeAmdSpecificBegin) == 10000);
static_assert(static_cast<long long>(hipDeviceAttributeClockInstructionRate) == 10000);
static_assert(static_cast<long long>(hipDeviceAttributeUnused3) == 10001);
static_assert(static_cast<long long>(hipDeviceAttributeMaxSharedMemoryPerMultiprocessor) == 10002);
static_assert(static_cast<long long>(hipDeviceAttributeUnused4) == 10003);
static_assert(static_cast<long long>(hipDeviceAttributeUnused5) == 10004);
static_assert(static_cast<long long>(hipDeviceAttributeHdpMemFlushCntl) == 10005);
static_assert(static_cast<long long>(hipDeviceAttributeHdpRegFlushCntl) == 10006);
static_assert(static_cast<long long>(hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc) ==
              10007);
static_assert(static_cast<long long>(hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim) ==
              10008);
static_assert(static_cast<long long>(hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim) ==
              10009);
static_assert(static_cast<long long>(hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem) ==
              10010);
static_assert(static_cast<long long>(hipDeviceAttributeIsLargeBar) == 10011);
static_assert(static_cast<long long>(hipDeviceAttributeAsicRevision) == 10012);
static_assert(static_cast<long long>(hipDeviceAttributeCanUseStreamWaitValue) == 10013);
static_assert(static_cast<long long>(hipDeviceAttributeImageSupport) == 10014);
static_assert(static_cast<long long>(hipDeviceAttributePhysicalMultiProcessorCount) == 10015);
static_assert(static_cast<long long>(hipDeviceAttributeFineGrainSupport) == 10016);
static_assert(static_cast<long long>(hipDeviceAttributeWallClockRate) == 10017);
static_assert(static_cast<long long>(hipDeviceAttributeNumberOfXccs) == 10018);
static_assert(static_cast<long long>(hipDeviceAttributeMaxAvailableVgprsPerThread) == 10019);
static_assert(static_cast<long long>(hipDeviceAttributePciChipId) == 10020);
static_assert(static_cast<long long>(hipDeviceAttributeAmdSpecificEnd) == 19999);
static_assert(static_cast<long long>(hipDeviceAttributeVendorSpecificBegin) == 20000);

// hipDriverProcAddressQueryResult
static_assert(static_cast<long long>(HIP_GET_PROC_ADDRESS_SUCCESS) == 0);
static_assert(static_cast<long long>(HIP_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND) == 1);
static_assert(static_cast<long long>(HIP_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT) == 2);

// hipComputeMode
static_assert(static_cast<long long>(hipComputeModeDefault) == 0);
static_assert(static_cast<long long>(hipComputeModeExclusive) == 1);
static_assert(static_cast<long long>(hipComputeModeProhibited) == 2);
static_assert(static_cast<long long>(hipComputeModeExclusiveProcess) == 3);

// hipFlushGPUDirectRDMAWritesOptions
static_assert(static_cast<long long>(hipFlushGPUDirectRDMAWritesOptionHost) == 1);
static_assert(static_cast<long long>(hipFlushGPUDirectRDMAWritesOptionMemOps) == 2);

// hipGPUDirectRDMAWritesOrdering
static_assert(static_cast<long long>(hipGPUDirectRDMAWritesOrderingNone) == 0);
static_assert(static_cast<long long>(hipGPUDirectRDMAWritesOrderingOwner) == 100);
static_assert(static_cast<long long>(hipGPUDirectRDMAWritesOrderingAllDevices) == 200);

// hipDeviceP2PAttr
static_assert(static_cast<long long>(hipDevP2PAttrPerformanceRank) == 0);
static_assert(static_cast<long long>(hipDevP2PAttrAccessSupported) == 1);
static_assert(static_cast<long long>(hipDevP2PAttrNativeAtomicSupported) == 2);
static_assert(static_cast<long long>(hipDevP2PAttrHipArrayAccessSupported) == 3);

// hipDriverEntryPointQueryResult
static_assert(static_cast<long long>(hipDriverEntryPointSuccess) == 0);
static_assert(static_cast<long long>(hipDriverEntryPointSymbolNotFound) == 1);
static_assert(static_cast<long long>(hipDriverEntryPointVersionNotSufficent) == 2);

// hipLimit_t
static_assert(static_cast<long long>(hipLimitStackSize) == 0);
static_assert(static_cast<long long>(hipLimitPrintfFifoSize) == 1);
static_assert(static_cast<long long>(hipLimitMallocHeapSize) == 2);
static_assert(static_cast<long long>(hipExtLimitScratchMin) == 4096);
static_assert(static_cast<long long>(hipExtLimitScratchMax) == 4097);
static_assert(static_cast<long long>(hipExtLimitScratchCurrent) == 4098);
static_assert(static_cast<long long>(hipLimitRange) == 4099);

// hipStreamBatchMemOpType
static_assert(static_cast<long long>(hipStreamMemOpWaitValue32) == 1);
static_assert(static_cast<long long>(hipStreamMemOpWriteValue32) == 2);
static_assert(static_cast<long long>(hipStreamMemOpWaitValue64) == 4);
static_assert(static_cast<long long>(hipStreamMemOpWriteValue64) == 5);
static_assert(static_cast<long long>(hipStreamMemOpBarrier) == 6);
static_assert(static_cast<long long>(hipStreamMemOpFlushRemoteWrites) == 3);

// hipMemoryAdvise
static_assert(static_cast<long long>(hipMemAdviseSetReadMostly) == 1);
static_assert(static_cast<long long>(hipMemAdviseUnsetReadMostly) == 2);
static_assert(static_cast<long long>(hipMemAdviseSetPreferredLocation) == 3);
static_assert(static_cast<long long>(hipMemAdviseUnsetPreferredLocation) == 4);
static_assert(static_cast<long long>(hipMemAdviseSetAccessedBy) == 5);
static_assert(static_cast<long long>(hipMemAdviseUnsetAccessedBy) == 6);
static_assert(static_cast<long long>(hipMemAdviseSetCoarseGrain) == 100);
static_assert(static_cast<long long>(hipMemAdviseUnsetCoarseGrain) == 101);

// hipMemRangeCoherencyMode
static_assert(static_cast<long long>(hipMemRangeCoherencyModeFineGrain) == 0);
static_assert(static_cast<long long>(hipMemRangeCoherencyModeCoarseGrain) == 1);
static_assert(static_cast<long long>(hipMemRangeCoherencyModeIndeterminate) == 2);

// hipMemRangeAttribute
static_assert(static_cast<long long>(hipMemRangeAttributeReadMostly) == 1);
static_assert(static_cast<long long>(hipMemRangeAttributePreferredLocation) == 2);
static_assert(static_cast<long long>(hipMemRangeAttributeAccessedBy) == 3);
static_assert(static_cast<long long>(hipMemRangeAttributeLastPrefetchLocation) == 4);
static_assert(static_cast<long long>(hipMemRangeAttributeCoherencyMode) == 100);

// hipMemPoolAttr
static_assert(static_cast<long long>(hipMemPoolReuseFollowEventDependencies) == 1);
static_assert(static_cast<long long>(hipMemPoolReuseAllowOpportunistic) == 2);
static_assert(static_cast<long long>(hipMemPoolReuseAllowInternalDependencies) == 3);
static_assert(static_cast<long long>(hipMemPoolAttrReleaseThreshold) == 4);
static_assert(static_cast<long long>(hipMemPoolAttrReservedMemCurrent) == 5);
static_assert(static_cast<long long>(hipMemPoolAttrReservedMemHigh) == 6);
static_assert(static_cast<long long>(hipMemPoolAttrUsedMemCurrent) == 7);
static_assert(static_cast<long long>(hipMemPoolAttrUsedMemHigh) == 8);

// hipMemAccessFlags
static_assert(static_cast<long long>(hipMemAccessFlagsProtNone) == 0);
static_assert(static_cast<long long>(hipMemAccessFlagsProtRead) == 1);
static_assert(static_cast<long long>(hipMemAccessFlagsProtReadWrite) == 3);

// hipMemAllocationType
static_assert(static_cast<long long>(hipMemAllocationTypeInvalid) == 0);
static_assert(static_cast<long long>(hipMemAllocationTypePinned) == 1);
static_assert(static_cast<long long>(hipMemAllocationTypeUncached) == 1073741824);
static_assert(static_cast<long long>(hipMemAllocationTypeMax) == 2147483647);

// hipMemAllocationHandleType
static_assert(static_cast<long long>(hipMemHandleTypeNone) == 0);
static_assert(static_cast<long long>(hipMemHandleTypePosixFileDescriptor) == 1);
static_assert(static_cast<long long>(hipMemHandleTypeWin32) == 2);
static_assert(static_cast<long long>(hipMemHandleTypeWin32Kmt) == 4);

// hipFuncAttribute
static_assert(static_cast<long long>(hipFuncAttributeMaxDynamicSharedMemorySize) == 8);
static_assert(static_cast<long long>(hipFuncAttributePreferredSharedMemoryCarveout) == 9);
static_assert(static_cast<long long>(hipFuncAttributeMax) == 10);

// hipFuncCache_t
static_assert(static_cast<long long>(hipFuncCachePreferNone) == 0);
static_assert(static_cast<long long>(hipFuncCachePreferShared) == 1);
static_assert(static_cast<long long>(hipFuncCachePreferL1) == 2);
static_assert(static_cast<long long>(hipFuncCachePreferEqual) == 3);

// hipSharedMemConfig
static_assert(static_cast<long long>(hipSharedMemBankSizeDefault) == 0);
static_assert(static_cast<long long>(hipSharedMemBankSizeFourByte) == 1);
static_assert(static_cast<long long>(hipSharedMemBankSizeEightByte) == 2);

// hipExternalMemoryHandleType
static_assert(static_cast<long long>(hipExternalMemoryHandleTypeOpaqueFd) == 1);
static_assert(static_cast<long long>(hipExternalMemoryHandleTypeOpaqueWin32) == 2);
static_assert(static_cast<long long>(hipExternalMemoryHandleTypeOpaqueWin32Kmt) == 3);
static_assert(static_cast<long long>(hipExternalMemoryHandleTypeD3D12Heap) == 4);
static_assert(static_cast<long long>(hipExternalMemoryHandleTypeD3D12Resource) == 5);
static_assert(static_cast<long long>(hipExternalMemoryHandleTypeD3D11Resource) == 6);
static_assert(static_cast<long long>(hipExternalMemoryHandleTypeD3D11ResourceKmt) == 7);
static_assert(static_cast<long long>(hipExternalMemoryHandleTypeNvSciBuf) == 8);

// hipExternalSemaphoreHandleType
static_assert(static_cast<long long>(hipExternalSemaphoreHandleTypeOpaqueFd) == 1);
static_assert(static_cast<long long>(hipExternalSemaphoreHandleTypeOpaqueWin32) == 2);
static_assert(static_cast<long long>(hipExternalSemaphoreHandleTypeOpaqueWin32Kmt) == 3);
static_assert(static_cast<long long>(hipExternalSemaphoreHandleTypeD3D12Fence) == 4);
static_assert(static_cast<long long>(hipExternalSemaphoreHandleTypeD3D11Fence) == 5);
static_assert(static_cast<long long>(hipExternalSemaphoreHandleTypeNvSciSync) == 6);
static_assert(static_cast<long long>(hipExternalSemaphoreHandleTypeKeyedMutex) == 7);
static_assert(static_cast<long long>(hipExternalSemaphoreHandleTypeKeyedMutexKmt) == 8);
static_assert(static_cast<long long>(hipExternalSemaphoreHandleTypeTimelineSemaphoreFd) == 9);
static_assert(static_cast<long long>(hipExternalSemaphoreHandleTypeTimelineSemaphoreWin32) == 10);

// hipGraphicsRegisterFlags
static_assert(static_cast<long long>(hipGraphicsRegisterFlagsNone) == 0);
static_assert(static_cast<long long>(hipGraphicsRegisterFlagsReadOnly) == 1);
static_assert(static_cast<long long>(hipGraphicsRegisterFlagsWriteDiscard) == 2);
static_assert(static_cast<long long>(hipGraphicsRegisterFlagsSurfaceLoadStore) == 4);
static_assert(static_cast<long long>(hipGraphicsRegisterFlagsTextureGather) == 8);

// hipGraphNodeType
static_assert(static_cast<long long>(hipGraphNodeTypeKernel) == 0);
static_assert(static_cast<long long>(hipGraphNodeTypeMemcpy) == 1);
static_assert(static_cast<long long>(hipGraphNodeTypeMemset) == 2);
static_assert(static_cast<long long>(hipGraphNodeTypeHost) == 3);
static_assert(static_cast<long long>(hipGraphNodeTypeGraph) == 4);
static_assert(static_cast<long long>(hipGraphNodeTypeEmpty) == 5);
static_assert(static_cast<long long>(hipGraphNodeTypeWaitEvent) == 6);
static_assert(static_cast<long long>(hipGraphNodeTypeEventRecord) == 7);
static_assert(static_cast<long long>(hipGraphNodeTypeExtSemaphoreSignal) == 8);
static_assert(static_cast<long long>(hipGraphNodeTypeExtSemaphoreWait) == 9);
static_assert(static_cast<long long>(hipGraphNodeTypeMemAlloc) == 10);
static_assert(static_cast<long long>(hipGraphNodeTypeMemFree) == 11);
static_assert(static_cast<long long>(hipGraphNodeTypeMemcpyFromSymbol) == 12);
static_assert(static_cast<long long>(hipGraphNodeTypeMemcpyToSymbol) == 13);
static_assert(static_cast<long long>(hipGraphNodeTypeBatchMemOp) == 14);
static_assert(static_cast<long long>(hipGraphNodeTypeCount) == 15);

// hipAccessProperty
static_assert(static_cast<long long>(hipAccessPropertyNormal) == 0);
static_assert(static_cast<long long>(hipAccessPropertyStreaming) == 1);
static_assert(static_cast<long long>(hipAccessPropertyPersisting) == 2);

// hipLaunchMemSyncDomain
static_assert(static_cast<long long>(hipLaunchMemSyncDomainDefault) == 0);
static_assert(static_cast<long long>(hipLaunchMemSyncDomainRemote) == 1);

// hipSynchronizationPolicy
static_assert(static_cast<long long>(hipSyncPolicyAuto) == 1);
static_assert(static_cast<long long>(hipSyncPolicySpin) == 2);
static_assert(static_cast<long long>(hipSyncPolicyYield) == 3);
static_assert(static_cast<long long>(hipSyncPolicyBlockingSync) == 4);

// hipLaunchAttributeID
static_assert(static_cast<long long>(hipLaunchAttributeAccessPolicyWindow) == 1);
static_assert(static_cast<long long>(hipLaunchAttributeCooperative) == 2);
static_assert(static_cast<long long>(hipLaunchAttributeSynchronizationPolicy) == 3);
static_assert(static_cast<long long>(hipLaunchAttributePriority) == 8);
static_assert(static_cast<long long>(hipLaunchAttributeMemSyncDomainMap) == 9);
static_assert(static_cast<long long>(hipLaunchAttributeMemSyncDomain) == 10);
static_assert(static_cast<long long>(hipLaunchAttributeMax) == 11);

// hipGraphExecUpdateResult
static_assert(static_cast<long long>(hipGraphExecUpdateSuccess) == 0);
static_assert(static_cast<long long>(hipGraphExecUpdateError) == 1);
static_assert(static_cast<long long>(hipGraphExecUpdateErrorTopologyChanged) == 2);
static_assert(static_cast<long long>(hipGraphExecUpdateErrorNodeTypeChanged) == 3);
static_assert(static_cast<long long>(hipGraphExecUpdateErrorFunctionChanged) == 4);
static_assert(static_cast<long long>(hipGraphExecUpdateErrorParametersChanged) == 5);
static_assert(static_cast<long long>(hipGraphExecUpdateErrorNotSupported) == 6);
static_assert(static_cast<long long>(hipGraphExecUpdateErrorUnsupportedFunctionChange) == 7);

// hipStreamCaptureMode
static_assert(static_cast<long long>(hipStreamCaptureModeGlobal) == 0);
static_assert(static_cast<long long>(hipStreamCaptureModeThreadLocal) == 1);
static_assert(static_cast<long long>(hipStreamCaptureModeRelaxed) == 2);

// hipStreamCaptureStatus
static_assert(static_cast<long long>(hipStreamCaptureStatusNone) == 0);
static_assert(static_cast<long long>(hipStreamCaptureStatusActive) == 1);
static_assert(static_cast<long long>(hipStreamCaptureStatusInvalidated) == 2);

// hipStreamUpdateCaptureDependenciesFlags
static_assert(static_cast<long long>(hipStreamAddCaptureDependencies) == 0);
static_assert(static_cast<long long>(hipStreamSetCaptureDependencies) == 1);

// hipGraphMemAttributeType
static_assert(static_cast<long long>(hipGraphMemAttrUsedMemCurrent) == 0);
static_assert(static_cast<long long>(hipGraphMemAttrUsedMemHigh) == 1);
static_assert(static_cast<long long>(hipGraphMemAttrReservedMemCurrent) == 2);
static_assert(static_cast<long long>(hipGraphMemAttrReservedMemHigh) == 3);

// hipUserObjectFlags
static_assert(static_cast<long long>(hipUserObjectNoDestructorSync) == 1);

// hipUserObjectRetainFlags
static_assert(static_cast<long long>(hipGraphUserObjectMove) == 1);

// hipGraphInstantiateFlags
static_assert(static_cast<long long>(hipGraphInstantiateFlagAutoFreeOnLaunch) == 1);
static_assert(static_cast<long long>(hipGraphInstantiateFlagUpload) == 2);
static_assert(static_cast<long long>(hipGraphInstantiateFlagDeviceLaunch) == 4);
static_assert(static_cast<long long>(hipGraphInstantiateFlagUseNodePriority) == 8);

// hipGraphDebugDotFlags
static_assert(static_cast<long long>(hipGraphDebugDotFlagsVerbose) == 1);
static_assert(static_cast<long long>(hipGraphDebugDotFlagsKernelNodeParams) == 4);
static_assert(static_cast<long long>(hipGraphDebugDotFlagsMemcpyNodeParams) == 8);
static_assert(static_cast<long long>(hipGraphDebugDotFlagsMemsetNodeParams) == 16);
static_assert(static_cast<long long>(hipGraphDebugDotFlagsHostNodeParams) == 32);
static_assert(static_cast<long long>(hipGraphDebugDotFlagsEventNodeParams) == 64);
static_assert(static_cast<long long>(hipGraphDebugDotFlagsExtSemasSignalNodeParams) == 128);
static_assert(static_cast<long long>(hipGraphDebugDotFlagsExtSemasWaitNodeParams) == 256);
static_assert(static_cast<long long>(hipGraphDebugDotFlagsKernelNodeAttributes) == 512);
static_assert(static_cast<long long>(hipGraphDebugDotFlagsHandles) == 1024);

// hipGraphInstantiateResult
static_assert(static_cast<long long>(hipGraphInstantiateSuccess) == 0);
static_assert(static_cast<long long>(hipGraphInstantiateError) == 1);
static_assert(static_cast<long long>(hipGraphInstantiateInvalidStructure) == 2);
static_assert(static_cast<long long>(hipGraphInstantiateNodeOperationNotSupported) == 3);
static_assert(static_cast<long long>(hipGraphInstantiateMultipleDevicesNotSupported) == 4);

// hipMemAllocationGranularity_flags
static_assert(static_cast<long long>(hipMemAllocationGranularityMinimum) == 0);
static_assert(static_cast<long long>(hipMemAllocationGranularityRecommended) == 1);

// hipMemHandleType
static_assert(static_cast<long long>(hipMemHandleTypeGeneric) == 0);

// hipMemOperationType
static_assert(static_cast<long long>(hipMemOperationTypeMap) == 1);
static_assert(static_cast<long long>(hipMemOperationTypeUnmap) == 2);

// hipArraySparseSubresourceType
static_assert(static_cast<long long>(hipArraySparseSubresourceTypeSparseLevel) == 0);
static_assert(static_cast<long long>(hipArraySparseSubresourceTypeMiptail) == 1);

// hipGraphDependencyType
static_assert(static_cast<long long>(hipGraphDependencyTypeDefault) == 0);
static_assert(static_cast<long long>(hipGraphDependencyTypeProgrammatic) == 1);

// hipMemRangeHandleType
static_assert(static_cast<long long>(hipMemRangeHandleTypeDmaBufFd) == 1);
static_assert(static_cast<long long>(hipMemRangeHandleTypeMax) == 2147483647);

// hipMemRangeFlags
static_assert(static_cast<long long>(hipMemRangeFlagDmaBufMappingTypePcie) == 1);
static_assert(static_cast<long long>(hipMemRangeFlagsMax) == 2147483647);

// hipChannelFormatKind
static_assert(static_cast<long long>(hipChannelFormatKindSigned) == 0);
static_assert(static_cast<long long>(hipChannelFormatKindUnsigned) == 1);
static_assert(static_cast<long long>(hipChannelFormatKindFloat) == 2);
static_assert(static_cast<long long>(hipChannelFormatKindNone) == 3);

// hipArray_Format
static_assert(static_cast<long long>(HIP_AD_FORMAT_UNSIGNED_INT8) == 1);
static_assert(static_cast<long long>(HIP_AD_FORMAT_UNSIGNED_INT16) == 2);
static_assert(static_cast<long long>(HIP_AD_FORMAT_UNSIGNED_INT32) == 3);
static_assert(static_cast<long long>(HIP_AD_FORMAT_SIGNED_INT8) == 8);
static_assert(static_cast<long long>(HIP_AD_FORMAT_SIGNED_INT16) == 9);
static_assert(static_cast<long long>(HIP_AD_FORMAT_SIGNED_INT32) == 10);
static_assert(static_cast<long long>(HIP_AD_FORMAT_HALF) == 16);
static_assert(static_cast<long long>(HIP_AD_FORMAT_FLOAT) == 32);

// hipResourceType
static_assert(static_cast<long long>(hipResourceTypeArray) == 0);
static_assert(static_cast<long long>(hipResourceTypeMipmappedArray) == 1);
static_assert(static_cast<long long>(hipResourceTypeLinear) == 2);
static_assert(static_cast<long long>(hipResourceTypePitch2D) == 3);

// HIPresourcetype
static_assert(static_cast<long long>(HIP_RESOURCE_TYPE_ARRAY) == 0);
static_assert(static_cast<long long>(HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY) == 1);
static_assert(static_cast<long long>(HIP_RESOURCE_TYPE_LINEAR) == 2);
static_assert(static_cast<long long>(HIP_RESOURCE_TYPE_PITCH2D) == 3);

// HIPaddress_mode
static_assert(static_cast<long long>(HIP_TR_ADDRESS_MODE_WRAP) == 0);
static_assert(static_cast<long long>(HIP_TR_ADDRESS_MODE_CLAMP) == 1);
static_assert(static_cast<long long>(HIP_TR_ADDRESS_MODE_MIRROR) == 2);
static_assert(static_cast<long long>(HIP_TR_ADDRESS_MODE_BORDER) == 3);

// HIPfilter_mode
static_assert(static_cast<long long>(HIP_TR_FILTER_MODE_POINT) == 0);
static_assert(static_cast<long long>(HIP_TR_FILTER_MODE_LINEAR) == 1);

// hipResourceViewFormat
static_assert(static_cast<long long>(hipResViewFormatNone) == 0);
static_assert(static_cast<long long>(hipResViewFormatUnsignedChar1) == 1);
static_assert(static_cast<long long>(hipResViewFormatUnsignedChar2) == 2);
static_assert(static_cast<long long>(hipResViewFormatUnsignedChar4) == 3);
static_assert(static_cast<long long>(hipResViewFormatSignedChar1) == 4);
static_assert(static_cast<long long>(hipResViewFormatSignedChar2) == 5);
static_assert(static_cast<long long>(hipResViewFormatSignedChar4) == 6);
static_assert(static_cast<long long>(hipResViewFormatUnsignedShort1) == 7);
static_assert(static_cast<long long>(hipResViewFormatUnsignedShort2) == 8);
static_assert(static_cast<long long>(hipResViewFormatUnsignedShort4) == 9);
static_assert(static_cast<long long>(hipResViewFormatSignedShort1) == 10);
static_assert(static_cast<long long>(hipResViewFormatSignedShort2) == 11);
static_assert(static_cast<long long>(hipResViewFormatSignedShort4) == 12);
static_assert(static_cast<long long>(hipResViewFormatUnsignedInt1) == 13);
static_assert(static_cast<long long>(hipResViewFormatUnsignedInt2) == 14);
static_assert(static_cast<long long>(hipResViewFormatUnsignedInt4) == 15);
static_assert(static_cast<long long>(hipResViewFormatSignedInt1) == 16);
static_assert(static_cast<long long>(hipResViewFormatSignedInt2) == 17);
static_assert(static_cast<long long>(hipResViewFormatSignedInt4) == 18);
static_assert(static_cast<long long>(hipResViewFormatHalf1) == 19);
static_assert(static_cast<long long>(hipResViewFormatHalf2) == 20);
static_assert(static_cast<long long>(hipResViewFormatHalf4) == 21);
static_assert(static_cast<long long>(hipResViewFormatFloat1) == 22);
static_assert(static_cast<long long>(hipResViewFormatFloat2) == 23);
static_assert(static_cast<long long>(hipResViewFormatFloat4) == 24);
static_assert(static_cast<long long>(hipResViewFormatUnsignedBlockCompressed1) == 25);
static_assert(static_cast<long long>(hipResViewFormatUnsignedBlockCompressed2) == 26);
static_assert(static_cast<long long>(hipResViewFormatUnsignedBlockCompressed3) == 27);
static_assert(static_cast<long long>(hipResViewFormatUnsignedBlockCompressed4) == 28);
static_assert(static_cast<long long>(hipResViewFormatSignedBlockCompressed4) == 29);
static_assert(static_cast<long long>(hipResViewFormatUnsignedBlockCompressed5) == 30);
static_assert(static_cast<long long>(hipResViewFormatSignedBlockCompressed5) == 31);
static_assert(static_cast<long long>(hipResViewFormatUnsignedBlockCompressed6H) == 32);
static_assert(static_cast<long long>(hipResViewFormatSignedBlockCompressed6H) == 33);
static_assert(static_cast<long long>(hipResViewFormatUnsignedBlockCompressed7) == 34);

// HIPresourceViewFormat
static_assert(static_cast<long long>(HIP_RES_VIEW_FORMAT_NONE) == 0);
static_assert(static_cast<long long>(HIP_RES_VIEW_FORMAT_UINT_1X8) == 1);
static_assert(static_cast<long long>(HIP_RES_VIEW_FORMAT_UINT_2X8) == 2);
static_assert(static_cast<long long>(HIP_RES_VIEW_FORMAT_UINT_4X8) == 3);
static_assert(static_cast<long long>(HIP_RES_VIEW_FORMAT_SINT_1X8) == 4);
static_assert(static_cast<long long>(HIP_RES_VIEW_FORMAT_SINT_2X8) == 5);
static_assert(static_cast<long long>(HIP_RES_VIEW_FORMAT_SINT_4X8) == 6);
static_assert(static_cast<long long>(HIP_RES_VIEW_FORMAT_UINT_1X16) == 7);
static_assert(static_cast<long long>(HIP_RES_VIEW_FORMAT_UINT_2X16) == 8);
static_assert(static_cast<long long>(HIP_RES_VIEW_FORMAT_UINT_4X16) == 9);
static_assert(static_cast<long long>(HIP_RES_VIEW_FORMAT_SINT_1X16) == 10);
static_assert(static_cast<long long>(HIP_RES_VIEW_FORMAT_SINT_2X16) == 11);
static_assert(static_cast<long long>(HIP_RES_VIEW_FORMAT_SINT_4X16) == 12);
static_assert(static_cast<long long>(HIP_RES_VIEW_FORMAT_UINT_1X32) == 13);
static_assert(static_cast<long long>(HIP_RES_VIEW_FORMAT_UINT_2X32) == 14);
static_assert(static_cast<long long>(HIP_RES_VIEW_FORMAT_UINT_4X32) == 15);
static_assert(static_cast<long long>(HIP_RES_VIEW_FORMAT_SINT_1X32) == 16);
static_assert(static_cast<long long>(HIP_RES_VIEW_FORMAT_SINT_2X32) == 17);
static_assert(static_cast<long long>(HIP_RES_VIEW_FORMAT_SINT_4X32) == 18);
static_assert(static_cast<long long>(HIP_RES_VIEW_FORMAT_FLOAT_1X16) == 19);
static_assert(static_cast<long long>(HIP_RES_VIEW_FORMAT_FLOAT_2X16) == 20);
static_assert(static_cast<long long>(HIP_RES_VIEW_FORMAT_FLOAT_4X16) == 21);
static_assert(static_cast<long long>(HIP_RES_VIEW_FORMAT_FLOAT_1X32) == 22);
static_assert(static_cast<long long>(HIP_RES_VIEW_FORMAT_FLOAT_2X32) == 23);
static_assert(static_cast<long long>(HIP_RES_VIEW_FORMAT_FLOAT_4X32) == 24);
static_assert(static_cast<long long>(HIP_RES_VIEW_FORMAT_UNSIGNED_BC1) == 25);
static_assert(static_cast<long long>(HIP_RES_VIEW_FORMAT_UNSIGNED_BC2) == 26);
static_assert(static_cast<long long>(HIP_RES_VIEW_FORMAT_UNSIGNED_BC3) == 27);
static_assert(static_cast<long long>(HIP_RES_VIEW_FORMAT_UNSIGNED_BC4) == 28);
static_assert(static_cast<long long>(HIP_RES_VIEW_FORMAT_SIGNED_BC4) == 29);
static_assert(static_cast<long long>(HIP_RES_VIEW_FORMAT_UNSIGNED_BC5) == 30);
static_assert(static_cast<long long>(HIP_RES_VIEW_FORMAT_SIGNED_BC5) == 31);
static_assert(static_cast<long long>(HIP_RES_VIEW_FORMAT_UNSIGNED_BC6H) == 32);
static_assert(static_cast<long long>(HIP_RES_VIEW_FORMAT_SIGNED_BC6H) == 33);
static_assert(static_cast<long long>(HIP_RES_VIEW_FORMAT_UNSIGNED_BC7) == 34);

// hipMemcpyKind
static_assert(static_cast<long long>(hipMemcpyHostToHost) == 0);
static_assert(static_cast<long long>(hipMemcpyHostToDevice) == 1);
static_assert(static_cast<long long>(hipMemcpyDeviceToHost) == 2);
static_assert(static_cast<long long>(hipMemcpyDeviceToDevice) == 3);
static_assert(static_cast<long long>(hipMemcpyDefault) == 4);
static_assert(static_cast<long long>(hipMemcpyDeviceToDeviceNoCU) == 1024);

// hipMemLocationType
static_assert(static_cast<long long>(hipMemLocationTypeInvalid) == 0);
static_assert(static_cast<long long>(hipMemLocationTypeNone) == 0);
static_assert(static_cast<long long>(hipMemLocationTypeDevice) == 1);
static_assert(static_cast<long long>(hipMemLocationTypeHost) == 2);
static_assert(static_cast<long long>(hipMemLocationTypeHostNuma) == 3);
static_assert(static_cast<long long>(hipMemLocationTypeHostNumaCurrent) == 4);

// hipMemcpyFlags
static_assert(static_cast<long long>(hipMemcpyFlagDefault) == 0);
static_assert(static_cast<long long>(hipMemcpyFlagPreferOverlapWithCompute) == 1);

// hipMemcpySrcAccessOrder
static_assert(static_cast<long long>(hipMemcpySrcAccessOrderInvalid) == 0);
static_assert(static_cast<long long>(hipMemcpySrcAccessOrderStream) == 1);
static_assert(static_cast<long long>(hipMemcpySrcAccessOrderDuringApiCall) == 2);
static_assert(static_cast<long long>(hipMemcpySrcAccessOrderAny) == 3);
static_assert(static_cast<long long>(hipMemcpySrcAccessOrderMax) == 2147483647);

// hipMemcpy3DOperandType
static_assert(static_cast<long long>(hipMemcpyOperandTypePointer) == 1);
static_assert(static_cast<long long>(hipMemcpyOperandTypeArray) == 2);
static_assert(static_cast<long long>(hipMemcpyOperandTypeMax) == 2147483647);

// hipFunction_attribute
static_assert(static_cast<long long>(HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK) == 0);
static_assert(static_cast<long long>(HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES) == 1);
static_assert(static_cast<long long>(HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES) == 2);
static_assert(static_cast<long long>(HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES) == 3);
static_assert(static_cast<long long>(HIP_FUNC_ATTRIBUTE_NUM_REGS) == 4);
static_assert(static_cast<long long>(HIP_FUNC_ATTRIBUTE_PTX_VERSION) == 5);
static_assert(static_cast<long long>(HIP_FUNC_ATTRIBUTE_BINARY_VERSION) == 6);
static_assert(static_cast<long long>(HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA) == 7);
static_assert(static_cast<long long>(HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES) == 8);
static_assert(static_cast<long long>(HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT) == 9);
static_assert(static_cast<long long>(HIP_FUNC_ATTRIBUTE_MAX) == 10);

// hipPointer_attribute
static_assert(static_cast<long long>(HIP_POINTER_ATTRIBUTE_CONTEXT) == 1);
static_assert(static_cast<long long>(HIP_POINTER_ATTRIBUTE_MEMORY_TYPE) == 2);
static_assert(static_cast<long long>(HIP_POINTER_ATTRIBUTE_DEVICE_POINTER) == 3);
static_assert(static_cast<long long>(HIP_POINTER_ATTRIBUTE_HOST_POINTER) == 4);
static_assert(static_cast<long long>(HIP_POINTER_ATTRIBUTE_P2P_TOKENS) == 5);
static_assert(static_cast<long long>(HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS) == 6);
static_assert(static_cast<long long>(HIP_POINTER_ATTRIBUTE_BUFFER_ID) == 7);
static_assert(static_cast<long long>(HIP_POINTER_ATTRIBUTE_IS_MANAGED) == 8);
static_assert(static_cast<long long>(HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL) == 9);
static_assert(static_cast<long long>(HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE) == 10);
static_assert(static_cast<long long>(HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR) == 11);
static_assert(static_cast<long long>(HIP_POINTER_ATTRIBUTE_RANGE_SIZE) == 12);
static_assert(static_cast<long long>(HIP_POINTER_ATTRIBUTE_MAPPED) == 13);
static_assert(static_cast<long long>(HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES) == 14);
static_assert(static_cast<long long>(HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE) == 15);
static_assert(static_cast<long long>(HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS) == 16);
static_assert(static_cast<long long>(HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE) == 17);

// hipTextureAddressMode
static_assert(static_cast<long long>(hipAddressModeWrap) == 0);
static_assert(static_cast<long long>(hipAddressModeClamp) == 1);
static_assert(static_cast<long long>(hipAddressModeMirror) == 2);
static_assert(static_cast<long long>(hipAddressModeBorder) == 3);

// hipTextureFilterMode
static_assert(static_cast<long long>(hipFilterModePoint) == 0);
static_assert(static_cast<long long>(hipFilterModeLinear) == 1);

// hipTextureReadMode
static_assert(static_cast<long long>(hipReadModeElementType) == 0);
static_assert(static_cast<long long>(hipReadModeNormalizedFloat) == 1);

// hipSurfaceBoundaryMode
static_assert(static_cast<long long>(hipBoundaryModeZero) == 0);
static_assert(static_cast<long long>(hipBoundaryModeTrap) == 1);
static_assert(static_cast<long long>(hipBoundaryModeClamp) == 2);

// ──────────────────────────────────────────────────────────────────────
// Opaque handle types
// ──────────────────────────────────────────────────────────────────────

// Almost all HIP opaque handles are pointer-sized struct pointers, same shape
// as CUDA's. Two differences verified here, not assumed:
//   * hipDevice_t is `int` (a plain ordinal), not a pointer -- same as CUDA's cudaDevice_t equivalent concept
//   * hipTextureObject_t / hipSurfaceObject_t are POINTERS in HIP
//     (struct __hip_texture* / struct __hip_surface*), unlike CUDA where
//     cudaTextureObject_t / cudaSurfaceObject_t are a 64-bit integral handle
static_assert(std::is_pointer_v<hipStream_t>);
static_assert(sizeof(hipStream_t) == sizeof(void *));
static_assert(std::is_pointer_v<hipEvent_t>);
static_assert(sizeof(hipEvent_t) == sizeof(void *));
static_assert(std::is_pointer_v<hipCtx_t>);
static_assert(sizeof(hipCtx_t) == sizeof(void *));
static_assert(std::is_pointer_v<hipModule_t>);
static_assert(sizeof(hipModule_t) == sizeof(void *));
static_assert(std::is_pointer_v<hipFunction_t>);
static_assert(sizeof(hipFunction_t) == sizeof(void *));
static_assert(std::is_pointer_v<hipLinkState_t>);
static_assert(sizeof(hipLinkState_t) == sizeof(void *));
static_assert(std::is_pointer_v<hipLibrary_t>);
static_assert(sizeof(hipLibrary_t) == sizeof(void *));
static_assert(std::is_pointer_v<hipKernel_t>);
static_assert(sizeof(hipKernel_t) == sizeof(void *));
static_assert(std::is_pointer_v<hipMemPool_t>);
static_assert(sizeof(hipMemPool_t) == sizeof(void *));
static_assert(std::is_pointer_v<hipGraph_t>);
static_assert(sizeof(hipGraph_t) == sizeof(void *));
static_assert(std::is_pointer_v<hipGraphNode_t>);
static_assert(sizeof(hipGraphNode_t) == sizeof(void *));
static_assert(std::is_pointer_v<hipGraphExec_t>);
static_assert(sizeof(hipGraphExec_t) == sizeof(void *));
static_assert(std::is_pointer_v<hipUserObject_t>);
static_assert(sizeof(hipUserObject_t) == sizeof(void *));
static_assert(std::is_same_v<hipDevice_t, int>);
static_assert(std::is_same_v<hipDeviceptr_t, void *>);
static_assert(std::is_pointer_v<hipArray_t>);
static_assert(sizeof(hipArray_t) == sizeof(void *));
static_assert(std::is_pointer_v<hipArray_const_t>);
static_assert(sizeof(hipArray_const_t) == sizeof(void *));
static_assert(std::is_pointer_v<hipMipmappedArray_t>);
static_assert(sizeof(hipMipmappedArray_t) == sizeof(void *));
static_assert(std::is_pointer_v<hipMipmappedArray_const_t>);
static_assert(sizeof(hipMipmappedArray_const_t) == sizeof(void *));
static_assert(std::is_pointer_v<hipMemGenericAllocationHandle_t>);
static_assert(sizeof(hipMemGenericAllocationHandle_t) == sizeof(void *));
static_assert(std::is_same_v<hipExternalMemory_t, void *>);
static_assert(std::is_same_v<hipExternalSemaphore_t, void *>);
static_assert(std::is_pointer_v<hipGraphicsResource_t>);
static_assert(sizeof(hipGraphicsResource_t) == sizeof(void *));
static_assert(std::is_pointer_v<hipTexRef>);
static_assert(sizeof(hipTexRef) == sizeof(void *));
static_assert(std::is_pointer_v<hipTextureObject_t>);
static_assert(sizeof(hipTextureObject_t) == sizeof(void *));
static_assert(std::is_pointer_v<hipSurfaceObject_t>);
static_assert(sizeof(hipSurfaceObject_t) == sizeof(void *));
static_assert(std::is_pointer_v<hipHostFn_t>);
static_assert(sizeof(hipHostFn_t) == sizeof(void *));
static_assert(std::is_pointer_v<hipStreamCallback_t>);
static_assert(sizeof(hipStreamCallback_t) == sizeof(void *));

// IPC handles are fixed-size byte arrays (64-byte ABI contract, HIP_IPC_HANDLE_SIZE)
static_assert(!std::is_pointer_v<hipIpcMemHandle_t>);
static_assert(!std::is_pointer_v<hipIpcEventHandle_t>);
static_assert(sizeof(hipIpcMemHandle_t) == 64);
static_assert(sizeof(hipIpcEventHandle_t) == 64);

// ──────────────────────────────────────────────────────────────────────
// Struct traits: trivial copyability and standard layout
// All HIP structs are passed by value / memcpy'd -- they must be trivially
// copyable and standard layout for C-interop. A failure here means our
// module wrapper broke the C-ABI contract.
// ──────────────────────────────────────────────────────────────────────

// Trivial copyability
static_assert(std::is_trivially_copyable_v<hipDeviceArch_t>);
static_assert(std::is_trivially_copyable_v<hipUUID>);
static_assert(std::is_trivially_copyable_v<hipDeviceProp_tR0600>);
static_assert(std::is_trivially_copyable_v<hipPointerAttribute_t>);
static_assert(std::is_trivially_copyable_v<hipFuncAttributes>);
static_assert(std::is_trivially_copyable_v<HIP_ARRAY_DESCRIPTOR>);
static_assert(std::is_trivially_copyable_v<HIP_ARRAY3D_DESCRIPTOR>);
static_assert(std::is_trivially_copyable_v<hip_Memcpy2D>);
static_assert(std::is_trivially_copyable_v<hipMipmappedArray>);
static_assert(std::is_trivially_copyable_v<HIP_TEXTURE_DESC>);
static_assert(std::is_trivially_copyable_v<hipResourceDesc>);
static_assert(std::is_trivially_copyable_v<HIP_RESOURCE_DESC>);
static_assert(std::is_trivially_copyable_v<hipResourceViewDesc>);
static_assert(std::is_trivially_copyable_v<HIP_RESOURCE_VIEW_DESC>);
static_assert(std::is_trivially_copyable_v<hipChannelFormatDesc>);
static_assert(std::is_trivially_copyable_v<textureReference>);
static_assert(std::is_trivially_copyable_v<hipTextureDesc>);
static_assert(std::is_trivially_copyable_v<hipPitchedPtr>);
static_assert(std::is_trivially_copyable_v<hipExtent>);
static_assert(std::is_trivially_copyable_v<hipPos>);
static_assert(std::is_trivially_copyable_v<hipMemcpy3DParms>);
static_assert(std::is_trivially_copyable_v<HIP_MEMCPY3D>);
static_assert(std::is_trivially_copyable_v<hipMemLocation>);
static_assert(std::is_trivially_copyable_v<hipMemcpyAttributes>);
static_assert(std::is_trivially_copyable_v<hipOffset3D>);
static_assert(std::is_trivially_copyable_v<hipMemcpy3DOperand>);
static_assert(std::is_trivially_copyable_v<hipMemcpy3DBatchOp>);
static_assert(std::is_trivially_copyable_v<hipMemcpy3DPeerParms>);
static_assert(std::is_trivially_copyable_v<dim3>);
static_assert(std::is_trivially_copyable_v<hipLaunchParams>);
static_assert(std::is_trivially_copyable_v<hipFunctionLaunchParams>);
static_assert(std::is_trivially_copyable_v<hipExternalMemoryHandleDesc>);
static_assert(std::is_trivially_copyable_v<hipExternalMemoryBufferDesc>);
static_assert(std::is_trivially_copyable_v<hipExternalMemoryMipmappedArrayDesc>);
static_assert(std::is_trivially_copyable_v<hipExternalSemaphoreHandleDesc>);
static_assert(std::is_trivially_copyable_v<hipExternalSemaphoreSignalParams>);
static_assert(std::is_trivially_copyable_v<hipExternalSemaphoreWaitParams>);
static_assert(std::is_trivially_copyable_v<hipHostNodeParams>);
static_assert(std::is_trivially_copyable_v<hipKernelNodeParams>);
static_assert(std::is_trivially_copyable_v<hipMemsetParams>);
static_assert(std::is_trivially_copyable_v<hipMemAllocNodeParams>);
static_assert(std::is_trivially_copyable_v<hipAccessPolicyWindow>);
static_assert(std::is_trivially_copyable_v<hipLaunchMemSyncDomainMap>);
static_assert(std::is_trivially_copyable_v<hipLaunchAttributeValue>);
static_assert(std::is_trivially_copyable_v<hipGraphInstantiateParams>);
static_assert(std::is_trivially_copyable_v<hipExternalSemaphoreSignalNodeParams>);
static_assert(std::is_trivially_copyable_v<hipExternalSemaphoreWaitNodeParams>);
static_assert(std::is_trivially_copyable_v<hipMemcpyNodeParams>);
static_assert(std::is_trivially_copyable_v<hipChildGraphNodeParams>);
static_assert(std::is_trivially_copyable_v<hipEventWaitNodeParams>);
static_assert(std::is_trivially_copyable_v<hipEventRecordNodeParams>);
static_assert(std::is_trivially_copyable_v<hipMemFreeNodeParams>);
static_assert(std::is_trivially_copyable_v<hipGraphNodeParams>);
static_assert(std::is_trivially_copyable_v<hipGraphEdgeData>);
static_assert(std::is_trivially_copyable_v<hipLaunchAttribute>);
static_assert(std::is_trivially_copyable_v<hipLaunchConfig_t>);
static_assert(std::is_trivially_copyable_v<HIP_LAUNCH_CONFIG>);
static_assert(std::is_trivially_copyable_v<hipMemPoolProps>);
static_assert(std::is_trivially_copyable_v<hipMemPoolPtrExportData>);
static_assert(std::is_trivially_copyable_v<hipMemAllocationProp>);
static_assert(std::is_trivially_copyable_v<hipMemAccessDesc>);
static_assert(std::is_trivially_copyable_v<hipStreamBatchMemOpParams>);
static_assert(std::is_trivially_copyable_v<hipBatchMemOpNodeParams>);
static_assert(std::is_trivially_copyable_v<hipArrayMapInfo>);
static_assert(std::is_trivially_copyable_v<hipIpcMemHandle_t>);
static_assert(std::is_trivially_copyable_v<hipIpcEventHandle_t>);

// Standard layout
static_assert(std::is_standard_layout_v<hipDeviceArch_t>);
static_assert(std::is_standard_layout_v<hipUUID>);
static_assert(std::is_standard_layout_v<hipDeviceProp_tR0600>);
static_assert(std::is_standard_layout_v<hipPointerAttribute_t>);
static_assert(std::is_standard_layout_v<hipFuncAttributes>);
static_assert(std::is_standard_layout_v<HIP_ARRAY_DESCRIPTOR>);
static_assert(std::is_standard_layout_v<HIP_ARRAY3D_DESCRIPTOR>);
static_assert(std::is_standard_layout_v<hip_Memcpy2D>);
static_assert(std::is_standard_layout_v<hipMipmappedArray>);
static_assert(std::is_standard_layout_v<HIP_TEXTURE_DESC>);
static_assert(std::is_standard_layout_v<hipResourceDesc>);
static_assert(std::is_standard_layout_v<HIP_RESOURCE_DESC>);
static_assert(std::is_standard_layout_v<hipResourceViewDesc>);
static_assert(std::is_standard_layout_v<HIP_RESOURCE_VIEW_DESC>);
static_assert(std::is_standard_layout_v<hipChannelFormatDesc>);
static_assert(std::is_standard_layout_v<textureReference>);
static_assert(std::is_standard_layout_v<hipTextureDesc>);
static_assert(std::is_standard_layout_v<hipPitchedPtr>);
static_assert(std::is_standard_layout_v<hipExtent>);
static_assert(std::is_standard_layout_v<hipPos>);
static_assert(std::is_standard_layout_v<hipMemcpy3DParms>);
static_assert(std::is_standard_layout_v<HIP_MEMCPY3D>);
static_assert(std::is_standard_layout_v<hipMemLocation>);
static_assert(std::is_standard_layout_v<hipMemcpyAttributes>);
static_assert(std::is_standard_layout_v<hipOffset3D>);
static_assert(std::is_standard_layout_v<hipMemcpy3DOperand>);
static_assert(std::is_standard_layout_v<hipMemcpy3DBatchOp>);
static_assert(std::is_standard_layout_v<hipMemcpy3DPeerParms>);
static_assert(std::is_standard_layout_v<dim3>);
static_assert(std::is_standard_layout_v<hipLaunchParams>);
static_assert(std::is_standard_layout_v<hipFunctionLaunchParams>);
static_assert(std::is_standard_layout_v<hipExternalMemoryHandleDesc>);
static_assert(std::is_standard_layout_v<hipExternalMemoryBufferDesc>);
static_assert(std::is_standard_layout_v<hipExternalMemoryMipmappedArrayDesc>);
static_assert(std::is_standard_layout_v<hipExternalSemaphoreHandleDesc>);
static_assert(std::is_standard_layout_v<hipExternalSemaphoreSignalParams>);
static_assert(std::is_standard_layout_v<hipExternalSemaphoreWaitParams>);
static_assert(std::is_standard_layout_v<hipHostNodeParams>);
static_assert(std::is_standard_layout_v<hipKernelNodeParams>);
static_assert(std::is_standard_layout_v<hipMemsetParams>);
static_assert(std::is_standard_layout_v<hipMemAllocNodeParams>);
static_assert(std::is_standard_layout_v<hipAccessPolicyWindow>);
static_assert(std::is_standard_layout_v<hipLaunchMemSyncDomainMap>);
static_assert(std::is_standard_layout_v<hipLaunchAttributeValue>);
static_assert(std::is_standard_layout_v<hipGraphInstantiateParams>);
static_assert(std::is_standard_layout_v<hipExternalSemaphoreSignalNodeParams>);
static_assert(std::is_standard_layout_v<hipExternalSemaphoreWaitNodeParams>);
static_assert(std::is_standard_layout_v<hipMemcpyNodeParams>);
static_assert(std::is_standard_layout_v<hipChildGraphNodeParams>);
static_assert(std::is_standard_layout_v<hipEventWaitNodeParams>);
static_assert(std::is_standard_layout_v<hipEventRecordNodeParams>);
static_assert(std::is_standard_layout_v<hipMemFreeNodeParams>);
static_assert(std::is_standard_layout_v<hipGraphNodeParams>);
static_assert(std::is_standard_layout_v<hipGraphEdgeData>);
static_assert(std::is_standard_layout_v<hipLaunchAttribute>);
static_assert(std::is_standard_layout_v<hipLaunchConfig_t>);
static_assert(std::is_standard_layout_v<HIP_LAUNCH_CONFIG>);
static_assert(std::is_standard_layout_v<hipMemPoolProps>);
static_assert(std::is_standard_layout_v<hipMemPoolPtrExportData>);
static_assert(std::is_standard_layout_v<hipMemAllocationProp>);
static_assert(std::is_standard_layout_v<hipMemAccessDesc>);
static_assert(std::is_standard_layout_v<hipStreamBatchMemOpParams>);
static_assert(std::is_standard_layout_v<hipBatchMemOpNodeParams>);
static_assert(std::is_standard_layout_v<hipArrayMapInfo>);
static_assert(std::is_standard_layout_v<hipIpcMemHandle_t>);
static_assert(std::is_standard_layout_v<hipIpcEventHandle_t>);

// ──────────────────────────────────────────────────────────────────────
// Link-time symbol resolution
// Forces the linker to resolve every re-exported function symbol, catching
// missing or unresolvable exports that type-only checks miss. The 3 names
// hip_runtime_api.h itself preprocessor-renames (hipGetDeviceProperties ->
// hipGetDevicePropertiesR0600, hipChooseDevice -> hipChooseDeviceR0600) are
// checked under their real, ABI-stable names -- the un-renamed spelling
// doesn't exist as a symbol once the macro has done its work.
// ──────────────────────────────────────────────────────────────────────

// Error Handling
WWR_LINK_CHECK(hipGetErrorName)
WWR_LINK_CHECK(hipGetErrorString)
WWR_LINK_CHECK(hipGetLastError)
WWR_LINK_CHECK(hipPeekAtLastError)
WWR_LINK_CHECK(hipExtGetLastError)
WWR_LINK_CHECK(hipDrvGetErrorName)
WWR_LINK_CHECK(hipDrvGetErrorString)

// Initialization, Version, and Driver Context Management
WWR_LINK_CHECK(hipInit)
WWR_LINK_CHECK(hipDriverGetVersion)
WWR_LINK_CHECK(hipRuntimeGetVersion)

// Deprecated by ROCm; re-exported with the [[deprecated]] attribute intact, so real
// consumers are still warned. These checks only prove the symbols still link, so silence it.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
WWR_LINK_CHECK(hipCtxCreate)
WWR_LINK_CHECK(hipCtxDestroy)
WWR_LINK_CHECK(hipCtxPopCurrent)
WWR_LINK_CHECK(hipCtxPushCurrent)
WWR_LINK_CHECK(hipCtxSetCurrent)
WWR_LINK_CHECK(hipCtxGetCurrent)
WWR_LINK_CHECK(hipCtxGetDevice)
WWR_LINK_CHECK(hipCtxGetApiVersion)
WWR_LINK_CHECK(hipCtxGetCacheConfig)
WWR_LINK_CHECK(hipCtxSetCacheConfig)
WWR_LINK_CHECK(hipCtxSetSharedMemConfig)
WWR_LINK_CHECK(hipCtxGetSharedMemConfig)
WWR_LINK_CHECK(hipCtxSynchronize)
WWR_LINK_CHECK(hipCtxGetFlags)
WWR_LINK_CHECK(hipCtxEnablePeerAccess)
WWR_LINK_CHECK(hipCtxDisablePeerAccess)
WWR_LINK_CHECK(hipDevicePrimaryCtxGetState)
WWR_LINK_CHECK(hipDevicePrimaryCtxRelease)
WWR_LINK_CHECK(hipDevicePrimaryCtxReset)
WWR_LINK_CHECK(hipDevicePrimaryCtxRetain)
WWR_LINK_CHECK(hipDevicePrimaryCtxSetFlags)
#pragma clang diagnostic pop
WWR_LINK_CHECK(hipSetValidDevices)

// Device Management
WWR_LINK_CHECK(hipGetDevice)
WWR_LINK_CHECK(hipSetDevice)
WWR_LINK_CHECK(hipGetDeviceCount)
WWR_LINK_CHECK(hipGetDevicePropertiesR0600)
WWR_LINK_CHECK(hipDeviceSynchronize)
WWR_LINK_CHECK(hipDeviceReset)
WWR_LINK_CHECK(hipSetDeviceFlags)
WWR_LINK_CHECK(hipGetDeviceFlags)
WWR_LINK_CHECK(hipChooseDeviceR0600)
WWR_LINK_CHECK(hipDeviceGetAttribute)
WWR_LINK_CHECK(hipDeviceGetByPCIBusId)
WWR_LINK_CHECK(hipDeviceGetPCIBusId)
WWR_LINK_CHECK(hipDeviceGet)
WWR_LINK_CHECK(hipDeviceComputeCapability)
WWR_LINK_CHECK(hipDeviceTotalMem)
WWR_LINK_CHECK(hipDeviceGetName)
WWR_LINK_CHECK(hipDeviceGetUuid)
WWR_LINK_CHECK(hipDeviceGetDefaultMemPool)
WWR_LINK_CHECK(hipDeviceSetMemPool)
WWR_LINK_CHECK(hipDeviceGetMemPool)
WWR_LINK_CHECK(hipDeviceSetLimit)
WWR_LINK_CHECK(hipDeviceGetLimit)
WWR_LINK_CHECK(hipDeviceSetCacheConfig)
WWR_LINK_CHECK(hipDeviceGetCacheConfig)
WWR_LINK_CHECK(hipDeviceSetSharedMemConfig)
WWR_LINK_CHECK(hipDeviceGetSharedMemConfig)
WWR_LINK_CHECK(hipDeviceGetStreamPriorityRange)
WWR_LINK_CHECK(hipDeviceGetTexture1DLinearMaxWidth)
WWR_LINK_CHECK(hipDeviceCanAccessPeer)
WWR_LINK_CHECK(hipDeviceEnablePeerAccess)
WWR_LINK_CHECK(hipDeviceDisablePeerAccess)
WWR_LINK_CHECK(hipDeviceGetP2PAttribute)
WWR_LINK_CHECK(hipDeviceGetGraphMemAttribute)
WWR_LINK_CHECK(hipDeviceSetGraphMemAttribute)
WWR_LINK_CHECK(hipDeviceGraphMemTrim)
WWR_LINK_CHECK(hipExtGetLinkTypeAndHopCount)

// Memory Management — Allocation / Free
WWR_LINK_CHECK(hipMalloc)
WWR_LINK_CHECK(hipMallocPitch)
WWR_LINK_CHECK(hipMalloc3D)
WWR_LINK_CHECK(hipMalloc3DArray)
WWR_LINK_CHECK(hipMallocArray)
WWR_LINK_CHECK(hipMallocMipmappedArray)
WWR_LINK_CHECK(hipMallocManaged)
WWR_LINK_CHECK(hipMallocAsync)
WWR_LINK_CHECK(hipMallocFromPoolAsync)
// Deprecated by ROCm; re-exported [[deprecated]] (consumers still warned), linkage-only check.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
WWR_LINK_CHECK(hipMallocHost)
#pragma clang diagnostic pop
WWR_LINK_CHECK(hipExtMallocWithFlags)
WWR_LINK_CHECK(hipHostAlloc)
WWR_LINK_CHECK(hipHostMalloc)
WWR_LINK_CHECK(hipHostFree)
WWR_LINK_CHECK(hipHostRegister)
WWR_LINK_CHECK(hipHostUnregister)
WWR_LINK_CHECK(hipHostGetDevicePointer)
WWR_LINK_CHECK(hipHostGetFlags)
WWR_LINK_CHECK(hipFree)
WWR_LINK_CHECK(hipFreeArray)
WWR_LINK_CHECK(hipFreeAsync)
WWR_LINK_CHECK(hipFreeMipmappedArray)
WWR_LINK_CHECK(hipFreeHost)
// Deprecated by ROCm; re-exported [[deprecated]] (consumers still warned), linkage-only check.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
WWR_LINK_CHECK(hipMemAllocHost)
#pragma clang diagnostic pop
WWR_LINK_CHECK(hipMemAllocPitch)

// Memory Management — Copy
WWR_LINK_CHECK(hipMemcpy)
WWR_LINK_CHECK(hipMemcpyAsync)
WWR_LINK_CHECK(hipMemcpy2D)
WWR_LINK_CHECK(hipMemcpy2DAsync)
WWR_LINK_CHECK(hipMemcpy2DToArray)
WWR_LINK_CHECK(hipMemcpy2DToArrayAsync)
WWR_LINK_CHECK(hipMemcpy2DFromArray)
WWR_LINK_CHECK(hipMemcpy2DFromArrayAsync)
WWR_LINK_CHECK(hipMemcpy2DArrayToArray)
WWR_LINK_CHECK(hipMemcpy3D)
WWR_LINK_CHECK(hipMemcpy3DAsync)
WWR_LINK_CHECK(hipMemcpy3DPeer)
WWR_LINK_CHECK(hipMemcpy3DPeerAsync)
WWR_LINK_CHECK(hipMemcpy3DBatchAsync)
WWR_LINK_CHECK(hipMemcpyPeer)
WWR_LINK_CHECK(hipMemcpyPeerAsync)
// Deprecated by ROCm; re-exported [[deprecated]] (consumers still warned), linkage-only checks.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
WWR_LINK_CHECK(hipMemcpyToArray)
WWR_LINK_CHECK(hipMemcpyFromArray)
#pragma clang diagnostic pop
WWR_LINK_CHECK(hipMemcpyToSymbol)
WWR_LINK_CHECK(hipMemcpyToSymbolAsync)
WWR_LINK_CHECK(hipMemcpyFromSymbol)
WWR_LINK_CHECK(hipMemcpyFromSymbolAsync)
WWR_LINK_CHECK(hipMemcpyWithStream)
WWR_LINK_CHECK(hipMemcpyBatchAsync)
WWR_LINK_CHECK(hipMemcpyParam2D)
WWR_LINK_CHECK(hipMemcpyParam2DAsync)
WWR_LINK_CHECK(hipMemcpyAtoA)
WWR_LINK_CHECK(hipMemcpyAtoD)
WWR_LINK_CHECK(hipMemcpyAtoH)
WWR_LINK_CHECK(hipMemcpyAtoHAsync)
WWR_LINK_CHECK(hipMemcpyDtoA)
WWR_LINK_CHECK(hipMemcpyDtoD)
WWR_LINK_CHECK(hipMemcpyDtoDAsync)
WWR_LINK_CHECK(hipMemcpyDtoH)
WWR_LINK_CHECK(hipMemcpyDtoHAsync)
WWR_LINK_CHECK(hipMemcpyHtoA)
WWR_LINK_CHECK(hipMemcpyHtoAAsync)
WWR_LINK_CHECK(hipMemcpyHtoD)
WWR_LINK_CHECK(hipMemcpyHtoDAsync)
WWR_LINK_CHECK(hipDrvMemcpy2DUnaligned)
WWR_LINK_CHECK(hipDrvMemcpy3D)
WWR_LINK_CHECK(hipDrvMemcpy3DAsync)

// Memory Management — Set
WWR_LINK_CHECK(hipMemset)
WWR_LINK_CHECK(hipMemsetAsync)
WWR_LINK_CHECK(hipMemset2D)
WWR_LINK_CHECK(hipMemset2DAsync)
WWR_LINK_CHECK(hipMemset3D)
WWR_LINK_CHECK(hipMemset3DAsync)
WWR_LINK_CHECK(hipMemsetD8)
WWR_LINK_CHECK(hipMemsetD8Async)
WWR_LINK_CHECK(hipMemsetD16)
WWR_LINK_CHECK(hipMemsetD16Async)
WWR_LINK_CHECK(hipMemsetD32)
WWR_LINK_CHECK(hipMemsetD32Async)
WWR_LINK_CHECK(hipMemsetD2D8)
WWR_LINK_CHECK(hipMemsetD2D8Async)
WWR_LINK_CHECK(hipMemsetD2D16)
WWR_LINK_CHECK(hipMemsetD2D16Async)
WWR_LINK_CHECK(hipMemsetD2D32)
WWR_LINK_CHECK(hipMemsetD2D32Async)

// Memory Management — Info / Advise / Query / Pointer Attributes
WWR_LINK_CHECK(hipMemGetInfo)
WWR_LINK_CHECK(hipMemPtrGetInfo)
WWR_LINK_CHECK(hipMemPrefetchAsync)
WWR_LINK_CHECK(hipMemPrefetchAsync_v2)
WWR_LINK_CHECK(hipMemAdvise)
WWR_LINK_CHECK(hipMemAdvise_v2)
WWR_LINK_CHECK(hipMemRangeGetAttribute)
WWR_LINK_CHECK(hipMemRangeGetAttributes)
WWR_LINK_CHECK(hipPointerGetAttribute)
WWR_LINK_CHECK(hipPointerGetAttributes)
WWR_LINK_CHECK(hipPointerSetAttribute)
WWR_LINK_CHECK(hipDrvPointerGetAttributes)
WWR_LINK_CHECK(hipMemGetAddressRange)
WWR_LINK_CHECK(hipMemGetHandleForAddressRange)

// Memory Management — Arrays
WWR_LINK_CHECK(hipArrayCreate)
WWR_LINK_CHECK(hipArrayDestroy)
WWR_LINK_CHECK(hipArrayGetDescriptor)
WWR_LINK_CHECK(hipArray3DCreate)
WWR_LINK_CHECK(hipArray3DGetDescriptor)
WWR_LINK_CHECK(hipArrayGetInfo)
WWR_LINK_CHECK(hipGetMipmappedArrayLevel)
WWR_LINK_CHECK(hipMipmappedArrayCreate)
WWR_LINK_CHECK(hipMipmappedArrayDestroy)
WWR_LINK_CHECK(hipMipmappedArrayGetLevel)

// Memory Management — Virtual Memory / Pools
WWR_LINK_CHECK(hipMemAddressFree)
WWR_LINK_CHECK(hipMemAddressReserve)
WWR_LINK_CHECK(hipMemCreate)
WWR_LINK_CHECK(hipMemRelease)
WWR_LINK_CHECK(hipMemMap)
WWR_LINK_CHECK(hipMemMapArrayAsync)
WWR_LINK_CHECK(hipMemUnmap)
WWR_LINK_CHECK(hipMemSetAccess)
WWR_LINK_CHECK(hipMemGetAccess)
WWR_LINK_CHECK(hipMemExportToShareableHandle)
WWR_LINK_CHECK(hipMemImportFromShareableHandle)
WWR_LINK_CHECK(hipMemGetAllocationGranularity)
WWR_LINK_CHECK(hipMemGetAllocationPropertiesFromHandle)
WWR_LINK_CHECK(hipMemRetainAllocationHandle)
WWR_LINK_CHECK(hipMemPoolCreate)
WWR_LINK_CHECK(hipMemPoolDestroy)
WWR_LINK_CHECK(hipMemPoolExportPointer)
WWR_LINK_CHECK(hipMemPoolExportToShareableHandle)
WWR_LINK_CHECK(hipMemPoolGetAccess)
WWR_LINK_CHECK(hipMemPoolGetAttribute)
WWR_LINK_CHECK(hipMemPoolImportFromShareableHandle)
WWR_LINK_CHECK(hipMemPoolImportPointer)
WWR_LINK_CHECK(hipMemPoolSetAccess)
WWR_LINK_CHECK(hipMemPoolSetAttribute)
WWR_LINK_CHECK(hipMemPoolTrimTo)

// Memory Management — External / IPC
WWR_LINK_CHECK(hipImportExternalMemory)
WWR_LINK_CHECK(hipExternalMemoryGetMappedBuffer)
WWR_DECLARED_CHECK(
    hipExternalMemoryGetMappedMipmappedArray) // not exported by libamdhip64.so.7.2.70204
WWR_LINK_CHECK(hipDestroyExternalMemory)
WWR_LINK_CHECK(hipIpcCloseMemHandle)
WWR_LINK_CHECK(hipIpcGetEventHandle)
WWR_LINK_CHECK(hipIpcGetMemHandle)
WWR_LINK_CHECK(hipIpcOpenEventHandle)
WWR_LINK_CHECK(hipIpcOpenMemHandle)

// Stream Management
WWR_LINK_CHECK(hipStreamCreate)
WWR_LINK_CHECK(hipStreamCreateWithFlags)
WWR_LINK_CHECK(hipStreamCreateWithPriority)
WWR_LINK_CHECK(hipStreamDestroy)
WWR_LINK_CHECK(hipStreamSynchronize)
WWR_LINK_CHECK(hipStreamWaitEvent)
WWR_LINK_CHECK(hipStreamQuery)
WWR_LINK_CHECK(hipStreamGetFlags)
WWR_LINK_CHECK(hipStreamGetPriority)
WWR_LINK_CHECK(hipStreamGetId)
WWR_LINK_CHECK(hipStreamGetDevice)
WWR_LINK_CHECK(hipStreamAddCallback)
WWR_LINK_CHECK(hipStreamAttachMemAsync)
WWR_LINK_CHECK(hipStreamBeginCapture)
WWR_LINK_CHECK(hipStreamBeginCaptureToGraph)
WWR_LINK_CHECK(hipStreamEndCapture)
WWR_LINK_CHECK(hipStreamIsCapturing)
WWR_LINK_CHECK(hipStreamGetCaptureInfo)
WWR_LINK_CHECK(hipStreamGetCaptureInfo_v2)
WWR_LINK_CHECK(hipStreamUpdateCaptureDependencies)
WWR_LINK_CHECK(hipThreadExchangeStreamCaptureMode)
#if WWR_HIP_SINCE_7_2
WWR_LINK_CHECK(hipStreamCopyAttributes)
#endif
WWR_LINK_CHECK(hipStreamGetAttribute)
WWR_LINK_CHECK(hipStreamSetAttribute)
WWR_LINK_CHECK(hipStreamBatchMemOp)
WWR_LINK_CHECK(hipStreamWaitValue32)
WWR_LINK_CHECK(hipStreamWaitValue64)
WWR_LINK_CHECK(hipStreamWriteValue32)
WWR_LINK_CHECK(hipStreamWriteValue64)
WWR_LINK_CHECK(hipExtStreamCreateWithCUMask)
WWR_LINK_CHECK(hipExtStreamGetCUMask)
WWR_LINK_CHECK(hipGetStreamDeviceId)

// Event Management
WWR_LINK_CHECK(hipEventCreate)
WWR_LINK_CHECK(hipEventCreateWithFlags)
WWR_LINK_CHECK(hipEventDestroy)
WWR_LINK_CHECK(hipEventRecord)
WWR_LINK_CHECK(hipEventRecordWithFlags)
WWR_LINK_CHECK(hipEventSynchronize)
WWR_LINK_CHECK(hipEventQuery)
WWR_LINK_CHECK(hipEventElapsedTime)

// External Semaphores
WWR_LINK_CHECK(hipImportExternalSemaphore)
WWR_LINK_CHECK(hipSignalExternalSemaphoresAsync)
WWR_LINK_CHECK(hipWaitExternalSemaphoresAsync)
WWR_LINK_CHECK(hipDestroyExternalSemaphore)

// Execution Control (Kernel Launch)
WWR_LINK_CHECK(hipLaunchKernel)
WWR_LINK_CHECK(hipLaunchKernelExC)
WWR_LINK_CHECK(hipLaunchCooperativeKernel)
WWR_LINK_CHECK(hipLaunchCooperativeKernelMultiDevice)
WWR_LINK_CHECK(hipLaunchHostFunc)
WWR_LINK_CHECK(hipConfigureCall)
WWR_LINK_CHECK(hipSetupArgument)
WWR_LINK_CHECK(hipLaunchByPtr)
WWR_LINK_CHECK(hipExtLaunchKernel)
WWR_LINK_CHECK(hipExtLaunchMultiKernelMultiDevice)
WWR_LINK_CHECK(hipDrvLaunchKernelEx)
WWR_LINK_CHECK(hipModuleLaunchKernel)
WWR_LINK_CHECK(hipModuleLaunchCooperativeKernel)
WWR_LINK_CHECK(hipModuleLaunchCooperativeKernelMultiDevice)
WWR_LINK_CHECK(hipKernelNameRef)
WWR_LINK_CHECK(hipKernelNameRefByPtr)

// Occupancy and Function Configuration
WWR_LINK_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor)
WWR_LINK_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags)
WWR_LINK_CHECK(hipOccupancyMaxPotentialBlockSize)
#if WWR_HIP_SINCE_7_2
WWR_LINK_CHECK(hipOccupancyAvailableDynamicSMemPerBlock)
#endif
WWR_LINK_CHECK(hipModuleOccupancyMaxActiveBlocksPerMultiprocessor)
WWR_LINK_CHECK(hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags)
WWR_LINK_CHECK(hipModuleOccupancyMaxPotentialBlockSize)
WWR_LINK_CHECK(hipModuleOccupancyMaxPotentialBlockSizeWithFlags)
WWR_LINK_CHECK(hipFuncGetAttribute)
WWR_LINK_CHECK(hipFuncGetAttributes)
WWR_LINK_CHECK(hipFuncSetAttribute)
WWR_LINK_CHECK(hipFuncSetCacheConfig)
WWR_LINK_CHECK(hipFuncSetSharedMemConfig)

// Symbol and Kernel Management
WWR_LINK_CHECK(hipGetSymbolAddress)
WWR_LINK_CHECK(hipGetSymbolSize)
WWR_LINK_CHECK(hipGetFuncBySymbol)

// Module / Library / Linker Management
WWR_LINK_CHECK(hipModuleLoad)
WWR_LINK_CHECK(hipModuleLoadData)
WWR_LINK_CHECK(hipModuleLoadDataEx)
WWR_LINK_CHECK(hipModuleLoadFatBinary)
WWR_LINK_CHECK(hipModuleUnload)
WWR_LINK_CHECK(hipModuleGetFunction)
WWR_LINK_CHECK(hipModuleGetFunctionCount)
WWR_LINK_CHECK(hipModuleGetGlobal)
WWR_LINK_CHECK(hipModuleGetTexRef)
WWR_LINK_CHECK(hipLibraryLoadData)
WWR_LINK_CHECK(hipLibraryLoadFromFile)
WWR_LINK_CHECK(hipLibraryUnload)
WWR_LINK_CHECK(hipLibraryGetKernel)
WWR_LINK_CHECK(hipLibraryGetKernelCount)
#if WWR_HIP_SINCE_7_2
WWR_LINK_CHECK(hipLibraryEnumerateKernels)
WWR_LINK_CHECK(hipKernelGetLibrary)
WWR_LINK_CHECK(hipKernelGetName)
#endif
WWR_LINK_CHECK(hipLinkAddData)
WWR_LINK_CHECK(hipLinkAddFile)
WWR_LINK_CHECK(hipLinkComplete)
WWR_LINK_CHECK(hipLinkCreate)
WWR_LINK_CHECK(hipLinkDestroy)
WWR_LINK_CHECK(hipApiName)

// Graph Management — Lifecycle / Dependencies
WWR_LINK_CHECK(hipGraphCreate)
WWR_LINK_CHECK(hipGraphDestroy)
WWR_LINK_CHECK(hipGraphAddDependencies)
WWR_LINK_CHECK(hipGraphRemoveDependencies)
WWR_LINK_CHECK(hipGraphGetEdges)
WWR_LINK_CHECK(hipGraphGetNodes)
WWR_LINK_CHECK(hipGraphGetRootNodes)
WWR_LINK_CHECK(hipGraphNodeGetDependencies)
WWR_LINK_CHECK(hipGraphNodeGetDependentNodes)
WWR_LINK_CHECK(hipGraphNodeGetType)
WWR_LINK_CHECK(hipGraphNodeGetEnabled)
WWR_LINK_CHECK(hipGraphNodeSetEnabled)
WWR_LINK_CHECK(hipGraphClone)
WWR_LINK_CHECK(hipGraphNodeFindInClone)
WWR_LINK_CHECK(hipGraphDebugDotPrint)
WWR_LINK_CHECK(hipGraphDestroyNode)

// Graph Management — Node Addition
WWR_LINK_CHECK(hipGraphAddEmptyNode)
WWR_LINK_CHECK(hipGraphAddKernelNode)
WWR_LINK_CHECK(hipGraphAddMemcpyNode)
WWR_LINK_CHECK(hipGraphAddMemcpyNode1D)
WWR_LINK_CHECK(hipGraphAddMemcpyNodeFromSymbol)
WWR_LINK_CHECK(hipGraphAddMemcpyNodeToSymbol)
WWR_LINK_CHECK(hipGraphAddMemsetNode)
WWR_LINK_CHECK(hipGraphAddHostNode)
WWR_LINK_CHECK(hipGraphAddChildGraphNode)
WWR_LINK_CHECK(hipGraphAddEventRecordNode)
WWR_LINK_CHECK(hipGraphAddEventWaitNode)
WWR_LINK_CHECK(hipGraphAddExternalSemaphoresSignalNode)
WWR_LINK_CHECK(hipGraphAddExternalSemaphoresWaitNode)
WWR_LINK_CHECK(hipGraphAddMemAllocNode)
WWR_LINK_CHECK(hipGraphAddMemFreeNode)
WWR_LINK_CHECK(hipGraphAddNode)
WWR_LINK_CHECK(hipGraphAddBatchMemOpNode)
WWR_LINK_CHECK(hipDrvGraphAddMemFreeNode)
WWR_LINK_CHECK(hipDrvGraphAddMemcpyNode)
WWR_LINK_CHECK(hipDrvGraphAddMemsetNode)

// Graph Management — Node Params
WWR_LINK_CHECK(hipGraphKernelNodeGetParams)
WWR_LINK_CHECK(hipGraphKernelNodeSetParams)
WWR_LINK_CHECK(hipGraphKernelNodeCopyAttributes)
WWR_LINK_CHECK(hipGraphKernelNodeGetAttribute)
WWR_LINK_CHECK(hipGraphKernelNodeSetAttribute)
WWR_LINK_CHECK(hipGraphMemcpyNodeGetParams)
WWR_LINK_CHECK(hipGraphMemcpyNodeSetParams)
WWR_LINK_CHECK(hipGraphMemcpyNodeSetParams1D)
WWR_LINK_CHECK(hipGraphMemcpyNodeSetParamsFromSymbol)
WWR_LINK_CHECK(hipGraphMemcpyNodeSetParamsToSymbol)
WWR_LINK_CHECK(hipGraphMemsetNodeGetParams)
WWR_LINK_CHECK(hipGraphMemsetNodeSetParams)
WWR_LINK_CHECK(hipGraphHostNodeGetParams)
WWR_LINK_CHECK(hipGraphHostNodeSetParams)
WWR_LINK_CHECK(hipGraphChildGraphNodeGetGraph)
WWR_LINK_CHECK(hipGraphEventRecordNodeGetEvent)
WWR_LINK_CHECK(hipGraphEventRecordNodeSetEvent)
WWR_LINK_CHECK(hipGraphEventWaitNodeGetEvent)
WWR_LINK_CHECK(hipGraphEventWaitNodeSetEvent)
WWR_LINK_CHECK(hipGraphExternalSemaphoresSignalNodeGetParams)
WWR_LINK_CHECK(hipGraphExternalSemaphoresSignalNodeSetParams)
WWR_LINK_CHECK(hipGraphExternalSemaphoresWaitNodeGetParams)
WWR_LINK_CHECK(hipGraphExternalSemaphoresWaitNodeSetParams)
WWR_LINK_CHECK(hipGraphMemAllocNodeGetParams)
WWR_LINK_CHECK(hipGraphMemFreeNodeGetParams)
WWR_LINK_CHECK(hipGraphNodeSetParams)
WWR_LINK_CHECK(hipGraphBatchMemOpNodeGetParams)
WWR_LINK_CHECK(hipGraphBatchMemOpNodeSetParams)
WWR_LINK_CHECK(hipDrvGraphMemcpyNodeGetParams)
WWR_LINK_CHECK(hipDrvGraphMemcpyNodeSetParams)

// Graph Management — Execution
WWR_LINK_CHECK(hipGraphInstantiate)
WWR_LINK_CHECK(hipGraphInstantiateWithFlags)
WWR_LINK_CHECK(hipGraphInstantiateWithParams)
WWR_LINK_CHECK(hipGraphExecDestroy)
WWR_LINK_CHECK(hipGraphExecUpdate)
WWR_LINK_CHECK(hipGraphExecGetFlags)
WWR_LINK_CHECK(hipGraphLaunch)
WWR_LINK_CHECK(hipGraphUpload)
WWR_LINK_CHECK(hipGraphExecKernelNodeSetParams)
WWR_LINK_CHECK(hipGraphExecMemcpyNodeSetParams)
WWR_LINK_CHECK(hipGraphExecMemcpyNodeSetParams1D)
WWR_LINK_CHECK(hipGraphExecMemcpyNodeSetParamsFromSymbol)
WWR_LINK_CHECK(hipGraphExecMemcpyNodeSetParamsToSymbol)
WWR_LINK_CHECK(hipGraphExecMemsetNodeSetParams)
WWR_LINK_CHECK(hipGraphExecHostNodeSetParams)
WWR_LINK_CHECK(hipGraphExecChildGraphNodeSetParams)
WWR_LINK_CHECK(hipGraphExecEventRecordNodeSetEvent)
WWR_LINK_CHECK(hipGraphExecEventWaitNodeSetEvent)
WWR_LINK_CHECK(hipGraphExecExternalSemaphoresSignalNodeSetParams)
WWR_LINK_CHECK(hipGraphExecExternalSemaphoresWaitNodeSetParams)
WWR_LINK_CHECK(hipGraphExecNodeSetParams)
WWR_LINK_CHECK(hipGraphExecBatchMemOpNodeSetParams)
WWR_LINK_CHECK(hipDrvGraphExecMemcpyNodeSetParams)
WWR_LINK_CHECK(hipDrvGraphExecMemsetNodeSetParams)

// Graph Management — User Objects
WWR_LINK_CHECK(hipGraphRetainUserObject)
WWR_LINK_CHECK(hipGraphReleaseUserObject)
WWR_LINK_CHECK(hipUserObjectCreate)
WWR_LINK_CHECK(hipUserObjectRetain)
WWR_LINK_CHECK(hipUserObjectRelease)

// Texture / Surface Management
WWR_LINK_CHECK(hipCreateTextureObject)
WWR_LINK_CHECK(hipDestroyTextureObject)
WWR_LINK_CHECK(hipGetTextureObjectResourceDesc)
WWR_LINK_CHECK(hipGetTextureObjectResourceViewDesc)
WWR_LINK_CHECK(hipGetTextureObjectTextureDesc)
WWR_LINK_CHECK(hipCreateSurfaceObject)
WWR_LINK_CHECK(hipDestroySurfaceObject)
WWR_LINK_CHECK(hipGetChannelDesc)
WWR_LINK_CHECK(hipBindTexture)
WWR_LINK_CHECK(hipBindTexture2D)
WWR_LINK_CHECK(hipBindTextureToArray)
WWR_LINK_CHECK(hipBindTextureToMipmappedArray)
WWR_LINK_CHECK(hipUnbindTexture)
// Deprecated by ROCm; re-exported [[deprecated]] (consumers still warned), linkage-only checks.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
WWR_LINK_CHECK(hipGetTextureAlignmentOffset)
WWR_LINK_CHECK(hipGetTextureReference)
#pragma clang diagnostic pop
WWR_LINK_CHECK(hipTexObjectCreate)
WWR_LINK_CHECK(hipTexObjectDestroy)
WWR_LINK_CHECK(hipTexObjectGetResourceDesc)
WWR_LINK_CHECK(hipTexObjectGetResourceViewDesc)
WWR_LINK_CHECK(hipTexObjectGetTextureDesc)
// Deprecated by ROCm; re-exported [[deprecated]] (consumers still warned), linkage-only checks.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
WWR_LINK_CHECK(hipTexRefGetAddress)
WWR_LINK_CHECK(hipTexRefGetAddressMode)
WWR_LINK_CHECK(hipTexRefGetArray)
WWR_LINK_CHECK(hipTexRefGetBorderColor)
WWR_LINK_CHECK(hipTexRefGetFilterMode)
WWR_LINK_CHECK(hipTexRefGetFlags)
WWR_LINK_CHECK(hipTexRefGetFormat)
WWR_LINK_CHECK(hipTexRefGetMaxAnisotropy)
WWR_LINK_CHECK(hipTexRefGetMipMappedArray)
WWR_LINK_CHECK(hipTexRefGetMipmapFilterMode)
WWR_LINK_CHECK(hipTexRefGetMipmapLevelBias)
WWR_LINK_CHECK(hipTexRefGetMipmapLevelClamp)
WWR_LINK_CHECK(hipTexRefSetAddress)
WWR_LINK_CHECK(hipTexRefSetAddress2D)
WWR_LINK_CHECK(hipTexRefSetAddressMode)
WWR_LINK_CHECK(hipTexRefSetArray)
WWR_LINK_CHECK(hipTexRefSetBorderColor)
WWR_LINK_CHECK(hipTexRefSetFilterMode)
WWR_LINK_CHECK(hipTexRefSetFlags)
WWR_LINK_CHECK(hipTexRefSetFormat)
WWR_LINK_CHECK(hipTexRefSetMaxAnisotropy)
WWR_LINK_CHECK(hipTexRefSetMipmapFilterMode)
WWR_LINK_CHECK(hipTexRefSetMipmapLevelBias)
WWR_LINK_CHECK(hipTexRefSetMipmapLevelClamp)
WWR_LINK_CHECK(hipTexRefSetMipmappedArray)
#pragma clang diagnostic pop

// Graphics Interoperability
WWR_LINK_CHECK(hipGraphicsMapResources)
WWR_LINK_CHECK(hipGraphicsUnmapResources)
WWR_LINK_CHECK(hipGraphicsUnregisterResource)
WWR_LINK_CHECK(hipGraphicsResourceGetMappedPointer)
WWR_LINK_CHECK(hipGraphicsSubResourceGetMappedArray)

// Driver Entry Points / Proc Address
WWR_LINK_CHECK(hipGetDriverEntryPoint)
WWR_LINK_CHECK(hipGetProcAddress)

// Profiler Control
// Deprecated by ROCm; re-exported [[deprecated]] (consumers still warned), linkage-only checks.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
WWR_LINK_CHECK(hipProfilerStart)
WWR_LINK_CHECK(hipProfilerStop)
#pragma clang diagnostic pop

} // namespace wwr::hip::test
