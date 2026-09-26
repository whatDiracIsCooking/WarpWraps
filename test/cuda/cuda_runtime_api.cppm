// cuda_runtime_api.cppm - Compile-time tests for gpumod.cuda.cuda_runtime_api

module;

#include "test/shared/link_check.h"

export module gpumod.test.cuda.cuda_runtime_api;

import std;
import gpumod.cuda.cuda_runtime_api;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time tests for gpumod.cuda.cuda_runtime_api
//
// The module is pure re-export (using declarations + constexpr flag values).
// Runtime tests for the underlying CUDA API would just test CUDA itself.
// We verify at compile-time that:
//   1. Constexpr flags have the correct values (the #undef/re-export is correct)
//   2. Key enum values with CUDA-specified values are correct
//   3. Opaque handle types have the expected type traits
//   4. Struct types satisfy trivial copyability and standard layout (C-interop guarantee)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace gpumod::cuda::test {

using namespace gpumod::cuda;

// ────────────────────────────────────────────────────────────────────────
// Constexpr flag values
// ────────────────────────────────────────────────────────────────────────

// cudaHostAlloc flags
static_assert(cudaHostAllocDefault == 0x00u);
static_assert(cudaHostAllocPortable == 0x01u);
static_assert(cudaHostAllocMapped == 0x02u);
static_assert(cudaHostAllocWriteCombined == 0x04u);

// cudaEvent flags
static_assert(cudaEventDefault == 0x00u);
static_assert(cudaEventBlockingSync == 0x01u);
static_assert(cudaEventDisableTiming == 0x02u);
static_assert(cudaEventInterprocess == 0x04u);

// cudaStream flags
static_assert(cudaStreamDefault == 0x00u);
static_assert(cudaStreamNonBlocking == 0x01u);

// cudaMallocManaged flags
static_assert(cudaMemAttachGlobal == 0x01u);
static_assert(cudaMemAttachHost == 0x02u);

// ────────────────────────────────────────────────────────────────────────
// Enum type checks
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_enum_v<cudaError_t>);
static_assert(std::is_enum_v<cudaMemcpyKind>);
static_assert(std::is_enum_v<cudaMemoryType>);
static_assert(std::is_enum_v<cudaChannelFormatKind>);
static_assert(std::is_enum_v<cudaComputeMode>);
static_assert(std::is_enum_v<cudaDeviceAttr>);
static_assert(std::is_enum_v<cudaLimit>);
static_assert(std::is_enum_v<cudaFuncCache>);
static_assert(std::is_enum_v<cudaFuncAttribute>);
static_assert(std::is_enum_v<cudaSharedMemConfig>);
static_assert(std::is_enum_v<cudaStreamCaptureStatus>);
static_assert(std::is_enum_v<cudaStreamCaptureMode>);
static_assert(std::is_enum_v<cudaGraphNodeType>);
static_assert(std::is_enum_v<cudaGraphExecUpdateResult>);
static_assert(std::is_enum_v<cudaMemPoolAttr>);
static_assert(std::is_enum_v<cudaMemAllocationType>);
static_assert(std::is_enum_v<cudaMemLocationType>);
static_assert(std::is_enum_v<cudaMemAllocationHandleType>);
static_assert(std::is_enum_v<cudaMemoryAdvise>);
static_assert(std::is_enum_v<cudaMemRangeAttribute>);
static_assert(std::is_enum_v<cudaAccessProperty>);
static_assert(std::is_enum_v<cudaResourceType>);
static_assert(std::is_enum_v<cudaResourceViewFormat>);
static_assert(std::is_enum_v<cudaSynchronizationPolicy>);
static_assert(std::is_enum_v<cudaExternalMemoryHandleType>);
static_assert(std::is_enum_v<cudaExternalSemaphoreHandleType>);
static_assert(std::is_enum_v<cudaGraphicsRegisterFlags>);
static_assert(std::is_enum_v<cudaGraphicsMapFlags>);
static_assert(std::is_enum_v<cudaUserObjectFlags>);
static_assert(std::is_enum_v<cudaUserObjectRetainFlags>);
static_assert(std::is_enum_v<cudaDeviceP2PAttr>);
static_assert(std::is_enum_v<cudaStreamUpdateCaptureDependenciesFlags>);
static_assert(std::is_enum_v<cudaLaunchAttributeID>);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cudaError_t
// ────────────────────────────────────────────────────────────────────────

// cudaSuccess == 0 is a guaranteed CUDA API contract
static_assert(static_cast<int>(cudaSuccess) == 0);
static_assert(static_cast<int>(cudaErrorInvalidValue) == 1);
static_assert(static_cast<int>(cudaErrorMemoryAllocation) == 2);
static_assert(static_cast<int>(cudaErrorInitializationError) == 3);
static_assert(static_cast<int>(cudaErrorInvalidConfiguration) == 9);
static_assert(static_cast<int>(cudaErrorInvalidDevicePointer) == 17);
static_assert(static_cast<int>(cudaErrorInvalidMemcpyDirection) == 21);
static_assert(static_cast<int>(cudaErrorNotReady) == 600);
static_assert(static_cast<int>(cudaErrorDeviceAlreadyInUse) == 216);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cudaMemcpyKind
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(cudaMemcpyHostToHost) == 0);
static_assert(static_cast<int>(cudaMemcpyHostToDevice) == 1);
static_assert(static_cast<int>(cudaMemcpyDeviceToHost) == 2);
static_assert(static_cast<int>(cudaMemcpyDeviceToDevice) == 3);
static_assert(static_cast<int>(cudaMemcpyDefault) == 4);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cudaMemoryType
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(cudaMemoryTypeUnregistered) == 0);
static_assert(static_cast<int>(cudaMemoryTypeHost) == 1);
static_assert(static_cast<int>(cudaMemoryTypeDevice) == 2);
static_assert(static_cast<int>(cudaMemoryTypeManaged) == 3);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cudaChannelFormatKind
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(cudaChannelFormatKindSigned) == 0);
static_assert(static_cast<int>(cudaChannelFormatKindUnsigned) == 1);
static_assert(static_cast<int>(cudaChannelFormatKindFloat) == 2);
static_assert(static_cast<int>(cudaChannelFormatKindNone) == 3);
static_assert(static_cast<int>(cudaChannelFormatKindNV12) == 4);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cudaComputeMode
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(cudaComputeModeDefault) == 0);
static_assert(static_cast<int>(cudaComputeModeExclusive) == 1);
static_assert(static_cast<int>(cudaComputeModeProhibited) == 2);
static_assert(static_cast<int>(cudaComputeModeExclusiveProcess) == 3);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cudaLimit
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(cudaLimitStackSize) == 0x00);
static_assert(static_cast<int>(cudaLimitPrintfFifoSize) == 0x01);
static_assert(static_cast<int>(cudaLimitMallocHeapSize) == 0x02);
static_assert(static_cast<int>(cudaLimitDevRuntimeSyncDepth) == 0x03);
static_assert(static_cast<int>(cudaLimitDevRuntimePendingLaunchCount) == 0x04);
static_assert(static_cast<int>(cudaLimitMaxL2FetchGranularity) == 0x05);
static_assert(static_cast<int>(cudaLimitPersistingL2CacheSize) == 0x06);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cudaFuncCache
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(cudaFuncCachePreferNone) == 0);
static_assert(static_cast<int>(cudaFuncCachePreferShared) == 1);
static_assert(static_cast<int>(cudaFuncCachePreferL1) == 2);
static_assert(static_cast<int>(cudaFuncCachePreferEqual) == 3);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cudaSharedMemConfig
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(cudaSharedMemBankSizeDefault) == 0);
static_assert(static_cast<int>(cudaSharedMemBankSizeFourByte) == 1);
static_assert(static_cast<int>(cudaSharedMemBankSizeEightByte) == 2);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cudaFuncAttribute
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(cudaFuncAttributeMaxDynamicSharedMemorySize) == 8);
static_assert(static_cast<int>(cudaFuncAttributePreferredSharedMemoryCarveout) == 9);
static_assert(static_cast<int>(cudaFuncAttributeClusterDimMustBeSet) == 10);
static_assert(static_cast<int>(cudaFuncAttributeRequiredClusterWidth) == 11);
static_assert(static_cast<int>(cudaFuncAttributeRequiredClusterHeight) == 12);
static_assert(static_cast<int>(cudaFuncAttributeRequiredClusterDepth) == 13);
static_assert(static_cast<int>(cudaFuncAttributeNonPortableClusterSizeAllowed) == 14);
static_assert(static_cast<int>(cudaFuncAttributeClusterSchedulingPolicyPreference) == 15);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cudaMemoryAdvise
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(cudaMemAdviseSetReadMostly) == 1);
static_assert(static_cast<int>(cudaMemAdviseUnsetReadMostly) == 2);
static_assert(static_cast<int>(cudaMemAdviseSetPreferredLocation) == 3);
static_assert(static_cast<int>(cudaMemAdviseUnsetPreferredLocation) == 4);
static_assert(static_cast<int>(cudaMemAdviseSetAccessedBy) == 5);
static_assert(static_cast<int>(cudaMemAdviseUnsetAccessedBy) == 6);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cudaMemRangeAttribute
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(cudaMemRangeAttributeReadMostly) == 1);
static_assert(static_cast<int>(cudaMemRangeAttributePreferredLocation) == 2);
static_assert(static_cast<int>(cudaMemRangeAttributeAccessedBy) == 3);
static_assert(static_cast<int>(cudaMemRangeAttributeLastPrefetchLocation) == 4);
static_assert(static_cast<int>(cudaMemRangeAttributePreferredLocationType) == 5);
static_assert(static_cast<int>(cudaMemRangeAttributePreferredLocationId) == 6);
static_assert(static_cast<int>(cudaMemRangeAttributeLastPrefetchLocationType) == 7);
static_assert(static_cast<int>(cudaMemRangeAttributeLastPrefetchLocationId) == 8);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cudaMemPoolAttr
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(cudaMemPoolReuseFollowEventDependencies) == 0x1);
static_assert(static_cast<int>(cudaMemPoolReuseAllowOpportunistic) == 0x2);
static_assert(static_cast<int>(cudaMemPoolReuseAllowInternalDependencies) == 0x3);
static_assert(static_cast<int>(cudaMemPoolAttrReleaseThreshold) == 0x4);
static_assert(static_cast<int>(cudaMemPoolAttrReservedMemCurrent) == 0x5);
static_assert(static_cast<int>(cudaMemPoolAttrReservedMemHigh) == 0x6);
static_assert(static_cast<int>(cudaMemPoolAttrUsedMemCurrent) == 0x7);
static_assert(static_cast<int>(cudaMemPoolAttrUsedMemHigh) == 0x8);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cudaMemAllocationType
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(cudaMemAllocationTypeInvalid) == 0x0);
static_assert(static_cast<int>(cudaMemAllocationTypePinned) == 0x1);
static_assert(static_cast<int>(cudaMemAllocationTypeManaged) == 0x2);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cudaMemAllocationHandleType
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(cudaMemHandleTypeNone) == 0x0);
static_assert(static_cast<int>(cudaMemHandleTypePosixFileDescriptor) == 0x1);
static_assert(static_cast<int>(cudaMemHandleTypeWin32) == 0x2);
static_assert(static_cast<int>(cudaMemHandleTypeWin32Kmt) == 0x4);
static_assert(static_cast<int>(cudaMemHandleTypeFabric) == 0x8);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cudaMemLocationType
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(cudaMemLocationTypeInvalid) == 0);
static_assert(static_cast<int>(cudaMemLocationTypeNone) == 0); // alias for Invalid
static_assert(static_cast<int>(cudaMemLocationTypeDevice) == 1);
static_assert(static_cast<int>(cudaMemLocationTypeHost) == 2);
static_assert(static_cast<int>(cudaMemLocationTypeHostNuma) == 3);
static_assert(static_cast<int>(cudaMemLocationTypeHostNumaCurrent) == 4);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cudaStreamCaptureStatus
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(cudaStreamCaptureStatusNone) == 0);
static_assert(static_cast<int>(cudaStreamCaptureStatusActive) == 1);
static_assert(static_cast<int>(cudaStreamCaptureStatusInvalidated) == 2);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cudaStreamCaptureMode
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(cudaStreamCaptureModeGlobal) == 0);
static_assert(static_cast<int>(cudaStreamCaptureModeThreadLocal) == 1);
static_assert(static_cast<int>(cudaStreamCaptureModeRelaxed) == 2);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cudaStreamUpdateCaptureDependenciesFlags
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(cudaStreamAddCaptureDependencies) == 0x0);
static_assert(static_cast<int>(cudaStreamSetCaptureDependencies) == 0x1);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cudaSynchronizationPolicy
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(cudaSyncPolicyAuto) == 1);
static_assert(static_cast<int>(cudaSyncPolicySpin) == 2);
static_assert(static_cast<int>(cudaSyncPolicyYield) == 3);
static_assert(static_cast<int>(cudaSyncPolicyBlockingSync) == 4);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cudaGraphNodeType
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(cudaGraphNodeTypeKernel) == 0x00);
static_assert(static_cast<int>(cudaGraphNodeTypeMemcpy) == 0x01);
static_assert(static_cast<int>(cudaGraphNodeTypeMemset) == 0x02);
static_assert(static_cast<int>(cudaGraphNodeTypeHost) == 0x03);
static_assert(static_cast<int>(cudaGraphNodeTypeGraph) == 0x04);
static_assert(static_cast<int>(cudaGraphNodeTypeEmpty) == 0x05);
static_assert(static_cast<int>(cudaGraphNodeTypeWaitEvent) == 0x06);
static_assert(static_cast<int>(cudaGraphNodeTypeEventRecord) == 0x07);
static_assert(static_cast<int>(cudaGraphNodeTypeExtSemaphoreSignal) == 0x08);
static_assert(static_cast<int>(cudaGraphNodeTypeExtSemaphoreWait) == 0x09);
static_assert(static_cast<int>(cudaGraphNodeTypeMemAlloc) == 0x0a);
static_assert(static_cast<int>(cudaGraphNodeTypeMemFree) == 0x0b);
static_assert(static_cast<int>(cudaGraphNodeTypeConditional) == 0x0d);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cudaGraphExecUpdateResult
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(cudaGraphExecUpdateSuccess) == 0x0);
static_assert(static_cast<int>(cudaGraphExecUpdateError) == 0x1);
static_assert(static_cast<int>(cudaGraphExecUpdateErrorTopologyChanged) == 0x2);
static_assert(static_cast<int>(cudaGraphExecUpdateErrorNodeTypeChanged) == 0x3);
static_assert(static_cast<int>(cudaGraphExecUpdateErrorFunctionChanged) == 0x4);
static_assert(static_cast<int>(cudaGraphExecUpdateErrorParametersChanged) == 0x5);
static_assert(static_cast<int>(cudaGraphExecUpdateErrorNotSupported) == 0x6);
static_assert(static_cast<int>(cudaGraphExecUpdateErrorUnsupportedFunctionChange) == 0x7);
static_assert(static_cast<int>(cudaGraphExecUpdateErrorAttributesChanged) == 0x8);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cudaLaunchAttributeID
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(cudaLaunchAttributeIgnore) == 0);
static_assert(static_cast<int>(cudaLaunchAttributeAccessPolicyWindow) == 1);
static_assert(static_cast<int>(cudaLaunchAttributeCooperative) == 2);
static_assert(static_cast<int>(cudaLaunchAttributeSynchronizationPolicy) == 3);
static_assert(static_cast<int>(cudaLaunchAttributeClusterDimension) == 4);
static_assert(static_cast<int>(cudaLaunchAttributeClusterSchedulingPolicyPreference) == 5);
static_assert(static_cast<int>(cudaLaunchAttributeProgrammaticStreamSerialization) == 6);
static_assert(static_cast<int>(cudaLaunchAttributeProgrammaticEvent) == 7);
static_assert(static_cast<int>(cudaLaunchAttributePriority) == 8);
static_assert(static_cast<int>(cudaLaunchAttributeMemSyncDomainMap) == 9);
static_assert(static_cast<int>(cudaLaunchAttributeMemSyncDomain) == 10);
static_assert(static_cast<int>(cudaLaunchAttributePreferredClusterDimension) == 11);
static_assert(static_cast<int>(cudaLaunchAttributeLaunchCompletionEvent) == 12);
static_assert(static_cast<int>(cudaLaunchAttributeDeviceUpdatableKernelNode) == 13);
static_assert(static_cast<int>(cudaLaunchAttributePreferredSharedMemoryCarveout) == 14);
static_assert(static_cast<int>(cudaLaunchAttributeNvlinkUtilCentricScheduling) == 16);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cudaResourceType
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(cudaResourceTypeArray) == 0x00);
static_assert(static_cast<int>(cudaResourceTypeMipmappedArray) == 0x01);
static_assert(static_cast<int>(cudaResourceTypeLinear) == 0x02);
static_assert(static_cast<int>(cudaResourceTypePitch2D) == 0x03);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cudaResourceViewFormat
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(cudaResViewFormatNone) == 0x00);
static_assert(static_cast<int>(cudaResViewFormatUnsignedChar1) == 0x01);
static_assert(static_cast<int>(cudaResViewFormatUnsignedChar2) == 0x02);
static_assert(static_cast<int>(cudaResViewFormatUnsignedChar4) == 0x03);
static_assert(static_cast<int>(cudaResViewFormatSignedChar1) == 0x04);
static_assert(static_cast<int>(cudaResViewFormatSignedChar2) == 0x05);
static_assert(static_cast<int>(cudaResViewFormatSignedChar4) == 0x06);
static_assert(static_cast<int>(cudaResViewFormatUnsignedShort1) == 0x07);
static_assert(static_cast<int>(cudaResViewFormatUnsignedShort2) == 0x08);
static_assert(static_cast<int>(cudaResViewFormatUnsignedShort4) == 0x09);
static_assert(static_cast<int>(cudaResViewFormatSignedShort1) == 0x0a);
static_assert(static_cast<int>(cudaResViewFormatSignedShort2) == 0x0b);
static_assert(static_cast<int>(cudaResViewFormatSignedShort4) == 0x0c);
static_assert(static_cast<int>(cudaResViewFormatUnsignedInt1) == 0x0d);
static_assert(static_cast<int>(cudaResViewFormatUnsignedInt2) == 0x0e);
static_assert(static_cast<int>(cudaResViewFormatUnsignedInt4) == 0x0f);
static_assert(static_cast<int>(cudaResViewFormatSignedInt1) == 0x10);
static_assert(static_cast<int>(cudaResViewFormatSignedInt2) == 0x11);
static_assert(static_cast<int>(cudaResViewFormatSignedInt4) == 0x12);
static_assert(static_cast<int>(cudaResViewFormatHalf1) == 0x13);
static_assert(static_cast<int>(cudaResViewFormatHalf2) == 0x14);
static_assert(static_cast<int>(cudaResViewFormatHalf4) == 0x15);
static_assert(static_cast<int>(cudaResViewFormatFloat1) == 0x16);
static_assert(static_cast<int>(cudaResViewFormatFloat2) == 0x17);
static_assert(static_cast<int>(cudaResViewFormatFloat4) == 0x18);
static_assert(static_cast<int>(cudaResViewFormatUnsignedBlockCompressed1) == 0x19);
static_assert(static_cast<int>(cudaResViewFormatUnsignedBlockCompressed2) == 0x1a);
static_assert(static_cast<int>(cudaResViewFormatUnsignedBlockCompressed3) == 0x1b);
static_assert(static_cast<int>(cudaResViewFormatUnsignedBlockCompressed4) == 0x1c);
static_assert(static_cast<int>(cudaResViewFormatSignedBlockCompressed4) == 0x1d);
static_assert(static_cast<int>(cudaResViewFormatUnsignedBlockCompressed5) == 0x1e);
static_assert(static_cast<int>(cudaResViewFormatSignedBlockCompressed5) == 0x1f);
static_assert(static_cast<int>(cudaResViewFormatUnsignedBlockCompressed6H) == 0x20);
static_assert(static_cast<int>(cudaResViewFormatSignedBlockCompressed6H) == 0x21);
static_assert(static_cast<int>(cudaResViewFormatUnsignedBlockCompressed7) == 0x22);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cudaGraphicsRegisterFlags
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(cudaGraphicsRegisterFlagsNone) == 0);
static_assert(static_cast<int>(cudaGraphicsRegisterFlagsReadOnly) == 1);
static_assert(static_cast<int>(cudaGraphicsRegisterFlagsWriteDiscard) == 2);
static_assert(static_cast<int>(cudaGraphicsRegisterFlagsSurfaceLoadStore) == 4);
static_assert(static_cast<int>(cudaGraphicsRegisterFlagsTextureGather) == 8);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cudaGraphicsMapFlags
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(cudaGraphicsMapFlagsNone) == 0);
static_assert(static_cast<int>(cudaGraphicsMapFlagsReadOnly) == 1);
static_assert(static_cast<int>(cudaGraphicsMapFlagsWriteDiscard) == 2);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cudaAccessProperty
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(cudaAccessPropertyNormal) == 0);
static_assert(static_cast<int>(cudaAccessPropertyStreaming) == 1);
static_assert(static_cast<int>(cudaAccessPropertyPersisting) == 2);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cudaUserObjectFlags / cudaUserObjectRetainFlags
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(cudaUserObjectNoDestructorSync) == 0x1);
static_assert(static_cast<int>(cudaGraphUserObjectMove) == 0x1);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cudaDeviceP2PAttr
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(cudaDevP2PAttrPerformanceRank) == 1);
static_assert(static_cast<int>(cudaDevP2PAttrAccessSupported) == 2);
static_assert(static_cast<int>(cudaDevP2PAttrNativeAtomicSupported) == 3);
static_assert(static_cast<int>(cudaDevP2PAttrCudaArrayAccessSupported) == 4);
static_assert(static_cast<int>(cudaDevP2PAttrOnlyPartialNativeAtomicSupported) == 5);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cudaExternalMemoryHandleType
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(cudaExternalMemoryHandleTypeOpaqueFd) == 1);
static_assert(static_cast<int>(cudaExternalMemoryHandleTypeOpaqueWin32) == 2);
static_assert(static_cast<int>(cudaExternalMemoryHandleTypeOpaqueWin32Kmt) == 3);
static_assert(static_cast<int>(cudaExternalMemoryHandleTypeD3D12Heap) == 4);
static_assert(static_cast<int>(cudaExternalMemoryHandleTypeD3D12Resource) == 5);
static_assert(static_cast<int>(cudaExternalMemoryHandleTypeD3D11Resource) == 6);
static_assert(static_cast<int>(cudaExternalMemoryHandleTypeD3D11ResourceKmt) == 7);
static_assert(static_cast<int>(cudaExternalMemoryHandleTypeNvSciBuf) == 8);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cudaExternalSemaphoreHandleType
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(cudaExternalSemaphoreHandleTypeOpaqueFd) == 1);
static_assert(static_cast<int>(cudaExternalSemaphoreHandleTypeOpaqueWin32) == 2);
static_assert(static_cast<int>(cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt) == 3);
static_assert(static_cast<int>(cudaExternalSemaphoreHandleTypeD3D12Fence) == 4);
static_assert(static_cast<int>(cudaExternalSemaphoreHandleTypeD3D11Fence) == 5);
static_assert(static_cast<int>(cudaExternalSemaphoreHandleTypeNvSciSync) == 6);
static_assert(static_cast<int>(cudaExternalSemaphoreHandleTypeKeyedMutex) == 7);
static_assert(static_cast<int>(cudaExternalSemaphoreHandleTypeKeyedMutexKmt) == 8);
static_assert(static_cast<int>(cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd) == 9);
static_assert(static_cast<int>(cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32) == 10);

// ────────────────────────────────────────────────────────────────────────
// Enum values: cudaDeviceAttr (representative subset of stable ABI values)
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(cudaDevAttrMaxThreadsPerBlock) == 1);
static_assert(static_cast<int>(cudaDevAttrMaxBlockDimX) == 2);
static_assert(static_cast<int>(cudaDevAttrMaxBlockDimY) == 3);
static_assert(static_cast<int>(cudaDevAttrMaxBlockDimZ) == 4);
static_assert(static_cast<int>(cudaDevAttrMaxGridDimX) == 5);
static_assert(static_cast<int>(cudaDevAttrMaxGridDimY) == 6);
static_assert(static_cast<int>(cudaDevAttrMaxGridDimZ) == 7);
static_assert(static_cast<int>(cudaDevAttrMaxSharedMemoryPerBlock) == 8);
static_assert(static_cast<int>(cudaDevAttrTotalConstantMemory) == 9);
static_assert(static_cast<int>(cudaDevAttrWarpSize) == 10);
static_assert(static_cast<int>(cudaDevAttrMaxPitch) == 11);
static_assert(static_cast<int>(cudaDevAttrMaxRegistersPerBlock) == 12);
static_assert(static_cast<int>(cudaDevAttrClockRate) == 13);
static_assert(static_cast<int>(cudaDevAttrTextureAlignment) == 14);
static_assert(static_cast<int>(cudaDevAttrGpuOverlap) == 15);
static_assert(static_cast<int>(cudaDevAttrMultiProcessorCount) == 16);
static_assert(static_cast<int>(cudaDevAttrKernelExecTimeout) == 17);
static_assert(static_cast<int>(cudaDevAttrIntegrated) == 18);
static_assert(static_cast<int>(cudaDevAttrCanMapHostMemory) == 19);
static_assert(static_cast<int>(cudaDevAttrComputeMode) == 20);
static_assert(static_cast<int>(cudaDevAttrMaxTexture1DWidth) == 21);
static_assert(static_cast<int>(cudaDevAttrMaxTexture2DWidth) == 22);
static_assert(static_cast<int>(cudaDevAttrMaxTexture2DHeight) == 23);
static_assert(static_cast<int>(cudaDevAttrMaxTexture3DWidth) == 24);
static_assert(static_cast<int>(cudaDevAttrMaxTexture3DHeight) == 25);
static_assert(static_cast<int>(cudaDevAttrMaxTexture3DDepth) == 26);
static_assert(static_cast<int>(cudaDevAttrMaxTexture2DLayeredWidth) == 27);
static_assert(static_cast<int>(cudaDevAttrMaxTexture2DLayeredHeight) == 28);
static_assert(static_cast<int>(cudaDevAttrMaxTexture2DLayeredLayers) == 29);
static_assert(static_cast<int>(cudaDevAttrSurfaceAlignment) == 30);
static_assert(static_cast<int>(cudaDevAttrConcurrentKernels) == 31);
static_assert(static_cast<int>(cudaDevAttrEccEnabled) == 32);
static_assert(static_cast<int>(cudaDevAttrPciBusId) == 33);
static_assert(static_cast<int>(cudaDevAttrPciDeviceId) == 34);
static_assert(static_cast<int>(cudaDevAttrTccDriver) == 35);
static_assert(static_cast<int>(cudaDevAttrMemoryClockRate) == 36);
static_assert(static_cast<int>(cudaDevAttrGlobalMemoryBusWidth) == 37);
static_assert(static_cast<int>(cudaDevAttrL2CacheSize) == 38);
static_assert(static_cast<int>(cudaDevAttrMaxThreadsPerMultiProcessor) == 39);
static_assert(static_cast<int>(cudaDevAttrAsyncEngineCount) == 40);
static_assert(static_cast<int>(cudaDevAttrUnifiedAddressing) == 41);
static_assert(static_cast<int>(cudaDevAttrMaxTexture1DLayeredWidth) == 42);
static_assert(static_cast<int>(cudaDevAttrMaxTexture1DLayeredLayers) == 43);
static_assert(static_cast<int>(cudaDevAttrMaxTexture2DGatherWidth) == 45);
static_assert(static_cast<int>(cudaDevAttrMaxTexture2DGatherHeight) == 46);
static_assert(static_cast<int>(cudaDevAttrMaxTexture3DWidthAlt) == 47);
static_assert(static_cast<int>(cudaDevAttrMaxTexture3DHeightAlt) == 48);
static_assert(static_cast<int>(cudaDevAttrMaxTexture3DDepthAlt) == 49);
static_assert(static_cast<int>(cudaDevAttrPciDomainId) == 50);
static_assert(static_cast<int>(cudaDevAttrTexturePitchAlignment) == 51);
static_assert(static_cast<int>(cudaDevAttrMaxTextureCubemapWidth) == 52);
static_assert(static_cast<int>(cudaDevAttrMaxTextureCubemapLayeredWidth) == 53);
static_assert(static_cast<int>(cudaDevAttrMaxTextureCubemapLayeredLayers) == 54);
static_assert(static_cast<int>(cudaDevAttrMaxSurface1DWidth) == 55);
static_assert(static_cast<int>(cudaDevAttrMaxSurface2DWidth) == 56);
static_assert(static_cast<int>(cudaDevAttrMaxSurface2DHeight) == 57);
static_assert(static_cast<int>(cudaDevAttrMaxSurface3DWidth) == 58);
static_assert(static_cast<int>(cudaDevAttrMaxSurface3DHeight) == 59);
static_assert(static_cast<int>(cudaDevAttrMaxSurface3DDepth) == 60);
static_assert(static_cast<int>(cudaDevAttrMaxSurface1DLayeredWidth) == 61);
static_assert(static_cast<int>(cudaDevAttrMaxSurface1DLayeredLayers) == 62);
static_assert(static_cast<int>(cudaDevAttrMaxSurface2DLayeredWidth) == 63);
static_assert(static_cast<int>(cudaDevAttrMaxSurface2DLayeredHeight) == 64);
static_assert(static_cast<int>(cudaDevAttrMaxSurface2DLayeredLayers) == 65);
static_assert(static_cast<int>(cudaDevAttrMaxSurfaceCubemapWidth) == 66);
static_assert(static_cast<int>(cudaDevAttrMaxSurfaceCubemapLayeredWidth) == 67);
static_assert(static_cast<int>(cudaDevAttrMaxSurfaceCubemapLayeredLayers) == 68);
static_assert(static_cast<int>(cudaDevAttrMaxTexture1DLinearWidth) == 69);
static_assert(static_cast<int>(cudaDevAttrMaxTexture2DLinearWidth) == 70);
static_assert(static_cast<int>(cudaDevAttrMaxTexture2DLinearHeight) == 71);
static_assert(static_cast<int>(cudaDevAttrMaxTexture2DLinearPitch) == 72);
static_assert(static_cast<int>(cudaDevAttrMaxTexture2DMipmappedWidth) == 73);
static_assert(static_cast<int>(cudaDevAttrMaxTexture2DMipmappedHeight) == 74);
static_assert(static_cast<int>(cudaDevAttrComputeCapabilityMajor) == 75);
static_assert(static_cast<int>(cudaDevAttrComputeCapabilityMinor) == 76);
static_assert(static_cast<int>(cudaDevAttrMaxTexture1DMipmappedWidth) == 77);
static_assert(static_cast<int>(cudaDevAttrStreamPrioritiesSupported) == 78);
static_assert(static_cast<int>(cudaDevAttrGlobalL1CacheSupported) == 79);
static_assert(static_cast<int>(cudaDevAttrLocalL1CacheSupported) == 80);
static_assert(static_cast<int>(cudaDevAttrMaxSharedMemoryPerMultiprocessor) == 81);
static_assert(static_cast<int>(cudaDevAttrMaxRegistersPerMultiprocessor) == 82);
static_assert(static_cast<int>(cudaDevAttrManagedMemory) == 83);
static_assert(static_cast<int>(cudaDevAttrIsMultiGpuBoard) == 84);
static_assert(static_cast<int>(cudaDevAttrMultiGpuBoardGroupID) == 85);
static_assert(static_cast<int>(cudaDevAttrHostNativeAtomicSupported) == 86);
static_assert(static_cast<int>(cudaDevAttrSingleToDoublePrecisionPerfRatio) == 87);
static_assert(static_cast<int>(cudaDevAttrPageableMemoryAccess) == 88);
static_assert(static_cast<int>(cudaDevAttrConcurrentManagedAccess) == 89);
static_assert(static_cast<int>(cudaDevAttrComputePreemptionSupported) == 90);
static_assert(static_cast<int>(cudaDevAttrCanUseHostPointerForRegisteredMem) == 91);
static_assert(static_cast<int>(cudaDevAttrReserved92) == 92);
static_assert(static_cast<int>(cudaDevAttrReserved93) == 93);
static_assert(static_cast<int>(cudaDevAttrReserved94) == 94);
static_assert(static_cast<int>(cudaDevAttrCooperativeLaunch) == 95);
static_assert(static_cast<int>(cudaDevAttrReserved96) == 96);
static_assert(static_cast<int>(cudaDevAttrMaxSharedMemoryPerBlockOptin) == 97);
static_assert(static_cast<int>(cudaDevAttrCanFlushRemoteWrites) == 98);
static_assert(static_cast<int>(cudaDevAttrHostRegisterSupported) == 99);
static_assert(static_cast<int>(cudaDevAttrPageableMemoryAccessUsesHostPageTables) == 100);
static_assert(static_cast<int>(cudaDevAttrDirectManagedMemAccessFromHost) == 101);
static_assert(static_cast<int>(cudaDevAttrMaxBlocksPerMultiprocessor) == 106);
static_assert(static_cast<int>(cudaDevAttrMaxPersistingL2CacheSize) == 108);
static_assert(static_cast<int>(cudaDevAttrMaxAccessPolicyWindowSize) == 109);
static_assert(static_cast<int>(cudaDevAttrReservedSharedMemoryPerBlock) == 111);
static_assert(static_cast<int>(cudaDevAttrSparseCudaArraySupported) == 112);
static_assert(static_cast<int>(cudaDevAttrHostRegisterReadOnlySupported) == 113);
static_assert(static_cast<int>(cudaDevAttrTimelineSemaphoreInteropSupported) == 114);
static_assert(static_cast<int>(cudaDevAttrMemoryPoolsSupported) == 115);
static_assert(static_cast<int>(cudaDevAttrGPUDirectRDMASupported) == 116);
static_assert(static_cast<int>(cudaDevAttrGPUDirectRDMAFlushWritesOptions) == 117);
static_assert(static_cast<int>(cudaDevAttrGPUDirectRDMAWritesOrdering) == 118);
static_assert(static_cast<int>(cudaDevAttrMemoryPoolSupportedHandleTypes) == 119);
static_assert(static_cast<int>(cudaDevAttrClusterLaunch) == 120);
static_assert(static_cast<int>(cudaDevAttrDeferredMappingCudaArraySupported) == 121);
static_assert(static_cast<int>(cudaDevAttrReserved122) == 122);
static_assert(static_cast<int>(cudaDevAttrReserved123) == 123);
static_assert(static_cast<int>(cudaDevAttrReserved124) == 124);
static_assert(static_cast<int>(cudaDevAttrIpcEventSupport) == 125);
static_assert(static_cast<int>(cudaDevAttrMemSyncDomainCount) == 126);
static_assert(static_cast<int>(cudaDevAttrReserved127) == 127);
static_assert(static_cast<int>(cudaDevAttrReserved128) == 128);
static_assert(static_cast<int>(cudaDevAttrReserved129) == 129);
static_assert(static_cast<int>(cudaDevAttrNumaConfig) == 130);
static_assert(static_cast<int>(cudaDevAttrNumaId) == 131);
static_assert(static_cast<int>(cudaDevAttrReserved132) == 132);
static_assert(static_cast<int>(cudaDevAttrMpsEnabled) == 133);
static_assert(static_cast<int>(cudaDevAttrHostNumaId) == 134);
static_assert(static_cast<int>(cudaDevAttrD3D12CigSupported) == 135);
static_assert(static_cast<int>(cudaDevAttrVulkanCigSupported) == 138);
static_assert(static_cast<int>(cudaDevAttrGpuPciDeviceId) == 139);
static_assert(static_cast<int>(cudaDevAttrGpuPciSubsystemId) == 140);
static_assert(static_cast<int>(cudaDevAttrReserved141) == 141);
static_assert(static_cast<int>(cudaDevAttrHostNumaMemoryPoolsSupported) == 142);
static_assert(static_cast<int>(cudaDevAttrHostNumaMultinodeIpcSupported) == 143);
static_assert(static_cast<int>(cudaDevAttrHostMemoryPoolsSupported) == 144);
static_assert(static_cast<int>(cudaDevAttrReserved145) == 145);
static_assert(static_cast<int>(cudaDevAttrOnlyPartialHostNativeAtomicSupported) == 147);

// ────────────────────────────────────────────────────────────────────────
// Opaque handle types: stream/event/graph/pool — all pointer-sized handles
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_pointer_v<cudaStream_t>);
static_assert(std::is_pointer_v<cudaEvent_t>);
static_assert(std::is_pointer_v<cudaGraph_t>);
static_assert(std::is_pointer_v<cudaGraphNode_t>);
static_assert(std::is_pointer_v<cudaGraphExec_t>);
static_assert(std::is_pointer_v<cudaMemPool_t>);
static_assert(std::is_pointer_v<cudaUserObject_t>);

// All handles must fit in a pointer
static_assert(sizeof(cudaStream_t) == sizeof(void *));
static_assert(sizeof(cudaEvent_t) == sizeof(void *));
static_assert(sizeof(cudaGraph_t) == sizeof(void *));
static_assert(sizeof(cudaGraphNode_t) == sizeof(void *));
static_assert(sizeof(cudaGraphExec_t) == sizeof(void *));
static_assert(sizeof(cudaMemPool_t) == sizeof(void *));

// Array handles
static_assert(std::is_pointer_v<cudaArray_t>);
static_assert(std::is_pointer_v<cudaArray_const_t>);
static_assert(std::is_pointer_v<cudaMipmappedArray_t>);
static_assert(std::is_pointer_v<cudaMipmappedArray_const_t>);

// Texture/surface objects are NOT pointers — CUDA defines them as uint64
// (driver-managed indices, not host pointers)
static_assert(!std::is_pointer_v<cudaTextureObject_t>);
static_assert(!std::is_pointer_v<cudaSurfaceObject_t>);
static_assert(std::is_integral_v<cudaTextureObject_t>);
static_assert(std::is_integral_v<cudaSurfaceObject_t>);
static_assert(sizeof(cudaTextureObject_t) == 8);
static_assert(sizeof(cudaSurfaceObject_t) == 8);

// IPC handles are fixed-size byte arrays (64-byte ABI contract, CUDA_IPC_HANDLE_SIZE)
static_assert(!std::is_pointer_v<cudaIpcEventHandle_t>);
static_assert(!std::is_pointer_v<cudaIpcMemHandle_t>);
static_assert(sizeof(cudaIpcEventHandle_t) == 64);
static_assert(sizeof(cudaIpcMemHandle_t) == 64);

// ────────────────────────────────────────────────────────────────────────
// Function-pointer types
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_pointer_v<cudaHostFn_t>);
static_assert(std::is_pointer_v<cudaStreamCallback_t>);

// ────────────────────────────────────────────────────────────────────────
// Struct traits: trivial copyability
// All CUDA structs are passed by value / memcpy'd — they must be trivially
// copyable. A failure here means our module wrapper broke the C-ABI contract.
// ────────────────────────────────────────────────────────────────────────

// Geometry / memory layout primitives
static_assert(std::is_trivially_copyable_v<cudaPitchedPtr>);
static_assert(std::is_trivially_copyable_v<cudaExtent>);
static_assert(std::is_trivially_copyable_v<cudaPos>);
static_assert(std::is_trivially_copyable_v<cudaChannelFormatDesc>);

// Memory pool / allocation
static_assert(std::is_trivially_copyable_v<cudaMemPoolProps>);
static_assert(std::is_trivially_copyable_v<cudaMemPoolPtrExportData>);

// Pointer query result
static_assert(std::is_trivially_copyable_v<cudaPointerAttributes>);

// Function / kernel metadata
static_assert(std::is_trivially_copyable_v<cudaFuncAttributes>);

// Array metadata
static_assert(std::is_trivially_copyable_v<cudaArraySparseProperties>);
static_assert(std::is_trivially_copyable_v<cudaArrayMemoryRequirements>);

// Memcpy / memset parameter blocks
static_assert(std::is_trivially_copyable_v<cudaMemcpy3DParms>);
static_assert(std::is_trivially_copyable_v<cudaMemcpy3DPeerParms>);
static_assert(std::is_trivially_copyable_v<cudaMemcpyNodeParams>);
static_assert(std::is_trivially_copyable_v<cudaMemsetParams>);

// Graph node parameter blocks
static_assert(std::is_trivially_copyable_v<cudaHostNodeParams>);
static_assert(std::is_trivially_copyable_v<cudaKernelNodeParams>);
static_assert(std::is_trivially_copyable_v<cudaLaunchAttributeValue>);
static_assert(std::is_trivially_copyable_v<cudaMemAllocNodeParams>);

// Graph lifecycle
static_assert(std::is_trivially_copyable_v<cudaGraphInstantiateParams>);
static_assert(std::is_trivially_copyable_v<cudaGraphExecUpdateResultInfo>);

// Resource / texture descriptors
static_assert(std::is_trivially_copyable_v<cudaResourceDesc>);
static_assert(std::is_trivially_copyable_v<cudaResourceViewDesc>);
static_assert(std::is_trivially_copyable_v<cudaTextureDesc>);

// Access policy
static_assert(std::is_trivially_copyable_v<cudaAccessPolicyWindow>);

// External memory / semaphore descriptors
static_assert(std::is_trivially_copyable_v<cudaExternalMemoryHandleDesc>);
static_assert(std::is_trivially_copyable_v<cudaExternalMemoryBufferDesc>);
static_assert(std::is_trivially_copyable_v<cudaExternalMemoryMipmappedArrayDesc>);
static_assert(std::is_trivially_copyable_v<cudaExternalSemaphoreHandleDesc>);
static_assert(std::is_trivially_copyable_v<cudaExternalSemaphoreSignalParams>);
static_assert(std::is_trivially_copyable_v<cudaExternalSemaphoreWaitParams>);

// Device properties (large but still a plain C struct)
static_assert(std::is_trivially_copyable_v<cudaDeviceProp>);

// ────────────────────────────────────────────────────────────────────────
// Struct traits: standard layout
// All CUDA structs must be standard layout for C/CUDA interop.
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_standard_layout_v<cudaPitchedPtr>);
static_assert(std::is_standard_layout_v<cudaExtent>);
static_assert(std::is_standard_layout_v<cudaPos>);
static_assert(std::is_standard_layout_v<cudaChannelFormatDesc>);
static_assert(std::is_standard_layout_v<cudaMemPoolProps>);
static_assert(std::is_standard_layout_v<cudaMemPoolPtrExportData>);
static_assert(std::is_standard_layout_v<cudaPointerAttributes>);
static_assert(std::is_standard_layout_v<cudaFuncAttributes>);
static_assert(std::is_standard_layout_v<cudaArraySparseProperties>);
static_assert(std::is_standard_layout_v<cudaArrayMemoryRequirements>);
static_assert(std::is_standard_layout_v<cudaMemcpy3DParms>);
static_assert(std::is_standard_layout_v<cudaMemcpy3DPeerParms>);
static_assert(std::is_standard_layout_v<cudaMemcpyNodeParams>);
static_assert(std::is_standard_layout_v<cudaMemsetParams>);
static_assert(std::is_standard_layout_v<cudaHostNodeParams>);
static_assert(std::is_standard_layout_v<cudaKernelNodeParams>);
static_assert(std::is_standard_layout_v<cudaMemAllocNodeParams>);
static_assert(std::is_standard_layout_v<cudaGraphInstantiateParams>);
static_assert(std::is_standard_layout_v<cudaGraphExecUpdateResultInfo>);
static_assert(std::is_standard_layout_v<cudaResourceDesc>);
static_assert(std::is_standard_layout_v<cudaResourceViewDesc>);
static_assert(std::is_standard_layout_v<cudaTextureDesc>);
static_assert(std::is_standard_layout_v<cudaAccessPolicyWindow>);
static_assert(std::is_standard_layout_v<cudaExternalMemoryHandleDesc>);
static_assert(std::is_standard_layout_v<cudaExternalMemoryBufferDesc>);
static_assert(std::is_standard_layout_v<cudaExternalMemoryMipmappedArrayDesc>);
static_assert(std::is_standard_layout_v<cudaExternalSemaphoreHandleDesc>);
static_assert(std::is_standard_layout_v<cudaExternalSemaphoreSignalParams>);
static_assert(std::is_standard_layout_v<cudaExternalSemaphoreWaitParams>);
static_assert(std::is_standard_layout_v<cudaLaunchAttributeValue>);
static_assert(std::is_standard_layout_v<cudaFuncAttributes>);
static_assert(std::is_standard_layout_v<cudaDeviceProp>);
static_assert(std::is_standard_layout_v<cudaIpcEventHandle_t>);
static_assert(std::is_standard_layout_v<cudaIpcMemHandle_t>);

// ────────────────────────────────────────────────────────────────────────
// Link-time symbol resolution
// Forces the linker to resolve every re-exported function symbol,
// catching missing or unresolvable exports that type-only checks miss.
// ────────────────────────────────────────────────────────────────────────

// Error Handling
GPUMOD_LINK_CHECK(cudaGetErrorName)
GPUMOD_LINK_CHECK(cudaGetErrorString)
GPUMOD_LINK_CHECK(cudaGetLastError)
GPUMOD_LINK_CHECK(cudaPeekAtLastError)

// Device Management
GPUMOD_LINK_CHECK(cudaGetDevice)
GPUMOD_LINK_CHECK(cudaSetDevice)
GPUMOD_LINK_CHECK(cudaGetDeviceCount)
GPUMOD_LINK_CHECK(cudaGetDeviceProperties)
GPUMOD_LINK_CHECK(cudaDeviceSynchronize)
GPUMOD_LINK_CHECK(cudaDeviceReset)
GPUMOD_LINK_CHECK(cudaSetDeviceFlags)
GPUMOD_LINK_CHECK(cudaGetDeviceFlags)
GPUMOD_LINK_CHECK(cudaChooseDevice)
GPUMOD_LINK_CHECK(cudaDeviceGetAttribute)
GPUMOD_LINK_CHECK(cudaDeviceGetByPCIBusId)
GPUMOD_LINK_CHECK(cudaDeviceGetPCIBusId)
GPUMOD_LINK_CHECK(cudaDeviceGetDefaultMemPool)
GPUMOD_LINK_CHECK(cudaDeviceSetMemPool)
GPUMOD_LINK_CHECK(cudaDeviceGetMemPool)
GPUMOD_LINK_CHECK(cudaDeviceSetLimit)
GPUMOD_LINK_CHECK(cudaDeviceGetLimit)
GPUMOD_LINK_CHECK(cudaDeviceSetCacheConfig)
GPUMOD_LINK_CHECK(cudaDeviceGetCacheConfig)
GPUMOD_LINK_CHECK(cudaDeviceSetSharedMemConfig)
GPUMOD_LINK_CHECK(cudaDeviceGetSharedMemConfig)
GPUMOD_LINK_CHECK(cudaDeviceGetStreamPriorityRange)
GPUMOD_LINK_CHECK(cudaDeviceGetTexture1DLinearMaxWidth)
GPUMOD_LINK_CHECK(cudaDeviceCanAccessPeer)
GPUMOD_LINK_CHECK(cudaDeviceEnablePeerAccess)
GPUMOD_LINK_CHECK(cudaDeviceDisablePeerAccess)
GPUMOD_LINK_CHECK(cudaDeviceGetP2PAttribute)
GPUMOD_LINK_CHECK(cudaDeviceFlushGPUDirectRDMAWrites)
GPUMOD_LINK_CHECK(cudaCtxResetPersistingL2Cache)
GPUMOD_LINK_CHECK(cudaDeviceGetNvSciSyncAttributes)
GPUMOD_LINK_CHECK(cudaDeviceRegisterAsyncNotification)
GPUMOD_LINK_CHECK(cudaDeviceUnregisterAsyncNotification)
GPUMOD_LINK_CHECK(cudaDeviceGetGraphMemAttribute)
GPUMOD_LINK_CHECK(cudaDeviceSetGraphMemAttribute)
GPUMOD_LINK_CHECK(cudaDeviceGraphMemTrim)
GPUMOD_LINK_CHECK(cudaDeviceGetHostAtomicCapabilities)
GPUMOD_LINK_CHECK(cudaDeviceGetP2PAtomicCapabilities)
GPUMOD_LINK_CHECK(cudaDriverGetVersion)
GPUMOD_LINK_CHECK(cudaRuntimeGetVersion)

// Memory — Device Allocation
GPUMOD_LINK_CHECK(cudaMalloc)
GPUMOD_LINK_CHECK(cudaMallocAsync)
GPUMOD_LINK_CHECK(cudaFree)
GPUMOD_LINK_CHECK(cudaFreeAsync)
GPUMOD_LINK_CHECK(cudaMallocPitch)
GPUMOD_LINK_CHECK(cudaMalloc3D)
GPUMOD_LINK_CHECK(cudaMalloc3DArray)
GPUMOD_LINK_CHECK(cudaMallocArray)
GPUMOD_LINK_CHECK(cudaFreeArray)
GPUMOD_LINK_CHECK(cudaMallocMipmappedArray)
GPUMOD_LINK_CHECK(cudaFreeMipmappedArray)
GPUMOD_LINK_CHECK(cudaMallocManaged)

// Memory — Host (Pinned)
GPUMOD_LINK_CHECK(cudaMallocHost)
GPUMOD_LINK_CHECK(cudaHostAlloc)
GPUMOD_LINK_CHECK(cudaFreeHost)
GPUMOD_LINK_CHECK(cudaHostRegister)
GPUMOD_LINK_CHECK(cudaHostUnregister)
GPUMOD_LINK_CHECK(cudaHostGetDevicePointer)
GPUMOD_LINK_CHECK(cudaHostGetFlags)

// Memory — Copy
GPUMOD_LINK_CHECK(cudaMemcpy)
GPUMOD_LINK_CHECK(cudaMemcpyAsync)
GPUMOD_LINK_CHECK(cudaMemcpy2D)
GPUMOD_LINK_CHECK(cudaMemcpy2DAsync)
GPUMOD_LINK_CHECK(cudaMemcpy2DToArray)
GPUMOD_LINK_CHECK(cudaMemcpy2DToArrayAsync)
GPUMOD_LINK_CHECK(cudaMemcpy2DFromArray)
GPUMOD_LINK_CHECK(cudaMemcpy2DFromArrayAsync)
GPUMOD_LINK_CHECK(cudaMemcpy2DArrayToArray)
GPUMOD_LINK_CHECK(cudaMemcpy3D)
GPUMOD_LINK_CHECK(cudaMemcpy3DAsync)
GPUMOD_LINK_CHECK(cudaMemcpy3DPeer)
GPUMOD_LINK_CHECK(cudaMemcpy3DPeerAsync)
GPUMOD_LINK_CHECK(cudaMemcpyPeer)
GPUMOD_LINK_CHECK(cudaMemcpyPeerAsync)
GPUMOD_LINK_CHECK(cudaMemcpyToArray)
GPUMOD_LINK_CHECK(cudaMemcpyToArrayAsync)
GPUMOD_LINK_CHECK(cudaMemcpyFromArray)
GPUMOD_LINK_CHECK(cudaMemcpyFromArrayAsync)
GPUMOD_LINK_CHECK(cudaMemcpyArrayToArray)
GPUMOD_LINK_CHECK(cudaMemcpyToSymbol)
GPUMOD_LINK_CHECK(cudaMemcpyToSymbolAsync)
GPUMOD_LINK_CHECK(cudaMemcpyFromSymbol)
GPUMOD_LINK_CHECK(cudaMemcpyFromSymbolAsync)

// Memory — Set
GPUMOD_LINK_CHECK(cudaMemset)
GPUMOD_LINK_CHECK(cudaMemsetAsync)
GPUMOD_LINK_CHECK(cudaMemset2D)
GPUMOD_LINK_CHECK(cudaMemset2DAsync)
GPUMOD_LINK_CHECK(cudaMemset3D)
GPUMOD_LINK_CHECK(cudaMemset3DAsync)

// Memory — Info / Advise / Query
GPUMOD_LINK_CHECK(cudaMemGetInfo)
GPUMOD_LINK_CHECK(cudaMemPrefetchAsync)
GPUMOD_LINK_CHECK(cudaMemAdvise)
GPUMOD_LINK_CHECK(cudaMemRangeGetAttribute)
GPUMOD_LINK_CHECK(cudaMemRangeGetAttributes)
GPUMOD_LINK_CHECK(cudaPointerGetAttributes)

// Memory — Array Operations
GPUMOD_LINK_CHECK(cudaArrayGetInfo)
GPUMOD_LINK_CHECK(cudaArrayGetPlane)
GPUMOD_LINK_CHECK(cudaArrayGetMemoryRequirements)
GPUMOD_LINK_CHECK(cudaArrayGetSparseProperties)
GPUMOD_LINK_CHECK(cudaGetMipmappedArrayLevel)

// Memory — Pools
GPUMOD_LINK_CHECK(cudaMemPoolCreate)
GPUMOD_LINK_CHECK(cudaMemPoolDestroy)
GPUMOD_LINK_CHECK(cudaMallocFromPoolAsync)
GPUMOD_LINK_CHECK(cudaMemPoolTrimTo)
GPUMOD_LINK_CHECK(cudaMemPoolSetAttribute)
GPUMOD_LINK_CHECK(cudaMemPoolGetAttribute)
GPUMOD_LINK_CHECK(cudaMemPoolSetAccess)
GPUMOD_LINK_CHECK(cudaMemPoolGetAccess)
GPUMOD_LINK_CHECK(cudaMemPoolExportToShareableHandle)
GPUMOD_LINK_CHECK(cudaMemPoolImportFromShareableHandle)
GPUMOD_LINK_CHECK(cudaMemPoolExportPointer)
GPUMOD_LINK_CHECK(cudaMemPoolImportPointer)

// Memory — External
GPUMOD_LINK_CHECK(cudaImportExternalMemory)
GPUMOD_LINK_CHECK(cudaExternalMemoryGetMappedBuffer)
GPUMOD_LINK_CHECK(cudaExternalMemoryGetMappedMipmappedArray)
GPUMOD_LINK_CHECK(cudaDestroyExternalMemory)

// Memory — IPC
GPUMOD_LINK_CHECK(cudaIpcGetMemHandle)
GPUMOD_LINK_CHECK(cudaIpcOpenMemHandle)
GPUMOD_LINK_CHECK(cudaIpcCloseMemHandle)
GPUMOD_LINK_CHECK(cudaIpcGetEventHandle)
GPUMOD_LINK_CHECK(cudaIpcOpenEventHandle)

// Stream Management
GPUMOD_LINK_CHECK(cudaStreamCreate)
GPUMOD_LINK_CHECK(cudaStreamCreateWithFlags)
GPUMOD_LINK_CHECK(cudaStreamCreateWithPriority)
GPUMOD_LINK_CHECK(cudaStreamDestroy)
GPUMOD_LINK_CHECK(cudaStreamSynchronize)
GPUMOD_LINK_CHECK(cudaStreamWaitEvent)
GPUMOD_LINK_CHECK(cudaStreamQuery)
GPUMOD_LINK_CHECK(cudaStreamGetFlags)
GPUMOD_LINK_CHECK(cudaStreamGetPriority)
GPUMOD_LINK_CHECK(cudaStreamGetId)
GPUMOD_LINK_CHECK(cudaStreamAddCallback)
GPUMOD_LINK_CHECK(cudaStreamAttachMemAsync)
GPUMOD_LINK_CHECK(cudaStreamBeginCapture)
GPUMOD_LINK_CHECK(cudaStreamEndCapture)
GPUMOD_LINK_CHECK(cudaStreamIsCapturing)
GPUMOD_LINK_CHECK(cudaStreamGetCaptureInfo)
GPUMOD_LINK_CHECK(cudaStreamUpdateCaptureDependencies)
GPUMOD_LINK_CHECK(cudaThreadExchangeStreamCaptureMode)
GPUMOD_LINK_CHECK(cudaStreamCopyAttributes)
GPUMOD_LINK_CHECK(cudaStreamGetAttribute)
GPUMOD_LINK_CHECK(cudaStreamSetAttribute)

// Event Management
GPUMOD_LINK_CHECK(cudaEventCreate)
GPUMOD_LINK_CHECK(cudaEventCreateWithFlags)
GPUMOD_LINK_CHECK(cudaEventDestroy)
GPUMOD_LINK_CHECK(cudaEventRecord)
GPUMOD_LINK_CHECK(cudaEventRecordWithFlags)
GPUMOD_LINK_CHECK(cudaEventSynchronize)
GPUMOD_LINK_CHECK(cudaEventQuery)
GPUMOD_LINK_CHECK(cudaEventElapsedTime)

// External Semaphores
GPUMOD_LINK_CHECK(cudaImportExternalSemaphore)
GPUMOD_LINK_CHECK(cudaSignalExternalSemaphoresAsync)
GPUMOD_LINK_CHECK(cudaWaitExternalSemaphoresAsync)
GPUMOD_LINK_CHECK(cudaDestroyExternalSemaphore)

// Execution Control
GPUMOD_LINK_CHECK(cudaLaunchKernel)
GPUMOD_LINK_CHECK(cudaLaunchCooperativeKernel)
GPUMOD_LINK_CHECK(cudaLaunchHostFunc)

// Occupancy
GPUMOD_LINK_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor)
GPUMOD_LINK_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags)
GPUMOD_LINK_CHECK(cudaOccupancyAvailableDynamicSMemPerBlock)
GPUMOD_LINK_CHECK(cudaOccupancyMaxPotentialClusterSize)
GPUMOD_LINK_CHECK(cudaOccupancyMaxActiveClusters)

// Function Configuration
GPUMOD_LINK_CHECK(cudaFuncGetAttributes)
GPUMOD_LINK_CHECK(cudaFuncSetAttribute)
GPUMOD_LINK_CHECK(cudaFuncSetCacheConfig)
GPUMOD_LINK_CHECK(cudaFuncSetSharedMemConfig)
GPUMOD_LINK_CHECK(cudaFuncGetName)
GPUMOD_LINK_CHECK(cudaFuncGetParamInfo)

// Graph — Graph / Node Lifecycle
GPUMOD_LINK_CHECK(cudaGraphCreate)
GPUMOD_LINK_CHECK(cudaGraphDestroy)
GPUMOD_LINK_CHECK(cudaGraphAddDependencies)
GPUMOD_LINK_CHECK(cudaGraphRemoveDependencies)
GPUMOD_LINK_CHECK(cudaGraphGetEdges)
GPUMOD_LINK_CHECK(cudaGraphGetNodes)
GPUMOD_LINK_CHECK(cudaGraphGetRootNodes)
GPUMOD_LINK_CHECK(cudaGraphNodeGetDependencies)
GPUMOD_LINK_CHECK(cudaGraphNodeGetDependentNodes)
GPUMOD_LINK_CHECK(cudaGraphNodeGetType)
GPUMOD_LINK_CHECK(cudaGraphNodeGetEnabled)
GPUMOD_LINK_CHECK(cudaGraphNodeSetEnabled)
GPUMOD_LINK_CHECK(cudaGraphClone)
GPUMOD_LINK_CHECK(cudaGraphNodeFindInClone)
GPUMOD_LINK_CHECK(cudaGraphDebugDotPrint)
GPUMOD_LINK_CHECK(cudaGraphDestroyNode)

// Graph — Node Addition
GPUMOD_LINK_CHECK(cudaGraphAddEmptyNode)
GPUMOD_LINK_CHECK(cudaGraphAddKernelNode)
GPUMOD_LINK_CHECK(cudaGraphAddMemcpyNode)
GPUMOD_LINK_CHECK(cudaGraphAddMemcpyNode1D)
GPUMOD_LINK_CHECK(cudaGraphAddMemcpyNodeFromSymbol)
GPUMOD_LINK_CHECK(cudaGraphAddMemcpyNodeToSymbol)
GPUMOD_LINK_CHECK(cudaGraphAddMemsetNode)
GPUMOD_LINK_CHECK(cudaGraphAddHostNode)
GPUMOD_LINK_CHECK(cudaGraphAddChildGraphNode)
GPUMOD_LINK_CHECK(cudaGraphAddEventRecordNode)
GPUMOD_LINK_CHECK(cudaGraphAddEventWaitNode)
GPUMOD_LINK_CHECK(cudaGraphAddExternalSemaphoresSignalNode)
GPUMOD_LINK_CHECK(cudaGraphAddExternalSemaphoresWaitNode)
GPUMOD_LINK_CHECK(cudaGraphAddMemAllocNode)
GPUMOD_LINK_CHECK(cudaGraphAddMemFreeNode)
GPUMOD_LINK_CHECK(cudaGraphAddNode)

// Graph — Node Params
GPUMOD_LINK_CHECK(cudaGraphKernelNodeGetParams)
GPUMOD_LINK_CHECK(cudaGraphKernelNodeSetParams)
GPUMOD_LINK_CHECK(cudaGraphKernelNodeCopyAttributes)
GPUMOD_LINK_CHECK(cudaGraphKernelNodeGetAttribute)
GPUMOD_LINK_CHECK(cudaGraphKernelNodeSetAttribute)
GPUMOD_LINK_CHECK(cudaGraphMemcpyNodeGetParams)
GPUMOD_LINK_CHECK(cudaGraphMemcpyNodeSetParams)
GPUMOD_LINK_CHECK(cudaGraphMemcpyNodeSetParams1D)
GPUMOD_LINK_CHECK(cudaGraphMemcpyNodeSetParamsFromSymbol)
GPUMOD_LINK_CHECK(cudaGraphMemcpyNodeSetParamsToSymbol)
GPUMOD_LINK_CHECK(cudaGraphMemsetNodeGetParams)
GPUMOD_LINK_CHECK(cudaGraphMemsetNodeSetParams)
GPUMOD_LINK_CHECK(cudaGraphHostNodeGetParams)
GPUMOD_LINK_CHECK(cudaGraphHostNodeSetParams)
GPUMOD_LINK_CHECK(cudaGraphChildGraphNodeGetGraph)
GPUMOD_LINK_CHECK(cudaGraphEventRecordNodeGetEvent)
GPUMOD_LINK_CHECK(cudaGraphEventRecordNodeSetEvent)
GPUMOD_LINK_CHECK(cudaGraphEventWaitNodeGetEvent)
GPUMOD_LINK_CHECK(cudaGraphEventWaitNodeSetEvent)
GPUMOD_LINK_CHECK(cudaGraphExternalSemaphoresSignalNodeGetParams)
GPUMOD_LINK_CHECK(cudaGraphExternalSemaphoresSignalNodeSetParams)
GPUMOD_LINK_CHECK(cudaGraphExternalSemaphoresWaitNodeGetParams)
GPUMOD_LINK_CHECK(cudaGraphExternalSemaphoresWaitNodeSetParams)
GPUMOD_LINK_CHECK(cudaGraphMemAllocNodeGetParams)
GPUMOD_LINK_CHECK(cudaGraphMemFreeNodeGetParams)
GPUMOD_LINK_CHECK(cudaGraphNodeSetParams)

// Graph — Execution
GPUMOD_LINK_CHECK(cudaGraphInstantiate)
GPUMOD_LINK_CHECK(cudaGraphInstantiateWithFlags)
GPUMOD_LINK_CHECK(cudaGraphInstantiateWithParams)
GPUMOD_LINK_CHECK(cudaGraphExecDestroy)
GPUMOD_LINK_CHECK(cudaGraphExecUpdate)
GPUMOD_LINK_CHECK(cudaGraphExecGetFlags)
GPUMOD_LINK_CHECK(cudaGraphLaunch)
GPUMOD_LINK_CHECK(cudaGraphUpload)
GPUMOD_LINK_CHECK(cudaGraphExecKernelNodeSetParams)
GPUMOD_LINK_CHECK(cudaGraphExecMemcpyNodeSetParams)
GPUMOD_LINK_CHECK(cudaGraphExecMemcpyNodeSetParams1D)
GPUMOD_LINK_CHECK(cudaGraphExecMemcpyNodeSetParamsFromSymbol)
GPUMOD_LINK_CHECK(cudaGraphExecMemcpyNodeSetParamsToSymbol)
GPUMOD_LINK_CHECK(cudaGraphExecMemsetNodeSetParams)
GPUMOD_LINK_CHECK(cudaGraphExecHostNodeSetParams)
GPUMOD_LINK_CHECK(cudaGraphExecChildGraphNodeSetParams)
GPUMOD_LINK_CHECK(cudaGraphExecEventRecordNodeSetEvent)
GPUMOD_LINK_CHECK(cudaGraphExecEventWaitNodeSetEvent)
GPUMOD_LINK_CHECK(cudaGraphExecExternalSemaphoresSignalNodeSetParams)
GPUMOD_LINK_CHECK(cudaGraphExecExternalSemaphoresWaitNodeSetParams)
GPUMOD_LINK_CHECK(cudaGraphExecNodeSetParams)
GPUMOD_LINK_CHECK(cudaGraphConditionalHandleCreate)

// Texture / Surface
GPUMOD_LINK_CHECK(cudaCreateTextureObject)
GPUMOD_LINK_CHECK(cudaDestroyTextureObject)
GPUMOD_LINK_CHECK(cudaGetTextureObjectResourceDesc)
GPUMOD_LINK_CHECK(cudaGetTextureObjectTextureDesc)
GPUMOD_LINK_CHECK(cudaGetTextureObjectResourceViewDesc)
GPUMOD_LINK_CHECK(cudaCreateSurfaceObject)
GPUMOD_LINK_CHECK(cudaDestroySurfaceObject)
GPUMOD_LINK_CHECK(cudaGetSurfaceObjectResourceDesc)
GPUMOD_LINK_CHECK(cudaCreateChannelDesc)
GPUMOD_LINK_CHECK(cudaGetChannelDesc)

// Symbol / Kernel
GPUMOD_LINK_CHECK(cudaGetSymbolAddress)
GPUMOD_LINK_CHECK(cudaGetSymbolSize)
GPUMOD_LINK_CHECK(cudaGetKernel)
GPUMOD_LINK_CHECK(cudaGetFuncBySymbol)

// User Objects
GPUMOD_LINK_CHECK(cudaUserObjectCreate)
GPUMOD_LINK_CHECK(cudaUserObjectRetain)
GPUMOD_LINK_CHECK(cudaUserObjectRelease)
GPUMOD_LINK_CHECK(cudaGraphRetainUserObject)
GPUMOD_LINK_CHECK(cudaGraphReleaseUserObject)

// Graphics Interop
GPUMOD_LINK_CHECK(cudaGraphicsUnregisterResource)
GPUMOD_LINK_CHECK(cudaGraphicsResourceSetMapFlags)
GPUMOD_LINK_CHECK(cudaGraphicsMapResources)
GPUMOD_LINK_CHECK(cudaGraphicsUnmapResources)
GPUMOD_LINK_CHECK(cudaGraphicsResourceGetMappedPointer)
GPUMOD_LINK_CHECK(cudaGraphicsSubResourceGetMappedArray)

// Driver Entry Points
GPUMOD_LINK_CHECK(cudaGetDriverEntryPoint)
GPUMOD_LINK_CHECK(cudaGetDriverEntryPointByVersion)
GPUMOD_LINK_CHECK(cudaGetExportTable)

} // namespace gpumod::cuda::test
