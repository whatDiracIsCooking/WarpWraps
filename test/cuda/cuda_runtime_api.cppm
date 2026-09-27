// cuda_runtime_api.cppm - Compile-time tests for wwr.cuda.cuda_runtime_api

module;

#include "test/shared/link_check.h"

export module wwr.test.cuda.cuda_runtime_api;

import std;
import wwr.cuda.cuda_runtime_api;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time tests for wwr.cuda.cuda_runtime_api
//
// The module is pure re-export (using declarations + constexpr flag values).
// Runtime tests for the underlying CUDA API would just test CUDA itself.
// We verify at compile-time that:
//   1. Constexpr flags have the correct values (the #undef/re-export is correct)
//   2. Key enum values with CUDA-specified values are correct
//   3. Opaque handle types have the expected type traits
//   4. Struct types satisfy trivial copyability and standard layout (C-interop guarantee)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace wwr::cuda::test {

using namespace wwr::cuda;

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
WWR_LINK_CHECK(cudaGetErrorName)
WWR_LINK_CHECK(cudaGetErrorString)
WWR_LINK_CHECK(cudaGetLastError)
WWR_LINK_CHECK(cudaPeekAtLastError)

// Device Management
WWR_LINK_CHECK(cudaGetDevice)
WWR_LINK_CHECK(cudaSetDevice)
WWR_LINK_CHECK(cudaGetDeviceCount)
WWR_LINK_CHECK(cudaGetDeviceProperties)
WWR_LINK_CHECK(cudaDeviceSynchronize)
WWR_LINK_CHECK(cudaDeviceReset)
WWR_LINK_CHECK(cudaSetDeviceFlags)
WWR_LINK_CHECK(cudaGetDeviceFlags)
WWR_LINK_CHECK(cudaChooseDevice)
WWR_LINK_CHECK(cudaDeviceGetAttribute)
WWR_LINK_CHECK(cudaDeviceGetByPCIBusId)
WWR_LINK_CHECK(cudaDeviceGetPCIBusId)
WWR_LINK_CHECK(cudaDeviceGetDefaultMemPool)
WWR_LINK_CHECK(cudaDeviceSetMemPool)
WWR_LINK_CHECK(cudaDeviceGetMemPool)
WWR_LINK_CHECK(cudaDeviceSetLimit)
WWR_LINK_CHECK(cudaDeviceGetLimit)
WWR_LINK_CHECK(cudaDeviceSetCacheConfig)
WWR_LINK_CHECK(cudaDeviceGetCacheConfig)
WWR_LINK_CHECK(cudaDeviceSetSharedMemConfig)
WWR_LINK_CHECK(cudaDeviceGetSharedMemConfig)
WWR_LINK_CHECK(cudaDeviceGetStreamPriorityRange)
WWR_LINK_CHECK(cudaDeviceGetTexture1DLinearMaxWidth)
WWR_LINK_CHECK(cudaDeviceCanAccessPeer)
WWR_LINK_CHECK(cudaDeviceEnablePeerAccess)
WWR_LINK_CHECK(cudaDeviceDisablePeerAccess)
WWR_LINK_CHECK(cudaDeviceGetP2PAttribute)
WWR_LINK_CHECK(cudaDeviceFlushGPUDirectRDMAWrites)
WWR_LINK_CHECK(cudaCtxResetPersistingL2Cache)
WWR_LINK_CHECK(cudaDeviceGetNvSciSyncAttributes)
WWR_LINK_CHECK(cudaDeviceRegisterAsyncNotification)
WWR_LINK_CHECK(cudaDeviceUnregisterAsyncNotification)
WWR_LINK_CHECK(cudaDeviceGetGraphMemAttribute)
WWR_LINK_CHECK(cudaDeviceSetGraphMemAttribute)
WWR_LINK_CHECK(cudaDeviceGraphMemTrim)
WWR_LINK_CHECK(cudaDeviceGetHostAtomicCapabilities)
WWR_LINK_CHECK(cudaDeviceGetP2PAtomicCapabilities)
WWR_LINK_CHECK(cudaDriverGetVersion)
WWR_LINK_CHECK(cudaRuntimeGetVersion)

// Memory — Device Allocation
WWR_LINK_CHECK(cudaMalloc)
WWR_LINK_CHECK(cudaMallocAsync)
WWR_LINK_CHECK(cudaFree)
WWR_LINK_CHECK(cudaFreeAsync)
WWR_LINK_CHECK(cudaMallocPitch)
WWR_LINK_CHECK(cudaMalloc3D)
WWR_LINK_CHECK(cudaMalloc3DArray)
WWR_LINK_CHECK(cudaMallocArray)
WWR_LINK_CHECK(cudaFreeArray)
WWR_LINK_CHECK(cudaMallocMipmappedArray)
WWR_LINK_CHECK(cudaFreeMipmappedArray)
WWR_LINK_CHECK(cudaMallocManaged)

// Memory — Host (Pinned)
WWR_LINK_CHECK(cudaMallocHost)
WWR_LINK_CHECK(cudaHostAlloc)
WWR_LINK_CHECK(cudaFreeHost)
WWR_LINK_CHECK(cudaHostRegister)
WWR_LINK_CHECK(cudaHostUnregister)
WWR_LINK_CHECK(cudaHostGetDevicePointer)
WWR_LINK_CHECK(cudaHostGetFlags)

// Memory — Copy
WWR_LINK_CHECK(cudaMemcpy)
WWR_LINK_CHECK(cudaMemcpyAsync)
WWR_LINK_CHECK(cudaMemcpy2D)
WWR_LINK_CHECK(cudaMemcpy2DAsync)
WWR_LINK_CHECK(cudaMemcpy2DToArray)
WWR_LINK_CHECK(cudaMemcpy2DToArrayAsync)
WWR_LINK_CHECK(cudaMemcpy2DFromArray)
WWR_LINK_CHECK(cudaMemcpy2DFromArrayAsync)
WWR_LINK_CHECK(cudaMemcpy2DArrayToArray)
WWR_LINK_CHECK(cudaMemcpy3D)
WWR_LINK_CHECK(cudaMemcpy3DAsync)
WWR_LINK_CHECK(cudaMemcpy3DPeer)
WWR_LINK_CHECK(cudaMemcpy3DPeerAsync)
WWR_LINK_CHECK(cudaMemcpyPeer)
WWR_LINK_CHECK(cudaMemcpyPeerAsync)
WWR_LINK_CHECK(cudaMemcpyToArray)
WWR_LINK_CHECK(cudaMemcpyToArrayAsync)
WWR_LINK_CHECK(cudaMemcpyFromArray)
WWR_LINK_CHECK(cudaMemcpyFromArrayAsync)
WWR_LINK_CHECK(cudaMemcpyArrayToArray)
WWR_LINK_CHECK(cudaMemcpyToSymbol)
WWR_LINK_CHECK(cudaMemcpyToSymbolAsync)
WWR_LINK_CHECK(cudaMemcpyFromSymbol)
WWR_LINK_CHECK(cudaMemcpyFromSymbolAsync)

// Memory — Set
WWR_LINK_CHECK(cudaMemset)
WWR_LINK_CHECK(cudaMemsetAsync)
WWR_LINK_CHECK(cudaMemset2D)
WWR_LINK_CHECK(cudaMemset2DAsync)
WWR_LINK_CHECK(cudaMemset3D)
WWR_LINK_CHECK(cudaMemset3DAsync)

// Memory — Info / Advise / Query
WWR_LINK_CHECK(cudaMemGetInfo)
WWR_LINK_CHECK(cudaMemPrefetchAsync)
WWR_LINK_CHECK(cudaMemAdvise)
WWR_LINK_CHECK(cudaMemRangeGetAttribute)
WWR_LINK_CHECK(cudaMemRangeGetAttributes)
WWR_LINK_CHECK(cudaPointerGetAttributes)

// Memory — Array Operations
WWR_LINK_CHECK(cudaArrayGetInfo)
WWR_LINK_CHECK(cudaArrayGetPlane)
WWR_LINK_CHECK(cudaArrayGetMemoryRequirements)
WWR_LINK_CHECK(cudaArrayGetSparseProperties)
WWR_LINK_CHECK(cudaGetMipmappedArrayLevel)

// Memory — Pools
WWR_LINK_CHECK(cudaMemPoolCreate)
WWR_LINK_CHECK(cudaMemPoolDestroy)
WWR_LINK_CHECK(cudaMallocFromPoolAsync)
WWR_LINK_CHECK(cudaMemPoolTrimTo)
WWR_LINK_CHECK(cudaMemPoolSetAttribute)
WWR_LINK_CHECK(cudaMemPoolGetAttribute)
WWR_LINK_CHECK(cudaMemPoolSetAccess)
WWR_LINK_CHECK(cudaMemPoolGetAccess)
WWR_LINK_CHECK(cudaMemPoolExportToShareableHandle)
WWR_LINK_CHECK(cudaMemPoolImportFromShareableHandle)
WWR_LINK_CHECK(cudaMemPoolExportPointer)
WWR_LINK_CHECK(cudaMemPoolImportPointer)

// Memory — External
WWR_LINK_CHECK(cudaImportExternalMemory)
WWR_LINK_CHECK(cudaExternalMemoryGetMappedBuffer)
WWR_LINK_CHECK(cudaExternalMemoryGetMappedMipmappedArray)
WWR_LINK_CHECK(cudaDestroyExternalMemory)

// Memory — IPC
WWR_LINK_CHECK(cudaIpcGetMemHandle)
WWR_LINK_CHECK(cudaIpcOpenMemHandle)
WWR_LINK_CHECK(cudaIpcCloseMemHandle)
WWR_LINK_CHECK(cudaIpcGetEventHandle)
WWR_LINK_CHECK(cudaIpcOpenEventHandle)

// Stream Management
WWR_LINK_CHECK(cudaStreamCreate)
WWR_LINK_CHECK(cudaStreamCreateWithFlags)
WWR_LINK_CHECK(cudaStreamCreateWithPriority)
WWR_LINK_CHECK(cudaStreamDestroy)
WWR_LINK_CHECK(cudaStreamSynchronize)
WWR_LINK_CHECK(cudaStreamWaitEvent)
WWR_LINK_CHECK(cudaStreamQuery)
WWR_LINK_CHECK(cudaStreamGetFlags)
WWR_LINK_CHECK(cudaStreamGetPriority)
WWR_LINK_CHECK(cudaStreamGetId)
WWR_LINK_CHECK(cudaStreamAddCallback)
WWR_LINK_CHECK(cudaStreamAttachMemAsync)
WWR_LINK_CHECK(cudaStreamBeginCapture)
WWR_LINK_CHECK(cudaStreamEndCapture)
WWR_LINK_CHECK(cudaStreamIsCapturing)
WWR_LINK_CHECK(cudaStreamGetCaptureInfo)
WWR_LINK_CHECK(cudaStreamUpdateCaptureDependencies)
WWR_LINK_CHECK(cudaThreadExchangeStreamCaptureMode)
WWR_LINK_CHECK(cudaStreamCopyAttributes)
WWR_LINK_CHECK(cudaStreamGetAttribute)
WWR_LINK_CHECK(cudaStreamSetAttribute)

// Event Management
WWR_LINK_CHECK(cudaEventCreate)
WWR_LINK_CHECK(cudaEventCreateWithFlags)
WWR_LINK_CHECK(cudaEventDestroy)
WWR_LINK_CHECK(cudaEventRecord)
WWR_LINK_CHECK(cudaEventRecordWithFlags)
WWR_LINK_CHECK(cudaEventSynchronize)
WWR_LINK_CHECK(cudaEventQuery)
WWR_LINK_CHECK(cudaEventElapsedTime)

// External Semaphores
WWR_LINK_CHECK(cudaImportExternalSemaphore)
WWR_LINK_CHECK(cudaSignalExternalSemaphoresAsync)
WWR_LINK_CHECK(cudaWaitExternalSemaphoresAsync)
WWR_LINK_CHECK(cudaDestroyExternalSemaphore)

// Execution Control
WWR_LINK_CHECK(cudaLaunchKernel)
WWR_LINK_CHECK(cudaLaunchCooperativeKernel)
WWR_LINK_CHECK(cudaLaunchHostFunc)

// Occupancy
WWR_LINK_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor)
WWR_LINK_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags)
WWR_LINK_CHECK(cudaOccupancyAvailableDynamicSMemPerBlock)
WWR_LINK_CHECK(cudaOccupancyMaxPotentialClusterSize)
WWR_LINK_CHECK(cudaOccupancyMaxActiveClusters)

// Function Configuration
WWR_LINK_CHECK(cudaFuncGetAttributes)
WWR_LINK_CHECK(cudaFuncSetAttribute)
WWR_LINK_CHECK(cudaFuncSetCacheConfig)
WWR_LINK_CHECK(cudaFuncSetSharedMemConfig)
WWR_LINK_CHECK(cudaFuncGetName)
WWR_LINK_CHECK(cudaFuncGetParamInfo)

// Graph — Graph / Node Lifecycle
WWR_LINK_CHECK(cudaGraphCreate)
WWR_LINK_CHECK(cudaGraphDestroy)
WWR_LINK_CHECK(cudaGraphAddDependencies)
WWR_LINK_CHECK(cudaGraphRemoveDependencies)
WWR_LINK_CHECK(cudaGraphGetEdges)
WWR_LINK_CHECK(cudaGraphGetNodes)
WWR_LINK_CHECK(cudaGraphGetRootNodes)
WWR_LINK_CHECK(cudaGraphNodeGetDependencies)
WWR_LINK_CHECK(cudaGraphNodeGetDependentNodes)
WWR_LINK_CHECK(cudaGraphNodeGetType)
WWR_LINK_CHECK(cudaGraphNodeGetEnabled)
WWR_LINK_CHECK(cudaGraphNodeSetEnabled)
WWR_LINK_CHECK(cudaGraphClone)
WWR_LINK_CHECK(cudaGraphNodeFindInClone)
WWR_LINK_CHECK(cudaGraphDebugDotPrint)
WWR_LINK_CHECK(cudaGraphDestroyNode)

// Graph — Node Addition
WWR_LINK_CHECK(cudaGraphAddEmptyNode)
WWR_LINK_CHECK(cudaGraphAddKernelNode)
WWR_LINK_CHECK(cudaGraphAddMemcpyNode)
WWR_LINK_CHECK(cudaGraphAddMemcpyNode1D)
WWR_LINK_CHECK(cudaGraphAddMemcpyNodeFromSymbol)
WWR_LINK_CHECK(cudaGraphAddMemcpyNodeToSymbol)
WWR_LINK_CHECK(cudaGraphAddMemsetNode)
WWR_LINK_CHECK(cudaGraphAddHostNode)
WWR_LINK_CHECK(cudaGraphAddChildGraphNode)
WWR_LINK_CHECK(cudaGraphAddEventRecordNode)
WWR_LINK_CHECK(cudaGraphAddEventWaitNode)
WWR_LINK_CHECK(cudaGraphAddExternalSemaphoresSignalNode)
WWR_LINK_CHECK(cudaGraphAddExternalSemaphoresWaitNode)
WWR_LINK_CHECK(cudaGraphAddMemAllocNode)
WWR_LINK_CHECK(cudaGraphAddMemFreeNode)
WWR_LINK_CHECK(cudaGraphAddNode)

// Graph — Node Params
WWR_LINK_CHECK(cudaGraphKernelNodeGetParams)
WWR_LINK_CHECK(cudaGraphKernelNodeSetParams)
WWR_LINK_CHECK(cudaGraphKernelNodeCopyAttributes)
WWR_LINK_CHECK(cudaGraphKernelNodeGetAttribute)
WWR_LINK_CHECK(cudaGraphKernelNodeSetAttribute)
WWR_LINK_CHECK(cudaGraphMemcpyNodeGetParams)
WWR_LINK_CHECK(cudaGraphMemcpyNodeSetParams)
WWR_LINK_CHECK(cudaGraphMemcpyNodeSetParams1D)
WWR_LINK_CHECK(cudaGraphMemcpyNodeSetParamsFromSymbol)
WWR_LINK_CHECK(cudaGraphMemcpyNodeSetParamsToSymbol)
WWR_LINK_CHECK(cudaGraphMemsetNodeGetParams)
WWR_LINK_CHECK(cudaGraphMemsetNodeSetParams)
WWR_LINK_CHECK(cudaGraphHostNodeGetParams)
WWR_LINK_CHECK(cudaGraphHostNodeSetParams)
WWR_LINK_CHECK(cudaGraphChildGraphNodeGetGraph)
WWR_LINK_CHECK(cudaGraphEventRecordNodeGetEvent)
WWR_LINK_CHECK(cudaGraphEventRecordNodeSetEvent)
WWR_LINK_CHECK(cudaGraphEventWaitNodeGetEvent)
WWR_LINK_CHECK(cudaGraphEventWaitNodeSetEvent)
WWR_LINK_CHECK(cudaGraphExternalSemaphoresSignalNodeGetParams)
WWR_LINK_CHECK(cudaGraphExternalSemaphoresSignalNodeSetParams)
WWR_LINK_CHECK(cudaGraphExternalSemaphoresWaitNodeGetParams)
WWR_LINK_CHECK(cudaGraphExternalSemaphoresWaitNodeSetParams)
WWR_LINK_CHECK(cudaGraphMemAllocNodeGetParams)
WWR_LINK_CHECK(cudaGraphMemFreeNodeGetParams)
WWR_LINK_CHECK(cudaGraphNodeSetParams)

// Graph — Execution
WWR_LINK_CHECK(cudaGraphInstantiate)
WWR_LINK_CHECK(cudaGraphInstantiateWithFlags)
WWR_LINK_CHECK(cudaGraphInstantiateWithParams)
WWR_LINK_CHECK(cudaGraphExecDestroy)
WWR_LINK_CHECK(cudaGraphExecUpdate)
WWR_LINK_CHECK(cudaGraphExecGetFlags)
WWR_LINK_CHECK(cudaGraphLaunch)
WWR_LINK_CHECK(cudaGraphUpload)
WWR_LINK_CHECK(cudaGraphExecKernelNodeSetParams)
WWR_LINK_CHECK(cudaGraphExecMemcpyNodeSetParams)
WWR_LINK_CHECK(cudaGraphExecMemcpyNodeSetParams1D)
WWR_LINK_CHECK(cudaGraphExecMemcpyNodeSetParamsFromSymbol)
WWR_LINK_CHECK(cudaGraphExecMemcpyNodeSetParamsToSymbol)
WWR_LINK_CHECK(cudaGraphExecMemsetNodeSetParams)
WWR_LINK_CHECK(cudaGraphExecHostNodeSetParams)
WWR_LINK_CHECK(cudaGraphExecChildGraphNodeSetParams)
WWR_LINK_CHECK(cudaGraphExecEventRecordNodeSetEvent)
WWR_LINK_CHECK(cudaGraphExecEventWaitNodeSetEvent)
WWR_LINK_CHECK(cudaGraphExecExternalSemaphoresSignalNodeSetParams)
WWR_LINK_CHECK(cudaGraphExecExternalSemaphoresWaitNodeSetParams)
WWR_LINK_CHECK(cudaGraphExecNodeSetParams)
WWR_LINK_CHECK(cudaGraphConditionalHandleCreate)

// Texture / Surface
WWR_LINK_CHECK(cudaCreateTextureObject)
WWR_LINK_CHECK(cudaDestroyTextureObject)
WWR_LINK_CHECK(cudaGetTextureObjectResourceDesc)
WWR_LINK_CHECK(cudaGetTextureObjectTextureDesc)
WWR_LINK_CHECK(cudaGetTextureObjectResourceViewDesc)
WWR_LINK_CHECK(cudaCreateSurfaceObject)
WWR_LINK_CHECK(cudaDestroySurfaceObject)
WWR_LINK_CHECK(cudaGetSurfaceObjectResourceDesc)
WWR_LINK_CHECK(cudaCreateChannelDesc)
WWR_LINK_CHECK(cudaGetChannelDesc)

// Symbol / Kernel
WWR_LINK_CHECK(cudaGetSymbolAddress)
WWR_LINK_CHECK(cudaGetSymbolSize)
WWR_LINK_CHECK(cudaGetKernel)
WWR_LINK_CHECK(cudaGetFuncBySymbol)

// User Objects
WWR_LINK_CHECK(cudaUserObjectCreate)
WWR_LINK_CHECK(cudaUserObjectRetain)
WWR_LINK_CHECK(cudaUserObjectRelease)
WWR_LINK_CHECK(cudaGraphRetainUserObject)
WWR_LINK_CHECK(cudaGraphReleaseUserObject)

// Graphics Interop
WWR_LINK_CHECK(cudaGraphicsUnregisterResource)
WWR_LINK_CHECK(cudaGraphicsResourceSetMapFlags)
WWR_LINK_CHECK(cudaGraphicsMapResources)
WWR_LINK_CHECK(cudaGraphicsUnmapResources)
WWR_LINK_CHECK(cudaGraphicsResourceGetMappedPointer)
WWR_LINK_CHECK(cudaGraphicsSubResourceGetMappedArray)

// Driver Entry Points
WWR_LINK_CHECK(cudaGetDriverEntryPoint)
WWR_LINK_CHECK(cudaGetDriverEntryPointByVersion)
WWR_LINK_CHECK(cudaGetExportTable)

} // namespace wwr::cuda::test
