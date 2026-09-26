// cuda.cppm - Compile-time tests for gpumod.cuda.cuda_h

module;

#include "test/shared/link_check.h"
#include <cuda.h>

export module gpumod.test.cuda.cuda_h;

import std;
import gpumod.cuda.cuda_h;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time tests for gpumod.cuda.cuda_h
//
// The module is pure re-export (using declarations).
// Runtime tests for the underlying CUDA Driver API would just test CUDA itself.
// We verify at compile-time that:
//   1. Key enum values with CUDA-specified values are correct
//   2. Opaque handle types have the expected type traits
//   3. Struct types satisfy trivial copyability and standard layout (C-interop guarantee)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace gpumod::test {

using namespace gpumod;

// ────────────────────────────────────────────────────────────────────────
// Enum type checks
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_enum_v<CUresult>);
static_assert(std::is_enum_v<CUarray_format>);
static_assert(std::is_enum_v<CUaddress_mode>);
static_assert(std::is_enum_v<CUfilter_mode>);
static_assert(std::is_enum_v<CUdevice_attribute>);
static_assert(std::is_enum_v<CUctx_flags>);
static_assert(std::is_enum_v<CUstream_flags>);
static_assert(std::is_enum_v<CUevent_flags>);
static_assert(std::is_enum_v<CUevent_sched_flags>);
static_assert(std::is_enum_v<CUjit_option>);
static_assert(std::is_enum_v<CUfunction_attribute>);
static_assert(std::is_enum_v<CUmemorytype>);
static_assert(std::is_enum_v<CUgraphNodeType>);
static_assert(std::is_enum_v<CUgraphExecUpdateResult>);
static_assert(std::is_enum_v<CUmemPool_attribute>);
static_assert(std::is_enum_v<CUmemLocationType>);
static_assert(std::is_enum_v<CUmemAllocationType>);
static_assert(std::is_enum_v<CUmemAllocationHandleType>);
static_assert(std::is_enum_v<CUmemAccess_flags>);
static_assert(std::is_enum_v<CUstreamCaptureStatus>);
static_assert(std::is_enum_v<CUstreamCaptureMode>);
static_assert(std::is_enum_v<CUexternalMemoryHandleType>);
static_assert(std::is_enum_v<CUexternalSemaphoreHandleType>);
static_assert(std::is_enum_v<CUdevice_P2PAttribute>);
static_assert(std::is_enum_v<CUgraphMem_attribute>);
static_assert(std::is_enum_v<CUsynchronizationPolicy>);
static_assert(std::is_enum_v<CUlaunchAttributeID>);
static_assert(std::is_enum_v<CUprocessState>);

// ────────────────────────────────────────────────────────────────────────
// Enum values: CUresult
// ────────────────────────────────────────────────────────────────────────

// CUDA_SUCCESS == 0 is a guaranteed Driver API contract
static_assert(static_cast<int>(CUDA_SUCCESS) == 0);
static_assert(static_cast<int>(CUDA_ERROR_INVALID_VALUE) == 1);
static_assert(static_cast<int>(CUDA_ERROR_OUT_OF_MEMORY) == 2);
static_assert(static_cast<int>(CUDA_ERROR_NOT_INITIALIZED) == 3);
static_assert(static_cast<int>(CUDA_ERROR_NO_DEVICE) == 100);
static_assert(static_cast<int>(CUDA_ERROR_INVALID_DEVICE) == 101);

// ────────────────────────────────────────────────────────────────────────
// Enum values: CUctx_flags
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CU_CTX_SCHED_AUTO) == 0x00);
static_assert(static_cast<int>(CU_CTX_SCHED_SPIN) == 0x01);
static_assert(static_cast<int>(CU_CTX_SCHED_YIELD) == 0x02);
static_assert(static_cast<int>(CU_CTX_BLOCKING_SYNC) == 0x04);

// ────────────────────────────────────────────────────────────────────────
// Enum values: CUstream_flags
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CU_STREAM_DEFAULT) == 0x0);
static_assert(static_cast<int>(CU_STREAM_NON_BLOCKING) == 0x1);

// ────────────────────────────────────────────────────────────────────────
// Enum values: CUevent_flags
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CU_EVENT_DEFAULT) == 0x0);
static_assert(static_cast<int>(CU_EVENT_BLOCKING_SYNC) == 0x1);
static_assert(static_cast<int>(CU_EVENT_DISABLE_TIMING) == 0x2);
static_assert(static_cast<int>(CU_EVENT_INTERPROCESS) == 0x4);

// ────────────────────────────────────────────────────────────────────────
// Enum values: CUarray_format
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CU_AD_FORMAT_UNSIGNED_INT8) == 0x01);
static_assert(static_cast<int>(CU_AD_FORMAT_HALF) == 0x10);
static_assert(static_cast<int>(CU_AD_FORMAT_FLOAT) == 0x20);

// ────────────────────────────────────────────────────────────────────────
// Enum values: CUmemorytype
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CU_MEMORYTYPE_HOST) == 0x01);
static_assert(static_cast<int>(CU_MEMORYTYPE_DEVICE) == 0x02);
static_assert(static_cast<int>(CU_MEMORYTYPE_ARRAY) == 0x03);
static_assert(static_cast<int>(CU_MEMORYTYPE_UNIFIED) == 0x04);

// ────────────────────────────────────────────────────────────────────────
// Enum values: CUdevice_attribute (representative subset)
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK) == 1);
static_assert(static_cast<int>(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X) == 2);
static_assert(static_cast<int>(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y) == 3);
static_assert(static_cast<int>(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z) == 4);
static_assert(static_cast<int>(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X) == 5);
static_assert(static_cast<int>(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y) == 6);
static_assert(static_cast<int>(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z) == 7);
static_assert(static_cast<int>(CU_DEVICE_ATTRIBUTE_WARP_SIZE) == 10);

// ────────────────────────────────────────────────────────────────────────
// Enum values: CUfunction_attribute
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK) == 0);
static_assert(static_cast<int>(CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES) == 1);

// ────────────────────────────────────────────────────────────────────────
// Enum values: CUjit_option
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CU_JIT_MAX_REGISTERS) == 0);
static_assert(static_cast<int>(CU_JIT_THREADS_PER_BLOCK) == 1);

// ────────────────────────────────────────────────────────────────────────
// Enum values: CUmemLocationType
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CU_MEM_LOCATION_TYPE_INVALID) == 0x0);
static_assert(static_cast<int>(CU_MEM_LOCATION_TYPE_DEVICE) == 0x1);
static_assert(static_cast<int>(CU_MEM_LOCATION_TYPE_HOST) == 0x2);

// ────────────────────────────────────────────────────────────────────────
// Enum values: CUmemAllocationType
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CU_MEM_ALLOCATION_TYPE_INVALID) == 0x0);
static_assert(static_cast<int>(CU_MEM_ALLOCATION_TYPE_PINNED) == 0x1);

// ────────────────────────────────────────────────────────────────────────
// Enum values: CUmemAllocationHandleType
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CU_MEM_HANDLE_TYPE_NONE) == 0x0);
static_assert(static_cast<int>(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) == 0x1);
static_assert(static_cast<int>(CU_MEM_HANDLE_TYPE_WIN32) == 0x2);
static_assert(static_cast<int>(CU_MEM_HANDLE_TYPE_WIN32_KMT) == 0x4);
static_assert(static_cast<int>(CU_MEM_HANDLE_TYPE_FABRIC) == 0x8);

// ────────────────────────────────────────────────────────────────────────
// Enum values: CUstreamCaptureStatus
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CU_STREAM_CAPTURE_STATUS_NONE) == 0);
static_assert(static_cast<int>(CU_STREAM_CAPTURE_STATUS_ACTIVE) == 1);
static_assert(static_cast<int>(CU_STREAM_CAPTURE_STATUS_INVALIDATED) == 2);

// ────────────────────────────────────────────────────────────────────────
// Enum values: CUexternalMemoryHandleType
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD) == 1);
static_assert(static_cast<int>(CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32) == 2);
static_assert(static_cast<int>(CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT) == 3);
static_assert(static_cast<int>(CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP) == 4);
static_assert(static_cast<int>(CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE) == 5);
static_assert(static_cast<int>(CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE) == 6);
static_assert(static_cast<int>(CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT) == 7);
static_assert(static_cast<int>(CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF) == 8);

// ────────────────────────────────────────────────────────────────────────
// Enum values: CUexternalSemaphoreHandleType
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD) == 1);
static_assert(static_cast<int>(CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32) == 2);
static_assert(static_cast<int>(CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT) == 3);
static_assert(static_cast<int>(CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE) == 4);
static_assert(static_cast<int>(CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE) == 5);
static_assert(static_cast<int>(CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC) == 6);
static_assert(static_cast<int>(CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX) == 7);
static_assert(static_cast<int>(CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX_KMT) == 8);
static_assert(static_cast<int>(CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD) == 9);
static_assert(static_cast<int>(CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32) == 10);

// ────────────────────────────────────────────────────────────────────────
// Enum values: CUgraphNodeType
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CU_GRAPH_NODE_TYPE_KERNEL) == 0);
static_assert(static_cast<int>(CU_GRAPH_NODE_TYPE_MEMCPY) == 1);
static_assert(static_cast<int>(CU_GRAPH_NODE_TYPE_MEMSET) == 2);
static_assert(static_cast<int>(CU_GRAPH_NODE_TYPE_HOST) == 3);
static_assert(static_cast<int>(CU_GRAPH_NODE_TYPE_GRAPH) == 4);
static_assert(static_cast<int>(CU_GRAPH_NODE_TYPE_EMPTY) == 5);
static_assert(static_cast<int>(CU_GRAPH_NODE_TYPE_WAIT_EVENT) == 6);
static_assert(static_cast<int>(CU_GRAPH_NODE_TYPE_EVENT_RECORD) == 7);
static_assert(static_cast<int>(CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL) == 8);
static_assert(static_cast<int>(CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT) == 9);
static_assert(static_cast<int>(CU_GRAPH_NODE_TYPE_MEM_ALLOC) == 10);
static_assert(static_cast<int>(CU_GRAPH_NODE_TYPE_MEM_FREE) == 11);

// ────────────────────────────────────────────────────────────────────────
// Enum values: CUdevice_P2PAttribute
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK) == 1);
static_assert(static_cast<int>(CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED) == 2);
static_assert(static_cast<int>(CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED) == 3);
static_assert(static_cast<int>(CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED) == 4);
static_assert(static_cast<int>(CU_DEVICE_P2P_ATTRIBUTE_ONLY_PARTIAL_NATIVE_ATOMIC_SUPPORTED) == 5);

// ────────────────────────────────────────────────────────────────────────
// Opaque handle types: all driver API handles are pointers
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_pointer_v<CUcontext>);
static_assert(std::is_pointer_v<CUmodule>);
static_assert(std::is_pointer_v<CUfunction>);
static_assert(std::is_pointer_v<CUlibrary>);
static_assert(std::is_pointer_v<CUkernel>);
static_assert(std::is_pointer_v<CUarray>);
static_assert(std::is_pointer_v<CUmipmappedArray>);
static_assert(std::is_pointer_v<CUevent>);
static_assert(std::is_pointer_v<CUstream>);
static_assert(std::is_pointer_v<CUgraph>);
static_assert(std::is_pointer_v<CUgraphNode>);
static_assert(std::is_pointer_v<CUgraphExec>);
static_assert(std::is_pointer_v<CUmemoryPool>);
static_assert(std::is_pointer_v<CUlinkState>);
static_assert(std::is_pointer_v<CUgraphicsResource>);
static_assert(std::is_pointer_v<CUexternalMemory>);
static_assert(std::is_pointer_v<CUexternalSemaphore>);
static_assert(std::is_pointer_v<CUuserObject>);

// All handles must fit in a pointer
static_assert(sizeof(CUcontext) == sizeof(void *));
static_assert(sizeof(CUmodule) == sizeof(void *));
static_assert(sizeof(CUfunction) == sizeof(void *));
static_assert(sizeof(CUstream) == sizeof(void *));
static_assert(sizeof(CUevent) == sizeof(void *));
static_assert(sizeof(CUgraph) == sizeof(void *));
static_assert(sizeof(CUgraphNode) == sizeof(void *));
static_assert(sizeof(CUgraphExec) == sizeof(void *));
static_assert(sizeof(CUmemoryPool) == sizeof(void *));

// CUdeviceptr is an unsigned integer (device virtual address), not a host pointer
static_assert(!std::is_pointer_v<CUdeviceptr>);
static_assert(std::is_integral_v<CUdeviceptr>);
static_assert(sizeof(CUdeviceptr) == 8);

// Texture/surface objects are uint64 (driver-managed indices, not host pointers)
static_assert(!std::is_pointer_v<CUtexObject>);
static_assert(!std::is_pointer_v<CUsurfObject>);
static_assert(std::is_integral_v<CUtexObject>);
static_assert(std::is_integral_v<CUsurfObject>);
static_assert(sizeof(CUtexObject) == 8);
static_assert(sizeof(CUsurfObject) == 8);

// IPC handles are fixed-size byte arrays (CU_IPC_HANDLE_SIZE = 64)
static_assert(!std::is_pointer_v<CUipcEventHandle>);
static_assert(!std::is_pointer_v<CUipcMemHandle>);
static_assert(sizeof(CUipcEventHandle) == 64);
static_assert(sizeof(CUipcMemHandle) == 64);

// ────────────────────────────────────────────────────────────────────────
// Callback / function-pointer types
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_pointer_v<CUhostFn>);
static_assert(std::is_pointer_v<CUstreamCallback>);
static_assert(std::is_pointer_v<CUoccupancyB2DSize>);

// ────────────────────────────────────────────────────────────────────────
// Struct traits: trivial copyability
// All CUDA Driver API structs are passed by value / memcpy'd — they must be
// trivially copyable. A failure here means our module wrapper broke the C-ABI
// contract.
// ────────────────────────────────────────────────────────────────────────

// Memory copy parameter blocks
static_assert(std::is_trivially_copyable_v<CUDA_MEMCPY2D>);
static_assert(std::is_trivially_copyable_v<CUDA_MEMCPY3D>);
static_assert(std::is_trivially_copyable_v<CUDA_MEMCPY3D_PEER>);
static_assert(std::is_trivially_copyable_v<CUDA_MEMCPY_NODE_PARAMS>);

// Array descriptors
static_assert(std::is_trivially_copyable_v<CUDA_ARRAY_DESCRIPTOR>);
static_assert(std::is_trivially_copyable_v<CUDA_ARRAY3D_DESCRIPTOR>);
static_assert(std::is_trivially_copyable_v<CUDA_ARRAY_SPARSE_PROPERTIES>);
static_assert(std::is_trivially_copyable_v<CUDA_ARRAY_MEMORY_REQUIREMENTS>);

// Kernel / launch parameter blocks
static_assert(std::is_trivially_copyable_v<CUDA_KERNEL_NODE_PARAMS>);
static_assert(std::is_trivially_copyable_v<CUDA_KERNEL_NODE_PARAMS_v2>);
static_assert(std::is_trivially_copyable_v<CUDA_KERNEL_NODE_PARAMS_v3>);
static_assert(std::is_trivially_copyable_v<CUDA_LAUNCH_PARAMS>);

// Memset / host node params
static_assert(std::is_trivially_copyable_v<CUDA_MEMSET_NODE_PARAMS>);
static_assert(std::is_trivially_copyable_v<CUDA_MEMSET_NODE_PARAMS_v2>);
static_assert(std::is_trivially_copyable_v<CUDA_HOST_NODE_PARAMS>);
static_assert(std::is_trivially_copyable_v<CUDA_HOST_NODE_PARAMS_v2>);

// Graph lifecycle
static_assert(std::is_trivially_copyable_v<CUDA_GRAPH_INSTANTIATE_PARAMS>);
static_assert(std::is_trivially_copyable_v<CUgraphExecUpdateResultInfo>);
static_assert(std::is_trivially_copyable_v<CUgraphEdgeData>);

// Resource / texture descriptors
static_assert(std::is_trivially_copyable_v<CUDA_RESOURCE_DESC>);
static_assert(std::is_trivially_copyable_v<CUDA_RESOURCE_VIEW_DESC>);
static_assert(std::is_trivially_copyable_v<CUDA_TEXTURE_DESC>);

// Access policy
static_assert(std::is_trivially_copyable_v<CUaccessPolicyWindow>);

// External memory / semaphore descriptors
static_assert(std::is_trivially_copyable_v<CUDA_EXTERNAL_MEMORY_HANDLE_DESC>);
static_assert(std::is_trivially_copyable_v<CUDA_EXTERNAL_MEMORY_BUFFER_DESC>);
static_assert(std::is_trivially_copyable_v<CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC>);
static_assert(std::is_trivially_copyable_v<CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC>);
static_assert(std::is_trivially_copyable_v<CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS>);
static_assert(std::is_trivially_copyable_v<CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS>);

// Memory allocation / pool
static_assert(std::is_trivially_copyable_v<CUmemLocation>);
static_assert(std::is_trivially_copyable_v<CUmemAllocationProp>);
static_assert(std::is_trivially_copyable_v<CUmemAccessDesc>);
static_assert(std::is_trivially_copyable_v<CUmemPoolProps>);
static_assert(std::is_trivially_copyable_v<CUmemPoolPtrExportData>);
static_assert(std::is_trivially_copyable_v<CUDA_MEM_ALLOC_NODE_PARAMS>);
static_assert(std::is_trivially_copyable_v<CUDA_MEM_FREE_NODE_PARAMS>);

// Launch attribute
static_assert(std::is_trivially_copyable_v<CUlaunchAttributeValue>);
static_assert(std::is_trivially_copyable_v<CUlaunchAttribute>);
static_assert(std::is_trivially_copyable_v<CUlaunchConfig>);

// Device properties (legacy)
static_assert(std::is_trivially_copyable_v<CUdevprop>);

// ────────────────────────────────────────────────────────────────────────
// Struct traits: standard layout
// All CUDA structs must be standard layout for C/CUDA interop.
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_standard_layout_v<CUDA_MEMCPY2D>);
static_assert(std::is_standard_layout_v<CUDA_MEMCPY3D>);
static_assert(std::is_standard_layout_v<CUDA_MEMCPY3D_PEER>);
static_assert(std::is_standard_layout_v<CUDA_MEMCPY_NODE_PARAMS>);
static_assert(std::is_standard_layout_v<CUDA_ARRAY_DESCRIPTOR>);
static_assert(std::is_standard_layout_v<CUDA_ARRAY3D_DESCRIPTOR>);
static_assert(std::is_standard_layout_v<CUDA_ARRAY_SPARSE_PROPERTIES>);
static_assert(std::is_standard_layout_v<CUDA_ARRAY_MEMORY_REQUIREMENTS>);
static_assert(std::is_standard_layout_v<CUDA_KERNEL_NODE_PARAMS>);
static_assert(std::is_standard_layout_v<CUDA_KERNEL_NODE_PARAMS_v2>);
static_assert(std::is_standard_layout_v<CUDA_KERNEL_NODE_PARAMS_v3>);
static_assert(std::is_standard_layout_v<CUDA_LAUNCH_PARAMS>);
static_assert(std::is_standard_layout_v<CUDA_MEMSET_NODE_PARAMS>);
static_assert(std::is_standard_layout_v<CUDA_MEMSET_NODE_PARAMS_v2>);
static_assert(std::is_standard_layout_v<CUDA_HOST_NODE_PARAMS>);
static_assert(std::is_standard_layout_v<CUDA_HOST_NODE_PARAMS_v2>);
static_assert(std::is_standard_layout_v<CUDA_GRAPH_INSTANTIATE_PARAMS>);
static_assert(std::is_standard_layout_v<CUgraphExecUpdateResultInfo>);
static_assert(std::is_standard_layout_v<CUgraphEdgeData>);
static_assert(std::is_standard_layout_v<CUDA_RESOURCE_DESC>);
static_assert(std::is_standard_layout_v<CUDA_RESOURCE_VIEW_DESC>);
static_assert(std::is_standard_layout_v<CUDA_TEXTURE_DESC>);
static_assert(std::is_standard_layout_v<CUaccessPolicyWindow>);
static_assert(std::is_standard_layout_v<CUDA_EXTERNAL_MEMORY_HANDLE_DESC>);
static_assert(std::is_standard_layout_v<CUDA_EXTERNAL_MEMORY_BUFFER_DESC>);
static_assert(std::is_standard_layout_v<CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC>);
static_assert(std::is_standard_layout_v<CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC>);
static_assert(std::is_standard_layout_v<CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS>);
static_assert(std::is_standard_layout_v<CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS>);
static_assert(std::is_standard_layout_v<CUmemLocation>);
static_assert(std::is_standard_layout_v<CUmemAllocationProp>);
static_assert(std::is_standard_layout_v<CUmemAccessDesc>);
static_assert(std::is_standard_layout_v<CUmemPoolProps>);
static_assert(std::is_standard_layout_v<CUmemPoolPtrExportData>);
static_assert(std::is_standard_layout_v<CUDA_MEM_ALLOC_NODE_PARAMS>);
static_assert(std::is_standard_layout_v<CUDA_MEM_FREE_NODE_PARAMS>);
static_assert(std::is_standard_layout_v<CUlaunchAttributeValue>);
static_assert(std::is_standard_layout_v<CUlaunchAttribute>);
static_assert(std::is_standard_layout_v<CUlaunchConfig>);
static_assert(std::is_standard_layout_v<CUipcEventHandle>);
static_assert(std::is_standard_layout_v<CUipcMemHandle>);
static_assert(std::is_standard_layout_v<CUdevprop>);

// ────────────────────────────────────────────────────────────────────────
// Link-time symbol resolution
// Forces the linker to resolve every re-exported function symbol,
// catching missing or unresolvable exports that type-only checks miss.
// ────────────────────────────────────────────────────────────────────────

// Initialization
GPUMOD_LINK_CHECK(cuInit)

// Version
GPUMOD_LINK_CHECK(cuDriverGetVersion)

// Device Management
GPUMOD_LINK_CHECK(cuDeviceGet)
GPUMOD_LINK_CHECK(cuDeviceGetCount)
GPUMOD_LINK_CHECK(cuDeviceGetName)
GPUMOD_LINK_CHECK(cuDeviceGetUuid)
GPUMOD_LINK_CHECK(cuDeviceTotalMem)
GPUMOD_LINK_CHECK(cuDeviceGetAttribute)
GPUMOD_LINK_CHECK(cuDeviceGetP2PAttribute)
GPUMOD_LINK_CHECK(cuDeviceGetDefaultMemPool)
GPUMOD_LINK_CHECK(cuDeviceSetMemPool)
GPUMOD_LINK_CHECK(cuDeviceGetMemPool)
GPUMOD_LINK_CHECK(cuDeviceGetTexture1DLinearMaxWidth)
GPUMOD_LINK_CHECK(cuDeviceGetNvSciSyncAttributes)
GPUMOD_LINK_CHECK(cuDeviceRegisterAsyncNotification)
GPUMOD_LINK_CHECK(cuDeviceUnregisterAsyncNotification)

// Context Management
GPUMOD_LINK_CHECK(cuCtxCreate)
GPUMOD_LINK_CHECK(cuCtxDestroy)
GPUMOD_LINK_CHECK(cuCtxGetCurrent)
GPUMOD_LINK_CHECK(cuCtxSetCurrent)
GPUMOD_LINK_CHECK(cuCtxPushCurrent)
GPUMOD_LINK_CHECK(cuCtxPopCurrent)
GPUMOD_LINK_CHECK(cuCtxGetDevice)
GPUMOD_LINK_CHECK(cuCtxGetFlags)
GPUMOD_LINK_CHECK(cuCtxSetFlags)
GPUMOD_LINK_CHECK(cuCtxGetId)
GPUMOD_LINK_CHECK(cuCtxSynchronize)
GPUMOD_LINK_CHECK(cuCtxGetLimit)
GPUMOD_LINK_CHECK(cuCtxSetLimit)
GPUMOD_LINK_CHECK(cuCtxGetCacheConfig)
GPUMOD_LINK_CHECK(cuCtxSetCacheConfig)
GPUMOD_LINK_CHECK(cuCtxGetStreamPriorityRange)
GPUMOD_LINK_CHECK(cuCtxGetSharedMemConfig)
GPUMOD_LINK_CHECK(cuCtxSetSharedMemConfig)
GPUMOD_LINK_CHECK(cuCtxEnablePeerAccess)
GPUMOD_LINK_CHECK(cuCtxDisablePeerAccess)
GPUMOD_LINK_CHECK(cuCtxResetPersistingL2Cache)
GPUMOD_LINK_CHECK(cuCtxGetExecAffinity)

// Module Management
GPUMOD_LINK_CHECK(cuModuleLoad)
GPUMOD_LINK_CHECK(cuModuleLoadData)
GPUMOD_LINK_CHECK(cuModuleLoadDataEx)
GPUMOD_LINK_CHECK(cuModuleLoadFatBinary)
GPUMOD_LINK_CHECK(cuModuleUnload)
GPUMOD_LINK_CHECK(cuModuleGetFunction)
GPUMOD_LINK_CHECK(cuModuleGetGlobal)

// Library Management
GPUMOD_LINK_CHECK(cuLibraryLoadData)
GPUMOD_LINK_CHECK(cuLibraryLoadFromFile)
GPUMOD_LINK_CHECK(cuLibraryUnload)
GPUMOD_LINK_CHECK(cuLibraryGetKernel)
GPUMOD_LINK_CHECK(cuLibraryGetModule)
GPUMOD_LINK_CHECK(cuLibraryGetGlobal)
GPUMOD_LINK_CHECK(cuKernelGetFunction)
GPUMOD_LINK_CHECK(cuKernelGetName)
GPUMOD_LINK_CHECK(cuKernelGetParamInfo)
GPUMOD_LINK_CHECK(cuKernelGetAttribute)
GPUMOD_LINK_CHECK(cuKernelSetAttribute)
GPUMOD_LINK_CHECK(cuKernelSetCacheConfig)
GPUMOD_LINK_CHECK(cuFuncGetName)
GPUMOD_LINK_CHECK(cuFuncGetParamInfo)
GPUMOD_LINK_CHECK(cuFuncIsLoaded)
GPUMOD_LINK_CHECK(cuFuncLoad)

// Memory Management
GPUMOD_LINK_CHECK(cuMemGetInfo_v2)
GPUMOD_LINK_CHECK(cuMemAlloc_v2)
GPUMOD_LINK_CHECK(cuMemFree_v2)
GPUMOD_LINK_CHECK(cuMemAllocHost_v2)
GPUMOD_LINK_CHECK(cuMemFreeHost)
GPUMOD_LINK_CHECK(cuMemHostRegister_v2)
GPUMOD_LINK_CHECK(cuMemHostUnregister)
GPUMOD_LINK_CHECK(cuMemHostGetDevicePointer_v2)
GPUMOD_LINK_CHECK(cuMemHostGetFlags)
GPUMOD_LINK_CHECK(cuMemAllocManaged)
GPUMOD_LINK_CHECK(cuMemAllocPitch_v2)
GPUMOD_LINK_CHECK(cuMemGetAddressRange_v2)
GPUMOD_LINK_CHECK(cuMemsetD8_v2)
GPUMOD_LINK_CHECK(cuMemsetD16_v2)
GPUMOD_LINK_CHECK(cuMemsetD32_v2)
GPUMOD_LINK_CHECK(cuMemsetD8Async)
GPUMOD_LINK_CHECK(cuMemsetD16Async)
GPUMOD_LINK_CHECK(cuMemsetD32Async)
GPUMOD_LINK_CHECK(cuMemcpy)
GPUMOD_LINK_CHECK(cuMemcpyAsync)
GPUMOD_LINK_CHECK(cuMemcpyPeer)
GPUMOD_LINK_CHECK(cuMemcpyPeerAsync)
GPUMOD_LINK_CHECK(cuMemcpyHtoD_v2)
GPUMOD_LINK_CHECK(cuMemcpyDtoH_v2)
GPUMOD_LINK_CHECK(cuMemcpyDtoD_v2)
GPUMOD_LINK_CHECK(cuMemcpyHtoDAsync_v2)
GPUMOD_LINK_CHECK(cuMemcpyDtoHAsync_v2)
GPUMOD_LINK_CHECK(cuMemcpyDtoDAsync_v2)
GPUMOD_LINK_CHECK(cuMemcpy2D_v2)
GPUMOD_LINK_CHECK(cuMemcpy2DAsync_v2)
GPUMOD_LINK_CHECK(cuMemcpy2DUnaligned_v2)
GPUMOD_LINK_CHECK(cuMemcpy3D_v2)
GPUMOD_LINK_CHECK(cuMemcpy3DAsync_v2)
GPUMOD_LINK_CHECK(cuMemcpy3DPeer)
GPUMOD_LINK_CHECK(cuMemcpy3DPeerAsync)
GPUMOD_LINK_CHECK(cuMemAdvise_v2)
GPUMOD_LINK_CHECK(cuMemPrefetchAsync_v2)
GPUMOD_LINK_CHECK(cuMemRangeGetAttribute)
GPUMOD_LINK_CHECK(cuMemRangeGetAttributes)
GPUMOD_LINK_CHECK(cuPointerGetAttribute)
GPUMOD_LINK_CHECK(cuPointerGetAttributes)

// Stream Management
GPUMOD_LINK_CHECK(cuStreamCreate)
GPUMOD_LINK_CHECK(cuStreamCreateWithPriority)
GPUMOD_LINK_CHECK(cuStreamDestroy_v2)
GPUMOD_LINK_CHECK(cuStreamSynchronize)
GPUMOD_LINK_CHECK(cuStreamWaitEvent)
GPUMOD_LINK_CHECK(cuStreamQuery)
GPUMOD_LINK_CHECK(cuStreamGetFlags)
GPUMOD_LINK_CHECK(cuStreamGetPriority)
GPUMOD_LINK_CHECK(cuStreamGetId)
GPUMOD_LINK_CHECK(cuStreamGetCtx)
GPUMOD_LINK_CHECK(cuStreamAddCallback)
GPUMOD_LINK_CHECK(cuStreamAttachMemAsync)
GPUMOD_LINK_CHECK(cuStreamBeginCapture_v2)
GPUMOD_LINK_CHECK(cuStreamEndCapture)
GPUMOD_LINK_CHECK(cuStreamIsCapturing)
GPUMOD_LINK_CHECK(cuStreamGetCaptureInfo_v3)
GPUMOD_LINK_CHECK(cuStreamUpdateCaptureDependencies_v2)
GPUMOD_LINK_CHECK(cuThreadExchangeStreamCaptureMode)
GPUMOD_LINK_CHECK(cuStreamCopyAttributes)
GPUMOD_LINK_CHECK(cuStreamGetAttribute)
GPUMOD_LINK_CHECK(cuStreamSetAttribute)
GPUMOD_LINK_CHECK(cuStreamWaitValue32_v2)
GPUMOD_LINK_CHECK(cuStreamWaitValue64_v2)
GPUMOD_LINK_CHECK(cuStreamWriteValue32)
GPUMOD_LINK_CHECK(cuStreamWriteValue64)
GPUMOD_LINK_CHECK(cuStreamBatchMemOp)

// Event Management
GPUMOD_LINK_CHECK(cuEventCreate)
GPUMOD_LINK_CHECK(cuEventDestroy)
GPUMOD_LINK_CHECK(cuEventRecord)
GPUMOD_LINK_CHECK(cuEventRecordWithFlags)
GPUMOD_LINK_CHECK(cuEventSynchronize)
GPUMOD_LINK_CHECK(cuEventQuery)
GPUMOD_LINK_CHECK(cuEventElapsedTime)

// External Memory
GPUMOD_LINK_CHECK(cuImportExternalMemory)
GPUMOD_LINK_CHECK(cuExternalMemoryGetMappedBuffer)
GPUMOD_LINK_CHECK(cuExternalMemoryGetMappedMipmappedArray)
GPUMOD_LINK_CHECK(cuDestroyExternalMemory)

// External Semaphores
GPUMOD_LINK_CHECK(cuImportExternalSemaphore)
GPUMOD_LINK_CHECK(cuSignalExternalSemaphoresAsync)
GPUMOD_LINK_CHECK(cuWaitExternalSemaphoresAsync)
GPUMOD_LINK_CHECK(cuDestroyExternalSemaphore)

// IPC
GPUMOD_LINK_CHECK(cuIpcGetMemHandle)
GPUMOD_LINK_CHECK(cuIpcOpenMemHandle)
GPUMOD_LINK_CHECK(cuIpcCloseMemHandle)
GPUMOD_LINK_CHECK(cuIpcGetEventHandle)
GPUMOD_LINK_CHECK(cuIpcOpenEventHandle)

// Kernel Execution
GPUMOD_LINK_CHECK(cuLaunchKernel)
GPUMOD_LINK_CHECK(cuLaunchKernelEx)
GPUMOD_LINK_CHECK(cuLaunchCooperativeKernel)
GPUMOD_LINK_CHECK(cuLaunchHostFunc)
GPUMOD_LINK_CHECK(cuFuncSetAttribute)
GPUMOD_LINK_CHECK(cuFuncSetCacheConfig)
GPUMOD_LINK_CHECK(cuFuncGetAttribute)
GPUMOD_LINK_CHECK(cuFuncSetSharedMemConfig)

// Occupancy
GPUMOD_LINK_CHECK(cuOccupancyMaxActiveBlocksPerMultiprocessor)
GPUMOD_LINK_CHECK(cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags)
GPUMOD_LINK_CHECK(cuOccupancyAvailableDynamicSMemPerBlock)
GPUMOD_LINK_CHECK(cuOccupancyMaxPotentialClusterSize)
GPUMOD_LINK_CHECK(cuOccupancyMaxActiveClusters)

// Texture Reference API (legacy)
GPUMOD_LINK_CHECK(cuTexRefSetArray)
GPUMOD_LINK_CHECK(cuTexRefSetMipmappedArray)
GPUMOD_LINK_CHECK(cuTexRefSetAddress)
GPUMOD_LINK_CHECK(cuTexRefSetAddress2D)
GPUMOD_LINK_CHECK(cuTexRefSetFormat)
GPUMOD_LINK_CHECK(cuTexRefSetAddressMode)
GPUMOD_LINK_CHECK(cuTexRefSetFilterMode)
GPUMOD_LINK_CHECK(cuTexRefSetMipmapFilterMode)
GPUMOD_LINK_CHECK(cuTexRefSetMipmapLevelBias)
GPUMOD_LINK_CHECK(cuTexRefSetMipmapLevelClamp)
GPUMOD_LINK_CHECK(cuTexRefSetMaxAnisotropy)
GPUMOD_LINK_CHECK(cuTexRefSetBorderColor)
GPUMOD_LINK_CHECK(cuTexRefSetFlags)
GPUMOD_LINK_CHECK(cuTexRefGetAddress)
GPUMOD_LINK_CHECK(cuTexRefGetArray)
GPUMOD_LINK_CHECK(cuTexRefGetMipmappedArray)
GPUMOD_LINK_CHECK(cuTexRefGetAddressMode)
GPUMOD_LINK_CHECK(cuTexRefGetFilterMode)
GPUMOD_LINK_CHECK(cuTexRefGetFormat)
GPUMOD_LINK_CHECK(cuTexRefGetMipmapFilterMode)
GPUMOD_LINK_CHECK(cuTexRefGetMipmapLevelBias)
GPUMOD_LINK_CHECK(cuTexRefGetMipmapLevelClamp)
GPUMOD_LINK_CHECK(cuTexRefGetMaxAnisotropy)
GPUMOD_LINK_CHECK(cuTexRefGetBorderColor)
GPUMOD_LINK_CHECK(cuTexRefGetFlags)

// Texture Object API
GPUMOD_LINK_CHECK(cuTexObjectCreate)
GPUMOD_LINK_CHECK(cuTexObjectDestroy)
GPUMOD_LINK_CHECK(cuTexObjectGetResourceDesc)
GPUMOD_LINK_CHECK(cuTexObjectGetTextureDesc)
GPUMOD_LINK_CHECK(cuTexObjectGetResourceViewDesc)

// Surface Reference API (legacy)
GPUMOD_LINK_CHECK(cuSurfRefSetArray)
GPUMOD_LINK_CHECK(cuSurfRefGetArray)

// Surface Object API
GPUMOD_LINK_CHECK(cuSurfObjectCreate)
GPUMOD_LINK_CHECK(cuSurfObjectDestroy)
GPUMOD_LINK_CHECK(cuSurfObjectGetResourceDesc)

// Virtual Memory Management
GPUMOD_LINK_CHECK(cuMemAddressReserve)
GPUMOD_LINK_CHECK(cuMemAddressFree)
GPUMOD_LINK_CHECK(cuMemCreate)
GPUMOD_LINK_CHECK(cuMemRelease)
GPUMOD_LINK_CHECK(cuMemMap)
GPUMOD_LINK_CHECK(cuMemUnmap)
GPUMOD_LINK_CHECK(cuMemSetAccess)
GPUMOD_LINK_CHECK(cuMemGetAccess)
GPUMOD_LINK_CHECK(cuMemGetAllocationGranularity)
GPUMOD_LINK_CHECK(cuMemGetAllocationPropertiesFromHandle)
GPUMOD_LINK_CHECK(cuMemRetainAllocationHandle)
GPUMOD_LINK_CHECK(cuMemExportToShareableHandle)
GPUMOD_LINK_CHECK(cuMemImportFromShareableHandle)

// Memory Pool Management
GPUMOD_LINK_CHECK(cuMemPoolCreate)
GPUMOD_LINK_CHECK(cuMemPoolDestroy)
GPUMOD_LINK_CHECK(cuMemAllocFromPoolAsync)
GPUMOD_LINK_CHECK(cuMemFreeAsync)
GPUMOD_LINK_CHECK(cuMemPoolTrimTo)
GPUMOD_LINK_CHECK(cuMemPoolSetAttribute)
GPUMOD_LINK_CHECK(cuMemPoolGetAttribute)
GPUMOD_LINK_CHECK(cuMemPoolSetAccess)
GPUMOD_LINK_CHECK(cuMemPoolGetAccess)
GPUMOD_LINK_CHECK(cuMemPoolExportToShareableHandle)
GPUMOD_LINK_CHECK(cuMemPoolImportFromShareableHandle)
GPUMOD_LINK_CHECK(cuMemPoolExportPointer)
GPUMOD_LINK_CHECK(cuMemPoolImportPointer)

// Array Management
GPUMOD_LINK_CHECK(cuArrayCreate_v2)
GPUMOD_LINK_CHECK(cuArrayDestroy)
GPUMOD_LINK_CHECK(cuArray3DCreate_v2)
GPUMOD_LINK_CHECK(cuArrayGetDescriptor_v2)
GPUMOD_LINK_CHECK(cuArray3DGetDescriptor_v2)
GPUMOD_LINK_CHECK(cuArrayGetSparseProperties)
GPUMOD_LINK_CHECK(cuMipmappedArrayGetSparseProperties)
GPUMOD_LINK_CHECK(cuArrayGetMemoryRequirements)
GPUMOD_LINK_CHECK(cuMipmappedArrayGetMemoryRequirements)
GPUMOD_LINK_CHECK(cuArrayGetPlane)
GPUMOD_LINK_CHECK(cuMipmappedArrayCreate)
GPUMOD_LINK_CHECK(cuMipmappedArrayDestroy)
GPUMOD_LINK_CHECK(cuMipmappedArrayGetLevel)

// Graph Management — Graph / Node Lifecycle
GPUMOD_LINK_CHECK(cuGraphCreate)
GPUMOD_LINK_CHECK(cuGraphDestroy)
GPUMOD_LINK_CHECK(cuGraphAddDependencies)
GPUMOD_LINK_CHECK(cuGraphRemoveDependencies)
GPUMOD_LINK_CHECK(cuGraphGetEdges)
GPUMOD_LINK_CHECK(cuGraphGetNodes)
GPUMOD_LINK_CHECK(cuGraphGetRootNodes)
GPUMOD_LINK_CHECK(cuGraphNodeGetDependencies)
GPUMOD_LINK_CHECK(cuGraphNodeGetDependentNodes)
GPUMOD_LINK_CHECK(cuGraphNodeGetType)
GPUMOD_LINK_CHECK(cuGraphDestroyNode)
GPUMOD_LINK_CHECK(cuGraphClone)
GPUMOD_LINK_CHECK(cuGraphNodeFindInClone)
GPUMOD_LINK_CHECK(cuGraphDebugDotPrint)
GPUMOD_LINK_CHECK(cuGraphAddNode)
GPUMOD_LINK_CHECK(cuGraphNodeSetParams)

// Graph Management — Node Addition
GPUMOD_LINK_CHECK(cuGraphAddEmptyNode)
GPUMOD_LINK_CHECK(cuGraphAddKernelNode)
GPUMOD_LINK_CHECK(cuGraphAddMemcpyNode)
GPUMOD_LINK_CHECK(cuGraphAddMemsetNode)
GPUMOD_LINK_CHECK(cuGraphAddHostNode)
GPUMOD_LINK_CHECK(cuGraphAddChildGraphNode)
GPUMOD_LINK_CHECK(cuGraphAddEventRecordNode)
GPUMOD_LINK_CHECK(cuGraphAddEventWaitNode)
GPUMOD_LINK_CHECK(cuGraphAddExternalSemaphoresSignalNode)
GPUMOD_LINK_CHECK(cuGraphAddExternalSemaphoresWaitNode)
GPUMOD_LINK_CHECK(cuGraphAddBatchMemOpNode)
GPUMOD_LINK_CHECK(cuGraphAddMemAllocNode)
GPUMOD_LINK_CHECK(cuGraphAddMemFreeNode)

// Graph Management — Node Params
GPUMOD_LINK_CHECK(cuGraphKernelNodeGetParams)
GPUMOD_LINK_CHECK(cuGraphKernelNodeSetParams)
GPUMOD_LINK_CHECK(cuGraphKernelNodeCopyAttributes)
GPUMOD_LINK_CHECK(cuGraphKernelNodeGetAttribute)
GPUMOD_LINK_CHECK(cuGraphKernelNodeSetAttribute)
GPUMOD_LINK_CHECK(cuGraphMemcpyNodeGetParams)
GPUMOD_LINK_CHECK(cuGraphMemcpyNodeSetParams)
GPUMOD_LINK_CHECK(cuGraphMemsetNodeGetParams)
GPUMOD_LINK_CHECK(cuGraphMemsetNodeSetParams)
GPUMOD_LINK_CHECK(cuGraphHostNodeGetParams)
GPUMOD_LINK_CHECK(cuGraphHostNodeSetParams)
GPUMOD_LINK_CHECK(cuGraphChildGraphNodeGetGraph)
GPUMOD_LINK_CHECK(cuGraphEventRecordNodeGetEvent)
GPUMOD_LINK_CHECK(cuGraphEventRecordNodeSetEvent)
GPUMOD_LINK_CHECK(cuGraphEventWaitNodeGetEvent)
GPUMOD_LINK_CHECK(cuGraphEventWaitNodeSetEvent)
GPUMOD_LINK_CHECK(cuGraphExternalSemaphoresSignalNodeGetParams)
GPUMOD_LINK_CHECK(cuGraphExternalSemaphoresSignalNodeSetParams)
GPUMOD_LINK_CHECK(cuGraphExternalSemaphoresWaitNodeGetParams)
GPUMOD_LINK_CHECK(cuGraphExternalSemaphoresWaitNodeSetParams)
GPUMOD_LINK_CHECK(cuGraphBatchMemOpNodeGetParams)
GPUMOD_LINK_CHECK(cuGraphBatchMemOpNodeSetParams)
GPUMOD_LINK_CHECK(cuGraphMemAllocNodeGetParams)
GPUMOD_LINK_CHECK(cuGraphMemFreeNodeGetParams)
GPUMOD_LINK_CHECK(cuGraphNodeGetEnabled)
GPUMOD_LINK_CHECK(cuGraphNodeSetEnabled)

// Graph Management — Execution
GPUMOD_LINK_CHECK(cuGraphInstantiate)
GPUMOD_LINK_CHECK(cuGraphInstantiateWithFlags)
GPUMOD_LINK_CHECK(cuGraphInstantiateWithParams)
GPUMOD_LINK_CHECK(cuGraphExecGetFlags)
GPUMOD_LINK_CHECK(cuGraphExecDestroy)
GPUMOD_LINK_CHECK(cuGraphExecUpdate)
GPUMOD_LINK_CHECK(cuGraphLaunch)
GPUMOD_LINK_CHECK(cuGraphUpload)
GPUMOD_LINK_CHECK(cuGraphExecKernelNodeSetParams)
GPUMOD_LINK_CHECK(cuGraphExecMemcpyNodeSetParams)
GPUMOD_LINK_CHECK(cuGraphExecMemsetNodeSetParams)
GPUMOD_LINK_CHECK(cuGraphExecHostNodeSetParams)
GPUMOD_LINK_CHECK(cuGraphExecChildGraphNodeSetParams)
GPUMOD_LINK_CHECK(cuGraphExecEventRecordNodeSetEvent)
GPUMOD_LINK_CHECK(cuGraphExecEventWaitNodeSetEvent)
GPUMOD_LINK_CHECK(cuGraphExecExternalSemaphoresSignalNodeSetParams)
GPUMOD_LINK_CHECK(cuGraphExecExternalSemaphoresWaitNodeSetParams)
GPUMOD_LINK_CHECK(cuGraphExecBatchMemOpNodeSetParams)
GPUMOD_LINK_CHECK(cuGraphExecNodeSetParams)
GPUMOD_LINK_CHECK(cuGraphConditionalHandleCreate)

// User Objects
GPUMOD_LINK_CHECK(cuUserObjectCreate)
GPUMOD_LINK_CHECK(cuUserObjectRetain)
GPUMOD_LINK_CHECK(cuUserObjectRelease)
GPUMOD_LINK_CHECK(cuGraphRetainUserObject)
GPUMOD_LINK_CHECK(cuGraphReleaseUserObject)
GPUMOD_LINK_CHECK(cuDeviceGraphMemTrim)
GPUMOD_LINK_CHECK(cuDeviceGetGraphMemAttribute)
GPUMOD_LINK_CHECK(cuDeviceSetGraphMemAttribute)

// Driver Entry Points
GPUMOD_LINK_CHECK(cuGetProcAddress)

// Peer Device Memory Access
GPUMOD_LINK_CHECK(cuDeviceCanAccessPeer)

// Multicast
GPUMOD_LINK_CHECK(cuMulticastCreate)
GPUMOD_LINK_CHECK(cuMulticastAddDevice)
GPUMOD_LINK_CHECK(cuMulticastBindMem)
GPUMOD_LINK_CHECK(cuMulticastBindAddr)
GPUMOD_LINK_CHECK(cuMulticastUnbind)
GPUMOD_LINK_CHECK(cuMulticastGetGranularity)

// Process State / Checkpointing
GPUMOD_LINK_CHECK(cuCheckpointProcessGetState)
GPUMOD_LINK_CHECK(cuCheckpointProcessLock)
GPUMOD_LINK_CHECK(cuCheckpointProcessCheckpoint)
GPUMOD_LINK_CHECK(cuCheckpointProcessRestore)
GPUMOD_LINK_CHECK(cuCheckpointProcessUnlock)

} // namespace gpumod::test
