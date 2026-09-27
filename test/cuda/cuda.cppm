// cuda.cppm - Compile-time tests for wwr.cuda.cuda_h

module;

#include "test/shared/link_check.h"
#include <cuda.h>

export module wwr.test.cuda.cuda_h;

import std;
import wwr.cuda.cuda_h;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time tests for wwr.cuda.cuda_h
//
// The module is pure re-export (using declarations).
// Runtime tests for the underlying CUDA Driver API would just test CUDA itself.
// We verify at compile-time that:
//   1. Key enum values with CUDA-specified values are correct
//   2. Opaque handle types have the expected type traits
//   3. Struct types satisfy trivial copyability and standard layout (C-interop guarantee)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace wwr::test {

using namespace wwr;

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
WWR_LINK_CHECK(cuInit)

// Version
WWR_LINK_CHECK(cuDriverGetVersion)

// Device Management
WWR_LINK_CHECK(cuDeviceGet)
WWR_LINK_CHECK(cuDeviceGetCount)
WWR_LINK_CHECK(cuDeviceGetName)
WWR_LINK_CHECK(cuDeviceGetUuid)
WWR_LINK_CHECK(cuDeviceTotalMem)
WWR_LINK_CHECK(cuDeviceGetAttribute)
WWR_LINK_CHECK(cuDeviceGetP2PAttribute)
WWR_LINK_CHECK(cuDeviceGetDefaultMemPool)
WWR_LINK_CHECK(cuDeviceSetMemPool)
WWR_LINK_CHECK(cuDeviceGetMemPool)
WWR_LINK_CHECK(cuDeviceGetTexture1DLinearMaxWidth)
WWR_LINK_CHECK(cuDeviceGetNvSciSyncAttributes)
WWR_LINK_CHECK(cuDeviceRegisterAsyncNotification)
WWR_LINK_CHECK(cuDeviceUnregisterAsyncNotification)

// Context Management
WWR_LINK_CHECK(cuCtxCreate)
WWR_LINK_CHECK(cuCtxDestroy)
WWR_LINK_CHECK(cuCtxGetCurrent)
WWR_LINK_CHECK(cuCtxSetCurrent)
WWR_LINK_CHECK(cuCtxPushCurrent)
WWR_LINK_CHECK(cuCtxPopCurrent)
WWR_LINK_CHECK(cuCtxGetDevice)
WWR_LINK_CHECK(cuCtxGetFlags)
WWR_LINK_CHECK(cuCtxSetFlags)
WWR_LINK_CHECK(cuCtxGetId)
WWR_LINK_CHECK(cuCtxSynchronize)
WWR_LINK_CHECK(cuCtxGetLimit)
WWR_LINK_CHECK(cuCtxSetLimit)
WWR_LINK_CHECK(cuCtxGetCacheConfig)
WWR_LINK_CHECK(cuCtxSetCacheConfig)
WWR_LINK_CHECK(cuCtxGetStreamPriorityRange)
WWR_LINK_CHECK(cuCtxGetSharedMemConfig)
WWR_LINK_CHECK(cuCtxSetSharedMemConfig)
WWR_LINK_CHECK(cuCtxEnablePeerAccess)
WWR_LINK_CHECK(cuCtxDisablePeerAccess)
WWR_LINK_CHECK(cuCtxResetPersistingL2Cache)
WWR_LINK_CHECK(cuCtxGetExecAffinity)

// Module Management
WWR_LINK_CHECK(cuModuleLoad)
WWR_LINK_CHECK(cuModuleLoadData)
WWR_LINK_CHECK(cuModuleLoadDataEx)
WWR_LINK_CHECK(cuModuleLoadFatBinary)
WWR_LINK_CHECK(cuModuleUnload)
WWR_LINK_CHECK(cuModuleGetFunction)
WWR_LINK_CHECK(cuModuleGetGlobal)

// Library Management
WWR_LINK_CHECK(cuLibraryLoadData)
WWR_LINK_CHECK(cuLibraryLoadFromFile)
WWR_LINK_CHECK(cuLibraryUnload)
WWR_LINK_CHECK(cuLibraryGetKernel)
WWR_LINK_CHECK(cuLibraryGetModule)
WWR_LINK_CHECK(cuLibraryGetGlobal)
WWR_LINK_CHECK(cuKernelGetFunction)
WWR_LINK_CHECK(cuKernelGetName)
WWR_LINK_CHECK(cuKernelGetParamInfo)
WWR_LINK_CHECK(cuKernelGetAttribute)
WWR_LINK_CHECK(cuKernelSetAttribute)
WWR_LINK_CHECK(cuKernelSetCacheConfig)
WWR_LINK_CHECK(cuFuncGetName)
WWR_LINK_CHECK(cuFuncGetParamInfo)
WWR_LINK_CHECK(cuFuncIsLoaded)
WWR_LINK_CHECK(cuFuncLoad)

// Memory Management
WWR_LINK_CHECK(cuMemGetInfo_v2)
WWR_LINK_CHECK(cuMemAlloc_v2)
WWR_LINK_CHECK(cuMemFree_v2)
WWR_LINK_CHECK(cuMemAllocHost_v2)
WWR_LINK_CHECK(cuMemFreeHost)
WWR_LINK_CHECK(cuMemHostRegister_v2)
WWR_LINK_CHECK(cuMemHostUnregister)
WWR_LINK_CHECK(cuMemHostGetDevicePointer_v2)
WWR_LINK_CHECK(cuMemHostGetFlags)
WWR_LINK_CHECK(cuMemAllocManaged)
WWR_LINK_CHECK(cuMemAllocPitch_v2)
WWR_LINK_CHECK(cuMemGetAddressRange_v2)
WWR_LINK_CHECK(cuMemsetD8_v2)
WWR_LINK_CHECK(cuMemsetD16_v2)
WWR_LINK_CHECK(cuMemsetD32_v2)
WWR_LINK_CHECK(cuMemsetD8Async)
WWR_LINK_CHECK(cuMemsetD16Async)
WWR_LINK_CHECK(cuMemsetD32Async)
WWR_LINK_CHECK(cuMemcpy)
WWR_LINK_CHECK(cuMemcpyAsync)
WWR_LINK_CHECK(cuMemcpyPeer)
WWR_LINK_CHECK(cuMemcpyPeerAsync)
WWR_LINK_CHECK(cuMemcpyHtoD_v2)
WWR_LINK_CHECK(cuMemcpyDtoH_v2)
WWR_LINK_CHECK(cuMemcpyDtoD_v2)
WWR_LINK_CHECK(cuMemcpyHtoDAsync_v2)
WWR_LINK_CHECK(cuMemcpyDtoHAsync_v2)
WWR_LINK_CHECK(cuMemcpyDtoDAsync_v2)
WWR_LINK_CHECK(cuMemcpy2D_v2)
WWR_LINK_CHECK(cuMemcpy2DAsync_v2)
WWR_LINK_CHECK(cuMemcpy2DUnaligned_v2)
WWR_LINK_CHECK(cuMemcpy3D_v2)
WWR_LINK_CHECK(cuMemcpy3DAsync_v2)
WWR_LINK_CHECK(cuMemcpy3DPeer)
WWR_LINK_CHECK(cuMemcpy3DPeerAsync)
WWR_LINK_CHECK(cuMemAdvise_v2)
WWR_LINK_CHECK(cuMemPrefetchAsync_v2)
WWR_LINK_CHECK(cuMemRangeGetAttribute)
WWR_LINK_CHECK(cuMemRangeGetAttributes)
WWR_LINK_CHECK(cuPointerGetAttribute)
WWR_LINK_CHECK(cuPointerGetAttributes)

// Stream Management
WWR_LINK_CHECK(cuStreamCreate)
WWR_LINK_CHECK(cuStreamCreateWithPriority)
WWR_LINK_CHECK(cuStreamDestroy_v2)
WWR_LINK_CHECK(cuStreamSynchronize)
WWR_LINK_CHECK(cuStreamWaitEvent)
WWR_LINK_CHECK(cuStreamQuery)
WWR_LINK_CHECK(cuStreamGetFlags)
WWR_LINK_CHECK(cuStreamGetPriority)
WWR_LINK_CHECK(cuStreamGetId)
WWR_LINK_CHECK(cuStreamGetCtx)
WWR_LINK_CHECK(cuStreamAddCallback)
WWR_LINK_CHECK(cuStreamAttachMemAsync)
WWR_LINK_CHECK(cuStreamBeginCapture_v2)
WWR_LINK_CHECK(cuStreamEndCapture)
WWR_LINK_CHECK(cuStreamIsCapturing)
WWR_LINK_CHECK(cuStreamGetCaptureInfo_v3)
WWR_LINK_CHECK(cuStreamUpdateCaptureDependencies_v2)
WWR_LINK_CHECK(cuThreadExchangeStreamCaptureMode)
WWR_LINK_CHECK(cuStreamCopyAttributes)
WWR_LINK_CHECK(cuStreamGetAttribute)
WWR_LINK_CHECK(cuStreamSetAttribute)
WWR_LINK_CHECK(cuStreamWaitValue32_v2)
WWR_LINK_CHECK(cuStreamWaitValue64_v2)
WWR_LINK_CHECK(cuStreamWriteValue32)
WWR_LINK_CHECK(cuStreamWriteValue64)
WWR_LINK_CHECK(cuStreamBatchMemOp)

// Event Management
WWR_LINK_CHECK(cuEventCreate)
WWR_LINK_CHECK(cuEventDestroy)
WWR_LINK_CHECK(cuEventRecord)
WWR_LINK_CHECK(cuEventRecordWithFlags)
WWR_LINK_CHECK(cuEventSynchronize)
WWR_LINK_CHECK(cuEventQuery)
WWR_LINK_CHECK(cuEventElapsedTime)

// External Memory
WWR_LINK_CHECK(cuImportExternalMemory)
WWR_LINK_CHECK(cuExternalMemoryGetMappedBuffer)
WWR_LINK_CHECK(cuExternalMemoryGetMappedMipmappedArray)
WWR_LINK_CHECK(cuDestroyExternalMemory)

// External Semaphores
WWR_LINK_CHECK(cuImportExternalSemaphore)
WWR_LINK_CHECK(cuSignalExternalSemaphoresAsync)
WWR_LINK_CHECK(cuWaitExternalSemaphoresAsync)
WWR_LINK_CHECK(cuDestroyExternalSemaphore)

// IPC
WWR_LINK_CHECK(cuIpcGetMemHandle)
WWR_LINK_CHECK(cuIpcOpenMemHandle)
WWR_LINK_CHECK(cuIpcCloseMemHandle)
WWR_LINK_CHECK(cuIpcGetEventHandle)
WWR_LINK_CHECK(cuIpcOpenEventHandle)

// Kernel Execution
WWR_LINK_CHECK(cuLaunchKernel)
WWR_LINK_CHECK(cuLaunchKernelEx)
WWR_LINK_CHECK(cuLaunchCooperativeKernel)
WWR_LINK_CHECK(cuLaunchHostFunc)
WWR_LINK_CHECK(cuFuncSetAttribute)
WWR_LINK_CHECK(cuFuncSetCacheConfig)
WWR_LINK_CHECK(cuFuncGetAttribute)
WWR_LINK_CHECK(cuFuncSetSharedMemConfig)

// Occupancy
WWR_LINK_CHECK(cuOccupancyMaxActiveBlocksPerMultiprocessor)
WWR_LINK_CHECK(cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags)
WWR_LINK_CHECK(cuOccupancyAvailableDynamicSMemPerBlock)
WWR_LINK_CHECK(cuOccupancyMaxPotentialClusterSize)
WWR_LINK_CHECK(cuOccupancyMaxActiveClusters)

// Texture Reference API (legacy)
WWR_LINK_CHECK(cuTexRefSetArray)
WWR_LINK_CHECK(cuTexRefSetMipmappedArray)
WWR_LINK_CHECK(cuTexRefSetAddress)
WWR_LINK_CHECK(cuTexRefSetAddress2D)
WWR_LINK_CHECK(cuTexRefSetFormat)
WWR_LINK_CHECK(cuTexRefSetAddressMode)
WWR_LINK_CHECK(cuTexRefSetFilterMode)
WWR_LINK_CHECK(cuTexRefSetMipmapFilterMode)
WWR_LINK_CHECK(cuTexRefSetMipmapLevelBias)
WWR_LINK_CHECK(cuTexRefSetMipmapLevelClamp)
WWR_LINK_CHECK(cuTexRefSetMaxAnisotropy)
WWR_LINK_CHECK(cuTexRefSetBorderColor)
WWR_LINK_CHECK(cuTexRefSetFlags)
WWR_LINK_CHECK(cuTexRefGetAddress)
WWR_LINK_CHECK(cuTexRefGetArray)
WWR_LINK_CHECK(cuTexRefGetMipmappedArray)
WWR_LINK_CHECK(cuTexRefGetAddressMode)
WWR_LINK_CHECK(cuTexRefGetFilterMode)
WWR_LINK_CHECK(cuTexRefGetFormat)
WWR_LINK_CHECK(cuTexRefGetMipmapFilterMode)
WWR_LINK_CHECK(cuTexRefGetMipmapLevelBias)
WWR_LINK_CHECK(cuTexRefGetMipmapLevelClamp)
WWR_LINK_CHECK(cuTexRefGetMaxAnisotropy)
WWR_LINK_CHECK(cuTexRefGetBorderColor)
WWR_LINK_CHECK(cuTexRefGetFlags)

// Texture Object API
WWR_LINK_CHECK(cuTexObjectCreate)
WWR_LINK_CHECK(cuTexObjectDestroy)
WWR_LINK_CHECK(cuTexObjectGetResourceDesc)
WWR_LINK_CHECK(cuTexObjectGetTextureDesc)
WWR_LINK_CHECK(cuTexObjectGetResourceViewDesc)

// Surface Reference API (legacy)
WWR_LINK_CHECK(cuSurfRefSetArray)
WWR_LINK_CHECK(cuSurfRefGetArray)

// Surface Object API
WWR_LINK_CHECK(cuSurfObjectCreate)
WWR_LINK_CHECK(cuSurfObjectDestroy)
WWR_LINK_CHECK(cuSurfObjectGetResourceDesc)

// Virtual Memory Management
WWR_LINK_CHECK(cuMemAddressReserve)
WWR_LINK_CHECK(cuMemAddressFree)
WWR_LINK_CHECK(cuMemCreate)
WWR_LINK_CHECK(cuMemRelease)
WWR_LINK_CHECK(cuMemMap)
WWR_LINK_CHECK(cuMemUnmap)
WWR_LINK_CHECK(cuMemSetAccess)
WWR_LINK_CHECK(cuMemGetAccess)
WWR_LINK_CHECK(cuMemGetAllocationGranularity)
WWR_LINK_CHECK(cuMemGetAllocationPropertiesFromHandle)
WWR_LINK_CHECK(cuMemRetainAllocationHandle)
WWR_LINK_CHECK(cuMemExportToShareableHandle)
WWR_LINK_CHECK(cuMemImportFromShareableHandle)

// Memory Pool Management
WWR_LINK_CHECK(cuMemPoolCreate)
WWR_LINK_CHECK(cuMemPoolDestroy)
WWR_LINK_CHECK(cuMemAllocFromPoolAsync)
WWR_LINK_CHECK(cuMemFreeAsync)
WWR_LINK_CHECK(cuMemPoolTrimTo)
WWR_LINK_CHECK(cuMemPoolSetAttribute)
WWR_LINK_CHECK(cuMemPoolGetAttribute)
WWR_LINK_CHECK(cuMemPoolSetAccess)
WWR_LINK_CHECK(cuMemPoolGetAccess)
WWR_LINK_CHECK(cuMemPoolExportToShareableHandle)
WWR_LINK_CHECK(cuMemPoolImportFromShareableHandle)
WWR_LINK_CHECK(cuMemPoolExportPointer)
WWR_LINK_CHECK(cuMemPoolImportPointer)

// Array Management
WWR_LINK_CHECK(cuArrayCreate_v2)
WWR_LINK_CHECK(cuArrayDestroy)
WWR_LINK_CHECK(cuArray3DCreate_v2)
WWR_LINK_CHECK(cuArrayGetDescriptor_v2)
WWR_LINK_CHECK(cuArray3DGetDescriptor_v2)
WWR_LINK_CHECK(cuArrayGetSparseProperties)
WWR_LINK_CHECK(cuMipmappedArrayGetSparseProperties)
WWR_LINK_CHECK(cuArrayGetMemoryRequirements)
WWR_LINK_CHECK(cuMipmappedArrayGetMemoryRequirements)
WWR_LINK_CHECK(cuArrayGetPlane)
WWR_LINK_CHECK(cuMipmappedArrayCreate)
WWR_LINK_CHECK(cuMipmappedArrayDestroy)
WWR_LINK_CHECK(cuMipmappedArrayGetLevel)

// Graph Management — Graph / Node Lifecycle
WWR_LINK_CHECK(cuGraphCreate)
WWR_LINK_CHECK(cuGraphDestroy)
WWR_LINK_CHECK(cuGraphAddDependencies)
WWR_LINK_CHECK(cuGraphRemoveDependencies)
WWR_LINK_CHECK(cuGraphGetEdges)
WWR_LINK_CHECK(cuGraphGetNodes)
WWR_LINK_CHECK(cuGraphGetRootNodes)
WWR_LINK_CHECK(cuGraphNodeGetDependencies)
WWR_LINK_CHECK(cuGraphNodeGetDependentNodes)
WWR_LINK_CHECK(cuGraphNodeGetType)
WWR_LINK_CHECK(cuGraphDestroyNode)
WWR_LINK_CHECK(cuGraphClone)
WWR_LINK_CHECK(cuGraphNodeFindInClone)
WWR_LINK_CHECK(cuGraphDebugDotPrint)
WWR_LINK_CHECK(cuGraphAddNode)
WWR_LINK_CHECK(cuGraphNodeSetParams)

// Graph Management — Node Addition
WWR_LINK_CHECK(cuGraphAddEmptyNode)
WWR_LINK_CHECK(cuGraphAddKernelNode)
WWR_LINK_CHECK(cuGraphAddMemcpyNode)
WWR_LINK_CHECK(cuGraphAddMemsetNode)
WWR_LINK_CHECK(cuGraphAddHostNode)
WWR_LINK_CHECK(cuGraphAddChildGraphNode)
WWR_LINK_CHECK(cuGraphAddEventRecordNode)
WWR_LINK_CHECK(cuGraphAddEventWaitNode)
WWR_LINK_CHECK(cuGraphAddExternalSemaphoresSignalNode)
WWR_LINK_CHECK(cuGraphAddExternalSemaphoresWaitNode)
WWR_LINK_CHECK(cuGraphAddBatchMemOpNode)
WWR_LINK_CHECK(cuGraphAddMemAllocNode)
WWR_LINK_CHECK(cuGraphAddMemFreeNode)

// Graph Management — Node Params
WWR_LINK_CHECK(cuGraphKernelNodeGetParams)
WWR_LINK_CHECK(cuGraphKernelNodeSetParams)
WWR_LINK_CHECK(cuGraphKernelNodeCopyAttributes)
WWR_LINK_CHECK(cuGraphKernelNodeGetAttribute)
WWR_LINK_CHECK(cuGraphKernelNodeSetAttribute)
WWR_LINK_CHECK(cuGraphMemcpyNodeGetParams)
WWR_LINK_CHECK(cuGraphMemcpyNodeSetParams)
WWR_LINK_CHECK(cuGraphMemsetNodeGetParams)
WWR_LINK_CHECK(cuGraphMemsetNodeSetParams)
WWR_LINK_CHECK(cuGraphHostNodeGetParams)
WWR_LINK_CHECK(cuGraphHostNodeSetParams)
WWR_LINK_CHECK(cuGraphChildGraphNodeGetGraph)
WWR_LINK_CHECK(cuGraphEventRecordNodeGetEvent)
WWR_LINK_CHECK(cuGraphEventRecordNodeSetEvent)
WWR_LINK_CHECK(cuGraphEventWaitNodeGetEvent)
WWR_LINK_CHECK(cuGraphEventWaitNodeSetEvent)
WWR_LINK_CHECK(cuGraphExternalSemaphoresSignalNodeGetParams)
WWR_LINK_CHECK(cuGraphExternalSemaphoresSignalNodeSetParams)
WWR_LINK_CHECK(cuGraphExternalSemaphoresWaitNodeGetParams)
WWR_LINK_CHECK(cuGraphExternalSemaphoresWaitNodeSetParams)
WWR_LINK_CHECK(cuGraphBatchMemOpNodeGetParams)
WWR_LINK_CHECK(cuGraphBatchMemOpNodeSetParams)
WWR_LINK_CHECK(cuGraphMemAllocNodeGetParams)
WWR_LINK_CHECK(cuGraphMemFreeNodeGetParams)
WWR_LINK_CHECK(cuGraphNodeGetEnabled)
WWR_LINK_CHECK(cuGraphNodeSetEnabled)

// Graph Management — Execution
WWR_LINK_CHECK(cuGraphInstantiate)
WWR_LINK_CHECK(cuGraphInstantiateWithFlags)
WWR_LINK_CHECK(cuGraphInstantiateWithParams)
WWR_LINK_CHECK(cuGraphExecGetFlags)
WWR_LINK_CHECK(cuGraphExecDestroy)
WWR_LINK_CHECK(cuGraphExecUpdate)
WWR_LINK_CHECK(cuGraphLaunch)
WWR_LINK_CHECK(cuGraphUpload)
WWR_LINK_CHECK(cuGraphExecKernelNodeSetParams)
WWR_LINK_CHECK(cuGraphExecMemcpyNodeSetParams)
WWR_LINK_CHECK(cuGraphExecMemsetNodeSetParams)
WWR_LINK_CHECK(cuGraphExecHostNodeSetParams)
WWR_LINK_CHECK(cuGraphExecChildGraphNodeSetParams)
WWR_LINK_CHECK(cuGraphExecEventRecordNodeSetEvent)
WWR_LINK_CHECK(cuGraphExecEventWaitNodeSetEvent)
WWR_LINK_CHECK(cuGraphExecExternalSemaphoresSignalNodeSetParams)
WWR_LINK_CHECK(cuGraphExecExternalSemaphoresWaitNodeSetParams)
WWR_LINK_CHECK(cuGraphExecBatchMemOpNodeSetParams)
WWR_LINK_CHECK(cuGraphExecNodeSetParams)
WWR_LINK_CHECK(cuGraphConditionalHandleCreate)

// User Objects
WWR_LINK_CHECK(cuUserObjectCreate)
WWR_LINK_CHECK(cuUserObjectRetain)
WWR_LINK_CHECK(cuUserObjectRelease)
WWR_LINK_CHECK(cuGraphRetainUserObject)
WWR_LINK_CHECK(cuGraphReleaseUserObject)
WWR_LINK_CHECK(cuDeviceGraphMemTrim)
WWR_LINK_CHECK(cuDeviceGetGraphMemAttribute)
WWR_LINK_CHECK(cuDeviceSetGraphMemAttribute)

// Driver Entry Points
WWR_LINK_CHECK(cuGetProcAddress)

// Peer Device Memory Access
WWR_LINK_CHECK(cuDeviceCanAccessPeer)

// Multicast
WWR_LINK_CHECK(cuMulticastCreate)
WWR_LINK_CHECK(cuMulticastAddDevice)
WWR_LINK_CHECK(cuMulticastBindMem)
WWR_LINK_CHECK(cuMulticastBindAddr)
WWR_LINK_CHECK(cuMulticastUnbind)
WWR_LINK_CHECK(cuMulticastGetGranularity)

// Process State / Checkpointing
WWR_LINK_CHECK(cuCheckpointProcessGetState)
WWR_LINK_CHECK(cuCheckpointProcessLock)
WWR_LINK_CHECK(cuCheckpointProcessCheckpoint)
WWR_LINK_CHECK(cuCheckpointProcessRestore)
WWR_LINK_CHECK(cuCheckpointProcessUnlock)

} // namespace wwr::test
