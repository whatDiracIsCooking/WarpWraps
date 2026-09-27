/**
 * @file interface.cppm
 * @brief Primary interface for wwr.extension.memory_buffer
 *
 * This module provides RAII-based memory buffer management for different
 * memory kinds with automatic allocation and deallocation. It aggregates
 * all buffer types:
 * - :device_buffer - GPU device memory buffers (gpuMallocFromPoolAsync/gpuFreeAsync)
 * - :pinned_buffer - Pinned host memory buffers (gpuHostAlloc/gpuFreeHost)
 * - :unified_buffer - Unified memory buffers (gpuMallocManaged)
 * - :host_buffer - Standard host memory buffers (std::malloc/std::free)
 * - :convenience_memory_buffer - Default-policy aliases (DeviceBuffer, HostBuffer, PinnedBuffer, UnifiedBuffer, and views)
 *
 * Usage:
 *   import wwr.extension.memory_buffer;
 *   using namespace wwr::extension;
 *
 *   auto dev = std::make_shared<DeviceHandle>(0);
 *   DeviceBuffer<float> dev_buf(1024, dev);  // Device memory (from the handle's pool)
 *   PinnedBuffer<float> pin_buf(1024);       // Pinned host memory
 *   UnifiedBuffer<float> uni_buf(1024);      // Unified memory
 *   HostBuffer<float> host_buf(1024);        // Standard host memory
 */

export module wwr.extension.memory_buffer;

import std;

// MemoryKind, BaseBuffer/BufferViewWrapper and stdHostMemoryError_t are all part
// of the documented public API - MemoryKind and the policy types are needed to
// name a buffer or view with a non-default error policy, and copy() returns
// stdHostMemoryError_t - so consumers must be able to name them.
export import :memory_kind;
export import :base_buffer;
export import :host_memory;

export import :device_buffer;
export import :pinned_buffer;
export import :unified_buffer;
export import :host_buffer;
export import :copy;
export import :memset;
export import :convenience_memory_buffer;

export namespace wwr::extension {
using wwr::extension::buffer_base;
using wwr::extension::buffer_typename;
using wwr::extension::same_value_type;
} // namespace wwr::extension
