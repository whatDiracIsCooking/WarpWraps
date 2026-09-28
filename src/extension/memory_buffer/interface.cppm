/**
 * @file interface.cppm
 * @brief Primary interface for wwr.extension.memory_buffer
 *
 * This module provides RAII-based memory buffer management for different
 * memory kinds with automatic allocation and deallocation. It aggregates
 * all buffer types:
 * - :device_buffer - GPU device memory buffers (alloc call set by the handle's tier)
 * - :pinned_buffer - Pinned host memory buffers (wwrHostAlloc/wwrFreeHost)
 * - :unified_buffer - Unified memory buffers (wwrMallocManaged)
 * - :host_buffer - Standard host memory buffers (std::malloc/std::free)
 *
 * Usage:
 *   import wwr.extension.memory_buffer;
 *   using namespace wwr::extension;
 *
 *   auto dev = std::make_shared<DeviceHandle>(0);
 *   using Abort = AbortPolicy<wwrError_t>;
 *   DeviceBufferWrapper<float, Abort, Abort> dev_buf(1024, dev);  // Device memory (DeviceHandle -> pool)
 *   PinnedBufferWrapper<float, Abort, Abort> pin_buf(1024);       // Pinned host memory
 *   UnifiedBufferWrapper<float, Abort, Abort> uni_buf(1024);      // Unified memory
 *   HostBufferWrapper<float, AbortPolicy<stdHostMemoryError_t>,
 *                     AbortPolicy<stdHostMemoryError_t>> host_buf(1024);  // Standard host memory
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

// :reinterpret_tag is intentionally NOT re-exported. base_buffer imports it
// with a plain `import` so the reinterpreting view constructor and
// reinterpret_buffer_view() can name the tag, but keeping it off the public
// interface means consumers reach the reinterpreting view only through the
// factory - naming reinterpret_view directly is an in-module detail.

export import :device_buffer;
export import :pinned_buffer;
export import :unified_buffer;
export import :host_buffer;
export import :copy;
export import :memset;

export namespace wwr::extension {
using wwr::extension::buffer_base;
using wwr::extension::buffer_typename;
using wwr::extension::same_value_type;
} // namespace wwr::extension
