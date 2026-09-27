/**
 * @file convenience_memory_buffer.cppm
 * @brief Default-policy aliases for the typed memory-buffer wrappers
 *
 * The `DeviceBuffer` / `HostBuffer` / `PinnedBuffer` / `UnifiedBuffer` families
 * (and their non-owning `*View` counterparts) bind the buffer wrappers to the
 * default error policy. This partition is the curated home for those default
 * bindings; alternative-policy aliases belong here too.
 */

export module gpumod.extension.memory_buffer:convenience_memory_buffer;

import :base_buffer;
import :memory_kind;
import :device_buffer;
import :host_buffer;
import :pinned_buffer;
import :unified_buffer;
import gpumod.extension.common;

export namespace gpumod::extension {

/**
 * @brief Convenient alias for DeviceBufferWrapper with default error policies
 *
 * Usage:
 *   auto device = std::make_shared<DeviceHandle>(0);
 *   DeviceBuffer<float> buffer(1024, device);  // 1024 floats from the handle's pool
 */
template<typename T>
using DeviceBuffer = DeviceBufferWrapper<T>;

/**
 * @brief Non-owning view alias for device memory
 *
 * Usage:
 *   DeviceBufferView<float> view(device_buffer);           // Full view
 *   DeviceBufferView<float> sub_view(device_buffer, 4, 8); // Sub-view: 8 elements starting at offset 4
 */
template<typename T>
using DeviceBufferView = BufferViewWrapper<T, MemoryKind::Device>;

/**
 * @brief Convenient alias for HostBufferWrapper with default error policies
 *
 * Usage:
 *   HostBuffer<float> buffer(1024);  // Allocate 1024 floats in host memory
 */
template<typename T>
using HostBuffer = HostBufferWrapper<T>;

/**
 * @brief Non-owning view alias for host memory
 *
 * Usage:
 *   HostBufferView<float> view(host_buffer);           // Full view
 *   HostBufferView<float> sub_view(host_buffer, 4, 8); // Sub-view: 8 elements starting at offset 4
 */
template<typename T>
using HostBufferView = BufferViewWrapper<T, MemoryKind::Host>;

/**
 * @brief Convenient alias for PinnedBufferWrapper with default error policies
 *
 * Usage:
 *   PinnedBuffer<float> buffer(1024);  // Allocate 1024 floats in pinned memory
 *   PinnedBuffer<float> mapped_buffer(1024, gpuHostAllocMapped);  // With flags
 */
template<typename T>
using PinnedBuffer = PinnedBufferWrapper<T>;

/**
 * @brief Non-owning view alias for pinned host memory
 *
 * Usage:
 *   PinnedBufferView<float> view(pinned_buffer);           // Full view
 *   PinnedBufferView<float> sub_view(pinned_buffer, 4, 8); // Sub-view: 8 elements starting at offset 4
 */
template<typename T>
using PinnedBufferView = BufferViewWrapper<T, MemoryKind::Pinned>;

/**
 * @brief Convenient alias for UnifiedBufferWrapper with default error policies
 *
 * Usage:
 *   UnifiedBuffer<float> buffer(1024);  // Allocate 1024 floats in unified memory
 *   UnifiedBuffer<float> host_buffer(1024, gpuMemAttachHost);  // With flags
 */
template<typename T>
using UnifiedBuffer = UnifiedBufferWrapper<T>;

/**
 * @brief Non-owning view alias for unified memory
 *
 * Usage:
 *   UnifiedBufferView<float> view(unified_buffer);           // Full view
 *   UnifiedBufferView<float> sub_view(unified_buffer, 4, 8); // Sub-view: 8 elements starting at offset 4
 */
template<typename T>
using UnifiedBufferView = BufferViewWrapper<T, MemoryKind::Unified>;

} // namespace gpumod::extension
