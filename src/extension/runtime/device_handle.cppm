/**
 * @file device_handle.cppm
 * @brief Value object bundling one physical GPU's identity, properties, a
 *        default allocation stream and a default memory pool
 */

export module gpumod.extension.runtime:device_handle;

import :gpu_stream;
import :gpu_mem_pool;
import :convenience_runtime;
import gpumod.runtime_api;
import gpumod.extension.common;
import gpumod.extension.handle;
import std;

export namespace wwr::extension {

/**
 * @brief Identity, static properties and default allocation stream of one physical GPU
 *
 * Constructed from a device index; queries the runtime for that device's
 * properties once, at construction, and eagerly creates a GpuStream and a
 * GpuMemPool on that device -- alloc_stream() and mem_pool() -- for callers
 * that want to allocate device memory without managing their own stream or
 * pool (e.g. DeviceBuffer, which is built from a DeviceHandle and draws from
 * its pool on its stream). An out-of-range index
 * or a driver failure aborts through the default error policy, matching the
 * GPU handle wrappers -- constructing a DeviceHandle means you intend to *use*
 * that device, which is not a recoverable operation in this layer.
 *
 * Move-only, because it owns a GpuStream and a GpuMemPool: two DeviceHandle
 * instances must never both claim ownership of the same underlying stream or
 * pool. Every device-bound resource already takes its device by `int dev_idx`
 * (see DeviceBoundHandle in gpumod.extension.common) rather than by DeviceHandle,
 * so this bundles a device rather than gating access to one.
 *
 * The full cudaDeviceProp / hipDeviceProp_t is held directly and exposed via
 * props(); this class deliberately does not mirror individual fields (name,
 * total memory, compute capability) behind their own accessors.
 */
class DeviceHandle {
public:
  explicit DeviceHandle(int index = 0,
                        std::source_location location = std::source_location::current())
      : index_(index), props_(query_props(index, location)), alloc_stream_(index, location),
        mem_pool_(index, location) {}

  int index() const noexcept { return index_; }

  /// @brief This device's static properties, queried once at construction
  const gpuDeviceProp &props() const noexcept { return props_; }

  /// @brief The default allocation stream, created on this device at construction
  GpuStream &alloc_stream() noexcept { return alloc_stream_; }
  /// @copydoc alloc_stream()
  const GpuStream &alloc_stream() const noexcept { return alloc_stream_; }

  /// @brief The default memory pool, created on this device at construction
  GpuMemPool &mem_pool() noexcept { return mem_pool_; }
  /// @copydoc mem_pool()
  const GpuMemPool &mem_pool() const noexcept { return mem_pool_; }

private:
  /// @brief Query one device's properties, aborting on failure
  static gpuDeviceProp query_props(int index, std::source_location location) {
    gpuDeviceProp prop{};
    gpu_check(gpuGetDeviceProperties(&prop, index), location);
    return prop;
  }

  int index_ = 0;
  gpuDeviceProp props_{};
  GpuStream alloc_stream_;
  GpuMemPool mem_pool_;
};

} // namespace wwr::extension
