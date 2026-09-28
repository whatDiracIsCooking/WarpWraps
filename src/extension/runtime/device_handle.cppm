/**
 * @file device_handle.cppm
 * @brief Value object bundling one physical GPU's identity, properties, a
 *        default allocation stream and a default memory pool
 */

export module wwr.extension.runtime:device_handle;

import :gpu_stream;
import :gpu_mem_pool;
import wwr.runtime_api;
import wwr.extension.common;
import wwr.extension.handle;
import std;

export namespace wwr::extension {

/**
 * @brief A type usable as the backing device handle for a DeviceBuffer
 *
 * Structural, like error_policy: any type that exposes an `index()`, an
 * `alloc_stream()` and a `mem_pool()` yielding the raw backend handles a pool
 * allocation needs. Downstream code can model this with its own handle type
 * instead of wwr's DeviceHandle -- returning raw wwrStream_t / wwrMemPool_t
 * directly is fine; GpuStream / GpuMemPool satisfy it via their implicit
 * conversions (BaseHandle::operator T).
 *
 * @note The accessors are required noexcept because DeviceBuffer::deallocate()
 *       reads index() and alloc_stream() on the destructor path -- the same
 *       destructor-safety rule that motivates nothrow_error_policy.
 */
template<typename H>
concept device_handle = requires(const H h) {
  { h.index() } noexcept -> std::convertible_to<int>;
  { h.alloc_stream() } noexcept -> std::convertible_to<wwrStream_t>;
  { h.mem_pool() } noexcept -> std::convertible_to<wwrMemPool_t>;
};

/**
 * @brief Identity, static properties and default allocation stream of one physical GPU
 *
 * Constructed from a device index; queries the runtime for that device's
 * properties once, at construction, and eagerly creates a stream and a memory
 * pool on that device -- alloc_stream() and mem_pool() -- for callers that want
 * to allocate device memory without managing their own stream or pool (e.g. a
 * device buffer, which is built from a DeviceHandle and draws from its pool on
 * its stream). An out-of-range index
 * or a driver failure aborts through the default error policy, matching the
 * GPU handle wrappers -- constructing a DeviceHandle means you intend to *use*
 * that device, which is not a recoverable operation in this layer.
 *
 * Move-only, because it owns a stream and a memory pool: two DeviceHandle
 * instances must never both claim ownership of the same underlying stream or
 * pool. Every device-bound resource already takes its device by `int dev_idx`
 * (see DeviceBoundHandle in wwr.extension.common) rather than by DeviceHandle,
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
  const wwrDeviceProp &props() const noexcept { return props_; }

  /// @brief The default allocation stream, created on this device at construction
  GpuStreamWrapper<AbortPolicy<wwrError_t>, AbortPolicy<wwrError_t>> &alloc_stream() noexcept { return alloc_stream_; }
  /// @copydoc alloc_stream()
  const GpuStreamWrapper<AbortPolicy<wwrError_t>, AbortPolicy<wwrError_t>> &alloc_stream() const noexcept { return alloc_stream_; }

  /// @brief The default memory pool, created on this device at construction
  GpuMemPoolWrapper<AbortPolicy<wwrError_t>, AbortPolicy<wwrError_t>> &mem_pool() noexcept { return mem_pool_; }
  /// @copydoc mem_pool()
  const GpuMemPoolWrapper<AbortPolicy<wwrError_t>, AbortPolicy<wwrError_t>> &mem_pool() const noexcept { return mem_pool_; }

private:
  /// @brief Query one device's properties, aborting on failure
  static wwrDeviceProp query_props(int index, std::source_location location) {
    wwrDeviceProp prop{};
    gpu_check(wwrGetDeviceProperties(&prop, index), location);
    return prop;
  }

  int index_ = 0;
  wwrDeviceProp props_{};
  GpuStreamWrapper<AbortPolicy<wwrError_t>, AbortPolicy<wwrError_t>> alloc_stream_;
  GpuMemPoolWrapper<AbortPolicy<wwrError_t>, AbortPolicy<wwrError_t>> mem_pool_;
};

/// The in-tree reference model must satisfy the concept it inspired -- guards
/// against an accessor later turning throwing or non-const-callable.
static_assert(device_handle<DeviceHandle>);

} // namespace wwr::extension
