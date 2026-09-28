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
 * @brief The device-handle capability ladder backing a DeviceBuffer
 *
 * Three refining concepts, each adding one accessor and thereby unlocking one
 * more allocation strategy in DeviceBufferWrapper:
 *
 *   device_handle         dev_idx()   -> synchronous wwrMalloc / wwrFree
 *   device_handle_stream  + stream()  -> wwrMallocAsync / wwrFreeAsync
 *   device_handle_pool    + pool()    -> wwrMallocFromPoolAsync (+ async free)
 *
 * Structural, like error_policy: any type exposing the accessors qualifies.
 * They yield raw backend handles, so downstream code can model a tier with its
 * own type instead of wwr's DeviceHandle -- returning a raw wwrStream_t /
 * wwrMemPool_t is fine, and GpuStream / GpuMemPool satisfy it via their
 * implicit conversions (BaseHandle::operator T). The requirement is stated as
 * convertible_to, never a specific `.get()`, precisely to keep that raw-handle
 * path valid.
 *
 * @note The accessors are required noexcept because DeviceBuffer reads them on
 *       the destructor (deallocate) path -- the same destructor-safety rule
 *       that motivates nothrow_error_policy.
 */
template<typename H>
concept device_handle = requires(const H h) {
  { h.dev_idx() } noexcept -> std::convertible_to<int>;
};

/// @brief A device_handle that also offers a stream -> enables async alloc/free
template<typename H>
concept device_handle_stream = device_handle<H> && requires(const H h) {
  { h.stream() } noexcept -> std::convertible_to<wwrStream_t>;
};

/// @brief A device_handle_stream that also offers a pool -> enables pool alloc
template<typename H>
concept device_handle_pool = device_handle_stream<H> && requires(const H h) {
  { h.pool() } noexcept -> std::convertible_to<wwrMemPool_t>;
};

/**
 * @brief Identity, static properties and default allocation stream of one physical GPU
 *
 * Constructed from a device index; queries the runtime for that device's
 * properties once, at construction, and eagerly creates a stream and a memory
 * pool on that device -- stream() and pool() -- for callers that want
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
      : index_(index), props_(query_props(index, location)),
        // The canonical DeviceBoundHandle ctor puts source_location after the
        // (defaulted) policy block, so forwarding `location` names the three
        // default policies first.
        stream_(index, {}, {}, {}, location), pool_(index, {}, {}, {}, location) {}

  int dev_idx() const noexcept { return index_; }

  /// @brief This device's static properties, queried once at construction
  const wwrDeviceProp &props() const noexcept { return props_; }

  /// @brief The default allocation stream, created on this device at construction
  GpuStreamWrapper<AbortPolicy<wwrError_t>, AbortPolicy<wwrError_t>> &stream() noexcept { return stream_; }
  /// @copydoc stream()
  const GpuStreamWrapper<AbortPolicy<wwrError_t>, AbortPolicy<wwrError_t>> &stream() const noexcept { return stream_; }

  /// @brief The default memory pool, created on this device at construction
  GpuMemPoolWrapper<AbortPolicy<wwrError_t>, AbortPolicy<wwrError_t>> &pool() noexcept { return pool_; }
  /// @copydoc pool()
  const GpuMemPoolWrapper<AbortPolicy<wwrError_t>, AbortPolicy<wwrError_t>> &pool() const noexcept { return pool_; }

private:
  /// @brief Query one device's properties, aborting on failure
  static wwrDeviceProp query_props(int index, std::source_location location) {
    wwrDeviceProp prop{};
    gpu_check(wwrGetDeviceProperties(&prop, index), location);
    return prop;
  }

  int index_ = 0;
  wwrDeviceProp props_{};
  GpuStreamWrapper<AbortPolicy<wwrError_t>, AbortPolicy<wwrError_t>> stream_;
  GpuMemPoolWrapper<AbortPolicy<wwrError_t>, AbortPolicy<wwrError_t>> pool_;
};

/// The in-tree reference model is the fullest tier -- pinning it also pins the
/// two tiers it refines. Guards against an accessor later turning throwing or
/// non-const-callable.
static_assert(device_handle_pool<DeviceHandle>);

} // namespace wwr::extension
