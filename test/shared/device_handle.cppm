/**
 * @file device_handle.cppm
 * @brief Test-side device handle: one physical GPU's identity, properties, a
 *        default allocation stream and a default memory pool
 *
 * A DeviceBuffer needs some concrete type satisfying the device_handle ladder to
 * back it, but the library ships none -- the handle is the caller's to provide.
 * This is the reference model the extension test suites allocate against: the
 * fullest tier (dev_idx() + stream() + pool()), so a buffer built on it draws
 * from that pool on that stream. It lived in src/extension/runtime before the
 * ladder and the concrete handle were separated; it now lives here so the
 * shipped library carries no concrete handle (and no forced AbortPolicy
 * instantiation) of its own.
 *
 * Usage:
 *   import wwr.test.shared.device_handle;
 *   using namespace wwr::extension::test;
 */

export module wwr.test.shared.device_handle;

import wwr.runtime_api;
import wwr.extension.common;
import wwr.extension.handle;
import wwr.extension.runtime;
import wwr.test.shared.abort_policy; // AbortPolicy for this fixture's stream/pool
import std;

export namespace wwr::extension::test {

/**
 * @brief Identity, static properties and default allocation stream of one physical GPU
 *
 * Constructed from a device index; queries the runtime for that device's
 * properties once, at construction, and eagerly creates a stream and a memory
 * pool on that device -- stream() and pool() -- for callers that want
 * to allocate device memory without managing their own stream or pool (e.g. a
 * device buffer, which is built from a DeviceHandle and draws from its pool on
 * its stream). An out-of-range index or a driver failure aborts through the
 * default error policy, matching the GPU handle wrappers -- constructing a
 * DeviceHandle means you intend to *use* that device, which is not a recoverable
 * operation in this layer.
 *
 * Move-only, because it owns a stream and a memory pool: two DeviceHandle
 * instances must never both claim ownership of the same underlying stream or
 * pool. Every device-bound resource already takes its device by `int dev_idx`
 * (see DeviceBoundHandle in wwr.extension.handle) rather than by DeviceHandle,
 * so this bundles a device rather than gating access to one.
 *
 * The full cudaDeviceProp / hipDeviceProp_t is held directly and exposed via
 * props(); this class deliberately does not mirror individual fields (name,
 * total memory, compute capability) behind their own accessors.
 */
class DeviceHandle {
  // stream_ and pool_ bind all three policy slots (create, destroy,
  // device-access) to abort-on-failure.
  using Abort = AbortPolicy<wwrError_t>;
  using Stream = StreamWrapper<Abort, Abort, Abort>;
  using Pool = MemPoolWrapper<Abort, Abort, Abort>;

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
  Stream &stream() noexcept { return stream_; }
  /// @copydoc stream()
  const Stream &stream() const noexcept { return stream_; }

  /// @brief The default memory pool, created on this device at construction
  Pool &pool() noexcept { return pool_; }
  /// @copydoc pool()
  const Pool &pool() const noexcept { return pool_; }

private:
  /// @brief Query one device's properties, aborting on failure
  static wwrDeviceProp query_props(int index, std::source_location location) {
    wwrDeviceProp prop{};
    gpu_check(wwrGetDeviceProperties(&prop, index), Abort{}, location);
    return prop;
  }

  int index_ = 0;
  wwrDeviceProp props_{};
  Stream stream_;
  Pool pool_;
};

/// The reference model is the fullest tier -- pinning it also pins the two tiers
/// it refines. Guards against an accessor later turning throwing or
/// non-const-callable.
static_assert(device_handle_pool<DeviceHandle>);

} // namespace wwr::extension::test
