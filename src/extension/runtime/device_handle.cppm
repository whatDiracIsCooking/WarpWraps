/**
 * @file device_handle.cppm
 * @brief DeviceHandle: a ready-made pool-tier device handle, so a consumer need
 *        not hand-roll one
 *
 * A DeviceBuffer (and every stream-bound library handle) is backed by some type
 * satisfying the device_handle ladder in wwr.extension.handle. The stream tier
 * needs no concrete type -- a StreamWrapper already exposes dev_idx() and stream(),
 * so it *is* a device_handle_stream. This supplies the fullest rung: an owned
 * stream and memory pool on one device (dev_idx() + stream() + pool()), so a buffer
 * built on it draws from that pool on that stream (wwrMallocFromPoolAsync).
 *
 * This is the DeviceHandle that lived in wwr.extension.runtime until #93 moved it
 * to test/shared -- to keep the shipped library free of a forced AbortPolicy. It
 * returns here policy-templated: it bakes no policy (the consumer brings one, as
 * every runtime wrapper does), so the reason it left no longer applies.
 *
 * It lives here, not in wwr.extension.handle: it composes StreamWrapper /
 * MemPoolWrapper, and runtime already imports handle, so defining it there would
 * close a module cycle.
 *
 * It is declared in the nested namespace wwr::extension::kit -- the opt-in layer
 * of ready-made helpers -- so a plain `using namespace wwr::extension;` does not
 * pull the generic name `DeviceHandle` into a consumer's scope (a downstream GPU
 * codebase is likely to have its own). A consumer opts in with
 * `using namespace wwr::extension::kit;` or names `kit::DeviceHandle` explicitly.
 *
 * Usage:
 *   import wwr.extension.runtime;
 *   using namespace wwr::extension;
 *
 *   using Abort = kit::AbortPolicy<wwrError_t>;         // kit's opt-in policy, or your own
 *   auto h = std::make_shared<kit::DeviceHandle<Abort, Abort, Abort>>();  // dev + stream + pool
 */

export module wwr.extension.runtime:device_handle;

import :stream;
import :mem_pool;
import wwr.runtime_api;
import wwr.extension.common; // error_policy / nothrow_error_policy, gpu_check
import std;

export namespace wwr::extension::kit {

/**
 * @brief Pool-tier device handle: an owned stream and memory pool on one device
 *
 * The fullest rung of the device_handle ladder (dev_idx() + stream() + pool()).
 * Owns a StreamWrapper and a MemPoolWrapper created on the same device and forwards
 * both, so a DeviceBuffer backed by it allocates from that pool on that stream.
 * The stream tier wants no such type -- a StreamWrapper is already a
 * device_handle_stream on its own.
 *
 * Move-only (it owns two handles) and templated on one policy triple shared by both
 * wrappers; the three policies are named directly, as on StreamWrapper /
 * MemPoolWrapper themselves. props() carries the device's static properties, queried
 * once at construction.
 *
 * @tparam P_create Error policy for the wrappers' creation
 * @tparam P_destroy Error policy for their destruction (nothrow -- destructor path)
 * @tparam P_device Error policy for the wrappers' device set/get calls
 */
template<error_policy<wwrError_t> P_create, nothrow_error_policy<wwrError_t> P_destroy,
         error_policy<wwrError_t> P_device>
class DeviceHandle {
  using Stream = StreamWrapper<P_create, P_destroy, P_device>;
  using Pool = MemPoolWrapper<P_create, P_destroy, P_device>;

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
  /// @brief Query one device's properties through the device policy
  static wwrDeviceProp query_props(int index, std::source_location location) {
    wwrDeviceProp prop{};
    gpu_check(wwrGetDeviceProperties(&prop, index), P_device{}, location);
    return prop;
  }

  int index_ = 0;
  wwrDeviceProp props_{};
  Stream stream_;
  Pool pool_;
};

} // namespace wwr::extension::kit
