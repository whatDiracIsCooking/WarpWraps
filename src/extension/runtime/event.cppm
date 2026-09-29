/**
 * @file event.cppm
 * @brief RAII wrapper for GPU event handles
 *
 * Provides EventWrapper for automatic GPU event management. The borrow-safe
 * operations (record/sync) are free functions taking a raw wwrEvent_t, so one
 * definition serves the owner, its view, and a bare handle alike -- see the
 * free functions below and runtime/README.md.
 */

export module wwr.extension.runtime:event;

import wwr.runtime_api;
import wwr.extension.common;
import wwr.extension.handle;
import std;

export namespace wwr::extension {

/// @brief Non-owning, copyable view of an event handle (carries its device
///        index). Returned by EventWrapper::view() (from DeviceBoundHandle); the
///        borrow-safe event operations are the free functions below, which act
///        on it, on an owning EventWrapper, or on a raw wwrEvent_t.
using EventView = DeviceBoundHandleView<wwrEvent_t>;

/**
 * @brief RAII wrapper for GPU event
 *
 * Automatically creates a GPU event on construction and destroys it on destruction.
 * Supports move semantics for transferring ownership.
 *
 * @tparam P_create Error policy type for creation
 * @tparam P_destroy Error policy type for destruction
 * @tparam P_device_access Error policy for the device set/get calls (defaults to abort)
 *
 * @note P_destroy MUST NOT THROW - it is called from the destructor.
 */
template<error_policy<wwrError_t> P_create,
         nothrow_error_policy<wwrError_t> P_destroy,
         error_policy<wwrError_t> P_device_access>
class EventWrapper
    : public DeviceBoundHandle<wwrEvent_t, EventWrapper<P_create, P_destroy, P_device_access>,
                               P_create, P_destroy, P_device_access> {
private:
  using Base = DeviceBoundHandle<wwrEvent_t, EventWrapper<P_create, P_destroy, P_device_access>,
                                 P_create, P_destroy, P_device_access>;

public:
  // The `EventWrapper(int dev_idx = 0)` default/per-device constructor, inherited
  // from DeviceBoundHandle, which selects and records the owning device. view()
  // (device-aware, deleted on rvalues) is inherited from DeviceBoundHandle too.
  using DeviceBoundHandle<wwrEvent_t, EventWrapper<P_create, P_destroy, P_device_access>,
                          P_create, P_destroy, P_device_access>::DeviceBoundHandle;

  /// @brief Create a GPU event on `dev_idx` with flags
  /// @param dev_idx Device to create the event on
  /// @param flags Flags for event creation (e.g., wwrEventDisableTiming, wwrEventBlockingSync)
  /// @param location Source location where creation was requested
  EventWrapper(const int dev_idx, const unsigned int flags,
                  std::source_location location = std::source_location::current())
      : Base(typename Base::skip_default_create_t{}) {
    Base::select_device(dev_idx, location);
    gpu_check(wwrEventCreateWithFlags(&this->handle_, flags), this->policy_create_, location);
    this->record_device();
  }

  /// @brief Create a GPU event
  /// @param handle Output parameter for the created event
  /// @param location Source location where creation was requested
  void create(wwrEvent_t *handle, std::source_location location) {
    gpu_check(wwrEventCreate(handle), this->policy_create_, location);
  }

  /// @brief Destroy a GPU event
  /// @param handle The event to destroy
  void destroy(wwrEvent_t handle) {
    if (handle != nullptr) {
      gpu_check(wwrEventSynchronize(handle), this->destroy_policy());
      gpu_check(wwrEventDestroy(handle), this->destroy_policy());
    }
  }
};

// Borrow-safe event operations. Free functions on the raw wwrEvent_t: an owning
// EventWrapper and a EventView both convert to it, so each op has one definition
// that works on the owner, the view, or a bare handle. `sync` overloads with the
// stream `sync` -- wwrEvent_t and wwrStream_t are distinct vendor pointer types
// on both backends, so the overload is unambiguous.

/// @brief Record an event on a stream
inline wwrError_t record(wwrEvent_t event, wwrStream_t stream) {
  return wwrEventRecord(event, stream);
}

/// @brief Record an event on a stream with flags
inline wwrError_t record(wwrEvent_t event, wwrStream_t stream, const unsigned int flags) {
  return wwrEventRecordWithFlags(event, stream, flags);
}

/// @brief Synchronize a GPU event
inline wwrError_t sync(wwrEvent_t event) { return wwrEventSynchronize(event); }

} // namespace wwr::extension
