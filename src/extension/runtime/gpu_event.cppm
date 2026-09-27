/**
 * @file gpu_event.cppm
 * @brief RAII wrapper for GPU event handles
 *
 * Provides GpuEvent class for automatic GPU event management.
 */

export module wwr.extension.runtime:gpu_event;

import wwr.runtime_api;
import wwr.extension.common;
import wwr.extension.handle;
import std;

export namespace wwr::extension {

/**
 * @brief Borrow-safe event operations, shared by the owner and the view
 *
 * CRTP mixin keyed on Derived::get(): every method forwards to the borrowed
 * wwrEvent_t and touches no ownership state, so it is correct for both
 * GpuEventWrapper (owns the event) and GpuEventView (borrows it) with no
 * duplication. Ownership-producing operations, if any, stay on the owner.
 *
 * Methods are const: they mutate the GPU event, not the C++ object, exactly as
 * a pointer's operations are const on the pointer.
 */
template<typename Derived>
class GpuEventAccess {
private:
  const Derived &self() const noexcept { return static_cast<const Derived &>(*this); }

public:
  /// @brief Record the event on a stream
  wwrError_t record(wwrStream_t stream) const { return wwrEventRecord(self().get(), stream); }

  /// @brief Record the event on a stream with flags
  wwrError_t record(wwrStream_t stream, const unsigned int flags) const {
    return wwrEventRecordWithFlags(self().get(), stream, flags);
  }

  /// @brief Synchronize the GPU event
  /// @return wwrSuccess on success, or a GPU error code on failure
  wwrError_t sync() const { return wwrEventSynchronize(self().get()); }
};

/**
 * @brief Non-owning, copyable view over a GPU event
 *
 * Carries the borrowed handle plus its device index (via DeviceBoundHandleView) and
 * the borrow-safe event operations (via GpuEventAccess). Construct one from an
 * owning GpuEvent with `.view()`, or directly from a raw wwrEvent_t you did not
 * create. It destroys nothing, so it must not outlive the event it borrows.
 */
class GpuEventView : public DeviceBoundHandleView<wwrEvent_t>, public GpuEventAccess<GpuEventView> {
public:
  using DeviceBoundHandleView<wwrEvent_t>::DeviceBoundHandleView;
};

/**
 * @brief RAII wrapper for GPU event
 *
 * Automatically creates a GPU event on construction and destroys it on destruction.
 * Supports move semantics for transferring ownership.
 *
 * @tparam P_create Error policy type for creation (defaults to DefaultErrorPolicy<wwrError_t>)
 * @tparam P_destroy Error policy type for destruction (defaults to P_create)
 *
 * @note P_destroy MUST NOT THROW - it is called from the destructor.
 */
template<error_policy<wwrError_t> P_create = DefaultErrorPolicy<wwrError_t>,
         nothrow_error_policy<wwrError_t> P_destroy = P_create>
class GpuEventWrapper
    : public DeviceBoundHandle<wwrEvent_t, GpuEventWrapper<P_create, P_destroy>, P_create, P_destroy>,
      public GpuEventAccess<GpuEventWrapper<P_create, P_destroy>> {
private:
  using Base =
      DeviceBoundHandle<wwrEvent_t, GpuEventWrapper<P_create, P_destroy>, P_create, P_destroy>;

public:
  // The `GpuEvent(int dev_idx = 0)` default/per-device constructor, inherited
  // from DeviceBoundHandle, which selects and records the owning device.
  using DeviceBoundHandle<wwrEvent_t, GpuEventWrapper<P_create, P_destroy>, P_create,
                       P_destroy>::DeviceBoundHandle;

  /// @brief Create a GPU event on `dev_idx` with flags
  /// @param dev_idx Device to create the event on
  /// @param flags Flags for event creation (e.g., wwrEventDisableTiming, wwrEventBlockingSync)
  /// @param location Source location where creation was requested
  GpuEventWrapper(const int dev_idx, const unsigned int flags,
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

  // record()/sync() come from GpuEventAccess, shared with GpuEventView.

  /// @brief A non-owning, copyable view of this event (handle + device index)
  ///
  /// Deleted on rvalues so a view cannot be taken from a temporary event, which
  /// would dangle immediately: `GpuEvent{}.view()` does not compile.
  GpuEventView view() const & noexcept { return GpuEventView{this->get(), this->dev_idx()}; }
  GpuEventView view() && = delete;

  /// @brief Destroy a GPU event
  /// @param handle The event to destroy
  void destroy(wwrEvent_t handle) {
    if (handle != nullptr) {
      gpu_check(wwrEventSynchronize(handle), this->policy_destroy_);
      gpu_check(wwrEventDestroy(handle), this->policy_destroy_);
    }
  }
};

} // namespace wwr::extension
