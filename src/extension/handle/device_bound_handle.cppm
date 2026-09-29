/**
 * @file device_bound_handle.cppm
 * @brief CRTP layer recording the physical GPU a handle is bound to
 *
 * Provides DeviceBoundHandle, a thin layer over BaseHandle that records the
 * device index a device-bound handle (stream, event, memory pool, library
 * handle) was created on.
 *
 * Usage:
 *   import wwr.extension.handle;
 *
 *   class StreamWrapper : public DeviceBoundHandle<wwrStream_t, StreamWrapper, ...> { ... };
 */

export module wwr.extension.handle:device_bound_handle;

import :handle;
import :device_bound_handle_view;
import wwr.extension.common; // gpu_check, error_policy
import wwr.runtime_api;
import std;

export namespace wwr::extension {

/**
 * @brief CRTP layer over BaseHandle for handles bound to one physical device
 *
 * A GPU stream, event, cuBLAS/cuSOLVER handle or memory pool is created on --
 * and belongs to -- whatever device was current at creation time. This layer
 * records that device index so it can be queried later. BaseHandle itself
 * stays device-agnostic, which is what lets resources that are *not*
 * device-bound (graphs, graph execs, solver params) sit on it without carrying
 * a device index they have no meaning for.
 *
 * Device placement is always intentional: `dev_idx` is the first constructor
 * argument, defaulted to 0. There is no device-agnostic constructor -- a child
 * of this layer cannot be created without naming (or defaulting) its device, so
 * no handle silently lands on whatever device happened to be current. The
 * recorded index is then read back from the runtime (wwrGetDevice), so it
 * reflects the device the handle was *actually* created on.
 *
 * Derived classes that create the handle through the default create() path get
 * the device selected and recorded automatically. Derived classes that create
 * the raw handle themselves (via skip_default_create_t) must call
 * select_device(dev_idx) *before* creating it and record_device() *after* --
 * including when the device is implied by the handle's own descriptor (e.g. a
 * memory pool's props.location.id, which is selected before the pool is created).
 *
 * @tparam T The underlying GPU handle type (e.g., wwrStream_t)
 * @tparam Derived The concrete class inheriting from this layer (CRTP)
 * @tparam P_create The error policy type for creation
 * @tparam P_destroy The error policy type for destruction
 * @tparam P_device_access The error policy for the device (wwrSetDevice/
 *         wwrGetDevice) calls; typed to wwrError_t regardless of the handle's own
 *         status type. No default -- the caller names the policy.
 */
template<typename T, typename Derived, typed_error_policy P_create,
         nothrow_error_policy<typename P_create::error_type> P_destroy,
         error_policy<wwrError_t> P_device_access>
class DeviceBoundHandle : public BaseHandle<T, Derived, P_create, P_destroy> {
private:
  using Base = BaseHandle<T, Derived, P_create, P_destroy>;

  // Select `dev_idx` as the current device through `dev_policy`, then yield
  // `loc`. A return-value adapter: it exists only to carry the void select into
  // the base-initializer argument slot, which is evaluated -- i.e. the device is
  // selected -- BEFORE Base runs Derived::create, so the handle is created on
  // `dev_idx`. It runs before any DeviceBoundHandle member is alive, so it takes
  // the constructor's `dev_policy` *parameter* by reference (policy_device_ does
  // not exist yet); the constructor then moves that same policy into
  // policy_device_, carrying forward whatever select recorded into it.
  static std::source_location on_device(P_device_access &dev_policy, int dev_idx,
                                        std::source_location loc) {
    gpu_check(wwrSetDevice(dev_idx), dev_policy, loc);
    return loc;
  }

protected:
  int dev_idx_ = -1; ///< Index of the device the handle was created on (-1 until recorded)
  [[no_unique_address]] P_device_access policy_device_{}; ///< Policy for the device (set/get) calls

  /// @brief Construct without creating the handle; derived selects the device,
  ///        creates the raw handle, then records it (see the class note). Leaves
  ///        policy_device_ default-constructed -- the skip-create wrappers take no
  ///        device policy, matching how they take no create/destroy policy either.
  DeviceBoundHandle(typename Base::skip_default_create_t tag) noexcept : Base(tag) {}

  /// @brief Make `dev_idx` the current device (call before creating a handle on it).
  ///        Routes wwrSetDevice through policy_device_, so device-selection error
  ///        handling is under the caller's control.
  void select_device(int dev_idx,
                     std::source_location location = std::source_location::current()) {
    gpu_check(wwrSetDevice(dev_idx), policy_device_, location);
  }

  /// @brief Record the current device as this handle's owner (call after creating the handle).
  ///        Routes wwrGetDevice through policy_device_ (typed to wwrError_t), NOT
  ///        policy_create_: a library handle's create policy is typed to its own
  ///        status enum (wwrblasStatus_t and the like), which wwrGetDevice's
  ///        wwrError_t would not satisfy. A dedicated wwrError_t device policy is
  ///        exactly what lets both device calls carry error handling on every handle.
  void record_device(std::source_location location = std::source_location::current()) {
    gpu_check(wwrGetDevice(&dev_idx_), policy_device_, location);
  }

public:
  /// @brief The one create path: select `dev_idx`, run Derived::create, record it.
  ///
  /// A single canonical constructor with every policy in a fixed positional slot,
  /// all defaulted. This is deliberate: because a runtime handle's create-error
  /// type IS wwrError_t, P_create and P_device_access can be the same type, so a
  /// pair of ctors distinguished only by "which policy" would share a signature
  /// -- ill-formed. One constructor sidesteps that collision entirely.
  ///
  /// `dev_idx` stays first so the common `Derived{}` / `Derived{5}` cases are
  /// unchanged; source_location stays last so passing policies still captures the
  /// caller automatically. `policy_device` is threaded into on_device by reference
  /// (it must -- policy_device_ is not alive yet) and then moved into policy_device_.
  explicit DeviceBoundHandle(int dev_idx = 0, P_create policy_create = {},
                             P_destroy policy_destroy = {}, P_device_access policy_device = {},
                             std::source_location location = std::source_location::current())
      : Base(std::move(policy_create), std::move(policy_destroy),
             on_device(policy_device, dev_idx, location)),
        policy_device_(std::move(policy_device)) {
    record_device(location);
  }

  DeviceBoundHandle(DeviceBoundHandle &&other) noexcept
      : Base(std::move(other)), dev_idx_(other.dev_idx_),
        policy_device_(std::move(other.policy_device_)) {
    other.dev_idx_ = -1;
  }

  DeviceBoundHandle &operator=(DeviceBoundHandle &&other) noexcept {
    if (this != &other) {
      Base::operator=(std::move(other));
      dev_idx_ = other.dev_idx_;
      policy_device_ = std::move(other.policy_device_);
      other.dev_idx_ = -1;
    }
    return *this;
  }

  /// @brief Index of the physical device this handle belongs to (-1 if not yet recorded)
  int dev_idx() const noexcept { return dev_idx_; }

  /// @brief The policy handling this handle's device (set/get) calls, for reading
  ///        back any state a stateful device policy accumulated during construction.
  const P_device_access &device_policy() const noexcept { return policy_device_; }

  /// @brief A non-owning, copyable view of this handle, carrying its device index.
  ///
  /// Hides BaseHandle::view() to return the device-aware view. Deleted on
  /// rvalues, as the base is, so a temporary cannot be viewed.
  DeviceBoundHandleView<T> view() const & noexcept {
    return DeviceBoundHandleView<T>{this->get(), dev_idx_};
  }
  DeviceBoundHandleView<T> view() && = delete;
};

} // namespace wwr::extension
