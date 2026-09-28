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
 *   class GpuStreamWrapper : public DeviceBoundHandle<wwrStream_t, GpuStreamWrapper, ...> { ... };
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
 * @tparam P_destroy The error policy type for destruction (defaults to P_create)
 */
template<typename T, typename Derived, typed_error_policy P_create,
         nothrow_error_policy<typename P_create::error_type> P_destroy = P_create>
class DeviceBoundHandle : public BaseHandle<T, Derived, P_create, P_destroy> {
private:
  using Base = BaseHandle<T, Derived, P_create, P_destroy>;

  // Select `dev_idx` as the current device, then yield `loc`. A return-value
  // adapter over select_device: it exists only to carry the void select into
  // the base-initializer argument slot, which is evaluated -- i.e. the device
  // is selected -- BEFORE Base runs Derived::create, so the handle is created
  // on `dev_idx`.
  static std::source_location on_device(int dev_idx, std::source_location loc) {
    select_device(dev_idx, loc);
    return loc;
  }

protected:
  int dev_idx_ = -1; ///< Index of the device the handle was created on (-1 until recorded)

  /// @brief Construct without creating the handle; derived selects the device,
  ///        creates the raw handle, then records it (see the class note).
  DeviceBoundHandle(typename Base::skip_default_create_t tag) noexcept : Base(tag) {}

  /// @brief Make `dev_idx` the current device (call before creating a handle on it).
  ///        Static, so it can run before the instance exists; uses the default
  ///        checker, which aborts on failure.
  static void select_device(int dev_idx,
                            std::source_location location = std::source_location::current()) {
    gpu_check(wwrSetDevice(dev_idx), location);
  }

  /// @brief Record the current device as this handle's owner (call after creating the handle).
  ///        Uses the default checker (aborts on failure), not policy_create_:
  ///        wwrGetDevice returns wwrError_t, which a library handle's error
  ///        policy is not typed to accept (it is typed to wwrblasStatus_t and
  ///        the like), so routing it through policy_create_ only compiles for
  ///        the wwrError_t-typed handles -- the same reason select_device is a
  ///        plain default-checked call.
  void record_device() { gpu_check(wwrGetDevice(&dev_idx_)); }

public:
  /// @brief Create on `dev_idx` (default 0): select it, run Derived::create, record it.
  ///
  /// This is the only create path DeviceBoundHandle offers, and it is why every
  /// child names its device as the first constructor argument. `dev_idx`
  /// defaults to 0 so the common single-GPU case stays `Derived{}`, but there is
  /// deliberately no overload that omits it.
  explicit DeviceBoundHandle(int dev_idx = 0,
                          std::source_location location = std::source_location::current())
      : Base(on_device(dev_idx, location)) {
    record_device();
  }

  /// @brief Create on `dev_idx` (default 0) with a custom error policy.
  DeviceBoundHandle(P_create policy, int dev_idx = 0,
                 std::source_location location = std::source_location::current())
      : Base(std::move(policy), on_device(dev_idx, location)) {
    record_device();
  }

  /// @brief Create on `dev_idx` (default 0) with distinct create/destroy policies.
  DeviceBoundHandle(P_create policy_create, P_destroy policy_destroy, int dev_idx = 0,
                 std::source_location location = std::source_location::current())
      : Base(std::move(policy_create), std::move(policy_destroy), on_device(dev_idx, location)) {
    record_device();
  }

  DeviceBoundHandle(DeviceBoundHandle &&other) noexcept
      : Base(std::move(other)), dev_idx_(other.dev_idx_) {
    other.dev_idx_ = -1;
  }

  DeviceBoundHandle &operator=(DeviceBoundHandle &&other) noexcept {
    if (this != &other) {
      Base::operator=(std::move(other));
      dev_idx_ = other.dev_idx_;
      other.dev_idx_ = -1;
    }
    return *this;
  }

  /// @brief Index of the physical device this handle belongs to (-1 if not yet recorded)
  int dev_idx() const noexcept { return dev_idx_; }

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
