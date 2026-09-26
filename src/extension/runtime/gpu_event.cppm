/**
 * @file gpu_event.cppm
 * @brief RAII wrapper for GPU event handles
 *
 * Provides GpuEvent class for automatic GPU event management.
 */

export module gpumod.extension.runtime:gpu_event;

import gpumod.runtime_api;
import gpumod.extension.common;
import std;

export namespace gpumod::extension {

// Specialize HandleErrorType for gpuEvent_t
template<>
struct HandleErrorType<gpuEvent_t> {
  using type = gpuError_t;
};

/**
 * @brief RAII wrapper for GPU event
 *
 * Automatically creates a GPU event on construction and destroys it on destruction.
 * Supports move semantics for transferring ownership.
 *
 * @tparam P_create Error policy type for creation (defaults to DefaultErrorPolicy<gpuError_t>)
 * @tparam P_destroy Error policy type for destruction (defaults to P_create)
 *
 * @note P_destroy MUST NOT THROW - it is called from the destructor.
 */
template<error_policy<gpuError_t> P_create = DefaultErrorPolicy<gpuError_t>,
         error_policy<gpuError_t> P_destroy = P_create>
class GpuEventWrapper
    : public GpuBoundHandle<gpuEvent_t, GpuEventWrapper<P_create, P_destroy>, P_create, P_destroy> {
private:
  using Base =
      GpuBoundHandle<gpuEvent_t, GpuEventWrapper<P_create, P_destroy>, P_create, P_destroy>;

public:
  // The `GpuEvent(int dev_idx = 0)` default/per-device constructor, inherited
  // from GpuBoundHandle, which selects and records the owning device.
  using GpuBoundHandle<gpuEvent_t, GpuEventWrapper<P_create, P_destroy>, P_create,
                       P_destroy>::GpuBoundHandle;

  /// @brief Create a GPU event on `dev_idx` with flags
  /// @param dev_idx Device to create the event on
  /// @param flags Flags for event creation (e.g., gpuEventDisableTiming, gpuEventBlockingSync)
  /// @param location Source location where creation was requested
  GpuEventWrapper(const int dev_idx, const unsigned int flags,
                  std::source_location location = std::source_location::current())
      : Base(typename Base::skip_default_create_t{}) {
    Base::select_device(dev_idx, location);
    gpu_check(gpuEventCreateWithFlags(&this->handle_, flags), this->policy_create_, location);
    this->record_device();
  }

  /// @brief Create a GPU event
  /// @param handle Output parameter for the created event
  /// @param location Source location where creation was requested
  void create(gpuEvent_t *handle, std::source_location location) {
    gpu_check(gpuEventCreate(handle), this->policy_create_, location);
  }

  /// @brief Record the event on a stream
  gpuError_t record(gpuStream_t stream) { return gpuEventRecord(this->handle_, stream); }

  /// @brief Record the event on a stream with flags
  gpuError_t record(gpuStream_t stream, const unsigned int flags) {
    return gpuEventRecordWithFlags(this->handle_, stream, flags);
  }

  /// @brief Synchronize the GPU event
  /// @return gpuSuccess on success, or a GPU error code on failure
  gpuError_t sync() { return gpuEventSynchronize(this->handle_); }

  /// @brief Destroy a GPU event
  /// @param handle The event to destroy
  void destroy(gpuEvent_t handle) {
    if (handle != nullptr) {
      gpu_check(gpuEventSynchronize(handle), this->policy_destroy_);
      gpu_check(gpuEventDestroy(handle), this->policy_destroy_);
    }
  }
};

/**
 * @brief Convenient alias for GpuEventWrapper with default error policies
 *
 * Usage:
 *   GpuEvent event;  // Instead of GpuEventWrapper<>
 */
using GpuEvent = GpuEventWrapper<>;

} // namespace gpumod::extension
