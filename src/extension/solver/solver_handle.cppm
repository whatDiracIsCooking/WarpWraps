/**
 * @file solver_handle.cppm
 * @brief RAII wrapper for a GPU solver handle
 *
 * Provides WwrsolverDnHandleWrapper class for automatic GPU solver handle management.
 */

export module wwr.extension.solver:solver_handle;

import :solver_error;
import wwr.solver;
import wwr.extension.common;
import wwr.extension.handle;
import std;

export namespace wwr::extension {

/**
 * @brief RAII wrapper for a GPU solver handle
 *
 * A cuSOLVER/rocSOLVER handle belongs to whatever device was current when it
 * was created, so this derives from DeviceBoundHandle: construction selects
 * dev_idx (the first constructor argument, default 0), creates the handle
 * there, and records it -- read it back with dev_idx(). Destroys the handle on
 * destruction; supports move semantics, copy is deleted.
 *
 * @tparam P_create Error policy type for creation
 * @tparam P_destroy Error policy type for destruction
 *
 * @note P_destroy MUST NOT THROW - it is called from the destructor.
 */
template<error_policy<wwrsolverStatus_t> P_create,
         nothrow_error_policy<wwrsolverStatus_t> P_destroy>
class WwrsolverDnHandleWrapper
    : public DeviceBoundHandle<wwrsolverDnHandle_t, WwrsolverDnHandleWrapper<P_create, P_destroy>,
                            P_create, P_destroy> {
private:
  using Base = DeviceBoundHandle<wwrsolverDnHandle_t, WwrsolverDnHandleWrapper<P_create, P_destroy>,
                              P_create, P_destroy>;

public:
  // The `WwrsolverDnHandleWrapper(int dev_idx = 0)` default/per-device constructor,
  // inherited from DeviceBoundHandle, which selects and records the owning device.
  using DeviceBoundHandle<wwrsolverDnHandle_t, WwrsolverDnHandleWrapper<P_create, P_destroy>, P_create,
                       P_destroy>::DeviceBoundHandle;

  /// @brief Create a GPU solver handle
  /// @param handle Output parameter for the created handle
  /// @param location Source location where creation was requested
  void create(wwrsolverDnHandle_t *handle, std::source_location location) {
    gpu_check(wwrsolverDnCreate(handle), this->policy_create_, location);
  }

  /// @brief Destroy a GPU solver handle
  /// @param handle The handle to destroy
  void destroy(wwrsolverDnHandle_t handle) {
    if (handle != nullptr) {
      gpu_check(wwrsolverDnDestroy(handle), this->policy_destroy_);
    }
  }
};

} // namespace wwr::extension
