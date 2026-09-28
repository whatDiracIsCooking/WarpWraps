/**
 * @file sparse_handle.cppm
 * @brief RAII wrapper for a GPU sparse handle
 *
 * Provides WwrsparseHandleWrapper class for automatic GPU sparse handle management.
 */

export module wwr.extension.sparse:sparse_handle;

import :sparse_error;
import wwr.sparse;
import wwr.runtime_api; // wwrError_t (the device-access policy's error type)
import wwr.extension.common;
import wwr.extension.handle;
import std;

export namespace wwr::extension {

/**
 * @brief RAII wrapper for a GPU sparse handle
 *
 * A cuSPARSE/rocSPARSE handle belongs to whatever device was current when it
 * was created, so this derives from DeviceBoundHandle: construction selects
 * dev_idx (the first constructor argument, default 0), creates the handle
 * there, and records it -- read it back with dev_idx(). Destroys the handle on
 * destruction; supports move semantics, copy is deleted.
 *
 * @tparam P_create Error policy type for creation
 * @tparam P_destroy Error policy type for destruction
 * @tparam P_device_access Error policy for the device set/get calls (defaults to abort)
 *
 * @note P_destroy MUST NOT THROW - it is called from the destructor.
 */
template<error_policy<wwrsparseStatus_t> P_create,
         nothrow_error_policy<wwrsparseStatus_t> P_destroy,
         error_policy<wwrError_t> P_device_access = AbortPolicy<wwrError_t>>
class WwrsparseHandleWrapper
    : public DeviceBoundHandle<wwrsparseHandle_t,
                            WwrsparseHandleWrapper<P_create, P_destroy, P_device_access>, P_create,
                            P_destroy, P_device_access> {
private:
  using Base = DeviceBoundHandle<wwrsparseHandle_t,
                              WwrsparseHandleWrapper<P_create, P_destroy, P_device_access>,
                              P_create, P_destroy, P_device_access>;

public:
  // The `WwrsparseHandleWrapper(int dev_idx = 0)` default/per-device constructor,
  // inherited from DeviceBoundHandle, which selects and records the owning device.
  using DeviceBoundHandle<wwrsparseHandle_t,
                       WwrsparseHandleWrapper<P_create, P_destroy, P_device_access>, P_create,
                       P_destroy, P_device_access>::DeviceBoundHandle;

  /// @brief Create a GPU sparse handle
  /// @param handle Output parameter for the created handle
  /// @param location Source location where creation was requested
  void create(wwrsparseHandle_t *handle, std::source_location location) {
    gpu_check(wwrsparseCreate(handle), this->policy_create_, location);
  }

  /// @brief Destroy a GPU sparse handle
  /// @param handle The handle to destroy
  void destroy(wwrsparseHandle_t handle) {
    if (handle != nullptr) {
      gpu_check(wwrsparseDestroy(handle), this->policy_destroy_);
    }
  }
};

} // namespace wwr::extension
