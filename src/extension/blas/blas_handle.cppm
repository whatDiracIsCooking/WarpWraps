/**
 * @file blas_handle.cppm
 * @brief RAII wrapper for a GPU BLAS handle
 *
 * Provides WwrblasHandleWrapper class for automatic GPU BLAS handle management.
 */

export module wwr.extension.blas:blas_handle;

import :blas_error;
import wwr.blas;
import wwr.extension.common;
import wwr.extension.handle;
import std;

export namespace wwr::extension {

/**
 * @brief RAII wrapper for a GPU BLAS handle
 *
 * A cuBLAS/rocBLAS handle belongs to whatever device was current when it was
 * created, so this derives from DeviceBoundHandle: construction selects dev_idx
 * (the first constructor argument, default 0), creates the handle there, and
 * records it -- read it back with dev_idx(). Destroys the handle on
 * destruction; supports move semantics, copy is deleted.
 *
 * @tparam P_create Error policy type for creation (defaults to DefaultErrorPolicy<wwrblasStatus_t>)
 * @tparam P_destroy Error policy type for destruction (defaults to P_create)
 *
 * @note P_destroy MUST NOT THROW - it is called from the destructor.
 */
template<error_policy<wwrblasStatus_t> P_create = DefaultErrorPolicy<wwrblasStatus_t>,
         nothrow_error_policy<wwrblasStatus_t> P_destroy = P_create>
class WwrblasHandleWrapper
    : public DeviceBoundHandle<wwrblasHandle_t, WwrblasHandleWrapper<P_create, P_destroy>, P_create,
                            P_destroy> {
private:
  using Base = DeviceBoundHandle<wwrblasHandle_t, WwrblasHandleWrapper<P_create, P_destroy>, P_create,
                              P_destroy>;

public:
  // The `WwrblasHandleWrapper(int dev_idx = 0)` default/per-device constructor,
  // inherited from DeviceBoundHandle, which selects and records the owning device.
  using DeviceBoundHandle<wwrblasHandle_t, WwrblasHandleWrapper<P_create, P_destroy>, P_create,
                       P_destroy>::DeviceBoundHandle;

  /// @brief Create a GPU BLAS handle
  /// @param handle Output parameter for the created handle
  /// @param location Source location where creation was requested
  void create(wwrblasHandle_t *handle, std::source_location location) {
    gpu_check(wwrblasCreate(handle), this->policy_create_, location);
  }

  /// @brief Destroy a GPU BLAS handle
  /// @param handle The handle to destroy
  void destroy(wwrblasHandle_t handle) {
    if (handle != nullptr) {
      gpu_check(wwrblasDestroy(handle), this->policy_destroy_);
    }
  }
};

} // namespace wwr::extension
