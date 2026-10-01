/**
 * @file blaslt_handle.cppm
 * @brief RAII wrapper for a cuBLASLt / hipBLASLt library handle
 *
 * Provides BlasLtHandleWrapper for automatic cuBLASLt / hipBLASLt handle
 * management.
 */

export module wwr.extension.blaslt:blaslt_handle;

import :blaslt_error; // re-exports wwr.extension.blas, which specializes the
                      // wwrblasLtStatus_t (== wwrblasStatus_t) error policy
import wwr.blaslt;
import wwr.extension.common;
import wwr.extension.handle;
import std;

export namespace wwr::extension {

/**
 * @brief RAII wrapper for a cuBLASLt / hipBLASLt library handle
 *
 * Unlike BlasHandleWrapper, this is a PLAIN BaseHandle, not a StreamBoundHandle:
 * wwrblasLtCreate(&h) takes no stream or device, and cuBLASLt binds no work
 * stream to the handle -- the stream is passed per call to wwrblasLtMatmul
 * instead. So the handle owns no stream and no device index; it is created with
 * the default create() path, destroyed on scope exit, move-only, copy deleted.
 *
 * wwrblasLtHandle_t is a pointer on both backends, so it rides BaseHandle's
 * null-sentinel liveness path (no explicit ownership flag).
 *
 * @tparam P_create Error policy type for creation
 * @tparam P_destroy Error policy type for destruction
 *
 * @note P_destroy MUST NOT THROW - it is called from the destructor.
 */
template<error_policy<wwrblasLtStatus_t> P_create,
         nothrow_error_policy<wwrblasLtStatus_t> P_destroy>
class BlasLtHandleWrapper
    : public BaseHandle<wwrblasLtHandle_t, BlasLtHandleWrapper<P_create, P_destroy>, P_create,
                        P_destroy> {
public:
  // Inherit BaseHandle's constructors, including the default create path that
  // runs create() below.
  using BaseHandle<wwrblasLtHandle_t, BlasLtHandleWrapper<P_create, P_destroy>, P_create,
                   P_destroy>::BaseHandle;

  /// @brief Create a cuBLASLt / hipBLASLt library handle
  /// @param handle Output parameter for the created handle
  /// @param location Source location where creation was requested
  void create(wwrblasLtHandle_t *handle, std::source_location location) {
    gpu_check(wwrblasLtCreate(handle), this->policy_create_, location);
  }

  /// @brief Destroy a cuBLASLt / hipBLASLt library handle
  /// @param handle The handle to destroy
  void destroy(wwrblasLtHandle_t handle) {
    if (handle != nullptr) {
      gpu_check(wwrblasLtDestroy(handle), this->destroy_policy());
    }
  }
};

} // namespace wwr::extension
