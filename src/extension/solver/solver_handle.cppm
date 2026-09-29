/**
 * @file solver_handle.cppm
 * @brief RAII wrapper for a GPU solver handle
 *
 * Provides SolverDnHandleWrapper class for automatic GPU solver handle management.
 */

export module wwr.extension.solver:solver_handle;

import :solver_error;
import wwr.solver;
import wwr.runtime_api; // wwrError_t (the device-access policy's error type)
import wwr.extension.common;
import wwr.extension.handle;
import std;

export namespace wwr::extension {

/**
 * @brief RAII wrapper for a GPU solver handle bound to a work stream
 *
 * A cuSOLVER/rocSOLVER handle binds a work stream and enqueues asynchronous work
 * on it, so this derives from StreamBoundHandle: it is constructed from a shared
 * stream owner (never a bare device index), created on that owner's device,
 * bound to that owner's stream via wwrsolverDnSetStream, and retains the owner so
 * the stream outlives it. Read the owning device back with dev_idx() and the
 * bound stream with stream() (it is a device_handle_stream). Destroys the handle
 * on destruction; supports move semantics, copy is deleted.
 *
 * @tparam P_create Error policy type for creation (and the wwrsolverDnSetStream bind)
 * @tparam P_destroy Error policy type for destruction
 * @tparam S The stream owner's type (a device_handle_stream), retained so the
 *         bound stream outlives the handle -- as DeviceBufferWrapper retains H.
 * @tparam P_device_access Error policy for the device set/get calls
 *
 * @note P_destroy MUST NOT THROW - it is called from the destructor.
 */
template<error_policy<wwrsolverStatus_t> P_create,
         nothrow_error_policy<wwrsolverStatus_t> P_destroy, device_handle_stream S,
         error_policy<wwrError_t> P_device_access>
class SolverDnHandleWrapper
    : public StreamBoundHandle<wwrsolverDnHandle_t,
                            SolverDnHandleWrapper<P_create, P_destroy, S, P_device_access>,
                            P_create, P_destroy, S, P_device_access> {
public:
  // The `SolverDnHandleWrapper(shared_ptr<S> owner)` stream-owner constructor,
  // inherited from StreamBoundHandle, which creates the handle on the owner's
  // device and binds its stream (see set_stream below).
  using StreamBoundHandle<wwrsolverDnHandle_t,
                       SolverDnHandleWrapper<P_create, P_destroy, S, P_device_access>, P_create,
                       P_destroy, S, P_device_access>::StreamBoundHandle;

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

  /// @brief Bind the work stream onto the handle (StreamBoundHandle hook)
  /// @param handle The handle to bind the stream onto
  /// @param stream The work stream to bind
  /// @param location Source location where the bind was requested
  void set_stream(wwrsolverDnHandle_t handle, wwrStream_t stream, std::source_location location) {
    gpu_check(wwrsolverDnSetStream(handle, stream), this->policy_create_, location);
  }
};

} // namespace wwr::extension
