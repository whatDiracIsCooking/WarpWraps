/**
 * @file solver_handle.cppm
 * @brief RAII wrapper for a GPU solver handle
 *
 * Provides GpusolverDnHandle class for automatic GPU solver handle management.
 */

export module gpumod.extension.solver:solver_handle;

import :solver_error;
import gpumod.solver;
import gpumod.extension.common;
import std;

export namespace gpumod::extension {

// Specialize HandleErrorType for gpusolverDnHandle_t
template<>
struct HandleErrorType<gpusolverDnHandle_t> {
  using type = gpusolverStatus_t;
};

/**
 * @brief RAII wrapper for a GPU solver handle
 *
 * Automatically creates a GPU solver handle on construction and destroys it on destruction.
 * Supports move semantics for transferring ownership.
 *
 * @tparam P_create Error policy type for creation (defaults to DefaultErrorPolicy<gpusolverStatus_t>)
 * @tparam P_destroy Error policy type for destruction (defaults to P_create)
 *
 * @note P_destroy MUST NOT THROW - it is called from the destructor.
 */
template<error_policy<gpusolverStatus_t> P_create = DefaultErrorPolicy<gpusolverStatus_t>,
         error_policy<gpusolverStatus_t> P_destroy = P_create>
class GpusolverDnHandleWrapper
    : public BaseGpuHandle<gpusolverDnHandle_t, GpusolverDnHandleWrapper<P_create, P_destroy>,
                           P_create, P_destroy> {
private:
  using Base = BaseGpuHandle<gpusolverDnHandle_t, GpusolverDnHandleWrapper<P_create, P_destroy>,
                             P_create, P_destroy>;

public:
  // Default constructors - inherited from base
  using BaseGpuHandle<gpusolverDnHandle_t, GpusolverDnHandleWrapper<P_create, P_destroy>, P_create,
                      P_destroy>::BaseGpuHandle;

  /// @brief Create a GPU solver handle
  /// @param handle Output parameter for the created handle
  /// @param location Source location where creation was requested
  void create(gpusolverDnHandle_t *handle, std::source_location location) {
    gpu_check(gpusolverDnCreate(handle), this->policy_create_, location);
  }

  /// @brief Destroy a GPU solver handle
  /// @param handle The handle to destroy
  void destroy(gpusolverDnHandle_t handle) {
    if (handle != nullptr) {
      gpu_check(gpusolverDnDestroy(handle), this->policy_destroy_);
    }
  }
};

/**
 * @brief Convenient alias for GpusolverDnHandleWrapper with default error policies
 *
 * Usage:
 *   GpusolverDnHandle handle;  // Instead of GpusolverDnHandleWrapper<>
 */
using GpusolverDnHandle = GpusolverDnHandleWrapper<>;

} // namespace gpumod::extension
