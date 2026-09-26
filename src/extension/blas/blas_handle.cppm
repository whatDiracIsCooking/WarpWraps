/**
 * @file blas_handle.cppm
 * @brief RAII wrapper for a GPU BLAS handle
 *
 * Provides GpublasHandle class for automatic GPU BLAS handle management.
 */

export module gpumod.extension.blas:blas_handle;

import :blas_error;
import gpumod.blas;
import gpumod.extension.common;
import std;

export namespace gpumod::extension {

// Specialize HandleErrorType for gpublasHandle_t
template<>
struct HandleErrorType<gpublasHandle_t> {
  using type = gpublasStatus_t;
};

/**
 * @brief RAII wrapper for a GPU BLAS handle
 *
 * Automatically creates a GPU BLAS handle on construction and destroys it on destruction.
 * Supports move semantics for transferring ownership.
 *
 * @tparam P_create Error policy type for creation (defaults to DefaultErrorPolicy<gpublasStatus_t>)
 * @tparam P_destroy Error policy type for destruction (defaults to P_create)
 *
 * @note P_destroy MUST NOT THROW - it is called from the destructor.
 */
template<error_policy<gpublasStatus_t> P_create = DefaultErrorPolicy<gpublasStatus_t>,
         error_policy<gpublasStatus_t> P_destroy = P_create>
class GpublasHandleWrapper
    : public BaseGpuHandle<gpublasHandle_t, GpublasHandleWrapper<P_create, P_destroy>, P_create,
                           P_destroy> {
private:
  using Base = BaseGpuHandle<gpublasHandle_t, GpublasHandleWrapper<P_create, P_destroy>, P_create,
                             P_destroy>;

public:
  // Default constructors - inherited from base
  using BaseGpuHandle<gpublasHandle_t, GpublasHandleWrapper<P_create, P_destroy>, P_create,
                      P_destroy>::BaseGpuHandle;

  /// @brief Create a GPU BLAS handle
  /// @param handle Output parameter for the created handle
  /// @param location Source location where creation was requested
  void create(gpublasHandle_t *handle, std::source_location location) {
    gpu_check(gpublasCreate(handle), this->policy_create_, location);
  }

  /// @brief Destroy a GPU BLAS handle
  /// @param handle The handle to destroy
  void destroy(gpublasHandle_t handle) {
    if (handle != nullptr) {
      gpu_check(gpublasDestroy(handle), this->policy_destroy_);
    }
  }
};

/**
 * @brief Convenient alias for GpublasHandleWrapper with default error policies
 *
 * Usage:
 *   GpublasHandle handle;  // Instead of GpublasHandleWrapper<>
 */
using GpublasHandle = GpublasHandleWrapper<>;

} // namespace gpumod::extension
