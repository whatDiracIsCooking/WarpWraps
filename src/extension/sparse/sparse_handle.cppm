/**
 * @file sparse_handle.cppm
 * @brief RAII wrapper for a GPU sparse handle
 *
 * Provides GpusparseHandle class for automatic GPU sparse handle management.
 */

export module gpumod.extension.sparse:sparse_handle;

import :sparse_error;
import gpumod.sparse;
import gpumod.extension.common;
import std;

export namespace gpumod::extension {

// Specialize HandleErrorType for gpusparseHandle_t
template<>
struct HandleErrorType<gpusparseHandle_t> {
  using type = gpusparseStatus_t;
};

/**
 * @brief RAII wrapper for a GPU sparse handle
 *
 * Automatically creates a GPU sparse handle on construction and destroys it on destruction.
 * Supports move semantics for transferring ownership.
 *
 * @tparam P_create Error policy type for creation (defaults to DefaultErrorPolicy<gpusparseStatus_t>)
 * @tparam P_destroy Error policy type for destruction (defaults to P_create)
 *
 * @note P_destroy MUST NOT THROW - it is called from the destructor.
 */
template<error_policy<gpusparseStatus_t> P_create = DefaultErrorPolicy<gpusparseStatus_t>,
         error_policy<gpusparseStatus_t> P_destroy = P_create>
class GpusparseHandleWrapper
    : public BaseGpuHandle<gpusparseHandle_t, GpusparseHandleWrapper<P_create, P_destroy>, P_create,
                           P_destroy> {
private:
  using Base = BaseGpuHandle<gpusparseHandle_t, GpusparseHandleWrapper<P_create, P_destroy>,
                             P_create, P_destroy>;

public:
  // Default constructors - inherited from base
  using BaseGpuHandle<gpusparseHandle_t, GpusparseHandleWrapper<P_create, P_destroy>, P_create,
                      P_destroy>::BaseGpuHandle;

  /// @brief Create a GPU sparse handle
  /// @param handle Output parameter for the created handle
  /// @param location Source location where creation was requested
  void create(gpusparseHandle_t *handle, std::source_location location) {
    gpu_check(gpusparseCreate(handle), this->policy_create_, location);
  }

  /// @brief Destroy a GPU sparse handle
  /// @param handle The handle to destroy
  void destroy(gpusparseHandle_t handle) {
    if (handle != nullptr) {
      gpu_check(gpusparseDestroy(handle), this->policy_destroy_);
    }
  }
};

/**
 * @brief Convenient alias for GpusparseHandleWrapper with default error policies
 *
 * Usage:
 *   GpusparseHandle handle;  // Instead of GpusparseHandleWrapper<>
 */
using GpusparseHandle = GpusparseHandleWrapper<>;

} // namespace gpumod::extension
