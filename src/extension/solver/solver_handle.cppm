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

/**
 * @brief RAII wrapper for a GPU solver handle
 *
 * A cuSOLVER/rocSOLVER handle belongs to whatever device was current when it
 * was created, so this derives from GpuBoundHandle: construction selects
 * dev_idx (the first constructor argument, default 0), creates the handle
 * there, and records it -- read it back with dev_idx(). Destroys the handle on
 * destruction; supports move semantics, copy is deleted.
 *
 * @tparam P_create Error policy type for creation (defaults to DefaultErrorPolicy<gpusolverStatus_t>)
 * @tparam P_destroy Error policy type for destruction (defaults to P_create)
 *
 * @note P_destroy MUST NOT THROW - it is called from the destructor.
 */
template<error_policy<gpusolverStatus_t> P_create = DefaultErrorPolicy<gpusolverStatus_t>,
         error_policy<gpusolverStatus_t> P_destroy = P_create>
class GpusolverDnHandleWrapper
    : public GpuBoundHandle<gpusolverDnHandle_t, GpusolverDnHandleWrapper<P_create, P_destroy>,
                            P_create, P_destroy> {
private:
  using Base = GpuBoundHandle<gpusolverDnHandle_t, GpusolverDnHandleWrapper<P_create, P_destroy>,
                              P_create, P_destroy>;

public:
  // The `GpusolverDnHandle(int dev_idx = 0)` default/per-device constructor,
  // inherited from GpuBoundHandle, which selects and records the owning device.
  using GpuBoundHandle<gpusolverDnHandle_t, GpusolverDnHandleWrapper<P_create, P_destroy>, P_create,
                       P_destroy>::GpuBoundHandle;

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

/// @brief Non-owning, copyable view of a solver handle, carrying its device index.
///        Returned by GpusolverDnHandle::view(); converts to gpusolverDnHandle_t
///        for the gpusolverDn* wrappers, so a borrowed handle can be used without
///        owning it.
using GpusolverDnHandleView = GpuBoundHandleView<gpusolverDnHandle_t>;

} // namespace gpumod::extension
