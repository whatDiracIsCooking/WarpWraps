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
import gpumod.extension.common.handle;
import std;

export namespace gpumod::extension {

/**
 * @brief RAII wrapper for a GPU BLAS handle
 *
 * A cuBLAS/rocBLAS handle belongs to whatever device was current when it was
 * created, so this derives from GpuBoundHandle: construction selects dev_idx
 * (the first constructor argument, default 0), creates the handle there, and
 * records it -- read it back with dev_idx(). Destroys the handle on
 * destruction; supports move semantics, copy is deleted.
 *
 * @tparam P_create Error policy type for creation (defaults to DefaultErrorPolicy<gpublasStatus_t>)
 * @tparam P_destroy Error policy type for destruction (defaults to P_create)
 *
 * @note P_destroy MUST NOT THROW - it is called from the destructor.
 */
template<error_policy<gpublasStatus_t> P_create = DefaultErrorPolicy<gpublasStatus_t>,
         nothrow_error_policy<gpublasStatus_t> P_destroy = P_create>
class GpublasHandleWrapper
    : public GpuBoundHandle<gpublasHandle_t, GpublasHandleWrapper<P_create, P_destroy>, P_create,
                            P_destroy> {
private:
  using Base = GpuBoundHandle<gpublasHandle_t, GpublasHandleWrapper<P_create, P_destroy>, P_create,
                              P_destroy>;

public:
  // The `GpublasHandle(int dev_idx = 0)` default/per-device constructor,
  // inherited from GpuBoundHandle, which selects and records the owning device.
  using GpuBoundHandle<gpublasHandle_t, GpublasHandleWrapper<P_create, P_destroy>, P_create,
                       P_destroy>::GpuBoundHandle;

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

/// @brief Non-owning, copyable view of a BLAS handle, carrying its device index.
///        Returned by GpublasHandle::view(); converts to gpublasHandle_t for the
///        gpublas* wrappers, so a borrowed handle can be used without owning it.
using GpublasHandleView = GpuBoundHandleView<gpublasHandle_t>;

} // namespace gpumod::extension
