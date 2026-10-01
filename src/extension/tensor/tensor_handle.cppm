/**
 * @file tensor_handle.cppm
 * @brief RAII wrapper for a GPU tensor library handle
 *
 * Provides TensorHandleWrapper, which creates a cuTENSOR/hipTensor library
 * context with wwrtensorCreate on construction and destroys it with
 * wwrtensorDestroy on scope exit, removing the leak-prone hand-paired
 * create/destroy the raw API demands.
 *
 * Unlike the BLAS/FFT handles this does NOT derive from StreamBoundHandle:
 * wwrtensorCreate takes no device or stream argument (cuTENSOR binds the stream
 * per-operation, at wwrtensorContract/Permute/... time, not on the handle), so
 * there is nothing for a stream-bound layer to own. It is therefore the simplest
 * kind of library handle -- a plain BaseHandle subclass on the default create
 * path, like a bare pool/context handle. The handle is a pointer on CUDA and HIP,
 * so liveness rides BaseHandle's null sentinel; no ownership flag materialises.
 *
 * Usage:
 *   import wwr.extension.tensor;
 *
 *   // the kit's opt-in policy, or your own; the core forces none.
 *   using wwr::extension::kit::AbortPolicy;
 *   wwr::extension::TensorHandleWrapper<AbortPolicy<wwrtensorStatus_t>,
 *                                       AbortPolicy<wwrtensorStatus_t>> handle;
 *   wwrtensorCreateTensorDescriptor(handle, ...);  // implicit conversion
 *   // wwrtensorDestroy on scope exit
 */

export module wwr.extension.tensor:tensor_handle;

import :tensor_error;
import wwr.tensor;
import wwr.extension.common;
import wwr.extension.handle;
import std;

export namespace wwr::extension {

/**
 * @brief RAII wrapper for a GPU tensor library handle
 *
 * @tparam P_create Error policy type for creation, typed to wwrtensorStatus_t
 * @tparam P_destroy Error policy type for destruction, typed to wwrtensorStatus_t
 *
 * @note P_destroy MUST NOT THROW -- it is called from the destructor.
 */
template<error_policy<wwrtensorStatus_t> P_create,
         nothrow_error_policy<wwrtensorStatus_t> P_destroy>
class TensorHandleWrapper
    : public BaseHandle<wwrtensorHandle_t, TensorHandleWrapper<P_create, P_destroy>, P_create,
                        P_destroy> {
public:
  // The default BaseHandle constructors (which run create() below). A library
  // handle needs no construction arguments, so unlike the texture/surface
  // objects it does not take the skip_default_create path.
  using BaseHandle<wwrtensorHandle_t, TensorHandleWrapper<P_create, P_destroy>, P_create,
                   P_destroy>::BaseHandle;

  /// @brief Create a GPU tensor library handle (BaseHandle default-create hook)
  /// @param handle Output parameter for the created handle
  /// @param location Source location where creation was requested
  void create(wwrtensorHandle_t *handle, std::source_location location) {
    gpu_check(wwrtensorCreate(handle), this->policy_create_, location);
  }

  /// @brief Destroy a GPU tensor library handle (BaseHandle destructor hook)
  /// @param handle The handle to destroy
  void destroy(wwrtensorHandle_t handle) {
    gpu_check(wwrtensorDestroy(handle), this->destroy_policy());
  }
};

} // namespace wwr::extension
