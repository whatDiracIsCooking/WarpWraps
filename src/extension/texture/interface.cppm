/**
 * @file interface.cppm
 * @brief Primary interface for wwr.extension.texture
 *
 * Owning RAII wrapper for a GPU texture object (cudaTextureObject_t /
 * hipTextureObject_t, per WWR_GPU_BACKEND). TextureObject creates the object
 * with wwrCreateTextureObject on construction from a resource + texture
 * descriptor, and destroys it with wwrDestroyTextureObject on scope exit --
 * removing the leak-prone hand-paired create/destroy the raw API demands.
 *
 * It is layered on BaseHandle exactly like FftPlanWrapper, and for the same
 * reason: a texture object is an integer with no reserved invalid value on CUDA
 * (typedef unsigned long long) and a pointer on HIP, so it cannot ride a single
 * liveness convention. BaseHandle already tracks liveness two ways -- the null
 * sentinel for a pointer, an explicit ownership flag otherwise -- and this
 * wrapper is the second consumer of the flagged path after the FFT plan.
 *
 * Unlike a stream, a texture object needs its descriptors at creation, so it
 * takes BaseHandle's skip-default-create path: the constructor builds the object
 * itself and records ownership with adopt(). The descriptors are copied by the
 * create call, so they are not retained. A plain BaseHandle (not
 * DeviceBoundHandle) is the base: wwrCreateTextureObject takes no device
 * argument -- the object's device is implied by the memory its resource
 * descriptor references -- so there is no device index for this layer to own.
 * A resource-view descriptor (the optional fourth create argument) is niche and
 * omitted here; nullptr is passed.
 *
 * Usage:
 *   import wwr.extension.texture;
 *
 *   wwr::extension::TextureObject<AbortPolicy<wwrError_t>, AbortPolicy<wwrError_t>>
 *       tex{res_desc, tex_desc};
 *   kernel<<<...>>>(tex.get());   // or the implicit conversion to the handle
 *   // wwrDestroyTextureObject on scope exit
 */

export module wwr.extension.texture;

import wwr.runtime_api;
import wwr.extension.common;
import wwr.extension.handle;
import std;

export namespace wwr::extension {

/**
 * @brief RAII wrapper for a GPU texture object
 *
 * @tparam P_create Error policy for creation, typed to wwrError_t
 * @tparam P_destroy Error policy for destruction, typed to wwrError_t
 *
 * @note P_destroy MUST NOT THROW -- it is called from the destructor.
 */
template<error_policy<wwrError_t> P_create, nothrow_error_policy<wwrError_t> P_destroy>
class TextureObject
    : public BaseHandle<wwrTextureObject_t, TextureObject<P_create, P_destroy>, P_create, P_destroy> {
  using Base = BaseHandle<wwrTextureObject_t, TextureObject<P_create, P_destroy>, P_create, P_destroy>;

public:
  /// @brief Create a texture object from a resource and texture descriptor
  /// @param res_desc Describes the memory the texture samples (array or linear)
  /// @param tex_desc Describes addressing, filtering and read mode
  /// @param policy_create Error policy for the create call
  /// @param policy_destroy Error policy for the destroy call (must not throw)
  /// @param location Source location where creation was requested
  TextureObject(const wwrResourceDesc &res_desc, const wwrTextureDesc &tex_desc,
                P_create policy_create = {}, P_destroy policy_destroy = {},
                std::source_location location = std::source_location::current())
      : Base(typename Base::skip_default_create_t{}, std::move(policy_create),
             std::move(policy_destroy)) {
    this->adopt(gpu_check(wwrCreateTextureObject(&this->handle_, &res_desc, &tex_desc, nullptr),
                          this->create_policy(), location));
  }

  /// @brief Destroy the texture object (BaseHandle destructor hook)
  /// @param handle The texture object to destroy
  void destroy(wwrTextureObject_t handle) {
    gpu_check(wwrDestroyTextureObject(handle), this->destroy_policy());
  }
};

} // namespace wwr::extension
