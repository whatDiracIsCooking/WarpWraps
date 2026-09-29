/**
 * @file interface.cppm
 * @brief Primary interface for wwr.extension.surface
 *
 * Owning RAII wrapper for a GPU surface object (cudaSurfaceObject_t /
 * hipSurfaceObject_t, per WWR_GPU_BACKEND). SurfaceObject creates the object
 * with wwrCreateSurfaceObject on construction from a resource descriptor and
 * destroys it with wwrDestroySurfaceObject on scope exit.
 *
 * It is the sibling of TextureObject (see wwr.extension.texture) and shares its
 * design: layered on BaseHandle, riding the flagged-liveness path on CUDA (the
 * handle is an integer with no invalid value) and the null-sentinel path on HIP
 * (a pointer). A surface object takes only a resource descriptor -- there is no
 * texture descriptor, since a surface is an unfiltered read/write view -- and
 * the resource must be a CUDA array allocated with wwrArraySurfaceLoadStore.
 *
 * A separate wrapper rather than a shared "object" base with TextureObject: the
 * two differ in their create signatures (surface takes no texture descriptor)
 * and destroy calls, and a shared base buys little over two thin BaseHandle
 * subclasses.
 *
 * Usage:
 *   import wwr.extension.surface;
 *
 *   wwr::extension::SurfaceObject<AbortPolicy<wwrError_t>, AbortPolicy<wwrError_t>>
 *       surf{res_desc};
 *   kernel<<<...>>>(surf.get());   // or the implicit conversion to the handle
 *   // wwrDestroySurfaceObject on scope exit
 */

export module wwr.extension.surface;

import wwr.runtime_api;
import wwr.extension.common;
import wwr.extension.handle;
import std;

export namespace wwr::extension {

/**
 * @brief RAII wrapper for a GPU surface object
 *
 * @tparam P_create Error policy for creation, typed to wwrError_t
 * @tparam P_destroy Error policy for destruction, typed to wwrError_t
 *
 * @note P_destroy MUST NOT THROW -- it is called from the destructor.
 */
template<error_policy<wwrError_t> P_create, nothrow_error_policy<wwrError_t> P_destroy>
class SurfaceObject
    : public BaseHandle<wwrSurfaceObject_t, SurfaceObject<P_create, P_destroy>, P_create, P_destroy> {
  using Base = BaseHandle<wwrSurfaceObject_t, SurfaceObject<P_create, P_destroy>, P_create, P_destroy>;

public:
  /// @brief Create a surface object from a resource descriptor
  /// @param res_desc Describes the array the surface reads and writes; the array
  ///        must have been allocated with wwrArraySurfaceLoadStore
  /// @param policy_create Error policy for the create call
  /// @param policy_destroy Error policy for the destroy call (must not throw)
  /// @param location Source location where creation was requested
  SurfaceObject(const wwrResourceDesc &res_desc, P_create policy_create = {},
                P_destroy policy_destroy = {},
                std::source_location location = std::source_location::current())
      : Base(typename Base::skip_default_create_t{}, std::move(policy_create),
             std::move(policy_destroy)) {
    this->adopt(gpu_check(wwrCreateSurfaceObject(&this->handle_, &res_desc), this->create_policy(),
                          location));
  }

  /// @brief Destroy the surface object (BaseHandle destructor hook)
  /// @param handle The surface object to destroy
  void destroy(wwrSurfaceObject_t handle) {
    gpu_check(wwrDestroySurfaceObject(handle), this->destroy_policy());
  }
};

} // namespace wwr::extension
