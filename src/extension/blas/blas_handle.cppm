/**
 * @file blas_handle.cppm
 * @brief RAII wrapper for a GPU BLAS handle
 *
 * Provides BlasHandleWrapper class for automatic GPU BLAS handle management.
 */

export module wwr.extension.blas:blas_handle;

import :blas_error;
import wwr.blas;
import wwr.runtime_api; // wwrError_t (the device-access policy's error type)
import wwr.extension.common;
import wwr.extension.handle;
import std;

export namespace wwr::extension {

/**
 * @brief A type that yields a raw wwrblasHandle_t -- the co-ownable BLAS handle
 *
 * Structural, like device_handle: any type exposing a noexcept get() returning
 * wwrblasHandle_t qualifies, which BlasHandleWrapper does through BaseHandle. A
 * consumer that only needs "something a wwrblasHandle_t comes from" -- e.g.
 * PointerModeScope, which shared-owns one for its lifetime -- constrains on this
 * instead of templating on BlasHandleWrapper's full <P_create, P_destroy, S,
 * P_device_access> list, none of which it uses.
 *
 * The return must be *exactly* wwrblasHandle_t (same_as), not merely convertible:
 * on HIP wwrblasHandle_t is void*, so a convertible_to check would admit any
 * object pointer -- e.g. a StreamWrapper's wwrStream_t (ihipStream_t*) -- and
 * silently accept the wrong handle. same_as still cannot separate blas from
 * solver/sparse on HIP (all three vendor handles ARE void* there) -- the same
 * irreducible void* aliasing the typed_error_policy note describes -- but it does
 * keep the distinct stream/event/pool handle types out.
 */
template<typename H>
concept blas_handle = requires(const H h) {
  { h.get() } noexcept -> std::same_as<wwrblasHandle_t>;
};

/**
 * @brief RAII wrapper for a GPU BLAS handle bound to a work stream
 *
 * A cuBLAS/rocBLAS handle binds a work stream and enqueues asynchronous work on
 * it, so this derives from StreamBoundHandle: it is constructed from a shared
 * stream owner (never a bare device index), created on that owner's device,
 * bound to that owner's stream via wwrblasSetStream, and retains the owner so
 * the stream outlives it. Read the owning device back with dev_idx() and the
 * bound stream with stream() (it is a device_handle_stream). Destroys the handle
 * on destruction; supports move semantics, copy is deleted.
 *
 * @tparam P_create Error policy type for creation (and the wwrblasSetStream bind)
 * @tparam P_destroy Error policy type for destruction
 * @tparam S The stream owner's type (a device_handle_stream), retained so the
 *         bound stream outlives the handle -- as DeviceBufferWrapper retains H.
 * @tparam P_device_access Error policy for the device set/get calls
 *
 * @note P_destroy MUST NOT THROW - it is called from the destructor.
 */
template<error_policy<wwrblasStatus_t> P_create,
         nothrow_error_policy<wwrblasStatus_t> P_destroy, device_handle_stream S,
         error_policy<wwrError_t> P_device_access>
class BlasHandleWrapper
    : public StreamBoundHandle<wwrblasHandle_t,
                            BlasHandleWrapper<P_create, P_destroy, S, P_device_access>, P_create,
                            P_destroy, S, P_device_access> {
public:
  // The `BlasHandleWrapper(shared_ptr<S> owner)` stream-owner constructor,
  // inherited from StreamBoundHandle, which creates the handle on the owner's
  // device and binds its stream (see set_stream below).
  using StreamBoundHandle<wwrblasHandle_t,
                       BlasHandleWrapper<P_create, P_destroy, S, P_device_access>, P_create,
                       P_destroy, S, P_device_access>::StreamBoundHandle;

  /// @brief Create a GPU BLAS handle
  /// @param handle Output parameter for the created handle
  /// @param location Source location where creation was requested
  void create(wwrblasHandle_t *handle, std::source_location location) {
    gpu_check(wwrblasCreate(handle), this->policy_create_, location);
  }

  /// @brief Destroy a GPU BLAS handle
  /// @param handle The handle to destroy
  void destroy(wwrblasHandle_t handle) {
    if (handle != nullptr) {
      gpu_check(wwrblasDestroy(handle), this->destroy_policy());
    }
  }

  /// @brief Bind the work stream onto the handle (StreamBoundHandle hook)
  /// @param handle The handle to bind the stream onto
  /// @param stream The work stream to bind
  /// @param location Source location where the bind was requested
  void set_stream(wwrblasHandle_t handle, wwrStream_t stream, std::source_location location) {
    gpu_check(wwrblasSetStream(handle, stream), this->policy_create_, location);
  }
};

} // namespace wwr::extension
