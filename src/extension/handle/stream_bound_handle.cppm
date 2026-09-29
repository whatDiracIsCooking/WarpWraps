/**
 * @file stream_bound_handle.cppm
 * @brief CRTP mid-layer for library handles that carry, and outlive-anchor, a work stream
 *
 * Provides StreamBoundHandle, the layer between DeviceBoundHandle and the
 * stream-bound library handles (BLAS/solver/sparse/FFT). A library handle binds
 * a work stream once (vendor …SetStream), then queues asynchronous work on it,
 * so the stream must outlive the handle. This layer captures that shared
 * pattern: it binds the stream in its constructor, exposes stream() -- so it
 * satisfies device_handle_stream and can back a DeviceBuffer's async alloc/free
 * -- and shared-owns the stream's owner so the bound stream outlives it.
 *
 * Usage:
 *   import wwr.extension.handle;
 *
 *   class BlasHandleWrapper : public StreamBoundHandle<wwrblasHandle_t, BlasHandleWrapper, ..., S> {
 *     void create(wwrblasHandle_t *h, std::source_location loc);   // BaseHandle hook
 *     void destroy(wwrblasHandle_t h);                             // BaseHandle hook
 *     void set_stream(wwrblasHandle_t h, wwrStream_t s, std::source_location loc); // this layer's hook
 *   };
 */

export module wwr.extension.handle:stream_bound_handle;

import :handle;
import :device_bound_handle;
import :stream_bound_handle_view;
import :device_handle; // the device_handle_stream concept the owner must satisfy
import wwr.extension.common; // error_policy concepts
import wwr.runtime_api; // wwrStream_t, wwrError_t
import std;

export namespace wwr::extension {

/**
 * @brief CRTP layer over DeviceBoundHandle for a library handle bound to a work stream
 *
 * A cuBLAS/rocBLAS, cuSOLVER/rocSOLVER, cuSPARSE/rocSPARSE handle or cuFFT/hipFFT
 * plan binds a single work stream (via its vendor …SetStream) and then enqueues
 * asynchronous work on it. That makes two things true of every such handle, which
 * this layer factors out of all four:
 *
 *   1. the bound stream must outlive the handle -- so this shared-owns the
 *      stream's owner (std::shared_ptr<S>), exactly the retention
 *      DeviceBufferWrapper uses to keep whatever backs an allocation alive past
 *      the buffer (this layer is that pattern, moved from buffers to handles), and
 *   2. the handle *is* a device_handle_stream once bound -- it exposes stream(),
 *      so a DeviceBufferWrapper built on it draws async on that stream.
 *
 * Like DeviceBufferWrapper on its handle H, this templates on the owner type S
 * (constrained to device_handle_stream) and stores std::shared_ptr<S>. stream()
 * reads back through the retained owner -- owner_->stream() -- rather than caching
 * a copy, so there is one source of truth and nothing to keep in sync; it is the
 * same live read DeviceBufferWrapper does at alloc/free time. (This also sidesteps
 * that FFT's vendor surface offers …SetStream but no …GetStream: the stream never
 * comes back from the library handle, only from the owner it was taken from.)
 *
 * A StreamBoundHandle is always constructed *from* a shared stream owner (like a
 * DeviceBufferWrapper from a shared handle); there is no stream-less path. The
 * handle is created on -- and the stream is verified to belong to -- the owner's
 * device: the constructor forwards owner->dev_idx() to DeviceBoundHandle, whose
 * select-create-record discipline lands the handle on that device, and the bound
 * stream is a device-scoped resource of the same owner. So "stream-bound" entails
 * "same-device-bound" by construction.
 *
 * Derived classes provide the BaseHandle create()/destroy() hooks plus one more
 * hook this layer calls after creation: set_stream(handle, stream, location),
 * which invokes the appropriate vendor …SetStream. It is routed through the
 * derived handle's create policy -- …SetStream returns the handle's own library
 * status type, which that policy is typed to.
 *
 * @tparam T The underlying GPU handle type (e.g., wwrblasHandle_t)
 * @tparam Derived The concrete class inheriting from this layer (CRTP)
 * @tparam P_create The error policy type for creation (and for set_stream)
 * @tparam P_destroy The error policy type for destruction
 * @tparam S The stream owner's type; a device_handle_stream, retained by
 *         shared_ptr so the bound stream outlives the handle -- as
 *         DeviceBufferWrapper retains its handle H.
 * @tparam P_device_access The error policy for the device (set/get) calls; typed
 *         to wwrError_t regardless of the handle's own status type. No default --
 *         the caller names the policy, as everywhere since the shipped library
 *         stopped carrying one.
 */
template<typename T, typename Derived, typed_error_policy P_create,
         nothrow_error_policy<typename P_create::error_type> P_destroy, device_handle_stream S,
         error_policy<wwrError_t> P_device_access>
class StreamBoundHandle
    : public DeviceBoundHandle<T, Derived, P_create, P_destroy, P_device_access> {
private:
  using Base = DeviceBoundHandle<T, Derived, P_create, P_destroy, P_device_access>;

  /// Retains the stream owner so the bound stream outlives this handle; stream()
  /// reads back through it. Null only in the moved-from state.
  std::shared_ptr<S> owner_;

public:
  /**
   * @brief Create the handle on `owner`'s device and bind `owner`'s work stream
   *
   * The one construction path: forwards owner->dev_idx() to DeviceBoundHandle
   * (which selects the device, runs Derived::create there, and records the
   * index), retains the owner, then calls the Derived set_stream hook to bind
   * owner->stream() onto the freshly created handle.
   *
   * @warning Ownership runs ONE direction only: this handle -> shared_ptr ->
   *          stream owner, NEVER owner -> handle. A stream owner that (directly
   *          or transitively) owned a StreamBoundHandle anchored back to it forms
   *          a reference cycle and leaks. This cannot be enforced structurally --
   *          it is a documented contract; see docs/architecture.md section 18
   *          ("Handle lifetime / ownership direction").
   *
   * @param owner Shared stream owner; its device hosts the handle and its stream is bound
   * @param policy_create Error policy for creation and for the set_stream call
   * @param policy_destroy Error policy for destruction
   * @param policy_device Error policy for the device set/get calls
   * @param location Source location where construction was requested
   */
  explicit StreamBoundHandle(std::shared_ptr<S> owner, P_create policy_create = {},
                             P_destroy policy_destroy = {}, P_device_access policy_device = {},
                             std::source_location location = std::source_location::current())
      : Base(owner->dev_idx(), std::move(policy_create), std::move(policy_destroy),
             std::move(policy_device), location),
        owner_(std::move(owner)) {
    // Base has created the handle on the owner's device and recorded dev_idx();
    // the handle is live, so bind the (retained) owner's stream through the hook.
    static_cast<Derived *>(this)->set_stream(this->get(), owner_->stream(), location);
  }

  StreamBoundHandle(StreamBoundHandle &&) noexcept = default;
  StreamBoundHandle &operator=(StreamBoundHandle &&) noexcept = default;

  /// @brief The bound work stream, read back through the retained owner -- makes
  ///        this a device_handle_stream. Undefined on a moved-from handle (null
  ///        owner), matching DeviceBufferWrapper's moved-from contract.
  wwrStream_t stream() const noexcept { return owner_->stream(); }

  /// @brief A non-owning, copyable view of this handle, carrying its device index
  ///        and bound work stream.
  ///
  /// Hides DeviceBoundHandle::view() to return the stream-aware view. The stream
  /// is captured as a bare handle at view() time; the view retains nothing, so it
  /// must not outlive this handle's owner. Deleted on rvalues, as the bases are.
  StreamBoundHandleView<T> view() const & noexcept {
    return StreamBoundHandleView<T>{this->get(), this->dev_idx(), stream()};
  }
  StreamBoundHandleView<T> view() && = delete;
};

} // namespace wwr::extension
