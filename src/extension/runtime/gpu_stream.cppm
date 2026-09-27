/**
 * @file gpu_stream.cppm
 * @brief RAII wrapper for GPU stream handles
 *
 * Provides GpuStream class for automatic GPU stream management.
 */

export module wwr.extension.runtime:gpu_stream;

import :gpu_graph;
import wwr.runtime_api;
import wwr.extension.common;
import wwr.extension.handle;
import std;

export namespace wwr::extension {

/**
 * @brief Borrow-safe stream operations, shared by the owner and the view
 *
 * CRTP mixin keyed on Derived::get(): every method forwards to the borrowed
 * wwrStream_t and touches no ownership state, so it is correct for both
 * GpuStreamWrapper (owns the stream) and GpuStreamView (borrows it) with no
 * duplication. end_capture stays on the owner: it produces an owned GpuGraph
 * through the create policy, which a non-owning view has no business doing.
 *
 * Methods are const: they mutate the GPU stream, not the C++ object.
 */
template<typename Derived>
class GpuStreamAccess {
private:
  const Derived &self() const noexcept { return static_cast<const Derived &>(*this); }

public:
  /// @brief Wait for an event on this stream
  wwrError_t wait_event(wwrEvent_t event, const unsigned int flags = 0) const {
    return wwrStreamWaitEvent(self().get(), event, flags);
  }

  /// @brief Begin capturing work submitted to this stream into a graph
  /// @param mode Capture mode (defaults to wwrStreamCaptureModeGlobal)
  /// @return wwrSuccess on success, or a GPU error code on failure
  wwrError_t begin_capture(const wwrStreamCaptureMode mode = wwrStreamCaptureModeGlobal) const {
    return wwrStreamBeginCapture(self().get(), mode);
  }

  /// @brief Synchronize the GPU stream
  /// @return wwrSuccess on success, or a GPU error code on failure
  wwrError_t sync() const { return wwrStreamSynchronize(self().get()); }
};

/**
 * @brief Non-owning, copyable view over a GPU stream
 *
 * Carries the borrowed handle plus its device index (via DeviceBoundHandleView) and
 * the borrow-safe stream operations (via GpuStreamAccess). Construct one from an
 * owning GpuStream with `.view()`, or directly from a raw wwrStream_t you did not
 * create -- the default stream (0), or a stream owned elsewhere. It destroys
 * nothing, so it must not outlive the stream it borrows.
 */
class GpuStreamView : public DeviceBoundHandleView<wwrStream_t>,
                      public GpuStreamAccess<GpuStreamView> {
public:
  using DeviceBoundHandleView<wwrStream_t>::DeviceBoundHandleView;
};

/**
 * @brief RAII wrapper for GPU stream
 *
 * Automatically creates a GPU stream on construction and destroys it on destruction.
 * Supports move semantics for transferring ownership.
 *
 * @tparam P_create Error policy type for creation (defaults to DefaultErrorPolicy<wwrError_t>)
 * @tparam P_destroy Error policy type for destruction (defaults to P_create)
 *
 * @note P_destroy MUST NOT THROW - it is called from the destructor.
 */
template<error_policy<wwrError_t> P_create = DefaultErrorPolicy<wwrError_t>,
         nothrow_error_policy<wwrError_t> P_destroy = P_create>
class GpuStreamWrapper : public DeviceBoundHandle<wwrStream_t, GpuStreamWrapper<P_create, P_destroy>,
                                               P_create, P_destroy>,
                         public GpuStreamAccess<GpuStreamWrapper<P_create, P_destroy>> {
private:
  using Base =
      DeviceBoundHandle<wwrStream_t, GpuStreamWrapper<P_create, P_destroy>, P_create, P_destroy>;

public:
  // The `GpuStream(int dev_idx = 0)` default/per-device constructor, inherited
  // from DeviceBoundHandle, which selects and records the owning device.
  using DeviceBoundHandle<wwrStream_t, GpuStreamWrapper<P_create, P_destroy>, P_create,
                       P_destroy>::DeviceBoundHandle;

  /// @brief Create a GPU stream on `dev_idx` with flags
  /// @param dev_idx Device to create the stream on
  /// @param flags Flags for stream creation (e.g., wwrStreamNonBlocking)
  /// @param location Source location where creation was requested
  GpuStreamWrapper(const int dev_idx, const unsigned int flags,
                   std::source_location location = std::source_location::current())
      : Base(typename Base::skip_default_create_t{}) {
    Base::select_device(dev_idx, location);
    gpu_check(wwrStreamCreateWithFlags(&this->handle_, flags), this->policy_create_, location);
    this->record_device();
  }

  /// @brief Create a GPU stream on `dev_idx` with flags and priority
  /// @param dev_idx Device to create the stream on
  /// @param flags Flags for stream creation (e.g., wwrStreamNonBlocking)
  /// @param priority Stream priority (lower values indicate higher priority)
  /// @param location Source location where creation was requested
  GpuStreamWrapper(const int dev_idx, const unsigned int flags, const int priority,
                   std::source_location location = std::source_location::current())
      : Base(typename Base::skip_default_create_t{}) {
    Base::select_device(dev_idx, location);
    gpu_check(wwrStreamCreateWithPriority(&this->handle_, flags, priority), this->policy_create_,
              location);
    this->record_device();
  }

  /// @brief Create a GPU stream (non-blocking by default)
  ///
  /// Uses wwrStreamNonBlocking so a default-constructed stream does not
  /// serialize against the legacy default stream (0). Callers that want the
  /// legacy blocking behaviour construct with an explicit `flags` argument.
  ///
  /// @param handle Output parameter for the created stream
  /// @param location Source location where creation was requested
  void create(wwrStream_t *handle, std::source_location location) {
    gpu_check(wwrStreamCreateWithFlags(handle, wwrStreamNonBlocking), this->policy_create_,
              location);
  }

  // wait_event()/begin_capture()/sync() come from GpuStreamAccess, shared with
  // GpuStreamView.

  /// @brief End capture on this stream and return the captured graph
  ///
  /// Wraps the graph the runtime produced in an owning GpuGraph (via
  /// GpuGraph::adopt), so the whole capture -> instantiate -> launch flow stays
  /// RAII. Errors go through the create policy, matching construction.
  ///
  /// @param location Source location where capture end was requested
  /// @return A GpuGraph owning the captured graph
  GpuGraphWrapper<P_create, P_destroy>
  end_capture(std::source_location location = std::source_location::current()) {
    wwrGraph_t graph = nullptr;
    gpu_check(wwrStreamEndCapture(this->handle_, &graph), this->policy_create_, location);
    return GpuGraphWrapper<P_create, P_destroy>::adopt(graph);
  }

  /// @brief A non-owning, copyable view of this stream (handle + device index)
  ///
  /// Deleted on rvalues so a view cannot be taken from a temporary stream, which
  /// would dangle immediately.
  GpuStreamView view() const & noexcept { return GpuStreamView{this->get(), this->dev_idx()}; }
  GpuStreamView view() && = delete;

  /// @brief Destroy a GPU stream
  /// @param handle The stream to destroy
  void destroy(wwrStream_t handle) {
    if (handle != nullptr) {
      gpu_check(wwrStreamSynchronize(handle), this->policy_destroy_);
      gpu_check(wwrStreamDestroy(handle), this->policy_destroy_);
    }
  }
};

} // namespace wwr::extension
