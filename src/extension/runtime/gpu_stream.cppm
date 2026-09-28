/**
 * @file gpu_stream.cppm
 * @brief RAII wrapper for GPU stream handles
 *
 * Provides GpuStreamWrapper for automatic GPU stream management. The borrow-safe
 * operations (sync/wait_event/begin_capture) are free functions taking a raw
 * wwrStream_t, so one definition serves the owner, its view, and a bare handle
 * (the default stream, or one owned elsewhere) alike -- see the free functions
 * below and runtime/README.md. end_capture stays a member: it mints an owned
 * GpuGraphWrapper through the create policy, which a borrow has no business doing.
 */

export module wwr.extension.runtime:gpu_stream;

import :gpu_graph;
import wwr.runtime_api;
import wwr.extension.common;
import wwr.extension.handle;
import std;

export namespace wwr::extension {

/// @brief Non-owning, copyable view of a stream handle (carries its device
///        index). Returned by GpuStreamWrapper::view() (from DeviceBoundHandle); the
///        borrow-safe stream operations are the free functions below, which act
///        on it, on an owning GpuStreamWrapper, or on a raw wwrStream_t.
using GpuStreamView = DeviceBoundHandleView<wwrStream_t>;

/**
 * @brief RAII wrapper for GPU stream
 *
 * Automatically creates a GPU stream on construction and destroys it on destruction.
 * Supports move semantics for transferring ownership.
 *
 * @tparam P_create Error policy type for creation (defaults to AbortPolicy<wwrError_t>)
 * @tparam P_destroy Error policy type for destruction (defaults to P_create)
 *
 * @note P_destroy MUST NOT THROW - it is called from the destructor.
 */
template<error_policy<wwrError_t> P_create = AbortPolicy<wwrError_t>,
         nothrow_error_policy<wwrError_t> P_destroy = P_create>
class GpuStreamWrapper : public DeviceBoundHandle<wwrStream_t, GpuStreamWrapper<P_create, P_destroy>,
                                                  P_create, P_destroy> {
private:
  using Base =
      DeviceBoundHandle<wwrStream_t, GpuStreamWrapper<P_create, P_destroy>, P_create, P_destroy>;

public:
  // The `GpuStreamWrapper(int dev_idx = 0)` default/per-device constructor, inherited
  // from DeviceBoundHandle, which selects and records the owning device. view()
  // (device-aware, deleted on rvalues) is inherited from DeviceBoundHandle too.
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

  /// @brief End capture on this stream and return the captured graph
  ///
  /// Wraps the graph the runtime produced in an owning GpuGraphWrapper (via
  /// GpuGraphWrapper::adopt), so the whole capture -> instantiate -> launch flow stays
  /// RAII. Errors go through the create policy, matching construction. Stays a
  /// member (not a free function) because it produces an owned graph.
  ///
  /// @param location Source location where capture end was requested
  /// @return A GpuGraphWrapper owning the captured graph
  GpuGraphWrapper<P_create, P_destroy>
  end_capture(std::source_location location = std::source_location::current()) {
    wwrGraph_t graph = nullptr;
    gpu_check(wwrStreamEndCapture(this->handle_, &graph), this->policy_create_, location);
    return GpuGraphWrapper<P_create, P_destroy>::adopt(graph);
  }

  /// @brief Destroy a GPU stream
  /// @param handle The stream to destroy
  void destroy(wwrStream_t handle) {
    if (handle != nullptr) {
      gpu_check(wwrStreamSynchronize(handle), this->policy_destroy_);
      gpu_check(wwrStreamDestroy(handle), this->policy_destroy_);
    }
  }
};

// Borrow-safe stream operations. Free functions on the raw wwrStream_t: an
// owning GpuStreamWrapper and a GpuStreamView both convert to it, so each op has one
// definition that works on the owner, the view, or a bare handle. Found by ADL
// on the wrapper/view types (both in wwr::extension); a bare handle needs
// qualification. They mutate the GPU stream, not any C++ object.

/// @brief Wait for an event on a stream
inline wwrError_t wait_event(wwrStream_t stream, wwrEvent_t event, const unsigned int flags = 0) {
  return wwrStreamWaitEvent(stream, event, flags);
}

/// @brief Begin capturing work submitted to a stream into a graph
inline wwrError_t begin_capture(wwrStream_t stream,
                                const wwrStreamCaptureMode mode = wwrStreamCaptureModeGlobal) {
  return wwrStreamBeginCapture(stream, mode);
}

/// @brief Synchronize a GPU stream
inline wwrError_t sync(wwrStream_t stream) { return wwrStreamSynchronize(stream); }

} // namespace wwr::extension
