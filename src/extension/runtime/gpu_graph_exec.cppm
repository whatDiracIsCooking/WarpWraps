/**
 * @file gpu_graph_exec.cppm
 * @brief RAII wrapper for GPU executable graph handles
 *
 * Provides GpuGraphExecWrapper, an RAII wrapper for wwrGraphExec_t -- the executable
 * graph produced by instantiating a wwrGraph_t. The borrow-safe operations
 * (launch/upload) are free functions taking a raw wwrGraphExec_t, so one
 * definition serves the owner, its view, and a bare handle alike -- see the
 * free functions below and runtime/README.md.
 */

export module wwr.extension.runtime:gpu_graph_exec;

import wwr.runtime_api;
import wwr.extension.common;
import wwr.extension.handle;
import std;

export namespace wwr::extension {

/// @brief Non-owning, copyable view of an executable-graph handle. Returned by
///        GpuGraphExecWrapper::view() (from BaseHandle; an exec is not device-bound, so
///        it carries no device index); the borrow-safe operations are the free
///        functions below, which act on it, on an owning GpuGraphExecWrapper, or on a
///        raw wwrGraphExec_t.
using GpuGraphExecView = HandleView<wwrGraphExec_t>;

/**
 * @brief RAII wrapper for a GPU executable graph
 *
 * Instantiates a graph template into an executable graph on construction and
 * destroys it on destruction. Supports move semantics for transferring
 * ownership.
 *
 * @note An executable graph is NOT device-bound, so this sits on BaseHandle
 *       rather than DeviceBoundHandle: wwrGraphInstantiate* takes no device, a
 *       graph may span multiple devices, and the exec runs on whatever device
 *       the stream passed to launch() belongs to. There is no device index to
 *       record. See device_bound_handle.cppm.
 *
 * @tparam P_create Error policy type for creation (defaults to AbortPolicy<wwrError_t>)
 * @tparam P_destroy Error policy type for destruction (defaults to P_create)
 *
 * @note P_destroy MUST NOT THROW - it is called from the destructor.
 */
template<error_policy<wwrError_t> P_create = AbortPolicy<wwrError_t>,
         nothrow_error_policy<wwrError_t> P_destroy = P_create>
class GpuGraphExecWrapper
    : public BaseHandle<wwrGraphExec_t, GpuGraphExecWrapper<P_create, P_destroy>, P_create,
                        P_destroy> {
private:
  using Base =
      BaseHandle<wwrGraphExec_t, GpuGraphExecWrapper<P_create, P_destroy>, P_create, P_destroy>;

public:
  // view() (deleted on rvalues) is inherited from BaseHandle.

  /// @brief Instantiate an executable graph from a graph template
  /// @param graph The source graph to instantiate (not owned; used only for this call)
  /// @param flags Instantiation flags (0 for none)
  /// @param location Source location where instantiation was requested
  explicit GpuGraphExecWrapper(wwrGraph_t graph, const unsigned long long flags = 0,
                               std::source_location location = std::source_location::current())
      : Base(typename Base::skip_default_create_t{}) {
    gpu_check(wwrGraphInstantiate(&this->handle_, graph, flags), this->policy_create_, location);
  }

  /// @brief Destroy the executable graph
  /// @param handle The executable graph to destroy
  void destroy(wwrGraphExec_t handle) {
    if (handle != nullptr) {
      gpu_check(wwrGraphExecDestroy(handle), this->policy_destroy_);
    }
  }
};

// Borrow-safe executable-graph operations. Free functions on the raw
// wwrGraphExec_t: an owning GpuGraphExecWrapper and a GpuGraphExecView both convert to
// it, so each op has one definition that works on the owner, the view, or a
// bare handle.

/// @brief Launch an executable graph on a stream
inline wwrError_t launch(wwrGraphExec_t exec, wwrStream_t stream) {
  return wwrGraphLaunch(exec, stream);
}

/// @brief Upload an executable graph to a stream's device without launching it
inline wwrError_t upload(wwrGraphExec_t exec, wwrStream_t stream) {
  return wwrGraphUpload(exec, stream);
}

} // namespace wwr::extension
