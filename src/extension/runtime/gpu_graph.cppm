/**
 * @file gpu_graph.cppm
 * @brief RAII wrapper for GPU graph handles
 *
 * Provides GpuGraph, an RAII wrapper for gpuGraph_t -- the mutable DAG that is
 * instantiated into an executable graph (GpuGraphExec).
 */

export module wwr.extension.runtime:gpu_graph;

import :gpu_graph_exec;
import wwr.runtime_api;
import wwr.extension.common;
import wwr.extension.handle;
import std;

export namespace wwr::extension {

/**
 * @brief RAII wrapper for a GPU graph
 *
 * Creates an empty graph on construction and destroys it on destruction.
 * Supports move semantics for transferring ownership.
 *
 * @note A graph is NOT device-bound, so this sits on BaseHandle rather than
 *       DeviceBoundHandle: gpuGraphCreate takes no device, and a graph is a
 *       description of work whose nodes may target different devices. There is
 *       no device index to record. See device_bound_handle.cppm.
 *
 * @tparam P_create Error policy type for creation (defaults to DefaultErrorPolicy<gpuError_t>)
 * @tparam P_destroy Error policy type for destruction (defaults to P_create)
 *
 * @note P_destroy MUST NOT THROW - it is called from the destructor.
 */
template<error_policy<gpuError_t> P_create = DefaultErrorPolicy<gpuError_t>,
         nothrow_error_policy<gpuError_t> P_destroy = P_create>
class GpuGraphWrapper
    : public BaseHandle<gpuGraph_t, GpuGraphWrapper<P_create, P_destroy>, P_create, P_destroy> {
private:
  using Base = BaseHandle<gpuGraph_t, GpuGraphWrapper<P_create, P_destroy>, P_create, P_destroy>;

protected:
  // Construct without creating a handle; used by adopt() below.
  GpuGraphWrapper(typename Base::skip_default_create_t tag) noexcept : Base(tag) {}

public:
  // The default/policy constructors, inherited from BaseHandle; each routes
  // through create() below to build an empty graph.
  using BaseHandle<gpuGraph_t, GpuGraphWrapper<P_create, P_destroy>, P_create,
                      P_destroy>::BaseHandle;

  /// @brief Take ownership of an already-created raw graph handle.
  ///
  /// The handle is not created here -- it is one the runtime produced, e.g. by
  /// gpuStreamEndCapture. The returned wrapper owns it and destroys it with
  /// gpuGraphDestroy like any other GpuGraph. Used by GpuStream::end_capture.
  static GpuGraphWrapper adopt(gpuGraph_t raw) noexcept {
    GpuGraphWrapper graph{typename Base::skip_default_create_t{}};
    graph.handle_ = raw;
    return graph;
  }

  /// @brief Create an empty graph
  /// @param handle Output parameter for the created graph
  /// @param location Source location where creation was requested
  void create(gpuGraph_t *handle, std::source_location location) {
    // gpuGraphCreate's flags parameter is reserved and must be 0.
    gpu_check(gpuGraphCreate(handle, 0), this->policy_create_, location);
  }

  /// @brief Instantiate this graph into an executable graph
  /// @param flags Instantiation flags (0 for none)
  /// @param location Source location where instantiation was requested
  /// @return A GpuGraphExec owning the instantiated executable graph
  GpuGraphExecWrapper<P_create, P_destroy>
  instantiate(const unsigned long long flags = 0,
              std::source_location location = std::source_location::current()) {
    return GpuGraphExecWrapper<P_create, P_destroy>(this->handle_, flags, location);
  }

  /// @brief Destroy the graph
  /// @param handle The graph to destroy
  void destroy(gpuGraph_t handle) {
    if (handle != nullptr) {
      gpu_check(gpuGraphDestroy(handle), this->policy_destroy_);
    }
  }
};

} // namespace wwr::extension
