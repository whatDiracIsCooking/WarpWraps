/**
 * @file gpu_graph_exec.cppm
 * @brief RAII wrapper for GPU executable graph handles
 *
 * Provides GpuGraphExec, an RAII wrapper for gpuGraphExec_t -- the executable
 * graph produced by instantiating a gpuGraph_t.
 */

export module gpumod.extension.runtime:gpu_graph_exec;

import gpumod.runtime_api;
import gpumod.extension.common;
import std;

export namespace gpumod::extension {

/**
 * @brief Borrow-safe executable-graph operations, shared by the owner and view
 *
 * CRTP mixin keyed on Derived::get(): launch/upload forward to the borrowed
 * gpuGraphExec_t and touch no ownership state, so they are correct for both
 * GpuGraphExecWrapper (owns the exec) and GpuGraphExecView (borrows it).
 *
 * Methods are const: they mutate the GPU exec, not the C++ object.
 */
template<typename Derived>
class GpuGraphExecAccess {
private:
  const Derived &self() const noexcept { return static_cast<const Derived &>(*this); }

public:
  /// @brief Launch the executable graph on a stream
  gpuError_t launch(gpuStream_t stream) const { return gpuGraphLaunch(self().get(), stream); }

  /// @brief Upload the executable graph to a stream's device without launching it
  gpuError_t upload(gpuStream_t stream) const { return gpuGraphUpload(self().get(), stream); }
};

/**
 * @brief Non-owning, copyable view over a GPU executable graph
 *
 * Carries the borrowed handle (via GpuHandleView; an exec is not device-bound)
 * and the borrow-safe operations (via GpuGraphExecAccess). Construct one from an
 * owning GpuGraphExec with `.view()`, or from a raw gpuGraphExec_t. It destroys
 * nothing, so it must not outlive the exec it borrows.
 */
class GpuGraphExecView : public GpuHandleView<gpuGraphExec_t>,
                         public GpuGraphExecAccess<GpuGraphExecView> {
public:
  using GpuHandleView<gpuGraphExec_t>::GpuHandleView;
};

/**
 * @brief RAII wrapper for a GPU executable graph
 *
 * Instantiates a graph template into an executable graph on construction and
 * destroys it on destruction. Supports move semantics for transferring
 * ownership.
 *
 * @note An executable graph is NOT device-bound, so this sits on BaseGpuHandle
 *       rather than GpuBoundHandle: gpuGraphInstantiate* takes no device, a
 *       graph may span multiple devices, and the exec runs on whatever device
 *       the stream passed to launch() belongs to. There is no device index to
 *       record. See device_bound_handle.cppm.
 *
 * @tparam P_create Error policy type for creation (defaults to DefaultErrorPolicy<gpuError_t>)
 * @tparam P_destroy Error policy type for destruction (defaults to P_create)
 *
 * @note P_destroy MUST NOT THROW - it is called from the destructor.
 */
template<error_policy<gpuError_t> P_create = DefaultErrorPolicy<gpuError_t>,
         error_policy<gpuError_t> P_destroy = P_create>
class GpuGraphExecWrapper
    : public BaseGpuHandle<gpuGraphExec_t, GpuGraphExecWrapper<P_create, P_destroy>, P_create,
                           P_destroy>,
      public GpuGraphExecAccess<GpuGraphExecWrapper<P_create, P_destroy>> {
private:
  using Base =
      BaseGpuHandle<gpuGraphExec_t, GpuGraphExecWrapper<P_create, P_destroy>, P_create, P_destroy>;

public:
  /// @brief Instantiate an executable graph from a graph template
  /// @param graph The source graph to instantiate (not owned; used only for this call)
  /// @param flags Instantiation flags (0 for none)
  /// @param location Source location where instantiation was requested
  explicit GpuGraphExecWrapper(gpuGraph_t graph, const unsigned long long flags = 0,
                               std::source_location location = std::source_location::current())
      : Base(typename Base::skip_default_create_t{}) {
    gpu_check(gpuGraphInstantiate(&this->handle_, graph, flags), this->policy_create_, location);
  }

  // launch()/upload() come from GpuGraphExecAccess, shared with GpuGraphExecView.

  /// @brief A non-owning, copyable view of this executable graph
  ///
  /// Deleted on rvalues so a view cannot be taken from a temporary exec, which
  /// would dangle immediately.
  GpuGraphExecView view() const & noexcept { return GpuGraphExecView{this->get()}; }
  GpuGraphExecView view() && = delete;

  /// @brief Destroy the executable graph
  /// @param handle The executable graph to destroy
  void destroy(gpuGraphExec_t handle) {
    if (handle != nullptr) {
      gpu_check(gpuGraphExecDestroy(handle), this->policy_destroy_);
    }
  }
};

/**
 * @brief Convenient alias for GpuGraphExecWrapper with default error policies
 *
 * Usage:
 *   GpuGraphExec exec{graph};  // Instead of GpuGraphExecWrapper<>{graph}
 */
using GpuGraphExec = GpuGraphExecWrapper<>;

} // namespace gpumod::extension
