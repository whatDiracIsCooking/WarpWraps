/**
 * @file convenience_solver.cppm
 * @brief Default-policy aliases for the GPU solver handle / params wrappers
 *
 * `GpusolverDnHandle` / `GpusolverDnHandleView` / `GpusolverDnParams` bind the
 * solver wrappers to the default error policy. This partition is the curated home
 * for those default bindings; alternative-policy aliases belong here too.
 */

export module wwr.extension.solver:convenience_solver;

import :solver_handle;
import :solver_params;
import wwr.solver;
import wwr.extension.common;
import wwr.extension.handle;

export namespace wwr::extension {

/**
 * @brief Convenient alias for GpusolverDnHandleWrapper with default error policies
 *
 * Usage:
 *   GpusolverDnHandle handle;  // Instead of GpusolverDnHandleWrapper<>
 */
using GpusolverDnHandle = GpusolverDnHandleWrapper<>;

/// @brief Non-owning, copyable view of a solver handle, carrying its device index.
///        Returned by GpusolverDnHandle::view(); converts to gpusolverDnHandle_t
///        for the gpusolverDn* wrappers, so a borrowed handle can be used without
///        owning it.
using GpusolverDnHandleView = DeviceBoundHandleView<gpusolverDnHandle_t>;

/**
 * @brief Convenient alias for GpusolverDnParamsWrapper with default error policies
 *
 * Usage:
 *   GpusolverDnParams params;  // Instead of GpusolverDnParamsWrapper<>
 */
using GpusolverDnParams = GpusolverDnParamsWrapper<>;

} // namespace wwr::extension
