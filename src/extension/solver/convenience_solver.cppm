/**
 * @file convenience_solver.cppm
 * @brief Default-policy aliases for the GPU solver handle / params wrappers
 *
 * `WwrsolverDnHandle` / `WwrsolverDnHandleView` / `WwrsolverDnParams` bind the
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
 * @brief Convenient alias for WwrsolverDnHandleWrapper with default error policies
 *
 * Usage:
 *   WwrsolverDnHandle handle;  // Instead of WwrsolverDnHandleWrapper<>
 */
using WwrsolverDnHandle = WwrsolverDnHandleWrapper<>;

/// @brief Non-owning, copyable view of a solver handle, carrying its device index.
///        Returned by WwrsolverDnHandle::view(); converts to wwrsolverDnHandle_t
///        for the wwrsolverDn* wrappers, so a borrowed handle can be used without
///        owning it.
using WwrsolverDnHandleView = DeviceBoundHandleView<wwrsolverDnHandle_t>;

/**
 * @brief Convenient alias for WwrsolverDnParamsWrapper with default error policies
 *
 * Usage:
 *   WwrsolverDnParams params;  // Instead of WwrsolverDnParamsWrapper<>
 */
using WwrsolverDnParams = WwrsolverDnParamsWrapper<>;

} // namespace wwr::extension
