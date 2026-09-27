/**
 * @file convenience_blas.cppm
 * @brief Default-policy aliases for the GPU BLAS handle wrapper
 *
 * `WwrblasHandle` / `WwrblasHandleView` bind WwrblasHandleWrapper to the default
 * error policy. This partition is the curated home for those default bindings;
 * alternative-policy aliases (e.g. a throwing or logging policy) belong here too.
 */

export module wwr.extension.blas:convenience_blas;

import :blas_handle;
import wwr.blas;
import wwr.extension.common;
import wwr.extension.handle;

export namespace wwr::extension {

/**
 * @brief Convenient alias for WwrblasHandleWrapper with default error policies
 *
 * Usage:
 *   WwrblasHandle handle;  // Instead of WwrblasHandleWrapper<>
 */
using WwrblasHandle = WwrblasHandleWrapper<>;

/// @brief Non-owning, copyable view of a BLAS handle, carrying its device index.
///        Returned by WwrblasHandle::view(); converts to wwrblasHandle_t for the
///        wwrblas* wrappers, so a borrowed handle can be used without owning it.
using WwrblasHandleView = DeviceBoundHandleView<wwrblasHandle_t>;

} // namespace wwr::extension
