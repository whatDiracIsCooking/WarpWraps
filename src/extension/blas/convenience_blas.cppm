/**
 * @file convenience_blas.cppm
 * @brief Default-policy aliases for the GPU BLAS handle wrapper
 *
 * `GpublasHandle` / `GpublasHandleView` bind GpublasHandleWrapper to the default
 * error policy. This partition is the curated home for those default bindings;
 * alternative-policy aliases (e.g. a throwing or logging policy) belong here too.
 */

export module gpumod.extension.blas:convenience_blas;

import :blas_handle;
import gpumod.blas;
import gpumod.extension.common;
import gpumod.extension.handle;

export namespace wwr::extension {

/**
 * @brief Convenient alias for GpublasHandleWrapper with default error policies
 *
 * Usage:
 *   GpublasHandle handle;  // Instead of GpublasHandleWrapper<>
 */
using GpublasHandle = GpublasHandleWrapper<>;

/// @brief Non-owning, copyable view of a BLAS handle, carrying its device index.
///        Returned by GpublasHandle::view(); converts to gpublasHandle_t for the
///        gpublas* wrappers, so a borrowed handle can be used without owning it.
using GpublasHandleView = DeviceBoundHandleView<gpublasHandle_t>;

} // namespace wwr::extension
