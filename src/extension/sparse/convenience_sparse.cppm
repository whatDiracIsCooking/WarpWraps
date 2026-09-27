/**
 * @file convenience_sparse.cppm
 * @brief Default-policy aliases for the GPU sparse handle wrapper
 *
 * `GpusparseHandle` / `GpusparseHandleView` bind GpusparseHandleWrapper to the
 * default error policy. This partition is the curated home for those default
 * bindings; alternative-policy aliases belong here too.
 */

export module gpumod.extension.sparse:convenience_sparse;

import :sparse_handle;
import gpumod.sparse;
import gpumod.extension.common;
import gpumod.extension.handle;

export namespace gpumod::extension {

/**
 * @brief Convenient alias for GpusparseHandleWrapper with default error policies
 *
 * Usage:
 *   GpusparseHandle handle;  // Instead of GpusparseHandleWrapper<>
 */
using GpusparseHandle = GpusparseHandleWrapper<>;

/// @brief Non-owning, copyable view of a sparse handle, carrying its device index.
///        Returned by GpusparseHandle::view(); converts to gpusparseHandle_t for
///        the gpusparse* wrappers, so a borrowed handle can be used without owning it.
using GpusparseHandleView = DeviceBoundHandleView<gpusparseHandle_t>;

} // namespace gpumod::extension
