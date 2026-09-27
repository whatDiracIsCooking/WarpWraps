/**
 * @file convenience_sparse.cppm
 * @brief Default-policy aliases for the GPU sparse handle wrapper
 *
 * `WwrsparseHandle` / `WwrsparseHandleView` bind WwrsparseHandleWrapper to the
 * default error policy. This partition is the curated home for those default
 * bindings; alternative-policy aliases belong here too.
 */

export module wwr.extension.sparse:convenience_sparse;

import :sparse_handle;
import wwr.sparse;
import wwr.extension.common;
import wwr.extension.handle;

export namespace wwr::extension {

/**
 * @brief Convenient alias for WwrsparseHandleWrapper with default error policies
 *
 * Usage:
 *   WwrsparseHandle handle;  // Instead of WwrsparseHandleWrapper<>
 */
using WwrsparseHandle = WwrsparseHandleWrapper<>;

/// @brief Non-owning, copyable view of a sparse handle, carrying its device index.
///        Returned by WwrsparseHandle::view(); converts to wwrsparseHandle_t for
///        the wwrsparse* wrappers, so a borrowed handle can be used without owning it.
using WwrsparseHandleView = DeviceBoundHandleView<wwrsparseHandle_t>;

} // namespace wwr::extension
