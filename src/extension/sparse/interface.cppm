/**
 * @file interface.cppm
 * @brief Primary interface for wwr.extension.sparse
 *
 * The error-handling and RAII-handle layer for GPU sparse linear algebra
 * (cuSPARSE or hipSPARSE, per WWR_GPU_BACKEND). The type-safe dispatch
 * wrappers built on top of it live separately in wwr.wrappers.sparse. It
 * aggregates:
 * - :sparse_error - Error code specializations for wwrsparseStatus_t
 * - :sparse_handle - RAII wrapper for wwrsparseHandle_t
 *
 * Usage:
 *   import wwr.extension.sparse;
 *   using namespace wwr::extension;
 */

export module wwr.extension.sparse;

import std;

// Re-export the vendor sparse module: wwrsparseHandle_t is the return type of
// SparseHandleWrapper::get() and its conversion operator, so a consumer can name it
// without importing wwr.sparse separately.
export import wwr.sparse;
export import :sparse_error;
export import :sparse_handle;
