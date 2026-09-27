/**
 * @file interface.cppm
 * @brief Primary interface for gpumod.extension.sparse
 *
 * The error-handling and RAII-handle layer for GPU sparse linear algebra
 * (cuSPARSE or hipSPARSE, per WWR_GPU_BACKEND). The type-safe dispatch
 * wrappers built on top of it live separately in gpumod.wrappers.sparse. It
 * aggregates:
 * - :sparse_error - Error code specializations for gpusparseStatus_t
 * - :sparse_handle - RAII wrapper for gpusparseHandle_t
 * - :convenience_sparse - Default-policy aliases (GpusparseHandle, GpusparseHandleView)
 *
 * Usage:
 *   import gpumod.extension.sparse;
 *   using namespace wwr::extension;
 */

export module gpumod.extension.sparse;

import std;

// Re-export the vendor sparse module: gpusparseHandle_t is the return type of
// GpusparseHandle::get() and its conversion operator, so a consumer can name it
// without importing gpumod.sparse separately.
export import gpumod.sparse;
export import :sparse_error;
export import :sparse_handle;
export import :convenience_sparse;
