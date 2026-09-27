/**
 * @file interface.cppm
 * @brief Primary interface for wwr.extension.blas
 *
 * The error-handling and RAII-handle layer for GPU BLAS (cuBLAS or hipBLAS,
 * per WWR_GPU_BACKEND). The type-safe dispatch wrappers built on top of it
 * live separately in wwr.wrappers.blas. It aggregates:
 * - :blas_error - Error code specializations for gpublasStatus_t
 * - :blas_handle - RAII wrapper for gpublasHandle_t
 * - :convenience_blas - Default-policy aliases (GpublasHandle, GpublasHandleView)
 *
 * Usage:
 *   import wwr.extension.blas;
 *   using namespace wwr::extension;
 */

export module wwr.extension.blas;

import std;

// Re-export the vendor BLAS module: gpublasHandle_t is the return type of
// GpublasHandle::get() and its conversion operator, so a consumer of this
// module can name it without importing wwr.blas separately.
export import wwr.blas;
export import :blas_error;
export import :blas_handle;
export import :convenience_blas;
