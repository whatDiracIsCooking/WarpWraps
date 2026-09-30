/**
 * @file interface.cppm
 * @brief Primary interface for wwr.extension.blas
 *
 * The error-handling and RAII-handle layer for GPU BLAS (cuBLAS or hipBLAS,
 * per WWR_GPU_BACKEND). The type-safe dispatch wrappers built on top of it
 * live separately in wwr.wrappers.blas. It aggregates:
 * - :blas_error - Error code specializations for wwrblasStatus_t
 * - :blas_handle - RAII wrapper for wwrblasHandle_t
 * - :scoped_pointer_mode - RAII guard toggling a handle's pointer mode
 *
 * Usage:
 *   import wwr.extension.blas;
 *   using namespace wwr::extension;
 */

export module wwr.extension.blas;

import std;

// Re-export the vendor BLAS module: wwrblasHandle_t is the return type of
// BlasHandleWrapper::get() and its conversion operator, so a consumer of this
// module can name it without importing wwr.blas separately.
export import wwr.blas;
export import :blas_error;
export import :blas_handle;
export import :scoped_pointer_mode;
