/**
 * @file interface.cppm
 * @brief Primary interface for gpumod.extension.blas
 *
 * The error-handling and RAII-handle layer for GPU BLAS (cuBLAS or hipBLAS,
 * per GPUMOD_GPU_BACKEND). The type-safe dispatch wrappers built on top of it
 * live separately in gpumod.wrappers.blas. It aggregates:
 * - :blas_error - Error code specializations for gpublasStatus_t
 * - :blas_handle - RAII wrapper for gpublasHandle_t
 *
 * Usage:
 *   import gpumod.extension.blas;
 *   using namespace gpumod::extension;
 */

export module gpumod.extension.blas;

import std;

// Re-export the vendor BLAS module: gpublasHandle_t is the return type of
// GpublasHandle::get() and its conversion operator, so a consumer of this
// module can name it without importing gpumod.blas separately.
export import gpumod.blas;
export import :blas_error;
export import :blas_handle;
