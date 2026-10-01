/**
 * @file interface.cppm
 * @brief Primary interface for wwr.extension.blaslt
 *
 * The error-handling and RAII layer for the modern cuBLASLt / hipBLASLt matmul
 * surface (per WWR_GPU_BACKEND). cuBLASLt uses runtime datatypes and descriptors
 * rather than typed entry points, so there is no layer-2 generic wrapper -- the
 * value here is RAII over the descriptor zoo (the library handle, the matmul
 * descriptor, matrix layouts, and the matmul preference). It aggregates:
 * - :blaslt_error  - Reuses wwr.extension.blas's wwrblasStatus_t specializations
 *                    (wwrblasLtStatus_t IS cublasStatus_t / hipblasStatus_t)
 * - :blaslt_handle - RAII wrapper for wwrblasLtHandle_t
 * - :blaslt_matmul - RAII wrappers for the matmul descriptor / matrix layout /
 *                    matmul preference
 *
 * Usage:
 *   import wwr.extension.blaslt;
 *   using namespace wwr::extension;
 */

export module wwr.extension.blaslt;

import std;

// Re-export the vendor cuBLASLt module: wwrblasLtHandle_t is the return type of
// BlasLtHandleWrapper::get() and its conversion operator, and the descriptor
// handle types are the matmul wrappers' get() types, so a consumer of this
// module can name them without importing wwr.blaslt separately.
export import wwr.blaslt;
export import :blaslt_error;
export import :blaslt_handle;
export import :blaslt_matmul;
