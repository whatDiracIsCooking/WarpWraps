/**
 * @file interface.cppm
 * @brief Primary interface for wwr.extension.error_handling
 *
 * Backend-neutral error-handling primitives, shared by the whole extension
 * layer. Imports only std -- it knows nothing of wwrError_t or any library
 * status type; those specialize its templates from their own modules
 * (wwr.extension.common:gpu_error for wwrError_t, the blas/solver/fft/sparse
 * error modules for the library status codes). It aggregates:
 * - :error_code - success_code / error_name / error_string templates + the error_type concept
 * - :error_policy - the error_policy / nothrow_error_policy / typed_error_policy concepts
 * - :gpu_check - no-throw error checks routed through an error policy
 *
 * Usage:
 *   import wwr.extension.error_handling;
 *   using namespace wwr::extension;
 *
 * wwr.extension.common re-exports this module, so
 * `import wwr.extension.common;` also brings these names in.
 */

export module wwr.extension.error_handling;

export import :error_code;
export import :error_policy;
export import :gpu_check;
