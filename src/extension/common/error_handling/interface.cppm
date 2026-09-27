/**
 * @file interface.cppm
 * @brief Primary interface for gpumod.extension.common.error_handling
 *
 * Backend-neutral error-handling primitives, shared by the whole extension
 * layer. Imports only std -- it knows nothing of gpuError_t or any library
 * status type; those specialize its templates from their own modules
 * (gpumod.extension.common:gpu_error for gpuError_t, the blas/solver/fft/sparse
 * error modules for the library status codes). It aggregates:
 * - :error_code - success_code / error_name / error_string templates
 * - :error_policy - the error_policy / nothrow_error_policy / typed_error_policy concepts
 * - :default_error_policy - DefaultErrorPolicy (prints to stderr, aborts)
 * - :gpu_check - no-throw error checks routed through an error policy
 *
 * Usage:
 *   import gpumod.extension.common.error_handling;
 *   using namespace wwr::extension;
 *
 * gpumod.extension.common re-exports this module, so
 * `import gpumod.extension.common;` also brings these names in.
 */

export module gpumod.extension.common.error_handling;

export import :error_code;
export import :error_policy;
export import :default_error_policy;
export import :gpu_check;
