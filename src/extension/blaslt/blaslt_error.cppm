/**
 * @file blaslt_error.cppm
 * @brief Error-code policy for cuBLASLt / hipBLASLt -- reused from wwr.extension.blas
 *
 * cuBLASLt has no status type of its own: wwrblasLtStatus_t is defined (in
 * src/blaslt.cppm) as cublasStatus_t / hipblasStatus_t -- the EXACT SAME
 * underlying type as wwrblasStatus_t (src/blas.cppm), and cuBLASLt reuses
 * cuBLAS's GetStatusName / GetStatusString. C++ explicit specializations key on
 * the underlying type, not the alias name, so wwr.extension.blas's :blas_error
 * partition already provides success_code / error_name / error_string for this
 * exact type.
 *
 * Defining our own would therefore be a DUPLICATE explicit specialization of an
 * identical type -- an ODR violation that breaks the build wherever both modules
 * are visible (a TU importing both, or the installed-package consume). So this
 * partition adds nothing: it re-exports wwr.extension.blas, re-exposing those
 * specializations for wwrblasLtStatus_t. The blaslt library LINK_PUBLICs
 * wwr::extension::blas to back the import.
 */

export module wwr.extension.blaslt:blaslt_error;

// Re-export the blas error policy: it already specializes success_code /
// error_name / error_string for wwrblasStatus_t, which IS wwrblasLtStatus_t, so
// gpu_check<wwrblasLtStatus_t> resolves through these. Do NOT add specializations
// here -- they would duplicate blas's for the identical type.
export import wwr.extension.blas;
