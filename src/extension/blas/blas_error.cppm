/**
 * @file blas_error.cppm
 * @brief GPU BLAS (cuBLAS / hipBLAS) error code specializations
 *
 * Provides specializations of error handling templates for gpublasStatus_t.
 */

export module wwr.extension.blas:blas_error;

import wwr.blas;
import wwr.extension.common;
import std;

export namespace wwr::extension {

// ============================================================================
// Success Code Specialization
// ============================================================================

/**
 * @brief Specialization for gpublasStatus_t
 *
 * @return GPUBLAS_STATUS_SUCCESS
 */
template<>
constexpr gpublasStatus_t success_code<gpublasStatus_t>() noexcept {
  return GPUBLAS_STATUS_SUCCESS;
}

// Every vendor status enum uses 0 for success (CUBLAS_STATUS_SUCCESS,
// HIPBLAS_STATUS_SUCCESS, ...). Pin the specialization to that contract,
// independently of which named enumerator it returns: a wrong-enumerator typo
// would make gpu_check treat every success as a failure (abort) or every failure
// as success, and nothing else here would catch it.
static_assert(std::to_underlying(success_code<gpublasStatus_t>()) == 0);

// ============================================================================
// Error Name Specialization
// ============================================================================

/**
 * @brief Specialization for gpublasStatus_t
 *
 * @param error The GPU BLAS error code
 * @return The error name string (e.g., "CUBLAS_STATUS_NOT_INITIALIZED"; hipBLAS returns its descriptive string here too)
 */
template<>
const char *error_name<gpublasStatus_t>(gpublasStatus_t error) noexcept {
  return gpublasGetStatusName(error);
}

// ============================================================================
// Error String Specialization
// ============================================================================

/**
 * @brief Specialization for gpublasStatus_t
 *
 * @param error The GPU BLAS error code
 * @return The error description string (e.g., "the library was not initialized")
 */
template<>
const char *error_string<gpublasStatus_t>(gpublasStatus_t error) noexcept {
  return gpublasGetStatusString(error);
}

// ============================================================================
// Template Instantiations
// ============================================================================

// Explicitly instantiate DefaultErrorPolicy for gpublasStatus_t
template class DefaultErrorPolicy<gpublasStatus_t>;

// Explicitly instantiate gpu_check for gpublasStatus_t
template bool gpu_check<gpublasStatus_t>(const gpublasStatus_t error,
                                         std::source_location location);

template bool gpu_check<gpublasStatus_t, DefaultErrorPolicy<gpublasStatus_t>>(
    const gpublasStatus_t error, DefaultErrorPolicy<gpublasStatus_t> &policy,
    std::source_location location);

} // namespace wwr::extension
