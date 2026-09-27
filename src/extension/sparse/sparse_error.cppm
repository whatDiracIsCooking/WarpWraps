/**
 * @file sparse_error.cppm
 * @brief GPU sparse (cuSPARSE / hipSPARSE) error code specializations
 *
 * Provides specializations of error handling templates for gpusparseStatus_t.
 */

export module wwr.extension.sparse:sparse_error;

import wwr.sparse;
import wwr.extension.common;
import std;

export namespace wwr::extension {

// ============================================================================
// Success Code Specialization
// ============================================================================

/**
 * @brief Specialization for gpusparseStatus_t
 *
 * @return GPUSPARSE_STATUS_SUCCESS
 */
template<>
constexpr gpusparseStatus_t success_code<gpusparseStatus_t>() noexcept {
  return GPUSPARSE_STATUS_SUCCESS;
}

// Every vendor status enum uses 0 for success (CUSPARSE_STATUS_SUCCESS,
// HIPSPARSE_STATUS_SUCCESS, ...). Pin the specialization to that contract,
// independently of which named enumerator it returns: a wrong-enumerator typo
// would make gpu_check treat every success as a failure (abort) or every failure
// as success, and nothing else here would catch it.
static_assert(std::to_underlying(success_code<gpusparseStatus_t>()) == 0);

// ============================================================================
// Error Name Specialization
// ============================================================================

/**
 * @brief Specialization for gpusparseStatus_t
 *
 * @param error The GPU sparse error code
 * @return The error name string (e.g., "CUSPARSE_STATUS_NOT_INITIALIZED" or "HIPSPARSE_STATUS_NOT_INITIALIZED")
 */
template<>
const char *error_name<gpusparseStatus_t>(gpusparseStatus_t error) noexcept {
  return gpusparseGetErrorName(error);
}

// ============================================================================
// Error String Specialization
// ============================================================================

/**
 * @brief Specialization for gpusparseStatus_t
 *
 * @param error The GPU sparse error code
 * @return The error description string (e.g., "the library was not initialized")
 */
template<>
const char *error_string<gpusparseStatus_t>(gpusparseStatus_t error) noexcept {
  return gpusparseGetErrorString(error);
}

// ============================================================================
// Template Instantiations
// ============================================================================

// Explicitly instantiate DefaultErrorPolicy for gpusparseStatus_t
template class DefaultErrorPolicy<gpusparseStatus_t>;

// Explicitly instantiate gpu_check for gpusparseStatus_t
template bool gpu_check<gpusparseStatus_t>(const gpusparseStatus_t error,
                                           std::source_location location);

template bool gpu_check<gpusparseStatus_t, DefaultErrorPolicy<gpusparseStatus_t>>(
    const gpusparseStatus_t error, DefaultErrorPolicy<gpusparseStatus_t> &policy,
    std::source_location location);

} // namespace wwr::extension
