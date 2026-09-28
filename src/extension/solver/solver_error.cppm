/**
 * @file solver_error.cppm
 * @brief GPU solver (cuSOLVER Dense / hipSOLVER Dense) error code specializations
 *
 * Provides specializations of error handling templates for wwrsolverStatus_t.
 */

export module wwr.extension.solver:solver_error;

import wwr.solver;
import wwr.extension.common;
import std;

export namespace wwr::extension {

// ============================================================================
// Success Code Specialization
// ============================================================================

/**
 * @brief Specialization for wwrsolverStatus_t
 *
 * @return WWRSOLVER_STATUS_SUCCESS
 */
template<>
constexpr wwrsolverStatus_t success_code<wwrsolverStatus_t>() noexcept {
  return WWRSOLVER_STATUS_SUCCESS;
}

// Vendor success enumerators are always 0. Pin that contract: a wrong-enumerator
// typo would silently invert gpu_check's success/failure, and nothing else catches it.
static_assert(std::to_underlying(success_code<wwrsolverStatus_t>()) == 0);

// ============================================================================
// Error Name Specialization
// ============================================================================

/**
 * @brief Specialization for wwrsolverStatus_t
 *
 * @param error The GPU solver error code
 * @return The error name string (e.g., "CUSOLVER_STATUS_NOT_INITIALIZED" or "HIPSOLVER_STATUS_NOT_INITIALIZED")
 */
template<>
const char *error_name<wwrsolverStatus_t>(wwrsolverStatus_t error) noexcept {
  return wwrsolverGetStatusName(error);
}

// ============================================================================
// Error String Specialization
// ============================================================================

/**
 * @brief Specialization for wwrsolverStatus_t
 *
 * @param error The GPU solver error code
 * @return The error description string (e.g., "the library was not initialized")
 */
template<>
const char *error_string<wwrsolverStatus_t>(wwrsolverStatus_t error) noexcept {
  return wwrsolverGetStatusString(error);
}

// ============================================================================
// Template Instantiations
// ============================================================================

// Explicitly instantiate DefaultErrorPolicy for wwrsolverStatus_t
template class DefaultErrorPolicy<wwrsolverStatus_t>;

// Explicitly instantiate gpu_check for wwrsolverStatus_t
template bool gpu_check<wwrsolverStatus_t>(const wwrsolverStatus_t error,
                                           std::source_location location);

template bool gpu_check<wwrsolverStatus_t, DefaultErrorPolicy<wwrsolverStatus_t>>(
    const wwrsolverStatus_t error, DefaultErrorPolicy<wwrsolverStatus_t> &policy,
    std::source_location location);

} // namespace wwr::extension
