/**
 * @file solver_error.cppm
 * @brief GPU solver (cuSOLVER Dense / hipSOLVER Dense) error code specializations
 *
 * Provides specializations of error handling templates for gpusolverStatus_t.
 */

export module gpumod.extension.solver:solver_error;

import gpumod.solver;
import gpumod.extension.common;
import std;

export namespace gpumod::extension {

// ============================================================================
// Success Code Specialization
// ============================================================================

/**
 * @brief Specialization for gpusolverStatus_t
 *
 * @return GPUSOLVER_STATUS_SUCCESS
 */
template<>
constexpr gpusolverStatus_t success_code<gpusolverStatus_t>() noexcept {
  return GPUSOLVER_STATUS_SUCCESS;
}

// Every vendor status enum uses 0 for success (CUSOLVER_STATUS_SUCCESS,
// HIPSOLVER_STATUS_SUCCESS, ...). Pin the specialization to that contract,
// independently of which named enumerator it returns: a wrong-enumerator typo
// would make gpu_check treat every success as a failure (abort) or every failure
// as success, and nothing else here would catch it.
static_assert(std::to_underlying(success_code<gpusolverStatus_t>()) == 0);

// ============================================================================
// Error Name Specialization
// ============================================================================

/**
 * @brief Specialization for gpusolverStatus_t
 *
 * @param error The GPU solver error code
 * @return The error name string (e.g., "CUSOLVER_STATUS_NOT_INITIALIZED" or "HIPSOLVER_STATUS_NOT_INITIALIZED")
 */
template<>
const char *error_name<gpusolverStatus_t>(gpusolverStatus_t error) noexcept {
  return gpusolverGetStatusName(error);
}

// ============================================================================
// Error String Specialization
// ============================================================================

/**
 * @brief Specialization for gpusolverStatus_t
 *
 * @param error The GPU solver error code
 * @return The error description string (e.g., "the library was not initialized")
 */
template<>
const char *error_string<gpusolverStatus_t>(gpusolverStatus_t error) noexcept {
  return gpusolverGetStatusString(error);
}

// ============================================================================
// Template Instantiations
// ============================================================================

// Explicitly instantiate DefaultErrorPolicy for gpusolverStatus_t
template class DefaultErrorPolicy<gpusolverStatus_t>;

// Explicitly instantiate gpu_check for gpusolverStatus_t
template bool gpu_check<gpusolverStatus_t>(const gpusolverStatus_t error,
                                           std::source_location location);

template bool gpu_check<gpusolverStatus_t, DefaultErrorPolicy<gpusolverStatus_t>>(
    const gpusolverStatus_t error, DefaultErrorPolicy<gpusolverStatus_t> &policy,
    std::source_location location);

} // namespace gpumod::extension
