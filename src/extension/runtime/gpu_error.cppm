/**
 * @file gpu_error.cppm
 * @brief GPU runtime API error code specializations
 *
 * Provides specializations of error handling templates for gpuError_t.
 */

export module gpumod.extension.runtime:gpu_error;

import gpumod.runtime_api;
import gpumod.extension.common;
import std;

export namespace gpumod::extension {

// ============================================================================
// Success Code Specialization
// ============================================================================

/**
 * @brief Specialization for gpuError_t
 *
 * @return gpuSuccess
 */
template<>
gpuError_t success_code<gpuError_t>() noexcept {
  return gpuSuccess;
}

// ============================================================================
// Error Name Specialization
// ============================================================================

/**
 * @brief Specialization for gpuError_t
 *
 * @param error The GPU error code
 * @return The error name string (e.g., "cudaErrorMemoryAllocation" or "hipErrorOutOfMemory")
 */
template<>
const char *error_name<gpuError_t>(gpuError_t error) noexcept {
  return gpuGetErrorName(error);
}

// ============================================================================
// Error String Specialization
// ============================================================================

/**
 * @brief Specialization for gpuError_t
 *
 * @param error The GPU error code
 * @return The error description string (e.g., "out of memory")
 */
template<>
const char *error_string<gpuError_t>(gpuError_t error) noexcept {
  return gpuGetErrorString(error);
}

// ============================================================================
// Template Instantiations
// ============================================================================

// Explicitly instantiate DefaultErrorPolicy for gpuError_t
template class DefaultErrorPolicy<gpuError_t>;

// Explicitly instantiate gpu_check for gpuError_t
template bool gpu_check<gpuError_t>(const gpuError_t error, std::source_location location);

template bool gpu_check<gpuError_t, DefaultErrorPolicy<gpuError_t>>(
    const gpuError_t error, DefaultErrorPolicy<gpuError_t> &policy, std::source_location location);

} // namespace gpumod::extension
