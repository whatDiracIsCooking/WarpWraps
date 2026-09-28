/**
 * @file gpu_error.cppm
 * @brief GPU runtime API error code specializations (wwrError_t)
 *
 * Provides the success_code / error_name / error_string specializations for
 * wwrError_t and the gpu_check<wwrError_t> instantiation. These live in common,
 * not the runtime extension module, because :device_bound_handle selects and
 * records the owning device (wwrSetDevice / wwrGetDevice, both wwrError_t) for
 * every device-bound handle -- including the library handles (blas, solver,
 * sparse, fft) that link common but not the runtime module. Keeping them here
 * makes the device-bound base self-sufficient for all of its users.
 */

export module wwr.extension.common:gpu_error;

import wwr.extension.error_handling;
import wwr.runtime_api;
import std;

export namespace wwr::extension {

// ============================================================================
// Success Code Specialization
// ============================================================================

/**
 * @brief Specialization for wwrError_t
 *
 * @return wwrSuccess
 */
template<>
wwrError_t success_code<wwrError_t>() noexcept {
  return wwrSuccess;
}

// ============================================================================
// Error Name Specialization
// ============================================================================

/**
 * @brief Specialization for wwrError_t
 *
 * @param error The GPU error code
 * @return The error name string (e.g., "cudaErrorMemoryAllocation" or "hipErrorOutOfMemory")
 */
template<>
const char *error_name<wwrError_t>(wwrError_t error) noexcept {
  return wwrGetErrorName(error);
}

// ============================================================================
// Error String Specialization
// ============================================================================

/**
 * @brief Specialization for wwrError_t
 *
 * @param error The GPU error code
 * @return The error description string (e.g., "out of memory")
 */
template<>
const char *error_string<wwrError_t>(wwrError_t error) noexcept {
  return wwrGetErrorString(error);
}

// ============================================================================
// Template Instantiations
// ============================================================================

// Explicitly instantiate AbortPolicy for wwrError_t
template class AbortPolicy<wwrError_t>;

// Explicitly instantiate gpu_check for wwrError_t
template bool gpu_check<wwrError_t>(const wwrError_t error, std::source_location location);

template bool gpu_check<wwrError_t, AbortPolicy<wwrError_t> &>(
    const wwrError_t error, AbortPolicy<wwrError_t> &policy, std::source_location location);

} // namespace wwr::extension
