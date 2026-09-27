/**
 * @file gpu_error.cppm
 * @brief GPU runtime API error code specializations (gpuError_t)
 *
 * Provides the success_code / error_name / error_string specializations for
 * gpuError_t and the gpu_check<gpuError_t> instantiation. These live in common,
 * not the runtime extension module, because :device_bound_handle selects and
 * records the owning device (gpuSetDevice / gpuGetDevice, both gpuError_t) for
 * every device-bound handle -- including the library handles (blas, solver,
 * sparse, fft) that link common but not the runtime module. Keeping them here
 * makes the device-bound base self-sufficient for all of its users.
 */

export module gpumod.extension.common:gpu_error;

import gpumod.extension.common.error_handling;
import gpumod.runtime_api;
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
