/**
 * @file fft_error.cppm
 * @brief GPU FFT (cuFFT / hipFFT) error code specializations
 *
 * Provides specializations of error handling templates for gpufftResult_t.
 */

export module gpumod.extension.fft:fft_error;

import gpumod.fft;
import gpumod.extension.common;
import std;

export namespace wwr::extension {

// ============================================================================
// Success Code Specialization
// ============================================================================

/**
 * @brief Specialization for gpufftResult_t
 *
 * @return GPUFFT_SUCCESS
 */
template<>
constexpr gpufftResult_t success_code<gpufftResult_t>() noexcept {
  return GPUFFT_SUCCESS;
}

// Every vendor status enum uses 0 for success (CUFFT_SUCCESS, HIPFFT_SUCCESS).
// Pin the specialization to that contract, independently of which named
// enumerator it returns: a wrong-enumerator typo would make gpu_check treat every
// success as a failure (abort) or every failure as success, and nothing else here
// would catch it.
static_assert(std::to_underlying(success_code<gpufftResult_t>()) == 0);

// ============================================================================
// Error Name Specialization
// ============================================================================

/**
 * @brief Specialization for gpufftResult_t
 *
 * @param error The GPU FFT error code
 * @return The error name string (e.g., "CUFFT_INVALID_PLAN" or "HIPFFT_INVALID_PLAN")
 */
template<>
const char *error_name<gpufftResult_t>(gpufftResult_t error) noexcept {
  return gpufftGetStatusName(error);
}

// ============================================================================
// Error String Specialization
// ============================================================================

/**
 * @brief Specialization for gpufftResult_t
 *
 * @param error The GPU FFT error code
 * @return The error description string (e.g., "the plan handle is invalid")
 */
template<>
const char *error_string<gpufftResult_t>(gpufftResult_t error) noexcept {
  return gpufftGetStatusString(error);
}

// ============================================================================
// Template Instantiations
// ============================================================================

// Explicitly instantiate DefaultErrorPolicy for gpufftResult_t
template class DefaultErrorPolicy<gpufftResult_t>;

// Explicitly instantiate gpu_check for gpufftResult_t
template bool gpu_check<gpufftResult_t>(const gpufftResult_t error, std::source_location location);

template bool gpu_check<gpufftResult_t, DefaultErrorPolicy<gpufftResult_t>>(
    const gpufftResult_t error, DefaultErrorPolicy<gpufftResult_t> &policy,
    std::source_location location);

} // namespace wwr::extension
