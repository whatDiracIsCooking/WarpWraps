/**
 * @file fft_error.cppm
 * @brief GPU FFT (cuFFT / hipFFT) error code specializations
 *
 * Provides specializations of error handling templates for wwrfftResult_t.
 */

export module wwr.extension.fft:fft_error;

import wwr.fft;
import wwr.extension.common;
import std;

export namespace wwr::extension {

// ============================================================================
// Success Code Specialization
// ============================================================================

/**
 * @brief Specialization for wwrfftResult_t
 *
 * @return WWRFFT_SUCCESS
 */
template<>
constexpr wwrfftResult_t success_code<wwrfftResult_t>() noexcept {
  return WWRFFT_SUCCESS;
}

// Vendor success enumerators are always 0. Pin that contract: a wrong-enumerator
// typo would silently invert gpu_check's success/failure, and nothing else catches it.
static_assert(std::to_underlying(success_code<wwrfftResult_t>()) == 0);

// ============================================================================
// Error Name Specialization
// ============================================================================

/**
 * @brief Specialization for wwrfftResult_t
 *
 * @param error The GPU FFT error code
 * @return The error name string (e.g., "CUFFT_INVALID_PLAN" or "HIPFFT_INVALID_PLAN")
 */
template<>
const char *error_name<wwrfftResult_t>(wwrfftResult_t error) noexcept {
  return wwrfftGetStatusName(error);
}

// ============================================================================
// Error String Specialization
// ============================================================================

/**
 * @brief Specialization for wwrfftResult_t
 *
 * @param error The GPU FFT error code
 * @return The error description string (e.g., "the plan handle is invalid")
 */
template<>
const char *error_string<wwrfftResult_t>(wwrfftResult_t error) noexcept {
  return wwrfftGetStatusString(error);
}

// ============================================================================
// Template Instantiations
// ============================================================================

// Explicitly instantiate AbortPolicy for wwrfftResult_t
template class AbortPolicy<wwrfftResult_t>;

// Explicitly instantiate gpu_check for wwrfftResult_t
template bool gpu_check<wwrfftResult_t, AbortPolicy<wwrfftResult_t> &>(
    const wwrfftResult_t error, AbortPolicy<wwrfftResult_t> &policy,
    std::source_location location);

} // namespace wwr::extension
