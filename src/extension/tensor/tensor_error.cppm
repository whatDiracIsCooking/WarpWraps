/**
 * @file tensor_error.cppm
 * @brief GPU tensor (cuTENSOR / hipTensor) error code specializations
 *
 * Provides specializations of the error-handling templates for wwrtensorStatus_t.
 *
 * Unlike cuFFT/cuBLAS (a separate GetStatusName / GetErrorName), cuTENSOR and
 * hipTensor expose ONE string function -- wwrtensorGetErrorString -- so both the
 * error_name and error_string specializations route to it. The names therefore
 * carry the same description text, not an enumerator spelling; there is no
 * vendor function that yields the bare "CUTENSOR_STATUS_*" name.
 */

export module wwr.extension.tensor:tensor_error;

import wwr.tensor;
import wwr.extension.common;
import std;

export namespace wwr::extension {

// ============================================================================
// Success Code Specialization
// ============================================================================

/**
 * @brief Specialization for wwrtensorStatus_t
 *
 * @return WWRTENSOR_STATUS_SUCCESS
 */
template<>
constexpr wwrtensorStatus_t success_code<wwrtensorStatus_t>() noexcept {
  return WWRTENSOR_STATUS_SUCCESS;
}

// Vendor success enumerators are always 0. Pin that contract: a wrong-enumerator
// typo would silently invert gpu_check's success/failure, and nothing else catches it.
static_assert(std::to_underlying(success_code<wwrtensorStatus_t>()) == 0);

// ============================================================================
// Error Name Specialization
// ============================================================================

/**
 * @brief Specialization for wwrtensorStatus_t
 *
 * @param error The GPU tensor error code
 * @return A human-readable description of the error
 *
 * @note cuTENSOR/hipTensor expose only wwrtensorGetErrorString (no name/spelling
 *       function), so this returns the same description text as error_string.
 */
template<>
const char *error_name<wwrtensorStatus_t>(wwrtensorStatus_t error) noexcept {
  return wwrtensorGetErrorString(error);
}

// ============================================================================
// Error String Specialization
// ============================================================================

/**
 * @brief Specialization for wwrtensorStatus_t
 *
 * @param error The GPU tensor error code
 * @return The error description string (e.g., "the operation is not supported")
 */
template<>
const char *error_string<wwrtensorStatus_t>(wwrtensorStatus_t error) noexcept {
  return wwrtensorGetErrorString(error);
}

} // namespace wwr::extension
