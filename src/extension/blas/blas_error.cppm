/**
 * @file blas_error.cppm
 * @brief GPU BLAS (cuBLAS / hipBLAS) error code specializations
 *
 * Provides specializations of error handling templates for wwrblasStatus_t.
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
 * @brief Specialization for wwrblasStatus_t
 *
 * @return WWRBLAS_STATUS_SUCCESS
 */
template<>
constexpr wwrblasStatus_t success_code<wwrblasStatus_t>() noexcept {
  return WWRBLAS_STATUS_SUCCESS;
}

// Vendor success enumerators are always 0. Pin that contract: a wrong-enumerator
// typo would silently invert gpu_check's success/failure, and nothing else catches it.
static_assert(std::to_underlying(success_code<wwrblasStatus_t>()) == 0);

// ============================================================================
// Error Name Specialization
// ============================================================================

/**
 * @brief Specialization for wwrblasStatus_t
 *
 * @param error The GPU BLAS error code
 * @return The error name string (e.g., "CUBLAS_STATUS_NOT_INITIALIZED"; hipBLAS returns its descriptive string here too)
 */
template<>
const char *error_name<wwrblasStatus_t>(wwrblasStatus_t error) noexcept {
  return wwrblasGetStatusName(error);
}

// ============================================================================
// Error String Specialization
// ============================================================================

/**
 * @brief Specialization for wwrblasStatus_t
 *
 * @param error The GPU BLAS error code
 * @return The error description string (e.g., "the library was not initialized")
 */
template<>
const char *error_string<wwrblasStatus_t>(wwrblasStatus_t error) noexcept {
  return wwrblasGetStatusString(error);
}

} // namespace wwr::extension
