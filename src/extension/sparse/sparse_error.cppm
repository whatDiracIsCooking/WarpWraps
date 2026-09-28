/**
 * @file sparse_error.cppm
 * @brief GPU sparse (cuSPARSE / hipSPARSE) error code specializations
 *
 * Provides specializations of error handling templates for wwrsparseStatus_t.
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
 * @brief Specialization for wwrsparseStatus_t
 *
 * @return WWRSPARSE_STATUS_SUCCESS
 */
template<>
constexpr wwrsparseStatus_t success_code<wwrsparseStatus_t>() noexcept {
  return WWRSPARSE_STATUS_SUCCESS;
}

// Vendor success enumerators are always 0. Pin that contract: a wrong-enumerator
// typo would silently invert gpu_check's success/failure, and nothing else catches it.
static_assert(std::to_underlying(success_code<wwrsparseStatus_t>()) == 0);

// ============================================================================
// Error Name Specialization
// ============================================================================

/**
 * @brief Specialization for wwrsparseStatus_t
 *
 * @param error The GPU sparse error code
 * @return The error name string (e.g., "CUSPARSE_STATUS_NOT_INITIALIZED" or "HIPSPARSE_STATUS_NOT_INITIALIZED")
 */
template<>
const char *error_name<wwrsparseStatus_t>(wwrsparseStatus_t error) noexcept {
  return wwrsparseGetErrorName(error);
}

// ============================================================================
// Error String Specialization
// ============================================================================

/**
 * @brief Specialization for wwrsparseStatus_t
 *
 * @param error The GPU sparse error code
 * @return The error description string (e.g., "the library was not initialized")
 */
template<>
const char *error_string<wwrsparseStatus_t>(wwrsparseStatus_t error) noexcept {
  return wwrsparseGetErrorString(error);
}

// ============================================================================
// Template Instantiations
// ============================================================================

// Explicitly instantiate DefaultErrorPolicy for wwrsparseStatus_t
template class DefaultErrorPolicy<wwrsparseStatus_t>;

// Explicitly instantiate gpu_check for wwrsparseStatus_t
template bool gpu_check<wwrsparseStatus_t>(const wwrsparseStatus_t error,
                                           std::source_location location);

template bool gpu_check<wwrsparseStatus_t, DefaultErrorPolicy<wwrsparseStatus_t> &>(
    const wwrsparseStatus_t error, DefaultErrorPolicy<wwrsparseStatus_t> &policy,
    std::source_location location);

} // namespace wwr::extension
