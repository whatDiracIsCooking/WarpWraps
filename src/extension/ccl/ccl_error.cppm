/**
 * @file ccl_error.cppm
 * @brief GPU collectives (NCCL / RCCL) error code specializations
 *
 * Provides specializations of error handling templates for wwrcclResult_t.
 */

export module wwr.extension.ccl:ccl_error;

import wwr.ccl;
import wwr.extension.common;
import std;

export namespace wwr::extension {

// ============================================================================
// Success Code Specialization
// ============================================================================

/**
 * @brief Specialization for wwrcclResult_t
 *
 * @return WWRCCL_SUCCESS
 */
template<>
constexpr wwrcclResult_t success_code<wwrcclResult_t>() noexcept {
  return WWRCCL_SUCCESS;
}

// Vendor success enumerators are always 0. Pin that contract: a wrong-enumerator
// typo would silently invert gpu_check's success/failure, and nothing else catches it.
static_assert(std::to_underlying(success_code<wwrcclResult_t>()) == 0);

// ============================================================================
// Error Name Specialization
// ============================================================================

/**
 * @brief Specialization for wwrcclResult_t
 *
 * @param error The GPU collectives error code
 * @return The error description string
 *
 * @note NCCL/RCCL expose a single string function (wwrcclGetErrorString), with
 *       no separate error-name query -- so error_name routes to it just as
 *       error_string does below. hipBLAS reuses one function the same way.
 */
template<>
const char *error_name<wwrcclResult_t>(wwrcclResult_t error) noexcept {
  return wwrcclGetErrorString(error);
}

// ============================================================================
// Error String Specialization
// ============================================================================

/**
 * @brief Specialization for wwrcclResult_t
 *
 * @param error The GPU collectives error code
 * @return The error description string (e.g., "invalid argument")
 */
template<>
const char *error_string<wwrcclResult_t>(wwrcclResult_t error) noexcept {
  return wwrcclGetErrorString(error);
}

} // namespace wwr::extension
