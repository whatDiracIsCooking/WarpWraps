/**
 * @file comp_error.cppm
 * @brief GPU compression (nvCOMP / hipCOMP) error code specializations
 *
 * Provides the success_code / error_name / error_string specializations for
 * wwrcompStatus_t, so a batched-LLIF status rides the same gpu_check / error_policy
 * path as every other wwr error type.
 *
 * Unlike every sibling (fft_error, the common wwrError_t specializations) this
 * one does NOT forward to a vendor status->string function, because neither
 * backend exposes one through wwr.comp: nvCOMP/hipCOMP ship no
 * nvcompGetStatusString counterpart to cudaGetErrorString, and wwr.comp carries
 * no such shim. So error_name / error_string are a hand-written switch over the
 * six WWRCOMP_* codes wwr.comp defines (the nvCOMP 2.2 / hipCOMP intersection),
 * returning string literals; an unrecognised value maps to "unknown".
 */

export module wwr.extension.comp:comp_error;

import wwr.comp;
import wwr.extension.common;
import std;

export namespace wwr::extension {

// ============================================================================
// Success Code Specialization
// ============================================================================

/**
 * @brief Specialization for wwrcompStatus_t
 *
 * @return WWRCOMP_SUCCESS
 */
template<>
constexpr wwrcompStatus_t success_code<wwrcompStatus_t>() noexcept {
  return WWRCOMP_SUCCESS;
}

// Vendor success enumerators are always 0. Pin that contract: a wrong-enumerator
// typo would silently invert gpu_check's success/failure, and nothing else catches it.
static_assert(std::to_underlying(success_code<wwrcompStatus_t>()) == 0);

// ============================================================================
// Error Name Specialization
// ============================================================================

/**
 * @brief Specialization for wwrcompStatus_t
 *
 * @param error The GPU compression status code
 * @return The enumerator name (e.g. "WWRCOMP_ERROR_INVALID_VALUE"), or "unknown"
 *
 * @note Hand-written, not a vendor forward: wwr.comp exposes no status->string
 *       function. The switch covers exactly the six codes wwr.comp defines.
 */
template<>
const char *error_name<wwrcompStatus_t>(wwrcompStatus_t error) noexcept {
  switch (error) {
  case WWRCOMP_SUCCESS:
    return "WWRCOMP_SUCCESS";
  case WWRCOMP_ERROR_INVALID_VALUE:
    return "WWRCOMP_ERROR_INVALID_VALUE";
  case WWRCOMP_ERROR_NOT_SUPPORTED:
    return "WWRCOMP_ERROR_NOT_SUPPORTED";
  case WWRCOMP_ERROR_CANNOT_DECOMPRESS:
    return "WWRCOMP_ERROR_CANNOT_DECOMPRESS";
  case WWRCOMP_ERROR_CUDA_ERROR:
    return "WWRCOMP_ERROR_CUDA_ERROR";
  case WWRCOMP_ERROR_INTERNAL:
    return "WWRCOMP_ERROR_INTERNAL";
  default:
    return "unknown";
  }
}

// ============================================================================
// Error String Specialization
// ============================================================================

/**
 * @brief Specialization for wwrcompStatus_t
 *
 * @param error The GPU compression status code
 * @return A human-readable description, or "unknown compression status"
 *
 * @note Hand-written for the same reason as error_name above.
 */
template<>
const char *error_string<wwrcompStatus_t>(wwrcompStatus_t error) noexcept {
  switch (error) {
  case WWRCOMP_SUCCESS:
    return "success";
  case WWRCOMP_ERROR_INVALID_VALUE:
    return "invalid value";
  case WWRCOMP_ERROR_NOT_SUPPORTED:
    return "operation not supported";
  case WWRCOMP_ERROR_CANNOT_DECOMPRESS:
    return "cannot decompress";
  case WWRCOMP_ERROR_CUDA_ERROR:
    return "underlying CUDA/HIP runtime error";
  case WWRCOMP_ERROR_INTERNAL:
    return "internal error";
  default:
    return "unknown compression status";
  }
}

} // namespace wwr::extension
