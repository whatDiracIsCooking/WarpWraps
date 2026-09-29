/**
 * @file gpu_check.cppm
 * @brief GPU error checking for wwr project
 *
 * Provides simple no-throw error checking for any error code type with a
 * success_code<T>() specialization (wwrError_t, library status codes, ...).
 * Hands failures to an error policy and returns bool indicating success.
 *
 * Usage:
 *   import wwr.extension.common;
 *
 *   if (!gpu_check(wwrStreamCreate(&stream), MyPolicy{})) {
 *       return;  // handle error
 *   }
 */

export module wwr.extension.error_handling:gpu_check;

import :error_code;
import :error_policy;
import std;

// ============================================================================
// Error Checking Functions
// ============================================================================
export namespace wwr::extension {

/**
 * @brief Check an error code, handing any failure to an error policy
 *
 * @tparam T The error code type
 * @tparam ErrorPolicy The error policy type (must satisfy error_policy<T>)
 * @param error The error code to check
 * @param policy The error policy to use for handling errors, taken by forwarding
 *        reference: a stateless policy temporary binds (the natural spelling
 *        gpu_check(code, MyPolicy{})), while a named policy is passed through and
 *        mutated in place -- handle_error records into the caller's own instance,
 *        which the buffer/handle wrappers and the counting-policy tests rely on
 * @param location Source location (automatically captured)
 * @return true if error == success code, false otherwise
 *
 * @example
 *   if (!gpu_check(wwrStreamCreate(&stream), CustomErrorPolicy<wwrError_t>{})) {
 *       // Handle error using custom policy
 *       return;
 *   }
 */
template<typename T, typename ErrorPolicy>
  requires error_policy<std::remove_cvref_t<ErrorPolicy>, T>
bool gpu_check(const T error, ErrorPolicy &&policy,
               std::source_location location = std::source_location::current()) {
  if (error != success_code<T>()) {
    policy.handle_error(error, location);
    return false;
  }
  return true;
}

} // namespace wwr::extension
