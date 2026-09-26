/**
 * @file gpu_check.cppm
 * @brief GPU error checking for gpumod project
 *
 * Provides simple no-throw error checking for any error code type with a
 * success_code<T>() specialization (gpuError_t, library status codes, ...).
 * Hands failures to an error policy and returns bool indicating success.
 *
 * Usage:
 *   import gpumod.extension.common;
 *
 *   if (!gpu_check(gpuStreamCreate(&stream))) {
 *       return;  // handle error
 *   }
 */

export module gpumod.extension.common:gpu_check;

import :error_code;
import :error_policy;
import :default_error_policy;
import std;

// ============================================================================
// Error Checking Functions
// ============================================================================
export namespace gpumod::extension {

/**
 * @brief Check an error code with the default error policy
 *
 * @param error The error code to check
 * @param location Source location (automatically captured)
 * @return true if error == success code, false otherwise
 *
 * @example
 *   if (!gpu_check(gpuStreamCreate(&stream))) {
 *       // Handle error (will print to stderr and abort)
 *       return;
 *   }
 */
template<typename T>
bool gpu_check(const T error, std::source_location location = std::source_location::current()) {
  if (error != success_code<T>()) {
    DefaultErrorPolicy<T> policy;
    policy.handle_error(error, location);
    return false;
  }
  return true;
}

/**
 * @brief Check an error code with a custom error policy
 *
 * @tparam T The error code type
 * @tparam ErrorPolicy The error policy type (must satisfy error_policy<T>)
 * @param error The error code to check
 * @param policy The error policy to use for handling errors
 * @param location Source location (automatically captured)
 * @return true if error == success code, false otherwise
 *
 * @example
 *   CustomErrorPolicy<gpuError_t> policy;
 *   if (!gpu_check(gpuStreamCreate(&stream), policy)) {
 *       // Handle error using custom policy
 *       return;
 *   }
 */
template<typename T, error_policy<T> ErrorPolicy>
bool gpu_check(const T error, ErrorPolicy &policy,
               std::source_location location = std::source_location::current()) {
  if (error != success_code<T>()) {
    policy.handle_error(error, location);
    return false;
  }
  return true;
}

} // namespace gpumod::extension
