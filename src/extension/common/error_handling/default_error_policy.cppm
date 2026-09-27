/**
 * @file default_error_policy.cppm
 * @brief Default error policy implementation
 *
 * Usage:
 *   import gpumod.extension.common;
 *   using namespace gpumod::extension;
 */

export module gpumod.extension.common.error_handling:default_error_policy;

import std;
import :error_code;

export namespace gpumod::extension {

// ============================================================================
// Default Error Policy
// ============================================================================

/**
 * @brief Default error policy implementation
 *
 * @tparam T The error code type
 *
 * @note Prints error information to std::cerr when an error occurs
 */
template<typename T>
class DefaultErrorPolicy {
public:
  /// @brief The error code type this policy handles (see typed_error_policy).
  using error_type = T;

  /**
     * @brief Handle an error by printing to stderr
     *
     * @param error The error code to handle
     * @param location Source location where the error occurred
     *
     * @note noexcept: usable as a destruction-slot (nothrow_error_policy) policy.
     */
  void handle_error(const T error, std::source_location location) noexcept {
    if (error != success_code<T>()) {
      std::print(std::cerr, "GPU error at {}:{} in {}: {} ({})\n", location.file_name(),
                 location.line(), location.function_name(), error_name(error), error_string(error));
      std::abort();
    }
  }
};

} // namespace gpumod::extension
