/**
 * @file default_error_policy.cppm
 * @brief Default error policy implementation
 *
 * Usage:
 *   import gpumod.extension.common;
 *   using namespace gpumod::extension;
 */

export module gpumod.extension.common:default_error_policy;

import std;
import :error_code;
import :error_policy;

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
class DefaultErrorPolicy : public BaseErrorPolicy<T> {
public:
  /**
     * @brief Handle an error by printing to stderr
     *
     * @param error The error code to handle
     * @param location Source location where the error occurred
     */
  void handle_error(const T error, std::source_location location) override {
    if (error != success_code<T>()) {
      std::print(std::cerr, "GPU error at {}:{} in {}: {} ({})\n", location.file_name(),
                 location.line(), location.function_name(), error_name(error), error_string(error));
      std::abort();
    }
  }
};

} // namespace gpumod::extension
