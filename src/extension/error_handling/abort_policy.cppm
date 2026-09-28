/**
 * @file abort_policy.cppm
 * @brief Abort-on-error policy (the default error policy)
 *
 * Usage:
 *   import wwr.extension.common;
 *   using namespace wwr::extension;
 */

export module wwr.extension.error_handling:abort_policy;

import std;
import :error_code;

export namespace wwr::extension {

// ============================================================================
// Abort Policy
// ============================================================================

/**
 * @brief Error policy that prints to stderr and aborts (the default policy)
 *
 * @tparam T The error code type
 *
 * @note Prints error information to std::cerr when an error occurs, then aborts.
 *
 * AbortPolicy is the name every default-policy slot resolves to (the
 * argless gpu_check and the P_create / P_alloc / P_free / P_destroy
 * template-argument defaults on the RAII wrappers). A build that wants different
 * behaviour passes an explicit policy at the call site instead. handle_error is
 * noexcept because AbortPolicy is also the default for the destruction
 * slots (P_free / P_destroy), which require nothrow_error_policy.
 */
template<typename T>
class AbortPolicy {
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

} // namespace wwr::extension
