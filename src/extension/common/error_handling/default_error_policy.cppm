/**
 * @file default_error_policy.cppm
 * @brief Default error policy implementation
 *
 * Usage:
 *   import wwr.extension.common;
 *   using namespace wwr::extension;
 *
 * The default policy body is a customization point -- a build may replace it
 * without editing this file; see the note on DefaultErrorPolicy below.
 */

module;

// Customization point. Defining WWR_DEFAULT_ERROR_POLICY_IMPL to a header path
// (e.g. via target_compile_definitions) makes DefaultErrorPolicy derive from
// that header's wwr::extension::DefaultErrorPolicyImpl<T> instead of the
// built-in body below -- see the note on DefaultErrorPolicy. The include lives
// in the global module fragment so the header is an ordinary header with its
// own #includes, not a module unit.
#ifdef WWR_DEFAULT_ERROR_POLICY_IMPL
#include WWR_DEFAULT_ERROR_POLICY_IMPL
#endif

export module wwr.extension.common.error_handling:default_error_policy;

import std;
import :error_code;

export namespace wwr::extension {

// ============================================================================
// Default Error Policy
// ============================================================================

/**
 * @brief Default error policy implementation
 *
 * @tparam T The error code type
 *
 * @note Prints error information to std::cerr when an error occurs, then aborts.
 *
 * The body is a customization point. A build that defines
 * WWR_DEFAULT_ERROR_POLICY_IMPL to a header providing
 * `wwr::extension::DefaultErrorPolicyImpl<T>` gets that type as the base in
 * place of the built-in body. DefaultErrorPolicy stays the public name, so
 * every default template argument, explicit instantiation, and concept
 * conformance is unaffected by the swap. The replacement's handle_error MUST be
 * noexcept -- the destruction-slot policies require nothrow_error_policy.
 */
#ifdef WWR_DEFAULT_ERROR_POLICY_IMPL
template<typename T>
class DefaultErrorPolicy : public DefaultErrorPolicyImpl<T> {};
#else
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
#endif

} // namespace wwr::extension
