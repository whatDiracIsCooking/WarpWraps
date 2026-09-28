/**
 * @file default_error_policy.cppm
 * @brief Default error policy implementation
 *
 * Usage:
 *   import wwr.extension.common;
 *   using namespace wwr::extension;
 *
 * The whole partition is a customization point: a build may REPLACE this file
 * without editing any other source. Point WWR_DEFAULT_ERROR_POLICY_MODULE at a
 * .cppm that declares this same partition
 * (`wwr.extension.common.error_handling:default_error_policy`) and exports its
 * own DefaultErrorPolicy<T>; see this directory's CMakeLists and
 * example/custom_default_error_policy. Because the replacement is a real module
 * unit -- not a header pulled into a global module fragment -- its purview may
 * `import` anything (std, the sibling :error_code partition, or a module of your
 * own, named in WWR_DEFAULT_ERROR_POLICY_LINK): imports go after the module
 * declaration, exactly like every other module in this tree.
 */

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
 * DefaultErrorPolicy is the name every default-policy slot resolves to (the
 * argless gpu_check and the P_create / P_alloc / P_free / P_destroy
 * template-argument defaults on the RAII wrappers). A build that swaps this
 * partition file keeps that name, so every default template argument, explicit
 * instantiation, and concept conformance is unaffected by the swap. A
 * replacement's handle_error MUST be noexcept -- the destruction-slot policies
 * require nothrow_error_policy.
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

} // namespace wwr::extension
