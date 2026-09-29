/**
 * @file abort_policy.cppm
 * @brief Test-side abort-on-error policy
 *
 * The shipped library carries no default error policy: every gpu_check call and
 * every RAII wrapper names the policy it wants, and policy choice belongs to the
 * consumer. The test suites want a plain abort-on-failure policy for the common
 * "this must not fail" case, so they own one here rather than reaching back into
 * the library for a default that no longer exists.
 *
 * handle_error is noexcept so this policy is usable in the destruction slots
 * (nothrow_error_policy) as well as the create/device slots.
 *
 * Usage:
 *   import wwr.test.shared.abort_policy;
 *   using namespace wwr::extension::test;
 */

export module wwr.test.shared.abort_policy;

import wwr.extension.common; // success_code, error_name, error_string
import std;

export namespace wwr::extension::test {

/**
 * @brief Error policy that prints to stderr and aborts
 *
 * @tparam T The error code type (any error_type: wwrError_t, a library status, ...)
 */
template<typename T>
class AbortPolicy {
public:
  using error_type = T;
  void handle_error(const T error, std::source_location location) noexcept {
    if (error != success_code<T>()) {
      std::print(std::cerr, "GPU error at {}:{} in {}: {} ({})\n", location.file_name(),
                 location.line(), location.function_name(), error_name(error), error_string(error));
      std::abort();
    }
  }
};

} // namespace wwr::extension::test
