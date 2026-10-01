/**
 * @file abort_policy.cppm
 * @brief kit::AbortPolicy -- the opt-in abort-on-error policy
 *
 * The extension layer's *core* ships no error policy and forces none: every
 * gpu_check call and every RAII wrapper names the policy it wants, and nothing
 * defaults to or instantiates a policy on the consumer's behalf (that is the
 * invariant #93/#97 established, and it still holds). This is not that -- it is
 * an *opt-in* member of the kit (wwr::extension::kit): a consumer reaches for it
 * by name, out of a namespace they deliberately pulled in, when they want the
 * common "this must not fail" behaviour instead of writing their own policy. The
 * core never names it, so the no-forced-policy invariant is intact; the kit just
 * stops every consumer from re-typing the same abort policy.
 *
 * It lives as a partition of wwr.extension.common (the universally-imported
 * foundation) so `import wwr.extension.common;` reaches it, and because it depends
 * only on the error_handling primitives (success_code / error_name / error_string).
 *
 * handle_error is noexcept, so it is usable in the destruction slots
 * (nothrow_error_policy) as well as the create/device slots.
 *
 * Usage:
 *   import wwr.extension.common;
 *   using namespace wwr::extension;
 *
 *   kit::AbortPolicy<wwrError_t> policy;   // or name it in a wrapper's policy slot
 */

export module wwr.extension.common:abort_policy;

import wwr.extension.error_handling; // success_code, error_name, error_string
import std;

export namespace wwr::extension::kit {

/**
 * @brief Error policy that prints to stderr and aborts on any non-success code
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

} // namespace wwr::extension::kit
