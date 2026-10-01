/**
 * @file abort_policy.cppm
 * @brief Test-side alias to the shipped kit::AbortPolicy
 *
 * The shipped library now carries an opt-in abort-on-error policy,
 * wwr::extension::kit::AbortPolicy (the kit's batteries-included policy; the
 * error-handling *core* still forces none). The test suites want exactly that
 * "this must not fail" behaviour, so this is now just a thin alias of the kit
 * policy into the test namespace -- the test files that spell `AbortPolicy` keep
 * working unchanged, and the test side no longer owns a separate copy.
 *
 * Usage:
 *   import wwr.test.shared.abort_policy;
 *   using namespace wwr::extension::test;
 */

export module wwr.test.shared.abort_policy;

export import wwr.extension.common; // kit::AbortPolicy

export namespace wwr::extension::test {

/// Alias to the shipped kit policy, in the test namespace the suites already use.
template<typename T>
using AbortPolicy = ::wwr::extension::kit::AbortPolicy<T>;

} // namespace wwr::extension::test
