/**
 * @file recording_policy.cppm
 * @brief Test-side record-and-continue error policy
 *
 * The mirror image of AbortPolicy: where that one aborts on failure, this one
 * stashes the error and returns, so construction of a wrapper whose create()
 * fails completes instead of terminating the process. That is the non-throwing
 * recoverable style the policy surface is meant to allow, and it is what makes
 * BaseHandle::valid() observable -- a wrapper built with this policy over a
 * failing create() is constructed-but-un-owned, and the test asks valid().
 *
 * handle_error is noexcept, so like AbortPolicy this is usable in the
 * destruction slot (nothrow_error_policy) as well as create/device.
 *
 * Usage:
 *   import wwr.test.shared.recording_policy;
 *   using namespace wwr::extension::test;
 */

export module wwr.test.shared.recording_policy;

import wwr.extension.common; // success_code
import std;

export namespace wwr::extension::test {

/**
 * @brief Error policy that records the last error and continues
 *
 * @tparam T The error code type (any error_type: wwrError_t, a library status, ...)
 */
template<typename T>
class RecordingPolicy {
public:
  using error_type = T;
  T last_error = success_code<T>();
  int calls = 0;
  void handle_error(const T error, std::source_location) noexcept {
    ++calls;
    last_error = error; // record and continue -- never throws, never aborts
  }
};

} // namespace wwr::extension::test
