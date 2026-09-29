/**
 * @file device_scope.cppm
 * @brief RAII guard that makes a device current and restores the previous one
 *
 * Usage:
 *   import wwr.extension.common;
 *
 *   {
 *     DeviceScope scope{target_idx};  // target_idx is now current
 *     ...                             // work on target_idx
 *   }                                 // previous device restored
 */

export module wwr.extension.common:device_scope;

import wwr.extension.error_handling;
// The AbortPolicy<wwrError_t> path -- the default P_device_access, and the
// explicit restore policy in the destructor below -- odr-uses
// success_code<wwrError_t>() (and error_name/error_string). That specialization
// lives in :gpu_error; without it reachable the compiler falls back to the
// inline-but-undefined primary template (-Wundefined-inline, and an ill-formed
// implicit instantiation). Acyclic: :gpu_error imports none of :device_scope's
// chain.
import :gpu_error;
import :noncopyable;
import wwr.runtime_api;
import std;

export namespace wwr::extension {

/**
 * @brief Make `target_idx` the current device for the guard's lifetime
 *
 * Records the device current at construction, switches to `target_idx`, and
 * restores the recorded device on destruction. wwrGetDevice/wwrSetDevice are
 * wwrError_t-returning and carry [[nodiscard]] on some backends, so the return
 * cannot simply be dropped -- each goes through gpu_check.
 *
 * The two calls that *enter* the scope (record the original, switch to the
 * target) route through P_device_access, so device-selection error handling is
 * under the caller's control (defaulting to abort). This mirrors
 * DeviceBoundHandle's P_device_access, and like it stays the plain
 * error_policy: a caller may hand in a throwing policy to propagate a bad-index
 * failure out of the constructor as an exception.
 *
 * The restore on destruction deliberately does *not* use that policy: it runs
 * from the destructor, where a throwing policy would std::terminate, and a
 * failed restore means the runtime context is already unusable -- at which
 * point aborting is the only sane response. So the restore passes an explicit
 * AbortPolicy<wwrError_t> (abort-on-failure) regardless of P_device_access.
 *
 * @tparam P_device_access The error policy for the entering wwrGetDevice/
 *         wwrSetDevice calls; typed to wwrError_t, defaulted to
 *         AbortPolicy<wwrError_t>.
 */
template<error_policy<wwrError_t> P_device_access = AbortPolicy<wwrError_t>>
struct DeviceScope : private NonCopyable {
  int original_idx = -1; ///< Device current at construction, restored on destruction

  /// @brief Switch to `target_idx`, recording the previous device to restore.
  ///        Routes both entering calls through `policy_device`, then retains it
  ///        (so a stateful policy's accumulated state survives, readable via
  ///        device_policy()).
  explicit DeviceScope(const int target_idx, P_device_access policy_device = {},
                       std::source_location location = std::source_location::current())
      : policy_device_(std::move(policy_device)) {
    gpu_check(wwrGetDevice(&original_idx), policy_device_, location);
    gpu_check(wwrSetDevice(target_idx), policy_device_, location);
  }

  ~DeviceScope() { gpu_check(wwrSetDevice(original_idx), AbortPolicy<wwrError_t>{}); }

  // Copy operations are implicitly deleted via the NonCopyable base. The
  // user-declared destructor suppresses the implicit moves, so the guard stays
  // non-movable as well.

  /// @brief The policy handling this guard's device (get/set) calls, for reading
  ///        back any state a stateful device policy accumulated on entry.
  const P_device_access &device_policy() const noexcept { return policy_device_; }

private:
  [[no_unique_address]] P_device_access policy_device_{}; ///< Policy for the entering (get/set) calls
};

} // namespace wwr::extension
