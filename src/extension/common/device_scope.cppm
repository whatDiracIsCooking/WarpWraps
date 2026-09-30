/**
 * @file device_scope.cppm
 * @brief RAII guard that makes a device current and restores the previous one
 *
 * Usage:
 *   import wwr.extension.common;
 *
 *   {
 *     DeviceScope<MyPolicy> scope{target_idx};  // target_idx is now current
 *     ...                                       // work on target_idx
 *   }                                           // previous device restored
 *
 *   // MyPolicy is your own error_policy<wwrError_t>; the library ships none.
 */

export module wwr.extension.common:device_scope;

import wwr.extension.error_handling;
// The constructor and destructor route their gpu_check calls through
// policy_device_, which odr-uses success_code<wwrError_t>(). That specialization
// lives in :error; without it reachable the compiler falls back to the
// inline-but-undefined primary template (-Wundefined-inline, and an ill-formed
// implicit instantiation). Acyclic: :error imports none of :device_scope's
// chain.
import :error;
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
 * Every wwrGetDevice/wwrSetDevice call -- the two that *enter* the scope and the
 * one that restores on destruction -- routes through the same retained
 * P, so device-selection error handling is entirely the caller's,
 * and the library keeps no error policy of its own. This mirrors
 * DeviceBoundHandle's P_device_access; it stays the plain error_policy so a
 * caller may hand in a throwing policy to propagate a bad-index failure out of
 * the constructor as an exception.
 *
 * The restore runs from the destructor, which is noexcept: a policy that
 * *throws* on a failed restore therefore terminates rather than propagates, so
 * a device policy that must survive destruction should be nothrow (one that
 * aborts, say). The buffer wrappers that nest a DeviceScope on their free path
 * already require exactly that (nothrow_error_policy).
 *
 * @tparam P The error policy for the wwrGetDevice/wwrSetDevice
 *         calls; typed to wwrError_t (what those calls return), independent of
 *         any handle's own status type. No default -- the caller names the policy.
 */
template<error_policy<wwrError_t> P>
struct DeviceScope : private NonCopyable {
  int original_idx = -1; ///< Device current at construction, restored on destruction

  /// @brief Switch to `target_idx`, recording the previous device to restore.
  ///        Routes both entering calls through `policy_device`, then retains it
  ///        (so a stateful policy's accumulated state survives, readable via
  ///        device_policy()).
  explicit DeviceScope(const int target_idx, P policy_device = {},
                       std::source_location location = std::source_location::current())
      : policy_device_(std::move(policy_device)) {
    gpu_check(wwrGetDevice(&original_idx), policy_device_, location);
    gpu_check(wwrSetDevice(target_idx), policy_device_, location);
  }

  // Restore symmetrically, through the same policy the entry calls used (see the
  // class note on the noexcept-destructor caveat for throwing policies).
  ~DeviceScope() { gpu_check(wwrSetDevice(original_idx), policy_device_); }

  // Copy operations are implicitly deleted via the NonCopyable base. The
  // user-declared destructor suppresses the implicit moves, so the guard stays
  // non-movable as well.

  /// @brief The policy handling this guard's device (get/set) calls, for reading
  ///        back any state a stateful device policy accumulated on entry.
  const P &device_policy() const noexcept { return policy_device_; }

private:
  [[no_unique_address]] P policy_device_{}; ///< Policy for the entering (get/set) calls
};

} // namespace wwr::extension
