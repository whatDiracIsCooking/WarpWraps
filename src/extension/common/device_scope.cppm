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
// gpu_check here uses the default AbortPolicy<wwrError_t>, which odr-uses
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
 * restores the recorded device on destruction. Each runtime call goes through
 * gpu_check with the default (abort-on-failure) policy -- wwrGetDevice/
 * wwrSetDevice are wwrError_t-returning and carry [[nodiscard]] on some
 * backends, so the return cannot simply be dropped.
 *
 * There is deliberately no error-policy hook. A device switch fails only when
 * the runtime context is already unusable, at which point aborting is the only
 * sane response; routing that through a caller's (typically allocation) policy
 * would defend a near-unreachable failure with the wrong tool.
 */
struct DeviceScope : private NonCopyable {
  int original_idx = -1; ///< Device current at construction, restored on destruction

  /// @brief Switch to `target_idx`, recording the previous device to restore
  explicit DeviceScope(const int target_idx,
                       std::source_location location = std::source_location::current()) {
    gpu_check(wwrGetDevice(&original_idx), location);
    gpu_check(wwrSetDevice(target_idx), location);
  }

  ~DeviceScope() { gpu_check(wwrSetDevice(original_idx)); }

  // Copy operations are implicitly deleted via the NonCopyable base. The
  // user-declared destructor suppresses the implicit moves, so the guard stays
  // non-movable as well.
};

} // namespace wwr::extension
