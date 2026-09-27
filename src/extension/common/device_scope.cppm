/**
 * @file device_scope.cppm
 * @brief RAII guard that makes a device current and restores the previous one
 *
 * Usage:
 *   import gpumod.extension.common;
 *
 *   {
 *     DeviceScope scope{target_idx};  // target_idx is now current
 *     ...                             // work on target_idx
 *   }                                 // previous device restored
 */

export module gpumod.extension.common:device_scope;

import gpumod.extension.common.error_handling;
// The default policy DefaultErrorPolicy<gpuError_t> is concept-checked on this
// template's default argument, which instantiates its vtable and thus its
// virtual handle_error -- and that odr-uses success_code<gpuError_t>(). That
// specialization lives in :gpu_error; without it reachable here the compiler
// falls back to the inline-but-undefined primary template (-Wundefined-inline,
// and an ill-formed implicit instantiation). Acyclic: :gpu_error imports none
// of :device_scope's chain.
import :gpu_error;
import gpumod.runtime_api;
import std;

export namespace gpumod::extension {

/**
 * @brief Make `target_idx` the current device for the guard's lifetime
 *
 * Records the device current at construction, switches to `target_idx`, and
 * restores the recorded device on destruction. Each runtime call goes through
 * gpu_check against the guard's error policy P -- gpuGetDevice/gpuSetDevice are
 * gpuError_t-returning and carry [[nodiscard]] on some backends, so the return
 * cannot simply be dropped.
 *
 * P defaults to DefaultErrorPolicy<gpuError_t> (aborts on failure), matching
 * the rest of this module. Pass an existing policy by reference (the
 * two-argument constructor) to thread a caller's own policy through the guard
 * instead: the guard borrows that instance rather than copying it, so a
 * stateful policy observes the guard's device switch in the same object the
 * caller reads back -- this is how DeviceBuffer routes the guard's device
 * selection through its allocation / deallocation policy. With no policy
 * argument the guard owns a default-constructed P.
 *
 * The borrowed policy (and, in the destructor, the device restore) must not
 * outlive the referenced policy; a guard is a short-lived scope object, so in
 * practice the policy is a caller stack/member that plainly does.
 */
template<error_policy<gpuError_t> P = DefaultErrorPolicy<gpuError_t>>
struct DeviceScopeWrapper {
  int original_idx = -1; ///< Device current at construction, restored on destruction

  /// @brief Switch to `target_idx`, reporting failures through an owned default P
  explicit DeviceScopeWrapper(const int target_idx,
                              std::source_location location = std::source_location::current())
    requires std::default_initializable<P>
      : owned_(std::in_place), policy_(*owned_) {
    enter(target_idx, location);
  }

  /// @brief Switch to `target_idx`, reporting failures through `policy`
  ///
  /// The guard borrows `policy` (which must outlive it) rather than copying it,
  /// so a stateful policy records the guard's calls in place -- this is how a
  /// caller threads its own error policy through the device switch.
  DeviceScopeWrapper(const int target_idx, P &policy,
                     std::source_location location = std::source_location::current())
      : policy_(policy) {
    enter(target_idx, location);
  }

  ~DeviceScopeWrapper() { gpu_check(gpuSetDevice(original_idx), policy_); }

  DeviceScopeWrapper(const DeviceScopeWrapper &) = delete;
  DeviceScopeWrapper &operator=(const DeviceScopeWrapper &) = delete;

private:
  /// @brief Record the current device and make `target_idx` current
  void enter(const int target_idx, std::source_location location) {
    gpu_check(gpuGetDevice(&original_idx), policy_, location);
    gpu_check(gpuSetDevice(target_idx), policy_, location);
  }

  std::optional<P> owned_; ///< Engaged only when the guard owns its policy
  P &policy_;              ///< The policy every runtime call is checked against
};

/// @brief DeviceScope with the default (abort-on-failure) error policy
using DeviceScope = DeviceScopeWrapper<>;

} // namespace gpumod::extension
