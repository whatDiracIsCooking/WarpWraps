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

import :gpu_check;
import gpumod.runtime_api;
import std;

export namespace gpumod::extension {

/**
 * @brief Make `target_idx` the current device for the guard's lifetime
 *
 * Records the device current at construction, switches to `target_idx`, and
 * restores the recorded device on destruction. Each runtime call goes through
 * gpu_check (default policy: aborts on failure), matching the rest of this
 * module -- gpuGetDevice/gpuSetDevice are gpuError_t-returning and carry
 * [[nodiscard]] on some backends, so the return cannot simply be dropped.
 */
struct DeviceScope {
  int original_idx = -1; ///< Device current at construction, restored on destruction

  explicit DeviceScope(const int target_idx) {
    gpu_check(gpuGetDevice(&original_idx));
    gpu_check(gpuSetDevice(target_idx));
  }

  ~DeviceScope() { gpu_check(gpuSetDevice(original_idx)); }
};

} // namespace gpumod::extension
