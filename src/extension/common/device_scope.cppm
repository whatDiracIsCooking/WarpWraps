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

import gpumod.runtime_api;
import std;

export namespace gpumod::extension {

/**
 * @brief Make `target_idx` the current device for the guard's lifetime
 *
 * Records the device current at construction, switches to `target_idx`, and
 * restores the recorded device on destruction.
 */
struct DeviceScope {
  int original_idx = -1; ///< Device current at construction, restored on destruction

  explicit DeviceScope(const int target_idx) {
    gpuGetDevice(&original_idx);
    gpuSetDevice(target_idx);
  }

  ~DeviceScope() { gpuSetDevice(original_idx); }
};

} // namespace gpumod::extension
