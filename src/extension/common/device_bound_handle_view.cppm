/**
 * @file device_bound_handle_view.cppm
 * @brief Non-owning view over a handle that also carries its owning device
 *
 * Provides GpuBoundHandleView, the copyable twin of GpuBoundHandle: a
 * GpuHandleView that additionally records the device index the borrowed handle
 * belongs to. A bare gpuStream_t/gpuEvent_t has lost that information; a view
 * keeps it so a borrowed handle can still answer dev_idx().
 *
 * Usage:
 *   import gpumod.extension.common;
 *
 *   class GpuEventView : public GpuBoundHandleView<gpuEvent_t>, ... { ... };
 */

export module gpumod.extension.common:device_bound_handle_view;

import :gpu_handle_view;
import std;

export namespace gpumod::extension {

/// @brief Non-owning view over a device-bound handle, carrying its device index
/// @tparam T The underlying GPU handle type (e.g., gpuEvent_t)
///
/// Mirrors GpuBoundHandle's role over BaseGpuHandle: it adds the recorded device
/// index and nothing else. Unlike the owner, the view never queries the runtime
/// -- the device is whatever the owner recorded (or -1 when constructed from a
/// raw handle whose device is unknown).
template<typename T>
class GpuBoundHandleView : public GpuHandleView<T> {
protected:
  int dev_idx_ = -1; ///< Index of the device the borrowed handle belongs to (-1 if unknown)

public:
  GpuBoundHandleView() noexcept = default;

  /// @brief View a raw handle whose owning device is unknown (dev_idx() == -1)
  explicit GpuBoundHandleView(T handle) noexcept : GpuHandleView<T>(handle) {}

  /// @brief View a handle known to belong to `dev_idx`
  GpuBoundHandleView(T handle, int dev_idx) noexcept
      : GpuHandleView<T>(handle), dev_idx_(dev_idx) {}

  /// @brief Index of the physical device the borrowed handle belongs to (-1 if unknown)
  int dev_idx() const noexcept { return dev_idx_; }
};

} // namespace gpumod::extension
