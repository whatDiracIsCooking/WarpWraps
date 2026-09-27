/**
 * @file device_bound_handle_view.cppm
 * @brief Non-owning view over a handle that also carries its owning device
 *
 * Provides DeviceBoundHandleView, the copyable twin of DeviceBoundHandle: a
 * HandleView that additionally records the device index the borrowed handle
 * belongs to. A bare wwrStream_t/wwrEvent_t has lost that information; a view
 * keeps it so a borrowed handle can still answer dev_idx().
 *
 * Usage:
 *   import wwr.extension.handle;
 *
 *   class GpuEventView : public DeviceBoundHandleView<wwrEvent_t>, ... { ... };
 */

export module wwr.extension.handle:device_bound_handle_view;

import :handle_view;
import std;

export namespace wwr::extension {

/// @brief Non-owning view over a device-bound handle, carrying its device index
/// @tparam T The underlying GPU handle type (e.g., wwrEvent_t)
///
/// Mirrors DeviceBoundHandle's role over BaseHandle: it adds the recorded device
/// index and nothing else. Unlike the owner, the view never queries the runtime
/// -- the device is whatever the owner recorded (or -1 when constructed from a
/// raw handle whose device is unknown).
template<typename T>
class DeviceBoundHandleView : public HandleView<T> {
protected:
  int dev_idx_ = -1; ///< Index of the device the borrowed handle belongs to (-1 if unknown)

public:
  DeviceBoundHandleView() noexcept = default;

  /// @brief View a raw handle whose owning device is unknown (dev_idx() == -1)
  explicit DeviceBoundHandleView(T handle) noexcept : HandleView<T>(handle) {}

  /// @brief View a handle known to belong to `dev_idx`
  DeviceBoundHandleView(T handle, int dev_idx) noexcept
      : HandleView<T>(handle), dev_idx_(dev_idx) {}

  /// @brief Index of the physical device the borrowed handle belongs to (-1 if unknown)
  int dev_idx() const noexcept { return dev_idx_; }
};

} // namespace wwr::extension
