/**
 * @file stream_bound_handle_view.cppm
 * @brief Non-owning view over a handle that also carries its bound work stream
 *
 * Provides StreamBoundHandleView, the copyable twin of StreamBoundHandle: a
 * DeviceBoundHandleView that additionally records the work stream the borrowed
 * handle is bound to. A bare library handle (wwrblasHandle_t and the like) has
 * lost that stream -- and FFT's vendor surface offers no …GetStream to recover
 * it -- so a view keeps it, letting a borrowed handle still answer stream() and
 * thereby model a device_handle_stream.
 *
 * Usage:
 *   import wwr.extension.handle;
 *
 *   using BlasHandleView = StreamBoundHandleView<wwrblasHandle_t>;
 */

export module wwr.extension.handle:stream_bound_handle_view;

import :device_bound_handle_view;
import wwr.runtime_api; // wwrStream_t
import std;

export namespace wwr::extension {

/// @brief Non-owning view over a stream-bound handle, carrying its work stream
/// @tparam T The underlying GPU handle type (e.g., wwrblasHandle_t)
///
/// Mirrors StreamBoundHandle's role over DeviceBoundHandle: it adds the bound
/// work stream and nothing else. Where the owner reads its stream back through a
/// retained owner (owner_->stream()), the view holds a bare wwrStream_t captured
/// at view() time -- it is non-owning, so it retains nothing and, like any view,
/// must not outlive what backs the borrowed handle or its stream. Exposing both
/// dev_idx() (inherited) and stream() makes the view a device_handle_stream, so a
/// DeviceBuffer can draw async allocations on a borrowed handle's stream.
template<typename T>
class StreamBoundHandleView : public DeviceBoundHandleView<T> {
protected:
  wwrStream_t stream_ = nullptr; ///< The work stream the borrowed handle is bound to (nullptr if unknown)

public:
  StreamBoundHandleView() noexcept = default;

  /// @brief View a raw handle whose device and stream are both unknown
  ///        (dev_idx() == -1, stream() == nullptr)
  explicit StreamBoundHandleView(T handle) noexcept : DeviceBoundHandleView<T>(handle) {}

  /// @brief View a handle known to belong to `dev_idx` and be bound to `stream`
  StreamBoundHandleView(T handle, int dev_idx, wwrStream_t stream) noexcept
      : DeviceBoundHandleView<T>(handle, dev_idx), stream_(stream) {}

  /// @brief The work stream the borrowed handle is bound to (nullptr if unknown)
  wwrStream_t stream() const noexcept { return stream_; }
};

} // namespace wwr::extension
