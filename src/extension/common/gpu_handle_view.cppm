/**
 * @file gpu_handle_view.cppm
 * @brief Non-owning, copyable view over a GPU handle
 *
 * Provides GpuHandleView, the copyable twin of BaseGpuHandle: it holds a raw
 * GPU handle but does not own it, so it is freely copyable and movable and
 * destroys nothing. Use it to pass a borrowed handle (a stream/event you do not
 * own, the default stream, an interop handle) while keeping get()/operator T().
 *
 * Usage:
 *   import gpumod.extension.common;
 *
 *   void wait(GpuHandleView<gpuStream_t> s) { gpuStreamSynchronize(s); }
 */

export module gpumod.extension.common:gpu_handle_view;

import std;

export namespace gpumod::extension {

/// @brief Non-owning, copyable view over a GPU handle
/// @tparam T The underlying GPU handle type
///
/// The value semantics are deliberately pointer-like: copying a view copies the
/// borrowed handle, and a view can outlive the owner it was taken from -- it is
/// the caller's responsibility not to use a dangling view, exactly as with a
/// raw pointer or std::string_view. Domain operations (sync, record, ...) are
/// added by inheriting an accessor mixin alongside this base.
template<typename T>
class GpuHandleView {
protected:
  T handle_ = nullptr;

public:
  GpuHandleView() noexcept = default;
  explicit GpuHandleView(T handle) noexcept : handle_(handle) {}

  // Copyable AND movable -- the whole point of a view. All defaulted (trivial).

  T get() const noexcept { return handle_; }
  operator T() const noexcept { return handle_; }
};

} // namespace gpumod::extension
