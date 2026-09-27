/**
 * @file handle_view.cppm
 * @brief Non-owning, copyable view over a GPU handle
 *
 * Provides HandleView, the copyable twin of BaseHandle: it holds a raw
 * GPU handle but does not own it, so it is freely copyable and movable and
 * destroys nothing. Use it to pass a borrowed handle (a stream/event you do not
 * own, the default stream, an interop handle) while keeping get()/operator T().
 *
 * Usage:
 *   import wwr.extension.handle;
 *
 *   void wait(HandleView<wwrStream_t> s) { wwrStreamSynchronize(s); }
 */

export module wwr.extension.handle:handle_view;

import std;

export namespace wwr::extension {

/// @brief Non-owning, copyable view over a GPU handle
/// @tparam T The underlying GPU handle type
///
/// The value semantics are deliberately pointer-like: copying a view copies the
/// borrowed handle, and a view can outlive the owner it was taken from -- it is
/// the caller's responsibility not to use a dangling view, exactly as with a
/// raw pointer or std::string_view. Domain operations (sync, record, ...) are
/// added by inheriting an accessor mixin alongside this base.
template<typename T>
class HandleView {
protected:
  T handle_{}; // nullptr for a pointer handle, 0 for an integer one (cufftHandle)

public:
  HandleView() noexcept = default;
  explicit HandleView(T handle) noexcept : handle_(handle) {}

  // Copyable AND movable -- the whole point of a view. All defaulted (trivial).

  T get() const noexcept { return handle_; }
  operator T() const noexcept { return handle_; }
};

} // namespace wwr::extension
