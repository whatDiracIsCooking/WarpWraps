/**
 * @file handle.cppm
 * @brief RAII wrapper for GPU handles
 *
 * Provides generic RAII wrapper for managing GPU handles (streams, events, library handles)
 * with automatic resource cleanup.
 *
 * Usage:
 *   import wwr.extension.handle;
 *
 *   class GpuStreamWrapper : public BaseHandle<wwrStream_t, GpuStreamWrapper, ...> { ... };
 */

export module wwr.extension.handle:handle;

import wwr.extension.common; // BaseErrorPolicy, NonCopyable
import :handle_view;
import std;

export namespace wwr::extension {

// ============================================================================
// RAII Handle Wrapper
// ============================================================================

/// @brief RAII wrapper for GPU handles using CRTP
/// @tparam T The underlying GPU handle type
/// @tparam Derived The derived class type
/// @tparam P_create The error policy for creation; its error_type is the error
///         type this handle checks against
/// @tparam P_destroy The error policy for destruction, constrained to
///         P_create's error type
///
/// @note P_destroy MUST NOT THROW exceptions, as it is invoked from the destructor
///       (a throwing handle_error would std::terminate). The nothrow_error_policy
///       constraint enforces this at compile time.
/// @note The error type is deduced from the policy (P_create::error_type), not
///       from the handle type T: on HIP the vendor handles are all `void*`, so a
///       handle-type -> error-type table cannot tell them apart. The policy
///       always carries its own error type.
/// @note Liveness (does this wrapper own a handle to destroy?) is tracked one of
///       two ways, chosen by the handle type. A pointer handle uses null as the
///       sentinel -- no extra state, so `owns_` is an empty member. A handle type
///       with no reserved invalid value (cufftHandle is a plain `int` on CUDA)
///       has nowhere to encode "empty" in the handle itself, so an explicit bool
///       tracks it. On HIP every vendor handle is a pointer, so the flag only
///       ever materialises for cuFFT.
template<typename T, typename Derived, typed_error_policy P_create,
         nothrow_error_policy<typename P_create::error_type> P_destroy>
class BaseHandle : private NonCopyable {
protected:
  T handle_{};

private:
  static constexpr bool has_null_sentinel = std::is_pointer_v<T>;
  struct no_flag {};
  [[no_unique_address]] std::conditional_t<has_null_sentinel, no_flag, bool> owns_{};

  // Does this wrapper currently own a handle that must be destroyed?
  bool live() const noexcept {
    if constexpr (has_null_sentinel) {
      return handle_ != nullptr;
    } else {
      return owns_;
    }
  }

  // Give up ownership without destroying -- applied to the moved-from source so
  // its destructor becomes a no-op. Only the sentinel is cleared; a non-pointer
  // handle keeps its (now unowned) integer value, exactly as the pointer case
  // keeps nothing observable.
  void release() noexcept {
    if constexpr (has_null_sentinel) {
      handle_ = nullptr;
    } else {
      owns_ = false;
    }
  }

  // Run Derived::create and record whether it produced a live handle. For a
  // pointer handle that is "is it non-null" (create() returns void); for a
  // flagged handle it is create()'s own success bool, which it returns for
  // exactly this purpose.
  void run_create(std::source_location location) {
    auto *derived = static_cast<Derived *>(this);
    if constexpr (has_null_sentinel) {
      derived->create(&handle_, location);
    } else {
      owns_ = derived->create(&handle_, location);
    }
  }

protected:
  [[no_unique_address]] P_create policy_create_{};
  [[no_unique_address]] P_destroy policy_destroy_{};

  // Tag type for derived classes to skip default handle creation
  struct skip_default_create_t {};

  // Protected constructor that skips automatic handle creation
  // Allows derived classes to manually create handles with custom parameters
  BaseHandle(skip_default_create_t) noexcept {}

public:
  BaseHandle(std::source_location location = std::source_location::current()) {
    run_create(location);
  }

  BaseHandle(P_create policy, std::source_location location = std::source_location::current())
      : policy_create_(policy), policy_destroy_(std::move(policy)) {
    run_create(location);
  }

  BaseHandle(P_create policy_create, P_destroy policy_destroy,
                std::source_location location = std::source_location::current())
      : policy_create_(std::move(policy_create)), policy_destroy_(std::move(policy_destroy)) {
    run_create(location);
  }

  ~BaseHandle() {
    if (live()) {
      static_cast<Derived *>(this)->destroy(handle_);
    }
  }

  BaseHandle(BaseHandle &&other) noexcept
      : handle_(other.handle_), owns_(other.owns_),
        policy_create_(std::move(other.policy_create_)),
        policy_destroy_(std::move(other.policy_destroy_)) {
    // Take ownership; leave other holding nothing.
    other.release();
  }

  BaseHandle &operator=(BaseHandle &&other) noexcept {
    // Self-assignment check
    if (this != &other) {
      // Destroy current handle if we own one
      if (live()) {
        static_cast<Derived *>(this)->destroy(handle_);
      }

      // Take ownership from other
      handle_ = other.handle_;
      owns_ = other.owns_;
      policy_create_ = std::move(other.policy_create_);
      policy_destroy_ = std::move(other.policy_destroy_);
      other.release();
    }

    return *this;
  }

  // Copy operations are implicitly deleted via the NonCopyable base.

  operator T() const noexcept { return handle_; }
  T get() const noexcept { return handle_; }

  /// @brief A non-owning, copyable view of this handle.
  ///
  /// Deleted on rvalues so a view cannot be taken from a temporary handle, which
  /// would dangle immediately. Device-bound layers return a richer view that also
  /// carries the device index; borrow-safe operations are free functions.
  HandleView<T> view() const & noexcept { return HandleView<T>{handle_}; }
  HandleView<T> view() && = delete;
};

} // namespace wwr::extension
