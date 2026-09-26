/**
 * @file gpu_handle.cppm
 * @brief RAII wrapper for GPU handles
 *
 * Provides generic RAII wrapper for managing GPU handles (streams, events, library handles)
 * with automatic resource cleanup.
 *
 * Usage:
 *   import gpumod.extension.common;
 *
 *   class GpuStream : public BaseGpuHandle<gpuStream_t, GpuStream, ...> { ... };
 */

export module gpumod.extension.common:gpu_handle;

import :error_policy;
import :noncopyable;
import :gpu_handle_view;
import std;

export namespace gpumod::extension {

// ============================================================================
// RAII Handle Wrapper
// ============================================================================

/// @brief RAII wrapper for GPU handles using CRTP
/// @tparam T The underlying GPU handle type
/// @tparam Derived The derived class type
/// @tparam P_create The error policy for creation; its error_type is the error
///         type this handle checks against
/// @tparam P_destroy The error policy for destruction (defaults to P_create),
///         constrained to P_create's error type
///
/// @note P_destroy MUST NOT THROW exceptions, as it is invoked from the destructor.
///       Throwing from P_destroy::handle_error() will result in program termination.
/// @note The error type is deduced from the policy (P_create::error_type), not
///       from the handle type T: on HIP the vendor handles are all `void*`, so a
///       handle-type -> error-type table cannot tell them apart. The policy
///       always carries its own error type.
template<typename T, typename Derived, typed_error_policy P_create,
         error_policy<typename P_create::error_type> P_destroy = P_create>
class BaseGpuHandle : private NonCopyable {
protected:
  T handle_ = nullptr;
  P_create policy_create_{};
  P_destroy policy_destroy_{};

  // Tag type for derived classes to skip default handle creation
  struct skip_default_create_t {};

  // Protected constructor that skips automatic handle creation
  // Allows derived classes to manually create handles with custom parameters
  BaseGpuHandle(skip_default_create_t) noexcept {}

public:
  BaseGpuHandle(std::source_location location = std::source_location::current()) {
    static_cast<Derived *>(this)->create(&handle_, location);
  }

  BaseGpuHandle(P_create policy, std::source_location location = std::source_location::current())
      : policy_create_(policy), policy_destroy_(std::move(policy)) {
    static_cast<Derived *>(this)->create(&handle_, location);
  }

  BaseGpuHandle(P_create policy_create, P_destroy policy_destroy,
                std::source_location location = std::source_location::current())
      : policy_create_(std::move(policy_create)), policy_destroy_(std::move(policy_destroy)) {
    static_cast<Derived *>(this)->create(&handle_, location);
  }

  ~BaseGpuHandle() {
    if (handle_) {
      static_cast<Derived *>(this)->destroy(handle_);
    }
  }

  BaseGpuHandle(BaseGpuHandle &&other) noexcept
      : handle_(other.handle_), policy_create_(std::move(other.policy_create_)),
        policy_destroy_(std::move(other.policy_destroy_)) {
    // Take ownership by moving the handle
    // Leave other in valid null state
    other.handle_ = nullptr;
  }

  BaseGpuHandle &operator=(BaseGpuHandle &&other) noexcept {
    // Self-assignment check
    if (this != &other) {
      // Destroy current handle if valid
      if (handle_ != nullptr) {
        static_cast<Derived *>(this)->destroy(handle_);
      }

      // Take ownership from other
      handle_ = other.handle_;
      other.handle_ = nullptr;
      policy_create_ = std::move(other.policy_create_);
      policy_destroy_ = std::move(other.policy_destroy_);
    }

    return *this;
  }

  // Copy operations are implicitly deleted via the NonCopyable base.

  operator T() const noexcept { return handle_; }
  T get() const noexcept { return handle_; }

  /// @brief A non-owning, copyable view of this handle.
  ///
  /// Deleted on rvalues so a view cannot be taken from a temporary handle, which
  /// would dangle immediately. Derived layers hide this with a richer view type
  /// (device index, borrow-safe operations) where they have one.
  GpuHandleView<T> view() const & noexcept { return GpuHandleView<T>{handle_}; }
  GpuHandleView<T> view() && = delete;
};

} // namespace gpumod::extension
