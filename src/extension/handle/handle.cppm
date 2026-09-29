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
 *   class StreamWrapper : public BaseHandle<wwrStream_t, StreamWrapper, ...> { ... };
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
  // policy_create_ is the canonical, always-stored slot. policy_destroy_ elides
  // itself into it when both are the same empty type (the common default), so
  // reach the destroy policy through destroy_policy(), never the raw member.
  [[no_unique_address]] P_create policy_create_{};
  [[no_unique_address]] policy_slot<P_destroy, P_create> policy_destroy_{};

  P_create &create_policy() noexcept { return policy_create_; }
  const P_create &create_policy() const noexcept { return policy_create_; }
  P_destroy &destroy_policy() noexcept { return policy_destroy_.resolve(policy_create_); }
  const P_destroy &destroy_policy() const noexcept { return policy_destroy_.resolve(policy_create_); }

  // Tag type for derived classes to skip default handle creation
  struct skip_default_create_t {};

  // Protected constructor that skips automatic handle creation
  // Allows derived classes to manually create handles with custom parameters
  BaseHandle(skip_default_create_t) noexcept {}

  // Skip automatic creation but still store the create/destroy policies. The
  // parameterless skip constructor above leaves both policies default-built,
  // which is all a pointer handle whose creation cannot fail through a stateful
  // policy needs (gpu_mem_pool). A handle that must create itself in its own
  // constructor -- because it needs arguments create() cannot carry -- and wants
  // a caller-supplied policy on that create (e.g. a texture/surface object under
  // a counting policy) threads them through here, then calls adopt().
  BaseHandle(skip_default_create_t, P_create policy_create, P_destroy policy_destroy)
      : policy_create_(std::move(policy_create)), policy_destroy_(std::move(policy_destroy)) {}

  /// @brief Record that a handle the derived class created for itself (through a
  ///        skip_default_create constructor) is live and owned. For a flagged
  ///        handle -- no in-band null, e.g. a CUDA texture object -- this sets the
  ///        ownership bit from the create call's success bool; for a pointer
  ///        handle liveness is the null sentinel, already carried by handle_, so
  ///        the bool is ignored. This is the skip-create counterpart to
  ///        run_create's flag handling, for wrappers that cannot route creation
  ///        through the create() hook.
  void adopt(bool created) noexcept {
    if constexpr (!has_null_sentinel) {
      owns_ = created;
    }
  }

public:
  BaseHandle(std::source_location location = std::source_location::current()) {
    run_create(location);
  }

  BaseHandle(P_create policy, std::source_location location = std::source_location::current())
      : policy_create_(policy), policy_destroy_(P_destroy(std::move(policy))) {
    run_create(location);
  }

  BaseHandle(P_create policy_create, P_destroy policy_destroy,
                std::source_location location = std::source_location::current())
      : policy_create_(std::move(policy_create)), policy_destroy_(std::move(policy_destroy)) {
    run_create(location);
  }

  ~BaseHandle() {
    if (valid()) {
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
      if (valid()) {
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

  /// @brief Does this wrapper own a live handle -- one its destructor will
  ///        destroy? False after a create() that failed without throwing (the
  ///        recoverable record-and-continue policy path) or when moved-from.
  ///        The single liveness predicate: the destructor and move-assignment
  ///        gate on it too. Works for the value-handle case that get() cannot
  ///        express -- a failed cufftHandle create leaves a garbage int, but
  ///        owns_ still reads false here. Named rather than `explicit operator
  ///        bool` because the non-explicit operator T() above would make
  ///        `if (h)` ambiguous for pointer T.
  bool valid() const noexcept {
    if constexpr (has_null_sentinel) {
      return handle_ != nullptr;
    } else {
      return owns_;
    }
  }

  /// @brief A non-owning, copyable view of this handle.
  ///
  /// Deleted on rvalues so a view cannot be taken from a temporary handle, which
  /// would dangle immediately. Device-bound layers return a richer view that also
  /// carries the device index; borrow-safe operations are free functions.
  HandleView<T> view() const & noexcept { return HandleView<T>{handle_}; }
  HandleView<T> view() && = delete;
};

} // namespace wwr::extension
