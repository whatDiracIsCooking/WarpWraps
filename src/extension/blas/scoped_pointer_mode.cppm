/**
 * @file scoped_pointer_mode.cppm
 * @brief RAII guard that sets a BLAS handle's pointer mode and restores the previous one
 *
 * Usage:
 *   import wwr.extension.blas;
 *
 *   {
 *     ScopedPointerMode scope{handle_owner, WWRBLAS_POINTER_MODE_DEVICE, MyPolicy{}};
 *     ...                       // handle reads alpha/beta from device pointers
 *   }                           // previous pointer mode restored on the handle
 *
 *   // handle_owner is a shared_ptr to any blas_handle (e.g. a BlasHandleWrapper);
 *   // MyPolicy is any error_policy<wwrblasStatus_t>: the kit's opt-in kit::AbortPolicy,
 *   // or your own -- the core forces none.
 */

export module wwr.extension.blas:scoped_pointer_mode;

import :blas_error;
// The constructor and destructor route their gpu_check calls through policy_mode_,
// which odr-uses success_code<wwrblasStatus_t>(). That specialization lives in
// :blas_error; without it reachable the compiler falls back to the
// inline-but-undefined primary template (-Wundefined-inline, and an ill-formed
// implicit instantiation). Acyclic: :blas_error imports none of
// :scoped_pointer_mode's chain. Mirrors :scoped_device_index's dependence on :error.
import :blas_handle; // the blas_handle concept the co-owned handle must satisfy
import wwr.blas;
import wwr.extension.common; // gpu_check, error_policy, NonCopyable
import std;

export namespace wwr::extension {

/**
 * @brief Set a BLAS handle's pointer mode for the guard's lifetime, then restore it
 *
 * Records the pointer mode current on the handle at construction, switches it to
 * `target_mode`, and restores the recorded mode on destruction. This is the
 * per-handle analogue of ScopedDeviceIndex: the pointer mode is a property of the BLAS
 * handle (whether scalar arguments like alpha/beta live in host or device
 * memory), not implicit thread state.
 *
 * The handle is taken as a std::shared_ptr and retained, so it is guaranteed to
 * outlive the guard -- the restore on destruction can never run against a
 * freed handle. This shared-owns exactly as StreamBoundHandle retains its stream
 * owner. Constraining the pointee on the blas_handle concept (rather than the
 * concrete BlasHandleWrapper) keeps this to one handle-axis parameter H instead
 * of threading the wrapper's <P_create, P_destroy, S, P_device_access> list, none
 * of which this guard uses; H is reached only through the private raw_handle() to
 * name the raw handle, so a bare wwrblasHandle_t works as the pointee too. That
 * bare form comes with a lifetime caveat -- a shared_ptr<wwrblasHandle_t> owns a
 * copy of the pointer, not the GPU handle, so it borrows rather than co-owns; the
 * bare-handle convenience constructor documents this in full.
 *
 * wwrblasGetPointerMode/wwrblasSetPointerMode are wwrblasStatus_t-returning, so
 * the return cannot simply be dropped -- each of the three calls (the two that
 * *enter* the scope and the one that restores on destruction) goes through the
 * same retained P, so pointer-mode error handling is entirely the caller's and
 * the library keeps no error policy of its own. It stays the plain error_policy
 * (not nothrow_error_policy) so a caller may hand in a throwing policy to
 * propagate a bad-handle failure out of the constructor as an exception.
 *
 * The restore runs from the destructor, which is noexcept: a policy that
 * *throws* on a failed restore therefore terminates rather than propagates, so a
 * pointer-mode policy that must survive destruction should be nothrow (one that
 * aborts, say). This mirrors ScopedDeviceIndex's noexcept-destructor caveat exactly.
 *
 * @tparam H The co-owned handle type; a blas_handle (yields a wwrblasHandle_t),
 *         retained by shared_ptr so it outlives the guard.
 * @tparam P The error policy for the get/set pointer-mode calls; typed to
 *         wwrblasStatus_t (what those calls return). No default -- the caller
 *         names the policy.
 */
template<blas_handle H, error_policy<wwrblasStatus_t> P>
struct ScopedPointerMode : private NonCopyable {
  wwrblasPointerMode_t original_mode{}; ///< Mode current at construction, restored on destruction

  /// @brief Switch `handle`'s pointer mode to `target_mode`, recording the
  ///        previous mode to restore. Routes both entering calls through
  ///        `policy_mode`, then retains it -- and the co-owned handle -- for the
  ///        symmetric restore call on destruction.
  /// @param handle Shared-owned BLAS handle; retained, so it outlives the guard
  ///        (must be non-null)
  explicit ScopedPointerMode(std::shared_ptr<H> handle, const wwrblasPointerMode_t target_mode,
                            P policy_mode = {},
                            std::source_location location = std::source_location::current())
      : handle_(std::move(handle)), policy_mode_(std::move(policy_mode)) {
    const wwrblasHandle_t raw = raw_handle(*handle_);
    gpu_check(wwrblasGetPointerMode(raw, &original_mode), policy_mode_, location);
    gpu_check(wwrblasSetPointerMode(raw, target_mode), policy_mode_, location);
  }

  /// @brief Convenience overload for a bare wwrblasHandle_t: wrap it in a
  ///        shared_ptr and delegate to the shared-owner constructor above.
  ///
  /// Only for H == wwrblasHandle_t (a wrapper like BlasHandleWrapper is
  /// non-copyable, so make_shared could not copy it here anyway -- pass such a
  /// handle through the shared_ptr constructor, which genuinely co-owns it).
  ///
  /// UNLIKE that constructor, this does NOT extend the handle's lifetime: a
  /// shared_ptr<wwrblasHandle_t> owns a copy of the raw *pointer*, not the GPU
  /// handle it names, so the guard here BORROWS -- the caller must keep the real
  /// handle alive until the guard is destroyed, or the restore runs against a
  /// freed handle. Reach for it only when that lifetime is already assured.
  explicit ScopedPointerMode(H handle, const wwrblasPointerMode_t target_mode, P policy_mode = {},
                            std::source_location location = std::source_location::current())
    requires std::same_as<H, wwrblasHandle_t>
      : ScopedPointerMode(std::make_shared<H>(handle), target_mode, std::move(policy_mode),
                         location) {}

  // Restore symmetrically, through the same policy the entry calls used (see the
  // class note on the noexcept-destructor caveat for throwing policies). The
  // retained handle is still alive by construction, so the restore is well-defined.
  ~ScopedPointerMode() {
    gpu_check(wwrblasSetPointerMode(raw_handle(*handle_), original_mode), policy_mode_);
  }

  // Copy operations are implicitly deleted via the NonCopyable base. The
  // user-declared destructor suppresses the implicit moves, so the guard stays
  // non-movable as well -- handle_ is thus never null after construction.

private:
  // Name the raw wwrblasHandle_t regardless of which blas_handle shape H is: the
  // concept admits both a bare wwrblasHandle_t and a wrapper exposing get(), so
  // `handle_->get()` would not compile on the bare form. noexcept mirrors the
  // get() arm's own noexcept requirement. Private because it is an implementation
  // detail of this guard, not part of the module's surface.
  static constexpr wwrblasHandle_t raw_handle(const H &h) noexcept {
    if constexpr (std::same_as<H, wwrblasHandle_t>)
      return h;
    else
      return h.get();
  }

  std::shared_ptr<H> handle_; ///< Co-owned handle whose pointer mode this guard toggles and restores
  [[no_unique_address]] P policy_mode_{}; ///< Policy for the get/set calls
};

} // namespace wwr::extension
