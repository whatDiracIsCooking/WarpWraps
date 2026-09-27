/**
 * @file error_policy.cppm
 * @brief Error policy concepts for error handling strategies
 *
 * An error policy is any type with a `handle_error(T, source_location)` member
 * and a `using error_type = T;` -- no base class. The concepts here are
 * structural (duck-typed), so writing a policy is just declaring that member;
 * the destruction slot additionally requires it to be noexcept.
 *
 * Usage:
 *   import gpumod.extension.common;
 *   using namespace wwr::extension;
 */

export module gpumod.extension.common.error_handling:error_policy;

import std;

export namespace wwr::extension {

// ============================================================================
// Error Policy Concepts
// ============================================================================

/**
 * @brief A type usable as an error policy for error type T
 *
 * @tparam P The policy type to check
 * @tparam T The error code type
 *
 * @note Structural, not inheritance-based: P need only expose
 *       `handle_error(T, source_location)` returning void. Nothrow move is
 *       required so a policy is safe as a member of the move-only RAII wrappers.
 */
template<typename P, typename T>
concept error_policy = requires(P p, T e, std::source_location loc) {
  { p.handle_error(e, loc) } -> std::same_as<void>;
} && std::is_nothrow_move_constructible_v<P> && std::is_nothrow_move_assignable_v<P>;

/**
 * @brief An error policy whose handle_error is additionally noexcept
 *
 * @tparam P The policy type to check
 * @tparam T The error code type
 *
 * @note Required of the *destruction* slot (P_destroy / P_free): it runs from a
 *       destructor, so a throwing handle_error would std::terminate. The
 *       creation slot deliberately stays the weaker error_policy -- a create
 *       policy may throw to propagate a failure, which is only unsafe on the
 *       destroy path. This turns the "MUST NOT THROW" rule from a comment into
 *       a compile-time constraint.
 */
template<typename P, typename T>
concept nothrow_error_policy = error_policy<P, T> && requires(P p, T e, std::source_location loc) {
  { p.handle_error(e, loc) } noexcept;
};

/**
 * @brief An error policy that publishes the error type it handles as `error_type`
 *
 * @tparam P The policy type to check
 *
 * @note A handle deduces its error type from its policy through this concept,
 *       rather than from a separate handle-type -> error-type table. That table
 *       is unusable on HIP, where hipblas/hipsolver/hipsparse handles are all
 *       `void*` and so cannot key distinct error types; the policy always can.
 */
template<typename P>
concept typed_error_policy =
    requires { typename P::error_type; } && error_policy<P, typename P::error_type>;

} // namespace wwr::extension
