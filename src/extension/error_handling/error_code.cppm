/**
 * @file error_code.cppm
 * @brief Error-code facilities (success_code / error_name / error_string) and
 *        the error_type concept derived from them
 *
 * Provides three per-type facilities and the concept that recognizes a type as
 * one the error-handling layer supports. The primaries are `= delete`: a type is
 * an error type exactly when it specializes all three, so the specializations
 * *are* the registration -- there is no separate registry to keep in sync, and a
 * half-specialized type (success_code but no error_string) is rejected by the
 * concept rather than crashing later in an error policy's handle_error.
 *
 * The wwrError_t specializations live in wwr.extension.common:gpu_error (they
 * back the device-bound handle base); library status types are specialized in
 * their own extension modules. error_type<T> is therefore only satisfied where
 * the type's module is reachable -- the same reachability every call already
 * needs.
 *
 * Usage:
 *   import wwr.extension.common;  // wwrError_t specializations come with it
 *   auto success = success_code<wwrError_t>();
 *   static_assert(error_type<wwrError_t>);
 */

export module wwr.extension.error_handling:error_code;

import std;

export namespace wwr::extension {

// ============================================================================
// Success Code Template
// ============================================================================

/**
 * @brief Get the success code for a given error type
 *
 * @tparam T The error code type
 * @return The success value for type T
 *
 * @note Deleted primary: a type must specialize this to be an error_type. An
 *       unspecialized T fails the error_type concept (and, if called directly,
 *       fails to compile here rather than at link time).
 */
template<typename T>
constexpr T success_code() noexcept = delete;

// ============================================================================
// Error Name Template
// ============================================================================

/**
 * @brief Get the string name of an error code
 *
 * @tparam T The error code type
 * @param code The error code value
 * @return String representation of the error name
 *
 * @note Deleted primary: see success_code above.
 */
template<typename T>
const char *error_name(T code) noexcept = delete;

// ============================================================================
// Error String Template
// ============================================================================

/**
 * @brief Get the descriptive string of an error code
 *
 * @tparam T The error code type
 * @param code The error code value
 * @return Descriptive string explaining the error
 *
 * @note Deleted primary: see success_code above.
 */
template<typename T>
const char *error_string(T code) noexcept = delete;

// ============================================================================
// Error Type Concept
// ============================================================================

/**
 * @brief A type the error-handling layer supports
 *
 * @tparam T The type to check
 *
 * @note Satisfied iff T specializes all three facilities above. Deriving the
 *       concept from the facilities (rather than a separate opt-in flag) makes
 *       the specializations the single source of truth: a type cannot be an
 *       error_type without actually providing what gpu_check and an error policy
 *       call, so partial registration is a concept failure, not a runtime crash.
 *       Calling a deleted primary in this requires-expression is a soft
 *       non-match, so error_type<T> is usable to constrain overloads and branch
 *       in `if constexpr`, not just to hard-fail on misuse.
 */
template<typename T>
concept error_type = requires(T code) {
  { success_code<T>() } -> std::same_as<T>;
  { error_name(code) } -> std::same_as<const char *>;
  { error_string(code) } -> std::same_as<const char *>;
};

} // namespace wwr::extension
