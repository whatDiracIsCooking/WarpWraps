/**
 * @file error_policy.cppm
 * @brief Error policy base class for error handling strategies
 *
 * Provides abstract base class template for implementing error handling policies.
 *
 * Usage:
 *   import gpumod.extension.common;
 *   using namespace gpumod::extension;
 */

export module gpumod.extension.common:error_policy;

import std;

export namespace gpumod::extension {

// ============================================================================
// Base Error Policy
// ============================================================================

/**
 * @brief Abstract base class template for error handling policies
 *
 * @tparam T The error code type
 *
 * @note This is an abstract base class that should be specialized for
 *       specific error handling strategies
 */
template<typename T>
class BaseErrorPolicy {
public:
  /**
     * @brief Handle an error condition
     *
     * @param error The error code to handle
     * @param location Source location where the error occurred
     *
     * @note Pure virtual function - must be implemented by derived classes
     */
  virtual void handle_error(const T error, std::source_location location) = 0;

  virtual ~BaseErrorPolicy() = default;
};

// ============================================================================
// Error Policy Concept
// ============================================================================

/**
 * @brief Concept constraining type P to be derived from BaseErrorPolicy<T>
 *
 * @tparam P The policy type to check
 * @tparam T The error code type
 *
 * @note Ensures that P is a valid error policy for error type T
 * @note Requires nothrow move operations to ensure safe use in RAII wrappers
 */
template<typename P, typename T>
concept error_policy =
    std::derived_from<P, BaseErrorPolicy<T>> && std::is_nothrow_move_constructible_v<P> &&
    std::is_nothrow_move_assignable_v<P>;

} // namespace gpumod::extension
