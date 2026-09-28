/**
 * @file error_code.cppm
 * @brief Success code utilities for error handling
 *
 * Provides template function declarations to retrieve success codes for various error types.
 * The wwrError_t specializations live in wwr.extension.common:gpu_error (they
 * back the device-bound handle base); library status types are specialized in
 * their own extension modules.
 *
 * Usage:
 *   import wwr.extension.common;  // wwrError_t specializations come with it
 *   auto success = success_code<wwrError_t>();
 */

export module wwr.extension.error_handling:error_code;

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
 * @note This is a template that requires specialization for each error type
 */
template<typename T>
constexpr T success_code() noexcept;

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
 * @note This is a template that requires specialization for each error type
 */
template<typename T>
const char *error_name(T code) noexcept;

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
 * @note This is a template that requires specialization for each error type
 */
template<typename T>
const char *error_string(T code) noexcept;

} // namespace wwr::extension
