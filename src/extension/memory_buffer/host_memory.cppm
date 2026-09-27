/**
 * @file host_memory.cppm
 * @brief Error codes for standard host memory operations
 *
 * Provides error status codes analogous to wwrError_t for host memory
 * allocation and deallocation operations.
 *
 * Usage:
 *   import wwr.extension.memory_buffer;
 *   using namespace wwr::extension;
 */

export module wwr.extension.memory_buffer:host_memory;

import wwr.extension.common;
import std;

export namespace wwr::extension {

// ============================================================================
// Error Code Enumeration
// ============================================================================

/**
 * @brief Error codes for standard host memory operations
 *
 * Provides error status codes analogous to wwrError_t for host memory
 * allocation and deallocation operations.
 */
enum class stdHostMemoryError_t {
  stdHostMemSuccess,        ///< Operation completed successfully
  stdHostMemAllocFailure,   ///< Memory allocation failed
  stdHostMemDeallocFailure, ///< Memory deallocation failed
  stdHostMemInvalidValue    ///< Invalid parameter value
};

// Bring enumerators into namespace scope
using enum stdHostMemoryError_t;

// ============================================================================
// Success Code Specialization
// ============================================================================

/**
 * @brief Specialization for stdHostMemoryError_t
 *
 * @return stdHostMemoryError_t::Success
 */
template<>
stdHostMemoryError_t success_code<stdHostMemoryError_t>() noexcept {
  return stdHostMemSuccess;
}

// ============================================================================
// Error Name Specialization
// ============================================================================

/**
 * @brief Specialization for stdHostMemoryError_t
 *
 * @param error The host memory error code
 * @return The error name string (e.g., "Success", "AllocFailure")
 */
template<>
const char *error_name<stdHostMemoryError_t>(stdHostMemoryError_t error) noexcept {
  switch (error) {
  case stdHostMemSuccess:
    return "stdHostMemSuccess";
  case stdHostMemAllocFailure:
    return "stdHostMemAllocFailure";
  case stdHostMemDeallocFailure:
    return "stdHostMemDeallocFailure";
  case stdHostMemInvalidValue:
    return "stdHostMemInvalidValue";
  default:
    return "UnknownError";
  }
}

// ============================================================================
// Error String Specialization
// ============================================================================

/**
 * @brief Specialization for stdHostMemoryError_t
 *
 * @param error The host memory error code
 * @return The error description string
 */
template<>
const char *error_string<stdHostMemoryError_t>(stdHostMemoryError_t error) noexcept {
  switch (error) {
  case stdHostMemSuccess:
    return "operation completed successfully";
  case stdHostMemAllocFailure:
    return "std::malloc failed";
  case stdHostMemDeallocFailure:
    return "std::free failed";
  case stdHostMemInvalidValue:
    return "invalid parameter value";
  default:
    return "unknown error";
  }
}

// ============================================================================
// Stream Insertion
// ============================================================================

/**
 * @brief Stream a host memory error code by name
 *
 * @param os The output stream
 * @param error The host memory error code
 * @return The stream, for chaining
 *
 * @note Found by ADL, which is what lets the test framework's expect::eq
 *       accept this enum: its comparison helpers are constrained on a
 *       `streamable` concept so a failed assertion can print both operands.
 *       A scoped enum has no implicit operator<<, so without this the
 *       constraint fails and the call is simply not viable.
 * @note Delegates to error_name rather than repeating the switch, so the
 *       streamed spelling cannot drift from the canonical one.
 */
inline std::ostream &operator<<(std::ostream &os, const stdHostMemoryError_t error) {
  return os << error_name(error);
}

// ============================================================================
// Host Memory Allocation Wrappers
// ============================================================================

/**
 * @brief Wrapper for std::malloc with error code return
 *
 * @param ptr Pointer to receive the allocated memory address
 * @param size Number of bytes to allocate
 * @return stdHostMemoryError_t::Success if allocation succeeded,
 *         stdHostMemoryError_t::AllocFailure if allocation failed
 */
stdHostMemoryError_t std_malloc(void **ptr, const std::size_t size) noexcept {
  *ptr = std::malloc(size);
  if (*ptr == nullptr) {
    return stdHostMemAllocFailure;
  }
  return stdHostMemSuccess;
}

/**
 * @brief Wrapper for std::free with error code return
 *
 * @param ptr Pointer to memory to deallocate
 * @return stdHostMemoryError_t::Success if deallocation succeeded,
 *         stdHostMemoryError_t::DeallocFailure if ptr is nullptr
 *
 * @note std::free with nullptr is well-defined (no-op), but we treat it as
 *       an error to catch potential programming mistakes
 */
stdHostMemoryError_t std_free(void *ptr) noexcept {
  if (ptr == nullptr) {
    return stdHostMemDeallocFailure;
  }
  std::free(ptr);
  return stdHostMemSuccess;
}

// ============================================================================
// Template Instantiations
// ============================================================================

// Explicitly instantiate DefaultErrorPolicy for stdHostMemoryError_t
template class DefaultErrorPolicy<stdHostMemoryError_t>;

// Explicitly instantiate gpu_check for stdHostMemoryError_t
template bool gpu_check<stdHostMemoryError_t>(const stdHostMemoryError_t error,
                                              std::source_location location);

template bool gpu_check<stdHostMemoryError_t, DefaultErrorPolicy<stdHostMemoryError_t>>(
    const stdHostMemoryError_t error, DefaultErrorPolicy<stdHostMemoryError_t> &policy,
    std::source_location location);

} // namespace wwr::extension
