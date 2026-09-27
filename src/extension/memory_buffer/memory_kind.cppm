/**
 * @file memory_kind.cppm
 * @brief Memory kind enumeration and error type mappings
 *
 * Provides memory kind classification and associated error type mappings
 * for different allocation strategies.
 *
 * Usage:
 *   import wwr.extension.memory_buffer;
 *   using namespace wwr::extension;
 */

export module wwr.extension.memory_buffer:memory_kind;

import std;
import wwr.runtime_api;
import :host_memory;

export namespace wwr::extension {

// ============================================================================
// Memory Kind Enumeration
// ============================================================================

/**
 * @brief Specifies the type of memory allocation strategy
 *
 * Defines the memory allocation backend and operational characteristics
 * for buffer management.
 */
enum class MemoryKind {
  Device, ///< GPU device memory (supports async operations via gpuMallocAsync/gpuFreeAsync)
  Pinned, ///< Page-locked host memory (faster DMA transfers, no async support)
  Host,   ///< Standard host memory via std::malloc/std::free (no async support)
  Unified ///< Managed memory accessible from both CPU and GPU via gpuMallocManaged
};

// ============================================================================
// Memory Error Type Mapping
// ============================================================================

template<MemoryKind K>
struct MemoryErrorType {
  using type = void;
};

// Specializations for each memory kind
template<>
struct MemoryErrorType<MemoryKind::Device> {
  using type = gpuError_t;
};

template<>
struct MemoryErrorType<MemoryKind::Pinned> {
  using type = gpuError_t;
};

template<>
struct MemoryErrorType<MemoryKind::Host> {
  using type = stdHostMemoryError_t;
};

template<>
struct MemoryErrorType<MemoryKind::Unified> {
  using type = gpuError_t;
};

// ============================================================================
// Memory Invalid Value Mapping
// ============================================================================

/**
 * @brief Maps a MemoryKind to the sentinel "invalid value" error code for that kind.
 *
 * Used by BaseBuffer to report out-of-bounds sub-view construction.
 *
 * @tparam K The memory kind
 */
template<MemoryKind K>
struct MemoryInvalidValue;

template<>
struct MemoryInvalidValue<MemoryKind::Device> {
  static constexpr gpuError_t value = gpuErrorInvalidValue;
};

template<>
struct MemoryInvalidValue<MemoryKind::Pinned> {
  static constexpr gpuError_t value = gpuErrorInvalidValue;
};

template<>
struct MemoryInvalidValue<MemoryKind::Unified> {
  static constexpr gpuError_t value = gpuErrorInvalidValue;
};

template<>
struct MemoryInvalidValue<MemoryKind::Host> {
  static constexpr stdHostMemoryError_t value = stdHostMemInvalidValue;
};

} // namespace wwr::extension
