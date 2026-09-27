/**
 * @file copy.cppm
 * @brief Buffer copy validation utilities
 *
 * Provides validation functions for buffer copy operations to ensure
 * bounds safety and parameter correctness.
 *
 * Usage:
 *   import wwr.extension.memory_buffer;
 *   using namespace wwr::extension;
 */

export module wwr.extension.memory_buffer:copy;

import std;
import :base_buffer;
import :host_memory;
import wwr.runtime_api;

export namespace wwr::extension {

/**
 * @brief Validate buffer copy parameters
 * @return true if copy parameters are valid, false if invalid (bounds violation)
 * @note Zero-count copies are considered valid
 */
template<buffer_base B1, buffer_base B2>
constexpr bool validate_copy(const B1 &dst, const std::size_t dst_offset, const B2 &src,
                             const std::size_t src_offset, const std::size_t count) noexcept {
  // Zero count is valid (no-op)
  if (count == 0)
    return true;

  // Check offset bounds
  if (src_offset > src.num_elements() || dst_offset > dst.num_elements())
    return false;

  // Check count overflow
  if (count > (src.num_elements() - src_offset) || count > (dst.num_elements() - dst_offset))
    return false;

  return true;
}

/**
 * @brief Synchronous host memory copy for full buffer
 * @note Both buffers must have the same element type
 * @note Uses std::memcpy for synchronous host-to-host copy
 * @param dst Destination buffer
 * @param src Source buffer
 * @return stdHostMemSuccess on success, stdHostMemInvalidValue if validation fails
 */
template<buffer_base B1, buffer_base B2>
  requires same_value_type<B1, B2> && (!B1::is_device) && (!B2::is_device)
stdHostMemoryError_t copy(B1 &dst, const B2 &src) noexcept {
  if (!validate_copy(dst, 0, src, 0, src.num_elements()))
    return stdHostMemInvalidValue;

  // std::memcpy is undefined for null pointers even with a zero length, and an
  // empty buffer has data() == nullptr.
  if (src.num_elements() == 0)
    return stdHostMemSuccess;

  std::memcpy(dst.data(), src.data(), src.size_bytes());
  return stdHostMemSuccess;
}

/**
 * @brief Generic stream-ordered async copy for any buffer combination
 * @note Both buffers must have the same element type
 * @note Uses gpuMemcpyDefault to automatically determine copy direction via UVA
 * @note Supports all buffer combinations with stream-ordered semantics
 * @note Host-to-host copies maintain stream ordering but execute synchronously
 * @param dst Destination buffer
 * @param src Source buffer
 * @param stream GPU stream for the copy (use stream 0 for default stream)
 * @return gpuError_t from gpuMemcpyAsync, or gpuErrorInvalidValue if validation fails
 */
template<buffer_base B1, buffer_base B2>
  requires same_value_type<B1, B2>
gpuError_t copy(B1 &dst, const B2 &src, gpuStream_t stream) noexcept {
  if (!validate_copy(dst, 0, src, 0, src.num_elements()))
    return gpuErrorInvalidValue;

  return gpuMemcpyAsync(dst.data(), src.data(), src.size_bytes(), gpuMemcpyDefault, stream);
}

/**
 * @brief Generic stream-ordered async copy with offsets and count
 * @note Both buffers must have the same element type
 * @note Uses gpuMemcpyDefault to automatically determine copy direction via UVA
 * @note Supports all buffer combinations with stream-ordered semantics
 * @note Host-to-host copies maintain stream ordering but execute synchronously
 * @param dst Destination buffer
 * @param dst_offset Offset in destination buffer (in elements)
 * @param src Source buffer
 * @param src_offset Offset in source buffer (in elements)
 * @param count Number of elements to copy
 * @param stream GPU stream for the copy (use stream 0 for default stream)
 * @return gpuError_t from gpuMemcpyAsync, or gpuErrorInvalidValue if validation fails
 */
template<buffer_base B1, buffer_base B2>
  requires same_value_type<B1, B2>
gpuError_t copy(B1 &dst, const std::size_t dst_offset, const B2 &src, const std::size_t src_offset,
                const std::size_t count, gpuStream_t stream) noexcept {
  if (!validate_copy(dst, dst_offset, src, src_offset, count))
    return gpuErrorInvalidValue;
  if (count == 0)
    return gpuSuccess;

  // Offset through storage_type: for a void buffer, "dst.data() + dst_offset"
  // is arithmetic on void* - a GNU extension rather than portable C++.
  return gpuMemcpyAsync(static_cast<typename B1::storage_type *>(dst.data()) + dst_offset,
                        static_cast<const typename B2::storage_type *>(src.data()) + src_offset,
                        count * B1::element_size, gpuMemcpyDefault, stream);
}

} // namespace wwr::extension
