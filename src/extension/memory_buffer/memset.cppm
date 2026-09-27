/**
 * @file memset.cppm
 * @brief Buffer memset utilities
 *
 * Provides memset functions for buffer zero/fill operations with the same
 * patterns as copy.cppm: synchronous host variants and stream-ordered async
 * variants covering all buffer kind combinations.
 *
 * Usage:
 *   import wwr.extension.memory_buffer;
 *   using namespace wwr::extension;
 */

export module wwr.extension.memory_buffer:memset;

import std;
import :base_buffer;
import :host_memory;
import wwr.runtime_api;

export namespace wwr::extension {

/**
 * @brief Validate buffer memset parameters
 * @return true if parameters are valid, false if bounds are violated
 * @note Zero-count fills are considered valid
 */
template<buffer_base B>
constexpr bool validate_memset(const B &buf, const std::size_t offset,
                               const std::size_t count) noexcept {
  if (count == 0)
    return true;
  if (offset > buf.num_elements())
    return false;
  if (count > (buf.num_elements() - offset))
    return false;
  return true;
}

/**
 * @brief Synchronous host memset for full buffer
 * @note Only enabled for non-device buffers (Host, Pinned)
 * @note Uses std::memset
 * @param buf Buffer to fill
 * @param value Byte value to fill with
 * @return stdHostMemSuccess always (std::memset cannot fail)
 */
template<buffer_base B>
  requires(!B::is_device)
stdHostMemoryError_t memset(B &buf, const int value) noexcept {
  // std::memset is undefined for a null pointer even with a zero length, and an
  // empty buffer has data() == nullptr.
  if (buf.num_elements() == 0)
    return stdHostMemSuccess;

  std::memset(buf.data(), value, buf.size_bytes());
  return stdHostMemSuccess;
}

/**
 * @brief Stream-ordered async memset for full buffer
 * @note Uses wwrMemsetAsync; works for all GPU-accessible buffer kinds
 * @param buf Buffer to fill
 * @param value Byte value to fill with
 * @param stream GPU stream for the operation (use stream 0 for default stream)
 * @return wwrError_t from wwrMemsetAsync
 */
template<buffer_base B>
wwrError_t memset(B &buf, const int value, const wwrStream_t stream) noexcept {
  return wwrMemsetAsync(buf.data(), value, buf.size_bytes(), stream);
}

/**
 * @brief Stream-ordered async memset with offset and count
 * @note Uses wwrMemsetAsync; works for all GPU-accessible buffer kinds
 * @param buf Buffer to fill
 * @param offset Offset in buffer (in elements)
 * @param count Number of elements to fill
 * @param value Byte value to fill with
 * @param stream GPU stream for the operation (use stream 0 for default stream)
 * @return wwrError_t from wwrMemsetAsync, or wwrErrorInvalidValue if validation fails
 */
template<buffer_base B>
wwrError_t memset(B &buf, const std::size_t offset, const std::size_t count, const int value,
                  const wwrStream_t stream) noexcept {
  if (!validate_memset(buf, offset, count))
    return wwrErrorInvalidValue;
  return wwrMemsetAsync(static_cast<typename B::storage_type *>(buf.data()) + offset, value,
                        count * B::element_size, stream);
}

} // namespace wwr::extension
