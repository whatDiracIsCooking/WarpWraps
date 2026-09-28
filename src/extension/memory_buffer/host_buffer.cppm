/**
 * @file host_buffer.cppm
 * @brief RAII wrapper for standard host memory buffers
 *
 * Provides HostBufferWrapper class for automatic host memory management.
 * Host memory uses standard std::malloc/std::free for allocation.
 */

export module wwr.extension.memory_buffer:host_buffer;

import :base_buffer;
import :memory_kind;
import :host_memory;
import wwr.extension.common;
import std;

export namespace wwr::extension {

/**
 * @brief RAII wrapper for standard host memory buffer
 *
 * Automatically allocates host memory on construction and deallocates it on destruction.
 * Uses standard std::malloc/std::free for allocation. This is pageable memory that
 * may be swapped to disk by the OS.
 *
 * @tparam T The element type stored in the buffer
 * @tparam P_alloc Error policy type for allocation (defaults to DefaultErrorPolicy<stdHostMemoryError_t>)
 * @tparam P_free Error policy type for deallocation (defaults to P_alloc)
 *
 * @note P_free MUST NOT THROW - it is called from the destructor.
 * @note For faster host-device transfers, consider using PinnedBufferWrapper instead
 */
template<typename T,
         error_policy<stdHostMemoryError_t> P_alloc = DefaultErrorPolicy<stdHostMemoryError_t>,
         nothrow_error_policy<stdHostMemoryError_t> P_free = P_alloc>
class HostBufferWrapper
    : public BaseBuffer<T, MemoryKind::Host, HostBufferWrapper<T, P_alloc, P_free>, P_alloc,
                        P_free> {
private:
  using Base =
      BaseBuffer<T, MemoryKind::Host, HostBufferWrapper<T, P_alloc, P_free>, P_alloc, P_free>;

public:
  // Inherit constructors from base
  using BaseBuffer<T, MemoryKind::Host, HostBufferWrapper<T, P_alloc, P_free>, P_alloc,
                   P_free>::BaseBuffer;

  /** @brief Releases the allocation while this object is still alive */
  ~HostBufferWrapper() { this->destroy_(); }

  HostBufferWrapper(HostBufferWrapper &&) noexcept = default;
  HostBufferWrapper &operator=(HostBufferWrapper &&) noexcept = default;

  /**
     * @brief Allocate host memory
     *
     * @param ptr Output parameter for the allocated memory pointer
     * @param num_elements Number of elements to allocate
     * @param policy Error policy for allocation
     * @param location Source location where allocation was requested
     *
     * @note Static: it runs from a BaseBuffer constructor, before this object exists.
     */
  static void allocate(T **ptr, std::size_t num_elements, P_alloc &policy,
                       std::source_location location) {
    *ptr = nullptr;
    const std::size_t size_bytes = num_elements * Base::element_size;
    const auto error = std_malloc(reinterpret_cast<void **>(ptr), size_bytes);
    // Must return on failure: std::malloc left *ptr null, and a policy that
    // reports without aborting would otherwise fall through to memset it.
    if (error != stdHostMemSuccess) {
      *ptr = nullptr;
      policy.handle_error(error, location);
      return;
    }
    std::memset(*ptr, 0, size_bytes);
  }

  /**
     * @brief Deallocate host memory
     *
     * @param ptr Pointer to memory to deallocate
     * @param num_elements Number of elements (unused, kept for interface consistency)
     */
  void deallocate(T *ptr, std::size_t num_elements) {
    if (ptr != nullptr) {
      const auto error = std_free(ptr);
      if (error != stdHostMemSuccess) {
        this->policy_free_.handle_error(error, std::source_location::current());
      }
    }
  }
};

} // namespace wwr::extension
