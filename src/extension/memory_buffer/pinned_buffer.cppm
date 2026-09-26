/**
 * @file pinned_buffer.cppm
 * @brief RAII wrapper for pinned (page-locked) host memory buffers
 *
 * Provides PinnedBuffer class for automatic pinned memory management.
 * Pinned memory enables faster DMA transfers between host and device.
 */

export module gpumod.extension.memory_buffer:pinned_buffer;

import :buffer_base;
import :memory_kind;
import gpumod.runtime_api;
import gpumod.extension.common;
import gpumod.extension.runtime;
import std;

export namespace gpumod::extension {

/**
 * @brief RAII wrapper for pinned (page-locked) host memory buffer
 *
 * Automatically allocates pinned memory on construction and deallocates it on destruction.
 * Pinned memory provides faster DMA transfers between host and device compared to
 * pageable host memory.
 *
 * @tparam T The element type stored in the buffer
 * @tparam P_alloc Error policy type for allocation (defaults to DefaultErrorPolicy<gpuError_t>)
 * @tparam P_free Error policy type for deallocation (defaults to P_alloc)
 *
 * @note P_free MUST NOT THROW - it is called from the destructor.
 * @note Pinned memory is a limited resource - allocate conservatively
 */
template<typename T, error_policy<gpuError_t> P_alloc = DefaultErrorPolicy<gpuError_t>,
         error_policy<gpuError_t> P_free = P_alloc>
class PinnedBufferWrapper
    : public BufferBase<T, MemoryKind::Pinned, PinnedBufferWrapper<T, P_alloc, P_free>, P_alloc,
                        P_free> {
private:
  using Base =
      BufferBase<T, MemoryKind::Pinned, PinnedBufferWrapper<T, P_alloc, P_free>, P_alloc, P_free>;

public:
  // Inherit constructors from base
  using BufferBase<T, MemoryKind::Pinned, PinnedBufferWrapper<T, P_alloc, P_free>, P_alloc,
                   P_free>::BufferBase;

  /**
     * @brief Allocate pinned host memory with flags
     *
     * @param num_elements Number of elements to allocate
     * @param flags Flags for gpuHostAlloc (e.g., gpuHostAllocWriteCombined, gpuHostAllocMapped)
     * @param location Source location where allocation was requested
     */
  PinnedBufferWrapper(std::size_t num_elements, unsigned int flags,
                      std::source_location location = std::source_location::current())
      : Base(typename Base::skip_default_alloc_t{}) {
    if (num_elements > Base::max_num_elements) {
      this->policy_alloc_.handle_error(MemoryInvalidValue<MemoryKind::Pinned>::value, location);
      return;
    }
    if (num_elements == 0)
      return;
    const std::size_t size_bytes = num_elements * Base::element_size;
    if (!gpu_check(gpuHostAlloc(reinterpret_cast<void **>(&this->data_), size_bytes, flags),
                   this->policy_alloc_, location)) {
      this->data_ = nullptr;
      return;
    }
    this->num_elements_ = num_elements;
    std::memset(this->data_, 0, size_bytes);
  }

  /** @brief Releases the allocation while this object is still alive */
  ~PinnedBufferWrapper() { this->destroy_(); }

  PinnedBufferWrapper(PinnedBufferWrapper &&) noexcept = default;
  PinnedBufferWrapper &operator=(PinnedBufferWrapper &&) noexcept = default;

  /**
     * @brief Allocate pinned host memory
     *
     * @param ptr Output parameter for the allocated memory pointer
     * @param num_elements Number of elements to allocate
     * @param policy Error policy for allocation
     * @param location Source location where allocation was requested
     *
     * @note Static: it runs from a BufferBase constructor, before this object exists.
     */
  static void allocate(T **ptr, std::size_t num_elements, P_alloc &policy,
                       std::source_location location) {
    *ptr = nullptr;
    const std::size_t size_bytes = num_elements * Base::element_size;
    // Must return on failure: *ptr is still null, and a policy that reports
    // without aborting would otherwise fall through to memset a null pointer.
    if (!gpu_check(gpuHostAlloc(reinterpret_cast<void **>(ptr), size_bytes, gpuHostAllocDefault),
                   policy, location)) {
      *ptr = nullptr;
      return;
    }
    std::memset(*ptr, 0, size_bytes);
  }

  /**
     * @brief Deallocate pinned host memory
     *
     * @param ptr Pointer to memory to deallocate
     * @param num_elements Number of elements (unused, kept for interface consistency)
     */
  void deallocate(T *ptr, std::size_t num_elements) {
    if (ptr != nullptr) {
      gpu_check(gpuFreeHost(ptr), this->policy_free_);
    }
  }
};

/**
 * @brief Convenient alias for PinnedBufferWrapper with default error policies
 *
 * Usage:
 *   PinnedBuffer<float> buffer(1024);  // Allocate 1024 floats in pinned memory
 *   PinnedBuffer<float> mapped_buffer(1024, gpuHostAllocMapped);  // With flags
 */
template<typename T>
using PinnedBuffer = PinnedBufferWrapper<T>;

/**
 * @brief Non-owning view alias for pinned host memory
 *
 * Usage:
 *   PinnedBufferView<float> view(pinned_buffer);           // Full view
 *   PinnedBufferView<float> sub_view(pinned_buffer, 4, 8); // Sub-view: 8 elements starting at offset 4
 */
template<typename T>
using PinnedBufferView = BufferViewWrapper<T, MemoryKind::Pinned>;

} // namespace gpumod::extension
