/**
 * @file unified_buffer.cppm
 * @brief RAII wrapper for unified (managed) memory buffers
 *
 * Provides UnifiedBuffer class for automatic unified memory management.
 * Unified memory is accessible from both CPU and GPU with automatic migration.
 */

export module gpumod.extension.memory_buffer:unified_buffer;

import :base_buffer;
import :memory_kind;
import gpumod.runtime_api;
import gpumod.extension.common;
import gpumod.extension.runtime;
import std;

export namespace gpumod::extension {

/**
 * @brief RAII wrapper for unified (managed) memory buffer
 *
 * Automatically allocates unified memory on construction and deallocates it on destruction.
 * Unified memory is accessible from both host and device code with automatic page migration.
 * Simplifies memory management at the cost of potential performance overhead.
 *
 * @tparam T The element type stored in the buffer
 * @tparam P_alloc Error policy type for allocation (defaults to DefaultErrorPolicy<gpuError_t>)
 * @tparam P_free Error policy type for deallocation (defaults to P_alloc)
 *
 * @note P_free MUST NOT THROW - it is called from the destructor.
 * @note Unified memory requires compute capability 6.0 or higher for full functionality
 */
template<typename T, error_policy<gpuError_t> P_alloc = DefaultErrorPolicy<gpuError_t>,
         nothrow_error_policy<gpuError_t> P_free = P_alloc>
class UnifiedBufferWrapper
    : public BaseBuffer<T, MemoryKind::Unified, UnifiedBufferWrapper<T, P_alloc, P_free>, P_alloc,
                        P_free> {
private:
  using Base =
      BaseBuffer<T, MemoryKind::Unified, UnifiedBufferWrapper<T, P_alloc, P_free>, P_alloc, P_free>;

public:
  // Inherit constructors from base
  using BaseBuffer<T, MemoryKind::Unified, UnifiedBufferWrapper<T, P_alloc, P_free>, P_alloc,
                   P_free>::BaseBuffer;

  /**
     * @brief Allocate unified memory with flags
     *
     * @param num_elements Number of elements to allocate
     * @param flags Flags for gpuMallocManaged (e.g., gpuMemAttachGlobal, gpuMemAttachHost)
     * @param location Source location where allocation was requested
     */
  UnifiedBufferWrapper(std::size_t num_elements, unsigned int flags,
                       std::source_location location = std::source_location::current())
      : Base(typename Base::skip_default_alloc_t{}) {
    if (num_elements > Base::max_num_elements) {
      this->policy_alloc_.handle_error(MemoryInvalidValue<MemoryKind::Unified>::value, location);
      return;
    }
    if (num_elements == 0)
      return;
    const std::size_t size_bytes = num_elements * Base::element_size;
    if (!gpu_check(gpuMallocManaged(reinterpret_cast<void **>(&this->data_), size_bytes, flags),
                   this->policy_alloc_, location)) {
      this->data_ = nullptr;
      return;
    }
    this->num_elements_ = num_elements;
    std::memset(this->data_, 0, size_bytes);
  }

  /** @brief Releases the allocation while this object is still alive */
  ~UnifiedBufferWrapper() { this->destroy_(); }

  UnifiedBufferWrapper(UnifiedBufferWrapper &&) noexcept = default;
  UnifiedBufferWrapper &operator=(UnifiedBufferWrapper &&) noexcept = default;

  /**
     * @brief Allocate unified memory
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
    // Must return on failure: *ptr is still null, and a policy that reports
    // without aborting would otherwise fall through to memset a null pointer.
    if (!gpu_check(gpuMallocManaged(reinterpret_cast<void **>(ptr), size_bytes, gpuMemAttachGlobal),
                   policy, location)) {
      *ptr = nullptr;
      return;
    }
    std::memset(*ptr, 0, size_bytes);
  }

  /**
     * @brief Deallocate unified memory
     *
     * @param ptr Pointer to memory to deallocate
     * @param num_elements Number of elements (unused, kept for interface consistency)
     */
  void deallocate(T *ptr, std::size_t num_elements) {
    if (ptr != nullptr) {
      gpu_check(gpuFree(ptr), this->policy_free_);
    }
  }
};

} // namespace gpumod::extension
