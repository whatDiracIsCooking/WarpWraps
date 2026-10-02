/**
 * @file unified_buffer.cppm
 * @brief RAII wrapper for unified (managed) memory buffers
 *
 * Provides UnifiedBufferWrapper class for automatic unified memory management.
 * Unified memory is accessible from both CPU and GPU with automatic migration.
 */

export module wwr.extension.memory_buffer:unified_buffer;

import :base_buffer;
import :memory_kind;
import wwr.runtime_api;
import wwr.extension.common;
import wwr.extension.handle;
import std;

export namespace wwr::extension {

/**
 * @brief RAII wrapper for unified (managed) memory buffer
 *
 * Allocates unified memory on construction and releases it on destruction.
 * Unified memory is accessible from both host and device code with automatic page
 * migration, so this buffer is host-accessible (operator[]) like the host and
 * pinned kinds -- but, like DeviceBufferWrapper, it is drawn from a shared
 * device_handle, retained so it outlives the buffer, and its device is made
 * current via a ScopedDeviceIndex guard across both the alloc and the free.
 *
 * Unlike DeviceBufferWrapper, the handle's *tier* does NOT pick the alloc/free
 * calls. Managed memory has only one allocation primitive -- wwrMallocManaged,
 * which is synchronous and takes no stream (there is no async managed alloc) --
 * and it can only be released with the synchronous wwrFree: wwrFreeAsync rejects
 * a managed pointer outright (CUDA cudaErrorNotSupported). So the free is always
 * wwrFree, whatever the handle offers; a stream- or pool-tier handle is accepted
 * (it satisfies device_handle), but its stream/pool are simply unused here. What
 * the handle buys a unified buffer is purely the device binding and the retained
 * lifetime -- not the free call that it dictates for a device buffer.
 *
 * H has no default: this layer ships no concrete handle, so the caller supplies
 * one satisfying device_handle -- a StreamWrapper, the shipped kit::DeviceHandle
 * in wwr.extension.runtime, or a type of their own against the device_handle
 * ladder in wwr.extension.handle. Only dev_idx() is ever read.
 *
 * @tparam T The element type stored in the buffer
 * @tparam P_alloc Error policy type for allocation
 * @tparam P_free Error policy type for deallocation
 * @tparam P_device_access Error policy for the ScopedDeviceIndex guard's device
 *         switch (wwrGetDevice/wwrSetDevice) on the alloc and free paths. As in
 *         DeviceBufferWrapper this is type-level only: a fresh instance is
 *         default-constructed per guard, not stored on the buffer. A failed
 *         *device switch* means the runtime context is already unusable -- a
 *         catastrophic, near-unreachable case needing a reaction, not per-buffer
 *         state.
 * @tparam H Device handle backing the buffer; only dev_idx() is used.
 *
 * @note P_free MUST NOT THROW - it is called from the destructor. P_device_access
 *       carries the same nothrow constraint: the free-path ScopedDeviceIndex's
 *       switch also runs during destruction.
 * @note wwrFree performs an implicit device synchronisation, so the destructor
 *       blocks the host until outstanding work on the managed region is done --
 *       the behaviour that makes the sync free safe, there being no async
 *       alternative for managed memory.
 * @note Unified memory requires compute capability 6.0 or higher for full functionality
 */
template<typename T, error_policy<wwrError_t> P_alloc,
         nothrow_error_policy<wwrError_t> P_free,
         nothrow_error_policy<wwrError_t> P_device_access, device_handle H>
class UnifiedBufferWrapper
    : public BaseBuffer<T, MemoryKind::Unified,
                        UnifiedBufferWrapper<T, P_alloc, P_free, P_device_access, H>, P_alloc,
                        P_free> {
private:
  using Base = BaseBuffer<T, MemoryKind::Unified,
                          UnifiedBufferWrapper<T, P_alloc, P_free, P_device_access, H>, P_alloc,
                          P_free>;

public:
  /**
     * @brief Allocate unified memory on a shared handle
     *
     * Makes the handle's device current via a ScopedDeviceIndex guard -- restoring
     * the caller's previous device afterward -- then allocates with
     * wwrMallocManaged and zero-initialises. The handle is retained (shared_ptr)
     * so it outlives this buffer; the destructor releases the block, see
     * deallocate().
     *
     * @param num_elements Number of elements to allocate
     * @param handle Device to allocate on (only dev_idx() is read)
     * @param flags Flags for wwrMallocManaged (e.g., wwrMemAttachGlobal, wwrMemAttachHost)
     * @param location Source location where allocation was requested
     */
  UnifiedBufferWrapper(std::size_t num_elements, std::shared_ptr<H> handle,
                       unsigned int flags = wwrMemAttachGlobal,
                       std::source_location location = std::source_location::current())
      : Base(typename Base::skip_default_alloc_t{}), handle_(std::move(handle)) {
    allocate_block(num_elements, flags, location);
  }

  /**
     * @brief As above, with a custom allocation/deallocation error policy
     *
     * The policy instance is installed before allocating so it observes any
     * failure. As with BaseBuffer's single-policy constructor, the deallocation
     * policy is a copy of it.
     *
     * @param num_elements Number of elements to allocate
     * @param handle Device to allocate on (only dev_idx() is read)
     * @param policy Error policy for both allocation and deallocation
     * @param flags Flags for wwrMallocManaged (e.g., wwrMemAttachGlobal, wwrMemAttachHost)
     * @param location Source location where allocation was requested
     */
  UnifiedBufferWrapper(std::size_t num_elements, std::shared_ptr<H> handle, P_alloc policy,
                       unsigned int flags = wwrMemAttachGlobal,
                       std::source_location location = std::source_location::current())
      : Base(typename Base::skip_default_alloc_t{}), handle_(std::move(handle)) {
    this->policy_alloc_ = std::move(policy);
    this->free_policy_ref() = this->policy_alloc_;
    allocate_block(num_elements, flags, location);
  }

  /** @brief Releases the allocation while this object is still alive */
  ~UnifiedBufferWrapper() { this->destroy_(); }

  UnifiedBufferWrapper(UnifiedBufferWrapper &&) noexcept = default;
  UnifiedBufferWrapper &operator=(UnifiedBufferWrapper &&) noexcept = default;

  /**
     * @brief Release the managed block with the synchronous wwrFree
     *
     * @param ptr Pointer to memory to deallocate
     * @param num_elements Number of elements (unused, kept for interface consistency)
     *
     * @note A ScopedDeviceIndex guard makes the handle's device current for the
     *       free and restores the caller's previous device afterward. The free is
     *       wwrFree regardless of the handle's tier: managed memory cannot be
     *       released on a stream (wwrFreeAsync rejects a managed pointer), and
     *       wwrFree's implicit device sync is what makes it safe -- and what makes
     *       this destructor block (see the class @note).
     */
  void deallocate(T *ptr, std::size_t num_elements) {
    if (ptr == nullptr)
      return;
    ScopedDeviceIndex<P_device_access> scope{handle_->dev_idx()};
    gpu_check(wwrFree(ptr), this->free_policy_ref());
  }

private:
  /// @brief Allocate `num_elements` of managed memory on the retained handle
  ///
  /// The body shared by both constructors; runs after handle_ (and, for the
  /// policy overload, the error policy) is in place. A ScopedDeviceIndex guard
  /// makes the handle's device current (restoring the caller's previous device on
  /// return), then allocates with wwrMallocManaged and zero-inits with a plain
  /// std::memset -- managed memory is host-addressable, so no device memset is
  /// needed.
  ///
  /// @param num_elements Number of elements to allocate
  /// @param flags Flags for wwrMallocManaged
  /// @param location Source location where allocation was requested
  void allocate_block(std::size_t num_elements, unsigned int flags,
                      std::source_location location) {
    if (!should_allocate(num_elements, location))
      return;
    ScopedDeviceIndex<P_device_access> scope{handle_->dev_idx(), {}, location};
    const std::size_t size_bytes = num_elements * Base::element_size;
    // Must return on failure: data_ is still null, and a policy that reports
    // without aborting would otherwise fall through to memset a null pointer.
    if (!gpu_check(wwrMallocManaged(reinterpret_cast<void **>(&this->data_), size_bytes, flags),
                   this->policy_alloc_, location)) {
      this->data_ = nullptr;
      return;
    }
    this->num_elements_ = num_elements;
    std::memset(this->data_, 0, size_bytes);
  }

  /// @brief Whether to proceed with an allocation of `num_elements`
  ///
  /// The guard the constructor runs before touching the device: an over-large
  /// request is reported through the allocation policy and skipped, and a
  /// zero-element request allocates nothing. Returns true only when a real
  /// allocation should follow.
  ///
  /// @param num_elements Number of elements the caller asked for
  /// @param location Source location where allocation was requested
  bool should_allocate(std::size_t num_elements, std::source_location location) {
    if (num_elements > Base::max_num_elements) {
      this->policy_alloc_.handle_error(MemoryInvalidValue<MemoryKind::Unified>::value, location);
      return false;
    }
    return num_elements != 0;
  }

  /// Retains the device handle so its device binding outlives this buffer's free.
  /// Null only in the moved-from state.
  std::shared_ptr<H> handle_;
};

} // namespace wwr::extension
