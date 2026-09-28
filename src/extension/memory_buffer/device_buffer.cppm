/**
 * @file device_buffer.cppm
 * @brief RAII wrapper for GPU device memory buffers
 *
 * Provides DeviceBufferWrapper class for automatic device memory management.
 * Device memory resides on the GPU and is accessible only by device code.
 */

export module wwr.extension.memory_buffer:device_buffer;

import :base_buffer;
import :memory_kind;
import wwr.runtime_api;
import wwr.extension.common;
import wwr.extension.runtime;
import std;

export namespace wwr::extension {

/**
 * @brief RAII wrapper for GPU device memory buffer
 *
 * Allocates device memory on construction and releases it on destruction. The
 * only way to build one is from a shared handle satisfying the device_handle
 * concept: the block is drawn from that handle's memory pool
 * (wwrMallocFromPoolAsync) on its allocation stream, and the handle is retained
 * so both outlive the buffer. H defaults to wwr's DeviceHandle; downstream code
 * may substitute its own conforming handle type. Device memory resides on the
 * GPU and provides the fastest access for device code.
 *
 * @tparam T The element type stored in the buffer
 * @tparam P_alloc Error policy type for allocation
 * @tparam P_free Error policy type for deallocation
 * @tparam H Device handle type backing the pool/stream (defaults to DeviceHandle)
 *
 * @note P_free MUST NOT THROW - it is called from the destructor.
 */
template<typename T, error_policy<wwrError_t> P_alloc,
         nothrow_error_policy<wwrError_t> P_free, device_handle H = DeviceHandle>
class DeviceBufferWrapper
    : public BaseBuffer<T, MemoryKind::Device, DeviceBufferWrapper<T, P_alloc, P_free, H>, P_alloc,
                        P_free> {
private:
  using Base =
      BaseBuffer<T, MemoryKind::Device, DeviceBufferWrapper<T, P_alloc, P_free, H>, P_alloc, P_free>;

public:
  /**
     * @brief Allocate device memory from a DeviceHandle's pool on its stream
     *
     * A DeviceBufferWrapper is always drawn from a shared DeviceHandle -- this and the
     * policy-taking overload below are the only constructors. Makes the handle's
     * device current for the allocation via a DeviceScope guard -- restoring the
     * caller's previous device afterward -- then allocates from its memory pool
     * (wwrMallocFromPoolAsync) on its default allocation stream. The handle
     * is retained (shared_ptr) so the pool and stream outlive this buffer -- the
     * destructor returns the block to the pool on that stream via wwrFreeAsync;
     * see deallocate(). Sharing lives at the handle level: neither pool nor
     * stream is ever owned independently of its DeviceHandle.
     *
     * @param num_elements Number of elements to allocate
     * @param handle Device to allocate on; its mem_pool() backs the block and
     *        its alloc_stream() carries it
     * @param location Source location where allocation was requested
     */
  DeviceBufferWrapper(std::size_t num_elements, std::shared_ptr<H> handle,
                      std::source_location location = std::source_location::current())
      : Base(typename Base::skip_default_alloc_t{}), handle_(std::move(handle)) {
    allocate_from_pool(num_elements, location);
  }

  /**
     * @brief As above, with a custom allocation/deallocation error policy
     *
     * The pool draw is otherwise identical; the policy instance is installed
     * before allocating so it observes any failure. As with BaseBuffer's
     * single-policy constructor, the deallocation policy is a copy of it.
     *
     * @param num_elements Number of elements to allocate
     * @param handle Device to allocate on; its mem_pool() backs the block and
     *        its alloc_stream() carries it
     * @param policy Error policy for both allocation and deallocation
     * @param location Source location where allocation was requested
     */
  DeviceBufferWrapper(std::size_t num_elements, std::shared_ptr<H> handle,
                      P_alloc policy,
                      std::source_location location = std::source_location::current())
      : Base(typename Base::skip_default_alloc_t{}), handle_(std::move(handle)) {
    this->policy_alloc_ = std::move(policy);
    this->policy_free_ = this->policy_alloc_;
    allocate_from_pool(num_elements, location);
  }

  /** @brief Releases the allocation while this object is still alive */
  ~DeviceBufferWrapper() { this->destroy_(); }

  DeviceBufferWrapper(DeviceBufferWrapper &&) noexcept = default;
  DeviceBufferWrapper &operator=(DeviceBufferWrapper &&) noexcept = default;

  /**
     * @brief Return the block to its pool on the handle's stream
     *
     * @param ptr Pointer to memory to deallocate
     * @param num_elements Number of elements (unused, kept for interface consistency)
     *
     * @note A DeviceScope guard makes the handle's device current for the free
     *       and restores the caller's previous device afterward. Released with
     *       wwrFreeAsync on the handle's stream -- the same stream the block was
     *       drawn on. wwrFree performs no implicit synchronisation for a pointer
     *       from wwrMallocFromPoolAsync, so using it here would hand the block
     *       back to the pool while work still queued on the stream was reading
     *       and writing it.
     */
  void deallocate(T *ptr, std::size_t num_elements) {
    if (ptr == nullptr)
      return;
    DeviceScope scope{handle_->index()};
    const wwrStream_t stream = handle_->alloc_stream();
    gpu_check(wwrFreeAsync(ptr, stream), this->policy_free_);
  }

private:
  /// @brief Draw `num_elements` from the retained handle's pool on its stream
  ///
  /// The body shared by both constructors; runs after handle_ (and, for the
  /// policy overload, the error policy) is in place. A DeviceScope guard makes
  /// the handle's device current (restoring the caller's previous device on
  /// return), then allocates from its pool on its allocation stream.
  ///
  /// @param num_elements Number of elements to allocate
  /// @param location Source location where allocation was requested
  void allocate_from_pool(std::size_t num_elements, std::source_location location) {
    if (!should_allocate(num_elements, location))
      return;
    DeviceScope scope{handle_->index(), location};
    const wwrStream_t stream = handle_->alloc_stream();
    const std::size_t size_bytes = num_elements * Base::element_size;
    if (!gpu_check(wwrMallocFromPoolAsync(reinterpret_cast<void **>(&this->data_), size_bytes,
                                          handle_->mem_pool(), stream),
                   this->policy_alloc_, location)) {
      this->data_ = nullptr;
      return;
    }
    this->num_elements_ = num_elements;
    gpu_check(wwrMemsetAsync(this->data_, 0, size_bytes, stream), this->policy_alloc_, location);
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
      this->policy_alloc_.handle_error(MemoryInvalidValue<MemoryKind::Device>::value, location);
      return false;
    }
    return num_elements != 0;
  }

  /// Retains the device handle (and thus its memory pool and allocation stream)
  /// so both outlive this buffer's wwrFreeAsync. Null only in the moved-from state.
  std::shared_ptr<H> handle_;
};

} // namespace wwr::extension
