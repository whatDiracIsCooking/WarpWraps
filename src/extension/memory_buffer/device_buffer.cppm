/**
 * @file device_buffer.cppm
 * @brief RAII wrapper for GPU device memory buffers
 *
 * Provides DeviceBuffer class for automatic device memory management.
 * Device memory resides on the GPU and is accessible only by device code.
 */

export module gpumod.extension.memory_buffer:device_buffer;

import :buffer_base;
import :memory_kind;
import gpumod.runtime_api;
import gpumod.extension.common;
import gpumod.extension.runtime;
import std;

export namespace gpumod::extension {

/**
 * @brief RAII wrapper for GPU device memory buffer
 *
 * Allocates device memory on construction and releases it on destruction. The
 * only way to build one is from a shared DeviceHandle: the block is drawn from
 * that handle's memory pool (gpuMallocFromPoolAsync) on its allocation stream,
 * and the handle is retained so both outlive the buffer. Device memory resides
 * on the GPU and provides the fastest access for device code.
 *
 * @tparam T The element type stored in the buffer
 * @tparam P_alloc Error policy type for allocation (defaults to DefaultErrorPolicy<gpuError_t>)
 * @tparam P_free Error policy type for deallocation (defaults to P_alloc)
 *
 * @note P_free MUST NOT THROW - it is called from the destructor.
 */
template<typename T, error_policy<gpuError_t> P_alloc = DefaultErrorPolicy<gpuError_t>,
         nothrow_error_policy<gpuError_t> P_free = P_alloc>
class DeviceBufferWrapper
    : public BufferBase<T, MemoryKind::Device, DeviceBufferWrapper<T, P_alloc, P_free>, P_alloc,
                        P_free> {
private:
  using Base =
      BufferBase<T, MemoryKind::Device, DeviceBufferWrapper<T, P_alloc, P_free>, P_alloc, P_free>;

public:
  /**
     * @brief Allocate device memory from a DeviceHandle's pool on its stream
     *
     * A DeviceBuffer is always drawn from a shared DeviceHandle -- this and the
     * policy-taking overload below are the only constructors. Makes the handle's
     * device current for the allocation via a DeviceScope guard -- restoring the
     * caller's previous device afterward -- then allocates from its memory pool
     * (gpuMallocFromPoolAsync) on its default allocation stream. The handle
     * is retained (shared_ptr) so the pool and stream outlive this buffer -- the
     * destructor returns the block to the pool on that stream via gpuFreeAsync;
     * see deallocate(). Sharing lives at the handle level: neither pool nor
     * stream is ever owned independently of its DeviceHandle.
     *
     * @param num_elements Number of elements to allocate
     * @param handle Device to allocate on; its mem_pool() backs the block and
     *        its alloc_stream() carries it
     * @param location Source location where allocation was requested
     */
  DeviceBufferWrapper(std::size_t num_elements, std::shared_ptr<DeviceHandle> handle,
                      std::source_location location = std::source_location::current())
      : Base(typename Base::skip_default_alloc_t{}), handle_(std::move(handle)) {
    allocate_from_pool(num_elements, location);
  }

  /**
     * @brief As above, with a custom allocation/deallocation error policy
     *
     * The pool draw is otherwise identical; the policy instance is installed
     * before allocating so it observes any failure. As with BufferBase's
     * single-policy constructor, the deallocation policy is a copy of it.
     *
     * @param num_elements Number of elements to allocate
     * @param handle Device to allocate on; its mem_pool() backs the block and
     *        its alloc_stream() carries it
     * @param policy Error policy for both allocation and deallocation
     * @param location Source location where allocation was requested
     */
  DeviceBufferWrapper(std::size_t num_elements, std::shared_ptr<DeviceHandle> handle,
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
     *       gpuFreeAsync on the handle's stream -- the same stream the block was
     *       drawn on. gpuFree performs no implicit synchronisation for a pointer
     *       from gpuMallocFromPoolAsync, so using it here would hand the block
     *       back to the pool while work still queued on the stream was reading
     *       and writing it.
     */
  void deallocate(T *ptr, std::size_t num_elements) {
    if (ptr == nullptr)
      return;
    DeviceScope scope{handle_->index()};
    gpu_check(gpuFreeAsync(ptr, handle_->alloc_stream().get()), this->policy_free_);
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
    const gpuStream_t stream = handle_->alloc_stream().get();
    const std::size_t size_bytes = num_elements * Base::element_size;
    if (!gpu_check(gpuMallocFromPoolAsync(reinterpret_cast<void **>(&this->data_), size_bytes,
                                          handle_->mem_pool(), stream),
                   this->policy_alloc_, location)) {
      this->data_ = nullptr;
      return;
    }
    this->num_elements_ = num_elements;
    gpu_check(gpuMemsetAsync(this->data_, 0, size_bytes, stream), this->policy_alloc_, location);
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

  /// Retains the DeviceHandle (and thus its memory pool and allocation stream)
  /// so both outlive this buffer's gpuFreeAsync. Null only in the moved-from state.
  std::shared_ptr<DeviceHandle> handle_;
};

/**
 * @brief Convenient alias for DeviceBufferWrapper with default error policies
 *
 * Usage:
 *   auto device = std::make_shared<DeviceHandle>(0);
 *   DeviceBuffer<float> buffer(1024, device);  // 1024 floats from the handle's pool
 */
template<typename T>
using DeviceBuffer = DeviceBufferWrapper<T>;

/**
 * @brief Non-owning view alias for device memory
 *
 * Usage:
 *   DeviceBufferView<float> view(device_buffer);           // Full view
 *   DeviceBufferView<float> sub_view(device_buffer, 4, 8); // Sub-view: 8 elements starting at offset 4
 */
template<typename T>
using DeviceBufferView = BufferViewWrapper<T, MemoryKind::Device>;

} // namespace gpumod::extension
