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
import wwr.extension.handle;
import std;

export namespace wwr::extension {

/**
 * @brief RAII wrapper for GPU device memory buffer
 *
 * Allocates device memory on construction and releases it on destruction. The
 * only way to build one is from a shared handle, which is retained so it
 * outlives the buffer. The allocation strategy is chosen at compile time by how
 * much the handle offers -- the device_handle capability ladder:
 *
 *   - device_handle_pool   -> wwrMallocFromPoolAsync from pool() on stream()
 *   - device_handle_stream -> wwrMallocAsync on stream()
 *   - device_handle (only) -> synchronous wwrMalloc
 *
 * H has no default: this layer ships no concrete handle, so the caller supplies
 * one satisfying the tier they want (see the device_handle ladder in
 * wwr.extension.handle). Device memory resides on the GPU and provides the
 * fastest access for device code.
 *
 * @tparam T The element type stored in the buffer
 * @tparam P_alloc Error policy type for allocation
 * @tparam P_free Error policy type for deallocation
 * @tparam H Device handle backing the buffer; its tier picks the strategy.
 * @tparam P_device_access Error policy for the DeviceScope guard's device
 *         switch (wwrGetDevice/wwrSetDevice) on the alloc and free paths. Unlike
 *         P_alloc/P_free, this is type-level only: a fresh instance is
 *         default-constructed for each guard, not stored on the buffer. An
 *         alloc/free failure is expected and handled, so its policy accumulates
 *         state per buffer; a failed *device switch* means the runtime context
 *         is already unusable -- a catastrophic, near-unreachable case whose
 *         policy needs no per-buffer state, only a reaction.
 *
 * @note P_free MUST NOT THROW - it is called from the destructor. P_device_access
 *       carries the same nothrow constraint: the free-path DeviceScope's switch
 *       also runs during destruction.
 * @warning The synchronous (device_handle-only) tier frees with wwrFree, which
 *          implicitly synchronises the whole device -- so its destructor blocks
 *          the host. Prefer a stream-bearing handle in hot alloc/free paths.
 */
template<typename T, error_policy<wwrError_t> P_alloc,
         nothrow_error_policy<wwrError_t> P_free, device_handle H,
         nothrow_error_policy<wwrError_t> P_device_access>
class DeviceBufferWrapper
    : public BaseBuffer<T, MemoryKind::Device,
                        DeviceBufferWrapper<T, P_alloc, P_free, H, P_device_access>, P_alloc,
                        P_free> {
private:
  using Base = BaseBuffer<T, MemoryKind::Device,
                          DeviceBufferWrapper<T, P_alloc, P_free, H, P_device_access>, P_alloc,
                          P_free>;

public:
  /**
     * @brief Allocate device memory on a shared handle
     *
     * A DeviceBufferWrapper is always drawn from a shared handle -- this and the
     * policy-taking overload below are the only constructors. Makes the handle's
     * device current via a DeviceScope guard -- restoring the caller's previous
     * device afterward -- then allocates by whichever strategy the handle's tier
     * selects (see the class comment and allocate_block()). The handle is
     * retained (shared_ptr) so whatever backs the allocation -- stream, pool --
     * outlives this buffer; the destructor releases the block, see deallocate().
     *
     * @param num_elements Number of elements to allocate
     * @param handle Device to allocate on; its tier picks the allocation call
     * @param location Source location where allocation was requested
     */
  DeviceBufferWrapper(std::size_t num_elements, std::shared_ptr<H> handle,
                      std::source_location location = std::source_location::current())
      : Base(typename Base::skip_default_alloc_t{}), handle_(std::move(handle)) {
    allocate_block(num_elements, location);
  }

  /**
     * @brief As above, with a custom allocation/deallocation error policy
     *
     * The pool draw is otherwise identical; the policy instance is installed
     * before allocating so it observes any failure. As with BaseBuffer's
     * single-policy constructor, the deallocation policy is a copy of it.
     *
     * @param num_elements Number of elements to allocate
     * @param handle Device to allocate on; its tier picks the allocation call
     * @param policy Error policy for both allocation and deallocation
     * @param location Source location where allocation was requested
     */
  DeviceBufferWrapper(std::size_t num_elements, std::shared_ptr<H> handle,
                      P_alloc policy,
                      std::source_location location = std::source_location::current())
      : Base(typename Base::skip_default_alloc_t{}), handle_(std::move(handle)) {
    this->policy_alloc_ = std::move(policy);
    this->policy_free_ = this->policy_alloc_;
    allocate_block(num_elements, location);
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
     *       and restores the caller's previous device afterward. A stream-bearing
     *       handle releases with wwrFreeAsync on the handle's stream -- the same
     *       stream the block was drawn on; wwrFree there would perform no implicit
     *       sync for a stream-ordered pointer and so hand the block back while work
     *       queued on the stream still read and wrote it. A stream-less handle has
     *       only wwrFree, whose implicit device sync is what makes it safe (and
     *       what makes this destructor block -- see the class @warning).
     */
  void deallocate(T *ptr, std::size_t num_elements) {
    if (ptr == nullptr)
      return;
    DeviceScope<P_device_access> scope{handle_->dev_idx()};
    if constexpr (device_handle_stream<H>)
      gpu_check(wwrFreeAsync(ptr, handle_->stream()), this->policy_free_);
    else
      gpu_check(wwrFree(ptr), this->policy_free_);
  }

private:
  /// @brief Allocate `num_elements` on the retained handle by its tier's strategy
  ///
  /// The body shared by both constructors; runs after handle_ (and, for the
  /// policy overload, the error policy) is in place. A DeviceScope guard makes
  /// the handle's device current (restoring the caller's previous device on
  /// return), then allocates by the compile-time-selected strategy and zero-inits
  /// the block. The pool tier draws from pool() on stream(); the stream tier from
  /// the device default pool on stream(); the bare tier synchronously. The
  /// zero-init follows the same async/sync split.
  ///
  /// @param num_elements Number of elements to allocate
  /// @param location Source location where allocation was requested
  void allocate_block(std::size_t num_elements, std::source_location location) {
    if (!should_allocate(num_elements, location))
      return;
    DeviceScope<P_device_access> scope{handle_->dev_idx(), {}, location};
    const std::size_t size_bytes = num_elements * Base::element_size;
    auto ptr = reinterpret_cast<void **>(&this->data_);
    bool ok;
    if constexpr (device_handle_pool<H>)
      ok = gpu_check(wwrMallocFromPoolAsync(ptr, size_bytes, handle_->pool(), handle_->stream()),
                     this->policy_alloc_, location);
    else if constexpr (device_handle_stream<H>)
      ok = gpu_check(wwrMallocAsync(ptr, size_bytes, handle_->stream()), this->policy_alloc_,
                     location);
    else
      ok = gpu_check(wwrMalloc(ptr, size_bytes), this->policy_alloc_, location);
    if (!ok) {
      this->data_ = nullptr;
      return;
    }
    this->num_elements_ = num_elements;
    if constexpr (device_handle_stream<H>)
      gpu_check(wwrMemsetAsync(this->data_, 0, size_bytes, handle_->stream()), this->policy_alloc_,
                location);
    else
      gpu_check(wwrMemset(this->data_, 0, size_bytes), this->policy_alloc_, location);
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

  /// Retains the device handle (and thus whatever backs the allocation -- pool,
  /// stream) so it outlives this buffer's free. Null only in the moved-from state.
  std::shared_ptr<H> handle_;
};

} // namespace wwr::extension
