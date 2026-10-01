/**
 * @file comp_manager.cppm
 * @brief RAII owner of the device scratch buffer the batched LLIF borrows
 *
 * ## Scope -- what this wraps, and why it is the honest neutral surface
 *
 * The issue asked for "RAII over the nvCOMP/hipCOMP manager (and any required
 * temp/scratch allocation handle)". But wwr.comp is NOT a manager object: it is a
 * stateless batched low-level interface (LLIF) -- a family of free
 * wwrcompBatched<Algo>* functions. The high-level C++ *manager* classes
 * (nvcompManagerBase & friends) live only in the backend-specific
 * wwr.cuda.nvcomp / wwr.hip.hipcomp, whose APIs diverge and are not neutral, so
 * there is no neutral manager OBJECT to wrap. Fabricating a wrapper over a
 * backend-specific class would not be backend-portable, which is the whole point
 * of this layer.
 *
 * What DOES exist neutrally, and is the one resource the batched calls genuinely
 * require, is the caller-allocated device TEMP/SCRATCH buffer: every
 * wwrcompBatched<Algo>CompressAsync / DecompressAsync takes a (device_temp_ptr,
 * temp_bytes) pair the caller must size (via the matching
 * ...CompressGetTempSize / ...DecompressGetTempSize) and allocate. CompScratch is
 * RAII over exactly that buffer -- acquire a sized device block on construction,
 * free it on destruction, hand it to the batched call as a borrow in between.
 *
 * It does NOT re-roll wwrMalloc/wwrFree: it composes the existing device-buffer
 * RAII (DeviceBufferWrapper<std::byte, ...>), so it inherits the handle
 * capability-ladder allocation strategy (pool / stream / synchronous), the
 * device-access guard, the move-only / destroy-exactly-once contract and the
 * error-policy surface already proven in wwr.extension.memory_buffer. CompScratch
 * adds only the compression-specific front: a factory that sizes the block from a
 * ...GetTempSize query, and bytes()/data() borrow accessors shaped for the
 * batched signatures.
 *
 * Usage:
 *   import wwr.extension.comp;
 *   using namespace wwr::extension;
 *
 *   // Size the scratch from the matching temp-size query, then let the batched
 *   // call borrow it:
 *   std::size_t temp_bytes = 0;
 *   wwrcompBatchedLZ4CompressGetTempSize(batch, max_chunk, opts, &temp_bytes);
 *   CompScratch<Abort, Abort, Abort, Handle> scratch(temp_bytes, handle);
 *   wwrcompBatchedLZ4CompressAsync(..., scratch.data(), scratch.bytes(), ...);
 */

export module wwr.extension.comp:comp_manager;

import :comp_error; // success_code<wwrcompStatus_t> et al. travel with the module
import wwr.comp;
import wwr.runtime_api; // wwrError_t (the device-buffer error type)
import wwr.extension.common;
import wwr.extension.handle;        // device_handle ladder
import wwr.extension.memory_buffer; // DeviceBufferWrapper
import std;

export namespace wwr::extension {

/**
 * @brief RAII owner of the device temp/scratch buffer the batched LLIF borrows
 *
 * Owns a device byte block sized for a batched compress/decompress call, drawn
 * from a shared device handle whose tier picks the allocation strategy (pool /
 * stream / synchronous), exactly as a DeviceBufferWrapper -- which it holds. The
 * block is freed on destruction; move-only, copy deleted, with the
 * destroy-exactly-once contract inherited from the underlying buffer.
 *
 * A zero-byte request (a query returning temp_bytes == 0, which happens for some
 * algorithm/shape pairs) allocates nothing and leaves data() == nullptr with
 * bytes() == 0 -- the exact (nullptr, 0) pair the batched calls accept for "no
 * temp needed".
 *
 * @tparam P_alloc Error policy for the device allocation (wwrError_t-typed)
 * @tparam P_free Error policy for the device free; runs from the destructor, so
 *         it must be a nothrow_error_policy
 * @tparam P_device_access Error policy for the alloc/free-path device switch
 *         (wwrError_t-typed, nothrow -- the free path runs during destruction)
 * @tparam H Device handle backing the allocation; its tier picks the strategy
 *
 * @note The error type here is wwrError_t, not wwrcompStatus_t: the scratch is a
 *       plain device allocation, and allocating it goes through the runtime
 *       allocator. wwrcompStatus_t is what the batched *calls themselves* return,
 *       and comp_error specializes the error-handling templates for it so a
 *       caller checks those with gpu_check(status, policy) directly -- the two
 *       error domains stay distinct, as they do everywhere in wwr.
 */
template<error_policy<wwrError_t> P_alloc, nothrow_error_policy<wwrError_t> P_free,
         nothrow_error_policy<wwrError_t> P_device_access, device_handle H>
class CompScratch {
public:
  /// @brief The underlying device-buffer type this scratch owns
  using buffer_type = DeviceBufferWrapper<std::byte, P_alloc, P_free, P_device_access, H>;

  /**
   * @brief Allocate a scratch block of `size_bytes` on a shared device handle
   *
   * @param size_bytes Byte size the batched call's temp-size query returned
   * @param handle Device to allocate on; its tier picks the allocation call
   * @param location Source location where allocation was requested
   *
   * @note size_bytes == 0 allocates nothing (data() == nullptr, bytes() == 0).
   */
  CompScratch(std::size_t size_bytes, std::shared_ptr<H> handle,
              std::source_location location = std::source_location::current())
      : buffer_(size_bytes, std::move(handle), location) {}

  /**
   * @brief As above, with a custom allocation/deallocation error policy
   *
   * @param size_bytes Byte size the batched call's temp-size query returned
   * @param handle Device to allocate on; its tier picks the allocation call
   * @param policy Error policy for both allocation and deallocation
   * @param location Source location where allocation was requested
   */
  CompScratch(std::size_t size_bytes, std::shared_ptr<H> handle, P_alloc policy,
              std::source_location location = std::source_location::current())
      : buffer_(size_bytes, std::move(handle), std::move(policy), location) {}

  /// @brief Device pointer to the scratch block, or nullptr when empty.
  ///        Shaped for the batched calls' `void *device_temp_ptr` parameter.
  void *data() noexcept { return buffer_.data(); }

  /// @brief Const device pointer to the scratch block, or nullptr when empty.
  const void *data() const noexcept { return buffer_.data(); }

  /// @brief Scratch size in bytes -- the batched calls' `temp_bytes` argument.
  std::size_t bytes() const noexcept { return buffer_.size_bytes(); }

  /// @brief Whether a non-empty block is held (false for a zero-byte request).
  bool empty() const noexcept { return buffer_.num_elements() == 0; }

  /// @brief The owned device buffer, for callers wanting its richer surface.
  buffer_type &buffer() noexcept { return buffer_; }
  const buffer_type &buffer() const noexcept { return buffer_; }

private:
  /// The device-buffer RAII this scratch is a thin, compression-shaped front for.
  /// Move-only and destroy-exactly-once by construction, so CompScratch inherits
  /// both without restating them.
  buffer_type buffer_;
};

} // namespace wwr::extension
