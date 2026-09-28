/**
 * @file device_handle.cppm
 * @brief The device-handle capability ladder that backs a DeviceBuffer
 */

export module wwr.extension.handle:device_handle;

import wwr.runtime_api;
import std;

export namespace wwr::extension {

/**
 * @brief The device-handle capability ladder backing a DeviceBuffer
 *
 * Three refining concepts, each adding one accessor and thereby unlocking one
 * more allocation strategy in DeviceBufferWrapper:
 *
 *   device_handle         dev_idx()   -> synchronous wwrMalloc / wwrFree
 *   device_handle_stream  + stream()  -> wwrMallocAsync / wwrFreeAsync
 *   device_handle_pool    + pool()    -> wwrMallocFromPoolAsync (+ async free)
 *
 * Structural, like error_policy: any type exposing the accessors qualifies.
 * They yield raw backend handles, so downstream code can model a tier with its
 * own type -- returning a raw wwrStream_t / wwrMemPool_t is fine, and
 * GpuStream / GpuMemPool satisfy it via their implicit conversions
 * (BaseHandle::operator T). The requirement is stated as convertible_to, never
 * a specific `.get()`, precisely to keep that raw-handle path valid.
 *
 * The ladder lives here, alongside the RAII handle machinery it describes,
 * rather than in wwr.extension.runtime: it names only raw backend types
 * (wwrStream_t / wwrMemPool_t), so it needs none of the stream/pool wrappers,
 * and keeping it here lets DeviceBuffer constrain its handle axis without
 * pulling the runtime module in.
 *
 * @note The accessors are required noexcept because DeviceBuffer reads them on
 *       the destructor (deallocate) path -- the same destructor-safety rule
 *       that motivates nothrow_error_policy.
 */
template<typename H>
concept device_handle = requires(const H h) {
  { h.dev_idx() } noexcept -> std::convertible_to<int>;
};

/// @brief A device_handle that also offers a stream -> enables async alloc/free
template<typename H>
concept device_handle_stream = device_handle<H> && requires(const H h) {
  { h.stream() } noexcept -> std::convertible_to<wwrStream_t>;
};

/// @brief A device_handle_stream that also offers a pool -> enables pool alloc
template<typename H>
concept device_handle_pool = device_handle_stream<H> && requires(const H h) {
  { h.pool() } noexcept -> std::convertible_to<wwrMemPool_t>;
};

} // namespace wwr::extension
