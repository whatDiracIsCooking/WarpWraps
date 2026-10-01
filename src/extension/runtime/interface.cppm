/**
 * @file interface.cppm
 * @brief Primary interface for wwr.extension.runtime
 *
 * This module provides GPU runtime API extensions (backend-neutral) including:
 * - :stream - RAII wrapper for GPU streams
 * - :event - RAII wrapper for GPU events
 * - :mem_pool - RAII wrapper for GPU memory pools
 * - :graph - RAII wrapper for GPU graphs
 * - :graph_exec - RAII wrapper for GPU executable graphs
 * - :device_handle - DeviceHandle, a ready-made pool-tier device handle, so a
 *   consumer need not hand-roll one (the stream tier is just StreamWrapper)
 *
 * Usage:
 *   import wwr.extension.runtime;
 *   using namespace wwr::extension;
 *
 *   // AbortPolicy here is your own abort-on-failure policy; the library ships none.
 *   // These wrappers are device-bound, so the third arg is the device-access policy.
 *   StreamWrapper<AbortPolicy<wwrError_t>, AbortPolicy<wwrError_t>, AbortPolicy<wwrError_t>> stream;
 *   EventWrapper<AbortPolicy<wwrError_t>, AbortPolicy<wwrError_t>, AbortPolicy<wwrError_t>> event;
 *   MemPoolWrapper<AbortPolicy<wwrError_t>, AbortPolicy<wwrError_t>, AbortPolicy<wwrError_t>> mem_pool;
 */

export module wwr.extension.runtime;

export import :stream;
export import :event;
export import :mem_pool;
export import :graph_exec;
export import :graph;
export import :device_handle;
