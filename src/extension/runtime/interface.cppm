/**
 * @file interface.cppm
 * @brief Primary interface for wwr.extension.runtime
 *
 * This module provides GPU runtime API extensions (backend-neutral) including:
 * - :gpu_stream - RAII wrapper for GPU streams
 * - :gpu_event - RAII wrapper for GPU events
 * - :gpu_mem_pool - RAII wrapper for GPU memory pools
 * - :gpu_graph - RAII wrapper for GPU graphs
 * - :gpu_graph_exec - RAII wrapper for GPU executable graphs
 * - :device_handle - device identity, properties, default allocation stream and memory pool
 *
 * Usage:
 *   import wwr.extension.runtime;
 *   using namespace wwr::extension;
 *
 *   GpuStreamWrapper<AbortPolicy<wwrError_t>, AbortPolicy<wwrError_t>> stream;
 *   GpuEventWrapper<AbortPolicy<wwrError_t>, AbortPolicy<wwrError_t>> event;
 *   GpuMemPoolWrapper<AbortPolicy<wwrError_t>, AbortPolicy<wwrError_t>> mem_pool;
 */

export module wwr.extension.runtime;

export import :gpu_stream;
export import :gpu_event;
export import :gpu_mem_pool;
export import :gpu_graph_exec;
export import :gpu_graph;
export import :device_handle;
