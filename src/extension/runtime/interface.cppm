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
 * - :convenience_runtime - Default-policy aliases (GpuStream, GpuEvent, GpuMemPool, GpuGraph, GpuGraphExec, and views)
 *
 * Usage:
 *   import wwr.extension.runtime;
 *   using namespace wwr::extension;
 *
 *   GpuStream stream;
 *   GpuEvent event;
 *   GpuMemPool mem_pool;
 */

export module wwr.extension.runtime;

export import :gpu_stream;
export import :gpu_event;
export import :gpu_mem_pool;
export import :gpu_graph_exec;
export import :gpu_graph;
export import :stream_event_pair;
export import :device_handle;
export import :convenience_runtime;
