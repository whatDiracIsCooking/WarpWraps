/**
 * @file convenience_runtime.cppm
 * @brief Default-policy aliases for the GPU runtime handle wrappers
 *
 * `GpuStream`, `GpuEvent`, `GpuMemPool`, `GpuGraph`, `GpuGraphExec` (and the view
 * aliases) bind the runtime wrappers to the default error policy. This partition
 * is the curated home for those default bindings; alternative-policy aliases
 * belong here too.
 */

export module gpumod.extension.runtime:convenience_runtime;

import :gpu_stream;
import :gpu_event;
import :gpu_mem_pool;
import :gpu_graph;
import :gpu_graph_exec;
import gpumod.runtime_api;
import gpumod.extension.common;
import gpumod.extension.handle;

export namespace wwr::extension {

/**
 * @brief Convenient alias for GpuStreamWrapper with default error policies
 *
 * Usage:
 *   GpuStream stream;  // Instead of GpuStreamWrapper<>
 */
using GpuStream = GpuStreamWrapper<>;

/**
 * @brief Convenient alias for GpuEventWrapper with default error policies
 *
 * Usage:
 *   GpuEvent event;  // Instead of GpuEventWrapper<>
 */
using GpuEvent = GpuEventWrapper<>;

/**
 * @brief Convenient alias for GpuMemPoolWrapper with default error policies
 *
 * Usage:
 *   GpuMemPool pool;  // Instead of GpuMemPoolWrapper<>
 */
using GpuMemPool = GpuMemPoolWrapper<>;

/// @brief Non-owning, copyable view of a memory pool handle (carries its device
///        index). Returned by GpuMemPool::view(); has no borrow-safe operations
///        of its own -- a pool handle is consumed by allocation calls.
using GpuMemPoolView = DeviceBoundHandleView<gpuMemPool_t>;

/**
 * @brief Convenient alias for GpuGraphWrapper with default error policies
 *
 * Usage:
 *   GpuGraph graph;  // Instead of GpuGraphWrapper<>
 */
using GpuGraph = GpuGraphWrapper<>;

/// @brief Non-owning, copyable view of a graph handle. Returned by
///        GpuGraph::view(); a graph is not device-bound, so it carries no device
///        index, and instantiate() stays on the owner (it produces an owned exec).
using GpuGraphView = HandleView<gpuGraph_t>;

/**
 * @brief Convenient alias for GpuGraphExecWrapper with default error policies
 *
 * Usage:
 *   GpuGraphExec exec{graph};  // Instead of GpuGraphExecWrapper<>{graph}
 */
using GpuGraphExec = GpuGraphExecWrapper<>;

} // namespace wwr::extension
