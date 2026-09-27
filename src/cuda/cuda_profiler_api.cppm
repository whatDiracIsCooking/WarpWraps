/**
 * @file cuda_profiler_api.cppm
 * @brief Primary interface for wwr.cuda.cuda_profiler_api
 *
 * This module wraps the CUDA Runtime profiler control API and exports the two
 * programmatic profiling boundary functions: cudaProfilerStart and
 * cudaProfilerStop.  These allow host code to demarcate the regions of interest
 * for Nsight / nvprof collection without modifying launch parameters.
 *
 * Usage:
 *   import wwr.cuda.cuda_profiler_api;
 */

module;

#include <cuda_profiler_api.h>

export module wwr.cuda.cuda_profiler_api;

// ========================================================================
// Export all cuda_profiler_api types and functions in wwr namespace
// ========================================================================

export namespace wwr::cuda {

// ========================================================================
// Dependent types from driver_types.h that appear in function signatures
// ========================================================================

using ::cudaError_t;

// ========================================================================
// Profiler Control Functions
// ========================================================================

// Enable profiling for the current CUDA context.  If profiling is already
// enabled this call has no effect.  Returns cudaSuccess on success.
using ::cudaProfilerStart;

// Disable profiling for the current CUDA context.  If profiling is already
// disabled this call has no effect.  Returns cudaSuccess on success.
using ::cudaProfilerStop;

} // namespace wwr::cuda
