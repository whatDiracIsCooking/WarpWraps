/**
 * @file tx.cppm
 * @brief Backend-neutral tools extension: gputx* names for NVTX / rocTX
 *
 * Exports gputx-prefixed aliases of the marker-and-range profiler-annotation
 * core -- NVTX (nvtx3/nvToolsExt.h) on CUDA, rocTX (roctracer/roctx.h) on HIP,
 * whichever backend this build is configured for. See gpu_backend.h for the
 * switch. This is the neutral layer src/wrappers is written against; it covers
 * the marker/range surface both vendors share (see src/cuda/nvToolsExt.cppm and
 * src/hip/roctx.cppm for what is deliberately left out of that surface).
 *
 * Each GPUMOD_FUNCTION is written out in full, one per line -- no local
 * prefix-pasting helper -- because test/shared/alias_coverage.py reads these
 * lines to enforce that every alias has a GPUMOD_SAME_FUNCTION restatement in
 * test/gpu/tx.cppm, and it only recognises literal GPUMOD_FUNCTION invocations.
 *
 * Two backend differences are resolved here, not above -- the same way
 * blas.cppm resolves its own:
 *   - the async-range terminator is nvtxRangeEnd on CUDA but roctxRangeStop on
 *     HIP; the neutral name is gputxRangeStop (pairing start/stop).
 *   - the range-id type is nvtxRangeId_t on CUDA but roctx_range_id_t on HIP;
 *     both are uint64_t. The neutral name is gputxRangeId_t.
 *
 * The A suffix is carried through from the vendor names (nvtxMarkA / roctxMarkA
 * etc.) -- both vendors spell the ASCII entry points that way, so the neutral
 * name is the vendor name with only the prefix swapped. There is no wide-char
 * variant at this layer for the suffix to distinguish against.
 *
 * Usage:
 *   import gpumod.tx;
 *
 *   gputxRangePushA("phase 1");
 *   // ... work ...
 *   gputxRangePop();
 */

module;

#include "gpu_backend.h"

export module gpumod.tx;

#if defined(GPUMOD_GPU_BACKEND_CUDA)
import gpumod.cuda.nvToolsExt;
#else
import gpumod.hip.roctx;
#endif

export namespace gpumod {

// ========================================================================
// Types
// ========================================================================

// Opaque handle identifying a process-wide asynchronous range (uint64_t). The
// HIP name differs beyond the nvtx/roctx prefix (roctx_range_id_t).
GPUMOD_TYPE(gputxRangeId_t, nvtxRangeId_t, roctx_range_id_t)

// ========================================================================
// Markers -- an instantaneous event at a point in time
// ========================================================================

GPUMOD_FUNCTION(gputxMarkA, nvtxMarkA, roctxMarkA)

// ========================================================================
// Ranges -- nested (stack) push/pop on the calling thread
// ========================================================================

GPUMOD_FUNCTION(gputxRangePushA, nvtxRangePushA, roctxRangePushA)
GPUMOD_FUNCTION(gputxRangePop, nvtxRangePop, roctxRangePop)

// ========================================================================
// Ranges -- process-wide asynchronous start/stop
// ========================================================================

GPUMOD_FUNCTION(gputxRangeStartA, nvtxRangeStartA, roctxRangeStartA)

// Async-range terminator. The CUDA name is nvtxRangeEnd, not nvtxRangeStop --
// the one name that does not prefix-swap.
GPUMOD_FUNCTION(gputxRangeStop, nvtxRangeEnd, roctxRangeStop)

} // namespace gpumod
