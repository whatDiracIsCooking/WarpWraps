/**
 * @file tx.cppm
 * @brief Backend-neutral tools extension: wwrtx* names for NVTX / rocTX
 *
 * Exports wwrtx-prefixed aliases of the marker-and-range profiler-annotation
 * core -- NVTX (nvtx3/nvToolsExt.h) on CUDA, rocTX (roctracer/roctx.h) on HIP,
 * whichever backend this build is configured for. See backend.h for the
 * switch. This is the neutral layer src/wrappers is written against; it covers
 * the marker/range surface both vendors share (see src/cuda/nvToolsExt.cppm and
 * src/hip/roctx.cppm for what is deliberately left out of that surface).
 *
 * Each WWR_FUNCTION is written out in full, one per line -- no local
 * prefix-pasting helper -- because test/shared/alias_coverage.py reads these
 * lines to enforce that every alias has a WWR_SAME_FUNCTION restatement in
 * test/gpu/tx.cppm, and it only recognises literal WWR_FUNCTION invocations.
 *
 * Two backend differences are resolved here, not above -- the same way
 * blas.cppm resolves its own:
 *   - the async-range terminator is nvtxRangeEnd on CUDA but roctxRangeStop on
 *     HIP; the neutral name is wwrtxRangeStop (pairing start/stop).
 *   - the range-id type is nvtxRangeId_t on CUDA but roctx_range_id_t on HIP;
 *     both are uint64_t. The neutral name is wwrtxRangeId_t.
 *
 * The A suffix is carried through from the vendor names (nvtxMarkA / roctxMarkA
 * etc.) -- both vendors spell the ASCII entry points that way, so the neutral
 * name is the vendor name with only the prefix swapped. There is no wide-char
 * variant at this layer for the suffix to distinguish against.
 *
 * Usage:
 *   import wwr.tx;
 *
 *   wwrtxRangePushA("phase 1");
 *   // ... work ...
 *   wwrtxRangePop();
 */

module;

#include "backend.h"

export module wwr.tx;

#if defined(WWR_GPU_BACKEND_CUDA)
import wwr.cuda.nvToolsExt;
#else
import wwr.hip.roctx;
#endif

export namespace wwr {

// ========================================================================
// Types
// ========================================================================

// Opaque handle identifying a process-wide asynchronous range (uint64_t). The
// HIP name differs beyond the nvtx/roctx prefix (roctx_range_id_t).
WWR_TYPE(wwrtxRangeId_t, nvtxRangeId_t, roctx_range_id_t)

// ========================================================================
// Markers -- an instantaneous event at a point in time
// ========================================================================

WWR_FUNCTION(wwrtxMarkA, nvtxMarkA, roctxMarkA)

// ========================================================================
// Ranges -- nested (stack) push/pop on the calling thread
// ========================================================================

WWR_FUNCTION(wwrtxRangePushA, nvtxRangePushA, roctxRangePushA)
WWR_FUNCTION(wwrtxRangePop, nvtxRangePop, roctxRangePop)

// ========================================================================
// Ranges -- process-wide asynchronous start/stop
// ========================================================================

WWR_FUNCTION(wwrtxRangeStartA, nvtxRangeStartA, roctxRangeStartA)

// Async-range terminator. The CUDA name is nvtxRangeEnd, not nvtxRangeStop --
// the one name that does not prefix-swap.
WWR_FUNCTION(wwrtxRangeStop, nvtxRangeEnd, roctxRangeStop)

} // namespace wwr
