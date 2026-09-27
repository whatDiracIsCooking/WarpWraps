// tx.cppm - Compile-time tests for gpumod.tx
//
// Every exported gputx* name is checked against the backend's own entity: the
// same range-id type, and the same function (see gpu_check_macros.h). The
// surface is the shared marker/range core, short enough to list in full.
//
// Two names carry the backend difference tx.cppm resolves: the async-range
// terminator gputxRangeStop maps to nvtxRangeEnd on CUDA (not "Stop"), and the
// range-id type gputxRangeId_t maps to roctx_range_id_t on HIP (not
// "roctxRangeId_t") -- this pins both to the right backend entity.

module;

#include "gpu_check_macros.h"

export module gpumod.test.gpu.tx;

import std;
import gpumod.tx;
#if defined(GPUMOD_GPU_BACKEND_CUDA)
import gpumod.cuda.nvToolsExt;
#else
import gpumod.hip.roctx;
#endif

namespace gpumod::test {

using namespace gpumod;

#if defined(GPUMOD_GPU_BACKEND_CUDA)

using namespace gpumod::cuda;

// ────────────────────────────────────────────────────────────────────────
// CUDA backend
// ────────────────────────────────────────────────────────────────────────

GPUMOD_SAME_TYPE(gputxRangeId_t, nvtxRangeId_t)

GPUMOD_SAME_FUNCTION(gputxMarkA, nvtxMarkA)
GPUMOD_SAME_FUNCTION(gputxRangePushA, nvtxRangePushA)
GPUMOD_SAME_FUNCTION(gputxRangePop, nvtxRangePop)
GPUMOD_SAME_FUNCTION(gputxRangeStartA, nvtxRangeStartA)
// gputxRangeStop maps to nvtxRangeEnd -- the one name that does not prefix-swap.
GPUMOD_SAME_FUNCTION(gputxRangeStop, nvtxRangeEnd)

#else

// ────────────────────────────────────────────────────────────────────────
// HIP backend
// ────────────────────────────────────────────────────────────────────────

using namespace gpumod::hip;

GPUMOD_SAME_TYPE(gputxRangeId_t, roctx_range_id_t)

GPUMOD_SAME_FUNCTION(gputxMarkA, roctxMarkA)
GPUMOD_SAME_FUNCTION(gputxRangePushA, roctxRangePushA)
GPUMOD_SAME_FUNCTION(gputxRangePop, roctxRangePop)
GPUMOD_SAME_FUNCTION(gputxRangeStartA, roctxRangeStartA)
GPUMOD_SAME_FUNCTION(gputxRangeStop, roctxRangeStop)

#endif

} // namespace gpumod::test
