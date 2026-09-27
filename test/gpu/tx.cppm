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
#if defined(WWR_GPU_BACKEND_CUDA)
import gpumod.cuda.nvToolsExt;
#else
import gpumod.hip.roctx;
#endif

namespace wwr::test {

using namespace wwr;

#if defined(WWR_GPU_BACKEND_CUDA)

using namespace wwr::cuda;

// ────────────────────────────────────────────────────────────────────────
// CUDA backend
// ────────────────────────────────────────────────────────────────────────

WWR_SAME_TYPE(gputxRangeId_t, nvtxRangeId_t)

WWR_SAME_FUNCTION(gputxMarkA, nvtxMarkA)
WWR_SAME_FUNCTION(gputxRangePushA, nvtxRangePushA)
WWR_SAME_FUNCTION(gputxRangePop, nvtxRangePop)
WWR_SAME_FUNCTION(gputxRangeStartA, nvtxRangeStartA)
// gputxRangeStop maps to nvtxRangeEnd -- the one name that does not prefix-swap.
WWR_SAME_FUNCTION(gputxRangeStop, nvtxRangeEnd)

#else

// ────────────────────────────────────────────────────────────────────────
// HIP backend
// ────────────────────────────────────────────────────────────────────────

using namespace wwr::hip;

WWR_SAME_TYPE(gputxRangeId_t, roctx_range_id_t)

WWR_SAME_FUNCTION(gputxMarkA, roctxMarkA)
WWR_SAME_FUNCTION(gputxRangePushA, roctxRangePushA)
WWR_SAME_FUNCTION(gputxRangePop, roctxRangePop)
WWR_SAME_FUNCTION(gputxRangeStartA, roctxRangeStartA)
WWR_SAME_FUNCTION(gputxRangeStop, roctxRangeStop)

#endif

} // namespace wwr::test
