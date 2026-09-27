// tx.cppm - Compile-time tests for wwr.tx
//
// Every exported wwrtx* name is checked against the backend's own entity: the
// same range-id type, and the same function (see gpu_check_macros.h). The
// surface is the shared marker/range core, short enough to list in full.
//
// Two names carry the backend difference tx.cppm resolves: the async-range
// terminator wwrtxRangeStop maps to nvtxRangeEnd on CUDA (not "Stop"), and the
// range-id type wwrtxRangeId_t maps to roctx_range_id_t on HIP (not
// "roctxRangeId_t") -- this pins both to the right backend entity.

module;

#include "gpu_check_macros.h"

export module wwr.test.gpu.tx;

import std;
import wwr.tx;
#if defined(WWR_GPU_BACKEND_CUDA)
import wwr.cuda.nvToolsExt;
#else
import wwr.hip.roctx;
#endif

namespace wwr::test {

using namespace wwr;

#if defined(WWR_GPU_BACKEND_CUDA)

using namespace wwr::cuda;

// ────────────────────────────────────────────────────────────────────────
// CUDA backend
// ────────────────────────────────────────────────────────────────────────

WWR_SAME_TYPE(wwrtxRangeId_t, nvtxRangeId_t)

WWR_SAME_FUNCTION(wwrtxMarkA, nvtxMarkA)
WWR_SAME_FUNCTION(wwrtxRangePushA, nvtxRangePushA)
WWR_SAME_FUNCTION(wwrtxRangePop, nvtxRangePop)
WWR_SAME_FUNCTION(wwrtxRangeStartA, nvtxRangeStartA)
// wwrtxRangeStop maps to nvtxRangeEnd -- the one name that does not prefix-swap.
WWR_SAME_FUNCTION(wwrtxRangeStop, nvtxRangeEnd)

#else

// ────────────────────────────────────────────────────────────────────────
// HIP backend
// ────────────────────────────────────────────────────────────────────────

using namespace wwr::hip;

WWR_SAME_TYPE(wwrtxRangeId_t, roctx_range_id_t)

WWR_SAME_FUNCTION(wwrtxMarkA, roctxMarkA)
WWR_SAME_FUNCTION(wwrtxRangePushA, roctxRangePushA)
WWR_SAME_FUNCTION(wwrtxRangePop, roctxRangePop)
WWR_SAME_FUNCTION(wwrtxRangeStartA, roctxRangeStartA)
WWR_SAME_FUNCTION(wwrtxRangeStop, roctxRangeStop)

#endif

} // namespace wwr::test
