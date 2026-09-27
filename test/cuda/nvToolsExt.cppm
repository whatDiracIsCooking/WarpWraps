// nvToolsExt.cppm - Compile-time tests for gpumod.cuda.nvToolsExt

module;

#include "test/shared/link_check.h"

export module gpumod.test.cuda.nvToolsExt;

import std;
import gpumod.cuda.nvToolsExt;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time tests for gpumod.cuda.nvToolsExt
//
// The module wraps NVTX's static-inline marker/range core through thin
// forwarding functions (nvToolsExt.h has internal linkage, so it cannot be
// re-exported with `using`, unlike roctx's real extern "C" symbols). We verify
// at compile/link time that:
//   1. nvtxRangeId_t is the expected 64-bit unsigned integral typedef
//   2. Each forwarding function has the expected signature
//   3. Link-time resolution of every exported function -- taking the wrapper's
//      address forces it to be emitted and linked, which in turn pulls in the
//      header-only NVTX implementation it forwards to
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace gpumod::cuda::test {

using namespace gpumod::cuda;

// ──────────────────────────────────────────────────────────────────────
// Type shape
// ──────────────────────────────────────────────────────────────────────

// nvtxRangeId_t is `typedef uint64_t nvtxRangeId_t;`
static_assert(std::is_same_v<nvtxRangeId_t, std::uint64_t>);

// ──────────────────────────────────────────────────────────────────────
// Forwarding-function signatures
// ──────────────────────────────────────────────────────────────────────

static_assert(std::is_same_v<decltype(nvtxMarkA), void(const char*)>);
static_assert(std::is_same_v<decltype(nvtxRangePushA), int(const char*)>);
static_assert(std::is_same_v<decltype(nvtxRangePop), int()>);
static_assert(std::is_same_v<decltype(nvtxRangeStartA), nvtxRangeId_t(const char*)>);
static_assert(std::is_same_v<decltype(nvtxRangeEnd), void(nvtxRangeId_t)>);

// ──────────────────────────────────────────────────────────────────────
// Link-time symbol resolution
// ──────────────────────────────────────────────────────────────────────

GPUMOD_LINK_CHECK(nvtxMarkA)
GPUMOD_LINK_CHECK(nvtxRangePushA)
GPUMOD_LINK_CHECK(nvtxRangePop)
GPUMOD_LINK_CHECK(nvtxRangeStartA)
GPUMOD_LINK_CHECK(nvtxRangeEnd)

} // namespace gpumod::cuda::test
