// roctx.cppm - Compile-time tests for gpumod.hip.roctx

module;

#include "test/shared/link_check.h"

export module gpumod.test.hip.roctx;

import std;
import gpumod.hip.roctx;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time tests for gpumod.hip.roctx
//
// The module is pure re-export (using-declarations -- roctx.h is a plain
// extern "C" API in libroctx64, with none of the convenience-template
// collisions a runtime header has). We verify at compile/link time that:
//   1. roctx_range_id_t is the expected 64-bit unsigned integral typedef
//   2. Link-time symbol resolution for every exported function (libroctx64)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace gpumod::hip::test {

using namespace gpumod::hip;

// ──────────────────────────────────────────────────────────────────────
// Type shape
// ──────────────────────────────────────────────────────────────────────

// roctx_range_id_t is `typedef uint64_t roctx_range_id_t;`
static_assert(std::is_same_v<roctx_range_id_t, std::uint64_t>);

// ──────────────────────────────────────────────────────────────────────
// Link-time symbol resolution
// ──────────────────────────────────────────────────────────────────────

GPUMOD_LINK_CHECK(roctxMarkA)
GPUMOD_LINK_CHECK(roctxRangePushA)
GPUMOD_LINK_CHECK(roctxRangePop)
GPUMOD_LINK_CHECK(roctxRangeStartA)
GPUMOD_LINK_CHECK(roctxRangeStop)

} // namespace gpumod::hip::test
