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

namespace wwr::hip::test {

using namespace wwr::hip;

// ──────────────────────────────────────────────────────────────────────
// Type shape
// ──────────────────────────────────────────────────────────────────────

// roctx_range_id_t is `typedef uint64_t roctx_range_id_t;`
static_assert(std::is_same_v<roctx_range_id_t, std::uint64_t>);

// ──────────────────────────────────────────────────────────────────────
// Link-time symbol resolution
// ──────────────────────────────────────────────────────────────────────

WWR_LINK_CHECK(roctxMarkA)
WWR_LINK_CHECK(roctxRangePushA)
WWR_LINK_CHECK(roctxRangePop)
WWR_LINK_CHECK(roctxRangeStartA)
WWR_LINK_CHECK(roctxRangeStop)

} // namespace wwr::hip::test
