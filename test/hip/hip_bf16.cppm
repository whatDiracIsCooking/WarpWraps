// hip_bf16.cppm - Compile-time tests for gpumod.hip.hip_bf16

module;

#include "test/shared/link_check.h"

export module gpumod.test.hip.hip_bf16;

import std;
import gpumod.hip.hip_bf16;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time tests for gpumod.hip.hip_bf16
//
// Mirrors test/cuda/cuda_bf16.cppm: __hip_bfloat16/__hip_bfloat162 are
// re-exported by `using` declaration only; their arithmetic/comparison
// operators are ordinary `static inline` free functions in
// amd_hip_bf16.h's global namespace, so hip_bf16.cppm provides thin
// forwarding operators in wwr::hip (parity with cuda_bf16.cppm's
// treatment of cuda_bf16.h). Whether those forwarders are actually reachable
// and correct at run time is out of scope for this compile-time-only
// directory (parity with cuda_bf16.cppm); this file covers the layout and alias guarantees that static_assert can reach.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace wwr::hip::test {

using namespace wwr::hip;

// ──────────────────────────────────────────────────────────────────────
// Struct traits: C-interop guarantees
// ──────────────────────────────────────────────────────────────────────

static_assert(std::is_trivially_copyable_v<__hip_bfloat16>);
static_assert(std::is_standard_layout_v<__hip_bfloat16>);
// __hip_bfloat162 has a user-provided copy constructor and copy-assignment
// operator in amd_hip_bf16.h (elementwise x/y, not a defaulted bit-copy), so
// unlike __hip_bfloat16 it is NOT trivially copyable -- a fact about the
// vendor type (matching CUDA's __nv_bfloat162 in this respect, though for a
// different reason: CUDA's is non-trivial for a converting move/copy pair,
// HIP's is non-trivial because the copy constructor is user-provided at all).
static_assert(!std::is_trivially_copyable_v<__hip_bfloat162>);
static_assert(std::is_standard_layout_v<__hip_bfloat162>);
static_assert(sizeof(__hip_bfloat16) == 2);
static_assert(sizeof(__hip_bfloat162) == 4);

} // namespace wwr::hip::test
