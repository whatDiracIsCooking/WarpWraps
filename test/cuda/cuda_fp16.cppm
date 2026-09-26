// cuda_fp16.cppm - Compile-time tests for gpumod.cuda.cuda_fp16

module;

#include "test/shared/link_check.h"

export module gpumod.test.cuda.cuda_fp16;

import std;
import gpumod.cuda.cuda_fp16;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Tests for gpumod.cuda.cuda_fp16
//
// The module re-exports __half/__half2 by `using` declaration only; their
// arithmetic/comparison operators are free functions defined in the module's
// global module fragment (from <cuda_fp16.h>), and the module's own doc
// comment claims they "work automatically via Argument-Dependent Lookup" from
// an importing TU. Whether ADL actually resolves them is not tested -- the
// operators forward to non-constexpr intrinsics, so it cannot be done here.
// This file covers the layout and alias guarantees that static_assert can
// reach.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace gpumod::cuda::test {

using namespace gpumod::cuda;

// ────────────────────────────────────────────────────────────────────────
// Struct traits: C-interop guarantees
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_trivially_copyable_v<__half>);
static_assert(std::is_standard_layout_v<__half>);
static_assert(std::is_standard_layout_v<__half2>);
// __half2 has a user-provided copy/move constructor in <cuda_fp16.h> (to
// bit-copy the packed pair without going through FP conversion), so unlike
// __half it is NOT trivially copyable — this is a fact about the vendor
// type, not something this wrapper could or should paper over.
static_assert(!std::is_trivially_copyable_v<__half2>);
static_assert(sizeof(__half) == 2);
static_assert(sizeof(__half2) == 4);

// ────────────────────────────────────────────────────────────────────────
// Type aliases
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_same_v<half, __half>);
static_assert(std::is_same_v<half2, __half2>);
static_assert(std::is_same_v<__nv_half, __half>);
static_assert(std::is_same_v<__nv_half2, __half2>);
static_assert(std::is_same_v<nv_half, __half>);
static_assert(std::is_same_v<nv_half2, __half2>);

} // namespace gpumod::cuda::test
