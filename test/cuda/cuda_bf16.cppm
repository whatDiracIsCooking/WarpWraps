// cuda_bf16.cppm - Compile-time tests for gpumod.cuda.cuda_bf16

module;

#include "test/shared/link_check.h"

export module gpumod.test.cuda.cuda_bf16;

import std;
import gpumod.cuda.cuda_bf16;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Tests for gpumod.cuda.cuda_bf16
//
// Mirrors cuda_fp16.cppm: __nv_bfloat16/__nv_bfloat162 are re-exported by
// `using` declaration only, and their arithmetic/comparison operators are
// free functions in the module's global module fragment (from
// <cuda_bf16.h>). The module's doc comment claims those operators "work
// automatically via Argument-Dependent Lookup" from an importing TU. Whether
// that holds under this compiler/module setup is not tested (the operators
// forward to non-constexpr intrinsics); this file covers the layout and alias
// guarantees that static_assert can reach.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace wwr::cuda::test {

using namespace wwr::cuda;

// ────────────────────────────────────────────────────────────────────────
// Struct traits: C-interop guarantees
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_trivially_copyable_v<__nv_bfloat16>);
static_assert(std::is_standard_layout_v<__nv_bfloat16>);
static_assert(std::is_standard_layout_v<__nv_bfloat162>);
// __nv_bfloat162 has a user-provided copy/move constructor in <cuda_bf16.h>
// (to bit-copy the packed pair without going through FP conversion), so
// unlike __nv_bfloat16 it is NOT trivially copyable — this is a fact about
// the vendor type, not something this wrapper could or should paper over.
static_assert(!std::is_trivially_copyable_v<__nv_bfloat162>);
static_assert(sizeof(__nv_bfloat16) == 2);
static_assert(sizeof(__nv_bfloat162) == 4);

// ────────────────────────────────────────────────────────────────────────
// Type aliases
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_same_v<nv_bfloat16, __nv_bfloat16>);
static_assert(std::is_same_v<nv_bfloat162, __nv_bfloat162>);

} // namespace wwr::cuda::test
