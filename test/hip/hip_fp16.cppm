// hip_fp16.cppm - Compile-time tests for wwr.hip.hip_fp16

module;

#include "test/shared/link_check.h"

export module wwr.test.hip.hip_fp16;

import std;
import wwr.hip.hip_fp16;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time tests for wwr.hip.hip_fp16
//
// The module re-exports __half/__half2 by `using` declaration only. As
// src/hip/hip_fp16.cppm's doc comment explains, this project's plain C++23
// module compile takes amd_hip_fp16.h's `__GNUC__`-fallback branch (clang
// always predefines `__GNUC__`, and `__HIP__` is never defined without
// `-x hip`), which pulls in hip_fp16_gcc.h -- a portable `__half`/`__half2`
// with NO arithmetic/comparison operators of their own (only an implicit
// `operator float()`). There is nothing to WWR_LINK_CHECK or ADL-test for
// operators that do not exist on this code path; this file covers the
// layout and alias guarantees that static_assert can reach.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace wwr::hip::test {

using namespace wwr::hip;

// ──────────────────────────────────────────────────────────────────────
// Struct traits: C-interop guarantees
// ──────────────────────────────────────────────────────────────────────

static_assert(std::is_trivially_copyable_v<__half>);
static_assert(std::is_standard_layout_v<__half>);
static_assert(std::is_standard_layout_v<__half2>);
// Unlike CUDA's __half2 (which has a user-provided copy/move constructor and
// so is NOT trivially copyable), the hip_fp16_gcc.h fallback __half2 this
// module actually compiles against (see src/hip/hip_fp16.cppm's doc comment)
// defaults its copy/move constructor and assignment -- it IS trivially
// copyable. A fact about the actual compiled type, verified by direct
// compilation, not assumed to match CUDA.
static_assert(std::is_trivially_copyable_v<__half2>);
static_assert(sizeof(__half) == 2);
static_assert(sizeof(__half2) == 4);

// ──────────────────────────────────────────────────────────────────────
// Type aliases
// ──────────────────────────────────────────────────────────────────────

static_assert(std::is_same_v<half, __half>);
static_assert(std::is_same_v<half2, __half2>);

} // namespace wwr::hip::test
