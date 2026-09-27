// cuda_profiler_api.cppm - Compile-time tests for wwr.cuda.cuda_profiler_api

module;

#include "test/shared/link_check.h"

export module wwr.test.cuda.cuda_profiler_api;

import std;
import wwr.cuda.cuda_profiler_api;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time tests for wwr.cuda.cuda_profiler_api
//
// The API surface of cuda_profiler_api.h is intentionally minimal: it
// declares exactly two functions — cudaProfilerStart and cudaProfilerStop —
// both returning cudaError_t and taking no arguments.  There are no macros,
// no enums beyond cudaError_t (which lives in driver_types.h), and no
// custom types.
//
// We verify at compile-time that:
//   1. cudaError_t is recognised as an enum type
//   2. cudaProfilerStart and cudaProfilerStop have the expected signatures
//      (callable with no arguments, returning cudaError_t)
//   3. Both functions resolve to linkable external symbols (WWR_LINK_CHECK)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace wwr::cuda::test {

using namespace wwr::cuda;

// ────────────────────────────────────────────────────────────────────────
// Return type
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_enum_v<cudaError_t>);

// ────────────────────────────────────────────────────────────────────────
// Signature checks: both functions take no arguments and return cudaError_t
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_invocable_v<decltype(cudaProfilerStart)>);
static_assert(std::is_invocable_v<decltype(cudaProfilerStop)>);

static_assert(std::is_same_v<std::invoke_result_t<decltype(cudaProfilerStart)>, cudaError_t>);

static_assert(std::is_same_v<std::invoke_result_t<decltype(cudaProfilerStop)>, cudaError_t>);

// ────────────────────────────────────────────────────────────────────────
// Link-time symbol resolution
// Both functions are external (non-inline), so WWR_LINK_CHECK applies.
// ────────────────────────────────────────────────────────────────────────

WWR_LINK_CHECK(cudaProfilerStart)
WWR_LINK_CHECK(cudaProfilerStop)

} // namespace wwr::cuda::test
