// solver_handle_view.cppm - Compile-time tests for GpusolverDnHandleView

export module gpumod.test.extension.solver_handle_view;

import std;
import gpumod.extension.common;
import gpumod.extension.solver;

// One TU per vendor-handle view on purpose: on HIP the hipblas/hipsolver/
// hipsparse handles are all `void*`, so importing two of these library modules
// together is ill-formed -- their HandleErrorType<void*> specializations collide.
// Each view is the device-bound, no-ops shape (a GpuBoundHandleView alias, as
// the library handles are device-bound); this pins that the alias instantiates
// and that .view() yields it on both backends.

namespace gpumod::extension::test {

static_assert(std::is_copy_constructible_v<GpusolverDnHandleView>);
static_assert(std::is_copy_assignable_v<GpusolverDnHandleView>);
static_assert(std::is_nothrow_move_constructible_v<GpusolverDnHandleView>);
static_assert(std::is_nothrow_default_constructible_v<GpusolverDnHandleView>);
static_assert(std::is_trivially_destructible_v<GpusolverDnHandleView>);

static_assert(!std::is_copy_constructible_v<GpusolverDnHandle>); // owner is move-only

static_assert(std::is_same_v<decltype(std::declval<const GpusolverDnHandle &>().view()),
                             GpusolverDnHandleView>);

} // namespace gpumod::extension::test
