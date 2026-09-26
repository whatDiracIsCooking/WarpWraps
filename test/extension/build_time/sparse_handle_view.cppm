// sparse_handle_view.cppm - Compile-time tests for GpusparseHandleView

export module gpumod.test.extension.sparse_handle_view;

import std;
import gpumod.extension.common;
import gpumod.extension.sparse;

// One TU per vendor-handle view on purpose: on HIP the hipblas/hipsolver/
// hipsparse handles are all `void*`, so importing two of these library modules
// together is ill-formed -- their HandleErrorType<void*> specializations collide.
// Each view is the device-bound, no-ops shape (a GpuBoundHandleView alias, as
// the library handles are device-bound); this pins that the alias instantiates
// and that .view() yields it on both backends.

namespace gpumod::extension::test {

static_assert(std::is_copy_constructible_v<GpusparseHandleView>);
static_assert(std::is_copy_assignable_v<GpusparseHandleView>);
static_assert(std::is_nothrow_move_constructible_v<GpusparseHandleView>);
static_assert(std::is_nothrow_default_constructible_v<GpusparseHandleView>);
static_assert(std::is_trivially_destructible_v<GpusparseHandleView>);

static_assert(!std::is_copy_constructible_v<GpusparseHandle>); // owner is move-only

static_assert(
    std::is_same_v<decltype(std::declval<const GpusparseHandle &>().view()), GpusparseHandleView>);

} // namespace gpumod::extension::test
