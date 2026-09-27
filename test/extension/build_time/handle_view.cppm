// handle_view.cppm - Compile-time tests for the non-owning GPU handle views

export module gpumod.test.extension.handle_view;

import std;
import gpumod.runtime_api;
import gpumod.extension.common;
import gpumod.extension.common.handle;
import gpumod.extension.runtime;
import gpumod.extension.blas;
import gpumod.extension.solver;
import gpumod.extension.sparse;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time contract of the handle views
//
// An owning handle is move-only and never owns twice; a view borrows the same
// handle but is a freely copyable, trivially destructible value. Where the owner
// has borrow-safe operations they come from one shared CRTP mixin, so owner and
// view expose them identically with no duplication; ownership-producing
// operations (end_capture, instantiate) stay on the owner. `.view()` returns the
// richest view a handle has -- device-aware and/or operation-carrying -- and is
// deleted on rvalues so a temporary cannot be viewed.
//
// This TU deliberately imports all three vendor-handle extension modules
// (blas/solver/sparse) TOGETHER. That used to be ill-formed on HIP, where those
// handles are all `void*`: a per-handle-type HandleErrorType<void*> table gave
// them conflicting error types in one TU. The error type is now deduced from the
// policy instead (typed_error_policy), so the table is gone and the three
// coexist -- this test is the regression guard for that.
//
// These are compile-time contracts; the runtime tests live beside the wrappers.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace gpumod::extension::test {

// A view is a freely copyable, trivially destructible value -- the whole point.
template<typename View>
concept is_view_value = std::is_copy_constructible_v<View> && std::is_copy_assignable_v<View> &&
                        std::is_nothrow_move_constructible_v<View> &&
                        std::is_nothrow_default_constructible_v<View> &&
                        std::is_trivially_destructible_v<View>;

static_assert(is_view_value<GpuEventView>);
static_assert(is_view_value<GpuStreamView>);
static_assert(is_view_value<GpuGraphExecView>);
static_assert(is_view_value<GpuMemPoolView>);
static_assert(is_view_value<GpuGraphView>);
static_assert(is_view_value<GpublasHandleView>);
static_assert(is_view_value<GpusolverDnHandleView>);
static_assert(is_view_value<GpusparseHandleView>);

// The generic view bases carry the same semantics.
static_assert(is_view_value<GpuHandleView<gpuEvent_t>>);
static_assert(is_view_value<GpuBoundHandleView<gpuEvent_t>>);

// Owners are move-only; a view is never taken by copying an owner.
static_assert(!std::is_copy_constructible_v<GpuEvent>);
static_assert(!std::is_copy_constructible_v<GpuStream>);
static_assert(!std::is_copy_constructible_v<GpublasHandle>);
static_assert(std::is_nothrow_move_constructible_v<GpuEvent>);

// Views convert to the raw handle, exactly as the owners do.
static_assert(std::is_convertible_v<GpuEventView, gpuEvent_t>);
static_assert(std::is_convertible_v<GpuStreamView, gpuStream_t>);
static_assert(std::is_convertible_v<GpuGraphExecView, gpuGraphExec_t>);

// A view is constructible from a raw handle (device unknown) or, when bound,
// from a handle plus its device index.
static_assert(std::is_constructible_v<GpuStreamView, gpuStream_t>);
static_assert(std::is_constructible_v<GpuStreamView, gpuStream_t, int>);
static_assert(std::is_constructible_v<GpuGraphExecView, gpuGraphExec_t>);

// .view() is offered on an lvalue owner and returns the richest view the handle
// has. Its rvalue overload is deleted so a view cannot be taken from a temporary
// (which would dangle immediately); that guard fires at the call site, not as a
// trait here -- the deleted overload still wins overload resolution, so calling
// it is a hard error rather than a detectable unsatisfied requirement.
static_assert(requires(const GpuEvent &e) { e.view(); });
static_assert(std::is_same_v<decltype(std::declval<const GpuEvent &>().view()), GpuEventView>);
static_assert(std::is_same_v<decltype(std::declval<const GpuStream &>().view()), GpuStreamView>);
static_assert(
    std::is_same_v<decltype(std::declval<const GpuGraphExec &>().view()), GpuGraphExecView>);
static_assert(std::is_same_v<decltype(std::declval<const GpuMemPool &>().view()), GpuMemPoolView>);
static_assert(std::is_same_v<decltype(std::declval<const GpuGraph &>().view()), GpuGraphView>);
static_assert(
    std::is_same_v<decltype(std::declval<const GpublasHandle &>().view()), GpublasHandleView>);
static_assert(std::is_same_v<decltype(std::declval<const GpusolverDnHandle &>().view()),
                             GpusolverDnHandleView>);
static_assert(
    std::is_same_v<decltype(std::declval<const GpusparseHandle &>().view()), GpusparseHandleView>);

// Never called: exists only to instantiate and type-check the borrow-safe
// operations shared by each owner and its view, without a device.
[[maybe_unused]] void exercise(const GpuEvent &event, const GpuStream &stream,
                               const GpuGraphExec &exec, gpuStream_t raw_stream,
                               gpuEvent_t raw_event) {
  const GpuEventView ev = event.view();
  (void)ev.get();
  (void)ev.dev_idx();
  (void)ev.record(raw_stream);
  (void)ev.record(raw_stream, 0u);
  (void)ev.sync();
  (void)event.sync(); // same op on the owner, via the shared mixin

  const GpuStreamView sv = stream.view();
  (void)sv.dev_idx();
  (void)sv.wait_event(raw_event);
  (void)sv.begin_capture();
  (void)sv.sync();
  (void)stream.wait_event(raw_event); // owner shares the mixin

  const GpuGraphExecView xv = exec.view();
  (void)xv.launch(raw_stream);
  (void)xv.upload(raw_stream);
  (void)exec.launch(raw_stream); // owner shares the mixin
}

} // namespace gpumod::extension::test
