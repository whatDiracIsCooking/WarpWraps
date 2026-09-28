// handle_view.cppm - Compile-time tests for the non-owning GPU handle views

export module wwr.test.extension.handle_view;

import std;
import wwr.runtime_api;
import wwr.extension.common;
import wwr.extension.handle;
import wwr.extension.runtime;
import wwr.extension.blas;
import wwr.extension.solver;
import wwr.extension.sparse;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time contract of the handle views
//
// An owning handle is move-only and never owns twice; a view borrows the same
// handle but is a freely copyable, trivially destructible value. Borrow-safe
// operations are free functions taking the raw handle, so the owner, its view
// and a bare handle all reach one definition; ownership-producing operations
// (end_capture, instantiate) stay members of the owner. `.view()` returns the
// richest view a handle has -- device-aware where the handle is device-bound --
// and is deleted on rvalues so a temporary cannot be viewed.
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

namespace wwr::extension::test {
// Bind abort-on-failure once per error type, for this file's instantiations.
using Abort = AbortPolicy<wwrError_t>;
using BlasAbort = AbortPolicy<wwrblasStatus_t>;
using SolverAbort = AbortPolicy<wwrsolverStatus_t>;
using SparseAbort = AbortPolicy<wwrsparseStatus_t>;

// A view is a freely copyable, trivially destructible value -- the whole point.
template<typename View>
concept is_view_value = std::is_copy_constructible_v<View> && std::is_copy_assignable_v<View> &&
                        std::is_nothrow_move_constructible_v<View> &&
                        std::is_nothrow_default_constructible_v<View> &&
                        std::is_trivially_destructible_v<View>;

static_assert(is_view_value<GpuEventView>);
static_assert(is_view_value<GpuStreamView>);
static_assert(is_view_value<GpuGraphExecView>);
static_assert(is_view_value<DeviceBoundHandleView<wwrMemPool_t>>);
static_assert(is_view_value<HandleView<wwrGraph_t>>);
static_assert(is_view_value<DeviceBoundHandleView<wwrblasHandle_t>>);
static_assert(is_view_value<DeviceBoundHandleView<wwrsolverDnHandle_t>>);
static_assert(is_view_value<DeviceBoundHandleView<wwrsparseHandle_t>>);

// The generic view bases carry the same semantics.
static_assert(is_view_value<HandleView<wwrEvent_t>>);
static_assert(is_view_value<DeviceBoundHandleView<wwrEvent_t>>);

// Owners are move-only; a view is never taken by copying an owner.
static_assert(!std::is_copy_constructible_v<GpuEventWrapper<Abort, Abort>>);
static_assert(!std::is_copy_constructible_v<GpuStreamWrapper<Abort, Abort>>);
static_assert(!std::is_copy_constructible_v<WwrblasHandleWrapper<BlasAbort, BlasAbort>>);
static_assert(std::is_nothrow_move_constructible_v<GpuEventWrapper<Abort, Abort>>);

// Views convert to the raw handle, exactly as the owners do.
static_assert(std::is_convertible_v<GpuEventView, wwrEvent_t>);
static_assert(std::is_convertible_v<GpuStreamView, wwrStream_t>);
static_assert(std::is_convertible_v<GpuGraphExecView, wwrGraphExec_t>);

// A view is constructible from a raw handle (device unknown) or, when bound,
// from a handle plus its device index.
static_assert(std::is_constructible_v<GpuStreamView, wwrStream_t>);
static_assert(std::is_constructible_v<GpuStreamView, wwrStream_t, int>);
static_assert(std::is_constructible_v<GpuGraphExecView, wwrGraphExec_t>);

// .view() is offered on an lvalue owner and returns the richest view the handle
// has. Its rvalue overload is deleted so a view cannot be taken from a temporary
// (which would dangle immediately); that guard fires at the call site, not as a
// trait here -- the deleted overload still wins overload resolution, so calling
// it is a hard error rather than a detectable unsatisfied requirement.
static_assert(requires(const GpuEventWrapper<Abort, Abort> &e) { e.view(); });
static_assert(std::is_same_v<decltype(std::declval<const GpuEventWrapper<Abort, Abort> &>().view()), GpuEventView>);
static_assert(std::is_same_v<decltype(std::declval<const GpuStreamWrapper<Abort, Abort> &>().view()), GpuStreamView>);
static_assert(
    std::is_same_v<decltype(std::declval<const GpuGraphExecWrapper<Abort, Abort> &>().view()), GpuGraphExecView>);
static_assert(std::is_same_v<decltype(std::declval<const GpuMemPoolWrapper<Abort, Abort> &>().view()), DeviceBoundHandleView<wwrMemPool_t>>);
static_assert(std::is_same_v<decltype(std::declval<const GpuGraphWrapper<Abort, Abort> &>().view()), HandleView<wwrGraph_t>>);
static_assert(
    std::is_same_v<decltype(std::declval<const WwrblasHandleWrapper<BlasAbort, BlasAbort> &>().view()), DeviceBoundHandleView<wwrblasHandle_t>>);
static_assert(std::is_same_v<decltype(std::declval<const WwrsolverDnHandleWrapper<SolverAbort, SolverAbort> &>().view()),
                             DeviceBoundHandleView<wwrsolverDnHandle_t>>);
static_assert(
    std::is_same_v<decltype(std::declval<const WwrsparseHandleWrapper<SparseAbort, SparseAbort> &>().view()), DeviceBoundHandleView<wwrsparseHandle_t>>);

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// [[no_unique_address]] on BaseHandle's policy members (issue #67)
//
// A pointer handle's only real state is the pointer itself (owns_ is already an
// elided empty flag). Two DIFFERENT empty policy types overlap other subobjects
// and vanish, so a custom create/destroy pair leaves the wrapper the size of the
// bare handle. The DEFAULT (P_destroy = P_create, one stateless type) does NOT
// shrink: [intro.object] forbids two same-type empty subobjects from sharing an
// address, so the second still costs a slot. Both are pinned here so the
// attribute is not mistaken for zero-cost on the common path (issue #71
// collapses the same-type slot). A minimal local wrapper isolates the
// policy-member cost -- the shipped GpuStreamWrapper etc. are DeviceBoundHandles
// that also carry a device index.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

template<typename E>
struct SizeEmptyPolicy {
  using error_type = E;
  void handle_error(E, std::source_location) noexcept {}
};
template<typename E>
struct SizeEmptyPolicy2 {
  using error_type = E;
  void handle_error(E, std::source_location) noexcept {}
};
template<typename Pc, typename Pd>
struct SizeProbeHandle : BaseHandle<void *, SizeProbeHandle<Pc, Pd>, Pc, Pd> {};

// Distinct empty policies cost nothing: the wrapper is just the pointer handle.
static_assert(sizeof(SizeProbeHandle<SizeEmptyPolicy<wwrError_t>, SizeEmptyPolicy2<wwrError_t>>) ==
              sizeof(void *));
// The same-type default still pays for the second, otherwise-elided slot.
static_assert(sizeof(SizeProbeHandle<SizeEmptyPolicy<wwrError_t>, SizeEmptyPolicy<wwrError_t>>) >
              sizeof(SizeProbeHandle<SizeEmptyPolicy<wwrError_t>, SizeEmptyPolicy2<wwrError_t>>));

// Never called: exists only to instantiate and type-check the borrow-safe free
// functions on each owner, its view, and a raw handle, without a device.
[[maybe_unused]] void exercise(const GpuEventWrapper<Abort, Abort> &event,
                               const GpuStreamWrapper<Abort, Abort> &stream,
                               const GpuGraphExecWrapper<Abort, Abort> &exec,
                               wwrStream_t raw_stream, wwrEvent_t raw_event) {
  const GpuEventView ev = event.view();
  (void)ev.get();
  (void)ev.dev_idx();
  (void)record(ev, raw_stream);
  (void)record(ev, raw_stream, 0u);
  (void)sync(ev);
  (void)sync(event); // same free function on the owner

  const GpuStreamView sv = stream.view();
  (void)sv.dev_idx();
  (void)wait_event(sv, raw_event);
  (void)begin_capture(sv);
  (void)sync(sv);
  (void)wait_event(stream, raw_event); // owner: same free function

  const GpuGraphExecView xv = exec.view();
  (void)launch(xv, raw_stream);
  (void)upload(xv, raw_stream);
  (void)launch(exec, raw_stream); // owner: same free function
}

} // namespace wwr::extension::test
