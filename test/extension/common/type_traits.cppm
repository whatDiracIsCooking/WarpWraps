// type_traits.cppm - Compile-time tests for wwr.extension.common's
// error/handle layer
//
// static_asserts on the error policy and RAII handle base (error_code,
// error_policy, abort_policy, gpu_check, handle). The build is the
// test: this file is a compile_time_tests dependency (see CMakeLists.txt). The
// fp/int concept and type-map asserts live in test/wrappers/common.

export module wwr.test.extension.common_error_handle;

import std;
import wwr.extension.common;
import wwr.extension.handle;

namespace wwr::extension::test {

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// error_policy / AbortPolicy
//
// The whole point of AbortPolicy is to be a valid policy: it is the
// default template argument of every extension's handle wrapper, so if it
// stopped satisfying error_policy (say a member made it throw on move) every
// handle would fail to compile with a constraint error far from the cause. Pin
// it here, and pin that the concept actually rejects a non-policy type.
//
// The policy is duck-typed, not base-derived (no BaseErrorPolicy): what a handle
// actually requires is typed_error_policy -- a handle_error member plus the
// error_type alias it deduces its error type from -- so assert that, not an
// implementation detail. AbortPolicy is also usable in the destruction
// slot, so pin nothrow_error_policy (its handle_error is noexcept) too.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

static_assert(error_policy<AbortPolicy<int>, int>);
static_assert(typed_error_policy<AbortPolicy<int>>);
static_assert(nothrow_error_policy<AbortPolicy<int>, int>);
static_assert(std::is_nothrow_move_constructible_v<AbortPolicy<int>>);
static_assert(std::is_nothrow_move_assignable_v<AbortPolicy<int>>);
static_assert(!error_policy<int, int>); // a bare int is not a policy

} // namespace wwr::extension::test

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// BaseHandle RAII semantics
//
// Every extension instantiates this CRTP base, so it compiles -- but nothing
// asserts the ownership contract it exists to provide: move-only (copy deleted),
// nothrow move (it is used as a member of move-only wrappers), and an implicit
// conversion back to the underlying handle. This is the "compiles clean,
// semantics silently wrong" class the build-time tier is for. We instantiate it
// exactly as the real wrappers do -- a CRTP derived type plus an error policy --
// against a stand-in handle so the check depends on no vendor type. The error
// type is deduced from the policy (AbortPolicy<int>::error_type), not from
// the handle type, so no per-handle table is needed here or in the real wrappers.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace wwr::extension::test {

struct fake_handle_tag;
using FakeHandle = fake_handle_tag *;

class FakeHandleWrapper
    : public BaseHandle<FakeHandle, FakeHandleWrapper, AbortPolicy<int>> {
public:
  using BaseHandle::BaseHandle;
  void create(FakeHandle *h, std::source_location) { *h = nullptr; }
  void destroy(FakeHandle) {}
};

static_assert(!std::is_copy_constructible_v<FakeHandleWrapper>);
static_assert(!std::is_copy_assignable_v<FakeHandleWrapper>);
static_assert(std::is_move_constructible_v<FakeHandleWrapper>);
static_assert(std::is_move_assignable_v<FakeHandleWrapper>);
static_assert(std::is_nothrow_move_constructible_v<FakeHandleWrapper>);
static_assert(std::is_nothrow_move_assignable_v<FakeHandleWrapper>);
static_assert(std::is_convertible_v<FakeHandleWrapper, FakeHandle>);

} // namespace wwr::extension::test
