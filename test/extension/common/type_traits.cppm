// type_traits.cppm - Compile-time tests for gpumod.extension.common's
// error/handle layer
//
// static_asserts on the error policy and RAII handle base (error_code,
// error_policy, default_error_policy, gpu_check, gpu_handle). The build is the
// test: this file is a compile_time_tests dependency (see CMakeLists.txt). The
// fp/int concept and type-map asserts live in test/wrappers/common.

export module gpumod.test.extension.common_error_handle;

import std;
import gpumod.extension.common;

namespace gpumod::extension::test {

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// error_policy / DefaultErrorPolicy
//
// The whole point of DefaultErrorPolicy is to be a valid policy: it is the
// default template argument of every extension's handle wrapper, so if it
// stopped satisfying error_policy (say a member made it throw on move) every
// handle would fail to compile with a constraint error far from the cause. Pin
// it here, and pin that the concept actually rejects a non-policy type.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

static_assert(error_policy<DefaultErrorPolicy<int>, int>);
static_assert(std::derived_from<DefaultErrorPolicy<int>, BaseErrorPolicy<int>>);
static_assert(std::is_nothrow_move_constructible_v<DefaultErrorPolicy<int>>);
static_assert(std::is_nothrow_move_assignable_v<DefaultErrorPolicy<int>>);
static_assert(!error_policy<int, int>); // a bare int is not a policy

} // namespace gpumod::extension::test

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// BaseGpuHandle RAII semantics
//
// Every extension instantiates this CRTP base, so it compiles -- but nothing
// asserts the ownership contract it exists to provide: move-only (copy deleted),
// nothrow move (it is used as a member of move-only wrappers), and an implicit
// conversion back to the underlying handle. This is the "compiles clean,
// semantics silently wrong" class the build-time tier is for. We instantiate it
// exactly as the real wrappers do -- a HandleErrorType specialization plus a
// CRTP derived type -- against a stand-in handle so the check depends on no
// vendor type. HandleErrorType is specialized in the gpumod::extension namespace,
// matching how blas_handle.cppm / solver_handle.cppm wire a handle to its error.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace gpumod::extension {

struct fake_handle_tag;
using FakeHandle = fake_handle_tag *;

template<>
struct HandleErrorType<FakeHandle> {
  using type = int;
};

namespace test {

class FakeHandleWrapper
    : public BaseGpuHandle<FakeHandle, FakeHandleWrapper, DefaultErrorPolicy<int>> {
public:
  using BaseGpuHandle::BaseGpuHandle;
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

} // namespace test
} // namespace gpumod::extension
