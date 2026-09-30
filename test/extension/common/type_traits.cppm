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
import wwr.runtime_api; // wwrError_t, for the device-access policy's error type
import wwr.test.shared.abort_policy; // the test suite's abort-on-failure policy

// A registered stand-in error type. typed_error_policy now requires the policy's
// published error_type to be a registered error_type, so a policy used with a
// handle needs one -- but registration is just the three specializations (see
// error_code.cppm), so we mint one here and stay free of any vendor/runtime type.
namespace wwr::extension {
enum class TestError { ok };
template<>
constexpr TestError success_code<TestError>() noexcept {
  return TestError::ok;
}
template<>
const char *error_name<TestError>(TestError) noexcept {
  return "TestError";
}
template<>
const char *error_string<TestError>(TestError) noexcept {
  return "test error";
}
} // namespace wwr::extension

namespace wwr::extension::test {

using TestError = wwr::extension::TestError;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// error_policy / AbortPolicy
//
// The whole point of AbortPolicy is to be a valid policy: the test suites hand
// it to every handle wrapper (the create, destroy and device slots), so if it
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

// error_policy stays purely structural: an unregistered stand-in like int is a
// valid policy target, because the concept checks P's shape, not T's membership.
static_assert(error_policy<AbortPolicy<int>, int>);
static_assert(nothrow_error_policy<AbortPolicy<int>, int>);
static_assert(std::is_nothrow_move_constructible_v<AbortPolicy<int>>);
static_assert(std::is_nothrow_move_assignable_v<AbortPolicy<int>>);
static_assert(!error_policy<int, int>); // a bare int is not a policy

// typed_error_policy additionally requires the *published* error_type to be a
// registered error_type -- this is the check every handle's P_create goes through.
static_assert(typed_error_policy<AbortPolicy<TestError>>);
static_assert(!typed_error_policy<AbortPolicy<int>>); // int carries no error facilities

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// policy_map
//
// A buffer suite (#177) takes one map and instantiates it per memory kind with
// that kind's error type, so the concept checks the map's two member alias
// templates -- alloc / free -- yield valid policies for a given error type. It is
// structural in E exactly as error_policy is (an unregistered int is a fine
// probe): the map's *shape* is checked here, and the suite enforces typing on the
// buffer. The free slot inherits the destructor-path nothrow requirement.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// A policy whose handle_error may throw: a valid alloc policy, never a free one.
template<typename E>
struct ThrowingPolicy {
  using error_type = E;
  void handle_error(E, std::source_location) {}
};
static_assert(error_policy<ThrowingPolicy<int>, int>);
static_assert(!nothrow_error_policy<ThrowingPolicy<int>, int>);

// One policy for both slots -- the single-policy shape the suite's convenience
// front door builds. Valid for any error type the policy accepts.
struct AbortMap {
  template<typename E>
  using alloc = AbortPolicy<E>;
  template<typename E>
  using free = AbortPolicy<E>;
};
static_assert(policy_map<AbortMap, int>);
static_assert(policy_map<AbortMap, TestError>);

// A map missing a key is not a policy map, and neither is a plain type.
struct AllocOnlyMap {
  template<typename E>
  using alloc = AbortPolicy<E>;
};
static_assert(!policy_map<AllocOnlyMap, int>);
static_assert(!policy_map<int, int>);

// The free slot must not throw: a map whose free policy can throw is rejected,
// even though the same policy is fine in the alloc slot.
struct ThrowingFreeMap {
  template<typename E>
  using alloc = AbortPolicy<E>;
  template<typename E>
  using free = ThrowingPolicy<E>;
};
static_assert(!policy_map<ThrowingFreeMap, int>);

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
// type is deduced from the policy (AbortPolicy<TestError>::error_type), not from
// the handle type, so no per-handle table is needed here or in the real wrappers.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace wwr::extension::test {

struct fake_handle_tag;
using FakeHandle = fake_handle_tag *;

class FakeHandleWrapper
    : public BaseHandle<FakeHandle, FakeHandleWrapper, AbortPolicy<TestError>,
                        AbortPolicy<TestError>> {
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

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// DeviceBoundHandle: the device-bound layer's compile-time contract
//
// The CRTP layer adds a member (dev_idx_), a second policy (policy_device_) and a
// hand-written move on top of BaseHandle. None of its interesting guarantees are
// runtime -- they are type-level -- so pin them here, where no GPU is needed:
//   * it stays move-only and nothrow-movable through the added member and the
//     hand-written move (a defaulted move that silently became a copy, or a
//     policy that stopped being nothrow-movable, is the regression);
//   * the conversion-to-handle it inherits survives the extra layer;
//   * the device-access policy is typed to wwrError_t *independently* of the
//     create policy's error type (TestError here) -- the very decoupling the
//     single-constructor design exists to allow, and
//   * view() is narrowed to the device-aware DeviceBoundHandleView (not the base
//     HandleView) and is still deleted on rvalues.
// The runtime ordering trick (select-before-create) needs a live device; it is
// tested in test/extension/handle.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace wwr::extension::test {

// Instantiated exactly as a real device-bound wrapper is: a CRTP derived type,
// a create policy typed to the handle's own status (TestError here, a registered
// stand-in for a library handle's status enum), and a device policy typed to
// wwrError_t.
class FakeDeviceHandleWrapper
    : public DeviceBoundHandle<FakeHandle, FakeDeviceHandleWrapper, AbortPolicy<TestError>,
                               AbortPolicy<TestError>, AbortPolicy<wwrError_t>> {
public:
  using DeviceBoundHandle::DeviceBoundHandle;
  void create(FakeHandle *h, std::source_location) { *h = nullptr; }
  void destroy(FakeHandle) {}
};

static_assert(!std::is_copy_constructible_v<FakeDeviceHandleWrapper>);
static_assert(!std::is_copy_assignable_v<FakeDeviceHandleWrapper>);
static_assert(std::is_move_constructible_v<FakeDeviceHandleWrapper>);
static_assert(std::is_move_assignable_v<FakeDeviceHandleWrapper>);
static_assert(std::is_nothrow_move_constructible_v<FakeDeviceHandleWrapper>);
static_assert(std::is_nothrow_move_assignable_v<FakeDeviceHandleWrapper>);
static_assert(std::is_convertible_v<FakeDeviceHandleWrapper, FakeHandle>);

// The create policy is AbortPolicy<TestError>, yet the device policy is
// AbortPolicy<wwrError_t>: the device (set/get) calls carry error handling typed
// to the runtime's own error enum regardless of the handle's status type.
static_assert(
    std::is_same_v<std::remove_cvref_t<decltype(std::declval<const FakeDeviceHandleWrapper &>()
                                                    .device_policy())>,
                   AbortPolicy<wwrError_t>>);

// view() on an lvalue returns the device-aware view, hiding BaseHandle::view()'s
// plain HandleView. (The rvalue overload is =delete, as in the base, so a
// temporary cannot be viewed -- not asserted here because naming a deleted
// function inside a requires-expression is a hard error under clang, not an
// unsatisfied requirement.)
static_assert(
    std::is_same_v<decltype(std::declval<const FakeDeviceHandleWrapper &>().view()),
                   DeviceBoundHandleView<FakeHandle>>);

} // namespace wwr::extension::test
