// handle_tests.cpp - RAII contract of wwr.extension.solver's two wrappers
//
// WwrsolverDnHandle is a DeviceBoundHandle specialisation (over
// wwrsolverDnHandle_t): it records the device it was created on, since a
// cuSOLVER handle is device-bound. WwrsolverDnParams stays a BaseHandle
// specialisation (over wwrsolverDnParams_t) -- params carry no device. See
// test/extension/blas/handle_tests.cpp for the shape and why get() nulling on
// the moved-from object is the double-free guard. Params is the second live
// type the solver module owns, so it gets the same RAII contract as the handle.
//
// Runtime, device-requiring: wwrsolverDnCreate needs a live GPU context.
// Backend-neutral -- built and run for either WWR_GPU_BACKEND.

#include <gtest/gtest.h>

import std;
import wwr.extension.common; // the error_policy concept, for the counting policy
import wwr.extension.handle; // BaseHandle, DeviceBoundHandle(View)
import wwr.extension.solver; // re-exports wwr.solver, so the raw handle/params types are in scope

namespace wwr::extension::test {

// A counting policy for the destroy-exactly-once check below; see
// test/extension/blas/handle_tests.cpp for why the counter is a static (the
// DeviceBoundHandle-inherited constructor takes no policy instance).
struct CountingSolverPolicy {
  using error_type = wwrsolverStatus_t;
  static inline int errors = 0;
  static void reset() { errors = 0; }
  void handle_error(wwrsolverStatus_t, std::source_location) noexcept { ++errors; }
};
using CountingSolverHandle = WwrsolverDnHandleWrapper<CountingSolverPolicy>;

static_assert(!std::is_copy_constructible_v<WwrsolverDnHandle>);
static_assert(!std::is_copy_assignable_v<WwrsolverDnHandle>);
static_assert(std::is_nothrow_move_constructible_v<WwrsolverDnHandle>);
static_assert(std::is_nothrow_move_assignable_v<WwrsolverDnHandle>);

static_assert(!std::is_copy_constructible_v<WwrsolverDnParams>);
static_assert(!std::is_copy_assignable_v<WwrsolverDnParams>);
static_assert(std::is_nothrow_move_constructible_v<WwrsolverDnParams>);
static_assert(std::is_nothrow_move_assignable_v<WwrsolverDnParams>);

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// WwrsolverDnHandle
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TEST(WwrsolverDnHandleTests, DefaultConstructorCreatesHandle) {
  WwrsolverDnHandle handle;
  EXPECT_NE(handle.get(), nullptr);
}

TEST(WwrsolverDnHandleTests, ImplicitConversionMatchesGet) {
  WwrsolverDnHandle handle;
  wwrsolverDnHandle_t raw = handle; // operator wwrsolverDnHandle_t()
  EXPECT_EQ(raw, handle.get());
}

TEST(WwrsolverDnHandleTests, MoveConstructorTransfersOwnership) {
  WwrsolverDnHandle handle1;
  wwrsolverDnHandle_t raw = handle1.get();

  WwrsolverDnHandle handle2(std::move(handle1));
  EXPECT_EQ(handle2.get(), raw);
  EXPECT_EQ(handle1.get(), nullptr);
}

TEST(WwrsolverDnHandleTests, MoveAssignmentTransfersOwnership) {
  WwrsolverDnHandle handle1;
  WwrsolverDnHandle handle2;
  wwrsolverDnHandle_t raw = handle1.get();

  handle2 = std::move(handle1);
  EXPECT_EQ(handle2.get(), raw);
  EXPECT_EQ(handle1.get(), nullptr);
}

TEST(WwrsolverDnHandleTests, SelfMoveAssignmentKeepsHandle) {
  WwrsolverDnHandle handle;
  wwrsolverDnHandle_t raw = handle.get();

  handle = std::move(handle);
  EXPECT_EQ(handle.get(), raw);
}

TEST(WwrsolverDnHandleTests, RecordsCreationDevice) {
  // The default constructor creates on device 0.
  WwrsolverDnHandle handle;
  EXPECT_EQ(handle.dev_idx(), 0);

  // dev_idx is the (defaulted) first constructor argument. Device 0 always exists.
  WwrsolverDnHandle on0(0);
  EXPECT_EQ(on0.dev_idx(), 0);
}

TEST(WwrsolverDnHandleTests, MovePreservesDevice) {
  WwrsolverDnHandle handle1;
  const int dev = handle1.dev_idx();

  WwrsolverDnHandle handle2(std::move(handle1));
  EXPECT_EQ(handle2.dev_idx(), dev);
  EXPECT_EQ(handle1.dev_idx(), -1);
}

TEST(WwrsolverDnHandleTests, ViewMirrorsOwnerHandleAndDevice) {
  // view() is inherited from DeviceBoundHandle and only compile-tested elsewhere;
  // this reads the borrowed handle/device back from a live handle. The view is
  // a bare DeviceBoundHandleView with no borrow-safe ops (a cuSOLVER call consumes
  // the raw handle), so mirroring get()/dev_idx() is its whole job.
  WwrsolverDnHandle handle;
  const WwrsolverDnHandleView view = handle.view();
  EXPECT_EQ(view.get(), handle.get());
  EXPECT_EQ(view.dev_idx(), handle.dev_idx());
}

TEST(WwrsolverDnHandleTests, CustomPolicyFreesExactlyOnceAcrossMove) {
  // Proves the custom P_create/P_destroy thread through the handle and that a
  // move-then-destroy frees exactly once -- a double-free would route a failing
  // wwrsolverDnDestroy through the policy and bump the counter. The existing
  // move tests only check get() == nullptr as an indirect proxy.
  CountingSolverPolicy::reset();
  {
    CountingSolverHandle source;
    const wwrsolverDnHandle_t raw = source.get();

    CountingSolverHandle dest(std::move(source));
    EXPECT_EQ(dest.get(), raw);
  }
  EXPECT_EQ(CountingSolverPolicy::errors, 0);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// WwrsolverDnParams
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TEST(WwrsolverDnParamsTests, DefaultConstructorCreatesParams) {
  WwrsolverDnParams params;
  EXPECT_NE(params.get(), nullptr);
}

TEST(WwrsolverDnParamsTests, ImplicitConversionMatchesGet) {
  WwrsolverDnParams params;
  wwrsolverDnParams_t raw = params; // operator wwrsolverDnParams_t()
  EXPECT_EQ(raw, params.get());
}

TEST(WwrsolverDnParamsTests, MoveConstructorTransfersOwnership) {
  WwrsolverDnParams params1;
  wwrsolverDnParams_t raw = params1.get();

  WwrsolverDnParams params2(std::move(params1));
  EXPECT_EQ(params2.get(), raw);
  EXPECT_EQ(params1.get(), nullptr);
}

TEST(WwrsolverDnParamsTests, MoveAssignmentTransfersOwnership) {
  WwrsolverDnParams params1;
  WwrsolverDnParams params2;
  wwrsolverDnParams_t raw = params1.get();

  params2 = std::move(params1);
  EXPECT_EQ(params2.get(), raw);
  EXPECT_EQ(params1.get(), nullptr);
}

TEST(WwrsolverDnParamsTests, SelfMoveAssignmentKeepsParams) {
  WwrsolverDnParams params;
  wwrsolverDnParams_t raw = params.get();

  params = std::move(params);
  EXPECT_EQ(params.get(), raw);
}

} // namespace wwr::extension::test
