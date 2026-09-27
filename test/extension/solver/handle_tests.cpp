// handle_tests.cpp - RAII contract of gpumod.extension.solver's two wrappers
//
// GpusolverDnHandle is a GpuBoundHandle specialisation (over
// gpusolverDnHandle_t): it records the device it was created on, since a
// cuSOLVER handle is device-bound. GpusolverDnParams stays a BaseHandle
// specialisation (over gpusolverDnParams_t) -- params carry no device. See
// test/extension/blas/handle_tests.cpp for the shape and why get() nulling on
// the moved-from object is the double-free guard. Params is the second live
// type the solver module owns, so it gets the same RAII contract as the handle.
//
// Runtime, device-requiring: gpusolverDnCreate needs a live GPU context.
// Backend-neutral -- built and run for either GPUMOD_GPU_BACKEND.

#include <gtest/gtest.h>

import std;
import gpumod.extension.common; // the error_policy concept, for the counting policy
import gpumod.extension.handle; // BaseHandle, GpuBoundHandle(View)
import gpumod.extension.solver; // re-exports gpumod.solver, so the raw handle/params types are in scope

namespace gpumod::extension::test {

// A counting policy for the destroy-exactly-once check below; see
// test/extension/blas/handle_tests.cpp for why the counter is a static (the
// GpuBoundHandle-inherited constructor takes no policy instance).
struct CountingSolverPolicy {
  using error_type = gpusolverStatus_t;
  static inline int errors = 0;
  static void reset() { errors = 0; }
  void handle_error(gpusolverStatus_t, std::source_location) noexcept { ++errors; }
};
using CountingSolverHandle = GpusolverDnHandleWrapper<CountingSolverPolicy>;

static_assert(!std::is_copy_constructible_v<GpusolverDnHandle>);
static_assert(!std::is_copy_assignable_v<GpusolverDnHandle>);
static_assert(std::is_nothrow_move_constructible_v<GpusolverDnHandle>);
static_assert(std::is_nothrow_move_assignable_v<GpusolverDnHandle>);

static_assert(!std::is_copy_constructible_v<GpusolverDnParams>);
static_assert(!std::is_copy_assignable_v<GpusolverDnParams>);
static_assert(std::is_nothrow_move_constructible_v<GpusolverDnParams>);
static_assert(std::is_nothrow_move_assignable_v<GpusolverDnParams>);

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// GpusolverDnHandle
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TEST(GpusolverDnHandleTests, DefaultConstructorCreatesHandle) {
  GpusolverDnHandle handle;
  EXPECT_NE(handle.get(), nullptr);
}

TEST(GpusolverDnHandleTests, ImplicitConversionMatchesGet) {
  GpusolverDnHandle handle;
  gpusolverDnHandle_t raw = handle; // operator gpusolverDnHandle_t()
  EXPECT_EQ(raw, handle.get());
}

TEST(GpusolverDnHandleTests, MoveConstructorTransfersOwnership) {
  GpusolverDnHandle handle1;
  gpusolverDnHandle_t raw = handle1.get();

  GpusolverDnHandle handle2(std::move(handle1));
  EXPECT_EQ(handle2.get(), raw);
  EXPECT_EQ(handle1.get(), nullptr);
}

TEST(GpusolverDnHandleTests, MoveAssignmentTransfersOwnership) {
  GpusolverDnHandle handle1;
  GpusolverDnHandle handle2;
  gpusolverDnHandle_t raw = handle1.get();

  handle2 = std::move(handle1);
  EXPECT_EQ(handle2.get(), raw);
  EXPECT_EQ(handle1.get(), nullptr);
}

TEST(GpusolverDnHandleTests, SelfMoveAssignmentKeepsHandle) {
  GpusolverDnHandle handle;
  gpusolverDnHandle_t raw = handle.get();

  handle = std::move(handle);
  EXPECT_EQ(handle.get(), raw);
}

TEST(GpusolverDnHandleTests, RecordsCreationDevice) {
  // The default constructor creates on device 0.
  GpusolverDnHandle handle;
  EXPECT_EQ(handle.dev_idx(), 0);

  // dev_idx is the (defaulted) first constructor argument. Device 0 always exists.
  GpusolverDnHandle on0(0);
  EXPECT_EQ(on0.dev_idx(), 0);
}

TEST(GpusolverDnHandleTests, MovePreservesDevice) {
  GpusolverDnHandle handle1;
  const int dev = handle1.dev_idx();

  GpusolverDnHandle handle2(std::move(handle1));
  EXPECT_EQ(handle2.dev_idx(), dev);
  EXPECT_EQ(handle1.dev_idx(), -1);
}

TEST(GpusolverDnHandleTests, ViewMirrorsOwnerHandleAndDevice) {
  // view() is inherited from GpuBoundHandle and only compile-tested elsewhere;
  // this reads the borrowed handle/device back from a live handle. The view is
  // a bare GpuBoundHandleView with no borrow-safe ops (a cuSOLVER call consumes
  // the raw handle), so mirroring get()/dev_idx() is its whole job.
  GpusolverDnHandle handle;
  const GpusolverDnHandleView view = handle.view();
  EXPECT_EQ(view.get(), handle.get());
  EXPECT_EQ(view.dev_idx(), handle.dev_idx());
}

TEST(GpusolverDnHandleTests, CustomPolicyFreesExactlyOnceAcrossMove) {
  // Proves the custom P_create/P_destroy thread through the handle and that a
  // move-then-destroy frees exactly once -- a double-free would route a failing
  // gpusolverDnDestroy through the policy and bump the counter. The existing
  // move tests only check get() == nullptr as an indirect proxy.
  CountingSolverPolicy::reset();
  {
    CountingSolverHandle source;
    const gpusolverDnHandle_t raw = source.get();

    CountingSolverHandle dest(std::move(source));
    EXPECT_EQ(dest.get(), raw);
  }
  EXPECT_EQ(CountingSolverPolicy::errors, 0);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// GpusolverDnParams
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TEST(GpusolverDnParamsTests, DefaultConstructorCreatesParams) {
  GpusolverDnParams params;
  EXPECT_NE(params.get(), nullptr);
}

TEST(GpusolverDnParamsTests, ImplicitConversionMatchesGet) {
  GpusolverDnParams params;
  gpusolverDnParams_t raw = params; // operator gpusolverDnParams_t()
  EXPECT_EQ(raw, params.get());
}

TEST(GpusolverDnParamsTests, MoveConstructorTransfersOwnership) {
  GpusolverDnParams params1;
  gpusolverDnParams_t raw = params1.get();

  GpusolverDnParams params2(std::move(params1));
  EXPECT_EQ(params2.get(), raw);
  EXPECT_EQ(params1.get(), nullptr);
}

TEST(GpusolverDnParamsTests, MoveAssignmentTransfersOwnership) {
  GpusolverDnParams params1;
  GpusolverDnParams params2;
  gpusolverDnParams_t raw = params1.get();

  params2 = std::move(params1);
  EXPECT_EQ(params2.get(), raw);
  EXPECT_EQ(params1.get(), nullptr);
}

TEST(GpusolverDnParamsTests, SelfMoveAssignmentKeepsParams) {
  GpusolverDnParams params;
  gpusolverDnParams_t raw = params.get();

  params = std::move(params);
  EXPECT_EQ(params.get(), raw);
}

} // namespace gpumod::extension::test
