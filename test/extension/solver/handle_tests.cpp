// handle_tests.cpp - RAII contract of wwr.extension.solver's two wrappers
//
// SolverDnHandleWrapper is a StreamBoundHandle specialisation (over
// wwrsolverDnHandle_t): created from a shared stream owner, it records the
// owner's device, binds the owner's stream (stream()), and retains the owner.
// SolverDnParamsWrapper stays a BaseHandle specialisation (over
// wwrsolverDnParams_t) -- params carry no device and no stream, so they are
// untouched by the stream-bound layer. See test/extension/blas/handle_tests.cpp
// for the shape and why get() nulling on the moved-from object is the
// double-free guard. Params is the second live type the solver module owns, so
// it gets the same RAII contract as the handle.
//
// Runtime, device-requiring: wwrsolverDnCreate needs a live GPU context.
// Backend-neutral -- built and run for either WWR_GPU_BACKEND.

#include <gtest/gtest.h>

import std;
import wwr.runtime_api; // wwrError_t, for the device-access policy
import wwr.extension.common; // the error_policy concept, for the counting policy
import wwr.extension.handle; // BaseHandle, StreamBoundHandle, StreamBoundHandleView
import wwr.extension.runtime; // StreamWrapper, so owner->stream().get() has a complete type
import wwr.extension.solver; // re-exports wwr.solver, so the raw handle/params types are in scope
import wwr.test.shared.abort_policy; // AbortPolicy for this file's instantiations
import wwr.test.shared.device_handle; // the reference stream owner

namespace wwr::extension::test {
// Bind abort-on-failure once, for this file's wrapper instantiations. The
// handle's create/destroy policies are wwrsolverStatus_t-typed; P_device_access
// is wwrError_t-typed (the device set/get calls), so it takes its own binding.
using Abort = AbortPolicy<wwrsolverStatus_t>;
using AbortDev = AbortPolicy<wwrError_t>;

// A counting policy for the destroy-exactly-once check below; see
// test/extension/blas/handle_tests.cpp for why the counter is a shared static.
struct CountingSolverPolicy {
  using error_type = wwrsolverStatus_t;
  static inline int errors = 0;
  static void reset() { errors = 0; }
  void handle_error(wwrsolverStatus_t, std::source_location) noexcept { ++errors; }
};
using CountingSolverHandle = SolverDnHandleWrapper<CountingSolverPolicy, CountingSolverPolicy, DeviceHandle, AbortDev>;

static_assert(!std::is_copy_constructible_v<SolverDnHandleWrapper<Abort, Abort, DeviceHandle, AbortDev>>);
static_assert(!std::is_copy_assignable_v<SolverDnHandleWrapper<Abort, Abort, DeviceHandle, AbortDev>>);
static_assert(std::is_nothrow_move_constructible_v<SolverDnHandleWrapper<Abort, Abort, DeviceHandle, AbortDev>>);
static_assert(std::is_nothrow_move_assignable_v<SolverDnHandleWrapper<Abort, Abort, DeviceHandle, AbortDev>>);
// A bound handle is itself a stream owner -- it can back a DeviceBuffer's async tier.
static_assert(device_handle_stream<SolverDnHandleWrapper<Abort, Abort, DeviceHandle, AbortDev>>);

static_assert(!std::is_copy_constructible_v<SolverDnParamsWrapper<Abort, Abort>>);
static_assert(!std::is_copy_assignable_v<SolverDnParamsWrapper<Abort, Abort>>);
static_assert(std::is_nothrow_move_constructible_v<SolverDnParamsWrapper<Abort, Abort>>);
static_assert(std::is_nothrow_move_assignable_v<SolverDnParamsWrapper<Abort, Abort>>);

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// SolverDnHandleWrapper
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TEST(SolverDnHandleTests, ConstructsLiveHandleFromOwner) {
  auto owner = std::make_shared<DeviceHandle>(0);
  SolverDnHandleWrapper<Abort, Abort, DeviceHandle, AbortDev> handle{owner};
  EXPECT_NE(handle.get(), nullptr);
}

TEST(SolverDnHandleTests, ImplicitConversionMatchesGet) {
  auto owner = std::make_shared<DeviceHandle>(0);
  SolverDnHandleWrapper<Abort, Abort, DeviceHandle, AbortDev> handle{owner};
  wwrsolverDnHandle_t raw = handle; // operator wwrsolverDnHandle_t()
  EXPECT_EQ(raw, handle.get());
}

TEST(SolverDnHandleTests, BindsOwnerStream) {
  auto owner = std::make_shared<DeviceHandle>(0);
  SolverDnHandleWrapper<Abort, Abort, DeviceHandle, AbortDev> handle{owner};
  EXPECT_EQ(handle.stream(), owner->stream().get());
}

TEST(SolverDnHandleTests, MoveConstructorTransfersOwnership) {
  auto owner = std::make_shared<DeviceHandle>(0);
  SolverDnHandleWrapper<Abort, Abort, DeviceHandle, AbortDev> handle1{owner};
  wwrsolverDnHandle_t raw = handle1.get();

  SolverDnHandleWrapper<Abort, Abort, DeviceHandle, AbortDev> handle2(std::move(handle1));
  EXPECT_EQ(handle2.get(), raw);
  EXPECT_EQ(handle1.get(), nullptr);
}

TEST(SolverDnHandleTests, MoveAssignmentTransfersOwnership) {
  auto owner = std::make_shared<DeviceHandle>(0);
  SolverDnHandleWrapper<Abort, Abort, DeviceHandle, AbortDev> handle1{owner};
  SolverDnHandleWrapper<Abort, Abort, DeviceHandle, AbortDev> handle2{owner};
  wwrsolverDnHandle_t raw = handle1.get();

  handle2 = std::move(handle1);
  EXPECT_EQ(handle2.get(), raw);
  EXPECT_EQ(handle1.get(), nullptr);
}

TEST(SolverDnHandleTests, SelfMoveAssignmentKeepsHandle) {
  auto owner = std::make_shared<DeviceHandle>(0);
  SolverDnHandleWrapper<Abort, Abort, DeviceHandle, AbortDev> handle{owner};
  wwrsolverDnHandle_t raw = handle.get();

  handle = std::move(handle);
  EXPECT_EQ(handle.get(), raw);
}

TEST(SolverDnHandleTests, RecordsCreationDevice) {
  // The handle is created on -- and records -- the owner's device.
  auto owner = std::make_shared<DeviceHandle>(0);
  SolverDnHandleWrapper<Abort, Abort, DeviceHandle, AbortDev> handle{owner};
  EXPECT_EQ(handle.dev_idx(), owner->dev_idx());
  EXPECT_EQ(handle.dev_idx(), 0); // device 0 always exists
}

TEST(SolverDnHandleTests, MovePreservesDevice) {
  auto owner = std::make_shared<DeviceHandle>(0);
  SolverDnHandleWrapper<Abort, Abort, DeviceHandle, AbortDev> handle1{owner};
  const int dev = handle1.dev_idx();

  SolverDnHandleWrapper<Abort, Abort, DeviceHandle, AbortDev> handle2(std::move(handle1));
  EXPECT_EQ(handle2.dev_idx(), dev);
  EXPECT_EQ(handle1.dev_idx(), -1);
}

TEST(SolverDnHandleTests, ViewMirrorsOwnerHandleDeviceAndStream) {
  // view() is inherited from StreamBoundHandle and only compile-tested elsewhere;
  // this reads the borrowed handle/device/stream back from a live handle. The view
  // is a StreamBoundHandleView with no borrow-safe ops (a cuSOLVER call consumes
  // the raw handle), so mirroring get()/dev_idx()/stream() is its whole job.
  auto owner = std::make_shared<DeviceHandle>(0);
  SolverDnHandleWrapper<Abort, Abort, DeviceHandle, AbortDev> handle{owner};
  const StreamBoundHandleView<wwrsolverDnHandle_t> view = handle.view();
  EXPECT_EQ(view.get(), handle.get());
  EXPECT_EQ(view.dev_idx(), handle.dev_idx());
  EXPECT_EQ(view.stream(), handle.stream());
}

TEST(SolverDnHandleTests, CustomPolicyFreesExactlyOnceAcrossMove) {
  // Proves the custom P_create/P_destroy thread through the handle and that a
  // move-then-destroy frees exactly once -- a double-free would route a failing
  // wwrsolverDnDestroy through the policy and bump the counter. The existing
  // move tests only check get() == nullptr as an indirect proxy.
  CountingSolverPolicy::reset();
  {
    auto owner = std::make_shared<DeviceHandle>(0);
    CountingSolverHandle source{owner};
    const wwrsolverDnHandle_t raw = source.get();

    CountingSolverHandle dest(std::move(source));
    EXPECT_EQ(dest.get(), raw);
  }
  EXPECT_EQ(CountingSolverPolicy::errors, 0);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// SolverDnParamsWrapper
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TEST(WwrsolverDnParamsTests, DefaultConstructorCreatesParams) {
  SolverDnParamsWrapper<Abort, Abort> params;
  EXPECT_NE(params.get(), nullptr);
}

TEST(WwrsolverDnParamsTests, ImplicitConversionMatchesGet) {
  SolverDnParamsWrapper<Abort, Abort> params;
  wwrsolverDnParams_t raw = params; // operator wwrsolverDnParams_t()
  EXPECT_EQ(raw, params.get());
}

TEST(WwrsolverDnParamsTests, MoveConstructorTransfersOwnership) {
  SolverDnParamsWrapper<Abort, Abort> params1;
  wwrsolverDnParams_t raw = params1.get();

  SolverDnParamsWrapper<Abort, Abort> params2(std::move(params1));
  EXPECT_EQ(params2.get(), raw);
  EXPECT_EQ(params1.get(), nullptr);
}

TEST(WwrsolverDnParamsTests, MoveAssignmentTransfersOwnership) {
  SolverDnParamsWrapper<Abort, Abort> params1;
  SolverDnParamsWrapper<Abort, Abort> params2;
  wwrsolverDnParams_t raw = params1.get();

  params2 = std::move(params1);
  EXPECT_EQ(params2.get(), raw);
  EXPECT_EQ(params1.get(), nullptr);
}

TEST(WwrsolverDnParamsTests, SelfMoveAssignmentKeepsParams) {
  SolverDnParamsWrapper<Abort, Abort> params;
  wwrsolverDnParams_t raw = params.get();

  params = std::move(params);
  EXPECT_EQ(params.get(), raw);
}

} // namespace wwr::extension::test
