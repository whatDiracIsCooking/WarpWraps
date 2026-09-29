// handle_tests.cpp - RAII contract of wwr.extension.sparse's SparseHandleWrapper
//
// SparseHandleWrapper is a StreamBoundHandle specialisation over wwrsparseHandle_t:
// created from a shared stream owner, it records the owner's device, binds the
// owner's stream (stream()), and retains the owner. See
// test/extension/blas/handle_tests.cpp for the shape and why get() nulling on the
// moved-from object is the double-free guard.
//
// Runtime, device-requiring: wwrsparseCreate needs a live GPU context.
// Backend-neutral -- built and run for either WWR_GPU_BACKEND.

#include <gtest/gtest.h>

import std;
import wwr.runtime_api; // wwrError_t, for the device-access policy
import wwr.extension.common; // the error_policy concept, for the counting policy
import wwr.extension.handle; // StreamBoundHandle, device_handle_stream, StreamBoundHandleView
import wwr.extension.runtime; // StreamWrapper, so owner->stream().get() has a complete type
import wwr.extension.sparse; // re-exports wwr.sparse, so wwrsparseHandle_t is in scope
import wwr.test.shared.abort_policy; // AbortPolicy for this file's instantiations
import wwr.test.shared.device_handle; // the reference stream owner

namespace wwr::extension::test {
// Bind abort-on-failure once, for this file's wrapper instantiations. The
// handle's create/destroy policies are wwrsparseStatus_t-typed; P_device_access
// is wwrError_t-typed (the device set/get calls), so it takes its own binding.
using Abort = AbortPolicy<wwrsparseStatus_t>;
using AbortDev = AbortPolicy<wwrError_t>;

// A counting policy for the destroy-exactly-once check below; see
// test/extension/blas/handle_tests.cpp for why the counter is a shared static.
struct CountingSparsePolicy {
  using error_type = wwrsparseStatus_t;
  static inline int errors = 0;
  static void reset() { errors = 0; }
  void handle_error(wwrsparseStatus_t, std::source_location) noexcept { ++errors; }
};
using CountingSparseHandle = SparseHandleWrapper<CountingSparsePolicy, CountingSparsePolicy, DeviceHandle, AbortDev>;

static_assert(!std::is_copy_constructible_v<SparseHandleWrapper<Abort, Abort, DeviceHandle, AbortDev>>);
static_assert(!std::is_copy_assignable_v<SparseHandleWrapper<Abort, Abort, DeviceHandle, AbortDev>>);
static_assert(std::is_nothrow_move_constructible_v<SparseHandleWrapper<Abort, Abort, DeviceHandle, AbortDev>>);
static_assert(std::is_nothrow_move_assignable_v<SparseHandleWrapper<Abort, Abort, DeviceHandle, AbortDev>>);
// A bound handle is itself a stream owner -- it can back a DeviceBuffer's async tier.
static_assert(device_handle_stream<SparseHandleWrapper<Abort, Abort, DeviceHandle, AbortDev>>);

TEST(SparseHandleTests, ConstructsLiveHandleFromOwner) {
  auto owner = std::make_shared<DeviceHandle>(0);
  SparseHandleWrapper<Abort, Abort, DeviceHandle, AbortDev> handle{owner};
  EXPECT_NE(handle.get(), nullptr);
}

TEST(SparseHandleTests, ImplicitConversionMatchesGet) {
  auto owner = std::make_shared<DeviceHandle>(0);
  SparseHandleWrapper<Abort, Abort, DeviceHandle, AbortDev> handle{owner};
  wwrsparseHandle_t raw = handle; // operator wwrsparseHandle_t()
  EXPECT_EQ(raw, handle.get());
}

TEST(SparseHandleTests, BindsOwnerStream) {
  auto owner = std::make_shared<DeviceHandle>(0);
  SparseHandleWrapper<Abort, Abort, DeviceHandle, AbortDev> handle{owner};
  EXPECT_EQ(handle.stream(), owner->stream().get());
}

TEST(SparseHandleTests, MoveConstructorTransfersOwnership) {
  auto owner = std::make_shared<DeviceHandle>(0);
  SparseHandleWrapper<Abort, Abort, DeviceHandle, AbortDev> handle1{owner};
  wwrsparseHandle_t raw = handle1.get();

  SparseHandleWrapper<Abort, Abort, DeviceHandle, AbortDev> handle2(std::move(handle1));
  EXPECT_EQ(handle2.get(), raw);
  EXPECT_EQ(handle1.get(), nullptr);
}

TEST(SparseHandleTests, MoveAssignmentTransfersOwnership) {
  auto owner = std::make_shared<DeviceHandle>(0);
  SparseHandleWrapper<Abort, Abort, DeviceHandle, AbortDev> handle1{owner};
  SparseHandleWrapper<Abort, Abort, DeviceHandle, AbortDev> handle2{owner};
  wwrsparseHandle_t raw = handle1.get();

  handle2 = std::move(handle1);
  EXPECT_EQ(handle2.get(), raw);
  EXPECT_EQ(handle1.get(), nullptr);
}

TEST(SparseHandleTests, SelfMoveAssignmentKeepsHandle) {
  auto owner = std::make_shared<DeviceHandle>(0);
  SparseHandleWrapper<Abort, Abort, DeviceHandle, AbortDev> handle{owner};
  wwrsparseHandle_t raw = handle.get();

  handle = std::move(handle);
  EXPECT_EQ(handle.get(), raw);
}

TEST(SparseHandleTests, RecordsCreationDevice) {
  // The handle is created on -- and records -- the owner's device.
  auto owner = std::make_shared<DeviceHandle>(0);
  SparseHandleWrapper<Abort, Abort, DeviceHandle, AbortDev> handle{owner};
  EXPECT_EQ(handle.dev_idx(), owner->dev_idx());
  EXPECT_EQ(handle.dev_idx(), 0); // device 0 always exists
}

TEST(SparseHandleTests, MovePreservesDevice) {
  auto owner = std::make_shared<DeviceHandle>(0);
  SparseHandleWrapper<Abort, Abort, DeviceHandle, AbortDev> handle1{owner};
  const int dev = handle1.dev_idx();

  SparseHandleWrapper<Abort, Abort, DeviceHandle, AbortDev> handle2(std::move(handle1));
  EXPECT_EQ(handle2.dev_idx(), dev);
  EXPECT_EQ(handle1.dev_idx(), -1);
}

TEST(SparseHandleTests, ViewMirrorsOwnerHandleDeviceAndStream) {
  // view() is inherited from StreamBoundHandle and only compile-tested elsewhere;
  // this reads the borrowed handle/device/stream back from a live handle. The view
  // is a StreamBoundHandleView with no borrow-safe ops (a cuSPARSE call consumes
  // the raw handle), so mirroring get()/dev_idx()/stream() is its whole job.
  auto owner = std::make_shared<DeviceHandle>(0);
  SparseHandleWrapper<Abort, Abort, DeviceHandle, AbortDev> handle{owner};
  const StreamBoundHandleView<wwrsparseHandle_t> view = handle.view();
  EXPECT_EQ(view.get(), handle.get());
  EXPECT_EQ(view.dev_idx(), handle.dev_idx());
  EXPECT_EQ(view.stream(), handle.stream());
}

TEST(SparseHandleTests, CustomPolicyFreesExactlyOnceAcrossMove) {
  // Sparse had no policy test at all (fft already proves this for its plan).
  // Substitutes a counting policy for the default (aborting) one, proving the
  // custom P_create/P_destroy thread through the handle and that a
  // move-then-destroy frees exactly once -- a double-free would route a failing
  // wwrsparseDestroy through the policy and bump the counter.
  CountingSparsePolicy::reset();
  {
    auto owner = std::make_shared<DeviceHandle>(0);
    CountingSparseHandle source{owner};
    const wwrsparseHandle_t raw = source.get();

    CountingSparseHandle dest(std::move(source));
    EXPECT_EQ(dest.get(), raw);
  }
  EXPECT_EQ(CountingSparsePolicy::errors, 0);
}

} // namespace wwr::extension::test
