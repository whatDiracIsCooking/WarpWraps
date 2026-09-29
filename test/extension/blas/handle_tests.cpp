// handle_tests.cpp - RAII contract of wwr.extension.blas's BlasHandleWrapper
//
// BlasHandleWrapper is a StreamBoundHandle specialisation over wwrblasHandle_t, so
// its whole behaviour is that layer's: created from a shared stream owner, it
// creates a live cuBLAS/hipBLAS handle on the owner's device (recorded as
// dev_idx()), binds the owner's stream (reported by stream()), retains the owner,
// hands ownership across on move (leaving the source null and dev_idx() == -1 so
// its destructor is a no-op), and destroys exactly once. These cases pin that
// contract the same way test/extension/runtime/basic.cpp pins GpuStreamWrapper's --
// through get(), whose nulling on the moved-from object is what proves the
// destructor will not double-free.
//
// The reference stream owner is the test DeviceHandle (wwr.test.shared), which
// satisfies device_handle_pool and so device_handle_stream; a shared_ptr to one
// is the sole construction path now that the stream-less (int dev_idx) ctor is
// gone.
//
// Runtime, device-requiring: wwrblasCreate needs a live GPU context, so there
// is no compile-time half. Backend-neutral -- built and run for either
// WWR_GPU_BACKEND.

#include <gtest/gtest.h>

import std;
import wwr.runtime_api; // wwrError_t, for the device-access policy
import wwr.extension.common; // the error_policy concept, for the counting policy
import wwr.extension.handle; // StreamBoundHandle, device_handle_stream, DeviceBoundHandleView
import wwr.extension.runtime; // GpuStreamWrapper, so owner->stream().get() has a complete type
import wwr.extension.blas; // re-exports wwr.blas, so wwrblasHandle_t is in scope
import wwr.test.shared.abort_policy; // AbortPolicy for this file's instantiations
import wwr.test.shared.device_handle; // the reference stream owner

namespace wwr::extension::test {
// Bind abort-on-failure once, for this file's wrapper instantiations. The
// handle's create/destroy policies are wwrblasStatus_t-typed; P_device_access is
// wwrError_t-typed (the device set/get calls), so it takes its own binding.
using Abort = AbortPolicy<wwrblasStatus_t>;
using AbortDev = AbortPolicy<wwrError_t>;

// A counting policy for the destroy-exactly-once check below. The stream-owner
// constructor accepts policy instances, but a counting policy tracking a shared
// static needs no injection -- each default-constructed instance bumps the same
// counter. It tallies failures instead of aborting (AbortPolicy would terminate
// the process), so a botched destroy is observable after the objects are gone.
struct CountingBlasPolicy {
  using error_type = wwrblasStatus_t;
  static inline int errors = 0;
  static void reset() { errors = 0; }
  void handle_error(wwrblasStatus_t, std::source_location) noexcept { ++errors; }
};
using CountingBlasHandle = BlasHandleWrapper<CountingBlasPolicy, CountingBlasPolicy, DeviceHandle, AbortDev>;

// Copy is deleted at the base; a copyable RAII handle would double-free.
static_assert(!std::is_copy_constructible_v<BlasHandleWrapper<Abort, Abort, DeviceHandle, AbortDev>>);
static_assert(!std::is_copy_assignable_v<BlasHandleWrapper<Abort, Abort, DeviceHandle, AbortDev>>);
static_assert(std::is_nothrow_move_constructible_v<BlasHandleWrapper<Abort, Abort, DeviceHandle, AbortDev>>);
static_assert(std::is_nothrow_move_assignable_v<BlasHandleWrapper<Abort, Abort, DeviceHandle, AbortDev>>);
// A bound handle is itself a stream owner -- it can back a DeviceBuffer's async tier.
static_assert(device_handle_stream<BlasHandleWrapper<Abort, Abort, DeviceHandle, AbortDev>>);

TEST(BlasHandleTests, ConstructsLiveHandleFromOwner) {
  auto owner = std::make_shared<DeviceHandle>(0);
  BlasHandleWrapper<Abort, Abort, DeviceHandle, AbortDev> handle{owner};
  EXPECT_NE(handle.get(), nullptr);
}

TEST(BlasHandleTests, ImplicitConversionMatchesGet) {
  auto owner = std::make_shared<DeviceHandle>(0);
  BlasHandleWrapper<Abort, Abort, DeviceHandle, AbortDev> handle{owner};
  wwrblasHandle_t raw = handle; // operator wwrblasHandle_t()
  EXPECT_EQ(raw, handle.get());
}

TEST(BlasHandleTests, BindsOwnerStream) {
  // stream() reports the retained owner's stream -- this is what makes the handle
  // a device_handle_stream, and what wwrblasSetStream received.
  auto owner = std::make_shared<DeviceHandle>(0);
  BlasHandleWrapper<Abort, Abort, DeviceHandle, AbortDev> handle{owner};
  EXPECT_EQ(handle.stream(), owner->stream().get());
}

TEST(BlasHandleTests, MoveConstructorTransfersOwnership) {
  auto owner = std::make_shared<DeviceHandle>(0);
  BlasHandleWrapper<Abort, Abort, DeviceHandle, AbortDev> handle1{owner};
  wwrblasHandle_t raw = handle1.get();

  BlasHandleWrapper<Abort, Abort, DeviceHandle, AbortDev> handle2(std::move(handle1));
  EXPECT_EQ(handle2.get(), raw);
  EXPECT_EQ(handle1.get(), nullptr);
}

TEST(BlasHandleTests, MoveAssignmentTransfersOwnership) {
  auto owner = std::make_shared<DeviceHandle>(0);
  BlasHandleWrapper<Abort, Abort, DeviceHandle, AbortDev> handle1{owner};
  BlasHandleWrapper<Abort, Abort, DeviceHandle, AbortDev> handle2{owner};
  wwrblasHandle_t raw = handle1.get();

  handle2 = std::move(handle1);
  EXPECT_EQ(handle2.get(), raw);
  EXPECT_EQ(handle1.get(), nullptr);
}

TEST(BlasHandleTests, SelfMoveAssignmentKeepsHandle) {
  auto owner = std::make_shared<DeviceHandle>(0);
  BlasHandleWrapper<Abort, Abort, DeviceHandle, AbortDev> handle{owner};
  wwrblasHandle_t raw = handle.get();

  handle = std::move(handle);
  EXPECT_EQ(handle.get(), raw);
}

TEST(BlasHandleTests, RecordsCreationDevice) {
  // The handle is created on -- and records -- the owner's device.
  auto owner = std::make_shared<DeviceHandle>(0);
  BlasHandleWrapper<Abort, Abort, DeviceHandle, AbortDev> handle{owner};
  EXPECT_EQ(handle.dev_idx(), owner->dev_idx());
  EXPECT_EQ(handle.dev_idx(), 0); // device 0 always exists
}

TEST(BlasHandleTests, MovePreservesDevice) {
  auto owner = std::make_shared<DeviceHandle>(0);
  BlasHandleWrapper<Abort, Abort, DeviceHandle, AbortDev> handle1{owner};
  const int dev = handle1.dev_idx();

  BlasHandleWrapper<Abort, Abort, DeviceHandle, AbortDev> handle2(std::move(handle1));
  EXPECT_EQ(handle2.dev_idx(), dev);
  EXPECT_EQ(handle1.dev_idx(), -1);
}

TEST(BlasHandleTests, ViewMirrorsOwnerHandleAndDevice) {
  // view() is inherited from DeviceBoundHandle and only compile-tested elsewhere
  // (test/extension/build_time/handle_view.cppm); nothing constructs a live
  // handle and reads the borrowed handle/device back. The view is a bare
  // DeviceBoundHandleView with no borrow-safe ops of its own -- a cuBLAS call
  // consumes the raw handle -- so mirroring get()/dev_idx() is its whole job.
  auto owner = std::make_shared<DeviceHandle>(0);
  BlasHandleWrapper<Abort, Abort, DeviceHandle, AbortDev> handle{owner};
  const DeviceBoundHandleView<wwrblasHandle_t> view = handle.view();
  EXPECT_EQ(view.get(), handle.get());
  EXPECT_EQ(view.dev_idx(), handle.dev_idx());
}

TEST(BlasHandleTests, CustomPolicyFreesExactlyOnceAcrossMove) {
  // Substitutes a counting policy for the default (aborting) one, proving the
  // custom P_create/P_destroy actually compile into and thread through the
  // handle, and that a move-then-destroy frees exactly once: a double-free
  // (source re-destroying a handle already freed by dest) would route a failing
  // wwrblasDestroy through the policy and bump the counter. The existing move
  // tests only check get() == nullptr as an indirect proxy for this.
  CountingBlasPolicy::reset();
  {
    auto owner = std::make_shared<DeviceHandle>(0);
    CountingBlasHandle source{owner};
    const wwrblasHandle_t raw = source.get();

    CountingBlasHandle dest(std::move(source));
    EXPECT_EQ(dest.get(), raw);
  }
  EXPECT_EQ(CountingBlasPolicy::errors, 0);
}

} // namespace wwr::extension::test
