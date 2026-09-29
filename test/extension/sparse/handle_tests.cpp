// handle_tests.cpp - RAII contract of wwr.extension.sparse's SparseHandleWrapper
//
// SparseHandleWrapper is a DeviceBoundHandle specialisation over wwrsparseHandle_t: it
// records the device it was created on, since a cuSPARSE handle is
// device-bound. See test/extension/blas/handle_tests.cpp for the shape and why
// get() nulling on the moved-from object is the double-free guard.
//
// Runtime, device-requiring: wwrsparseCreate needs a live GPU context.
// Backend-neutral -- built and run for either WWR_GPU_BACKEND.

#include <gtest/gtest.h>

import std;
import wwr.runtime_api; // wwrError_t, for the device-access policy
import wwr.extension.common; // the error_policy concept, for the counting policy
import wwr.extension.handle; // DeviceBoundHandle(View)
import wwr.extension.sparse; // re-exports wwr.sparse, so wwrsparseHandle_t is in scope
import wwr.test.shared.abort_policy; // AbortPolicy for this file's instantiations

namespace wwr::extension::test {
// Bind abort-on-failure once, for this file's wrapper instantiations. The
// handle's create/destroy policies are wwrsparseStatus_t-typed; P_device_access
// is wwrError_t-typed (the device set/get calls), so it takes its own binding.
using Abort = AbortPolicy<wwrsparseStatus_t>;
using AbortDev = AbortPolicy<wwrError_t>;

// A counting policy for the destroy-exactly-once check below; see
// test/extension/blas/handle_tests.cpp for why the counter is a static (the
// DeviceBoundHandle-inherited constructor takes no policy instance).
struct CountingSparsePolicy {
  using error_type = wwrsparseStatus_t;
  static inline int errors = 0;
  static void reset() { errors = 0; }
  void handle_error(wwrsparseStatus_t, std::source_location) noexcept { ++errors; }
};
using CountingSparseHandle = SparseHandleWrapper<CountingSparsePolicy, CountingSparsePolicy, AbortDev>;

static_assert(!std::is_copy_constructible_v<SparseHandleWrapper<Abort, Abort, AbortDev>>);
static_assert(!std::is_copy_assignable_v<SparseHandleWrapper<Abort, Abort, AbortDev>>);
static_assert(std::is_nothrow_move_constructible_v<SparseHandleWrapper<Abort, Abort, AbortDev>>);
static_assert(std::is_nothrow_move_assignable_v<SparseHandleWrapper<Abort, Abort, AbortDev>>);

TEST(SparseHandleTests, DefaultConstructorCreatesHandle) {
  SparseHandleWrapper<Abort, Abort, AbortDev> handle;
  EXPECT_NE(handle.get(), nullptr);
}

TEST(SparseHandleTests, ImplicitConversionMatchesGet) {
  SparseHandleWrapper<Abort, Abort, AbortDev> handle;
  wwrsparseHandle_t raw = handle; // operator wwrsparseHandle_t()
  EXPECT_EQ(raw, handle.get());
}

TEST(SparseHandleTests, MoveConstructorTransfersOwnership) {
  SparseHandleWrapper<Abort, Abort, AbortDev> handle1;
  wwrsparseHandle_t raw = handle1.get();

  SparseHandleWrapper<Abort, Abort, AbortDev> handle2(std::move(handle1));
  EXPECT_EQ(handle2.get(), raw);
  EXPECT_EQ(handle1.get(), nullptr);
}

TEST(SparseHandleTests, MoveAssignmentTransfersOwnership) {
  SparseHandleWrapper<Abort, Abort, AbortDev> handle1;
  SparseHandleWrapper<Abort, Abort, AbortDev> handle2;
  wwrsparseHandle_t raw = handle1.get();

  handle2 = std::move(handle1);
  EXPECT_EQ(handle2.get(), raw);
  EXPECT_EQ(handle1.get(), nullptr);
}

TEST(SparseHandleTests, SelfMoveAssignmentKeepsHandle) {
  SparseHandleWrapper<Abort, Abort, AbortDev> handle;
  wwrsparseHandle_t raw = handle.get();

  handle = std::move(handle);
  EXPECT_EQ(handle.get(), raw);
}

TEST(SparseHandleTests, RecordsCreationDevice) {
  // The default constructor creates on device 0.
  SparseHandleWrapper<Abort, Abort, AbortDev> handle;
  EXPECT_EQ(handle.dev_idx(), 0);

  // dev_idx is the (defaulted) first constructor argument. Device 0 always exists.
  SparseHandleWrapper<Abort, Abort, AbortDev> on0(0);
  EXPECT_EQ(on0.dev_idx(), 0);
}

TEST(SparseHandleTests, MovePreservesDevice) {
  SparseHandleWrapper<Abort, Abort, AbortDev> handle1;
  const int dev = handle1.dev_idx();

  SparseHandleWrapper<Abort, Abort, AbortDev> handle2(std::move(handle1));
  EXPECT_EQ(handle2.dev_idx(), dev);
  EXPECT_EQ(handle1.dev_idx(), -1);
}

TEST(SparseHandleTests, ViewMirrorsOwnerHandleAndDevice) {
  // view() is inherited from DeviceBoundHandle and only compile-tested elsewhere;
  // this reads the borrowed handle/device back from a live handle. The view is
  // a bare DeviceBoundHandleView with no borrow-safe ops (a cuSPARSE call consumes
  // the raw handle), so mirroring get()/dev_idx() is its whole job.
  SparseHandleWrapper<Abort, Abort, AbortDev> handle;
  const DeviceBoundHandleView<wwrsparseHandle_t> view = handle.view();
  EXPECT_EQ(view.get(), handle.get());
  EXPECT_EQ(view.dev_idx(), handle.dev_idx());
}

TEST(SparseHandleTests, CustomPolicyFreesExactlyOnceAcrossMove) {
  // Sparse had no policy test at all (fft already proves this for its plan).
  // Substitutes a counting policy for the default (aborting) one, proving the
  // custom P_create/P_destroy thread through the handle and that a
  // move-then-destroy frees exactly once -- a double-free would route a failing
  // wwrsparseDestroy through the policy and bump the counter.
  CountingSparsePolicy::reset();
  {
    CountingSparseHandle source;
    const wwrsparseHandle_t raw = source.get();

    CountingSparseHandle dest(std::move(source));
    EXPECT_EQ(dest.get(), raw);
  }
  EXPECT_EQ(CountingSparsePolicy::errors, 0);
}

} // namespace wwr::extension::test
