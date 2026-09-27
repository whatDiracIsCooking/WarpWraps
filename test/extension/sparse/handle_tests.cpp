// handle_tests.cpp - RAII contract of wwr.extension.sparse's WwrsparseHandle
//
// WwrsparseHandle is a DeviceBoundHandle specialisation over wwrsparseHandle_t: it
// records the device it was created on, since a cuSPARSE handle is
// device-bound. See test/extension/blas/handle_tests.cpp for the shape and why
// get() nulling on the moved-from object is the double-free guard.
//
// Runtime, device-requiring: wwrsparseCreate needs a live GPU context.
// Backend-neutral -- built and run for either WWR_GPU_BACKEND.

#include <gtest/gtest.h>

import std;
import wwr.extension.common; // the error_policy concept, for the counting policy
import wwr.extension.handle; // DeviceBoundHandle(View)
import wwr.extension.sparse; // re-exports wwr.sparse, so wwrsparseHandle_t is in scope

namespace wwr::extension::test {

// A counting policy for the destroy-exactly-once check below; see
// test/extension/blas/handle_tests.cpp for why the counter is a static (the
// DeviceBoundHandle-inherited constructor takes no policy instance).
struct CountingSparsePolicy {
  using error_type = wwrsparseStatus_t;
  static inline int errors = 0;
  static void reset() { errors = 0; }
  void handle_error(wwrsparseStatus_t, std::source_location) noexcept { ++errors; }
};
using CountingSparseHandle = WwrsparseHandleWrapper<CountingSparsePolicy>;

static_assert(!std::is_copy_constructible_v<WwrsparseHandle>);
static_assert(!std::is_copy_assignable_v<WwrsparseHandle>);
static_assert(std::is_nothrow_move_constructible_v<WwrsparseHandle>);
static_assert(std::is_nothrow_move_assignable_v<WwrsparseHandle>);

TEST(WwrsparseHandleTests, DefaultConstructorCreatesHandle) {
  WwrsparseHandle handle;
  EXPECT_NE(handle.get(), nullptr);
}

TEST(WwrsparseHandleTests, ImplicitConversionMatchesGet) {
  WwrsparseHandle handle;
  wwrsparseHandle_t raw = handle; // operator wwrsparseHandle_t()
  EXPECT_EQ(raw, handle.get());
}

TEST(WwrsparseHandleTests, MoveConstructorTransfersOwnership) {
  WwrsparseHandle handle1;
  wwrsparseHandle_t raw = handle1.get();

  WwrsparseHandle handle2(std::move(handle1));
  EXPECT_EQ(handle2.get(), raw);
  EXPECT_EQ(handle1.get(), nullptr);
}

TEST(WwrsparseHandleTests, MoveAssignmentTransfersOwnership) {
  WwrsparseHandle handle1;
  WwrsparseHandle handle2;
  wwrsparseHandle_t raw = handle1.get();

  handle2 = std::move(handle1);
  EXPECT_EQ(handle2.get(), raw);
  EXPECT_EQ(handle1.get(), nullptr);
}

TEST(WwrsparseHandleTests, SelfMoveAssignmentKeepsHandle) {
  WwrsparseHandle handle;
  wwrsparseHandle_t raw = handle.get();

  handle = std::move(handle);
  EXPECT_EQ(handle.get(), raw);
}

TEST(WwrsparseHandleTests, RecordsCreationDevice) {
  // The default constructor creates on device 0.
  WwrsparseHandle handle;
  EXPECT_EQ(handle.dev_idx(), 0);

  // dev_idx is the (defaulted) first constructor argument. Device 0 always exists.
  WwrsparseHandle on0(0);
  EXPECT_EQ(on0.dev_idx(), 0);
}

TEST(WwrsparseHandleTests, MovePreservesDevice) {
  WwrsparseHandle handle1;
  const int dev = handle1.dev_idx();

  WwrsparseHandle handle2(std::move(handle1));
  EXPECT_EQ(handle2.dev_idx(), dev);
  EXPECT_EQ(handle1.dev_idx(), -1);
}

TEST(WwrsparseHandleTests, ViewMirrorsOwnerHandleAndDevice) {
  // view() is inherited from DeviceBoundHandle and only compile-tested elsewhere;
  // this reads the borrowed handle/device back from a live handle. The view is
  // a bare DeviceBoundHandleView with no borrow-safe ops (a cuSPARSE call consumes
  // the raw handle), so mirroring get()/dev_idx() is its whole job.
  WwrsparseHandle handle;
  const WwrsparseHandleView view = handle.view();
  EXPECT_EQ(view.get(), handle.get());
  EXPECT_EQ(view.dev_idx(), handle.dev_idx());
}

TEST(WwrsparseHandleTests, CustomPolicyFreesExactlyOnceAcrossMove) {
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
