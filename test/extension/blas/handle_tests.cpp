// handle_tests.cpp - RAII contract of gpumod.extension.blas's GpublasHandle
//
// GpublasHandle is a GpuBoundHandle specialisation over gpublasHandle_t, so its
// whole behaviour is that layer's: create a live cuBLAS/hipBLAS handle on the
// selected device (recorded as dev_idx()), hand ownership across on move
// (leaving the source null and dev_idx() == -1 so its destructor is a no-op),
// and destroy exactly once. These cases pin that contract the same way
// test/extension/runtime/basic.cpp pins GpuStream's -- through get(), whose
// nulling on the moved-from object is what proves the destructor will not
// double-free.
//
// Runtime, device-requiring: gpublasCreate needs a live GPU context, so there
// is no compile-time half. Backend-neutral -- built and run for either
// GPUMOD_GPU_BACKEND.

#include <gtest/gtest.h>

import std;
import gpumod.extension.blas; // re-exports gpumod.blas, so gpublasHandle_t is in scope

namespace gpumod::extension::test {

// Copy is deleted at the base; a copyable RAII handle would double-free.
static_assert(!std::is_copy_constructible_v<GpublasHandle>);
static_assert(!std::is_copy_assignable_v<GpublasHandle>);
static_assert(std::is_nothrow_move_constructible_v<GpublasHandle>);
static_assert(std::is_nothrow_move_assignable_v<GpublasHandle>);

TEST(GpublasHandleTests, DefaultConstructorCreatesHandle) {
  GpublasHandle handle;
  EXPECT_NE(handle.get(), nullptr);
}

TEST(GpublasHandleTests, ImplicitConversionMatchesGet) {
  GpublasHandle handle;
  gpublasHandle_t raw = handle; // operator gpublasHandle_t()
  EXPECT_EQ(raw, handle.get());
}

TEST(GpublasHandleTests, MoveConstructorTransfersOwnership) {
  GpublasHandle handle1;
  gpublasHandle_t raw = handle1.get();

  GpublasHandle handle2(std::move(handle1));
  EXPECT_EQ(handle2.get(), raw);
  EXPECT_EQ(handle1.get(), nullptr);
}

TEST(GpublasHandleTests, MoveAssignmentTransfersOwnership) {
  GpublasHandle handle1;
  GpublasHandle handle2;
  gpublasHandle_t raw = handle1.get();

  handle2 = std::move(handle1);
  EXPECT_EQ(handle2.get(), raw);
  EXPECT_EQ(handle1.get(), nullptr);
}

TEST(GpublasHandleTests, SelfMoveAssignmentKeepsHandle) {
  GpublasHandle handle;
  gpublasHandle_t raw = handle.get();

  handle = std::move(handle);
  EXPECT_EQ(handle.get(), raw);
}

TEST(GpublasHandleTests, RecordsCreationDevice) {
  // The default constructor creates on device 0.
  GpublasHandle handle;
  EXPECT_EQ(handle.dev_idx(), 0);

  // dev_idx is the (defaulted) first constructor argument. Device 0 always exists.
  GpublasHandle on0(0);
  EXPECT_EQ(on0.dev_idx(), 0);
}

TEST(GpublasHandleTests, MovePreservesDevice) {
  GpublasHandle handle1;
  const int dev = handle1.dev_idx();

  GpublasHandle handle2(std::move(handle1));
  EXPECT_EQ(handle2.dev_idx(), dev);
  EXPECT_EQ(handle1.dev_idx(), -1);
}

} // namespace gpumod::extension::test
