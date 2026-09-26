// handle_tests.cpp - RAII contract of gpumod.extension.sparse's GpusparseHandle
//
// GpusparseHandle is a GpuBoundHandle specialisation over gpusparseHandle_t: it
// records the device it was created on, since a cuSPARSE handle is
// device-bound. See test/extension/blas/handle_tests.cpp for the shape and why
// get() nulling on the moved-from object is the double-free guard.
//
// Runtime, device-requiring: gpusparseCreate needs a live GPU context.
// Backend-neutral -- built and run for either GPUMOD_GPU_BACKEND.

#include <gtest/gtest.h>

import std;
import gpumod.extension.sparse; // re-exports gpumod.sparse, so gpusparseHandle_t is in scope

namespace gpumod::extension::test {

static_assert(!std::is_copy_constructible_v<GpusparseHandle>);
static_assert(!std::is_copy_assignable_v<GpusparseHandle>);
static_assert(std::is_nothrow_move_constructible_v<GpusparseHandle>);
static_assert(std::is_nothrow_move_assignable_v<GpusparseHandle>);

TEST(GpusparseHandleTests, DefaultConstructorCreatesHandle) {
  GpusparseHandle handle;
  EXPECT_NE(handle.get(), nullptr);
}

TEST(GpusparseHandleTests, ImplicitConversionMatchesGet) {
  GpusparseHandle handle;
  gpusparseHandle_t raw = handle; // operator gpusparseHandle_t()
  EXPECT_EQ(raw, handle.get());
}

TEST(GpusparseHandleTests, MoveConstructorTransfersOwnership) {
  GpusparseHandle handle1;
  gpusparseHandle_t raw = handle1.get();

  GpusparseHandle handle2(std::move(handle1));
  EXPECT_EQ(handle2.get(), raw);
  EXPECT_EQ(handle1.get(), nullptr);
}

TEST(GpusparseHandleTests, MoveAssignmentTransfersOwnership) {
  GpusparseHandle handle1;
  GpusparseHandle handle2;
  gpusparseHandle_t raw = handle1.get();

  handle2 = std::move(handle1);
  EXPECT_EQ(handle2.get(), raw);
  EXPECT_EQ(handle1.get(), nullptr);
}

TEST(GpusparseHandleTests, SelfMoveAssignmentKeepsHandle) {
  GpusparseHandle handle;
  gpusparseHandle_t raw = handle.get();

  handle = std::move(handle);
  EXPECT_EQ(handle.get(), raw);
}

TEST(GpusparseHandleTests, RecordsCreationDevice) {
  // The default constructor creates on device 0.
  GpusparseHandle handle;
  EXPECT_EQ(handle.dev_idx(), 0);

  // dev_idx is the (defaulted) first constructor argument. Device 0 always exists.
  GpusparseHandle on0(0);
  EXPECT_EQ(on0.dev_idx(), 0);
}

TEST(GpusparseHandleTests, MovePreservesDevice) {
  GpusparseHandle handle1;
  const int dev = handle1.dev_idx();

  GpusparseHandle handle2(std::move(handle1));
  EXPECT_EQ(handle2.dev_idx(), dev);
  EXPECT_EQ(handle1.dev_idx(), -1);
}

} // namespace gpumod::extension::test
