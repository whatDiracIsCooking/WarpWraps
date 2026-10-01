// scratch_tests.cpp - RAII contract of wwr.extension.comp's CompScratch
//
// CompScratch owns the device temp/scratch buffer the batched LLIF borrows. It is
// a thin front over DeviceBufferWrapper<std::byte, ...>, so its RAII contract --
// alloc on construct, free on destroy, move-only, destroy-exactly-once -- is the
// underlying buffer's. These cases pin that the composition does not drop it: a
// scratch round-trips (allocate, hand out a device pointer + byte count, free)
// cleanly, a zero-byte request is the valid (nullptr, 0) "no temp" pair, and a
// move does not double-free (a counting error policy on the free path would tally
// a second free; it must stay zero).
//
// Runtime, device-requiring: the scratch is a real device allocation. Its error
// type is wwrError_t (the allocation path), distinct from wwrcompStatus_t (what
// the batched calls return) -- so the policies here are wwrError_t-typed, exactly
// as DeviceBufferWrapper's are. Backend-neutral -- built and run for either
// WWR_GPU_BACKEND.

#include <gtest/gtest.h>

import std;
import wwr.runtime_api;   // wwrError_t
import wwr.extension.common; // error_policy concept, for the counting policy
import wwr.extension.handle; // device_handle concepts (static_asserts)
import wwr.extension.comp;    // CompScratch
import wwr.test.shared.abort_policy;  // AbortPolicy for this file's instantiations
import wwr.test.shared.device_handle; // the reference device handle (pool tier)

namespace wwr::extension::test {

using Abort = AbortPolicy<wwrError_t>;

// An error policy that tallies failures into an external counter instead of
// aborting, so a botched free is observable after the objects are gone rather
// than terminating the process (which AbortPolicy would). wwrError_t-typed,
// because the scratch's alloc/free is a runtime-allocator call.
struct CountingErrorPolicy {
  using error_type = wwrError_t;
  int *errors = nullptr;

  CountingErrorPolicy() = default;
  explicit CountingErrorPolicy(int *counter) : errors(counter) {}

  void handle_error(wwrError_t, std::source_location) noexcept {
    if (errors != nullptr) {
      ++*errors;
    }
  }
};

using Scratch = CompScratch<Abort, Abort, Abort, DeviceHandle>;
using CountingScratch = CompScratch<CountingErrorPolicy, CountingErrorPolicy, Abort, DeviceHandle>;

static_assert(!std::is_copy_constructible_v<Scratch>);
static_assert(!std::is_copy_assignable_v<Scratch>);
static_assert(std::is_nothrow_move_constructible_v<Scratch>);

TEST(CompScratchTests, AllocatesRequestedBytes) {
  auto handle = std::make_shared<DeviceHandle>(0);
  constexpr std::size_t n = 4096;
  Scratch scratch(n, handle);
  EXPECT_EQ(scratch.bytes(), n);
  EXPECT_NE(scratch.data(), nullptr);
  EXPECT_FALSE(scratch.empty());
}

TEST(CompScratchTests, ZeroBytesIsTheEmptyNoTempPair) {
  // Some algorithm/shape pairs report temp_bytes == 0; the batched calls accept
  // (nullptr, 0) for "no temp needed", which is exactly what an empty scratch is.
  auto handle = std::make_shared<DeviceHandle>(0);
  Scratch scratch(0, handle);
  EXPECT_EQ(scratch.bytes(), 0u);
  EXPECT_EQ(scratch.data(), nullptr);
  EXPECT_TRUE(scratch.empty());
}

TEST(CompScratchTests, ConstructAndDestroyReportNoError) {
  auto handle = std::make_shared<DeviceHandle>(0);
  int errors = 0;
  {
    CountingScratch scratch(4096, handle, CountingErrorPolicy{&errors});
  }
  EXPECT_EQ(errors, 0);
}

TEST(CompScratchTests, MoveConstructorTransfersOwnershipAndFreesOnce) {
  auto handle = std::make_shared<DeviceHandle>(0);
  int errors = 0;
  {
    CountingScratch source(4096, handle, CountingErrorPolicy{&errors});
    void *raw = source.data();

    CountingScratch dest(std::move(source));
    EXPECT_EQ(dest.data(), raw);  // the block moved across
    EXPECT_EQ(dest.bytes(), 4096u);
  }
  // A move that failed to clear the source's ownership would free the same block
  // twice; the second free fails and the policy counts it.
  EXPECT_EQ(errors, 0);
}

TEST(CompScratchTests, BorrowAccessorsMatchUnderlyingBuffer) {
  // data()/bytes() are the borrow the batched calls take; they must agree with
  // the underlying buffer the scratch composes.
  auto handle = std::make_shared<DeviceHandle>(0);
  Scratch scratch(2048, handle);
  EXPECT_EQ(scratch.data(), scratch.buffer().data());
  EXPECT_EQ(scratch.bytes(), scratch.buffer().size_bytes());
}

} // namespace wwr::extension::test
