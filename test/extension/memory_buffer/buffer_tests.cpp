// buffer_tests.cpp - Tests for gpumod.extension.memory_buffer
//
// Covers the RAII contract of every buffer kind, non-owning views, the
// bounds/overflow rejection paths, and the stream-ordered allocate/free pairing
// for asynchronous device allocations.
//
// A plain .cpp, not a .cppm: GoogleTest's headers and `import std;` coexist
// fine in an ordinary TU, but `#include <gtest/gtest.h>` in a module interface
// unit's global module fragment runs into the same std-module conflict this
// project already hit with CUDA headers. Compiling straight into the executable
// also means the self-registering TEST() objects cannot be dropped by the
// linker, so none of the --whole-archive machinery is needed.

#include <gtest/gtest.h>

import std;
import gpumod.runtime_api;
import gpumod.extension.common;
import gpumod.extension.runtime;
import gpumod.extension.memory_buffer;

#include "counting_policy.h"

namespace gpumod::extension::test {

namespace ext = gpumod::extension;

// CountingPolicy and the Counted* buffer aliases used across the failure-path
// suites live in counting_policy.h, shared with allocation_failure_tests.cpp.

// The compile-time contract of the buffer types (move-only, nothrow moves,
// views copyable, element_size) is checked by static_assert in
// test/extension/build_time/memory_buffer.cppm, which builds without this
// executable.

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Owning host-accessible buffers: the shared RAII contract, once per kind
//
// HostBuffer, PinnedBuffer and UnifiedBuffer are all owning, host-accessible
// (operator[]), zero-initialising, and built from a bare element count. Their
// allocate/zero-init and the move that transfers ownership are one shared
// BufferBase implementation, so a typed suite proves that contract once per
// kind instead of copying three near-identical TEST()s per property. It also
// covers move-ASSIGNMENT for the pinned and unified kinds, which the old
// per-kind tests never did -- only the host kind's was exercised.
//
// Device memory is excluded on purpose: it is not host-accessible, is built
// from a DeviceHandle rather than a count, and its move frees on a stream --
// see DeviceBufferTests, which carries its own move-assignment test.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

template<typename Buf>
class HostAccessibleOwningBufferTest : public ::testing::Test {};

using HostAccessibleOwningBufferTypes =
    ::testing::Types<HostBuffer<float>, PinnedBuffer<float>, UnifiedBuffer<float>>;
TYPED_TEST_SUITE(HostAccessibleOwningBufferTest, HostAccessibleOwningBufferTypes);

TYPED_TEST(HostAccessibleOwningBufferTest, AllocatesAndZeroInitialises) {
  TypeParam buf(16);
  ASSERT_NE(buf.data(), nullptr);
  EXPECT_EQ(buf.num_elements(), std::size_t{16});
  EXPECT_EQ(buf.size_bytes(), std::size_t{16 * sizeof(float)});
  for (std::size_t i = 0; i < buf.num_elements(); ++i) {
    EXPECT_EQ(buf[i], 0.0f) << "at index " << i;
  }
}

TYPED_TEST(HostAccessibleOwningBufferTest, MoveConstructorTransfersOwnership) {
  TypeParam src(8);
  float *const raw = src.data();

  TypeParam dst(std::move(src));
  EXPECT_EQ(dst.data(), raw);
  EXPECT_EQ(dst.num_elements(), std::size_t{8});
  EXPECT_EQ(src.data(), nullptr);
  EXPECT_EQ(src.num_elements(), std::size_t{0});
}

TYPED_TEST(HostAccessibleOwningBufferTest, MoveAssignmentReleasesThenTakesOwnership) {
  TypeParam src(8);
  TypeParam dst(32);
  float *const raw = src.data();

  dst = std::move(src);
  EXPECT_EQ(dst.data(), raw);
  EXPECT_EQ(dst.num_elements(), std::size_t{8});
  EXPECT_EQ(src.data(), nullptr);
  EXPECT_EQ(src.num_elements(), std::size_t{0});
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// HostBuffer
//
// Zero-init, the move constructor and move-assignment are covered for every
// owning host-accessible kind by HostAccessibleOwningBufferTest above; what
// stays here is host-specific -- the empty/zero-element invariants, the
// error-policy path, and the move edge cases that exist only once.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TEST(HostBufferTests, DefaultConstructedIsEmpty) {
  HostBuffer<float> buf;
  EXPECT_EQ(buf.data(), nullptr);
  EXPECT_EQ(buf.num_elements(), std::size_t{0});
  EXPECT_EQ(buf.size_bytes(), std::size_t{0});
}

TEST(HostBufferTests, ZeroElementsIsEmptyNotAnError) {
  HostPolicy policy;
  CountedHostBuffer<float> buf(0, policy);
  EXPECT_EQ(buf.alloc_policy().count(), std::size_t{0});
  EXPECT_EQ(buf.data(), nullptr);
  EXPECT_EQ(buf.num_elements(), std::size_t{0});
}

TEST(HostBufferTests, SelfMoveAssignmentKeepsBuffer) {
  HostBuffer<float> buf(8);
  float *const raw = buf.data();

  buf = std::move(buf);
  EXPECT_EQ(buf.data(), raw);
  EXPECT_EQ(buf.num_elements(), std::size_t{8});
}

TEST(HostBufferTests, MoveAssignmentOntoEmptyBuffer) {
  HostBuffer<float> dst;
  HostBuffer<float> src(4);
  float *const raw = src.data();

  dst = std::move(src);
  EXPECT_EQ(dst.data(), raw);
  EXPECT_EQ(dst.num_elements(), std::size_t{4});
}

TEST(HostBufferTests, SubscriptReadsAndWrites) {
  HostBuffer<float> buf(4);
  for (std::size_t i = 0; i < 4; ++i)
    buf[i] = static_cast<float>(i) + 1.0f;
  EXPECT_EQ(buf[0], 1.0f);
  EXPECT_EQ(buf[3], 4.0f);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Size-overflow rejection
//
// These reject an oversized element count BEFORE any allocation, so they are
// safe under every sanitizer and stay here. The sibling cases that force a
// real allocation to fail live in allocation_failure_tests.cpp, which carries
// the no_sanitizer label -- see this directory's CMakeLists.txt.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TEST(SizeOverflowRejectionTests, ElementCountOverflowingSizeBytesIsRejected) {
  // n * sizeof(float) wraps to a small byte count. Unchecked, this
  // yields a tiny allocation behind a buffer still advertising n
  // elements, so every later bounds check passes.
  HostPolicy policy;
  CountedHostBuffer<float> buf(CountedHostBuffer<float>::max_num_elements + 1, policy);

  EXPECT_EQ(buf.alloc_policy().count(), std::size_t{1});
  EXPECT_EQ(buf.alloc_policy().last(), stdHostMemInvalidValue);
  EXPECT_EQ(buf.data(), nullptr);
  EXPECT_EQ(buf.num_elements(), std::size_t{0});
}

TEST(SizeOverflowRejectionTests, DeviceElementCountOverflowIsRejected) {
  GpuPolicy policy;
  auto dev = std::make_shared<DeviceHandle>(0);
  CountedDeviceBuffer<float> buf(CountedDeviceBuffer<float>::max_num_elements + 1, dev, policy);

  EXPECT_EQ(buf.alloc_policy().count(), std::size_t{1});
  EXPECT_EQ(buf.alloc_policy().last(), gpuErrorInvalidValue);
  EXPECT_EQ(buf.data(), nullptr);
  EXPECT_EQ(buf.num_elements(), std::size_t{0});
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Buffer views
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TEST(BufferViewTests, FullViewAliasesSourceBuffer) {
  HostBuffer<float> buf(16);
  HostBufferView<float> view(buf);

  EXPECT_EQ(view.data(), buf.data());
  EXPECT_EQ(view.num_elements(), std::size_t{16});
}

TEST(BufferViewTests, SubViewOffsetsIntoSourceBuffer) {
  HostBuffer<float> buf(16);
  for (std::size_t i = 0; i < 16; ++i)
    buf[i] = static_cast<float>(i);

  HostBufferView<float> view(buf, 4, 8);
  ASSERT_EQ(view.data(), buf.data() + 4);
  EXPECT_EQ(view.num_elements(), std::size_t{8});
  EXPECT_EQ(view[0], 4.0f);
  EXPECT_EQ(view[7], 11.0f);
}

TEST(BufferViewTests, OffsetOnlyViewCoversRemainingElements) {
  HostBuffer<float> buf(16);
  HostBufferView<float> view(buf, 12);

  EXPECT_EQ(view.data(), buf.data() + 12);
  EXPECT_EQ(view.num_elements(), std::size_t{4});
}

TEST(BufferViewTests, ViewWritesAreVisibleThroughSourceBuffer) {
  HostBuffer<float> buf(8);
  HostBufferView<float> view(buf, 2, 4);
  view[0] = 42.0f;

  EXPECT_EQ(buf[2], 42.0f);
}

TEST(BufferViewTests, ViewsAreCopyable) {
  HostBuffer<float> buf(16);
  HostBufferView<float> view(buf, 4, 8);
  HostBufferView<float> copy(view);

  EXPECT_EQ(copy.data(), view.data());
  EXPECT_EQ(copy.num_elements(), view.num_elements());
}

TEST(BufferViewTests, ViewOfViewNarrowsFurther) {
  HostBuffer<float> buf(16);
  HostBufferView<float> view(buf, 4, 8);
  HostBufferView<float> inner(view, 2, 3);

  EXPECT_EQ(inner.data(), buf.data() + 6);
  EXPECT_EQ(inner.num_elements(), std::size_t{3});
}

TEST(BufferViewTests, OutOfRangeViewIsRejected) {
  HostPolicy policy;
  CountedHostBuffer<float> buf(16, policy);
  CountedHostView<float> view(buf, 10, 10);

  EXPECT_EQ(view.alloc_policy().count(), std::size_t{1});
  EXPECT_EQ(view.num_elements(), std::size_t{0});
  EXPECT_EQ(view.data(), nullptr);
}

TEST(BufferViewTests, ViewBoundsCheckDoesNotOverflow) {
  // offset + count wraps past zero; the check must still reject it.
  HostPolicy policy;
  CountedHostBuffer<float> buf(16, policy);
  CountedHostView<float> view(buf, 8, std::numeric_limits<std::size_t>::max() - 4);

  EXPECT_EQ(view.alloc_policy().count(), std::size_t{1});
  EXPECT_EQ(view.num_elements(), std::size_t{0});
}

TEST(BufferViewTests, OffsetPastEndDoesNotUnderflow) {
  HostPolicy policy;
  CountedHostBuffer<float> buf(16, policy);
  CountedHostView<float> view(buf, 99);

  EXPECT_EQ(view.alloc_policy().count(), std::size_t{1});
  EXPECT_EQ(view.num_elements(), std::size_t{0});
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Device, pinned and unified buffers
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// NOTE: copy() and memset() return gpuError_t, where gpuSuccess is 0.
// EXPECT_TRUE on one of those is inverted -- it passes only when the call
// FAILED. Always compare against gpuSuccess explicitly.

TEST(DeviceBufferTests, PoolAllocationZeroInitialises) {
  auto dev_h = std::make_shared<DeviceHandle>(0);
  DeviceBuffer<float> dev(64, dev_h);
  ASSERT_NE(dev.data(), nullptr);
  EXPECT_EQ(dev.num_elements(), std::size_t{64});

  // The buffer zero-inits on the handle's stream, so read it back on that same
  // stream rather than stream 0, which would race the async memset.
  HostBuffer<float> host(64);
  const gpuStream_t stream = dev_h->alloc_stream().get();
  ASSERT_EQ(ext::copy(host, dev, stream), gpuSuccess);
  ASSERT_EQ(gpuStreamSynchronize(stream), gpuSuccess);
  for (std::size_t i = 0; i < host.num_elements(); ++i) {
    EXPECT_EQ(host[i], 0.0f) << "at index " << i;
  }
}

TEST(DeviceBufferTests, MoveTransfersDeviceOwnership) {
  auto dev_h = std::make_shared<DeviceHandle>(0);
  DeviceBuffer<float> src(64, dev_h);
  float *const raw = src.data();

  DeviceBuffer<float> dst(std::move(src));
  EXPECT_EQ(dst.data(), raw);
  EXPECT_EQ(src.data(), nullptr);
}

TEST(DeviceBufferTests, MoveAssignmentReleasesDeviceOwnership) {
  // Move-assignment onto a non-empty buffer takes the destroy_() -> gpuFreeAsync
  // path the move constructor never reaches: dst's original block must be
  // returned to the pool on its stream before dst takes src's block. The move
  // constructor has no prior allocation to free, so this is the only test that
  // frees a live device block on a move.
  auto dev_h = std::make_shared<DeviceHandle>(0);
  GpuStream &stream = dev_h->alloc_stream();
  DeviceBuffer<float> src(64, dev_h);
  DeviceBuffer<float> dst(32, dev_h);
  float *const raw = src.data();

  dst = std::move(src);
  EXPECT_EQ(dst.data(), raw);
  EXPECT_EQ(dst.num_elements(), std::size_t{64});
  EXPECT_EQ(src.data(), nullptr);
  EXPECT_EQ(src.num_elements(), std::size_t{0});

  // The async free of dst's old block drains cleanly - no double free, no
  // error left on the stream.
  EXPECT_EQ(gpuStreamSynchronize(stream.get()), gpuSuccess);
  EXPECT_EQ(gpuGetLastError(), gpuSuccess);
}

TEST(DeviceBufferTests, RoundTripsThroughDeviceMemory) {
  HostBuffer<float> up(32);
  HostBuffer<float> down(32);
  for (std::size_t i = 0; i < 32; ++i)
    up[i] = static_cast<float>(i) * 2.0f;

  auto dev_h = std::make_shared<DeviceHandle>(0);
  DeviceBuffer<float> dev(32, dev_h);
  const gpuStream_t stream = dev_h->alloc_stream().get();
  ASSERT_EQ(ext::copy(dev, up, stream), gpuSuccess);
  ASSERT_EQ(ext::copy(down, dev, stream), gpuSuccess);
  ASSERT_EQ(gpuStreamSynchronize(stream), gpuSuccess);

  for (std::size_t i = 0; i < 32; ++i) {
    EXPECT_EQ(down[i], up[i]) << "at index " << i;
  }
}

TEST(DeviceBufferTests, StreamOrderedAllocationsSurviveRepeatedChurn) {
  // Regression test for the stream-ordered allocate/free pairing.
  // gpuFree performs no implicit synchronisation for a pointer from
  // gpuMallocAsync, so releasing with it handed blocks back to the
  // pool while queued work was still using them - and the pool then
  // reissued them to the next allocation. Freeing on the same stream
  // keeps each block alive until its work has drained.
  auto dev_h = std::make_shared<DeviceHandle>(0);
  GpuStream &stream = dev_h->alloc_stream();
  for (int iter = 0; iter < 64; ++iter) {
    DeviceBuffer<float> buf(4096, dev_h);
    ASSERT_NE(buf.data(), nullptr) << "at iteration " << iter;
    EXPECT_EQ(buf.num_elements(), std::size_t{4096}) << "at iteration " << iter;
    EXPECT_EQ(ext::memset(buf, 0xAB, stream.get()), gpuSuccess) << "at iteration " << iter;
  }
  EXPECT_EQ(gpuStreamSynchronize(stream.get()), gpuSuccess);
  EXPECT_EQ(gpuGetLastError(), gpuSuccess);
}

TEST(DeviceBufferTests, StreamOrderedBufferHoldsItsContents) {
  auto dev_h = std::make_shared<DeviceHandle>(0);
  GpuStream &stream = dev_h->alloc_stream();
  HostBuffer<float> host(128);
  {
    DeviceBuffer<float> dev(128, dev_h);
    ASSERT_EQ(ext::memset(dev, 0, stream.get()), gpuSuccess);
    ASSERT_EQ(ext::copy(host, dev, stream.get()), gpuSuccess);
    ASSERT_EQ(gpuStreamSynchronize(stream.get()), gpuSuccess);
  }
  for (std::size_t i = 0; i < host.num_elements(); ++i) {
    EXPECT_EQ(host[i], 0.0f) << "at index " << i;
  }
  EXPECT_EQ(gpuStreamSynchronize(stream.get()), gpuSuccess);
}

TEST(DeviceBufferTests, MovedStreamOrderedBufferFreesOnce) {
  auto dev_h = std::make_shared<DeviceHandle>(0);
  GpuStream &stream = dev_h->alloc_stream();
  {
    DeviceBuffer<float> src(1024, dev_h);
    DeviceBuffer<float> dst(std::move(src));
    EXPECT_EQ(src.data(), nullptr);
    EXPECT_NE(dst.data(), nullptr);
  }
  EXPECT_EQ(gpuStreamSynchronize(stream.get()), gpuSuccess);
  EXPECT_EQ(gpuGetLastError(), gpuSuccess);
}

TEST(DeviceHandleTests, ReportsIndexAndQueriesProperties) {
  DeviceHandle dev(0);
  EXPECT_EQ(dev.index(), 0);
  // props() holds the queried cudaDeviceProp / hipDeviceProp_t; a real device
  // names itself, which also exercises the field across both backends.
  EXPECT_NE(dev.props().name[0], '\0');
  // The stream created eagerly on that device is usable.
  EXPECT_EQ(dev.alloc_stream().sync(), gpuSuccess);
  // The memory pool created eagerly on that device is a live handle.
  EXPECT_NE(dev.mem_pool().get(), nullptr);
}

TEST(DeviceBufferTests, HandleAllocationHoldsItsContents) {
  auto dev = std::make_shared<DeviceHandle>(0);
  HostBuffer<float> host(128);
  {
    DeviceBuffer<float> buf(128, dev);
    ASSERT_NE(buf.data(), nullptr);
    EXPECT_EQ(buf.num_elements(), std::size_t{128});
    ASSERT_EQ(ext::memset(buf, 0, dev->alloc_stream().get()), gpuSuccess);
    ASSERT_EQ(ext::copy(host, buf, dev->alloc_stream().get()), gpuSuccess);
    ASSERT_EQ(dev->alloc_stream().sync(), gpuSuccess);
  }
  for (std::size_t i = 0; i < host.num_elements(); ++i) {
    EXPECT_EQ(host[i], 0.0f) << "at index " << i;
  }
  EXPECT_EQ(gpuGetLastError(), gpuSuccess);
}

TEST(DeviceBufferTests, HandleBufferRetainsStreamAfterLocalHandleReset) {
  // The buffer keeps a shared_ptr to the handle, so dropping the local
  // reference must not destroy the stream the destructor frees on.
  auto dev = std::make_shared<DeviceHandle>(0);
  const gpuStream_t stream = dev->alloc_stream().get();
  DeviceBuffer<float> buf(256, dev);
  dev.reset(); // the buffer's retained handle is now the sole owner
  ASSERT_NE(buf.data(), nullptr);
  EXPECT_EQ(gpuStreamSynchronize(stream), gpuSuccess); // stream still alive
  // ~buf frees on that same stream, then releases the handle - no use-after-free.
}

// Allocate+zero-init and the ownership-transferring move are covered for the
// pinned and unified kinds by HostAccessibleOwningBufferTest above. What stays
// here is what those cannot express: the flags-taking constructors, and that
// unified memory is genuinely host-writable.

TEST(HostAccessibleBufferTests, PinnedBufferWithFlags) {
  PinnedBuffer<float> buf(64, gpuHostAllocMapped);
  EXPECT_NE(buf.data(), nullptr);
  EXPECT_EQ(buf.num_elements(), std::size_t{64});
}

TEST(HostAccessibleBufferTests, UnifiedBufferIsHostAccessible) {
  UnifiedBuffer<float> buf(64);
  ASSERT_NE(buf.data(), nullptr);
  for (std::size_t i = 0; i < 64; ++i)
    buf[i] = static_cast<float>(i);
  EXPECT_EQ(buf[10], 10.0f);
}

TEST(HostAccessibleBufferTests, UnifiedBufferWithFlags) {
  UnifiedBuffer<float> buf(64, gpuMemAttachHost);
  EXPECT_NE(buf.data(), nullptr);
  EXPECT_EQ(buf.num_elements(), std::size_t{64});
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// copy() and memset()
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TEST(CopyAndMemsetTests, SynchronousHostCopy) {
  HostBuffer<float> src(16);
  HostBuffer<float> dst(16);
  for (std::size_t i = 0; i < 16; ++i)
    src[i] = static_cast<float>(i);

  ASSERT_EQ(ext::copy(dst, src), stdHostMemSuccess);
  for (std::size_t i = 0; i < 16; ++i) {
    EXPECT_EQ(dst[i], src[i]) << "at index " << i;
  }
}

TEST(CopyAndMemsetTests, SynchronousHostCopyRejectsUndersizedDestination) {
  HostBuffer<float> src(16);
  HostBuffer<float> dst(4);
  EXPECT_EQ(ext::copy(dst, src), stdHostMemInvalidValue);
}

TEST(CopyAndMemsetTests, EmptyBufferCopyIsANoOp) {
  HostBuffer<float> src;
  HostBuffer<float> dst;
  EXPECT_EQ(ext::copy(dst, src), stdHostMemSuccess);
}

TEST(CopyAndMemsetTests, OffsetCopyMovesOnlyTheRequestedRange) {
  HostBuffer<float> host(16);
  for (std::size_t i = 0; i < 16; ++i)
    host[i] = static_cast<float>(i) + 1.0f;

  auto dev_h = std::make_shared<DeviceHandle>(0);
  DeviceBuffer<float> dev(16, dev_h);
  HostBuffer<float> out(16);

  const gpuStream_t stream = dev_h->alloc_stream().get();
  ASSERT_EQ(ext::copy(dev, 0, host, 4, 8, stream), gpuSuccess);
  ASSERT_EQ(ext::copy(out, 0, dev, 0, 8, stream), gpuSuccess);
  ASSERT_EQ(gpuStreamSynchronize(stream), gpuSuccess);

  for (std::size_t i = 0; i < 8; ++i) {
    EXPECT_EQ(out[i], static_cast<float>(i + 4) + 1.0f) << "at index " << i;
  }
}

TEST(CopyAndMemsetTests, OffsetCopyRejectsOutOfBoundsRange) {
  auto dev_h = std::make_shared<DeviceHandle>(0);
  HostBuffer<float> host(16);
  DeviceBuffer<float> dev(16, dev_h);
  EXPECT_EQ(ext::copy(dev, 0, host, 12, 8, gpuStream_t{0}), gpuErrorInvalidValue);
}

TEST(CopyAndMemsetTests, HostMemsetFillsBuffer) {
  HostBuffer<std::byte> buf(16);
  ASSERT_EQ(ext::memset(buf, 0x5A), stdHostMemSuccess);
  for (std::size_t i = 0; i < 16; ++i) {
    EXPECT_EQ(static_cast<int>(buf[i]), 0x5A) << "at index " << i;
  }
}

TEST(CopyAndMemsetTests, EmptyBufferMemsetIsANoOp) {
  HostBuffer<std::byte> buf;
  EXPECT_EQ(ext::memset(buf, 0x5A), stdHostMemSuccess);
}

TEST(CopyAndMemsetTests, OffsetMemsetRejectsOutOfBoundsRange) {
  auto dev_h = std::make_shared<DeviceHandle>(0);
  DeviceBuffer<float> dev(16, dev_h);
  EXPECT_EQ(ext::memset(dev, 12, 8, 0, gpuStream_t{0}), gpuErrorInvalidValue);
}

TEST(CopyAndMemsetTests, OffsetMemsetFillsOnlyTheRequestedRange) {
  // The success path of the offset+count device memset: only [offset,
  // offset+count) is written, and the bytes outside it keep the pool draw's
  // zero-init. Previously only the rejection path was exercised, so a memset
  // that ignored its offset and filled from the start would have passed.
  // A std::byte buffer makes the fill byte observable directly; 0xAB in a
  // float would read back as a NaN bit pattern.
  auto dev_h = std::make_shared<DeviceHandle>(0);
  const gpuStream_t stream = dev_h->alloc_stream().get();
  DeviceBuffer<std::byte> dev(16, dev_h); // zero-initialised by the pool draw

  ASSERT_EQ(ext::memset(dev, 4, 8, 0xAB, stream), gpuSuccess);

  HostBuffer<std::byte> host(16);
  ASSERT_EQ(ext::copy(host, dev, stream), gpuSuccess);
  ASSERT_EQ(gpuStreamSynchronize(stream), gpuSuccess);

  for (std::size_t i = 0; i < 16; ++i) {
    const int expected = (i >= 4 && i < 12) ? 0xAB : 0x00;
    EXPECT_EQ(static_cast<int>(host[i]), expected) << "at index " << i;
  }
}

TEST(CopyAndMemsetTests, VoidBufferCopiesByBytes) {
  HostBuffer<void> src(32);
  HostBuffer<void> dst(32);
  EXPECT_EQ(src.size_bytes(), std::size_t{32});
  EXPECT_EQ(ext::copy(dst, src), stdHostMemSuccess);
  EXPECT_EQ(ext::copy(dst, 8, src, 8, 16, gpuStream_t{0}), gpuSuccess);
  EXPECT_EQ(gpuStreamSynchronize(gpuStream_t{0}), gpuSuccess);
}

} // namespace gpumod::extension::test
