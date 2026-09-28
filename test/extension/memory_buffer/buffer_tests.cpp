// buffer_tests.cpp - Tests for wwr.extension.memory_buffer
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
import wwr.runtime_api;
import wwr.extension.common;
import wwr.extension.handle;
import wwr.extension.runtime;
import wwr.extension.memory_buffer;
import wwr.test.shared.device_handle;

#include "counting_policy.h"

namespace wwr::extension::test {
// Bind abort-on-failure once per error type, for this file's instantiations.
using Abort = AbortPolicy<wwrError_t>;
using HostAbort = AbortPolicy<stdHostMemoryError_t>;

namespace ext = wwr::extension;

// CountingPolicy and the Counted* buffer aliases used across the failure-path
// suites live in counting_policy.h, shared with allocation_failure_tests.cpp.

// The compile-time contract of the buffer types (move-only, nothrow moves,
// views copyable, element_size) is checked by static_assert in
// test/extension/build_time/memory_buffer.cppm, which builds without this
// executable.

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Owning host-accessible buffers: the shared RAII contract, once per kind
//
// HostBufferWrapper, PinnedBufferWrapper and UnifiedBufferWrapper are all owning, host-accessible
// (operator[]), zero-initialising, and built from a bare element count. Their
// allocate/zero-init and the move that transfers ownership are one shared
// BaseBuffer implementation, so a typed suite proves that contract once per
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
    ::testing::Types<HostBufferWrapper<float, HostAbort, HostAbort>, PinnedBufferWrapper<float, Abort, Abort>, UnifiedBufferWrapper<float, Abort, Abort>>;
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
// HostBufferWrapper
//
// Zero-init, the move constructor and move-assignment are covered for every
// owning host-accessible kind by HostAccessibleOwningBufferTest above; what
// stays here is host-specific -- the empty/zero-element invariants, the
// error-policy path, and the move edge cases that exist only once.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TEST(HostBufferTests, DefaultConstructedIsEmpty) {
  HostBufferWrapper<float, HostAbort, HostAbort> buf;
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
  HostBufferWrapper<float, HostAbort, HostAbort> buf(8);
  float *const raw = buf.data();

  buf = std::move(buf);
  EXPECT_EQ(buf.data(), raw);
  EXPECT_EQ(buf.num_elements(), std::size_t{8});
}

TEST(HostBufferTests, MoveAssignmentOntoEmptyBuffer) {
  HostBufferWrapper<float, HostAbort, HostAbort> dst;
  HostBufferWrapper<float, HostAbort, HostAbort> src(4);
  float *const raw = src.data();

  dst = std::move(src);
  EXPECT_EQ(dst.data(), raw);
  EXPECT_EQ(dst.num_elements(), std::size_t{4});
}

TEST(HostBufferTests, SubscriptReadsAndWrites) {
  HostBufferWrapper<float, HostAbort, HostAbort> buf(4);
  for (std::size_t i = 0; i < 4; ++i)
    buf[i] = static_cast<float>(i) + 1.0f;
  EXPECT_EQ(buf[0], 1.0f);
  EXPECT_EQ(buf[3], 4.0f);
}

TEST(HostBufferTests, ConstAndPointerAccessorsMatchData) {
  // Three public accessors that no other test exercised (every other reaches
  // storage through data() and the non-const operator[]): operator T*(),
  // operator const T*(), and the const operator[].
  HostBufferWrapper<float, HostAbort, HostAbort> buf(8);
  buf[0] = 3.5f;
  float *p = buf; // operator T*()
  EXPECT_EQ(p, buf.data());

  const HostBufferWrapper<float, HostAbort, HostAbort> &cbuf = buf;
  const float *cp = cbuf; // operator const T*()
  EXPECT_EQ(cp, cbuf.data());
  EXPECT_EQ(cbuf[0], 3.5f); // const operator[]
}

// ── The separate allocation/deallocation policy axis ────────────────
//
// The four-argument (num, P_alloc, P_free) constructor and the whole P_free
// template parameter were untested: every prior test used one policy for both
// slots, and free_policy() was never called. A policy carrying a public tag lets
// us check which instance landed in which slot without forcing a failure.
namespace {
struct TaggedHostPolicy {
  using error_type = stdHostMemoryError_t;
  int tag = 0;
  TaggedHostPolicy() = default;
  explicit TaggedHostPolicy(int t) : tag(t) {}
  void handle_error(stdHostMemoryError_t, std::source_location) noexcept {}
};
} // namespace

TEST(HostBufferTests, SeparateAllocAndFreePoliciesReachTheirOwnSlots) {
  HostBufferWrapper<float, TaggedHostPolicy, TaggedHostPolicy> buf(8, TaggedHostPolicy{1}, TaggedHostPolicy{2});
  EXPECT_EQ(buf.alloc_policy().tag, 1);
  EXPECT_EQ(buf.free_policy().tag, 2); // free_policy() accessor + distinct P_free instance
}

// ── The base destructor's leaked-allocation safety net ──────────────
//
// ~BaseBuffer reports (via policy_free_) when an owning buffer still has a live
// allocation at base-destruction time -- i.e. the derived class forgot to call
// destroy_(). No other test reaches that branch, because every real buffer kind
// releases correctly. A deliberately-broken owning buffer whose destructor omits
// destroy_() is the only way in. The policy records to statics because the base
// destructor fires after any instance policy would already be gone.
namespace {
struct StaticHostPolicy {
  using error_type = stdHostMemoryError_t;
  static inline int fires = 0;
  static inline stdHostMemoryError_t last{};
  static void reset() {
    fires = 0;
    last = stdHostMemoryError_t{};
  }
  void handle_error(stdHostMemoryError_t error, std::source_location) noexcept {
    ++fires;
    last = error;
  }
};

// Broken on purpose: allocates like a real host buffer but its (implicit)
// destructor never calls destroy_(), so the allocation is still live when
// ~BaseBuffer runs. The block is freed by hand after the object dies, so no
// actual leak remains and the case stays sanitizer-clean.
template<typename T, typename P>
class LeakyBuffer : public BaseBuffer<T, MemoryKind::Host, LeakyBuffer<T, P>, P, P> {
public:
  using Base = BaseBuffer<T, MemoryKind::Host, LeakyBuffer<T, P>, P, P>;
  using Base::Base;
  static void allocate(T **ptr, std::size_t n, P &, std::source_location) {
    *ptr = static_cast<T *>(std::malloc(n * sizeof(T)));
  }
  void deallocate(T *ptr, std::size_t) { std::free(ptr); } // contract-required; unused here
};
} // namespace

TEST(HostBufferTests, LeakedAllocationIsReportedByBaseDestructor) {
  StaticHostPolicy::reset();
  void *raw = nullptr;
  {
    LeakyBuffer<float, StaticHostPolicy> buf(8);
    raw = buf.data();
    ASSERT_NE(raw, nullptr);
    // buf destructs here WITHOUT calling destroy_(); ~BaseBuffer sees data_ set
    // and must report through policy_free_.
  }
  EXPECT_EQ(StaticHostPolicy::fires, 1) << "base destructor did not report the leak";
  EXPECT_EQ(StaticHostPolicy::last, stdHostMemInvalidValue);
  std::free(raw); // release the intentionally-leaked block -- no real leak remains
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
  EXPECT_EQ(buf.alloc_policy().last(), wwrErrorInvalidValue);
  EXPECT_EQ(buf.data(), nullptr);
  EXPECT_EQ(buf.num_elements(), std::size_t{0});
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Buffer views
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TEST(BufferViewTests, FullViewAliasesSourceBuffer) {
  HostBufferWrapper<float, HostAbort, HostAbort> buf(16);
  BufferViewWrapper<float, MemoryKind::Host, HostAbort, HostAbort> view(buf);

  EXPECT_EQ(view.data(), buf.data());
  EXPECT_EQ(view.num_elements(), std::size_t{16});
}

TEST(BufferViewTests, SubViewOffsetsIntoSourceBuffer) {
  HostBufferWrapper<float, HostAbort, HostAbort> buf(16);
  for (std::size_t i = 0; i < 16; ++i)
    buf[i] = static_cast<float>(i);

  BufferViewWrapper<float, MemoryKind::Host, HostAbort, HostAbort> view(buf, 4, 8);
  ASSERT_EQ(view.data(), buf.data() + 4);
  EXPECT_EQ(view.num_elements(), std::size_t{8});
  EXPECT_EQ(view[0], 4.0f);
  EXPECT_EQ(view[7], 11.0f);
}

TEST(BufferViewTests, OffsetOnlyViewCoversRemainingElements) {
  HostBufferWrapper<float, HostAbort, HostAbort> buf(16);
  BufferViewWrapper<float, MemoryKind::Host, HostAbort, HostAbort> view(buf, 12);

  EXPECT_EQ(view.data(), buf.data() + 12);
  EXPECT_EQ(view.num_elements(), std::size_t{4});
}

TEST(BufferViewTests, ViewWritesAreVisibleThroughSourceBuffer) {
  HostBufferWrapper<float, HostAbort, HostAbort> buf(8);
  BufferViewWrapper<float, MemoryKind::Host, HostAbort, HostAbort> view(buf, 2, 4);
  view[0] = 42.0f;

  EXPECT_EQ(buf[2], 42.0f);
}

TEST(BufferViewTests, ViewsAreCopyable) {
  HostBufferWrapper<float, HostAbort, HostAbort> buf(16);
  BufferViewWrapper<float, MemoryKind::Host, HostAbort, HostAbort> view(buf, 4, 8);
  BufferViewWrapper<float, MemoryKind::Host, HostAbort, HostAbort> copy(view);

  EXPECT_EQ(copy.data(), view.data());
  EXPECT_EQ(copy.num_elements(), view.num_elements());
}

TEST(BufferViewTests, ViewOfViewNarrowsFurther) {
  HostBufferWrapper<float, HostAbort, HostAbort> buf(16);
  BufferViewWrapper<float, MemoryKind::Host, HostAbort, HostAbort> view(buf, 4, 8);
  BufferViewWrapper<float, MemoryKind::Host, HostAbort, HostAbort> inner(view, 2, 3);

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

// ── Reinterpreting (cross-type) views ──────────────────────────────

TEST(BufferViewTests, ReinterpretViewPreservesByteSpan) {
  HostBufferWrapper<float, HostAbort, HostAbort> buf(16); // 64 bytes
  // For a default-policy source the factory's return type is exactly the
  // per-kind alias, so it can be named instead of deduced with auto.
  BufferViewWrapper<std::byte, MemoryKind::Host, HostAbort, HostAbort> bytes = reinterpret_buffer_view<std::byte>(buf);

  EXPECT_EQ(static_cast<void *>(bytes.data()), static_cast<void *>(buf.data()));
  EXPECT_EQ(bytes.num_elements(), std::size_t{64});
  EXPECT_EQ(bytes.size_bytes(), buf.size_bytes());
}

TEST(BufferViewTests, ReinterpretViewWritesAreVisibleThroughSource) {
  HostBufferWrapper<std::uint16_t, HostAbort, HostAbort> buf(4);
  buf[0] = 0;
  BufferViewWrapper<std::byte, MemoryKind::Host, HostAbort, HostAbort> bytes = reinterpret_buffer_view<std::byte>(buf); // 8 bytes

  ASSERT_EQ(bytes.num_elements(), std::size_t{8});
  bytes[0] = std::byte{0xFF};
  bytes[1] = std::byte{0xFF};
  EXPECT_EQ(buf[0], std::uint16_t{0xFFFF}); // endianness-independent
}

TEST(BufferViewTests, ReinterpretViewRejectsPartialTrailingElement) {
  HostPolicy policy;
  CountedHostBuffer<std::byte> buf(6, policy); // 6 bytes, not a multiple of 4
  auto ints = reinterpret_buffer_view<std::uint32_t>(buf);

  EXPECT_EQ(ints.alloc_policy().count(), std::size_t{1});
  EXPECT_EQ(ints.num_elements(), std::size_t{0});
  EXPECT_EQ(ints.data(), nullptr);
}

TEST(BufferViewTests, ReinterpretViewRejectsMisalignedBase) {
  HostPolicy policy;
  CountedHostBuffer<std::byte> buf(16, policy);
  // A sub-view one byte into an (over-)aligned allocation cannot be aligned for
  // a 4-byte type -- this is the reachable misalignment case.
  BufferViewWrapper<std::byte, MemoryKind::Host, HostPolicy, HostPolicy> shifted(buf, 1, 8);
  ASSERT_EQ(shifted.num_elements(), std::size_t{8});

  auto ints = reinterpret_buffer_view<std::uint32_t>(shifted);
  EXPECT_EQ(ints.alloc_policy().count(), std::size_t{1});
  EXPECT_EQ(ints.num_elements(), std::size_t{0});
}

TEST(BufferViewTests, ReinterpretViewComposesWithSubView) {
  HostBufferWrapper<float, HostAbort, HostAbort> buf(16); // 64 bytes
  // A reinterpreting view is itself a buffer, so the ordinary offset+count
  // sub-view constructor narrows it -- in the TARGET type's units (bytes here).
  BufferViewWrapper<std::byte, MemoryKind::Host, HostAbort, HostAbort> bytes = reinterpret_buffer_view<std::byte>(buf);
  BufferViewWrapper<std::byte, MemoryKind::Host, HostAbort, HostAbort> mid(bytes, 8, 16); // bytes [8, 24)

  EXPECT_EQ(static_cast<void *>(mid.data()),
            static_cast<void *>(reinterpret_cast<std::byte *>(buf.data()) + 8));
  EXPECT_EQ(mid.num_elements(), std::size_t{16});
}

TEST(BufferViewTests, EmptyBufferReinterpretsToEmptyView) {
  HostBufferWrapper<float, HostAbort, HostAbort> buf; // null, 0 elements
  BufferViewWrapper<std::byte, MemoryKind::Host, HostAbort, HostAbort> bytes = reinterpret_buffer_view<std::byte>(buf);

  EXPECT_EQ(bytes.data(), nullptr);
  EXPECT_EQ(bytes.num_elements(), std::size_t{0});
}

TEST(BufferViewTests, ReinterpretViewUpcastsToWiderElementType) {
  // The cross-type tests above all narrow (float/uint16 -> byte) or reject; the
  // widening success path was never taken. A byte buffer whose base is suitably
  // aligned (std::malloc is aligned for any type) and whose size is a whole
  // multiple of 4 reinterprets to a uint32 view -- num_elements() shrinks from
  // bytes to elements, and a write through the wide view is visible byte-for-byte
  // through the source.
  HostBufferWrapper<std::byte, HostAbort, HostAbort> buf(16); // 16 bytes, zero-initialised
  BufferViewWrapper<std::uint32_t, MemoryKind::Host, HostAbort, HostAbort> words = reinterpret_buffer_view<std::uint32_t>(buf);

  ASSERT_EQ(static_cast<void *>(words.data()), static_cast<void *>(buf.data()));
  ASSERT_EQ(words.num_elements(), std::size_t{4}); // 16 / 4

  words[0] = 0xFFFFFFFFu; // all-ones is endianness-independent
  EXPECT_EQ(buf[0], std::byte{0xFF});
  EXPECT_EQ(buf[1], std::byte{0xFF});
  EXPECT_EQ(buf[2], std::byte{0xFF});
  EXPECT_EQ(buf[3], std::byte{0xFF});
  EXPECT_EQ(buf[4], std::byte{0x00}); // the next word is untouched
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Device, pinned and unified buffers
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// NOTE: copy() and memset() return wwrError_t, where wwrSuccess is 0.
// EXPECT_TRUE on one of those is inverted -- it passes only when the call
// FAILED. Always compare against wwrSuccess explicitly.

// Downstream handles below DeviceHandle's tier, backing the other two rungs of
// the ladder at runtime (DeviceHandle itself covers the pool tier throughout).

// Stream tier: a real owned stream + dev_idx, no pool -> wwrMallocAsync path.
struct StreamTierHandle {
  explicit StreamTierHandle(int dev) : dev_(dev), stream_(dev) {}
  int dev_idx() const noexcept { return dev_; }
  wwrStream_t stream() const noexcept { return stream_.get(); }
  int dev_;
  GpuStreamWrapper<Abort, Abort> stream_;
};
static_assert(device_handle_stream<StreamTierHandle>);
static_assert(!device_handle_pool<StreamTierHandle>);

// Bare tier: dev_idx only -> synchronous wwrMalloc / wwrFree path.
struct SyncTierHandle {
  int dev_idx() const noexcept { return 0; }
};
static_assert(device_handle<SyncTierHandle>);
static_assert(!device_handle_stream<SyncTierHandle>);

TEST(DeviceBufferTests, PoolAllocationZeroInitialises) {
  auto dev_h = std::make_shared<DeviceHandle>(0);
  DeviceBufferWrapper<float, Abort, Abort, DeviceHandle> dev(64, dev_h);
  ASSERT_NE(dev.data(), nullptr);
  EXPECT_EQ(dev.num_elements(), std::size_t{64});

  // The buffer zero-inits on the handle's stream, so read it back on that same
  // stream rather than stream 0, which would race the async memset.
  HostBufferWrapper<float, HostAbort, HostAbort> host(64);
  const wwrStream_t stream = dev_h->stream().get();
  ASSERT_EQ(ext::copy(host, dev, stream), wwrSuccess);
  ASSERT_EQ(wwrStreamSynchronize(stream), wwrSuccess);
  for (std::size_t i = 0; i < host.num_elements(); ++i) {
    EXPECT_EQ(host[i], 0.0f) << "at index " << i;
  }
}

TEST(DeviceBufferTests, StreamTierRoundTripsWithoutPool) {
  // A stream-only handle drives the wwrMallocAsync/wwrFreeAsync branch: same
  // stream-ordered contract as the pool tier, just from the device default pool.
  auto h = std::make_shared<StreamTierHandle>(0);
  DeviceBufferWrapper<float, Abort, Abort, StreamTierHandle> dev(32, h);
  ASSERT_NE(dev.data(), nullptr);
  EXPECT_EQ(dev.num_elements(), std::size_t{32});

  HostBufferWrapper<float, HostAbort, HostAbort> up(32), down(32);
  for (std::size_t i = 0; i < 32; ++i)
    up[i] = static_cast<float>(i) * 2.0f;

  const wwrStream_t stream = h->stream();
  ASSERT_EQ(ext::copy(dev, up, stream), wwrSuccess);
  ASSERT_EQ(ext::copy(down, dev, stream), wwrSuccess);
  ASSERT_EQ(wwrStreamSynchronize(stream), wwrSuccess);
  for (std::size_t i = 0; i < 32; ++i)
    EXPECT_EQ(down[i], up[i]) << "at index " << i;
}

TEST(DeviceBufferTests, SyncTierAllocatesZeroInitialised) {
  // A dev_idx-only handle drives the synchronous wwrMalloc/wwrMemset/wwrFree
  // branch. The zero-init is synchronous, so it is already visible on readback;
  // the block frees via wwrFree in the destructor at scope exit.
  auto h = std::make_shared<SyncTierHandle>();
  DeviceBufferWrapper<float, Abort, Abort, SyncTierHandle> dev(48, h);
  ASSERT_NE(dev.data(), nullptr);
  EXPECT_EQ(dev.num_elements(), std::size_t{48});

  // The sync-tier block is a plain allocation, readable on any stream.
  GpuStreamWrapper<Abort, Abort> stream(0);
  HostBufferWrapper<float, HostAbort, HostAbort> host(48);
  ASSERT_EQ(ext::copy(host, dev, stream.get()), wwrSuccess);
  ASSERT_EQ(wwrStreamSynchronize(stream.get()), wwrSuccess);
  for (std::size_t i = 0; i < host.num_elements(); ++i)
    EXPECT_EQ(host[i], 0.0f) << "at index " << i;
}

TEST(DeviceBufferTests, ReinterpretViewAliasesDeviceStorage) {
  auto dev_h = std::make_shared<DeviceHandle>(0);
  const wwrStream_t stream = dev_h->stream().get();

  HostBufferWrapper<float, HostAbort, HostAbort> up(16);
  for (std::size_t i = 0; i < up.num_elements(); ++i)
    up[i] = static_cast<float>(i) * 3.0f;

  DeviceBufferWrapper<float, Abort, Abort, DeviceHandle> dev(16, dev_h);
  ASSERT_EQ(ext::copy(dev, up, stream), wwrSuccess);

  // A device pointer passes the alignment check (it never gets dereferenced on
  // the host); read the block back through a byte view to prove it aliases.
  BufferViewWrapper<std::byte, MemoryKind::Device, Abort, Abort> dev_bytes = reinterpret_buffer_view<std::byte>(dev);
  ASSERT_EQ(static_cast<void *>(dev_bytes.data()), static_cast<void *>(dev.data()));
  ASSERT_EQ(dev_bytes.num_elements(), dev.size_bytes());

  HostBufferWrapper<std::byte, HostAbort, HostAbort> host_bytes(dev_bytes.num_elements());
  ASSERT_EQ(ext::copy(host_bytes, dev_bytes, stream), wwrSuccess);
  ASSERT_EQ(wwrStreamSynchronize(stream), wwrSuccess);

  EXPECT_EQ(std::memcmp(host_bytes.data(), up.data(), up.size_bytes()), 0);
}

TEST(DeviceBufferTests, MoveTransfersDeviceOwnership) {
  auto dev_h = std::make_shared<DeviceHandle>(0);
  DeviceBufferWrapper<float, Abort, Abort, DeviceHandle> src(64, dev_h);
  float *const raw = src.data();

  DeviceBufferWrapper<float, Abort, Abort, DeviceHandle> dst(std::move(src));
  EXPECT_EQ(dst.data(), raw);
  EXPECT_EQ(src.data(), nullptr);
}

TEST(DeviceBufferTests, MoveAssignmentReleasesDeviceOwnership) {
  // Move-assignment onto a non-empty buffer takes the destroy_() -> wwrFreeAsync
  // path the move constructor never reaches: dst's original block must be
  // returned to the pool on its stream before dst takes src's block. The move
  // constructor has no prior allocation to free, so this is the only test that
  // frees a live device block on a move.
  auto dev_h = std::make_shared<DeviceHandle>(0);
  GpuStreamWrapper<Abort, Abort> &stream = dev_h->stream();
  DeviceBufferWrapper<float, Abort, Abort, DeviceHandle> src(64, dev_h);
  DeviceBufferWrapper<float, Abort, Abort, DeviceHandle> dst(32, dev_h);
  float *const raw = src.data();

  dst = std::move(src);
  EXPECT_EQ(dst.data(), raw);
  EXPECT_EQ(dst.num_elements(), std::size_t{64});
  EXPECT_EQ(src.data(), nullptr);
  EXPECT_EQ(src.num_elements(), std::size_t{0});

  // The async free of dst's old block drains cleanly - no double free, no
  // error left on the stream.
  EXPECT_EQ(wwrStreamSynchronize(stream.get()), wwrSuccess);
  EXPECT_EQ(wwrGetLastError(), wwrSuccess);
}

TEST(DeviceBufferTests, RoundTripsThroughDeviceMemory) {
  HostBufferWrapper<float, HostAbort, HostAbort> up(32);
  HostBufferWrapper<float, HostAbort, HostAbort> down(32);
  for (std::size_t i = 0; i < 32; ++i)
    up[i] = static_cast<float>(i) * 2.0f;

  auto dev_h = std::make_shared<DeviceHandle>(0);
  DeviceBufferWrapper<float, Abort, Abort, DeviceHandle> dev(32, dev_h);
  const wwrStream_t stream = dev_h->stream().get();
  ASSERT_EQ(ext::copy(dev, up, stream), wwrSuccess);
  ASSERT_EQ(ext::copy(down, dev, stream), wwrSuccess);
  ASSERT_EQ(wwrStreamSynchronize(stream), wwrSuccess);

  for (std::size_t i = 0; i < 32; ++i) {
    EXPECT_EQ(down[i], up[i]) << "at index " << i;
  }
}

TEST(DeviceBufferTests, StreamOrderedAllocationsSurviveRepeatedChurn) {
  // Regression test for the stream-ordered allocate/free pairing.
  // wwrFree performs no implicit synchronisation for a pointer from
  // wwrMallocAsync, so releasing with it handed blocks back to the
  // pool while queued work was still using them - and the pool then
  // reissued them to the next allocation. Freeing on the same stream
  // keeps each block alive until its work has drained.
  auto dev_h = std::make_shared<DeviceHandle>(0);
  GpuStreamWrapper<Abort, Abort> &stream = dev_h->stream();
  for (int iter = 0; iter < 64; ++iter) {
    DeviceBufferWrapper<float, Abort, Abort, DeviceHandle> buf(4096, dev_h);
    ASSERT_NE(buf.data(), nullptr) << "at iteration " << iter;
    EXPECT_EQ(buf.num_elements(), std::size_t{4096}) << "at iteration " << iter;
    EXPECT_EQ(ext::memset(buf, 0xAB, stream.get()), wwrSuccess) << "at iteration " << iter;
  }
  EXPECT_EQ(wwrStreamSynchronize(stream.get()), wwrSuccess);
  EXPECT_EQ(wwrGetLastError(), wwrSuccess);
}

TEST(DeviceBufferTests, StreamOrderedBufferHoldsItsContents) {
  auto dev_h = std::make_shared<DeviceHandle>(0);
  GpuStreamWrapper<Abort, Abort> &stream = dev_h->stream();
  HostBufferWrapper<float, HostAbort, HostAbort> host(128);
  {
    DeviceBufferWrapper<float, Abort, Abort, DeviceHandle> dev(128, dev_h);
    ASSERT_EQ(ext::memset(dev, 0, stream.get()), wwrSuccess);
    ASSERT_EQ(ext::copy(host, dev, stream.get()), wwrSuccess);
    ASSERT_EQ(wwrStreamSynchronize(stream.get()), wwrSuccess);
  }
  for (std::size_t i = 0; i < host.num_elements(); ++i) {
    EXPECT_EQ(host[i], 0.0f) << "at index " << i;
  }
  EXPECT_EQ(wwrStreamSynchronize(stream.get()), wwrSuccess);
}

TEST(DeviceBufferTests, MovedStreamOrderedBufferFreesOnce) {
  auto dev_h = std::make_shared<DeviceHandle>(0);
  GpuStreamWrapper<Abort, Abort> &stream = dev_h->stream();
  {
    DeviceBufferWrapper<float, Abort, Abort, DeviceHandle> src(1024, dev_h);
    DeviceBufferWrapper<float, Abort, Abort, DeviceHandle> dst(std::move(src));
    EXPECT_EQ(src.data(), nullptr);
    EXPECT_NE(dst.data(), nullptr);
  }
  EXPECT_EQ(wwrStreamSynchronize(stream.get()), wwrSuccess);
  EXPECT_EQ(wwrGetLastError(), wwrSuccess);
}

TEST(DeviceHandleTests, ReportsIndexAndQueriesProperties) {
  DeviceHandle dev(0);
  EXPECT_EQ(dev.dev_idx(), 0);
  // props() holds the queried cudaDeviceProp / hipDeviceProp_t; a real device
  // names itself, which also exercises the field across both backends.
  EXPECT_NE(dev.props().name[0], '\0');
  // The stream created eagerly on that device is usable.
  EXPECT_EQ(sync(dev.stream()), wwrSuccess);
  // The memory pool created eagerly on that device is a live handle.
  EXPECT_NE(dev.pool().get(), nullptr);
}

TEST(DeviceBufferTests, HandleAllocationHoldsItsContents) {
  auto dev = std::make_shared<DeviceHandle>(0);
  HostBufferWrapper<float, HostAbort, HostAbort> host(128);
  {
    DeviceBufferWrapper<float, Abort, Abort, DeviceHandle> buf(128, dev);
    ASSERT_NE(buf.data(), nullptr);
    EXPECT_EQ(buf.num_elements(), std::size_t{128});
    ASSERT_EQ(ext::memset(buf, 0, dev->stream().get()), wwrSuccess);
    ASSERT_EQ(ext::copy(host, buf, dev->stream().get()), wwrSuccess);
    ASSERT_EQ(sync(dev->stream()), wwrSuccess);
  }
  for (std::size_t i = 0; i < host.num_elements(); ++i) {
    EXPECT_EQ(host[i], 0.0f) << "at index " << i;
  }
  EXPECT_EQ(wwrGetLastError(), wwrSuccess);
}

TEST(DeviceBufferTests, HandleBufferRetainsStreamAfterLocalHandleReset) {
  // The buffer keeps a shared_ptr to the handle, so dropping the local
  // reference must not destroy the stream the destructor frees on.
  auto dev = std::make_shared<DeviceHandle>(0);
  const wwrStream_t stream = dev->stream().get();
  DeviceBufferWrapper<float, Abort, Abort, DeviceHandle> buf(256, dev);
  dev.reset(); // the buffer's retained handle is now the sole owner
  ASSERT_NE(buf.data(), nullptr);
  EXPECT_EQ(wwrStreamSynchronize(stream), wwrSuccess); // stream still alive
  // ~buf frees on that same stream, then releases the handle - no use-after-free.
}

// Allocate+zero-init and the ownership-transferring move are covered for the
// pinned and unified kinds by HostAccessibleOwningBufferTest above. What stays
// here is what those cannot express: the flags-taking constructors, and that
// unified memory is genuinely host-writable.

TEST(HostAccessibleBufferTests, PinnedBufferWithFlags) {
  PinnedBufferWrapper<float, Abort, Abort> buf(64, wwrHostAllocMapped);
  EXPECT_NE(buf.data(), nullptr);
  EXPECT_EQ(buf.num_elements(), std::size_t{64});
}

TEST(HostAccessibleBufferTests, UnifiedBufferIsHostAccessible) {
  UnifiedBufferWrapper<float, Abort, Abort> buf(64);
  ASSERT_NE(buf.data(), nullptr);
  for (std::size_t i = 0; i < 64; ++i)
    buf[i] = static_cast<float>(i);
  EXPECT_EQ(buf[10], 10.0f);
}

TEST(HostAccessibleBufferTests, UnifiedBufferWithFlags) {
  UnifiedBufferWrapper<float, Abort, Abort> buf(64, wwrMemAttachHost);
  EXPECT_NE(buf.data(), nullptr);
  EXPECT_EQ(buf.num_elements(), std::size_t{64});
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// copy() and memset()
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TEST(CopyAndMemsetTests, SynchronousHostCopy) {
  HostBufferWrapper<float, HostAbort, HostAbort> src(16);
  HostBufferWrapper<float, HostAbort, HostAbort> dst(16);
  for (std::size_t i = 0; i < 16; ++i)
    src[i] = static_cast<float>(i);

  ASSERT_EQ(ext::copy(dst, src), stdHostMemSuccess);
  for (std::size_t i = 0; i < 16; ++i) {
    EXPECT_EQ(dst[i], src[i]) << "at index " << i;
  }
}

TEST(CopyAndMemsetTests, SynchronousHostCopyRejectsUndersizedDestination) {
  HostBufferWrapper<float, HostAbort, HostAbort> src(16);
  HostBufferWrapper<float, HostAbort, HostAbort> dst(4);
  EXPECT_EQ(ext::copy(dst, src), stdHostMemInvalidValue);
}

TEST(CopyAndMemsetTests, EmptyBufferCopyIsANoOp) {
  HostBufferWrapper<float, HostAbort, HostAbort> src;
  HostBufferWrapper<float, HostAbort, HostAbort> dst;
  EXPECT_EQ(ext::copy(dst, src), stdHostMemSuccess);
}

TEST(CopyAndMemsetTests, OffsetCopyMovesOnlyTheRequestedRange) {
  HostBufferWrapper<float, HostAbort, HostAbort> host(16);
  for (std::size_t i = 0; i < 16; ++i)
    host[i] = static_cast<float>(i) + 1.0f;

  auto dev_h = std::make_shared<DeviceHandle>(0);
  DeviceBufferWrapper<float, Abort, Abort, DeviceHandle> dev(16, dev_h);
  HostBufferWrapper<float, HostAbort, HostAbort> out(16);

  const wwrStream_t stream = dev_h->stream().get();
  ASSERT_EQ(ext::copy(dev, 0, host, 4, 8, stream), wwrSuccess);
  ASSERT_EQ(ext::copy(out, 0, dev, 0, 8, stream), wwrSuccess);
  ASSERT_EQ(wwrStreamSynchronize(stream), wwrSuccess);

  for (std::size_t i = 0; i < 8; ++i) {
    EXPECT_EQ(out[i], static_cast<float>(i + 4) + 1.0f) << "at index " << i;
  }
}

TEST(CopyAndMemsetTests, OffsetCopyRejectsOutOfBoundsRange) {
  auto dev_h = std::make_shared<DeviceHandle>(0);
  HostBufferWrapper<float, HostAbort, HostAbort> host(16);
  DeviceBufferWrapper<float, Abort, Abort, DeviceHandle> dev(16, dev_h);
  EXPECT_EQ(ext::copy(dev, 0, host, 12, 8, wwrStream_t{0}), wwrErrorInvalidValue);
}

// ── copy() through reinterpreting views ────────────────────────────
// copy() sees a reinterpret view as an ordinary buffer, so it moves the view's
// byte span and reads offsets/counts in the view's (reinterpreted) element
// units. These pin both, with a reinterpret view on each end.

TEST(CopyAndMemsetTests, SynchronousHostCopyThroughReinterpretViews) {
  HostBufferWrapper<std::uint32_t, HostAbort, HostAbort> src(4);
  HostBufferWrapper<std::uint32_t, HostAbort, HostAbort> dst(4);
  for (std::size_t i = 0; i < 4; ++i) {
    src[i] = 0xDEAD0000u + static_cast<std::uint32_t>(i);
    dst[i] = 0;
  }

  // Byte views over both ends: the sync overload memcpy's the whole 16-byte span.
  BufferViewWrapper<std::byte, MemoryKind::Host, HostAbort, HostAbort> src_bytes = reinterpret_buffer_view<std::byte>(src);
  BufferViewWrapper<std::byte, MemoryKind::Host, HostAbort, HostAbort> dst_bytes = reinterpret_buffer_view<std::byte>(dst);
  ASSERT_EQ(src_bytes.num_elements(), std::size_t{16});
  ASSERT_EQ(ext::copy(dst_bytes, src_bytes), stdHostMemSuccess);

  for (std::size_t i = 0; i < 4; ++i)
    EXPECT_EQ(dst[i], src[i]) << "at index " << i;
}

TEST(CopyAndMemsetTests, OffsetCopyThroughReinterpretViewUsesByteUnits) {
  HostBufferWrapper<std::uint32_t, HostAbort, HostAbort> src(4);
  HostBufferWrapper<std::uint32_t, HostAbort, HostAbort> dst(4);
  for (std::size_t i = 0; i < 4; ++i) {
    src[i] = 0x11111111u * static_cast<std::uint32_t>(i + 1);
    dst[i] = 0;
  }

  BufferViewWrapper<std::byte, MemoryKind::Host, HostAbort, HostAbort> src_bytes = reinterpret_buffer_view<std::byte>(src);
  BufferViewWrapper<std::byte, MemoryKind::Host, HostAbort, HostAbort> dst_bytes = reinterpret_buffer_view<std::byte>(dst);

  // 8 bytes (two uint32) from src[0..] into dst starting at byte 4 (= dst[1]).
  // Offsets and count are in the view's element units, which are bytes here.
  ASSERT_EQ(ext::copy(dst_bytes, 4, src_bytes, 0, 8, wwrStream_t{0}), wwrSuccess);
  ASSERT_EQ(wwrStreamSynchronize(wwrStream_t{0}), wwrSuccess);

  EXPECT_EQ(dst[0], 0u);
  EXPECT_EQ(dst[1], src[0]);
  EXPECT_EQ(dst[2], src[1]);
  EXPECT_EQ(dst[3], 0u);
}

TEST(CopyAndMemsetTests, HostMemsetFillsBuffer) {
  HostBufferWrapper<std::byte, HostAbort, HostAbort> buf(16);
  ASSERT_EQ(ext::memset(buf, 0x5A), stdHostMemSuccess);
  for (std::size_t i = 0; i < 16; ++i) {
    EXPECT_EQ(static_cast<int>(buf[i]), 0x5A) << "at index " << i;
  }
}

TEST(CopyAndMemsetTests, EmptyBufferMemsetIsANoOp) {
  HostBufferWrapper<std::byte, HostAbort, HostAbort> buf;
  EXPECT_EQ(ext::memset(buf, 0x5A), stdHostMemSuccess);
}

TEST(CopyAndMemsetTests, OffsetMemsetRejectsOutOfBoundsRange) {
  auto dev_h = std::make_shared<DeviceHandle>(0);
  DeviceBufferWrapper<float, Abort, Abort, DeviceHandle> dev(16, dev_h);
  EXPECT_EQ(ext::memset(dev, 12, 8, 0, wwrStream_t{0}), wwrErrorInvalidValue);
}

TEST(CopyAndMemsetTests, OffsetMemsetFillsOnlyTheRequestedRange) {
  // The success path of the offset+count device memset: only [offset,
  // offset+count) is written, and the bytes outside it keep the pool draw's
  // zero-init. Previously only the rejection path was exercised, so a memset
  // that ignored its offset and filled from the start would have passed.
  // A std::byte buffer makes the fill byte observable directly; 0xAB in a
  // float would read back as a NaN bit pattern.
  auto dev_h = std::make_shared<DeviceHandle>(0);
  const wwrStream_t stream = dev_h->stream().get();
  DeviceBufferWrapper<std::byte, Abort, Abort, DeviceHandle> dev(16, dev_h); // zero-initialised by the pool draw

  ASSERT_EQ(ext::memset(dev, 4, 8, 0xAB, stream), wwrSuccess);

  HostBufferWrapper<std::byte, HostAbort, HostAbort> host(16);
  ASSERT_EQ(ext::copy(host, dev, stream), wwrSuccess);
  ASSERT_EQ(wwrStreamSynchronize(stream), wwrSuccess);

  for (std::size_t i = 0; i < 16; ++i) {
    const int expected = (i >= 4 && i < 12) ? 0xAB : 0x00;
    EXPECT_EQ(static_cast<int>(host[i]), expected) << "at index " << i;
  }
}

TEST(CopyAndMemsetTests, VoidBufferCopiesByBytes) {
  HostBufferWrapper<void, HostAbort, HostAbort> src(32);
  HostBufferWrapper<void, HostAbort, HostAbort> dst(32);
  EXPECT_EQ(src.size_bytes(), std::size_t{32});
  EXPECT_EQ(ext::copy(dst, src), stdHostMemSuccess);
  EXPECT_EQ(ext::copy(dst, 8, src, 8, 16, wwrStream_t{0}), wwrSuccess);
  EXPECT_EQ(wwrStreamSynchronize(wwrStream_t{0}), wwrSuccess);
}

} // namespace wwr::extension::test
