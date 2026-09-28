// basic.cpp - Basic GPU runtime wrapper tests
// Tests for the GpuStreamWrapper, GpuEventWrapper, GpuGraphWrapper(Exec) and
// GpuMemPoolWrapper wrappers
//
// A plain TU, not a module interface unit. Its self-registering test objects
// are compiled straight into the executable, so it needs no module for a
// main.cpp to import.
//
// No global Environment here: these tests build their own stream and event and
// share nothing, so this binary links GTest::gtest_main and has no main.cpp.

#include <gtest/gtest.h>

import std;
import wwr.runtime_api;
import wwr.extension.common;
import wwr.extension.handle;
import wwr.extension.runtime;

namespace wwr::extension::test {
// Bind abort-on-failure once, for this file's wrapper instantiations.
using Abort = AbortPolicy<wwrError_t>;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// GpuStreamWrapper Tests
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TEST(GpuStreamTests, DefaultConstructor) {
  GpuStreamWrapper<Abort, Abort> stream;
  EXPECT_NE(stream.get(), nullptr);

  // Default-constructed streams are non-blocking: they do not serialize against
  // the legacy default stream (0).
  unsigned int flags = 0;
  ASSERT_EQ(wwrStreamGetFlags(stream.get(), &flags), wwrSuccess);
  EXPECT_EQ(flags & wwrStreamNonBlocking, wwrStreamNonBlocking);
}

TEST(GpuStreamTests, WithFlags) {
  GpuStreamWrapper<Abort, Abort> stream(0, wwrStreamNonBlocking);
  EXPECT_NE(stream.get(), nullptr);
}

TEST(GpuStreamTests, RecordsCreationDevice) {
  // The default constructor creates on device 0.
  GpuStreamWrapper<Abort, Abort> stream;
  EXPECT_EQ(stream.dev_idx(), 0);

  // The (dev_idx, flags) constructor records the device it was told to use.
  GpuStreamWrapper<Abort, Abort> flagged(0, wwrStreamNonBlocking);
  EXPECT_EQ(flagged.dev_idx(), 0);
}

TEST(GpuStreamTests, ConstructOnDevice) {
  // dev_idx is the (defaulted) first constructor argument. Device 0 always exists.
  GpuStreamWrapper<Abort, Abort> stream(0);
  EXPECT_NE(stream.get(), nullptr);
  EXPECT_EQ(stream.dev_idx(), 0);
}

TEST(GpuStreamTests, MovePreservesDevice) {
  GpuStreamWrapper<Abort, Abort> stream1;
  const int dev = stream1.dev_idx();

  GpuStreamWrapper<Abort, Abort> stream2(std::move(stream1));
  EXPECT_EQ(stream2.dev_idx(), dev);
  EXPECT_EQ(stream1.dev_idx(), -1);
}

TEST(GpuStreamTests, MoveConstructor) {
  GpuStreamWrapper<Abort, Abort> stream1;
  wwrStream_t handle = stream1.get();

  GpuStreamWrapper<Abort, Abort> stream2(std::move(stream1));
  EXPECT_EQ(stream2.get(), handle);
  EXPECT_EQ(stream1.get(), nullptr);
}

TEST(GpuStreamTests, MoveAssignment) {
  GpuStreamWrapper<Abort, Abort> stream1;
  GpuStreamWrapper<Abort, Abort> stream2;
  wwrStream_t handle1 = stream1.get();

  stream2 = std::move(stream1);
  EXPECT_EQ(stream2.get(), handle1);
  EXPECT_EQ(stream1.get(), nullptr);
}

TEST(GpuStreamTests, CaptureToGraph) {
  // The full capture -> instantiate -> launch -> synchronize round-trip:
  // work enqueued between begin_capture and end_capture is recorded into a
  // graph rather than run, then replayed by launching the instantiated exec.
  GpuStreamWrapper<Abort, Abort> stream;

  void *buf = nullptr;
  ASSERT_EQ(wwrMalloc(&buf, sizeof(int)), wwrSuccess);

  ASSERT_EQ(begin_capture(stream), wwrSuccess);
  ASSERT_EQ(wwrMemsetAsync(buf, 0, sizeof(int), stream.get()), wwrSuccess);
  GpuGraphWrapper<Abort, Abort> graph = stream.end_capture();
  EXPECT_NE(graph.get(), nullptr);

  GpuGraphExecWrapper<Abort, Abort> exec = graph.instantiate();
  ASSERT_EQ(launch(exec, stream.get()), wwrSuccess);
  EXPECT_EQ(sync(stream), wwrSuccess);

  EXPECT_EQ(wwrFree(buf), wwrSuccess);
}

TEST(GpuStreamTests, WaitEventOrdersWorkAcrossStreams) {
  // wait_event() is the cross-stream ordering primitive and nothing else here
  // exercises it (the other tests reach the runtime through the raw API on
  // .get()). Enqueue work on `producer`, record an event on it, then make
  // `consumer` wait on that event through the wrapper method before its own
  // work. The whole chain draining with success is the observable contract.
  GpuStreamWrapper<Abort, Abort> producer;
  GpuStreamWrapper<Abort, Abort> consumer;
  GpuEventWrapper<Abort, Abort> event;

  void *buf = nullptr;
  ASSERT_EQ(wwrMalloc(&buf, sizeof(int)), wwrSuccess);

  ASSERT_EQ(wwrMemsetAsync(buf, 0, sizeof(int), producer.get()), wwrSuccess);
  ASSERT_EQ(record(event, producer.get()), wwrSuccess);
  EXPECT_EQ(wait_event(consumer, event.get()), wwrSuccess);
  ASSERT_EQ(wwrMemsetAsync(buf, 1, sizeof(int), consumer.get()), wwrSuccess);
  EXPECT_EQ(sync(consumer), wwrSuccess);
  EXPECT_EQ(sync(producer), wwrSuccess);

  EXPECT_EQ(wwrFree(buf), wwrSuccess);
}

// The handle views are proven to compile and convert in
// test/extension/build_time/handle_view.cppm (static_asserts only). Each owner
// suite carries one ViewBorrows* case proving the view mirrors the owner's
// handle/device AND that a borrowed op actually runs against the device -- the
// The borrow-safe ops are free functions on the raw handle, so a view and its
// owner reach the same definition; one op per view is enough to prove the view
// converts to the handle, and the ops themselves are exercised by the owner
// cases above.
TEST(GpuStreamTests, ViewBorrowsHandleAndDrivesWork) {
  GpuStreamWrapper<Abort, Abort> stream;
  GpuStreamView view = stream.view();
  ASSERT_EQ(view.get(), stream.get());
  ASSERT_EQ(view.dev_idx(), stream.dev_idx());

  GpuEventWrapper<Abort, Abort> event;
  ASSERT_EQ(record(event, stream.get()), wwrSuccess);
  EXPECT_EQ(wait_event(view, event.get()), wwrSuccess);
  EXPECT_EQ(sync(view), wwrSuccess);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// GpuEventWrapper Tests
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TEST(GpuEventTests, DefaultConstructor) {
  GpuEventWrapper<Abort, Abort> event;
  EXPECT_NE(event.get(), nullptr);
}

TEST(GpuEventTests, WithFlags) {
  GpuEventWrapper<Abort, Abort> event(0, wwrEventDisableTiming);
  EXPECT_NE(event.get(), nullptr);
}

TEST(GpuEventTests, RecordsCreationDevice) {
  // The default constructor creates on device 0.
  GpuEventWrapper<Abort, Abort> event;
  EXPECT_EQ(event.dev_idx(), 0);

  // The (dev_idx, flags) constructor records the device it was told to use.
  GpuEventWrapper<Abort, Abort> flagged(0, wwrEventDisableTiming);
  EXPECT_EQ(flagged.dev_idx(), 0);
}

TEST(GpuEventTests, ConstructOnDevice) {
  // dev_idx is the (defaulted) first constructor argument. Device 0 always exists.
  GpuEventWrapper<Abort, Abort> event(0);
  EXPECT_NE(event.get(), nullptr);
  EXPECT_EQ(event.dev_idx(), 0);
}

TEST(GpuEventTests, MoveConstructor) {
  GpuEventWrapper<Abort, Abort> event1;
  wwrEvent_t handle = event1.get();

  GpuEventWrapper<Abort, Abort> event2(std::move(event1));
  EXPECT_EQ(event2.get(), handle);
  EXPECT_EQ(event1.get(), nullptr);
}

TEST(GpuEventTests, MoveAssignment) {
  GpuEventWrapper<Abort, Abort> event1;
  GpuEventWrapper<Abort, Abort> event2;
  wwrEvent_t handle1 = event1.get();

  event2 = std::move(event1);
  EXPECT_EQ(event2.get(), handle1);
  EXPECT_EQ(event1.get(), nullptr);
}

TEST(GpuEventTests, RecordAndSynchronize) {
  GpuStreamWrapper<Abort, Abort> stream;
  GpuEventWrapper<Abort, Abort> event;

  // Record event on stream
  ASSERT_EQ(wwrEventRecord(event.get(), stream.get()), wwrSuccess);

  // Synchronize on event
  EXPECT_EQ(wwrEventSynchronize(event.get()), wwrSuccess);
}

TEST(GpuEventTests, QueryEvent) {
  GpuStreamWrapper<Abort, Abort> stream;
  GpuEventWrapper<Abort, Abort> event;

  // Record event on stream
  ASSERT_EQ(wwrEventRecord(event.get(), stream.get()), wwrSuccess);

  // Wait for event to complete
  ASSERT_EQ(wwrEventSynchronize(event.get()), wwrSuccess);

  // Query should now return success
  EXPECT_EQ(wwrEventQuery(event.get()), wwrSuccess);
}

TEST(GpuEventTests, MemberRecordAndSync) {
  // The suite records through the raw API elsewhere; this drives the record()/
  // sync() free functions on an owner, including the two-arg record(event,
  // stream, flags) overload (flag 0 is always valid).
  GpuStreamWrapper<Abort, Abort> stream;
  GpuEventWrapper<Abort, Abort> event;

  ASSERT_EQ(record(event, stream.get()), wwrSuccess);
  EXPECT_EQ(sync(event), wwrSuccess);

  ASSERT_EQ(record(event, stream.get(), 0), wwrSuccess);
  EXPECT_EQ(sync(event), wwrSuccess);
}

TEST(GpuEventTests, ViewBorrowsHandleAndDrivesWork) {
  // One borrowed op proves the view converts; record(event, stream, flags) is
  // already covered on the owner by MemberRecordAndSync (same free functions).
  GpuStreamWrapper<Abort, Abort> stream;
  GpuEventWrapper<Abort, Abort> event;
  GpuEventView view = event.view();
  ASSERT_EQ(view.get(), event.get());
  ASSERT_EQ(view.dev_idx(), event.dev_idx());

  ASSERT_EQ(record(view, stream.get()), wwrSuccess);
  EXPECT_EQ(sync(view), wwrSuccess);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// GpuGraphWrapper Tests
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TEST(GpuGraphTests, DefaultConstructor) {
  GpuGraphWrapper<Abort, Abort> graph;
  EXPECT_NE(graph.get(), nullptr);
}

TEST(GpuGraphTests, MoveConstructor) {
  GpuGraphWrapper<Abort, Abort> graph1;
  wwrGraph_t handle = graph1.get();

  GpuGraphWrapper<Abort, Abort> graph2(std::move(graph1));
  EXPECT_EQ(graph2.get(), handle);
  EXPECT_EQ(graph1.get(), nullptr);
}

TEST(GpuGraphTests, MoveAssignment) {
  GpuGraphWrapper<Abort, Abort> graph1;
  GpuGraphWrapper<Abort, Abort> graph2;
  wwrGraph_t handle1 = graph1.get();

  graph2 = std::move(graph1);
  EXPECT_EQ(graph2.get(), handle1);
  EXPECT_EQ(graph1.get(), nullptr);
}

TEST(GpuGraphTests, Instantiate) {
  GpuGraphWrapper<Abort, Abort> graph;
  GpuGraphExecWrapper<Abort, Abort> exec = graph.instantiate();
  EXPECT_NE(exec.get(), nullptr);
}

TEST(GpuGraphTests, ViewBorrowsHandle) {
  // HandleView<wwrGraph_t> carries only the handle: a graph is not device-bound and its
  // ops (instantiate) produce owned objects, so the view has no borrow-safe
  // operations of its own. Check it mirrors the owner's handle.
  GpuGraphWrapper<Abort, Abort> graph;
  HandleView<wwrGraph_t> view = graph.view();
  EXPECT_EQ(view.get(), graph.get());
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// GpuGraphExecWrapper Tests
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TEST(GpuGraphExecTests, ConstructFromGraph) {
  GpuGraphWrapper<Abort, Abort> graph;
  GpuGraphExecWrapper<Abort, Abort> exec(graph.get());
  EXPECT_NE(exec.get(), nullptr);
}

TEST(GpuGraphExecTests, MoveConstructor) {
  GpuGraphWrapper<Abort, Abort> graph;
  GpuGraphExecWrapper<Abort, Abort> exec1(graph.get());
  wwrGraphExec_t handle = exec1.get();

  GpuGraphExecWrapper<Abort, Abort> exec2(std::move(exec1));
  EXPECT_EQ(exec2.get(), handle);
  EXPECT_EQ(exec1.get(), nullptr);
}

TEST(GpuGraphExecTests, MoveAssignment) {
  GpuGraphWrapper<Abort, Abort> graph1;
  GpuGraphWrapper<Abort, Abort> graph2;
  GpuGraphExecWrapper<Abort, Abort> exec1(graph1.get());
  GpuGraphExecWrapper<Abort, Abort> exec2(graph2.get());
  wwrGraphExec_t handle1 = exec1.get();

  exec2 = std::move(exec1);
  EXPECT_EQ(exec2.get(), handle1);
  EXPECT_EQ(exec1.get(), nullptr);
}

TEST(GpuGraphExecTests, LaunchEmptyGraph) {
  // An empty graph instantiates and launches as a no-op; this exercises the
  // full create -> instantiate -> launch -> synchronize round-trip.
  GpuStreamWrapper<Abort, Abort> stream;
  GpuGraphWrapper<Abort, Abort> graph;
  GpuGraphExecWrapper<Abort, Abort> exec = graph.instantiate();

  ASSERT_EQ(launch(exec, stream.get()), wwrSuccess);
  EXPECT_EQ(sync(stream), wwrSuccess);
}

TEST(GpuGraphExecTests, UploadThenLaunch) {
  // upload() places the exec on the stream's device without launching it, and
  // nothing else exercises it. Following it with launch proves the uploaded
  // exec is the one that runs.
  GpuStreamWrapper<Abort, Abort> stream;
  GpuGraphWrapper<Abort, Abort> graph;
  GpuGraphExecWrapper<Abort, Abort> exec = graph.instantiate();

  ASSERT_EQ(upload(exec, stream.get()), wwrSuccess);
  ASSERT_EQ(launch(exec, stream.get()), wwrSuccess);
  EXPECT_EQ(sync(stream), wwrSuccess);
}

TEST(GpuGraphExecTests, ViewBorrowsHandleAndDrivesWork) {
  // One borrowed op proves the view converts; upload() is covered on the owner
  // by UploadThenLaunch (same free functions).
  GpuStreamWrapper<Abort, Abort> stream;
  GpuGraphWrapper<Abort, Abort> graph;
  GpuGraphExecWrapper<Abort, Abort> exec = graph.instantiate();
  GpuGraphExecView view = exec.view();
  ASSERT_EQ(view.get(), exec.get());

  ASSERT_EQ(launch(view, stream.get()), wwrSuccess);
  EXPECT_EQ(sync(stream), wwrSuccess);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// GpuMemPoolWrapper Tests
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//
// GpuMemPoolWrapper is a DeviceBoundHandle, like GpuStreamWrapper/GpuEventWrapper, so it carries the
// same device-recording and move contract on top of the base handle -- these
// mirror the stream/event cases and add the pool's own two extra constructors
// (release-threshold and explicit-props) plus a live allocation round-trip.

TEST(GpuMemPoolTests, DefaultConstructor) {
  GpuMemPoolWrapper<Abort, Abort> pool;
  EXPECT_NE(pool.get(), nullptr);
}

TEST(GpuMemPoolTests, RecordsCreationDevice) {
  GpuMemPoolWrapper<Abort, Abort> pool;
  EXPECT_EQ(pool.dev_idx(), 0);
}

TEST(GpuMemPoolTests, ConstructWithReleaseThreshold) {
  GpuMemPoolWrapper<Abort, Abort> pool(0, 2u * 1024u * 1024u);
  EXPECT_NE(pool.get(), nullptr);
  EXPECT_EQ(pool.dev_idx(), 0);
}

TEST(GpuMemPoolTests, ConstructFromProps) {
  wwrMemPoolProps props = {};
  props.allocType = wwrMemAllocationTypePinned;
  props.handleTypes = wwrMemHandleTypeNone;
  props.location.type = wwrMemLocationTypeDevice;
  props.location.id = 0;

  GpuMemPoolWrapper<Abort, Abort> pool(props);
  EXPECT_NE(pool.get(), nullptr);
  EXPECT_EQ(pool.dev_idx(), 0);
}

TEST(GpuMemPoolTests, MoveConstructor) {
  GpuMemPoolWrapper<Abort, Abort> pool1;
  wwrMemPool_t handle = pool1.get();

  GpuMemPoolWrapper<Abort, Abort> pool2(std::move(pool1));
  EXPECT_EQ(pool2.get(), handle);
  EXPECT_EQ(pool1.get(), nullptr);
}

TEST(GpuMemPoolTests, MoveAssignment) {
  GpuMemPoolWrapper<Abort, Abort> pool1;
  GpuMemPoolWrapper<Abort, Abort> pool2;
  wwrMemPool_t handle1 = pool1.get();

  pool2 = std::move(pool1);
  EXPECT_EQ(pool2.get(), handle1);
  EXPECT_EQ(pool1.get(), nullptr);
}

TEST(GpuMemPoolTests, MovePreservesDevice) {
  GpuMemPoolWrapper<Abort, Abort> pool1;
  const int dev = pool1.dev_idx();

  GpuMemPoolWrapper<Abort, Abort> pool2(std::move(pool1));
  EXPECT_EQ(pool2.dev_idx(), dev);
  EXPECT_EQ(pool1.dev_idx(), -1);
}

TEST(GpuMemPoolTests, StreamOrderedAllocationRoundTrips) {
  // A live pool: allocate from it on a stream, then free back to it. Proves
  // the wrapped handle is a usable pool, not just a non-null pointer.
  GpuMemPoolWrapper<Abort, Abort> pool;
  GpuStreamWrapper<Abort, Abort> stream;

  void *ptr = nullptr;
  ASSERT_EQ(wwrMallocFromPoolAsync(&ptr, 1024, pool.get(), stream.get()), wwrSuccess);
  ASSERT_EQ(sync(stream), wwrSuccess);
  EXPECT_NE(ptr, nullptr);

  ASSERT_EQ(wwrFreeAsync(ptr, stream.get()), wwrSuccess);
  EXPECT_EQ(sync(stream), wwrSuccess);
}

TEST(GpuMemPoolTests, ViewBorrowsHandle) {
  // DeviceBoundHandleView<wwrMemPool_t> carries the handle and device index but has no borrow-safe
  // ops (a pool handle is consumed by allocation calls). Check it mirrors the
  // owner.
  GpuMemPoolWrapper<Abort, Abort> pool;
  DeviceBoundHandleView<wwrMemPool_t> view = pool.view();
  EXPECT_EQ(view.get(), pool.get());
  EXPECT_EQ(view.dev_idx(), pool.dev_idx());
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// DeviceScope Tests
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//
// DeviceScope (wwr.extension.common) makes a device current for its lifetime
// and restores the previously-current device on destruction. These check the
// three parts of that contract: the original device is recorded, the target
// becomes current, and the original is restored. Device 0 always exists, so the
// tests target it; on a single-GPU box target and original coincide, which
// still exercises record and restore but does not distinguish a cross-device
// switch -- that distinction only shows on multi-GPU hardware.

TEST(DeviceScopeTests, RecordsOriginalDevice) {
  int before = -1;
  ASSERT_EQ(wwrGetDevice(&before), wwrSuccess);

  DeviceScope scope(before);
  EXPECT_EQ(scope.original_idx, before);
}

TEST(DeviceScopeTests, MakesTargetCurrent) {
  DeviceScope scope(0);

  int current = -1;
  ASSERT_EQ(wwrGetDevice(&current), wwrSuccess);
  EXPECT_EQ(current, 0);
}

TEST(DeviceScopeTests, RestoresPreviousDeviceOnDestruction) {
  int before = -1;
  ASSERT_EQ(wwrGetDevice(&before), wwrSuccess);

  {
    DeviceScope scope(before);
  }

  int after = -1;
  ASSERT_EQ(wwrGetDevice(&after), wwrSuccess);
  EXPECT_EQ(after, before);
}

TEST(DeviceScopeTests, NestedScopesRestore) {
  int before = -1;
  ASSERT_EQ(wwrGetDevice(&before), wwrSuccess);

  {
    DeviceScope outer(before);
    {
      DeviceScope inner(before);
      int inside = -1;
      ASSERT_EQ(wwrGetDevice(&inside), wwrSuccess);
      EXPECT_EQ(inside, before);
    }
    int after_inner = -1;
    ASSERT_EQ(wwrGetDevice(&after_inner), wwrSuccess);
    EXPECT_EQ(after_inner, before);
  }

  int after_outer = -1;
  ASSERT_EQ(wwrGetDevice(&after_outer), wwrSuccess);
  EXPECT_EQ(after_outer, before);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Runtime wrapper error-policy Tests
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//
// Every other test here uses the default (aborting) policy on the success path,
// so nothing proves a custom P_create is actually threaded through a runtime
// wrapper and fires on a real failure -- the counterpart of
// test/extension/memory_buffer's allocation-failure suite.
//
// The policy records to statics because the flag constructors default-construct
// it (there is no flags+policy overload) and the wrapper exposes no accessor to
// read an instance back. AbortPolicy aborts, so it cannot observe the
// path. An invalid creation-flag mask forces wwrEventCreateWithFlags to return
// wwrErrorInvalidValue -- a recoverable error that allocates nothing, so unlike
// the memory allocation-failure suite this needs no no_sanitizer label.

template<typename T>
class ProbePolicy {
public:
  using error_type = T;
  static inline std::size_t count = 0;
  static inline T last{};
  static void reset() {
    count = 0;
    last = T{};
  }
  void handle_error(const T error, std::source_location) noexcept {
    ++count;
    last = error;
  }
};

TEST(RuntimePolicyTests, CreationFailureFiresCreatePolicy) {
  ProbePolicy<wwrError_t>::reset();
  {
    // 0xFFFFFFFF is not a valid event-creation flag mask, so
    // wwrEventCreateWithFlags fails and leaves the handle null -- the
    // destructor then frees nothing, and only the create policy fires.
    GpuEventWrapper<ProbePolicy<wwrError_t>, ProbePolicy<wwrError_t>> event(0, 0xFFFFFFFFu);
    EXPECT_EQ(event.get(), nullptr);
  }
  EXPECT_GE(ProbePolicy<wwrError_t>::count, std::size_t{1});
  EXPECT_EQ(ProbePolicy<wwrError_t>::last, wwrErrorInvalidValue);

  // The failed create left a sticky error; clear it so later tests see a clean
  // context (as the memory allocation-failure suite does).
  static_cast<void>(wwrGetLastError());
}

} // namespace wwr::extension::test
