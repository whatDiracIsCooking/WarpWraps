// basic.cpp - Basic GPU runtime wrapper tests
// Tests for the GpuStream, GpuEvent, GpuGraph(Exec), GpuMemPool and
// StreamEventPair wrappers
//
// A plain TU, not a module interface unit. Its self-registering test objects
// are compiled straight into the executable, so it needs no module for a
// main.cpp to import.
//
// No global Environment here: these tests build their own stream and event and
// share nothing, so this binary links GTest::gtest_main and has no main.cpp.

#include <gtest/gtest.h>

import std;
import gpumod.runtime_api;
import gpumod.extension.common;
import gpumod.extension.handle;
import gpumod.extension.runtime;

namespace gpumod::extension::test {

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// GpuStream Tests
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TEST(GpuStreamTests, DefaultConstructor) {
  GpuStream stream;
  EXPECT_NE(stream.get(), nullptr);

  // Default-constructed streams are non-blocking: they do not serialize against
  // the legacy default stream (0).
  unsigned int flags = 0;
  ASSERT_EQ(gpuStreamGetFlags(stream.get(), &flags), gpuSuccess);
  EXPECT_EQ(flags & gpuStreamNonBlocking, gpuStreamNonBlocking);
}

TEST(GpuStreamTests, WithFlags) {
  GpuStream stream(0, gpuStreamNonBlocking);
  EXPECT_NE(stream.get(), nullptr);
}

TEST(GpuStreamTests, RecordsCreationDevice) {
  // The default constructor creates on device 0.
  GpuStream stream;
  EXPECT_EQ(stream.dev_idx(), 0);

  // The (dev_idx, flags) constructor records the device it was told to use.
  GpuStream flagged(0, gpuStreamNonBlocking);
  EXPECT_EQ(flagged.dev_idx(), 0);
}

TEST(GpuStreamTests, ConstructOnDevice) {
  // dev_idx is the (defaulted) first constructor argument. Device 0 always exists.
  GpuStream stream(0);
  EXPECT_NE(stream.get(), nullptr);
  EXPECT_EQ(stream.dev_idx(), 0);
}

TEST(GpuStreamTests, MovePreservesDevice) {
  GpuStream stream1;
  const int dev = stream1.dev_idx();

  GpuStream stream2(std::move(stream1));
  EXPECT_EQ(stream2.dev_idx(), dev);
  EXPECT_EQ(stream1.dev_idx(), -1);
}

TEST(GpuStreamTests, MoveConstructor) {
  GpuStream stream1;
  gpuStream_t handle = stream1.get();

  GpuStream stream2(std::move(stream1));
  EXPECT_EQ(stream2.get(), handle);
  EXPECT_EQ(stream1.get(), nullptr);
}

TEST(GpuStreamTests, MoveAssignment) {
  GpuStream stream1;
  GpuStream stream2;
  gpuStream_t handle1 = stream1.get();

  stream2 = std::move(stream1);
  EXPECT_EQ(stream2.get(), handle1);
  EXPECT_EQ(stream1.get(), nullptr);
}

TEST(GpuStreamTests, CaptureToGraph) {
  // The full capture -> instantiate -> launch -> synchronize round-trip:
  // work enqueued between begin_capture and end_capture is recorded into a
  // graph rather than run, then replayed by launching the instantiated exec.
  GpuStream stream;

  void *buf = nullptr;
  ASSERT_EQ(gpuMalloc(&buf, sizeof(int)), gpuSuccess);

  ASSERT_EQ(stream.begin_capture(), gpuSuccess);
  ASSERT_EQ(gpuMemsetAsync(buf, 0, sizeof(int), stream.get()), gpuSuccess);
  GpuGraph graph = stream.end_capture();
  EXPECT_NE(graph.get(), nullptr);

  GpuGraphExec exec = graph.instantiate();
  ASSERT_EQ(exec.launch(stream.get()), gpuSuccess);
  EXPECT_EQ(stream.sync(), gpuSuccess);

  EXPECT_EQ(gpuFree(buf), gpuSuccess);
}

TEST(GpuStreamTests, WaitEventOrdersWorkAcrossStreams) {
  // wait_event() is the cross-stream ordering primitive and nothing else here
  // exercises it (the other tests reach the runtime through the raw API on
  // .get()). Enqueue work on `producer`, record an event on it, then make
  // `consumer` wait on that event through the wrapper method before its own
  // work. The whole chain draining with success is the observable contract.
  GpuStream producer;
  GpuStream consumer;
  GpuEvent event;

  void *buf = nullptr;
  ASSERT_EQ(gpuMalloc(&buf, sizeof(int)), gpuSuccess);

  ASSERT_EQ(gpuMemsetAsync(buf, 0, sizeof(int), producer.get()), gpuSuccess);
  ASSERT_EQ(event.record(producer.get()), gpuSuccess);
  EXPECT_EQ(consumer.wait_event(event.get()), gpuSuccess);
  ASSERT_EQ(gpuMemsetAsync(buf, 1, sizeof(int), consumer.get()), gpuSuccess);
  EXPECT_EQ(consumer.sync(), gpuSuccess);
  EXPECT_EQ(producer.sync(), gpuSuccess);

  EXPECT_EQ(gpuFree(buf), gpuSuccess);
}

// The handle views are proven to compile and convert in
// test/extension/build_time/handle_view.cppm (static_asserts only). Each owner
// suite carries one ViewBorrows* case proving the view mirrors the owner's
// handle/device AND that a borrowed op actually runs against the device -- the
// GpuStreamAccess/GpuEventAccess/GpuGraphExecAccess mixins are shared with the
// owner, so one borrowed op per view is enough to prove the forwarding wiring;
// the mixin methods themselves are exercised by the owner cases above.
TEST(GpuStreamTests, ViewBorrowsHandleAndDrivesWork) {
  GpuStream stream;
  GpuStreamView view = stream.view();
  ASSERT_EQ(view.get(), stream.get());
  ASSERT_EQ(view.dev_idx(), stream.dev_idx());

  GpuEvent event;
  ASSERT_EQ(event.record(stream.get()), gpuSuccess);
  EXPECT_EQ(view.wait_event(event.get()), gpuSuccess);
  EXPECT_EQ(view.sync(), gpuSuccess);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// GpuEvent Tests
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TEST(GpuEventTests, DefaultConstructor) {
  GpuEvent event;
  EXPECT_NE(event.get(), nullptr);
}

TEST(GpuEventTests, WithFlags) {
  GpuEvent event(0, gpuEventDisableTiming);
  EXPECT_NE(event.get(), nullptr);
}

TEST(GpuEventTests, RecordsCreationDevice) {
  // The default constructor creates on device 0.
  GpuEvent event;
  EXPECT_EQ(event.dev_idx(), 0);

  // The (dev_idx, flags) constructor records the device it was told to use.
  GpuEvent flagged(0, gpuEventDisableTiming);
  EXPECT_EQ(flagged.dev_idx(), 0);
}

TEST(GpuEventTests, ConstructOnDevice) {
  // dev_idx is the (defaulted) first constructor argument. Device 0 always exists.
  GpuEvent event(0);
  EXPECT_NE(event.get(), nullptr);
  EXPECT_EQ(event.dev_idx(), 0);
}

TEST(GpuEventTests, MoveConstructor) {
  GpuEvent event1;
  gpuEvent_t handle = event1.get();

  GpuEvent event2(std::move(event1));
  EXPECT_EQ(event2.get(), handle);
  EXPECT_EQ(event1.get(), nullptr);
}

TEST(GpuEventTests, MoveAssignment) {
  GpuEvent event1;
  GpuEvent event2;
  gpuEvent_t handle1 = event1.get();

  event2 = std::move(event1);
  EXPECT_EQ(event2.get(), handle1);
  EXPECT_EQ(event1.get(), nullptr);
}

TEST(GpuEventTests, RecordAndSynchronize) {
  GpuStream stream;
  GpuEvent event;

  // Record event on stream
  ASSERT_EQ(gpuEventRecord(event.get(), stream.get()), gpuSuccess);

  // Synchronize on event
  EXPECT_EQ(gpuEventSynchronize(event.get()), gpuSuccess);
}

TEST(GpuEventTests, QueryEvent) {
  GpuStream stream;
  GpuEvent event;

  // Record event on stream
  ASSERT_EQ(gpuEventRecord(event.get(), stream.get()), gpuSuccess);

  // Wait for event to complete
  ASSERT_EQ(gpuEventSynchronize(event.get()), gpuSuccess);

  // Query should now return success
  EXPECT_EQ(gpuEventQuery(event.get()), gpuSuccess);
}

TEST(GpuEventTests, MemberRecordAndSync) {
  // The suite records through the raw API elsewhere; this drives the wrapper's
  // own record()/sync() members, including the two-arg record(stream, flags)
  // overload (flag 0 is always valid).
  GpuStream stream;
  GpuEvent event;

  ASSERT_EQ(event.record(stream.get()), gpuSuccess);
  EXPECT_EQ(event.sync(), gpuSuccess);

  ASSERT_EQ(event.record(stream.get(), 0), gpuSuccess);
  EXPECT_EQ(event.sync(), gpuSuccess);
}

TEST(GpuEventTests, ViewBorrowsHandleAndDrivesWork) {
  // One borrowed op proves forwarding; record(stream, flags) is already covered
  // on the owner by MemberRecordAndSync (same GpuEventAccess mixin).
  GpuStream stream;
  GpuEvent event;
  GpuEventView view = event.view();
  ASSERT_EQ(view.get(), event.get());
  ASSERT_EQ(view.dev_idx(), event.dev_idx());

  ASSERT_EQ(view.record(stream.get()), gpuSuccess);
  EXPECT_EQ(view.sync(), gpuSuccess);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// GpuGraph Tests
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TEST(GpuGraphTests, DefaultConstructor) {
  GpuGraph graph;
  EXPECT_NE(graph.get(), nullptr);
}

TEST(GpuGraphTests, MoveConstructor) {
  GpuGraph graph1;
  gpuGraph_t handle = graph1.get();

  GpuGraph graph2(std::move(graph1));
  EXPECT_EQ(graph2.get(), handle);
  EXPECT_EQ(graph1.get(), nullptr);
}

TEST(GpuGraphTests, MoveAssignment) {
  GpuGraph graph1;
  GpuGraph graph2;
  gpuGraph_t handle1 = graph1.get();

  graph2 = std::move(graph1);
  EXPECT_EQ(graph2.get(), handle1);
  EXPECT_EQ(graph1.get(), nullptr);
}

TEST(GpuGraphTests, Instantiate) {
  GpuGraph graph;
  GpuGraphExec exec = graph.instantiate();
  EXPECT_NE(exec.get(), nullptr);
}

TEST(GpuGraphTests, ViewBorrowsHandle) {
  // GpuGraphView carries only the handle: a graph is not device-bound and its
  // ops (instantiate) produce owned objects, so the view has no borrow-safe
  // operations of its own. Check it mirrors the owner's handle.
  GpuGraph graph;
  GpuGraphView view = graph.view();
  EXPECT_EQ(view.get(), graph.get());
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// GpuGraphExec Tests
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TEST(GpuGraphExecTests, ConstructFromGraph) {
  GpuGraph graph;
  GpuGraphExec exec(graph.get());
  EXPECT_NE(exec.get(), nullptr);
}

TEST(GpuGraphExecTests, MoveConstructor) {
  GpuGraph graph;
  GpuGraphExec exec1(graph.get());
  gpuGraphExec_t handle = exec1.get();

  GpuGraphExec exec2(std::move(exec1));
  EXPECT_EQ(exec2.get(), handle);
  EXPECT_EQ(exec1.get(), nullptr);
}

TEST(GpuGraphExecTests, MoveAssignment) {
  GpuGraph graph1;
  GpuGraph graph2;
  GpuGraphExec exec1(graph1.get());
  GpuGraphExec exec2(graph2.get());
  gpuGraphExec_t handle1 = exec1.get();

  exec2 = std::move(exec1);
  EXPECT_EQ(exec2.get(), handle1);
  EXPECT_EQ(exec1.get(), nullptr);
}

TEST(GpuGraphExecTests, LaunchEmptyGraph) {
  // An empty graph instantiates and launches as a no-op; this exercises the
  // full create -> instantiate -> launch -> synchronize round-trip.
  GpuStream stream;
  GpuGraph graph;
  GpuGraphExec exec = graph.instantiate();

  ASSERT_EQ(exec.launch(stream.get()), gpuSuccess);
  EXPECT_EQ(stream.sync(), gpuSuccess);
}

TEST(GpuGraphExecTests, UploadThenLaunch) {
  // upload() places the exec on the stream's device without launching it, and
  // nothing else exercises it. Following it with launch proves the uploaded
  // exec is the one that runs.
  GpuStream stream;
  GpuGraph graph;
  GpuGraphExec exec = graph.instantiate();

  ASSERT_EQ(exec.upload(stream.get()), gpuSuccess);
  ASSERT_EQ(exec.launch(stream.get()), gpuSuccess);
  EXPECT_EQ(stream.sync(), gpuSuccess);
}

TEST(GpuGraphExecTests, ViewBorrowsHandleAndDrivesWork) {
  // One borrowed op proves forwarding; upload() is covered on the owner by
  // UploadThenLaunch (same GpuGraphExecAccess mixin).
  GpuStream stream;
  GpuGraph graph;
  GpuGraphExec exec = graph.instantiate();
  GpuGraphExecView view = exec.view();
  ASSERT_EQ(view.get(), exec.get());

  ASSERT_EQ(view.launch(stream.get()), gpuSuccess);
  EXPECT_EQ(stream.sync(), gpuSuccess);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// GpuMemPool Tests
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//
// GpuMemPool is a GpuBoundHandle, like GpuStream/GpuEvent, so it carries the
// same device-recording and move contract on top of the base handle -- these
// mirror the stream/event cases and add the pool's own two extra constructors
// (release-threshold and explicit-props) plus a live allocation round-trip.

TEST(GpuMemPoolTests, DefaultConstructor) {
  GpuMemPool pool;
  EXPECT_NE(pool.get(), nullptr);
}

TEST(GpuMemPoolTests, RecordsCreationDevice) {
  GpuMemPool pool;
  EXPECT_EQ(pool.dev_idx(), 0);
}

TEST(GpuMemPoolTests, ConstructWithReleaseThreshold) {
  GpuMemPool pool(0, 2u * 1024u * 1024u);
  EXPECT_NE(pool.get(), nullptr);
  EXPECT_EQ(pool.dev_idx(), 0);
}

TEST(GpuMemPoolTests, ConstructFromProps) {
  gpuMemPoolProps props = {};
  props.allocType = gpuMemAllocationTypePinned;
  props.handleTypes = gpuMemHandleTypeNone;
  props.location.type = gpuMemLocationTypeDevice;
  props.location.id = 0;

  GpuMemPool pool(props);
  EXPECT_NE(pool.get(), nullptr);
  EXPECT_EQ(pool.dev_idx(), 0);
}

TEST(GpuMemPoolTests, MoveConstructor) {
  GpuMemPool pool1;
  gpuMemPool_t handle = pool1.get();

  GpuMemPool pool2(std::move(pool1));
  EXPECT_EQ(pool2.get(), handle);
  EXPECT_EQ(pool1.get(), nullptr);
}

TEST(GpuMemPoolTests, MoveAssignment) {
  GpuMemPool pool1;
  GpuMemPool pool2;
  gpuMemPool_t handle1 = pool1.get();

  pool2 = std::move(pool1);
  EXPECT_EQ(pool2.get(), handle1);
  EXPECT_EQ(pool1.get(), nullptr);
}

TEST(GpuMemPoolTests, MovePreservesDevice) {
  GpuMemPool pool1;
  const int dev = pool1.dev_idx();

  GpuMemPool pool2(std::move(pool1));
  EXPECT_EQ(pool2.dev_idx(), dev);
  EXPECT_EQ(pool1.dev_idx(), -1);
}

TEST(GpuMemPoolTests, StreamOrderedAllocationRoundTrips) {
  // A live pool: allocate from it on a stream, then free back to it. Proves
  // the wrapped handle is a usable pool, not just a non-null pointer.
  GpuMemPool pool;
  GpuStream stream;

  void *ptr = nullptr;
  ASSERT_EQ(gpuMallocFromPoolAsync(&ptr, 1024, pool.get(), stream.get()), gpuSuccess);
  ASSERT_EQ(stream.sync(), gpuSuccess);
  EXPECT_NE(ptr, nullptr);

  ASSERT_EQ(gpuFreeAsync(ptr, stream.get()), gpuSuccess);
  EXPECT_EQ(stream.sync(), gpuSuccess);
}

TEST(GpuMemPoolTests, ViewBorrowsHandle) {
  // GpuMemPoolView carries the handle and device index but has no borrow-safe
  // ops (a pool handle is consumed by allocation calls). Check it mirrors the
  // owner.
  GpuMemPool pool;
  GpuMemPoolView view = pool.view();
  EXPECT_EQ(view.get(), pool.get());
  EXPECT_EQ(view.dev_idx(), pool.dev_idx());
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// StreamEventPair Tests
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//
// StreamEventPair bundles a GpuStream and a GpuEvent. It is not a handle
// wrapper of its own -- ownership is the two members', moved member-wise -- so
// these check that each StreamEventConfig branch builds a live pair, that the
// accessors agree with the raw getters, that record/sync work end to end, and
// that a move empties the source's members (each GpuStream/GpuEvent nulls on
// move).

TEST(StreamEventPairTests, DefaultConstructor) {
  StreamEventPair pair;
  EXPECT_NE(pair.stream_raw(), nullptr);
  EXPECT_NE(pair.event_raw(), nullptr);
  EXPECT_EQ(pair.gpu_stream().dev_idx(), 0);
  EXPECT_EQ(pair.gpu_event().dev_idx(), 0);
}

TEST(StreamEventPairTests, ConstructWithStreamFlags) {
  StreamEventConfig cfg;
  cfg.device = 0;
  cfg.stream_flags = gpuStreamNonBlocking;

  StreamEventPair pair(cfg);
  EXPECT_NE(pair.stream_raw(), nullptr);
  EXPECT_NE(pair.event_raw(), nullptr);
}

TEST(StreamEventPairTests, ConstructWithStreamPriority) {
  // Priority 0 is always in range; this drives make_stream's priority branch.
  StreamEventConfig cfg;
  cfg.device = 0;
  cfg.stream_priority = 0;

  StreamEventPair pair(cfg);
  EXPECT_NE(pair.stream_raw(), nullptr);
  EXPECT_NE(pair.event_raw(), nullptr);
}

TEST(StreamEventPairTests, ConstructWithEventFlags) {
  StreamEventConfig cfg;
  cfg.device = 0;
  cfg.event_flags = gpuEventDisableTiming;

  StreamEventPair pair(cfg);
  EXPECT_NE(pair.stream_raw(), nullptr);
  EXPECT_NE(pair.event_raw(), nullptr);
}

TEST(StreamEventPairTests, AccessorsMatchRawGetters) {
  StreamEventPair pair;
  EXPECT_EQ(pair.gpu_stream().get(), pair.stream_raw());
  EXPECT_EQ(pair.gpu_event().get(), pair.event_raw());
}

TEST(StreamEventPairTests, RecordAndSync) {
  StreamEventPair pair;
  ASSERT_EQ(pair.record(), gpuSuccess);
  EXPECT_EQ(pair.event_sync(), gpuSuccess);
  EXPECT_EQ(pair.stream_sync(), gpuSuccess);
}

TEST(StreamEventPairTests, RecordWithFlagsAndSync) {
  // The flagged record(flags) overload, which routes to the event's two-arg
  // record; flag 0 is always valid.
  StreamEventPair pair;
  ASSERT_EQ(pair.record(0), gpuSuccess);
  EXPECT_EQ(pair.event_sync(), gpuSuccess);
  EXPECT_EQ(pair.stream_sync(), gpuSuccess);
}

TEST(StreamEventPairTests, MoveConstructor) {
  StreamEventPair pair1;
  gpuStream_t stream = pair1.stream_raw();
  gpuEvent_t event = pair1.event_raw();

  StreamEventPair pair2(std::move(pair1));
  EXPECT_EQ(pair2.stream_raw(), stream);
  EXPECT_EQ(pair2.event_raw(), event);
  EXPECT_EQ(pair1.stream_raw(), nullptr);
  EXPECT_EQ(pair1.event_raw(), nullptr);
}

TEST(StreamEventPairTests, MoveAssignment) {
  StreamEventPair pair1;
  StreamEventPair pair2;
  gpuStream_t stream = pair1.stream_raw();
  gpuEvent_t event = pair1.event_raw();

  pair2 = std::move(pair1);
  EXPECT_EQ(pair2.stream_raw(), stream);
  EXPECT_EQ(pair2.event_raw(), event);
  EXPECT_EQ(pair1.stream_raw(), nullptr);
  EXPECT_EQ(pair1.event_raw(), nullptr);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// DeviceScope Tests
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//
// DeviceScope (gpumod.extension.common) makes a device current for its lifetime
// and restores the previously-current device on destruction. These check the
// three parts of that contract: the original device is recorded, the target
// becomes current, and the original is restored. Device 0 always exists, so the
// tests target it; on a single-GPU box target and original coincide, which
// still exercises record and restore but does not distinguish a cross-device
// switch -- that distinction only shows on multi-GPU hardware.

TEST(DeviceScopeTests, RecordsOriginalDevice) {
  int before = -1;
  ASSERT_EQ(gpuGetDevice(&before), gpuSuccess);

  DeviceScope scope(before);
  EXPECT_EQ(scope.original_idx, before);
}

TEST(DeviceScopeTests, MakesTargetCurrent) {
  DeviceScope scope(0);

  int current = -1;
  ASSERT_EQ(gpuGetDevice(&current), gpuSuccess);
  EXPECT_EQ(current, 0);
}

TEST(DeviceScopeTests, RestoresPreviousDeviceOnDestruction) {
  int before = -1;
  ASSERT_EQ(gpuGetDevice(&before), gpuSuccess);

  {
    DeviceScope scope(before);
  }

  int after = -1;
  ASSERT_EQ(gpuGetDevice(&after), gpuSuccess);
  EXPECT_EQ(after, before);
}

TEST(DeviceScopeTests, NestedScopesRestore) {
  int before = -1;
  ASSERT_EQ(gpuGetDevice(&before), gpuSuccess);

  {
    DeviceScope outer(before);
    {
      DeviceScope inner(before);
      int inside = -1;
      ASSERT_EQ(gpuGetDevice(&inside), gpuSuccess);
      EXPECT_EQ(inside, before);
    }
    int after_inner = -1;
    ASSERT_EQ(gpuGetDevice(&after_inner), gpuSuccess);
    EXPECT_EQ(after_inner, before);
  }

  int after_outer = -1;
  ASSERT_EQ(gpuGetDevice(&after_outer), gpuSuccess);
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
// read an instance back. DefaultErrorPolicy aborts, so it cannot observe the
// path. An invalid creation-flag mask forces gpuEventCreateWithFlags to return
// gpuErrorInvalidValue -- a recoverable error that allocates nothing, so unlike
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
  ProbePolicy<gpuError_t>::reset();
  {
    // 0xFFFFFFFF is not a valid event-creation flag mask, so
    // gpuEventCreateWithFlags fails and leaves the handle null -- the
    // destructor then frees nothing, and only the create policy fires.
    GpuEventWrapper<ProbePolicy<gpuError_t>> event(0, 0xFFFFFFFFu);
    EXPECT_EQ(event.get(), nullptr);
  }
  EXPECT_GE(ProbePolicy<gpuError_t>::count, std::size_t{1});
  EXPECT_EQ(ProbePolicy<gpuError_t>::last, gpuErrorInvalidValue);

  // The failed create left a sticky error; clear it so later tests see a clean
  // context (as the memory allocation-failure suite does).
  static_cast<void>(gpuGetLastError());
}

} // namespace gpumod::extension::test
