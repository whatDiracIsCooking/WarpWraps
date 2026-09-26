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
import gpumod.extension.runtime;

namespace gpumod::extension::test {

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// GpuStream Tests
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TEST(GpuStreamTests, DefaultConstructor) {
  GpuStream stream;
  EXPECT_NE(stream.get(), nullptr);
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

  gpuFree(buf);
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

// A recording policy for the threaded-policy guard below. DefaultErrorPolicy
// aborts, so it cannot observe the guard's behaviour; this one counts instead,
// which is all the error_policy concept asks for.
template<typename T>
class RecordingScopePolicy : public BaseErrorPolicy<T> {
public:
  void handle_error(const T error, std::source_location) override {
    ++count_;
    last_ = error;
  }
  std::size_t count() const noexcept { return count_; }

private:
  std::size_t count_ = 0;
  T last_{};
};

// DeviceScopeWrapper's two-argument constructor threads a caller's own error
// policy through the guard, borrowing it by reference rather than copying, so a
// stateful policy observes the guard's gpuGetDevice/gpuSetDevice in place. This
// is the path DeviceBuffer uses to route the device switch through its
// allocation/deallocation policy. Device 0 always exists, so the switch here
// succeeds and the policy stays untouched -- which is exactly the contract on
// the success path: the guard reports nothing it was not asked to.
TEST(DeviceScopeTests, ThreadsBorrowedPolicy) {
  int before = -1;
  ASSERT_EQ(gpuGetDevice(&before), gpuSuccess);

  RecordingScopePolicy<gpuError_t> policy;
  {
    DeviceScopeWrapper<RecordingScopePolicy<gpuError_t>> scope(before, policy);
    EXPECT_EQ(scope.original_idx, before);
  }

  EXPECT_EQ(policy.count(), std::size_t{0});

  int after = -1;
  ASSERT_EQ(gpuGetDevice(&after), gpuSuccess);
  EXPECT_EQ(after, before);
}

} // namespace gpumod::extension::test
