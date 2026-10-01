// memory_buffer.cppm - Compile-time tests for wwr.extension.memory_buffer

export module wwr.test.extension.memory_buffer;

import std;
import wwr.runtime_api;
import wwr.extension.common;
import wwr.extension.handle;
import wwr.extension.memory_buffer;
import wwr.test.shared.abort_policy;
import wwr.test.shared.device_handle;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time contract of the buffer types
//
// Owning buffers are move-only and never throw from a move; views share the
// buffer_base interface but are copyable and never own. These compile-time
// contracts live here; test/extension/memory_buffer/buffer_tests.cpp keeps the
// runtime tests.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace wwr::extension::test {
// Bind abort-on-failure once per error type, for this file's instantiations.
using Abort = AbortPolicy<wwrError_t>;
using HostAbort = AbortPolicy<stdHostMemoryError_t>;

static_assert(!std::is_copy_constructible_v<HostBufferWrapper<float, HostAbort, HostAbort>>);
static_assert(!std::is_copy_assignable_v<HostBufferWrapper<float, HostAbort, HostAbort>>);
static_assert(!std::is_copy_constructible_v<DeviceBufferWrapper<float, Abort, Abort, Abort, DeviceHandle>>);
static_assert(std::is_nothrow_move_constructible_v<HostBufferWrapper<float, HostAbort, HostAbort>>);
static_assert(std::is_nothrow_move_assignable_v<HostBufferWrapper<float, HostAbort, HostAbort>>);
static_assert(std::is_nothrow_move_constructible_v<DeviceBufferWrapper<float, Abort, Abort, Abort, DeviceHandle>>);
static_assert(std::is_nothrow_move_assignable_v<DeviceBufferWrapper<float, Abort, Abort, Abort, DeviceHandle>>);
static_assert(std::is_nothrow_move_constructible_v<PinnedBufferWrapper<float, Abort, Abort>>);
static_assert(std::is_nothrow_move_constructible_v<UnifiedBufferWrapper<float, Abort, Abort>>);

// Views share the buffer_base interface but are copyable and never own.
static_assert(std::is_copy_constructible_v<BufferViewWrapper<float, MemoryKind::Host, HostAbort, HostAbort>>);
static_assert(std::is_copy_assignable_v<BufferViewWrapper<float, MemoryKind::Host, HostAbort, HostAbort>>);
static_assert(BufferViewWrapper<float, MemoryKind::Host, HostAbort, HostAbort>::is_view);
static_assert(!HostBufferWrapper<float, HostAbort, HostAbort>::is_view);
static_assert(buffer_base<HostBufferWrapper<float, HostAbort, HostAbort>>);
static_assert(buffer_base<BufferViewWrapper<float, MemoryKind::Host, HostAbort, HostAbort>>);
static_assert(buffer_base<DeviceBufferWrapper<float, Abort, Abort, Abort, DeviceHandle>>);
static_assert(same_value_type<HostBufferWrapper<float, HostAbort, HostAbort>, DeviceBufferWrapper<float, Abort, Abort, Abort, DeviceHandle>>);
static_assert(buffer_typename<HostBufferWrapper<float, HostAbort, HostAbort>, float>);

// void buffers store bytes.
static_assert(HostBufferWrapper<void, HostAbort, HostAbort>::element_size == 1);
static_assert(HostBufferWrapper<float, HostAbort, HostAbort>::element_size == sizeof(float));

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Zero-cost alloc/free policy slots (issues #67, #71)
//
// A buffer's non-policy state is just data_ + num_elements_. Two DIFFERENT empty
// policy types overlap other subobjects and vanish under [[no_unique_address]]
// (#67). The same-type DEFAULT (P_free = P_alloc) cannot share an address --
// [intro.object] forbids two same-type empty subobjects from doing so -- so #71
// collapses the free slot into the alloc slot via policy_slot: same-type empty
// policies store a single instance, making the common path as free as the
// distinct case. A stateful policy is never collapsed and still costs its bytes.
// The 16-byte data floor has no slack, so its sizeof witnesses the win directly.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

template<typename E>
struct EmptyPolicyA {
  using error_type = E;
  void handle_error(E, std::source_location) noexcept {}
};
template<typename E>
struct EmptyPolicyB {
  using error_type = E;
  void handle_error(E, std::source_location) noexcept {}
};
template<typename E>
struct StatefulPolicy {
  using error_type = E;
  int count = 0; // real state -- must survive, so this slot is never collapsed
  void handle_error(E, std::source_location) noexcept { ++count; }
};
using HErr = stdHostMemoryError_t;

// policy_slot contract: a same-type empty free slot elides; a stateful one stays.
static_assert(std::is_empty_v<policy_slot<EmptyPolicyA<HErr>, EmptyPolicyA<HErr>>>);
static_assert(!std::is_empty_v<policy_slot<StatefulPolicy<HErr>, StatefulPolicy<HErr>>>);

// Distinct empty policies cost nothing: the buffer is exactly its data members.
static_assert(sizeof(HostBufferWrapper<float, EmptyPolicyA<HErr>, EmptyPolicyB<HErr>>) ==
              sizeof(float *) + sizeof(std::size_t));
// #71: the same-type default is now equally free -- the collapsed slot vanishes.
static_assert(sizeof(HostBufferWrapper<float, HostAbort, HostAbort>) ==
              sizeof(float *) + sizeof(std::size_t));
// A stateful policy is not collapsed, so it still enlarges the buffer.
static_assert(sizeof(HostBufferWrapper<float, StatefulPolicy<HErr>, StatefulPolicy<HErr>>) >
              sizeof(float *) + sizeof(std::size_t));

// A reinterpreting view is never formed by an implicit cross-type conversion,
// and the reinterpret_view tag that selects the constructor is an in-module
// detail (not re-exported), so the factory is the only public door. Assert both:
// no implicit conversion, and reinterpret_buffer_view carries the source's kind
// and policies onto a view of the requested element type.
static_assert(!std::is_constructible_v<BufferViewWrapper<std::byte, MemoryKind::Host, HostAbort, HostAbort>, HostBufferWrapper<float, HostAbort, HostAbort> &>);
static_assert(std::same_as<decltype(reinterpret_buffer_view<std::byte>(
                               std::declval<HostBufferWrapper<float, HostAbort, HostAbort> &>())),
                           BufferViewWrapper<std::byte, MemoryKind::Host, HostAbort, HostAbort>>);

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// The device_handle capability ladder and the DeviceBuffer handle axis
//
// DeviceBufferWrapper's fourth parameter lets downstream code back the buffer
// with its own handle type. The test DeviceHandle is the reference model (the
// fullest tier); the fakes below stand in for downstream handles and pin the
// three concept tiers -- each adds one accessor and unlocks one strategy.
// Raw backend handles straight out of the accessors, no Stream/MemPool
// wrappers in sight -- which is what the convertible_to (not .get()) shape buys.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// Bare tier: dev_idx() only -> synchronous wwrMalloc/wwrFree.
struct IndexOnlyFake {
  int dev_idx() const noexcept { return 0; }
};
static_assert(device_handle<IndexOnlyFake>);
static_assert(!device_handle_stream<IndexOnlyFake>);

// Stream tier: adds stream() -> wwrMallocAsync/wwrFreeAsync.
struct StreamOnlyFake {
  int dev_idx() const noexcept { return 0; }
  wwrStream_t stream() const noexcept { return wwrStream_t{}; }
};
static_assert(device_handle_stream<StreamOnlyFake>);
static_assert(!device_handle_pool<StreamOnlyFake>);

// Pool tier: adds pool() -> wwrMallocFromPoolAsync. Matches DeviceHandle's shape.
struct FakeHandle {
  int dev_idx() const noexcept { return 0; }
  wwrStream_t stream() const noexcept { return wwrStream_t{}; }
  wwrMemPool_t pool() const noexcept { return wwrMemPool_t{}; }
};
static_assert(device_handle_pool<FakeHandle>);
static_assert(device_handle_pool<DeviceHandle>);

// The concept bites when the contract is broken: a throwing accessor (unsafe on
// the destructor path) fails the tier that needs it, and a handle missing even
// dev_idx() fails the root.
struct ThrowingHandle {
  int dev_idx() const { return 0; } // not noexcept
  wwrStream_t stream() const noexcept { return wwrStream_t{}; }
};
static_assert(!device_handle<ThrowingHandle>);
static_assert(!device_handle_stream<ThrowingHandle>);

struct MissingHandle {
  wwrStream_t stream() const noexcept { return wwrStream_t{}; }
};
static_assert(!device_handle<MissingHandle>);

// The buffer instantiates over a downstream handle at every tier and keeps its
// contract; construction from a shared handle holds throughout.
template<typename H>
using FakeBuffer = DeviceBufferWrapper<float, Abort, Abort, Abort, H>;
static_assert(buffer_base<FakeBuffer<IndexOnlyFake>>);
static_assert(buffer_base<FakeBuffer<StreamOnlyFake>>);
static_assert(buffer_base<FakeBuffer<FakeHandle>>);
static_assert(std::is_nothrow_move_constructible_v<FakeBuffer<FakeHandle>>);
static_assert(!std::is_copy_constructible_v<FakeBuffer<FakeHandle>>);
static_assert(std::is_constructible_v<FakeBuffer<IndexOnlyFake>, std::size_t,
                                      std::shared_ptr<IndexOnlyFake>>);
static_assert(std::is_constructible_v<FakeBuffer<FakeHandle>, std::size_t,
                                      std::shared_ptr<FakeHandle>>);

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// The buffer/view suite (#177)
//
// The suite binds the alias prelude from one place; the contract is that it emits
// *exactly* the wrappers a consumer would spell by hand -- the right error family
// per kind, and each view paired to its buffer's policies (the drift footgun the
// suite exists to close). The convenience is the map form under a single-policy
// map, so the two must agree; and buffer_suite stops before the device kind while
// device_buffer_suite adds it.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

using ConvBuf = kit::device_buffers<AbortPolicy, DeviceHandle>;

// The right wrapper and error family per kind: host speaks stdHostMemoryError_t,
// the GPU-managed kinds speak wwrError_t, and device carries its access policy.
static_assert(std::same_as<ConvBuf::host<float>, HostBufferWrapper<float, HostAbort, HostAbort>>);
static_assert(std::same_as<ConvBuf::pinned<float>, PinnedBufferWrapper<float, Abort, Abort>>);
static_assert(std::same_as<ConvBuf::unified<float>, UnifiedBufferWrapper<float, Abort, Abort>>);
static_assert(std::same_as<ConvBuf::device<float>,
                           DeviceBufferWrapper<float, Abort, Abort, Abort, DeviceHandle>>);

// Each view pairs with its buffer's kind and policies -- derived via view_of from
// the buffer alias, so it cannot drift from it.
static_assert(std::same_as<ConvBuf::host_view<float>,
                           BufferViewWrapper<float, MemoryKind::Host, HostAbort, HostAbort>>);
static_assert(std::same_as<ConvBuf::device_view<float>,
                           BufferViewWrapper<float, MemoryKind::Device, Abort, Abort>>);
static_assert(std::same_as<ConvBuf::host_view<float>, kit::view_of<ConvBuf::host<float>>>);

// The convenience is the map form under a single-policy map -- the two agree.
struct SingleAbortMap {
  template<typename E>
  using alloc = AbortPolicy<E>;
  template<typename E>
  using free = AbortPolicy<E>;
};
static_assert(std::same_as<kit::device_buffers<AbortPolicy, DeviceHandle>,
                           kit::device_buffer_suite<kit::single_policy_map<AbortPolicy>, DeviceHandle>>);
static_assert(std::same_as<ConvBuf::host<float>, kit::buffer_suite<SingleAbortMap>::host<float>>);

// buffer_suite and device_buffer_suite share the host/pinned/unified aliases; the
// device kind lives only on the device suite (buffer_suite<M>::device does not
// exist, which is what lets a host-only consumer skip the handle).
static_assert(std::same_as<kit::buffer_suite<SingleAbortMap>::host<float>,
                           kit::device_buffer_suite<SingleAbortMap, DeviceHandle>::host<float>>);

} // namespace wwr::extension::test
