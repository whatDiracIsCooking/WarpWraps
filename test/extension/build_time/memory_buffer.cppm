// memory_buffer.cppm - Compile-time tests for wwr.extension.memory_buffer

export module wwr.test.extension.memory_buffer;

import std;
import wwr.runtime_api;
import wwr.extension.common;
import wwr.extension.runtime;
import wwr.extension.memory_buffer;

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
static_assert(!std::is_copy_constructible_v<DeviceBufferWrapper<float, Abort, Abort>>);
static_assert(std::is_nothrow_move_constructible_v<HostBufferWrapper<float, HostAbort, HostAbort>>);
static_assert(std::is_nothrow_move_assignable_v<HostBufferWrapper<float, HostAbort, HostAbort>>);
static_assert(std::is_nothrow_move_constructible_v<DeviceBufferWrapper<float, Abort, Abort>>);
static_assert(std::is_nothrow_move_assignable_v<DeviceBufferWrapper<float, Abort, Abort>>);
static_assert(std::is_nothrow_move_constructible_v<PinnedBufferWrapper<float, Abort, Abort>>);
static_assert(std::is_nothrow_move_constructible_v<UnifiedBufferWrapper<float, Abort, Abort>>);

// Views share the buffer_base interface but are copyable and never own.
static_assert(std::is_copy_constructible_v<BufferViewWrapper<float, MemoryKind::Host, HostAbort, HostAbort>>);
static_assert(std::is_copy_assignable_v<BufferViewWrapper<float, MemoryKind::Host, HostAbort, HostAbort>>);
static_assert(BufferViewWrapper<float, MemoryKind::Host, HostAbort, HostAbort>::is_view);
static_assert(!HostBufferWrapper<float, HostAbort, HostAbort>::is_view);
static_assert(buffer_base<HostBufferWrapper<float, HostAbort, HostAbort>>);
static_assert(buffer_base<BufferViewWrapper<float, MemoryKind::Host, HostAbort, HostAbort>>);
static_assert(buffer_base<DeviceBufferWrapper<float, Abort, Abort>>);
static_assert(same_value_type<HostBufferWrapper<float, HostAbort, HostAbort>, DeviceBufferWrapper<float, Abort, Abort>>);
static_assert(buffer_typename<HostBufferWrapper<float, HostAbort, HostAbort>, float>);

// void buffers store bytes.
static_assert(HostBufferWrapper<void, HostAbort, HostAbort>::element_size == 1);
static_assert(HostBufferWrapper<float, HostAbort, HostAbort>::element_size == sizeof(float));

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// [[no_unique_address]] on the alloc/free policy members (issue #67)
//
// A buffer's non-policy state is just data_ + num_elements_. Two DIFFERENT empty
// policy types overlap other subobjects and vanish entirely, so a custom
// alloc/free pair adds nothing. The DEFAULT (P_free = P_alloc, one stateless
// type in both slots) does NOT shrink: [intro.object] forbids two same-type
// empty subobjects from sharing an address, so the second still forces a byte
// plus padding. Both facts are pinned so the attribute is not mistaken for
// zero-cost on the common path (issue #71 collapses the same-type slot).
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
using HErr = stdHostMemoryError_t;

// Distinct empty policies cost nothing: the buffer is exactly its data members.
static_assert(sizeof(HostBufferWrapper<float, EmptyPolicyA<HErr>, EmptyPolicyB<HErr>>) ==
              sizeof(float *) + sizeof(std::size_t));
// The same-type default still pays for the second, otherwise-elided slot.
static_assert(sizeof(HostBufferWrapper<float, HostAbort, HostAbort>) >
              sizeof(HostBufferWrapper<float, EmptyPolicyA<HErr>, EmptyPolicyB<HErr>>));

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
// The device_handle concept and the DeviceBuffer handle axis
//
// DeviceBufferWrapper's fourth parameter lets downstream code back the buffer
// with its own handle type. wwr's DeviceHandle is the reference model; the fake
// handles below stand in for a downstream one and pin the concept's shape.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// A minimal downstream handle: raw backend handles straight out of the
// accessors, no GpuStream/GpuMemPool wrappers in sight.
struct FakeHandle {
  int index() const noexcept { return 0; }
  wwrStream_t alloc_stream() const noexcept { return wwrStream_t{}; }
  wwrMemPool_t mem_pool() const noexcept { return wwrMemPool_t{}; }
};
static_assert(device_handle<FakeHandle>);
static_assert(device_handle<DeviceHandle>);

// The concept bites when the contract is broken: a throwing accessor (unsafe on
// the destructor path) and a missing accessor both fail it.
struct ThrowingHandle {
  int index() const { return 0; } // not noexcept
  wwrStream_t alloc_stream() const noexcept { return wwrStream_t{}; }
  wwrMemPool_t mem_pool() const noexcept { return wwrMemPool_t{}; }
};
static_assert(!device_handle<ThrowingHandle>);

struct MissingHandle {
  int index() const noexcept { return 0; }
};
static_assert(!device_handle<MissingHandle>);

// The buffer instantiates over a downstream handle and keeps its contract.
using FakeDeviceBuffer =
    DeviceBufferWrapper<float, Abort, Abort,
                        FakeHandle>;
static_assert(buffer_base<FakeDeviceBuffer>);
static_assert(std::is_nothrow_move_constructible_v<FakeDeviceBuffer>);
static_assert(!std::is_copy_constructible_v<FakeDeviceBuffer>);
static_assert(std::is_constructible_v<FakeDeviceBuffer, std::size_t, std::shared_ptr<FakeHandle>>);

} // namespace wwr::extension::test
