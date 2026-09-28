// memory_buffer.cppm - Compile-time tests for wwr.extension.memory_buffer

export module wwr.test.extension.memory_buffer;

import std;
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

static_assert(!std::is_copy_constructible_v<HostBuffer<float>>);
static_assert(!std::is_copy_assignable_v<HostBuffer<float>>);
static_assert(!std::is_copy_constructible_v<DeviceBuffer<float>>);
static_assert(std::is_nothrow_move_constructible_v<HostBuffer<float>>);
static_assert(std::is_nothrow_move_assignable_v<HostBuffer<float>>);
static_assert(std::is_nothrow_move_constructible_v<DeviceBuffer<float>>);
static_assert(std::is_nothrow_move_assignable_v<DeviceBuffer<float>>);
static_assert(std::is_nothrow_move_constructible_v<PinnedBuffer<float>>);
static_assert(std::is_nothrow_move_constructible_v<UnifiedBuffer<float>>);

// Views share the buffer_base interface but are copyable and never own.
static_assert(std::is_copy_constructible_v<HostBufferView<float>>);
static_assert(std::is_copy_assignable_v<HostBufferView<float>>);
static_assert(HostBufferView<float>::is_view);
static_assert(!HostBuffer<float>::is_view);
static_assert(buffer_base<HostBuffer<float>>);
static_assert(buffer_base<HostBufferView<float>>);
static_assert(buffer_base<DeviceBuffer<float>>);
static_assert(same_value_type<HostBuffer<float>, DeviceBuffer<float>>);
static_assert(buffer_typename<HostBuffer<float>, float>);

// void buffers store bytes.
static_assert(HostBuffer<void>::element_size == 1);
static_assert(HostBuffer<float>::element_size == sizeof(float));

// A reinterpreting view is never formed by an implicit cross-type conversion,
// and the reinterpret_view tag that selects the constructor is an in-module
// detail (not re-exported), so the factory is the only public door. Assert both:
// no implicit conversion, and reinterpret_buffer_view carries the source's kind
// and policies onto a view of the requested element type.
static_assert(!std::is_constructible_v<HostBufferView<std::byte>, HostBuffer<float> &>);
static_assert(std::same_as<decltype(reinterpret_buffer_view<std::byte>(
                               std::declval<HostBuffer<float> &>())),
                           HostBufferView<std::byte>>);

} // namespace wwr::extension::test
