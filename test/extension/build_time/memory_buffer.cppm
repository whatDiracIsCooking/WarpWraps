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

static_assert(!std::is_copy_constructible_v<HostBufferWrapper<float>>);
static_assert(!std::is_copy_assignable_v<HostBufferWrapper<float>>);
static_assert(!std::is_copy_constructible_v<DeviceBufferWrapper<float>>);
static_assert(std::is_nothrow_move_constructible_v<HostBufferWrapper<float>>);
static_assert(std::is_nothrow_move_assignable_v<HostBufferWrapper<float>>);
static_assert(std::is_nothrow_move_constructible_v<DeviceBufferWrapper<float>>);
static_assert(std::is_nothrow_move_assignable_v<DeviceBufferWrapper<float>>);
static_assert(std::is_nothrow_move_constructible_v<PinnedBufferWrapper<float>>);
static_assert(std::is_nothrow_move_constructible_v<UnifiedBufferWrapper<float>>);

// Views share the buffer_base interface but are copyable and never own.
static_assert(std::is_copy_constructible_v<BufferViewWrapper<float, MemoryKind::Host>>);
static_assert(std::is_copy_assignable_v<BufferViewWrapper<float, MemoryKind::Host>>);
static_assert(BufferViewWrapper<float, MemoryKind::Host>::is_view);
static_assert(!HostBufferWrapper<float>::is_view);
static_assert(buffer_base<HostBufferWrapper<float>>);
static_assert(buffer_base<BufferViewWrapper<float, MemoryKind::Host>>);
static_assert(buffer_base<DeviceBufferWrapper<float>>);
static_assert(same_value_type<HostBufferWrapper<float>, DeviceBufferWrapper<float>>);
static_assert(buffer_typename<HostBufferWrapper<float>, float>);

// void buffers store bytes.
static_assert(HostBufferWrapper<void>::element_size == 1);
static_assert(HostBufferWrapper<float>::element_size == sizeof(float));

// A reinterpreting view is never formed by an implicit cross-type conversion,
// and the reinterpret_view tag that selects the constructor is an in-module
// detail (not re-exported), so the factory is the only public door. Assert both:
// no implicit conversion, and reinterpret_buffer_view carries the source's kind
// and policies onto a view of the requested element type.
static_assert(!std::is_constructible_v<BufferViewWrapper<std::byte, MemoryKind::Host>, HostBufferWrapper<float> &>);
static_assert(std::same_as<decltype(reinterpret_buffer_view<std::byte>(
                               std::declval<HostBufferWrapper<float> &>())),
                           BufferViewWrapper<std::byte, MemoryKind::Host>>);

} // namespace wwr::extension::test
