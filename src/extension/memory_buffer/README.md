# wwr.extension.memory_buffer

RAII-based memory buffer management for all GPU-relevant memory kinds. Provides strongly typed, move-only buffer wrappers with pluggable error policies, a unified copy API, and C++20 concepts for generic buffer programming.

## Module

`wwr.extension.memory_buffer`

## Buffer Types

All buffer types live in the `wwr::extension` namespace and are template aliases over their respective `*Wrapper` classes with default error policies.

| Alias | Memory kind | Allocation API | Host-accessible |
|---|---|---|---|
| `DeviceBuffer<T>` | GPU device memory | `wwrMallocFromPoolAsync` (from a `DeviceHandle`'s pool) | No |
| `PinnedBuffer<T>` | Page-locked host memory | `wwrHostAlloc` (`wwrHostAllocDefault` unless flags are given) | Yes |
| `UnifiedBuffer<T>` | Unified (managed) memory | `wwrMallocManaged` | Yes |
| `HostBuffer<T>` | Standard host memory | `std::malloc` | Yes |

### Common Interface (from `BaseBuffer`)

All buffer types inherit from `BaseBuffer<T, K, Derived, P_alloc, P_free>` (CRTP) and expose:

```cpp
T*             data() noexcept;
const T*       data() const noexcept;
std::size_t    num_elements() const noexcept;
std::size_t    size_bytes() const noexcept;
storage_type&  operator[](std::size_t index) noexcept;   // host-accessible, non-void buffers only
operator T*() noexcept;
operator const T*() const noexcept;

static constexpr std::size_t max_num_elements;  // largest count whose byte size fits in size_t
```

Buffers are move-constructible and move-assignable; copy construction and assignment are deleted.

`operator[]` is declared as `storage_type&` rather than `T&` so that `Buffer<void>` remains
instantiable; `storage_type` is `T` for every case in which the operator is enabled.

### Invariants

- The element type must be trivially copyable (`std::byte` stands in for `void`). These
  buffers fill and copy raw storage bytewise and never construct a `T`, so a
  non-trivially-copyable element type is rejected at construction with a `static_assert`.
- `data() == nullptr` if and only if `num_elements() == 0`.
- Constructing with more than `max_num_elements` elements is rejected before any
  allocation is attempted: `num_elements * element_size` would otherwise wrap and produce a
  small allocation behind a buffer still reporting the requested count.
- A failed allocation leaves an empty buffer. This matters only for error policies that
  return rather than terminate; the default policy aborts.
- Constructing with `0` elements yields an empty buffer without calling the allocator.

### Constructors

**Default (all types except `DeviceBuffer`):** `Buffer<T> buf;` — empty buffer (`nullptr`, 0 elements).

**Synchronous allocation (all types except `DeviceBuffer`):** `Buffer<T> buf(n);` — allocates `n` elements.

**With error policy (all types except `DeviceBuffer`):** `Buffer<T> buf(n, policy);` or `Buffer<T> buf(n, policy_alloc, policy_free);`

**Device buffers are always drawn from a shared `DeviceHandle`:**
```cpp
auto device = std::make_shared<DeviceHandle>(0);
DeviceBuffer<T> buf(n, device);          // wwrMallocFromPoolAsync from the handle's pool
DeviceBuffer<T> buf(n, device, policy);  // ...with a custom error policy
```

A `DeviceBuffer` selects the handle's device with `wwrSetDevice`, allocates `n` elements from
the handle's memory pool (`mem_pool()`) on its `alloc_stream()`, and zero-initialises them on
that stream. It keeps a `shared_ptr` to the handle, so the pool and stream — owned solely by
the `DeviceHandle`, never shared on their own — outlive the buffer's `wwrFreeAsync`. The block
is released with `wwrFreeAsync` on that same stream; `wwrFree` performs no implicit
synchronisation for a pointer from the stream-ordered pool allocator, so using it would return
the block while work still queued on the stream was reading and writing it.

**Pinned-specific with flags:**
```cpp
PinnedBuffer<T> buf(n, wwrHostAllocMapped);  // wwrHostAlloc with flags
```

**Unified-specific with flags:**
```cpp
UnifiedBuffer<T> buf(n, wwrMemAttachHost);   // wwrMallocManaged with flags
```

## Error Handling

Error policies are parameterised via `P_alloc` and `P_free` template arguments. The default is `DefaultErrorPolicy<ErrorType>`. The error type varies by memory kind:

| Kind | Error type |
|---|---|
| `Device`, `Pinned`, `Unified` | `wwrError_t` |
| `Host` | `stdHostMemoryError_t` |

`stdHostMemoryError_t` is a project-defined enum providing `stdHostMemSuccess`, `stdHostMemAllocFailure`, `stdHostMemDeallocFailure`, and `stdHostMemInvalidValue`.

**Constraint:** `P_free` must not throw, as it is invoked from the destructor. This is enforced at compile time by the `nothrow_error_policy` concept on the `P_free` slot.

## Copy Functions

Declared in the `:copy` partition, exported from the primary interface.

### Synchronous host-to-host copy

```cpp
template <buffer_base B1, buffer_base B2>
    requires same_value_type<B1, B2> && (!B1::is_device) && (!B2::is_device) && ...
[[nodiscard]] stdHostMemoryError_t copy(B1& dst, const B2& src) noexcept;
```

Uses `std::memcpy`. Validates bounds before copying.

### Stream-ordered async copy (full buffer)

```cpp
template <buffer_base B1, buffer_base B2>
    requires same_value_type<B1, B2> && ...
[[nodiscard]] wwrError_t copy(B1& dst, const B2& src, wwrStream_t stream) noexcept;
```

### Stream-ordered async copy (with offsets)

```cpp
template <buffer_base B1, buffer_base B2>
    requires same_value_type<B1, B2> && ...
[[nodiscard]] wwrError_t copy(B1& dst, std::size_t dst_offset,
                                const B2& src, std::size_t src_offset,
                                std::size_t count,
                                wwrStream_t stream) noexcept;
```

Both async overloads use `wwrMemcpyAsync` with `wwrMemcpyDefault` (direction inferred from Unified Virtual Addressing). All copy functions validate bounds and return an error code rather than throwing.

## Concepts

| Concept | Description |
|---|---|
| `buffer_base<B>` | `B` has `value_type`, `memory_kind`, `data()`, `num_elements()`, `size_bytes()` |
| `buffer_typename<B, T>` | `buffer_base<B>` and `B::value_type == T` |
| `same_value_type<B1, B2>` | Both satisfy `buffer_base` and have the same `value_type` |

## Buffer Views

`BufferViewWrapper<T, K, P_alloc, P_free>` is a non-owning view over any buffer of the same
`T` and `K`, aliased per kind as `DeviceBufferView<T>`, `PinnedBufferView<T>`,
`HostBufferView<T>` and `UnifiedBufferView<T>`.

```cpp
HostBufferView<float> full(buf);            // whole buffer
HostBufferView<float> tail(buf, 12);        // from element 12 to the end
HostBufferView<float> sub(buf, 4, 8);       // 8 elements starting at element 4
HostBufferView<float> inner(sub, 2, 3);     // views compose
```

Views are copyable and never deallocate. The source is taken by non-const reference, since a
view hands out a mutable `T*`. Out-of-range ranges are reported through `P_alloc` and leave the
view empty; the bounds check does not overflow for large offsets or counts.

### Reinterpreting (cross-type) views

A view's element type may differ from the source buffer's. This is a byte
reinterpretation, so it is never an implicit conversion — form it with the
`reinterpret_buffer_view<T2>` factory (or, equivalently, the `reinterpret_view`
tag constructor):

```cpp
HostBuffer<float> buf(16);                            // 64 bytes
auto bytes = reinterpret_buffer_view<std::byte>(buf); // HostBufferView<std::byte>, 64 elements
auto ints  = reinterpret_buffer_view<int>(buf);       // HostBufferView<int>, 16 elements
HostBufferView<std::byte> b(reinterpret_view, buf);   // the tag form, same result
```

The byte span is preserved: `num_elements()` becomes `src.size_bytes() / sizeof(T2)`, and the
source's memory kind and error policies are carried onto the view. Reinterpretation is reported
through `P_alloc` and leaves the view empty when the source's byte size is not a whole multiple
of `sizeof(T2)`, or when its base pointer is not aligned for `T2` — the latter is reachable for
a sub-view, whose offset into an allocation aligned for one type need not be aligned for a
wider one. A device buffer can be reinterpreted too: the alignment check inspects the pointer
value only and never dereferences it.

## Extending: the `BaseBuffer` contract

An owning buffer deriving from `BaseBuffer<T, K, Derived, P_alloc, P_free>` must:

1. provide `static void allocate(T**, std::size_t, P_alloc&, std::source_location)` -- static,
   because it runs from a `BaseBuffer` constructor, before the derived object exists;
2. provide `void deallocate(T*, std::size_t)`;
3. call `destroy_()` from its own destructor.

Point 3 is what lets `deallocate()` see derived state such as `DeviceBufferWrapper`'s stream. A
base destructor cannot call it: the derived sub-object is already gone by then. `~BaseBuffer`
reports through `P_free` if a derived class fails to do this, rather than leaking silently.

Views pass `IsView = true` as the trailing template argument of `BaseBuffer`. The flag is a
template parameter rather than a member read back out of `Derived`, because the
constrained/deleted copy-constructor pair is resolved while `Derived` is still incomplete.

## Memory Kind Enumeration

```cpp
enum class MemoryKind { Device, Pinned, Host, Unified };
```

Each buffer type exposes `static constexpr MemoryKind memory_kind` and the boolean flags `is_device`, `is_host`, and `is_gpu_managed`.

## Dependencies

| Dependency | Purpose |
|---|---|
| `wwr.extension.common` | Error policy concepts, `gpu_check`, `DefaultErrorPolicy` |
| `wwr.extension.runtime` | `gpu_check` overloads for `wwrError_t` |
| `wwr.runtime_api` | gpu* memory allocation APIs (CUDA or HIP runtime) |
