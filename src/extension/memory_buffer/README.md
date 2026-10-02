# wwr.extension.memory_buffer

RAII-based memory buffer management for all GPU-relevant memory kinds. Provides strongly typed, move-only buffer wrappers with pluggable error policies, a unified copy API, and C++20 concepts for generic buffer programming.

## Module

`wwr.extension.memory_buffer`

## Buffer Types

The module ships the `*Wrapper` classes (in the `wwr::extension` namespace); each takes its `P_alloc`/`P_free` error policies as explicit template arguments — neither has a default, so every use names both. It ships **no** error policy either: `AbortPolicy` below is the consumer's own abort-on-failure policy (see `example/warp_reduce`). Binding a wrapper to a policy is a one-line `using` a consumer writes once, for the names it uses:

```cpp
// DeviceBufferWrapper and UnifiedBufferWrapper are both device-bound: parameter order is
// T, P_alloc, P_free, P_device_access, H — the device-access policy precedes the handle.
template<typename T> using DeviceBuffer  = DeviceBufferWrapper<T, AbortPolicy<wwrError_t>, AbortPolicy<wwrError_t>, AbortPolicy<wwrError_t>, MyDeviceHandle>;
template<typename T> using UnifiedBuffer = UnifiedBufferWrapper<T, AbortPolicy<wwrError_t>, AbortPolicy<wwrError_t>, AbortPolicy<wwrError_t>, MyDeviceHandle>;
template<typename T> using PinnedBuffer  = PinnedBufferWrapper<T, AbortPolicy<wwrError_t>, AbortPolicy<wwrError_t>>;
template<typename T> using HostBuffer    = HostBufferWrapper<T, AbortPolicy<stdHostMemoryError_t>, AbortPolicy<stdHostMemoryError_t>>;
```

`DeviceBufferWrapper` and `UnifiedBufferWrapper` take two arguments the host and
pinned kinds don't: because they are device-bound, the device-access error policy
(fourth) and the handle type backing them (fifth). This layer ships **no**
concrete handle — `MyDeviceHandle` above is the consumer's own, any type
satisfying the `device_handle` ladder (see `handle/device_handle.cppm` and the one
`example/warp_reduce/main.cpp` defines). The difference between them is what the
handle's *tier* buys: a device buffer picks its alloc/free calls from it, while a
unified buffer reads only `dev_idx()` — managed memory has one allocation call and
a sync-only free, so a stream or pool on the handle goes unused.

Writing this prelude by hand is what the **suite** below removes: it binds these
aliases (and their views) from one place, so the wrapper → error-family mapping
and the buffer/view policy pairing live in the library. The raw wrappers stay
exactly as documented here; the suite is an addition, not a replacement.

The examples below use those names (see also `example/warp_reduce`). Each wrapper class:

| Wrapper | Memory kind | Allocation API | Host-accessible |
|---|---|---|---|
| `DeviceBufferWrapper<T, …, H>` | GPU device memory | pool / async / sync, by the caller-supplied handle's tier | No |
| `UnifiedBufferWrapper<T, …, H>` | Unified (managed) memory | `wwrMallocManaged` (handle-backed, device-bound; always sync `wwrFree`) | Yes |
| `PinnedBufferWrapper<T>` | Page-locked host memory | `wwrHostAlloc` (`wwrHostAllocDefault` unless flags are given) | Yes |
| `HostBufferWrapper<T>` | Standard host memory | `std::malloc` | Yes |

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

**Default (host and pinned only):** `Buffer<T> buf;` — empty buffer (`nullptr`, 0 elements).

**Synchronous allocation (host and pinned only):** `Buffer<T> buf(n);` — allocates `n` elements.

**With error policy (host and pinned only):** `Buffer<T> buf(n, policy);` or `Buffer<T> buf(n, policy_alloc, policy_free);`

**From a `std::vector` (`HostBuffer` and `PinnedBuffer` only):** `Buffer<T> buf(values);` — sizes the buffer to `values.size()` and copies the bytes in with `std::memcpy`. The same `policy` / `policy_alloc, policy_free` overloads are offered. Restricted to the host-reachable non-device kinds (`!is_device`): the copy writes through an ordinary host pointer, so `UnifiedBuffer` (host-reachable but `is_device`) and `DeviceBuffer` are excluded, as are views.

**Device and unified buffers are drawn from a shared handle** (`MyDeviceHandle` is the consumer's own — see above):
```cpp
auto device = std::make_shared<MyDeviceHandle>(0);
DeviceBuffer<T>  buf(n, device);          // strategy follows the handle's tier
DeviceBuffer<T>  buf(n, device, policy);  // ...with a custom error policy
UnifiedBuffer<T> buf(n, device);          // managed memory, device-bound by the handle
UnifiedBuffer<T> buf(n, device, policy);  // ...with a custom error policy
```

A `DeviceBuffer` selects the handle's device with `wwrSetDevice`, allocates `n` elements, and
zero-initialises them, keeping a `shared_ptr` to the handle so whatever backs the allocation
outlives the buffer's free. *How* it allocates is chosen at compile time by the handle's tier
on the `device_handle` capability ladder (see `handle/device_handle.cppm`):

| Handle tier (concept it satisfies) | allocate | free |
|---|---|---|
| `device_handle_pool` | `wwrMallocFromPoolAsync` from `pool()` on `stream()` | `wwrFreeAsync` on `stream()` |
| `device_handle_stream` | `wwrMallocAsync` on `stream()` | `wwrFreeAsync` on `stream()` |
| `device_handle` (`dev_idx()` only) | synchronous `wwrMalloc` | synchronous `wwrFree` |

The two stream-bearing tiers release with `wwrFreeAsync` on the same stream the block was drawn
on; `wwrFree` there would perform no implicit synchronisation for a stream-ordered pointer, so
using it would return the block while work still queued on the stream was reading and writing
it. The bare tier has only `wwrFree`, whose implicit device synchronisation is what makes it
safe — and what makes that tier's destructor block the host, so prefer a stream-bearing handle
in hot alloc/free paths.

A `UnifiedBuffer` is handle-backed the same way — it makes the handle's device current with a
`ScopedDeviceIndex` guard and retains the `shared_ptr` — but its tier does **not** pick the calls.
Managed memory has a single allocation primitive (`wwrMallocManaged`, synchronous, no stream) and
can only be released with the synchronous `wwrFree`: `wwrFreeAsync` rejects a managed pointer
(`cudaErrorNotSupported`). So the free is always `wwrFree`, whatever the handle offers; a
stream/pool-tier handle is accepted but its stream and pool go unused. `wwrFree`'s implicit device
synchronisation means the destructor blocks the host — there being no async alternative for managed
memory.

**Pinned-specific with flags:**
```cpp
PinnedBuffer<T> buf(n, wwrHostAllocMapped);  // wwrHostAlloc with flags
```

**Unified-specific with flags:**
```cpp
UnifiedBuffer<T> buf(n, device, wwrMemAttachHost);  // wwrMallocManaged with flags, on the handle
```

## Error Handling

Error policies are parameterised via `P_alloc` and `P_free` template arguments; there is no default, so every use names them (the core forces no policy; `wwr::extension::kit::AbortPolicy` is a ready one, or bring your own). The error type varies by memory kind:

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
`T` and `K`. A consumer names one per kind with the same alias pattern —
`template<typename T> using HostBufferView = BufferViewWrapper<T, MemoryKind::Host, AbortPolicy<stdHostMemoryError_t>, AbortPolicy<stdHostMemoryError_t>>;` and the
`Device` / `Pinned` / `Unified` equivalents — and the examples below use those names.

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
`reinterpret_buffer_view<T2>` factory:

```cpp
HostBuffer<float> buf(16);                            // 64 bytes
auto bytes = reinterpret_buffer_view<std::byte>(buf); // HostBufferView<std::byte>, 64 elements
auto ints  = reinterpret_buffer_view<int>(buf);       // HostBufferView<int>, 16 elements
```

The factory is the only public door. A `reinterpret_view` tag constructor backs
it, but that tag lives in an in-module partition (`:reinterpret_tag`) that the
module does not re-export, so it cannot be named from outside.

The byte span is preserved: `num_elements()` becomes `src.size_bytes() / sizeof(T2)`, and the
source's memory kind and error policies are carried onto the view. Reinterpretation is reported
through `P_alloc` and leaves the view empty when the source's byte size is not a whole multiple
of `sizeof(T2)`, or when its base pointer is not aligned for `T2` — the latter is reachable for
a sub-view, whose offset into an allocation aligned for one type need not be aligned for a
wider one. A device buffer can be reinterpreted too: the alignment check inspects the pointer
value only and never dereferences it.

## The suite: binding a whole prelude at once

The `:suite` partition emits the buffer *and* view aliases from one place, so a
consumer names a policy and (for device buffers) a handle instead of writing one
alias per kind plus a matching view alias for each. These live in the opt-in
namespace `wwr::extension::kit` (not `wwr::extension` directly), so a plain
`using namespace wwr::extension;` does not pull generic names like `buffers` into
scope — opt in with `using namespace wwr::extension::kit;` or name `kit::`. Two
entry points over one mechanism.

**Single policy for every kind — the convenience.** `kit::buffers<P>` /
`kit::device_buffers<P, H>` take one policy template and use it across all kinds:

```cpp
using Buf = kit::device_buffers<kit::AbortPolicy, MyDeviceHandle>;   // kit::AbortPolicy is template<class E>

Buf::device<float>    d(1024, dev);
Buf::unified<float>   u(1024, dev);  // handle-backed too (reads only dev_idx())
Buf::host<float>      h(1024);       // stdHostMemoryError_t, chosen by the suite
Buf::host_view<float> v(h, 4, 8);    // policies follow Buf::host, cannot drift
```

**Different policies per error family — the map.** When the host and GPU families
want different policies, bring a map with `alloc` / `free` keys:

```cpp
struct MyPolicies {
  template<typename E> using alloc = AbortPolicy<E>;
  template<typename E> using free  = LogPolicy<E>;
};
using Buf = device_buffer_suite<MyPolicies, MyDeviceHandle>;
```

`buffer_suite<M>` / `buffers<P>` stop at host/pinned (plus views); the `device_*`
forms add the handle-backed kinds — `device` / `unified` and their views — and take
the handle. A host-only consumer therefore needs no handle, and
`buffer_suite<M>::device` / `::unified` simply do not exist. The suite ships **no**
policy — the consumer still brings one — and the raw wrappers stay usable
underneath; the suite is an addition, not a replacement.

Four things trip people up:

- **The policy is a *template* on the error type** — `template<class E> class P`,
  not a concrete `AbortPolicy<wwrError_t>`. The suite instantiates it per kind
  with that kind's error type.
- **The map's keys are `alloc` and `free`** — buffer vocabulary. The map never
  names a handle; `H` arrives already built. The device and unified buffers' third
  policy (`P_device_access`) is bound by the suite from the `free` policy, so the
  two-key map suffices.
- **In a dependent context, spell it `typename Suite::template device<float>`.**
  When `Suite` is a template parameter both `typename` and `template` are
  required; in non-dependent code (a concrete `using Buf = …`) neither is.
- **`copy` / `memset` cross suite boundaries; views do not.** `copy` and `memset`
  constrain on `same_value_type`, which checks only `value_type`, so a buffer from
  one suite copies into a buffer from another. A *view* carries its buffer's kind
  and policies, so `Buf::host_view<T>` only views a `Buf::host<T>`; a view over a
  differently-policied buffer is a different type.

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
| `wwr.extension.common` | Error policy concepts, `gpu_check` |
| `wwr.extension.runtime` | `gpu_check` overloads for `wwrError_t` |
| `wwr.runtime_api` | wwr* memory allocation APIs (CUDA or HIP runtime) |
