# wwr.extension.runtime

C++23 module providing RAII wrappers for the GPU runtime API's core objects. Backend-neutral: written against `wwr.runtime_api`'s `wwr*` names, so the same source builds for the CUDA and the HIP backend (see `src/README.md`).

## Module Name

`wwr.extension.runtime`

## Overview

This module exposes type-safe, RAII-managed wrappers for core GPU runtime objects and integrates them with the project-wide error policy infrastructure. All types are in the `wwr::extension` namespace.

**Borrow-safe operations are free functions.** `sync`, `wait_event`, `begin_capture` (stream), `record`, `sync` (event), and `launch`, `upload` (executable graph) are free functions taking the raw handle (`wwrStream_t`/`wwrEvent_t`/`wwrGraphExec_t`). An owning wrapper and its view both convert to that handle, so one definition serves the owner, its view, and a bare handle alike (found by ADL on the wrapper/view types). Operations that *produce* an owned handle — `end_capture` (stream), `instantiate` (graph) — stay members, since they need the wrapper's error policy. A handle's `view()` (from `BaseHandle`/`DeviceBoundHandle`) returns its non-owning, copyable, trivially-destructible view.

**The module ships the wrappers, not error policies for them.** Each `*Wrapper` takes its create and destroy error policies as explicit template arguments — neither has a default, so every use names both; the device-bound wrappers (stream, event, mem pool) take a third, the device-access policy. The error-handling *core* forces no policy; `wwr::extension::kit::AbortPolicy` is the kit's opt-in one, or bring your own. Binding is a one-line `using` a consumer writes once, for exactly the names it uses (`using Stream = StreamWrapper<kit::AbortPolicy<wwrError_t>, kit::AbortPolicy<wwrError_t>, kit::AbortPolicy<wwrError_t>>;`) — see the [Usage](#usage) block and `example/warp_reduce`.

## Partitions

| Partition | Description |
|-----------|-------------|
| `:stream` | RAII wrapper for `wwrStream_t` |
| `:event` | RAII wrapper for `wwrEvent_t` |
| `:mem_pool` | RAII wrapper for `wwrMemPool_t` |
| `:graph` | RAII wrapper for `wwrGraph_t` |
| `:graph_exec` | RAII wrapper for `wwrGraphExec_t` |
| `:device_handle` | `kit::DeviceHandle`, a ready-made pool-tier device handle (the stream tier is just `StreamWrapper`) |

## Exported Types and Functions

### Stream (`stream`)

```cpp
template<error_policy<wwrError_t> P_create,
         nothrow_error_policy<wwrError_t> P_destroy>
class StreamWrapper;
```

Constructors:
- `(int dev_idx = 0)` — default; creates a non-blocking stream (`wwrStreamCreateWithFlags` with `wwrStreamNonBlocking`) on `dev_idx`, so it does not serialize against the legacy default stream (0)
- `(int dev_idx, unsigned int flags)` — creates with `wwrStreamCreateWithFlags`
- `(int dev_idx, unsigned int flags, int priority)` — creates with `wwrStreamCreateWithPriority`

Destruction calls `wwrStreamDestroy`. Supports move semantics; copy is deleted.

Graph capture: the free function `begin_capture(stream, mode = wwrStreamCaptureModeGlobal)` starts recording work submitted to the stream, and the owner's `end_capture()` member ends it and returns a `GraphWrapper<P_create, P_destroy>` (the stream's own policies) owning the captured graph (via `GraphWrapper::adopt`), so the whole `begin_capture → end_capture → instantiate → launch` flow stays RAII. `end_capture` stays a member because it mints an owned graph through the create policy.

### Event (`event`)

```cpp
template<error_policy<wwrError_t> P_create,
         nothrow_error_policy<wwrError_t> P_destroy>
class EventWrapper;
```

Constructors:
- Default — creates a timing-disabled event (`wwrEventCreateWithFlags` with
  `wwrEventDisableTiming`)
- `(int dev_idx, unsigned int flags)` — creates with `wwrEventCreateWithFlags`
  and the given flags (pass `0` for a timing-capable event)

Destruction calls `wwrEventDestroy`.

### Memory pool (`mem_pool`)

```cpp
template<error_policy<wwrError_t> P_create,
         nothrow_error_policy<wwrError_t> P_destroy>
class MemPoolWrapper;
```

Constructors:
- `(int dev_idx = 0)` — default; creates a pool with pinned allocation on `dev_idx`; sets `wwrMemPoolAttrReleaseThreshold` to 1 GB
- `(int dev_idx, unsigned int release_threshold)` — default properties on `dev_idx` with a caller-supplied threshold
- `(const wwrMemPoolProps& props, unsigned int release_threshold = 1GB)` — creates with caller-supplied properties (`props.location.id` names the device)

Destruction calls `wwrMemPoolDestroy`.

### Graph (`graph`)

```cpp
template<error_policy<wwrError_t> P_create,
         nothrow_error_policy<wwrError_t> P_destroy>
class GraphWrapper;
```

Constructors:
- Default — creates an empty graph with `wwrGraphCreate`

`instantiate(unsigned long long flags = 0)` returns a `GraphExecWrapper<P_create, P_destroy>` (the graph's own policies) for this graph. Destruction calls `wwrGraphDestroy`.

### Executable graph (`graph_exec`)

```cpp
template<error_policy<wwrError_t> P_create,
         nothrow_error_policy<wwrError_t> P_destroy>
class GraphExecWrapper;
```

Constructors:
- `(wwrGraph_t graph, unsigned long long flags = 0)` — instantiates `graph` with `wwrGraphInstantiate`

The free functions `launch(exec, stream)` and `upload(exec, stream)` run the graph and upload it without launching, respectively. Destruction calls `wwrGraphExecDestroy`.

> Neither graph type is device-bound — both sit on `BaseHandle`, not `DeviceBoundHandle`. A graph describes work whose nodes may target different devices, and an executable graph runs on whatever device the stream passed to `launch()` belongs to, so there is no owning device to record.
>
> `wwrGraphInstantiate` is a hand-written forwarding function in `wwr.runtime_api`, not a plain alias: the backends' plain `*Instantiate` entry points disagree on signature beyond the prefix (CUDA takes flags, HIP takes an error-node/log-buffer triple), so `wwrGraphInstantiate(exec, graph, flags = 0)` forwards to `cudaGraphInstantiate` on CUDA and `hipGraphInstantiateWithFlags` on HIP — both of which take `(GraphExec_t*, Graph_t, unsigned long long)`.

### Device handle (`kit::DeviceHandle`)

```cpp
namespace wwr::extension::kit {
template<error_policy<wwrError_t> P_create,
         nothrow_error_policy<wwrError_t> P_destroy,
         error_policy<wwrError_t> P_device>
class DeviceHandle;
}
```

A ready-made backing for a `DeviceBuffer` (or a stream-bound library handle): one device's index, properties, an owned `StreamWrapper` and an owned `MemPoolWrapper`, exposed as `dev_idx()` / `props()` / `stream()` / `pool()`. It is the fullest rung of the `device_handle` ladder (`device_handle_pool`), so a buffer built on it draws from that pool on that stream. Move-only; constructed from a device index (`DeviceHandle(int index = 0)`). It lives in the opt-in namespace `wwr::extension::kit` (with the buffer suite), out of `wwr::extension` so a plain `using namespace wwr::extension;` does not pull the generic name `DeviceHandle` into scope.

> The **stream tier needs no type here** — a `StreamWrapper` already exposes `dev_idx()` and `stream()`, so it *is* a `device_handle_stream` and backs a buffer directly. `DeviceHandle` is for the pool tier.
>
> This is the handle that lived here until [#93](https://github.com/whatDiracIsCooking/WarpWraps/pull/93) moved it to `test/shared` to keep a forced `AbortPolicy` out of the library. It is back because it is now **policy-templated** — it bakes no policy, so the reason it left no longer applies.

## Usage

```cpp
import wwr.extension.runtime;
using namespace wwr::extension;

// Bind the wrappers to an error policy — your names, defined once. kit::AbortPolicy
// is the kit's opt-in abort-on-failure policy; the core forces none, so bring your
// own if you prefer.
using Abort = kit::AbortPolicy<wwrError_t>;
// Device-bound wrappers take a third policy: device access (wwrSetDevice/wwrGetDevice).
using Stream = StreamWrapper<Abort, Abort, Abort>;
using Event = EventWrapper<Abort, Abort, Abort>;
using MemPool = MemPoolWrapper<Abort, Abort, Abort>;
// Graphs and graph execs are not device-bound: two policies only.
using Graph = GraphWrapper<Abort, Abort>;
using GraphExec = GraphExecWrapper<Abort, Abort>;

Stream stream;
Event  event;
MemPool pool;

Graph graph;                     // empty graph
// ... add nodes / stream-capture into `graph` ...
GraphExec exec = graph.instantiate();
exec.launch(stream);

auto ok = success_code<wwrError_t>();  // wwrSuccess
```

## Dependencies

- `wwr.runtime_api` (backend-neutral runtime API names)
- `wwr.extension.common` (error policy infrastructure)
