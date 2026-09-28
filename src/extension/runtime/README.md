# wwr.extension.runtime

C++23 module providing RAII wrappers for the GPU runtime API's core objects. Backend-neutral: written against `wwr.runtime_api`'s `gpu*` names, so the same source builds for the CUDA and the HIP backend (see `src/README.md`).

## Module Name

`wwr.extension.runtime`

## Overview

This module exposes type-safe, RAII-managed wrappers for core GPU runtime objects and integrates them with the project-wide error policy infrastructure. All types are in the `wwr::extension` namespace.

**Borrow-safe operations are free functions.** `sync`, `wait_event`, `begin_capture` (stream), `record`, `sync` (event), and `launch`, `upload` (executable graph) are free functions taking the raw handle (`wwrStream_t`/`wwrEvent_t`/`wwrGraphExec_t`). An owning wrapper and its view both convert to that handle, so one definition serves the owner, its view, and a bare handle alike (found by ADL on the wrapper/view types). Operations that *produce* an owned handle — `end_capture` (stream), `instantiate` (graph) — stay members, since they need the wrapper's error policy. A handle's `view()` (from `BaseHandle`/`DeviceBoundHandle`) returns its non-owning, copyable, trivially-destructible view.

**The module ships the wrappers, not default-policy aliases for them.** Each `*Wrapper` takes its create and destroy error policies as explicit template arguments — neither has a default, so every use names both. Binding them (e.g. to `AbortPolicy`) is a one-line `using` a consumer writes once, for exactly the names it uses (`using GpuStream = GpuStreamWrapper<AbortPolicy<wwrError_t>, AbortPolicy<wwrError_t>>;`) — see the [Usage](#usage) block and `example/warp_reduce`.

## Partitions

| Partition | Description |
|-----------|-------------|
| `:gpu_stream` | RAII wrapper for `wwrStream_t` |
| `:gpu_event` | RAII wrapper for `wwrEvent_t` |
| `:gpu_mem_pool` | RAII wrapper for `wwrMemPool_t` |
| `:gpu_graph` | RAII wrapper for `wwrGraph_t` |
| `:gpu_graph_exec` | RAII wrapper for `wwrGraphExec_t` |
| `:stream_event_pair` | Bundles a `wwrStream_t` and a `wwrEvent_t` (`StreamEventPair`) |
| `:device_handle` | device identity, properties, default allocation stream and memory pool |

## Exported Types and Functions

### Stream (`gpu_stream`)

```cpp
template<error_policy<wwrError_t> P_create,
         nothrow_error_policy<wwrError_t> P_destroy>
class GpuStreamWrapper;
```

Constructors:
- `(int dev_idx = 0)` — default; creates a non-blocking stream (`wwrStreamCreateWithFlags` with `wwrStreamNonBlocking`) on `dev_idx`, so it does not serialize against the legacy default stream (0)
- `(int dev_idx, unsigned int flags)` — creates with `wwrStreamCreateWithFlags`
- `(int dev_idx, unsigned int flags, int priority)` — creates with `wwrStreamCreateWithPriority`

Destruction calls `wwrStreamDestroy`. Supports move semantics; copy is deleted.

Graph capture: the free function `begin_capture(stream, mode = wwrStreamCaptureModeGlobal)` starts recording work submitted to the stream, and the owner's `end_capture()` member ends it and returns a `GpuGraphWrapper<P_create, P_destroy>` (the stream's own policies) owning the captured graph (via `GpuGraphWrapper::adopt`), so the whole `begin_capture → end_capture → instantiate → launch` flow stays RAII. `end_capture` stays a member because it mints an owned graph through the create policy.

### Event (`gpu_event`)

```cpp
template<error_policy<wwrError_t> P_create,
         nothrow_error_policy<wwrError_t> P_destroy>
class GpuEventWrapper;
```

Constructors:
- Default — creates an event with `wwrEventCreate`
- `(unsigned int flags)` — creates with `wwrEventCreateWithFlags`

Destruction calls `wwrEventDestroy`.

### Memory pool (`gpu_mem_pool`)

```cpp
template<error_policy<wwrError_t> P_create,
         nothrow_error_policy<wwrError_t> P_destroy>
class GpuMemPoolWrapper;
```

Constructors:
- `(int dev_idx = 0)` — default; creates a pool with pinned allocation on `dev_idx`; sets `wwrMemPoolAttrReleaseThreshold` to 1 GB
- `(int dev_idx, unsigned int release_threshold)` — default properties on `dev_idx` with a caller-supplied threshold
- `(const wwrMemPoolProps& props, unsigned int release_threshold = 1GB)` — creates with caller-supplied properties (`props.location.id` names the device)

Destruction calls `wwrMemPoolDestroy`.

### Graph (`gpu_graph`)

```cpp
template<error_policy<wwrError_t> P_create,
         nothrow_error_policy<wwrError_t> P_destroy>
class GpuGraphWrapper;
```

Constructors:
- Default — creates an empty graph with `wwrGraphCreate`

`instantiate(unsigned long long flags = 0)` returns a `GpuGraphExecWrapper<P_create, P_destroy>` (the graph's own policies) for this graph. Destruction calls `wwrGraphDestroy`.

### Executable graph (`gpu_graph_exec`)

```cpp
template<error_policy<wwrError_t> P_create,
         nothrow_error_policy<wwrError_t> P_destroy>
class GpuGraphExecWrapper;
```

Constructors:
- `(wwrGraph_t graph, unsigned long long flags = 0)` — instantiates `graph` with `wwrGraphInstantiate`

The free functions `launch(exec, stream)` and `upload(exec, stream)` run the graph and upload it without launching, respectively. Destruction calls `wwrGraphExecDestroy`.

> Neither graph type is device-bound — both sit on `BaseHandle`, not `DeviceBoundHandle`. A graph describes work whose nodes may target different devices, and an executable graph runs on whatever device the stream passed to `launch()` belongs to, so there is no owning device to record.
>
> `wwrGraphInstantiate` is a hand-written forwarding function in `wwr.runtime_api`, not a plain alias: the backends' plain `*Instantiate` entry points disagree on signature beyond the prefix (CUDA takes flags, HIP takes an error-node/log-buffer triple), so `wwrGraphInstantiate(exec, graph, flags = 0)` forwards to `cudaGraphInstantiate` on CUDA and `hipGraphInstantiateWithFlags` on HIP — both of which take `(GraphExec_t*, Graph_t, unsigned long long)`.

### Stream/event pair (`stream_event_pair`)

```cpp
struct StreamEventConfig;   // { int device; optional stream_flags/stream_priority/event_flags }
class StreamEventPair;      // move-only
```

Bundles an owned stream and an owned event, both created (with `AbortPolicy`) on the device named by the `StreamEventConfig` (default device 0). Default-constructible, or from a `StreamEventConfig` for optional flags/priority. Exposes the owners via `gpu_stream()` / `gpu_event()`, the raw handles via `stream_raw()` / `event_raw()`, and convenience `record()` / `record(flags)`, `stream_sync()`, `event_sync()` wrapping the borrow-safe free functions.

### Device handle (`device_handle`)

```cpp
class DeviceHandle;   // move-only
```

Constructors:
- `(int index = 0)` — queries `index`'s properties once (`wwrGetDeviceProperties`) and eagerly creates a stream and a memory pool on that device

`dev_idx()` returns the device index, `props()` returns the full `wwrDeviceProp` (`cudaDeviceProp` / `hipDeviceProp_t`) held directly — individual fields are not mirrored behind their own accessors — `stream()` returns the owned default allocation stream (usable anywhere a `wwrStream_t` is), and `pool()` returns the owned default memory pool. `DeviceHandle` is thus the fullest tier of the `device_handle` capability ladder (`dev_idx()` + `stream()` + `pool()`), so a device buffer built from a `std::shared_ptr<DeviceHandle>` draws from that pool on that stream. Move-only, because it owns the stream and pool; a bad index or driver failure aborts through the default error policy.

## Usage

```cpp
import wwr.extension.runtime;
using namespace wwr::extension;

// Bind the wrappers to an error policy — your names, defined once.
using Abort = AbortPolicy<wwrError_t>;
using GpuStream = GpuStreamWrapper<Abort, Abort>;
using GpuEvent = GpuEventWrapper<Abort, Abort>;
using GpuMemPool = GpuMemPoolWrapper<Abort, Abort>;
using GpuGraph = GpuGraphWrapper<Abort, Abort>;
using GpuGraphExec = GpuGraphExecWrapper<Abort, Abort>;

GpuStream stream;
GpuEvent  event;
GpuMemPool pool;

GpuGraph graph;                     // empty graph
// ... add nodes / stream-capture into `graph` ...
GpuGraphExec exec = graph.instantiate();
exec.launch(stream);

auto ok = success_code<wwrError_t>();  // wwrSuccess
```

## Dependencies

- `wwr.runtime_api` (backend-neutral runtime API names)
- `wwr.extension.common` (error policy infrastructure)
