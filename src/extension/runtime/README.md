# wwr.extension.runtime

C++23 module providing RAII wrappers and error handling utilities for the GPU runtime API. Backend-neutral: written against `wwr.runtime_api`'s `gpu*` names, so the same source builds for the CUDA and the HIP backend (see `src/README.md`).

## Module Name

`wwr.extension.runtime`

## Overview

This module exposes type-safe, RAII-managed wrappers for core GPU runtime objects and integrates them with the project-wide error policy infrastructure. All types are in the `wwr::extension` namespace.

**Borrow-safe operations are free functions.** `sync`, `wait_event`, `begin_capture` (stream), `record`, `sync` (event), and `launch`, `upload` (executable graph) are free functions taking the raw handle (`wwrStream_t`/`wwrEvent_t`/`wwrGraphExec_t`). An owning wrapper and its view both convert to that handle, so one definition serves the owner, its view, and a bare handle alike (found by ADL on the wrapper/view types). Operations that *produce* an owned handle — `end_capture` (stream), `instantiate` (graph) — stay members, since they need the wrapper's error policy. A handle's `view()` (from `BaseHandle`/`DeviceBoundHandle`) returns its non-owning, copyable, trivially-destructible view.

## Partitions

| Partition | Description |
|-----------|-------------|
| `:gpu_error` | `wwrError_t` specializations for the common error handling templates |
| `:gpu_stream` | RAII wrapper for `wwrStream_t` |
| `:gpu_event` | RAII wrapper for `wwrEvent_t` |
| `:gpu_mem_pool` | RAII wrapper for `wwrMemPool_t` |
| `:gpu_graph` | RAII wrapper for `wwrGraph_t` |
| `:gpu_graph_exec` | RAII wrapper for `wwrGraphExec_t` |
| `:device_handle` | device identity, properties and default allocation stream |

## Exported Types and Functions

### Error handling (`gpu_error`)

Specializes three function templates from `wwr.extension.common` for `wwrError_t`:

- `success_code<wwrError_t>()` — returns `wwrSuccess`
- `error_name<wwrError_t>(error)` — delegates to `wwrGetErrorName`
- `error_string<wwrError_t>(error)` — delegates to `wwrGetErrorString`

Also explicitly instantiates `DefaultErrorPolicy<wwrError_t>` and both overloads of `gpu_check<wwrError_t>`.

### Stream (`gpu_stream`)

```cpp
template<error_policy<wwrError_t> P_create = DefaultErrorPolicy<wwrError_t>,
         nothrow_error_policy<wwrError_t> P_destroy = P_create>
class GpuStreamWrapper;

using GpuStream = GpuStreamWrapper<>;
```

Constructors:
- Default — creates a non-blocking stream (`wwrStreamCreateWithFlags` with `wwrStreamNonBlocking`), so it does not serialize against the legacy default stream (0)
- `(unsigned int flags)` — creates with `wwrStreamCreateWithFlags`
- `(unsigned int flags, int priority)` — creates with `wwrStreamCreateWithPriority`

Destruction calls `wwrStreamDestroy`. Supports move semantics; copy is deleted.

Graph capture: `begin_capture(mode = wwrStreamCaptureModeGlobal)` starts recording work submitted to the stream, and `end_capture()` ends it and returns a `GpuGraph` owning the captured graph (via `GpuGraph::adopt`), so the whole `begin_capture → end_capture → instantiate → launch` flow stays RAII.

### Event (`gpu_event`)

```cpp
template<error_policy<wwrError_t> P_create = DefaultErrorPolicy<wwrError_t>,
         nothrow_error_policy<wwrError_t> P_destroy = P_create>
class GpuEventWrapper;

using GpuEvent = GpuEventWrapper<>;
```

Constructors:
- Default — creates an event with `wwrEventCreate`
- `(unsigned int flags)` — creates with `wwrEventCreateWithFlags`

Destruction calls `wwrEventDestroy`.

### Memory pool (`gpu_mem_pool`)

```cpp
template<error_policy<wwrError_t> P_create = DefaultErrorPolicy<wwrError_t>,
         nothrow_error_policy<wwrError_t> P_destroy = P_create>
class GpuMemPoolWrapper;

using GpuMemPool = GpuMemPoolWrapper<>;
```

Constructors:
- Default — creates a pool with pinned allocation on the current device; sets `wwrMemPoolAttrReleaseThreshold` to 1 GB
- `(const wwrMemPoolProps& props, unsigned int release_threshold = 1GB)` — creates with caller-supplied properties

Destruction calls `wwrMemPoolDestroy`.

### Graph (`gpu_graph`)

```cpp
template<error_policy<wwrError_t> P_create = DefaultErrorPolicy<wwrError_t>,
         nothrow_error_policy<wwrError_t> P_destroy = P_create>
class GpuGraphWrapper;

using GpuGraph = GpuGraphWrapper<>;
```

Constructors:
- Default — creates an empty graph with `wwrGraphCreate`

`instantiate(unsigned long long flags = 0)` returns a `GpuGraphExec` for this graph. Destruction calls `wwrGraphDestroy`.

### Executable graph (`gpu_graph_exec`)

```cpp
template<error_policy<wwrError_t> P_create = DefaultErrorPolicy<wwrError_t>,
         nothrow_error_policy<wwrError_t> P_destroy = P_create>
class GpuGraphExecWrapper;

using GpuGraphExec = GpuGraphExecWrapper<>;
```

Constructors:
- `(wwrGraph_t graph, unsigned long long flags = 0)` — instantiates `graph` with `wwrGraphInstantiate`

`launch(wwrStream_t)` runs the graph; `upload(wwrStream_t)` uploads it without launching. Destruction calls `wwrGraphExecDestroy`.

> Neither graph type is device-bound — both sit on `BaseHandle`, not `DeviceBoundHandle`. A graph describes work whose nodes may target different devices, and an executable graph runs on whatever device the stream passed to `launch()` belongs to, so there is no owning device to record.
>
> `wwrGraphInstantiate` is a hand-written forwarding function in `wwr.runtime_api`, not a plain alias: the backends' plain `*Instantiate` entry points disagree on signature beyond the prefix (CUDA takes flags, HIP takes an error-node/log-buffer triple), so `wwrGraphInstantiate(exec, graph, flags = 0)` forwards to `cudaGraphInstantiate` on CUDA and `hipGraphInstantiateWithFlags` on HIP — both of which take `(GraphExec_t*, Graph_t, unsigned long long)`.

### Device handle (`device_handle`)

```cpp
class DeviceHandle;   // move-only
```

Constructors:
- `(int index = 0)` — queries `index`'s properties once (`wwrGetDeviceProperties`) and eagerly creates a `GpuStream` and a `GpuMemPool` on that device

`activate()` makes this device current (`wwrSetDevice`) without restoring. `index()` returns the device index, `props()` returns the full `wwrDeviceProp` (`cudaDeviceProp` / `hipDeviceProp_t`) held directly — individual fields are not mirrored behind their own accessors — `alloc_stream()` returns the owned default allocation stream (usable anywhere a `wwrStream_t` is), and `mem_pool()` returns the owned default memory pool. A `DeviceBuffer` is built from a `std::shared_ptr<DeviceHandle>` and draws from that pool on that stream. Move-only, because it owns the stream and pool; a bad index or driver failure aborts through the default error policy.

## Usage

```cpp
import wwr.extension.runtime;
using namespace wwr::extension;

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
