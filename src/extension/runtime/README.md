# gpumod.extension.runtime

C++23 module providing RAII wrappers and error handling utilities for the GPU runtime API. Backend-neutral: written against `gpumod.runtime_api`'s `gpu*` names, so the same source builds for the CUDA and the HIP backend (see `src/README.md`).

## Module Name

`gpumod.extension.runtime`

## Overview

This module exposes type-safe, RAII-managed wrappers for core GPU runtime objects and integrates them with the project-wide error policy infrastructure. All types are in the `gpumod::extension` namespace.

## Partitions

| Partition | Description |
|-----------|-------------|
| `:gpu_error` | `gpuError_t` specializations for the common error handling templates |
| `:gpu_stream` | RAII wrapper for `gpuStream_t` |
| `:gpu_event` | RAII wrapper for `gpuEvent_t` |
| `:gpu_mem_pool` | RAII wrapper for `gpuMemPool_t` |
| `:gpu_graph` | RAII wrapper for `gpuGraph_t` |
| `:gpu_graph_exec` | RAII wrapper for `gpuGraphExec_t` |
| `:device_handle` | device identity, properties and default allocation stream |

## Exported Types and Functions

### Error handling (`gpu_error`)

Specializes three function templates from `gpumod.extension.common` for `gpuError_t`:

- `success_code<gpuError_t>()` — returns `gpuSuccess`
- `error_name<gpuError_t>(error)` — delegates to `gpuGetErrorName`
- `error_string<gpuError_t>(error)` — delegates to `gpuGetErrorString`

Also explicitly instantiates `DefaultErrorPolicy<gpuError_t>` and both overloads of `gpu_check<gpuError_t>`.

### Stream (`gpu_stream`)

```cpp
template<error_policy<gpuError_t> P_create = DefaultErrorPolicy<gpuError_t>,
         error_policy<gpuError_t> P_destroy = P_create>
class GpuStreamWrapper;

using GpuStream = GpuStreamWrapper<>;
```

Constructors:
- Default — creates a non-blocking stream (`gpuStreamCreateWithFlags` with `gpuStreamNonBlocking`), so it does not serialize against the legacy default stream (0)
- `(unsigned int flags)` — creates with `gpuStreamCreateWithFlags`
- `(unsigned int flags, int priority)` — creates with `gpuStreamCreateWithPriority`

Destruction calls `gpuStreamDestroy`. Supports move semantics; copy is deleted.

Graph capture: `begin_capture(mode = gpuStreamCaptureModeGlobal)` starts recording work submitted to the stream, and `end_capture()` ends it and returns a `GpuGraph` owning the captured graph (via `GpuGraph::adopt`), so the whole `begin_capture → end_capture → instantiate → launch` flow stays RAII.

### Event (`gpu_event`)

```cpp
template<error_policy<gpuError_t> P_create = DefaultErrorPolicy<gpuError_t>,
         error_policy<gpuError_t> P_destroy = P_create>
class GpuEventWrapper;

using GpuEvent = GpuEventWrapper<>;
```

Constructors:
- Default — creates an event with `gpuEventCreate`
- `(unsigned int flags)` — creates with `gpuEventCreateWithFlags`

Destruction calls `gpuEventDestroy`.

### Memory pool (`gpu_mem_pool`)

```cpp
template<error_policy<gpuError_t> P_create = DefaultErrorPolicy<gpuError_t>,
         error_policy<gpuError_t> P_destroy = P_create>
class GpuMemPoolWrapper;

using GpuMemPool = GpuMemPoolWrapper<>;
```

Constructors:
- Default — creates a pool with pinned allocation on the current device; sets `gpuMemPoolAttrReleaseThreshold` to 1 GB
- `(const gpuMemPoolProps& props, unsigned int release_threshold = 1GB)` — creates with caller-supplied properties

Destruction calls `gpuMemPoolDestroy`.

### Graph (`gpu_graph`)

```cpp
template<error_policy<gpuError_t> P_create = DefaultErrorPolicy<gpuError_t>,
         error_policy<gpuError_t> P_destroy = P_create>
class GpuGraphWrapper;

using GpuGraph = GpuGraphWrapper<>;
```

Constructors:
- Default — creates an empty graph with `gpuGraphCreate`

`instantiate(unsigned long long flags = 0)` returns a `GpuGraphExec` for this graph. Destruction calls `gpuGraphDestroy`.

### Executable graph (`gpu_graph_exec`)

```cpp
template<error_policy<gpuError_t> P_create = DefaultErrorPolicy<gpuError_t>,
         error_policy<gpuError_t> P_destroy = P_create>
class GpuGraphExecWrapper;

using GpuGraphExec = GpuGraphExecWrapper<>;
```

Constructors:
- `(gpuGraph_t graph, unsigned long long flags = 0)` — instantiates `graph` with `gpuGraphInstantiate`

`launch(gpuStream_t)` runs the graph; `upload(gpuStream_t)` uploads it without launching. Destruction calls `gpuGraphExecDestroy`.

> Neither graph type is device-bound — both sit on `BaseGpuHandle`, not `GpuBoundHandle`. A graph describes work whose nodes may target different devices, and an executable graph runs on whatever device the stream passed to `launch()` belongs to, so there is no owning device to record.
>
> `gpuGraphInstantiate` is a hand-written forwarding function in `gpumod.runtime_api`, not a plain alias: the backends' plain `*Instantiate` entry points disagree on signature beyond the prefix (CUDA takes flags, HIP takes an error-node/log-buffer triple), so `gpuGraphInstantiate(exec, graph, flags = 0)` forwards to `cudaGraphInstantiate` on CUDA and `hipGraphInstantiateWithFlags` on HIP — both of which take `(GraphExec_t*, Graph_t, unsigned long long)`.

### Device handle (`device_handle`)

```cpp
class DeviceHandle;   // move-only
```

Constructors:
- `(int index = 0)` — queries `index`'s properties once (`gpuGetDeviceProperties`) and eagerly creates a `GpuStream` and a `GpuMemPool` on that device

`activate()` makes this device current (`gpuSetDevice`) without restoring. `index()` returns the device index, `props()` returns the full `gpuDeviceProp` (`cudaDeviceProp` / `hipDeviceProp_t`) held directly — individual fields are not mirrored behind their own accessors — `alloc_stream()` returns the owned default allocation stream (usable anywhere a `gpuStream_t` is), and `mem_pool()` returns the owned default memory pool. A `DeviceBuffer` is built from a `std::shared_ptr<DeviceHandle>` and draws from that pool on that stream. Move-only, because it owns the stream and pool; a bad index or driver failure aborts through the default error policy.

## Usage

```cpp
import gpumod.extension.runtime;
using namespace gpumod::extension;

GpuStream stream;
GpuEvent  event;
GpuMemPool pool;

GpuGraph graph;                     // empty graph
// ... add nodes / stream-capture into `graph` ...
GpuGraphExec exec = graph.instantiate();
exec.launch(stream);

auto ok = success_code<gpuError_t>();  // gpuSuccess
```

## Dependencies

- `gpumod.runtime_api` (backend-neutral runtime API names)
- `gpumod.extension.common` (error policy infrastructure)
