# src/thrust — the backend-neutral Thrust re-export layer

Device-includable headers that re-export Thrust (CUDA/CCCL) and rocThrust (HIP)
under one backend-neutral vocabulary in namespace `wwr::thrust`. Part of the
`wwr*` layer directly under `src/`, alongside the `wwr.*` modules: the only
place a consumer writes to reach Thrust without naming a backend.

## Why `using`, not wrappers or a module

The algorithm names are **identical on both backends** — a caller writes
`thrust::sort` either way — so each header is a plain `using`-re-export, not a
set of forwarding templates. Every overload comes across, and there is nothing
to keep in sync with Thrust's evolving overload set. The leading `::` on each
`using ::thrust::<name>` is load-bearing: inside `namespace wwr::thrust` a bare
`thrust` would name *this* namespace.

It is **not** a named module: the Thrust headers assume a device pass and a
device TU cannot `import` (`docs/architecture.md` §8). These are headers a `.cu`
(or `-x hip`) TU `#include`s and links `wwr::thrust` for.

The **one** spelling that diverges between the backends is the stream-bound
execution policy — `thrust::cuda::par.on(stream)` vs `thrust::hip::par.on(stream)`.
That is not a name, so it is not re-exported; `wwr::par_on(stream)`
(`execution_policy.cuh`) is the one-line shim over it, passed as the leading
argument:

```cpp
#include "thrust/reorder.cuh"
wwr::thrust::sort(wwr::par_on(stream), first, last);
```

## Layout

| File | Re-exports |
|---|---|
| `execution_policy.cuh` | `wwr::par_on(stream)` — the sole backend neutralization, plus the rocThrust `_VSTD` / `hipStreamDefault` toolchain bridging every HIP device TU needs |
| `reorder.cuh` | sort, unique, partition, remove, copy, reverse |
| `reduce.cuh` | reduce, transform_reduce, count, inner_product, min/max/minmax_element |
| `scan.cuh` | inclusive/exclusive scan, their transform_* fused and *_by_key variants |
| `transform.cuh` | transform, for_each, fill, generate, sequence, tabulate, replace |

The four family headers mirror the audited-portable algorithm set (the
portability audit lives in `docs/architecture.md` §22's history); `CMakeLists.txt`
carries the header-only `wwr::thrust` INTERFACE target that resolves the
backend's Thrust package and the `src/` include root.

## Scope

This is the device-side re-export only. Exposing a host-callable surface
(`src/*.cppm`) and the behavioural tests are a later step; the compile-time
acceptance that every re-exported name exists on both backends is
`test/gpu/thrust.cu`.
