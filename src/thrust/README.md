# src/thrust — the backend-neutral Thrust re-export layer

Device-includable headers that re-export Thrust (CUDA/CCCL) and rocThrust (HIP)
under one backend-neutral vocabulary in namespace `wwr::thrust`. Part of the
`wwr*` layer directly under `src/`, alongside the `wwr.*` modules: the place a
device TU writes to reach Thrust without naming a backend.

## Device-only, by construction

This layer has **no host-callable surface, and cannot have one** as a module.
Two facts box GPU Thrust out of C++ modules from both sides: a module interface
unit is compiled as ordinary host C++ (no device pass, so it cannot instantiate
a Thrust device algorithm), and a device TU cannot `import` (so it could not
consume one either). The device-instantiated template can therefore neither live
in a module nor be delivered through one — this is true of any device-template
library, not just Thrust (`docs/architecture.md` §22).

So the headers here are exactly that: headers a `.cu` (or `-x hip`) TU
`#include`s and links `wwr::thrust` for. A host TU that wants to call one of
these algorithms must itself go through the device pass (be a `.cu`, or be
compiled `-x cuda` / `-x hip`); a plain host compile cannot — Thrust rejects a
device execution policy in a host pass with a hard `static_assert`.

## Why `using`, and 1:1 with Thrust's headers

The algorithm names are identical on both backends — a caller writes
`thrust::sort` either way — so each header is a plain `using ::thrust::<name>`
re-export: every overload comes across, nothing to keep in sync. The leading
`::` is load-bearing — inside `namespace wwr::thrust` a bare `thrust` names *this*
namespace.

There is **one header per Thrust header, same name**, so the spelling a caller
already knows carries straight over and the layer adds no vocabulary of its own:

```cpp
#include "thrust/sort.cuh"        // mirrors <thrust/sort.h>
wwr::thrust::sort(wwr::par_on(stream), first, last);
```

The **one** spelling that diverges between the backends is the stream-bound
execution policy (`thrust::cuda::par.on` vs `thrust::hip::par.on`). That is not a
name, so it is not re-exported; `wwr::par_on(stream)` (`execution_policy.cuh`) is
the one-line shim over it, passed as the leading argument. It is the sole wwr
addition in this layer — everything else is Thrust under its own names.

## Layout

| File | Mirrors / provides |
|---|---|
| `execution_policy.cuh` | `wwr::par_on(stream)` — the sole backend neutralization, plus the rocThrust `_VSTD` / `hipStreamDefault` toolchain bridging every HIP device TU needs |
| `sort.cuh` `unique.cuh` `partition.cuh` `remove.cuh` `copy.cuh` `reverse.cuh` | the reordering / compaction headers |
| `reduce.cuh` `transform_reduce.cuh` `count.cuh` `inner_product.cuh` `extrema.cuh` | the folds and element extrema |
| `scan.cuh` `transform_scan.cuh` | the prefix scans |
| `fill.cuh` `for_each.cuh` `generate.cuh` `sequence.cuh` `tabulate.cuh` `replace.cuh` `transform.cuh` | the elementwise maps |

`CMakeLists.txt` carries the header-only `wwr::thrust` INTERFACE target (the
backend's Thrust package + the `src/` include root). The compile-time acceptance
that every re-exported name exists on both backends is `test/gpu/thrust.cu`.
