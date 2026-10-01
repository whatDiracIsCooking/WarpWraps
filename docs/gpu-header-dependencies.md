# `src/` header dependency tree

The `.h` and `.cuh` switch-point and shared-type headers of the `wwr*` layer
(directly under `src/`), as opposed to the `.cppm` modules. This records their
include graph as it stands, so the flatness is visible at a glance and a new edge
stands out in review. `complex.h`, `runtime.h` and `rand.h` are *shared-type*
headers — a host TU that cannot `import` in context (and a device `.cu` / `.cuh`)
includes one to name the same vendor type the modules export; they reach
`selected_backend.h` directly, so they compile in a host TU without the
device-pass `#error`. Each also carries device content in a section gated behind
the device-pass macros: `rand.h` the rand layer's `__device__` generators (so a
device `.cu` includes `rand.h` directly and there is no `rand.cuh`), `complex.h`
the complex `__device__` wrappers, and `runtime.h` the `WWR_GRID_CONSTANT` /
`WWR_WARP_SIZE` macros and the full runtime header (folded in from the former
`runtime.cuh`).

`cooperative_groups.h` and `wmma.h` are a second kind of `.h`, and the reason the
extension tracks host-*safety* rather than whether host symbols exist: they are
device-only (no host-usable type or function — one defines nothing, the other a
device-namespace alias), but their *whole* body sits behind the device-pass
macros, so a host TU that includes one sees an empty header rather than the
`#error` a `.cuh` carries. They are host-safe without being host-usable. That
makes them `.h`, not `.cuh`, even though — unlike the shared-type trio — a host
consumer gets nothing from them.

See also `src/README.md` ("The switch points"), which explains *why* the graph
has this shape; this file only states *what* it is.

## The graph

`selected_backend.h` is the leaf everything rests on. The device-only `.cuh`
files — `atomic.cuh`, `parallel_for.cuh`, `math.cuh` — reach it through
`device_guard.h`, which adds the "must be a device pass" `#error` guard a
host-safe header must not have. `cooperative_groups.h` and `wmma.h` need no such
`#error`: inside their own device gate they `#include` `runtime.h` — host-safe,
reaching `selected_backend.h` directly — so the edge exists in a device pass and
vanishes in a host compile, where the gate skips it and the header is empty. They
want `runtime.h` for `WWR_WARP_SIZE` (a portable tile size for one, a wave index
for the other, both because the API is a whole-warp collective) and the backend
switch, through that one include. **Every edge *into* a device header stays
inside a single target** — the device-pass edges `cooperative_groups.h` →
`runtime.h` and `wmma.h` → `runtime.h` are both within `wwr.device`, so neither
leaks an include path across targets. That is the concern that keeps each `.cuh`
rooted directly at `device_guard.h` for its guard, and that lets these two gated
`.h` reach `runtime.h` without a leak.

`complex.h`, `runtime.h`, `rand.h`, `fp16.h`, `bf16.h` and `fp8.h` reach
`selected_backend.h` *directly*, not through `device_guard.h`: each compiles in a
host TU and so must not carry the "must be a device pass" `#error`. They appear
below as leaves of `selected_backend.h` alongside `device_guard.h`.
(`cooperative_groups.h` and `wmma.h` differ again: in a host compile their gate is
shut, so they reach `selected_backend.h` on *no* path and define nothing; they
reach it through `runtime.h` only in a device pass, which is why they appear below
nested under `runtime.h` rather than as leaves here.)
`complex.h` is included by
`complex.cppm` (host module) and device `.cu` that reach its gated wrappers
(`random_normal.cu`, `test/gpu/complex.cu`); `runtime.h` by the device headers
(`atomic.cuh`, `parallel_for.cuh`, and `cooperative_groups.h` / `wmma.h` inside
their gate) and the host TUs that declare a stream-taking function across the
boundary in a GMF or plain `.cu` (the two `*_bridge.h`, `example/warp_reduce`) —
`runtime_api.cppm` does *not* include `runtime.h`: it reaches the same handle, and
the rest of the runtime surface, through its own vendor-include header
`runtime_api.h`, which `#undef`s the allocation-flag macros that would otherwise
collide with its `WWR_RT_VALUE` expansions (`runtime.h`, which binds no flag
values, leaves them defined). `rand.h` is included by `rand.cppm` (host module,
which re-exports the state types and binds the host API), the two `*_bridge.h`,
and the two extension device `.cu` (which reach its gated `__device__`
generators); `rand.cppm` includes `rand.h` directly, with no `#undef` needed —
the vendor kernel headers define no colliding macros.

`fp16.h`, `bf16.h` and `fp8.h` are the same shape: each is included by its device
`.cu` consumers for the gated wrappers, and `fp16.h` / `bf16.h` are also included
by `fp16.cppm` / `bf16.cppm` (which import no raw module — their conversions are
external-linkage). `fp8.cppm` is the lone holdout: its conversions are
static-inline (§12), so it imports `wwr.{cuda,hip}.*fp8` for the host wrappers and
does *not* include `fp8.h`.

Unlike `complex.h` and `runtime.h`, whose always-on vendor header is host-cheap,
`rand.h` pulls the heavy `curand_kernel.h` / `hiprand_kernel.h`. That weight lands
only in the TUs that `#include rand.h` and is firewalled from every `import
wwr.rand` consumer by the module BMI — the trade is spelled out in `rand.h`'s own
header and `src/README.md`.

```
selected_backend.h        no #includes — the leaf the switch/shared-type layer rests on
├── device_guard.h         + the "is this a device pass?" #error guard; no vendor header
│                            (included directly by the device-only .cuh: atomic.cuh, parallel_for.cuh, math.cuh)
├── complex.h            + <cuComplex.h>               | <array> <hip/hip_complex.h>
│                          (+ __device__ wrappers, gated behind the device-pass macros)
├── runtime.h           + <cuda_runtime_api.h>        | <hip/hip_runtime_api.h>   (the stream type, always)
│   │                      (+ <cuda_runtime.h> | <hip/hip_runtime.h>, WWR_GRID_CONSTANT, WWR_WARP_SIZE — gated behind the device-pass macros)
│   ├── cooperative_groups.h  (.h; whole body gated — empty in a host TU, includes runtime.h only in a device pass)
│   │                  + <cooperative_groups.h>      | <hip/hip_cooperative_groups.h>
│   └── wmma.h         (.h; whole body gated — empty in a host TU, includes runtime.h only in a device pass)
│                          + <mma.h>                     | <rocwmma/rocwmma.hpp>
├── runtime_api.h       + <cuda_runtime_api.h>        | <hip/hip_runtime_api.h>   (runtime_api.cppm's GMF only; #undef's the colliding flag macros)
├── rand.h               + <curand.h> <curand_kernel.h> | <cstdio> <hiprand/hiprand.h> <hiprand/hiprand_kernel.h>
│                          (+ __device__ generators, gated behind the device-pass macros)
├── fp16.h               + <cuda_fp16.h>               | <array> <hip/hip_fp16.h>
│                          (+ __device__ conversions, gated behind the device-pass macros)
├── bf16.h               + <cuda_bf16.h>               | <array> <hip/hip_bf16.h>
│                          (+ __device__ conversions, gated behind the device-pass macros)
└── fp8.h                + <cuda_fp8.h>                | <algorithm> <array> <hip/hip_fp8.h>
                           (+ __device__ narrowing, gated behind the device-pass macros)

backend.h                 no #includes — independent; consumed only by the .cppm modules
```

The two columns after each `+` are the CUDA branch (`WWR_SELECTED_CUDA`) and
the HIP branch (`WWR_SELECTED_HIP` / the `#else`); a translation unit sees
exactly one — or, for the two fully-gated `.h`, neither, in a host compile where
the gate is shut. `cooperative_groups.h`, `wmma.h`, `complex.h`,
`runtime.h`, `rand.h`, `fp16.h`, `bf16.h` and `fp8.h` pull vendor headers
(`cooperative_groups.h` and `wmma.h` only inside their device gate);
`runtime.h` adds the full runtime (`<cuda_runtime.h>` / `<hip/hip_runtime.h>`) on
top of its always-on stream-type header in that gated section.
`complex.h`, `rand.h`, `fp16.h`, `bf16.h` and `fp8.h` each carry their
`__device__` wrappers themselves, in a device-pass-gated section, alongside their
types and vendor headers. `device_guard.h` pulls none — it carries only the guard. That guard is a check on
`__CUDACC__` / `__HIP__` / `__HIPCC__`, separate from the backend selection: it
answers "is this a device pass?", not "which backend?", which is why it lives in
`device_guard.h` and not in `selected_backend.h` (a host-safe header includes the
latter from a host compile and must not `#error` there — which is exactly why
`complex.h`, `runtime.h` and `rand.h` reach `selected_backend.h` directly rather
than through `device_guard.h`).

## Consumers (reverse edges)

Who reaches each header from outside `src/`. `selected_backend.h` is reached
transitively — through `device_guard.h` internally, and directly through
`complex.h`, `runtime.h` and `rand.h`.

| Header | Included by | CMake target that carries it |
|---|---|---|
| `selected_backend.h` | (internal — `device_guard.h`; and `complex.h`, `runtime.h`, `rand.h`, `fp16.h`, `bf16.h`, `fp8.h`) | — (header-only, no target of its own) |
| `device_guard.h` | (internal — the device-only `.cuh`: `atomic.cuh`, `parallel_for.cuh`, `math.cuh`) | — (header-only; rides the target of each including `.cuh`) |
| `backend.h` | `blas.cppm`, `bf16.cppm`, `complex.cppm`, `fp16.cppm`, `rand.cppm`, `runtime_api.cppm`, `solver.cppm` | each module's own target |
| `complex.h` | `complex.cppm`, `extension/random_normal/random_normal.cu`, `test/gpu/complex.cu` (device, for the gated wrappers) | header-only (rides `wwr.device` / `wwr.complex`) |
| `runtime.h` | `atomic.cuh`, `extension/parallel_for/parallel_for.cuh`, `cooperative_groups.h` / `wmma.h` (device, inside their gate), `extension/init_state/init_state_bridge.h`, `extension/random_normal/random_normal_bridge.h`, `example/warp_reduce/warp_reduce_bridge.h` | header-only (rides `wwr.device` and the src/ include root each consumer carries) |
| `rand.h` | `rand.cppm`, the two `*_bridge.h`, `extension/init_state/init_state.cu`, `extension/random_normal/random_normal.cu` (device, for the gated generators) | header-only (rides the src/ include root each consumer carries; the device `.cu` also link `wwr.rand.device` for the RNG library; installed by the `src/*.h` glob) |
| `atomic.cuh` | `test/gpu/atomic.cu` (compile test) | `wwr.device` |
| `cooperative_groups.h` | `example/warp_reduce/warp_reduce.cu`, `test/gpu/cooperative_groups.cu` (compile test) | `wwr.device` |
| `wmma.h` | (no functional caller yet — only `test/gpu/wmma.cu` compiles it) | `wwr.device` |
| `fp16.h` | `fp16.cppm` (host, for the type + conversions), `random_normal.cu`, `test/gpu/wmma.cu`, `test/gpu/fp16.cu` (device) | header-only (rides `wwr.fp16` / `wwr.device`) |
| `bf16.h` | `bf16.cppm` (host), `random_normal.cu`, `test/gpu/wmma.cu`, `test/gpu/bf16.cu` (device) | header-only (rides `wwr.bf16` / `wwr.device`) |
| `fp8.h` | `test/gpu/fp8.cu` (device, for the gated narrowing). `fp8.cppm` does **not** include it — it imports the raw module instead (static-inline conversions, §12) | header-only (rides `wwr.device`) |

`cooperative_groups.h`, `wmma.h`, `atomic.cuh` and `runtime.h`'s device section
share the `wwr.device` target, as do the device sections of `complex.h` /
`fp16.h` / `bf16.h` / `fp8.h`; the rand device generators (in `rand.h`) sit behind
a separate `wwr.rand.device` target on
purpose (an RNG device TU should not be forced to carry the runtime/warp-size
machinery). That target split is why
`device_guard.h` — not `runtime.h` — is where the shared guard lives: it carries
only the guard and `selected_backend.h`, no vendor headers and no target-specific
content, so each `.cuh` can include it without one target's include path or
vendor headers leaking into another's consumers. Routing a device header in one
target through one in another would do exactly that leaking, which is why every
cross-target pair is kept apart. The device-pass edges from the gated `.h`,
`cooperative_groups.h` and `wmma.h` → `runtime.h` (taken only in a device pass),
are within `wwr.device`, so they carry no such leak: a consumer of any of the
three already links that one target.
