# `src/` header dependency tree

The `.h` and `.cuh` switch-point and shared-type headers of the `wwr*` layer
(directly under `src/`), as opposed to the `.cppm` modules. This records their
include graph as it stands, so the flatness is visible at a glance and a new edge
stands out in review. `complex.h`, `runtime.h` and `rand.h` are *shared-type*
headers — a host TU that cannot `import` in context (and, for `complex`/`runtime`,
a device `.cuh`) includes one to name the same vendor type the modules export;
they reach `selected_backend.h` directly, so they compile in a host TU without the
device-pass `#error`. `rand.h` goes further: it also carries the rand layer's
`__device__` generators, in a section gated behind the device-pass macros, so a
device `.cu` includes `rand.h` directly and there is no `rand.cuh`.

See also `src/README.md` ("The switch points"), which explains *why* the graph
has this shape; this file only states *what* it is.

## The graph

`selected_backend.h` is the leaf everything rests on. The device `.cuh` files
reach it through `device_guard.h`, which adds the "must be a device pass"
`#error` guard a host-safe header must not have. `cooperative_groups.cuh` and
`wmma.cuh` reach `device_guard.h` through `runtime.cuh`, whose `WWR_WARP_SIZE`
they also want — a portable tile size for one, a wave index for the other, both
because the API is a whole-warp collective. **No host-safe header includes
another, and every `.cuh`-to-`.cuh` edge stays inside a single target** — the two
such edges are both into `runtime.cuh`, and all three files are `wwr.device`, so
neither leaks an include path across targets. That is the concern that keeps every
`.cuh` rooted directly at `device_guard.h`.

`complex.h`, `runtime.h` and `rand.h` reach `selected_backend.h` *directly*, not
through `device_guard.h`: each compiles in a host TU and so must not carry the
"must be a device pass" `#error`. They appear below as leaves of
`selected_backend.h` alongside `device_guard.h`. `complex.h` is included by
`complex.cppm` (host module) and device `.cu` that reach its gated wrappers
(`random_normal.cu`, `test/gpu/complex.cu`); `runtime.h` by
`runtime.cuh` (device) and the host TUs that declare a stream-taking function
across the boundary in a GMF or plain `.cu` (the two `*_bridge.h`,
`example/warp_reduce`) — `runtime_api.cppm` exports the same type via `import`
rather than including it, because the vendor runtime macros would collide with
its `WWR_RT_VALUE` expansions. `rand.h` is included by `rand.cppm` (host module,
which re-exports the state types and binds the host API), the two `*_bridge.h`,
and the two extension device `.cu` (which reach its gated `__device__`
generators); unlike `runtime_api.cppm`, `rand.cppm` *does* include its shared-type
header — the vendor kernel headers define no colliding macros.

Unlike `complex.h` and `runtime.h`, whose vendor header is host-cheap, `rand.h`
pulls the heavy `curand_kernel.h` / `hiprand_kernel.h`. That weight lands only in
the TUs that `#include rand.h` and is firewalled from every `import wwr.rand`
consumer by the module BMI — the trade is spelled out in `rand.h`'s own header
and `src/README.md`.

```
selected_backend.h        no #includes — the leaf the switch/shared-type layer rests on
├── device_guard.h         + the "is this a device pass?" #error guard; no vendor header
│   ├── runtime.cuh  + <cuda_runtime.h>            | <hip/hip_runtime.h>   (also includes runtime.h)
│   │   ├── cooperative_groups.cuh
│   │   │                  + <cooperative_groups.h>      | <hip/hip_cooperative_groups.h>
│   │   └── wmma.cuh       + <mma.h>                     | <rocwmma/rocwmma.hpp>
│   ├── fp16.cuh           + <cuda_fp16.h>               | <hip/hip_fp16.h>
│   └── bf16.cuh           + <cuda_bf16.h>               | <hip/hip_bf16.h>
├── complex.h            + <cuComplex.h>               | <array> <hip/hip_complex.h>
│                          (+ __device__ wrappers, gated behind the device-pass macros)
├── runtime.h           + <cuda_runtime_api.h>        | <hip/hip_runtime_api.h>
└── rand.h               + <curand.h> <curand_kernel.h> | <cstdio> <hiprand/hiprand.h> <hiprand/hiprand_kernel.h>
                          (+ __device__ generators, gated behind the device-pass macros)

backend.h                 no #includes — independent; consumed only by the .cppm modules
```

The two columns after each `+` are the CUDA branch (`WWR_SELECTED_CUDA`) and
the HIP branch (`WWR_SELECTED_HIP` / the `#else`); a translation unit sees
exactly one. `fp16.cuh`, `bf16.cuh`, `runtime.cuh`, `cooperative_groups.cuh`,
`wmma.cuh`, `complex.h`, `runtime.h` and `rand.h` pull vendor headers;
`runtime.cuh` shares `runtime.h`'s type while adding the full runtime on top.
`complex.h` and `rand.h` each carry their `__device__` wrappers themselves, in a
device-pass-gated section, alongside their types and vendor headers. `device_guard.h` pulls none — it carries only the guard. That guard is a check on
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
| `selected_backend.h` | (internal — `device_guard.h`; and `complex.h`, `runtime.h`, `rand.h`) | — (header-only, no target of its own) |
| `device_guard.h` | (internal only — `runtime.cuh`, `fp16.cuh`, `bf16.cuh`) | — (header-only, rides each `.cuh`'s target) |
| `backend.h` | `blas.cppm`, `bf16.cppm`, `complex.cppm`, `fp16.cppm`, `rand.cppm`, `runtime_api.cppm`, `solver.cppm` | each module's own target |
| `complex.h` | `complex.cppm`, `extension/random_normal/random_normal.cu`, `test/gpu/complex.cu` (device, for the gated wrappers) | header-only (rides `wwr.device` / `wwr.complex`) |
| `runtime.h` | `runtime.cuh`, `extension/init_state/init_state_bridge.h`, `extension/random_normal/random_normal_bridge.h`, `example/warp_reduce/warp_reduce_bridge.h` | header-only (rides `wwr.device` and the src/ include root each consumer carries) |
| `rand.h` | `rand.cppm`, the two `*_bridge.h`, `extension/init_state/init_state.cu`, `extension/random_normal/random_normal.cu` (device, for the gated generators) | header-only (rides the src/ include root each consumer carries; the device `.cu` also link `wwr.rand.device` for the RNG library; installed by the `src/*.h` glob) |
| `runtime.cuh` | `extension/parallel_for/parallel_for.cuh` (and, internally, `cooperative_groups.cuh`) | `wwr.device` |
| `cooperative_groups.cuh` | (none yet — the warp-reduction example will be its first caller) | `wwr.device` |
| `wmma.cuh` | (none yet — only `test/gpu/wmma.cu` compiles it) | `wwr.device` |
| `fp16.cuh` | `extension/random_normal/random_normal.cu` | `wwr.device` |
| `bf16.cuh` | `extension/random_normal/random_normal.cu` | `wwr.device` |

`runtime.cuh`, `cooperative_groups.cuh`, `wmma.cuh`, `fp16.cuh` and `bf16.cuh`
share the `wwr.device` target, as does `complex.h`'s device section; the rand
device generators (in `rand.h`) sit behind a separate `wwr.rand.device` target on
purpose (an RNG device TU should not be forced to carry the runtime/warp-size
machinery). That target split is why
`device_guard.h` — not
`runtime.cuh` — is where the shared guard lives: it carries only the guard
and `selected_backend.h`, no vendor headers and no target-specific content, so
each `.cuh` can include it without one target's include path or vendor headers
leaking into another's consumers. Routing a `.cuh` in one target through a `.cuh`
in another would do exactly that leaking, which is why every cross-target pair is
kept apart. The two `.cuh`-to-`.cuh` edges, `cooperative_groups.cuh` and `wmma.cuh` →
`runtime.cuh`, are within `wwr.device`, so they carry no such leak: a
consumer of any of the three already links that one target.
