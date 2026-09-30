# `src/` header dependency tree

The `.h` and `.cuh` switch-point and shared-type headers of the `wwr*` layer
(directly under `src/`), plus the one remaining bridge (`rand_state_bridge.h`,
under `src/extension/bridge/`), as opposed to the `.cppm` modules. This records
their include graph as it stands, so the flatness is visible at a glance and a
new edge stands out in review. `complex.h` and `runtime.h` are *shared-type*
headers — a device `.cuh` and a host TU that cannot `import` in context both
include one to name the same vendor type the modules export; like the bridge they
reach `selected_backend.h` directly, so they compile in a host TU without the
device-pass `#error`. The bridge lives in the extension layer but is documented
here for the same reason.

See also `src/README.md` ("The switch points"), which explains *why* the graph
has this shape; this file only states *what* it is.

## The graph

`selected_backend.h` is the leaf everything rests on. The device `.cuh` files
reach it through `device_guard.h`, which adds the "must be a device pass"
`#error` guard a host-safe header must not have. `cooperative_groups.cuh` and
`wmma.cuh` reach `device_guard.h` through `runtime.cuh`, whose `WWR_WARP_SIZE`
they also want — a portable tile size for one, a wave index for the other, both
because the API is a whole-warp collective. **No host-safe header includes
another, and every `.cuh`-to-`.cuh` edge stays inside a single target** — both
are into `runtime.cuh`, and all three files are `wwr.device`, so neither leaks an
include path across targets, which is the concern that keeps every other `.cuh`
rooted directly at `device_guard.h`.

`complex.h`, `runtime.h` and `rand_state_bridge.h` reach `selected_backend.h`
*directly*, not through `device_guard.h`: each compiles in a host TU and so must
not carry the "must be a device pass" `#error`. They appear below as leaves of
`selected_backend.h` alongside `device_guard.h`. `complex.h` is included by
`complex.cuh` (device) and `complex.cppm` (host module); `runtime.h` by
`runtime.cuh` (device) and the host TUs that declare a stream-taking function
across the boundary in a GMF or plain `.cu` (the two `*_bridge.h`,
`example/warp_reduce`) — `runtime_api.cppm` exports the same type via `import`
rather than including it, because the vendor runtime macros would collide with
its `WWR_RT_VALUE` expansions; `rand_state_bridge.h` lives in
`src/extension/bridge/` (carried by the
`wwr.extension.bridge` INTERFACE target) and is included by
`extension/init_state/init_state_bridge.h` and
`extension/random_normal/random_normal_bridge.h`.

```
selected_backend.h        no #includes — the leaf the switch/shared-type layer rests on
├── device_guard.h         + the "is this a device pass?" #error guard; no vendor header
│   ├── runtime.cuh  + <cuda_runtime.h>            | <hip/hip_runtime.h>   (also includes runtime.h)
│   │   ├── cooperative_groups.cuh
│   │   │                  + <cooperative_groups.h>      | <hip/hip_cooperative_groups.h>
│   │   └── wmma.cuh       + <mma.h>                     | <rocwmma/rocwmma.hpp>
│   ├── complex.cuh        (vendor headers via complex.h)
│   ├── fp16.cuh           + <cuda_fp16.h>               | <hip/hip_fp16.h>
│   ├── bf16.cuh           + <cuda_bf16.h>               | <hip/hip_bf16.h>
│   └── rand.cuh           + <curand_kernel.h>           | <cstdio> <hiprand/hiprand_kernel.h>
├── complex.h            + <cuComplex.h>               | <array> <hip/hip_complex.h>
├── runtime.h           + <cuda_runtime_api.h>        | <hip/hip_runtime_api.h>
└── rand_state_bridge.h   forward-declares the vendor struct; no vendor header

backend.h                 no #includes — independent; consumed only by the .cppm modules
```

The two columns after each `+` are the CUDA branch (`WWR_SELECTED_CUDA`) and
the HIP branch (`WWR_SELECTED_HIP` / the `#else`); a translation unit sees
exactly one. `fp16.cuh`, `bf16.cuh`, `runtime.cuh`, `cooperative_groups.cuh`,
`wmma.cuh`, `rand.cuh`, `complex.h` and `runtime.h` pull vendor headers;
`complex.cuh` pulls its via `complex.h`, and `runtime.cuh` shares `runtime.h`'s
type while adding the full runtime on top; `rand_state_bridge.h` pulls none (it
forward-declares the vendor struct), and `device_guard.h` pulls none — it carries
only the guard. That guard is a check on `__CUDACC__` / `__HIP__` / `__HIPCC__`,
separate from the backend selection: it answers "is this a device pass?", not
"which backend?", which is why it lives in `device_guard.h` and not in
`selected_backend.h` (a host-safe header includes the latter from a host compile
and must not `#error` there — which is exactly why `complex.h`, `runtime.h` and
`rand_state_bridge.h` reach `selected_backend.h` directly rather than through
`device_guard.h`).

## Consumers (reverse edges)

Who reaches each header from outside `src/`. `selected_backend.h` is reached
transitively — through `device_guard.h` internally, and directly through
`complex.h`, `runtime.h` and `rand_state_bridge.h`.

| Header | Included by | CMake target that carries it |
|---|---|---|
| `selected_backend.h` | (internal — `device_guard.h`; and `complex.h`, `runtime.h`, `rand_state_bridge.h`) | — (header-only, no target of its own) |
| `device_guard.h` | (internal only — `runtime.cuh`, `complex.cuh`, `fp16.cuh`, `bf16.cuh`, `rand.cuh`) | — (header-only, rides each `.cuh`'s target) |
| `backend.h` | `blas.cppm`, `bf16.cppm`, `complex.cppm`, `fp16.cppm`, `rand.cppm`, `runtime_api.cppm`, `solver.cppm` | each module's own target |
| `complex.h` | `complex.cuh`, `complex.cppm` | header-only (rides `wwr.device` / `wwr.complex`) |
| `runtime.h` | `runtime.cuh`, `extension/init_state/init_state_bridge.h`, `extension/random_normal/random_normal_bridge.h`, `example/warp_reduce/warp_reduce_bridge.h` | header-only (rides `wwr.device` and the src/ include root each consumer carries) |
| `rand_state_bridge.h` | `extension/init_state/init_state_bridge.h`, `extension/random_normal/random_normal_bridge.h` | `wwr.extension.bridge` |
| `runtime.cuh` | `extension/parallel_for/parallel_for.cuh` (and, internally, `cooperative_groups.cuh`) | `wwr.device` |
| `cooperative_groups.cuh` | (none yet — the warp-reduction example will be its first caller) | `wwr.device` |
| `wmma.cuh` | (none yet — only `test/gpu/wmma.cu` compiles it) | `wwr.device` |
| `complex.cuh` | `extension/random_normal/random_normal.cu` | `wwr.device` |
| `fp16.cuh` | `extension/random_normal/random_normal.cu` | `wwr.device` |
| `bf16.cuh` | `extension/random_normal/random_normal.cu` | `wwr.device` |
| `rand.cuh` | `extension/init_state/init_state.cu`, `extension/random_normal/random_normal.cu` | `wwr.rand.device` |

`runtime.cuh`, `cooperative_groups.cuh`, `wmma.cuh`, `complex.cuh`, `fp16.cuh` and
`bf16.cuh` share the `wwr.device` target; `rand.cuh` sits in a separate `wwr.rand.device`
target on purpose (an RNG device TU should not be forced to carry the
runtime/warp-size machinery). That target split is why `device_guard.h` — not
`runtime.cuh` — is where the shared guard lives: it carries only the guard
and `selected_backend.h`, no vendor headers and no target-specific content, so
each `.cuh` can include it without one target's include path or vendor headers
leaking into another's consumers. Routing a `.cuh` in one target through a `.cuh`
in another would do exactly that leaking, which is why every cross-target pair is
kept apart. The two `.cuh`-to-`.cuh` edges, `cooperative_groups.cuh` and `wmma.cuh` →
`runtime.cuh`, are within `wwr.device`, so they carry no such leak: a
consumer of any of the three already links that one target.
