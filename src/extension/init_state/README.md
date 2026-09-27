# gpumod.extension.init_state

Backend-neutral parallel initialization of cuRAND / hipRAND **device**-API
generator states: one state per element, each seeded onto its own subsequence,
ready for a kernel to draw an independent stream per thread.

```cpp
import gpumod.extension.init_state;
import gpumod.extension.random_normal;
using namespace wwr::extension;

auto device = std::make_shared<DeviceHandle>();
auto stream = device->alloc_stream().get();
DeviceBuffer<gpurandState> states(n, device);
DeviceBuffer<float> values(n, device);

init_state(stream, n, states.data(), seed);
random_normal(stream, n, states.data(), values.data());
```

| Exports |
|---|
| `init_state(stream, count, states, seed = 0, sequence_offset = 0, offset = 0)` |

`count == 0` is a no-op and launches nothing.

## The state array is the caller's, on purpose

`init_state` seeds element `i` onto subsequence `sequence_offset + i` of the
given seed. States on **different subsequences of one seed are independent**;
states seeded identically produce identical values, which is the single most
damaging way this can be got wrong (`RandTests.ElementsAreIndependent` exists
for exactly that).

Because the caller holds the states:

- a `random_normal` (see `gpumod.extension.random_normal`) advances the states
  as it draws, so a second draw on the same array **continues** the streams
  rather than repeating them (`RandTests.StatesAdvanceAcrossCalls`);
- one array can be shared by several kernels;
- `sequence_offset` carves further disjoint blocks out of one seed.

`gpurandState` comes from `gpumod.rand`. Note that it is `gpurandStateXORWOW`
on CUDA and a *distinct type* on HIP — see `src/README.md`.

## How it is built, and why it is shaped this way

Two kinds of translation unit, because two different compilers have to be
satisfied:

| File | Compiled as | Why |
|---|---|---|
| `interface.cppm` | host C++23 module | the public API |
| `init_state.cu` | **device** code (nvcc under CUDA, `-x hip` under HIP) | it contains the kernel |
| `init_state_bridge.h` | included by both | the declaration they share |

`init_state.cu` is shared unchanged between backends. `.cu`/`.cuh` here means
*device-compiled, whichever backend* — the same sense as the `.cuh` headers it
includes (`src/rand.cuh`, `parallel_for.cuh`), none of which is CUDA-only
either. CMake maps `.cu` to CUDA on its own; under HIP the `LANGUAGE` is
overridden back to `CXX` (a HIP build enables no CUDA language at all) and
`hip::device` is linked, into a small dedicated static library so those flags
never reach the module unit. All of that lives in
`wwr_add_gpu_device_library` (`cmake/`), which this `CMakeLists.txt` calls.
There is no per-backend `#if` in the source: `src/rand.cuh` and
`src/fp_types.cuh` resolve every difference, so the functor is written once.

`init_state_bridge.h` is deliberately `.h`, not `.cuh`: every `.cuh` in this
project is device-pass-only and `#error`s outside one, and that header has to be
includable from the module unit's host compile.

Two things in here are worth knowing before editing:

- **`init_state_bridge.h` declares the device half in the module's GLOBAL MODULE
  FRAGMENT.** Forced: a name declared in a module's *purview* has module linkage
  — clang mangles it `f@gpumod.extension.init_state` and it can never resolve to
  a definition from a plain TU, which is what `init_state.cu` is. A GMF can
  `#include` but not `import`, so the two types in the signature come from
  `src/gpu_stream_bridge.h` and `src/rand_state_bridge.h` rather than from
  `import gpumod.rand` — they are the same types, so nothing is cast anywhere.
  Reading the define those bridges need is why this module links `gpumod_backend`
  PRIVATE.
- **The device-side function is `device::init_state`** — same name as the
  exported wrapper, one namespace down. Keep the `device::` qualification at the
  call site: dropping it is infinite recursion, not a compile error.

## Relation to what it replaced

This and `gpumod.extension.random_normal` were split out of the single
`gpumod.extension.rand` module, which had them as two partitions
(`:init_state`, `:random_normal`) over one device library. That module was the
port of the CUDA-only `gpumod.extension.curand.*` (removed;
`src/cuda/extension/curand`).
