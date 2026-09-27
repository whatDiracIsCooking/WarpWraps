# wwr.extension.random_normal

Backend-neutral parallel generation of standard-normal values into a typed
device array, one value per element, drawing from cuRAND / hipRAND
**device**-API states initialized by `wwr.extension.init_state`.

```cpp
import wwr.extension.init_state;
import wwr.extension.random_normal;
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
| `random_normal<T>(stream, count, states, output, scale = T{1})` |

`count == 0` is a no-op and launches nothing. Each drawn value is multiplied by
`scale` before being stored; it defaults to 1 (an unscaled standard normal).

## Which RNG API is this, and when to use the other one

`wwr.rand` wraps the **host** API — `gpurandGenerateNormal(gen, out, n,
…)` fills a buffer with one library call. That is the right tool when a filled
buffer is all that is wanted, and it is faster for that.

This module is the **device** API: the caller owns an array of generator
states (see `wwr.extension.init_state`), and a kernel draws from them. Reach
for it when the draw has to happen inside a kernel — each thread consuming its
own value as part of a larger computation — or when the same states must
advance across several kernels.

## The states are the caller's, so a second draw continues the streams

`random_normal` consumes one state per element: `states[i]` advances as
`output[i]` is drawn, so the same array fed to a second call **continues** the
streams rather than repeating them (`RandTests.StatesAdvanceAcrossCalls`). The
states must have been initialized by `wwr.extension.init_state`, which is
also where `sequence_offset` and per-element independence are documented.

`gpurandState` comes from `wwr.rand`. Note that it is `gpurandStateXORWOW`
on CUDA and a *distinct type* on HIP — see `src/README.md`.

## Supported output types

The wrapper is an unconstrained template; the supported types are exactly those
it explicitly instantiates, and any other type fails to link:

| Output type | Drawn with |
|---|---|
| `float` | `gpurand_normal` |
| `double` | `gpurand_normal_double` |
| `gpuFloatComplex` | `gpurand_normal2` |
| `gpuDoubleComplex` | `gpurand_normal2_double` |
| `gpuHalf` | `gpurand_normal`, converted |
| `gpuBfloat16` | `gpurand_normal`, converted |

For the complex types each component is drawn as an independent standard
normal, so a component has variance 1 and `E[|z|^2] = 2` — the value is **not**
standard normal as a whole. Pass `scale = 1/sqrt(2)` to normalize the
magnitude to unit variance.

The half types are drawn in single precision and converted — neither vendor
offers a native half-precision normal generator.

## How it is built, and why it is shaped this way

Three kinds of translation unit, because three different compilers have to be
satisfied:

| File | Compiled as | Why |
|---|---|---|
| `interface.cppm`, `instantiations.cpp` | host C++23 modules | the public API |
| `random_normal.cu` | **device** code (nvcc under CUDA, `-x hip` under HIP) | it contains the kernel |
| `random_normal_bridge.h` | included by both | the declaration they share |

`random_normal.cu` is shared unchanged between backends. `.cu`/`.cuh` here means
*device-compiled, whichever backend* — the same sense as the `.cuh` headers it
includes (`src/rand.cuh`, `parallel_for.cuh`), none of which is CUDA-only
either. CMake maps `.cu` to CUDA on its own; under HIP the `LANGUAGE` is
overridden back to `CXX` (a HIP build enables no CUDA language at all) and
`hip::device` is linked, into a small dedicated static library so those flags
never reach the module units. All of that lives in
`wwr_add_gpu_device_library` (`cmake/`), which this `CMakeLists.txt` calls.
There is no per-backend `#if` in the source: `src/rand.cuh` and
`src/fp_types.cuh` resolve every difference, so the functor is written once.

`random_normal_bridge.h` is deliberately `.h`, not `.cuh`: every `.cuh` in this
project is device-pass-only and `#error`s outside one, and that header has to be
includable from the module units' host compiles.

Two things in here are worth knowing before editing:

- **`random_normal_bridge.h` declares the device half in the modules' GLOBAL
  MODULE FRAGMENT.** Forced: a name declared in a module's *purview* has module
  linkage — clang mangles it `f@wwr.extension.random_normal` and it can never
  resolve to a definition from a plain TU, which is what `random_normal.cu` is.
  A GMF can `#include` but not `import`, so the two types in the signatures come
  from `src/gpu_stream_bridge.h` and `src/rand_state_bridge.h` rather than from
  `import wwr.rand` — they are the same types, so nothing is cast anywhere.
  Reading the define those bridges need is why this module links `wwr_backend`
  PRIVATE.
- **The device-side function is `device::random_normal`** — same name as the
  exported wrapper, one namespace down. Keep the `device::` qualification at the
  call site: dropping it is infinite recursion, not a compile error.
- **Three instantiation lists, all six types, all have to stay in step:** the
  `extern template` declarations in `interface.cppm`, the definitions in
  `instantiations.cpp`, and the device-side ones in `random_normal.cu`. Adding a
  type to one and not the others links against nothing.

## Relation to what it replaced

This and `wwr.extension.init_state` were split out of the single
`wwr.extension.rand` module, which had them as two partitions
(`:init_state`, `:random_normal`) over one device library. That module was the
port of the CUDA-only `wwr.extension.curand.*` (removed;
`src/cuda/extension/curand`).
