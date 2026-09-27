# src/wrappers/fft — Type-Safe GPU FFT Extension

This directory provides the C++23 module `wwr.wrappers.fft`: type-safe
execution wrappers over the base (single-GPU) cuFFT /
hipFFT API. Callers write `exec_c2c<float>(...)` instead of picking
`cufftExecC2C` / `hipfftExecC2C` (or the `Z2Z` double-precision sibling); the
correct typed entry point is selected at compile time via `if constexpr`
dispatch.

It is backend-neutral: written once against `src/fft`'s `wwrfft*` names
(`wwrfftExecC2C` is `cufftExecC2C` on a CUDA build and `hipfftExecC2C` on a HIP
build), so the same source builds for either `WWR_GPU_BACKEND`.

**Import:** `import wwr.wrappers.fft;`
**Namespace:** `wwr`

## Backend differences, and where they are resolved

None of them are resolved here — all live in `src/fft.cppm`: the
`INVERSE`/`BACKWARD` direction-flag spelling, the differing result-code sets,
and the hand-written status-string functions. See that file and
`src/README.md`.

The multi-GPU `Xt` surface (`cufftXt` / `hipfftXt`) is not wrapped by this
module or `src/fft`; reach it through `wwr.cuda.cufftXt` /
`wwr.hip.hipfftXt`.

The exec wrappers take the raw `wwrfftHandle` plan and return the raw
`wwrfftResult_t` — a caller creates, configures, destroys and error-checks the
plan itself. RAII ownership and typed error handling live in the sibling
`wwr.extension.fft` module (`:fft_plan` wraps `wwrfftHandle`; `:fft_error`
specialises the error policy for `wwrfftResult_t`).

## Module Partitions

### `:type_traits` — `type_traits.cppm`

Re-exports `wwr.wrappers.common` and adds `FftComplex<T>`, mapping a real
precision (`float`/`double`) to the FFT library's own complex element type
(`wwrfftComplex`/`wwrfftDoubleComplex`). It is `wwrfftComplex`, not the shared
`wwrFloatComplex`, on purpose: on HIP `hipfftComplex` is a distinct type from
`hipComplex`, and the `wwrfftExec*` signatures name the former.

### `:exec` — `exec.cppm` — Execution

Three functions, each templated on the real precision `T` (`float`/`double`):

| Function | float | double |
|---|---|---|
| `exec_c2c<T>(plan, in, out, direction)` | `C2C` | `Z2Z` |
| `exec_r2c<T>(plan, in, out)` | `R2C` | `D2Z` |
| `exec_c2r<T>(plan, in, out)` | `C2R` | `Z2D` |

`direction` is `WWRFFT_FORWARD` or `WWRFFT_INVERSE`. Plan creation and
configuration are not wrapped — they take no element-type template argument, so
call the raw `wwrfftCreate` / `wwrfftPlan*` / `wwrfftMakePlan*` /
`wwrfftGetSize*` / `wwrfftSetWorkArea` / `wwrfftDestroy` functions on the
`wwrfftHandle` directly.

## Template Instantiation

Explicit instantiations are hand-written, one per `(function, precision)` over
`{float, double}`. The `extern template` declarations live in `exec.cppm` next
to each function; the matching `template` instantiations live in
`instantiations.cpp`. A new instantiation also needs its entry in
`test/wrappers/fft/fft_dispatch.toml` (see Tests), or the build fails.

## Tests

- `test/wrappers/fft/fft_dispatch.toml` — build-time, both backends, no GPU:
  `test/shared/dispatch.py` disassembles this target's objects and checks that
  every explicit instantiation calls exactly the `wwrfftExec*` function the
  table names — catching a precision widened the wrong way (`exec_c2c<float>`
  calling `Z2Z`) or a transform kind mixed up (`R2C` vs `C2R`).
- `test/gpu/fft.cppm` — compile-time checks on `src/fft`'s names.

There are no runtime tests, by design: a wrapper's body is the token-paste
forward and nothing else, so the pasted name is the whole of what it can get
wrong, and the dispatch check proves that for every instantiation without a
device.

## Build

```
wwr_add_cxx_module_library(
  NAME wwr.wrappers.fft
  PRIMARY_INTERFACE interface.cppm
  PARTITIONS type_traits.cppm exec.cppm
  IMPLEMENTATION instantiations.cpp
  LINK_PUBLIC wwr.fft wwr.wrappers.common
  IMPORT_STD
)
```

`dispatch_macros.h` (the `wwrfftExec` dispatch macro) sits next to the sources
and is included by relative path.
