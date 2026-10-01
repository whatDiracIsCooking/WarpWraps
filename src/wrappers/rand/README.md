# src/wrappers/rand — Type-Safe GPU RNG Host Generation

This directory provides the C++23 module `wwr.wrappers.rand`: type-safe host
generation wrappers over the cuRAND / hipRAND host API. Callers write
`generate_uniform<float>(...)` instead of picking `curandGenerateUniform` /
`hiprandGenerateUniform` (or the `...Double` double-precision sibling); the
correct typed entry point is selected at compile time via `if constexpr`
dispatch.

It is backend-neutral: written once against `src/rand`'s `wwrrand*` names
(`wwrrandGenerateUniform` is `curandGenerateUniform` on a CUDA build and
`hiprandGenerateUniform` on a HIP build), so the same source builds for either
`WWR_GPU_BACKEND`.

**Import:** `import wwr.wrappers.rand;`
**Namespace:** `wwr`

## Scope — real only

The host generation surface is real-only: `float` and `double`. cuRAND / hipRAND
expose no complex host generate, no `_64` index-type variant, and the
double entry points are a `Double` name suffix rather than a distinct element
type. So this module's `:type_traits` needs no complex mapping (no `FftComplex`
analogue the way `wwr.wrappers.fft` has) — it only re-exports the `real_fp`
concept from `wwr.wrappers.common`.

The integer (`wwrrandGenerate` / `wwrrandGenerateLongLong`), Poisson and
quasirandom-direction surfaces carry no S/D letter dispatch, so they are not
wrapped here — call the raw `wwrrand*` functions directly.

The generate wrappers take the raw `wwrrandGenerator_t` and return the raw
`wwrrandStatus_t` — a caller creates, seeds, stream-binds, destroys and
error-checks the generator itself. RAII ownership and typed error handling are
companion extension work, not this layer.

## Module Partitions

### `:type_traits` — `type_traits.cppm`

Re-exports `wwr.wrappers.common` (the `real_fp` concept). Nothing else: rand's
host surface is real-only, so there is no element-type mapping to add.

### `:exec` — `exec.cppm` — Host generation

Three functions, each templated on the real precision `T` (`float`/`double`):

| Function | float | double |
|---|---|---|
| `generate_uniform<T>(gen, out, n)` | `Uniform` | `UniformDouble` |
| `generate_normal<T>(gen, out, n, mean, stddev)` | `Normal` | `NormalDouble` |
| `generate_lognormal<T>(gen, out, n, mean, stddev)` | `LogNormal` | `LogNormalDouble` |

The suffixes paste onto `wwrrandGenerate` (so float `generate_uniform` calls
`wwrrandGenerateUniform`). Generator creation, seeding and configuration are not
wrapped — they take no element-type template argument, so call the raw
`wwrrandCreateGenerator` / `wwrrandSetPseudoRandomGeneratorSeed` /
`wwrrandSetStream` / `wwrrandDestroyGenerator` functions on the
`wwrrandGenerator_t` directly.

## Template Instantiation

Explicit instantiations are hand-written, one per `(function, precision)` over
`{float, double}`. The `extern template` declarations live in `exec.cppm` next
to each function; the matching `template` instantiations live in
`instantiations.cpp`. A new instantiation also needs its entry in
`test/wrappers/rand/rand_dispatch.toml` (see Tests), or the build fails.

## Tests

- `test/wrappers/rand/rand_dispatch.toml` — build-time, both backends, no GPU:
  `test/shared/dispatch.py` disassembles this target's objects and checks that
  every explicit instantiation calls exactly the `wwrrandGenerate*` function the
  table names — catching a precision widened the wrong way
  (`generate_uniform<float>` calling `UniformDouble`) or a distribution mixed up
  (`Normal` vs `LogNormal`).
- `test/wrappers/rand/rand_tests.cpp` — runtime (`REQUIRES_GPU`): creates a
  generator, seeds it, and generates uniform/normal/lognormal into a device
  buffer for both `float` and `double`, checking the returned status is success.

## Build

```
wwr_add_cxx_module_library(
  NAME wwr.wrappers.rand
  PRIMARY_INTERFACE interface.cppm
  PARTITIONS type_traits.cppm exec.cppm
  IMPLEMENTATION instantiations.cpp
  LINK_PUBLIC wwr.rand wwr.wrappers.common
  IMPORT_STD
)
```

`dispatch_macros.h` (the `wwrrandGenerate` dispatch macro) sits next to the
sources and is included by relative path.
