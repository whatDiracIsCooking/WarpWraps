# src/wrappers/common — Shared Extension Utilities

This directory provides the foundational C++23 module
`gpumod.wrappers.common` that is imported by every other extension
module. It is the shared **type vocabulary** the four extensions (blas, solver,
sparse, fft) are written against — the floating-point and index-width concepts
and the real/complex/half type maps — and nothing more. It aggregates two module
partitions. It is backend-neutral: it imports only `std` and `src` (never
`src/cuda` or `src/hip`), so the same source builds for the CUDA and the HIP
backend.

**Import:** `import gpumod.wrappers.common;`
**Namespace:** `gpumod`

> The wrappers here take the raw vendor handle (`gpublasHandle_t`, …) and return
> the raw vendor status (`gpublasStatus_t`, …); this module carries no
> error-handling or RAII layer of its own. Typed error policies and RAII handle
> ownership — `error_code`, `error_policy`, `default_error_policy`, `gpu_check`
> and the `BaseGpuHandle` CRTP base — live in `gpumod.extension.common`, and each
> extension's `:*_error` / `:*_handle` partitions build on it.

## Module Partitions

### `:fp_types` — `fp_types.cppm`

C++20 concepts and type mappings for the floating-point types used across GPU
BLAS and solver operations. The complex and half-precision types are `src`'s
backend-neutral aliases: `gpuFloatComplex` is `cuFloatComplex` on a CUDA build
and `hipFloatComplex` on a HIP build, and so on.

| Concept | Matches |
|---|---|
| `real_fp<T>` | `float`, `double` |
| `complex_fp<T>` | `gpuFloatComplex`, `gpuDoubleComplex` |
| `usual_fp<T>` | `real_fp<T>` or `complex_fp<T>` |
| `half_fp<T>` | `gpuHalf`, `gpuBfloat16` |
| `usual_and_half_fp<T>` | `usual_fp<T>` or `half_fp<T>` |

Also exports:

```cpp
template<real_fp T>  using RealToComplexType = ...;  // float -> gpuFloatComplex, double -> gpuDoubleComplex
template<usual_fp T> using ComplexToRealType = ...;  // T -> its real scalar component type
template<half_fp T>  using HalfToFloatType   = float;
```

`gpuFloatComplex`, `gpuDoubleComplex`, and `gpuComplex` are re-exported from this
partition for convenience.

The cuBLAS/cuSOLVER `dispatch_macros` header is not shared from here — it
hardcodes the `cublas` prefix and `cuComplex`, so each wrapper keeps its own
copy next to its sources, in `src/wrappers/blas/dispatch_macros.h` and
`src/wrappers/solver/dispatch_macros.h`.

What stayed is `dispatch_sdcz.h` — the prefix-agnostic
`GPUMOD_REAL_DISPATCH` / `GPUMOD_COMPLEX_DISPATCH` cores those per-wrapper
headers build on. It is a non-module header, so those wrappers reach it by the
owned-header spelling `#include "wrappers/common/dispatch_sdcz.h"`, resolved
through the `src/` include root this target exports (see `CMakeLists.txt`).

### `:int_types` — `int_types.cppm`

The index-width concept the BLAS wrappers constrain their `IntT` parameter with.

```cpp
template<typename T> concept int_type = ...;  // exactly {int, int64_t}
```

The domain is exactly `int` and `int64_t` because that is exactly what the
`*_DISPATCH_64` macros discriminate on: any other type satisfying the concept
would match neither branch and expand the wrapper body to nothing. See #58.

## Build

```
gpumod_add_cxx_module_library(
  NAME gpumod.wrappers.common
  PRIMARY_INTERFACE interface.cppm
  PARTITIONS fp_types.cppm int_types.cppm
  LINK_PUBLIC gpumod.complex gpumod.fp16 gpumod.bf16
  IMPORT_STD
)
```
