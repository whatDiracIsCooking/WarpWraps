# src/wrappers/complex — Ergonomic complex value type

This directory provides `wwr::complex<T>`, an ergonomic, operator-carrying value
type over the gpu\* complex layer (`wwr.complex` / `complex.cuh`). It is
backend-neutral: the module imports only `std`, `wwr.complex` and
`wwr.wrappers.common` (never `src/cuda` or `src/hip`), so the same source builds
for the CUDA and the HIP backend.

**Import (host):** `import wwr.wrappers.complex;`
**Include (device):** `#include "wrappers/complex/device_complex.cuh"`
**Namespace:** `wwr`

## Why this exists

The gpu\* complex layer is deliberately **operator-less**: `cuComplex` is an
operator-less `float2` aggregate where `hipComplex` is a class, so `a * b` and
brace-initialisation are not portable, and arithmetic lives in the C-style
`wwrC*` functions (`docs/architecture.md` §3). That is correct for the neutral
layer but awkward to use.

`wwr::complex<T>` buys the ergonomics back on a type this project owns, without
reintroducing the portability hazard:

```cpp
using namespace wwr;

const complex<float> z = complex<float>{1.0f, 2.0f} * complex<float>{3.0f, -1.0f};
const wwrFloatComplex v = z;               // implicit -> vendor type, for a BLAS/FFT call
const complex<float> back = to_complex(v); // and back
```

- **Aggregate `{re, im}`.** `complex<float>{1, 2}` is plain aggregate init. `T`
  is constrained to `wwr::real_fp` (`float` / `double`) and mapped to its vendor
  type through `wwr::RealToComplexType` — both from `wwr.wrappers.common`.
- **Operators** — `+ - * /`, unary `-`, `==`/`!=`, scaling and offset by a real
  scalar (either order), and compound assignment (`+= -= *= /=`, complex or
  scalar) — computed on the components as hidden friends / members. A declared
  friend or member does not disqualify the aggregate.
- **Implicit conversion** to `wwrFloatComplex` / `wwrDoubleComplex` (through the
  portable `make_wwr*Complex`, *not* a `reinterpret_cast`), so a value drops
  straight into a call site that expects one. `to_complex()` reads a vendor
  value back — layout is `{T, T}`, matching the vendor type.

The vendor `wwrC*` functions remain the spelling for a raw `wwrFloatComplex`;
this type is the convenience over them, not a replacement.

## Host module and device mirror

| File | For | Link |
|---|---|---|
| `complex.cppm` — `wwr.wrappers.complex` | host TUs (`import`) | — |
| `device_complex.cuh` | device-compiled `.cu` kernels (`#include`) | `wwr.device` |

The two are line-for-line mirrors: the same aggregate, the same bodies,
`__device__ __forceinline__` where the module says `inline`. Component
arithmetic is `constexpr` on **both** sides; only the conversion operator and
`to_complex` are non-`constexpr`, because they route through the vendor
`make_*`/accessor functions, which are not.

Two asymmetries, both forced by the header/module split:

- The header re-defines the `real_fp` concept locally, because a header cannot
  `import wwr.wrappers.common`.
- It is named `device_complex.cuh`, **not** `complex.cuh`: a same-basename
  header would shadow the gpu\* `src/complex.cuh` it must include, since a
  quoted `#include "complex.cuh"` searches its own directory first.

## Build

```
wwr_add_cxx_module_library(
  NAME wwr.wrappers.complex
  PRIMARY_INTERFACE complex.cppm
  LINK_PUBLIC wwr.complex wwr.wrappers.common
  IMPORT_STD
)
```

`device_complex.cuh` is header-only and rides `wwr.device`'s `src/` include root
(like `wrappers/math/math.cuh`), so it needs no target of its own. It is
compile-tested on both backends by `test/gpu/device_complex.cu`, a compile-only
device TU on the `compile_time_tests` target.
