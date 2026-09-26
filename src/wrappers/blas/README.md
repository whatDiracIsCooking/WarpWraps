# src/wrappers/blas — Type-Safe GPU BLAS Extension

This directory provides the C++23 module `gpumod.wrappers.blas`, which
wraps every BLAS operation cuBLAS and hipBLAS have in common in a single
generic C++ template function. Callers write `gemm<float>(...)` instead of
`cublasSgemm_v2(...)` / `hipblasSgemm(...)`; the correct typed function is
selected at compile time via `if constexpr` dispatch.

It is backend-neutral: written once against `src/blas`'s `gpublas*` names
(`gpublasSgemm` is `cublasSgemm_v2` on a CUDA build and `hipblasSgemm` on a HIP
build), so the same source builds for either `GPUMOD_GPU_BACKEND`.

**Import:** `import gpumod.wrappers.blas;`
**Namespace:** `gpumod`

## Backend differences, and where they are resolved

None of them are resolved here -- all live in `src/blas.cppm`:

- **Names.** cuBLAS's `_v2` suffix and hipBLAS's bare names both map to
  `gpublas<X>`; dispatch basenames here carry no `_v2`.
- **Status strings.** hipBLAS has only `hipblasStatusToString`;
  `gpublasGetStatusName` and `gpublasGetStatusString` both map to it on HIP.
- **`getrsBatched` / `getriBatched` constness.** hipBLAS declares the input
  arrays (and `getriBatched`'s pivots) non-const. The wrappers here keep
  cuBLAS's const-correct signature on both backends; on HIP,
  `src/blas.cppm` supplies forwarding functions that `const_cast` into
  hipBLAS (which only reads them).
- **cuBLAS-only functions.** `gemm3m`, `gemmGroupedBatched`, `matinvBatched`,
  `tpttr` and `trttp` have no hipBLAS counterpart, so they are not wrapped here
  or anywhere — this module adapts what both vendors offer. Call them through
  `import gpumod.cuda.cublas_v2;`, which exposes the whole cuBLAS API on a CUDA
  build. The repository README lists them under "Vendor-only functions".

The wrappers take the raw `gpublasHandle_t` and return the raw `gpublasStatus_t`
— a caller creates, destroys and error-checks the handle itself. RAII ownership
and typed error handling live in the sibling `gpumod.extension.blas` module
(`:blas_handle` wraps `gpublasHandle_t`; `:blas_error` specialises the error
policy for `gpublasStatus_t`).

## Module Partitions

### `:type_traits` — `type_traits.cppm`

Re-exports concepts from `gpumod.wrappers.common` and adds:

```cpp
// Constrains IntT to int, int64_t, or anything convertible to either
template<typename T>
concept int_type = ...;

// Maps gpuHalf and gpuBfloat16 to float (used by gemvStridedBatched)
template<half_fp T> struct GetSinglePrecisionType { using type = T; };
template<> struct GetSinglePrecisionType<gpuHalf>     { using type = float; };
template<> struct GetSinglePrecisionType<gpuBfloat16> { using type = float; };

template<half_fp T>
using SinglePrecisionType = typename GetSinglePrecisionType<T>::type;
```

Also re-exports `gpuFloatComplex`, `gpuDoubleComplex`, `gpuComplex`, and
`std::int64_t` for use by callers that import only this extension.

### `:level_1` — `level_1.cppm` — Vector-Vector Operations

All functions are templated on `usual_fp T` (or a sub-concept) and `int_type IntT`.
The `IntT` parameter selects between the 32-bit (`int`) and 64-bit (`int64_t`,
`_64`-suffixed) entry points automatically.

| Function | Operation |
|---|---|
| `iamax<T,IntT>(handle, n, x, incx, result)` | Index of element with maximum absolute value |
| `iamin<T,IntT>(handle, n, x, incx, result)` | Index of element with minimum absolute value |
| `asum<T,IntT>(handle, n, x, incx, result)` | Sum of absolute values; result is `RealType<T>` |
| `axpy<T,IntT>(handle, n, alpha, x, incx, y, incy)` | `y = alpha*x + y` |
| `copy<T,IntT>(handle, n, x, incx, y, incy)` | Vector copy |
| `dot<T,IntT>(handle, n, x, incx, y, incy, result)` | Dot product (real types) |
| `dotc<T,IntT>(...)` | Conjugated dot product (complex types) |
| `dotu<T,IntT>(...)` | Unconjugated dot product (complex types) |
| `nrm2<T,IntT>(handle, n, x, incx, result)` | Euclidean norm; result is `RealType<T>` |
| `rot<T,IntT>(...)` | Apply Givens rotation (multiple overloads for real/complex `s`) |
| `rotg<T>(handle, a, b, c, s)` | Generate Givens rotation parameters |
| `rotm<T,IntT>(...)` | Apply modified Givens rotation (real only) |
| `rotmg<T>(...)` | Generate modified Givens rotation parameters (real only) |
| `scal<T,IntT>(handle, n, alpha, x, incx)` | `x = alpha*x`; overload with `RealType<T>* alpha` for complex |
| `swap<T,IntT>(handle, n, x, incx, y, incy)` | Swap two vectors |

### `:level_2` — `level_2.cppm` — Matrix-Vector Operations

| Function | Operation |
|---|---|
| `gemv` | General matrix-vector: `y = alpha*op(A)*x + beta*y` |
| `gbmv` | General banded matrix-vector multiply |
| `ger` | Rank-1 update (real): `A = alpha*x*y^T + A` |
| `geru` | Rank-1 update (complex, unconjugated) |
| `gerc` | Rank-1 update (complex, conjugated) |
| `symv` | Symmetric matrix-vector multiply (real) |
| `syr` / `syr2` | Symmetric rank-1/rank-2 update (real) |
| `sbmv` | Symmetric banded matrix-vector multiply (real) |
| `spmv` | Symmetric packed matrix-vector multiply (real) |
| `spr` / `spr2` | Symmetric packed rank-1/rank-2 update (real) |
| `trmv` | Triangular matrix-vector multiply |
| `trsv` | Triangular solve: `op(A)*x = b` |
| `tbmv` / `tbsv` | Triangular banded matrix-vector multiply / solve |
| `tpmv` / `tpsv` | Triangular packed matrix-vector multiply / solve |
| `hemv` | Hermitian matrix-vector multiply (complex) |
| `hbmv` | Hermitian banded matrix-vector multiply (complex) |
| `hpmv` | Hermitian packed matrix-vector multiply (complex) |
| `her` / `her2` | Hermitian rank-1/rank-2 update (complex) |
| `hpr` / `hpr2` | Hermitian packed rank-1/rank-2 update (complex) |
| `gemvBatched` | Batched general matrix-vector multiply |
| `gemvStridedBatched` | Strided batched general matrix-vector multiply |

### `:level_3` — `level_3.cppm` — Matrix-Matrix Operations

| Function | Operation |
|---|---|
| `gemm` | General matrix-matrix: `C = alpha*op(A)*op(B) + beta*C` |
| `gemmBatched` | Batched GEMM (pointer-array variant) |
| `gemmStridedBatched` | Batched GEMM with uniform stride |
| `symm` | Symmetric matrix-matrix multiply |
| `syrk` | Symmetric rank-k update: `C = alpha*op(A)*op(A)^T + beta*C` |
| `syr2k` | Symmetric rank-2k update |
| `syrkx` | Symmetric rank-k update variant |
| `trmm` | Triangular matrix-matrix multiply |
| `trsm` | Triangular solve with multiple RHS |
| `trsmBatched` | Batched triangular solve |
| `hemm` | Hermitian matrix-matrix multiply (complex) |
| `herk` | Hermitian rank-k update (complex) |
| `her2k` | Hermitian rank-2k update (complex) |
| `herkx` | Hermitian rank-k update variant (complex) |

### `:extension` — `extension.cppm` — BLAS-like Extensions

| Function | Operation |
|---|---|
| `geam` | Matrix addition/transposition: `C = alpha*op(A) + beta*op(B)` |
| `dgmm` | Diagonal matrix multiply: `C = A * diag(x)` or `diag(x) * A` |
| `getrfBatched` | Batched LU factorisation |
| `getrsBatched` | Batched LU solve |
| `getriBatched` | Batched matrix inversion via LU |
| `geqrfBatched` | Batched QR factorisation |
| `gelsBatched` | Batched least-squares solve |

## Template Instantiation

Explicit instantiations are written by hand for the type set (`float`, `double`,
`gpuFloatComplex`, `gpuDoubleComplex`), each crossed with an index type
(`int`, `int64_t`) where the underlying function is templated on both. The
`extern template` declarations live in `level_1.cppm`, `level_2.cppm`,
`level_3.cppm`, and `extension.cppm`, next to each function; the matching
`template` instantiations live in `instantiations.cpp`.

A new instantiation also needs its entry in
`test/wrappers/blas/blas_dispatch.toml` (see Tests), or the build fails.

## Tests

- `test/wrappers/blas/blas_dispatch.toml` -- build-time, both backends, no
  GPU: `test/shared/dispatch.py` disassembles this target's objects and checks
  that every explicit instantiation calls exactly the gpublas function the
  table names. Catches what the types let through: an `int` widened into a
  `_64` entry point, a basename mixed up with one of the same signature
  (`iamax`/`iamin`), a dispatch branch missing.
- `test/gpu/blas.cppm` -- compile-time checks on `src/blas`'s names.

There are no runtime tests, by design: a wrapper's body is the token-paste
forward and nothing else, so the pasted name is the whole of what it can get
wrong, and the dispatch check proves that for every instantiation without a
device.

## Build

```
gpumod_add_cxx_module_library(
  NAME gpumod.wrappers.blas
  PRIMARY_INTERFACE interface.cppm
  PARTITIONS type_traits.cppm
             level_1.cppm level_2.cppm level_3.cppm extension.cppm
  IMPLEMENTATION instantiations.cpp
  LINK_PUBLIC gpumod.blas gpumod.complex gpumod.fp16
              gpumod.bf16 gpumod.wrappers.common
  IMPORT_STD
)
```

`dispatch_macros.h` (the `gpublas`-prefixed `GPUMOD_USUAL_DISPATCH` plus the `_64`
index variants) sits next to the sources and is included same-dir; it
builds on the prefix-agnostic `GPUMOD_REAL_DISPATCH` / `GPUMOD_COMPLEX_DISPATCH` cores shared
from `wrappers/common/dispatch_sdcz.h`.
