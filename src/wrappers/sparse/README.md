# src/wrappers/sparse — Type-Safe GPU Sparse Extension

This directory provides the C++23 module `wwr.wrappers.sparse`, which wraps
the legacy typed (S/D/C/Z) sparse operations cuSPARSE and hipSPARSE have in
common in a single generic C++ template function. Callers write `bsrmv<float>(...)`
instead of `cusparseSbsrmv(...)` / `hipsparseSbsrmv(...)`; the correct typed
function is selected at compile time via `if constexpr` dispatch.

It is backend-neutral: written once against `src/sparse`'s `gpusparse*` names
(`gpusparseSbsrmv` is `cusparseSbsrmv` on a CUDA build and `hipsparseSbsrmv` on a
HIP build), so the same source builds for either `WWR_GPU_BACKEND`.

**Import:** `import wwr.wrappers.sparse;`
**Namespace:** `wwr`

## Scope: the shared typed API only

This module wraps the S/D/C/Z-prefixed functions that (a) exist in both cuSPARSE
and hipSPARSE and (b) cuSPARSE has not removed or deprecated. That is a small,
current slice of cuSPARSE — the library removed most of its legacy typed API
(`csrmv`, `csrsv2`, `csrmm`, `csrsm2`, `csric02`, `csrilu02`, ...) in CUDA 12, so
those are absent here even though hipSPARSE keeps them, on the same
"adapt what both vendors offer" policy the BLAS and solver extensions use.

The **modern generic API** (`SpMV`, `SpMM`, `SpGEMM`, `SDDMM`, `SpSV`, `SpSM`,
the descriptor-creation routines) is **not** wrapped here. Its element type is a
runtime `cudaDataType`/`hipDataType` argument, not a name letter, so there is no
S/D/C/Z entry point to dispatch to and nothing for a token-paste wrapper to add.
It is fully usable directly from `wwr.sparse` / the raw vendor modules.

## Backend differences, and where they are resolved

Almost all live in `src/sparse.cppm`, not here:

- **Names.** `cusparse<X><name>` and `hipsparse<X><name>` both map to
  `gpusparse<X><name>`.
- **`gebsr2gebsc_bufferSize` / `csr2gebsr_bufferSize` buffer size.** cuSPARSE
  writes the byte count as `int*`, hipSPARSE as `size_t*`. The wrappers here take
  `size_t*` on both backends; `src/sparse.cppm` forwards through an `int` on
  CUDA.
- **cuSPARSE-only / hipSPARSE-only functions.** Not wrapped here or in
  `src/sparse`; see that module's file header.

The wrappers take the raw `gpusparseHandle_t` and return the raw
`gpusparseStatus_t` — a caller creates, destroys and error-checks the handle
itself. RAII ownership and typed error handling live in the sibling
`wwr.extension.sparse` module (`:sparse_handle` wraps `gpusparseHandle_t`;
`:sparse_error` specialises the error policy for `gpusparseStatus_t`).

## Module Partitions

### `:level_2` — `level_2.cppm` — Matrix-Vector

| Function | Operation |
|---|---|
| `bsrmv<T>(...)` | BSR matrix-vector multiply: `y = alpha*op(A)*x + beta*y` |

### `:solvers` — `solvers.cppm` — Tridiagonal / Pentadiagonal Batch Solvers

| Function | Operation |
|---|---|
| `gtsv2<T>` / `gtsv2_bufferSizeExt<T>` | Tridiagonal solve `A*X = B` (with pivoting) |
| `gtsv2_nopivot<T>` / `..._bufferSizeExt<T>` | Tridiagonal solve without pivoting |
| `gtsv2StridedBatch<T>` / `..._bufferSizeExt<T>` | Batched tridiagonal solve, strided storage |
| `gtsvInterleavedBatch<T>` / `..._bufferSizeExt<T>` | Batched tridiagonal solve, interleaved storage |
| `gpsvInterleavedBatch<T>` / `..._bufferSizeExt<T>` | Batched pentadiagonal solve, interleaved storage |

### `:extra` — `extra.cppm` — CSR Matrix Addition

| Function | Operation |
|---|---|
| `csrgeam2<T>` / `csrgeam2_bufferSizeExt<T>` | `C = alpha*A + beta*B` (CSR) |

The untyped structure query `Xcsrgeam2Nnz` is called directly through the raw
module (it takes no element type, so there is nothing to dispatch).

### `:conversion` — `conversion.cppm` — Format Conversion

| Function | Operation |
|---|---|
| `nnz<T>(...)` | Count nonzeros per row/column and in total, from a dense matrix |
| `gebsr2gebsc<T>` / `gebsr2gebsc_bufferSize<T>` | General-BSR to general-BSC (block transpose) |
| `csr2gebsr<T>` / `csr2gebsr_bufferSize<T>` | CSR to general-BSR |

## Template Instantiation

Explicit instantiations are written by hand for the type set (`float`, `double`,
`gpuFloatComplex`, `gpuDoubleComplex`). The `extern template` declarations live
next to each function in `level_2.cppm`, `solvers.cppm`, `extra.cppm` and
`conversion.cppm`; the matching `template` instantiations live in
`instantiations.cpp`. A new instantiation also needs its entry in
`test/wrappers/sparse/sparse_dispatch.toml` (see Tests), or the build fails.

## Tests

- `test/wrappers/sparse/sparse_dispatch.toml` — build-time, both backends, no
  GPU: `test/shared/dispatch.py` disassembles this target's objects and checks
  that every explicit instantiation calls exactly the `gpusparse*` function the
  table names. Catches a basename mixed up with one of the same signature
  (`gtsv2`/`gtsv2_nopivot`, `bsrmv` vs another BSR routine) or a dispatch branch
  missing.
- `test/gpu/sparse.cppm` — compile-time checks on `src/sparse`'s names.

There are no runtime tests, by design: a wrapper's body is the token-paste
forward and nothing else, so the pasted name is the whole of what it can get
wrong, and the dispatch check proves that for every instantiation without a
device.

## Build

```
wwr_add_cxx_module_library(
  NAME wwr.wrappers.sparse
  PRIMARY_INTERFACE interface.cppm
  PARTITIONS level_2.cppm solvers.cppm extra.cppm conversion.cppm
  IMPLEMENTATION instantiations.cpp
  LINK_PUBLIC wwr.sparse wwr.complex wwr.wrappers.common
  IMPORT_STD
)
```

`dispatch_macros.h` (the `gpusparse`-prefixed `WWR_USUAL_DISPATCH`) sits next to the
sources and is included same-dir; it builds on the prefix-agnostic
`WWR_REAL_DISPATCH` / `WWR_COMPLEX_DISPATCH` cores shared from `wrappers/common/dispatch_sdcz.h`.
