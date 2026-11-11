# `experimental/rust` — a C ABI over the `wwr*` layer, for Rust FFI

**Status: experimental spike, not wired into the build.** This is a proof that a
backend-neutral C ABI (`extern "C"`) can sit over the `wwr*` `.h` route so Rust
(or any FFI consumer) links one surface and carries *zero* CUDA-vs-HIP knowledge
— no `#[cfg]`, no vendor type, no per-backend pairing table. The divergences
`docs/architecture.md` catalogues are all resolved *below* the `extern "C"` line.

It is deliberately scoped to one surface (`complex`) and the **host** side only.
See "What this does not cover" before extending it.

## Why a shim at all — Rust can already `#[cfg]` two `extern "C"` blocks

It can, for the easy names. The shim earns its place on the three things a Rust
`#[cfg]` cannot express, all exercised here against `complex`:

1. **`static inline` vendor functions have no symbol** (architecture.md §12).
   `cuComplex.h` / `amd_hip_complex.h` declare their whole families `static
   inline`, so there is nothing for Rust to link. The thunk is the only way across.
2. **By-value ABI differs** (architecture.md §3). `cuFloatComplex` is a plain
   aggregate; `hipFloatComplex` is a `HIP_vector_type` *class* — a different
   by-value calling convention. Rust gets a stable `#[repr(C)]` POD; the thunk
   translates POD↔vendor at the boundary.
3. **Behaviour differs even where names match** (architecture.md §3). ROCm's
   `hipCdiv`/`hipCabs` overflow where cuComplex scales. The HIP thunk runs
   cuComplex's scaled algorithm instead — same answer on both backends, one
   function to Rust.

## Layout

```
spec/complex.py    the neutral surface as data: the (wwr, cuda, hip) tuples —
                   the same thing src/detail/*_names.h carries — plus the POD
                   layout and per-backend body each C-ABI entry needs.
generate.py        spec -> three artefacts. One source, so nothing drifts.
gen/wwr_capi.h     vendor-free C header for bindgen / a C consumer.
gen/wwr_capi.cpp   the extern "C" thunks (committed copy is the CUDA backend).
gen/ffi.rs         the Rust side: one extern "C" block, #[repr(C)] PODs, NO cfg.
driver.c           a pure-C consumer (stands in for Rust's FFI) that proves the
                   symbols link and the numbers are right.
```

`gen/` is a committed snapshot so the PR is reviewable without running anything;
regenerate it with the commands below.

## Build & run (host g++, no nvcc, CUDA headers only)

```bash
python3 generate.py cuda                       # (re)generate gen/ for CUDA
cd gen
g++ -std=c++23 -O2 -fPIC -I/usr/local/cuda/include -shared \
    wwr_capi.cpp -o libwwr_capi.so
nm -D -C libwwr_capi.so | grep wwr_            # unmangled C symbols
gcc -std=c11 -I. ../driver.c -L. -lwwr_capi -o driver
LD_LIBRARY_PATH=. ./driver                     # prints add/mul/abs/div, all correct
```

`python3 generate.py hip` emits the HIP thunks (cabs→hypotf, cdiv→scaled); they
need real ROCm headers to build, which live only in the `wwr:hip` image.

## What this does NOT cover (yet)

- **Host surface only.** The high-value `wwr*` code — `src/extension`'s device
  kernels, `parallel_for`, the sdcz dispatch in `src/wrappers` — is device C++
  *templates*. Each needs a monomorphised C entry per instantiation
  (`wwr_random_normal_f32`, …). The generator pattern extends to it; this spike
  does not implement it.
- **Not a CMake target.** No `wwr.capi` library, no soname/versioning, no
  install rules. Once Rust links it, symbol stability becomes a contract to own.
- **The spec is hand-written here.** A real surface would generate the spec FROM
  the vendor headers with libclang rather than restate the tuples.
- **Not validated on real HIP, and no `rustc` in the loop.** The HIP thunks were
  compiled against a class-type stub; `ffi.rs` is generated but not compiled.
  `driver.c` stands in for the Rust ABI exactly.
