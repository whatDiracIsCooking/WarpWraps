# `experimental/rust` — a C ABI over the `wwr` layer, for Rust FFI

**Status: experimental, wired into the build as `wwr.capi`.** This is a
backend-neutral C ABI (`extern "C"`) over the `wwr` layer so Rust (or any FFI
consumer) links one surface and carries *zero* CUDA-vs-HIP knowledge — no
`#[cfg]`, no vendor type, no per-backend pairing table. The divergences
`docs/architecture.md` catalogues are all resolved *below* the `extern "C"` line,
inside the `wwr` functions the thunk calls. The ABI is not yet a stable contract.

It is deliberately scoped to one surface (`complex`) and the **host** side only.
See "What this does not cover" before extending it.

## The one decision that shapes everything: the thunk calls *into* `wwr`

The thunk does not re-derive the backend divergence; it calls the host-callable
`wwr` functions that already resolve it once — `wwr::wwrCaddf` / `wwrCmulf` /
`wwrCabsf` / `wwrCdivf` and the `double` set, exported from the `wwr.complex`
module (architecture.md §3, §12, §25). So the generated `wwr_capi.cpp`:

- `import wwr.complex;` — it names **no vendor header** and has **no CUDA-vs-HIP
  `#ifdef`**. It is one shape over both backends.
- converts each POD arg to/from the neutral `wwr::wwrFloatComplex` /
  `wwr::wwrDoubleComplex` at the boundary (brace `{re, im}` in, `.x`/`.y` out —
  both backend-neutral per `complex.h`), and forwards into the `wwr` function.
- **The backend is chosen only by which `wwr` library `wwr.capi` links.** The
  split lives entirely on the link line, nowhere in the source.

Because the divergence is gone from *above* the `wwr::` call, every function is a
plain forward: the spike's `body: forward | custom` + `CUSTOM` mechanism and the
per-backend `.cpp` generation are gone, and the spec shrinks to
`(capi_name, wwr_name, POD_layout)`.

## Why a shim at all — Rust can already `#[cfg]` two `extern "C"` blocks

It can, for the easy names. The shim earns its place on three things a Rust
`#[cfg]` cannot express, all exercised here against `complex`:

1. **`static inline` vendor functions have no symbol** (architecture.md §12).
   `cuComplex.h` / `amd_hip_complex.h` declare their whole families `static
   inline`, so there is nothing for Rust to link. The thunk is the only way across
   — and it crosses by calling `wwr`, whose module re-exposes those static-inline
   families as external-linkage host wrappers.
2. **By-value ABI differs** (architecture.md §3). `cuFloatComplex` is a plain
   aggregate; `hipFloatComplex` is a `HIP_vector_type` *class* — a different
   by-value calling convention. Rust gets a stable `#[repr(C)]` POD; the thunk
   translates POD↔neutral at the boundary.
3. **Behaviour differs even where names match** (architecture.md §3). ROCm's
   `hipCdiv`/`hipCabs` overflow where cuComplex scales. The `wwr::wwrCdivf` /
   `wwrCabsf` the thunk calls already runs the scaled algorithm on HIP — one
   function to Rust, same answer on both backends.

## Layout

```
spec/complex.py    the neutral surface as data: (capi_name, wwr_name, POD layout).
                   Provisional scaffolding until it is sourced from the canonical
                   name surface (src/detail/*_names.h); architecture.md §25, #328.
generate.py        spec -> three artefacts. One source, so nothing drifts.
gen/wwr_capi.h     vendor-free C header for bindgen / a C consumer.
gen/wwr_capi.cpp   the backend-neutral extern "C" thunks (import wwr.complex).
gen/ffi.rs         the Rust side: one extern "C" block, #[repr(C)] PODs, NO cfg.
capi/CMakeLists.txt  the wwr.capi shared-library target + the driver ctest.
driver.c           a pure-C consumer (stands in for Rust's FFI) that links the
                   symbols and checks the numbers; run as the wwr.capi.driver ctest.
```

`gen/` is a committed snapshot so the PR is reviewable without running anything;
regenerate it with `python3 capi/generate.py` (no backend argument — the `.cpp`
is backend-neutral).

## Build & run

The standalone host-`g++` path is gone: "call into `wwr`" means the capi builds
against the module graph, so it builds as a CMake target like any other.

```bash
cmake --preset default            # CUDA (or --preset hip, inside the wwr:hip image)
cmake --build build --target wwr.capi wwr.capi.driver
ctest --test-dir build -R wwr.capi   # runs driver.c; checks add/mul/abs/div
```

`wwr.capi` (real dotted name, aliased `wwr::capi`) is a shared library gated ON
for a top-level build by `WWR_BUILD_EXPERIMENTAL`; the `wwr.capi.driver` ctest is
gated by `WWR_BUILD_TESTS` and is host-only, so it runs on no-GPU CI.

## What this does NOT cover (yet)

- **Host surface only.** The high-value `wwr` code — `src/extension`'s device
  kernels, `parallel_for`, the sdcz dispatch in `src/wrappers` — is device C++
  *templates*. Each needs a monomorphised C entry per instantiation; a later
  milestone (architecture.md §25, #329 out-of-scope).
- **No status-returning surface yet.** `complex` never forces mapping the `wwr`
  status enum to a stable C `int` so Rust can build a `Result`; that is #329
  increment 2.
- **Not installed.** `wwr.capi` is built but not part of the `find_package(wwr)`
  package — `wwr_install_package()` sweeps `src/` only. A later increment promotes
  it once the Rust/status story settles (#329).
- **The spec is hand-written here**, provisional until sourced from the canonical
  name surface (architecture.md §25, #328).
- **No `rustc` in the loop.** `ffi.rs` is generated but not compiled; `driver.c`
  stands in for the Rust ABI exactly. That is #329 increment 3 (optional).
