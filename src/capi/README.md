# `src/capi` — a C ABI over the `wwr` layer, for Rust FFI

A backend-neutral C ABI (`extern "C"`) over the `wwr` layer so Rust (or any FFI
consumer) links one surface and carries *zero* CUDA-vs-HIP knowledge — no
`#[cfg]`, no vendor type, no per-backend pairing table. The divergences
`docs/architecture.md` catalogues are all resolved *below* the `extern "C"` line,
inside the `wwr` functions the thunk calls.

Unlike the rest of `wwr` (which ships as module *source* a consumer compiles —
BMIs are not portable), the capi is a **prebuilt `libwwr_capi.so`** a non-C++
consumer links. It is installed as part of the `find_package(wwr)` package under
`-DWWR_INSTALL_CAPI=ON` (see "Install", below).

It covers two surfaces (`complex` and a thin status-returning slice of
`runtime_api`), **host** side only. See "What this does not cover" before
extending it.

## The status surface, and the one problem `complex` never forces

`complex` returns values; it never returns a status. The runtime slice
(`wwr_runtime_get_version`, `wwr_get_device_count`, `wwr_set_device`) does, and
that forces the design `complex` could dodge: the backend error-enum **values**
differ even where the names agree (architecture.md §6), so the ABI must **not**
pass the backend's raw status int through to Rust. The thunk's internal `_map`
helper turns the neutral `wwr::wwrError_t` into a **stable `wwr_status` C enum
with fixed integer values** (`WWR_STATUS_SUCCESS == 0`, a curated error set, a
`WWR_STATUS_UNKNOWN` catch-all) — resolved **below** the `extern "C"` line,
exactly as the neutral POD resolves the by-value divergence for `complex`. Rust
sees one `#[repr(i32)]` enum whichever backend `wwr.capi` links, and builds a
`Result` from it. The enum and the mapper are generated from one list
(`spec.STATUS`), so they cannot drift, and `_map` compares only the neutral
`wwr::wwr<Name>` constants — never a vendor name — so the thunk stays one shape
over both backends.

The `wwr_status` enumerator values are a **frozen ABI contract** (see "Stability
policy", below, and architecture.md §21).

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
plain forward: there is no per-backend `.cpp` generation, and the spec is just
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
spec/complex.py    the neutral surface as data: PODs (capi_name, wwr_name, layout),
                   the stable wwr_status enum map (C enumerator, wwr constant), and
                   the functions (complex + the runtime status slice). Provisional
                   scaffolding until it is sourced from the canonical name surface
                   (src/detail/*_names.h); architecture.md §25, #328.
generate.py        spec -> three artefacts. One source, so nothing drifts.
gen/wwr_capi.h     vendor-free C header for bindgen / a C consumer (PODs, the
                   wwr_status enum, prototypes). Installed under WWR_INSTALL_CAPI.
gen/wwr_capi.cpp   the backend-neutral extern "C" thunks (import wwr.complex +
                   wwr.runtime_api; the _map status helper).
gen/ffi.rs         the Rust side: one extern "C" block, #[repr(C)] PODs, the
                   #[repr(i32)] wwr_status enum, NO cfg.
CMakeLists.txt     the wwr.capi shared-library target, the install rules, and the
                   driver ctests.
driver.c           a pure-C consumer (stands in for Rust's FFI) that links the
                   symbols and checks the numbers; run as the wwr.capi.driver ctest.
driver.rs          the real Rust consumer: include!s gen/ffi.rs and links
                   libwwr_capi.so, compiled rustc-direct (no cargo) and run as the
                   wwr.capi.driver_rs ctest — rustc-gated (see "rustc in the loop").
wwr-sys/           the idiomatic Rust -sys crate (Cargo.toml + build.rs + src/):
                   the user-consumable binding over the installed libwwr_capi.so.
                   See "The wwr-sys crate", below.
```

`gen/` is a committed snapshot so the PR is reviewable without running anything;
regenerate it with `python3 src/capi/generate.py` (no backend argument — the
`.cpp` is backend-neutral).

## Build & run

"Call into `wwr`" means the capi builds against the module graph, so it builds as
a CMake target like any other. It builds **unconditionally** with the rest of
`src/` (it depends only on `wwr::complex` / `wwr::runtime_api`, which always
build); the install is the only thing gated (see below).

```bash
cmake --preset default            # CUDA (or --preset hip, inside the wwr:hip image)
cmake --build build --target wwr.capi wwr.capi.driver
ctest --test-dir build -R wwr.capi   # runs driver.c (+ driver_rs if rustc is found)
```

`wwr.capi` (real dotted name, aliased `wwr::capi`) is a shared library. Its
output file is `libwwr_capi.so` (an `OUTPUT_NAME` override, matching `ffi.rs`'s
`wwr_capi` link name and the C ABI's `wwr_` symbol prefix) while the target keeps
the dotted name. The `wwr.capi.driver` ctest is gated by `WWR_BUILD_TESTS` and is
host-only, so it runs on no-GPU CI.

## Install

The capi is a prebuilt binary, not module source, so it does not ride the
`src/`-sweep that packages every other `wwr` library. `wwr_install_package()`
excludes `wwr.capi` from the sweep (the sweep's non-recursive `src/` collection
never reaches a target defined in this subdirectory, like `src/thrust`) and
installs it **explicitly**, gated on `-DWWR_INSTALL_CAPI=ON` (mirroring
`WWR_INSTALL_EXTENSION`; effect only when `WWR_INSTALL` is ON). Under the gate:

- `install(TARGETS wwr.capi EXPORT wwr-targets ...)` puts `libwwr_capi.so` (+ its
  soname) in the lib dir and adds `wwr::capi` to the exported target set, so a
  consumer's `find_package(wwr)` + `target_link_libraries(... wwr::capi)`
  resolves;
- `install(FILES gen/wwr_capi.h ...)` puts the vendor-free C header in the `wwr`
  include dir.

A consumer that links `wwr::capi` compiles no module unit — it links the prebuilt
`.so` and `#include`s `wwr_capi.h` (or generates bindings from it). The `wwr-sys`
Rust crate skeleton installs here too (`share/wwr/rust/wwr-sys/`); see "The
wwr-sys crate", above.

## rustc in the loop

`driver.rs` is the real Rust consumer the C driver stands in for: it `include!`s
the generated `gen/ffi.rs` and links `libwwr_capi.so`, compiled rustc-direct (no
cargo) and run as the `wwr.capi.driver_rs` ctest. It mirrors `driver.c`'s
assertions — the complex arithmetic and the stable `wwr_status` surface — so the
two stand in for each other.

It is gated on `find_program(RUSTC rustc)`, **not** merely on `WWR_BUILD_TESTS`,
and that gate follows the repo's image/CI contract. `ci.yml`'s `cpp` jobs run
inside the container image `main` already published; `container:` is resolved
before any step, so a PR that adds rust to `docker/` is still tested against the
**old, rust-less** image (the `images.yml` PR check only *builds* the new
Dockerfile, it does not publish it — see both workflow headers). With no `rustc`
there, the Rust test cleanly does not register and CMake prints a visible skip,
exactly like the repo's GPU-gated tests. Once `images.yml` republishes the
rust-enabled base on `main`, every later `cpp` run finds `rustc` and runs it. The
toolchain is pinned in `docker/Dockerfile.base` (`ARG RUST_VERSION`), installed
via `rustup --profile minimal` to a shared `/opt/cargo` + `/opt/rustup` so
cuda/hip/combined all inherit it.

## The wwr-sys crate

`wwr-sys/` is the idiomatic Rust `-sys` crate over the prebuilt `libwwr_capi.so`
(#340). A Rust user adds one dependency, links the `.so`, and carries **zero**
`#[cfg]` / vendor knowledge — the backend was chosen when the linked `.so` was
built, exactly the ABI's thesis.

- **`Cargo.toml`** — crate `wwr-sys`, version `1.0.0` tracking the frozen ABI
  contract (not bumped independently), `links = "wwr_capi"` (the `-sys`
  convention — cargo enforces one crate per native lib).
- **`src/lib.rs`** — the binding surface. It `include!`s `src/ffi.rs`, the
  **single source**: `generate.py` writes that `ffi.rs` from the same
  `spec/complex.py` it writes `gen/ffi.rs` / the C header / the thunk from, so
  the crate cannot drift from the ABI. (The crate ships its own copy because,
  once installed away from this tree, a sibling `../gen/ffi.rs` `include!` would
  dangle.) Do not hand-edit `ffi.rs`; regenerate with `python3 src/capi/generate.py`.
- **`build.rs`** — reads `WWR_CAPI_LIB_DIR` and emits `rustc-link-search` +
  `rustc-link-lib=dylib=wwr_capi` (and an rpath, so the test binary finds the
  `.so` without `LD_LIBRARY_PATH`). The link name is the lib's `OUTPUT_NAME`
  `wwr_capi` — kept in step with `src/capi/CMakeLists.txt`, `Cargo.toml`'s
  `links`, and `ffi.rs`'s `#[link]` by a comment on each (the increment-3 lesson).

It is the raw `-sys` surface (every function `unsafe extern "C"`); a safe
`Result`-returning wrapper is deliberately out of scope. A crate-level smoke test
(`cargo test`) links the `.so` and calls through it on a no-GPU host.

**Build it against the built or installed lib:**

```bash
cmake --preset default                       # or --preset hip (inside wwr:hip)
cmake --build build --target wwr.capi        # produces libwwr_capi.so
WWR_CAPI_LIB_DIR=$PWD/build/src/capi \
  cargo test --manifest-path src/capi/wwr-sys/Cargo.toml
```

**Depend on the installed crate.** Under `-DWWR_INSTALL_CAPI=ON` the crate
skeleton installs to `<prefix>/share/wwr/rust/wwr-sys/` (`Cargo.toml`, `build.rs`,
`src/`) alongside `libwwr_capi.so` in `<prefix>/lib`. A downstream `Cargo.toml`:

```toml
[dependencies]
wwr-sys = { path = "<prefix>/share/wwr/rust/wwr-sys" }   # or a git dependency
```

then build with the lib dir pointed at the install:

```bash
WWR_CAPI_LIB_DIR=<prefix>/lib cargo build
```

That exact path is exercised end to end by `example/capi-consumer-rs` (#341): a
standalone crate that path-depends on the installed `wwr-sys`, links the
installed `libwwr_capi.so` and asserts `wwr_cadd64` / `wwr_runtime_get_version`.
`devtools/install-check.sh --capi` builds and runs it against a throwaway install
prefix (with an `ldd` check that it links the *installed* `.so`), gated on rustc
so the pre-rust-image CI path stays green.

## Stability policy

The `wwr_status` enumerator *values* are a frozen contract: once shipped, an
enumerator keeps its integer value forever, so a consumer linking any future
`libwwr_capi.so` reads the same `#[repr(i32)]` mapping. New error codes append
with the next value (never reorder, never reuse). `SOVERSION` bumps only on an
ABI break — a changed struct layout, a changed signature, a removed symbol —
never for an additive change. See architecture.md §21.

## What this does NOT cover (yet)

- **Host surface only.** The high-value `wwr` code — `src/extension`'s device
  kernels, `parallel_for`, the sdcz dispatch in `src/wrappers` — is device C++
  *templates*. Each needs a monomorphised C entry per instantiation; a later
  milestone (architecture.md §25).
- **The spec is hand-written here**, provisional until sourced from the canonical
  name surface (architecture.md §25, #328).
