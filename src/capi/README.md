# `src/capi` — a C ABI over the `wwr` layer, for Rust FFI

A backend-neutral C ABI (`extern "C"`) over the `wwr` layer so Rust (or any FFI
consumer) links one surface and carries *zero* CUDA-vs-HIP knowledge — no
`#[cfg]`, no vendor type, no per-backend pairing table. The divergences
`docs/architecture.md` catalogues are all resolved *below* the `extern "C"` line,
inside the `wwr` functions the thunk calls.

Unlike the rest of `wwr` (which ships as module *source* a consumer compiles —
BMIs are not portable), the capi is a **prebuilt shared object** a non-C++
consumer links. It is installed as part of the `find_package(wwr)` package under
`-DWWR_INSTALL_CAPI=ON` (see "Install", below).

**It is split per surface** so each surface's `.so` links only *its* vendor dep
(`libwwr_capi_complex.so` NEEDs no math library; `libwwr_capi_blas.so` NEEDs
cuBLAS / hipBLAS), mirroring the project's own per-`.cppm` dependency isolation —
plus an **umbrella `libwwr_capi.so`** that contains every unit's symbols, so an
existing `-lwwr_capi` consumer is unchanged (see "The per-unit split", below).

It covers seven surfaces: `complex`, a thin status-returning slice of
`runtime_api`, and five full curated vendor libraries — `blas` (cuBLAS /
hipBLAS), `blaslt` (cuBLASLt / hipBLASLt), `solver` (cuSOLVER Dn / hipSOLVER),
`rand` (cuRAND / hipRAND) and `rtc` (nvRTC / hipRTC). `complex`, `runtime_api`
and `rtc` are **host**-side; `blas` / `blaslt` / `solver` / `rand` execute on the
**device** at call time (a handle binds a device, a GEMM launches), so their
thunks are link-checked on no-GPU CI and run-checked on a GPU box. The five
library surfaces share ONE fragment-sourced path (`spec/library.py`'s
`LibrarySpec` + the generic emitters in `generate.py`); each `spec/<x>.py` is a
declaration. See "What this does not cover" before extending it.

## How the surface is sourced (and the one surface that is not)

The C ABI is a **consumption path** over the repo's one canonical name surface,
not a restatement of it (architecture.md §25). The generator takes two inputs and
cross-references them, so nothing hand-maintained can drift:

1. **Names come from `src/detail/*_names.h`.** These macro fragments are the
   authoritative `(wwr, cuda, hip)` surface every consumer binds to. `generate.py`
   parses one as text (`spec/fragments.py`, the way `devtools/header_intersection.py`
   reads the manifests — stdlib `re`, no clang), reproducing the preprocessor
   exactly: `WWR_RT_<KIND>(Stem)` expands to the stem with each backend's prefix,
   and `WWR_<KIND>_RAW(wwrX, cudaX, hipX)` carries all three literally. Delete a
   name from the fragment and it leaves the generated header, thunk and `ffi.rs`
   (a surface that still *requests* it then fails the build loudly, rather than
   inventing it).
2. **Signatures + enum values come from the committed `vendor/<pin>/*.json`
   manifests** (`vendor_harvest.py`'s output, SDK-free, already in CI).
   `spec/manifest.py` looks up the fragment's selected-backend vendor name and
   reads the `decl` clang recorded (e.g.
   `cudaError_t cudaRuntimeGetVersion(int * runtimeVersion)`) or an enum
   constant's folded `value`. No signature is hand-written.

`spec/captypes.py` is the extensible core that turns a vendor `decl` into the
ABI's vendor-free representation — the machinery a later surface (#349 blas)
reuses as-is:

| vendor type | capi representation |
|---|---|
| scalar (`int`, `float`, `size_t`, …) | passed through unchanged |
| pointer (`T*` / `const T*`) | passed through; a vendor pointee is erased to `void*` |
| vendor enum (`cudaError_t`/`hipError_t`) | a **stable** capi enum with fixed values + a `_map` at the boundary (§6) |
| opaque handle (`cublasHandle_t`, `cudaStream_t`) | `typedef void* wwr_<x>_handle;` |
| by-value struct | a capi POD whose **layout** is carried in a sidecar (`captypes.Struct`), the one datum a name/`decl` must not pin for ABI (§4/§12) |

A surface declares what it needs in a tiny spec: `spec/runtime_status.py` names
the fragment, which entries to expose and the capi name each gets, and registers
the one `EnumMap` (`wwrError_t` → `wwr_status`). The status slice exercises the
scalar / pointer / enum arms. **`spec/blas.py` exercises the rest** (#349): it
takes the *whole* curated `blas_names.h` fragment and registers the vendor-free
types blas forces — the two `Handle`s (`wwrblasHandle_t` / stream → `void*`),
nine `ValueEnum`s (`Operation` / `FillMode` / `DiagType` / `SideMode` /
`PointerMode` / `DataType` / `ComputeType` / `GemmAlgo` and the `status`, each a
stable fixed-value capi enum the thunk maps **both ways** at the boundary), and
three `PointeePOD`s (`cuComplex *` → `wwr_complex64 *`, `cuDoubleComplex *` →
`wwr_complex128 *`, `__half *` → a new `wwr_half` POD — a vendor pointee that
keeps a typed, layout-matched POD instead of erasing to `void*`). All 400
functions classify and forward; a function the classifier could not represent
would be skipped with a logged reason (none is).

**`complex` is the deliberate exception** (`spec/complex.py`, still hand-written).
Its thunks forward into wwr-*authored* inline wrappers (`wwr::wwrCaddf`), which
are **not** vendor functions and so appear in no `vendor/*.json` manifest — there
is no `decl` to source a signature from. So `complex` stays a small, documented
hand-written surface; every other surface is fragment-sourced.

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
`Result` from it. The enum and the mapper are generated from one `EnumMap`
(`spec/runtime_status.py`), so they cannot drift, and `_map` compares only the
neutral `wwr::wwr<Name>` constants — never a vendor name — so the thunk stays one
shape over both backends. (`set_device(-1)` returns the identical stable code on
both backends even though the raw vendor codes differ — that *is* the §6
guarantee, exercised by the drivers.)

The `wwr_status` enumerator values are a **frozen ABI contract** (see "Stability
policy", below, and architecture.md §21).

## The one decision that shapes everything: the thunk calls *into* `wwr`

The thunk does not re-derive the backend divergence; it calls the host-callable
`wwr` functions that already resolve it once — `wwr::wwrCaddf` / `wwrCmulf` /
`wwrCabsf` / `wwrCdivf` and the `double` set, exported from the `wwr.complex`
module (architecture.md §3, §12, §25). So the generated `complex` thunk TU:

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
spec/fragments.py  parse a src/detail/*_names.h fragment into its (wwr, cuda, hip)
                   tuples (text/regex, SDK-free) -- the authoritative name set.
spec/manifest.py   read a vendor/<pin>/*.json manifest; resolve a vendor name to
                   its decl / enum value -- the signatures the names lack.
spec/captypes.py   classify a vendor decl into the vendor-free capi representation
                   (scalar/pointer/enum/handle/struct) + the enum-map / layout
                   sidecar. The extensible core #349 reuses.
spec/runtime_status.py  the fragment-SOURCED runtime status slice: the fragment,
                   which entries to expose, and the wwrError_t -> wwr_status map.
spec/blas.py       the fragment-SOURCED blas surface (#349): the whole curated
                   blas_names.h fragment, plus the handle / value-enum / pointee-POD
                   registrations the first LARGE library needs. 400 functions.
spec/complex.py    the ONE hand-written surface: PODs (capi_name, wwr_name, layout)
                   + the complex functions. Hand-written because its thunks forward
                   into wwr-authored inline wrappers, absent from any vendor manifest
                   (see above and architecture.md §25).
generate.py        fragment + manifest -> the per-unit artefacts. Sourced, so
                   nothing drifts.
gen/common.h       the PODs MORE THAN ONE unit uses (wwr_complex64 / wwr_complex128:
                   complex declares them, blas takes them by pointer). A one-unit
                   POD stays in its unit header (wwr_half -> blas; wwr_status ->
                   runtime_api).
gen/<unit>/wwr_capi_<unit>.h   one vendor-free C header per unit (complex /
                   runtime_api / blas) -- its own PODs/enums/handles + prototypes,
                   #include "common.h" for the shared PODs it names.
gen/<unit>/wwr_capi_<unit>.cpp the backend-neutral extern "C" thunk TU per unit,
                   importing ONLY the wwr module(s) its surface needs (complex:
                   wwr.complex; runtime_api: wwr.runtime_api; blas: wwr.blas +
                   wwr.fp16) -- the _map status helper / the _fwd forwarder / the
                   per-enum capi<->neutral maps live in the unit that uses them.
gen/wwr_capi.h     the UMBRELLA header: #include "common.h" + every unit header, so
                   <wwr/wwr_capi.h> still declares the whole ABI. Installed under
                   WWR_INSTALL_CAPI.
gen/common.rs      the Rust shared PODs (wwr_complex64 / wwr_complex128), the dual
                   of common.h -- always compiled, mirrored into wwr-sys/src.
gen/<unit>/<unit>.rs  one Rust module per surface: its #[repr(C)] types + a
                   #[rustfmt::skip] extern "C" block (NO #[link] -- build.rs links
                   the unit's .so per feature). Mirrored into wwr-sys/src.
gen/ffi.rs         the COMBINED Rust binding the in-tree driver.rs include!s: every
                   POD/enum + ONE extern "C" block with #[link(name="wwr_capi")]
                   over the umbrella .so. Standalone (NOT the crate); mirrors the C
                   driver linking the umbrella.
CMakeLists.txt     the per-unit object libraries, the per-unit + umbrella shared
                   libraries, the install rules, and the driver ctests.
driver.c           a pure-C consumer (stands in for Rust's FFI) that links the
                   symbols and checks the numbers; run as the wwr.capi.driver ctest.
driver.rs          the real Rust consumer: include!s gen/ffi.rs (combined) and links
                   libwwr_capi.so, compiled rustc-direct (no cargo) and run as the
                   wwr.capi.driver_rs ctest — rustc-gated (see "rustc in the loop").
wwr-sys/           the idiomatic Rust -sys crate (Cargo.toml + build.rs + src/):
                   the user-consumable binding, SPLIT PER SURFACE into Cargo
                   features over the per-unit .so's. See "The wwr-sys crate", below.
```

`gen/` is a committed snapshot so the PR is reviewable without running anything;
regenerate it with `python3 src/capi/generate.py` (no backend argument — the
`.cpp`s are backend-neutral, and the generator reads both backend manifests to
cross-check that the surface resolves identically on each). It writes `common.h`,
the umbrella `wwr_capi.h`, the combined `ffi.rs`, `common.rs`, and the per-unit
`<unit>/wwr_capi_<unit>.{h,cpp}` + `<unit>/<unit>.rs` — and mirrors `common.rs`
and the per-unit `.rs` into `wwr-sys/src/` (the crate's single source).

## The per-unit split

One `libwwr_capi.so` that linked every surface's `wwr` target also **NEEDed**
every surface's vendor library — so a consumer of the pure-host `complex` surface
still dragged in `libcublas` through the `blas` surface. That collapses the
project's own per-target dependency isolation (`wwr.complex` links nothing extra;
`wwr.blas` links cuBLAS). So the capi is split: **one shared object per surface**
(`libwwr_capi_complex.so`, `libwwr_capi_runtime_api.so`, `libwwr_capi_blas.so`),
each linking only *its* surface's vendor dep, plus an **umbrella**
`libwwr_capi.so` that contains every unit's symbols so an existing `-lwwr_capi`
consumer resolves the whole ABI from one `.so`. The proof is on the link line:

```
$ readelf -d libwwr_capi_complex.so | grep NEEDED   # no cublas
$ readelf -d libwwr_capi_blas.so    | grep NEEDED   # libcublas.so.13 (CUDA)
                                                     # libhipblas.so.3 (HIP)
```

Each unit's thunk is compiled **once** into an object library
(`wwr.capi.<unit>.obj`), linked into both its own per-unit `.so` and the umbrella
(so the umbrella carries all symbols without recompiling). Target names keep the
dotted convention: `wwr.capi` / `wwr::capi` (umbrella) and `wwr.capi.<unit>` /
`wwr::capi::<unit>` (each unit). The **`wwr-sys` Rust crate mirrors this split**
into Cargo features, so a Rust consumer opts down to one surface's `.so` the same
way a C++ consumer links one `wwr::capi::<unit>` (see "The wwr-sys crate"); the
in-tree `driver.rs` keeps linking the umbrella through the combined `gen/ffi.rs`.

## Build & run

"Call into `wwr`" means the capi builds against the module graph, so it builds as
a CMake target like any other. It builds **unconditionally** with the rest of
`src/` (each unit depends only on its surface's `wwr` module, which always
builds); the install is the only thing gated (see below).

```bash
cmake --preset default            # CUDA (or --preset hip, inside the wwr:hip image)
cmake --build build --target wwr.capi wwr.capi.driver   # umbrella + the C driver
ctest --test-dir build -R wwr.capi   # runs driver.c (+ driver_rs if rustc is found)
```

The umbrella `wwr.capi` (real dotted name, aliased `wwr::capi`) is a shared
library. Its output file is `libwwr_capi.so` (an `OUTPUT_NAME` override, matching
`ffi.rs`'s `wwr_capi` link name and the C ABI's `wwr_` symbol prefix) while the
target keeps the dotted name; each unit `wwr.capi.<unit>` outputs
`libwwr_capi_<unit>.so` the same way. The `wwr.capi.driver` ctest is gated by
`WWR_BUILD_TESTS`, links the umbrella, and is host-only, so it runs on no-GPU CI.

## Install

The capi is a set of prebuilt binaries, not module source, so it does not ride
the `src/`-sweep that packages every other `wwr` library. `wwr_install_package()`
excludes the capi targets from the sweep (the sweep's non-recursive `src/`
collection never reaches a target defined in this subdirectory, like `src/thrust`)
and installs them **explicitly**, gated on `-DWWR_INSTALL_CAPI=ON` (mirroring
`WWR_INSTALL_EXTENSION`; effect only when `WWR_INSTALL` is ON). Under the gate:

- `install(TARGETS ... EXPORT wwr-targets ...)` puts the umbrella `libwwr_capi.so`
  AND each `libwwr_capi_<unit>.so` (+ sonames) in the lib dir and adds `wwr::capi`
  plus every `wwr::capi::<unit>` to the exported target set, so a consumer's
  `find_package(wwr)` + `target_link_libraries(... wwr::capi)` (the umbrella) or
  `... wwr::capi::blas` (one isolated surface) resolves;
- the vendor-free C headers install mirroring the `gen/` tree under the `wwr`
  include dir, so `<wwr/wwr_capi.h>` (the umbrella) and its root-relative
  `common.h` / `complex/...` includes all resolve unchanged.

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

`wwr-sys/` is the idiomatic Rust `-sys` crate over the prebuilt `libwwr_capi_*.so`
family (#340). A Rust user adds one dependency, links the `.so`(s), and carries
**zero** `#[cfg]` / vendor knowledge — the backend was chosen when the linked
`.so` was built, exactly the ABI's thesis.

**The crate is split per surface, mirroring the C-side per-unit split.** One
always-compiled `common` module holds the PODs more than one surface uses
(`wwr_complex64` / `wwr_complex128`), and each surface is its own module behind a
Cargo **feature**:

| feature       | module          | links (`build.rs`)                        |
|---------------|-----------------|-------------------------------------------|
| `complex`     | `complex`       | `libwwr_capi_complex.so` (no vendor math lib) |
| `runtime_api` | `runtime_api`   | `libwwr_capi_runtime_api.so`              |
| `blas`        | `blas`          | `libwwr_capi_blas.so` → cuBLAS / hipBLAS  |

**Default = all three surfaces**, so an existing user who wants the whole ABI is
unchanged — they just transparently link the per-unit `.so`s instead of the old
umbrella. A user who needs one surface opts **down**:

```toml
[dependencies]
wwr-sys = { path = "...", default-features = false, features = ["complex"] }
```

and then **nothing** of another surface's vendor dependency is linked — a
`complex`-only build NEEDs no cuBLAS. That is the Rust mirror of the C side's
`libwwr_capi_complex.so` NEEDing no math library:

```
$ readelf -d <complex-only-binary> | grep NEEDED
  libwwr_capi_complex.so.1       # the complex unit .so, and
  …                              # NO libcublas / NO libwwr_capi_blas.so
```

- **`Cargo.toml`** — crate `wwr-sys`, version `1.0.0` tracking the frozen ABI
  contract (not bumped independently), `links = "wwr_capi"` (the `-sys`
  convention — cargo enforces one crate per native-lib family), and a `[features]`
  table: `default = ["complex", "runtime_api", "blas"]`, each surface an empty
  feature list (it gates a module + its `build.rs` link, nothing more).
- **`src/lib.rs`** — the binding surface. It declares `pub mod common;` (always)
  plus a `#[cfg(feature = "…")] pub mod …;` per surface, each the **single source**
  `generate.py` writes (`src/common.rs`, `src/<unit>.rs`) from the same specs it
  writes the C headers / thunks from, so the crate cannot drift. (The crate ships
  its own copies because, once installed away from this tree, a sibling
  `../gen/*.rs` `include!` would dangle.) Do not hand-edit them; regenerate with
  `python3 src/capi/generate.py`.
- **`build.rs`** — reads `WWR_CAPI_LIB_DIR`, emits `rustc-link-search` + an rpath,
  and then **one `rustc-link-lib=dylib=wwr_capi_<unit>` per ENABLED feature**
  (detected via `CARGO_FEATURE_<UNIT>`). So a `--features complex` build links
  only `libwwr_capi_complex.so` — the per-feature isolation. The link names are
  the units' `OUTPUT_NAME`s, kept in step with `src/capi/CMakeLists.txt` by a
  comment (the increment-3 lesson).

It is the raw `-sys` surface (every function `unsafe extern "C"`); a safe
`Result`-returning wrapper is deliberately out of scope. Crate-level smoke tests
(`cargo test`) link the enabled surfaces' `.so`s and call through them on a
no-GPU host (each `#[cfg]`-gated on its feature).

**Build it against the built or installed libs:**

```bash
cmake --preset default                       # or --preset hip (inside wwr:hip)
cmake --build build --target wwr.capi.complex wwr.capi.runtime_api wwr.capi.blas
WWR_CAPI_LIB_DIR=$PWD/build/src/capi \
  cargo test --manifest-path src/capi/wwr-sys/Cargo.toml   # default = all surfaces
# or one surface in isolation:
WWR_CAPI_LIB_DIR=$PWD/build/src/capi cargo build \
  --manifest-path src/capi/wwr-sys/Cargo.toml --no-default-features --features complex
```

**Depend on the installed crate.** Under `-DWWR_INSTALL_CAPI=ON` the crate
skeleton installs to `<prefix>/share/wwr/rust/wwr-sys/` (`Cargo.toml`, `build.rs`,
`src/` — including the generated per-unit `.rs`) alongside the `libwwr_capi_*.so`s
in `<prefix>/lib`. A downstream `Cargo.toml`:

```toml
[dependencies]
wwr-sys = { path = "<prefix>/share/wwr/rust/wwr-sys" }   # or a git dependency
```

then build with the lib dir pointed at the install:

```bash
WWR_CAPI_LIB_DIR=<prefix>/lib cargo build
```

That exact path is exercised end to end by `example/capi-consumer-rs` (#341): a
standalone crate that path-depends on the installed `wwr-sys` with the `complex`
+ `runtime_api` features, links the installed per-unit `.so`s and asserts
`wwr_cadd64` / `wwr_runtime_get_version`. `devtools/install-check.sh --capi`
builds and runs it against a throwaway install prefix (with an `ldd` check that
it links the *installed* `.so`), **and** builds a `complex`-only consumer and
asserts its `readelf` NEEDED set carries no cuBLAS — both gated on rustc so the
pre-rust-image CI path stays green.

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
- **Only `complex` is hand-written**, and deliberately so (it forwards into
  wwr-authored wrappers absent from any vendor manifest). Every other surface is
  fragment-sourced; see "How the surface is sourced" (architecture.md §25, #346).
