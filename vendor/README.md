# vendor/ — committed vendor-surface manifests (machine output)

Every `*.json` here is the **declared + linkable surface of one wrapped vendor
library**, harvested by `devtools/vendor_harvest.py` from the pinned SDK headers
and the matching `.so`. They are **machine output — never hand-edit them.** Each
file's `"harvest"` block records the exact clang command, the config tuple
(backend, SDK version, arch, defines, includes), the clang version and the source
header that produced it, so a manifest is reproducible and a diff between two is
meaningful only when their tuples agree. See `devtools/vendor_harvest.py`'s
header for what a manifest is and is not.

## Schema

Two halves, because a header DECLARES a surface but the shared library DEFINES
one and the two genuinely differ:

- `"symbols"` — the **declared** surface: every declaration the AST found whose
  leading token is the manifest's `prefix` (or one of its `extra_prefixes` — a
  variant library's own token, e.g. `hipblaslt` alongside `hipblas`, #167), each
  with its `name`, `kind`, normalised `decl` (+ `decl_hash`), enum `value`, and
  `deprecated` / `exported` markers where they apply. The `"harvest"` block
  records `prefix` and, when a variant widened it, `extra_prefixes`.
- `"linkable"` — the **linkable** surface, from `nm -D --defined-only` over the
  library:
  - `lib` / `soname` — the library's SONAME (`libcusparse.so.12`), recorded
    instead of a path so it is stable across the two container mount points and
    the pin.
  - `available` — `false` for a header-only surface (nvtx3) or an aggregate with
    no discrete `.so`; then `reason` states why and `linkable` is omitted.
  - `linkable` / `linkable_count` — the names the `.so` defines that carry this
    manifest's prefix.
  - `declared_not_linkable` — **the point of the schema**: FUNCTION-kind names the
    header declares that the library does NOT export, so a program that calls one
    compiles but fails to link. This is the machine-checked counterpart to the
    test suite's `WWR_DECLARED_CHECK(sym)` sites (vs the usual `WWR_LINK_CHECK`),
    each of which appears here — e.g. `cusparse.json` lists all eight
    `*gebsr2gebsc_bufferSizeExt` / `*csr2gebsr_bufferSizeExt` variants, and
    `hip_runtime_api.json` lists `hipExternalMemoryGetMappedMipmappedArray`. A
    documented-superset manifest (`cublasLt`, whose `cublas` prefix also captures
    the base cublas_v2 API that lives in `libcublas.so`, not `libcublasLt.so`)
    reports those base names here too — expected, since it is named against the
    variant `.so`.

## Layout

One directory per SDK version, one file per wrapped library:

- `cuda-13.0.x/` — the CUDA backend, harvested against the toolkit the
  `:cuda-ci` image ships (CUDA 13.0.x; NCCL / cuTENSOR / nvCOMP arrive by apt).
  This is a within-major **band**, not a point: CUDA 13.0 → 13.3 adds rather
  than removes, so one dir serves both the 13.0.0 floor and every 13.0.x pin —
  there is no separate CUDA floor manifest (nor a CI cuda-floor leg).
- `rocm-7.2.4/` — the HIP **pin**, harvested against ROCm 7.2.4 (arch `gfx942`),
  plus hipCOMP from `/opt/rocm-ds`.
- `rocm-7.1.0/` — the HIP **floor** (the oldest ROCm `wwr` builds on;
  `CMakeLists.txt` enforces it). ROCm churns across minors exactly in the
  surfaces this repo wraps, so pin and floor are distinct manifests. Diffing the
  two is the changelog AMD does not publish (#116): at 7.1.0 → 7.2.4 nothing
  leaves the surface (0 floor-only symbols), 36 names are *added* in 7.2 (28
  amd_smi, 6 hip_runtime_api, 2 hipblaslt) — all behind `WWR_*_SINCE_*` /
  `__has_include` guards in `src/hip`, so the floor is clean. (The 2 hipblaslt
  SIGMOID constants became visible in this diff only once #167 widened the
  harvest to capture a variant library's own uppercase constants; before that
  they were in no manifest and the diff read 34.)

## Regenerating

`devtools/vendor_manifests.sh` is the single source of the harvest invocations
(prefix, defines, include roots per library). Run it **inside the matching
`-ci` image** — the manifests are only reproducible against those exact headers.
The HIP version dir is **read from the image** (`/opt/rocm/.info/version`), so
one `hip` invocation writes `rocm-7.2.4/` in the pin image and `rocm-7.1.0/` in
the floor image; CUDA stays the `13.0.x` band:

```bash
devtools/vendor_manifests.sh cuda   # rewrites cuda-13.0.x/  (in :cuda-ci)
devtools/vendor_manifests.sh hip    # rewrites rocm-<ver>/   (in :hip-ci or :hip-7.1.0-ci)
devtools/vendor_manifests.sh both   # both (needs the combined image)
```

The `vendor-manifests` CI job regenerates into a throwaway tree and fails on any
diff, printing it — the honesty mechanism that keeps these files from going
stale-but-trusted. It runs three legs — `cuda-13.0.x` (`:cuda-ci`), `rocm-7.2.4`
(`:hip-ci`) and `rocm-7.1.0` (`:hip-7.1.0-ci`). A red diff is either a manifest
nobody regenerated after an intended change, or a new SDK point release changing
the surface under a fixed pin/floor.

## Consumers

`devtools/header_intersection.py` reads these manifests (the `symbols` /
**declared** block) to answer "which names do the two backends share, and does a
module wrap them all?" — a set operation over committed data, so it needs **no
SDK**. That is a deliberate change from its old behaviour of regexing the vendor
`.h` files at run time: reading the full harvested surface makes its `--coverage`
numbers **more correct** than the old single-header scan and therefore different
from it. An umbrella like `cublas_v2.h` (a thin wrapper over `cublas_api.h` that
mostly `#define`s `_v2` aliases) hid the bulk of the API from a one-file text
scan; the manifest sees all of it, so the shared surface it reports is materially
larger. `test/shared/test_header_intersection.py` exercises this off the
committed manifests (rand and blas), so it runs on a bare CI runner with no GPU
SDK.

## Coverage

`devtools/vendor_manifests.sh`'s header lists the deliberate gaps: the
type-wrapper modules (`cuda_fp16`/`bf16`/`fp8`/`fp4`/`fp6`, `cuComplex` and
their HIP twins) and `cufile` have no single vendor prefix the name-mode
harvester can key on, so no meaningful manifest exists for them. `cublasLt` /
`cublasXt` / `cusolverMg` / `cusolverSp` are documented supersets (their prefix
also captures the shared base surface). Variant libraries additionally carry
their OWN uppercase constants via an `extra_prefix` (`cublaslt`, `cublasxt`,
`cusolverdn`, `cudalibmg`, `nvfatbin`, `nvjitlink`, `lib`, `hipblaslt`, `hiplib`,
`activity`), whose leading token is the variant, not the base (#167).
