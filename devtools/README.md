# `devtools/` — the host-side scripts

Everything here runs on the host (or inside the dev container) rather than as
part of a build. Each script documents its own rationale in its header; this
file is the map. The **test**, **devbox**, **doctor** and **pr** skills drive
most of them.

## Build / test drivers

| Script | What it does |
|---|---|
| `cpp-tier.sh` | Configure + build + `ctest` for one backend — the local gate for device behaviour CI cannot run. |
| `cross-backend-check.sh` | Does the *other* GPU backend still compile? |
| `install-check.sh` | Install the package, then consume it from a separate project (catches export regressions). |
| `prepush-tests.sh` | The fast pytest tier the pre-push hook runs on a `.py` change. |
| `coverage.sh` | Coverage build + report. |

## Container / worktree

| Script | What it does |
|---|---|
| `devcontainer.sh` | up / rebuild / shell / test / down for the dev container. |
| `worktree.sh` | Create and tear down container-backed sibling worktrees. |
| `doctor.sh` | Diagnose a degraded environment. |
| `config.sh`, `lib.sh` | Shared config (`PROJECT_NAME`) and helpers sourced by the above. |

## Vendor-surface analysis

Two complementary tools compare the CUDA / HIP vendor surfaces. Both key off the
vendor prefix (`curand*`/`CURAND_*`, `hiprand*`/`HIPRAND_*`), but they answer
different questions and read different inputs:

| Script | Question | How |
|---|---|---|
| `vendor_harvest.py` | What is one library's declared + linkable **surface** — signatures, enum **values**, deprecations, what the `.so` exports? | clang `-ast-dump=json` (+ `nm -D` over the `.so`). Needs the SDK; writes a committed manifest under `vendor/`. |
| `header_intersection.py` | Which **names** do the two backends share, and does a module wrap them all? | A set operation over the committed `vendor/*.json` manifests. **Needs no SDK** — pure data over the harvested declared surface. |

`vendor_harvest.py` is the SDK-side half: it emits a per-library JSON
**manifest** of the declared + linkable surface for **one** `(backend,
sdk_version, arch, defines)` configuration — an AST resolves the preprocessor for
exactly one config, so a `hipsparse` harvested with and without `CUDART_VERSION`
are two different surfaces. Each manifest records the exact command and config
tuple that produced it. It **only writes manifests** — nothing it emits is fed
back into source; it is a spec/diff artifact, not a code generator. One file per
library keeps diffs readable (src/cuda and src/hip carry ~10,700 `using ::` lines
between them). `devtools/vendor_manifests.sh` is the single source of the harvest
invocations; the `vendor-manifests` CI job regenerates and fails on any diff.

```bash
# Regenerate the committed manifests (inside the matching pinned image):
devtools/vendor_manifests.sh cuda   # -> vendor/cuda-13.0.x/*.json
devtools/vendor_manifests.sh hip    # -> vendor/rocm-7.2.4/*.json
```

`header_intersection.py` is the SDK-free completeness gate that reads those
manifests: a green `--coverage` run means every shared **name** is aliased in the
module source (see its header for which modules that actually proves anything
for). It used to regex the vendor `.h` files at run time, which meant it only ran
where the SDK was installed and only saw one header's text; reading the committed
manifests instead lets it run anywhere and see the full declared surface. That
makes its numbers **intentionally more correct** than — and different from — the
old header scan: an umbrella like `cublas_v2.h` (a thin wrapper over
`cublas_api.h`) hid the bulk of the API from a single-file text scan, so the
manifest reports a materially larger shared surface. See its header for the full
reasoning.

```bash
# Which names do the two rand backends share, and does src/rand.cppm wrap them?
devtools/header_intersection.py \
    --cuda vendor/cuda-13.0.x/curand.json vendor/cuda-13.0.x/curand_kernel.json \
    --hip  vendor/rocm-7.2.4/hiprand.json vendor/rocm-7.2.4/hiprand_kernel.json \
    --coverage src/rand.cppm src/rand.cuh
```
