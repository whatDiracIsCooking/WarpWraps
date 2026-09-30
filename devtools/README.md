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

Two complementary tools read the CUDA / HIP vendor headers directly. Both key
off the vendor prefix (`curand*`/`CURAND_*`, `hiprand*`/`HIPRAND_*`), but they
answer different questions and use different machinery:

| Script | Question | How |
|---|---|---|
| `header_intersection.py` | Which **names** do the two backends share? | Regex over header text — right for names, because the vendor headers pull in device intrinsics no host parser digests cleanly. |
| `vendor_harvest.py` | What is one library's declared **surface** — signatures, enum **values**, deprecations? | clang `-ast-dump=json`. A token is not enough for a signature, and the device-code objection is about *device* code; the host declarations parse fine. |

`header_intersection.py` is the completeness gate: a green `--coverage` run means
every shared name is aliased (see its header for which modules that actually
proves anything for). `vendor_harvest.py` emits a per-library JSON **manifest**
of the declared surface for **one** `(backend, sdk_version, arch, defines)`
configuration — an AST resolves the preprocessor for exactly one config, so a
`hipsparse` harvested with and without `CUDART_VERSION` are two different
surfaces. Each manifest records the exact command and config tuple that produced
it. It **only writes manifests** — nothing it emits is fed back into source; it
is a spec/diff artifact, not a code generator. One file per library keeps diffs
readable (src/cuda and src/hip carry ~10,700 `using ::` lines between them).

```bash
# The declared surface of hipRAND, for the AMD platform:
devtools/vendor_harvest.py --backend HIP --sdk-version "ROCm 7.2.4" --arch gfx942 \
    -D __HIP_PLATFORM_AMD__ -I /opt/rocm/include \
    /opt/rocm/include/hiprand/hiprand.h            # -> hiprand.harvest.json

# ...and of cuRAND, to diff enum values against it (they differ — see rand.cppm):
devtools/vendor_harvest.py --backend CUDA --sdk-version "CUDA 13.0" \
    -I /usr/local/cuda/include /usr/local/cuda/include/curand.h
```

The harvest is a clean **subset** of what `header_intersection.py` lists by name:
the regex counts every prefixed identifier that appears anywhere (a `#define`
body, a `#include "hiprand_rocm.h"` filename, a doc `@see`), whereas an AST only
sees actual **declarations**. See each script's header for the full reasoning.
