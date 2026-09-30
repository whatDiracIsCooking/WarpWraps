# Docker — the batch path

This directory is **one of two front ends onto the same image**
(`./Dockerfile.cuda`). Which one you want depends on what you are doing:

| | `docker/compose.yaml` (here) | `.devcontainer/cuda/` + `devtools/` |
|---|---|---|
| Shape | one-shot: configure, build, test, exit | long-lived container you stay in |
| Logs | tees everything to `.log/` | on your terminal; `cpp-tier.sh` also logs |
| Good for | CI, nightlies, sanitizer sweeps, a clean reproducible run | iterating, agents, debugging |
| Per worktree? | no — services are global | yes, container + volumes keyed on the workspace path |
| CPU bounds | none | `CPUSET`/`CPUS` in `devtools/config.sh` |
| Entry point | `dc test` | `devtools/devcontainer.sh shell -c devtools/cpp-tier.sh` |

They are not layered on each other and neither is deprecated. The rule of
thumb: if you want to read the result and throw the container away, use
compose; if you want to stay inside, use the devcontainer.

Every workflow here runs through `docker/compose.yaml`. All services mount the
repo root at `/workspace` and run as your host UID/GID, so nothing lands
root-owned.

Because the `user:` field is required, always export these two first:

```bash
export HOST_UID=$(id -u) HOST_GID=$(id -g)
```

A shell alias keeps the rest short:

```bash
alias dc='docker compose -f docker/compose.yaml run --rm'
```

## Configuration

Compose reads variables from your shell and, if the file exists, from
`docker/.env` — auto-loaded because it sits beside `compose.yaml`. That file is
gitignored and there is deliberately **no committed template** for it: every
variable except `HOST_UID`/`HOST_GID` already carries a default in
`compose.yaml` (`${VAR:-default}`), and those two are written `${VAR:?...}`
*without* one so a forgotten export fails loudly. A committed `.env` pinning
`HOST_UID=1000` would satisfy that check and then quietly write build output
owned by uid 1000 on any host where you are not uid 1000 — precisely the failure
the `:?` exists to prevent.

If you want one anyway, write only the part that is machine-specific:

```bash
printf 'HOST_UID=%s\nHOST_GID=%s\n' "$(id -u)" "$(id -g)" > docker/.env
```

The tables below list every variable the services read.

## Build the Image

Four files beside this one, in a diamond:

```
                Dockerfile.base          the vendor-neutral C++23 toolchain
                 /           \
  Dockerfile.cuda             Dockerfile.hip
            |                       :
  Dockerfile.combined ..............:  (reuses the HIP scripts, not the image)
```

`Dockerfile.cuda` is what these services and
`.devcontainer/cuda/devcontainer.json` both build, so the devcontainer and
compose cannot drift onto two different toolchains.

**They chain by tag, not by stage.** A child opens with
`FROM ${PARENT_IMAGE}`, so its parent has to be built and tagged first —
`./build.sh` is what walks the chain, and is the way to build any of them:

```bash
docker/build.sh cuda        # base, then cuda -> wwr:cuda and wwr:latest
docker/build.sh hip         # base, then hip  -> wwr:hip
docker/build.sh combined    # base, cuda, then combined -> wwr:combined
docker/build.sh base        # just the toolchain -> wwr:base
```

It takes the tag prefix from `PROJECT_NAME` in `devtools/config.sh`, so
`wwr:latest` — what `WWR_IMAGE` below defaults to — is always one of the
two tags the CUDA image gets. Run it from anywhere; the context is always the
repo root, because the files read `pyproject.toml`, `uv.lock` and
`docker/install-*.sh` relative to it.

### The vendor libraries, and where each one comes from

Four install scripts, because the libraries this project wraps arrive by four
different routes — and which route a library takes is not a style choice, it is
whatever its vendor publishes:

| Script | Mechanism | What it installs |
|---|---|---|
| `install-cuda.sh` | apt, NVIDIA's CUDA repo | the toolkit, plus NCCL, **cuTENSOR** and **nvCOMP** — three packages the `cuda-toolkit-*` meta does not pull |
| `install-cugraph.sh` | PyPI wheels → `/opt/rapids` | **cuGraph** and the RAPIDS libraries it links (raft, rmm, cuvs). There is no apt package; RAPIDS ships conda and wheels only |
| `install-rocm.sh` | apt, AMD's ROCm repo | the HIP SDK, which already includes hipTensor and RCCL |
| `install-rocm-ds.sh` | git + cmake → `/opt/rocm-ds` | **hipCOMP** — AMD packages it nowhere; ROCm-DS has no apt channel |

**cuGraph has no HIP counterpart in these images, on purpose.** rocGRAPH and
hipGRAPH were built here and removed: they cannot be compiled for an RDNA
target in usable time. Measured on one 24-core machine, same source, same
patches — `gfx942` (wave64) finished in **19 minutes**; `gfx1200` (wave32, the
default here) **stalled at 129/135 after ~2 hours** with single translation
units past 2h each. `install-rocm-ds.sh` carries the detail and issue #126
holds the API measurements for when ROCm-DS supports ROCm 7 on RDNA.

Three things follow from that table and are worth knowing before you touch it:

- **The CUDA image is now much heavier: 10.1GB → 13.1GB**, measured, split
  ~2.05GB into the apt layer (cuTENSOR, and nvCOMP for ~70MB of it) and 949MB
  into `/opt/rapids`. All of it lands in `:cuda-ci`, which `ci.yml` pulls on
  every run. If pull time becomes the bottleneck, the CUDA counterpart of
  `ROCM_PRUNE` starts with `libcugraph_mg.so` (277MB, multi-GPU).

  A third GB was uv's wheel cache: `uv pip install` without `--no-cache` left
  949MB at `/root/.cache/uv` — an exact copy of what it had just installed —
  taking that layer to 1.95GB. `install-cugraph.sh` passes `--no-cache`. Worth
  knowing the shape of that bug, because it is the same one `ROCM_PRUNE`
  documents: layers are additive, so deleting the cache in a later `RUN` frees
  nothing. It has to not be written in the first place.
- **`/opt/rocm-ds` needs the network during `docker build`** — it clones and
  compiles hipCOMP rather than installing a package. It costs ~3 minutes, all of
  it hipCOMP's 28 objects.
- **`GPU_TARGETS` must name exactly one target.** `USE_WARPSIZE_32` is a single
  flag for the whole of hipCOMP, so a list naming both an RDNA and a CDNA card
  has no correct value to take and `install-rocm-ds.sh` rejects it. The default
  `gfx1200` is RDNA4, so wave32.

### `WWR_WARP_SIZE`, and the one thing to know about it

hipCOMP's `USE_WARPSIZE_32` is **OFF by default** and documented "e.g., for
gfx1100 devices" — so an RDNA build without it is a silent wrong answer rather
than a build failure. `install-rocm-ds.sh` derives it from `GPU_TARGETS`, which
is the one place in the image that knows.

**That value is baked into `libhipcomp.so` at image build time.** Configuring
wwr later with a different `WWR_WARP_SIZE` cannot change it, and the two
disagreeing produces no error and no warning — just wrong results from the
compression kernels.

So the image records what it actually compiled with:

```cmake
# /opt/rocm-ds/wwr-image-warp-size.cmake
set(WWR_IMAGE_WARP_SIZE 32)
set(WWR_IMAGE_GPU_TARGETS "gfx1200")
```

**Follow-up, belonging with the wrappers rather than the image:** `CMakeLists.txt`
should include that file when it exists and fail when `WWR_WARP_SIZE` disagrees
with `WWR_IMAGE_WARP_SIZE`. Until it does, the coupling is documented but not
enforced on the wwr side.

| Build arg | Default | Declared in |
|---|---|---|
| `CUGRAPH_VERSION` | `26.8.0` | `Dockerfile.cuda` |
| `CUVS_VERSION` | `26.8.1` | `Dockerfile.cuda` (separate — RAPIDS releases cuVS on its own cadence) |
| `HIPCOMP_VERSION` | `v2.2.0` | `Dockerfile.hip` **and** `Dockerfile.combined` |

`HIPCOMP_VERSION` joins `ROCM_VERSION` and `GPU_TARGETS` on the list of args
declared in two files that must be bumped together, for the same reason:
`combined` cannot inherit from `hip`, so it re-runs the script instead.

hipCOMP is pinned by **tag**, which is more than the ROCm-DS graph libraries
offered — see the note above about why they are not here.

Consumers find all of this through `CMAKE_PREFIX_PATH`, set in the image. Note
that a configure passing its own `-DCMAKE_PREFIX_PATH` **shadows** the
environment variable rather than extending it — so `devtools/install-check.sh`
appends the image's prefixes to its temp prefix (translating the ENV's `:`
separators to the `;` a CMake list uses) rather than replacing them.

Anything after the target is passed through to every `docker build` in the
chain, and any build arg set in the environment is forwarded to the file that
declares it — so widening the CUDA architecture for a mixed fleet is:

```bash
CUDA_ARCH="80;86;90" docker/build.sh cuda
docker/build.sh cuda --no-cache --progress=plain   # flags reach every step
```

`BUILD_DRY_RUN=1` prints the `docker build` commands without running them.

By hand is still fine, as long as you build the parent yourself first:

```bash
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.base -t wwr:base .
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.cuda -t wwr:latest .
```

Skip that first line and docker does not fall back to building the parent — it
tries to *pull* it, and fails with `pull access denied for wwr, repository
does not exist`, which reads like a registry problem rather than a missing local
build. `docker/Dockerfile.base`'s header has the rest of the reasoning.

### The `-ci` variants, and the registry

Three more environment knobs turn a local build into a published one. Only
`.github/workflows/images.yml` sets them, and only to publish the two images
CI pulls:

```bash
IMAGE_TAG_SUFFIX=-ci IMAGE_REGISTRY=ghcr.io/<owner> BUILD_PUSH=1 \
  ROCM_PRUNE=1 docker/build.sh hip      # -> wwr:hip-ci, pushed to GHCR
```

| knob | effect |
|---|---|
| `IMAGE_TAG_SUFFIX` | appended to every tag in the chain (`wwr:hip-ci`) |
| `IMAGE_REGISTRY` | adds `<registry>/wwr:<tag>` as a second tag |
| `BUILD_PUSH=1` | pushes the **final target's** registry tags, not its parents' |

`:latest` is dropped when a suffix is set — `wwr:latest-ci` would be a lie,
since `latest` is what `WWR_IMAGE` resolves to and must keep meaning the
full CUDA dev image. Parents are not pushed because a child image is
self-contained; publishing `:base` too would upload 1.45GB nothing pulls.

**`ROCM_PRUNE=1` takes the ROCm image down** (~570MB larger than the old 7.05GB
now that rccl stays, and ~226MB larger again now that hiptensor does), dropping
Tensile/rocFFT kernel objects, composable-kernel archives and rocalution — none
of which a *compile* links. rccl and hiptensor both used to be on that list and
both came off it the same way: something started wrapping them, which makes them
link-time dependencies, which makes pruning them a `find_package` failure at
configure. `docker/install-rocm.sh` carries the list and the reason each entry
is safe.

The prune does **not** touch `/opt/rocm-ds` — hipCOMP is built from source into
its own prefix, so nothing apt removes can reach it.

It is an **optimisation**, worth ~13GB less to pull on every CI run and a
3-minute push instead of many. It is not what makes a HIP job possible: a
GitHub-hosted runner has a 145GB root with 86GB free before any cleanup, so the
unpruned image fits too. (This paragraph previously claimed otherwise, on an
estimate the first real run disproved.)

It has to happen inside that script's own `RUN`. Layers are additive, so an
`rm` in a later layer hides the files and frees nothing; `:hip` and `:hip-ci`
are therefore two full installs rather than a shared layer plus a delta.

## Build the Project

```bash
dc build
```

### Build Options

Pass environment variables before the command:

```bash
# Debug build
BUILD_PRESET=debug dc build

# Clean rebuild (recompile everything)
CLEAN=1 dc build

# Reconfigure only (e.g. after changing CMake options)
RECONFIGURE=1 dc build

# Full rebuild from scratch
REBUILD=1 dc build

# Skip configure, just recompile after code edits
BUILD_ONLY=1 dc build
```

| Env Var | Effect |
|---------|--------|
| `WWR_IMAGE` | Docker image to use (default: `wwr:latest`) |
| `BUILD_PRESET` | Select cmake preset: `default` (Release), `debug`, `asan` (default: `default`) |
| `CLEAN=1` | Remove compiled objects before building |
| `RECONFIGURE=1` | Wipe cmake cache and reconfigure from scratch |
| `REBUILD=1` | Both clean and reconfigure (nuclear) |
| `BUILD_ONLY=1` | Skip configure, just build |

## Run Tests

The `test` service configures, builds, then runs gtest followed by pytest.

The pytest half runs the whole suite — two files,
`test/shared/test_dispatch.py` and `.claude/hooks/test_protect_main.py`, the
same thing the push gate and `devcontainer.sh test` run. There is no `-m "not slow"` filter any more: the
fast/slow split and `devtools/slow-tier.sh` were retired when the suite shrank
to a second. The `compute-sanitizer` service still deselects `no_sanitizer`,
because instrumented runs turn a merely-slow case into an hours-long one.

```bash
# All tests (gtest + pytest)
dc test

# Gtest only
SKIP_PYTEST=1 dc test

# Gtest with filter
SKIP_PYTEST=1 TEST_FILTER='MemoryBuffer' dc test

# Pytest only
SKIP_GTEST=1 dc test

# Pytest with filter
SKIP_GTEST=1 PYTEST_ARGS='-k smoke' dc test
```

| Env Var | Effect |
|---------|--------|
| `TEST_FILTER` | Regex passed to `ctest -R` (NOT `--gtest_filter`: compose drives ctest, which registers one entry per suite) |
| `PYTEST_ARGS` | Extra arguments appended to the pytest invocation |
| `SKIP_GTEST=1` | Skip the C++ suite |
| `SKIP_PYTEST=1` | Skip the Python suite |
| `TIMEOUT_MULTIPLIER` | Scale test timeouts (default: `1`) |
| `CUDA_VISIBLE_DEVICES` | Which GPU to run on (default: `1`) |

### pytest cannot run from a `.claude/worktrees/` checkout

Every service mounts `..:/workspace` and nothing else. In one of the
lightweight worktrees the repo's `.git` is a *file* pointing at
`<repo-root>/main/.git/worktrees/<name>`, which is outside that mount, so
`git rev-parse --git-common-dir` fails inside the container and
`.claude/hooks/test_protect_main.py` raises during collection — which takes
the whole pytest run with it, `test_dispatch.py` included. The gtest half is
unaffected, and `dc test` exits 1.

Run the C++ half here and the Python half on the host, where git resolves:

```bash
SKIP_PYTEST=1 dc test     # in the worktree
uv run pytest -n auto -rs # on the host
```

No volume in `compose.yaml` fixes this: the path that would have to be mounted
is absolute and host-specific, so it cannot be committed. The
`.devcontainer/*.json` files do carry exactly such a path, hand-written per
machine, which is why the devcontainer has no such problem. Running compose
from the primary checkout also has none — there `.git` is a real directory
inside the mount.

## Run Tests under AddressSanitizer

```bash
dc asan
```

Uses the `asan` preset, built into `build-compose-asan/`. Accepts the same `TEST_FILTER`,
`SKIP_GTEST`, and `SKIP_PYTEST` variables.

## Run Tests under Compute Sanitizer

```bash
dc compute-sanitizer

# Select a tool other than memcheck
COMPUTE_SANITIZER_TOOL=racecheck dc compute-sanitizer
```

Uses the `compute-sanitizer` preset, built into `build-compose-compute-sanitizer/`.

## Interactive Shell

```bash
dc build bash
```

## Logs

All output is logged to `.log/` with timestamps:

```
.log/configure.{ts}.out.txt / .log/configure.{ts}.err.txt
.log/build.{ts}.out.txt     / .log/build.{ts}.err.txt
.log/gtest.{ts}.out.txt     / .log/gtest.{ts}.err.txt
.log/pytest.{ts}.out.txt    / .log/pytest.{ts}.err.txt
```

## Requirements

- Docker with BuildKit support
- NVIDIA Container Toolkit (for GPU access)
