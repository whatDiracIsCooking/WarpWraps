# CLAUDE.md

## What this is

`gpumod` — C++23 module wrappers for the CUDA and HIP GPU APIs, plus the
type-safe abstractions built on them. The vendor headers are exposed as
importable named modules (`import gpumod.cuda.cublas_v2;`), the `gpu*` layer
directly under `src/` maps backend-neutral `gpu*` names onto whichever backend
was selected, and `src/wrappers` is written once against those names.

**A build targets exactly one backend.** `GPUMOD_GPU_BACKEND` is `CUDA` or
`HIP`, and it is read *before* `project()` because it decides whether the CUDA
language is enabled at all. A HIP build needs no CUDA toolkit; a CUDA build
needs no ROCm.

The C++ and CMake identity is `gpumod`: namespace `gpumod`, modules `gpumod.*`,
macros `GPUMOD_*`, CMake targets `gpumod.*` aliased to `gpumod::*`.
`PROJECT_NAME` in `devtools/config.sh` is `gpumod` too — it names docker
volumes, images and the devcontainer, and `doctor.sh` warns when it and any
`.devcontainer/*/devcontainer.json` disagree.

## Setup

```bash
uv sync                                   # creates .venv from uv.lock
pre-commit install                        # commit-time lint + pre-push gate
devtools/devcontainer.sh rebuild          # the C++ toolchain lives in here
```

`uv` is the only assumed host tool. **`uv.lock` is tracked, and it is the only
place Python dependency versions are written down.** After editing
`pyproject.toml`, run `uv lock` and commit the result. Do not hand-install with
`uv pip install`.

**The C++ side has exactly one dependency: GoogleTest**, fetched and built from
source by `deps/CMakeLists.txt` at configure time. Nothing is expected prebuilt
in `/opt`, and `DOCTOR_REQUIRED_PATHS` is deliberately empty as a result. Do not
add a `/opt` path unless a build reads it.

GoogleTest is **unpatched** (v1.17.0, `CXX_SCAN_FOR_MODULES OFF`). A tempting
change is a patch swapping its `#include`s for `import std;` behind a PRIVATE
define — but only GoogleTest's own TUs would see it and consumers get the header
build regardless. Building with headers puts consumers in that same position
with no patch to re-roll on every upgrade. Do not add one.

`devtools/doctor.sh` reports what is degraded in the current environment. Run it
first when something behaves oddly — **and run it on both sides**, because this
repo has a host half and a container half and doctor is what tells you which one
you are on. On a bare host the whole C++ toolchain warns; that is the expected
healthy state, not a broken machine.

Doctor resolves the tools it reports through `VENV_PATHS`, not `PATH` — the
cmakelang pair lives in `.venv/bin` on the host and `/opt/venv/bin` in the
image. It also probes whether `gh`'s token can actually see *this* repository: a
fine-grained PAT authenticates fine and still 404s on a repo it was never
granted, and the first symptom used to be `gh pr create` failing after the
branch was already pushed.

## The two languages, and what verifies which

This is a C++ project with a Python tooling and test layer. There is no Python
package to import — `pyproject.toml` exists for the dev toolchain and the test
suite, and has no `[build-system]`.

| | C++ | Python |
|---|---|---|
| Source | `src/`, `cmake/`, `deps/`, `example/` | `test/shared/dispatch.py`, `test/shared/alias_coverage.py`, `.claude/hooks/protect-main.py` |
| Tests | `test/`, run by ctest | `test/shared/`, `.claude/hooks/`, run by pytest |
| Driver | `devtools/cpp-tier.sh` | `pytest` |
| Lint | `cmake-lint` (CI); `clang-tidy` via `cpp-tier.sh --tidy` | `ruff` |
| Gated by | nothing automatic — see below | pre-push hook (only on a `.py`) |

**A change under `src/` is verified by the C++ tier and by nothing else.** The
Python suite does not touch the built binaries at all — it tests three scripts.
Say which suite you ran.

**There is no `tests/` directory.** The Python suite is three files, each sitting
beside the script it tests, and `testpaths` in `pyproject.toml` names both
directories (`test/shared` and `.claude/hooks`):

| File | Tests |
|---|---|
| `test/shared/test_dispatch.py` | `test/shared/dispatch.py`, the build-time dispatch checker |
| `test/shared/test_alias_coverage.py` | `test/shared/alias_coverage.py`, the `gpu*` alias-coverage guard |
| `.claude/hooks/test_protect_main.py` | `.claude/hooks/protect-main.py`, the main-checkout guard |

Do not add a `pytest.ini` or `conftest.py` in either directory — a second
config silently overrides the root one, and which markers exist would then
depend on the directory pytest started from.

**Why they exist:** each tests a *checker*, and a checker's failure mode is
passing when it should fail — silent, and indistinguishable from working. Nothing about compiling the real objects
can catch that, because a broken checker and a correct tree look identical.
`test_dispatch.py` drives it with fake `llvm-objdump`/`llvm-cxxfilt` output and
asserts it *rejects* wrong dispatch, missing instantiations, overload
collisions and malformed tables; `test_alias_coverage.py` drives
`alias_coverage.py`'s parsers with crafted input and asserts the real `src/`
tree has full `gpu*` alias coverage. The regress stops there: a broken test goes
red on its own, so it needs no test of its own.

> **Everything else in that file trusts a hand-written imitation of
> `llvm-objdump -dr`, so two tests keep the imitation honest.** The gap is not
> theoretical: tightening `RELOC_RE`'s `\s+` to a literal space used to leave
> every fake-driven case green while the real check found nothing — the real
> relocation line pads its offset to 16 hex digits and puts **two** spaces after
> the colon.
>
> | Test | Needs | Catches |
> |---|---|---|
> | `test_parser_reads_recorded_real_objdump_output` | nothing | a parser edit, **on the host**, where `dispatch.py` is written |
> | `test_recorded_sample_still_matches_real_llvm_objdump` | `llvm-objdump` + a C compiler | the toolchain changing format under the recording |
>
> `test/shared/objdump_sample.txt` is real recorded output, committed so the
> first one never skips. The second compiles the *same* source and requires the
> two to parse identically, which is what stops the recording going quietly
> stale; regenerate it with `test/shared/gen-objdump-sample.sh` **inside the
> image**. Only that second test skips on a bare host — `llvm-objdump` is in
> `DOCTOR_OPTIONAL_TOOLS` — and a 1-skipped host run now means "the toolchain
> was not re-checked", not "the parser is unverified".

## Containers: four files, one diamond

```
                Dockerfile.base
                 /           \
  Dockerfile.cuda             Dockerfile.hip
            |                       :
  Dockerfile.combined ..............:  (reuses install-rocm.sh, not the image)
```

| File | What it is |
|---|---|
| `docker/Dockerfile.base` | The vendor-neutral toolchain: clang-20 + libc++, CMake 4.2, Ninja, ccache, uv/Python. No GPU SDK. |
| `docker/Dockerfile.cuda` | `base` + the CUDA toolkit. **The default backend**, and what the devcontainer and compose build. |
| `docker/Dockerfile.hip` | `base` + ROCm. No CUDA at all. |
| `docker/Dockerfile.combined` | `cuda` + ROCm (~40GB). |

> **Four files, but six images — the `-ci` variants are the same files with
> different build args.** `.github/workflows/images.yml` publishes
> `ghcr.io/<owner>/gpumod:{cuda-ci,hip-ci}`, which `.github/workflows/ci.yml`
> names in `container:`. `hip-ci` is built with `ROCM_PRUNE=1`, dropping ~13GB
> of kernel objects and unused libraries that no *compile* links: **20.5GB →
> 7.05GB**, measured. `docker/install-rocm.sh` has the list. That is an
> optimisation — ~13GB less to pull per CI run — **not** what makes the HIP
> job possible: a hosted runner has a 145GB root with 86GB free before any
> cleanup, so the unpruned image would fit. These docs claimed otherwise until
> the first real run produced a `df`.
>
> The suffix, the registry and the push are three environment knobs on
> `build.sh` (`IMAGE_TAG_SUFFIX`, `IMAGE_REGISTRY`, `BUILD_PUSH`), so the image
> names still live in exactly one table. The suffix is what stops a pruned
> image from overwriting the `gpumod:hip` a developer runs against a real card,
> and `:latest` is dropped when it is set. The prune has to run inside
> `install-rocm.sh`'s own `RUN` — layers are additive, so a later `rm` frees
> nothing — which is why `:hip` and `:hip-ci` are two full installs rather than
> a shared layer.

**Build them with `docker/build.sh <base|cuda|hip|combined>`, never by hand.**
The files chain by TAG, not by stage — each child opens `FROM ${PARENT_IMAGE}`
— so a parent must exist and be tagged before its child builds, and `build.sh`
is the only thing that walks the chain. Build a child directly with no parent
tagged and docker does not fall back to building it: it tries to **pull**, and
fails with `pull access denied for gpumod, repository does not exist`, which
reads like a registry problem rather than a missing local build.

`build.sh` takes the tag prefix from `PROJECT_NAME` in `devtools/config.sh`
(so `CROSS_CHECK_IMAGE` / `ROCM_IMAGE` agree by construction, not by
convention), tags the CUDA image both `:cuda` and `:latest`, forwards any build
arg set in the environment to the file that declares it, and passes trailing
flags to every step. `BUILD_DRY_RUN=1` prints the commands without running them.
**It deliberately does not skip a step whose tag already exists** — a cached
build is a second or two, and skipping would silently hand a child a stale
parent after a `Dockerfile.base` edit, which is the one failure the old single
file could not have.

> **The three `.devcontainer/*/devcontainer.json` each carry an
> `initializeCommand` that builds their parent.** A devcontainer build is one
> `docker build` with no way to produce a parent, so the hook runs `build.sh` on
> the host first — `base` for the cuda and hip variants, `cuda` for combined.
> That is also why `combined`'s `CUDA_ARCH` sits in the `initializeCommand`
> rather than in `build.args`: it is consumed by `Dockerfile.cuda`, which the
> hook builds, not by `Dockerfile.combined`.
>
> **`devcontainer build` does not run that hook; only `up` does.** Verified on
> @devcontainers/cli 0.89.0 — `up` runs it as its very first step, before it
> resolves the image, while a bare `devcontainer build` skips lifecycle hooks and
> dies with `Command failed: docker pull gpumod:base`. Harmless in practice,
> since `devcontainer.sh` only ever calls `up` and `exec`, but do not reach for
> `devcontainer build` to get just the image — `docker/build.sh cuda` is that.

**`combined` is a diamond only in intent.** Docker has no multiple inheritance,
so it takes `cuda` as its parent and re-runs `docker/install-rocm.sh`. Each SDK
install is a script rather than an inline `RUN` precisely so that reuse costs
nothing — Dockerfiles have no include. A `COPY --from=…:hip /opt/rocm` would be
a literal two-parent join and is the wrong one: `install-rocm.sh` also writes an
apt keyring and pin, `/etc/ld.so.conf.d/rocm.conf`, and the `render`/`video`
groups, none of which live under `/opt/rocm`. The consequence to remember is
that `ROCM_VERSION` and `GPU_TARGETS` are declared in `Dockerfile.combined` as
well as in `Dockerfile.hip` — bump them together.

The toolkit is installed from apt rather than inherited from `nvidia/cuda`,
because `Dockerfile.base` cannot be both `nvidia/cuda` and `rocm/dev-ubuntu` at
once, and duplicating the LLVM/CMake stages per vendor is how they drift. The
consequence to remember: the `NVIDIA_VISIBLE_DEVICES` /
`NVIDIA_DRIVER_CAPABILITIES` env that the `nvidia/cuda` images set is what the
NVIDIA container runtime reads to inject the driver under `--gpus all`.
`Dockerfile.cuda` sets both explicitly. Drop them and the container builds fine
and then has no GPU at runtime.

Two front ends, both onto the `cuda` image:

- **`docker/compose.yaml`** — the batch path. `build` / `test` / `asan` /
  `compute-sanitizer`, each a one-shot run that tees to `.log/` and exits.
  Requires `HOST_UID`/`HOST_GID` in the environment (`:?`, not a default, so a
  forgotten export fails loudly instead of writing root-owned files). Run from
  a `.claude/worktrees/` checkout its pytest half cannot work at all — it
  mounts no `.git` — so use `SKIP_PYTEST=1` there and run the Python tier on
  the host; `docker/README.md` has the why.
- **`.devcontainer/cuda/`** driven by `devtools/devcontainer.sh` — the
  interactive path, per worktree, with its own container, volumes and CPU
  bounds.

**They do not share a build directory, and cannot.** The devcontainer mounts the
workspace at its *host* path; compose mounts it at `/workspace`. A CMake cache
records absolute source and binary directories, so a tree configured by one is
rejected by the other with `The current CMakeCache.txt directory ... is
different than the directory ... where CMakeCache.txt was created`. Caches are
not relocatable, which is why compose configures with `-B $BUILD_DIR` into its
own `build-compose*/` directories.

## devtools/

Everything project-specific lives in **`devtools/config.sh`** — the project
name, which devcontainer, which presets, the test command, worker counts, the
container's CPU bounds, marker expressions, the doctor lists. Edit that file,
not the scripts; they are meant to be copied into the next repo unchanged.

| Script | What it does |
|---|---|
| `config.sh` | The settings above. Sourced by everything else. |
| `lib.sh` | Shared helpers (config splitting, venv/pytest discovery). |
| `doctor.sh` | Report what is degraded here. Non-zero only on a FAIL. |
| `devcontainer.sh` | `up` / `rebuild` / `shell` / `test` / `down` for this worktree. |
| `worktree.sh` | `add` / `rm` / `sync` / `gc` / `list` for sibling worktrees. |
| `cpp-tier.sh` | Configure + build + ctest, logged. The C++ tier. `--rocm` runs it on a real AMD card from the host. |
| `coverage.sh` | Build instrumented, run ctest, then an `llvm-cov` summary of `src/`. Runtime coverage only — see "Running tests". |
| `cross-backend-check.sh` | Compile the tree for the *other* backend. ~7s. |
| `install-check.sh` | Install to a throwaway prefix, then build `example/consumer` against it. |
| `prepush-tests.sh` | The fast Python gate that `git push` runs. |

The Claude Code hooks are **not** here — they live in `.claude/hooks/`
(`protect-main.py`, its `test_protect_main.py`, and `session-context.sh`),
beside the `.claude/settings.json` that wires them, since they belong to the
harness rather than to the copied-forward devtools toolkit.

Duplications worth knowing about:

- **`PROJECT_NAME` is spelled out again in every
  `.devcontainer/*/devcontainer.json`** — three today (`cuda`, `hip`,
  `combined`) — because JSON cannot source shell. If they drift, `worktree.sh
  rm` and `gc` stop recognising this project's volumes and silently leak one per
  worktree. `doctor.sh` finds the variants by glob rather than by a list, so a
  new one is checked without an edit, and it names the one that disagrees.
- **The `.git` bind mount in every `devcontainer.json` reads its host path from
  `${localEnv:GPUMOD_GIT_DIR}`**, which `devtools/devcontainer.sh` resolves with
  `git rev-parse` and exports on `up`/`rebuild` — so no host path is written into
  the tracked JSON (the literal after the colon is only a fallback for a direct
  VS Code "Reopen in Container"). A worktree's `.git` is a *file* pointing into
  the main checkout's `.git/worktrees/<name>`, which is not under the workspace
  folder, so without that mount every git command inside a worktree container
  fails with "not a git repository".
- **The preset names are in `CMakePresets.json` and the *choice* is in
  `config.sh`** (`CMAKE_PRESET`, `CTEST_PRESET`). Only the choice — never copy a
  compiler flag or an architecture into `config.sh`.

`protect-main.py` is **enabled by default** via `.claude/settings.json`. Set
`CLAUDE_ALLOW_MAIN_EDITS=1` for a session deliberately editing the primary
checkout.

> **It guards `Bash` as well as `Write|Edit|NotebookEdit`, and it has to.** The
> matcher used to list only the three file tools, so a session editing with
> `sed -i`, a `>` redirect or a `python3 - <<EOF` heredoc rewrote the primary
> checkout without the guard ever being invoked — and that is the *default*
> shape for an agent told to prefer shell tools, not an exotic one. The Bash
> half is necessarily a heuristic over an arbitrary command string; it errs
> toward denying, and `CLAUDE_ALLOW_MAIN_EDITS=1` is the way past a false
> positive. **A few git operations are judged by what they do**, since they
> name no path to be caught by: `git reset --hard`/`--merge` and `git clean -fd`
> in the primary checkout (they discard uncommitted work in a tree other
> sessions share), and `git commit` while that checkout is on `main` (this repo
> ships through PRs — a topic branch checked out there stays committable).
> `git add` is deliberately not guarded, being inert and one `git restore
> --staged` from undone; neither is `git push`, because a local hook binds
> neither other machines nor other clients and **branch protection is the real
> control there** — guarding it here would be false confidence.
> **Git-ignored paths are carved out** — `build*/`,
> `.slow-tier-reports/`, `.venv/` are generated and disposable, so `mkdir -p
> build`, a `--fresh` wipe and a build's own output are not "editing the
> checkout". Note `/build*/` is a *directory* pattern, so the guard asks git
> about the trailing-slash form too — but only for a name with no dot in it,
> since `/build*/` would otherwise match `build.log/` and make every root
> `build*` name writable. `.claude/hooks/test_protect_main.py` pins both halves, including
> that `.claude/settings.json` still names `Bash` — the script can be perfect
> and still never run.

## Worktrees

Two systems, deliberately:

- **`devtools/worktree.sh`** — heavyweight, container-backed. Each worktree is a
  **sibling** directory under the repo root, with its own devcontainer, named
  volumes and build image. `rm` tears down all of that; plain `git worktree
  remove` leaks it, which is what `gc` exists to reconcile.
- **`.claude/worktrees/<name>`** — lightweight, no container. Quick isolated
  checkouts and agent scratch space.

`sync` fast-forwards local `main` to `origin/main` (`--ff-only`). Nothing else
does this: after a PR merges, `origin/main` moves but local `main` does not, and
`add` forks from the calling checkout's HEAD — so a stale `main` silently seeds
stale branches.

Use `rebuild`, not `up`, after editing `devcontainer.json` or any
`docker/Dockerfile.*`:
`up` reuses the running container, applies none of the change, and reports
success with the same container id. (A `Dockerfile.base` edit needs nothing
extra beyond that — the `initializeCommand` rebuilds the parent every time.)

**Three separate "how parallel" knobs, and only one bounds a container.** `JOBS`
is the pytest worker count. `BUILD_JOBS` is `cmake --build`'s. The two that
actually bound a container are `CPUSET` and `CPUS`, applied on `up`/`rebuild` via
`docker update`. They belong to the container, which is keyed on the workspace
folder — so they are per worktree, not per shell.

> **Applying them costs the CUDA container a restart, and that is deliberate.**
> `docker update` regenerates the device cgroup from `HostConfig.Devices`, and
> for `--gpus all` that list is **empty** — the ask lives in `DeviceRequests`
> and is honoured by the NVIDIA runtime's OCI hook at container *start*. So
> updating the CPU limits silently revokes GPU access while leaving every
> `/dev/nvidia*` node visible inside, and `nvidia-smi` then reports
> `Failed to initialize NVML: Unknown Error` — which reads like a driver fault
> rather than a permissions one. On this repo it surfaces one step earlier
> still: `CMAKE_CUDA_ARCHITECTURES=native` queries the device at *configure*
> time, so a healthy tree fails to configure. `devcontainer.sh` restarts the
> container to re-run the hook; the limits survive, since they live in the
> container config rather than the cgroup. The HIP variant is untouched by
> this — `--device=/dev/kfd` populates `HostConfig.Devices`, which `docker
> update` preserves — so the restart is gated on `DeviceRequests`.

`BUILD_JOBS` is the one to set conservatively. A C++23 module build is
memory-hungry per job (`clang-scan-deps` plus a BMI cache), so the right number
is *lower* than the core count. Leaving it empty lets Ninja pick cores + 2,
which is what OOM-kills a 16-core box — and an OOM-killed compiler surfaces as a
bare `ninja: build stopped`, with nothing about memory in it.

**Killing `devcontainer.sh test` on the host does not kill the test run inside.**
The `npx` process dies; the pytest master and its workers keep running at 100%
CPU and nothing tells you. After interrupting, check with
`devtools/devcontainer.sh shell -c "ps -eo pid,pcpu,cmd --sort=-pcpu | head"`;
`pkill -f pytest` reaches only the master.

## Building the C++ tree

```bash
devtools/devcontainer.sh shell -c devtools/cpp-tier.sh   # the normal way
devtools/cpp-tier.sh --preset hip                        # the ROCm backend
devtools/cpp-tier.sh --preset compile-time               # no GPU, no GoogleTest
devtools/cpp-tier.sh --fresh                             # wipe the cache
devtools/cross-backend-check.sh                          # does the OTHER backend compile?
```

- **`CMAKE_CUDA_ARCHITECTURES=native` queries a live device at *configure*
  time.** No GPU, no configure — even for a change that touches no CUDA. Use
  `ci-cuda`, which pins `86`, when the build host has no card, or `-D` an
  architecture onto any preset.
- **The HIP counterpart fails *silently* instead.** ROCm's `hip-config-amd.cmake`
  reads the CMake variable `GPU_TARGETS`; unset, it runs `amdgpu-arch` to detect
  an installed card, and on a box with no AMD GPU that detection fails and ROCm
  picks its own default — verified with ROCm 7.2.4 to be **gfx906**, with no
  `--offload-arch` on the compile line and no warning. The `hip` image exports
  `GPU_TARGETS=gfx1200`, but an environment variable is not a CMake variable, so
  the top-level `CMakeLists.txt` seeds one from the other before
  `find_package(hip)`. Precedence: explicit `-DGPU_TARGETS` beats `$ENV{GPU_TARGETS}`
  beats ROCm's detection. It matters beyond performance — gfx906 is wave64 where
  gfx1200 is wave32, so a silent fallback also disagrees with
  `GPUMOD_WARP_SIZE`.
- **Running the `hip` image against a real AMD card needs NUMERIC group ids.**
  `--device=/dev/kfd --device=/dev/dri` is not enough: those nodes are
  `root:render` / `root:video`, so the container user must be in those groups.
  `--group-add render --group-add video` resolves the names *inside* the
  container, where they have different ids or do not exist, and the run then
  fails at the first device call with `hipErrorNoDevice (no ROCm-capable device
  is detected)` — which reads exactly like a box with no GPU. Pass the **host**
  gids instead (`getent group render video`). With them the `hip` preset's
  ctest passes in full on this machine; without them, every device-dependent
  suite aborts at the first device call.

  **`devtools/cpp-tier.sh --rocm` does all of that for you**, and exists
  because the warning above was not enough on its own: the failure is silent,
  looks like missing hardware, and has been read as one. It resolves the gids
  with `getent`, checks `/dev/kfd` exists, and re-execs `cpp-tier.sh` inside
  the `hip` image. Reach for the raw `docker run` only when you want something
  the flag does not cover. `ROCM_*` in `devtools/config.sh` holds the image,
  preset, device nodes and group names.
- **`default` and `workstation` are now the same configuration**, differing only
  in `binaryDir`, so a container build and a host build can coexist. `default`
  is the one that keeps `build/`, because compose and this file both name that
  path.
- **`default`, `workstation`, `debug`, `asan`, `hip`, `compile-time`,
  `ci-cuda`, `ci-hip` and `coverage` have test presets.** With any other preset
  `cpp-tier.sh` builds and then says there was nothing to ctest. That is not a
  pass. (`coverage` exists for `coverage.sh`, which drives its own ctest to
  collect profiles — see "Running tests".)
- **`ci-cuda` and `ci-hip` are what CI runs, and they run here too.** Reach for
  them to reproduce a CI result exactly. They are deliberately the SAME SHAPE:
  each builds the whole tree for its backend — GoogleTest and every runtime
  test binary included — and then runs `ctest -LE gpu`. Neither needs a
  device, so both work in a container started without `--gpus`; `ci-cuda`
  pins the architecture so configure never queries a driver.

  Read the result as **compile-and-link plus a thin runtime slice**, not as a
  test of GPU behaviour. Of what `-LE gpu` leaves, only `HalfConversion` and
  `Bfloat16Conversion` compute anything (host fp16/bf16, and *not* shared
  between backends — `gpumod.fp16`/`gpumod.bf16` map onto the chosen backend's
  own types, so running them on both legs is worth it). The rest are the
  `SuiteListIsComplete` guards, which prove each device binary loads and its
  registry matches the CMake list, and the `*_compile_tests` link checks. The
  four dispatch checks run at BUILD time on both legs, so they never appear in
  the ctest count but are covered.

  `ci-hip` was briefly `GPUMOD_COMPILE_TIME_ONLY=ON`. That left
  `test/extension/*.cpp` never compiled for HIP at all — under the *stricter*
  of the two front ends, which is most of why the leg exists.

  **Both legs report 12, and they are meant to stay equal.** Each is
  `{cuda,hip}_compile_tests` + `gpu_compile_tests` + the two conversion suites
  + eight `SuiteListIsComplete` guards. A difference between them is a signal,
  not a quirk — which is the point of making them match, since an 11 that is
  supposed to be 11 is the kind of number nobody notices drifting to 10.
- **There are no per-architecture presets, and nothing may go below sm_75.**
  `base` uses `native` and `ci-cuda` pins `86`; between them that is every case
  this project has, and `86` is what the reference box, `Dockerfile.cuda`'s
  `CUDA_ARCH` and `ci-cuda` already agree on. `volta` (70), `ampere` (80),
  `hopper` (90) and `portable` (70;80;90) were removed rather than corrected.

  Two of them *could not configure at all*: CUDA 13's nvcc floor is
  `compute_75`, so sm_70 dies at configure with `nvcc fatal : Unsupported gpu
  architecture 'compute_70'` — reported as "Check for working CUDA compiler -
  broken", with the real line further up the output. `Dockerfile.cuda`'s header
  predicted exactly this when the toolkit was bumped and nobody acted on it,
  which is the lesson worth keeping: **a preset nothing exercises can be dead
  for a whole toolkit generation.** What is exercised now is `default`
  (`native`, on a real card) and the two presets CI drives.

  Target another architecture with a `-D` and a build directory of its own.
  It has to be `-D`: that beats a preset's `cacheVariables`, where an
  environment variable does not — `CUDAARCHS` is read only when the cache
  variable is unset, and `base` always sets it.

  ```bash
  cmake --preset default -B build-h100 -DCMAKE_CUDA_ARCHITECTURES="90"
  ```
- **Every configure preset has its own `binaryDir`**, and it needs to stay that
  way — two presets sharing one directory silently reconfigure it back and forth,
  a full rebuild each way.
- **`CMAKE_EXPERIMENTAL_CXX_IMPORT_STD` is a UUID tied to the CMake version.**
  Upgrade CMake and the old UUID is ignored *silently*; every `import std` then
  fails to resolve. The new value is in that release's
  `Help/dev/experimental.rst`.

`cpp-tier.sh` logs every run to
`.slow-tier-reports/cpp-<stamp>-<rev>-<preset>.log` and appends a PASS/FAIL line
to `summary.log`. Quote the log path when reporting a failure — the summary has
none of the diagnostics.

## Running tests

Three tiers, plus a cheap portability check:

```bash
pytest -n auto -rs               # the Python suite: three files, under two seconds
devtools/cpp-tier.sh             # the C++ tier: configure, build, ctest
devtools/install-check.sh        # the package tier: install, then consume it
devtools/cross-backend-check.sh  # does the OTHER backend still compile? ~7s
devtools/cpp-tier.sh --rocm      # the C++ tier on a real AMD card, from the host
```

The fourth is not a tier — it runs no test and launches no kernel. It answers
the one question the other three cannot, because each of them builds exactly
one backend. See "A green CUDA build does not mean the code is portable". CI
now answers it too, on every PR, by building the `ci-hip` preset for real; keep
this one for the seconds-long local answer before you push.

**To reproduce what CI saw, run CI's presets rather than guessing:**

```bash
devtools/cpp-tier.sh --preset ci-cuda   # whole tree for CUDA, then -LE gpu
devtools/cpp-tier.sh --preset ci-hip    # whole tree for HIP,  then -LE gpu
```

Both work in a container with no `--gpus` and no `/dev/kfd`, which is the point
— they are the same selection the runner gets. `--preset default` is still what
you want before a PR, because it is the only thing that runs the seven
device-dependent suites.

**The third one is not a variant of the second.** `cpp-tier.sh` builds the tree
in place and never installs, so nothing in it can see an export regression —
and the way to cause one is ordinary: add a module whose `.cppm` has a `PRIVATE`
include directory or define. That passes `cpp-tier.sh` and breaks every consumer,
because a consumer of a C++23 module package *compiles the installed module
sources* and a `PRIVATE` requirement is never exported. `install-check.sh`
installs to a throwaway prefix and builds `example/consumer` against it, which
is the only thing that answers the question. Every defect found while the
install rules were being written was found that way rather than by reading them.

**Coverage is a separate tool, not a tier.** `devtools/coverage.sh` builds the
tree with the `coverage` preset (`GPUMOD_COVERAGE=ON` adds clang source-based
instrumentation to CXX only — nvcc's `.cu` units are untouched, exactly as ASAN
is), runs ctest so the binaries drop `.profraw`, then merges with `llvm-profdata`
and prints an `llvm-cov` summary of `src/` (`--html` for a browsable report).
The llvm tools must match the clang that built the tree — a `.profraw` carries a
format version — so it resolves the versioned `llvm-cov-20`/`llvm-profdata-20`
first. It gates nothing and launches no kernel; read it as *which `src/` lines
the runtime ctest suites execute*. The two purely compile-time groups — the
vendor re-exports (`src/cuda/*`, `src/hip/*`) and the dispatch-checked neutral
wrappers (`blas`/`complex`/`fft`/`rand`/`solver`/`sparse.cppm`) — never run, so
counting them only drags the number down with lines already tested another way;
they are dropped from the summary via `COVERAGE_IGNORE_REGEX` in
`devtools/config.sh` (keep that regex in sync if a neutral module is added).
`fp16`/`bf16`/`runtime_api.cppm` stay in — the runtime suites do exercise their
host-side logic. One caveat remains: an abort-on-error path reached only through
a GoogleTest `EXPECT_DEATH` shows as uncovered, because the forked child calls
`std::abort()` without flushing its `.profraw` — the death test still verifies
the behavior, it just cannot register as covered. Artifacts land under
`build-coverage/coverage/`, already ignored by `/build*/`.

**There is no fast/slow split.** The Python suite is a handful of fast files
with no expensive tier behind them, so `FAST_TEST_ARGS` in `devtools/config.sh`
is empty and still prepended to the default run; add a marker there only
alongside something that runs a slower half.

`SLOW_TIER_REPORTS` in `config.sh` (default `.slow-tier-reports/`) is where
`cpp-tier.sh` writes its logs — `.gitignore`, `.dockerignore` and the CI
artifact upload all name that path.

### Two kinds of C++ test, compile-time and runtime

**Compile-time tests are proved by building.** A failing `static_assert` fails
the build of its module. They live in `test/cuda/`, `test/hip/`, `test/gpu/`,
`test/extension/build_time/`, and `test/wrappers/{blas,solver,fft,sparse}/`. All
are reached by the `compile-time` preset or the `compile_time_tests` target. `test/gpu` in
particular checks that every `gpu*` name really is the chosen backend's own
symbol — that is a compile-time claim, not a runtime one.

> **"No GPU needed" applies to the BUILD, not to running every binary.** The
> `compile-time` preset configures and builds with no device and no GoogleTest
> fetch — verified. But `cuda_compile_tests` links the CUDA *driver* API stubs
> (nvml, cuda_h, …), so the linked binary needs `libcuda.so.1` at **load** time
> and dies with `error while loading shared libraries: libcuda.so.1` in a
> container started without `--gpus`. `gpu_compile_tests` has no such dependency
> and runs anywhere. So on a driverless box, build the compile-time tier and
> read the *build* as the result; do not read that one ctest failure as a broken
> change.

**The dispatch checks additionally disassemble.** `test/shared/dispatch.py`
reads the wrapper instantiations out of their compiled objects and demangles
what they call, to prove each one dispatches to exactly one `gpublas*` /
`gpusolverDn*` / `gpufft*` / `gpusparse*` function — the wrong-entry-point class
of bug that compiles and links. One TOML table per module —
`test/wrappers/{blas,solver,fft,sparse}/<name>_dispatch.toml` — holds both the
expectations and the module-specific configuration; `gpumod_add_dispatch_check()`
wires one up, at four call sites today. That needs `llvm-objdump` and
`llvm-cxxfilt`, which is why both are in `DOCTOR_OPTIONAL_TOOLS`.

> Each table is **hand-written on purpose** — it is only worth something as an
> independent statement of what *should* be called. `--print` dumps what the
> objects actually contain, in the table's format, for comparison; never paste
> that output unread.

> **`test/wrappers/{blas,solver,fft,sparse}/` are build-time only**,
> and deliberately: the wrappers there have no body but a token-paste forward
> (`return gpublas<X><name>(__VA_ARGS__)`), so which name is pasted — plus, for
> the solver's modern API, which `gpusolverDataType_t` is mapped — is the whole
> of what they can get wrong, and the dispatch check plus
> `test/wrappers/solver/type_traits.cppm` prove both for every instantiation
> without a device. The runtime `basic.cpp` suites that used to sit beside them
> compared a wrapper byte-for-byte against the direct call; they were dropped
> because they re-proved a tenth of the same claim, needed a GPU to do it, and
> rested on the vendor library being bitwise run-to-run reproducible, which
> neither cuBLAS nor rocBLAS promises in general.

**Runtime tests register one ctest entry per suite**, via
`gpumod_add_gtest_suite_tests()`, plus a guard test that fails when a suite
exists in the binary but is missing from the CMake list. The suite list is
hand-maintained — that guard is what keeps it honest, so when a failure says a
suite is missing from the list, add it rather than deleting the guard. Eight
targets use this machinery today: `test/gpu/conversions` (`HalfConversion`,
`Bfloat16Conversion`, CPU-only), and — all device-dependent —
`test/extension/{memory_buffer,runtime,rand,blas,solver,fft,sparse}`.

> **The runtime tier is no longer a single CPU-only suite.**
> `test/gpu/conversions` exercises the host fp16/bf16 conversion wrappers, which
> compute on the CPU, so it needs no device — but there are device-dependent
> suites again. The device-dependent suites are:
> `test/extension/memory_buffer` (RAII device/host buffers),
> `test/extension/runtime` (`GpuStream`/`GpuEvent`/`GpuGraph` RAII),
> `test/extension/rand` (kernels via `gpumod.extension.init_state` +
> `gpumod.extension.random_normal`) and the vendor-handle/plan RAII
> suites `test/extension/{blas,solver,fft,sparse}`. Those allocate device memory,
> launch kernels, or create vendor-library handles, so a GPU-less box now
> *fails* them.
>
> **The way to run without a device is the `gpu` ctest label, not a skip.**
> Nothing in `test/` calls `GTEST_SKIP` or gates on a device count, and nothing
> should: a skipped test blends into a green run, where an EXCLUDED one is
> named in the ctest output. `gpumod_add_gtest_suite_tests(... REQUIRES_GPU)`
> puts `gpu` on every entry it registers, the seven device-dependent targets
> above pass it, and `test/gpu/conversions` deliberately does not. So:
>
> ```
> ctest --preset default    42 entries — everything, needs a card
> ctest --preset ci-cuda    12 entries — `-LE gpu`, needs nothing
> ctest --preset ci-hip     12 entries — same, the other backend
> ```
>
> `cuda_compile_tests` is the one entry whose treatment depends on a cache
> variable, for a reason unrelated to devices: it links the CUDA driver stubs
> and cannot *load* without `libcuda.so.1` (see the "No GPU needed" applies to
> the BUILD note above). With `GPUMOD_CUDA_DRIVER_STUBS=OFF` (the default) it
> carries `gpu` and is excluded; with it ON — only the `ci-cuda` preset — that
> one test gets the toolkit's own stubs on `LD_LIBRARY_PATH` via
> `ENVIRONMENT_MODIFICATION` and runs instead. `test/cuda/CMakeLists.txt` has
> the full reasoning, including why the stubs are in the build directory
> rather than the image. `test/hip`'s counterpart needs none of it.
>
> Two traps around the label. A second label on the same test must be added
> with `set_property(TEST ... APPEND PROPERTY LABELS ...)`, never
> `set_tests_properties(... PROPERTIES LABELS ...)`, which REPLACES — that is
> how `no_sanitizer` on the two allocation-failure suites would silently strip
> their `gpu`. And the `<target>.SuiteListIsComplete` guards are intentionally
> left unlabeled: `--gtest_list_tests` touches no device, so they still run on
> a GPU-less runner and keep the hand-written suite lists honest there.
>
> None of this changes what a box with a card runs: no preset other than
> `ci-cuda` filters on `gpu`, so `devtools/cpp-tier.sh` still runs all 42.

### A green CUDA build does not mean the code is portable

**The two backends go through different front ends, and one is stricter.** A
`.cu` is compiled by nvcc under CUDA and by clang's `-x hip` under HIP. nvcc is
the more permissive of the two, so there is a class of error that compiles
clean, passes the entire ctest run, and ships — and only fails when someone
builds ROCm.

A concrete instance of that class: a functor holding a `const` member of
**class** type is non-trivially-copyable under clang and so fails
`parallel_for`'s `device_functor` concept, while nvcc accepts it. It passes the
CUDA tier and only fails on a ROCm build; `parallel_for.cuh`'s `device_functor`
comment states the rule.

```bash
devtools/cross-backend-check.sh               # ~7s, from the host
devtools/cross-backend-check.sh --device-only # ~4s, the .cu files only
```

**Run it before opening a PR that touches a `.cu`**, or anything under `src/`
that a `.cu` includes. It configures and builds the compile-time tier for the
other backend — no GoogleTest, no runtime tests, no device — in a one-shot
`docker run` against the `hip` image, and needs only docker on the host.

> **Seconds, not minutes — the cost note on `cpp-tier.sh` does not apply
> here.** Measured cold with ccache disabled: 4s for
> `--device-only` and 7s for the whole compile-time tier — 350 steps with the
> dispatch checks in it, and 7s is also the wall clock from the host, docker
> startup included. The "a gate
> that costs minutes is one people learn to bypass" reasoning that keeps the
> C++ tier off the push gate is about a full module build with separable
> compilation and a GoogleTest fetch; a compile-only check of the other backend
> is a different thing.

It is **not** a git hook, deliberately: it needs docker and a ~20GB image, so
on a machine without them it could only skip — and a gate that silently does
nothing on the machine that most needs it is worse than no gate. It exits 2
rather than 0 when it cannot run, so a skip is never mistaken for a pass.

It proves the other backend **compiles**, not that it runs. On a box with an
AMD card, `devtools/cpp-tier.sh --rocm` is the stronger statement — the same
tier, run against the hardware, with the device nodes and numeric gids handled
for you. This is the cheap check you can afford every time; that one is the
check worth running before a PR that touches a `.cu`, when you have a card.

**A green run means nothing until you know what skipped.** Tests that need an
absent tool skip *silently* on the Python side; `pytest -rs` lists the reasons.
Say what skipped, and say whether the C++ tier ran at all, before calling a
change verified.

## What is gated, and what is not

- **`git push`** runs the fast pytest tier — **only when the push touches a
  `.py`**. That condition is the `files: \.py$` filter on the pre-push hook in
  `.pre-commit-config.yaml`, not anything inside the script.
- **CI runs on every push to `main` and every PR**, all of it on
  GitHub-hosted runners (`workflow_dispatch` also keeps the manual trigger).
  `.github/workflows/ci.yml` has four jobs: `lint` (ruff + cmake-lint),
  `test` (the pytest tier on 3.11 and 3.13), `cpp` (a two-leg matrix, below)
  and `install-check`.
- **The C++ tree IS built on the server now, both backends, same shape.** The
  `cpp` job declares a `container:` rather than building one; each leg builds
  the whole tree for its backend and runs `ctest -LE gpu`. Neither needs a
  device — `ci-cuda` pins `CMAKE_CUDA_ARCHITECTURES` so configure never
  queries a driver. Read both as compile-and-link plus a thin runtime slice:
  nothing on a hosted runner tests GPU behaviour (see "Two kinds of C++ test"
  and the `ci.yml` header for exactly what the surviving entries prove).
  The **cuda leg also runs `--tidy`** after its ctest — the only place
  clang-tidy runs at all. It is advisory by construction (`cpp-tier.sh` never
  lets it touch the exit code), so its findings are in the uploaded log, not in
  the job's colour; a green `cpp (cuda)` says nothing about tidy.
- **The images come from `.github/workflows/images.yml`**, which publishes
  `ghcr.io/<owner>/gpumod:{cuda-ci,hip-ci}` on a change under `docker/` (plus
  weekly, plus on demand). They are the same Dockerfiles as the dev images
  with different build args: `hip-ci` is built with `ROCM_PRUNE=1`, which
  drops ~12.9GB of kernel objects and unused libraries that no *compile*
  needs — 20.5GB to 7.05GB, so there is that much less to pull per run.
  **Those tags move, and `container:` is resolved before any step runs** — so
  a PR that edits `docker/` is tested against the image `main` already
  published. Run `images` manually on the branch first.

  What such a PR *does* get is a **build-only run of that same workflow**: the
  `pull_request` trigger carries the same `paths:` filter and builds both
  images without pushing (`BUILD_PUSH` empty, the GHCR login skipped, and the
  job renames itself `build (<image>)`). So a Dockerfile that no longer builds
  fails on the PR instead of on the publish after the merge. It does **not**
  make the `cpp` legs use the new image — that is still the manual run above.

**So a PR touching `src/` is now gated on the server for compile, link, export
and every non-device test.** What is still not: the seven device-dependent
runtime suites — `test/extension/{memory_buffer,runtime,rand,blas,solver,fft,
sparse}` — which carry the `gpu` ctest label and are excluded by name. Only a
box with a card runs those, so **`devtools/cpp-tier.sh` before opening a PR
remains the gate for anything touching device behaviour**, it is local, and it
is bypassable with `--no-verify`. Run it and say what you ran.

That the local tier is not itself a git hook is deliberate rather than an
oversight: a C++23 named-module build with CUDA separable compilation costs
minutes even warm, and a gate that costs minutes is one people learn to bypass
with `--no-verify`, which is worse than no gate. CI is where that cost belongs,
and now carries it.

`devtools/cross-backend-check.sh` is no longer the only thing that compiles the
other backend — the `cpp (hip)` leg does the same work on every PR. Keep it for
the fast local answer before you push; it is seconds against CI's minutes.

## Conventions

- C++23, named modules, clang + libc++. `CMakeLists.txt` **refuses gcc**.
- Targets are declared through the macros in `cmake/`: `gpumod_add_cxx_module_library`
  (75 call sites), `gpumod_add_gtest_executable` / `gpumod_add_gtest_suite_tests`
  / `gpumod_add_test_executable` for tests, and `gpumod_add_gpu_device_library`
  (9 call sites, `src/extension/{random_normal,init_state}` among them) for a
  module's device-kernel `.cu` library. That last one replaced
  `gpumod_add_cuda_library`, a CUDA-only leftover with no HIP branch that no call
  site could use; device-kernel modules hand-rolled ~25 identical lines each
  until it existed. `gpumod_add_interface_library` is still wired and documented
  with **no call sites** — do not assume it is dead.
- Target names use dots and are aliased to `::`.
- **`#include` style tracks header ownership.** A header this project owns uses
  quotes; the standard library and third-party/vendor headers (the CUDA/HIP/ROCm
  SDKs, GoogleTest) use angle brackets. An owned header is spelled by the path
  its include root makes resolve — bare for a same-directory header
  (`#include "gpu_backend.h"`), root-relative otherwise
  (`#include "wrappers/common/dispatch_sdcz.h"`,
  `#include "test/shared/link_check.h"`) — **never a `../` relative climb**. Those
  roots are the `$<BUILD_INTERFACE>` / `$<INSTALL_INTERFACE>` include directories
  on the owning targets; `cmake/gpumod_install.cmake` mirrors `src/`'s shape under
  `include/gpumod/` so the one spelling resolves both in-tree and for an installed
  consumer recompiling the module units. A non-module header a `.cppm` includes
  from its global module fragment is reached this way, so its root must be
  exported (PUBLIC/INTERFACE), not PRIVATE — a PRIVATE include root is the export
  regression `install-check.sh` exists to catch.
- Python ≥3.11, `from __future__ import annotations` everywhere.
- Lint is deliberately narrow (`E,F,I,UP,B`) and there is **no formatter hook** —
  not ruff-format, not clang-format, not cmake-format, though all three are
  configured. A formatter rewrites whole files and buries real diffs in
  restyling.
- `ruff check .` is clean — keep it that way; the pre-commit hook fails on any
  finding in a file you touch. `src/` and `deps/` are excluded from ruff, being
  C++. The one Python file inside the C++ tree, `test/shared/dispatch.py`, is
  **not** excluded and is held to the same rules.
- **Install the pre-commit hook from wherever you actually commit.** The hook
  bakes in an absolute `INSTALL_PYTHON` and lives in the *common* git dir — one
  file shared by the main checkout and every worktree. Installing it from inside
  the container points it at `/opt/venv` and silently breaks committing from the
  host; the symptom is `ModuleNotFoundError: No module named 'pre_commit'`. Fix
  with `pre-commit install -f` from the side you commit on.
