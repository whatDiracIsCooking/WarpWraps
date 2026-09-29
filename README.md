# Warp Wraps (`wwr`)

C++23 module wrappers for the CUDA and HIP GPU APIs. Exposes the CUDA runtime,
cuBLAS, cuSOLVER, cuRAND (and their ROCm/HIP counterparts) and supporting
libraries as importable C++23 modules, then builds type-safe abstractions on top
of them.

A build targets exactly one GPU backend, CUDA or HIP (`WWR_GPU_BACKEND`).
The `wwr*` layer directly under `src/` maps backend-neutral `wwr*` names
(`wwrStream_t`, `wwrStreamCreate`, `wwrFloatComplex`, …) onto the chosen
backend, and `src/wrappers` is written once against those names.

## Quick start

Interactive, and the one to develop in:

```sh
devtools/devcontainer.sh rebuild
devtools/devcontainer.sh shell -c devtools/cpp-tier.sh   # configure + build + ctest
devtools/devcontainer.sh shell                           # or just live in it
```

The devcontainer bind-mounts your checkout's `.git` by absolute host path, so
before the first `rebuild` update the `// EDIT` markers in
`.devcontainer/<variant>/devcontainer.json` — see
[`.devcontainer/README.md`](.devcontainer/README.md) ("Per-project edits").

Batch, onto the same image:

```sh
export HOST_UID=$(id -u) HOST_GID=$(id -g)
docker/build.sh cuda          # builds docker/Dockerfile.base, then .cuda
docker compose -f docker/compose.yaml run --rm build
docker compose -f docker/compose.yaml run --rm test
```

Host-side, for editing and the Python suite:

```sh
uv sync && pre-commit install
pytest -n auto -rs
devtools/doctor.sh
```

---

## Using wwr in your project

Install it, then `find_package`:

```sh
cmake --preset default
cmake --build build
cmake --install build --prefix /where/you/want/it
```

```cmake
find_package(wwr 0.1 REQUIRED)

add_executable(app main.cpp)
target_link_libraries(app PRIVATE
  wwr::wwr.wrappers.blas
  wwr::wwr.extension.memory_buffer
)
```

```cpp
import wwr.wrappers.blas;
import wwr.extension.memory_buffer;
```

Configure with `-DCMAKE_PREFIX_PATH=/where/you/want/it` so `find_package` can see
it. Every target in the [Modules](#modules) tables is exported under the
`wwr::` namespace, keeping its dotted name — `wwr.wrappers.blas` is
`wwr::wwr.wrappers.blas`. Linking one is what makes the corresponding
`import` resolve; there is nothing else to configure.

[`example/consumer/`](example/consumer/) is a complete, buildable version of the
above, and `devtools/install-check.sh` runs it against a real install.

### What a consumer has to match, and why

**Your build compiles wwr's module interface units.** This is the one thing
worth understanding before depending on the package, because it is not how a
header-or-`.so` library behaves.

A BMI — the compiled form of a C++ module interface — is not portable. It is
tied to the exact compiler build, standard library and flags that produced it,
and no two toolchains promise to agree. So the package cannot ship BMIs; it
ships the `.cppm` **sources**, and your build compiles them. CMake's
`CXX_MODULES_DIRECTORY` export machinery is what re-attaches them to the
imported targets.

The consequence is that a consumer's build is less "linking against wwr" than
"continuing wwr's build", and it needs the same things that build needed:

- **Clang with libc++.** The module units `import std;`, which resolves against
  libc++'s own module manifest. Set `CMAKE_CXX_STANDARD_LIBRARY` to `libc++` and
  `CMAKE_CXX_COMPILER` to `clang++` **before `project()`**, as
  `example/consumer/CMakeLists.txt` does.
- **`CMAKE_EXPERIMENTAL_CXX_IMPORT_STD`**, the UUID for your CMake version.
  Also before `project()`. A stale UUID is ignored silently and every
  `import std` then fails to resolve.
- **CMake 4.2+**, and the CUDA Toolkit or ROCm that the install was built
  against.

`wwrConfig.cmake` checks what it can — it warns on a compiler or standard
library that does not match the one the package was built with, rather than
letting the mismatch surface as a wall of errors inside wwr's own sources.
Silence those with `-DWWR_SKIP_TOOLCHAIN_CHECK=ON` if you know your toolchain
is compatible.

### One backend per installation

An installation contains exactly one backend's wrapper modules, recorded in the
config as `WWR_GPU_BACKEND` along with `WWR_WARP_SIZE`. Requesting the
other one is refused rather than half-satisfied:

```cmake
find_package(wwr REQUIRED COMPONENTS CUDA)   # fails on a HIP installation
```

To use both, build and install wwr twice, to two prefixes.

### Embedding it instead

`add_subdirectory` and `FetchContent` also work, and link the targets directly
with no install step. `WWR_INSTALL` defaults to `OFF` in that case, so
wwr's headers and module sources do not follow your project into *its*
install tree.

## Requirements

Everything is pinned inside the container, which is the supported path:

- Docker with BuildKit, and the **NVIDIA Container Toolkit** for GPU access
  (or a ROCm-capable host with `/dev/kfd` for the HIP backend)
- An NVIDIA GPU for the CUDA backend. Not only for the test services:
  `CMAKE_CUDA_ARCHITECTURES=native` queries a live device at **configure** time,
  so even a CPU-only change needs one unless you pin a preset.
- `uv` on the host — the only assumed host tool, and the only thing the Python
  half needs.

Building on a bare host additionally needs Clang 20 with libc++ (including
`libc++.modules.json` and `clang-scan-deps`), CMake 4.2+, Ninja, and the CUDA
Toolkit or ROCm. The root `CMakeLists.txt` discovers the LLVM installation by
globbing `/usr/lib/llvm-*` and `/usr/local/llvm-*`; override
`CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS` and `CMAKE_CXX_STDLIB_MODULES_JSON` on the
command line if yours lives elsewhere.

GoogleTest is fetched and built from source as part of the configure step;
nothing needs to be installed for it, and nothing is expected prebuilt in
`/opt`.

`devtools/doctor.sh` tells you which of these you have. Run it on the host and
again inside the container — the difference between the two lists is the point.

---

## Layout

```
CMakeLists.txt        Toolchain discovery, backend switch, project(), options
CMakePresets.json     Configure/build/test presets
cmake/                The macros that define every target in the project
deps/                 GoogleTest (fetched from source at configure time)
src/*.cppm,*.cuh,*.h  The wwr* backend switch: wwr* names for the chosen backend
src/cuda/             Low-level CUDA API module wrappers (CUDA backend)
src/hip/              Low-level ROCm/HIP API module wrappers (HIP backend)
src/wrappers/         Backend-neutral higher-level abstractions
example/consumer/     A standalone project that uses an INSTALLED wwr
test/shared/          link_check.h; dispatch.py + alias_coverage.py and their pytest suites
test/cuda/, test/hip/ Compile-time checks for the low-level wrappers
test/gpu/             Compile-time checks that every wwr* name is the backend's, + fp16/bf16 conversions
test/wrappers/        Build-time dispatch checks for the blas/solver/fft/sparse wrappers (built, not run)
test/extension/       Extension tests: static_assert build-time checks and the runtime GoogleTest suites
docker/               The batch path: compose.yaml, the SDK install scripts
  ├── Dockerfile.base     The vendor-neutral toolchain; parent of the three below
  ├── Dockerfile.cuda     base + CUDA. The default backend
  ├── Dockerfile.hip      base + ROCm, no CUDA
  ├── Dockerfile.combined cuda + ROCm (~40GB)
  └── build.sh            Builds one of them, and its ancestors, in order
.devcontainer/        The interactive path
devtools/             Container, worktree, test-tier and doctor scripts
docs/                 CONTRIBUTING.md, architecture.md, gpu-header-dependencies.md
```

New here? **[`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md)** is the short version
of setup, which suite verifies what, and the gates to run before a PR.

---

## Modules

### Low-level wrappers (`src/cuda/`, `src/hip/`)

Thin C++23 module interfaces over the native vendor headers. All symbols are
placed in the `wwr` namespace.

| Module | Import | Wraps |
|--------|--------|-------|
| `wwr.cuda.cuda_runtime_api` | `import wwr.cuda.cuda_runtime_api;` | `cuda_runtime_api.h` — full CUDA runtime API |
| `wwr.cuda.cublas_v2` | `import wwr.cuda.cublas_v2;` | `cublas_v2.h` — complete cuBLAS API |
| `wwr.cuda.cusolverDn` | `import wwr.cuda.cusolverDn;` | `cusolverDn.h` — complete cuSOLVER Dense API |
| `wwr.cuda.curand` | `import wwr.cuda.curand;` | `curand.h` / `curand_kernel.h` |
| `wwr.cuda.cufft` | `import wwr.cuda.cufft;` | `cufft.h` — complete cuFFT API |
| `wwr.cuda.cusparse` | `import wwr.cuda.cusparse;` | `cusparse.h` — complete cuSPARSE API |
| `wwr.cuda.cuComplex` | `import wwr.cuda.cuComplex;` | `cuComplex.h` (forwarding wrappers for `static inline` symbols) |
| `wwr.cuda.cuda_fp16` | `import wwr.cuda.cuda_fp16;` | `cuda_fp16.h` — `__half`, `__half2` and operators |
| `wwr.cuda.cuda_bf16` | `import wwr.cuda.cuda_bf16;` | `cuda_bf16.h` — `__nv_bfloat16` and operators |

`src/hip` mirrors these against hipBLAS, hipSOLVER, hipRAND, hipFFT, hipSPARSE,
rocm_smi/amd_smi and hiprtc.

### Extensions (`src/wrappers/`)

Type-safe abstractions, RAII resource management and utility kernels, all in the
`wwr` namespace. **Every one of them is backend-neutral**: they are
written once against the wwr* layer's `wwr*` names and build for either backend.
There is no per-backend extension tree.

That is the layer's contract — `wwr.wrappers.*` adapts the functions **both**
vendors offer. See [Vendor-only functions](#vendor-only-functions) for the
handful that only one does, and how to reach them.

| Module | Purpose |
|--------|---------|
| `wwr.wrappers.common` | FP concepts and integer utilities the generic wrappers build on |
| `wwr.extension.common` | Error handling (`gpu_check`, pluggable policy), `DeviceScope` |
| `wwr.extension.handle` | The RAII handle base (`BaseHandle`, `DeviceBoundHandle`) and the non-owning `HandleView` |
| `wwr.extension.runtime` | RAII stream, event, graph and memory pool |
| `wwr.wrappers.blas` | Generic templated BLAS (`gemm<float>(…)` rather than `cublasSgemm_v2` / `hipblasSgemm`), either backend |
| `wwr.wrappers.solver` | Generic templated dense solver, either backend |
| `wwr.wrappers.fft` | Generic templated FFT, transform kind selected at compile time, either backend |
| `wwr.wrappers.sparse` | Generic templated sparse over the shared legacy-typed API (`bsrmv<float>(…)`, `gtsv2`, `csrgeam2`, …), either backend |
| `wwr.extension.blas` / `.solver` / `.fft` / `.sparse` | RAII, device-bound vendor handles and the FFT plan, either backend |
| `wwr.extension.memory_buffer` | `DeviceBufferWrapper<T>`, `PinnedBufferWrapper<T>`, `UnifiedBufferWrapper<T>`, `HostBufferWrapper<T>` and the view wrappers |
| `wwr.extension.init_state` | Per-thread RNG state initialization on the device API, either backend |
| `wwr.extension.random_normal` | Normal-distribution draws from those per-thread states, either backend |
| `wwr.fp16` / `wwr.bf16` | Host-side fp16 / bf16 conversions, mapped onto the chosen backend's own type |
| `parallel_for` | header-only device-side parallel iteration helper, either backend |

### Vendor-only functions

`wwr.wrappers.*` is an adapter over the **intersection** of what the two
vendors offer. A handful of cuBLAS and cuSOLVER entry points have no
hipBLAS/hipSOLVER counterpart, so they get no backend-neutral wrapper — wrapping
them would mean a module that compiles on one backend and not the other, which is
the property this layer exists to avoid.

They are still reachable. Import the low-level module, which exposes the vendor's
complete API 1:1, and call the typed entry point directly:

```cpp
import wwr.cuda.cublas_v2;   // the whole cuBLAS API; CUDA builds only
cublasCgemm3m(handle, /* … */);
```

The same applies in the other direction for rocBLAS/rocSOLVER calls cuBLAS and
cuSOLVER lack (`import wwr.hip.hipblas;`).

**cuBLAS-only** — verified against `hipblas.h`. `wwr.wrappers.blas` has
everything else:

| Function | Operation |
|---|---|
| `cublas{C,Z}gemm3m` | GEMM using the 3m (Gauss) algorithm — complex only |
| `cublas{S,D}gemmGroupedBatched` | Grouped batched GEMM, per-group dimensions — real only |
| `cublas{S,D,C,Z}matinvBatched` | Batched direct inversion (small matrices) |
| `cublas{S,D,C,Z}tpttr` | Triangular packed → full format |
| `cublas{S,D,C,Z}trttp` | Triangular full → packed format |

**cuSOLVER-only** — verified against `hipsolver-dense.h` /
`hipsolver-dense64.h`. All are in the modern (`X`-prefixed,
`cusolverDnParams_t`-based) API, and each has a matching `_bufferSize`:

| Function | Operation |
|---|---|
| `cusolverDnXsytrs` | Solve after symmetric indefinite factorization |
| `cusolverDnXtrtri` | Triangular matrix inversion |
| `cusolverDnXlarft` | Form the triangular factor T of a block reflector |
| `cusolverDnXgesvd` | Full SVD: A = U * Sigma * V^T |
| `cusolverDnXgesvdp` | SVD with partial pivoting, for improved accuracy |
| `cusolverDnXgesvdr` | Approximate SVD via randomized algorithm (rank-k) |
| `cusolverDnXsyevd` | Symmetric eigenvalue decomposition (all eigenvalues) |
| `cusolverDnXsyevdx` | Selective symmetric eigenvalue decomposition |
| `cusolverDnXsyevBatched` | Batched symmetric eigenvalue decomposition |
| `cusolverDnXgeev` | General (non-symmetric) eigenvalue decomposition |

hipsolverDn's eigenvalue/SVD surface is **legacy-typed only**, which is why the
entire modern eigen/SVD API is on that list. The legacy-typed eigenvalue/SVD API
is fully shared and *is* wrapped, as `wwr.wrappers.solver`'s
`:eigen_solver_legacy` partition (48 functions) — as are the 8 modern functions
both backends do have (potrf/potrs/getrf/getrs/geqrf plus `_bufferSize`).

**cuSPARSE / hipSPARSE** diverge more than the other pairs, so
`wwr.wrappers.sparse` wraps a deliberately narrow slice: the legacy typed
(S/D/C/Z) functions both vendors still have and cuSPARSE has **not** removed —
`bsrmv`, the `gtsv2`/`gpsvInterleavedBatch` batch solvers, `csrgeam2`, `nnz`,
`gebsr2gebsc`, `csr2gebsr`. cuSPARSE removed most of its legacy typed API
(`csrmv`, `csrsv2`, `csrmm`, `csrsm2`, `csric02`, `csrilu02`, …) in CUDA 12, so
those — though hipSPARSE keeps them — are out. The **modern generic API**
(`SpMV`, `SpMM`, `SpGEMM`, `SDDMM`, `SpSV`, `SpSM`) *is* shared, but its element
type is a runtime `cudaDataType`/`hipDataType` argument rather than a name
letter, so there is no S/D/C/Z entry point to dispatch to; call it directly
through `wwr.sparse` or the raw `wwr.cuda.cusparse` /
`wwr.hip.hipsparse` modules.

## Design

**RAII handles** — a CRTP base owns any opaque vendor handle. Move-only;
destruction calls the vendor destroy function.

**Error handling** — `gpu_check(error, policy[, location])` routes status codes
through an error policy the caller supplies; there is no default. `AbortPolicy`
(the commonly-bound choice) prints to `stderr` and calls `std::abort()`.

**Dispatch macros** — `dispatch_macros.h` (next to the sources in
`src/wrappers/blas`, `src/wrappers/solver`, `src/wrappers/sparse` and
`src/wrappers/fft`) selects the correctly-typed vendor function (S/D/C/Z
prefix, 32- or 64-bit integers, or FFT transform kind) at compile time from a
single generic template body.

**Explicit template instantiations** — modules that need them write `extern
template` declarations and `template` instantiations by hand, next to the
existing ones. There is no code generator.

---

## Presets

| Preset | What it is |
|---|---|
| `default` | Release, CUDA backend, `build/`. What compose names, and the one preset that runs every test. |
| `workstation` | Same, in `build-workstation/`, so a host build and a container build can coexist. |
| `debug` / `asan` | Debug, and Debug + AddressSanitizer. |
| `compute-sanitizer` | Debug build for the GPU memcheck/racecheck run. |
| `coverage` | Debug + `WWR_COVERAGE=ON` in `build-coverage/`: clang source-based coverage for host code. Driven by `devtools/coverage.sh`. |
| `hip` | The ROCm/HIP backend, in `build-hip/`. Needs ROCm, not CUDA. |
| `compile-time` | `WWR_COMPILE_TIME_ONLY=ON` in `build-compile-time/`: no GoogleTest fetch, no runtime tests, no GPU. |
| `ci-cuda` / `ci-hip` | What `.github/workflows/ci.yml` builds, and runnable here to reproduce it. Same shape as each other: the whole tree for that backend, runtime binaries included, then `ctest -LE gpu`. Neither needs a device (`ci-cuda` pins the architecture so configure never queries a driver). Read them as compile-and-link plus a thin runtime slice, not as a test of GPU behaviour. |

Every configure preset has its own `binaryDir`, and it needs to stay that way —
two presets sharing one build directory silently reconfigure it back and forth.

`default`, `workstation`, `debug`, `asan`, `hip`, `compile-time`, `ci-cuda`,
`ci-hip` and `coverage` have test presets. With any other preset `cpp-tier.sh`
builds and then reports that there was nothing to ctest; that is not a pass.

**There are no per-architecture presets.** `base` uses `native`, which resolves
to the card in the build machine, and `ci-cuda` pins `86` for a runner that has
none — between them that is every case this project has. The `volta` (sm_70),
`ampere` (sm_80), `hopper` (sm_90) and `portable` (sm_70;80;90) presets were
removed rather than corrected: nothing exercised them, two of them could not
configure at all, and one architecture is handled straightforwardly without
them.

Target something else with a command-line `-D` and a build directory of its
own. It has to be `-D`: that overrides a preset's `cacheVariables`, where an
environment variable does not — `CUDAARCHS` is consulted only when the cache
variable is unset, and `base` always sets it.

```bash
cmake --preset default -B build-h100 -DCMAKE_CUDA_ARCHITECTURES="90"
cmake --build build-h100
```

**Nothing may go below sm_80.** wwr wraps and tests bf16 tensor-core WMMA
(`test/gpu/wmma.cu`), an Ampere feature — `nvcuda::wmma::fragment<…,
__nv_bfloat16, …>` is only defined for sm_80+ — so the CUDA backend rejects an
explicit sub-80 `CMAKE_CUDA_ARCHITECTURES` at *configure* with `wwr requires
CUDA architecture 80 (Ampere) or newer for bf16 WMMA`. (`native` and the `all*`
keywords are resolved by nvcc, not that check, and pass through.) The toolkit
floor is lower still — CUDA 13's nvcc rejects anything below `compute_75` — but
wwr's is the binding one; a real guard here is why `volta` (sm_70) and
`portable` cannot quietly return the way they stayed broken across the CUDA 13
bump.

---

## Containers

Four files under `docker/`, in a diamond:

| File | What it adds |
|---|---|
| `Dockerfile.base` | The vendor-neutral C++23 toolchain: clang-20 + libc++, CMake, Ninja, ccache, uv/Python. No GPU SDK. |
| `Dockerfile.cuda` | `base` + the CUDA toolkit. The default, and what the devcontainer and compose build. |
| `Dockerfile.hip` | `base` + ROCm/HIP. No CUDA at all. |
| `Dockerfile.combined` | `cuda` + ROCm, for switching backends without switching containers (~40GB). |

```sh
docker/build.sh cuda        # base, then cuda -> wwr:cuda and wwr:latest
docker/build.sh hip         # base, then hip
docker/build.sh combined    # base, cuda, then combined
```

**They chain by tag, not by stage.** Each child opens with
`FROM ${PARENT_IMAGE}`, so the parent must be built and tagged before it —
`docker/build.sh` is what walks the chain, and building a child by hand with no
parent tagged fails with `pull access denied for wwr` rather than building
one. The `.devcontainer/*/devcontainer.json` files run a single `docker build`,
so each carries an `initializeCommand` that calls `build.sh` for its parent.

`combined` is the bottom of the diamond in intent only: docker has no multiple
inheritance, so it takes `cuda` as its parent and re-runs
`docker/install-rocm.sh` — which is why that block is a script rather than an
inline `RUN`. Copying `/opt/rocm` out of the `hip` image instead would miss the
apt keyring, the loader path and the `render`/`video` groups the script also
writes.

The toolkit comes from apt rather than from the `nvidia/cuda` base image,
because `Dockerfile.base` cannot be both `nvidia/cuda` and `rocm/dev-ubuntu` at
once and duplicating the toolchain stages per vendor is how they drift.
`Dockerfile.cuda` sets the `NVIDIA_*` runtime variables those images would
otherwise have provided.

Two front ends onto the same image, and they do **not** share a build directory:
the devcontainer mounts the workspace at its host path, compose mounts it at
`/workspace`, and a CMake cache records absolute paths. Compose configures into
its own `build-compose*/` directories for exactly that reason.

---

## Tests

Three tiers:

```sh
pytest -n auto -rs          # the Python suite: three files, under two seconds
devtools/cpp-tier.sh        # the C++ tier: configure, build, ctest
devtools/install-check.sh   # the package tier: install, then consume it
```

Plus one thing that is not a tier — it runs no test and launches no kernel:

```bash
devtools/cross-backend-check.sh   # does the OTHER GPU backend still compile? ~7s
```

Every tier above builds exactly one backend, so none of them can see a
portability break. A `.cu` goes through nvcc under CUDA and clang under HIP,
and nvcc is the more permissive of the two — a functor holding a `const` member
of class type, for instance, is non-trivially-copyable under clang and fails
`parallel_for`'s `device_functor` concept, while nvcc accepts it: it passes the
whole CUDA tier and only breaks on a ROCm build. Run this whenever a change
touches a `.cu`, or anything under `src/` that a `.cu` includes. It exits 2, not
0, when it cannot run, so a skip is never mistaken for a pass. CI now runs the
`ci-hip` leg on every PR too, but keep this for the seconds-long local answer
before you push.

And one measurement tool, also not a tier:

```bash
devtools/coverage.sh              # build instrumented, run ctest, llvm-cov summary of src/
```

It reports which `src/` lines the runtime ctest suites execute. Most of `src/`
is header-only modules proved at compile time, which never run — so they show as
uncovered and are not gaps. `--html` writes a browsable report under
`build-coverage/coverage/`.

The Python suite is three files, each beside the script it tests:
`test/shared/test_dispatch.py` for the build-time dispatch checker,
`test/shared/test_alias_coverage.py` for the `wwr*` alias-coverage guard, and
`.claude/hooks/test_protect_main.py` for the main-checkout guard. None touches a
built binary. There is no fast/slow split: `slow-tier.sh` and the `slow` marker
were retired once the suite fit in a second.

The package tier is the only one that looks at the *installed* result.
`cpp-tier.sh` builds the tree in place and never installs, so an export
regression passes it cleanly — and causing one is ordinary: a `.cppm` with a
`PRIVATE` include directory or define builds fine here and breaks every
consumer, because consumers compile the installed module sources and `PRIVATE`
requirements are not exported. Run it when a change touches `cmake/`, a target's
usage requirements, or anything under `src/` that a consumer imports.

### Compile-time tests

Some tests are proved by building: a failing `static_assert` fails the build of
its module. They live in `test/cuda/`, `test/hip/`, `test/gpu/` and
`test/extension/build_time/`, need no GoogleTest and no GPU to build, and can be
built alone:

```sh
devtools/cpp-tier.sh --preset compile-time        # own build dir, no GoogleTest fetch
cmake --build build --target compile_time_tests   # or, in an existing build
ctest --test-dir build -L compile_time            # runs the linked binaries
```

Note the asymmetry: *building* these needs no device, but `cuda_compile_tests`
links the CUDA driver API stubs, so *running* it needs `libcuda.so.1` — a
container started without `--gpus` fails it with `error while loading shared
libraries`. `gpu_compile_tests` runs anywhere.

`WWR_CUDA_DRIVER_STUBS=ON` (which the `ci-cuda` preset sets) resolves that
by pointing just that one test at the toolkit's own stubs, so it runs on a
driverless box too. It proves every *non-driver* dependency resolves — not
that a real driver would load. `test/cuda/CMakeLists.txt` explains why the
symlinks live in the build directory rather than the image.

### Runtime tests

A test translation unit is a plain `.cpp` compiled straight into its test
executable — not a `.cppm`, and not wrapped in a module library. That keeps
`#include <gtest/gtest.h>` out of a global module fragment, where it collides
with `import std;` in this project's GPU-heavy TUs, and means the self-registering
`TEST()` objects cannot be dropped by the linker.

Test binaries register with ctest through `wwr_add_gtest_suite_tests()`, which
creates **one ctest entry per suite** plus a guard test that fails if a suite
exists in the binary but is missing from the CMake list.

```sh
ctest --preset default                  # everything
devtools/cpp-tier.sh -- -R MemoryBuffer # one suite
```

**A green run means nothing until you know what skipped.** `pytest -rs` lists
the reasons on the Python side; on the C++ side ctest names each skip. Say what
skipped, and say whether the C++ tier ran at all, before calling a change
verified.

One thing worth knowing before you debug a failure: **`cudaError_t` is not a
boolean** — `cudaSuccess` is `0`, so `ASSERT_TRUE` on a CUDA or cuBLAS status
passes only when the call *failed*. Compare against `cudaSuccess` /
`CUBLAS_STATUS_SUCCESS` explicitly.

---

## Adding a module

```cmake
wwr_add_cxx_module_library(
  NAME wwr.wrappers.widget
  PRIMARY_INTERFACE widget.cppm
  IMPORT_STD
  LINK_PUBLIC wwr.wrappers.common
)
```

Target names use dots and are aliased to `::` — `wwr.wrappers.widget` is
consumable as `wwr::widget`. For a `.cu` of device-kernel code
use `wwr_add_gpu_device_library`, which compiles it for whichever backend
the build selected; for header-only code use `wwr_add_interface_library`.
See [`cmake/README.md`](cmake/README.md).

---

## Tooling

Everything project-specific lives in **`devtools/config.sh`** — the project name,
which devcontainer, which presets, the test command, worker counts, CPU bounds,
marker expressions, the doctor lists. Edit that file, not the scripts.

| Script | What it does |
|---|---|
| `doctor.sh` | Report what is degraded here. Non-zero only on a FAIL. |
| `devcontainer.sh` | `up` / `rebuild` / `shell` / `test` / `down` for this worktree. |
| `worktree.sh` | `add` / `rm` / `sync` / `gc` / `list` for container-backed worktrees. |
| `cpp-tier.sh` | Configure + build + ctest, logged. The C++ tier. |
| `cross-backend-check.sh` | Compile the tree for the *other* GPU backend. ~7s. |
| `install-check.sh` | Install to a throwaway prefix, then build `example/consumer` against it. |
| `prepush-tests.sh` | The fast Python gate that `git push` runs. |

`install-check.sh` is the only tier that can catch an export regression:
`cpp-tier.sh` builds the tree in place and never installs, so a module whose
compile requirements are `PRIVATE` passes it and breaks every consumer.

Use `rebuild`, not `up`, after editing `devcontainer.json` or any
`docker/Dockerfile.*`:
`up` reuses the running container and reports success having applied nothing.

`BUILD_JOBS` is the one to set conservatively — a C++23 module build is
memory-hungry per job, and an OOM-killed compiler surfaces as a bare
`ninja: build stopped` with nothing about memory in it.

---

## Notes on the toolchain

- **C++23, named modules, clang + libc++.** `CMakeLists.txt` refuses gcc.
- **`CMAKE_EXPERIMENTAL_CXX_IMPORT_STD` is a UUID tied to the CMake version.**
  Upgrade CMake and the old UUID is ignored *silently*; every `import std` then
  fails to resolve. The new value is in that release's
  `Help/dev/experimental.rst`.
- **`CMAKE_CUDA_ARCHITECTURES=native` queries a live device at configure time.**
  No GPU, no configure. Use `ci-cuda` (pinned `86`) when the build host has no
  card, or `-D` an architecture onto any preset.
- **wwr requires sm_80 (Ampere).** The CUDA backend rejects an explicit sub-80
  `CMAKE_CUDA_ARCHITECTURES` at configure, because it wraps and tests bf16 WMMA
  (an sm_80+ feature). The toolkit floor is lower (CUDA 13's nvcc bottoms out at
  `compute_75`), but wwr's is what binds. There are no per-architecture presets
  any more — see "Presets".
- Lint is deliberately narrow (`E,F,I,UP,B`) and there is **no formatter hook**.
  `ruff check .` is clean — keep it that way.
