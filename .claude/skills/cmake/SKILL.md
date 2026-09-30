---
name: cmake
description: >-
  How to work in this repo's build system — the wwr_* target macros in cmake/,
  the dotted target name / :: alias convention (and which spelling a link
  dependency uses vs a target definition), the read-the-buildsystem-back install
  sweep and the C++23-module package contract, and the three test macros. Use
  when adding or changing a CMakeLists.txt, declaring a module/interface/device
  library or a test target, wiring a new link dependency, touching install/export
  rules, or when a configure/link fails on a target name. Not for running the
  suites (that's `test`) or comment style (that's `docstyle`).
---

# Working in cmake/

The build is driven by a small set of `wwr_*` macros, one target per call. This
skill is the map and the conventions; the **full parameter tables live in
`cmake/README.md`** and the rationale in each macro's own file header. Reach for
those when you need a specific flag; reach for this when you need to know *which*
macro, *how* to name and link a target, and *what not to break*.

A build targets **exactly one backend** (`WWR_GPU_BACKEND` is `CUDA` or `HIP`,
read before `project()`). Everything below is backend-neutral unless it says
otherwise.

## The files

| File | What it gives you |
|---|---|
| `wwr_add_cxx_module_library.cmake` | The workhorse: a C++23 named-module library (`.cppm` primary + partitions + impl). |
| `wwr_add_gpu_device_library.cmake` | A module's device-kernel `.cu` archive (`*.device`), compiled for the selected backend. **No `::` alias** (see below). |
| `wwr_add_interface_library.cmake` | A header-only INTERFACE library. Wired and documented but has **no call sites** — do not assume it is dead. |
| `wwr_add_test_executable.cmake` | A plain ctest-registered executable — the compile-and-link tiers (`test/cuda`, `test/hip`, `test/gpu`). |
| `wwr_add_gtest_executable.cmake` | A GoogleTest binary (module props + shared `main.cpp` + libstdc++). |
| `wwr_add_gtest_suite_tests.cmake` | Registers a gtest binary with ctest, **one entry per suite**, plus a `SuiteListIsComplete` drift guard. |
| `wwr_add_dispatch_check.cmake` | Build-time check that each wrapper calls the vendor function its TOML table names (disassembles objects). |
| `wwr_check_gtest_suites.cmake` | Script-mode helper the drift guard runs. |
| `wwr_install.cmake` | Every install rule, the export set, and the generated package. |
| `wwrConfig.cmake.in` | Template for the installed `wwrConfig.cmake` (backend/toolchain/extension checks). |
| `wwr_internal_helpers.cmake` | `_wwr_*` helpers: `_wwr_create_alias`, `_wwr_require_args`, include/link config, `_wwr_disable_cuda_device_linking`. |

## Choosing a macro

- A module (`export module wwr.…;`) → `wwr_add_cxx_module_library`. Add
  `IMPORT_STD` if it `import std;`s. Add a sibling `wwr_add_gpu_device_library`
  if it owns a `.cu` kernel.
- Header-only, no module → `wwr_add_interface_library`.
- A test that only needs to **compile and link** → `wwr_add_test_executable`.
- A test that **runs** GoogleTest cases → `wwr_add_gtest_executable` **plus**
  `wwr_add_gtest_suite_tests` to register it. Add `REQUIRES_GPU` to the latter
  when the binary needs a live device (allocates, launches, or makes a vendor
  handle) — it labels every entry `gpu`, which CI excludes.
- A wrapper whose dispatch you want proven → `wwr_add_dispatch_check`.

Targets are declared **only** through these macros (except the handful of raw
`add_library(... INTERFACE)` aggregators in `src/CMakeLists.txt` and the
top-level `CMakeLists.txt`). `CMakeLists.txt` refuses gcc; C++23 + clang +
libc++ throughout.

## Target names and the `::` alias — the convention

**Target names are dotted and mirror the module name.** The target
`wwr.extension.fft` is what `import wwr.extension.fft;` imports. That identity is
a hard rule, not a coincidence — keep them equal.

**Every wwr target also has a `::` alias**, formed by turning every dot into
`::`: `wwr.extension.fft` → `wwr::extension::fft`. `_wwr_create_alias`
(`wwr_internal_helpers.cmake`) does this for the module and interface macros. The
raw `add_library` targets get no alias automatically, so five are aliased **by
hand** where they are defined:

| Raw target | Hand-written alias |
|---|---|
| `wwr.device` | `wwr::device` |
| `wwr.rand.device` | `wwr::rand::device` |
| `wwr.extension.parallel_for` | `wwr::extension::parallel_for` |
| `wwr_backend` | `wwr::backend` |
| `wwr_module_flags` | `wwr::module_flags` |

**A link dependency is always spelled with the `::` alias.** This is the whole
point of the convention: a link name containing `::` that resolves to no target
is a **hard configure-time error**, while a bare name (`wwr.fft`, `wwr_backend`)
silently degrades to a `-lwwr.fft` the linker only fails on later — or does not.
So `::` is the misspelled-dependency guard. Write:

```cmake
wwr_add_cxx_module_library(
  NAME       wwr.extension.fft          # definition: the REAL dotted name
  PRIMARY_INTERFACE interface.cppm
  LINK_PUBLIC  wwr::fft                  # dependencies: the :: alias
  LINK_PRIVATE wwr::backend
)
```

Dependencies passed through variables follow the same rule — the `set()` holds
the alias, so the expansion is guarded:

```cmake
set(_raw_blas wwr::cuda::cublas_v2)      # not wwr.cuda.cublas_v2
...  LINK_PUBLIC ${_raw_blas} wwr::cuda::${_lib}
```

### The three deliberate exceptions — spell these with the REAL dotted name

1. **Target definitions.** The `NAME` argument and the first argument of
   `add_library`/`add_executable`/`target_*` name the target itself, not a
   dependency — always the real dotted name.
2. **`wwr_add_gpu_device_library` archives (`*.device`).** These get **no
   alias** on purpose (the archive is pulled into its owning module with a
   `$<LINK_LIBRARY:WHOLE_ARCHIVE,…>` genex, not an ordinary link — see
   `cmake/README.md`, "Forcing a device archive into an exported target"). Link
   them by real name. Do **not** confuse these with `wwr.device` /
   `wwr.rand.device`, which are raw INTERFACE libraries that *do* have aliases.
3. **`wwr_add_dispatch_check`'s `TARGET`.** It names a target to disassemble,
   not to link — the real name.

### The installed spelling is different (and that is fine)

`install(EXPORT … NAMESPACE wwr::)` prepends `wwr::` to each target's dotted
export name, so a **consumer** links `wwr::wwr.extension.fft` — see
`example/consumer/CMakeLists.txt`. That is a *different* string from the in-tree
alias `wwr::extension::fft`, and both are correct: in-tree CMakeLists use the
nested alias, an installed consumer uses the export spelling. Linking the alias
in-tree does **not** disturb the export — CMake resolves an alias to its real
target before applying the export namespace, so the package records
`wwr::wwr.extension.fft` regardless. (Unifying the two spellings — making the
installed name `wwr::extension::fft` too, via a `::`-bearing `EXPORT_NAME` — is
tracked in issue #183; leave the export path alone until then.)

## Install and the package — read-the-buildsystem-back

`wwr_install_package()` runs **once, last** in the top-level `CMakeLists.txt`. It
does **not** take a list of targets: it sweeps `BUILDSYSTEM_TARGETS` under
`src/` and installs every library it finds, so a new module is packaged
automatically and cannot be forgotten. ALIAS targets are not in
`BUILDSYSTEM_TARGETS`, so the `::` names never double-install.

A consumer **compiles the installed `.cppm` sources** (a BMI is not portable, so
none ships). Three consequences bite anyone editing install rules:

- Every compile requirement of a `.cppm` must reach the consumer — a `PRIVATE`
  include dir or define is not exported and becomes a broken install. Grant it
  under `$<INSTALL_INTERFACE:>`.
- Module sources install to **per-target** destinations (many modules are rooted
  at a file literally named `interface.cppm`).
- A header a module unit `#include`s installs **next to** those sources.

None of this is provable by `cpp-tier.sh` (it never installs). **`devtools/
install-check.sh` is the only gate** — it installs to a throwaway prefix and
builds `example/consumer` against it. Run it after any change to `wwr_install.cmake`,
`wwrConfig.cmake.in`, a target's visibility, or the link graph. The extension
layer ships only with `-DWWR_INSTALL_EXTENSION=ON`; check both
(`install-check.sh` and `install-check.sh --extension`, or `--preset`).

## Tests — the three-macro split

`wwr_add_test_executable` proves a TU **compiles and links**;
`wwr_add_gtest_executable` builds a **running** GoogleTest binary; and
`wwr_add_gtest_suite_tests` **registers** that binary with ctest one entry per
suite (measured sweet spot between per-binary and per-case), with a
`SuiteListIsComplete` guard that fails when the binary has a suite the
hand-maintained `SUITES`/`TYPED_SUITES` list omits — **when it fails, add the
suite, don't delete the guard.** `TYPED_SUITES` needs `TYPES` in the same order
as the `::testing::Types` list. `REQUIRES_GPU` labels entries `gpu`; a second
label must **APPEND** (`set_property(TEST … APPEND …)`), or it strips `gpu`.
`cmake/README.md`, "Tests", has the timing numbers and the `gpu`-label CI story.

## Gotchas

- **Separable compilation is OFF on every target.** `_wwr_disable_cuda_device_linking`
  states that invariant; module and device macros apply it. Don't turn it on.
- **A module's include root that a `.cppm` uses from its global module fragment
  must be exported** (PUBLIC/INTERFACE), never PRIVATE — a PRIVATE root is the
  export regression `install-check.sh` exists to catch.
- **Don't hand-roll `-Wl,--whole-archive`** for a `.device` archive; use the
  `$<LINK_LIBRARY:WHOLE_ARCHIVE,…>` genex, and spell it both `$<BUILD_INTERFACE:…tgt>`
  and `$<INSTALL_INTERFACE:…wwr::tgt>` (README has why).
- **Two presets cannot share a `binaryDir`** — each has its own (`build/`,
  `build-hip/`, …); sharing one silently reconfigures back and forth.
- Verify a build-system change by **configuring both backends** (`cmake --preset
  default` and `cmake --preset hip`) — a HIP-only link list is invisible to a
  CUDA configure — then `install-check.sh` if you touched the package.

## Where the depth is

`cmake/README.md` (full parameter tables, the install contract, the test timing
study), each macro's file header (the reasoning behind its invariants), and
`docs/architecture.md` (why the layer exists). This skill is the map.
