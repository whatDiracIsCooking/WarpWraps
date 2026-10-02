# cmake/ — Build Utilities

CMake modules for C++23 module libraries and testing.

## Files

```
cmake/
├── README.md                                                  # This file
├── wwr_add_gpu_device_library.cmake                  # Macro for creating backend-neutral device-kernel static libraries
├── wwr_add_cxx_module_library.cmake                  # Macro for creating C++23 module libraries
├── wwr_add_dispatch_check.cmake                      # Build-time check that wrappers call the vendor functions their TOML table names
├── wwr_add_gtest_executable.cmake                    # Consolidates GoogleTest executable boilerplate
├── wwr_add_gtest_suite_tests.cmake                   # Registers a GoogleTest binary with ctest, one entry per suite
├── wwr_add_interface_library.cmake                   # Macro for creating INTERFACE libraries
├── wwr_add_test_executable.cmake                     # Plain (non-GoogleTest) ctest-registered executables
├── wwr_check_gtest_suites.cmake                      # Script mode: fails when a suite in the binary is missing from the CMake list
├── wwr_install.cmake                                 # Install rules, the export set and the CMake package
├── wwrConfig.cmake.in                                # Template for the installed wwrConfig.cmake
└── wwr_internal_helpers.cmake                        # Internal helper functions (alias creation, include dirs, linking)
```

---

## Build Directory Layout

After configuring and building, the `build/` directory has this structure:

```
build/
├── _deps/                  # CMake FetchContent -- GoogleTest, the only one
│   ├── googletest-build/
│   ├── googletest-src/
│   └── googletest-subbuild/
├── deps/                   # deps/CMakeLists.txt, added via add_subdirectory
├── lib/                    # Static libraries (gtest, gmock)
├── src/                    # Per-module build artifacts, one dir per backend
│   ├── cuda/               # (a CUDA build; a HIP build has src/hip instead)
│   ├── gpu/
│   └── wrappers/
└── test/                   # Test executables
```

Every configure preset has its own `binaryDir`, so the tree above is `build/`
for the `default` preset, `build-hip/` for `hip`, and so on — they must stay
separate, or two presets sharing one directory silently reconfigure it back and
forth, a full rebuild each way.

`_deps/` is managed by CMake's FetchContent — its name is a CMake convention and should not be changed. `deps/` holds local dependencies that live in the source tree under `deps/`.

---

## `wwr_add_gpu_device_library`

Creates the STATIC library that holds a module's device-kernel `.cu` sources, compiled for **whichever backend the build selected**. Every module that owns a kernel declares one next to its `wwr_add_cxx_module_library()` call.

A `.cu` in this tree means "device pass", not "nvcc" — the same sense as the `.cuh` headers it includes. Compiling one needs a language decision and, under HIP, two driver flags; a C++23 module interface unit must get neither, being an ordinary host CXX compile. Keeping the device sources in their own library is what keeps the two sets of flags apart.

### Usage

```cmake
wwr_add_gpu_device_library(
  NAME library_name
  SOURCES file1.cu [file2.cu ...]
  [LINK_PRIVATE lib1 lib2 ...]
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `NAME` | required | Target name, by convention the owning module's name plus `.device` |
| `SOURCES` | required | One or more `.cu` sources |
| `LINK_PRIVATE` | optional | Private link dependencies |

### Always applied

- `add_library(${NAME} STATIC ${SOURCES})`
- **CUDA backend:** `CUDA_STANDARD 20`, `CUDA_STANDARD_REQUIRED ON`, `CUDA_SEPARABLE_COMPILATION OFF`, `CUDA_RESOLVE_DEVICE_SYMBOLS OFF`, `POSITION_INDEPENDENT_CODE ON`
- **HIP backend:** `LANGUAGE CXX` on every source (a HIP build enables no CUDA language, and `hip::device`'s `-x hip` is gated on `$<COMPILE_LANGUAGE:CXX>`), `hip::device` linked PRIVATE, and `-B<rocm>/llvm/bin` so the `amdgcn-link` step finds ROCm's `lld`

### What it deliberately does not take

No `LINK_PUBLIC`, no `INCLUDE_DIRS_*`, no separable-compilation switch — each absence is a project invariant, not an oversight. Links are PRIVATE always (a device library's usage requirements are device-code include paths and `-x hip`; nothing linking it should inherit either), include directories arrive by linking `wwr.extension.parallel_for`, and separable compilation is OFF on every target in this tree. The macro's own header comment has the reasoning. Add a parameter when a real call site needs one.

It also creates no `::` alias, unlike the two macros below — its call site had none before the macro existed.

### Example

```cmake
wwr_add_gpu_device_library(
  NAME wwr.extension.random_normal.device
  SOURCES random_normal.cu
  LINK_PRIVATE wwr.extension.parallel_for wwr.rand.device)
```

> This replaced `WWR_ADD_CUDA_LIBRARY` (removed), a CUDA-only macro inherited from the template this repo grew from. It could not serve either call site — no HIP branch at all, and `CUDA_SEPARABLE_COMPILATION ON` — which is why it had no call sites while `rand` and `fill` hand-rolled 25 identical lines each.

---

## `WWR_ADD_INTERFACE_LIBRARY`

Creates an INTERFACE library that exposes header files (`.h`/`.cuh`) to consumers. Automatically adds the standard project include directories and creates a `::` alias.

### Usage

```cmake
WWR_ADD_INTERFACE_LIBRARY(
  NAME library_name
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `NAME` | required | Library target name |

### Always applied

- `add_library(${NAME} INTERFACE)`
- Include dirs: `INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>` + `$<INSTALL_INTERFACE:include>`
- `::` alias via `_wwr_create_alias`

### Example

```cmake
WWR_ADD_INTERFACE_LIBRARY(
  NAME wwr.core.parallel_for)
```

---

## `WWR_ADD_CXX_MODULE_LIBRARY`

Consolidates the common boilerplate for creating a C++23 module library.

### Usage

```cmake
WWR_ADD_CXX_MODULE_LIBRARY(
  NAME my_library
  PRIMARY_INTERFACE my_module.cppm
  [PARTITIONS partition1.cppm partition2.cppm ...]
  [IMPLEMENTATION impl1.cpp impl2.cpp ...]
  [LINK_PUBLIC lib1 lib2 ...]
  [LINK_PRIVATE lib3 lib4 ...]
  [IMPORT_STD]
  [NO_CUDA_DEVICE_LINKING]
  [INCLUDE_CUDA_TOOLKIT PUBLIC|PRIVATE]
  [INCLUDE_DIRS_PUBLIC dir1 ...]
  [INCLUDE_DIRS_PRIVATE dir2 ...]
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `NAME` | required | Library target name |
| `PRIMARY_INTERFACE` | required | Primary module interface file (`export module M;`) |
| `PARTITIONS` | optional | Partition interface files (`export module M:P;`) |
| `IMPLEMENTATION` | optional | Module implementation files (`module M;`) |
| `IMPORT_STD` | flag | Enable `import std;` (sets `CXX_MODULE_STD 1`) |
| `NO_CUDA_DEVICE_LINKING` | flag | Sets `CUDA_SEPARABLE_COMPILATION OFF` and `CUDA_RESOLVE_DEVICE_SYMBOLS OFF` |
| `INCLUDE_CUDA_TOOLKIT` | `PUBLIC\|PRIVATE` | Add `CUDAToolkit_INCLUDE_DIRS` as SYSTEM includes |
| `INCLUDE_DIRS_PUBLIC` | optional | Directories to add as PUBLIC include paths |
| `INCLUDE_DIRS_PRIVATE` | optional | Directories to add as PRIVATE include paths |

### Example

```cmake
WWR_ADD_CXX_MODULE_LIBRARY(
  NAME my_module.math.set_val
  PRIMARY_INTERFACE set_val.cppm
  IMPLEMENTATION set_val.cpp
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT PUBLIC
  LINK_PUBLIC cuda_runtime_api my_module::common
)
```

---

## Install and the CMake package

`wwr_install.cmake` emits every install rule and generates the package that
`find_package(wwr)` finds. The top-level `CMakeLists.txt` calls
`wwr_install_package()` once, last, when `WWR_INSTALL` is on — which it is
for a top-level build and is not when wwr is embedded via `add_subdirectory`
or `FetchContent`.

**Nothing has to be registered.** The function reads the buildsystem back and
installs every library target defined under `src/`, so a module added there is
in the package automatically and cannot be forgotten. The three rules below all
follow from one fact — a consumer of a C++23 module package *compiles the module
interface sources it installs*, because a BMI is not portable and so none can be
shipped:

1. Every compile requirement of a `.cppm` must reach the consumer. A `PRIVATE`
   include directory or define is not exported, which makes it a broken install
   rather than a private detail. `src`, `src/extension/init_state` and
   `src/extension/random_normal` grant theirs under `$<INSTALL_INTERFACE:>`,
   leaving the in-tree build unchanged.
2. Module sources need per-target install destinations — six modules under
   `src/wrappers` are each rooted at a file named `interface.cppm`.
3. A header a module unit `#include`s (`dispatch_macros.h`) is installed next to
   the installed sources, because that is where the relative include looks.

**`devtools/install-check.sh` is what verifies all of this**, by installing to a
throwaway prefix and building `example/consumer` against it. `cpp-tier.sh`
cannot: it never installs, so an export regression passes it cleanly. Every
defect in the list above was found that way rather than reasoned about.

### The optional layers — wrappers and extension

The sweep always installs the backend dir and the gpu\* (`wwr*`) core. Two
layers on top of it can be added to the package independently, each gated by
its own option and exposed as a `find_package` component. Both are off by
default: a default install is the raw core alone.

**The wrappers layer (`src/wrappers`)** ships only with
`-DWWR_INSTALL_WRAPPERS=ON` — off by default. Nothing else in `src/` links a
`wwr.wrappers.*` target, so omitting it leaves the rest of the export set
intact. When off the sweep skips `src/wrappers`, its header subtree is not
mirrored under `include/wwr/wrappers`, and the package records
`WWR_HAS_WRAPPERS` OFF — exposed by `wwrConfig.cmake` both as a plain variable
and as the `wrappers` component, so `find_package(wwr COMPONENTS wrappers)` is
refused on a core-only install.

**The extension layer (`src/extension`)** ships **only** when the build sets
`-DWWR_INSTALL_EXTENSION=ON` — off by default, because the RAII/handle/buffer/
error abstractions are a far larger surface than the core and a consumer that
wants only the core should not pay to install them. When on, the sweep also
collects `src/extension` (its module libraries *and* the
`wwr_add_gpu_device_library` `.device` archives), the extension header subtree is
mirrored under `include/wwr/extension`, and the package records
`WWR_HAS_EXTENSION` — exposed both as a plain variable and as the `extension`
component.

`example/consumer` guards each optional layer behind its `WWR_HAS_*` variable, so
one source compiles against a core-only install and a full one — its core gemm
runs on the raw `wwr*` layer, needing neither. `devtools/install-check.sh`
proves each rule the same way the extension one was always proven: the default
run ships wrappers and the consumer links them, `--no-wrappers` drops them, and
`--extension` adds the extension layer.

### Forcing a device archive into an exported target

A module whose kernel lives in a separate `.device` library (`init_state`,
`random_normal`) must force that archive's members in, or the linker drops the
explicitly-instantiated device objects and consumers fail to link. Two things
make the force survive `install(EXPORT)`:

- Use the `$<LINK_LIBRARY:WHOLE_ARCHIVE,tgt>` genex, **not** the hand-written
  `-Wl,--whole-archive $<TARGET_FILE:tgt> -Wl,--no-whole-archive`: the
  hand-written form is copied into the export file unevaluated, and
  `$<TARGET_FILE:>` of a target the consumer does not have resolves to nothing.
- Spell the target **both** ways — `$<BUILD_INTERFACE:…tgt>` with the real
  dotted name, and `$<INSTALL_INTERFACE:…>` with the exported name (the nested
  `::` alias, e.g. `wwr::extension::init_state::device`). CMake does not
  namespace the name inside `LINK_LIBRARY` when it writes the export (it does for
  ordinary link entries), so the exported spelling has to be written out by hand
  here — a name that matches nothing in the consumer degrades to a plain `-ltgt`
  the linker cannot find. `_wwr_export_name` in `wwr_install.cmake` is what
  computes that exported name for every target.

Link it PUBLIC so it is also the ordinary link dependency (which is why the
`.device` target is not in the module's `LINK_PUBLIC`). `install-check.sh` is
what catches a regression here.

---

## Tests

Three macros, and which you want depends on whether the test *runs*.

`wwr_add_test_executable` builds a plain ctest-registered executable — used
by the compile-time tiers (`test/cuda`, `test/hip`, `test/gpu`), where the proof
is that the translation unit compiled and linked at all.

`wwr_add_gtest_executable` builds a GoogleTest binary, consolidating the
module properties, the shared `main.cpp` link list and libstdc++.

`wwr_add_gtest_suite_tests` registers that binary with ctest as **one entry
per suite**, plus a `SuiteListIsComplete` guard test that fails when a suite
exists in the binary but is missing from the CMake list. The list is
hand-maintained; the guard is what keeps it honest, so when it fails, add the
suite rather than deleting the guard. It runs
`wwr_check_gtest_suites.cmake`, whose path is resolved from
`${PROJECT_SOURCE_DIR}/cmake/`.

**Why per suite, not per binary or per case.** Measured on the math suite (276
cases, RTX 3080; date not recorded): one entry per binary is ~0.9s but
`ctest -R` then selects nothing useful and a whole binary reports as one
pass/fail; one entry per case is ~92s, almost all of it ctest's ~0.28s
per-entry process spawn (the binary itself starts in 0.09s); one entry per suite
is ~10s and keeps what the granularity is for — a `-R` that selects, and skips
that do not read as passes — at a tenth of the per-case wall clock.

Typed suites (`TYPED_TEST_SUITE`) register `Suite/0`, `Suite/1`, … one per type
in the `::testing::Types` list, in order; `TYPES` names them so an entry reads
`FooTests<double>` rather than `FooTests/1`, and must list the same types in the
same order as the alias the suites are instantiated over. A target whose typed
suites span more than one `Types` list needs one call per list (the cuBLAS tests
have eight suites over `IndexTypes` and five over `ElemsAndIndices`); the guard
still sees the union, because suite names accumulate on a global property and the
guard is deferred to the end of the directory scope.

Nothing invokes a test binary by path any more — `docker/compose.yaml` drives
`ctest`, so renaming a target does not need a change there.

### `REQUIRES_GPU`, and the `gpu` label

`wwr_add_gtest_suite_tests(... REQUIRES_GPU)` puts the `gpu` ctest label on
every entry it registers. Pass it when the binary needs a live device — it
allocates device memory, launches a kernel, or creates a vendor-library handle.
The device suites under `test/extension/*` do; `test/gpu/conversions` does not,
because host fp16/bf16 conversion computes on the CPU.

The label is what lets `.github/workflows/ci.yml` build and test on a
GitHub-hosted runner, which has no card:

```
ctest --preset default    every entry
ctest --preset ci-cuda    the `-LE gpu` subset — no device needed
ctest --preset ci-hip     the same subset, the other backend
```

The two ci-* legs run the same set and so report the same count; a divergence is
a signal rather than a quirk of one backend.

**A label, not a `GTEST_SKIP`.** Nothing in `test/` gates on a device count,
and nothing should: an excluded test is named in the ctest output, where a
skipped one blends into a green run. It is the same mechanism as
`no_sanitizer`, which the `asan` preset and both compose sanitizer services
already exclude.

Two things to get right when adding one:

- **A second label must APPEND.** `set_tests_properties(... PROPERTIES LABELS
  x)` *replaces* the property, so on a `REQUIRES_GPU` target it silently strips
  `gpu`. Use `set_property(TEST ... APPEND PROPERTY LABELS x)` — see
  `test/extension/memory_buffer/CMakeLists.txt`, where two suites carry both
  `gpu` and `no_sanitizer`.
- **The `SuiteListIsComplete` guard is deliberately left unlabeled**, even for
  a `REQUIRES_GPU` target. `--gtest_list_tests` enumerates the registry without
  constructing a fixture, so it needs the binary to load but never touches a
  device — verified: the guards all pass in a container started without
  `--gpus`. That keeps the hand-written suite lists honest on the runner too.
  If a target ever grows a static initializer that talks to the driver, label
  the guard rather than deleting it.

`cuda_compile_tests` is the one entry whose label depends on a cache variable,
for a reason unrelated to devices — it links the CUDA driver stubs and cannot
*load* without `libcuda.so.1`. It carries `gpu` by default, and with
`WWR_CUDA_DRIVER_STUBS=ON` (the `ci-cuda` preset) it instead gets the
toolkit's stubs on `LD_LIBRARY_PATH` for that one test and runs. Its own
CMakeLists has the why, including why the stubs go in the build directory and
why `ENVIRONMENT_MODIFICATION` rather than `ENVIRONMENT`.
