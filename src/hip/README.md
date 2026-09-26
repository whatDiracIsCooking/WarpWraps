# src/hip — ROCm/HIP API Module Wrappers (opt-in)

This directory mirrors `src/cuda` for ROCm/HIP: C++23 module interface units
wrapping the native HIP library headers so they can be consumed via `import`
statements. It is **opt-in** and not built by default — see "Build" below.

Filled in incrementally as the `src/hip` module tree grows; each module gets a
section here once it exists.

## Design decisions

### Namespace: `gpumod::hip`, not bare `gpumod`

`src/cuda/cuda_fp16.cppm` exports `using ::half;` from CUDA's `cuda_fp16.h`.
`hip/amd_detail/amd_hip_fp16.h` defines its own, distinct global `half`. A
same-shaped bare-`gpumod`-namespace `gpumod.hip.hip_fp16` module
would make `gpumod::half` ambiguous the moment both it and
`gpumod.cuda.cuda_fp16` are imported in one translation unit. So every
`src/hip` module exports into `gpumod::hip`, never bare `gpumod`,
even where (like `hip_bf16` vs `cuda_bf16`'s `__nv_bfloat16`/`hip_bfloat16`)
there happens to be no actual collision — the namespace choice doesn't depend
on checking each header case by case.

### `hip_runtime_api` collapses `cuda.cppm` + `cuda_runtime_api.cppm`

CUDA splits the driver API (`cuda.h`, wrapped by `gpumod.cuda.cuda_h`)
from the runtime API (`cuda_runtime_api.h`, wrapped by
`gpumod.cuda.cuda_runtime_api`). HIP does not make this split —
`hip/hip_runtime_api.h` covers both — so `gpumod.hip.hip_runtime_api` is
the one module covering what those two CUDA modules cover together.

### Link `hip::host`, never `hip::device`

`find_package(hip CONFIG REQUIRED)` (ROCm's AMD platform config) defines
`hip::device` as an INTERFACE target whose `INTERFACE_COMPILE_OPTIONS` inject
`-x hip --offload-arch=<gpu>` and whose `INTERFACE_LINK_OPTIONS` inject
`--hip-link`. Those flags are for compiling actual `.hip` device-code
translation units through `hipcc`/clang's HIP compilation mode. Applied to a
plain C++23 module interface unit (a `.cppm` that only `#include`s a
declarations-only header), `-x hip` overrides the source language entirely and
breaks module compilation. Every `src/hip` module wrapping a header-only /
host-API library links `hip::host` only — it carries the actual runtime
library (`libamdhip64`, via `hip::amdhip64`) and
`__HIP_PLATFORM_AMD__=1`, the correct analogue of `CUDA::cudart`. This has no
CUDA-side counterpart to get wrong by omission: `CUDA::cudart` never carries
device-compilation flags in the first place, so the mistake is specific to
HIP's CMake package shape.

### Pre-include `<array>` before any HIP header that reaches `hip_complex.h`

`amd_detail/amd_hip_vector_types.h` (pulled in transitively by
`hip/hip_complex.h`, and therefore by every HIP library header that
`#include`s it -- `hipblas.h`, `hipblaslt.h`, `hipsolver-types.h`,
`hipsparse.h`, `hipfft.h`, in addition to `hip_complex.h` itself)
`#include`s `hip/amd_detail/host_defines.h` immediately before `#include
<array>`. Outside real HIP device-compilation mode (`__HIP__` undefined --
see "Link `hip::host`, never `hip::device`" above), `host_defines.h`'s
non-HCC branch `#define`s `__noinline__` as an *empty* object-like macro. If
`<array>` (and libc++'s `__config` through it) is first included anywhere in
the TU *after* that point, `__has_attribute(__noinline__)` inside `__config`
macro-expands `__noinline__` to nothing first, becoming `__has_attribute()`
-- zero arguments, which clang rejects outright (`too few arguments provided
to function-like macro invocation`). `#include <array>` ourselves before the
HIP header sidesteps this: `__config` is fully processed, with the real
unpolluted `__noinline__`, under its own include guard before
`host_defines.h` ever runs, so the later re-inclusion from
`amd_hip_vector_types.h` is a no-op. Confirmed necessary and sufficient by
direct experiment (bare `#include <hip/hip_complex.h>`, or any header that
reaches it, fails this way; pre-including `<array>` fixes it) -- every
module below whose header reaches `hip_complex.h` carries this same
`#include <array>` line with a shorter pointer back to this entry.

### `hip_fp4` / `hip_fp6` must stay separate translation units

`amd_detail/amd_hip_fp4.h` and `amd_detail/amd_hip_fp6.h` each define
`half_to_f16`, `half2_to_f16x2`, `hipbf16_to_bf16` and `hipbf162_to_bf16x2` as
non-inline statics in the global namespace — including both headers in one TU
is a redefinition error (see project memory
`project-hip-fp4-fp6-cannot-share-tu`). Fine for this
module-per-library tree, where `hip_fp4.cppm` and `hip_fp6.cppm` are separate
targets exactly as `cuda_fp4.cppm`/`cuda_fp6.cppm` are — but do not merge them.

## Modules

### `gpumod.hip.hip_runtime_api`

**Import:** `import gpumod.hip.hip_runtime_api;`

Wraps `hip/hip_runtime_api.h`. Like `cuda_runtime_api.h`, the extern-`"C"`
surface is declarations-only. The header also carries a handful of
templated/typed convenience overloads outside the `extern "C"` block
(`hipMalloc<T>`, `hipHostMalloc<T>`, `hipMallocManaged<T>`,
`hipBindTexture`/`hipBindTextureToMipmappedArray`/`hipUnbindTexture`
templates, `hipMallocAsync`/`hipMallocFromPoolAsync` typed overloads,
`hipLaunchKernelEx`) — the `hip_runtime.h`-style layer CUDA keeps in a
separate header this project doesn't wrap. Parity with the `cuda_runtime_api`
wrapper (declarations only, no convenience templates) means these are not
exported.

Exports the full `extern "C"` HIP runtime surface: error handling, device
management, memory allocation/copy/set (including async, 2D, 3D, pitched,
managed, and pool variants), stream and event management, kernel launch,
occupancy helpers, HIP graph construction and execution, texture/surface
management, graphics interop, and driver entry-point queries — plus flag
constants (`hipHostAlloc*`, `hipEvent*`, `hipStream*`, `hipMemAttach*`)
converted from macros to typed `constexpr` values the same way
`cuda_runtime_api.cppm` does, with a compile-time `static_assert` validating
each value before the `#undef`.

### Why nvml became two modules

CUDA's `nvml` (`gpumod.cuda.nvml`) maps to *two* separate HIP libraries,
not one: `rocm_smi` (`rocm_smi/rocm_smi.h`, the legacy/stable AMD GPU
management and monitoring library) and `amd_smi` (`amd_smi/amdsmi.h`, the
newer one, meant to eventually supersede rocm_smi). They are real,
independently usable libraries -- separate CMake packages
(`find_package(rocm_smi CONFIG REQUIRED)` / `find_package(amd_smi CONFIG
REQUIRED)`), separate shared libraries (`rocm_smi64` / `amd_smi`), separate
headers -- with overlapping but not identical surfaces. `amd_smi` does not
wrap or supersede `rocm_smi` at the API level (both are still shipped and
usable independently in this ROCm release), so unifying them into one module
would mean picking a lossy subset or an artificial merged surface neither
vendor header actually presents. Each is exported faithfully as its own
module instead, at the same 1:1-with-a-vendor-header granularity every other
`src/cuda`/`src/hip` module uses.

### `gpumod.hip.rocm_smi`

**Import:** `import gpumod.hip.rocm_smi;`

Wraps `rocm_smi/rocm_smi.h`. Like `rocm_smi.h` itself, this is a pure C API
(its whole body, including the transitively-included `<cstdint>`, is wrapped
in `extern "C"`), so there is no `hip_runtime_api.h`-style convenience-template
collision to work around here -- every type, enumerator, and function the
header declares is exported by a plain `using` declaration.

Exports the status/init-flag/performance-level/event/clock/temperature/
voltage/power-profile/GPU-block/RAS/memory/firmware-block enumerations, the
frequency/version/range/OD-voltage-curve/GPU-metrics/error-count/process-info
structs, and the full device management and monitoring function surface:
identification, PCIe, power, memory, physical state (fan/temperature/voltage),
clock/performance-level control, versioning, error/RAS queries, performance
counters, system info, XGMI, hardware topology, compute/memory partitioning,
supported-function queries, and event notification.

### `gpumod.hip.amd_smi`

**Import:** `import gpumod.hip.amd_smi;`

Wraps `amd_smi/amdsmi.h` -- see "Why nvml became two modules" above for how
this relates to `gpumod.hip.rocm_smi`. Also a pure C `extern "C"` API,
exported the same way: every declared type, enumerator, and function via
`using`.

`amdsmi.h` guards a CPU/ESMI (E-SMS, AMD EPYC System Management Interface)
extension surface -- RAPL MSR energy counters, HSMP system statistics,
performance-boost-limit control, DDR bandwidth and DIMM statistics, xGMI/GMI3
link-width control, P-state selection -- behind `#ifdef ENABLE_ESMI_LIB`. This
project does not define that macro (consistent with the rest of `src/hip`,
which targets the GPU management surface, not AMD's separate CPU tooling), so
those declarations do not exist in the translation unit this module compiles
and are not (and cannot be) exported.

Exports the init-flag/status/processor-type/clock/partition/temperature/
firmware-block/VRAM-type/RAS/memory/event/power-profile enumerations, the
XGMI/VRAM/violation/frequency/cache/firmware/ASIC/KFD/board/power/clock/
engine-usage/P2P/GPU-metrics/CPER (CXL/PCIe error record) structs, and the GPU
discovery, identification, PCIe, power, memory, physical state, clock/
performance control, version, ECC/RAS, error, performance counter, system
info, XGMI, hardware topology, compute/memory/accelerator partitioning, event
notification, firmware/VBIOS, board/ASIC info, and driver control function
surface.

### `gpumod.hip.roctracer`

**Import:** `import gpumod.hip.roctracer;`

Wraps `roctracer/roctracer.h` -- HIP's rough counterpart to CUPTI
(`gpumod.cuda.cupti`): the generic, runtime-independent callback and
asynchronous-activity tracing API. `roctracer.h` itself `#include`s
`roctracer/ext/prof_protocol.h` (the domain/activity-record wire types), so
both headers' declarations are exported together, at the same granularity as
`cupti.cppm` wrapping `cupti.h` plus its own transitively included headers.

Deliberately out of scope, unlike `cupti.cppm` (which pulls in the NVTX
callback-ID enum from `cupti_nvtx_cbid.h` without needing to wrap `nvtx.h`):

- `roctracer_hip.h` -- exists only to define one small enum (`hip_op_id_t`)
  for op IDs within the HIP async-activity domains, but pulls in the *entire*
  HIP runtime plus the large generated `hip/amd_detail/hip_prof_str.h`
  string-table header to do it. That surface already belongs to
  `gpumod.hip.hip_runtime_api`; re-including it here for one enum would
  make this module a redundant superset of that one.
- `roctracer_roctx.h` -- exists only to type the callback `data` payload for
  `ACTIVITY_DOMAIN_ROCTX`, but doing so requires including `roctx.h`, a
  distinct marker/range-annotation API (AMD's analogue of NVTX) with its own
  library (`libroctx64`) and its own public function surface. Neither NVTX nor
  ROCTX is wrapped anywhere in this project yet, so pulling in `roctx.h` here
  would silently wrap a second, unrelated library's API to get two small
  types. `ACTIVITY_DOMAIN_ROCTX` itself (the domain enumerator) is still
  exported; only the ROCTX-specific callback-data struct is left out.

No CMake package exists for `roctracer` (no `/opt/rocm/lib/cmake/roctracer/`),
so `src/hip/CMakeLists.txt` uses `find_library(ROCTRACER_LIBRARY roctracer64
...)`, the same pattern `src/cuda/CMakeLists.txt` uses for `cusolverMg`.
Unlike `hip_runtime_api`/`rocm_smi`/`amd_smi`, this target links neither
`hip::host` nor any other HIP package target -- `roctracer64` needs the HIP
runtime for neither symbols nor its own header's transitive include
(`roctracer.h` resolves `ext/prof_protocol.h` relative to its own directory,
not through anything `hip::host` provides), verified by compiling and linking
a standalone probe against it. It still needs `/opt/rocm/include` on the
include path so `#include <roctracer/roctracer.h>` itself resolves --
`find_path(ROCTRACER_INCLUDE_DIR roctracer/roctracer.h ...)` gets that without
pulling in `hip::host` just for a side effect of linking it.

### `gpumod.hip.hip_complex`

**Import:** `import gpumod.hip.hip_complex;`

Wraps `hip/hip_complex.h` (a thin platform-selector that, on the AMD platform
`hip::host` configures for, includes `hip/amd_detail/amd_hip_complex.h`). Like
CUDA's `cuComplex.h` (`gpumod.cuda.cuComplex`), every function is
`static inline` in the global namespace -- internal linkage, unreachable via
`using` -- so this module provides thin forwarding functions with the same
names, exactly as `cuComplex.cppm` does. Verified by reading
`amd_hip_complex.h` directly rather than assuming the CUDA shape carried over;
it does. Two functions (`hipCsqabsf`/`hipCsqabs`, squared magnitude) have no
`cuComplex.h` counterpart but are wrapped anyway since they're part of the
public header surface, as are the fused multiply-add functions
`hipCfmaf`/`hipCfma`.

### `gpumod.hip.hip_fp16`

**Import:** `import gpumod.hip.hip_fp16;`

Wraps `hip/hip_fp16.h` -> `amd_detail/amd_hip_fp16.h`. This is the real
`half` collision the "Namespace" design decision above warns about against
`gpumod.cuda.cuda_fp16`.

**Reading `amd_hip_fp16.h` in isolation is misleading here** -- what actually
gets compiled depends on the preprocessor state, and it's not what a first
read suggests. The header's `__half`/`__half2` are defined by
`#if defined(__clang__) && defined(__HIP__) ... #elif defined(__GNUC__) ||
defined(_MSC_VER) ... #endif`. The `__clang__`+`__HIP__` branch defines
`__half`/`__half2` with `friend inline` "hidden friend" arithmetic operators
inside the class body -- but that branch requires `__HIP__`, which is only
ever defined under real `-x hip` device-code compilation, exactly what "link
hip::host, never hip::device" above rules out. clang, however, *always*
predefines `__GNUC__` (as `4`, for GCC source compatibility) independent of
`-x hip` -- so this project's plain C++23 module compile actually takes the
`__GNUC__` branch, which `#include`s `amd_detail/hip_fp16_gcc.h`: a
deliberately minimal, portable `__half`/`__half2` with **no
arithmetic/comparison operators of their own** (only an implicit
`operator float()` and a converting constructor from `__half_raw`). Confirmed
by direct compilation, not by reading the source: `__half a, b; auto c = a +
b;` compiles, but `decltype(c)` is `float`, not `__half` -- the add silently
happens in full float precision via the implicit conversion, not as a
half-precision operation. `hip_fp16.cppm` exports the types only; there is no
operator to forward or claim ADL-reachability for on this code path.

### `gpumod.hip.hip_bf16`

**Import:** `import gpumod.hip.hip_bf16;`

Wraps `hip/hip_bf16.h` -> `amd_detail/amd_hip_bf16.h`. HIP's bfloat16 type is
`__hip_bfloat16`/`__hip_bfloat162` -- distinct names from CUDA's
`__nv_bfloat16`/`__nv_bfloat162`, so there is no actual collision here (unlike
`hip_fp16`'s `half`), but the module still exports into `gpumod::hip`
per the namespace design decision above. Unlike `hip_fp16`, this header's
operators are ordinary `static inline` free functions (verified directly --
the two headers do not share the same shape), so this module needs thin
forwarding operators, as `cuda_bf16.cppm` does for CUDA.

### `gpumod.hip.hip_fp8`

**Import:** `import gpumod.hip.hip_fp8;`

Wraps `hip/hip_fp8.h` -> `amd_detail/amd_hip_fp8.h`. Unlike CUDA's
`cuda_fp8.h` (e4m3/e5m2/e8m0), HIP's header defines four struct formats
compiled in unconditionally for host code: OCP e4m3/e5m2
(`__hip_fp8_e4m3`/`__hip_fp8_e5m2`) and AMD's original fnuz-encoded e4m3/e5m2
(`__hip_fp8_e4m3_fnuz`/`__hip_fp8_e5m2_fnuz`), each with x2/x4 vector
variants -- there is no e8m0 scaling-factor format on the HIP side. The C++
struct types have converting constructors/operators that work automatically
via ADL, as in `cuda_fp8.cppm`; the C-style `__hip_cvt_*` conversion functions
are `static inline` and get thin forwarding wrappers, also as in
`cuda_fp8.cppm`.

### No ROCm analogue: `cusolverMg`, `cublasXt`

Per issue #177's resolved investigation (closed): neither `cusolverMg.cppm`
(cuSOLVER's multi-GPU dense-solve API) nor `cublasXt.cppm` (cuBLAS's
multi-GPU API) has a ROCm counterpart. There is no multi-GPU dense-solve API
anywhere in rocSOLVER or hipSOLVER, and no `Xt`-prefixed or otherwise
multi-device API anywhere in hipBLAS or hipBLASLt -- verified directly
against the installed ROCm CMake packages and headers, not assumed. Recorded
here -- like the other deliberate absences on the CUDA side
(`cufile`, `nvJitLink`, `nvFatbin` have no ROCm analogue either) -- so
nobody re-investigates this later.
No code follows from this -- it is a deliberate absence, not a gap to fill.

### `gpumod.hip.hip_fp4` / `gpumod.hip.hip_fp6` -- blocked, not built

`src/hip/hip_fp4.cppm` and `src/hip/hip_fp6.cppm` (with
`test/hip/hip_fp4.cppm` / `test/hip/hip_fp6.cppm`) exist as complete,
reviewed source -- same shape as CUDA's `cuda_fp4.cppm`/`cuda_fp6.cppm`:
struct types with ADL-reachable converting constructors, plus thin forwarding
wrappers for the `static inline` `__hip_cvt_*` conversion functions, each
taking an `enum hipRoundMode` parameter (from `amd_detail/amd_hip_mx_common.h`,
pulled in transitively) where CUDA's take `cudaRoundMode` -- the header's own
comment notes AMD GPUs do not currently honor it, but the parameter is still
part of the call signature. Per the same-TU rule (see "Design decisions"
above), each `.cppm`'s global module fragment includes only its own header,
and neither test `.cppm` imports the other.

**But neither is wired into `src/hip/CMakeLists.txt` / `test/hip/CMakeLists.txt`
/ `test/hip/main.cpp`, because neither compiles with this project's toolchain.**
`amd_detail/amd_hip_fp4.h` and `amd_detail/amd_hip_fp6.h` both pull in
`amd_detail/amd_hip_ocp_types.h`, which unconditionally does:

```c
#if (defined(__clang__) && (__clang_major__ > 17) && defined(__HIP__)) || \
    (defined(__GNUC__) && (__GNUC__ > 13))
  ... real vector-type typedefs ...
#else
#error "Only supported by HIPCC or GCC >= 13."
#endif
```

Neither disjunct is satisfiable here: the first needs `__HIP__`, defined only
under real `-x hip` device-code compilation -- exactly what "link hip::host,
never hip::device" above forbids for a plain C++23 module interface unit; the
second needs a *genuine* GCC >= 13 front end, and while clang always
predefines `__GNUC__` (as `4`, for source compatibility -- see the
`hip_fp16` entry above for where that matters too), `4 > 13` is false, so
clang never takes this branch either. Verified by direct `clang++`
compilation with this project's exact flags (`-D__HIP_PLATFORM_AMD__=1
-stdlib=libc++ -std=gnu++23`, no `-x hip`): the header hits the `#error`
verbatim. Spoofing `__GNUC__` to a large value on the command line
(`-U__GNUC__ -D__GNUC__=14`) does get past the `#error`, but breaks worse
downstream -- `amd_hip_fp16.h`'s own `__GNUC__`-fallback branch (see the
`hip_fp16` entry above) then assumes it's compiling against glibc/libstdc++
internals that don't match what's actually included (libc++), producing
unrelated redefinition errors in `<cmath>`/`math.h`, and confirming this
isn't a one-macro fix. Real GCC >= 13 is installed in this container
(`g++ --version` reports 13.3.0), but this project's whole module build
(`clang-scan-deps`, `CXX_MODULE_STD`, the libc++ `std` module BMI every other
target links against) is clang-specific; switching compiler for two targets
is not a change these two files can make on their own, and is out of scope
for this task (`scripts/build.sh`/presets are off limits).

**Net effect: 6 of the 8 modules in this issue are live; hip_fp4/hip_fp6 are
present as source but excluded from the build**, with a `BLOCKED` comment at
each exclusion point (`src/hip/CMakeLists.txt`, `test/hip/CMakeLists.txt`,
`test/hip/main.cpp`) pointing back here. Revisit if a future ROCm release
relaxes `amd_hip_ocp_types.h`'s guard for plain clang, or if this project
ever has a reason to compile part of `src/hip` through real HIP device mode.

### `gpumod.hip.hiprtc`

**Import:** `import gpumod.hip.hiprtc;`

Wraps `hip/hiprtc.h`, declarations-only `extern "C"` (parity with CUDA's
`nvrtc.h` / `gpumod.cuda.nvrtc`). Needs its own library: `hiprtc` ships
as a separate ROCm CMake package from `hip` (`find_package(hiprtc CONFIG
REQUIRED)`, added alongside the existing `find_package(hip CONFIG REQUIRED)`
in the top-level `CMakeLists.txt`, guarded by the same
`GPUMOD_GPU_BACKEND=HIP` check), providing the `hiprtc::hiprtc` imported
target. This module links `hiprtc::hiprtc` alone -- the runtime-compilation
API is self-contained and does not also need `hip::host`'s `libamdhip64`.
`hiprtcJIT_option`/`hiprtcJITInputType` are preprocessor macro aliases for
`hipJitOption`/`hipJitInputType` (defined in `hip/linker_types.h`, which
`hiprtc.h` includes directly, not in `hip_runtime_api.h`); both real enum
types are exported since the `hiprtcLink*` function signatures need them.

## Library-linked module batch (issue #178)

### `gpumod.hip.hipblas`

**Import:** `import gpumod.hip.hipblas;`

Wraps `hipblas/hipblas.h`. CUDA counterpart: `gpumod.cuda.cublas_v2`.
`find_package(hipblas CONFIG REQUIRED)`, links `roc::hipblas` (which pulls in
`hip::host` transitively, per `hipblas-targets.cmake`'s own
`INTERFACE_LINK_LIBRARIES`).

Unlike cuBLAS, hipBLAS has no `"_v2"`-suffixed name pair to `#undef`/redirect
around -- every function is declared once, under its final name -- so this
module needs none of `cublas_v2.cppm`'s convenience-wrapper machinery.
hipBLAS does carry its own scalar typedefs cuBLAS has no equivalent for --
`hipblasHalf`, `hipblasBfloat16`, `hipblasInt8`, and (used in essentially
every `StridedBatched` function's stride parameter, 1000+ call sites)
`hipblasStride` -- exported directly here, the same way `hipDataType` (from
`hip/library_types.h`, also with no home module in this project) is.

Categorized by hipblas.h's own `\brief BLAS Level 1/2/3 API`, `\brief SOLVER
API`, and `\brief BLAS_EX API` doc-comment groupings (verified against the
header directly) rather than `cublas_v2.cppm`'s finer hand-grouping -- at
hipBLAS's ~1200-function scale, grouping by the vendor's own section markers
is both authoritative and tractable. Every one of the 1191 `HIPBLAS_EXPORT`
declarations in the header is exported and `GPUMOD_LINK_CHECK`'d.

### `gpumod.hip.hipblaslt`

**Import:** `import gpumod.hip.hipblaslt;`

Wraps `hipblaslt/hipblaslt.h`. CUDA counterpart: `gpumod.cuda.cublasLt`.
`find_package(hipblaslt CONFIG REQUIRED)`, links `roc::hipblaslt` **and**
`hip::host` explicitly -- unlike every other library in this batch,
`roc::hipblaslt`'s own `INTERFACE_LINK_LIBRARIES` is just
`roc::hipblas-common`, it does not pull in `hip::host` transitively
(confirmed in `hipblaslt-targets.cmake`, and by the build: without `hip::host`
linked here too, `hipblaslt.h`'s transitive `#include`s of `hip_bfloat16.h`
/ `hip_complex.h` / `hip_runtime.h` / `library_types.h` all fail their
`__HIP_PLATFORM_AMD__` guard).

hipBLASLt's surface is much narrower than cuBLASLt's: 24 functions total
(library/handle management, matrix layout / matmul descriptor / preference
descriptor CRUD plus attribute get/set, heuristic algorithm search,
`hipblasLtMatmul` itself, matrix transform descriptor family). No logger
callback family, no heuristics-cache capacity control, no
`cublasLtDisableCpuInstructionsSetMask` analogue, no algo introspection
beyond the heuristic search -- `hipblasLtMatmulAlgo_t` is an opaque blob
obtained only from `hipblasLtMatmulAlgoGetHeuristic`, not independently
constructed or queried the way `cublasLtMatmulAlgoInit`/`Check`/
`CapGetAttribute`/`ConfigSetAttribute`/`ConfigGetAttribute` allow on the CUDA
side. `hipblasLtGetGitRevision` and `hipblasLtGetArchName` are hipBLASLt's
own additions with no cuBLASLt counterpart.

### `gpumod.hip.hipsolver`

**Import:** `import gpumod.hip.hipsolver;`

Wraps `hipsolver/hipsolver.h`. CUDA counterparts: **both**
`gpumod.cuda.cusolverDn` **and** `gpumod.cuda.cusolverSp`.
`find_package(hipsolver CONFIG REQUIRED)`, links `roc::hipsolver` (which
pulls in `hip::host` transitively).

**Collapse, matching the `hip_runtime_api` precedent:** CUDA splits
cuSOLVER's dense surface (`cusolverDn.h`) from its sparse surface
(`cusolverSp.h`) into two headers and two modules. hipSOLVER's single
`hipsolver.h` umbrella pulls in the dense surface (`internal/hipsolver-dense.h`
+ `internal/hipsolver-dense64.h`, `hipsolverDn*`) and the sparse surface
(`internal/hipsolver-sparse.h`, `hipsolverSp*`) together, so this is **one**
module covering what the two CUDA modules cover together. It does *not* wrap
two other API surfaces `hipsolver.h` also pulls in:
`internal/hipsolver-functions.h`'s native `hipsolver*` (no Dn/Sp prefix) API
is a rocSOLVER-optimized alternative with no cuSOLVER counterpart at all --
nothing on the CUDA side to mirror it against; `internal/hipsolver-refactor.h`'s
`hipsolverRf*` (refactorization) API has a real `cusolverRf` counterpart on
the CUDA side, but this project has never wrapped `cusolverRf` either (no
`cusolverRf.cppm` exists in `src/cuda`), so there is likewise nothing to
mirror it against. Both remain declared-but-unexported in this module's TU.

**Narrowing, per issue #177's resolved investigation (closed):** the sparse
side isn't just relocated, it's much smaller. hipSOLVER's sparse surface is
11 functions total -- `hipsolverSpCreate`/`Destroy`/`SetStream`,
`hipsolverSp[SD]csrlsvchol` in both device and Host variants (4 functions),
and `hipsolverSp[SDCZ]csrlsvqr` device-only (4 functions) -- verified by
reading `internal/hipsolver-sparse.h` directly. Nothing else in
`cusolverSp.cppm` (no LU solve, no host QR, no complex Cholesky, no
least-squares, no eigensolvers, no reordering) has a hipsolverSp
counterpart; those are not stubbed, simply not exported.

The dense side (`hipsolverDn*`) mirrors `cusolverDn.cppm`'s full surface
wherever a same-named function exists, but hipsolverDn is itself a real
subset of cusolverDn: no IRS (iterative refinement solver) parameter/info
objects or `IRSX*` solvers, no mixed-precision GESV/GELS beyond the
same-precision `SS`/`DD`/`CC`/`ZZ` variants (no `SH`/`SB`/`SX`/`DH`/etc.), no
`Laswp`, no `Lauum`, no `Sytri` (inverse), no logger functions, no
emulation-strategy or math-mode setters, and a much narrower
`hipsolverDnFunction_t` (one enumerator, `HIPSOLVERDN_GETRF`, vs cuSOLVER's
three). It also *adds* six functions `cusolverDn.cppm` does not export:
`hipsolverDnXgesvdjSetMaxSweeps`/`SetSortEig`/`SetTolerance` and
`hipsolverDnXsyevjSetMaxSweeps`/`SetSortEig`/`SetTolerance` (Jacobi
parameter setters for the gesvdj/syevj info objects) -- included since the
full hipsolverDn header surface is in scope here, not just the CUDA-side
shape. Net surface: 247 dense functions + 11 sparse functions = 258
`GPUMOD_LINK_CHECK`s total.

Types owned by `gpumod.hip.hipblas` (`hipblasOperation_t`,
`hipblasFillMode_t`, `hipblasSideMode_t` -- used directly in most hipsolverDn
signatures) are not re-exported here, mirroring `cusolverDn.cppm`'s own
choice not to re-export `cublasOperation_t`/`cublasFillMode_t`. hipsolverDn's
own `hipsolverOperation_t`/`hipsolverFillMode_t`/`hipsolverSideMode_t`
(typedef aliases of the hipblas types, used by the X-prefixed 64-bit generic
API) are hipsolver's own contribution under those names, so they *are*
exported here. `hipDataType` (used by the `Xgeqrf`/`Xgetrf`/`Xgetrs`/
`Xpotrf`/`Xpotrs` generic functions) has no home module in this project's
`src/hip` tree (mirroring `cudaDataType`'s situation, independently
re-exported by both `cusolverDn.cppm` and `cusolverMg.cppm` on the CUDA
side), so it is exported directly here too.

### `gpumod.hip.hipsparse`

**Import:** `import gpumod.hip.hipsparse;`

Wraps `hipsparse/hipsparse.h`. CUDA counterpart: `gpumod.cuda.cusparse`.
`find_package(hipsparse CONFIG REQUIRED)`, links `roc::hipsparse` (which
pulls in `hip::host` transitively).

`hipsparse.h` is itself just an umbrella of `#include`s (`hipsparse-types.h`,
`hipsparse-auxiliary.h`, `hipsparse-generic-types.h`,
`hipsparse-generic-auxiliary.h`, and one header per routine family under
`internal/{level1,level2,level3,extra,precond,conversion,reorder,generic}/`)
-- the categorization in `hipsparse.cppm` follows that directory layout
directly, the same granularity `cusparse.cppm` uses for its own category
comments. This project's TU never defines `CUDART_VERSION` (no CUDA header
is `#include`d here), so every `#if(!defined(CUDART_VERSION) || ...)` branch
in `hipsparse-types.h` and `hipsparse-generic-types.h` resolves to
hipSPARSE's own modern, full-feature enum definitions, never one of the
version-gated cuSPARSE-compatibility fallback branches.

`hipsparseBfloat16` (a bf16 scalar convenience struct with implicit float
conversions) is not referenced by any function signature in this header at
all -- it exists only for caller convenience, exactly the situation
`cublas_v2.cppm` faces with `cublasBfloat16`, which it also does not export.
Skipped here for the same reason. hipSPARSE also has no counterpart to
cuSPARSE's Preview SpMM-with-custom-operators API
(`cusparseSpMMOp_createPlan`/`cusparseSpMMOp`/`cusparseSpMMOp_destroyPlan`,
`cusparseSpMMOpAlg_t`) -- verified by grepping the installed hipSPARSE
headers; nothing exists. Not stubbed, simply not exported. All 546
`HIPSPARSE_EXPORT` declarations found across the header tree are exported
and `GPUMOD_LINK_CHECK`'d.

### `gpumod.hip.hipfft`

**Import:** `import gpumod.hip.hipfft;`

Wraps `hipfft/hipfft.h`. CUDA counterpart: `gpumod.cuda.cufft`.
`find_package(hipfft CONFIG REQUIRED)`, links **`hip::hipfft`** -- note the
namespace is `hip::`, *not* `roc::`, unlike every other library in this
batch (verified directly from `hipfft-targets.cmake`; don't assume `roc::`
just because the others use it).

hipFFT's call-site direction flags (`HIPFFT_FORWARD`/`HIPFFT_BACKWARD`) are
`#define`d plain ints, not enumerators -- handled the same way
`cufft.cppm` handles `CUFFT_FORWARD`/`CUFFT_INVERSE`: validated by
`static_assert` against the macro value before `#undef`, then re-declared as
`constexpr` replacements. Note the second flag's name is `HIPFFT_BACKWARD`,
not `CUFFT_INVERSE` -- hipFFT's own naming, kept as-is.

### `gpumod.hip.hipfftXt`

**Import:** `import gpumod.hip.hipfftXt;`

Wraps `hipfft/hipfftXt.h` (which itself `#include`s `hipfft/hipfft.h` and
`hipfft/hiplibxt.h`). CUDA counterpart: `gpumod.cuda.cufftXt`. Same
package as `hipfft` above (`find_package(hipfft ...)` already covers it, no
second `find_package` call), same `hip::hipfft` imported target.

**Kept as a separate module, not folded into `hipfft`:** `hipfftXt.h` is a
genuinely separate header with its own `extern "C"` surface (multi-GPU
descriptor allocation/copy/free, descriptor-based typed and generic
execution, extended plan creation with explicit per-buffer `hipDataType`,
callback registration) -- unlike the `hip_runtime_api` collapse (HIP
textually merges what CUDA keeps in two separate headers) or the hipsolver
collapse (one umbrella header `#include`ing what CUDA keeps in two), hipFFT
keeps the *same* two-header split CUDA does: `hipfft.h`/`hipfftXt.h` mirror
`cufft.h`/`cufftXt.h` one-for-one, both in what each header declares and in
why `cufftXt.cppm` already stays separate from `cufft.cppm` on the CUDA
side. So this stays two modules here too.

`hiplibxt.h`'s types (`hipXtDesc_t`/`hipXtDesc`, `hiplibFormat_t`/
`hiplibFormat`, `hipLibXtDesc_t`/`hipLibXtDesc`,
`hipfftXtCopyType_t`/`hipfftXtCopyType`) are hipFFT's counterpart to
`cudalibxt.h`'s `cudaXtDesc_t`/`cudaLibXtDesc_t`/`cudaXtCopyType_t` --
exported here alongside `hipfftXt.h`'s own declarations, the same way
`cufftXt.cppm` exports `cudalibxt.h`'s types alongside `cufftXt.h`'s. hipFFT
has no separate "JIT callback" (unsigned-long-long-offset) typedef family
the way `cufftXt.cppm` exports `cufftJITCallbackLoad*`/`cufftJITCallbackStore*`
-- verified by grepping `hipfftXt.h`; only the one (`size_t`-offset)
callback family exists. `hipfftXtSubFormat_t` also has no
`CUFFT_XT_FORMAT_DISTRIBUTED_INPUT`/`DISTRIBUTED_OUTPUT` counterpart
(multi-node distributed formats) -- hipFFT's enum is a strict subset.

### `gpumod.hip.hiprand`

**Import:** `import gpumod.hip.hiprand;`

Wraps `hiprand/hiprand.h` -- the hipRAND *host* API only. CUDA counterpart:
the host half of `gpumod.cuda.curand`. `find_package(hiprand CONFIG
REQUIRED)`, links **`hip::hiprand`** -- also `hip::`, not `roc::` (verified
from `hiprand-targets.cmake`).

`hiprandDirectionVectors32_t`/`hiprandDirectionVectors64_t` are, on the AMD
platform `hip::host` configures for, plain C array typedefs (`unsigned
int[32]` / `unsigned long long[64]`, from `hiprand_rocm.h`) -- not struct
types the way `curandDirectionVectors32_t`/`64_t` might suggest -- verified
by reading `hiprand_rocm.h` directly rather than assuming the shape carried
over.

### `gpumod.hip.hiprand_kernel`

**Import:** `import gpumod.hip.hiprand_kernel;`

Wraps `hiprand/hiprand_kernel.h` for the device-side generator **state types**
(`hiprandState`, `hiprandStateXORWOW`, `hiprandStatePhilox4_32_10`, ...).
CUDA counterpart: the "Device API Types" half of `gpumod.cuda.curand` --
cuRAND wraps `curand.h` and `curand_kernel.h` in one module, hipRAND gets two.
The split is the point: `hiprand.h` is a declarations-only `extern "C"`
header, while `hiprand_kernel.h` drags in the per-platform rocRAND
device-generator implementation (`hiprand_kernel_rocm.h`), and host-API-only
consumers should not pay for that. Same `hip::hiprand` imported target.

**Types only.** `hiprand_init`, `hiprand_normal`, `hiprand_normal2`,
`hiprand_normal_double`, `hiprand_normal2_double` and the rest of that
header's callable surface are `__device__`-qualified, so they can only be
called from a real device-compile pass -- and a kernel translation unit
imports no modules. `src/rand.cuh` is how a device TU reaches them; a
module unit is host code, and exporting them here would produce names nothing
could call. The *state types* are different: they are plain data, and the
host is what allocates and sizes the per-thread state array.

**`<cstdio>` is included before the header, and has to be.**
`hiprand_kernel.h` -> `hiprand_kernel_rocm.h` -> `rocrand/rocrand_kernel.h` ->
`rocrand/rocrand_mtgp32.h` calls bare `printf` without including `<cstdio>`
itself; in a plain C++23 module compile that is `error: use of undeclared
identifier 'printf'`. Verified by direct compilation: the bare include fails
that way, pre-including `<cstdio>` is necessary and sufficient. A real `-x hip`
compile drags in a `printf` declaration of its own, which is why this never
showed up in device code.

**`hiprandState` is NOT `hiprandStateXORWOW` here -- unlike cuRAND.** cuRAND
makes the default state an alias (`curandState` *is* `curandStateXORWOW`).
`hiprand_kernel_rocm.h` instead expands a `DEFINE_HIPRAND_STATE` macro once
per generator, each expansion emitting a *fresh* struct deriving from the
rocRAND state it wraps -- and it expands that macro for both `hiprandState`
and `hiprandStateXORWOW` over the same `rocrand_state_xorwow`. So the two are
distinct C++ types that merely share a base and therefore a layout.
`test/hip/hiprand_kernel.cppm` pins the difference deliberately, so a future
ROCm release that collapses them is caught rather than silently relied on.

Left out: `hipRandState_t` (the header's own `\deprecated` spelling of
`hiprandStateXORWOW_t`), the MTGP32 host-side parameter types and
`hiprandMakeMTGP32KernelState` (this project wraps no MTGP32 host-side
state-construction API on either backend), and every device function.

### `hip_profile.h`: no module -- nothing to wrap

`hip/hip_profile.h` was evaluated for a `gpumod.hip.hip_profile` module
(CUDA counterpart: `gpumod.cuda.cuda_profiler_api`) and deliberately
skipped. In this ROCm version the entire header is three empty
function-like macros -- `HIP_SCOPED_MARKER(markerName, group)`,
`HIP_BEGIN_MARKER(markerName, group)`, `HIP_END_MARKER()` -- with no
`extern "C"` declarations and no linkable symbols; there is nothing a module
interface unit could `using`-declare or forward. The functions that actually
correspond to CUDA's `cudaProfilerStart`/`cudaProfilerStop`,
`hipProfilerStart`/`hipProfilerStop`, already live in `hip/hip_runtime_api.h`
and are already exported and `GPUMOD_LINK_CHECK`'d by `gpumod.hip.hip_runtime_api`
(the PR #181 pilot module) -- see its `using ::hipProfilerStart;` /
`using ::hipProfilerStop;` and `test/hip/hip_runtime_api.cppm`'s
`GPUMOD_LINK_CHECK(hipProfilerStart)` / `GPUMOD_LINK_CHECK(hipProfilerStop)`. A module that
re-exported the three no-op macros as, say, `constexpr` no-op marker functions
would not provide API parity with anything real (there's no device-code
marker call site in this project to parallel), so it would be manufactured
scope rather than a wrapper of something that exists. If `HIP_SCOPED_MARKER`
and friends ever gain real definitions in a future ROCm release, this
decision should be revisited.

## Extension modules

### hipBLAS: `gpumod.wrappers.blas` (backend-neutral)

The hipBLAS wrappers are no longer a HIP-only mirror: `src/wrappers/blas`
(`import gpumod.wrappers.blas;`) is written once against `src/blas`'s
`gpublas*` names and builds for either backend. The hipBLAS-vs-cuBLAS
differences the old mirror documented -- no `_v2` suffix, a single
`hipblasStatusToString`, `getrsBatched`/`getriBatched`'s non-const arrays, and
the five cuBLAS-only functions (`gemm3m`, `gemmGroupedBatched`,
`matinvBatched`, `tpttr`, `trttp`) -- are now resolved in `src/blas.cppm`
and `src/wrappers/blas/README.md`.

### hipSOLVER: `gpumod.wrappers.solver` (backend-neutral)

Also no longer a HIP-only mirror: `src/wrappers/solver`
(`import gpumod.wrappers.solver;`) is written once against
`src/solver`'s `gpusolverDn*` names and builds for either backend, both
legacy partitions (linear + eigen/SVD, 32 + 48 functions, fully shared) plus
the 8 functions the two backends' modern (X-prefixed) APIs share. The
cuSOLVER-only remainder (the rest of the modern API: `sytrs`/`trtri`/`larft`
plus the entire modern eigenvalue/SVD surface, which hipsolverDn has no
counterpart for at all) is not wrapped at all -- reach it through
`gpumod.cuda.cusolverDn` on a CUDA build. Neither `src/hip/extension` nor
`src/cuda/extension` exists: the extension layer is backend-neutral in full.

## Build

Built when `GPUMOD_GPU_BACKEND=HIP`. Every other preset
(`default`/`debug`/`compile-time`/`asan`) builds the CUDA backend. The `hip`
preset (`scripts/build.sh --hip`, own `build-hip/` directory) builds this tree
*instead of* `src/cuda`, together with `src`, `src/wrappers` and
`test/wrappers`:

```
scripts/build.sh --hip --clean
ctest --test-dir build-hip --output-on-failure
```

The backends are exclusive: a HIP build neither enables the CUDA language nor
finds the CUDA toolkit, and a CUDA build never gains a ROCm dependency. See
`src/README.md`.

Each target is defined in `src/hip/CMakeLists.txt` using the same
project-local `gpumod_add_cxx_module_library` CMake function `src/cuda`
uses.
