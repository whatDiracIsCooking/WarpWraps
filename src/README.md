# The `wwr*` backend switch (top level of `src/`)

A build targets exactly one GPU backend, chosen at configure time:

```
cmake --preset default                               # WWR_GPU_BACKEND=CUDA
cmake --preset hip                                   # WWR_GPU_BACKEND=HIP
```

Each module here — the `wwr*` layer, living directly under `src/` alongside the
`src/cuda`, `src/hip` and `src/wrappers` subdirectories — re-exports one raw
library module (`src/cuda` or `src/hip`) under backend-neutral `wwr*` names in
namespace `wwr`. (Exceptions: `wwr.runtime_api`, `wwr.rand`, `wwr.fp16`,
`wwr.bf16` and `wwr.vector_types` import no raw module — the first four bind
directly to an external-linkage vendor API reached through the sibling
`runtime_api.h` / `rand.h` / `fp16.h` / `bf16.h`, and `wwr.vector_types` needs
nothing to bind (its types are plain data and its constructors are a portable
brace — see `vector_types.h`). See below.) These modules are the **only** place
in the project that names both
backends; everything above them (`src/wrappers`, `test/wrappers`) is written once
against `wwr*` names and builds unchanged for either backend.

| Module | CUDA backend wraps | HIP backend wraps |
|---|---|---|
| `wwr.runtime_api` | `cuda_runtime_api.h` via `runtime_api.h` (no import) | `hip/hip_runtime_api.h` via `runtime_api.h` (no import) |
| `wwr.complex` | `wwr.cuda.cuComplex` | `wwr.hip.hip_complex` |
| `wwr.fp16` | `cuda_fp16.h` via `fp16.h` (no import) | `hip/hip_fp16.h` via `fp16.h` (no import) |
| `wwr.bf16` | `cuda_bf16.h` via `bf16.h` (no import) | `hip/hip_bf16.h` via `bf16.h` (no import) |
| `wwr.fp8` | `wwr.cuda.cuda_fp8` | `wwr.hip.hip_fp8` |
| `wwr.vector_types` | `vector_functions.h` via `vector_types.h` (no import) | `hip/hip_vector_types.h` via `vector_types.h` (no import) |
| `wwr.blas` | `wwr.cuda.cublas_v2` | `wwr.hip.hipblas` |
| `wwr.blaslt` | `wwr.cuda.cublasLt` | `wwr.hip.hipblaslt` |
| `wwr.solver` | `wwr.cuda.cusolverDn` | `wwr.hip.hipsolver` |
| `wwr.sparse` | `wwr.cuda.cusparse` | `wwr.hip.hipsparse` |
| `wwr.fft` | `wwr.cuda.cufft` | `wwr.hip.hipfft` |
| `wwr.rand` | `curand.h` / `curand_kernel.h` via `rand.h` (no import) | `hiprand.h` / `hiprand_kernel.h` via `rand.h` (no import) |
| `wwr.ccl` | `wwr.cuda.nccl` | `wwr.hip.rccl` |
| `wwr.tensor` | `wwr.cuda.cutensor` | `wwr.hip.hiptensor` |
| `wwr.comp` | `wwr.cuda.nvcomp` | `wwr.hip.hipcomp` |
| `wwr.rtc` | `wwr.cuda.nvrtc` | `wwr.hip.hiprtc` |

`gpu.fp8` is the narrow-float scalar layer above `fp16` / `bf16`, scoped to the
intersection of the vendor type pair: the OCP `E4M3`/`E5M2` fp8 formats, plus
their storage typedefs, the saturation and interpretation enums, and
float/double *narrowing* conversions. HIP's fnuz fp8 formats and CUDA's `e8m0`
scaling factor have no counterpart and stay in the raw modules. Widening back to
float goes through the scalar types' own `operator float`, which both vendors
carry — there is no direct storage→float vendor call. Unlike `fp16.cppm`, this
module `import`s its raw counterpart rather than `#include`ing the vendor
header, so `docs/architecture.md` §10 (hip_fp8 needs `<algorithm>`) is handled
once in the raw module and never recurs here. See the file header.

**`gpu.fp6` and `gpu.fp4` are written but BLOCKED on the HIP backend**, so they
are not registered in `CMakeLists.txt` — `src/fp6.cppm` / `src/fp4.cppm` exist
as complete, reviewed source (same shape as `fp8.cppm`, kept `import`-based and
as separate modules per §11, with per-type `wwrFp4Round*` / `wwrFp6Round*`
rounding enums so both can be co-imported without an ODR clash). Their HIP raw
dependencies `wwr.hip.hip_fp6` / `wwr.hip.hip_fp4` do not build with this
toolchain (`amd_detail/amd_hip_ocp_types.h` `#error`s outside real `-x hip`
device mode or GCC ≥ 13 — see `src/hip/README.md`), and a build must compile for
either backend, so wiring them would fail the HIP path. They flip on with their
raw modules.

`gpu.vector_types` re-exports the CUDA/HIP vector types (`float2`, `int4`, …)
and their `make_*` constructors into namespace `wwr` — and, unlike every other
module here, keeps the vendors' own names rather than coining `wwr`-prefixed
ones. It can, because the two backends spell these types identically
(`wwr::float2` IS `::float2`), so there is nothing to translate and a parallel
vocabulary would only add noise; `complex` has no such luxury
(`cuFloatComplex` ≠ `hipFloatComplex`, so `wwrFloatComplex` is mandatory). The
surface is the `char` … `double` scalar bases in ranks 1–3, plus rank 4 for the
bases CUDA 13 does not deprecate (`char`/`uchar`/`short`/`ushort`/`int`/`uint`/
`float`). The 64-bit 4-vectors `long4` / `ulong4` / `longlong4` / `ulonglong4` /
`double4` are left out: CUDA 13 deprecates the plain names in favour of
`*_16a` / `*_32a` alignment-pinned forms HIP has no counterpart for, so no
spelling of a 64-bit 4-vector is portable. Construction is a portable brace
(`float2{x, y}` — a CUDA aggregate, a HIP `HIP_vector_type` constructor), which
is why this module — unlike `complex` — imports no raw module and the host
`make_*` call no vendor function. `dim3` is not here: it is launch geometry, not
a data vector, and only the HIP raw module exposes it today. See the file
header.

`gpu.blas` names follow cuBLAS's typed names without the `_v2` suffix:
`wwrblasSgemm` is `cublasSgemm_v2` or `hipblasSgemm`, and `wwrblasSgemm_64` is
`cublasSgemm_v2_64` or `hipblasSgemm_64`. Its ~390 functions are each written
out in full (`WWR_FUNCTION(wwrblasSgemm, cublasSgemm_v2, hipblasSgemm)`), one
line per name, rather than generated by a prefix-pasting shorthand. See the
file header for the two places it does more than rename.

`gpu.blaslt` covers the modern "Lt" GEMM surface (`cublasLt` / `hipblaslt`) --
the handle, the matmul / matrix-layout / preference / matrix-transform
descriptors with their attribute get/set, the algo-heuristic search, and
`wwrblasLtMatmul` / `wwrblasLtMatrixTransform` themselves, plus the epilogue /
order / pointer-mode / matrix-scale enums. The surface is exactly what
`devtools/header_intersection.py --cuda vendor/cuda-13.0.x/cublasLt.json --hip
vendor/rocm-7.2.4/hipblaslt.json` reports the two backends share by name; cuBLASLt's much larger private surface (algo
introspection, logger, tile/stages enums) and hipBLASLt's own additions are
reached through the raw modules. `wwrblasLtGetVersion` is left out (shared name,
irreconcilable signatures); status success codes live in `wwr.blas`. See the
file header.

`gpu.solver` covers cuSOLVER Dense / hipSOLVER Dense. It has every legacy
(int-based) function, since the two backends match 1:1 there, but only the 8
modern `X`-prefixed functions hipSOLVER also has (potrf/potrs/getrf/getrs/geqrf
plus their `_bufferSize` functions). The CUDA-only modern eigen/SVD API is not
covered here or by `wwr.wrappers.solver`; reach it through
`wwr.cuda.cusolverDn`. Neither library has a status-to-string
function, so `wwrsolverGetStatusName`/`String` are hand-written switches, one
per backend.

`gpu.sparse` covers the legacy typed (S/D/C/Z) cuSPARSE / hipSPARSE functions
the two backends still share and cuSPARSE has not deprecated: the BSR
matrix-vector multiply (`bsrmv`), the tridiagonal/pentadiagonal batch solvers
(`gtsv2*` / `gpsvInterleavedBatch*`), CSR matrix addition (`csrgeam2*`) and a
few format conversions (`nnz`, `gebsr2gebsc*`, `csr2gebsr*`), plus the handle,
stream, pointer-mode, error-string and matrix-descriptor helpers a caller needs.
The modern generic API (`SpMV`, `SpMM`, `SpGEMM`, ...) is deliberately *not*
wrapped here: its element type is a runtime `cudaDataType`/`hipDataType`
argument rather than a name letter, so it needs no S/D/C/Z dispatch -- reach it
through `wwr.cuda.cusparse` / `wwr.hip.hipsparse`. One spot does more than
rename: cuSPARSE spells the byte count of `gebsr2gebsc_bufferSize` /
`csr2gebsr_bufferSize` as `int*` where hipSPARSE spells it `size_t*`, so these
keep hipSPARSE's `size_t*` signature on both backends, forwarding through an
`int` on CUDA (the mirror of `gpu.blas`'s `getrsBatched` shims). cuSPARSE-only
functions (the Preview `SpMMOp` API, `CreateSlicedEll`) and the legacy typed
functions cuSPARSE removed but hipSPARSE keeps (`csrmv`, `csrsv2`, `csrmm`,
`csrsm2`, `csric02`, `csrilu02`, `gemvi`, the HYB path, ...) are left out.

`gpu.fft` covers the base (single-GPU) cuFFT / hipFFT API: the plan
lifecycle, one-shot and two-step plan creation, work-size estimation and query,
work-area management, and the six typed `wwrfftExec{C2C,R2C,C2R,Z2Z,D2Z,Z2D}`
functions. The multi-GPU `Xt` surface is not wrapped — reach it through
`wwr.cuda.cufftXt` / `wwr.hip.hipfftXt`. Two spots do more than rename:
`WWRFFT_FORWARD`/`WWRFFT_INVERSE` unify cuFFT's `INVERSE` and hipFFT's
`BACKWARD` spelling, and — since neither library has a status-to-string
function — `wwrfftGetStatusName`/`String` are hand-written switches, one per
backend. Only the 14 result codes both enums share get a `WWRFFT_*` alias; a
backend-only code (cuFFT's `MISSING_DEPENDENCY`, hipFFT's `PARSE_ERROR`, …) is
reached through the raw module. `wwrfftGetProperty` is likewise omitted — the
two disagree on its property-type enum. See the file header.

`gpu.rand` covers the cuRAND / hipRAND host API — generators, the
`wwrrandGenerate*` functions, Poisson distributions, quasirandom direction
vectors — plus the device generator **state types** (`wwrrandState`,
`wwrrandStatePhilox4_32_10`, …). The device **functions** (`curand_init`,
`curand_normal`, …) are not here and cannot be: they are `__device__`-qualified,
so only a real device-compile pass can call them, and a kernel translation unit
imports no modules. `rand.h`'s device-pass-gated section is where those live —
see "Device headers" below. The state types stay on the module side because they
are plain data and the *host* is what allocates and sizes the per-thread state
array.

`wwr.rand` imports no raw vendor module. It reaches the vendor headers through
`rand.h` — `curand.h` + `curand_kernel.h` on CUDA, `hiprand.h` +
`hiprand_kernel.h` on HIP (cuRAND packages host and device together, hipRAND
keeps them apart) — and binds its `wwr*` names straight to the `::curand*` /
`::hiprand*` declarations with `backend.h`'s `_RAW` macros. This is safe here and
nowhere else in the layer: the cuRAND / hipRAND host API is a real
external-linkage library, so a reference or type alias needs only the
declaration. A module wrapping static-inline vendor math (`complex`, `fp16`,
`bf16`) cannot do this and keeps its `import` (see "How a name is mapped" below
and `docs/architecture.md` §12). The single-vendor raw modules
(`wwr.cuda.curand`, `wwr.hip.hiprand`, `wwr.hip.hiprand_kernel`) remain
installable, off this path. Names that only one backend has are left out; the
file header lists them. Three things to watch:

- The RNG-type enumerator **values** differ: `WWRRAND_RNG_PSEUDO_DEFAULT` is
  100 on cuRAND and 400 on hipRAND. Use the names, never the numbers.
- `wwrrandGetScrambleConstants32`/`64` take hipRAND's const-correct
  `const unsigned int**` (or `const unsigned long long**`) on both backends. On
  CUDA they are forwarding functions around cuRAND's non-const signature.
- **`wwrrandState` is `wwrrandStateXORWOW` on CUDA and a distinct type on
  HIP.** cuRAND makes the default state an alias; hipRAND emits a separate
  struct per generator over a shared rocRAND base. Same layout, different C++
  type — so a `wwrrandState*` where a `wwrrandStateXORWOW*` is wanted compiles
  on CUDA and fails on HIP. Pick one spelling and keep it. Verified on ROCm
  7.2.4 and pinned by `test/hip/hiprand_kernel.cppm` and `test/gpu/rand.cppm`,
  which assert the two backends' opposite answers separately.

`gpu.rand` breaks the first rule below: it was ported whole, before
`src/wrappers` used any of it. Every name in it is therefore checked in full by
`test/gpu/rand.cppm`. `src/extension/init_state` and `src/extension/random_normal`
now use the state types and, through `rand.h`, the device functions; the rest
of the host API is still checked only by that test.

`gpu.ccl` covers multi-GPU **collectives** — NCCL / RCCL. This is the one
popular vendor library where the two backends genuinely agree on the API: RCCL
is a source-compatible reimplementation of NCCL, so both spell the entire
surface with the identical `nccl*` / `NCCL_*` names and there is no prefix
divergence to bridge (`wwrcclAllReduce` is `ncclAllReduce` on both). It carries
the measured intersection — 36 functions (the collectives, P2P
`wwrcclSend`/`Recv`, communicator lifecycle, grouping, custom reductions), the
10 shared types, the reduction-op / datatype enums, and the scalar flag
constants. Three things to watch:

- **The surface is measured, not assumed — against the installed libraries.**
  RCCL's version leads the packaged NCCL's, so a few RCCL names —
  `ncclGather`/`Scatter`, `ncclAllToAll{,v}`, `ncclAllReduceWithBias`, and
  `ncclResetDebugInit` (in NCCL's upstream header, but not the `libnccl-dev` the
  CUDA image ships) — have no counterpart in the installed NCCL and are therefore
  **absent from `wwr.ccl`**, reachable only through the raw `wwr.hip.rccl`
  module. Run `devtools/header_intersection.py --cuda vendor/cuda-13.0.x/nccl.json
  --hip vendor/rocm-7.2.4/rccl.json` to reproduce.
- The `NCCL_*` values are `#define` macros, which a module cannot re-export; the
  raw modules turn the scalar flag/param ones into `constexpr` (as `cufft` does
  for its direction flags) and `wwr.ccl` aliases those to `WWRCCL_*`. The
  version macros (`NCCL_MAJOR`, `NCCL_VERSION_CODE`) are not aliased — their
  values are backend-specific and `wwrcclGetVersion` is the portable query — and
  the two struct-initializer macros (`NCCL_CONFIG_INITIALIZER`,
  `NCCL_SIM_INFO_INITIALIZER`) are omitted too.
- Like `gpu.rand`, every name is checked in full by `test/gpu/ccl.cppm` (there
  is no runtime or dispatch test downstream — the collectives need a real
  multi-GPU job to run), and enumerator **values** are pinned there per backend.
  NCCL's CUDA library is not part of the CUDA toolkit; `docker/Dockerfile.cuda`
  installs `libnccl-dev`, while RCCL is already in the base ROCm image.

`gpu.tensor` covers **tensor primitives** — contraction, reduction, permutation,
element-wise — cuTENSOR / hipTensor. Unlike `ccl`, this is **not a hipify pair**:
cuTENSOR is closed-source over CUDA and hipTensor is built on composable-kernel,
two independent implementations that merely mirror each other's naming. The
intersection was therefore **measured, not assumed** (run
`devtools/header_intersection.py --cuda vendor/cuda-13.0.x/cutensor.json --hip
vendor/rocm-7.2.4/hiptensor.json` to reproduce): of
cuTENSOR 2.8.1.0's 45 functions and hipTensor 2.2.0's 38, **37 are shared by
name with positionally identical signatures**, and `wwr.tensor` carries 36 of
them plus 18 shared types and 93 value-agreeing constants (28 data types —
hipTensor numbers `HIPTENSOR_R_*` to match `cudaDataType_t` — 29 operators, 11
status codes, and the workspace/attribute/mode enums). The measurement made it a
clear yes, opposite to the cuDNN/MIOpen verdict (`docs/architecture.md` §19).
Three things to watch:

- **Two divergences need more than an alias** (`docs/architecture.md` §4, §5).
  The compute descriptors (`WWRTENSOR_COMPUTE_DESC_16F/16BF/32F/64F`) are
  `extern const` opaque *pointers* on cuTENSOR but enum *values* on hipTensor —
  no single macro spans both, so they sit in a per-backend `#if`; the neutral
  `wwrtensorComputeDescriptor_t` type still matches the backend's own
  `descCompute` parameter, so a caller passes them straight through.
  `wwrtensorLoggerSetLevel` is a forwarding function, not an alias: cuTENSOR
  takes `int32_t`, hipTensor its own `hiptensorLogLevel_t` enum, so the neutral
  signature is a uniform `int32_t` (HIP casts).
- **Backend-only names are absent**, reachable only through the raw modules: the
  cuTENSOR block-sparse and trinary-contraction APIs, the extra
  operators/algos/compute-descriptors, `cutensorGetCudartVersion` and
  `cutensorPlanPreferenceGetAttribute`; and on the HIP side
  `hiptensorGetHiprtVersion`, the `hiptensorLogLevel_t` enum, and
  `HIPTENSOR_ALGO_ACTOR_CRITIC`. The `CUTENSOR_VERSION` / `HIPTENSOR_VERSION`
  macros are backend-specific; `wwrtensorGetVersion` is the portable query.
- Like `gpu.rand` / `gpu.ccl`, every name is checked in full by
  `test/gpu/tensor.cppm`, with enumerator **values** pinned per backend. Both
  libraries are found by hand: cuTENSOR ships no CMake config and is not a
  CUDA-toolkit component (`docker/install-cuda.sh` republishes it under
  `/opt/nvidia/cutensor`); hipTensor *has* a package, but `hiptensor::hiptensor`
  drags `hip::device` into its interface, which would force device compilation
  on host consumers, so `wwr.hip.hiptensor` `find_library`s `libhiptensor` and
  links `hip::host` instead (see `src/hip/README.md`). hipTensor was kept off the
  `ROCM_PRUNE` list.

`gpu.rtc` covers **runtime compilation** — NVRTC / hipRTC. hipRTC was written
"for parity with nvrtc", so this is a hipify pair like `ccl`, but a partial one:
the surface is the measured intersection (run
`devtools/header_intersection.py --cuda vendor/cuda-13.0.x/nvrtc.json --hip
vendor/rocm-7.1.0/hiprtc.json` to reproduce) — 23 names shared by spelling: the
program lifecycle (`wwrrtcCreateProgram` / `DestroyProgram`), `wwrrtcCompileProgram`,
the log (`wwrrtcGetProgramLog` / `Size`) and name-expression
(`wwrrtcAddNameExpression` / `wwrrtcGetLoweredName`) queries, `wwrrtcVersion` /
`wwrrtcGetErrorString`, the `wwrrtcProgram` / `wwrrtcResult` types, and the 12
result codes both `nvrtcResult` / `hiprtcResult` enums share. One spot does more
than rename:

- **The compiled-output getter diverges in name.** NVRTC emits PTX
  (`nvrtcGetPTX` / `nvrtcGetPTXSize`), hipRTC emits a code object
  (`hiprtcGetCode` / `hiprtcGetCodeSize`). `wwr.rtc` picks hipRTC's neutral
  spelling, `wwrrtcGetCode` / `wwrrtcGetCodeSize`, mapping each to the backend's
  own — the one shape in the `WWR_FUNCTION` table whose CUDA and HIP names are
  not a prefix swap, so `test/gpu/rtc.cppm` pins the two separately.

Single-backend names are left out, reachable only through the raw modules:
NVRTC's CUBIN / LTO IR / OptiX IR getters, its PCH and flow-callback surface, its
supported-arch query and the `NVRTC_ERROR_CANCELLED` / PCH / time-trace result
codes; and hipRTC's **linking** surface (`hiprtcLink*`, with the `hipJitOption` /
`hipJitInputType` enums the `hiprtcJIT_option` / `hiprtcJITInputType` macro
aliases expand to — see `src/hip/README.md`), its bitcode getters and
`HIPRTC_ERROR_LINKING`. Like `gpu.rand` / `gpu.ccl`, `gpu.rtc` breaks the first
rule below — it was ported whole before any `src/wrappers` consumer exists — so
every name in it is checked in full by `test/gpu/rtc.cppm`.

## How a name is mapped

`backend.h` holds the switch. It is included only in a `src` module's
global module fragment, and module units do not export macros, so nothing
outside this directory can see it.

```cpp
WWR_TYPE(wwrStream_t, cudaStream_t, hipStream_t)          // using wwrStream_t = ...;
WWR_VALUE(wwrSuccess, cudaSuccess, hipSuccess)            // inline constexpr auto wwrSuccess = ...;
WWR_FUNCTION(wwrStreamCreate, cudaStreamCreate, hipStreamCreate)
                                                          // inline constexpr auto& wwrStreamCreate = ...;
```

Where the backends differ only by prefix, a module defines a one-argument
shorthand, so each name is written once:

```cpp
#define WWR_RT_FUNCTION(x) WWR_FUNCTION(wwr##x, cuda##x, hip##x)
WWR_RT_FUNCTION(StreamCreate)
```

`WWR_FUNCTION` binds a reference to the backend's own function. The signature
is never restated, so it cannot drift from the header: a call through
`wwrStreamCreate` compiles to a direct call of `cudaStreamCreate` /
`hipStreamCreate`, and a call that does not match that backend's signature
fails to compile on that backend.

The `_RAW` variants (`WWR_TYPE_RAW` / `WWR_VALUE_RAW` / `WWR_FUNCTION_RAW`) are
the same three, but resolving to the vendor's own global names (`::curand*` /
`::hiprand*`) instead of the `::wwr::cuda` / `::wwr::hip` module re-exports. They
are for a module that reaches its vendor header by `#include` — through a
src/-root sibling `.h` — rather than by importing the raw module, which is sound
only when the vendor host API has external linkage. `wwr.rand` is the sole user;
see its section above and `docs/architecture.md` §12.

## Device headers — the other half of the switch

A `.cppm` here is a module, and code that imports one is host code. A kernel
translation unit — a `.cu` under CUDA, a `-x hip` compiled source under HIP —
imports no modules at all, so none of the modules above can serve it. Three
device headers are the counterpart for that case: same `wwr*` names, reached by
`#include`, with the backend resolved through `selected_backend.h` (which reads
the compiler's own device-compile macro before `backend.h`'s CMake define)
rather than by an `import`. Under `src/extension`, `parallel_for.cuh` is a `.cuh`
and `#error`s if included outside a
device-compile pass, a *separate* check on `__CUDACC__` / `__HIP__` /
`__HIPCC__`, because `selected_backend.h` would otherwise resolve a backend in a
host compile too, and what it carries is device-only. `cooperative_groups.h` and
`wmma.h` are `.h`: just as device-only, but instead of the `#error` their *whole*
body sits behind that same `__CUDACC__` / `__HIP__` / `__HIPCC__` check, so a
host TU that includes one gets an empty header rather than an error. They are
host-*safe* without being host-*usable* — no `wwr*` name reaches a host consumer
— which is what makes them `.h`, not `.cuh`.

(`runtime`, `rand`, `complex`, `fp16`, `bf16`, `fp8`, `vector_types` and `atomic`
are the *shared-type* headers these build on, not device-only ones: their device
wrappers fold into `runtime.h` / `rand.h` / `complex.h` / `fp16.h` / `bf16.h` /
`fp8.h` / `vector_types.h` / `atomic.h`'s own device-pass-gated sections, so a
device consumer `#include`s the one neutral `.h` — the `.h` + `.cppm` shape for a
header with both host and device symbols. `atomic.h` is the leanest: its host
half is two project-owned enums (`wwrMemoryOrder`, `wwrThreadScope`) rather than a
vendor type, and `wwr.atomic` re-exports only those — the scoped-atomic
*operations* stay device-only (docs/architecture.md §20). `runtime.h` also carries
`WWR_GRID_CONSTANT`, `WWR_WARP_SIZE` and the full vendor runtime in that section,
the device half folded in from the former `runtime.cuh`. `rand.h` links
`wwr.rand.device` for its RNG library; the rest ride `wwr.device`. See the
`gpu.*` sections above and the respective `src/*.h`.)

The `.cuh` reach both — the backend selection and that guard — through one
header, `device_guard.h`, which is `selected_backend.h` plus the device-pass
`#error` and nothing else. `parallel_for.cuh` and `math.cuh` `#include` it
directly, beside `runtime.h` for the vendor runtime header their signatures /
intrinsics need. `cooperative_groups.h` and `wmma.h` need no `#error`: they
`#include` `runtime.h` from inside their own device gate, which gives them the
backend selection and `WWR_WARP_SIZE` in a device pass and nothing at all in a
host compile (`runtime.h` is host-safe and reaches `selected_backend.h` directly,
not `device_guard.h`). The guard cannot move into `selected_backend.h` itself,
which is "for anything" and must not `#error` in the host compiles the bridges
do; `device_guard.h` is the device-only layer above it that can. It carries no
vendor header and no target-specific content, so the `.cuh` that need it share it
regardless of which target they are in.

| Header | Provides | Link |
|---|---|---|
| `runtime.h` | `wwrStream_t` (always); `WWR_GRID_CONSTANT`, `WWR_WARP_SIZE` and the full vendor runtime header (device-pass-gated section) | `wwr.device` |
| `cooperative_groups.h` | nothing of its own — the vendor's `namespace cooperative_groups`, reached through the one `#include` that differs | `wwr.device` |
| `wmma.h` | `wwrwmma`, aliasing the vendor's `nvcuda::wmma` / `rocwmma` — the namespace is the only name here, the spellings inside it agree | `wwr.device` |
| `atomic.h` | `wwrMemoryOrder`, `wwrThreadScope` (always — also re-exported by `wwr.atomic`); the `wwrAtomic*` forwarders — `Load`/`Store`/`Exchange`, `CompareExchange{Strong,Weak}`, `Fetch{Add,Sub,And,Or,Xor,Min,Max}`, each carrying an explicit order and scope (device-pass-gated section) | `wwr.device` |

The `rand`, `complex`, `fp16`, `bf16`, `fp8` and `vector_types` device wrappers
are **not** in this table: each lives in its `.h`'s device-pass-gated section
alongside that module's types, reached by `#include "<name>.h"`. `rand`'s
(`wwrrand_init`, `wwrrand_normal`, …) link through `wwr.rand.device`; the rest
(`complex`'s `make_wwr*Complex`/`wwrC*`, `fp16`'s `wwrFloat2Half`/`wwrHalf2Float`,
`bf16`'s `wwrFloat2Bfloat16`/`wwrBfloat162Float`, `fp8`'s
`wwrFloat2Fp8`/`wwrDouble2Fp8`, `vector_types`'s `make_*`) ride `wwr.device`.

`cooperative_groups.h` and `wmma.h` are the two with no `.cppm` counterpart:
every entity they hand a kernel is `__device__`-only, structurally, so there is
nothing a module could export to host code. `atomic` was a third until its enums
were split out — its scoped-atomic *operations* are `__device__ __forceinline__`
and stay device-only (a host-side `cuda::atomic` operations module was weighed
and declined; docs/architecture.md §20), but the two enums they take are plain
data that crosses the boundary, so `wwr.atomic` exports those and `atomic.h`
became a shared-type `.h` — which is why it sits with `runtime.h` in the table
above, not here. All three device sections reach the vendor runtime header and
the backend switch through `runtime.h`, now that the former `runtime.cuh` is
folded into it, taking that `#include` from inside their device gate — so they
pull it only in a device pass and need no `#error` of their own.
`cooperative_groups.h` and `wmma.h` want it for `WWR_WARP_SIZE` — what a portable
tile size is built from for one and a wave index for the other, both APIs being
whole-warp collectives; `atomic.h`'s device section wants it for the runtime
header that on HIP declares the `__HIP_MEMORY_SCOPE_*` constants its builtins
take.

`cooperative_groups.h` is also the only one here that defines **no `wwr*`
names at all**. Both
vendors put cooperative groups in `namespace cooperative_groups` and agree on
the spellings inside it, so there is no vendor name for this layer to hide and
a forwarding function per entry point would rename each name to itself. The
file is the `#include` switch and a header listing the divergences callers must
avoid — two of which are silent. It carries no wrapper layer: nothing in the
tree would compile it, and `test/gpu`'s host-compiled module tests structurally
cannot, every entity being `__device__`-only.

`wmma.h` is that file one step short. AMD wrote rocWMMA to be
source-compatible with NVIDIA's WMMA, so the spellings inside agree exactly as
cooperative groups' do — but the namespaces do not (`nvcuda::wmma` against
`rocwmma`), and a caller aliasing one itself would need the backend `#if` that
only this layer may carry. So the file is the `#include` switch plus exactly one
name, `wwrwmma`, and nothing else; the divergences it cannot absorb — 16x16x16
being the only portable shape, `wwrBfloat16` not being a usable element type
under HIP, and `fragment::num_elements` changing between HIP's two compile
passes — are documented rather than wrapped.

`atomic.h` genuinely wraps, where `cooperative_groups.h` renames nothing and
`wmma.h` renames one namespace. Its device section forwards a real divergence:
the scoped, memory-ordered atomics are
`cuda::atomic_ref<T, Scope>` (from libcu++'s `<cuda/atomic>`) on CUDA and
`__hip_atomic_*` clang builtins on HIP — two spellings for one operation. That
clears the bar the **common** atomics do not: `atomicAdd`/`CAS`/… are spelled
identically on both, ride the vendor runtime header, and are wrapped by nothing
(docs/architecture.md §15, `test/gpu/atomics.cu`). It defines two enums of its
own — `wwrMemoryOrder` and `wwrThreadScope`, because neither vendor's spelling
is portable — and one `__device__ __forceinline__` forwarder per operation. The
enums sit *above* the device gate (they are plain, host-visible data, which is
what lets `wwr.atomic` re-export them and makes this a shared-type `.h`, not a
device-only `.cuh`); the forwarders and the vendor-constant mappings sit inside
it. The scope is a template parameter and the order a function argument, the
split libcu++ forces. The four portable scopes and the mapping to each vendor's
constant — silent to get wrong, the §16 class of trap — are in
docs/architecture.md §20, verified against both memory models; CUDA's `cluster`
and HIP's `wavefront` stay vendor-only, as §15 leaves `atomicAdd_block`/`_system`
out. libhipcxx (`<hip/std/atomic>`) is absent from the pinned ROCm, so the
builtin forwarding is the only portable design — measured, in §20.

`wwrStream_t` is not among them: it lives in `runtime.h`, a `.h` rather than a
`.cuh` precisely because it does *not* `#error` outside a device pass -- see "The
switch points" below. `runtime.h` is a src/-root *shared-type* header (a
companion to `complex.h`, now carrying a device-pass-gated section of its own):
`#include`d, not imported, by the device headers that reach the runtime through
it (`atomic.h`, `cooperative_groups.h`, `wmma.h`, `parallel_for.cuh`) and the
host TUs that declare a stream-taking function across the boundary in a GMF or
plain `.cu` (the `*_bridge.h`, `example/warp_reduce`), reached bare through the
`src/` include root. `wwr.runtime_api` exports the same `::cudaStream_t` /
`::hipStream_t` to importers -- but `runtime_api.cppm` does not include
`runtime.h`: it draws the handle (and the rest of the runtime surface) from its
own vendor-include header `runtime_api.h`, which `#undef`s the allocation-flag
macros that would otherwise collide with its `WWR_RT_VALUE` expansions.
`runtime.h`'s *always-on* part names only the handle type, so it binds no flag
values and needs no such `#undef`; its *device section*, though, now binds the
whole neutral runtime surface for device `.cu`/`.cuh` TUs (which cannot import
`wwr.runtime_api`) and reuses that same `runtime_api.h` flag dance to do it --
through the one list both sites share, `runtime_api_surface.h`. `wwrrandState`
lives the
same way, in the src/-root
header `rand.h`, `#include`d (not imported) by `rand.cppm` and the two
`*_bridge.h` (and reached in a device pass by `rand.h`'s own gated generators)
-- with one difference "`_bridge`, shared-type `.h`, and why the extension is not
enough" below spells out: unlike `complex.h` / `runtime.h`, `rand.h`'s vendor
header is not cheap.

The types are the *same types* the modules export under the same names, so a
buffer allocated by host code that imports `wwr.rand` is exactly what a
kernel naming `wwrrandState` expects, and an `extern template` declared in a
`.cppm` links against a definition compiled in a `.cu`.

Two things these headers do that the modules do not have to, and one they
deliberately decline to do. Each is stated once.

| | Why |
|---|---|
| **`WWR_WARP_SIZE`** | Neither backend offers a usable compile-time warp size: `warpSize` is not a constant expression on either, and HIP's `__AMDGCN_WAVEFRONT_SIZE__` disagrees between its own two passes. Set at configure time instead, and validated against no real device. |
| **Forwarding templates, not `WWR_FUNCTION`** | `curand_normal` and friends are an overload set on CUDA and a function template on hipRAND; a function reference can name neither, so `rand.h`'s device section writes a thin `__device__` template per name. Being `__device__`-only, they are why a `parallel_for` functor's `operator()` is `__device__` and why its callability is checked on the kernel, not the host-side concept. |
| **Declining to normalise a return type** | `ballot()` is 32-bit on CUDA and 64-bit on HIP, so a caller storing a wave64 ballot in an `unsigned` is silently wrong; `thread_rank()`/`num_threads()` vary too. `cooperative_groups.h` wraps none of it — the divergence is documented rather than wrapped. |

## Rules

- **Only list names something above this layer uses.** Add a name here, once,
  when `src/wrappers` needs it. (`gpu.rand` is the one exception; see above.)
- **Backend differences are resolved here, never above.** A name that differs
  beyond its prefix, a function with default arguments or overloads the
  reference cannot carry, or an API one backend lacks gets an explicit `#if`
  block in the module, with a hand-written forwarding function if needed.
  An overload set can instead be bound to one overload with a typed
  reference, as `wwrMalloc` does (`hipMalloc` has a `template<class T>`
  overload): `inline constexpr wwrError_t (&wwrMalloc)(void**, std::size_t) = ...;`.
- **The backend define is PRIVATE, and never PUBLIC.**
  `WWR_GPU_BACKEND_CUDA` / `WWR_GPU_BACKEND_HIP` come from the
  `wwr_backend` INTERFACE target. Every module here links it PRIVATE, so
  it never reaches an importer. `test/gpu`, *above* `src`, links it PRIVATE
  too, because it includes one of the switch headers here from a **host**
  compile and so has no `__CUDACC__`/`__HIP__` to switch on -- it names both
  backends deliberately. (`src/extension/init_state` and
  `src/extension/random_normal` link it PRIVATE for the same reason, their module
  units including a `*_bridge.h` in their global module fragment.)
  Linking it is not a licence to `#if` on the backend -- the `#if` belongs in the switch
  points below and nowhere else. That goes
  double for `selected_backend.h`, which is the most convenient switch in the
  project and therefore the easiest one to misuse from above this layer.

## The switch points

There are four, because not every consumer can `import` and not every consumer
is a device pass. `selected_backend.h` answers *which backend* once; the other
three build on its answer rather than re-deriving it.

| Header | For | Picks the backend from |
|---|---|---|
| `selected_backend.h` | **anything** | `__CUDACC__` / `__HIP__` / `__HIPCC__` first, then `WWR_GPU_BACKEND_*`. Defines `WWR_SELECTED_CUDA` / `WWR_SELECTED_HIP` and nothing else |
| `backend.h` | the `.cppm`s here | `WWR_GPU_BACKEND_*`; expands to names an `import` provides |
| `device_guard.h` | device-only `.cuh` (`parallel_for.cuh`, `math.cuh`) | `WWR_SELECTED_*` (via `selected_backend.h`), after `#error`ing outside a device pass |
| `complex.h`, `runtime.h`, `rand.h`, `atomic.h` | **anything** (`.cpp`, `.cppm`, `.cu`, `.cuh`) | `WWR_SELECTED_*` (via `selected_backend.h`), *without* `#error`ing outside a device pass |

These shared-type headers are the fourth kind: they name a type used on
both sides of the host/device boundary — a vendor type for three, the
project's own `wwrMemoryOrder` / `wwrThreadScope` enums for `atomic.h` — so
unlike the device `.cuh` they must
compile in a host TU too, which is why they pick the backend through
`selected_backend.h` directly rather than `device_guard.h`. See the next section
for what makes a header a `_bridge` (a *declaration* carrier) versus a
shared-type `.h` (a *type* carrier, like these four).

The device `.cuh` headers each ask two questions and keep them apart: "is
this a device pass?" (their own requirement — everything they define is
device-compile syntax) and "which backend?" (`selected_backend.h`'s). Both
answers arrive through one include, `device_guard.h` — `parallel_for.cuh` and
`math.cuh` include it directly — which pairs
`selected_backend.h` with the device-pass `#error`. (`cooperative_groups.h` and
`wmma.h` ask the same two questions but answer the first with a gate rather than
an `#error`, which is what lets them be `.h`; they reach the backend through
`runtime.h` from inside that gate, not through `device_guard.h`; see above.) The
two questions cannot be folded into one ladder, because `selected_backend.h`
answers the second from the CMake define in a host compile too, and would hand a
device header a backend it must still refuse to serve — so `device_guard.h`, not
`selected_backend.h`, is where the guard sits. It picks no backend of its own —
it is plumbing over `selected_backend.h`'s answer plus the `#error` — but it is
the header a pure-device `.cuh` reaches for to get that answer under the guard,
which is why it, and no longer the former `runtime.cuh`, is the device row
above.

### `_bridge`, shared-type `.h`, and why the extension is not enough

Every header in `src/` falls into one of five classes, and the file extension
does not fully separate them:

| Class | Compiles in a host TU | Compiles in a device pass | Named |
|---|:---:|:---:|---|
| host-only | ✅ | ❌ | `*.h` — `backend.h`, and `dispatch_macros.h` under `src/wrappers` |
| shared-type | ✅ | ✅ | `*.h` — `runtime.h`, `complex.h`, `rand.h`, `fp16.h`, `bf16.h`, `fp8.h`, `vector_types.h`, `atomic.h` |
| **bridge** | ✅ | ✅ | `*_bridge.h` |
| device-only | ❌ | ✅ | `*.cuh` — `parallel_for.cuh`, `math.cuh` |
| device-only, gated | ✅ (empty) | ✅ | `*.h` — `cooperative_groups.h`, `wmma.h` |

`runtime.h`, `rand.h`, `complex.h`, `fp16.h`, `bf16.h`, `fp8.h`,
`vector_types.h` and `atomic.h` are the shared-type row's hybrids: each carries
its crossing type(s) *and*, in a device-pass-gated section, the `__device__`
wrappers (and,
for `runtime.h`, the `WWR_GRID_CONSTANT` / `WWR_WARP_SIZE` macros and the full
runtime header) that used to live in the matching `.cuh`. Each still compiles in
both modes — the device section simply gates itself out in a host TU — so it
stays a `.h`, and a device consumer `#include`s it instead of a separate `.cuh`.
This is the `.h` + `.cppm` shape for a header carrying both host and
device symbols. (`atomic.h` is the newest fold and the one exception to "crossing
*vendor* type": its host half is the project's own enums, not a vendor type, and
its `.cppm` re-exports only those — the operations are the device half. `runtime`
was the last of the vendor-type candidates folded; its `.cppm`
is irregular — `runtime_api.cppm` binds the runtime surface through its own
`runtime_api.h` and does not include `runtime.h`, see its own note above.)

A wrapper with *no* host half can still be a `.h` — the device-only-gated row:
`cooperative_groups.h` and `wmma.h` put their *entire* body behind the device-pass
macros, so they compile to nothing in a host TU, host-*safe* without being
host-*usable*. What forces a `.cuh` instead is the hard `#error`:
`parallel_for.cuh` and `math.cuh` refuse a host compile outright rather than
gating to empty.

So `.h` on its own does *not* mean "safe from a `.cu`" (`backend.h` is `.h` and
host-only), and several `.h`s deliberately compile in both modes — the
shared-type `complex.h`, `runtime.h`, `rand.h`, `atomic.h`, and the device-only-gated
`cooperative_groups.h` / `wmma.h`. What separates the *shared-type* headers from a
`_bridge.h` is **what crosses**: a shared-type header names a *type* that
device code and a host TU (a module GMF, a plain `.cu`) both `#include` so they
agree with the type the module exports (`wwrFloatComplex` via `complex.h`,
`wwrStream_t` via `runtime.h`, `wwrrandState` via `rand.h`, and the
`wwrMemoryOrder` / `wwrThreadScope` enums — project-owned, not a vendor type —
via `atomic.h`); a `_bridge.h`
carries a *function declaration* across the boundary, so the host TU that
declares and the device TU that defines see one identical spelling and the symbol
mangles the same in both.

There are exactly two bridges, both under `src/extension`, consumed by
`src/extension/init_state` and `src/extension/random_normal`. Each carries a
device *function* declaration; the state *type* those functions take rides the
shared-type header `rand.h`, not a bridge — the distinction above, a bridge
carries a declaration and a shared-type `.h` carries a type:

| Bridge | Carries |
|---|---|
| `extension/init_state/init_state_bridge.h` | `device::init_state()` |
| `extension/random_normal/random_normal_bridge.h` | `device::random_normal()` |

`selected_backend.h` compiles in both modes too and is deliberately neither: it
declares nothing at all, so nothing of it crosses. It answers a configuration
question that is the same for everyone, and its two macros never appear in a
signature.

Both patterns exist because a module unit's **global module fragment** can
`#include` but cannot `import`, so a type or declaration it shares with a plain
(non-module) TU has to arrive by `#include`. That rules out `backend.h` (its
macros name imported entities) *and* `device_guard.h` with the device-only
`.cuh` that include it (a module interface unit is a host compile, so their
`#error` fires). `selected_backend.h` falling back to the CMake define only when
no device pass is in progress is what lets one header serve both, and is why
`wwr.device` still carries no define.

`wwrStream_t` moved from a bridge to the shared-type header `runtime.h` because
the device headers and the host GMF consumers all reach it by `#include` and its
vendor header is cheap. (`runtime_api.cppm` exports the same type, but reaches it through
its own vendor-include header `runtime_api.h`, not `runtime.h` -- `runtime_api.h`
`#undef`s the allocation-flag macros that would collide with its `WWR_RT_VALUE`
expansions, which `runtime.h` — binding no flag values — leaves defined.)
`wwrrandState`
followed into the shared-type header
`rand.h` -- but `rand.h` is the heavy exception: its vendor kernel headers
(`curand_kernel.h` / `hiprand_kernel.h`) are *not* cheap, so this is a deliberate
trade. The type once rode a *forward-declaring* bridge that kept those headers out
of every host compile; `rand.h` instead includes them, accepting the parse in the
three TUs that `#include` it -- `rand.cppm`'s own compile (which builds the
`wwr.rand` BMI) and the two wrapper module GMFs -- in exchange for one source of
truth for the state-type list, shared by `rand.cppm`, `rand.h`'s own device
generators and the two bridges. `rand.h` also includes the vendor *host* header
(`curand.h` /
`hiprand.h`, cheap beside the kernel one) so the same one-source-of-truth covers
the host API: `rand.cppm` binds its `wwr*` names straight to those declarations
and imports no raw vendor module (safe because the host API is external-linkage
-- see the `gpu.rand` section and docs/architecture.md §12). The BMI firewalls
the header from every `import wwr.rand` consumer, and two of those three TUs would
`#include` such a header in any design (a wrapper declaring the device boundary
names the state type), so the added cost is bounded and small. See `rand.h`'s own
header and docs/architecture.md §1.

`extension/init_state/init_state_bridge.h` and
`extension/random_normal/random_normal_bridge.h` are the consumers both were
written for; their header comments have the full reasoning.

## Tests

`test/gpu` has one compile-time module per module here. It checks that every
exported `wwr*` name is exactly the backend entity it stands for: the same
type, the same constant (type and value), the same function (plus
`WWR_LINK_CHECK`). The expected backend names are written out in full rather than
derived with `backend.h`'s macros, so a mistake in those macros shows up as
a failing test instead of being repeated in it. Add a line there whenever you
add a name here. (`wwr.atomic` is the one exception to "the backend entity it
stands for": it exports two project-owned enums with no vendor entity behind
them, so `test/gpu/atomic.cppm` pins only that the re-export carries both enums
intact; the scoped-atomic operations, being `__device__`-only, are compile-tested
separately by `test/gpu/atomic.cu`.)

Two claims are `__device__`-only and so cannot be a host-compiled module test;
each is tested instead by a device `.cu` built under the selected backend, where
building *is* the assertion (no ctest entry, no launch, a break reddening that
backend's compile-time tier):

- `test/gpu/cooperative_groups.cu` — that `cooperative_groups.h`'s include
  switch resolves the vendor `namespace cooperative_groups` on both backends.
- `test/gpu/atomics.cu` — that the portable common atomics (`atomicAdd`,
  `atomicCAS`, …) resolve for the common widths under both front ends. These
  ride the runtime header `runtime.h`'s device section switches and are spelled
  identically on both backends, so wwr wraps none of them; see
  `docs/architecture.md` section 15.
