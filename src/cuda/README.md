# src/cuda — Native CUDA API Module Wrappers

This directory contains C++23 module interface units that wrap the native CUDA
library headers (`cuda_runtime_api.h`, `cublas_v2.h`, `cusolverDn.h`,
`curand.h`, `cuComplex.h`, `cuda_fp16.h`, `cuda_bf16.h`) so they can be
consumed via `import` statements in module-based translation units.

Each `.cppm` file is a primary module interface unit built by
`gpumod_add_cxx_module_library`. All exported symbols are placed in the
`gpumod::cuda` namespace, except `gpumod.cuda.cuda_h` (the CUDA Driver API),
which exports directly into the global namespace.

## Modules

### `gpumod.cuda.cuda_runtime_api`

**Import:** `import gpumod.cuda.cuda_runtime_api;`

Wraps `cuda_runtime_api.h` (not `cuda_runtime.h`, which contains static inline
wrappers that cannot be exported from modules). Exports:

- **Flag constants** — `cudaHostAllocDefault/Portable/Mapped/WriteCombined`,
  `cudaEventDefault/BlockingSync/DisableTiming/Interprocess`,
  `cudaStreamDefault/NonBlocking`, `cudaMemAttachGlobal/Host`. These macros are
  replaced by typed `constexpr unsigned int` variables after compile-time
  `static_assert` validation of their values.
- **Core types** — `cudaError_t`, `cudaEvent_t`, `cudaStream_t`,
  `cudaDeviceProp`, `cudaMemcpyKind`, `cudaMemPool_t`, graph types
  (`cudaGraph_t`, `cudaGraphNode_t`, `cudaGraphExec_t`, …), texture and surface
  objects, IPC handles, external memory handles, and all associated structs and
  enumerations.
- **Functions** — the full CUDA runtime API surface: error handling, device
  management, memory allocation/copy/set (including async, 2D, 3D, pitched,
  managed, and pool variants), stream and event management, kernel launch,
  occupancy helpers, CUDA graph construction and execution, texture/surface
  management, graphics interop, and driver entry-point queries.

### `gpumod.cuda.cublas_v2`

**Import:** `import gpumod.cuda.cublas_v2;`

Wraps `cublas_v2.h`. Provides the complete cuBLAS public API:

- **Types and enumerations** — `cublasHandle_t`, `cublasStatus_t`,
  `cublasOperation_t`, `cublasFillMode_t`, `cublasDiagType_t`,
  `cublasSideMode_t`, `cublasPointerMode_t`, `cublasAtomicsMode_t`,
  `cublasGemmAlgo_t`, `cublasMath_t`, `cublasComputeType_t`,
  `cublasEmulationStrategy_t`, all associated enum values.
- **Handle management** — `cublasCreate`, `cublasDestroy`, `cublasGetVersion`
  (thin wrappers over the `_v2`-suffixed functions, which are also exported
  directly), stream, pointer mode, atomics mode, math mode, workspace, SM count,
  emulation strategy, and logging functions.
- **Level 1/2/3 BLAS** — all S/D/C/Z typed functions with both 32-bit and
  64-bit integer variants (the `_v2` and `_v2_64` suffixed symbols).
- **Extensions** — `geam`, `dgmm`, batched factorizations (`getrfBatched`,
  `getrsBatched`, `getriBatched`, `matinvBatched`, `geqrfBatched`,
  `gelsBatched`), and triangular format conversions (`tpttr`, `trttp`).

Note: macros that redirect non-`_v2` names to `_v2` functions are `#undef`'d
before the module interface; the module exports both the `_v2` function names
and clean convenience wrappers without the suffix.

### `gpumod.cuda.cusolverDn`

**Import:** `import gpumod.cuda.cusolverDn;`

Wraps `cusolverDn.h`. Exports the cuSOLVER Dense API in full:

- **Types** — `cusolverDnHandle_t`, `cusolverStatus_t`, parameter structs
  (`syevjInfo_t`, `gesvdjInfo_t`, `cusolverDnIRSParams_t`,
  `cusolverDnIRSInfos_t`, `cusolverDnParams_t`), and all enumerations
  (`cusolverEigMode_t`, `cusolverEigType_t`, `cusolverEigRange_t`,
  `cusolverIRSRefinement_t`, `cusolverPrecType_t`, etc.).
- **Handle and mode management** — create/destroy, stream assignment,
  deterministic mode, math mode, emulation strategy, advanced options, logging.
- **Generic (X-prefix) API** — `cusolverDnXgeqrf`, `cusolverDnXgetrf`,
  `cusolverDnXpotrf`, `cusolverDnXsyevd`, `cusolverDnXsyevdx`,
  `cusolverDnXsyevBatched`, `cusolverDnXgesvd`, `cusolverDnXgesvdp`,
  `cusolverDnXgesvdr`, `cusolverDnXgetrs`, `cusolverDnXpotrs`,
  `cusolverDnXlarft`, `cusolverDnXsytrs`, `cusolverDnXtrtri`,
  `cusolverDnXgeev`, plus associated `_bufferSize` variants.
- **IRS (iterative refinement) solvers** — `cusolverDnIRSXgesv`,
  `cusolverDnIRSXgels`, parameter and info management.
- **Mixed-precision GESV/GELS** — all SS/SH/SB/SX/DD/DS/DH/DB/DX/CC/CK/CE/CY/
  ZZ/ZC/ZK/ZE/ZY variants for both `gesv` and `gels`.
- **Legacy typed API** — GETRF, GETRS, POTRF, POTRS, POTRI, LAUUM, SYTRF,
  SYTRI, GEQRF, ORGQR/UNGQR, ORMQR/UNMQR, GEBRD, ORGBR/UNGBR, SYTRD/HETRD,
  ORGTR/UNGTR, ORMTR/UNMTR, GESVD, GESVDJ (including batched), GESVDA strided
  batched, SYEVD/HEEVD, SYEVDX/HEEVDX, SYEVJ/HEEVJ (including batched),
  SYGVD/HEGVD, SYGVDX/HEGVDX, SYGVJ/HEGVJ, LASWP — all S/D/C/Z variants.

### `gpumod.cuda.curand`

**Import:** `import gpumod.cuda.curand;`

Wraps `curand.h` and `curand_kernel.h`. Exports:

- **Types and status** — `curandGenerator_t`, `curandStatus_t`, all status
  values.
- **Enumerations** — `curandRngType_t` (XORWOW, MRG32K3A, MTGP32, MT19937,
  Philox4-32-10, Sobol32/64 and scrambled variants),
  `curandOrdering_t`, `curandDirectionVectorSet_t`, `curandMethod_t`.
- **Host API** — generator create/destroy, seed, offset, ordering, dimension
  configuration; generation functions for uniform integers, uniform floats
  (single and double), normal, log-normal, Poisson, and binomial distributions;
  discrete distribution management; direction vector and scramble constant
  queries.
- **Device state types** — `curandStateXORWOW_t`, `curandStateMRG32k3a_t`,
  `curandStateMtgp32_t`, `curandStatePhilox4_32_10_t`, Sobol32/64 and
  scrambled variants, and the default `curandState_t` alias.

### `gpumod.cuda.cuComplex`

**Import:** `import gpumod.cuda.cuComplex;`

Wraps `cuComplex.h`. Because all functions in that header have `static inline`
linkage, they cannot be directly re-exported from a module; this wrapper
provides thin forwarding functions:

- **Types** — `cuFloatComplex`, `cuDoubleComplex`, `cuComplex`.
- **Constructors** — `make_cuFloatComplex`, `make_cuDoubleComplex`,
  `make_cuComplex`.
- **Component access** — `cuCrealf`, `cuCimagf`, `cuCreal`, `cuCimag`.
- **Arithmetic (single precision)** — `cuCaddf`, `cuCsubf`, `cuCmulf`,
  `cuCdivf`, `cuCabsf`, `cuConjf`.
- **Arithmetic (double precision)** — `cuCadd`, `cuCsub`, `cuCmul`, `cuCdiv`,
  `cuCabs`, `cuConj`.
- **Type conversion** — `cuComplexDoubleToFloat`, `cuComplexFloatToDouble`.

### `gpumod.cuda.cuda_fp16`

**Import:** `import gpumod.cuda.cuda_fp16;`

Wraps `cuda_fp16.h`. Exports:

- **Types** — `__half`, `__half2`, `half`, `half2`, `__nv_half`, `__nv_half2`,
  `nv_half`, `nv_half2`, and the raw storage types `__half_raw`, `__half2_raw`,
  `__nv_half_raw`, `__nv_half2_raw`.
- Arithmetic and comparison operators are defined in the global namespace and
  reach exported types via Argument-Dependent Lookup (ADL).

### `gpumod.cuda.cuda_bf16`

**Import:** `import gpumod.cuda.cuda_bf16;`

Wraps `cuda_bf16.h`. Exports:

- **Types** — `__nv_bfloat16`, `__nv_bfloat162`, `nv_bfloat16`,
  `nv_bfloat162`, `__nv_bfloat16_raw`, `__nv_bfloat162_raw`.
- Operators are accessible via ADL from the global namespace, same as
  `cuda_fp16`.

BFloat16 uses 8 exponent bits (same range as `float`) and 7 mantissa bits,
making it well-suited for deep learning workloads.

## Build

Defined in `CMakeLists.txt` using the project-local
`gpumod_add_cxx_module_library` CMake function. Each target links the
appropriate CUDA toolkit component (`CUDA::cudart`, `CUDA::cublas`,
`CUDA::cusolver`, `CUDA::curand`) and exposes CUDA toolkit include directories
publicly.
