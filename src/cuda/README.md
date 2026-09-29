# src/cuda — Native CUDA API Module Wrappers

This directory contains C++23 module interface units that wrap the native CUDA
library headers (`cuda_runtime_api.h`, `cublas_v2.h`, `cusolverDn.h`,
`curand.h`, `cuComplex.h`, `cuda_fp16.h`, `cuda_bf16.h`) so they can be
consumed via `import` statements in module-based translation units.

Each `.cppm` file is a primary module interface unit built by
`wwr_add_cxx_module_library`. All exported symbols are placed in the
`wwr::cuda` namespace, except `wwr.cuda.cuda_h` (the CUDA Driver API),
which exports directly into the global namespace.

## Modules

### `wwr.cuda.cuda_runtime_api`

**Import:** `import wwr.cuda.cuda_runtime_api;`

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

### `wwr.cuda.cublas_v2`

**Import:** `import wwr.cuda.cublas_v2;`

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

### `wwr.cuda.cusolverDn`

**Import:** `import wwr.cuda.cusolverDn;`

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

### `wwr.cuda.curand`

**Import:** `import wwr.cuda.curand;`

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

### `wwr.cuda.cuComplex`

**Import:** `import wwr.cuda.cuComplex;`

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

### `wwr.cuda.cuda_fp16`

**Import:** `import wwr.cuda.cuda_fp16;`

Wraps `cuda_fp16.h`. Exports:

- **Types** — `__half`, `__half2`, `half`, `half2`, `__nv_half`, `__nv_half2`,
  `nv_half`, `nv_half2`, and the raw storage types `__half_raw`, `__half2_raw`,
  `__nv_half_raw`, `__nv_half2_raw`.
- Arithmetic and comparison operators are defined in the global namespace and
  reach exported types via Argument-Dependent Lookup (ADL).

### `wwr.cuda.cuda_bf16`

**Import:** `import wwr.cuda.cuda_bf16;`

Wraps `cuda_bf16.h`. Exports:

- **Types** — `__nv_bfloat16`, `__nv_bfloat162`, `nv_bfloat16`,
  `nv_bfloat162`, `__nv_bfloat16_raw`, `__nv_bfloat162_raw`.
- Operators are accessible via ADL from the global namespace, same as
  `cuda_fp16`.

BFloat16 uses 8 exponent bits (same range as `float`) and 7 mantissa bits,
making it well-suited for deep learning workloads.

### `wwr.cuda.nccl`

**Import:** `import wwr.cuda.nccl;`

Wraps `nccl.h` — NVIDIA's Collective Communications Library. HIP counterpart:
`wwr.hip.rccl`, AMD's source-compatible reimplementation, which spells the whole
surface with the identical `nccl*` / `NCCL_*` names. See `src/ccl.cppm` for the
backend-neutral `wwrccl*` layer. Exports:

- **Types** — `ncclComm_t`, `ncclUniqueId`, `ncclWindow_t`, `ncclResult_t`,
  `ncclConfig_t`, `ncclSimInfo_t`, the `ncclRedOp_t` / `ncclDataType_t` /
  `ncclScalarResidence_t` enums, and all their enumerators.
- **Host API** — version query, buffer alloc, communicator lifecycle
  (`ncclCommInitRank{,Config,Scalable}`, `ncclCommInitAll`,
  `ncclCommSplit/Shrink/Finalize/Destroy/Abort`), error checking, communicator
  info, buffer/window registration, custom reductions, the collectives
  (`ncclAllReduce`, `ncclBroadcast`, `ncclReduce`, `ncclAllGather`,
  `ncclReduceScatter`), P2P (`ncclSend`/`Recv`), and grouping.
- **Constexpr flag values** — the scalar `NCCL_*` macros (`NCCL_UNIQUE_ID_BYTES`,
  `NCCL_SPLIT_NOCOLOR`, the WIN/CTA/SHRINK flags, …), which a module cannot
  re-export as macros, mirroring `cufft`'s treatment of its direction flags.

NCCL is **not** part of the `cuda-toolkit` meta-package, and there is no
`CUDA::nccl` FindCUDAToolkit target, so `CMakeLists.txt` `find_library`s
`libnccl` and wraps it in an installable imported target (as it does for
`cusolverMg`); `docker/Dockerfile.cuda` installs `libnccl-dev`.

### `wwr.cuda.nvcomp`

**Import:** `import wwr.cuda.nvcomp;`

Wraps nvCOMP — NVIDIA's GPU lossless-compression library: the batched low-level
interface (LLIF) for LZ4, Snappy, Cascaded, Deflate, GZIP, Zstd, GDeflate,
Bitcomp and ANS, the CRC32 checksum API, and the shared status/type enums. HIP
counterpart: `wwr.hip.hipcomp`. There is deliberately **no** backend-neutral
`wwr.comp` layer: nvCOMP is 5.3 while hipCOMP is a hipify of nvCOMP 2.2, so the
batched signatures diverge (split compress/decompress opts, an extra
device-status parameter, Sync/Async temp-size queries) and a `wwr*` alias could
not present one portable signature — see issue #110. The C++ HLIF managers and
the version macros are not wrapped.

The vendor headers declare their default-option structs, per-algorithm
chunk-size limits, alignment requirements and CRC32 model presets as file-scope
`static const` (internal linkage), which a module cannot name in an `export`ed
declaration. They are re-declared in `wwr::cuda` (integral limits as `inline
constexpr`, struct values as `inline const`), the same forwarding workaround
`cuComplex` uses for `cuComplex.h`'s static-inline functions.

nvCOMP is not a FindCUDAToolkit component; it ships its own CMake config package
providing `nvcomp::nvcomp`, so `CMakeLists.txt` `find_package(nvcomp)`s it and
the module links that plus `CUDA::cudart`. `docker/install-cuda.sh` installs the
`nvcomp-cuda-*` apt package and republishes it under `/opt/nvidia/nvcomp` on
`CMAKE_PREFIX_PATH`.

## Build

Defined in `CMakeLists.txt` using the project-local
`wwr_add_cxx_module_library` CMake function. Each target links the
appropriate CUDA toolkit component (`CUDA::cudart`, `CUDA::cublas`,
`CUDA::cusolver`, `CUDA::curand`) and exposes CUDA toolkit include directories
publicly. The few libraries with no FindCUDAToolkit component (`cusolverMg`,
`nccl`) are `find_library`'d by hand instead.
