# ROCm/HIP API module wrappers (built when WWR_GPU_BACKEND is HIP)

# HIP Runtime API
#
# hip_runtime_api.h covers both driver and runtime (CUDA splits them into cuda.h
# vs cuda_runtime_api.h), so this one module covers what wwr.cuda.cuda_h and
# wwr.cuda.cuda_runtime_api cover together. Links hip::host, never
# hip::device -- see src/hip/README.md, "Link hip::host, never hip::device".
wwr_add_cxx_module_library(
  NAME
  wwr.hip.hip_runtime_api
  PRIMARY_INTERFACE
  hip_runtime_api.cppm
  LINK_PUBLIC
  hip::host
  IMPORT_STD
)

# ROCm SMI (System Management Interface)
#
# nvml's HIP counterpart is two independent modules: rocm_smi (this one) and
# amd_smi (below) -- see src/hip/README.md, "Why nvml became two modules". The
# imported target is the unnamespaced rocm_smi64 (from rocm_smiTargets.cmake);
# link that, not hip::host, since this library is independent of the HIP
# runtime.
wwr_add_cxx_module_library(
  NAME
  wwr.hip.rocm_smi
  PRIMARY_INTERFACE
  rocm_smi.cppm
  LINK_PUBLIC
  rocm_smi64
  IMPORT_STD
)

# AMD SMI (System Management Interface)
#
# The newer of the two nvml counterparts -- see the rocm_smi comment above. The
# imported target is the unnamespaced amd_smi (from amd_smiTargets.cmake); link
# that, not hip::host, since this library is also independent of the HIP
# runtime.
wwr_add_cxx_module_library(
  NAME
  wwr.hip.amd_smi
  PRIMARY_INTERFACE
  amd_smi.cppm
  LINK_PUBLIC
  amd_smi
  IMPORT_STD
)

# ROCtracer (ROC Profiling Tools Interface)
#
# HIP's rough counterpart to CUPTI (wwr.cuda.cupti). roctracer has no CMake
# package (no /opt/rocm/lib/cmake/roctracer/), so find_library it directly, as
# src/cuda/CMakeLists.txt does for cusolverMg. roctracer64 needs the HIP runtime
# for neither symbols nor its header's transitive includes, so this links no
# hip:: target -- but it still needs /opt/rocm/include on the include path for
# `#include <roctracer/roctracer.h>`, so find_path that directly too.
find_library(ROCTRACER_LIBRARY roctracer64 HINTS /opt/rocm/lib REQUIRED)
find_path(ROCTRACER_INCLUDE_DIR roctracer/roctracer.h HINTS /opt/rocm/include
                                                            REQUIRED
)
wwr_add_cxx_module_library(
  NAME
  wwr.hip.roctracer
  PRIMARY_INTERFACE
  roctracer.cppm
  LINK_PUBLIC
  ${ROCTRACER_LIBRARY}
  INCLUDE_DIRS_PRIVATE
  ${ROCTRACER_INCLUDE_DIR}
  IMPORT_STD
)

# rocTX (ROCm Tracing Extension) -- marker/range core
#
# HIP's counterpart to NVTX (wwr.cuda.nvToolsExt). Like roctracer above,
# rocTX has no CMake package, so find_library roctx64 and find_path its header
# directly. roctx.h pulls in only <stdint.h> -- neither the HIP runtime's
# symbols nor its headers -- so this links no hip:: target, only libroctx64.
find_library(ROCTX_LIBRARY roctx64 HINTS /opt/rocm/lib REQUIRED)
find_path(ROCTX_INCLUDE_DIR roctracer/roctx.h HINTS /opt/rocm/include REQUIRED)
wwr_add_cxx_module_library(
  NAME
  wwr.hip.roctx
  PRIMARY_INTERFACE
  roctx.cppm
  LINK_PUBLIC
  ${ROCTX_LIBRARY}
  INCLUDE_DIRS_PRIVATE
  ${ROCTX_INCLUDE_DIR}
  IMPORT_STD
)

# HIP Complex Numbers
#
# hip/hip_complex.h -> amd_detail/amd_hip_complex.h. All functions there are
# `static inline`, like CUDA's cuComplex.h -- thin forwarding functions, not
# `using` declarations. See src/hip/hip_complex.cppm.
wwr_add_cxx_module_library(
  NAME
  wwr.hip.hip_complex
  PRIMARY_INTERFACE
  hip_complex.cppm
  LINK_PUBLIC
  hip::host
  IMPORT_STD
)

# HIP Half-Precision Floating-Point
#
# hip/hip_fp16.h -> amd_detail/amd_hip_fp16.h. This project's plain C++23
# module compile takes amd_hip_fp16.h's __GNUC__-fallback branch (clang
# always predefines __GNUC__; __HIP__ is never defined without -x hip),
# which has no arithmetic/comparison operators on __half/__half2 at all --
# nothing to forward. See src/hip/hip_fp16.cppm's file header for the full
# story (discovered by compiling, not by reading the header in isolation).
wwr_add_cxx_module_library(
  NAME
  wwr.hip.hip_fp16
  PRIMARY_INTERFACE
  hip_fp16.cppm
  LINK_PUBLIC
  hip::host
  IMPORT_STD
)

# HIP BFloat16 Floating-Point
#
# hip/hip_bf16.h -> amd_detail/amd_hip_bf16.h. __hip_bfloat16/__hip_bfloat162's
# operators are `static inline` free functions (unlike hip_fp16's hidden
# friends) -- thin forwarding functions needed, as in cuda_bf16.cppm. See
# src/hip/hip_bf16.cppm.
wwr_add_cxx_module_library(
  NAME
  wwr.hip.hip_bf16
  PRIMARY_INTERFACE
  hip_bf16.cppm
  LINK_PUBLIC
  hip::host
  IMPORT_STD
)

# HIP FP8 Floating-Point
#
# hip/hip_fp8.h -> amd_detail/amd_hip_fp8.h. See src/hip/hip_fp8.cppm.
wwr_add_cxx_module_library(
  NAME
  wwr.hip.hip_fp8
  PRIMARY_INTERFACE
  hip_fp8.cppm
  LINK_PUBLIC
  hip::host
  IMPORT_STD
)

# HIP FP4 / FP6 Floating-Point -- BLOCKED, not registered as build targets
#
# src/hip/hip_fp4.cppm and hip_fp6.cppm exist as complete source but are NOT
# wired in: amd_detail/amd_hip_ocp_types.h #errors unless compiled through real
# `-x hip` device mode (which "Link hip::host, never hip::device" forbids) or a
# real GCC >= 13 front end -- a hard vendor-header requirement this toolchain
# cannot satisfy, not a CMake-flags problem. See src/hip/README.md, "hip_fp4 /
# hip_fp6: blocked, not built", for the full writeup and when to revisit.

# HIP Runtime Compilation (hiprtc)
#
# hip/hiprtc.h, declarations-only extern "C" (like CUDA's nvrtc.h). Needs its
# own library from ROCm's separate `hiprtc` find_package config (see the
# find_package(hiprtc CONFIG REQUIRED) call in the top-level CMakeLists.txt) --
# hiprtc::hiprtc is self-contained (the runtime-compilation API does not also
# need hip::host's libamdhip64), so it is linked alone, not alongside
# hip::host.
wwr_add_cxx_module_library(
  NAME
  wwr.hip.hiprtc
  PRIMARY_INTERFACE
  hiprtc.cppm
  LINK_PUBLIC
  hiprtc::hiprtc
  IMPORT_STD
)

# ------------------------------------------------------------------------
# Library-linked module batch (issue #178)
# ------------------------------------------------------------------------
# Modern imported CMake targets that self-contain their include dirs, so
# LINK_PUBLIC alone is sufficient -- no INCLUDE_CUDA_TOOLKIT-equivalent flag.

# hipBLAS
#
# CUDA counterpart: cublas_v2.cppm. Unlike cuBLAS, hipBLAS carries no
# "_v2"-suffixed name pair to redirect around -- every function is declared
# once, under its final name. See src/hip/hipblas.cppm.
wwr_add_cxx_module_library(
  NAME
  wwr.hip.hipblas
  PRIMARY_INTERFACE
  hipblas.cppm
  LINK_PUBLIC
  roc::hipblas
  IMPORT_STD
)

# hipBLASLt
#
# CUDA counterpart: cublasLt.cppm. hipBLASLt's surface is much narrower --
# 24 functions total, no logger/heuristics-cache/algo-introspection API.
# See src/hip/hipblaslt.cppm.
#
# Unlike every other library in this batch, roc::hipblaslt does NOT pull in
# hip::host transitively (its INTERFACE_LINK_LIBRARIES is just
# roc::hipblas-common), so without linking hip::host explicitly here,
# hipblaslt.h's transitive #includes fail their `__HIP_PLATFORM_AMD__` guard.
wwr_add_cxx_module_library(
  NAME
  wwr.hip.hipblaslt
  PRIMARY_INTERFACE
  hipblaslt.cppm
  LINK_PUBLIC
  roc::hipblaslt
  hip::host
  IMPORT_STD
)

# hipSOLVER
#
# CUDA counterparts: cusolverDn.cppm AND cusolverSp.cppm -- HIP has one
# hipsolver.h/hipSOLVER library covering both the dense (hipsolverDn*) and
# the narrow sparse (hipsolverSp*, only 11 functions -- issue #177) surface,
# unlike CUDA's two separate headers/modules, so this is one module. Neither
# hipsolver-functions.h's native (non-Dn/Sp) API nor hipsolver-refactor.h's
# hipsolverRf* API is wrapped -- this project has never wrapped cusolverRf
# either, so there is nothing on the CUDA side for a refactor module to
# mirror. See src/hip/hipsolver.cppm and src/hip/README.md.
wwr_add_cxx_module_library(
  NAME
  wwr.hip.hipsolver
  PRIMARY_INTERFACE
  hipsolver.cppm
  LINK_PUBLIC
  roc::hipsolver
  IMPORT_STD
)

# hipSPARSE
#
# CUDA counterpart: cusparse.cppm. See src/hip/hipsparse.cppm.
wwr_add_cxx_module_library(
  NAME
  wwr.hip.hipsparse
  PRIMARY_INTERFACE
  hipsparse.cppm
  LINK_PUBLIC
  roc::hipsparse
  IMPORT_STD
)

# hipFFT
#
# CUDA counterpart: cufft.cppm. See src/hip/hipfft.cppm.
#
# Note the imported target namespace is hip::, NOT roc:: -- unlike most of this
# batch.
wwr_add_cxx_module_library(
  NAME
  wwr.hip.hipfft
  PRIMARY_INTERFACE
  hipfft.cppm
  LINK_PUBLIC
  hip::hipfft
  IMPORT_STD
)

# hipFFT eXtended (hipfftXt)
#
# CUDA counterpart: cufftXt.cppm. hipfftXt.h is a genuinely separate header
# with its own extern "C" surface (multi-GPU descriptor alloc/copy/free,
# descriptor-based execution, extended plan creation, callback
# registration) -- mirrors why cuFFT keeps cufftXt.cppm separate from
# cufft.cppm, so this stays a separate module too, not folded into
# wwr.hip.hipfft. Same package as hipfft above (no second
# find_package call), same hip::hipfft imported target. See
# src/hip/hipfftXt.cppm.
wwr_add_cxx_module_library(
  NAME
  wwr.hip.hipfftXt
  PRIMARY_INTERFACE
  hipfftXt.cppm
  LINK_PUBLIC
  hip::hipfft
  IMPORT_STD
)

# hipRAND
#
# CUDA counterpart: the host-API half of curand.cppm. See src/hip/hiprand.cppm.
#
# Note the imported target namespace is hip::, NOT roc:: -- also unlike most of
# this batch.
wwr_add_cxx_module_library(
  NAME
  wwr.hip.hiprand
  PRIMARY_INTERFACE
  hiprand.cppm
  LINK_PUBLIC
  hip::hiprand
  IMPORT_STD
)

# hipRAND device generator state types
#
# CUDA counterpart: the "Device API Types" half of curand.cppm -- cuRAND wraps
# curand.h and curand_kernel.h in one module; hipRAND gets two, because
# hiprand_kernel.h drags in the whole rocRAND device-generator implementation
# and hiprand.h does not. Host-API-only consumers keep linking
# wwr.hip.hiprand alone and pay none of it.
#
# TYPES only -- hiprand_init/hiprand_normal/... are __device__-qualified and
# unreachable from a module unit; src/rand.cuh is how a device
# translation unit reaches those. See src/hip/hiprand_kernel.cppm.
#
# Same hip::hiprand imported target as above (one hipRAND package, two
# headers); it carries /opt/rocm/include and __HIP_PLATFORM_AMD__, which is
# all a types-only wrapper needs.
wwr_add_cxx_module_library(
  NAME
  wwr.hip.hiprand_kernel
  PRIMARY_INTERFACE
  hiprand_kernel.cppm
  LINK_PUBLIC
  hip::hiprand
  IMPORT_STD
)
