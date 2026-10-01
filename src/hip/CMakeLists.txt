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

# AMD SMI (System Management Interface)
#
# nvml's sole HIP counterpart: ROCm's legacy librocm_smi64 is deliberately NOT
# wrapped, because libamd_smi re-exports its symbols and a process holding both
# double-frees at teardown. src/hip/README.md, "nvml's HIP counterpart", has
# that plus which SMI library a wwr.ccl process can hold on which ROCm version.
# The imported target is the unnamespaced amd_smi (from amd_smiTargets.cmake);
# link that, not hip::host, since this library is independent of the HIP
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
# unreachable from a module unit; src/rand.h's device-pass-gated section is how
# a device translation unit reaches those. See src/hip/hiprand_kernel.cppm.
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

# RCCL (ROCm Communication Collectives Library)
#
# CUDA counterpart: nccl.cppm. RCCL is a source-compatible reimplementation of
# NCCL, so the two wrap the identical nccl* surface -- see src/ccl.cppm for the
# backend-neutral layer. roc::rccl (from find_package(rccl)) carries
# /opt/rocm/include, so `#include <rccl/rccl.h>` resolves; hip::host is linked
# alongside because rccl.h pulls in <hip/hip_runtime.h> and <hip/hip_fp16.h> and
# roc::rccl does not guarantee to bring the HIP runtime's include path itself.
# Gated on WWR_WITH_CCL (the find_package(rccl) in the top-level CMakeLists.txt
# is gated on the same flag), so a HIP build with the wrapper off needs no RCCL.
if(WWR_WITH_CCL)
  wwr_add_cxx_module_library(
    NAME
    wwr.hip.rccl
    PRIMARY_INTERFACE
    rccl.cppm
    LINK_PUBLIC
    roc::rccl
    hip::host
    IMPORT_STD
  )
endif()

# hipCOMP (GPU lossless compression)
#
# CUDA counterpart: nvcomp.cppm. hipCOMP is a hipify of nvCOMP branch-2.2 and is
# NOT part of ROCm proper -- it is built from source into /opt/rocm-ds (see
# docker/install-rocm-ds.sh), which ships the hipcomp::hipcomp imported target
# find_package(hipcomp) resolves off CMAKE_PREFIX_PATH. hip::host is linked
# alongside because hipcomp.h pulls in <hip/hip_runtime.h> for hipStream_t.
# Gated on WWR_WITH_COMP (the top-level find_package(hipcomp) is gated on the
# same flag), so a build with the wrapper off needs no hipCOMP.
if(WWR_WITH_COMP)
  wwr_add_cxx_module_library(
    NAME
    wwr.hip.hipcomp
    PRIMARY_INTERFACE
    hipcomp.cppm
    LINK_PUBLIC
    hipcomp::hipcomp
    hip::host
    IMPORT_STD
  )
endif()

# hipTensor (tensor contraction / reduction / permutation / element-wise)
#
# CUDA counterpart: cutensor.cppm. Unlike RCCL/NCCL, hipTensor is an
# independent implementation, not a source-compatible reimplementation -- see
# src/tensor.cppm for the measured intersection the wwrtensor* layer carries.
#
# hipTensor ships a CMake package, but its hiptensor::hiptensor target lists
# `hip::device` in INTERFACE_LINK_LIBRARIES -- and hip::device carries `-x hip`
# / --offload-arch as INTERFACE_COMPILE_OPTIONS. Linked PUBLIC, that would force
# DEVICE compilation onto every downstream HOST consumer of wwr.tensor, flipping
# e.g. gpu_compile_tests' main.cpp (which does `import std;`) into an
# amdgcn TU that cannot load the host `std` BMI (docs/architecture.md, section 8
# -- import std is not available to a device TU). rccl's roc::rccl does not do
# this; only hiptensor::hiptensor does. So -- exactly as cuda/CMakeLists.txt
# wraps the config-less cuTENSOR, and as nccl/cusolverMg wrap a bare library --
# find libhiptensor.so and its header by hand and link hip::host, so this module
# and its consumers compile as HOST like every other wwr.hip.* module. The
# `#include <array>` in hiptensor.cppm keeps the host compile of the HIP headers
# working (section 9). The header resolves via hip::host's /opt/rocm/include,
# but the wrapper carries the include dir explicitly too, so it is
# self-sufficient.
#
# TWO SPELLINGS, because hipTensor gained a second public header between the
# floor and the pin: 2.1.0 (ROCm 7.1) ships only hiptensor.hpp, and 2.2.0 (ROCm
# 7.2) added the C-linkage hiptensor.h beside it. NAMES tries both so the
# include dir resolves at either end; hiptensor.cppm picks the same way with
# __has_include. The re-exported surface is identical -- all 170 names this
# module exports are declared at 7.1 too -- so only the filename moves.
# Gated on WWR_WITH_TENSOR: OFF drops this module, the neutral wwr.tensor, and
# wwr.extension.tensor, so the REQUIRED finds below never run and a build on a
# box without hipTensor still configures.
if(WWR_WITH_TENSOR)
  find_path(
    HIP_hiptensor_INCLUDE_DIR
    NAMES hiptensor/hiptensor.h hiptensor/hiptensor.hpp
    HINTS ${ROCM_PATH} $ENV{ROCM_PATH} /opt/rocm
    PATH_SUFFIXES include REQUIRED
  )
  find_library(
    HIP_hiptensor_LIBRARY hiptensor
    HINTS ${ROCM_PATH} $ENV{ROCM_PATH} /opt/rocm
    PATH_SUFFIXES lib REQUIRED
  )
  if(NOT TARGET wwr::hip_hiptensor)
    add_library(wwr::hip_hiptensor UNKNOWN IMPORTED GLOBAL)
    set_target_properties(
      wwr::hip_hiptensor
      PROPERTIES IMPORTED_LOCATION "${HIP_hiptensor_LIBRARY}"
                 INTERFACE_INCLUDE_DIRECTORIES "${HIP_hiptensor_INCLUDE_DIR}"
    )
  endif()

  wwr_add_cxx_module_library(
    NAME
    wwr.hip.hiptensor
    PRIMARY_INTERFACE
    hiptensor.cppm
    LINK_PUBLIC
    wwr::hip_hiptensor
    hip::host
    IMPORT_STD
  )
endif()
