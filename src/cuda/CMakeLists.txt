# Native CUDA API module wrappers

# CUDA Driver API
wwr_add_cxx_module_library(
  NAME
  wwr.cuda.cuda_h
  PRIMARY_INTERFACE
  cuda.cppm
  LINK_PUBLIC
  CUDA::cuda_driver
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)

# CUDA Runtime API
wwr_add_cxx_module_library(
  NAME
  wwr.cuda.cuda_runtime_api
  PRIMARY_INTERFACE
  cuda_runtime_api.cppm
  LINK_PUBLIC
  CUDA::cudart
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)

# cuComplex API
wwr_add_cxx_module_library(
  NAME
  wwr.cuda.cuComplex
  PRIMARY_INTERFACE
  cuComplex.cppm
  LINK_PUBLIC
  CUDA::cudart
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)

# CUDA FP16 API
wwr_add_cxx_module_library(
  NAME
  wwr.cuda.cuda_fp16
  PRIMARY_INTERFACE
  cuda_fp16.cppm
  LINK_PUBLIC
  CUDA::cudart
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)

# CUDA BF16 API
wwr_add_cxx_module_library(
  NAME
  wwr.cuda.cuda_bf16
  PRIMARY_INTERFACE
  cuda_bf16.cppm
  LINK_PUBLIC
  CUDA::cudart
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)

# CUDA FP4 API
wwr_add_cxx_module_library(
  NAME
  wwr.cuda.cuda_fp4
  PRIMARY_INTERFACE
  cuda_fp4.cppm
  LINK_PUBLIC
  CUDA::cudart
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)

# CUDA FP6 API
wwr_add_cxx_module_library(
  NAME
  wwr.cuda.cuda_fp6
  PRIMARY_INTERFACE
  cuda_fp6.cppm
  LINK_PUBLIC
  CUDA::cudart
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)

# CUDA FP8 API
wwr_add_cxx_module_library(
  NAME
  wwr.cuda.cuda_fp8
  PRIMARY_INTERFACE
  cuda_fp8.cppm
  LINK_PUBLIC
  CUDA::cudart
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)

# cuBLAS API
wwr_add_cxx_module_library(
  NAME
  wwr.cuda.cublas_v2
  PRIMARY_INTERFACE
  cublas_v2.cppm
  LINK_PUBLIC
  wwr_module_flags
  CUDA::cublas
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)

# cuBLASLt API
wwr_add_cxx_module_library(
  NAME
  wwr.cuda.cublasLt
  PRIMARY_INTERFACE
  cublasLt.cppm
  LINK_PUBLIC
  wwr_module_flags
  CUDA::cublasLt
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)

# cuBLASXt API
wwr_add_cxx_module_library(
  NAME
  wwr.cuda.cublasXt
  PRIMARY_INTERFACE
  cublasXt.cppm
  LINK_PUBLIC
  wwr_module_flags
  CUDA::cublas
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)

# cuFFT API
wwr_add_cxx_module_library(
  NAME
  wwr.cuda.cufft
  PRIMARY_INTERFACE
  cufft.cppm
  LINK_PUBLIC
  CUDA::cufft
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)

# cuRAND API
wwr_add_cxx_module_library(
  NAME
  wwr.cuda.curand
  PRIMARY_INTERFACE
  curand.cppm
  LINK_PUBLIC
  CUDA::curand
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)

# cuSOLVER Dense API
wwr_add_cxx_module_library(
  NAME
  wwr.cuda.cusolverDn
  PRIMARY_INTERFACE
  cusolverDn.cppm
  LINK_PUBLIC
  CUDA::cusolver
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)

# cuSPARSE API
wwr_add_cxx_module_library(
  NAME
  wwr.cuda.cusparse
  PRIMARY_INTERFACE
  cusparse.cppm
  LINK_PUBLIC
  CUDA::cusparse
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)

# CUDA Profiler API
wwr_add_cxx_module_library(
  NAME
  wwr.cuda.cuda_profiler_api
  PRIMARY_INTERFACE
  cuda_profiler_api.cppm
  LINK_PUBLIC
  CUDA::cudart
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)

# cuFFT Extended (multi-GPU) API
wwr_add_cxx_module_library(
  NAME
  wwr.cuda.cufftXt
  PRIMARY_INTERFACE
  cufftXt.cppm
  LINK_PUBLIC
  CUDA::cufft
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)

# cuSOLVER Multi-GPU API
# libcusolverMg is not a FindCUDAToolkit component (there is no CUDA::cusolverMg
# target), so find it by hand. Without it nothing defines the cusolverMg*
# symbols and any consumer that calls them fails to link -- which this module
# did until test/cuda's LINK_CHECKs were made real.
find_library(
  CUDA_cusolverMg_LIBRARY cusolverMg HINTS ${CUDAToolkit_LIBRARY_DIR} REQUIRED
)

# Linked through an imported target rather than as the bare path find_library
# returned, because this target is installed. A path in a PUBLIC link is copied
# verbatim into the export set, which would hardcode THIS machine's CUDA
# location into the installed package and break it anywhere else. A target's
# NAME is what gets exported instead, and wwrConfig.cmake re-runs the
# find_library to recreate it on the consumer's machine. GLOBAL so the name is
# visible outside this directory, as the export set requires.
if(NOT TARGET wwr::cuda_cusolverMg)
  add_library(wwr::cuda_cusolverMg UNKNOWN IMPORTED GLOBAL)
  set_target_properties(
    wwr::cuda_cusolverMg PROPERTIES IMPORTED_LOCATION
                                       "${CUDA_cusolverMg_LIBRARY}"
  )
endif()

wwr_add_cxx_module_library(
  NAME
  wwr.cuda.cusolverMg
  PRIMARY_INTERFACE
  cusolverMg.cppm
  LINK_PUBLIC
  CUDA::cusolver
  wwr::cuda_cusolverMg
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)

# cuSOLVER Sparse API
wwr_add_cxx_module_library(
  NAME
  wwr.cuda.cusolverSp
  PRIMARY_INTERFACE
  cusolverSp.cppm
  LINK_PUBLIC
  CUDA::cusolver
  CUDA::cusparse
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)

# NVTX (NVIDIA Tools Extension) -- marker/range core
#
# HIP counterpart: roctx.cppm. Header-only: nvtx3/nvToolsExt.h is static inline
# with no libnvToolsExt and no CUDA::nvToolsExt target, so this links no CUDA
# library -- INCLUDE_CUDA_TOOLKIT alone puts nvtx3/ on the include path (needed
# both here and when a consumer recompiles the installed module unit). See
# src/cuda/nvToolsExt.cppm.
wwr_add_cxx_module_library(
  NAME
  wwr.cuda.nvToolsExt
  PRIMARY_INTERFACE
  nvToolsExt.cppm
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)

# CUPTI (CUDA Profiling Tools Interface)
wwr_add_cxx_module_library(
  NAME
  wwr.cuda.cupti
  PRIMARY_INTERFACE
  cupti.cppm
  LINK_PUBLIC
  CUDA::cupti
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)

# nvFatbin (NVIDIA Fatbinary Creation)
wwr_add_cxx_module_library(
  NAME
  wwr.cuda.nvFatbin
  PRIMARY_INTERFACE
  nvFatbin.cppm
  LINK_PUBLIC
  CUDA::nvfatbin
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)

# cuFile (GPUDirect Storage)
wwr_add_cxx_module_library(
  NAME
  wwr.cuda.cufile
  PRIMARY_INTERFACE
  cufile.cppm
  LINK_PUBLIC
  CUDA::cuFile
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)

# nvJitLink (NVIDIA JIT Linking)
wwr_add_cxx_module_library(
  NAME
  wwr.cuda.nvJitLink
  PRIMARY_INTERFACE
  nvJitLink.cppm
  LINK_PUBLIC
  CUDA::nvJitLink
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)

# nvJPEG
wwr_add_cxx_module_library(
  NAME
  wwr.cuda.nvjpeg
  PRIMARY_INTERFACE
  nvjpeg.cppm
  LINK_PUBLIC
  CUDA::nvjpeg
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)

# nvCOMP (GPU lossless compression) -- HIP counterpart: wwr.hip.hipcomp.
#
# nvCOMP is not part of the base CUDA toolkit and is NOT a FindCUDAToolkit
# component; it ships its own CMake config package providing the nvcomp::nvcomp
# imported target (see docker/install-cuda.sh, which republishes it under a
# fixed /opt/nvidia/nvcomp prefix on CMAKE_PREFIX_PATH). CUDA::cudart is linked
# alongside for the cudaStream_t the batched async signatures take.
wwr_add_cxx_module_library(
  NAME
  wwr.cuda.nvcomp
  PRIMARY_INTERFACE
  nvcomp.cppm
  LINK_PUBLIC
  nvcomp::nvcomp
  CUDA::cudart
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)

# NVML (NVIDIA Management Library)
wwr_add_cxx_module_library(
  NAME
  wwr.cuda.nvml
  PRIMARY_INTERFACE
  nvml.cppm
  LINK_PUBLIC
  CUDA::nvml
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)

# NVRTC (NVIDIA Runtime Compilation)
wwr_add_cxx_module_library(
  NAME
  wwr.cuda.nvrtc
  PRIMARY_INTERFACE
  nvrtc.cppm
  LINK_PUBLIC
  CUDA::nvrtc
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)

# NCCL (Collective Communications) -- HIP counterpart: wwr.hip.rccl.
#
# NCCL is NOT a FindCUDAToolkit component (there is no CUDA::nccl target), so
# find libnccl by hand and wrap it in an installable imported target, exactly as
# cusolverMg above -- a bare find_library path in a PUBLIC link would hardcode
# this machine's location into the exported package. libnccl-dev provides both
# nccl.h (on the default include path; INCLUDE_CUDA_TOOLKIT adds the toolkit
# include nccl.h needs for <cuda_runtime.h>) and libnccl.so. It is not in the
# base CUDA toolkit -- Dockerfile.cuda installs it (see docker/Dockerfile.cuda).
find_library(CUDA_nccl_LIBRARY nccl HINTS ${CUDAToolkit_LIBRARY_DIR} REQUIRED)
if(NOT TARGET wwr::cuda_nccl)
  add_library(wwr::cuda_nccl UNKNOWN IMPORTED GLOBAL)
  set_target_properties(
    wwr::cuda_nccl PROPERTIES IMPORTED_LOCATION "${CUDA_nccl_LIBRARY}"
  )
endif()

wwr_add_cxx_module_library(
  NAME
  wwr.cuda.nccl
  PRIMARY_INTERFACE
  nccl.cppm
  LINK_PUBLIC
  CUDA::cudart
  wwr::cuda_nccl
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)
