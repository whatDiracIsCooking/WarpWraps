# Native CUDA API module wrappers

# CUDA Driver API
gpumod_add_cxx_module_library(
  NAME
  gpumod.cuda.cuda_h
  PRIMARY_INTERFACE
  cuda.cppm
  LINK_PUBLIC
  CUDA::cuda_driver
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)

# CUDA Runtime API
gpumod_add_cxx_module_library(
  NAME
  gpumod.cuda.cuda_runtime_api
  PRIMARY_INTERFACE
  cuda_runtime_api.cppm
  LINK_PUBLIC
  CUDA::cudart
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)

# cuComplex API
gpumod_add_cxx_module_library(
  NAME
  gpumod.cuda.cuComplex
  PRIMARY_INTERFACE
  cuComplex.cppm
  LINK_PUBLIC
  CUDA::cudart
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)

# CUDA FP16 API
gpumod_add_cxx_module_library(
  NAME
  gpumod.cuda.cuda_fp16
  PRIMARY_INTERFACE
  cuda_fp16.cppm
  LINK_PUBLIC
  CUDA::cudart
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)

# CUDA BF16 API
gpumod_add_cxx_module_library(
  NAME
  gpumod.cuda.cuda_bf16
  PRIMARY_INTERFACE
  cuda_bf16.cppm
  LINK_PUBLIC
  CUDA::cudart
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)

# CUDA FP4 API
gpumod_add_cxx_module_library(
  NAME
  gpumod.cuda.cuda_fp4
  PRIMARY_INTERFACE
  cuda_fp4.cppm
  LINK_PUBLIC
  CUDA::cudart
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)

# CUDA FP6 API
gpumod_add_cxx_module_library(
  NAME
  gpumod.cuda.cuda_fp6
  PRIMARY_INTERFACE
  cuda_fp6.cppm
  LINK_PUBLIC
  CUDA::cudart
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)

# CUDA FP8 API
gpumod_add_cxx_module_library(
  NAME
  gpumod.cuda.cuda_fp8
  PRIMARY_INTERFACE
  cuda_fp8.cppm
  LINK_PUBLIC
  CUDA::cudart
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)

# cuBLAS API
gpumod_add_cxx_module_library(
  NAME
  gpumod.cuda.cublas_v2
  PRIMARY_INTERFACE
  cublas_v2.cppm
  LINK_PUBLIC
  gpumod_module_flags
  CUDA::cublas
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)

# cuBLASLt API
gpumod_add_cxx_module_library(
  NAME
  gpumod.cuda.cublasLt
  PRIMARY_INTERFACE
  cublasLt.cppm
  LINK_PUBLIC
  gpumod_module_flags
  CUDA::cublasLt
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)

# cuBLASXt API
gpumod_add_cxx_module_library(
  NAME
  gpumod.cuda.cublasXt
  PRIMARY_INTERFACE
  cublasXt.cppm
  LINK_PUBLIC
  gpumod_module_flags
  CUDA::cublas
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)

# cuFFT API
gpumod_add_cxx_module_library(
  NAME
  gpumod.cuda.cufft
  PRIMARY_INTERFACE
  cufft.cppm
  LINK_PUBLIC
  CUDA::cufft
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)

# cuRAND API
gpumod_add_cxx_module_library(
  NAME
  gpumod.cuda.curand
  PRIMARY_INTERFACE
  curand.cppm
  LINK_PUBLIC
  CUDA::curand
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)

# cuSOLVER Dense API
gpumod_add_cxx_module_library(
  NAME
  gpumod.cuda.cusolverDn
  PRIMARY_INTERFACE
  cusolverDn.cppm
  LINK_PUBLIC
  CUDA::cusolver
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)

# cuSPARSE API
gpumod_add_cxx_module_library(
  NAME
  gpumod.cuda.cusparse
  PRIMARY_INTERFACE
  cusparse.cppm
  LINK_PUBLIC
  CUDA::cusparse
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)

# CUDA Profiler API
gpumod_add_cxx_module_library(
  NAME
  gpumod.cuda.cuda_profiler_api
  PRIMARY_INTERFACE
  cuda_profiler_api.cppm
  LINK_PUBLIC
  CUDA::cudart
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)

# cuFFT Extended (multi-GPU) API
gpumod_add_cxx_module_library(
  NAME
  gpumod.cuda.cufftXt
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
# NAME is what gets exported instead, and gpumodConfig.cmake re-runs the
# find_library to recreate it on the consumer's machine. GLOBAL so the name is
# visible outside this directory, as the export set requires.
if(NOT TARGET gpumod::cuda_cusolverMg)
  add_library(gpumod::cuda_cusolverMg UNKNOWN IMPORTED GLOBAL)
  set_target_properties(
    gpumod::cuda_cusolverMg PROPERTIES IMPORTED_LOCATION
                                       "${CUDA_cusolverMg_LIBRARY}"
  )
endif()

gpumod_add_cxx_module_library(
  NAME
  gpumod.cuda.cusolverMg
  PRIMARY_INTERFACE
  cusolverMg.cppm
  LINK_PUBLIC
  CUDA::cusolver
  gpumod::cuda_cusolverMg
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)

# cuSOLVER Sparse API
gpumod_add_cxx_module_library(
  NAME
  gpumod.cuda.cusolverSp
  PRIMARY_INTERFACE
  cusolverSp.cppm
  LINK_PUBLIC
  CUDA::cusolver
  CUDA::cusparse
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)

# CUPTI (CUDA Profiling Tools Interface)
gpumod_add_cxx_module_library(
  NAME
  gpumod.cuda.cupti
  PRIMARY_INTERFACE
  cupti.cppm
  LINK_PUBLIC
  CUDA::cupti
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)

# nvFatbin (NVIDIA Fatbinary Creation)
gpumod_add_cxx_module_library(
  NAME
  gpumod.cuda.nvFatbin
  PRIMARY_INTERFACE
  nvFatbin.cppm
  LINK_PUBLIC
  CUDA::nvfatbin
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)

# cuFile (GPUDirect Storage)
gpumod_add_cxx_module_library(
  NAME
  gpumod.cuda.cufile
  PRIMARY_INTERFACE
  cufile.cppm
  LINK_PUBLIC
  CUDA::cuFile
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)

# nvJitLink (NVIDIA JIT Linking)
gpumod_add_cxx_module_library(
  NAME
  gpumod.cuda.nvJitLink
  PRIMARY_INTERFACE
  nvJitLink.cppm
  LINK_PUBLIC
  CUDA::nvJitLink
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)

# nvJPEG
gpumod_add_cxx_module_library(
  NAME
  gpumod.cuda.nvjpeg
  PRIMARY_INTERFACE
  nvjpeg.cppm
  LINK_PUBLIC
  CUDA::nvjpeg
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)

# NVML (NVIDIA Management Library)
gpumod_add_cxx_module_library(
  NAME
  gpumod.cuda.nvml
  PRIMARY_INTERFACE
  nvml.cppm
  LINK_PUBLIC
  CUDA::nvml
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)

# NVRTC (NVIDIA Runtime Compilation)
gpumod_add_cxx_module_library(
  NAME
  gpumod.cuda.nvrtc
  PRIMARY_INTERFACE
  nvrtc.cppm
  LINK_PUBLIC
  CUDA::nvrtc
  IMPORT_STD
  INCLUDE_CUDA_TOOLKIT
  PUBLIC
)
