# src -- the backend-neutral gpu* layer, plus the wiring of the trees around it
#
# The gpu* modules live directly in this directory. Each re-exports one raw
# library module (src/cuda or src/hip) under gpu-prefixed names, picked at
# configure time by GPUMOD_GPU_BACKEND. This is the only place in the project
# that names both backends; everything built on top (src/wrappers) is written
# once against gpu* names. See README.md.
#
# This file also owns the order the src/ trees are added: the chosen backend's
# raw 1:1 wrappers first (the gpu* modules below link them), then those modules,
# then src/wrappers (written once against the gpu* names).
if(GPUMOD_GPU_BACKEND STREQUAL "CUDA")
  add_subdirectory(cuda)
else()
  add_subdirectory(hip)
endif()

# The backend define (GPUMOD_GPU_BACKEND_*) and the src/-root include path for
# the gpu* layer's headers (selected_backend.h, gpu_backend.h, runtime.cuh,
# device_guard.h). Linked PRIVATE by every module below, and by test/gpu and the
# bridge-using modules (init_state, random_normal), which read selected_backend.h
# from a host compile with no __CUDACC__/__HIP__ to switch on. Linking it does NOT
# license a backend #if in the consumer -- the #if lives only in the gpu* layer's
# headers.
add_library(gpumod_backend INTERFACE)
target_compile_definitions(
  gpumod_backend INTERFACE GPUMOD_GPU_BACKEND_${GPUMOD_GPU_BACKEND}
)
target_include_directories(
  gpumod_backend
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            $<INSTALL_INTERFACE:${GPUMOD_INSTALL_INCLUDEDIR}>
)

# The include path, runtime libs and GPUMOD_WARP_SIZE for the device-side headers
# (runtime.cuh, complex.cuh, fp16.cuh, cooperative_groups.cuh, wmma.cuh) -- the
# counterpart to gpumod_backend for code reached by #include, not import (a kernel
# TU imports no modules). Carries NO GPUMOD_GPU_BACKEND define: these headers pick
# their backend from __CUDACC__/__HIP__, so this is safe to link from outside the
# gpu* layer (gpumod.extension.parallel_for is the intended consumer). Never
# hip::device here -- see src/hip/README.md, "Link hip::host, never hip::device",
# and cmake/gpumod_add_gpu_device_library.cmake for the per-kernel target that does.
#
# GPUMOD_WARP_SIZE is a configure-time constant, not a backend switch.
# cooperative_groups.cuh and wmma.cuh need no target of their own:
# both vendors ship them header-only on the include path CUDA::cudart / hip::host
# already bring (docs/architecture.md §2, §16 -- and do NOT add
# find_package(rocwmma), which drags in hip::device; see §17). The common atomics
# need no header either -- runtime.cuh's vendor runtime declares them (§15,
# test/gpu/atomics.cu).
add_library(gpumod.device INTERFACE)
target_include_directories(
  gpumod.device
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            $<INSTALL_INTERFACE:${GPUMOD_INSTALL_INCLUDEDIR}>
)
target_compile_definitions(
  gpumod.device
  INTERFACE GPUMOD_WARP_SIZE=${GPUMOD_WARP_SIZE}
)
if(GPUMOD_GPU_BACKEND STREQUAL "CUDA")
  target_link_libraries(gpumod.device INTERFACE CUDA::cudart)
else()
  target_link_libraries(gpumod.device INTERFACE hip::host)
endif()

# rand.cuh's include path and RNG library, for a device TU calling the
# cuRAND/hipRAND device API -- the #include counterpart to gpumod.rand. Separate
# from gpumod.device because the two headers are independent and a kernel needing
# no RNG should not link one. Carries no GPUMOD_GPU_BACKEND define and never
# hip::device, for the reasons on gpumod.device above.
add_library(gpumod.rand.device INTERFACE)
target_include_directories(
  gpumod.rand.device
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            $<INSTALL_INTERFACE:${GPUMOD_INSTALL_INCLUDEDIR}>
)
if(GPUMOD_GPU_BACKEND STREQUAL "CUDA")
  target_link_libraries(gpumod.rand.device INTERFACE CUDA::curand)
else()
  target_link_libraries(gpumod.rand.device INTERFACE hip::hiprand)
endif()

# Raw module each gpu module wraps, per backend.
if(GPUMOD_GPU_BACKEND STREQUAL "CUDA")
  set(_gpu_raw_runtime_api gpumod.cuda.cuda_runtime_api)
  set(_gpu_raw_complex gpumod.cuda.cuComplex)
  set(_gpu_raw_fp16 gpumod.cuda.cuda_fp16)
  set(_gpu_raw_bf16 gpumod.cuda.cuda_bf16)
  set(_gpu_raw_blas gpumod.cuda.cublas_v2)
  set(_gpu_raw_solver gpumod.cuda.cusolverDn)
  set(_gpu_raw_sparse gpumod.cuda.cusparse)
  set(_gpu_raw_fft gpumod.cuda.cufft)
  set(_gpu_raw_rand gpumod.cuda.curand)
else()
  set(_gpu_raw_runtime_api gpumod.hip.hip_runtime_api)
  set(_gpu_raw_complex gpumod.hip.hip_complex)
  set(_gpu_raw_fp16 gpumod.hip.hip_fp16)
  set(_gpu_raw_bf16 gpumod.hip.hip_bf16)
  set(_gpu_raw_blas gpumod.hip.hipblas)
  set(_gpu_raw_solver gpumod.hip.hipsolver)
  set(_gpu_raw_sparse gpumod.hip.hipsparse)
  set(_gpu_raw_fft gpumod.hip.hipfft)
  # Two modules where CUDA has one: cuRAND wraps curand.h and curand_kernel.h
  # together, hipRAND keeps hiprand_kernel.h separate. rand.cppm needs both
  # (host API + device generator state types) -- see src/hip/CMakeLists.txt.
  set(_gpu_raw_rand gpumod.hip.hiprand gpumod.hip.hiprand_kernel)
endif()

gpumod_add_cxx_module_library(
  NAME
  gpumod.runtime_api
  PRIMARY_INTERFACE
  runtime_api.cppm
  LINK_PUBLIC
  ${_gpu_raw_runtime_api}
  LINK_PRIVATE
  gpumod_backend
  IMPORT_STD
)

gpumod_add_cxx_module_library(
  NAME
  gpumod.complex
  PRIMARY_INTERFACE
  complex.cppm
  LINK_PUBLIC
  ${_gpu_raw_complex}
  LINK_PRIVATE
  gpumod_backend
  IMPORT_STD
)

gpumod_add_cxx_module_library(
  NAME
  gpumod.fp16
  PRIMARY_INTERFACE
  fp16.cppm
  LINK_PUBLIC
  ${_gpu_raw_fp16}
  LINK_PRIVATE
  gpumod_backend
  IMPORT_STD
)

gpumod_add_cxx_module_library(
  NAME
  gpumod.bf16
  PRIMARY_INTERFACE
  bf16.cppm
  LINK_PUBLIC
  ${_gpu_raw_bf16}
  LINK_PRIVATE
  gpumod_backend
  IMPORT_STD
)

gpumod_add_cxx_module_library(
  NAME
  gpumod.blas
  PRIMARY_INTERFACE
  blas.cppm
  LINK_PUBLIC
  ${_gpu_raw_blas}
  gpumod.complex
  LINK_PRIVATE
  gpumod_backend
  IMPORT_STD
)

gpumod_add_cxx_module_library(
  NAME
  gpumod.solver
  PRIMARY_INTERFACE
  solver.cppm
  LINK_PUBLIC
  ${_gpu_raw_solver}
  gpumod.complex
  gpumod.blas
  LINK_PRIVATE
  gpumod_backend
  IMPORT_STD
)

gpumod_add_cxx_module_library(
  NAME
  gpumod.sparse
  PRIMARY_INTERFACE
  sparse.cppm
  LINK_PUBLIC
  ${_gpu_raw_sparse}
  gpumod.complex
  LINK_PRIVATE
  gpumod_backend
  IMPORT_STD
)

gpumod_add_cxx_module_library(
  NAME
  gpumod.fft
  PRIMARY_INTERFACE
  fft.cppm
  LINK_PUBLIC
  ${_gpu_raw_fft}
  LINK_PRIVATE
  gpumod_backend
  IMPORT_STD
)

gpumod_add_cxx_module_library(
  NAME
  gpumod.rand
  PRIMARY_INTERFACE
  rand.cppm
  LINK_PUBLIC
  ${_gpu_raw_rand}
  LINK_PRIVATE
  gpumod_backend
  IMPORT_STD
)

# Installed consumers compile these module units themselves (a C++23 module
# package ships interface SOURCES, not BMIs), so every unit's compile requirements
# must be exported. gpumod_backend is linked PRIVATE above, and PRIVATE
# requirements are not exported -- so its include path and define would be missing
# when an installed consumer compiles a .cppm, hitting gpu_backend.h's #error case.
# Re-add it under $<INSTALL_INTERFACE:> only: BUILD and INSTALL interfaces are
# exclusive, so the in-tree build is unchanged (and the no-backend-#if rule stays
# enforced where it matters). See cmake/README.md, "Install and the CMake package".
#
# Derived from BUILDSYSTEM_TARGETS so a module added above cannot be forgotten;
# the INTERFACE libraries here have no units to compile and are skipped.
get_property(_gpu_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
foreach(_target IN LISTS _gpu_targets)
  get_target_property(_type ${_target} TYPE)
  if(_type STREQUAL "STATIC_LIBRARY")
    target_link_libraries(
      ${_target} PUBLIC $<INSTALL_INTERFACE:gpumod_backend>
    )
  endif()
endforeach()
unset(_gpu_targets)
unset(_target)
unset(_type)

# The wrappers, written once against the gpu* names above. Added last so the
# gpumod.* module targets they link already exist.
add_subdirectory(wrappers)

# The extension layer -- higher-level RAII/error-handling abstractions built on
# the gpu* names (runtime, memory_buffer, parallel_for, rand, and the
# error/handle layer for blas/solver/fft/sparse). Added after wrappers; it too
# is written once against the gpu* module targets above.
add_subdirectory(extension)
