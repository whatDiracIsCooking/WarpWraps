# src -- the backend-neutral wwr* layer, plus the wiring of the trees around it
#
# The wwr* modules live directly in this directory. Each re-exports one raw
# library module (src/cuda or src/hip) under wwr-prefixed names, picked at
# configure time by WWR_GPU_BACKEND. This is the only place in the project
# that names both backends; everything built on top (src/wrappers) is written
# once against wwr* names. See README.md.
#
# This file also owns the order the src/ trees are added: the chosen backend's
# raw 1:1 wrappers first (the wwr* modules below link them), then those modules,
# then src/wrappers (written once against the wwr* names).
if(WWR_GPU_BACKEND STREQUAL "CUDA")
  add_subdirectory(cuda)
else()
  add_subdirectory(hip)
endif()

# The backend define (WWR_GPU_BACKEND_*) and the src/-root include path for
# the wwr* layer's headers (selected_backend.h, backend.h, runtime.cuh,
# device_guard.h). Linked PRIVATE by every module below, and by test/gpu and the
# bridge-using modules (init_state, random_normal), which read
# selected_backend.h from a host compile with no __CUDACC__/__HIP__ to switch
# on. Linking it does NOT license a backend #if in the consumer -- the #if lives
# only in the wwr* layer's headers.
add_library(wwr_backend INTERFACE)
target_compile_definitions(
  wwr_backend INTERFACE WWR_GPU_BACKEND_${WWR_GPU_BACKEND}
)
target_include_directories(
  wwr_backend
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            $<INSTALL_INTERFACE:${WWR_INSTALL_INCLUDEDIR}>
)
# In-tree :: alias -- a raw add_library gets none automatically, and the
# dot-less name has no dot to convert, so it is spelled by hand. Dependants link
# the guarded wwr::backend spelling.
add_library(wwr::backend ALIAS wwr_backend)

# The include path, runtime libs and WWR_WARP_SIZE for the device-side
# headers (runtime.cuh, cooperative_groups.h, wmma.h, and the device-pass
# gated sections of complex.h / fp16.h / bf16.h / fp8.h)
# -- the counterpart to wwr_backend for code reached by #include, not import
# (a kernel TU imports no modules). Carries NO WWR_GPU_BACKEND define: these
# headers pick their backend from __CUDACC__/__HIP__, so this is safe to link
# from outside the wwr* layer (wwr.extension.parallel_for is the intended
# consumer). Never hip::device here -- see src/hip/README.md, "Link hip::host,
# never hip::device", and cmake/wwr_add_gpu_device_library.cmake for the
# per-kernel target that does.
#
# WWR_WARP_SIZE is a configure-time constant, not a backend switch.
# cooperative_groups.h and wmma.h need no target of their own:
# both vendors ship them header-only on the include path CUDA::cudart /
# hip::host already bring (docs/architecture.md §2, §16 -- and do NOT add
# find_package(rocwmma), which drags in hip::device; see §17). The common
# atomics need no header either -- runtime.cuh's vendor runtime declares them
# (§15, test/gpu/atomics.cu).
add_library(wwr.device INTERFACE)
target_include_directories(
  wwr.device
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            $<INSTALL_INTERFACE:${WWR_INSTALL_INCLUDEDIR}>
)
target_compile_definitions(
  wwr.device
  INTERFACE WWR_WARP_SIZE=${WWR_WARP_SIZE}
)
if(WWR_GPU_BACKEND STREQUAL "CUDA")
  # CUDA::cudart carries the toolkit's include path, which is where atomic.cuh's
  # <cuda/atomic> (libcu++) lives. CUDA 13.0 relocated that header from
  # include/cuda/atomic to include/cccl/cuda/atomic, but CMake 4.2's
  # FindCUDAToolkit adds the cccl dir to CUDA::cudart's interface for exactly
  # that reason (verified: CUDAToolkit_INCLUDE_DIRS and CUDA::cudart both list
  # .../include and .../include/cccl on 13.0.88), so no extra include dir is
  # needed here. See docs/architecture.md §20.
  target_link_libraries(wwr.device INTERFACE CUDA::cudart)
else()
  target_link_libraries(wwr.device INTERFACE hip::host)
endif()
# In-tree :: alias (raw add_library gets none automatically).
add_library(wwr::device ALIAS wwr.device)

# The include path and RNG library for a device TU calling the cuRAND/hipRAND
# device API through rand.h's device-pass-gated generators -- the #include
# counterpart to wwr.rand. Separate from wwr.device because the two are
# independent and a kernel needing no RNG should not link one. Carries no
# WWR_GPU_BACKEND define and never hip::device, for the reasons on wwr.device
# above.
add_library(wwr.rand.device INTERFACE)
target_include_directories(
  wwr.rand.device
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            $<INSTALL_INTERFACE:${WWR_INSTALL_INCLUDEDIR}>
)
if(WWR_GPU_BACKEND STREQUAL "CUDA")
  target_link_libraries(wwr.rand.device INTERFACE CUDA::curand)
else()
  target_link_libraries(wwr.rand.device INTERFACE hip::hiprand)
endif()
# In-tree :: alias (raw add_library gets none automatically).
add_library(wwr::rand::device ALIAS wwr.rand.device)

# Raw module each wwr* module wraps, per backend.
#
# _backend_rt_host is NOT a raw module: wwr.runtime_api / wwr.fp16 / wwr.bf16
# import no raw module. Their vendor API is external-linkage, so the .cppm binds
# its wwr* names directly to the declarations a src/-root sibling header pulls
# in (runtime_api.h / fp16.h / bf16.h -- the rand.h shape). They need only that
# header's include path plus the link library, both of which the backend runtime
# host lib carries (CUDA::cudart / hip::host, as wwr.device relies on) -- and it
# brings no WWR_WARP_SIZE, unlike wwr.device itself.
if(WWR_GPU_BACKEND STREQUAL "CUDA")
  set(_raw_complex wwr::cuda::cuComplex)
  set(_raw_fp8 wwr::cuda::cuda_fp8)
  set(_backend_rt_host CUDA::cudart)
  set(_raw_blas wwr::cuda::cublas_v2)
  set(_raw_blaslt wwr::cuda::cublasLt)
  set(_raw_solver wwr::cuda::cusolverDn)
  set(_raw_sparse wwr::cuda::cusparse)
  set(_raw_fft wwr::cuda::cufft)
  set(_raw_tx wwr::cuda::nvToolsExt)
  set(_raw_ccl wwr::cuda::nccl)
  set(_raw_tensor wwr::cuda::cutensor)
  set(_raw_comp wwr::cuda::nvcomp)
else()
  set(_raw_complex wwr::hip::hip_complex)
  set(_raw_fp8 wwr::hip::hip_fp8)
  set(_backend_rt_host hip::host)
  set(_raw_blas wwr::hip::hipblas)
  set(_raw_blaslt wwr::hip::hipblaslt)
  set(_raw_solver wwr::hip::hipsolver)
  set(_raw_sparse wwr::hip::hipsparse)
  set(_raw_fft wwr::hip::hipfft)
  set(_raw_tx wwr::hip::roctx)
  set(_raw_ccl wwr::hip::rccl)
  set(_raw_tensor wwr::hip::hiptensor)
  set(_raw_comp wwr::hip::hipcomp)
endif()

# wwr.runtime_api binds straight to the vendor runtime *library*, not a raw
# vendor module: its GMF #includes runtime_api.h, which pulls in
# cuda_runtime_api.h / hip_runtime_api.h, and the runtime host API is a real
# external-linkage library, so binding wwr* names to the declarations needs only
# the prototype plus the link library. _backend_rt_host (CUDA::cudart /
# hip::host) carries both, PUBLIC (a consumer calling wwr* runtime entry points
# links the runtime too). The raw module wwr.cuda.cuda_runtime_api /
# wwr.hip.hip_runtime_api stays as a single-vendor surface, off this path -- the
# rand.h shape. See src/runtime_api.h and src/README.md.
wwr_add_cxx_module_library(
  NAME
  wwr.runtime_api
  PRIMARY_INTERFACE
  runtime_api.cppm
  LINK_PUBLIC
  ${_backend_rt_host}
  LINK_PRIVATE
  wwr::backend
  IMPORT_STD
)

wwr_add_cxx_module_library(
  NAME
  wwr.complex
  PRIMARY_INTERFACE
  complex.cppm
  LINK_PUBLIC
  ${_raw_complex}
  LINK_PRIVATE
  wwr::backend
  IMPORT_STD
)

wwr_add_cxx_module_library(
  NAME
  wwr.fp16
  PRIMARY_INTERFACE
  fp16.cppm
  LINK_PUBLIC
  ${_backend_rt_host}
  LINK_PRIVATE
  wwr::backend
  IMPORT_STD
)

wwr_add_cxx_module_library(
  NAME
  wwr.bf16
  PRIMARY_INTERFACE
  bf16.cppm
  LINK_PUBLIC
  ${_backend_rt_host}
  LINK_PRIVATE
  wwr::backend
  IMPORT_STD
)

wwr_add_cxx_module_library(
  NAME
  wwr.fp8
  PRIMARY_INTERFACE
  fp8.cppm
  LINK_PUBLIC
  ${_raw_fp8}
  LINK_PRIVATE
  wwr::backend
  IMPORT_STD
)

# wwr.fp6 / wwr.fp4 -- BLOCKED on the HIP backend, not registered
#
# src/fp6.cppm and src/fp4.cppm exist as complete, reviewed source (same shape
# as fp8.cppm) but are NOT wired in, because their HIP raw dependencies
# wwr.hip.hip_fp6 / wwr.hip.hip_fp4 do not build with this toolchain:
# amd_detail/amd_hip_ocp_types.h #errors outside real `-x hip` device mode or a
# genuine GCC >= 13 front end. See src/hip/CMakeLists.txt's "HIP FP4 / FP6 --
# BLOCKED" comment and src/hip/README.md, "wwr.hip.hip_fp4 / wwr.hip.hip_fp6 --
# blocked, not built". A build targets one backend and CI compiles both plus the
# cross-backend check, so registering these would fail the HIP path. The CUDA
# raw modules (wwr.cuda.cuda_fp6 / cuda_fp4) do build, so these flip on the day
# the HIP raw modules do -- add the wwr_add_cxx_module_library() calls
# (LINK_PUBLIC wwr.cuda.cuda_fp6 / wwr.hip.hip_fp6, etc.) and the test/gpu
# entries then.

wwr_add_cxx_module_library(
  NAME
  wwr.blas
  PRIMARY_INTERFACE
  blas.cppm
  LINK_PUBLIC
  ${_raw_blas}
  wwr::complex
  LINK_PRIVATE
  wwr::backend
  IMPORT_STD
)

wwr_add_cxx_module_library(
  NAME
  wwr.blaslt
  PRIMARY_INTERFACE
  blaslt.cppm
  LINK_PUBLIC
  ${_raw_blaslt}
  LINK_PRIVATE
  wwr::backend
  IMPORT_STD
)

wwr_add_cxx_module_library(
  NAME
  wwr.solver
  PRIMARY_INTERFACE
  solver.cppm
  LINK_PUBLIC
  ${_raw_solver}
  wwr::complex
  wwr::blas
  LINK_PRIVATE
  wwr::backend
  IMPORT_STD
)

wwr_add_cxx_module_library(
  NAME
  wwr.sparse
  PRIMARY_INTERFACE
  sparse.cppm
  LINK_PUBLIC
  ${_raw_sparse}
  wwr::complex
  LINK_PRIVATE
  wwr::backend
  IMPORT_STD
)

wwr_add_cxx_module_library(
  NAME
  wwr.fft
  PRIMARY_INTERFACE
  fft.cppm
  LINK_PUBLIC
  ${_raw_fft}
  LINK_PRIVATE
  wwr::backend
  IMPORT_STD
)

# wwr.rand wraps the vendor RNG *library* directly, not a raw vendor module: its
# GMF #includes rand.h, which pulls in curand.h/curand_kernel.h (or the hipRAND
# pair), and the host API is a real external-linkage library, so binding wwr*
# references to the declarations needs only the prototype plus the link library.
# wwr::rand::device already carries exactly that pair -- the vendor lib and the
# src/-root include path -- so link it, PUBLIC (a consumer calling wwrrand*
# links the vendor lib too). The raw modules (wwr.cuda.curand, wwr.hip.hiprand,
# wwr.hip.hiprand_kernel) stay as single-vendor surfaces, off this path. See
# src/rand.h and src/README.md.
wwr_add_cxx_module_library(
  NAME
  wwr.rand
  PRIMARY_INTERFACE
  rand.cppm
  LINK_PUBLIC
  wwr::rand::device
  LINK_PRIVATE
  wwr::backend
  IMPORT_STD
)

wwr_add_cxx_module_library(
  NAME
  wwr.tx
  PRIMARY_INTERFACE
  tx.cppm
  LINK_PUBLIC
  ${_raw_tx}
  LINK_PRIVATE
  wwr::backend
  IMPORT_STD
)

wwr_add_cxx_module_library(
  NAME
  wwr.ccl
  PRIMARY_INTERFACE
  ccl.cppm
  LINK_PUBLIC
  ${_raw_ccl}
  LINK_PRIVATE
  wwr::backend
  IMPORT_STD
)

wwr_add_cxx_module_library(
  NAME
  wwr.tensor
  PRIMARY_INTERFACE
  tensor.cppm
  LINK_PUBLIC
  ${_raw_tensor}
  LINK_PRIVATE
  wwr::backend
  IMPORT_STD
)

wwr_add_cxx_module_library(
  NAME
  wwr.comp
  PRIMARY_INTERFACE
  comp.cppm
  LINK_PUBLIC
  ${_raw_comp}
  LINK_PRIVATE
  wwr::backend
  IMPORT_STD
)

# Installed consumers compile these module units themselves (a C++23 module
# package ships interface SOURCES, not BMIs), so every unit's compile
# requirements must be exported. wwr_backend is linked PRIVATE above, and
# PRIVATE requirements are not exported -- so its include path and define would
# be missing when an installed consumer compiles a .cppm, hitting
# backend.h's #error case. Re-add it under $<INSTALL_INTERFACE:> only: BUILD
# and INSTALL interfaces are exclusive, so the in-tree build is unchanged (and
# the no-backend-#if rule stays enforced where it matters). See cmake/README.md,
# "Install and the CMake package".
#
# Derived from BUILDSYSTEM_TARGETS so a module added above cannot be forgotten;
# the INTERFACE libraries here have no units to compile and are skipped.
get_property(_wwr_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
foreach(_target IN LISTS _wwr_targets)
  get_target_property(_type ${_target} TYPE)
  if(_type STREQUAL "STATIC_LIBRARY")
    target_link_libraries(
      ${_target} PUBLIC $<INSTALL_INTERFACE:wwr_backend>
    )
  endif()
endforeach()
unset(_wwr_targets)
unset(_target)
unset(_type)

# The wrappers, written once against the wwr* names above. Added last so the
# wwr.* module targets they link already exist.
add_subdirectory(wrappers)

# The extension layer -- higher-level RAII/error-handling abstractions built on
# the wwr* names (runtime, memory_buffer, parallel_for, rand, and the
# error/handle layer for blas/solver/fft/sparse). Added after wrappers; it too
# is written once against the wwr* module targets above.
add_subdirectory(extension)
