# wwr* layer compile-time tests (built for either WWR_GPU_BACKEND)
#
# One module per wwr* module, checking that every exported wwr* name is
# exactly the backend entity it stands for: same type, same constant, same
# function (plus WWR_LINK_CHECK). Same shape as test/cuda -- see that file's
# header comment for the WWR_LINK_CHECK/WHOLE_ARCHIVE rationale.
#
# Like the wwr* layer, unlike everything else above it, these tests name both
# backends, so they link wwr_backend PRIVATE for the backend
# define. They spell each expected backend name out in full instead of using
# backend.h's macros, so a mistake there cannot be mirrored here.
#
# gpu_compile_tests below is the one thing that runs (ctest label
# compile_time). Expected exit code: 0

if(WWR_GPU_BACKEND STREQUAL "CUDA")
  set(_gpu_test_raw_runtime_api wwr::cuda::cuda_runtime_api)
  set(_gpu_test_raw_complex wwr::cuda::cuComplex)
  set(_gpu_test_raw_fp16 wwr::cuda::cuda_fp16)
  set(_gpu_test_raw_bf16 wwr::cuda::cuda_bf16)
  set(_gpu_test_raw_fp8 wwr::cuda::cuda_fp8)
  set(_gpu_test_raw_blas wwr::cuda::cublas_v2)
  set(_gpu_test_raw_blaslt wwr::cuda::cublasLt)
  set(_gpu_test_raw_solver wwr::cuda::cusolverDn)
  set(_gpu_test_raw_sparse wwr::cuda::cusparse)
  set(_gpu_test_raw_fft wwr::cuda::cufft)
  set(_gpu_test_raw_rand wwr::cuda::curand)
  set(_gpu_test_raw_tx wwr::cuda::nvToolsExt)
  set(_gpu_test_raw_ccl wwr::cuda::nccl)
  set(_gpu_test_raw_tensor wwr::cuda::cutensor)
  set(_gpu_test_raw_comp wwr::cuda::nvcomp)
else()
  set(_gpu_test_raw_runtime_api wwr::hip::hip_runtime_api)
  set(_gpu_test_raw_complex wwr::hip::hip_complex)
  set(_gpu_test_raw_fp16 wwr::hip::hip_fp16)
  set(_gpu_test_raw_bf16 wwr::hip::hip_bf16)
  set(_gpu_test_raw_fp8 wwr::hip::hip_fp8)
  set(_gpu_test_raw_blas wwr::hip::hipblas)
  set(_gpu_test_raw_blaslt wwr::hip::hipblaslt)
  set(_gpu_test_raw_solver wwr::hip::hipsolver)
  set(_gpu_test_raw_sparse wwr::hip::hipsparse)
  set(_gpu_test_raw_fft wwr::hip::hipfft)
  # Two, matching src/CMakeLists.txt: hipRAND's device state types live in
  # their own module, where cuRAND keeps them in curand.cppm.
  set(_gpu_test_raw_rand wwr::hip::hiprand wwr::hip::hiprand_kernel)
  set(_gpu_test_raw_tx wwr::hip::roctx)
  set(_gpu_test_raw_ccl wwr::hip::rccl)
  set(_gpu_test_raw_tensor wwr::hip::hiptensor)
  set(_gpu_test_raw_comp wwr::hip::hipcomp)
endif()

set(GPU_TEST_LIBRARIES
    runtime_api
    complex
    fp16
    bf16
    fp8
    # fp6, fp4 deliberately NOT listed -- their wwr* modules are BLOCKED on the
    # HIP backend (see src/CMakeLists.txt's "wwr.fp6 / wwr.fp4 -- BLOCKED"
    # comment and src/hip/README.md). test/gpu/fp6.cppm and test/gpu/fp4.cppm
    # exist as reviewed source and flip on with the modules.
    blas
    blaslt
    solver
    sparse
    fft
    rand
    tx
    ccl
    comp
)

set(_gpu_test_targets "")
foreach(_lib IN LISTS GPU_TEST_LIBRARIES)
  wwr_add_cxx_module_library(
    NAME
    wwr.test.gpu.${_lib}
    IMPORT_STD
    NO_CUDA_DEVICE_LINKING
    PRIMARY_INTERFACE
    ${_lib}.cppm
    LINK_PUBLIC
    wwr::${_lib}
    ${_gpu_test_raw_${_lib}}
    LINK_PRIVATE
    wwr::backend
    # PROJECT_SOURCE_DIR (not CMAKE_SOURCE_DIR) is gpumod's own root, so the
    # root-relative #include "test/shared/link_check.h" resolves even when a
    # consumer adds gpumod as a subdirectory -- there CMAKE_SOURCE_DIR is the
    # consumer's root, which has no test/shared.
    INCLUDE_DIRS_PRIVATE
    ${PROJECT_SOURCE_DIR}
  )
  add_dependencies(compile_time_tests wwr.test.gpu.${_lib})
  list(APPEND _gpu_test_targets wwr.test.gpu.${_lib})
endforeach()

wwr_add_test_executable(
  NAME
  gpu_compile_tests
  TIMEOUT
  30
  LINK_PRIVATE
  ${_gpu_test_targets}
  WHOLE_ARCHIVE
  ${_gpu_test_targets}
)

add_dependencies(compile_time_tests gpu_compile_tests)

# tensor is the one wwr* compile-time test kept OUT of the shared, RUN
# gpu_compile_tests binary above: on HIP, libhiptensor.so runs a device probe in
# a global constructor that aborts at process startup on any host without a
# supported AMD GPU -- "unsupported host device detected" on an unsupported card
# (gfx1200 here), "no ROCm-capable device is detected" with none, e.g. CI's
# GPU-less runners (hiptensor/library/src/hip_device.cpp). A runnable binary
# that links it therefore cannot reach main(). So tensor is verified exactly
# like the device-library tests below: the module's static_asserts are proved
# when it COMPILES, and its own executable is BUILT (WWR_LINK_CHECK forces every
# wwrtensor* symbol to resolve against libcutensor / libhiptensor -- the link IS
# the assertion) but carries NO ctest entry and is never launched. Uniform
# across backends: cuTENSOR does not abort this way, but one code path is
# simpler, and the shared runner already proves the run/print path elsewhere.
wwr_add_cxx_module_library(
  NAME
  wwr.test.gpu.tensor
  IMPORT_STD
  NO_CUDA_DEVICE_LINKING
  PRIMARY_INTERFACE
  tensor.cppm
  LINK_PUBLIC
  wwr::tensor
  ${_gpu_test_raw_tensor}
  LINK_PRIVATE
  wwr::backend
  INCLUDE_DIRS_PRIVATE
  ${PROJECT_SOURCE_DIR}
)
add_dependencies(compile_time_tests wwr.test.gpu.tensor)

wwr_add_test_executable(
  NAME
  gpu_compile_tests_tensor
  MAIN
  tensor_link_check.cpp
  LINK_PRIVATE
  wwr::test::gpu::tensor
  WHOLE_ARCHIVE
  wwr.test.gpu.tensor
  NO_RUN
)
add_dependencies(compile_time_tests gpu_compile_tests_tensor)
set_tests_properties(gpu_compile_tests PROPERTIES LABELS compile_time)

# cooperative_groups.cuh is the odd one out among the wwr* tests above: it
# exports no names, so there is nothing for a .cppm module test to compare, and
# every cooperative-groups entity is __device__-only, so a host-compiled module
# test structurally cannot reach it. The claim it CAN break -- the vendor header
# resolving on both backends through one include switch -- is a device-compile
# claim, so the test is a device TU. Building it is the assertion: no ctest
# entry, no launch (runtime validation is the warp-reduction example's, per
# issue #37); if the header fails on a backend, that backend's compile-time tier
# goes red. Its only include path is wwr.device, which carries the
# header.
wwr_add_gpu_device_library(
  NAME
  wwr.test.gpu.cooperative_groups.device
  SOURCES
  cooperative_groups.cu
  LINK_PRIVATE
  wwr::device
)
add_dependencies(
  compile_time_tests wwr.test.gpu.cooperative_groups.device
)

# complex.cuh, fp16.cuh and bf16.cuh are the same shape of test for the same
# reason: their __device__ functions over cuComplex/hipComplex and __half /
# bfloat16 can only be named by a device pass, and the claim each can break --
# every wwrC* / conversion resolving on both backends through one include
# switch, plus (for complex) the §3 aggregate vs class divergence the header is
# built around -- is a device-compile claim. Building them is the assertion; no
# ctest entry, no launch. Reached through wwr.device, the same include path a
# real device consumer uses.
wwr_add_gpu_device_library(
  NAME
  wwr.test.gpu.complex.device
  SOURCES
  complex.cu
  LINK_PRIVATE
  wwr::device
)
add_dependencies(compile_time_tests wwr.test.gpu.complex.device)

wwr_add_gpu_device_library(
  NAME
  wwr.test.gpu.fp16.device
  SOURCES
  fp16.cu
  LINK_PRIVATE
  wwr::device
)
add_dependencies(compile_time_tests wwr.test.gpu.fp16.device)

wwr_add_gpu_device_library(
  NAME
  wwr.test.gpu.bf16.device
  SOURCES
  bf16.cu
  LINK_PRIVATE
  wwr::device
)
add_dependencies(compile_time_tests wwr.test.gpu.bf16.device)

# fp8.cuh is fp16.cuh / bf16.cuh's shape: its __device__ narrowing forwarders
# and the fp8 scalar/storage types they name can only be reached by a device
# pass, and the claim it can break -- every wwrFp8* name and narrowing resolving
# on both backends through one include switch, and (on HIP) <hip/hip_fp8.h>
# compiling cleanly in a real -x hip pass -- is a device-compile claim. Building
# it is the assertion; no ctest entry, no launch. Reached through wwr.device,
# the same include path a real device consumer uses.
wwr_add_gpu_device_library(
  NAME
  wwr.test.gpu.fp8.device
  SOURCES
  fp8.cu
  LINK_PRIVATE
  wwr::device
)
add_dependencies(compile_time_tests wwr.test.gpu.fp8.device)

# wmma.cu is cooperative_groups.cu's shape with a weaker starting guarantee.
# wmma.cuh exports one name -- the wwr::wwrwmma alias -- so there is still
# nothing for a .cppm module test to compare, and every WMMA entity is
# __device__-only, so a host-compiled module test structurally cannot reach it.
# But where cooperative_groups.cuh switches between two paths to the same vendor
# API, this one switches between NVIDIA's <mma.h> and AMD's separately written
# rocWMMA, which merely agrees on the spellings -- so the claim that they stay
# interchangeable is worth compiling on both backends rather than assuming.
# Building it is the assertion: no ctest entry, no launch (WMMA is arch-gated
# and this tier configures with no device). Its include path is wwr.device,
# which carries wmma.cuh along with the fp16.cuh/bf16.cuh it names element
# types from.
wwr_add_gpu_device_library(
  NAME
  wwr.test.gpu.wmma.device
  SOURCES
  wmma.cu
  LINK_PRIVATE
  wwr::device
)
add_dependencies(compile_time_tests wwr.test.gpu.wmma.device)

# atomics.cu tests the portable common atomics, which wwr does NOT wrap:
# atomicAdd/CAS/... are spelled identically on both backends and ride the vendor
# runtime header that runtime.cuh switches, so there is no atomics header
# to test. The claim it CAN break -- every operation resolving for the common
# widths under BOTH front ends, nvcc being the permissive one -- is a
# device-compile claim, so this is a device TU. Building it is the assertion; no
# ctest entry, no launch. Reached through wwr.device, which carries
# runtime.cuh and the runtime header.
wwr_add_gpu_device_library(
  NAME
  wwr.test.gpu.atomics.device
  SOURCES
  atomics.cu
  LINK_PRIVATE
  wwr::device
)
add_dependencies(
  compile_time_tests wwr.test.gpu.atomics.device
)

# atomic.cu (singular) tests src/atomic.cuh, the scoped/ordered atomics wwr
# DOES wrap -- the surface above atomics.cu's bare common atomics, where the two
# backends diverge (cuda::atomic_ref vs a __hip_atomic_* builtin). The claim it
# CAN break -- every forwarder x every portable scope resolving under BOTH front
# ends -- is a device-compile claim, so this is a device TU. Building it is the
# assertion; no ctest entry, no launch (ordering semantics need a card, not a
# compile). Reached through wwr.device, which carries atomic.cuh and, on CUDA
# 13+, the cccl include dir <cuda/atomic> needs.
wwr_add_gpu_device_library(
  NAME
  wwr.test.gpu.atomic.device
  SOURCES
  atomic.cu
  LINK_PRIVATE
  wwr::device
)
add_dependencies(
  compile_time_tests wwr.test.gpu.atomic.device
)

# math.cuh is the same shape of test as fp16.cuh / atomics.cu: its wrappers are
# __device__ templates over vendor math intrinsics (::expf/::exp, ...) that only
# a device pass can name, and the claim they can break -- every function
# instantiating for float and double, through one include switch, on both
# backends -- is a device-compile claim. Building it is the assertion; no ctest
# entry, no launch. Reached through wwr.device, which carries the header along
# with the runtime header its intrinsics ride.
wwr_add_gpu_device_library(
  NAME
  wwr.test.gpu.math.device
  SOURCES
  math.cu
  LINK_PRIVATE
  wwr::device
)
add_dependencies(compile_time_tests wwr.test.gpu.math.device)

# Guard: every wwr* function alias in the wwr* layer is independently verified
# by a WWR_SAME_FUNCTION restatement in the matching test/gpu module, or by a
# dispatch-table entry that test/shared/dispatch.py checks against the objects.
# The check is source-only (no objects), so it is a build-time custom target
# rather than a static_assert, added to compile_time_tests like the dispatch
# checks. test/shared/test_alias_coverage.py tests the guard itself.
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(_alias_script "${PROJECT_SOURCE_DIR}/test/shared/alias_coverage.py")
set(_alias_stamp "${CMAKE_CURRENT_BINARY_DIR}/gpu_alias_coverage.stamp")
file(
  GLOB _alias_inputs CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/src/*.cppm"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cppm"
  "${PROJECT_SOURCE_DIR}/test/wrappers/*/*_dispatch.toml"
)
add_custom_command(
  OUTPUT "${_alias_stamp}"
  COMMAND
    Python3::Interpreter "${_alias_script}" --root "${PROJECT_SOURCE_DIR}"
    --stamp "${_alias_stamp}"
  DEPENDS "${_alias_script}" ${_alias_inputs}
  COMMENT "Checking wwr* alias coverage"
  VERBATIM
)
add_custom_target(
  gpu_alias_coverage ALL
  DEPENDS "${_alias_stamp}"
  COMMENT "wwr* alias coverage check"
)
add_dependencies(compile_time_tests gpu_alias_coverage)

# The one runtime test under test/gpu. The fp16 / bf16 host conversion wrappers
# (unlike every other wwr* name here) run on the CPU, so they can be exercised
# for real with no device -- the .cppm tests above prove they link, this proves
# they compute. GoogleTest is not fetched under WWR_COMPILE_TIME_ONLY, so
# this target is skipped there, exactly like test/cuda and test/hip one level
# up.
#
# The ONLY runtime suite in the tree without REQUIRES_GPU, and the reason that
# flag is worth having: `ctest -LE gpu` on a GitHub-hosted runner still runs
# these two, because host conversion is host code. Every other runtime target
# (test/extension/*) allocates device memory, launches a kernel or creates a
# vendor handle, and is labeled accordingly.
if(NOT WWR_COMPILE_TIME_ONLY)
  wwr_add_gtest_executable(
    NAME
    wwr.test.gpu.conversions
    SOURCES
    conversions.cpp
    LINK
    wwr::fp16
    wwr::bf16
  )
  wwr_add_gtest_suite_tests(
    TARGET
    wwr.test.gpu.conversions
    SUITES
    HalfConversion
    Bfloat16Conversion
  )
endif()
