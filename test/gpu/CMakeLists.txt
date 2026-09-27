# gpu* layer compile-time tests (built for either WWR_GPU_BACKEND)
#
# One module per gpu* module, checking that every exported gpu* name is
# exactly the backend entity it stands for: same type, same constant, same
# function (plus WWR_LINK_CHECK). Same shape as test/cuda -- see that file's
# header comment for the WWR_LINK_CHECK/WHOLE_ARCHIVE rationale.
#
# Like the gpu* layer, unlike everything else above it, these tests name both
# backends, so they link gpumod_backend PRIVATE for the backend
# define. They spell each expected backend name out in full instead of using
# gpu_backend.h's macros, so a mistake there cannot be mirrored here.
#
# gpu_compile_tests below is the one thing that runs (ctest label
# compile_time). Expected exit code: 0

if(WWR_GPU_BACKEND STREQUAL "CUDA")
  set(_gpu_test_raw_runtime_api gpumod.cuda.cuda_runtime_api)
  set(_gpu_test_raw_complex gpumod.cuda.cuComplex)
  set(_gpu_test_raw_fp16 gpumod.cuda.cuda_fp16)
  set(_gpu_test_raw_bf16 gpumod.cuda.cuda_bf16)
  set(_gpu_test_raw_blas gpumod.cuda.cublas_v2)
  set(_gpu_test_raw_solver gpumod.cuda.cusolverDn)
  set(_gpu_test_raw_sparse gpumod.cuda.cusparse)
  set(_gpu_test_raw_fft gpumod.cuda.cufft)
  set(_gpu_test_raw_rand gpumod.cuda.curand)
  set(_gpu_test_raw_tx gpumod.cuda.nvToolsExt)
else()
  set(_gpu_test_raw_runtime_api gpumod.hip.hip_runtime_api)
  set(_gpu_test_raw_complex gpumod.hip.hip_complex)
  set(_gpu_test_raw_fp16 gpumod.hip.hip_fp16)
  set(_gpu_test_raw_bf16 gpumod.hip.hip_bf16)
  set(_gpu_test_raw_blas gpumod.hip.hipblas)
  set(_gpu_test_raw_solver gpumod.hip.hipsolver)
  set(_gpu_test_raw_sparse gpumod.hip.hipsparse)
  set(_gpu_test_raw_fft gpumod.hip.hipfft)
  # Two, matching src/CMakeLists.txt: hipRAND's device state types live in
  # their own module, where cuRAND keeps them in curand.cppm.
  set(_gpu_test_raw_rand gpumod.hip.hiprand gpumod.hip.hiprand_kernel)
  set(_gpu_test_raw_tx gpumod.hip.roctx)
endif()

set(GPU_TEST_LIBRARIES
    runtime_api
    complex
    fp16
    bf16
    blas
    solver
    sparse
    fft
    rand
    tx
)

set(_gpu_test_targets "")
foreach(_lib IN LISTS GPU_TEST_LIBRARIES)
  wwr_add_cxx_module_library(
    NAME
    gpumod.test.gpu.${_lib}
    IMPORT_STD
    NO_CUDA_DEVICE_LINKING
    PRIMARY_INTERFACE
    ${_lib}.cppm
    LINK_PUBLIC
    gpumod.${_lib}
    ${_gpu_test_raw_${_lib}}
    LINK_PRIVATE
    gpumod_backend
    INCLUDE_DIRS_PRIVATE
    ${CMAKE_SOURCE_DIR}
  )
  add_dependencies(compile_time_tests gpumod.test.gpu.${_lib})
  list(APPEND _gpu_test_targets gpumod.test.gpu.${_lib})
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
set_tests_properties(gpu_compile_tests PROPERTIES LABELS compile_time)

# cooperative_groups.cuh is the odd one out among the gpu* tests above: it
# exports no names, so there is nothing for a .cppm module test to compare, and
# every cooperative-groups entity is __device__-only, so a host-compiled module
# test structurally cannot reach it. The claim it CAN break -- the vendor header
# resolving on both backends through one include switch -- is a device-compile
# claim, so the test is a device TU. Building it is the assertion: no ctest
# entry, no launch (runtime validation is the warp-reduction example's, per
# issue #37); if the header fails on a backend, that backend's compile-time tier
# goes red. Its only include path is gpumod.device, which carries the
# header.
wwr_add_gpu_device_library(
  NAME
  gpumod.test.gpu.cooperative_groups.device
  SOURCES
  cooperative_groups.cu
  LINK_PRIVATE
  gpumod.device
)
add_dependencies(
  compile_time_tests gpumod.test.gpu.cooperative_groups.device
)

# complex.cuh, fp16.cuh and bf16.cuh are the same shape of test for the same
# reason: their __device__ functions over cuComplex/hipComplex and __half /
# bfloat16 can only be named by a device pass, and the claim each can break --
# every gpuC* / conversion resolving on both backends through one include
# switch, plus (for complex) the §3 aggregate vs class divergence the header is
# built around -- is a device-compile claim. Building them is the assertion; no
# ctest entry, no launch. Reached through gpumod.device, the same include path a
# real device consumer uses.
wwr_add_gpu_device_library(
  NAME
  gpumod.test.gpu.complex.device
  SOURCES
  complex.cu
  LINK_PRIVATE
  gpumod.device
)
add_dependencies(compile_time_tests gpumod.test.gpu.complex.device)

wwr_add_gpu_device_library(
  NAME
  gpumod.test.gpu.fp16.device
  SOURCES
  fp16.cu
  LINK_PRIVATE
  gpumod.device
)
add_dependencies(compile_time_tests gpumod.test.gpu.fp16.device)

wwr_add_gpu_device_library(
  NAME
  gpumod.test.gpu.bf16.device
  SOURCES
  bf16.cu
  LINK_PRIVATE
  gpumod.device
)
add_dependencies(compile_time_tests gpumod.test.gpu.bf16.device)

# wmma.cu is cooperative_groups.cu's shape with a weaker starting guarantee.
# wmma.cuh exports one name -- the wwr::gpuwmma alias -- so there is still
# nothing for a .cppm module test to compare, and every WMMA entity is
# __device__-only, so a host-compiled module test structurally cannot reach it.
# But where cooperative_groups.cuh switches between two paths to the same vendor
# API, this one switches between NVIDIA's <mma.h> and AMD's separately written
# rocWMMA, which merely agrees on the spellings -- so the claim that they stay
# interchangeable is worth compiling on both backends rather than assuming.
# Building it is the assertion: no ctest entry, no launch (WMMA is arch-gated
# and this tier configures with no device). Its include path is gpumod.device,
# which carries wmma.cuh along with the fp16.cuh/bf16.cuh it names element
# types from.
wwr_add_gpu_device_library(
  NAME
  gpumod.test.gpu.wmma.device
  SOURCES
  wmma.cu
  LINK_PRIVATE
  gpumod.device
)
add_dependencies(compile_time_tests gpumod.test.gpu.wmma.device)

# atomics.cu tests the portable common atomics, which gpumod does NOT wrap:
# atomicAdd/CAS/... are spelled identically on both backends and ride the vendor
# runtime header that runtime.cuh switches, so there is no atomics header
# to test. The claim it CAN break -- every operation resolving for the common
# widths under BOTH front ends, nvcc being the permissive one -- is a
# device-compile claim, so this is a device TU. Building it is the assertion; no
# ctest entry, no launch. Reached through gpumod.device, which carries
# runtime.cuh and the runtime header.
wwr_add_gpu_device_library(
  NAME
  gpumod.test.gpu.atomics.device
  SOURCES
  atomics.cu
  LINK_PRIVATE
  gpumod.device
)
add_dependencies(
  compile_time_tests gpumod.test.gpu.atomics.device
)

# Guard: every gpu* function alias in the gpu* layer is independently verified
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
  COMMENT "Checking gpu* alias coverage"
  VERBATIM
)
add_custom_target(
  gpu_alias_coverage ALL
  DEPENDS "${_alias_stamp}"
  COMMENT "gpu* alias coverage check"
)
add_dependencies(compile_time_tests gpu_alias_coverage)

# The one runtime test under test/gpu. The fp16 / bf16 host conversion wrappers
# (unlike every other gpu* name here) run on the CPU, so they can be exercised
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
    gpumod.test.gpu.conversions
    SOURCES
    conversions.cpp
    LINK
    gpumod.fp16
    gpumod.bf16
  )
  wwr_add_gtest_suite_tests(
    TARGET
    gpumod.test.gpu.conversions
    SUITES
    HalfConversion
    Bfloat16Conversion
  )
endif()
