# CUDA compile-time tests
# Verify compile-time flags, type traits, and link symbols for CUDA libraries.
#
# Compile-time only: a failing static_assert fails the build of that module, so
# building these targets IS running the tests. This directory is added even
# under GPUMOD_COMPILE_TIME_ONLY; build just it with
#   cmake --build build --target compile_time_tests
#
# cuda_compile_tests below is the one thing that runs (ctest label
# compile_time). It has no checks of its own -- reaching main() means every
# module compiled and the executable linked.
# Expected exit code: 0
#
# Each CUDA library gets one gpumod.test.cuda.<name> module wrapping its
# compile-time test (<name>.cppm) in an identical shape -- a foreach builds them
# all rather than repeating a 10-line gpumod_add_cxx_module_library() block for
# each. cuda.cppm is the one irregular name: its module is cuda_h (matching
# the wrapped module gpumod.cuda.cuda_h -- "cuda" on its own collides
# with this project's own top-level namespace).

set(CUDA_TEST_LIBRARIES
    cuda_runtime_api
    cuComplex
    cublasLt
    cublasXt
    cufft
    cufftXt
    cusparse
    cuda_fp4
    cuda_fp6
    cuda_fp8
    cuda_fp16
    cuda_bf16
    cuda_h
    cusolverMg
    cupti
    cuda_profiler_api
    cusolverSp
    nvFatbin
    cufile
    nvJitLink
    nvjpeg
    nvml
    nvrtc
)

foreach(_lib IN LISTS CUDA_TEST_LIBRARIES)
  if(_lib STREQUAL "cuda_h")
    set(_source "cuda.cppm")
  else()
    set(_source "${_lib}.cppm")
  endif()

  gpumod_add_cxx_module_library(
    NAME
    gpumod.test.cuda.${_lib}
    IMPORT_STD
    NO_CUDA_DEVICE_LINKING
    PRIMARY_INTERFACE
    ${_source}
    LINK_PUBLIC
    gpumod.cuda.${_lib}
    INCLUDE_DIRS_PRIVATE
    ${CMAKE_SOURCE_DIR}
  )
  add_dependencies(compile_time_tests gpumod.test.cuda.${_lib})
endforeach()

# Create test executable with main.cpp entry point.
#
# Every test library is linked WHOLE_ARCHIVE. They are static archives and
# main() calls nothing in them, so without it the linker pulls no member at all
# and the LINK_CHECKs (test/shared/link_check.h) never reach it. The list is
# derived from CUDA_TEST_LIBRARIES so a new library cannot be forgotten.
#
# Note: Link both libc++ (for the C++ standard library) and libstdc++ (for
# CUDA/thrust); libstdc++ must be linked explicitly after all CUDA libraries
set(_cuda_test_targets "")
foreach(_lib IN LISTS CUDA_TEST_LIBRARIES)
  list(APPEND _cuda_test_targets gpumod.test.cuda.${_lib})
endforeach()

gpumod_add_test_executable(
  NAME
  cuda_compile_tests
  TIMEOUT
  30
  LINK_PRIVATE
  ${_cuda_test_targets}
  WHOLE_ARCHIVE
  ${_cuda_test_targets}
  LINK_LIBSTDCXX
)

add_dependencies(compile_time_tests cuda_compile_tests)
set_tests_properties(cuda_compile_tests PROPERTIES LABELS compile_time)
