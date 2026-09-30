# CUDA compile-time tests
# Verify compile-time flags, type traits, and link symbols for CUDA libraries.
#
# Compile-time only: a failing static_assert fails the build of that module, so
# building these targets IS running the tests. This directory is added even
# under WWR_COMPILE_TIME_ONLY; build just it with
#   cmake --build build --target compile_time_tests
#
# cuda_compile_tests below is the one thing that runs (ctest label
# compile_time). It has no checks of its own -- reaching main() means every
# module compiled and the executable linked.
# Expected exit code: 0
#
# Each CUDA library gets one wwr.test.cuda.<name> module wrapping its
# compile-time test (<name>.cppm) in an identical shape -- a foreach builds them
# all rather than repeating a 10-line wwr_add_cxx_module_library() block for
# each. cuda.cppm is the one irregular name: its module is cuda_h (matching
# the wrapped module wwr.cuda.cuda_h -- "cuda" on its own collides
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
    nvToolsExt
    cuda_profiler_api
    cusolverSp
    nvFatbin
    cufile
    nvJitLink
    nvjpeg
    nvcomp
    nvml
    nvrtc
    cublas_v2
)

foreach(_lib IN LISTS CUDA_TEST_LIBRARIES)
  if(_lib STREQUAL "cuda_h")
    set(_source "cuda.cppm")
  else()
    set(_source "${_lib}.cppm")
  endif()

  wwr_add_cxx_module_library(
    NAME
    wwr.test.cuda.${_lib}
    IMPORT_STD
    NO_CUDA_DEVICE_LINKING
    PRIMARY_INTERFACE
    ${_source}
    LINK_PUBLIC
    wwr.cuda.${_lib}
    # PROJECT_SOURCE_DIR (not CMAKE_SOURCE_DIR) is gpumod's own root, so the
    # root-relative #include "test/shared/link_check.h" resolves even when a
    # consumer adds gpumod as a subdirectory -- there CMAKE_SOURCE_DIR is the
    # consumer's root, which has no test/shared.
    INCLUDE_DIRS_PRIVATE
    ${PROJECT_SOURCE_DIR}
  )
  add_dependencies(compile_time_tests wwr.test.cuda.${_lib})
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
  list(APPEND _cuda_test_targets wwr.test.cuda.${_lib})
endforeach()

wwr_add_test_executable(
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

# This is a COMPILE-time test that nonetheless cannot LOAD on a machine with
# no NVIDIA driver. It links the driver-API stubs (cuda_h, nvml), so the binary
# needs libcuda.so.1 and libnvidia-ml.so.1 at load and dies before main() with
# `error while loading shared libraries` in a container started without --gpus
# -- a GitHub-hosted runner, for instance. Measured with ldd: those two are the
# ONLY unresolved entries; every other DT_NEEDED comes from the toolkit.
# test/hip's counterpart has no such dependency and needs none of this.
#
# Two ways to cope, and the option picks between them.
#
# WWR_CUDA_DRIVER_STUBS=OFF (default) -- carry the `gpu` label and be
# excluded by `ctest -LE gpu`. Excluding it costs little: a compile error fails
# the build, and so does an undefined symbol, because reaching main() is
# exactly "it linked". Only the loader check is lost, and that is precisely
# what a driverless box cannot answer honestly.
#
# WWR_CUDA_DRIVER_STUBS=ON (the ci-cuda preset) -- point the test at the
# toolkit's own stubs and drop the label, so it RUNS. The binary never calls
# the driver; its main() prints and exits. This recovers the loader check for
# the other ~15 libraries, and makes the CUDA and HIP CI legs report the same
# ctest count, which is worth something on its own: an 11 that is supposed to
# be 11 is the kind of number nobody notices drifting to 10.
#
# WHAT IT DOES NOT PROVE: that the real driver would load. It is a stub. Read a
# pass here as "every non-driver dependency resolves".
#
# Three deliberate choices in how it is wired:
#
#   * The symlinks live in the BUILD directory, not the image. Stubs on a
#     container-wide LD_LIBRARY_PATH would beat the loader cache and shadow the
#     real driver the NVIDIA runtime injects on a machine that has one -- an
#     image-wide footgun for a one-test problem.
#   * ENVIRONMENT_MODIFICATION, not ENVIRONMENT. The latter REPLACES the
#     variable at configure time; the toolkit's own libraries are found through
#     LD_LIBRARY_PATH too, so replacing it would break the other 15 while
#     fixing these 2. path_list_prepend edits it at test time instead.
#   * The stub paths come from the cache variables FindCUDAToolkit already
#     resolved, never from a constructed path -- the stubs sit under
#     targets/<arch>-linux/lib/stubs, which is not where lib64 suggests.
if(WWR_CUDA_DRIVER_STUBS)
  if(NOT CUDA_cuda_driver_LIBRARY OR NOT CUDA_nvml_LIBRARY)
    message(
      FATAL_ERROR
        "WWR_CUDA_DRIVER_STUBS=ON but FindCUDAToolkit did not resolve the "
        "driver stubs (CUDA_cuda_driver_LIBRARY / CUDA_nvml_LIBRARY are "
        "empty). Turn the option off, or install the CUDA toolkit's stubs."
    )
  endif()

  # SONAME-named symlinks to the stubs the linker already used. The loader
  # wants libcuda.so.1; the toolkit ships libcuda.so.
  set(_wwr_stub_dir "${CMAKE_CURRENT_BINARY_DIR}/driver-stubs")
  file(MAKE_DIRECTORY "${_wwr_stub_dir}")
  file(CREATE_LINK "${CUDA_cuda_driver_LIBRARY}"
       "${_wwr_stub_dir}/libcuda.so.1" SYMBOLIC
  )
  file(CREATE_LINK "${CUDA_nvml_LIBRARY}"
       "${_wwr_stub_dir}/libnvidia-ml.so.1" SYMBOLIC
  )

  set_tests_properties(cuda_compile_tests PROPERTIES LABELS compile_time)
  set_tests_properties(
    cuda_compile_tests
    PROPERTIES ENVIRONMENT_MODIFICATION
               "LD_LIBRARY_PATH=path_list_prepend:${_wwr_stub_dir}"
  )
else()
  set_tests_properties(
    cuda_compile_tests PROPERTIES LABELS "compile_time;gpu"
  )
endif()
