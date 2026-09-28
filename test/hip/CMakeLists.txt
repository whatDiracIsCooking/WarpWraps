# HIP compile-time tests (built when WWR_GPU_BACKEND is HIP)
# Mirrors test/cuda/CMakeLists.txt exactly -- see that file's header comment
# for the WWR_LINK_CHECK/WWR_DECLARED_CHECK/WHOLE_ARCHIVE rationale, which
# applies unchanged here.
#
# hip_compile_tests below is the one thing that runs (ctest label
# compile_time). It has no checks of its own -- reaching main() means every
# module compiled and the executable linked.
# Expected exit code: 0

set(HIP_TEST_LIBRARIES
    hip_runtime_api
    rocm_smi
    amd_smi
    roctracer
    roctx
    hip_complex
    hip_fp16
    hip_bf16
    hip_fp8
    hiprtc
    # hip_fp4, hip_fp6 deliberately NOT listed -- see the BLOCKED comment in
    # src/hip/CMakeLists.txt and "hip_fp4 / hip_fp6: blocked, not built" in
    # src/hip/README.md. test/hip/hip_fp4.cppm and test/hip/hip_fp6.cppm exist
    # as reviewed source, ready to add here once that's resolved.
    hipblas
    hipblaslt
    hipsolver
    hipsparse
    hipfft
    hipfftXt
    hiprand
    hiprand_kernel
)

foreach(_lib IN LISTS HIP_TEST_LIBRARIES)
  wwr_add_cxx_module_library(
    NAME
    wwr.test.hip.${_lib}
    IMPORT_STD
    NO_CUDA_DEVICE_LINKING
    PRIMARY_INTERFACE
    ${_lib}.cppm
    LINK_PUBLIC
    wwr.hip.${_lib}
    # PROJECT_SOURCE_DIR (not CMAKE_SOURCE_DIR) is gpumod's own root, so the
    # root-relative #include "test/shared/link_check.h" resolves even when a
    # consumer adds gpumod as a subdirectory -- there CMAKE_SOURCE_DIR is the
    # consumer's root, which has no test/shared.
    INCLUDE_DIRS_PRIVATE
    ${PROJECT_SOURCE_DIR}
  )
  add_dependencies(compile_time_tests wwr.test.hip.${_lib})
endforeach()

# Every test library is linked WHOLE_ARCHIVE. They are static archives and
# main() calls nothing in them, so without it the linker pulls no member at all
# and the LINK_CHECKs (test/shared/link_check.h) never reach it. The list is
# derived from HIP_TEST_LIBRARIES so a new library cannot be forgotten.
set(_hip_test_targets "")
foreach(_lib IN LISTS HIP_TEST_LIBRARIES)
  list(APPEND _hip_test_targets wwr.test.hip.${_lib})
endforeach()

wwr_add_test_executable(
  NAME
  hip_compile_tests
  TIMEOUT
  30
  LINK_PRIVATE
  ${_hip_test_targets}
  WHOLE_ARCHIVE
  ${_hip_test_targets}
)

add_dependencies(compile_time_tests hip_compile_tests)
set_tests_properties(hip_compile_tests PROPERTIES LABELS compile_time)
