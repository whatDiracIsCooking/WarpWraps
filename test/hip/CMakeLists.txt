# HIP compile-time tests (built when WWR_GPU_BACKEND is HIP)
# Mirrors test/cuda/CMakeLists.txt -- see that file's header comment for the
# WWR_LINK_CHECK/WWR_DECLARED_CHECK/WHOLE_ARCHIVE rationale, which applies
# unchanged here.
#
# TWO executables (see the big comment on hip_compile_tests_gpu for the why):
#
#   hip_compile_tests      -- libraries whose .so loads with no GPU. Label
#                             `compile_time`; RUNS on the GPU-less CI runner, so
#                             reaching main() confirms every module compiled,
#                             the executable linked, AND its .so's load.
#   hip_compile_tests_gpu  -- hiptensor + rccl, whose load-time init aborts
#                             without a GPU. BUILT (so the link-check resolves
#                             all 80 symbols) but NO_RUN -- never launched,
#                             exactly like test/gpu's gpu_compile_tests_tensor.
# Expected exit code: 0 for hip_compile_tests; the other has no ctest entry.

set(HIP_TEST_LIBRARIES
    hip_runtime_api
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
    hipcomp
)

# hiptensor and rccl: link-checked in a NO_RUN executable (see below). Their
# shared objects load but their initialisation needs a GPU, so they cannot share
# hip_compile_tests, which must RUN GPU-less on CI.
set(HIP_TEST_LIBRARIES_GPU
    hiptensor
    rccl
)

# One wwr.test.hip.<lib> module per library, built identically for both groups.
foreach(_lib IN LISTS HIP_TEST_LIBRARIES HIP_TEST_LIBRARIES_GPU)
  wwr_add_cxx_module_library(
    NAME
    wwr.test.hip.${_lib}
    IMPORT_STD
    NO_CUDA_DEVICE_LINKING
    PRIMARY_INTERFACE
    ${_lib}.cppm
    LINK_PUBLIC
    wwr::hip::${_lib}
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
# and the LINK_CHECKs (test/shared/link_check.h) never reach it. Each target
# list is derived from its library list so a new library cannot be forgotten.
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

# ── hiptensor + rccl: link-checked, never run ────────────────────────────────
#
# These two compile and LINK with no GPU -- building this executable resolves
# all 80 re-exported symbols against the vendor .so, which IS the link-check
# (#117), and CI does that build. What they cannot do is RUN GPU-less: their
# load-time initialisation reaches for a device and aborts before main()
# (measured in wwr:hip with no --device: SIGSEGV; libhiptensor's constructor
# calls the HIP runtime -- "no ROCm-capable device is detected" at
# hiptensor/.../hip_device.cpp -- and librccl pulls librocm_smi64 /
# libhsa-runtime64 whose init faults the same way).
#
# This is NOT test/cuda's problem. There a driver .so is MISSING at load
# (libcuda.so.1), which a SONAME symlink to the toolkit stub cures
# (WWR_CUDA_DRIVER_STUBS). Here nothing is missing -- `ldd` resolves every
# dependency -- so there is nothing to stub; the library's own init needs real
# hardware.
#
# So they are verified exactly like test/gpu's gpu_compile_tests_tensor (whose
# wwr.tensor is hiptensor-backed and hits the same abort): a separate
# executable, WHOLE_ARCHIVE so WWR_LINK_CHECK forces every symbol to resolve at
# link time, built as a dependency of compile_time_tests, but NO_RUN -- no ctest
# entry, so never launched anywhere. Keeping them out of hip_compile_tests is
# what lets that binary keep RUNNING the other ~16 libraries' load check on the
# GPU-less CI runner; HIP has no stub-rescue like ci-cuda to bring it back.
set(_hip_test_targets_gpu "")
foreach(_lib IN LISTS HIP_TEST_LIBRARIES_GPU)
  list(APPEND _hip_test_targets_gpu wwr.test.hip.${_lib})
endforeach()

wwr_add_test_executable(
  NAME
  hip_compile_tests_gpu
  MAIN
  gpu_link_check.cpp
  LINK_PRIVATE
  ${_hip_test_targets_gpu}
  WHOLE_ARCHIVE
  ${_hip_test_targets_gpu}
  NO_RUN
)

add_dependencies(compile_time_tests hip_compile_tests_gpu)
