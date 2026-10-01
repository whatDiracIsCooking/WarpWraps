# wwr.extension.blaslt RAII wrapper + error-policy tests
#
# One binary, two ctest registrations. BlasLtErrorTests is host-only -- it
# checks the shared wwrblasLtStatus_t error policy (reused from
# wwr.extension.blas, since wwrblasLtStatus_t IS wwrblasStatus_t) and needs no
# device, so it runs in CI under `ctest -LE gpu`. BlasLtHandleTests is
# REQUIRES_GPU: wwrblasLtCreate and
# the descriptor creates need a live GPU context, and the destroy-exactly-once
# contract is the whole point.
#
# Links wwr::extension::blas alongside wwr::extension::blaslt so the error suite
# resolves the specializations with BOTH visible (the duplicate-specialization
# guard); wwr::backend for the WWR_GPU_BACKEND_* define the test's compute-type
# import is gated on; wwr::solver for WWRSOLVER_R_32F (the scale/element type).
#
# The raw backend module carries the compute-type value enumerator
# (CUBLAS_COMPUTE_32F / HIPBLAS_COMPUTE_32F) that cuBLASLt / hipBLASLt re-export
# the type but not the value of. It is imported as a BMI, not a textual vendor
# header, to dodge the libc++ <array> / `import std;` clash on HIP -- see
# matmul_tests.cpp's header -- so its module target is linked per backend here.
if(WWR_GPU_BACKEND STREQUAL "CUDA")
  set(_blaslt_compute_module wwr::cuda::cublas_v2)
else()
  set(_blaslt_compute_module wwr::hip::hipblas)
endif()

wwr_add_gtest_executable(
  NAME
  extension_blaslt_tests
  SOURCES
  matmul_tests.cpp
  LINK
  wwr::backend
  wwr::blaslt
  wwr::solver
  ${_blaslt_compute_module}
  wwr::extension::blas
  wwr::extension::blaslt
  wwr::extension::common
  wwr::extension::handle
  wwr::test::shared::abort_policy
)

# Host-only error-mapping suite -- no REQUIRES_GPU, so CI's `-LE gpu` runs it.
wwr_add_gtest_suite_tests(
  TARGET
  extension_blaslt_tests
  SUITES
  BlasLtErrorTests
  TIMEOUT
  30
)

# Device-requiring lifetime suite.
wwr_add_gtest_suite_tests(
  TARGET
  extension_blaslt_tests
  SUITES
  BlasLtHandleTests
  TIMEOUT
  30
  REQUIRES_GPU
)
