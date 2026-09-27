# gpumod.extension.solver RAII handle tests
#
# Runtime only: gpusolverDnCreate needs a live GPU context. Same shape as
# test/extension/blas -- one plain .cpp, one ctest entry per suite. Two suites
# here, one per live type the module owns (the Dn handle and its params).

gpumod_add_gtest_executable(
  NAME
  extension_solver_tests
  SOURCES
  handle_tests.cpp
  LINK
  gpumod.extension.solver
  gpumod.extension.common.handle
)

# One ctest entry per suite. Add a suite here when you add one to the test file,
# or the SuiteListIsComplete guard test will fail and say so.
gpumod_add_gtest_suite_tests(
  TARGET
  extension_solver_tests
  SUITES
  GpusolverDnHandleTests
  GpusolverDnParamsTests
  TIMEOUT
  30
  REQUIRES_GPU
)
