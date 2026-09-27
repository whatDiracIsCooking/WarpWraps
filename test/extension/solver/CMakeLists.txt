# wwr.extension.solver RAII handle tests
#
# Runtime only: wwrsolverDnCreate needs a live GPU context. Same shape as
# test/extension/blas -- one plain .cpp, one ctest entry per suite. Two suites
# here, one per live type the module owns (the Dn handle and its params).

wwr_add_gtest_executable(
  NAME
  extension_solver_tests
  SOURCES
  handle_tests.cpp
  LINK
  wwr.extension.solver
  wwr.extension.handle
)

# One ctest entry per suite. Add a suite here when you add one to the test file,
# or the SuiteListIsComplete guard test will fail and say so.
wwr_add_gtest_suite_tests(
  TARGET
  extension_solver_tests
  SUITES
  WwrsolverDnHandleTests
  WwrsolverDnParamsTests
  TIMEOUT
  30
  REQUIRES_GPU
)
