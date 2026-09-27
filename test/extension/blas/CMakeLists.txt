# wwr.extension.blas RAII handle tests
#
# Runtime only: gpublasCreate needs a live GPU context, so there is no
# compile-time half to configure under WWR_COMPILE_TIME_ONLY. Same shape as
# test/extension/runtime -- one plain .cpp into the executable, one ctest entry
# per suite, GTest::gtest_main supplies main().

wwr_add_gtest_executable(
  NAME
  extension_blas_tests
  SOURCES
  handle_tests.cpp
  LINK
  wwr.extension.blas
  wwr.extension.handle
)

# One ctest entry per suite. Add a suite here when you add one to the test file,
# or the SuiteListIsComplete guard test will fail and say so.
wwr_add_gtest_suite_tests(
  TARGET
  extension_blas_tests
  SUITES
  GpublasHandleTests
  TIMEOUT
  30
  REQUIRES_GPU
)
