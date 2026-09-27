# gpumod.extension.sparse RAII handle tests
#
# Runtime only: gpusparseCreate needs a live GPU context. Same shape as
# test/extension/blas -- one plain .cpp, one ctest entry per suite.

wwr_add_gtest_executable(
  NAME
  extension_sparse_tests
  SOURCES
  handle_tests.cpp
  LINK
  gpumod.extension.sparse
  gpumod.extension.handle
)

# One ctest entry per suite. Add a suite here when you add one to the test file,
# or the SuiteListIsComplete guard test will fail and say so.
wwr_add_gtest_suite_tests(
  TARGET
  extension_sparse_tests
  SUITES
  GpusparseHandleTests
  TIMEOUT
  30
  REQUIRES_GPU
)
