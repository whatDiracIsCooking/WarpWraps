# wwr.extension.sparse RAII handle tests
#
# Runtime only: wwrsparseCreate needs a live GPU context. Same shape as
# test/extension/blas -- one plain .cpp, one ctest entry per suite.

wwr_add_gtest_executable(
  NAME
  extension_sparse_tests
  SOURCES
  handle_tests.cpp
  LINK
  wwr::runtime_api
  wwr::extension::sparse
  wwr::extension::handle
  wwr::extension::runtime
  wwr::test::shared::abort_policy
  wwr::test::shared::device_handle
)

# One ctest entry per suite. Add a suite here when you add one to the test file,
# or the SuiteListIsComplete guard test will fail and say so.
wwr_add_gtest_suite_tests(
  TARGET
  extension_sparse_tests
  SUITES
  SparseHandleTests
  TIMEOUT
  30
  REQUIRES_GPU
)

# Host-only error-mapping suite: no device, so NOT REQUIRES_GPU -- this is the
# `ctest -LE gpu` (CI) coverage of the sparse_error specializations.
wwr_add_gtest_executable(
  NAME
  extension_sparse_error_tests
  SOURCES
  error_tests.cpp
  LINK
  wwr::extension::common
  wwr::extension::sparse
)

wwr_add_gtest_suite_tests(
  TARGET
  extension_sparse_error_tests
  SUITES
  SparseErrorTests
  TIMEOUT
  30
)
