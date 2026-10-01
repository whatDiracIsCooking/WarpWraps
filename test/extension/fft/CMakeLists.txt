# wwr.extension.fft RAII plan tests
#
# Runtime only: wwrfftCreate needs a live GPU context. Same shape as
# test/extension/blas -- one plain .cpp, one ctest entry per suite. Links
# wwr.extension.common as well, since the test substitutes its own
# a counting policy for the default one to observe destroy-exactly-once.

wwr_add_gtest_executable(
  NAME
  extension_fft_tests
  SOURCES
  plan_tests.cpp
  LINK
  wwr::runtime_api
  wwr::extension::fft
  wwr::extension::common
  wwr::extension::handle
  wwr::extension::runtime
  wwr::test::shared::abort_policy
  wwr::test::shared::device_handle
)

# One ctest entry per suite. Add a suite here when you add one to the test file,
# or the SuiteListIsComplete guard test will fail and say so.
wwr_add_gtest_suite_tests(
  TARGET
  extension_fft_tests
  SUITES
  FftPlanTests
  TIMEOUT
  30
  REQUIRES_GPU
)

# Host-only error-mapping suite: no device, so NOT REQUIRES_GPU -- this is the
# `ctest -LE gpu` (CI) coverage of the fft_error specializations.
wwr_add_gtest_executable(
  NAME
  extension_fft_error_tests
  SOURCES
  error_tests.cpp
  LINK
  wwr::extension::common
  wwr::extension::fft
)

wwr_add_gtest_suite_tests(
  TARGET
  extension_fft_error_tests
  SUITES
  FftErrorTests
  TIMEOUT
  30
)
