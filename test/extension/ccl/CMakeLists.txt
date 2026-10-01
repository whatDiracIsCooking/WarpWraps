# wwr.extension.ccl error-policy and RAII communicator tests
#
# One executable, two ctest groupings split by what each suite needs:
#   - CclErrorTests is host-only (error-code specialization mapping), so it is
#     NOT labeled gpu and runs under `ctest -LE gpu` / in CI -- the link check.
#   - CclCommTests is REQUIRES_GPU: wwrcclCommInitAll needs a live GPU context.
# Same shape as test/extension/fft otherwise -- one plain .cpp,
# GTest::gtest_main supplies main(). Links wwr.extension.common for its policy.

wwr_add_gtest_executable(
  NAME
  extension_ccl_tests
  SOURCES
  comm_tests.cpp
  LINK
  wwr::extension::ccl
  wwr::extension::common
  wwr::extension::handle
  wwr::test::shared::abort_policy
)

# Host-only: the error-code specialization mapping. No gpu label, so it runs
# under `ctest -LE gpu`.
wwr_add_gtest_suite_tests(
  TARGET
  extension_ccl_tests
  SUITES
  CclErrorTests
  TIMEOUT
  30
)

# Device-requiring: communicator construct/destroy. Labeled gpu by REQUIRES_GPU.
# Both calls accumulate onto the one SuiteListIsComplete guard for the target.
wwr_add_gtest_suite_tests(
  TARGET
  extension_ccl_tests
  SUITES
  CclCommTests
  TIMEOUT
  30
  REQUIRES_GPU
)
