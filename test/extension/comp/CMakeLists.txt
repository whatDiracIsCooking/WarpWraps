# Tests for compression: the vendor batched LLIF (wwr.comp) and the layer-3
# convenience over it (wwr.extension.comp).
#
# Three executables, split by what each needs:
#   * comp_roundtrip_tests - the wwr.comp batched round-trip, REQUIRES_GPU
#   * comp_error_tests     - wwr.extension.comp's error mapping, HOST-ONLY so it
#                            runs under `ctest -LE gpu` in CI (no device)
#   * comp_scratch_tests   - wwr.extension.comp's CompScratch RAII, REQUIRES_GPU
# Same shape as the sibling extension tests: plain .cpp straight into the
# executable, GTest::gtest_main supplies main().

wwr_add_gtest_executable(
  NAME
  comp_roundtrip_tests
  SOURCES
  roundtrip.cpp
  LINK
  wwr::runtime_api
  wwr::comp
)

wwr_add_gtest_suite_tests(
  TARGET
  comp_roundtrip_tests
  SUITES
  CompRoundTrip
  TIMEOUT
  30
  REQUIRES_GPU
)

# Host-only error-mapping suite: no device, so NOT REQUIRES_GPU -- this is the
# `ctest -LE gpu` (CI) coverage of the comp_error specializations.
wwr_add_gtest_executable(
  NAME
  comp_error_tests
  SOURCES
  error_tests.cpp
  LINK
  wwr::extension::common
  wwr::extension::comp
)

wwr_add_gtest_suite_tests(
  TARGET
  comp_error_tests
  SUITES
  CompErrorTests
  TIMEOUT
  30
)

# CompScratch RAII lifetime suite: allocates a real device block, REQUIRES_GPU.
wwr_add_gtest_executable(
  NAME
  comp_scratch_tests
  SOURCES
  scratch_tests.cpp
  LINK
  wwr::runtime_api
  wwr::extension::common
  wwr::extension::handle
  wwr::extension::comp
  wwr::test::shared::abort_policy
  wwr::test::shared::device_handle
)

wwr_add_gtest_suite_tests(
  TARGET
  comp_scratch_tests
  SUITES
  CompScratchTests
  TIMEOUT
  30
  REQUIRES_GPU
)
