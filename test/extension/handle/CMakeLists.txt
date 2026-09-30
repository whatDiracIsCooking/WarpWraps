# DeviceBoundHandle runtime tests: the construction-ordering trick the layer
# exists for (select-before-create) and the by-reference->move device-policy
# threading, both observed on a single device by forcing wwrSetDevice to fail on
# an out-of-range ordinal. The compile-time contract (move-only, the defaulted
# device policy, the device-aware view()) lives in test/extension/common.
#
# Same shape as test/extension/runtime: one plain .cpp compiled straight into
# the executable, no module library, one ctest entry per suite, and
# GTest::gtest_main for main(). REQUIRES_GPU because the wwrSetDevice /
# wwrGetDevice calls need a live runtime -- even the failing select does.

wwr_add_gtest_executable(
  NAME
  handle_device_bound_tests
  SOURCES
  basic.cpp
  LINK
  wwr::runtime_api
  wwr::extension::common
  wwr::extension::handle
  wwr::test::shared::abort_policy
)

# One ctest entry per suite. Add a suite here when you add one, or the
# SuiteListIsComplete guard test will fail and say so.
wwr_add_gtest_suite_tests(
  TARGET
  handle_device_bound_tests
  SUITES
  DeviceBoundHandleTests
  TIMEOUT
  30
  REQUIRES_GPU
)

# BaseHandle::valid() liveness query (see valid.cpp): a pure query with no
# runtime calls, so -- unlike the device-bound suite above -- it is NOT
# REQUIRES_GPU and runs in CI. The create() failures are simulated by a
# RecordingPolicy test double, no card needed.
wwr_add_gtest_executable(
  NAME
  handle_valid_tests
  SOURCES
  valid.cpp
  LINK
  wwr::runtime_api
  wwr::extension::common
  wwr::extension::handle
  wwr::test::shared::recording_policy
)

wwr_add_gtest_suite_tests(
  TARGET
  handle_valid_tests
  SUITES
  BaseHandleValidTests
  TIMEOUT
  30
)
