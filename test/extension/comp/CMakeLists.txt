# Device round-trip tests for wwr.comp (backend-neutral batched compression).
#
# Same shape as test/extension/runtime: one plain .cpp compiled straight into
# the executable, GTest::gtest_main supplies main(). The neutral shims run the
# vendor batched LLIF on a real device, so the one suite is REQUIRES_GPU.

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
