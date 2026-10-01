# wwr.extension.tensor tests, split by whether they need a live device.
#
# error_tests.cpp is host-only -- the wwrtensorStatus_t success_code/error_name/
# error_string mapping is a pure table lookup with no device contact -- so it is
# NOT REQUIRES_GPU and runs in CI (ctest -LE gpu). It is the same split as
# test/extension/handle (valid vs device-bound).
wwr_add_gtest_executable(
  NAME
  extension_tensor_error_tests
  SOURCES
  error_tests.cpp
  LINK
  wwr::extension::tensor
  wwr::extension::common
)

wwr_add_gtest_suite_tests(
  TARGET
  extension_tensor_error_tests
  SUITES
  TensorErrorTests
  TIMEOUT
  30
)

# lifetime_tests.cpp exercises the RAII handle + descriptor: wwrtensorCreate
# needs a live GPU context, so it is REQUIRES_GPU (labeled `gpu`, excluded in
# CI). The destroy-exactly-once contract through a counting policy is the whole
# test.
wwr_add_gtest_executable(
  NAME
  extension_tensor_lifetime_tests
  SOURCES
  lifetime_tests.cpp
  LINK
  wwr::extension::tensor
  wwr::extension::common
  wwr::extension::handle
)

# One ctest entry per suite. Add a suite here when you add one to the test file,
# or the SuiteListIsComplete guard test will fail and say so.
wwr_add_gtest_suite_tests(
  TARGET
  extension_tensor_lifetime_tests
  SUITES
  TensorHandleTests
  TensorDescriptorTests
  TIMEOUT
  30
  REQUIRES_GPU
)
