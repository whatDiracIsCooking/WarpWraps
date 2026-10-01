# Build-time checks on wwr.extension.common's error/handle concepts and
# type traits. The build is the test: static_asserts, compiled under either
# WWR_GPU_BACKEND and without a GPU or GoogleTest (target
# compile_time_tests).
wwr_add_cxx_module_library(
  NAME
  wwr.test.extension.common_error_handle
  IMPORT_STD
  NO_CUDA_DEVICE_LINKING
  PRIMARY_INTERFACE
  type_traits.cppm
  LINK_PUBLIC
  wwr::complex
  wwr::fp16
  wwr::bf16
  wwr::runtime_api
  wwr::extension::common
  wwr::extension::handle
  wwr::test::shared::abort_policy
)

add_dependencies(
  compile_time_tests wwr.test.extension.common_error_handle
)

# Everything below is a GoogleTest runtime suite. This directory is added before
# test/extension's own early return (its build-time half above must configure
# under WWR_COMPILE_TIME_ONLY), so it has to return early itself before the
# GoogleTest part -- which WWR_COMPILE_TIME_ONLY does not fetch. This is the
# idiom documented in test/extension/build_time/CMakeLists.txt.
if(WWR_COMPILE_TIME_ONLY)
  return()
endif()

# Host-only error-mapping suite: no device, so NOT REQUIRES_GPU -- this is the
# `ctest -LE gpu` (CI) coverage of common/error's wwrError_t specializations.
wwr_add_gtest_executable(
  NAME
  extension_common_error_tests
  SOURCES
  error_tests.cpp
  LINK
  wwr::runtime_api
  wwr::extension::common
)

wwr_add_gtest_suite_tests(
  TARGET
  extension_common_error_tests
  SUITES
  CommonErrorTests
  TIMEOUT
  30
)
