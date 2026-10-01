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

# CclCommTests drives wwrcclCommInitAll, whose NCCL backend probes the driver
# entry point (cudaGetDriverEntryPointByVersion) and per-arch kernel images
# (cudaFuncGetAttributes -> cudaErrorNoKernelImageForDevice) during init and
# handles the misses internally. compute-sanitizer memcheck still counts those
# benign CUDA API returns, so under --error-exitcode they fail an
# otherwise-clean tree. no_sanitizer excludes the suite from the sanitizer
# ctest runs (the asan and compute-sanitizer services both `-LE no_sanitizer`)
# while every other run keeps it. APPEND, not a plain set: REQUIRES_GPU above
# already put `gpu` on this entry and the plain form would replace it -- see
# test/extension/memory_buffer for the full rationale. To sanitize the suite
# anyway, filter the vendor frames in place with
# devtools/compute-sanitizer-suppressions.xml.
set_property(
  TEST CclCommTests
  APPEND
  PROPERTY LABELS no_sanitizer
)
