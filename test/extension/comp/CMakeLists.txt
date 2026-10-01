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

# CompRoundTrip exercises nvcomp's Snappy decompressor, whose unsnap_kernel has
# shared-memory read/write hazards internal to the vendor kernel -- benign (the
# round-trip verifies correct output), but compute-sanitizer racecheck reports
# them and so fails an otherwise-clean tree under --error-exitcode. no_sanitizer
# excludes the suite from the sanitizer ctest runs (both `-LE no_sanitizer`);
# every other run keeps it. APPEND keeps REQUIRES_GPU's `gpu` label (see
# test/extension/memory_buffer). devtools/compute-sanitizer-suppressions.xml
# filters the same unsnap_kernel frames in place for a targeted sanitized run.
set_property(
  TEST CompRoundTrip
  APPEND
  PROPERTY LABELS no_sanitizer
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
