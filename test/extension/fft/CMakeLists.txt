# gpumod.extension.fft RAII plan tests
#
# Runtime only: gpufftCreate needs a live GPU context. Same shape as
# test/extension/blas -- one plain .cpp, one ctest entry per suite. Links
# gpumod.extension.common as well, since the test substitutes its own
# BaseErrorPolicy for the default one to observe destroy-exactly-once.

gpumod_add_gtest_executable(
  NAME
  extension_fft_tests
  SOURCES
  plan_tests.cpp
  LINK
  gpumod.extension.fft
  gpumod.extension.common
  gpumod.extension.common.handle
)

# One ctest entry per suite. Add a suite here when you add one to the test file,
# or the SuiteListIsComplete guard test will fail and say so.
gpumod_add_gtest_suite_tests(
  TARGET
  extension_fft_tests
  SUITES
  FftPlanTests
  TIMEOUT
  30
  REQUIRES_GPU
)
