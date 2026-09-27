# gpumod.extension.init_state + gpumod.extension.random_normal tests
#
# One integration suite over the pair: random_normal draws from states that
# init_state seeds, so every case needs both modules.
#
# Runtime only: their whole point is device code, and the cuRAND / hipRAND
# device functions underneath them cannot be called from host code, so there is
# no compile-time counterpart to test/gpu/rand.cppm for them. Running the
# kernels and checking the numbers that come back IS the test.
#
# No device-compiled source of its own: the kernels live inside the two modules
# (src/extension/init_state/init_state.cu and
# src/extension/random_normal/random_normal.cu), so this test TU is an ordinary
# host CXX compile that just imports them.

wwr_add_gtest_executable(
  NAME
  extension_rand_tests
  SOURCES
  rand_tests.cpp
  LINK
  gpumod.runtime_api
  gpumod.rand
  gpumod.complex
  gpumod.fp16
  gpumod.bf16
  gpumod.extension.runtime
  gpumod.extension.memory_buffer
  gpumod.extension.init_state
  gpumod.extension.random_normal
)

# One ctest entry per suite. Add a suite here when you add one to the test
# file, or the SuiteListIsComplete guard test will fail and say so.
wwr_add_gtest_suite_tests(
  TARGET
  extension_rand_tests
  SUITES
  RandTests
  TIMEOUT
  60
  REQUIRES_GPU
)
