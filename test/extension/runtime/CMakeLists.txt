# GPU runtime wrapper tests (backend-neutral: built and run for either
# GPUMOD_GPU_BACKEND)
# RAII contract of GpuStream, GpuEvent, GpuGraph(Exec), GpuMemPool and the
# StreamEventPair bundle, plus DeviceScope (from gpumod.extension.common), the
# device-current guard that restores the previous device on destruction
#
# Same shape as test/extension/memory_buffer: one plain .cpp compiled straight
# into the executable, no module library, no WHOLE_ARCHIVE, one ctest entry per
# suite. These tests share no state, so GTest::gtest_main supplies main().

gpumod_add_gtest_executable(
  NAME
  runtime_basic_tests
  SOURCES
  basic.cpp
  LINK
  gpumod.runtime_api
  gpumod.extension.common
  gpumod.extension.runtime
)

# One ctest entry per suite. Add a suite here when you add one, or the
# SuiteListIsComplete guard test will fail and say so.
gpumod_add_gtest_suite_tests(
  TARGET
  runtime_basic_tests
  SUITES
  GpuEventTests
  GpuStreamTests
  GpuGraphTests
  GpuGraphExecTests
  GpuMemPoolTests
  StreamEventPairTests
  DeviceScopeTests
  TIMEOUT
  30
  REQUIRES_GPU
)
