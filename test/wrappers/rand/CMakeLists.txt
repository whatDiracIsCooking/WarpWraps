# GPU RNG wrapper tests (backend-neutral: built for either WWR_GPU_BACKEND)
#
# Two halves, in dependency-independent order:
#
#   * Build-time (both backends, no GPU): the typed generate wrappers pick their
#     entry point by token-pasting a precision-specific name suffix (Uniform /
#     UniformDouble, ...), which no static_assert can see, so rand_dispatch.toml
#     checks it in the compiled objects. This must configure under
#     WWR_COMPILE_TIME_ONLY too, so it comes before the early return below.
#   * Runtime (REQUIRES_GPU): the wrappers are thin forwarders, but a live
#     generator is the one thing the dispatch check cannot exercise -- that the
#     pasted call actually returns success against a real cuRAND / hipRAND
#     generator. rand_tests.cpp does that for float and double.

# GPU_SOURCE is the file carrying the WWR_FUNCTION_RAW(wwrrand...) alias lines
# the check maps vendor symbols back through -- the shared fragment, not
# rand.cppm, since the host surface moved there (rand.cppm now just #includes
# it, and dispatch.py scans one file's text, not its includes).
wwr_add_dispatch_check(
  NAME
  wwr.test.wrappers.rand_dispatch
  TARGET
  wwr.wrappers.rand
  TABLE
  rand_dispatch.toml
  GPU_SOURCE
  src/detail/rand_names.h
)

# Build-time targets are configured above; the GoogleTest runtime suite needs a
# GPU, so it is skipped under WWR_COMPILE_TIME_ONLY.
if(WWR_COMPILE_TIME_ONLY)
  return()
endif()

wwr_add_gtest_executable(
  NAME
  wrappers_rand_tests
  SOURCES
  rand_tests.cpp
  LINK
  wwr::runtime_api
  wwr::rand
  wwr::wrappers::rand
  wwr::extension::runtime
  wwr::extension::memory_buffer
  wwr::test::shared::device_handle
)

# One ctest entry per suite. Add a suite here when you add one to the test file,
# or the SuiteListIsComplete guard test will fail and say so.
wwr_add_gtest_suite_tests(
  TARGET
  wrappers_rand_tests
  SUITES
  WrappersRandTests
  TIMEOUT
  60
  REQUIRES_GPU
)
