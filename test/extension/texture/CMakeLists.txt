# wwr.extension.texture RAII wrapper tests
#
# Runtime only: wwrCreateTextureObject needs a live GPU context and a backing
# CUDA array. Same shape as test/extension/fft -- one plain .cpp, one ctest
# entry per suite. Links wwr.extension.common for its counting policy and
# wwr.runtime_api for the array + descriptor API the test builds against.

wwr_add_gtest_executable(
  NAME
  extension_texture_tests
  SOURCES
  object_tests.cpp
  LINK
  wwr::runtime_api
  wwr::extension::texture
  wwr::extension::common
  wwr::extension::handle
  wwr::test::shared::abort_policy
)

# One ctest entry per suite. Add a suite here when you add one to the test file,
# or the SuiteListIsComplete guard test will fail and say so.
wwr_add_gtest_suite_tests(
  TARGET
  extension_texture_tests
  SUITES
  TextureObjectTests
  TIMEOUT
  30
  REQUIRES_GPU
)
