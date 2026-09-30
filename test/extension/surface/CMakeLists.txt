# wwr.extension.surface RAII wrapper tests
#
# Runtime only: wwrCreateSurfaceObject needs a live GPU context and a
# surface-load-store CUDA array. Same shape as test/extension/texture.

wwr_add_gtest_executable(
  NAME
  extension_surface_tests
  SOURCES
  object_tests.cpp
  LINK
  wwr::runtime_api
  wwr::extension::surface
  wwr::extension::common
  wwr::extension::handle
  wwr::test::shared::abort_policy
)

# One ctest entry per suite. Add a suite here when you add one to the test file,
# or the SuiteListIsComplete guard test will fail and say so.
wwr_add_gtest_suite_tests(
  TARGET
  extension_surface_tests
  SUITES
  SurfaceObjectTests
  TIMEOUT
  30
  REQUIRES_GPU
)
