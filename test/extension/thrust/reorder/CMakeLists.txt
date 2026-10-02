# A-sort (issue #238) acceptance for wwr.wrappers.thrust.reorder.
#
# No device source of its own: the reorder kernels live inside the module
# (src/wrappers/thrust/reorder/reorder.cu), so this is an ordinary host CXX
# compile that imports the module and drives it -- same shape as
# test/extension/rand and the scan acceptance. REQUIRES_GPU: the point is device
# execution on a real stream.
wwr_add_gtest_executable(
  NAME
  extension_thrust_reorder_tests
  SOURCES
  reorder_tests.cpp
  LINK
  wwr::runtime_api
  wwr::wrappers::thrust::reorder
)
# Under CUDA, Thrust instantiates host-side error-category helpers that nvcc's
# default (gcc) host pass compiles against libstdc++ while the rest is libc++;
# link libstdc++ explicitly, AFTER the module's device archive, so those host
# symbols resolve -- the same link the F2 stream test makes, for the same
# reason. HIP compiles the source as CXX under libc++ and needs nothing here.
if(WWR_GPU_BACKEND STREQUAL "CUDA")
  target_link_libraries(
    extension_thrust_reorder_tests
    PRIVATE /usr/lib/x86_64-linux-gnu/libstdc++.so.6
  )
endif()

wwr_add_gtest_suite_tests(
  TARGET
  extension_thrust_reorder_tests
  SUITES
  ThrustReorderTests
  TIMEOUT
  60
  REQUIRES_GPU
)
