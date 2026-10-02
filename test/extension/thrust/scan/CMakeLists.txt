# A-scan (issue #240) acceptance for wwr.wrappers.thrust.scan.
#
# No device source of its own: the scan kernels live inside the module
# (src/wrappers/thrust/scan/scan.cu), so this is an ordinary host CXX compile
# that imports the module and drives it -- same shape as test/extension/rand.
# REQUIRES_GPU: the point is device execution on a real stream.
wwr_add_gtest_executable(
  NAME
  extension_thrust_scan_tests
  SOURCES
  scan_tests.cpp
  LINK
  wwr::runtime_api
  wwr::wrappers::thrust::scan
)

# Under CUDA, Thrust instantiates host-side error-category helpers that nvcc's
# default (gcc) host pass compiles against libstdc++, while the rest of the
# binary is libc++. Link libstdc++ explicitly, AFTER the module (which carries
# the whole-archived device library), so those host symbols resolve -- the same
# both-standard-libraries link the F2 stream test and test/cuda make, for the
# same CUDA/Thrust reason. HIP compiles the device TU as CXX under libc++, so it
# needs nothing here.
if(WWR_GPU_BACKEND STREQUAL "CUDA")
  target_link_libraries(
    extension_thrust_scan_tests PRIVATE /usr/lib/x86_64-linux-gnu/libstdc++.so.6
  )
endif()

wwr_add_gtest_suite_tests(
  TARGET
  extension_thrust_scan_tests
  SUITES
  ThrustScanTests
  TIMEOUT
  60
  REQUIRES_GPU
)
