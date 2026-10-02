# wwr* Thrust execution-policy shim (issue #237, F2) test.
#
# A device TU (stream_ops.cu) #includes src/wrappers/thrust/execution_policy.cuh
# and calls thrust::sort(wwr::par_on(stream), ...); the host suite
# (stream_tests.cpp) proves the stream is honored -- deterministically, by
# reading the bound stream back out of the policy, and behaviourally, by sorting
# on it. Runtime only -- the point is device execution on a real stream -- so
# every suite is REQUIRES_GPU, like test/extension/rand.

# The device-kernel half: stream_ops.cu, device-compiled for the selected
# backend. wwr::thrust carries the backend's Thrust include dirs, the src/
# include root (so "wrappers/thrust/execution_policy.cuh" and "runtime.h"
# resolve) and wwr::device's runtime. INCLUDE_DIRS is this directory, for the
# sibling stream_bridge.h.
wwr_add_gpu_device_library(
  NAME
  wwr.test.extension.thrust.device
  SOURCES
  stream_ops.cu
  LINK_PRIVATE
  wwr::thrust
)
target_include_directories(
  wwr.test.extension.thrust.device PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

# The host suite: ordinary CXX, imports the runtime API, drives the device
# archive through stream_bridge.h. The archive is referenced directly (not
# lazily), so it needs no WHOLE_ARCHIVE genex -- a plain link resolves it.
wwr_add_gtest_executable(
  NAME
  extension_thrust_tests
  SOURCES
  stream_tests.cpp
  LINK
  wwr::runtime_api
  wwr.test.extension.thrust.device
)
target_include_directories(
  extension_thrust_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)
# Under CUDA, thrust::sort instantiates Thrust's host-side error-category
# helpers, which nvcc's default (gcc) host pass compiles against libstdc++; the
# rest of the binary is libc++. Link libstdc++ explicitly, AFTER the CUDA device
# archive, so those host symbols resolve -- the same both-standard-libraries
# link test/cuda's cuda_compile_tests (LINK_LIBSTDCXX) makes, and for the same
# CUDA/Thrust reason. HIP compiles the source as CXX under libc++, so it needs
# nothing here.
if(WWR_GPU_BACKEND STREQUAL "CUDA")
  target_link_libraries(
    extension_thrust_tests PRIVATE /usr/lib/x86_64-linux-gnu/libstdc++.so.6
  )
endif()

wwr_add_gtest_suite_tests(
  TARGET
  extension_thrust_tests
  SUITES
  ThrustStreamTests
  TIMEOUT
  60
  REQUIRES_GPU
)

# Per-family A-algorithm acceptance suites (issues #238-#241), each a
# self-contained subdirectory importing its wwr.wrappers.thrust.* module. One
# add_subdirectory per family, one-per-line, so the four parallel family
# branches merge additively.
add_subdirectory(scan)
