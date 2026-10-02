# wwr.wrappers.thrust.reduce (#239, A-reduce) acceptance.
#
# Unlike the F2 shim test there is no device .cu here: the module's exported
# wrappers are host functions that forward to its device archive (pulled in
# whole by the module target), so the host suite just imports the module and
# calls them. Runtime only -- correctness and stream-honoring both need a live
# device -- so every suite is REQUIRES_GPU, like test/extension/rand and the F2
# stream suite.
wwr_add_gtest_executable(
  NAME
  extension_thrust_reduce_tests
  SOURCES
  reduce_tests.cpp
  LINK
  wwr::runtime_api
  wwr::wrappers::thrust::reduce
)

# Under CUDA, Thrust's host-side helpers instantiated through the reduction
# calls are compiled by nvcc's default (gcc) host pass against libstdc++, while
# the rest of the binary is libc++. Link libstdc++ explicitly so those host
# symbols resolve -- the same both-standard-libraries link the F2 stream suite
# makes, and for the same CUDA/Thrust reason. HIP compiles the device archive
# as CXX under libc++, so it needs nothing here.
if(WWR_GPU_BACKEND STREQUAL "CUDA")
  target_link_libraries(
    extension_thrust_reduce_tests
    PRIVATE /usr/lib/x86_64-linux-gnu/libstdc++.so.6
  )
endif()

wwr_add_gtest_suite_tests(
  TARGET
  extension_thrust_reduce_tests
  SUITES
  ThrustReduceStream
  TYPED_SUITES
  ReduceTyped
  TYPES
  float
  double
  int32_t
  int64_t
  TIMEOUT
  60
  REQUIRES_GPU
)
