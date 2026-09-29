# Build-time checks on wwr.extension.common's error/handle concepts and
# type traits. The build is the test: static_asserts, compiled under either
# WWR_GPU_BACKEND and without a GPU or GoogleTest (target
# compile_time_tests).
wwr_add_cxx_module_library(
  NAME
  wwr.test.extension.common_error_handle
  IMPORT_STD
  NO_CUDA_DEVICE_LINKING
  PRIMARY_INTERFACE
  type_traits.cppm
  LINK_PUBLIC
  wwr.complex
  wwr.fp16
  wwr.bf16
  wwr.runtime_api
  wwr.extension.common
  wwr.extension.handle
  wwr.test.shared.abort_policy
)

add_dependencies(
  compile_time_tests wwr.test.extension.common_error_handle
)
