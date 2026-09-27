# Build-time checks on gpumod.extension.common's error/handle concepts and
# type traits. The build is the test: static_asserts, compiled under either
# WWR_GPU_BACKEND and without a GPU or GoogleTest (target
# compile_time_tests).
wwr_add_cxx_module_library(
  NAME
  gpumod.test.extension.common_error_handle
  IMPORT_STD
  NO_CUDA_DEVICE_LINKING
  PRIMARY_INTERFACE
  type_traits.cppm
  LINK_PUBLIC
  gpumod.complex
  gpumod.fp16
  gpumod.bf16
  gpumod.extension.common
  gpumod.extension.handle
)

add_dependencies(
  compile_time_tests gpumod.test.extension.common_error_handle
)
