# Build-time checks on gpumod.extension.common's error/handle concepts and
# type traits. The build is the test: static_asserts, compiled under either
# GPUMOD_GPU_BACKEND and without a GPU or GoogleTest (target
# compile_time_tests).
gpumod_add_cxx_module_library(
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
  gpumod.extension.common.handle
)

add_dependencies(
  compile_time_tests gpumod.test.extension.common_error_handle
)
