# Tests for gpumod.wrappers.common, the backend-neutral foundation the four
# extensions are built on (built for either GPUMOD_GPU_BACKEND).
#
# Build-time only. common is the fp/int concepts and the real/complex/half type
# maps -- no compiled body, so there is nothing to disassemble and nothing to
# run: a failing static_assert in type_traits.cppm fails the build, which is the
# test. The checks import gpumod.wrappers.common directly rather than riding on a
# downstream extension's incidental use of a given name. type_traits.cppm also
# pins int_type to the two index widths the BLAS *_DISPATCH_64 macros handle; #58
# is what a wider domain costs. (See src/wrappers/common/README.md for the
# error-policy / RAII-handle design.)

gpumod_add_cxx_module_library(
  NAME
  gpumod.test.wrappers.common_type_traits
  IMPORT_STD
  NO_CUDA_DEVICE_LINKING
  PRIMARY_INTERFACE
  type_traits.cppm
  LINK_PUBLIC
  gpumod.complex
  gpumod.fp16
  gpumod.bf16
  gpumod.wrappers.common
)

add_dependencies(
  compile_time_tests gpumod.test.wrappers.common_type_traits
)
