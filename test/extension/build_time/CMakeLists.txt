# Backend-neutral extension build-time tests
#
# The build is the test: static_asserts on src/extension modules, one module per
# module under test. Built for either GPUMOD_GPU_BACKEND, and without GoogleTest
# or a GPU (GPUMOD_COMPILE_TIME_ONLY; target compile_time_tests).
#
# This directory is for modules whose only tests are build-time ones. A module
# that also has runtime tests keeps both kinds together in its own directory,
# adding its build-time half here and returning early under
# GPUMOD_COMPILE_TIME_ONLY before the GoogleTest part. (The blas/solver/sparse/
# fft handle suites under test/extension are runtime only -- their vendor
# handles need a live GPU; the dispatch checks that prove the wrappers call the
# right entry point live separately, in test/wrappers.)

gpumod_add_cxx_module_library(
  NAME
  gpumod.test.extension.memory_buffer
  IMPORT_STD
  NO_CUDA_DEVICE_LINKING
  PRIMARY_INTERFACE
  memory_buffer.cppm
  LINK_PUBLIC
  gpumod.extension.memory_buffer
)

add_dependencies(compile_time_tests gpumod.test.extension.memory_buffer)

gpumod_add_cxx_module_library(
  NAME
  gpumod.test.extension.handle_view
  IMPORT_STD
  NO_CUDA_DEVICE_LINKING
  PRIMARY_INTERFACE
  handle_view.cppm
  LINK_PUBLIC
  gpumod.extension.common
  gpumod.extension.handle
  gpumod.extension.runtime
  gpumod.extension.blas
  gpumod.extension.solver
  gpumod.extension.sparse
)

add_dependencies(compile_time_tests gpumod.test.extension.handle_view)
