# C++ Module Library
#
# Backend-neutral, shared foundation for the extension layer's error handling
# and RAII handles: imports only std and the gpu* names, never src/cuda or
# src/hip. The blas/solver/fft/sparse error+handle modules and the runtime and
# memory_buffer modules all link this.
gpumod_add_cxx_module_library(
  NAME
  gpumod.extension.common
  PRIMARY_INTERFACE
  interface.cppm
  PARTITIONS
  error_code.cppm
  gpu_error.cppm
  gpu_check.cppm
  error_policy.cppm
  default_error_policy.cppm
  gpu_handle.cppm
  device_bound_handle.cppm
  noncopyable.cppm
  LINK_PUBLIC
  gpumod.runtime_api
  IMPORT_STD
)
