# C++ Module Library
#
# Backend-neutral, shared foundation for the extension layer's error handling
# and RAII handles: imports only std and the gpu* names, never src/cuda or
# src/hip. The blas/solver/fft/sparse error+handle modules and the runtime and
# memory_buffer modules all link this.
#
# The error+check primitives (error_code, error_policy, default_error_policy,
# gpu_check) live in the gpumod.extension.common.error_handling submodule; this
# module links it PUBLIC and re-exports it, then adds the gpuError_t
# specializations (:gpu_error) and the RAII handle layers on top.
add_subdirectory(error_handling)

gpumod_add_cxx_module_library(
  NAME
  gpumod.extension.common
  PRIMARY_INTERFACE
  interface.cppm
  PARTITIONS
  gpu_error.cppm
  gpu_handle.cppm
  device_bound_handle.cppm
  gpu_handle_view.cppm
  device_bound_handle_view.cppm
  device_scope.cppm
  noncopyable.cppm
  LINK_PUBLIC
  gpumod.runtime_api
  gpumod.extension.common.error_handling
  IMPORT_STD
)
