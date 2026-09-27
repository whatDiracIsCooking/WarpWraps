# C++ Module Library
#
# Backend-neutral, shared foundation for the extension layer's error handling:
# imports only std and the gpu* names, never src/cuda or src/hip. The
# blas/solver/fft/sparse error+handle modules and the runtime and memory_buffer
# modules all link this.
#
# The error+check primitives (error_code, error_policy, default_error_policy,
# gpu_check) live in the gpumod.extension.common.error_handling submodule; this
# module links it PUBLIC and re-exports it, then adds the gpuError_t
# specializations (:gpu_error) on top. The RAII GPU-handle layer built on this
# foundation now lives in the handle/ subdirectory
# (gpumod.extension.common.handle), added below.
add_subdirectory(error_handling)

gpumod_add_cxx_module_library(
  NAME
  gpumod.extension.common
  PRIMARY_INTERFACE
  interface.cppm
  PARTITIONS
  gpu_error.cppm
  device_scope.cppm
  noncopyable.cppm
  LINK_PUBLIC
  gpumod.runtime_api
  gpumod.extension.common.error_handling
  IMPORT_STD
)

# The RAII GPU-handle layer, split out but living under common/ because it is a
# direct extension of this foundation. Added after the target above so it can
# link it.
add_subdirectory(handle)
