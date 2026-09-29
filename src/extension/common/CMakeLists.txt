# C++ Module Library
#
# Backend-neutral, shared foundation for the extension layer's error handling:
# imports only std and the gpu* names, never src/cuda or src/hip. The
# blas/solver/fft/sparse error+handle modules and the runtime and memory_buffer
# modules all link this.
#
# The error+check primitives (error_code, error_policy, gpu_check) live in the
# sibling wwr.extension.error_handling module (in
# src/extension/error_handling); this module links it PUBLIC and re-exports it,
# then adds the wwrError_t specializations (:gpu_error) on top. The RAII
# GPU-handle layer built on this foundation is another sibling module
# (wwr.extension.handle, in src/extension/handle).
wwr_add_cxx_module_library(
  NAME
  wwr.extension.common
  PRIMARY_INTERFACE
  interface.cppm
  PARTITIONS
  gpu_error.cppm
  device_scope.cppm
  noncopyable.cppm
  LINK_PUBLIC
  wwr.runtime_api
  wwr.extension.error_handling
  IMPORT_STD
)
