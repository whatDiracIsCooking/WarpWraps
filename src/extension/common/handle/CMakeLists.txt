# C++ Module Library
#
# The RAII GPU-handle layer (owning handles + non-owning views), split out of
# gpumod.extension.common so that module stays the pure error/policy foundation.
# Links common back for the error policies, gpu_check and NonCopyable it builds
# on, and runtime_api for the device queries GpuBoundHandle makes (gpuGetDevice/
# gpuSetDevice). The runtime, memory_buffer and blas/solver/fft/sparse handle
# modules that derive concrete handles from these bases link this.
gpumod_add_cxx_module_library(
  NAME
  gpumod.extension.common.handle
  PRIMARY_INTERFACE
  interface.cppm
  PARTITIONS
  gpu_handle.cppm
  device_bound_handle.cppm
  gpu_handle_view.cppm
  device_bound_handle_view.cppm
  LINK_PUBLIC
  gpumod.extension.common
  gpumod.runtime_api
  IMPORT_STD
)
