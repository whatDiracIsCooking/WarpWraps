# C++ Module Library
#
# The RAII GPU-handle layer (owning handles + non-owning views), split out of
# wwr.extension.common so that module stays the pure error/policy foundation.
# Links common back for the error policies, gpu_check and NonCopyable it builds
# on, and runtime_api for the device queries DeviceBoundHandle makes
# (wwrGetDevice/wwrSetDevice). The runtime and memory_buffer modules derive
# concrete handles from the device-bound base; the blas/solver/fft/sparse
# handle modules derive from the stream-bound base layered on top of it. All
# link this.
wwr_add_cxx_module_library(
  NAME
  wwr.extension.handle
  PRIMARY_INTERFACE
  interface.cppm
  PARTITIONS
  handle.cppm
  device_bound_handle.cppm
  stream_bound_handle.cppm
  handle_view.cppm
  device_bound_handle_view.cppm
  stream_bound_handle_view.cppm
  device_handle.cppm
  LINK_PUBLIC
  wwr::extension::common
  wwr::runtime_api
  IMPORT_STD
)
