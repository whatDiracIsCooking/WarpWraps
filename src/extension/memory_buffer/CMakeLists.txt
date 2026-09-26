# Memory buffer modules - RAII wrappers for different memory kinds

gpumod_add_cxx_module_library(
  NAME
  gpumod.extension.memory_buffer
  PRIMARY_INTERFACE
  interface.cppm
  PARTITIONS
  memory_kind.cppm
  host_memory.cppm
  buffer_base.cppm
  device_buffer.cppm
  pinned_buffer.cppm
  unified_buffer.cppm
  host_buffer.cppm
  copy.cppm
  memset.cppm
  LINK_PUBLIC
  gpumod.runtime_api
  gpumod.extension.common
  gpumod.extension.runtime
  IMPORT_STD
)
