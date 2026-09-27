# Memory buffer modules - RAII wrappers for different memory kinds

wwr_add_cxx_module_library(
  NAME
  gpumod.extension.memory_buffer
  PRIMARY_INTERFACE
  interface.cppm
  PARTITIONS
  memory_kind.cppm
  host_memory.cppm
  base_buffer.cppm
  device_buffer.cppm
  pinned_buffer.cppm
  unified_buffer.cppm
  host_buffer.cppm
  copy.cppm
  memset.cppm
  convenience_memory_buffer.cppm
  LINK_PUBLIC
  gpumod.runtime_api
  gpumod.extension.common
  gpumod.extension.runtime
  IMPORT_STD
)
