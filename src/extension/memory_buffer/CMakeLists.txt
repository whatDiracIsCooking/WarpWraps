# Memory buffer modules - RAII wrappers for different memory kinds

wwr_add_cxx_module_library(
  NAME
  wwr.extension.memory_buffer
  PRIMARY_INTERFACE
  interface.cppm
  PARTITIONS
  memory_kind.cppm
  host_memory.cppm
  reinterpret_tag.cppm
  base_buffer.cppm
  device_buffer.cppm
  pinned_buffer.cppm
  unified_buffer.cppm
  host_buffer.cppm
  copy.cppm
  memset.cppm
  LINK_PUBLIC
  wwr::runtime_api
  wwr::extension::common
  wwr::extension::handle
  wwr::extension::runtime
  IMPORT_STD
)
