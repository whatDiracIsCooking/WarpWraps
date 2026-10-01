# The error-handling and RAII layer for backend-neutral batched compression
# (nvCOMP / hipCOMP). comp_error specializes the error-handling templates for
# wwrcompStatus_t; comp_manager's CompScratch is RAII over the device scratch
# the batched LLIF borrows, composed from DeviceBufferWrapper (hence the
# memory_buffer link). Aggregated by interface.cppm.
wwr_add_cxx_module_library(
  NAME
  wwr.extension.comp
  PRIMARY_INTERFACE
  interface.cppm
  PARTITIONS
  comp_error.cppm
  comp_manager.cppm
  LINK_PUBLIC
  wwr::comp
  wwr::runtime_api
  wwr::extension::common
  wwr::extension::handle
  wwr::extension::memory_buffer
  IMPORT_STD
)
