# The error-handling and RAII-handle layer for GPU sparse linear algebra. The
# dispatch wrappers that once shared this module were split out to
# wwr.wrappers.sparse; what is left here is sparse_error + sparse_handle,
# aggregated by interface.cppm.
wwr_add_cxx_module_library(
  NAME
  wwr.extension.sparse
  PRIMARY_INTERFACE
  interface.cppm
  PARTITIONS
  sparse_error.cppm
  sparse_handle.cppm
  LINK_PUBLIC
  wwr::sparse
  wwr::extension::common
  wwr::extension::handle
  IMPORT_STD
)
