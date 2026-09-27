# The error-handling and RAII-handle layer for GPU sparse linear algebra. The
# dispatch wrappers that once shared this module were split out to
# gpumod.wrappers.sparse; what is left here is sparse_error + sparse_handle,
# aggregated by interface.cppm.
gpumod_add_cxx_module_library(
  NAME
  gpumod.extension.sparse
  PRIMARY_INTERFACE
  interface.cppm
  PARTITIONS
  sparse_error.cppm
  sparse_handle.cppm
  LINK_PUBLIC
  gpumod.sparse
  gpumod.extension.common
  gpumod.extension.common.handle
  IMPORT_STD
)
