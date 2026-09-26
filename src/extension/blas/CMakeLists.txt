# The error-handling and RAII-handle layer for GPU BLAS. The dispatch wrappers
# that once shared this module were split out to gpumod.wrappers.blas; what is
# left here is blas_error + blas_handle, aggregated by interface.cppm.
gpumod_add_cxx_module_library(
  NAME
  gpumod.extension.blas
  PRIMARY_INTERFACE
  interface.cppm
  PARTITIONS
  blas_error.cppm
  blas_handle.cppm
  LINK_PUBLIC
  gpumod.blas
  gpumod.extension.common
  IMPORT_STD
)
