# The error-handling and RAII-handle layer for GPU dense solvers. The dispatch
# wrappers that once shared this module were split out to
# wwr.wrappers.solver; what is left here is solver_error + solver_handle +
# solver_params, aggregated by interface.cppm.
wwr_add_cxx_module_library(
  NAME
  wwr.extension.solver
  PRIMARY_INTERFACE
  interface.cppm
  PARTITIONS
  solver_error.cppm
  solver_handle.cppm
  solver_params.cppm
  LINK_PUBLIC
  wwr.solver
  wwr.extension.common
  wwr.extension.handle
  IMPORT_STD
)
