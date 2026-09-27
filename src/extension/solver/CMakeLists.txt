# The error-handling and RAII-handle layer for GPU dense solvers. The dispatch
# wrappers that once shared this module were split out to
# gpumod.wrappers.solver; what is left here is solver_error + solver_handle +
# solver_params, aggregated by interface.cppm.
gpumod_add_cxx_module_library(
  NAME
  gpumod.extension.solver
  PRIMARY_INTERFACE
  interface.cppm
  PARTITIONS
  solver_error.cppm
  solver_handle.cppm
  solver_params.cppm
  convenience_solver.cppm
  LINK_PUBLIC
  gpumod.solver
  gpumod.extension.common
  gpumod.extension.handle
  IMPORT_STD
)
