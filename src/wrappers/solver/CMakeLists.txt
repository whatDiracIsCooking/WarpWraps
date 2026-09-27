# GPU dense solver extension - type-safe wrappers over gpusolverDn* (cuSOLVER
# Dense or hipSOLVER Dense, per WWR_GPU_BACKEND). dispatch_macros.h
# sits next to the sources and is #include'd by relative path, so it needs no
# include directory.
wwr_add_cxx_module_library(
  NAME
  gpumod.wrappers.solver
  PRIMARY_INTERFACE
  interface.cppm
  PARTITIONS
  type_traits.cppm
  linear_solver_legacy.cppm
  eigen_solver_legacy.cppm
  linear_solver.cppm
  IMPLEMENTATION
  instantiations.cpp
  LINK_PUBLIC
  gpumod.solver
  gpumod.blas
  gpumod.complex
  gpumod.wrappers.common
  IMPORT_STD
)
