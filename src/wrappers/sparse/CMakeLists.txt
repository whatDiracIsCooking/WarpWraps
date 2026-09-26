# GPU sparse extension - type-safe wrappers over the legacy typed (S/D/C/Z)
# gpusparse* functions cuSPARSE and hipSPARSE share (per GPUMOD_GPU_BACKEND).
# dispatch_macros.h sits next to the sources and is #include'd by relative
# path, so it needs no include directory.
gpumod_add_cxx_module_library(
  NAME
  gpumod.wrappers.sparse
  PRIMARY_INTERFACE
  interface.cppm
  PARTITIONS
  level_2.cppm
  solvers.cppm
  extra.cppm
  conversion.cppm
  IMPLEMENTATION
  instantiations.cpp
  LINK_PUBLIC
  gpumod.sparse
  gpumod.complex
  gpumod.wrappers.common
  IMPORT_STD
)
