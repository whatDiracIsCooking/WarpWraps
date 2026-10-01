# GPU RNG wrappers - type-safe host generation over wwrrandGenerate* (cuRAND or
# hipRAND, per WWR_GPU_BACKEND). dispatch_macros.h sits next to the sources and
# is #include'd by relative path, so it needs no include directory.
wwr_add_cxx_module_library(
  NAME
  wwr.wrappers.rand
  PRIMARY_INTERFACE
  interface.cppm
  PARTITIONS
  type_traits.cppm
  exec.cppm
  IMPLEMENTATION
  instantiations.cpp
  LINK_PUBLIC
  wwr::rand
  wwr::wrappers::common
  IMPORT_STD
)
