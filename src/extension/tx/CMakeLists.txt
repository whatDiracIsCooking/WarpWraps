# GPU tools-extension guard - a backend-neutral RAII nested-range over wwrtx*
# (NVTX or rocTX, per WWR_GPU_BACKEND). The tools API is untyped, so there is no
# dispatch machinery here: one interface unit, no partitions, instantiations, or
# dispatch table -- just the ScopedRange guard. See interface.cppm.
wwr_add_cxx_module_library(
  NAME
  wwr.extension.tx
  PRIMARY_INTERFACE
  interface.cppm
  LINK_PUBLIC
  wwr::tx
  IMPORT_STD
)
