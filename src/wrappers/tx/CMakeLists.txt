# GPU tools-extension wrapper - backend-neutral profiler annotations over
# gputx* (NVTX or rocTX, per WWR_GPU_BACKEND). The tools API is untyped, so
# there is no dispatch machinery here: one interface unit, no partitions,
# instantiations, or dispatch table -- just named free functions and the
# ScopedRange RAII guard. See src/wrappers/tx/interface.cppm.
wwr_add_cxx_module_library(
  NAME
  gpumod.wrappers.tx
  PRIMARY_INTERFACE
  interface.cppm
  LINK_PUBLIC
  gpumod.tx
  IMPORT_STD
)
