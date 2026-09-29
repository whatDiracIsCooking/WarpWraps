# RAII wrapper for a GPU texture object -- a BaseHandle subclass created from a
# resource + texture descriptor, following fft_plan as the template for a handle
# whose CUDA type is an integer with no invalid value. One interface unit, no
# partitions: the wrapper is a single thin class with no error type of its own
# (it checks against wwrError_t). See interface.cppm.
wwr_add_cxx_module_library(
  NAME
  wwr.extension.texture
  PRIMARY_INTERFACE
  interface.cppm
  LINK_PUBLIC
  wwr.runtime_api
  wwr.extension.common
  wwr.extension.handle
  IMPORT_STD
)
