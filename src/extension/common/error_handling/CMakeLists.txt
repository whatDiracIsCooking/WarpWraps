# C++ Module Library
#
# The extension layer's backend-neutral error-handling primitives, split out of
# gpumod.extension.common so the error+check machinery is one self-contained
# unit. Imports only std -- no gpu* names, no runtime API. gpumod.extension.common
# re-exports it (and adds the gpuError_t specializations in :gpu_error), so
# consumers reach these names through `import gpumod.extension.common;`.
gpumod_add_cxx_module_library(
  NAME
  gpumod.extension.common.error_handling
  PRIMARY_INTERFACE
  interface.cppm
  PARTITIONS
  error_code.cppm
  error_policy.cppm
  default_error_policy.cppm
  gpu_check.cppm
  IMPORT_STD
)
