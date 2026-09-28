# C++ Module Library
#
# The extension layer's backend-neutral error-handling primitives, split out of
# wwr.extension.common so the error+check machinery is one self-contained
# unit. Imports only std -- no gpu* names, no runtime API.
# wwr.extension.common re-exports it (and adds the wwrError_t specializations
# in :gpu_error), so consumers reach these names through
# `import wwr.extension.common;`.
#
# :abort_policy is the built-in "print and abort" AbortPolicy. A
# build that wants different behaviour does not swap it -- it passes an explicit
# policy at the call site (the gpu_check policy argument, or the P_* template
# arguments on the RAII wrappers).
wwr_add_cxx_module_library(
  NAME
  wwr.extension.common.error_handling
  PRIMARY_INTERFACE
  interface.cppm
  PARTITIONS
  error_code.cppm
  error_policy.cppm
  abort_policy.cppm
  gpu_check.cppm
  IMPORT_STD
)
