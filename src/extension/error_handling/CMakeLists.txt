# C++ Module Library
#
# The extension layer's backend-neutral error-handling primitives: a standalone
# sibling module under src/extension, imports only std -- no gpu* names, no
# runtime API. wwr.extension.common links it PUBLIC and re-exports it (and adds
# the wwrError_t specializations in :gpu_error), so consumers reach these names
# through `import wwr.extension.common;`.
#
# The layer ships no default error policy: every gpu_check call and every RAII
# wrapper names the policy it wants (the gpu_check policy argument, or the P_*
# template arguments on the wrappers). Policy choice belongs to the consumer.
wwr_add_cxx_module_library(
  NAME
  wwr.extension.error_handling
  PRIMARY_INTERFACE
  interface.cppm
  PARTITIONS
  error_code.cppm
  error_policy.cppm
  gpu_check.cppm
  IMPORT_STD
)
