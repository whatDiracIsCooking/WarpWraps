# The error-handling and RAII-communicator layer for GPU collectives (NCCL /
# RCCL). NCCL/RCCL get no layer-2 execution wrapper -- a collective's element
# type is a runtime enum, not a dispatch-on-type name letter -- so what lives
# here is ccl_error + ccl_comm, aggregated by interface.cppm.
wwr_add_cxx_module_library(
  NAME
  wwr.extension.ccl
  PRIMARY_INTERFACE
  interface.cppm
  PARTITIONS
  ccl_error.cppm
  ccl_comm.cppm
  LINK_PUBLIC
  wwr::ccl
  wwr::extension::common
  wwr::extension::handle
  IMPORT_STD
)
