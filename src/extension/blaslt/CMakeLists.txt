# The error-handling and RAII layer for the cuBLASLt / hipBLASLt matmul surface.
# blaslt_handle wraps the library handle; blaslt_matmul wraps the matmul
# descriptor / matrix layout / matmul preference. blaslt_error adds no
# specializations -- wwrblasLtStatus_t IS wwrblasStatus_t, so it re-exports
# wwr.extension.blas's policy (hence the LINK_PUBLIC on wwr::extension::blas).
wwr_add_cxx_module_library(
  NAME
  wwr.extension.blaslt
  PRIMARY_INTERFACE
  interface.cppm
  PARTITIONS
  blaslt_error.cppm
  blaslt_handle.cppm
  blaslt_matmul.cppm
  LINK_PUBLIC
  wwr::blaslt
  wwr::extension::blas
  wwr::extension::common
  wwr::extension::handle
  IMPORT_STD
)
