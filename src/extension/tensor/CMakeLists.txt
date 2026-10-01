# The error-handling and RAII layer for GPU tensor contractions (cuTENSOR /
# hipTensor). No layer-2 execution wrapper -- the API is descriptor- and
# runtime-datatype-driven, so RAII over the vendor create/destroy pairs plus the
# wwrtensorStatus_t error policy is the whole convenience. tensor_error +
# tensor_handle + tensor_plan, aggregated by interface.cppm.
wwr_add_cxx_module_library(
  NAME
  wwr.extension.tensor
  PRIMARY_INTERFACE
  interface.cppm
  PARTITIONS
  tensor_error.cppm
  tensor_handle.cppm
  tensor_plan.cppm
  LINK_PUBLIC
  wwr::tensor
  wwr::extension::common
  wwr::extension::handle
  IMPORT_STD
)
