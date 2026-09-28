# The error-handling and RAII-plan layer for GPU FFT. The execution wrappers
# that once shared this module were split out to wwr.wrappers.fft; what is
# left here is fft_error + fft_plan, aggregated by interface.cppm.
wwr_add_cxx_module_library(
  NAME
  wwr.extension.fft
  PRIMARY_INTERFACE
  interface.cppm
  PARTITIONS
  fft_error.cppm
  fft_plan.cppm
  LINK_PUBLIC
  wwr.fft
  wwr.extension.common
  wwr.extension.handle
  IMPORT_STD
)
